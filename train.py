from __future__ import annotations

import csv
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model import RecipDTI
from metric import PAPER_METRICS, classification_metrics, summarize_metrics
from utils import move_batch_to_device, save_json, seed_everything


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * int(count)
        self.count += int(count)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


@dataclass
class PredictionBuffer:
    labels: list[np.ndarray] = field(default_factory=list)
    probabilities: list[np.ndarray] = field(default_factory=list)

    def append(self, labels: torch.Tensor, logits: torch.Tensor) -> None:
        self.labels.append(labels.detach().cpu().numpy())
        self.probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.labels:
            raise ValueError("prediction buffer is empty")
        return np.concatenate(self.labels), np.concatenate(self.probabilities)

    def metrics(self, threshold: float = 0.5) -> dict[str, float]:
        labels, probabilities = self.arrays()
        return classification_metrics(labels, probabilities, threshold=threshold)


@dataclass(frozen=True)
class EpochResult:
    loss: float
    metrics: dict[str, float]
    samples: int
    seconds: float

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        base = {
            f"{prefix}loss": self.loss,
            f"{prefix}samples": float(self.samples),
            f"{prefix}seconds": self.seconds,
        }
        base.update({f"{prefix}{key}": value for key, value in self.metrics.items()})
        return base


@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 100
    learning_rate: float = 1e-4
    threshold: float = 0.5
    gradient_clip: float | None = None
    seed: int = 3407

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive when provided")


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_optimizer(
    model: nn.Module,
    learning_rate: float,
) -> torch.optim.Optimizer:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)


def forward_batch(
    model: RecipDTI,
    batch: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels"]
    if not torch.is_tensor(labels):
        raise TypeError("batch labels must be a tensor")
    output = model(
        batch["fragment_features"],
        batch["residue_features"],
        batch["fragment_mask"],
        batch["residue_mask"],
    )
    logits = output.get("logits")
    if not torch.is_tensor(logits):
        raise TypeError("model output does not contain logits")
    if logits.shape != labels.shape:
        raise ValueError("logit and label shapes do not match")
    return logits, labels


def train_epoch(
    model: RecipDTI,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    gradient_clip: float | None = None,
) -> EpochResult:
    model.train()
    loss_meter = AverageMeter()
    predictions = PredictionBuffer()
    started = time.perf_counter()

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            logits, labels = forward_batch(model, batch)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if gradient_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        size = int(labels.size(0))
        loss_meter.update(float(loss.item()), size)
        predictions.append(labels, logits)

    seconds = time.perf_counter() - started
    metrics = predictions.metrics() if loss_meter.count else {}
    return EpochResult(loss_meter.average, metrics, loss_meter.count, seconds)


@torch.inference_mode()
def evaluate_epoch(
    model: RecipDTI,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    description: str = "Evaluate",
) -> EpochResult:
    model.eval()
    loss_meter = AverageMeter()
    predictions = PredictionBuffer()
    started = time.perf_counter()

    for batch in tqdm(loader, desc=description, leave=False):
        batch = move_batch_to_device(batch, device)
        with autocast_context(device):
            logits, labels = forward_batch(model, batch)
            loss = criterion(logits, labels)
        size = int(labels.size(0))
        loss_meter.update(float(loss.item()), size)
        predictions.append(labels, logits)

    seconds = time.perf_counter() - started
    metrics = predictions.metrics(threshold=threshold) if loss_meter.count else {}
    return EpochResult(loss_meter.average, metrics, loss_meter.count, seconds)


class CheckpointManager:
    def __init__(
        self,
        directory: str | Path,
        monitor: str = "AUROC",
        mode: str = "max",
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be max or min")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.path = self.directory / "best_model.pt"

    def improved(self, value: float) -> bool:
        return value > self.best if self.mode == "max" else value < self.best

    def update(
        self,
        value: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metadata: Mapping[str, object] | None = None,
    ) -> bool:
        if not self.improved(value):
            return False
        self.best = float(value)
        payload: dict[str, object] = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "monitor": self.monitor,
            "monitor_value": self.best,
        }
        if metadata:
            payload.update(dict(metadata))
        torch.save(payload, self.path)
        return True

    def load(
        self,
        model: nn.Module,
        device: torch.device,
        strict: bool = True,
    ) -> dict[str, object]:
        payload = torch.load(self.path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"], strict=strict)
        return payload


def write_history(rows: Iterable[Mapping[str, object]], path: str | Path) -> Path:
    records = [dict(row) for row in rows]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        target.write_text("", encoding="utf-8")
        return target
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return target


def run_training(
    model: RecipDTI,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    options: TrainingOptions = TrainingOptions(),
    metadata: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, float]], Path]:
    options.validate()
    seed_everything(options.seed)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = make_optimizer(model, options.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    manager = CheckpointManager(output_dir, monitor="AUROC", mode="max")
    history: list[dict[str, float]] = []

    for epoch in range(1, options.epochs + 1):
        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            gradient_clip=options.gradient_clip,
        )
        val_result = evaluate_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold=options.threshold,
            description="Validation",
        )
        row: dict[str, float] = {"epoch": float(epoch)}
        row.update(train_result.to_dict(prefix="train_"))
        row.update(val_result.to_dict(prefix="val_"))
        history.append(row)
        manager.update(
            val_result.metrics["AUROC"],
            model,
            optimizer,
            epoch,
            metadata=metadata,
        )

    write_history(history, Path(output_dir) / "history.csv")
    save_json(
        {
            "epochs": options.epochs,
            "seed": options.seed,
            "best_AUROC": manager.best,
        },
        Path(output_dir) / "training_summary.json",
    )
    return history, manager.path


def evaluate_checkpoint(
    model: RecipDTI,
    loader: DataLoader,
    checkpoint_path: str | Path,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model = model.to(device)
    result = evaluate_epoch(
        model,
        loader,
        nn.BCEWithLogitsLoss(),
        device,
        threshold=threshold,
        description="Test",
    )
    output = {"loss": result.loss, **result.metrics}
    return output


def summarize_runs(
    runs: Iterable[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return summarize_metrics(runs, names=PAPER_METRICS)
