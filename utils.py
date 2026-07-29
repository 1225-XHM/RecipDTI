from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    os.environ["PYTHONHASHSEED"] = str(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(data: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def move_batch_to_device(
    batch: Mapping[str, object],
    device: torch.device,
    non_blocking: bool = True,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device, non_blocking=non_blocking)
        else:
            result[key] = value
    return result


def detach_to_cpu(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(item) for item in value)
    return value


def resolve_device(name: str | None = None) -> torch.device:
    if name:
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_memory_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def format_bytes(value: int) -> str:
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"
