from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


DATASET_FILES = {
    "BindingDB": ("BindingDB.txt",),
    "BioSNAP": ("BioSNAP.txt", "BIOSNAP.txt"),
    "Human": ("Human.txt",),
}

DATASET_ALIASES = {
    "bindingdb": "BindingDB",
    "biosnap": "BioSNAP",
    "human": "Human",
}

EVALUATION_SETTINGS = ("E1", "E2", "E3", "E4")


@dataclass(frozen=True)
class EncoderConfig:
    molformer_name: str = "ibm-research/MoLFormer-XL-both-10pct"
    esm2_name: str = "facebook/esm2_t33_650M_UR50D"
    max_fragment_tokens: int = 128
    max_protein_residues: int = 1000
    fragment_batch_size: int = 64
    protein_batch_size: int = 1
    cache_dtype: str = "float16"

    def validate(self) -> None:
        if self.max_fragment_tokens < 2:
            raise ValueError("max_fragment_tokens must be at least 2")
        if self.max_protein_residues < 1:
            raise ValueError("max_protein_residues must be positive")
        if self.fragment_batch_size < 1 or self.protein_batch_size < 1:
            raise ValueError("encoder batch sizes must be positive")
        if self.cache_dtype not in {"float16", "float32"}:
            raise ValueError("cache_dtype must be float16 or float32")


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 256
    dropout: float = 0.10
    top_k: int = 16

    def validate(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    num_workers: int = 0
    seeds: tuple[int, ...] = (3407, 3408, 3409, 3410, 3411)

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not self.seeds:
            raise ValueError("at least one random seed is required")


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    project_dir: Path
    encoder: EncoderConfig = EncoderConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

    def validate(self) -> None:
        if self.dataset not in DATASET_FILES:
            raise ValueError(f"unsupported dataset: {self.dataset}")
        self.encoder.validate()
        self.model.validate()
        self.train.validate()

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def data_path(self) -> Path:
        candidates = DATASET_FILES[self.dataset]
        for filename in candidates:
            path = self.data_dir / filename
            if path.exists():
                return path
        return self.data_dir / candidates[0]

    @property
    def cache_dir(self) -> Path:
        return self.project_dir / "cache" / self.dataset

    @property
    def split_dir(self) -> Path:
        return self.project_dir / "splits" / self.dataset

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "outputs" / self.dataset

    def split_path(self, setting: str, seed: int) -> Path:
        name = normalize_setting(setting)
        return self.split_dir / f"{name}_seed_{int(seed)}.json"

    def run_dir(self, setting: str, seed: int) -> Path:
        name = normalize_setting(setting)
        return self.output_dir / name / f"seed_{int(seed)}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["project_dir"] = str(self.project_dir)
        return payload


def normalize_dataset(dataset: str) -> str:
    key = dataset.strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"unsupported dataset: {dataset}")
    return DATASET_ALIASES[key]


def normalize_setting(setting: str) -> str:
    value = setting.strip().upper()
    if value not in EVALUATION_SETTINGS:
        raise ValueError(f"setting must be one of {EVALUATION_SETTINGS}")
    return value


def get_config(
    dataset: str,
    project_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    root = Path(project_dir or Path(__file__).resolve().parent).expanduser().resolve()
    canonical = normalize_dataset(dataset)
    encoder = EncoderConfig()
    model = ModelConfig()
    train = TrainConfig()

    if overrides:
        encoder_values = dict(overrides.get("encoder", {}))
        model_values = dict(overrides.get("model", {}))
        train_values = dict(overrides.get("train", {}))
        if encoder_values:
            encoder = EncoderConfig(**{**asdict(encoder), **encoder_values})
        if model_values:
            model = ModelConfig(**{**asdict(model), **model_values})
        if train_values:
            if "seeds" in train_values:
                train_values["seeds"] = tuple(int(v) for v in train_values["seeds"])
            train = TrainConfig(**{**asdict(train), **train_values})

    config = ExperimentConfig(
        dataset=canonical,
        project_dir=root,
        encoder=encoder,
        model=model,
        train=train,
    )
    config.validate()
    return config
