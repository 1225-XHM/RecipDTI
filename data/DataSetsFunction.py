from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import EVALUATION_SETTINGS, normalize_setting


SPLIT_FORMAT_VERSION = 2
FEATURE_CACHE_VERSION = 2


@dataclass(frozen=True)
class Sample:
    smiles: str
    protein: str
    label: int


@dataclass(frozen=True)
class SplitManifest:
    setting: str
    seed: int
    fingerprint: str
    indices: dict[str, list[int]]
    statistics: dict[str, dict[str, int]]

    @property
    def train(self) -> list[int]:
        return self.indices["train"]

    @property
    def val(self) -> list[int]:
        return self.indices["val"]

    @property
    def test(self) -> list[int]:
        return self.indices["test"]


@dataclass(frozen=True)
class FeatureRecord:
    key: str
    features: torch.Tensor
    metadata: dict[str, object]


def stable_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def canonicalize_smiles(smiles: str) -> str:
    value = smiles.strip()
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def normalize_protein_sequence(sequence: str) -> str:
    value = "".join(sequence.split()).upper()
    if not value:
        raise ValueError("protein sequence is empty")
    invalid = sorted({character for character in value if not "A" <= character <= "Z"})
    if invalid:
        raise ValueError(f"protein sequence contains unsupported characters: {invalid}")
    return value


def parse_binary_label(value: str | int | float) -> int:
    numeric = float(value)
    if numeric not in {0.0, 1.0}:
        raise ValueError(f"label must be 0 or 1, found {value!r}")
    return int(numeric)


def read_samples(path: str | Path) -> list[Sample]:
    source = Path(path)
    samples: list[Sample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 3:
                raise ValueError(
                    f"{source}:{line_number} expected 3 columns, found {len(fields)}"
                )
            try:
                samples.append(
                    Sample(
                        smiles=canonicalize_smiles(fields[0]),
                        protein=normalize_protein_sequence(fields[1]),
                        label=parse_binary_label(fields[2]),
                    )
                )
            except Exception as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if not samples:
        raise ValueError(f"no samples found in {source}")
    return samples


def dataset_fingerprint(samples: Sequence[Sample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.smiles.encode("utf-8"))
        digest.update(b"\t")
        digest.update(sample.protein.encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(sample.label).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _strip_dummy_atoms(molecule: Chem.Mol) -> Chem.Mol | None:
    editable = Chem.RWMol(molecule)
    indices = [
        atom.GetIdx()
        for atom in editable.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]
    for index in reversed(indices):
        editable.RemoveAtom(index)
    clean = editable.GetMol()
    if clean.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(clean)
    except Exception:
        return None
    return clean


@lru_cache(maxsize=100_000)
def brics_fragments(smiles: str) -> tuple[str, ...]:
    canonical = canonicalize_smiles(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    broken = BRICS.BreakBRICSBonds(molecule)
    fragment_molecules = Chem.GetMolFrags(
        broken,
        asMols=True,
        sanitizeFrags=False,
    )
    fragments: list[str] = []
    for fragment in fragment_molecules:
        clean = _strip_dummy_atoms(fragment)
        if clean is None:
            continue
        value = Chem.MolToSmiles(
            clean,
            canonical=True,
            isomericSmiles=True,
        )
        if value:
            fragments.append(value)
    return tuple(fragments) if fragments else (canonical,)


def _split_statistics(
    samples: Sequence[Sample],
    split: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, indices in split.items():
        selected = [samples[index] for index in indices]
        positives = sum(item.label for item in selected)
        result[name] = {
            "samples": len(selected),
            "positive": positives,
            "negative": len(selected) - positives,
            "drugs": len({item.smiles for item in selected}),
            "proteins": len({item.protein for item in selected}),
        }
    return result


def validate_split(
    samples: Sequence[Sample],
    split: Mapping[str, Sequence[int]],
    setting: str,
) -> None:
    name = normalize_setting(setting)
    expected = {"train", "val", "test"}
    if set(split) != expected:
        raise ValueError(f"split keys must be {expected}")

    index_sets = {key: set(int(i) for i in split[key]) for key in expected}
    for key in expected:
        if not index_sets[key]:
            raise ValueError(f"{key} split is empty")
        if len(index_sets[key]) != len(split[key]):
            raise ValueError(f"{key} split contains duplicate indices")

    for values in index_sets.values():
        if any(index < 0 or index >= len(samples) for index in values):
            raise IndexError("split contains an out-of-range sample index")

    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    if any(index_sets[left] & index_sets[right] for left, right in pairs):
        raise ValueError("train, validation and test indices overlap")

    selected = {
        key: [samples[index] for index in split[key]]
        for key in expected
    }
    for key, values in selected.items():
        if {item.label for item in values} != {0, 1}:
            raise ValueError(f"{name} {key} split must contain both classes")

    if name in {"E2", "E4"}:
        drugs = {key: {item.smiles for item in values} for key, values in selected.items()}
        if any(drugs[left] & drugs[right] for left, right in pairs):
            raise ValueError(f"{name} contains drug leakage")

    if name in {"E3", "E4"}:
        proteins = {key: {item.protein for item in values} for key, values in selected.items()}
        if any(proteins[left] & proteins[right] for left, right in pairs):
            raise ValueError(f"{name} contains protein leakage")


def load_split_manifest(
    path: str | Path,
    samples: Sequence[Sample],
    setting: str | None = None,
) -> SplitManifest:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    indices = payload.get("indices", payload)
    manifest_setting = normalize_setting(setting or payload.get("setting", "E1"))
    version = int(payload.get("split_version", SPLIT_FORMAT_VERSION))
    if version != SPLIT_FORMAT_VERSION:
        raise ValueError(f"unsupported split format version: {version}")

    expected_fingerprint = dataset_fingerprint(samples)
    stored_fingerprint = str(payload.get("dataset_fingerprint", expected_fingerprint))
    if stored_fingerprint != expected_fingerprint:
        raise ValueError("split manifest does not match the current dataset")

    normalized = {
        key: [int(index) for index in indices[key]]
        for key in ("train", "val", "test")
    }
    validate_split(samples, normalized, manifest_setting)
    statistics = payload.get("statistics") or _split_statistics(samples, normalized)
    return SplitManifest(
        setting=manifest_setting,
        seed=int(payload.get("seed", 0)),
        fingerprint=stored_fingerprint,
        indices=normalized,
        statistics=statistics,
    )


class FeatureStore:
    def __init__(self, cache_dir: str | Path) -> None:
        root = Path(cache_dir)
        self.drug_dir = root / "drugs"
        self.protein_dir = root / "proteins"

    @staticmethod
    def _read_payload(path: Path) -> dict[str, object]:
        if not path.exists():
            raise FileNotFoundError(f"missing cached feature: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"invalid cached feature payload: {path}")
        return payload

    @staticmethod
    def _extract_tensor(payload: Mapping[str, object], path: Path) -> torch.Tensor:
        value = payload.get("features")
        if not torch.is_tensor(value):
            raise KeyError(f"{path} does not contain a tensor named features")
        if value.ndim != 2 or value.size(0) < 1 or value.size(1) < 1:
            raise ValueError(f"invalid feature shape in {path}: {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite feature values in {path}")
        return value.float().contiguous()

    @lru_cache(maxsize=4096)
    def load_drug(self, smiles: str) -> FeatureRecord:
        canonical = canonicalize_smiles(smiles)
        path = self.drug_dir / f"{stable_key(canonical)}.pt"
        payload = self._read_payload(path)
        features = self._extract_tensor(payload, path)
        metadata = {key: value for key, value in payload.items() if key != "features"}
        return FeatureRecord(canonical, features, metadata)

    @lru_cache(maxsize=2048)
    def load_protein(self, sequence: str) -> FeatureRecord:
        normalized = normalize_protein_sequence(sequence)
        path = self.protein_dir / f"{stable_key(normalized)}.pt"
        payload = self._read_payload(path)
        features = self._extract_tensor(payload, path)
        metadata = {key: value for key, value in payload.items() if key != "features"}
        return FeatureRecord(normalized, features, metadata)

    def dimensions(self, samples: Sequence[Sample]) -> tuple[int, int]:
        first = samples[0]
        fragment_dim = int(self.load_drug(first.smiles).features.size(-1))
        residue_dim = int(self.load_protein(first.protein).features.size(-1))
        return fragment_dim, residue_dim


class RecipDTIDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        indices: Sequence[int],
        cache_dir: str | Path,
    ) -> None:
        self.samples = list(samples)
        self.indices = [int(index) for index in indices]
        self.store = FeatureStore(cache_dir)
        if not self.indices:
            raise ValueError("dataset indices cannot be empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        sample = self.samples[self.indices[item]]
        drug = self.store.load_drug(sample.smiles)
        protein = self.store.load_protein(sample.protein)
        return {
            "fragment_features": drug.features,
            "residue_features": protein.features,
            "label": float(sample.label),
            "smiles": sample.smiles,
            "protein": sample.protein,
        }


def collate_recipdti(batch: list[dict[str, object]]) -> dict[str, object]:
    if not batch:
        raise ValueError("batch cannot be empty")
    fragment_features = [item["fragment_features"] for item in batch]
    residue_features = [item["residue_features"] for item in batch]
    if not all(torch.is_tensor(value) for value in fragment_features + residue_features):
        raise TypeError("feature values must be tensors")

    fragment_lengths = torch.tensor(
        [int(value.size(0)) for value in fragment_features],
        dtype=torch.long,
    )
    residue_lengths = torch.tensor(
        [int(value.size(0)) for value in residue_features],
        dtype=torch.long,
    )
    fragments = pad_sequence(fragment_features, batch_first=True)
    residues = pad_sequence(residue_features, batch_first=True)
    fragment_mask = (
        torch.arange(fragments.size(1)).unsqueeze(0)
        < fragment_lengths.unsqueeze(1)
    )
    residue_mask = (
        torch.arange(residues.size(1)).unsqueeze(0)
        < residue_lengths.unsqueeze(1)
    )
    return {
        "fragment_features": fragments,
        "residue_features": residues,
        "fragment_mask": fragment_mask,
        "residue_mask": residue_mask,
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "smiles": [str(item["smiles"]) for item in batch],
        "proteins": [str(item["protein"]) for item in batch],
    }


def build_loaders(
    samples: Sequence[Sample],
    manifest: SplitManifest,
    cache_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    datasets = {
        key: RecipDTIDataset(samples, manifest.indices[key], cache_dir)
        for key in ("train", "val", "test")
    }
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "collate_fn": collate_recipdti,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(datasets["train"], shuffle=True, **common)
    val_loader = DataLoader(datasets["val"], shuffle=False, **common)
    test_loader = DataLoader(datasets["test"], shuffle=False, **common)
    return train_loader, val_loader, test_loader


def iter_unique_drugs(samples: Iterable[Sample]) -> list[str]:
    return sorted({sample.smiles for sample in samples})


def iter_unique_proteins(samples: Iterable[Sample]) -> list[str]:
    return sorted({sample.protein for sample in samples})
