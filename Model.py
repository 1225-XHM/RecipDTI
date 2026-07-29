from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelShape:
    fragment_dim: int
    residue_dim: int
    hidden_dim: int = 256
    dropout: float = 0.10
    top_k: int = 16

    def validate(self) -> None:
        if min(self.fragment_dim, self.residue_dim, self.hidden_dim) < 1:
            raise ValueError("feature dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")


def _validate_inputs(
    fragments: torch.Tensor,
    residues: torch.Tensor,
    fragment_mask: torch.Tensor,
    residue_mask: torch.Tensor,
) -> None:
    if fragments.ndim != 3 or residues.ndim != 3:
        raise ValueError("fragment and residue features must be rank-3 tensors")
    if fragment_mask.ndim != 2 or residue_mask.ndim != 2:
        raise ValueError("fragment and residue masks must be rank-2 tensors")
    if fragments.shape[:2] != fragment_mask.shape:
        raise ValueError("fragment feature and mask shapes do not match")
    if residues.shape[:2] != residue_mask.shape:
        raise ValueError("residue feature and mask shapes do not match")
    if fragments.size(0) != residues.size(0):
        raise ValueError("fragment and residue batch sizes do not match")
    if fragment_mask.dtype != torch.bool or residue_mask.dtype != torch.bool:
        raise TypeError("masks must use torch.bool")
    if not fragment_mask.any(dim=1).all():
        raise ValueError("every sample requires at least one fragment")
    if not residue_mask.any(dim=1).all():
        raise ValueError("every sample requires at least one residue")


def masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    if scores.shape != mask.shape:
        raise ValueError("score and mask shapes must match")
    minimum = torch.finfo(scores.dtype).min
    masked = scores.masked_fill(~mask, minimum)
    weights = torch.softmax(masked, dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    denominator = weights.sum(dim=dim, keepdim=True).clamp_min(1e-9)
    return weights / denominator


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if features.shape[:2] != mask.shape:
        raise ValueError("feature and mask shapes must match")
    weights = mask.unsqueeze(-1).to(features.dtype)
    numerator = (features * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def masked_weighted_mean(
    features: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if features.shape[:2] != weights.shape or weights.shape != mask.shape:
        raise ValueError("feature, weight and mask shapes are inconsistent")
    normalized = weights * mask.to(weights.dtype)
    numerator = (features * normalized.unsqueeze(-1)).sum(dim=1)
    denominator = normalized.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return numerator / denominator


class DimensionProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("projection dimensions must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.projection: nn.Module
        if self.input_dim == self.output_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.size(-1) != self.input_dim:
            raise ValueError(
                f"expected input dimension {self.input_dim}, found {features.size(-1)}"
            )
        return self.projection(features)


class ContextGate(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.fragment_gate = nn.Linear(self.dim * 2, self.dim)
        self.residue_gate = nn.Linear(self.dim * 2, self.dim)

    def forward(
        self,
        fragment_context: torch.Tensor,
        residue_context: torch.Tensor,
        fragment_mask: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fragment_summary = masked_mean(fragment_context, fragment_mask)
        residue_summary = masked_mean(residue_context, residue_mask)
        fragment_input = torch.cat([fragment_summary, residue_summary], dim=-1)
        residue_input = torch.cat([residue_summary, fragment_summary], dim=-1)
        fragment_gate = torch.sigmoid(self.fragment_gate(fragment_input))
        residue_gate = torch.sigmoid(self.residue_gate(residue_input))
        return fragment_gate, residue_gate


class FeatureCalibration(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.projection = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.dim)

    def forward(
        self,
        base: torch.Tensor,
        context: torch.Tensor,
        gate: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if base.shape != context.shape:
            raise ValueError("base and context shapes must match")
        if gate.shape != (base.size(0), base.size(-1)):
            raise ValueError("gate shape is inconsistent with the feature tensor")
        update = self.projection(context * gate.unsqueeze(1))
        output = self.norm(base + self.dropout(update))
        return output * mask.unsqueeze(-1).to(output.dtype)


class FragmentResidueCrossAttention(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.fragment_norm = nn.LayerNorm(self.dim)
        self.residue_norm = nn.LayerNorm(self.dim)
        self.context_gate = ContextGate(self.dim)
        self.fragment_calibration = FeatureCalibration(self.dim, dropout)
        self.residue_calibration = FeatureCalibration(self.dim, dropout)

    def affinity(
        self,
        fragments: torch.Tensor,
        residues: torch.Tensor,
    ) -> torch.Tensor:
        fragment_unit = F.normalize(self.fragment_norm(fragments), dim=-1)
        residue_unit = F.normalize(self.residue_norm(residues), dim=-1)
        return torch.matmul(fragment_unit, residue_unit.transpose(-1, -2))

    def forward(
        self,
        fragments: torch.Tensor,
        residues: torch.Tensor,
        fragment_mask: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        affinity = self.affinity(fragments, residues)
        pair_mask = fragment_mask.unsqueeze(-1) & residue_mask.unsqueeze(1)
        fragment_attention = masked_softmax(affinity, pair_mask, dim=-1)
        residue_attention = masked_softmax(
            affinity.transpose(-1, -2),
            pair_mask.transpose(-1, -2),
            dim=-1,
        )
        fragment_unit = F.normalize(self.fragment_norm(fragments), dim=-1)
        residue_unit = F.normalize(self.residue_norm(residues), dim=-1)
        fragment_context = torch.matmul(fragment_attention, residue_unit)
        residue_context = torch.matmul(residue_attention, fragment_unit)
        fragment_gate, residue_gate = self.context_gate(
            fragment_context,
            residue_context,
            fragment_mask,
            residue_mask,
        )
        fragment_output = self.fragment_calibration(
            fragments,
            fragment_context,
            fragment_gate,
            fragment_mask,
        )
        residue_output = self.residue_calibration(
            residues,
            residue_context,
            residue_gate,
            residue_mask,
        )
        return fragment_output, residue_output


class SelectiveAffinityFusion(nn.Module):
    def __init__(self, dim: int, top_k: int, dropout: float) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.dim = int(dim)
        self.top_k = int(top_k)
        self.projection = nn.Sequential(
            nn.Linear(self.dim * 3, self.dim),
            nn.LayerNorm(self.dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _masked_max(
        scores: torch.Tensor,
        mask: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        minimum = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(~mask, minimum)
        values = masked.max(dim=dim).values
        valid = mask.any(dim=dim)
        return torch.where(valid, values, torch.zeros_like(values))

    def _global_summary(
        self,
        fragments: torch.Tensor,
        residues: torch.Tensor,
        affinity: torch.Tensor,
        fragment_mask: torch.Tensor,
        residue_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fragment_scores = self._masked_max(affinity, pair_mask, dim=-1)
        residue_scores = self._masked_max(affinity, pair_mask, dim=-2)
        fragment_weights = torch.sigmoid(fragment_scores)
        residue_weights = torch.sigmoid(residue_scores)
        fragment_global = masked_weighted_mean(
            fragments,
            fragment_weights,
            fragment_mask,
        )
        residue_global = masked_weighted_mean(
            residues,
            residue_weights,
            residue_mask,
        )
        return fragment_global, residue_global

    def _local_summary(
        self,
        fragments: torch.Tensor,
        residues: torch.Tensor,
        affinity: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, residue_count = affinity.shape
        flat_scores = affinity.flatten(1)
        flat_mask = pair_mask.flatten(1)
        count = min(self.top_k, flat_scores.size(1))
        candidates = flat_scores.masked_fill(
            ~flat_mask,
            torch.finfo(flat_scores.dtype).min,
        )
        top_scores, top_indices = candidates.topk(count, dim=1)
        valid = flat_mask.gather(1, top_indices)
        fragment_indices = top_indices // residue_count
        residue_indices = top_indices % residue_count
        batch_indices = torch.arange(batch_size, device=affinity.device).unsqueeze(1)
        selected_fragments = fragments[batch_indices, fragment_indices]
        selected_residues = residues[batch_indices, residue_indices]
        interactions = selected_fragments * selected_residues
        weights = masked_softmax(top_scores, valid, dim=-1)
        return (interactions * weights.unsqueeze(-1)).sum(dim=1)

    def forward(
        self,
        fragments: torch.Tensor,
        residues: torch.Tensor,
        fragment_mask: torch.Tensor,
        residue_mask: torch.Tensor,
        affinity: torch.Tensor,
    ) -> torch.Tensor:
        pair_mask = fragment_mask.unsqueeze(-1) & residue_mask.unsqueeze(1)
        fragment_global, residue_global = self._global_summary(
            fragments,
            residues,
            affinity,
            fragment_mask,
            residue_mask,
            pair_mask,
        )
        local = self._local_summary(fragments, residues, affinity, pair_mask)
        merged = torch.cat([fragment_global, residue_global, local], dim=-1)
        return self.projection(merged)


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        bottleneck = max(1, hidden_dim // 2)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


class RecipDTI(nn.Module):
    def __init__(
        self,
        fragment_dim: int,
        residue_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.10,
        top_k: int = 16,
    ) -> None:
        super().__init__()
        self.shape = ModelShape(
            fragment_dim=fragment_dim,
            residue_dim=residue_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            top_k=top_k,
        )
        self.shape.validate()
        self.fragment_projector = DimensionProjector(fragment_dim, hidden_dim)
        self.residue_projector = DimensionProjector(residue_dim, hidden_dim)
        self.interaction = FragmentResidueCrossAttention(hidden_dim, dropout)
        self.fusion = SelectiveAffinityFusion(hidden_dim, top_k, dropout)
        self.classifier = PredictionHead(hidden_dim, dropout)

    @staticmethod
    def cosine_affinity(
        fragments: torch.Tensor,
        residues: torch.Tensor,
    ) -> torch.Tensor:
        fragment_unit = F.normalize(fragments, dim=-1)
        residue_unit = F.normalize(residues, dim=-1)
        return torch.matmul(fragment_unit, residue_unit.transpose(-1, -2))

    def encode_pair(
        self,
        fragment_features: torch.Tensor,
        residue_features: torch.Tensor,
        fragment_mask: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(
            fragment_features,
            residue_features,
            fragment_mask,
            residue_mask,
        )
        fragments = self.fragment_projector(fragment_features)
        residues = self.residue_projector(residue_features)
        fragments = fragments * fragment_mask.unsqueeze(-1).to(fragments.dtype)
        residues = residues * residue_mask.unsqueeze(-1).to(residues.dtype)
        fragments, residues = self.interaction(
            fragments,
            residues,
            fragment_mask,
            residue_mask,
        )
        affinity = self.cosine_affinity(fragments, residues)
        return self.fusion(
            fragments,
            residues,
            fragment_mask,
            residue_mask,
            affinity,
        )

    def forward(
        self,
        fragment_features: torch.Tensor,
        residue_features: torch.Tensor,
        fragment_mask: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = self.encode_pair(
            fragment_features,
            residue_features,
            fragment_mask,
            residue_mask,
        )
        return {
            "logits": self.classifier(fused),
            "z_fuse": fused,
        }


def build_model(
    fragment_dim: int,
    residue_dim: int,
    hidden_dim: int = 256,
    dropout: float = 0.10,
    top_k: int = 16,
) -> RecipDTI:
    return RecipDTI(
        fragment_dim=fragment_dim,
        residue_dim=residue_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        top_k=top_k,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Mapping[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint does not contain a valid model state")
    model.load_state_dict(state, strict=strict)
    return checkpoint
