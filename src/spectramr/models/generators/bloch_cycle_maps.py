"""Tissue-map head, acquisition sequence and refinement stage of ``bloch_cycle_network``.

The network predicts map logits, bounds them into physiological ranges in milliseconds
(an unbounded head read as ms makes ``exp(-TE/T2)`` underflow to zero), renders the
signal through the Bloch equation at the declared sequence, and optionally refines the
logits in further Bloch modules that see the input and the previous render.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

#: Physiological bounds of predicted maps, in ms; ``ParameterEstimator`` reads the same.
T1_RANGE_MS = (100.0, 4000.0)
T2_RANGE_MS = (10.0, 300.0)

SEQUENCE_TYPES = ("SE", "GRE")
_BLOCH_CONFIG_KEYS = frozenset({"TR", "TE", "flip_angle", "use_physics_backward"})
#: The sequence an arm without ``bloch_config`` has always rendered at.
_DEFAULT_TR_MS, _DEFAULT_TE_MS = 3000.0, 100.0


@dataclass(frozen=True)
class BlochSequence:
    """The acquisition the generator's internal simulator renders (times in ms)."""

    sequence_type: str
    tr_ms: float
    te_ms: float
    flip_deg: float

    @classmethod
    def from_config(
        cls, bloch_config: Mapping[str, Any] | None, sequence_type: str
    ) -> BlochSequence:
        """Validate ``model_kwargs.bloch_config``; ``None`` keeps the historical SE 3000/100."""
        if sequence_type not in SEQUENCE_TYPES:
            raise ValueError(f"sequence_type={sequence_type!r}; supported: {SEQUENCE_TYPES}.")
        if bloch_config is None:
            return cls(sequence_type, _DEFAULT_TR_MS, _DEFAULT_TE_MS, 90.0)
        cfg = dict(bloch_config)
        if unknown := sorted(set(cfg) - _BLOCH_CONFIG_KEYS):
            raise ValueError(
                f"bloch_config keys {unknown} have no reader; supported: "
                f"{sorted(_BLOCH_CONFIG_KEYS)}. field_strength is not read: the signal "
                "equation is field-independent at given T1/T2."
            )
        if missing := sorted({"TR", "TE"} - set(cfg)):
            raise ValueError(f"bloch_config declares a sequence but omits {missing}.")
        if cfg.get("use_physics_backward", True) is not True:
            raise ValueError(
                "bloch_config.use_physics_backward=false needs a learned decoder, which "
                "bloch_cycle_network does not have; the Bloch simulator is its decoder."
            )
        tr, te = float(cfg["TR"]), float(cfg["TE"])
        flip = float(cfg.get("flip_angle", 90.0))
        if tr <= 0.0 or te <= 0.0 or te >= tr:
            raise ValueError(f"bloch_config needs 0 < TE < TR; got TR={tr}, TE={te}.")
        if not 0.0 < flip <= 180.0:
            raise ValueError(f"bloch_config.flip_angle must be in (0, 180]; got {flip}.")
        if sequence_type == "SE" and flip != 90.0:
            raise ValueError(
                f"bloch_config.flip_angle={flip} with sequence_type=SE: the spin-echo "
                "equation assumes a 90 degree excitation; use sequence_type=GRE."
            )
        return cls(sequence_type, tr, te, flip)


def resolve_fixed_relaxation(name: str, learnable: bool, fixed_ms: float | None) -> float | None:
    """``None`` for a predicted map, else the validated constant (ms) it is held at."""
    lo, hi = T1_RANGE_MS if name == "t1" else T2_RANGE_MS
    if learnable:
        if fixed_ms is not None:
            raise ValueError(f"{name}_fixed_ms is set but {name}_learnable is true.")
        return None
    if fixed_ms is None:
        raise ValueError(f"{name}_learnable=false needs {name}_fixed_ms, the value it is held at.")
    if not lo <= float(fixed_ms) <= hi:
        raise ValueError(f"{name}_fixed_ms={fixed_ms} lies outside [{lo}, {hi}] ms.")
    return float(fixed_ms)


def maps_from_logits(
    logits: torch.Tensor, t1_fixed: float | None, t2_fixed: float | None
) -> torch.Tensor:
    """[B, 3, H, W] logits -> (PD > 0, T1 ms, T2 ms); a fixed map is a constant.

    PD is in the target's arbitrary units, so it is positive but unbounded: a PD capped
    at 1 would cap the render at ``exp(-TE/T2_max)`` below an unclamped target.
    """
    s = torch.sigmoid(logits)
    pd = F.softplus(logits[:, 0:1])
    (t1_lo, t1_hi), (t2_lo, t2_hi) = T1_RANGE_MS, T2_RANGE_MS
    t1 = t1_lo + (t1_hi - t1_lo) * s[:, 1:2] if t1_fixed is None else torch.full_like(pd, t1_fixed)
    t2 = t2_lo + (t2_hi - t2_lo) * s[:, 2:3] if t2_fixed is None else torch.full_like(pd, t2_fixed)
    return torch.cat([pd, t1, t2], dim=1)


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.body(x))


class BlochRefinementModule(nn.Module):
    """One further Bloch module: (input, previous render, map logits) -> refined logits."""

    def __init__(self, in_channels: int, feature_dim: int, num_residual_blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels + 1 + 3, feature_dim, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(feature_dim) for _ in range(num_residual_blocks)]
        )
        self.head = nn.Conv2d(feature_dim, 3, 1)

    def forward(self, x: torch.Tensor, signal: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(torch.cat([x, signal, logits], dim=1)))
        return logits + self.head(h)


def build_refinement_modules(
    in_channels: int,
    num_bloch_modules: int,
    feature_dim: int | None,
    num_residual_blocks: int | None,
) -> nn.ModuleList:
    """The ``num_bloch_modules - 1`` refinement stages after the UNet's first module."""
    if int(num_bloch_modules) < 1:
        raise ValueError(f"num_bloch_modules must be >= 1; got {num_bloch_modules}.")
    if num_bloch_modules == 1:
        if feature_dim is not None or num_residual_blocks is not None:
            raise ValueError(
                "feature_dim / num_residual_blocks size the refinement modules, which "
                "exist only when num_bloch_modules > 1."
            )
        return nn.ModuleList()
    if feature_dim is None or num_residual_blocks is None:
        raise ValueError("num_bloch_modules > 1 needs feature_dim and num_residual_blocks.")
    if feature_dim < 1 or num_residual_blocks < 0:
        raise ValueError(
            f"feature_dim must be >= 1 and num_residual_blocks >= 0; got "
            f"{feature_dim}, {num_residual_blocks}."
        )
    return nn.ModuleList(
        BlochRefinementModule(in_channels, int(feature_dim), int(num_residual_blocks))
        for _ in range(int(num_bloch_modules) - 1)
    )
