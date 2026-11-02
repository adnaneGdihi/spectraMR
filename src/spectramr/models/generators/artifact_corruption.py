"""Training-time artifact corruption for ``artifact_robust_generator``.

The model receives undersampled multi-coil k-space, real/imag interleaved
``[B, 2C, H, W]``. Each declared artifact is a registered degradation
(``infrastructure.physics.degradations``), applied to the coil images
``ifft2c(y)`` and brought back with ``fft2c``. The result is re-masked to the
input's acquired support, so a corruption never fills lines the loader left
empty. Draws come from a CPU generator seeded from the run seed, so a step's
corruption is reproducible and choosing which samples to corrupt needs no
device sync.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from spectramr.core.metrics.coil_manifold import as_complex_coils, as_interleaved
from spectramr.infrastructure.physics.degradations.registry import apply_degradation
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

#: Registered degradations that keep complex coil images complex. ``rician`` is
#: left out: it returns a magnitude and would strip the phase the model reads.
ADVERTISED_ARTIFACTS: tuple[str, ...] = (
    "complex_gaussian",
    "random_motion",
    "rigid_motion",
    "gibbs",
    "bias",
    "partial_fourier",
    "spike",
    "nyquist_ghost",
)

_SPEC_KEYS = frozenset({"artifacts", "detector_loss_weight", "eval_severity"})
_ARTIFACT_KEYS = frozenset({"probability", "severity"})


def _unit_float(value: Any, where: str, *, open_low: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number in [0, 1], got {value!r}.")
    out = float(value)
    if not math.isfinite(out) or out < 0.0 or out > 1.0 or (open_low and out == 0.0):
        bound = "(0, 1]" if open_low else "[0, 1]"
        raise ValueError(f"{where} must be in {bound}, got {out!r}.")
    return out


@dataclass(frozen=True)
class ArtifactDraw:
    """How often one artifact fires and the severity range it is drawn from."""

    name: str
    probability: float
    severity: tuple[float, float]


@dataclass(frozen=True)
class ArtifactAugmentationSpec:
    """``model.model_kwargs.artifact_augmentation``, validated."""

    artifacts: tuple[ArtifactDraw, ...]
    detector_loss_weight: float
    eval_severity: float | None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.artifacts)

    def record(self) -> dict[str, Any]:
        """The provenance record: the knob as the model applies it."""
        return {
            "artifacts": {
                a.name: {"probability": a.probability, "severity": list(a.severity)}
                for a in self.artifacts
            },
            "detector_loss_weight": self.detector_loss_weight,
            "eval_severity": self.eval_severity,
            "domain": "coil images (ifft2c of the input k-space), re-masked to the acquired support",
        }


def parse_artifact_augmentation(block: Any) -> ArtifactAugmentationSpec:
    """Validate the block; raises on a missing, unknown or out-of-range key."""
    where = "model.model_kwargs.artifact_augmentation"
    if not isinstance(block, Mapping):
        raise ValueError(
            f"{where} is required and must be a mapping with keys {sorted(_SPEC_KEYS)}; "
            f"got {block!r}. Without it the model would train on clean inputs only."
        )
    if set(block) != _SPEC_KEYS:
        raise ValueError(
            f"{where} must declare exactly {sorted(_SPEC_KEYS)}; "
            f"missing {sorted(_SPEC_KEYS - set(block))}, unknown {sorted(set(block) - _SPEC_KEYS)}."
        )
    raw = block["artifacts"]
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{where}.artifacts must be a non-empty mapping name -> draw.")
    draws = []
    for name, draw in raw.items():
        if name not in ADVERTISED_ARTIFACTS:
            raise ValueError(
                f"{where}.artifacts: unknown artifact {name!r}; valid: {list(ADVERTISED_ARTIFACTS)}."
            )
        if not isinstance(draw, Mapping) or set(draw) != _ARTIFACT_KEYS:
            raise ValueError(
                f"{where}.artifacts.{name} must declare exactly {sorted(_ARTIFACT_KEYS)}."
            )
        prob = _unit_float(
            draw["probability"], f"{where}.artifacts.{name}.probability", open_low=True
        )
        sev = draw["severity"]
        if not isinstance(sev, (list, tuple)) or len(sev) != 2:
            raise ValueError(f"{where}.artifacts.{name}.severity must be [low, high].")
        low = _unit_float(sev[0], f"{where}.artifacts.{name}.severity[0]")
        high = _unit_float(sev[1], f"{where}.artifacts.{name}.severity[1]")
        if low > high:
            raise ValueError(f"{where}.artifacts.{name}.severity has low {low} > high {high}.")
        draws.append(ArtifactDraw(str(name), prob, (low, high)))
    weight = block["detector_loss_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not weight >= 0:
        raise ValueError(f"{where}.detector_loss_weight must be a number >= 0, got {weight!r}.")
    eval_sev = block["eval_severity"]
    eval_severity = None if eval_sev is None else _unit_float(eval_sev, f"{where}.eval_severity")
    return ArtifactAugmentationSpec(tuple(draws), float(weight), eval_severity)


def fold_depth(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
    if x.ndim == 4:
        return x, None
    if x.ndim != 5:
        raise ValueError(f"expected [B, C, H, W] or [B, C, H, W, D], got {tuple(x.shape)}.")
    b, c, h, w, d = x.shape
    return x.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w), (b, d)


def unfold_depth(x: torch.Tensor, layout: tuple[int, int] | None) -> torch.Tensor:
    if layout is None:
        return x
    b, d = layout
    return x.reshape(b, d, *x.shape[1:]).permute(0, 2, 3, 4, 1)


class ArtifactCorruptor:
    """Applies the declared artifacts to interleaved coil k-space."""

    def __init__(self, spec: ArtifactAugmentationSpec, seed: int | None = None) -> None:
        self.spec = spec
        self._rng = torch.Generator().manual_seed(
            int(torch.initial_seed() if seed is None else seed) % (2**63)
        )
        self._forced: tuple[str, float, int] | None = None

    @contextlib.contextmanager
    def forced(self, name: str, severity: float, seed: int) -> Iterator[None]:
        """Apply only ``name`` at ``severity`` to every sample (the robustness eval)."""
        if name not in self.spec.names:
            raise ValueError(
                f"artifact {name!r} is not declared; declared: {list(self.spec.names)}."
            )
        self._forced = (name, float(severity), int(seed))
        try:
            yield
        finally:
            self._forced = None

    @property
    def active(self) -> bool:
        return self._forced is not None

    def coil_images(self, x: torch.Tensor) -> torch.Tensor:
        """Complex coil images ``[B, C, H, W]`` of a 4-D interleaved k-space input."""
        return ifft2c(as_complex_coils(x, name="artifact_robust_generator input"))

    def corrupt(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(k-space, coil images, labels)``; labels are ``[B, K]`` presence in ``{0, 1}``."""
        images = self.coil_images(x)
        batch = x.shape[0]
        labels = torch.zeros(batch, len(self.spec.artifacts))
        for index, draw in enumerate(self.spec.artifacts):
            if self._forced is not None:
                if draw.name != self._forced[0]:
                    continue
                selected = torch.ones(batch, dtype=torch.bool)
                theta, seed = self._forced[1], self._forced[2]
            else:
                selected = torch.rand(batch, generator=self._rng) < draw.probability
                if not bool(selected.any()):  # CPU tensor: no device sync
                    continue
                low, high = draw.severity
                theta = low + (high - low) * float(torch.rand((), generator=self._rng))
                seed = int(torch.randint(0, 2**31 - 1, (), generator=self._rng))
            degraded = apply_degradation(draw.name, images, theta, seed)
            if not torch.is_complex(degraded):
                raise TypeError(
                    f"degradation {draw.name!r} returned a real tensor for complex coil "
                    "images; it would strip the phase the model reads."
                )
            mask = selected.to(images.device, non_blocking=True).view(batch, 1, 1, 1)
            images = torch.where(mask, degraded.to(images.dtype), images)
            labels[:, index] = selected.float()
        measured = as_complex_coils(x, name="artifact_robust_generator input")
        kspace = fft2c(images) * (measured != 0)
        # The detector must see only the measured support, as on a real acquisition.
        images = ifft2c(kspace)
        return as_interleaved(kspace).to(x.dtype), images, labels.to(x.device, non_blocking=True)


__all__ = [
    "ADVERTISED_ARTIFACTS",
    "ArtifactAugmentationSpec",
    "ArtifactCorruptor",
    "ArtifactDraw",
    "fold_depth",
    "parse_artifact_augmentation",
    "unfold_depth",
]
