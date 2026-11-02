r"""DiffSliceSim: a differentiable through-plane slice-profile renderer.

The network predicts a latent tissue volume of ``D`` sub-slices around the imaged slice. Each
sub-slice voxel holds a proton density :math:`\rho_z \ge 0` and partial-volume fractions
:math:`f_{z,m}` (softmax over ``material_dim`` materials), and each material carries a learned
signal signature :math:`s_m \in \mathbb{R}^{C}`. The rendered image is

.. math::

    I(x, y) = \sum_z p(z)\, \rho_z(x, y) \sum_m f_{z,m}(x, y)\, s_m,

where :math:`p` is the slice-select profile sampled at a fixed physical sub-slice spacing.
``partial_volume_effect: false`` replaces :math:`p` with a unit impulse at the centre sub-slice,
keeping the parameter count. On single-slice data this is not an ablation: each free
:math:`\rho_z` absorbs any positive :math:`p(z)`, so the profile only becomes observable when
several slices with overlapping profiles are rendered from one shared volume.

The ``hamming_windowed_sinc`` profile is the small-tip excitation profile of a Hamming-windowed
sinc RF pulse (time-bandwidth product :data:`RF_TIME_BANDWIDTH`), i.e. the magnitude of the pulse's
Fourier transform, scaled so its FWHM equals ``slice_thickness_mm`` [Pauly 1991].

Reference:
    J. Pauly, P. Le Roux, D. Nishimura, A. Macovski, "Parameter relations for the Shinnar-Le Roux
    selective excitation pulse design algorithm," IEEE TMI 10(1), 1991.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.block_registry import register_block
from spectramr.models.blocks.interfaces import IBlockStrategy

__all__ = [
    "RENDER_ENGINES",
    "RF_TIME_BANDWIDTH",
    "SLICE_PROFILES",
    "DiffSliceSim",
    "build_renderer",
    "slice_profile_weights",
]

# Time-bandwidth product of the excitation pulse; a standard clinical windowed-sinc value.
RF_TIME_BANDWIDTH = 4.0
_RF_SAMPLES = 512


def _hamming_windowed_sinc(z_mm: torch.Tensor, thickness_mm: float) -> torch.Tensor:
    tau = (torch.arange(_RF_SAMPLES, dtype=torch.float64) + 0.5) / _RF_SAMPLES - 0.5
    pulse = torch.sinc(RF_TIME_BANDWIDTH * tau) * (0.54 + 0.46 * torch.cos(2 * math.pi * tau))
    freq = z_mm.to(torch.float64) * RF_TIME_BANDWIDTH / thickness_mm
    # The pulse is even, so its Fourier transform is the cosine sum (a small DFT matmul).
    return (torch.cos(2 * math.pi * freq[:, None] * tau[None, :]) @ pulse).abs()


def _rect(z_mm: torch.Tensor, thickness_mm: float) -> torch.Tensor:
    return (z_mm.abs() <= thickness_mm / 2).to(torch.float64)


SLICE_PROFILES: dict[str, Callable[[torch.Tensor, float], torch.Tensor]] = {
    "hamming_windowed_sinc": _hamming_windowed_sinc,
    "rect": _rect,
}


def slice_profile_weights(
    profile: str, thickness_mm: float, spacing_mm: float, partial_volume_effect: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(z_mm, weights)`` on sub-slices ``k * spacing_mm``, ``|z| <= thickness_mm``.

    The weights are non-negative and sum to one; without partial volume they are an impulse at
    ``z = 0``.
    """
    half = math.ceil(thickness_mm / spacing_mm)
    z_mm = torch.arange(-half, half + 1, dtype=torch.float64) * spacing_mm
    if partial_volume_effect:
        weights = SLICE_PROFILES[profile](z_mm, thickness_mm)
    else:
        weights = (z_mm == 0).to(torch.float64)
    return z_mm.float(), (weights / weights.sum()).float()


@register_block("diff_slice_sim")
class DiffSliceSim(IBlockStrategy):
    """Render ``(B, F, H, W)`` features to a ``(B, C, H, W)`` image through a slice profile."""

    def __init__(
        self,
        feature_dim: int,
        out_channels: int,
        material_dim: int,
        slice_profile: str = "hamming_windowed_sinc",
        slice_thickness_mm: float = 3.0,
        partial_volume_effect: bool = True,
        subslice_spacing_mm: float = 1.0,
    ) -> None:
        super().__init__()
        if slice_profile not in SLICE_PROFILES:
            raise ValueError(
                f"rendering.physics.slice_profile {slice_profile!r} is unknown; "
                f"choose one of {sorted(SLICE_PROFILES)}."
            )
        for key, value in (
            ("slice_thickness_mm", slice_thickness_mm),
            ("subslice_spacing_mm", subslice_spacing_mm),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(
                    f"rendering.physics.{key} must be a positive number, got {value!r}."
                )
        if not isinstance(partial_volume_effect, bool):
            raise ValueError(
                "rendering.physics.partial_volume_effect must be a bool, "
                f"got {partial_volume_effect!r}."
            )
        if isinstance(material_dim, bool) or not isinstance(material_dim, int) or material_dim < 1:
            raise ValueError(f"material_dim must be a positive int, got {material_dim!r}.")

        self.slice_profile = slice_profile
        self.slice_thickness_mm = float(slice_thickness_mm)
        self.subslice_spacing_mm = float(subslice_spacing_mm)
        self.partial_volume_effect = partial_volume_effect
        self.material_dim = material_dim
        z_mm, weights = slice_profile_weights(
            slice_profile, self.slice_thickness_mm, self.subslice_spacing_mm, partial_volume_effect
        )
        self.register_buffer("subslice_z_mm", z_mm, persistent=False)
        self.register_buffer("profile", weights, persistent=False)
        self.num_subslices = int(weights.numel())
        # Per sub-slice: one proton-density channel plus material_dim fraction logits.
        self.volume_head = nn.Conv2d(feature_dim, self.num_subslices * (material_dim + 1), 1)
        self.material_signal = nn.Parameter(torch.rand(material_dim, out_channels))

    @property
    def record(self) -> dict[str, Any]:
        """The resolved rendering physics, for provenance."""
        return {
            "engine": "DiffSliceSim",
            "slice_profile": self.slice_profile,
            "slice_thickness_mm": self.slice_thickness_mm,
            "partial_volume_effect": self.partial_volume_effect,
            "subslice_spacing_mm": self.subslice_spacing_mm,
            "num_subslices": self.num_subslices,
            "rf_time_bandwidth": RF_TIME_BANDWIDTH,
            "material_dim": self.material_dim,
        }

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Render features ``x`` ``(B, F, H, W)`` to ``(B, C, H, W)``."""
        b, _, h, w = x.shape
        vol = self.volume_head(x).view(b, self.num_subslices, self.material_dim + 1, h, w)
        weight = F.softplus(vol[:, :, 0]) * self.profile.to(vol.dtype).view(1, -1, 1, 1)
        fractions = torch.softmax(vol[:, :, 1:], dim=2)
        return torch.einsum(
            "bzmhw,bzhw,mc->bchw", fractions, weight, self.material_signal.to(vol.dtype)
        )


RENDER_ENGINES: dict[str, type[DiffSliceSim]] = {"DiffSliceSim": DiffSliceSim}
_RENDERING_KEYS = frozenset({"engine", "physics"})
_PHYSICS_KEYS = frozenset(
    {"slice_profile", "slice_thickness_mm", "partial_volume_effect", "subslice_spacing_mm"}
)


def build_renderer(
    rendering: Mapping[str, Any], *, feature_dim: int, out_channels: int, material_dim: int
) -> DiffSliceSim:
    """Validate a ``model_kwargs.rendering`` block and build its engine; unknown keys raise."""
    if not isinstance(rendering, Mapping):
        raise ValueError(f"rendering must be a mapping, got {type(rendering).__name__}.")
    unknown = set(rendering) - _RENDERING_KEYS
    if unknown:
        raise ValueError(
            f"rendering has unknown keys {sorted(unknown)}; valid: {sorted(_RENDERING_KEYS)}."
        )
    engine = rendering.get("engine", "DiffSliceSim")
    if engine not in RENDER_ENGINES:
        raise ValueError(
            f"rendering.engine {engine!r} is unknown; choose one of {sorted(RENDER_ENGINES)}."
        )
    physics = rendering.get("physics") or {}
    if not isinstance(physics, Mapping):
        raise ValueError(f"rendering.physics must be a mapping, got {type(physics).__name__}.")
    unknown = set(physics) - _PHYSICS_KEYS
    if unknown:
        raise ValueError(
            f"rendering.physics has unknown keys {sorted(unknown)}; valid: {sorted(_PHYSICS_KEYS)}."
        )
    return RENDER_ENGINES[engine](
        feature_dim=feature_dim, out_channels=out_channels, material_dim=material_dim, **physics
    )
