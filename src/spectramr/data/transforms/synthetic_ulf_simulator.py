"""HF→ULF forward simulator as a TorchIO transform.

Synthesises a paired ultra-low-field (ULF, e.g. 64 mT or 0.3 T)
volume from an existing high-field (HF, e.g. 3 T or HCP) volume,
so that any 3T-only dataset (HCP-1200, OASIS, IXI, …) can be used
for paired training without acquiring real ULF scans.

Composed degradations (in physically-meaningful order):

1. **B0 inhomogeneity** — applies geometric distortion along the
   readout axis (``sequence="spin_warp"``, default) or the phase-encode
   axis (``sequence="epi"``) (mirrors the existing 2D helper in
   :mod:`spectramr.data.datasets.low_field_simulation`, generalised to 3D).
2. **Maxwell concomitant distortion** — when ``field_strength_t`` is
   below ~1 T, adds the concomitant frequency offset from
   :mod:`spectramr.infrastructure.physics.maxwell_concomitant` and warps
   the volume by the implied pixel shift along the same axis.
3. **T2* blurring** — Gaussian smoothing whose σ scales with
   :math:`1/B_0` (ULF has shorter T2* and lower readout bandwidth
   in practice; longer effective sample acquisition produces blur).
4. **Rician noise** at the requested ULF SNR (in dB). At 64 mT the
   typical SNR is 8–15 dB, vs ≥30 dB at 3 T.

The transform is registered with the same ``tio.Transform`` base
class as the other transforms in :mod:`spectramr.data.transforms` so it
plugs into any TorchIO ``Compose`` pipeline. After application,
``subject[output_image]`` is the synthetic ULF and the original
HF image (renamed to ``target`` by default) is preserved.

This keeps the *existing* paired training pipeline
(``dataset_type: nifti_paired``) usable: you point
``data.data_root`` at an HF-only dataset and add this transform to
the augmentation stack — the strategy gets its (input=ULF,
target=HF) pairs without any new dataset class.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio

from spectramr.data.transforms.registry import register_transform
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import warp_volume
from spectramr.infrastructure.physics.maxwell_concomitant import (
    maxwell_concomitant_field_hz,
)

logger = logging.getLogger(__name__)


def _smooth_b0_field(
    shape: tuple[int, int, int], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Generate a realistic smooth B0 field map in normalized units."""
    D, H, W = shape
    z = torch.linspace(-1.0, 1.0, D, device=device)
    y = torch.linspace(-1.0, 1.0, H, device=device)
    x = torch.linspace(-1.0, 1.0, W, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    field = (
        torch.sin(yy * 1.5 * float(np.pi))
        * torch.cos(xx * 1.5 * float(np.pi))
        * torch.cos(zz * 0.7 * float(np.pi))
    )
    field = field + 0.05 * torch.randn_like(field)
    # 3D box-filter smoothing.
    field = (
        F.avg_pool3d(field.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2)
        .squeeze(0)
        .squeeze(0)
    )
    return field


# warp_volume displacement channels are (dz, dy, dx).
_SHIFT_CHANNEL = {"H": 1, "W": 2}


def _apply_axis_shift(
    volume: torch.Tensor,
    shift_pixels: torch.Tensor,
    axis: Literal["H", "W"] = "H",
) -> torch.Tensor:
    """Warp a (B, C, D, H, W) volume by a per-voxel pixel shift along ``axis``."""
    if volume.ndim != 5:
        raise ValueError(f"Expected (B,C,D,H,W); got {tuple(volume.shape)}")
    B, C, D, H, W = volume.shape
    disp = torch.zeros(B, 3, D, H, W, device=volume.device, dtype=volume.dtype)
    disp[:, _SHIFT_CHANNEL[axis]] = shift_pixels
    return warp_volume(volume, disp, padding_mode="border")


def _gaussian_blur_3d(volume: torch.Tensor, sigma: float, kernel_size: int = 5) -> torch.Tensor:
    """Separable 3D Gaussian blur on a (B, C, D, H, W) volume."""
    if sigma <= 0.0:
        return volume
    coords = torch.arange(kernel_size, dtype=volume.dtype, device=volume.device)
    coords = coords - (kernel_size - 1) / 2.0
    g = torch.exp(-0.5 * (coords / sigma).pow(2))
    g = g / g.sum()
    # Build a separable 3D kernel of shape (1, 1, k, k, k) by outer-product.
    g3 = g.view(-1, 1, 1) * g.view(1, -1, 1) * g.view(1, 1, -1)
    g3 = g3.view(1, 1, kernel_size, kernel_size, kernel_size)
    C = volume.shape[1]
    g3 = g3.expand(C, 1, -1, -1, -1)
    pad = kernel_size // 2
    return F.conv3d(volume, g3, padding=pad, groups=C)


def _add_rician_noise(volume: torch.Tensor, target_snr_db: float) -> torch.Tensor:
    """Add Rician noise to a magnitude volume to reach the requested SNR."""
    signal_power = volume.float().pow(2).mean().clamp_min(1e-12)
    snr_linear = 10.0 ** (float(target_snr_db) / 10.0)
    sigma = (signal_power / snr_linear).sqrt() / float(np.sqrt(2.0))
    noise_real = torch.randn_like(volume) * sigma
    noise_imag = torch.randn_like(volume) * sigma
    return torch.sqrt((volume + noise_real).pow(2) + noise_imag.pow(2))


@register_transform("simulate_ulf_from_hf", requires=("target",), modifies=("input",))
class SimulateULFFromHF(tio.Transform):
    """Synthesize a paired ULF image from an HF source image.

    Args:
        source_image: name of the HF image already in the subject
            (e.g. ``"target"`` or ``"image"``).
        output_image: name to attach the synthetic ULF under
            (default ``"input"``, matching the GeoMamba-ULF strategy).
        field_strength_t: simulated ULF B0 in Tesla (0.064 for
            Hyperfine SWOOP, 0.3 for M4Raw, 0.5 for Aspect).
        target_snr_db: desired Rician-noise SNR for the synthetic
            ULF magnitude image. Empirical Hyperfine SNR for adult
            brain ranges 8–15 dB depending on contrast.
        b0_distortion_strength: amplitude (in normalized voxel
            shifts) of the EPI-style B0 distortion. Set to 0 to
            skip B0 simulation.
        enable_maxwell: when True, adds the Maxwell concomitant
            offset from the physics module on top of the B0 shift.
            Negligible above ~1 T.
        gradient_amplitudes_mT_per_m: gradient amplitudes used by
            the Maxwell term. Typical clinical EPI 10–30 mT/m;
            ULF often 1–5 mT/m.
        voxel_size_mm: spatial voxel size (D, H, W) for the Maxwell
            geometry calculation. Defaults to (1.0, 1.0, 1.0).
        t2_star_blur_sigma: σ in voxels for the isotropic 3D
            Gaussian that mimics T2*-induced blurring at ULF.
        pe_axis: phase-encode direction, ``"H"`` or ``"W"``.
        sequence: ``"spin_warp"`` (default) displaces off-resonance along
            readout (the in-plane axis other than ``pe_axis``); ``"epi"``
            displaces along ``pe_axis``.
        keep_source: keep the original HF image in the subject under
            its original name. Almost always True (the strategy
            consumes it as ``target``).
        include / exclude / copy / kwargs: standard
            :class:`tio.Transform` arguments.
    """

    def __init__(
        self,
        source_image: str = "target",
        output_image: str = "input",
        field_strength_t: float = 0.064,
        target_snr_db: float = 12.0,
        b0_distortion_strength: float = 0.05,
        enable_maxwell: bool = True,
        gradient_amplitudes_mT_per_m: tuple[float, float, float] = (5.0, 5.0, 5.0),
        voxel_size_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
        t2_star_blur_sigma: float = 0.7,
        pe_axis: Literal["H", "W"] = "H",
        sequence: Literal["spin_warp", "epi"] = "spin_warp",
        keep_source: bool = True,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        copy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(include=include, exclude=exclude, copy=copy, **kwargs)
        if field_strength_t <= 0:
            raise ValueError(f"field_strength_t must be > 0; got {field_strength_t}")
        if pe_axis not in ("H", "W"):
            raise ValueError(f"pe_axis must be 'H' or 'W'; got {pe_axis!r}")
        if sequence not in ("spin_warp", "epi"):
            raise ValueError(f"sequence must be 'spin_warp' or 'epi'; got {sequence!r}")
        self.source_image = source_image
        self.output_image = output_image
        self.field_strength_t = float(field_strength_t)
        self.target_snr_db = float(target_snr_db)
        self.b0_distortion_strength = float(b0_distortion_strength)
        self.enable_maxwell = bool(enable_maxwell)
        self.gradient_amplitudes_mT_per_m = tuple(gradient_amplitudes_mT_per_m)
        self.voxel_size_mm = tuple(voxel_size_mm)
        self.t2_star_blur_sigma = float(t2_star_blur_sigma)
        self.pe_axis: Literal["H", "W"] = pe_axis
        self.sequence: Literal["spin_warp", "epi"] = sequence
        self.keep_source = bool(keep_source)
        # TorchIO's reproducibility contract: these land in the subject history.
        self.args_names = (
            "source_image",
            "output_image",
            "field_strength_t",
            "target_snr_db",
            "b0_distortion_strength",
            "enable_maxwell",
            "gradient_amplitudes_mT_per_m",
            "voxel_size_mm",
            "t2_star_blur_sigma",
            "pe_axis",
            "sequence",
            "keep_source",
        )

    @property
    def _shift_axis(self) -> Literal["H", "W"]:
        """In-plane axis along which off-resonance displaces signal."""
        if self.sequence == "epi":
            return self.pe_axis
        return "W" if self.pe_axis == "H" else "H"

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.source_image not in subject:
            raise KeyError(
                f"[SimulateULFFromHF] source image {self.source_image!r} missing from subject"
            )

        source = subject[self.source_image]
        # TorchIO image data is (C, W, H, D) — convert to (1, C, D, H, W) for 3D ops.
        data = source.data
        if data.ndim != 4:
            raise ValueError(
                f"[SimulateULFFromHF] expected 4D (C,W,H,D) tensor; got {tuple(data.shape)}"
            )
        C, W, H, D = data.shape
        vol = data.permute(0, 3, 2, 1).unsqueeze(0).contiguous()  # (1, C, D, H, W)

        # Normalise to [0, 1] for the noise-power calc (preserves
        # original scaling at the end).
        vmax = vol.float().abs().max().clamp_min(1e-6)
        vol_n = (vol.float() / vmax).clamp(0.0, 1.0)

        ulf = self._simulate(vol_n)

        # Rescale back to source units.
        ulf = ulf * vmax
        # (1, C, D, H, W) → (C, W, H, D) for TorchIO.
        ulf_tio = ulf.squeeze(0).permute(0, 3, 2, 1).contiguous()

        ulf_image = tio.ScalarImage(tensor=ulf_tio.cpu(), affine=source.affine)
        if self.output_image in subject and not self.keep_source:
            subject.remove_image(self.output_image)
        subject.add_image(ulf_image, self.output_image)

        if not self.keep_source:
            subject.remove_image(self.source_image)
        return subject

    def _simulate(self, vol: torch.Tensor) -> torch.Tensor:
        """Apply the (B0 → Maxwell → T2* → Rician) chain."""
        device = vol.device
        D, H, W = vol.shape[-3:]

        axis = self._shift_axis
        # 1. B0 inhomogeneity (shift along readout for spin-warp, PE for EPI).
        if self.b0_distortion_strength > 0:
            b0_norm = _smooth_b0_field((D, H, W), device=device)
            shift_pixels = b0_norm * self.b0_distortion_strength * (H if axis == "H" else W)
            vol = _apply_axis_shift(vol, shift_pixels, axis=axis)

        # 2. Maxwell concomitant — only meaningful below ~1 T.
        if self.enable_maxwell and self.field_strength_t < 1.0:
            mx_hz = maxwell_concomitant_field_hz(
                spatial_shape=(D, H, W),
                voxel_size_mm=self.voxel_size_mm,
                gradient_amplitudes_mT_per_m=self.gradient_amplitudes_mT_per_m,
                field_strength_t=self.field_strength_t,
                device=device,
                dtype=vol.dtype,
            )
            # Normalize so the max shift maps to ~1 pixel (proxy for
            # short EPI ESP at ULF). The strategy can recover the
            # absolute scale via ``training.geomamba_ulf.unwarp``.
            denom = mx_hz.abs().max().clamp_min(1e-6)
            mx_shift = (mx_hz / denom) * 0.3
            vol = _apply_axis_shift(vol, mx_shift, axis=axis)

        # 3. T2* blurring.
        vol = _gaussian_blur_3d(vol, sigma=self.t2_star_blur_sigma)

        # 4. Rician noise at the requested SNR.
        vol = _add_rician_noise(vol, target_snr_db=self.target_snr_db)

        return vol.clamp(0.0, 1.0)


__all__ = ["SimulateULFFromHF"]
