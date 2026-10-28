"""The ULF low-pass forward operator ``A``, resolved once from ``training.ulf_lowpass``.

``A x = ifft2c(H . fft2c(x))``. Under ``cutoff_mode: voxel_ratio`` the Gaussian OTF is
anisotropic and per contrast: a Gaussian PSF whose FWHM equals the native ULF voxel,
sampled on the stored grid, has an OTF std of ``k * grid_mm / native_mm`` in units of
the grid Nyquist frequency, with ``k = 2 sqrt(2 ln 2) / pi ~= 0.75``. Each sample's OTF
is gathered from a per-contrast table by ``contrast_id``, with no host sync.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

from .ulf_map_strategy import _fft2c_dc_index, _lowpass_otf

#: Gaussian PSF with FWHM = native voxel -> OTF std per grid Nyquist, per unit voxel ratio.
FWHM_CUTOFF_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0)) / math.pi


def _anisotropic_otf(
    h: int, w: int, cut_h: float, cut_w: float, device: torch.device
) -> torch.Tensor:
    """Centred real Gaussian OTF ``[1, 1, H, W]`` with one std per image axis."""
    dc_y, dc_x = _fft2c_dc_index(h, w, device)
    yy = (torch.arange(h, device=device, dtype=torch.float32) - dc_y) / max(h / 2.0, 1.0)
    xx = (torch.arange(w, device=device, dtype=torch.float32) - dc_x) / max(w / 2.0, 1.0)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    arg = gy * gy / (2.0 * cut_h * cut_h) + gx * gx / (2.0 * cut_w * cut_w)
    return torch.exp(-arg).view(1, 1, h, w)


class UlfLowpassOperator:
    """``A`` with a constant cutoff, or a per-contrast cutoff gathered by ``contrast_id``."""

    def __init__(
        self,
        *,
        constant: float | None = None,
        axis_cutoffs: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        if (constant is None) == (axis_cutoffs is None):
            raise ValueError("UlfLowpassOperator needs exactly one of constant / axis_cutoffs.")
        self.constant = constant
        self.axis_cutoffs = dict(axis_cutoffs or {})
        self._tables: dict[tuple, torch.Tensor] = {}

    @classmethod
    def constant_cutoff(cls, cutoff: float) -> UlfLowpassOperator:
        return cls(constant=float(cutoff))

    @property
    def per_contrast(self) -> bool:
        return self.constant is None

    def describe(self) -> dict[str, Any]:
        """The applied cutoffs, for the run log."""
        if not self.per_contrast:
            return {"cutoff_mode": "constant", "blur_cutoff": self.constant}
        return {"cutoff_mode": "voxel_ratio", "axis_cutoffs": self.axis_cutoffs}

    def otf(
        self, h: int, w: int, device: torch.device, contrast_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``[1, 1, H, W]`` (constant) or ``[B, 1, H, W]`` (per contrast)."""
        if not self.per_contrast:
            return _lowpass_otf(h, w, float(self.constant), device, torch.float32)
        if contrast_id is None:
            raise ValueError(
                "ulf_lowpass.cutoff_mode='voxel_ratio' needs a per-sample contrast_id; the "
                "batch carries none (enable data.multi_contrast)."
            )
        key = (h, w, str(device))
        table = self._tables.get(key)
        if table is None:
            n = max(self.axis_cutoffs) + 1
            # NaN rows poison a contrast_id the map does not name instead of passing it through.
            table = torch.full((n, 1, h, w), float("nan"), device=device)
            for idx, (cut_h, cut_w) in self.axis_cutoffs.items():
                table[idx] = _anisotropic_otf(h, w, cut_h, cut_w, device)[0]
            self._tables[key] = table
        cid = torch.as_tensor(contrast_id, device=device).reshape(-1).long()
        return table.index_select(0, cid)

    def apply(self, x: torch.Tensor, contrast_id: torch.Tensor | None = None) -> torch.Tensor:
        """``A x`` (real output)."""
        h_otf = self.otf(x.shape[-2], x.shape[-1], x.device, contrast_id)
        return ifft2c(h_otf * fft2c(x)).real


def build_ulf_lowpass_operator(config: Any) -> UlfLowpassOperator:
    """Resolve ``training.ulf_lowpass``; raise when absent or inconsistent with the data."""
    cfg = getattr(getattr(config, "training", None), "ulf_lowpass", None)
    if cfg is None:
        raise ValueError(
            "This ULF strategy applies the low-pass forward operator A and requires a "
            "training.ulf_lowpass block (cutoff_mode, and blur_cutoff or the voxel "
            "sizes); none is declared."
        )
    if cfg.cutoff_mode == "constant":
        return UlfLowpassOperator.constant_cutoff(cfg.blur_cutoff)
    mc = getattr(getattr(config, "data", None), "multi_contrast", None)
    if mc is None or not getattr(mc, "enabled", False) or not getattr(mc, "contrast_map", None):
        raise ValueError(
            "ulf_lowpass.cutoff_mode='voxel_ratio' selects the cutoff by contrast_id, "
            "which the dataset stamps only when data.multi_contrast is enabled with a "
            "contrast_map."
        )
    contrast_map = dict(mc.contrast_map)
    missing = sorted(set(contrast_map) - set(cfg.grid_voxel_mm))
    unread = sorted(set(cfg.grid_voxel_mm) - set(contrast_map))
    if missing or unread:
        raise ValueError(
            f"ulf_lowpass voxel sizes must cover exactly the contrast_map "
            f"{sorted(contrast_map)}: missing {missing}, not in the map {unread}."
        )
    axis_cutoffs = {
        int(idx): tuple(
            FWHM_CUTOFF_FACTOR * g / n
            for g, n in zip(cfg.grid_voxel_mm[name], cfg.native_voxel_mm[name], strict=True)
        )
        for name, idx in contrast_map.items()
    }
    return UlfLowpassOperator(axis_cutoffs=axis_cutoffs)  # type: ignore[arg-type]
