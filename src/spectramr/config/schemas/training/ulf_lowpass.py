"""Config for the ULF forward operator ``A`` shared by the ULF restoration strategies.

``A x = ifft2c(H . fft2c(x))`` with a Gaussian OTF ``H`` models the 64 mT effective
resolution on the 3 T grid. It is the data term of ``ulf_map``, the DPS likelihood of
``ulf_dps``, the re-degradation loss of ``ulf_redegrad_tta`` and the ``field_lowpass``
operator of ``generative_refiner``; this block is the one owner of its cutoff.
Read via ``config.training.ulf_lowpass``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from spectramr.config.schemas.strictness import StrictSchema


class UlfLowpassConfig(StrictSchema):
    """Cutoff of the ULF low-pass forward operator."""

    cutoff_mode: Literal["constant", "voxel_ratio"] = Field(
        description=(
            "'constant': one isotropic cutoff (blur_cutoff) for every sample. "
            "'voxel_ratio': a per-contrast, per-axis cutoff derived from the stored "
            "grid spacing and the native ULF voxel size, selected per sample by "
            "contrast_id (needs data.multi_contrast). No default: the arm declares it."
        ),
    )
    blur_cutoff: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Gaussian OTF std in units of the grid Nyquist frequency ('constant' only). "
            "Smaller => more blur."
        ),
    )
    grid_voxel_mm: dict[str, tuple[float, float]] | None = Field(
        default=None,
        description=(
            "Per-contrast in-plane spacing (mm) of the STORED grid, along the image "
            "axes (H, W) of a 2-D slice ('voxel_ratio' only). From the manifest's "
            "hf_voxel_mm: both volumes are resampled onto the 3 T grid."
        ),
    )
    native_voxel_mm: dict[str, tuple[float, float]] | None = Field(
        default=None,
        description=(
            "Per-contrast in-plane voxel size (mm) the ULF scanner resolved, along the "
            "same (H, W) axes ('voxel_ratio' only). From the manifest's ulf_voxel_mm."
        ),
    )

    @model_validator(mode="after")
    def _knobs_match_mode(self) -> UlfLowpassConfig:
        voxel = (self.grid_voxel_mm, self.native_voxel_mm)
        if self.cutoff_mode == "constant":
            if self.blur_cutoff is None:
                raise ValueError("cutoff_mode='constant' requires blur_cutoff.")
            if any(v is not None for v in voxel):
                raise ValueError(
                    "grid_voxel_mm / native_voxel_mm are not read under "
                    "cutoff_mode='constant'; remove them."
                )
            return self
        if self.blur_cutoff is not None:
            raise ValueError(
                "blur_cutoff is not read under cutoff_mode='voxel_ratio' (the cutoff is "
                "derived from the voxel sizes); remove it."
            )
        if self.grid_voxel_mm is None or self.native_voxel_mm is None:
            raise ValueError(
                "cutoff_mode='voxel_ratio' requires grid_voxel_mm and native_voxel_mm."
            )
        if set(self.grid_voxel_mm) != set(self.native_voxel_mm):
            raise ValueError(
                f"grid_voxel_mm {sorted(self.grid_voxel_mm)} and native_voxel_mm "
                f"{sorted(self.native_voxel_mm)} must name the same contrasts."
            )
        for name, table in (
            ("grid_voxel_mm", self.grid_voxel_mm),
            ("native_voxel_mm", self.native_voxel_mm),
        ):
            for contrast, mm in table.items():
                if any(v <= 0.0 for v in mm):
                    raise ValueError(f"{name}[{contrast!r}]={mm} must be positive.")
        return self
