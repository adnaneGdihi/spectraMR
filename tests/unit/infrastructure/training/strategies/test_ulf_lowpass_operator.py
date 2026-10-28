"""Tests for the ULF low-pass forward operator resolved from training.ulf_lowpass."""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.config.schemas.training.ulf_lowpass import UlfLowpassConfig
from spectramr.infrastructure.training.strategies.ulf_lowpass_operator import (
    FWHM_CUTOFF_FACTOR,
    UlfLowpassOperator,
    build_ulf_lowpass_operator,
)
from spectramr.infrastructure.training.strategies.ulf_map_strategy import ulf_degrade

_MAP = {"T1w": 0, "T2w": 1, "FLAIR": 2}


def _config(
    ulf: UlfLowpassConfig | None, contrast_map: dict | None = _MAP
) -> types.SimpleNamespace:
    mc = types.SimpleNamespace(enabled=contrast_map is not None, contrast_map=contrast_map)
    return types.SimpleNamespace(
        training=types.SimpleNamespace(ulf_lowpass=ulf),
        data=types.SimpleNamespace(multi_contrast=mc),
    )


def _voxel_ratio() -> UlfLowpassConfig:
    return UlfLowpassConfig(
        cutoff_mode="voxel_ratio",
        grid_voxel_mm={"T1w": (0.49, 0.49), "T2w": (0.22, 0.22), "FLAIR": (0.43, 0.43)},
        native_voxel_mm={"T1w": (1.6, 1.6), "T2w": (1.6, 1.6), "FLAIR": (1.7, 1.7)},
    )


@pytest.mark.unit
def test_fwhm_factor_matches_the_gaussian_psf_identity() -> None:
    # FWHM = 2 sqrt(2 ln 2) sigma_x and sigma_f = 1 / (2 pi sigma_x), in Nyquist units 2 sigma_f.
    assert pytest.approx(0.7496, abs=1e-4) == FWHM_CUTOFF_FACTOR


@pytest.mark.unit
def test_constant_mode_equals_ulf_degrade() -> None:
    op = build_ulf_lowpass_operator(
        _config(UlfLowpassConfig(cutoff_mode="constant", blur_cutoff=0.3))
    )
    x = torch.rand(2, 1, 16, 16)
    torch.testing.assert_close(op.apply(x), ulf_degrade(x, 0.3))


@pytest.mark.unit
def test_voxel_ratio_derives_per_contrast_cutoffs() -> None:
    op = build_ulf_lowpass_operator(_config(_voxel_ratio()))
    assert op.axis_cutoffs[0] == pytest.approx((FWHM_CUTOFF_FACTOR * 0.49 / 1.6,) * 2)
    assert op.axis_cutoffs[1] == pytest.approx((FWHM_CUTOFF_FACTOR * 0.22 / 1.6,) * 2)


@pytest.mark.unit
def test_contrast_id_selects_the_otf_per_sample() -> None:
    op = build_ulf_lowpass_operator(_config(_voxel_ratio()))
    x = torch.rand(1, 1, 16, 16).expand(3, 1, 16, 16).contiguous()
    out = op.apply(x, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(out[0:1], op.apply(x[0:1], torch.tensor([0])))
    assert not torch.allclose(out[0], out[1])
    # T2w (finest grid, smallest ratio) is blurred hardest.
    hf = lambda im: (im - im.mean()).abs().mean()  # noqa: E731
    assert float(hf(out[1])) < float(hf(out[0]))


@pytest.mark.unit
def test_anisotropic_cutoff_blurs_one_axis_more() -> None:
    op = UlfLowpassOperator(axis_cutoffs={0: (0.1, 2.0)})
    x = torch.zeros(1, 1, 16, 16)
    x[..., 8, 8] = 1.0
    psf = op.apply(x, torch.tensor([0]))[0, 0]
    # Spread along H (small H cutoff), near-impulse along W.
    assert float(psf[7, 8] / psf[8, 8]) > 0.5 > float(psf[8, 7] / psf[8, 8])


@pytest.mark.unit
def test_voxel_ratio_without_contrast_id_raises() -> None:
    op = build_ulf_lowpass_operator(_config(_voxel_ratio()))
    with pytest.raises(ValueError, match="contrast_id"):
        op.apply(torch.rand(1, 1, 8, 8))


@pytest.mark.unit
def test_build_raises_on_missing_block_or_contrast_map_mismatch() -> None:
    with pytest.raises(ValueError, match=r"training\.ulf_lowpass"):
        build_ulf_lowpass_operator(_config(None))
    with pytest.raises(ValueError, match="multi_contrast"):
        build_ulf_lowpass_operator(_config(_voxel_ratio(), contrast_map=None))
    with pytest.raises(ValueError, match="missing"):
        build_ulf_lowpass_operator(_config(_voxel_ratio(), {**_MAP, "PD": 3}))
