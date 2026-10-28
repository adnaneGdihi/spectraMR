"""Unit tests for HW-Mamba: a fixed Haar decomposition with Mamba-predicted detail corrections."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.hw_mamba import HWMamba, haar_dwt, haar_idwt
from spectramr.models.registry import is_strict_kwargs

BATCH, H, W = 2, 8, 8
D_MODEL, LAYERS, D_STATE = 16, 1, 4


def _model(**kw) -> HWMamba:
    torch.manual_seed(0)
    return HWMamba(1, 1, d_model=D_MODEL, num_layers=LAYERS, d_state=D_STATE, **kw)


def _randomise_corrections(model: HWMamba) -> None:
    with torch.no_grad():
        for corr in model.detail_corrections:
            corr.weight.normal_(std=0.1)
            corr.bias.normal_(std=0.1)


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_haar_round_trip_is_exact(levels):
    x = torch.randn(2, 4, 64)
    approx, details = haar_dwt(x, levels)
    torch.testing.assert_close(haar_idwt(approx, details), x)


def test_band_count_and_lengths_follow_wavelet_levels():
    approx, details = haar_dwt(torch.randn(1, 2, 64), 3)
    assert len(details) == 3
    assert [d.shape[-1] for d in details] == [32, 16, 8]
    assert approx.shape[-1] == 8


def test_zero_corrections_reconstruct_the_ulf_features_exactly():
    model = _model(wavelet_levels=3)
    seq = torch.randn(BATCH, D_MODEL, H * W)
    torch.testing.assert_close(model._wavelet_reconstruct(seq), seq, atol=1e-5, rtol=1e-5)


def test_approximation_band_is_the_ulf_band_exactly():
    model = _model(wavelet_levels=3)
    _randomise_corrections(model)
    seq = torch.randn(BATCH, D_MODEL, H * W)
    recon = model._wavelet_reconstruct(seq)
    torch.testing.assert_close(haar_dwt(recon, 3)[0], haar_dwt(seq, 3)[0], atol=1e-5, rtol=1e-5)


def test_corrections_change_the_detail_bands():
    model = _model(wavelet_levels=3)
    _randomise_corrections(model)
    seq = torch.randn(BATCH, D_MODEL, H * W)
    recon = model._wavelet_reconstruct(seq)
    assert not torch.allclose(recon, seq)


def test_wavelet_levels_sets_the_number_of_correction_levels():
    assert len(_model(wavelet_levels=2).detail_corrections) == 2
    assert len(_model().detail_corrections) == 3  # default


def test_length_not_divisible_by_two_to_the_levels_raises():
    model = _model(wavelet_levels=7)
    with pytest.raises(ValueError, match="divisible"):
        model._wavelet_reconstruct(torch.randn(1, D_MODEL, H * W))


def test_forward_shape_and_gradients_reach_the_detail_predictor():
    model = _model(wavelet_levels=3)
    _randomise_corrections(model)
    x = torch.randn(BATCH, 1, H, W)
    out = model(x)
    assert out.shape == (BATCH, 1, H, W)
    out.sum().backward()
    assert model.detail_predictor.layers[0].in_proj.weight.grad is not None


def test_pixel_residual_control_has_no_wavelet_path():
    model = _model(wavelet_domain=False)
    assert not hasattr(model, "detail_corrections")
    assert model(torch.randn(BATCH, 1, H, W)).shape == (BATCH, 1, H, W)


def test_wavelet_levels_with_the_pixel_control_raises():
    with pytest.raises(ValueError, match="no effect"):
        _model(wavelet_domain=False, wavelet_levels=3)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_wavelet_levels_raise(bad):
    with pytest.raises(ValueError, match="wavelet_levels"):
        _model(wavelet_levels=bad)


def test_hw_mamba_is_registered_strict():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    assert is_strict_kwargs("hw_mamba")
