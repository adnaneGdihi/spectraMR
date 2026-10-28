"""Unit tests for INR-Mamba: a Mamba context queried at continuous coordinates."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.embeddings import CoordinatePositionalEncoding
from spectramr.models.generators.inr_mamba import INRMamba, pixel_centre_coords
from spectramr.models.registry import is_strict_kwargs

BATCH, H, W = 2, 8, 8
D_MODEL, LAYERS, D_STATE = 16, 1, 4


def _model(**kw) -> INRMamba:
    torch.manual_seed(0)
    return INRMamba(1, 1, d_model=D_MODEL, num_layers=LAYERS, d_state=D_STATE, num_freqs=6, **kw)


@pytest.fixture
def image() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(BATCH, 1, H, W)


@pytest.mark.parametrize("interp", ["hilbert_t", "spatial"])
def test_forward_is_the_query_at_pixel_centres(image, interp):
    model = _model(interp=interp).eval()
    centres = pixel_centre_coords(H, W, image.device).unsqueeze(0).expand(BATCH, -1, -1)
    via_query = model.query(image, centres).reshape(BATCH, 1, H, W)
    torch.testing.assert_close(model(image), via_query)


@pytest.mark.parametrize("interp", ["hilbert_t", "spatial"])
def test_off_grid_query_is_finite_and_not_nearest_neighbour(image, interp):
    model = _model(interp=interp).eval()
    off = torch.tensor([[[0.37, 0.61], [0.12, 0.88]]]).expand(BATCH, -1, -1)
    nearest = torch.stack(
        [torch.floor(off[..., 0] * W) / W + 0.5 / W, torch.floor(off[..., 1] * H) / H + 0.5 / H],
        dim=-1,
    )
    got = model.query(image, off)
    assert torch.isfinite(got).all()
    assert not torch.allclose(got, model.query(image, nearest))


def test_two_times_query_returns_twice_the_grid(image):
    model = _model().eval()
    centres = pixel_centre_coords(2 * H, 2 * W, image.device).unsqueeze(0).expand(BATCH, -1, -1)
    out = model.query(image, centres)
    assert out.shape == (BATCH, 1, 4 * H * W)
    assert out.reshape(BATCH, 1, 2 * H, 2 * W).shape == (BATCH, 1, 2 * H, 2 * W)


def test_hilbert_readout_at_a_cell_centre_is_that_cells_feature(image):
    """A cell centre's sequence position is an integer, so no blending happens there."""
    model = _model(interp="hilbert_t").eval()
    coords = pixel_centre_coords(H, W, image.device).unsqueeze(0).expand(BATCH, -1, -1)
    context = torch.randn(BATCH, H * W, D_MODEL)
    shape = (H, W)
    sampled = model._hilbert_readout(context, shape, coords)
    inverse = model.linearize._get_linearizer(shape, image.device).inverse_idx
    torch.testing.assert_close(sampled, context[:, inverse], atol=1e-5, rtol=1e-5)


def test_gradients_reach_the_context_encoder(image):
    model = _model().train()
    model(image).sum().backward()
    assert model.context_encoder.layers[0].in_proj.weight.grad is not None


def test_fourier_bands_are_the_capped_octaves():
    enc = INRMamba(1, 1, d_model=D_MODEL, num_layers=LAYERS, d_state=D_STATE, num_freqs=32).pos_enc
    assert isinstance(enc, CoordinatePositionalEncoding)
    assert enc.freq_bands.shape == (32,)
    assert enc.freq_bands.max().item() == pytest.approx(2.0**10)


def test_unknown_interp_raises():
    with pytest.raises(ValueError, match="interp"):
        _model(interp="nearest")


def test_inr_mamba_is_registered_strict():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    assert is_strict_kwargs("inr_mamba")
