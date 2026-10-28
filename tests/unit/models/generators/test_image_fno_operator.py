"""Unit tests for the plain image-domain FNO comparator of the CS-MNO PDE arms."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.image_fno_operator import ImageFNOOperator


@pytest.mark.unit
def test_one_dimensional_shape_round_trip() -> None:
    model = ImageFNOOperator(spatial_rank=1, n_layers=2, modes=4, dim=8)
    out = model(torch.randn(2, 1, 32))
    assert out.shape == (2, 1, 32)


@pytest.mark.unit
def test_two_dimensional_shape_round_trip() -> None:
    model = ImageFNOOperator(spatial_rank=2, n_layers=2, modes=4, dim=8)
    out = model(torch.randn(2, 1, 32, 32))
    assert out.shape == (2, 1, 32, 32)


@pytest.mark.unit
def test_torchio_trailing_singletons_are_squeezed_to_the_declared_rank() -> None:
    model = ImageFNOOperator(spatial_rank=1, n_layers=1, modes=4, dim=8)
    x = torch.randn(2, 1, 32)
    torch.testing.assert_close(model(x.view(2, 1, 32, 1, 1)), model(x))


@pytest.mark.unit
def test_rank_mismatch_raises() -> None:
    model = ImageFNOOperator(spatial_rank=1, n_layers=1, modes=4, dim=8)
    with pytest.raises(ValueError, match="rank-1"):
        model(torch.randn(2, 1, 16, 16))


@pytest.mark.unit
def test_spectral_weights_change_the_output() -> None:
    torch.manual_seed(0)
    model = ImageFNOOperator(spatial_rank=2, n_layers=1, modes=4, dim=8)
    x = torch.randn(1, 1, 32, 32)
    before = model(x).detach().clone()
    with torch.no_grad():
        model.layers[0].spectral.spectral.weights1.zero_()
        model.layers[0].spectral.spectral.weights2.zero_()
    assert not torch.allclose(model(x), before)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"spatial_rank": 3}, "spatial_rank"), ({"activation": "tanh"}, "activation")],
)
def test_unknown_values_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ImageFNOOperator(**kwargs)


@pytest.mark.unit
def test_registered_under_its_yaml_name() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert MODEL_REGISTRY["image_fno_operator"]["class"] is ImageFNOOperator
