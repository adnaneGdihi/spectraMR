"""multi_param_mapping: the registry builds the real model, and its maps feed the TSE render."""

from __future__ import annotations

import warnings

import pytest
import torch

from spectramr.models.generators.multi_param_mapping import MultiParamMapping

_TINY = {"base_channels": 4, "depth": 2}


def _factory():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from spectramr.models.factories.model_factory import ModelFactory

        return ModelFactory()


def test_registry_builds_the_real_model() -> None:
    from spectramr.models.registry import MODEL_REGISTRY

    model = _factory().create_generator(
        model_type="multi_param_mapping",
        in_channels=3,
        out_channels=1,
        _declared_keys=frozenset(_TINY),
        **_TINY,
    )
    assert MODEL_REGISTRY["multi_param_mapping"]["class"] is MultiParamMapping
    assert isinstance(model, MultiParamMapping)


def test_maps_are_positive_and_named_for_the_render() -> None:
    from spectramr.infrastructure.physics.tse_forward import tse_signal

    model = MultiParamMapping(**_TINY).eval()
    with torch.no_grad():
        maps = model.predict_parameters(torch.randn(2, 3, 16, 16))
    assert set(maps) == {"rho", "t1", "t2"}
    assert all(m.shape == (2, 1, 16, 16) and bool((m > 0).all()) for m in maps.values())
    render = tse_signal(
        maps["rho"],
        maps["t1"],
        maps["t2"],
        family=0,
        tr=5500.0,
        te=128.0,
        ti=0.0,
        esp=16.0,
        etl=15,
        refocus_deg=160.0,
    )
    assert torch.isfinite(render).all()


def test_segmentation_head_is_built_only_when_asked_and_bad_sizes_raise() -> None:
    assert "segmentation" not in MultiParamMapping(**_TINY)(torch.randn(1, 3, 16, 16))
    out = MultiParamMapping(**_TINY, num_seg_classes=3)(torch.randn(1, 3, 16, 16))
    assert out["segmentation"].shape == (1, 3, 16, 16)
    with pytest.raises(ValueError, match="2-D"):
        MultiParamMapping(spatial_dims=3)
