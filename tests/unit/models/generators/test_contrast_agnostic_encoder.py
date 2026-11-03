"""contrast_agnostic_encoder: the registry builds the real encoder, and it projects."""

from __future__ import annotations

import warnings

import pytest
import torch

from spectramr.models.generators.contrast_agnostic_encoder import ContrastAgnosticEncoder

_TINY = {"base_channels": 4, "depth": 2, "projector_depth": 2, "projector_hidden_dim": 16}


def _factory():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from spectramr.models.factories.model_factory import ModelFactory

        return ModelFactory()


def test_registry_builds_the_real_encoder_through_the_factory() -> None:
    from spectramr.models.registry import MODEL_REGISTRY

    model = _factory().create_generator(
        model_type="contrast_agnostic_encoder",
        in_channels=1,
        out_channels=8,
        _declared_keys=frozenset(_TINY),
        **_TINY,
    )
    assert MODEL_REGISTRY["contrast_agnostic_encoder"]["class"] is ContrastAgnosticEncoder
    assert isinstance(model, ContrastAgnosticEncoder)
    out = model.eval()(torch.randn(3, 1, 16, 16))
    assert out["projection"].shape == (3, 8)
    assert out["embedding"].shape == (3, 16)  # base 4 * 2**depth
    assert out["features"].shape == (3, 16, 4, 4)


def test_an_unread_knob_raises() -> None:
    from spectramr.models.factories.model_factory import ModelCreationError

    with pytest.raises(ModelCreationError, match="use_momentum_encoder"):
        _factory().create_generator(
            model_type="contrast_agnostic_encoder",
            in_channels=1,
            out_channels=8,
            _declared_keys=frozenset({"use_momentum_encoder"}),
            use_momentum_encoder=True,
        )


def test_torchio_depth_axis_is_accepted_and_bad_sizes_raise() -> None:
    model = ContrastAgnosticEncoder(**_TINY).eval()
    assert model(torch.randn(2, 1, 16, 16, 1))["projection"].shape == (2, 128)
    with pytest.raises(ValueError, match="depth"):
        ContrastAgnosticEncoder(depth=0)
    with pytest.raises(ValueError, match="2-D"):
        ContrastAgnosticEncoder(spatial_dims=3)
