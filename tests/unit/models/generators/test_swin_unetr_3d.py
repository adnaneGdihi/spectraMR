"""True-3D Swin UNETR: production build path, pad/crop, cross-slice context, knob validation."""

from __future__ import annotations

import warnings

import pytest
import torch

from spectramr.models.generators.swin_unetr_3d import SwinUNETR3D

_TINY = {"feature_size": 12, "depths": [1, 1, 1, 1], "num_heads": [1, 2, 2, 4], "window_size": 2}


def _factory():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from spectramr.models.factories.model_factory import ModelFactory

        return ModelFactory()


def test_factory_builds_and_runs_a_16_slice_patch() -> None:
    model = _factory().create_generator(
        model_type="monai_swin_unetr_3d",
        in_channels=1,
        out_channels=1,
        _declared_keys=frozenset(_TINY),
        **_TINY,
    )
    assert isinstance(model, SwinUNETR3D)
    # The list-valued num_heads must survive the factory's scalar coercion.
    assert model.swin.swinViT.layers4[0].blocks[0].attn.num_heads == 4
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 32, 16)  # depth 16 is not a multiple of 32: pad/crop path
    with torch.no_grad():
        y = model.eval()(x)
    assert y.shape == x.shape


def test_receptive_field_crosses_slices() -> None:
    torch.manual_seed(0)
    model = SwinUNETR3D(**_TINY).eval()
    x = torch.randn(1, 1, 64, 32, 16)
    x2 = x.clone()
    x2[..., 0] += 5.0
    with torch.no_grad():
        delta = (model(x2) - model(x))[..., -1].abs().max()
    assert delta > 0, "perturbing slice 0 must change the last slice in a 3-D model"


def test_strict_kwargs_rejects_unread_knob() -> None:
    from spectramr.models.factories.model_factory import ModelCreationError

    with pytest.raises(ModelCreationError, match="dim"):
        _factory().create_generator(
            model_type="monai_swin_unetr_3d",
            in_channels=1,
            out_channels=1,
            dim=64,
            _declared_keys=frozenset({"dim"}),
        )


@pytest.mark.parametrize(
    "override",
    [
        {"feature_size": 16},
        {"depths": [1, 1, 1]},
        {"num_heads": [5, 1, 1, 1]},
        {"pad_mode": "circular"},
        {"window_size": 0},
    ],
)
def test_invalid_knobs_raise(override: dict) -> None:
    with pytest.raises(ValueError):
        SwinUNETR3D(**{**_TINY, **override})


def test_rejects_non_volumetric_input() -> None:
    with pytest.raises(ValueError, match="5-D"):
        SwinUNETR3D(**_TINY)(torch.zeros(1, 1, 32, 32))
