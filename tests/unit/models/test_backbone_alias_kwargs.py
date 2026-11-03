"""`_BackboneAlias` wrappers receive model_kwargs flattened, as GeneratorBuilder.build passes them."""

from __future__ import annotations

import warnings

import pytest
import torch


def _factory():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from spectramr.models.factories.model_factory import ModelFactory

        return ModelFactory()


def test_flattened_backbone_key_reaches_the_wrapper() -> None:
    from spectramr.models.generators.unet_generator import UNetGenerator

    model = _factory().create_generator(
        model_type="loupe",
        in_channels=2,
        out_channels=1,
        _declared_keys=frozenset({"backbone"}),
        backbone="unet",
    )
    assert isinstance(model.backbone, UNetGenerator)


class _TinyBackbone(torch.nn.Module):
    """Explicit signature: an undeclared knob must raise, not vanish."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 4) -> None:
        super().__init__()
        self.width = width
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@pytest.fixture
def tiny_backbone(monkeypatch: pytest.MonkeyPatch) -> str:
    from spectramr.models.registry import MODEL_REGISTRY

    _factory()
    entry = {**MODEL_REGISTRY["configurable_unet"], "class": _TinyBackbone}
    monkeypatch.setitem(MODEL_REGISTRY, "_tiny_backbone", entry)
    return "_tiny_backbone"


def _build(model_type: str, **kwargs):
    return _factory().create_generator(
        model_type=model_type,
        in_channels=2,
        out_channels=1,
        _declared_keys=frozenset(kwargs),
        **kwargs,
    )


def test_declared_knob_reaches_the_backbone(tiny_backbone: str) -> None:
    model = _build("loupe", backbone=tiny_backbone, width=7)
    assert isinstance(model.backbone, _TinyBackbone)
    assert model.backbone.width == 7


def test_unread_knob_raises(tiny_backbone: str) -> None:
    with pytest.raises(Exception, match="field_conditioning_dim"):
        _build("scas", backbone=tiny_backbone, field_conditioning_dim=32)


def test_strategy_owned_keys_are_not_forwarded(tiny_backbone: str) -> None:
    model = _build("pilot", backbone=tiny_backbone, n_shots=50, samples_per_shot=256)
    assert isinstance(model.backbone, _TinyBackbone)
    model = _build("loupe", backbone=tiny_backbone, share_across_coils=True)
    assert isinstance(model.backbone, _TinyBackbone)


def test_swept_model_fields_do_not_raise(tiny_backbone: str) -> None:
    # GeneratorBuilder sweeps top-level config.model fields into every build.
    model = _factory().create_generator(
        model_type="scas",
        in_channels=2,
        out_channels=1,
        _declared_keys=frozenset({"backbone"}),
        backbone=tiny_backbone,
        input_type="image",
    )
    assert isinstance(model.backbone, _TinyBackbone)


def test_nested_model_kwargs_raises(tiny_backbone: str) -> None:
    with pytest.raises(Exception, match="model_kwargs"):
        _build("scas", backbone=tiny_backbone, model_kwargs={"width": 3})


def test_declared_key_named_like_a_model_field_still_raises(tiny_backbone: str) -> None:
    # `base_channels` is also a top-level ModelConfigSchema field; declared in model_kwargs it
    # is the arm's knob, and the tiny backbone does not read it.
    with pytest.raises(Exception, match="base_channels"):
        _build("scas", backbone=tiny_backbone, base_channels=64)


def test_swept_field_named_like_a_knob_does_not_raise(tiny_backbone: str) -> None:
    model = _factory().create_generator(
        model_type="scas",
        in_channels=2,
        out_channels=1,
        _declared_keys=frozenset({"backbone"}),
        backbone=tiny_backbone,
        base_channels=64,  # swept from config.model, not declared by the arm
    )
    assert isinstance(model.backbone, _TinyBackbone)
