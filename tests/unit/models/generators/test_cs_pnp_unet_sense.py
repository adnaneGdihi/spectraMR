"""SENSE proximal gradient with a backbone as the prox: knobs, the data step, and gradients.

The backbone is stubbed with a one-conv module, so no real UNet is built.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.infrastructure.physics.multicoil_ei_operator import (
    MulticoilEIOperator,
    interleaved_to_complex,
)
from spectramr.infrastructure.training.strategies.mixins.utils import _callable_accepts_kwarg
from spectramr.models.generators import cs_pnp_unet_sense as mod
from spectramr.models.generators.cs_pnp_unet_sense import CSPnPUNetSENSE
from spectramr.models.registry import get_model_capabilities
from tests.unit.models.generators.test_cs_l1_wavelet_sense import _problem


class _Factory:
    """Stands in for the model factory; records the kwargs the backbone is built with."""

    def __init__(self) -> None:
        self.built: list[tuple[str, dict]] = []

    def create_generator(self, name: str, **kwargs) -> nn.Module:
        self.built.append((name, kwargs))
        conv = nn.Conv2d(2, 2, 1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch) -> _Factory:
    stub = _Factory()
    monkeypatch.setattr("spectramr.models.factories.model_factory.get_model_factory", lambda: stub)
    return stub


_KNOBS = {"num_iterations": 3, "backbone": "standard_unet", "backbone_kwargs": {"features": 8}}


def _model(**overrides) -> CSPnPUNetSENSE:
    return CSPnPUNetSENSE(**{**_KNOBS, **overrides})


def test_backbone_is_built_with_the_models_channels(factory: _Factory) -> None:
    _model(backbone_kwargs={"features": 8, "in_channels": 9})
    assert factory.built == [
        ("standard_unet", {"features": 8, "in_channels": 2, "out_channels": 2})
    ]


def test_zero_backbone_is_landweber_on_sense(factory: _Factory) -> None:
    """A backbone that adds nothing leaves K gradient steps at 1/L: the data step is exact."""
    model_input, mask, maps, _ = _problem(acceleration=4, noise=0.0)
    op = MulticoilEIOperator(maps, mask, where="in the test")
    b = interleaved_to_complex(model_input)
    step = 1.0 / op.lipschitz_bound()
    expected = b
    for _ in range(3):
        expected = expected - step * (op.adjoint(op.forward(expected)) - b)
    with torch.no_grad():
        out = _model()(model_input, mask=mask, sensitivity_maps=maps)
    assert out.shape == model_input.shape
    assert torch.allclose(interleaved_to_complex(out), expected, atol=1e-5)


def test_backbone_receives_gradient_through_every_iteration(factory: _Factory) -> None:
    model_input, mask, maps, x_true = _problem(acceleration=4, noise=0.0)
    model = _model()
    out = model(model_input, mask=mask, sensitivity_maps=maps)
    (interleaved_to_complex(out) - x_true).abs().pow(2).mean().backward()
    grads = [p.grad for p in model.backbone.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)


def test_forward_raises_without_the_mask_or_complex_maps(factory: _Factory) -> None:
    model_input, mask, maps, _ = _problem(acceleration=4, noise=0.0)
    model = _model(num_iterations=1)
    with pytest.raises(ValueError, match="needs the sampling mask"):
        model(model_input, sensitivity_maps=maps)
    with pytest.raises(ValueError, match="requires complex coil"):
        model(model_input, mask=mask)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_iterations": 0}, "num_iterations must be >= 1"),
        ({"num_iterations": 2.5}, "num_iterations must be a positive integer"),
        ({"backbone": ""}, "backbone must be a model name"),
        ({"backbone": "cs_l1_wavelet_sense"}, "is itself a solver"),
        ({"backbone_kwargs": None}, "backbone_kwargs must be a mapping"),
        ({"in_channels": 8}, "must be 2"),
    ],
)
def test_unknown_or_out_of_range_knob_raises(factory: _Factory, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _model(**kwargs)
    assert factory.built == []


def test_a_missing_knob_raises_rather_than_defaulting() -> None:
    knobs = {k: v for k, v in _KNOBS.items() if k != "backbone"}
    with pytest.raises(TypeError, match="backbone"):
        CSPnPUNetSENSE(**knobs)  # type: ignore[call-arg]


def test_registered_and_fed_mask_and_maps(factory: _Factory) -> None:
    assert get_model_capabilities("cs_pnp_unet_sense") is not None
    forward = _model().forward
    assert _callable_accepts_kwarg(forward, "mask")
    assert _callable_accepts_kwarg(forward, "sensitivity_maps")
    assert mod.CSPnPUNetSENSE is CSPnPUNetSENSE
