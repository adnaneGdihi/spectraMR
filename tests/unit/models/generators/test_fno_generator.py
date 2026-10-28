"""FNOGenerator output head: ``output_form`` complex (default) vs magnitude."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.fno_generator import FNOGenerator


def _tiny(**kw) -> FNOGenerator:
    base = {"width": 8, "modes1": 4, "modes2": 4, "depth": 1, "output_domain": "image"}
    return FNOGenerator(**{**base, **kw})


def test_magnitude_head_emits_one_nonnegative_channel():
    model = _tiny(out_channels=1, output_form="magnitude").eval()
    out = model(torch.randn(1, 2, 16, 16))
    assert out.shape == (1, 1, 16, 16)
    assert bool((out >= 0).all())


def test_complex_head_is_default_and_emits_two_channels():
    model = _tiny().eval()
    assert model.output_form == "complex"
    assert model(torch.randn(1, 2, 16, 16)).shape == (1, 2, 16, 16)


def test_unknown_output_form_raises():
    with pytest.raises(ValueError, match="output_form"):
        _tiny(out_channels=2, output_form="phase")


def test_magnitude_head_requires_out_channels_one():
    with pytest.raises(ValueError, match="out_channels"):
        _tiny(out_channels=2, output_form="magnitude")
