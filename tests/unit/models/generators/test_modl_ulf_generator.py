"""``modl_ulf``: unrolled MoDL with simulated low-pass DC, and its no-DC control.

Paired with ``src/spectramr/models/generators/modl_ulf_generator.py``.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.ulf_lowpass import ulf_degrade
from spectramr.models.generators.modl_ulf_generator import MoDLULFGenerator


def _model(mode: str = "lowpass", **kw) -> MoDLULFGenerator:
    torch.manual_seed(0)
    cutoff = {"blur_cutoff": 0.5} if mode == "lowpass" else {}
    return MoDLULFGenerator(
        features=8, num_layers=3, iterations=3, data_consistency=mode, **cutoff, **kw
    )


def test_it_is_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert MODEL_REGISTRY["modl_ulf"]["class"] is MoDLULFGenerator


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"data_consistency": "cg"}, "data_consistency='cg'"),
        ({"data_consistency": "lowpass"}, "needs model_kwargs.blur_cutoff"),
        ({"data_consistency": "none", "blur_cutoff": 0.5}, "read only with"),
        (
            {"data_consistency": "none", "in_channels": 1, "out_channels": 2},
            "in_channels == out_channels",
        ),
    ],
)
def test_illegal_declarations_raise(kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MoDLULFGenerator(**kw)


def test_dc_changes_the_output_of_the_same_cnn() -> None:
    """The control is the same denoiser weights without the DC step."""
    torch.manual_seed(2)
    y = torch.rand(2, 1, 16, 16)
    dc, plain = _model("lowpass"), _model("none")
    plain.denoiser.load_state_dict(dc.denoiser.state_dict())
    with torch.no_grad():
        assert not torch.allclose(dc.eval()(y), plain.eval()(y))


def test_a_strong_dc_weight_makes_the_estimate_explain_the_input() -> None:
    """DC enforces A x = y: the 64 mT image is the (simulated) measurement of x."""
    torch.manual_seed(3)
    y = torch.rand(1, 1, 16, 16)
    weak, strong = (
        _model("lowpass", lambda_init=1e-3).eval(),
        _model("lowpass", lambda_init=1e3).eval(),
    )
    with torch.no_grad():
        err = [(ulf_degrade(m(y), 0.5) - y).abs().mean() for m in (weak, strong)]
    assert err[1] < err[0]


def test_lambda_is_learned_and_the_control_has_none() -> None:
    dc = _model("lowpass").train()
    y = torch.rand(2, 1, 16, 16)
    dc(y).pow(2).mean().backward()
    assert dc.log_lambda.grad is not None and dc.log_lambda.grad.abs() > 0
    assert _model("none").log_lambda is None


def test_torchio_depth_is_preserved() -> None:
    y = torch.rand(2, 1, 16, 16, 1)
    assert _model("lowpass").eval()(y).shape == y.shape
