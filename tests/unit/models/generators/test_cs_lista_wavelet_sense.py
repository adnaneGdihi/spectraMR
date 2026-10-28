"""Unrolled ISTA with learned wavelet thresholds: knobs, init, band layout, gradients, recovery."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from spectramr.infrastructure.physics.multicoil_ei_operator import interleaved_to_complex
from spectramr.infrastructure.training.strategies.mixins.utils import _callable_accepts_kwarg
from spectramr.models.generators.cs_lista_wavelet_sense import CSLISTAWaveletSENSE, _band_index
from spectramr.models.registry import get_model_capabilities
from tests.unit.models.generators.test_cs_l1_wavelet_sense import _error, _problem

_KNOBS = {"num_iterations": 4, "lambda_reg": 0.003, "wavelet": "haar", "levels": 3}


def _model(**overrides) -> CSLISTAWaveletSENSE:
    return CSLISTAWaveletSENSE(**{**_KNOBS, **overrides})


def test_parameters_are_per_iteration_and_per_band() -> None:
    model = _model()
    assert model.raw_threshold.shape == (4, 9)
    assert model.log_step_scale.shape == (4,)
    assert torch.allclose(F.softplus(model.raw_threshold), torch.full((4, 9), 0.003))


def test_band_index_marks_the_approximation_and_every_detail_band_once() -> None:
    index = _band_index(64, 64, 3)
    assert (index == -1).sum() == 8 * 8
    assert sorted(index[index >= 0].unique().tolist()) == list(range(9))
    assert (
        index[0, 0] == -1
        and index[0, 63] == 0
        and index[63, 0] == 1
        and index[63, 63] == 2
        and index[0, 8] == 6
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lambda_reg": 0.0}, "lambda_reg"),
        ({"lambda_reg": True}, "lambda_reg"),
        ({"num_iterations": 0}, "num_iterations"),
        ({"wavelet": "sym9"}, "wavelet"),
        ({"levels": 0}, "levels"),
        ({"in_channels": 4}, "must be 2"),
    ],
)
def test_unknown_or_out_of_range_knob_raises(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _model(**kwargs)


def test_a_missing_knob_raises_rather_than_defaulting() -> None:
    with pytest.raises(TypeError, match="levels"):
        CSLISTAWaveletSENSE(num_iterations=2, lambda_reg=0.1, wavelet="haar")  # type: ignore[call-arg]


def test_forward_raises_without_the_mask() -> None:
    model_input, _, maps, _ = _problem(acceleration=4, noise=0.0)
    with pytest.raises(ValueError, match="mask"):
        _model()(model_input, sensitivity_maps=maps)


def test_every_parameter_receives_gradient() -> None:
    problem = _problem(acceleration=4, noise=0.01)
    model_input, mask, maps, x_true = problem
    model = _model()
    out = model(model_input, mask=mask, sensitivity_maps=maps)
    (interleaved_to_complex(out) - x_true).abs().pow(2).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.raw_threshold.grad.abs().sum() > 0


def test_training_the_thresholds_beats_the_initial_solver() -> None:
    problem = _problem(acceleration=4, noise=0.02)
    model_input, mask, maps, x_true = problem
    model = _model(lambda_reg=0.2)
    before = _error(model, problem)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(40):
        opt.zero_grad()
        out = model(model_input, mask=mask, sensitivity_maps=maps)
        (interleaved_to_complex(out) - x_true).abs().pow(2).mean().backward()
        opt.step()
    assert _error(model, problem) < before


def test_registered_and_fed_mask_and_maps() -> None:
    assert get_model_capabilities("cs_lista_wavelet_sense") is not None
    forward = _model().forward
    assert _callable_accepts_kwarg(forward, "mask")
    assert _callable_accepts_kwarg(forward, "sensitivity_maps")
