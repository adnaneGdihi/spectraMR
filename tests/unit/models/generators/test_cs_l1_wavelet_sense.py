"""SENSE + L1-wavelet CS: knobs, the strategy seam it is fed through, and recovery.

Small synthetic problems only (4 coils x 64 x 64): the M4Raw arm is untested here.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.parameter_free import declares_parameter_free
from spectramr.infrastructure.logging.provenance import count_parameters
from spectramr.infrastructure.physics.multicoil_ei_operator import (
    MulticoilEIOperator,
    interleaved_to_complex,
)
from spectramr.infrastructure.training.strategies.mixins.utils import _callable_accepts_kwarg
from spectramr.models.generators.cs_l1_wavelet_sense import CSL1WaveletSENSE
from spectramr.models.registry import get_model_capabilities

N = 64


def _maps() -> torch.Tensor:
    """Four coils spread along the undersampled axis, normalised to sum_c |S_c|^2 = 1."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, N), torch.linspace(-1, 1, N), indexing="ij")
    coils = []
    for centre in (-1.2, -0.4, 0.4, 1.2):
        mag = torch.exp(-((xx - centre) ** 2) / 0.5 - yy**2 / 4)
        coils.append(torch.polar(mag, math.pi * (0.3 * yy + 0.4 * centre * xx)))
    maps = torch.stack(coils)[None]
    return maps / maps.abs().pow(2).sum(dim=1, keepdim=True).sqrt()


def _phantom() -> torch.Tensor:
    """Piecewise constant with a smooth phase: sparse in the Haar details."""
    x = torch.zeros(1, 1, N, N, dtype=torch.complex64)
    x[..., 8:56, 12:52] = 1.0
    x[..., 16:32, 20:36] = 0.4
    x[..., 36:52, 28:44] = 0.7
    return x * torch.polar(torch.ones(N, N), torch.linspace(0, 0.5, N).expand(N, N))


def _cartesian_mask(acceleration: int, seed: int = 0) -> torch.Tensor:
    """N / R phase-encode lines: an 8-line centre plus random lines, as [1, 1, 1, W]."""
    gen = torch.Generator().manual_seed(seed)
    lines = torch.zeros(N)
    lines[N // 2 - 4 : N // 2 + 4] = 1
    others = torch.nonzero(lines == 0).flatten()
    lines[others[torch.randperm(len(others), generator=gen)[: N // acceleration - 8]]] = 1
    return lines.view(1, 1, 1, N)


def _problem(acceleration: int, noise: float):
    torch.manual_seed(0)
    maps, mask, x_true = _maps(), _cartesian_mask(acceleration), _phantom()
    op = MulticoilEIOperator(maps, mask, where="in the test")
    y = op.forward(x_true)
    y = y + noise * mask * torch.randn_like(y)
    # The strategy's own seam: MulticoilEIOperator.prepare builds the model input.
    model_input, _ = op.prepare({"measured_kspace": y}, y)
    return model_input, mask, maps, x_true


_KNOBS = {"lambda_reg": 0.003, "num_iterations": 200, "wavelet": "haar", "levels": 3}


def _cs(**overrides) -> CSL1WaveletSENSE:
    return CSL1WaveletSENSE(**{**_KNOBS, **overrides})


def _error(model: CSL1WaveletSENSE, problem) -> float:
    model_input, mask, maps, x_true = problem
    with torch.no_grad():
        out = model(model_input, mask=mask, sensitivity_maps=maps)
    assert out.shape == model_input.shape
    return float((interleaved_to_complex(out) - x_true).norm() / x_true.norm())


def _zero_filled_error(problem) -> float:
    model_input, _, _, x_true = problem
    return float((interleaved_to_complex(model_input) - x_true).norm() / x_true.norm())


def test_r4_four_coils_beats_zero_filled() -> None:
    problem = _problem(acceleration=4, noise=0.01)
    assert _error(_cs(), problem) < 0.2 * _zero_filled_error(problem)


@pytest.mark.parametrize("acceleration", [4, 6])
def test_the_wavelet_prior_does_the_work_not_sense(acceleration: int) -> None:
    """Same solver and budget at lambda = 0 (pure SENSE): the prior must cut the error.

    Beating zero-filled alone could be SENSE unfolding; this isolates the prior.
    """
    problem = _problem(acceleration=acceleration, noise=0.01)
    assert _error(_cs(), problem) < 0.3 * _error(_cs(lambda_reg=0.0), problem)


def test_forward_raises_without_the_mask_or_complex_maps() -> None:
    model_input, mask, maps, _ = _problem(acceleration=4, noise=0.0)
    model = _cs(num_iterations=1)
    with pytest.raises(ValueError, match="needs the sampling mask"):
        model(model_input, sensitivity_maps=maps)
    with pytest.raises(ValueError, match="requires complex coil"):
        model(model_input, mask=mask)
    with pytest.raises(ValueError, match="COMPLEX coil"):
        model(model_input, mask=mask, sensitivity_maps=maps.abs())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"wavelet": "sym8"}, "wavelet 'sym8' is unknown"),
        ({"levels": 0}, "levels must be a positive integer"),
        ({"num_iterations": 0}, "num_iterations must be a positive integer"),
        ({"lambda_reg": -0.1}, "lambda_reg must be >= 0"),
        ({"lambda_reg": "auto"}, "lambda_reg must be a number"),
        ({"in_channels": 8}, "must be 2"),
    ],
)
def test_unknown_or_out_of_range_knob_raises(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _cs(**kwargs)


def test_a_missing_knob_raises_rather_than_defaulting() -> None:
    """The factory drops a misspelled key; no default may then stand in for it."""
    # What the factory hands over after dropping `lamda_reg`: the knob is absent.
    knobs = {k: v for k, v in _KNOBS.items() if k != "lambda_reg"}
    with pytest.raises(TypeError, match="lambda_reg"):
        CSL1WaveletSENSE(**knobs)  # type: ignore[call-arg]


def test_strategy_hands_it_mask_and_maps_and_nothing_it_ignores() -> None:
    """``_prepare_generator_inputs`` gates each kwarg on the forward signature."""
    forward = _cs().forward
    assert _callable_accepts_kwarg(forward, "mask")
    assert _callable_accepts_kwarg(forward, "sensitivity_maps")
    assert not _callable_accepts_kwarg(forward, "kspace_measured")


def test_registered_as_an_image_model_and_declared_parameter_free() -> None:
    caps = get_model_capabilities("cs_l1_wavelet_sense")
    assert caps is not None and caps.input_domain == "image" and caps.output_domain == "image"
    model = _cs()
    assert declares_parameter_free(model)
    assert sum(p.numel() for p in model.parameters()) == 0


def test_solver_knobs_reach_the_provenance_record() -> None:
    model = _cs(lambda_reg=0.02, num_iterations=7)
    record = count_parameters(model)
    assert record["total"] == 0
    assert record["solver"]["lambda_reg"] == 0.02
    assert record["solver"]["num_iterations"] == 7
    assert record["solver"]["wavelet"] == "haar" and record["solver"]["levels"] == 3
    assert record["solver"]["lambda_policy"] == "fixed"


# --- end to end: the arm's YAML, loader chain, collate and EI validation step -----


@pytest.mark.parametrize(
    ("arm", "two_d"),
    [
        ("cs_l1wavelet_sense_m4raw_r4", False),
        ("cs_l1wavelet_sense_m4raw_poisson_r16", True),
        ("cs_l1wavelet_sense_m4raw_radial_r8", True),
    ],
)
def test_the_arm_is_graded_by_the_ei_validation_step(monkeypatch, arm: str, two_d: bool) -> None:
    """The production validation path feeds it A^H y, the mask and the maps, and
    scores it on the cohort's columns; the second-repetition forward re-solves.
    The sweep cells hand it a 2-D mask that varies along the readout axis."""
    from tests.fixtures.self_supervised_batches import build_strategy, load_arm, real_batches

    config = load_arm(arm)
    _, val = real_batches(config, coils=4, shape=(64, 64))
    strategy = build_strategy(config)
    model = strategy.generator_model
    declared = config.model.model_kwargs
    assert isinstance(model, CSL1WaveletSENSE)
    assert (model.lambda_reg, model.num_iterations) == (
        declared["lambda_reg"],
        declared["num_iterations"],
    )
    assert (model.transform.wavelet, model.transform.levels) == (
        declared["wavelet"],
        declared["levels"],
    )
    calls: list[tuple[bool, bool]] = []
    forward = model.forward

    masks: list[torch.Tensor] = []

    def spy(x, mask=None, sensitivity_maps=None):
        calls.append((mask is not None, sensitivity_maps is not None))
        if mask is not None:
            masks.append(mask)
        return forward(x, mask=mask, sensitivity_maps=sensitivity_maps)

    monkeypatch.setattr(model, "forward", spy)
    metrics = strategy.validation_step(val.input, val.target, batch_data=val)
    assert calls and all(m and s for m, s in calls)
    varies_along_readout = bool((masks[0] != masks[0][..., :1, :]).any())
    assert varies_along_readout is two_d
    assert len(calls) >= 2, "the input_alt forward did not run"
    for key in ("val_psnr", "val_sense_psnr", "val_reference_noise_corrected_mse", "val_zf_psnr"):
        assert key in metrics and metrics[key] == metrics[key], f"{key} missing or NaN"
    assert metrics["val_rep_variance"] > 0
