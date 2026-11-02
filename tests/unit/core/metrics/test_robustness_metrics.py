"""Unit tests for attack-robustness / OOD reconstruction-stability metrics.

PR-6 of ``TODO/backlog_frontier_gap_audit_2026_05.md``. Two registered
metrics quantify *instability* of a reconstruction network that is
invisible to standard fidelity scores:

- ``adversarial_psnr_drop`` — PSNR loss under a small bounded (PGD,
  L-infinity) worst-case input perturbation. It needs the forward model
  (``model`` callable + ``model_input``); without them it is a documented
  no-op returning ``0.0`` because the perturbation cannot be applied.
- ``ood_acceleration_shift`` — PSNR degradation between an
  in-distribution acceleration and a held-out (OOD) acceleration. Without
  the OOD pair it is a documented no-op returning ``0.0``.

Both are ``higher_is_better == False`` (a larger drop is worse). The
toy models below are deliberately *fragile* (linear, ill-conditioned) so
the PGD attack and the OOD pair produce a strictly positive drop.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.core.metrics.registry import MetricsRegistry, get_metric
from spectramr.core.metrics.robustness_metrics import (
    AdversarialPSNRDropMetric,
    InputDependenceSpreadMetric,
    OODAccelerationShiftMetric,
    _psnr,
)


class _FragileLinear(nn.Module):
    """A high-gain linear map: tiny input perturbations blow up the output."""

    def __init__(self, gain: float = 50.0) -> None:
        super().__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain * x


# --------------------------------------------------------------------- #
# Registration / contract
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    ["adversarial_psnr_drop", "ood_acceleration_shift", "input_dependence_spread"],
)
def test_metrics_are_registered(name: str) -> None:
    assert MetricsRegistry.is_registered(name)
    assert get_metric(name) is not None


@pytest.mark.parametrize(
    "name", ["adversarial_psnr_drop", "ood_acceleration_shift"]
)
def test_lower_is_better(name: str) -> None:
    metric = get_metric(name)
    assert metric.higher_is_better is False
    assert metric.name == name


# --------------------------------------------------------------------- #
# PSNR helper
# --------------------------------------------------------------------- #


def test_psnr_perfect_is_finite_and_large() -> None:
    x = torch.rand(1, 1, 8, 8)
    val = _psnr(x, x)
    assert isinstance(val, float)
    assert val > 50.0  # near-perfect -> very high PSNR


# --------------------------------------------------------------------- #
# adversarial_psnr_drop
# --------------------------------------------------------------------- #


def test_adversarial_noop_without_model() -> None:
    metric = AdversarialPSNRDropMetric()
    pred = torch.rand(1, 1, 8, 8)
    target = torch.rand(1, 1, 8, 8)
    assert metric(pred, target) == 0.0


def test_adversarial_drop_positive_on_fragile_model() -> None:
    torch.manual_seed(0)
    model = _FragileLinear(gain=50.0)
    model_input = torch.rand(1, 1, 8, 8)
    target = model(model_input)  # clean output == target -> PSNR(clean) high
    pred = model(model_input)
    metric = AdversarialPSNRDropMetric(epsilon=1e-1, steps=5)
    drop = metric(pred, target, model=model, model_input=model_input)
    assert isinstance(drop, float)
    assert drop > 0.0


def test_adversarial_returns_float() -> None:
    model = _FragileLinear(gain=2.0)
    model_input = torch.rand(1, 1, 8, 8)
    target = model(model_input)
    pred = model(model_input)
    metric = AdversarialPSNRDropMetric()
    out = metric(pred, target, model=model, model_input=model_input)
    assert isinstance(out, float)


def test_adversarial_shape_mismatch_raises() -> None:
    metric = AdversarialPSNRDropMetric()
    pred = torch.rand(1, 1, 8, 8)
    target = torch.rand(1, 1, 4, 4)
    with pytest.raises(ValueError):
        metric(pred, target)


# --------------------------------------------------------------------- #
# ood_acceleration_shift
# --------------------------------------------------------------------- #


def test_ood_noop_without_ood_inputs() -> None:
    metric = OODAccelerationShiftMetric()
    pred = torch.rand(1, 1, 8, 8)
    target = torch.rand(1, 1, 8, 8)
    assert metric(pred, target) == 0.0


def test_ood_shift_positive_when_ood_worse() -> None:
    torch.manual_seed(0)
    target = torch.rand(1, 1, 8, 8)
    # In-distribution: nearly perfect reconstruction (high PSNR).
    pred = target + 1e-3 * torch.randn_like(target)
    # OOD: much worse reconstruction (low PSNR) at held-out acceleration.
    target_ood = torch.rand(1, 1, 8, 8)
    pred_ood = target_ood + 0.5 * torch.randn_like(target_ood)
    metric = OODAccelerationShiftMetric()
    shift = metric(
        pred,
        target,
        prediction_ood=pred_ood,
        target_ood=target_ood,
    )
    assert isinstance(shift, float)
    assert shift > 0.0


def test_ood_shift_via_callable() -> None:
    torch.manual_seed(1)
    target = torch.rand(1, 1, 8, 8)
    pred = target + 1e-3 * torch.randn_like(target)
    target_ood = torch.rand(1, 1, 8, 8)
    pred_ood = target_ood + 0.5 * torch.randn_like(target_ood)

    def make_ood() -> tuple[torch.Tensor, torch.Tensor]:
        return pred_ood, target_ood

    metric = OODAccelerationShiftMetric()
    shift = metric(pred, target, ood_fn=make_ood)
    assert isinstance(shift, float)
    assert shift > 0.0


def test_ood_shape_mismatch_raises() -> None:
    metric = OODAccelerationShiftMetric()
    pred = torch.rand(1, 1, 8, 8)
    target = torch.rand(1, 1, 4, 4)
    with pytest.raises(ValueError):
        metric(pred, target)


# --------------------------------------------------------------------- #
# InputDependenceSpreadMetric — DC-blob measurement-collapse gate (L4)
# --------------------------------------------------------------------- #


def test_input_dependence_higher_is_better() -> None:
    """We WANT the output to depend on the measurement, so higher is better."""
    metric = InputDependenceSpreadMetric()
    assert metric.higher_is_better is True
    assert metric.name == "input_dependence_spread"


def test_input_dependence_noop_without_evidence() -> None:
    """No cascade evidence -> documented 0.0 no-op (never raises)."""
    metric = InputDependenceSpreadMetric()
    assert metric(None, None) == 0.0
    assert metric(None, None, cascade_pred_means=[0.5]) == 0.0  # need >= 2


def test_input_dependence_scalar_fallback_flags_dc_blob() -> None:
    """The exact DC-blob signature collapses to a near-zero spread."""
    metric = InputDependenceSpreadMetric()
    spread = metric(None, None, cascade_pred_means=[0.1369, 0.1366, 0.1364])
    assert spread < 0.01


def test_input_dependence_per_pixel_catches_structural_collapse() -> None:
    """A constant tensor across the cascade -> per-pixel std 0 (caught)."""
    metric = InputDependenceSpreadMetric()
    blob = torch.full((1, 1, 8, 8), 0.137)
    spread = metric(None, None, cascade_predictions=[blob, blob.clone(), blob.clone()])
    assert spread == pytest.approx(0.0, abs=1e-6)


def test_input_dependence_healthy_output_not_flagged() -> None:
    """Distinct per-level predictions -> a clearly positive spread."""
    torch.manual_seed(0)
    metric = InputDependenceSpreadMetric()
    preds = [torch.rand(1, 1, 8, 8) for _ in range(3)]
    assert metric(None, None, cascade_predictions=preds) > 0.1


# ---------------------------------------------------------------------------
# certified_radius: registered, needs the reconstructor, raises without it
# ---------------------------------------------------------------------------


def test_certified_radius_registered_and_needs_reconstructor() -> None:
    assert MetricsRegistry.is_registered("certified_radius")
    assert "reconstructor" in MetricsRegistry.needs("certified_radius")


def test_certified_radius_raises_without_reconstructor() -> None:
    with pytest.raises(ValueError):
        get_metric("certified_radius")(torch.rand(1, 1, 4, 4), None)


def test_certified_radius_per_sample_and_mean() -> None:
    from spectramr.core.metrics.context import MetricContext
    from spectramr.core.metrics.robustness_metrics import CertifiedRadiusMetric

    torch.manual_seed(0)
    metric = get_metric("certified_radius", sigma=0.1, n_samples=4)
    assert isinstance(metric, CertifiedRadiusMetric)
    x = torch.full((3, 1, 4, 4), 2.0)
    radii = metric.per_sample(lambda z: z, x)
    assert radii.shape == (3,)
    assert bool((radii > 0).all())  # a positive mean response votes for a positive radius
    mean = metric(context=MetricContext(reconstructor=lambda z: z), model_input=x)
    assert mean == pytest.approx(float(radii.mean()), rel=0.2)


def test_adversarial_psnr_drop_runs_under_no_grad() -> None:
    torch.manual_seed(0)
    model = nn.Conv2d(1, 1, 3, padding=1)
    x = torch.rand(2, 1, 8, 8)
    target = model(x).detach()
    with torch.no_grad():  # the validation loop's context
        drop = get_metric("adversarial_psnr_drop", epsilon=0.05, steps=3)(
            target, target, model=model, model_input=x
        )
    assert drop > 0.0
