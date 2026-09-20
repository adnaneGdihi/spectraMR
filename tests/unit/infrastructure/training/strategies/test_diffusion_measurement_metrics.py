"""The measurement-aware metrics seam on the cold-diffusion validation path.

The trust functionals have been registered with ``needs=("mask", ...)`` since the
trust-functional work, and ``kspace_filling/PAPER.md`` advertises them as
reported, but **no shipped config could reach them** (2026-09-16). Three
independent reasons, one test each below:

1. ``_MEASUREMENT_AWARE_METRICS`` listed only two of the five names.
2. ``only=`` is an INTERSECTION with the computer's configured specs, and the
   validation computer is configured from ``validation.scoring.compute`` -- not
   ``metrics.compute``, which this seam's own comment named.
3. The context-free pass computes the same names as ``nan`` and its write-back
   ran *after* the seam, overwriting the measurement with the NaN.

These tests use the REAL ``ValidationMetricsComputer``. The stub they replaced
ignored ``only=`` and returned hard-coded values, so it proved the seam builds a
context and nothing at all about reachability from a YAML (non-negotiable 15:
a gate is only a gate for the violation shape you have watched it fail on).
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.computer import ValidationMetricsComputer
from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
    merge_metric_passes,
)

WANTED = DiffusionTrainingStrategy._MEASUREMENT_AWARE_METRICS

#: What the ten attention_shootout arms actually declare today.
_SHIPPED_SCORING_COMPUTE = ["psnr", "robust_mri_psnr", "hfen"]


class _Recorder:
    """Minimal logging_service stand-in: the seam warns through it."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def log_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class _SpyComputer:
    """Records the call the seam makes, so the context itself can be asserted."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def compute(self, pred, target, *, only=None, **kwargs):
        self.calls.append({"pred": pred, "target": target, "only": only, **kwargs})
        return dict.fromkeys(only or (), 0.5)


def _computer(names: list[str]) -> ValidationMetricsComputer:
    return ValidationMetricsComputer(
        config=ValidationMetricsConfig(metrics=[MetricSpec.from_name(n) for n in names])
    )


def _host(logging_service: _Recorder | None = None) -> DiffusionTrainingStrategy:
    host = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
    host._select_batch_compatible_smaps = lambda batch: None  # single-coil surrogate
    host.logging_service = logging_service or _Recorder()
    return host


def _interleaved(k: torch.Tensor) -> torch.Tensor:
    """Complex ``[B,C,H,W]`` -> real/imag interleaved ``[B,2C,H,W]`` (the model's layout)."""
    return torch.cat([k.real, k.imag], dim=1)


def _batch(fill: float = 1.0, h: int = 16, w: int = 16):
    """A hard-DC reconstruction whose null band is filled to ``fill`` x the truth."""
    torch.manual_seed(0)
    target = torch.randn(1, 1, h, w, dtype=torch.complex64)
    mask = torch.zeros(1, 1, h, w)
    mask[..., h // 2 - 4 : h // 2 + 4, :] = 1.0
    k = fft2c(target)
    pred = ifft2c(k * mask + fill * k * (1.0 - mask))
    measured = ifft2c(k * mask)
    return pred, target, measured, mask


# --- 1. the reachability bug -------------------------------------------------
def test_the_shipped_configured_set_intersects_to_nothing() -> None:
    """PLANTED: what every arm in the cohort got. Not NaN -- nothing at all."""
    pred, target, _, mask = _batch(fill=0.0)
    ctx = MetricContext(mask=mask, y_kspace=fft2c(target) * mask)
    out = _computer(_SHIPPED_SCORING_COMPUTE).compute(pred, target, only=WANTED, context=ctx)
    assert out == {}, (
        "an arm whose validation.scoring.compute omits these names must get NOTHING "
        "from the seam -- if this starts returning values, `only=` has stopped "
        "being an intersection and the seam has become a second owner of "
        "'which metrics does this arm grade on' (non-negotiable 17)"
    )


def test_the_widened_set_scores_on_the_context() -> None:
    """The fires-test: every declared name resolves to a finite number."""
    pred, target, _, mask = _batch(fill=0.0)
    ctx = MetricContext(mask=mask, y_kspace=fft2c(target) * mask)
    out = _computer([*_SHIPPED_SCORING_COMPUTE, *WANTED]).compute(
        pred, target, only=WANTED, context=ctx
    )
    assert set(out) == set(WANTED)
    assert all(math.isfinite(v) for v in out.values()), out
    # The do-nothing signature, on a literally zero-filled reconstruction.
    assert out["ndcr"] == pytest.approx(0.0, abs=1e-5)
    assert out["eta_null"] == pytest.approx(0.0, abs=1e-5)
    assert out["nse_hall"] == pytest.approx(1.0, abs=1e-3)
    assert out["null_band_energy_deficit"] == pytest.approx(1.0, abs=1e-5)


def test_without_the_context_every_name_is_nan() -> None:
    """PLANTED: the value the context-free write-back used to win with."""
    pred, target, _, _ = _batch(fill=0.0)
    out = _computer([*_SHIPPED_SCORING_COMPUTE, *WANTED]).compute(pred, target, only=WANTED)
    assert set(out) == set(WANTED)
    assert all(math.isnan(v) for v in out.values()), out


def test_the_measurement_aware_pass_owns_its_keys() -> None:
    """PLANTED: the overwrite, through the helper the production path calls.

    ``_compute_validation_metrics`` runs the context-free pass over every
    configured spec and then writes it back as ``val_<k>``. For a
    ``needs=("mask",)`` name that pass is structurally NaN, so whichever dict
    wins decides whether the column carries a number or a hole. Reverting
    ``merge_metric_passes`` to ``{**measurement_aware, **context_free}`` turns
    this red.
    """
    pred, target, _, mask = _batch(fill=0.0)
    computer = _computer([*_SHIPPED_SCORING_COMPUTE, *WANTED])
    ctx = MetricContext(mask=mask, y_kspace=fft2c(target) * mask)

    context_free = dict(computer.compute(pred, target))
    measurement_aware = dict(computer.compute(pred, target, only=WANTED, context=ctx))
    assert all(math.isnan(context_free[k]) for k in WANTED), (
        "precondition: without a context these names ARE NaN -- if that stops "
        "being true this test no longer exercises the overwrite"
    )

    merged = merge_metric_passes(context_free, measurement_aware)
    assert all(math.isfinite(merged[k]) for k in WANTED), merged
    # The names the main pass owns are untouched by the merge. (Compared
    # NaN-aware: robust_mri_psnr is itself N/A on this synthetic complex input,
    # and ``nan == nan`` is False -- which would make this assert about float
    # semantics rather than about ownership.)
    for name in _SHIPPED_SCORING_COMPUTE:
        before, after = context_free[name], merged[name]
        assert after == before or (math.isnan(before) and math.isnan(after)), name


def test_the_declared_names_cover_the_claims_the_cohort_publishes() -> None:
    """PLANTED: shrinking the tuple back to its 2026-09-02 pair turns this red.

    Every other test here reads ``WANTED`` off the class, so it would narrow
    silently with the tuple. This one pins the CONTENT against what
    ``experiments/inprogress/kspace_filling/PAPER.md`` promises the report gives
    -- the null-space error fraction and the data-consistency residual -- plus
    the two that answer "how much of this was filled in" at all.
    """
    assert set(WANTED) >= {
        "ndcr",
        "nse_hall",
        "eta_null",
        "fabrication_excess",
        "null_band_energy_deficit",
    }


# --- 2. the tuple stays honest ----------------------------------------------
def test_every_declared_name_is_registered_and_needs_the_measurement() -> None:
    """A name here that the context-free pass can compute does not belong here.

    The seam exists only for metrics the main pass cannot serve. Adding a plain
    full-reference metric to the tuple would make this seam a second owner of it.
    """
    assert len(WANTED) == len(set(WANTED))
    for name in WANTED:
        assert MetricsRegistry.is_registered(name), name
        assert MetricsRegistry.needs(name), f"{name} declares no measurement context"


# --- 3. the mask is declared, never inferred --------------------------------
def test_the_seam_uses_the_declared_mask_rather_than_the_measurement_support() -> None:
    """A partial readout window must not be read as unacquired lines.

    M4Raw stores 195 of 256 readout columns, so ~24.5% of every measurement is a
    structural zero. Inferring the support from non-zeros folds that window into
    the null band, and every null-band metric then reads it as invented content.
    """
    pred, target, measured, mask = _batch(fill=0.0)
    # A readout window: columns outside it are structurally zero in the
    # measurement, but they are NOT unacquired phase-encode lines.
    window = torch.zeros_like(mask)
    window[..., 3:13] = 1.0
    measured = ifft2c(fft2c(measured) * window)

    spy = _SpyComputer()
    declared = (mask * window).clamp(0.0, 1.0)
    _host()._measurement_aware_metrics(
        _interleaved(pred), _interleaved(target), _interleaved(measured), spy, declared
    )
    (call,) = spy.calls
    assert torch.equal(call["context"].mask, declared.to(torch.float32))

    spy_inferred = _SpyComputer()
    _host()._measurement_aware_metrics(
        _interleaved(pred), _interleaved(target), _interleaved(measured), spy_inferred, None
    )
    inferred = spy_inferred.calls[0]["context"].mask
    assert not torch.equal(inferred, declared), (
        "the inferred support must differ from the declared one here -- if it "
        "does not, this test has stopped exercising the readout-window shape"
    )


def test_the_seam_reports_an_absent_mask_instead_of_inferring_in_silence() -> None:
    """Absent is a state to report, never a state to infer (non-negotiable 18)."""
    pred, target, measured, _ = _batch(fill=0.0)
    recorder = _Recorder()
    _host(recorder)._measurement_aware_metrics(
        _interleaved(pred), _interleaved(target), _interleaved(measured), _SpyComputer(), None
    )
    assert len(recorder.warnings) == 1
    assert "rung mask" in recorder.warnings[0]

    recorder_ok = _Recorder()
    _, _, _, mask = _batch(fill=0.0)
    _host(recorder_ok)._measurement_aware_metrics(
        _interleaved(pred), _interleaved(target), _interleaved(measured), _SpyComputer(), mask
    )
    assert recorder_ok.warnings == []


def test_the_seam_passes_complex_images_and_the_acquired_kspace() -> None:
    """Plumbing: the null-space projector needs phase, which the arm's transform drops."""
    pred, target, measured, mask = _batch(fill=0.0)
    spy = _SpyComputer()
    out = _host()._measurement_aware_metrics(
        _interleaved(pred), _interleaved(target), _interleaved(measured), spy, mask
    )
    assert set(out) == set(WANTED)
    (call,) = spy.calls
    assert call["only"] == tuple(WANTED)
    ctx = call["context"]
    assert isinstance(ctx, MetricContext)
    assert torch.is_complex(ctx.y_kspace) and ctx.y_kspace.shape == (1, 1, 16, 16)
    assert torch.is_complex(call["pred"]) and call["pred"].shape == (1, 1, 16, 16)
