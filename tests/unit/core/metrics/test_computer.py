"""Tests for ``ValidationMetricsComputer`` (``core/metrics/computer.py``).

Covers the error / sentinel contract hardened in WS-2 6.2:

* a ``ValueError`` from a metric (virtually always a shape mismatch) is
  **re-raised** so the pipeline bug surfaces (CLAUDE.md pitfall #9);
* a non-``ValueError`` exception is logged and stored as ``float("nan")``
  (not a finite ``0.0`` that would masquerade as a real-but-terrible score);
* ``finalize()`` emits ``float("nan")`` for the three not-computed cases
  (``compute()`` returned ``None``, no ``compute`` method, or an exception),
  matching the NaN-means-not-computed contract in ``compute()``;
* ``get_direction`` is spec-first with a ``DEFAULT_METRIC_DIRECTIONS``
  fallback, and ``is_improvement`` respects both directions.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.computer import ValidationMetricsComputer
from spectramr.core.metrics.types import (
    MetricMode,
    MetricSpec,
    ValidationMetricsConfig,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes injected directly into ``_metric_instances`` so we exercise the
# error handling without touching the real registry.
# ---------------------------------------------------------------------------


class _FakeIncremental:
    """Non-summary metric whose ``__call__`` raises a chosen exception."""

    summarize = False

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __call__(self, preds, target, **kwargs):  # noqa: ANN001, ANN002
        raise self._exc


class _FakeSummary:
    """Summary metric with a configurable ``compute()`` outcome."""

    summarize = True

    def __init__(self, *, result=None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc

    def update(self, preds, target, **kwargs):  # noqa: ANN001, ANN002
        pass

    def compute(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeSummaryNoCompute:
    """Summary metric that lacks a ``compute`` method entirely."""

    summarize = True

    def update(self, preds, target, **kwargs):  # noqa: ANN001, ANN002
        pass


def _computer_with(metric_name: str, instance) -> ValidationMetricsComputer:
    cfg = ValidationMetricsConfig(
        metrics=[MetricSpec(name=metric_name)], primary_metric=metric_name
    )
    comp = ValidationMetricsComputer(cfg, device="cpu")
    # Pre-seed the cache so compute()/finalize() use our fake, not the registry.
    comp._metric_instances[metric_name] = instance
    return comp


_P = torch.zeros(1, 1, 4, 4)
_T = torch.zeros(1, 1, 4, 4)


# ---------------------------------------------------------------------------
# compute(): ValueError re-raises, other exceptions → NaN
# ---------------------------------------------------------------------------


def test_valueerror_from_metric_is_reraised() -> None:
    comp = _computer_with("psnr", _FakeIncremental(ValueError("shape mismatch")))
    with pytest.raises(ValueError, match="pitfall #9"):
        comp.compute(_P, _T)


def test_non_valueerror_from_metric_becomes_nan() -> None:
    comp = _computer_with("psnr", _FakeIncremental(RuntimeError("cuda oom")))
    out = comp.compute(_P, _T)
    assert "psnr" in out
    assert math.isnan(out["psnr"])


def test_repeated_metric_failure_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    # A metric that fails EVERY validation batch (e.g. LPIPS with no backend on a stale
    # env) must warn once, not once-per-batch — else the run log floods with hundreds of
    # identical lines (the 2026-07 cluster pull: 662 warnings).
    import logging

    comp = _computer_with("lpips", _FakeIncremental(RuntimeError("no backend")))
    with caplog.at_level(logging.WARNING, logger="spectramr.core.metrics.computer"):
        for _ in range(5):
            out = comp.compute(_P, _T)
            assert math.isnan(out["lpips"])  # still NaN every batch
    warnings = [r for r in caplog.records if "lpips" in r.getMessage()]
    assert len(warnings) == 1, f"expected 1 warning, got {len(warnings)}"
    assert comp._warned_metrics == {"lpips"}


# ---------------------------------------------------------------------------
# finalize(): the three not-computed sentinels are NaN, not 0.0
# ---------------------------------------------------------------------------


def test_finalize_none_result_is_nan() -> None:
    comp = _computer_with("fid", _FakeSummary(result=None))
    out = comp.finalize()
    assert math.isnan(out["fid"])


def test_finalize_exception_is_nan() -> None:
    comp = _computer_with("fid", _FakeSummary(exc=RuntimeError("no features")))
    out = comp.finalize()
    assert math.isnan(out["fid"])


def test_finalize_no_compute_method_is_nan() -> None:
    comp = _computer_with("fid", _FakeSummaryNoCompute())
    out = comp.finalize()
    assert math.isnan(out["fid"])


def test_finalize_valid_result_passes_through() -> None:
    comp = _computer_with("fid", _FakeSummary(result=12.5))
    out = comp.finalize()
    assert out["fid"] == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# get_direction: spec-first, then DEFAULT_METRIC_DIRECTIONS, then MAX
# ---------------------------------------------------------------------------


def test_get_direction_prefers_configured_spec() -> None:
    cfg = ValidationMetricsConfig(
        metrics=[MetricSpec(name="psnr", direction=MetricMode.MIN)],
        primary_metric="psnr",
    )
    comp = ValidationMetricsComputer(cfg, device="cpu")
    # Configured as MIN even though the default table says psnr is MAX.
    assert comp.get_direction("psnr") == MetricMode.MIN


def test_get_direction_falls_back_to_default_table() -> None:
    cfg = ValidationMetricsConfig(
        metrics=[MetricSpec(name="psnr")], primary_metric="psnr"
    )
    comp = ValidationMetricsComputer(cfg, device="cpu")
    # mse is not configured → resolved from the SSOT (MIN).
    assert comp.get_direction("mse") == MetricMode.MIN
    # An unknown metric used to silently resolve to MAX. That default is what
    # inverted best-checkpoint selection for every unlisted lower-is-better
    # metric (#208) — it must now raise.
    from spectramr.core.metrics.metric_directions import UnknownMetricDirectionError

    with pytest.raises(UnknownMetricDirectionError):
        comp.get_direction("totally_unknown_metric")


# ---------------------------------------------------------------------------
# is_improvement: both directions
# ---------------------------------------------------------------------------


def test_is_improvement_max_direction() -> None:
    cfg = ValidationMetricsConfig(
        metrics=[MetricSpec(name="psnr", direction=MetricMode.MAX)],
        primary_metric="psnr",
    )
    comp = ValidationMetricsComputer(cfg, device="cpu")
    assert comp.is_improvement(30.0, 28.0, "psnr") is True
    assert comp.is_improvement(27.0, 28.0, "psnr") is False


def test_is_improvement_min_direction() -> None:
    cfg = ValidationMetricsConfig(
        metrics=[MetricSpec(name="mse", direction=MetricMode.MIN)],
        primary_metric="mse",
    )
    comp = ValidationMetricsComputer(cfg, device="cpu")
    assert comp.is_improvement(0.1, 0.2, "mse") is True
    assert comp.is_improvement(0.3, 0.2, "mse") is False


# ---------------------------------------------------------------------------
# Regression: best-checkpoint selection was INVERTED for lower-is-better metrics
# (#208). get_direction ended in DEFAULT_METRIC_DIRECTIONS.get(name, MAX), so any
# lower-is-better metric absent from that 63-entry legacy table resolved to MAX
# and is_improvement() read `current > best` — retaining the WORST checkpoint.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    [
        "lpips",  # perceptual distance — the canonical `primary_metric` for GAN arms
        "fid",
        "nll_bits_per_dim",  # live in experiments/
        "val_lpips",  # the val_-prefixed form actually used in YAML
        "val_hfen",  # self-declared on the class, absent from the static map
        "val_kspace_error",
        "mad",
        "ms_gmsd",
        "expected_calibration_error",
        "gradient_entropy",  # the one metric the two legacy tables contradicted
    ],
)
def test_lower_is_better_metrics_are_minimized(metric: str) -> None:
    """A WORSE score must never count as an improvement."""
    comp = ValidationMetricsComputer(
        ValidationMetricsConfig(metrics=[], primary_metric=metric), device="cpu"
    )
    assert comp.get_direction(metric) == MetricMode.MIN
    # 0.9 is worse than 0.1 for a lower-is-better metric.
    assert comp.is_improvement(0.9, 0.1, metric) is False
    assert comp.is_improvement(0.1, 0.9, metric) is True


@pytest.mark.parametrize(
    "metric", ["psnr", "ssim", "val_psnr", "val_ssim", "val_ms_ssim", "val_dice"]
)
def test_higher_is_better_metrics_are_maximized(metric: str) -> None:
    comp = ValidationMetricsComputer(
        ValidationMetricsConfig(metrics=[], primary_metric=metric), device="cpu"
    )
    assert comp.get_direction(metric) == MetricMode.MAX
    assert comp.is_improvement(0.9, 0.1, metric) is True


def test_explicit_spec_direction_still_wins() -> None:
    """The user dictating `direction:` in YAML overrides the SSOT."""
    comp = ValidationMetricsComputer(
        ValidationMetricsConfig(
            metrics=[MetricSpec(name="lpips", direction=MetricMode.MAX)],
            primary_metric="lpips",
        ),
        device="cpu",
    )
    assert comp.get_direction("lpips") == MetricMode.MAX


def test_undeclared_monitor_key_raises() -> None:
    """An unresolvable monitor key must not silently default to MAX."""
    from spectramr.core.metrics.metric_directions import UnknownMetricDirectionError

    comp = ValidationMetricsComputer(
        ValidationMetricsConfig(metrics=[], primary_metric="psnr"), device="cpu"
    )
    with pytest.raises(UnknownMetricDirectionError):
        comp.get_direction("a_metric_nobody_declared")


# ---------------------------------------------------------------------------
# #173: an unregistered name reaching the computer is a wiring defect, not a typo.
#
# This used to warn-and-`continue`, producing a silently missing CSV column for
# the whole run. It fired routinely for six shipped `compute_*` flags naming
# unregistered metrics, so the warning was noise nobody read. Both upstream
# surfaces are now gated (explicit `metrics.compute` raises; dangling legacy flags
# are filtered with one log), so anything arriving here has passed both.
# ---------------------------------------------------------------------------


def test_unregistered_metric_raises_instead_of_being_skipped():
    import pytest

    from spectramr.core.metrics.computer import ValidationMetricsComputer
    from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig

    # Injected past the config surface: this models the WIRING defect the raise is
    # for, not a user typo (which the upstream gates now reject).
    computer = ValidationMetricsComputer(
        ValidationMetricsConfig(
            metrics=[
                MetricSpec(name="definitely_not_registered", direction=MetricMode.MAX)
            ],
            primary_metric="psnr",
        )
    )

    preds = torch.rand(1, 1, 16, 16)
    with pytest.raises(KeyError, match="wiring defect"):
        computer.compute(preds, preds.clone())


def test_registered_metrics_still_compute_normally():
    """The raise must not disturb the happy path."""
    from spectramr.core.metrics.computer import ValidationMetricsComputer
    from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig

    computer = ValidationMetricsComputer(
        ValidationMetricsConfig(
            metrics=[MetricSpec.from_name("psnr"), MetricSpec.from_name("ssim")],
            primary_metric="psnr",
        )
    )
    preds = torch.rand(1, 1, 32, 32)
    out = computer.compute(preds, preds.clone())
    assert set(out) >= {"psnr", "ssim"}


# ---------------------------------------------------------------------------
# A DECLARED not-applicable is not a crash.
#
# `outcome.py`'s own docstring calls collapsing scored / not-applicable / crashed
# into a bare NaN "pitfall #9 wearing a numeric disguise". The computer's broad
# `except Exception` did exactly that: it caught `MetricNotApplicableError` and
# logged "Failed to compute metric 'X'", so a metric that correctly declined to
# score an input read as a broken metric. The value is still NaN -- that IS the
# contract for a non-OK outcome -- but the machine-readable reason now survives.
# ---------------------------------------------------------------------------


def _unnormalized_pair():
    """A target honouring neither the [0,1] nor the [-1,1] contract (#180)."""
    target = torch.rand(1, 1, 32, 32) * 500.0
    return target + 5.0, target


def test_not_applicable_records_a_reason_rather_than_reading_as_a_crash():
    from spectramr.core.metrics.computer import ValidationMetricsComputer
    from spectramr.core.metrics.outcome import NotApplicableReason
    from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig

    computer = ValidationMetricsComputer(
        ValidationMetricsConfig(
            metrics=[MetricSpec.from_name("psnr"), MetricSpec.from_name("mse")],
            primary_metric="psnr",
        )
    )
    out = computer.compute(*_unnormalized_pair())

    assert math.isnan(out["psnr"]), "a non-OK outcome must carry NaN"
    assert (
        computer.last_not_applicable["psnr"]
        is NotApplicableReason.DATA_RANGE_UNRESOLVED
    )
    # Anti-vacuity: the range-insensitive sibling still scores, so this is a
    # per-metric exclusion and not a wholesale failure of the batch.
    assert math.isfinite(out["mse"])
    assert "mse" not in computer.last_not_applicable


def test_the_warning_says_not_applicable_not_failed_to_compute(caplog):
    """The log line is the only place a reader learns which of the two it was."""
    import logging

    from spectramr.core.metrics.computer import ValidationMetricsComputer
    from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig

    computer = ValidationMetricsComputer(
        ValidationMetricsConfig(
            metrics=[MetricSpec.from_name("psnr")], primary_metric="psnr"
        )
    )
    with caplog.at_level(logging.WARNING):
        computer.compute(*_unnormalized_pair())

    messages = [r.getMessage() for r in caplog.records]
    assert any("NOT APPLICABLE" in m for m in messages), messages
    assert not any("Failed to compute" in m for m in messages), (
        "a declared exclusion must not be reported as a failed computation"
    )


def test_the_reason_map_is_per_call_not_cumulative():
    """A metric that recovers must stop being reported as excluded."""
    from spectramr.core.metrics.computer import ValidationMetricsComputer
    from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig

    computer = ValidationMetricsComputer(
        ValidationMetricsConfig(
            metrics=[MetricSpec.from_name("psnr")], primary_metric="psnr"
        )
    )
    computer.compute(*_unnormalized_pair())
    assert "psnr" in computer.last_not_applicable

    normalized = torch.rand(1, 1, 32, 32)
    out = computer.compute(normalized, normalized.clone())

    assert computer.last_not_applicable == {}
    assert math.isfinite(out["psnr"])


def test_the_reason_dedupe_is_keyed_on_metric_and_reason(caplog):
    """Log-once per metric would hide the same metric failing a NEW way."""
    import logging

    from spectramr.core.metrics.computer import ValidationMetricsComputer
    from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig

    computer = ValidationMetricsComputer(
        ValidationMetricsConfig(
            metrics=[MetricSpec.from_name("psnr")], primary_metric="psnr"
        )
    )
    with caplog.at_level(logging.WARNING):
        computer.compute(*_unnormalized_pair())
        computer.compute(*_unnormalized_pair())

    na_lines = [r for r in caplog.records if "NOT APPLICABLE" in r.getMessage()]
    assert len(na_lines) == 1, "identical (metric, reason) must warn once"
    assert ("psnr", next(iter(computer.last_not_applicable.values()))) in (
        computer._not_applicable_seen
    )
# The fused host transfer (plan 4.3).
#
# `compute()` used to call `.item()` per metric per batch — one GPU sync each,
# ~500 per validation for 10 metrics over 50 batches, and each one drained the
# queue so metric k+1 could not launch while metric k was still running. Scalar
# results are now parked and transferred together.
#
# This is a PERFORMANCE change, so the tests that matter are the ones proving it
# changed nothing else: the two shapes that used to end in NaN must STILL end in
# NaN. A fused transfer that helpfully mean-reduces a vector-valued metric would
# convert a flagged defect into a plausible number (pitfall #18) — the exact
# trade this section exists to refuse.
# ---------------------------------------------------------------------------


class _FakeReturning:
    """Non-summary metric returning a fixed value."""

    summarize = False

    def __init__(self, value) -> None:
        self._value = value

    def __call__(self, preds, target, **kwargs):
        return self._value


def _multi_computer(**instances) -> ValidationMetricsComputer:
    cfg = ValidationMetricsConfig(
        metrics=[MetricSpec(name=n) for n in instances],
        primary_metric=next(iter(instances)),
    )
    comp = ValidationMetricsComputer(cfg, device="cpu")
    comp._metric_instances.update(instances)
    return comp


def test_scalar_tensor_metrics_are_transferred_in_one_sync() -> None:
    from unittest import mock

    from spectramr.core.metrics import scalar_transfer

    comp = _multi_computer(
        **{f"m{i}": _FakeReturning(torch.tensor(float(i))) for i in range(5)}
    )
    with mock.patch.object(
        scalar_transfer.torch, "stack", wraps=torch.stack
    ) as stack_spy:
        out = comp.compute(_P, _T)

    assert out == {f"m{i}": float(i) for i in range(5)}
    assert stack_spy.call_count == 1, "one sync for all five metrics, not five"


def test_no_per_metric_item_call_remains() -> None:
    """The anti-pattern itself, asserted absent rather than inferred gone."""
    from unittest import mock

    comp = _multi_computer(
        a=_FakeReturning(torch.tensor(1.0)), b=_FakeReturning(torch.tensor(2.0))
    )
    with mock.patch.object(
        torch.Tensor, "item", autospec=True, side_effect=AssertionError("GPU sync")
    ):
        assert comp.compute(_P, _T) == {"a": 1.0, "b": 2.0}


def test_metric_order_survives_the_deferred_write_back() -> None:
    """Deferred values are written back later; they must not be REORDERED.

    A seeded placeholder holds each dict slot. Without it the fused metrics would
    all land after the non-fused ones, silently permuting the console line and
    any consumer that trusts insertion order.
    """
    comp = _multi_computer(
        first=_FakeReturning(torch.tensor(1.0)),
        second=_FakeReturning(2.0),  # plain float: never deferred
        third=_FakeReturning(torch.tensor(3.0)),
    )
    assert list(comp.compute(_P, _T)) == ["first", "second", "third"]


def test_vector_valued_metric_still_becomes_nan() -> None:
    """Value-identity guard: a non-scalar return was NaN before, and stays NaN.

    `.item()` on a 4-element tensor raises RuntimeError, which the non-ValueError
    branch records as NaN. Mean-reducing it instead would be a silent change of
    meaning, not a speed-up.
    """
    comp = _computer_with("psnr", _FakeReturning(torch.zeros(4)))
    assert math.isnan(comp.compute(_P, _T)["psnr"])


def test_complex_scalar_metric_still_becomes_nan() -> None:
    """Same guard for complex: `float(complex)` raised before, so NaN.

    Taking `.real` here would be a guess about which projection is meaningful.
    """
    comp = _computer_with("psnr", _FakeReturning(torch.tensor(1 + 2j)))
    assert math.isnan(comp.compute(_P, _T)["psnr"])


def test_plain_python_floats_are_untouched() -> None:
    comp = _computer_with("psnr", _FakeReturning(3.5))
    assert comp.compute(_P, _T)["psnr"] == 3.5


def test_a_failing_metric_does_not_strand_its_neighbours() -> None:
    """The fused write-back runs after the loop — a mid-loop NaN must not skip it."""
    comp = _multi_computer(
        good=_FakeReturning(torch.tensor(9.0)),
        bad=_FakeIncremental(RuntimeError("boom")),
        alsogood=_FakeReturning(torch.tensor(4.0)),
    )
    out = comp.compute(_P, _T)
    assert out["good"] == 9.0
    assert out["alsogood"] == 4.0
    assert math.isnan(out["bad"])


# ---------------------------------------------------------------------------
# ``only=`` -- the metric subset used by the diffusion zero-filled baseline.
#
# The baseline grades a DIFFERENT tensor (the measurement the model was given)
# against the SAME target, so it needs a second pass over the batch. Running
# the arm's whole configured set twice would double every perceptual metric
# per rung per batch on a paradigm whose validation already dominates wall
# clock, so the second pass is restricted -- and restricted by INTERSECTION,
# never by addition, so this parameter cannot become a second owner of "which
# metrics does this arm grade on" (non-negotiable 17).
# ---------------------------------------------------------------------------


def test_only_restricts_the_call_to_the_named_subset() -> None:
    comp = _multi_computer(
        psnr=_FakeReturning(torch.tensor(30.0)),
        hfen=_FakeReturning(torch.tensor(0.5)),
        lpips=_FakeReturning(torch.tensor(0.1)),
    )
    out = comp.compute(_P, _T, only=("psnr", "hfen"))
    assert set(out) == {"psnr", "hfen"}, "the unnamed spec still ran"


def test_only_is_an_intersection_and_never_widens_the_set() -> None:
    """A name absent from the config is NOT computed just because it was asked for.

    The widening direction is the dangerous one: it would let a caller grade an
    arm on a metric the arm never declared, under the same ``val_*`` prefix the
    declared metrics use.
    """
    comp = _multi_computer(psnr=_FakeReturning(torch.tensor(30.0)))
    out = comp.compute(_P, _T, only=("psnr", "hfen"))
    assert set(out) == {"psnr"}


def test_default_call_still_runs_every_configured_spec() -> None:
    comp = _multi_computer(
        psnr=_FakeReturning(torch.tensor(30.0)),
        hfen=_FakeReturning(torch.tensor(0.5)),
        lpips=_FakeReturning(torch.tensor(0.1)),
    )
    assert set(comp.compute(_P, _T)) == {"psnr", "hfen", "lpips"}


def test_only_is_keyword_only() -> None:
    """A third POSITIONAL argument must not be able to become the subset.

    Every existing call site passes ``(predictions, targets)`` positionally;
    a positional-or-keyword ``only`` would make any future third positional
    argument silently narrow the metric set instead of raising.
    """
    import inspect

    param = inspect.signature(ValidationMetricsComputer.compute).parameters["only"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_a_subset_call_narrows_last_not_applicable() -> None:
    """The documented ordering hazard, executed rather than asserted in prose.

    ``last_not_applicable`` is per-call by contract, and ``training_loop.py``
    reads it after validation. A subset call therefore MUST run before the
    full one; this test is what makes that requirement a fact on the record
    rather than a comment.
    """
    comp = _multi_computer(
        psnr=_FakeReturning(torch.tensor(30.0)),
        lpips=_FakeReturning(torch.tensor(0.1)),
    )
    comp.last_not_applicable = {"lpips": "sentinel"}  # type: ignore[assignment]
    comp.compute(_P, _T, only=("psnr",))
    assert comp.last_not_applicable == {}, (
        "a subset call left the previous call's N/A record in place; if it is "
        "cleared (as here) it must run FIRST, not after the full compute"
    )


def test_factory_reads_scoring_compute_from_a_training_config() -> None:
    """A training config's ``.metrics`` schema block must not shadow ``validation.scoring``."""
    from types import SimpleNamespace

    from spectramr.core.metrics.computer import create_validation_metrics_computer

    cfg = SimpleNamespace(
        metrics=SimpleNamespace(compute=["mse"]),
        validation=SimpleNamespace(scoring=SimpleNamespace(compute=["psnr", "ssim"], primary="ssim")),
    )
    comp = create_validation_metrics_computer(cfg)
    assert comp.get_metric_names() == ["psnr", "ssim"]
    assert comp.config.primary_metric == "ssim"


def test_factory_refuses_a_metrics_attribute_that_names_nothing() -> None:
    from types import SimpleNamespace

    from spectramr.core.metrics.computer import create_validation_metrics_computer

    with pytest.raises(TypeError, match="not a list or dict"):
        create_validation_metrics_computer(SimpleNamespace(metrics=SimpleNamespace(compute=[])))
