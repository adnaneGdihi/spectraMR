"""Expansion tests for ``MetricsMixin``.

Targets ``spectramr.infrastructure.training.strategies.mixins.metrics_mixin``.

These tests exercise the mixin's publicly-callable seams *in isolation*
(no full ``BaseTrainingStrategy`` instantiation) so that the heavy
``TrainingSettings`` machinery is not pulled in for what should be unit
tests. The mixin is constructed via ``__new__`` and the minimal attrs
the methods touch (``config``, ``device``, the lazy metrics-computer
cache) are stitched on manually.

Categories
----------

- ``_extract_metrics_from_config``: dict / namespace / empty path
- ``_apply_metric_transforms``: every transform branch + autodetect
  (complex pred → magnitude, 2-ch real pred → magnitude,
  ``ifft_magnitude`` for even-channel real stacked, ``magnitude``
  for complex / 2-ch, no-op when ``domain == 'none'``)
- ``_convert_metrics_to_floats``: scalar tensor, multi-element tensor,
  complex tensor, native float / int, native complex, unsupported type
  raises ``TypeError``
- ``_validate_metrics_domain_consistency``: emits a warning through the
  attached logging service when training & validation domains
  disagree.
- ``_compute_training_metrics``: respects ``enable_tracking=False``,
  respects ``train_metric_interval``, unpacks dict pred/target, and
  returns the underlying computer's metrics keyed with the
  ``train_`` prefix.
- ``validation_step``: drives ``env.generator`` in eval mode, applies
  the configured transform, and delegates to the validation
  metrics computer.
- ``reset_validation_metrics`` / ``finalize_validation`` /
  ``get_primary_metric`` / ``is_metric_improvement`` proxy through to
  the validation computer.

Loud-fail expectations (CLAUDE.md #9)
-------------------------------------

The mixin promises that:

- ``_convert_metrics_to_floats`` refuses to silently coerce an unknown
  type to float — it raises ``TypeError``.
- ``_apply_metric_transforms('ifft_magnitude')`` against an odd-channel
  real tensor short-circuits to ``.abs()`` rather than feeding the
  raw real tensor to ``ifft2c`` (which would render the centro-
  symmetric "doubled-brain" artefact). The test pins this behaviour
  so a regression to a silent ``ifft2c`` call is detected.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


class _FakeLoggingService:
    """Minimal stand-in for the strategy's logging_service.

    Captures every log call so tests can assert on the warning emitted
    by ``_validate_metrics_domain_consistency``.
    """

    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def log_info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append(str(msg))

    def log_warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(str(msg))

    def log_error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.errors.append(str(msg))


class _FakeComputer:
    """Stand-in for the validation/training metrics computer."""

    def __init__(
        self,
        metrics: dict[str, float] | None = None,
        primary: str = "psnr",
    ) -> None:
        self._metrics = dict(metrics or {"psnr": 30.0, "ssim": 0.9})
        self._primary = primary
        self.reset_calls = 0
        self.finalize_calls = 0
        self.compute_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self._final: dict[str, float] = {}

    def get_metric_names(self) -> list[str]:
        return list(self._metrics)

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        self.compute_calls.append((tuple(pred.shape), tuple(target.shape)))
        return dict(self._metrics)

    def reset(self) -> None:
        self.reset_calls += 1

    def finalize(self) -> dict[str, float]:
        self.finalize_calls += 1
        return dict(self._final or self._metrics)

    def get_primary_metric(self) -> str:
        return self._primary

    def is_improvement(
        self,
        current: float,
        best: float,
        metric_name: str | None = None,
    ) -> bool:
        return current > best


def _make_mixin(
    *,
    config: Any | None = None,
    device: torch.device | None = None,
    training_computer: _FakeComputer | None = None,
    validation_computer: _FakeComputer | None = None,
    logging_service: _FakeLoggingService | None = None,
    env: Any | None = None,
) -> MetricsMixin:
    """Construct a ``MetricsMixin`` instance with minimal wiring.

    No call to ``MetricsMixin.__init__`` — we bypass it because the
    full strategy initializer requires a complete ``TrainingSettings``.
    """
    s = MetricsMixin.__new__(MetricsMixin)
    s.config = config if config is not None else SimpleNamespace()
    s.device = device or torch.device("cpu")
    if training_computer is not None:
        s._training_computer = training_computer
    if validation_computer is not None:
        s._validation_computer = validation_computer
    if logging_service is not None:
        s.logging_service = logging_service
    if env is not None:
        s.env = env
    return s


# ---------------------------------------------------------------------------
# _extract_metrics_from_config
# ---------------------------------------------------------------------------


class TestExtractMetricsFromConfig:
    def test_dict_with_true_flags_keeps_only_enabled(self) -> None:
        s = _make_mixin()
        out = s._extract_metrics_from_config({"psnr": True, "ssim": True, "mae": False})
        assert sorted(out) == ["psnr", "ssim"]

    def test_namespace_picks_up_compute_flags(self) -> None:
        s = _make_mixin()
        cfg = SimpleNamespace(
            compute_psnr=True,
            compute_ssim=False,
            compute_mae=True,
            compute_lpips=True,
        )
        out = s._extract_metrics_from_config(cfg)
        assert sorted(out) == ["lpips", "mae", "psnr"]

    def test_namespace_with_no_flags_falls_back_to_default(self) -> None:
        """If nothing is set, the extractor returns the documented
        ``['psnr', 'ssim']`` default — i.e. the mixin always gives
        the metrics computer *something* to compute."""
        s = _make_mixin()
        cfg = SimpleNamespace(unrelated="value")
        out = s._extract_metrics_from_config(cfg)
        assert out == ["psnr", "ssim"]

    def test_empty_dict_falls_back_to_default(self) -> None:
        s = _make_mixin()
        out = s._extract_metrics_from_config({})
        assert out == ["psnr", "ssim"]


# ---------------------------------------------------------------------------
# _convert_metrics_to_floats
# ---------------------------------------------------------------------------


class TestConvertMetricsToFloats:
    def test_scalar_tensor_uses_item(self) -> None:
        out = MetricsMixin._convert_metrics_to_floats({"psnr": torch.tensor(12.5)})
        assert out == {"psnr": 12.5}

    def test_multi_element_tensor_uses_mean(self) -> None:
        out = MetricsMixin._convert_metrics_to_floats(
            {"ssim": torch.tensor([0.8, 1.0, 0.6])}
        )
        assert out["ssim"] == pytest.approx((0.8 + 1.0 + 0.6) / 3.0, rel=1e-6)

    def test_complex_tensor_takes_real_part(self) -> None:
        ct = torch.complex(torch.tensor(2.0), torch.tensor(7.0))
        out = MetricsMixin._convert_metrics_to_floats({"phase_mse": ct})
        assert out == {"phase_mse": 2.0}

    def test_native_float_passes_through(self) -> None:
        out = MetricsMixin._convert_metrics_to_floats({"mae": 0.25})
        assert out == {"mae": 0.25}

    def test_native_int_promoted_to_float(self) -> None:
        out = MetricsMixin._convert_metrics_to_floats({"count": 3})
        assert out == {"count": 3.0}
        assert isinstance(out["count"], float)

    def test_native_complex_takes_real_part(self) -> None:
        out = MetricsMixin._convert_metrics_to_floats({"x": complex(1.5, -3.0)})
        assert out == {"x": 1.5}

    def test_fused_multi_tensor_matches_per_item(self) -> None:
        """TRAIN-1 (wasted-compute audit): many tensor metrics are transferred
        in ONE fused ``.cpu()`` instead of N ``.item()`` syncs. The fused result
        must be value-identical to the naive per-tensor conversion, including
        mixed dtypes (AMP fp16/fp32), non-scalar means, complex real-parts, and
        interleaved python scalars — and every key must survive.
        """
        metrics = {
            "g_total": torch.tensor(0.5),
            "l1": torch.tensor([0.2, 0.4]),  # mean = 0.3
            "fp16": torch.tensor(1.25, dtype=torch.float16),  # AMP component
            "phase": torch.complex(torch.tensor(3.0), torch.tensor(9.0)),
            "lr": 1e-4,  # python float, no sync
            "step": 7,  # python int
        }
        out = MetricsMixin._convert_metrics_to_floats(metrics)
        assert out["g_total"] == pytest.approx(0.5)
        assert out["l1"] == pytest.approx(0.3, rel=1e-6)
        assert out["fp16"] == pytest.approx(1.25, rel=1e-3)
        assert out["phase"] == pytest.approx(3.0)
        assert out["lr"] == pytest.approx(1e-4)
        assert out["step"] == pytest.approx(7.0)
        assert set(out) == set(metrics)
        assert all(isinstance(v, float) for v in out.values())

    def test_unsupported_type_raises_typeerror(self) -> None:
        """The mixin refuses silent coercion of unknown types — this is
        the CLAUDE.md #9 "no silent fallback" rule applied to the
        metrics aggregation path. A regression to a silent ``str`` →
        ``float('nan')`` would mask a broken metric computer."""
        with pytest.raises(TypeError, match="Cannot convert metric"):
            MetricsMixin._convert_metrics_to_floats({"bad": "not a float"})

    def test_unsupported_type_error_names_the_metric(self) -> None:
        with pytest.raises(TypeError) as exc_info:
            MetricsMixin._convert_metrics_to_floats({"weird": object()})
        assert "weird" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _apply_metric_transforms
# ---------------------------------------------------------------------------


def _config_with(
    *,
    target_channels: int | None = None,
    coil_mode: str | None = None,
    num_virtual_coils: int | None = None,
    validation_transform: str | None = None,
) -> SimpleNamespace:
    data = SimpleNamespace(
        target_channels=target_channels,
        coil_processing_mode=coil_mode,
        num_virtual_coils=num_virtual_coils,
    )
    validation = SimpleNamespace(
        transform=validation_transform,
        output_transform=None,
    )
    return SimpleNamespace(data=data, validation=validation)


class TestApplyMetricTransforms:
    def test_domain_none_short_circuits(self) -> None:
        """An explicit ``domain: none`` in the metrics config disables
        transforms regardless of dtype."""
        s = _make_mixin(config=_config_with())
        pred = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        target = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        cfg = SimpleNamespace(domain="none", transform=None)
        out_pred, out_target = s._apply_metric_transforms(pred, target, cfg)
        assert torch.equal(out_pred, pred)
        assert torch.equal(out_target, target)

    def test_no_transform_no_complex_no_op(self) -> None:
        """No explicit transform + already-real single-channel image →
        the method is a pure no-op."""
        s = _make_mixin(config=_config_with())
        pred = torch.randn(1, 1, 8, 8)
        target = torch.randn(1, 1, 8, 8)
        cfg = SimpleNamespace(domain=None, transform=None)
        out_pred, out_target = s._apply_metric_transforms(pred, target, cfg)
        assert torch.equal(out_pred, pred)
        assert torch.equal(out_target, target)

    def test_complex_pred_autodetects_magnitude_transform(self) -> None:
        """A complex pred without an explicit transform must default to
        ``magnitude`` — otherwise the metrics computer would receive a
        complex tensor it cannot handle."""
        s = _make_mixin(config=_config_with())
        pred = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        target = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        out_pred, out_target = s._apply_metric_transforms(pred, target, None)
        assert torch.is_floating_point(out_pred)
        assert torch.is_floating_point(out_target)
        assert out_pred.shape == pred.shape
        # |z| should equal sqrt(re² + im²)
        expected = torch.sqrt(pred.real**2 + pred.imag**2)
        assert torch.allclose(out_pred, expected, atol=1e-5)

    def test_two_channel_real_autodetects_magnitude_transform(self) -> None:
        """4D real with C=2 is the canonical "real/imag split" layout —
        the magnitude transform pairs the two channels."""
        s = _make_mixin(config=_config_with())
        re = torch.randn(1, 1, 4, 4)
        im = torch.randn(1, 1, 4, 4)
        pred = torch.cat([re, im], dim=1)
        target = torch.cat([re.clone(), im.clone()], dim=1)
        out_pred, out_target = s._apply_metric_transforms(pred, target, None)
        # Output collapses C=2 → C=1 magnitude
        assert out_pred.shape == (1, 1, 4, 4)
        expected = torch.sqrt(re**2 + im**2)
        assert torch.allclose(out_pred, expected, atol=1e-5)
        assert torch.allclose(out_target, expected, atol=1e-5)

    def test_explicit_magnitude_transform_on_complex(self) -> None:
        s = _make_mixin(config=_config_with())
        pred = torch.complex(
            torch.full((1, 1, 4, 4), 3.0), torch.full((1, 1, 4, 4), 4.0)
        )
        target = torch.complex(
            torch.full((1, 1, 4, 4), 6.0), torch.full((1, 1, 4, 4), 8.0)
        )
        cfg = SimpleNamespace(domain=None, transform="magnitude")
        out_pred, out_target = s._apply_metric_transforms(pred, target, cfg)
        assert torch.allclose(out_pred, torch.full((1, 1, 4, 4), 5.0), atol=1e-5)
        assert torch.allclose(out_target, torch.full((1, 1, 4, 4), 10.0), atol=1e-5)

    def test_ifft_magnitude_complex_input(self) -> None:
        """Complex input → IFFT → magnitude. Shape preserved
        spatially; channel count preserved."""
        s = _make_mixin(config=_config_with())
        torch.manual_seed(0)
        pred = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        target = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        cfg = SimpleNamespace(domain=None, transform="ifft_magnitude")
        out_pred, out_target = s._apply_metric_transforms(pred, target, cfg)
        # Output is real magnitude with same B/C/H/W.
        assert out_pred.shape == (1, 1, 8, 8)
        assert out_target.shape == (1, 1, 8, 8)
        assert torch.is_floating_point(out_pred)
        assert torch.is_floating_point(out_target)
        assert (out_pred >= 0).all()

    def test_ifft_magnitude_even_channel_real_stacked(self) -> None:
        """Real-stacked [R0, I0, R1, I1] layout (C=4, two coils) →
        pair → IFFT → RSS-combine to single-channel magnitude."""
        s = _make_mixin(config=_config_with())
        torch.manual_seed(0)
        pred = torch.randn(1, 4, 8, 8)
        target = torch.randn(1, 4, 8, 8)
        cfg = SimpleNamespace(domain=None, transform="ifft_magnitude")
        out_pred, out_target = s._apply_metric_transforms(pred, target, cfg)
        # 2 paired complex channels → RSS-combine → 1 channel
        assert out_pred.shape == (1, 1, 8, 8)
        assert out_target.shape == (1, 1, 8, 8)
        assert (out_pred >= 0).all()
        assert (out_target >= 0).all()

    def test_ifft_magnitude_single_channel_real_falls_back_to_abs(self) -> None:
        """Single real channel cannot be paired as R/I — the mixin must
        treat it as already-magnitude and return ``.abs()``, NOT call
        ``ifft2c`` on a strictly-real tensor (which would produce the
        Hermitian-symmetric "doubled-brain" artefact). This is the
        loud-fail / no-silent-fallback rule for the visualization
        path."""
        s = _make_mixin(config=_config_with())
        pred = torch.randn(1, 1, 8, 8)
        target = torch.randn(1, 1, 8, 8)
        cfg = SimpleNamespace(domain=None, transform="ifft_magnitude")
        out_pred, out_target = s._apply_metric_transforms(pred, target, cfg)
        assert torch.allclose(out_pred, pred.abs(), atol=1e-6)
        assert torch.allclose(out_target, target.abs(), atol=1e-6)

    def test_ifft_magnitude_virtual_coil_slice(self) -> None:
        """When virtual-coil compression is active (``svd``/``compress``)
        and the tensor carries more channels than ``2 * num_virtual_coils``,
        the mixin slices to the first contrast's R/I channels before
        the IFFT — preventing cross-contrast RSS mixing."""
        s = _make_mixin(
            config=_config_with(
                target_channels=4,
                coil_mode="svd",
                num_virtual_coils=2,
            )
        )
        torch.manual_seed(0)
        # 8 channels = 2 contrasts × 2 virtual coils × R/I.
        pred = torch.randn(1, 8, 4, 4)
        target = torch.randn(1, 8, 4, 4)
        cfg = SimpleNamespace(domain=None, transform="ifft_magnitude")
        out_pred, out_target = s._apply_metric_transforms(pred, target, cfg)
        # First contrast → 2 paired complex channels → RSS to C=1.
        assert out_pred.shape == (1, 1, 4, 4)
        assert out_target.shape == (1, 1, 4, 4)

    def test_explicit_transform_in_metrics_config_overrides_validation(
        self,
    ) -> None:
        """Metrics-config transform wins over validation-config
        transform."""
        s = _make_mixin(config=_config_with(validation_transform="ifft_magnitude"))
        pred = torch.randn(1, 1, 8, 8)
        target = torch.randn(1, 1, 8, 8)
        cfg = SimpleNamespace(
            domain=None,
            transform=None,
            output_transform="magnitude",
        )
        out_pred, out_target = s._apply_metric_transforms(pred, target, cfg)
        # `magnitude` on 1-channel real is a no-op; sanity: shape preserved.
        assert out_pred.shape == pred.shape

    def test_validation_config_transform_used_when_metrics_config_empty(
        self,
    ) -> None:
        s = _make_mixin(config=_config_with(validation_transform="ifft_magnitude"))
        torch.manual_seed(0)
        pred = torch.randn(1, 2, 4, 4)
        target = torch.randn(1, 2, 4, 4)
        out_pred, out_target = s._apply_metric_transforms(pred, target, None)
        # Even-channel real-stacked → ifft → RSS to single channel.
        assert out_pred.shape == (1, 1, 4, 4)
        assert out_target.shape == (1, 1, 4, 4)


# ---------------------------------------------------------------------------
# _validate_metrics_domain_consistency
# ---------------------------------------------------------------------------


class TestValidateMetricsDomainConsistency:
    def test_matching_domains_emits_no_warning(self) -> None:
        log = _FakeLoggingService()
        s = _make_mixin(logging_service=log)
        config = SimpleNamespace(
            metrics=SimpleNamespace(domain="image"),
            validation=SimpleNamespace(domain="image", output_transform=None),
        )
        s._validate_metrics_domain_consistency(config)
        assert log.warnings == []

    def test_mismatched_domains_warns(self) -> None:
        log = _FakeLoggingService()
        s = _make_mixin(logging_service=log)
        config = SimpleNamespace(
            metrics=SimpleNamespace(domain="image"),
            validation=SimpleNamespace(domain="kspace", output_transform=None),
        )
        s._validate_metrics_domain_consistency(config)
        assert len(log.warnings) == 1
        assert "Metrics Domain Mismatch" in log.warnings[0]
        assert "image" in log.warnings[0]
        assert "kspace" in log.warnings[0]

    def test_no_logging_service_is_silent(self) -> None:
        """The method must not crash when no logging_service is wired."""
        s = MetricsMixin.__new__(MetricsMixin)
        # No logging_service attribute — the method should early-return.
        config = SimpleNamespace(
            metrics=SimpleNamespace(domain="image"),
            validation=SimpleNamespace(domain="kspace", output_transform=None),
        )
        s._validate_metrics_domain_consistency(config)  # must not raise

    def test_ifft_magnitude_transform_normalizes_to_image_domain(self) -> None:
        """If the training metric path declares ``transform=ifft_magnitude``
        the effective output domain is image — i.e. the consistency check
        must NOT flag it as kspace just because the underlying physics is
        kspace."""
        log = _FakeLoggingService()
        s = _make_mixin(logging_service=log)
        config = SimpleNamespace(
            metrics=SimpleNamespace(domain="kspace", transform="ifft_magnitude"),
            validation=SimpleNamespace(domain=None, output_transform=None),
        )
        s._validate_metrics_domain_consistency(config)
        # train_domain_actual flips back to "image" → val falls through
        # to train_domain_actual → no mismatch.
        assert log.warnings == []


# ---------------------------------------------------------------------------
# _compute_training_metrics
# ---------------------------------------------------------------------------


class TestComputeTrainingMetrics:
    def test_disabled_tracking_short_circuits(self) -> None:
        """``metrics.enable_tracking == False`` returns ``{}``."""
        computer = _FakeComputer()
        s = _make_mixin(training_computer=computer)
        cfg = SimpleNamespace(
            metrics=SimpleNamespace(enable_tracking=False, train_metric_interval=1)
        )
        out = s._compute_training_metrics(
            torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 4),
            config=cfg,
            current_step=0,
        )
        assert out == {}
        assert computer.compute_calls == []

    def test_off_interval_returns_empty(self) -> None:
        """Steps that aren't multiples of ``train_metric_interval`` are
        skipped — this is the perf invariant ("don't recompute every
        step")."""
        computer = _FakeComputer()
        s = _make_mixin(training_computer=computer)
        cfg = SimpleNamespace(
            metrics=SimpleNamespace(enable_tracking=True, train_metric_interval=10)
        )
        out = s._compute_training_metrics(
            torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 4),
            config=cfg,
            current_step=3,
        )
        assert out == {}
        assert computer.compute_calls == []

    # ------------------------------------------------------------------
    # A budget under the cadence used to compute NOTHING. Measured on
    # `experiment_11_attention_none` (`max_iterations: 40`, interval at the
    # schema default 100): every `train_*` column in `training_metrics.csv` was
    # blank on all 40 rows while the loss columns filled normally.
    #
    # The throttle only LOOKED safe because it used to be a facade -- with
    # `self.env.step` frozen at 0, `0 % interval == 0` fired on every step
    # (pitfall #16). Fixing the step to be live (`resolve_loop_iteration`)
    # flipped short runs from always-on to never.
    # ------------------------------------------------------------------

    def test_first_iteration_fires_under_an_unsatisfiable_cadence(self) -> None:
        """The regression, at the exact numbers that produced it."""
        computer = _FakeComputer()
        s = _make_mixin(training_computer=computer)
        cfg = SimpleNamespace(
            metrics=SimpleNamespace(enable_tracking=True, train_metric_interval=100),
            training=SimpleNamespace(max_iterations=40),
        )
        out = s._compute_training_metrics(
            torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 4),
            config=cfg,
            current_step=1,
        )
        assert out, "iteration 1 must compute -- otherwise the column is empty"
        assert len(computer.compute_calls) == 1

    def test_final_iteration_fires_under_an_unsatisfiable_cadence(self) -> None:
        """Two points make a curve; one makes a dot. The final iteration is the
        other end, and it is read off the DECLARED budget."""
        computer = _FakeComputer()
        s = _make_mixin(training_computer=computer)
        cfg = SimpleNamespace(
            metrics=SimpleNamespace(enable_tracking=True, train_metric_interval=100),
            training=SimpleNamespace(max_iterations=40),
        )
        out = s._compute_training_metrics(
            torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 4),
            config=cfg,
            current_step=40,
        )
        assert out
        assert len(computer.compute_calls) == 1

    def test_the_whole_short_run_yields_exactly_two_points(self) -> None:
        """The perf invariant survives: the override adds TWO computations to a
        run, not one per step. Asserted over the real 1..40 sweep rather than at
        single steps, because "does not recompute every step" is a property of
        the sweep."""
        computer = _FakeComputer()
        s = _make_mixin(training_computer=computer)
        cfg = SimpleNamespace(
            metrics=SimpleNamespace(enable_tracking=True, train_metric_interval=100),
            training=SimpleNamespace(max_iterations=40),
        )
        fired = [
            step
            for step in range(1, 41)
            if s._compute_training_metrics(
                torch.randn(1, 1, 4, 4),
                torch.randn(1, 1, 4, 4),
                config=cfg,
                current_step=step,
            )
        ]
        assert fired == [1, 40]

    def test_a_normal_run_keeps_its_cadence_plus_the_two_ends(self) -> None:
        """The override must not perturb a correctly-configured run: the
        periodic points are unchanged and only the two ends are added."""
        computer = _FakeComputer()
        s = _make_mixin(training_computer=computer)
        cfg = SimpleNamespace(
            metrics=SimpleNamespace(enable_tracking=True, train_metric_interval=100),
            training=SimpleNamespace(max_iterations=250),
        )
        fired = [
            step
            for step in range(1, 251)
            if s._compute_training_metrics(
                torch.randn(1, 1, 4, 4),
                torch.randn(1, 1, 4, 4),
                config=cfg,
                current_step=step,
            )
        ]
        assert fired == [1, 100, 200, 250]

    def test_unresolvable_budget_still_yields_the_first_point(self) -> None:
        """The stated approximation: the mixin reads the DECLARED budget, so an
        epoch-bounded run that stops early never hits the final leg. A one-point
        curve is the documented floor -- not an empty column, and not a raise."""
        computer = _FakeComputer()
        s = _make_mixin(training_computer=computer)
        cfg = SimpleNamespace(
            metrics=SimpleNamespace(enable_tracking=True, train_metric_interval=100),
        )  # no `training` block at all
        assert s._compute_training_metrics(
            torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 4),
            config=cfg,
            current_step=1,
        )
        assert (
            s._compute_training_metrics(
                torch.randn(1, 1, 4, 4),
                torch.randn(1, 1, 4, 4),
                config=cfg,
                current_step=40,
            )
            == {}
        )

    def test_a_non_positive_interval_does_not_raise(self) -> None:
        """`current_step % 0` would be a `ZeroDivisionError`.

        Not reachable through the schema -- `MetricsConfigSchema` rejects 0
        (pinned in `tests/unit/config/schemas/test_metrics.py`) -- but
        `_get_config_value` also serves raw namespaces and dicts, and the mixin
        is called with those. A non-positive cadence has no periodic firing; the
        two ends still produce data."""
        computer = _FakeComputer()
        s = _make_mixin(training_computer=computer)
        cfg = SimpleNamespace(
            metrics=SimpleNamespace(enable_tracking=True, train_metric_interval=0),
            training=SimpleNamespace(max_iterations=40),
        )
        assert s._compute_training_metrics(
            torch.randn(1, 1, 4, 4),
            torch.randn(1, 1, 4, 4),
            config=cfg,
            current_step=1,
        )
        assert (
            s._compute_training_metrics(
                torch.randn(1, 1, 4, 4),
                torch.randn(1, 1, 4, 4),
                config=cfg,
                current_step=7,
            )
            == {}
        )

    def test_on_interval_returns_train_prefixed_metrics(self) -> None:
        """At step % interval == 0, every metric is forwarded with the
        ``train_`` prefix so downstream tensorboard scalars are
        disambiguated from validation metrics."""
        computer = _FakeComputer(metrics={"psnr": 25.0, "ssim": 0.8})
        s = _make_mixin(
            config=SimpleNamespace(
                data=SimpleNamespace(target_channels=None),
                validation=SimpleNamespace(transform=None, output_transform=None),
            ),
            training_computer=computer,
        )
        cfg = SimpleNamespace(
            metrics=SimpleNamespace(enable_tracking=True, train_metric_interval=5)
        )
        out = s._compute_training_metrics(
            torch.randn(1, 1, 8, 8),
            torch.randn(1, 1, 8, 8),
            config=cfg,
            current_step=10,
        )
        assert out == {"train_psnr": 25.0, "train_ssim": 0.8}
        assert len(computer.compute_calls) == 1

    def test_shape_mismatch_returns_empty(self) -> None:
        """If pred/target shapes disagree and the adapter cannot
        reconcile them, the computation is skipped (returns ``{}``)
        rather than crashing the training step."""
        computer = _FakeComputer()
        s = _make_mixin(
            config=SimpleNamespace(
                data=SimpleNamespace(target_channels=None),
                validation=SimpleNamespace(transform=None, output_transform=None),
            ),
            training_computer=computer,
        )
        cfg = SimpleNamespace(
            metrics=SimpleNamespace(enable_tracking=True, train_metric_interval=1)
        )
        # Different batch sizes — TorchIOAdapter cannot reconcile.
        pred = torch.randn(1, 1, 4, 4)
        target = torch.randn(2, 1, 8, 8)
        out = s._compute_training_metrics(pred, target, config=cfg, current_step=1)
        assert out == {}
        assert computer.compute_calls == []

    def test_dict_pred_unpacks_reconstruction_key(self) -> None:
        """Some generators (DisentangledMRIGenerator) emit dicts —
        the mixin unwraps the ``reconstruction`` entry before metric
        computation."""
        computer = _FakeComputer()
        s = _make_mixin(
            config=SimpleNamespace(
                data=SimpleNamespace(target_channels=None),
                validation=SimpleNamespace(transform=None, output_transform=None),
            ),
            training_computer=computer,
        )
        cfg = SimpleNamespace(
            metrics=SimpleNamespace(enable_tracking=True, train_metric_interval=1)
        )
        recon = torch.randn(1, 1, 4, 4)
        out = s._compute_training_metrics(
            {"reconstruction": recon},
            torch.randn(1, 1, 4, 4),
            config=cfg,
            current_step=1,
        )
        # Compute call was made — the dict was successfully unwrapped.
        assert "train_psnr" in out
        assert len(computer.compute_calls) == 1


# ---------------------------------------------------------------------------
# validation_step
# ---------------------------------------------------------------------------


class _ConstantGenerator(torch.nn.Module):
    """Stub generator that returns the input scaled by a constant."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.eval_calls = 0
        self.train_calls = 0
        # Need a parameter for ``.eval()``/``.train()`` to be meaningful.
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def eval(self) -> "_ConstantGenerator":  # noqa: D401
        self.eval_calls += 1
        return super().eval()

    def train(self, mode: bool = True) -> "_ConstantGenerator":  # noqa: D401
        # nn.Module.eval() is implemented as `self.train(False)`, so an
        # unconditional counter here double-counts: once via eval()'s
        # internal delegation, once via the explicit train() call at the
        # end of validation_step. Only count real "back to train mode"
        # transitions (mode=True), matching what this test actually means
        # to assert.
        if mode:
            self.train_calls += 1
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class TestValidationStep:
    def test_drives_generator_eval_and_back_to_train(self) -> None:
        """The validation step puts the generator into eval mode for
        the forward pass and returns it to train mode afterwards —
        this is how the strategy avoids BatchNorm pollution mid-step."""
        gen = _ConstantGenerator(scale=2.0)
        env = SimpleNamespace(generator=gen)
        computer = _FakeComputer(metrics={"psnr": 31.0})
        s = _make_mixin(
            config=SimpleNamespace(
                data=SimpleNamespace(target_channels=None),
                validation=SimpleNamespace(transform=None, output_transform=None),
            ),
            env=env,
            validation_computer=computer,
        )
        out = s.validation_step(torch.ones(1, 1, 4, 4), torch.full((1, 1, 4, 4), 2.0))
        assert out == {"psnr": 31.0}
        # eval() called once before fwd; train() called once after.
        assert gen.eval_calls == 1
        assert gen.train_calls == 1
        # Compute received pred with same shape as target.
        assert computer.compute_calls == [((1, 1, 4, 4), (1, 1, 4, 4))]

    def test_handles_tuple_output(self) -> None:
        """Some generators return ``(image, aux)`` — the first element
        is the reconstruction."""

        class _TupleGen(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> Any:
                return x, "auxiliary"

        env = SimpleNamespace(generator=_TupleGen())
        computer = _FakeComputer()
        s = _make_mixin(
            config=SimpleNamespace(
                data=SimpleNamespace(target_channels=None),
                validation=SimpleNamespace(transform=None, output_transform=None),
            ),
            env=env,
            validation_computer=computer,
        )
        s.validation_step(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
        assert len(computer.compute_calls) == 1

    def test_handles_dict_output(self) -> None:
        class _DictGen(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"reconstruction": x, "noise": torch.zeros_like(x)}

        env = SimpleNamespace(generator=_DictGen())
        computer = _FakeComputer()
        s = _make_mixin(
            config=SimpleNamespace(
                data=SimpleNamespace(target_channels=None),
                validation=SimpleNamespace(transform=None, output_transform=None),
            ),
            env=env,
            validation_computer=computer,
        )
        s.validation_step(torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
        assert len(computer.compute_calls) == 1


# ---------------------------------------------------------------------------
# Computer proxy methods
# ---------------------------------------------------------------------------


class TestComputerProxies:
    def test_reset_validation_metrics_no_op_when_no_computer(self) -> None:
        s = _make_mixin()
        s.reset_validation_metrics()  # must not raise

    def test_reset_validation_metrics_delegates(self) -> None:
        computer = _FakeComputer()
        s = _make_mixin(validation_computer=computer)
        s.reset_validation_metrics()
        assert computer.reset_calls == 1

    def test_finalize_validation_empty_when_no_computer(self) -> None:
        s = _make_mixin()
        assert s.finalize_validation() == {}

    def test_finalize_validation_delegates(self) -> None:
        computer = _FakeComputer(metrics={"final_psnr": 33.0})
        s = _make_mixin(validation_computer=computer)
        out = s.finalize_validation()
        assert out == {"final_psnr": 33.0}
        assert computer.finalize_calls == 1

    def test_get_primary_metric_delegates(self) -> None:
        computer = _FakeComputer(primary="ssim")
        s = _make_mixin(validation_computer=computer)
        # Bypass the lazy property: get_primary_metric calls
        # `self.validation_metrics_computer.get_primary_metric()`.
        assert s.get_primary_metric() == "ssim"

    def test_is_metric_improvement_delegates(self) -> None:
        computer = _FakeComputer()
        s = _make_mixin(validation_computer=computer)
        assert s.is_metric_improvement(30.0, 25.0) is True
        assert s.is_metric_improvement(20.0, 25.0) is False


# ---------------------------------------------------------------------------
# ifft_sense_adjoint smaps selection — 2026-05-31 exp11 EMA-ablation crash
# ---------------------------------------------------------------------------


class TestIfftSenseAdjointSmapsSelection:
    """Regression for the ``a or b`` tensor-truthiness crash.

    The ``ifft_sense_adjoint`` transform selected its sensitivity maps with
    ``getattr(self, "_cached_smaps", None) or getattr(self, "_current_smaps",
    None)``. ``_cached_smaps`` IS populated (diffusion.py:3123) with a
    multi-element tensor, so the ``or`` evaluated ``bool(tensor)`` and raised
    ``RuntimeError: Boolean value of Tensor with more than one value is
    ambiguous`` — the crash that killed the exp11 EMA-warmup ablation (both
    arms) right after the iter-1000 validation. The fix selects on ``is None``
    via ``pick_present``.

    Second chapter (2026-06-03): selecting on ``is None`` cured the truthiness
    crash but exposed a batch-dimension leak. ``_generate_validation_prediction``
    writes ``_cached_smaps`` with the *validation* batch and never clears it, so
    the next training-metric step (smaller batch) broadcast against the stale
    maps and crashed ``sense_adjoint`` at dim 0 (the iter-1001
    ``tensor a (2) vs b (36)`` failure). Selection now goes through
    ``_select_batch_compatible_smaps``: prefer the per-step ``_current_smaps``,
    accept a map only when its batch is broadcast-compatible, else drop it with
    a loud warning (per-coil iFFT magnitude fallback — CLAUDE.md #9/#10).
    """

    @staticmethod
    def _coil_kspace(coils: int = 4, size: int = 8) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.complex(
            torch.randn(1, coils, size, size), torch.randn(1, coils, size, size)
        )

    def test_multielement_cached_smaps_does_not_raise(self) -> None:
        s = _make_mixin(config=_config_with())
        pred = self._coil_kspace()
        target = self._coil_kspace()
        # The trigger: a present, multi-element _cached_smaps tensor. The old
        # ``_cached_smaps or ...`` would raise here; pick_present must not.
        s._cached_smaps = self._coil_kspace()
        cfg = SimpleNamespace(domain=None, transform="ifft_sense_adjoint")
        out_pred, out_target = s._apply_metric_transforms(pred, target, cfg)
        # SENSE-combined real magnitude, coils collapsed to 1.
        assert torch.is_floating_point(out_pred)
        assert torch.is_floating_point(out_target)
        assert out_pred.shape[1] == 1
        assert torch.isfinite(out_pred).all()

    def test_falls_back_to_current_smaps_when_cached_absent(self) -> None:
        s = _make_mixin(config=_config_with())
        pred = self._coil_kspace()
        target = self._coil_kspace()
        # _cached_smaps absent → must fall back to _current_smaps (the
        # canonical name), not silently degrade to per-coil iFFT.
        s._current_smaps = self._coil_kspace()
        cfg = SimpleNamespace(domain=None, transform="ifft_sense_adjoint")
        out_pred, _ = s._apply_metric_transforms(pred, target, cfg)
        assert out_pred.shape[1] == 1
        assert torch.isfinite(out_pred).all()

    def test_missing_maps_on_multicoil_data_is_announced_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """PLANTED: no maps at all fell back to per-coil magnitudes without a word."""
        s = _make_mixin(config=_config_with())
        s._current_smaps = s._cached_smaps = None
        cfg = SimpleNamespace(domain=None, transform="ifft_sense_adjoint")
        with caplog.at_level(logging.WARNING):
            for _ in range(2):
                s._apply_metric_transforms(self._coil_kspace(), self._coil_kspace(), cfg)
        assert caplog.text.count("no sensitivity maps for a 4-coil prediction") == 1

    def test_bare_or_on_multielement_tensor_is_the_documented_failure(self) -> None:
        # Documents the exact pre-fix failure mode the production code hit.
        smaps = self._coil_kspace()
        with pytest.raises(RuntimeError, match="ambiguous"):
            _ = bool(smaps)

    def test_stale_validation_batch_smaps_does_not_crash(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A validation-scoped ``_cached_smaps`` whose batch does not match the
        current prediction must NOT crash the training-metric step (the
        iter-1001 ``tensor a (2) vs b (36)`` failure). The mixin drops the
        incompatible maps, falls back to per-coil iFFT magnitude, and warns
        loudly (no silent degrade — CLAUDE.md #9/#10)."""
        s = _make_mixin(config=_config_with())
        # No per-step maps; only the leaked validation-batch maps (batch 6 ≠ 2).
        s._current_smaps = None
        s._cached_smaps = torch.randn(6, 2, 8, 8)
        pred = torch.randn(2, 2, 8, 8)
        target = torch.randn(2, 2, 8, 8)
        cfg = SimpleNamespace(domain=None, transform="ifft_sense_adjoint")
        with caplog.at_level(logging.WARNING):
            out_pred, out_target = s._apply_metric_transforms(pred, target, cfg)
        assert out_pred.shape[0] == 2  # batch preserved, no crash
        assert out_target.shape[0] == 2
        assert torch.is_floating_point(out_pred)
        assert (out_pred >= 0).all()
        assert "smaps batch" in caplog.text  # the drop is announced

    def test_prefers_current_smaps_over_stale_cached(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When per-step ``_current_smaps`` matches the prediction batch it is
        used for the SENSE combine even if a stale validation ``_cached_smaps``
        is also present — so SENSE training metrics survive the
        validation→training boundary rather than being disabled."""
        s = _make_mixin(config=_config_with())
        s._current_smaps = torch.randn(2, 2, 8, 8)  # current training batch
        s._cached_smaps = torch.randn(6, 2, 8, 8)  # stale validation batch
        pred = torch.randn(2, 2, 8, 8)
        target = torch.randn(2, 2, 8, 8)
        cfg = SimpleNamespace(domain=None, transform="ifft_sense_adjoint")
        with caplog.at_level(logging.WARNING):
            out_pred, out_target = s._apply_metric_transforms(pred, target, cfg)
        # SENSE combine over coils → single-channel magnitude, batch preserved.
        assert out_pred.shape == (2, 1, 8, 8)
        assert out_target.shape == (2, 1, 8, 8)
        assert (out_pred >= 0).all()
        assert "smaps batch" not in caplog.text  # compatible map → no warning


class TestValidationImageKspaceVeto:
    """The validation image saver (``_to_vis`` with ``force_ifft=True``) must not
    IFFT an already-image target into a k-space DC blob (smoke 2026-06-15
    exp_p7 / exp_p3 real-images). ``_to_vis`` is a nested closure, so this pins
    the shared decision logic it applies — the ``looks_like_kspace`` veto on the
    re-paired complex candidate — for both the image and the k-space case."""

    @staticmethod
    def _to_vis_force_ifft(t):
        from spectramr.infrastructure.physics.fft_ops import ifft2c
        from spectramr.infrastructure.training.utils.domain_inference import (
            looks_like_kspace,
        )

        if torch.is_complex(t):
            kcand = t
        else:
            kcand = torch.stack(
                [torch.complex(t[:, i], t[:, i + 1]) for i in range(0, t.shape[1], 2)],
                dim=1,
            )
        img = ifft2c(kcand) if looks_like_kspace(kcand.abs()) else kcand
        mag = torch.abs(img)
        if mag.shape[1] > 1:
            mag = torch.sqrt((mag**2).sum(dim=1, keepdim=True))
        return mag

    @staticmethod
    def _dc_blob_score(m):
        c = m[0, 0, 60:68, 60:68].mean()
        return float(c / (m.mean() + 1e-6))

    def _brain(self):
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 128), torch.linspace(-1, 1, 128), indexing="ij"
        )
        return (((xx**2 + yy**2) < 0.55).float() * 8.0).unsqueeze(0).unsqueeze(0)

    def test_image_target_not_ifftd_to_blob(self) -> None:
        brain = self._brain().to(torch.complex64)
        img_rs = torch.cat([brain.real, brain.imag], dim=1)  # already image domain
        out = self._to_vis_force_ifft(img_rs)
        assert self._dc_blob_score(out) < 4.0  # stays a brain, NOT a DC blob

    def test_kspace_target_still_ifftd(self) -> None:
        from spectramr.infrastructure.physics.fft_ops import fft2c

        ksp = fft2c(self._brain().to(torch.complex64))
        ksp_rs = torch.cat([ksp.real, ksp.imag], dim=1)
        out = self._to_vis_force_ifft(ksp_rs)
        assert self._dc_blob_score(out) < 4.0  # IFFT recovers the brain
