"""Regression tests for ModelValidationMixin.validation_step.

The load-bearing case here is the `batch_context` re-forwarding bug: line 95 calls
``self._validation_forward(input_batch, batch_context, **kwargs)``. If
``validation_step`` only ``.get``s ``batch_context`` (instead of ``.pop``), a caller
that supplies ``batch_context=`` (the cross-field / field-flow / field-bridge
``validation_step`` overrides do) leaves it in ``**kwargs`` too, so it is passed both
positionally and by keyword -> ``TypeError: ... multiple values for argument
'batch_context'``. This crashed EVERY validation pass for those three strategies, and
their per-strategy tests masked it by calling ``_validation_forward`` directly. This
test drives the FULL ``validation_step`` chain so it can never regress silently again.
"""

from __future__ import annotations

import types

from tests.utils.block_config_stub import ValidationConfigStub

import torch

from spectramr.infrastructure.training.strategies.mixins.model_validation import (
    ModelValidationMixin,
)
from spectramr.infrastructure.training.utils.metric_views import MetricViews


class _Stub(ModelValidationMixin):
    """Minimal carrier: image metrics are turned off so we exercise only the
    unpack -> batch_context -> _validation_forward path that holds the bug."""

    def __init__(self) -> None:
        self.generator_model = types.SimpleNamespace(eval=lambda: None)
        self.state = types.SimpleNamespace(
            config=types.SimpleNamespace(
                training=None,
                validation=ValidationConfigStub(scoring={"enable_image_metrics": False}),
            )
        )
        self.received: dict | None = None

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        self.received = {"bc": batch_context, "kw": dict(kwargs)}
        return torch.zeros_like(input_batch)


def test_validation_step_pops_batch_context_no_double_pass() -> None:
    s = _Stub()
    out = s.validation_step(
        torch.zeros(1, 1, 4, 4),
        torch.zeros(1, 1, 4, 4),
        batch_context={"k": 1},
        field_strength=torch.tensor([1.0]),
    )
    # compute_image_metrics is False -> no metric machinery -> {} (and no crash).
    assert out == {}
    # batch_context reached _validation_forward exactly once (positionally)...
    assert s.received is not None
    assert s.received["bc"] == {"k": 1}
    # ...and was NOT also left in **kwargs (the double-pass that raised TypeError).
    assert "batch_context" not in s.received["kw"]
    # unrelated kwargs (per-sample fields) still flow through.
    assert "field_strength" in s.received["kw"]


def test_validation_step_without_batch_context_uses_default() -> None:
    # The .get -> .pop change must not regress the common no-batch_context path:
    # the mixin synthesises a default context.
    s = _Stub()
    s.validation_step(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    assert s.received is not None
    assert s.received["bc"] == {"use_dc": False, "measured_kspace": None}


class _MetricStub(ModelValidationMixin):
    """Image metrics ON but no computer wired, so validation_step skips the metric
    machinery and falls through to the val_loss emission we exercise here."""

    def __init__(self) -> None:
        self.generator_model = types.SimpleNamespace(eval=lambda: None)
        self.state = types.SimpleNamespace(
            config=types.SimpleNamespace(
                training=None,
                validation=ValidationConfigStub(scoring={"enable_image_metrics": True}),
            )
        )
        self.validation_metrics_computer = None  # -> image-metric block is skipped

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        return torch.zeros_like(input_batch) + 0.5


def test_validation_step_emits_val_loss() -> None:
    # fig_1_2_learning_curves needs a validation loss series; the mixin emits one
    # (magnitude-matched L1 on pred/target) so the curve is no longer train-only.
    out = _MetricStub().validation_step(torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
    assert "val_loss" in out
    assert abs(out["val_loss"] - 0.5) < 1e-6  # L1(0.5, 1.0) over a constant field


def test_val_loss_absent_when_image_metrics_disabled() -> None:
    # When validation metric computation is turned off, no val_loss is emitted
    # (keeps the compute_image_metrics=False contract -> empty metrics).
    assert _Stub().validation_step(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)) == {}


def test_empty_computer_scores_nothing_and_keeps_val_loss() -> None:
    """#2839: ``compute: [val_loss]`` (all strategy-emitted) asks for no registry metric.

    The transform raising stands in for an output that is not an image; the
    image-metric block used to reach the fallback PSNR on it.
    """
    from spectramr.core.metrics.computer import ValidationMetricsComputer
    from spectramr.core.metrics.types import ValidationMetricsConfig

    s = _MetricStub()
    s.validation_metrics_computer = ValidationMetricsComputer(
        ValidationMetricsConfig.from_dict({"metrics": [], "primary_metric": "val_loss"})
    )

    def _not_an_image(*args, **kwargs):
        raise RuntimeError("no image-domain view of this output")

    s._metric_transform_views = _not_an_image
    out = s.validation_step(torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
    assert set(out) == {"val_loss"}


class _ImageVsCoilKspaceStub(_MetricStub):
    """An image network (2 channels) graded against 4-coil k-space (8 channels)."""

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        return torch.full((input_batch.shape[0], 2, 4, 4), 0.5)


def test_val_loss_grades_an_image_against_the_targets_image_not_channel_0() -> None:
    """Planted: the old collapse compared |x_hat| with Re(k_coil0), a number that rose
    with any change in the image's background and described no objective."""
    from spectramr.infrastructure.training.utils.metric_transform import (
        image_magnitude,
        kspace_rss_magnitude,
    )

    s = _ImageVsCoilKspaceStub()
    s._metric_transform_views = lambda pred, target, cfg, **_: MetricViews(
        image_magnitude(pred), kspace_rss_magnitude(target)
    )
    target = torch.randn(1, 8, 4, 4, generator=torch.Generator().manual_seed(0))
    out = s.validation_step(torch.zeros(1, 8, 4, 4), target)
    pred = torch.full((1, 2, 4, 4), 0.5)
    expected = torch.nn.functional.l1_loss(image_magnitude(pred), kspace_rss_magnitude(target))
    channel_0 = torch.nn.functional.l1_loss(image_magnitude(pred), target[:, :1])
    assert abs(float(out["val_loss"]) - float(expected)) < 1e-6
    assert abs(float(out["val_loss"]) - float(channel_0)) > 1e-3


def test_val_loss_is_skipped_when_no_transform_places_both() -> None:
    s = _ImageVsCoilKspaceStub()
    s._metric_transform_views = lambda pred, target, cfg, **_: MetricViews(pred, target)
    out = s.validation_step(torch.zeros(1, 8, 4, 4), torch.ones(1, 8, 4, 4))
    assert "val_loss" not in out


class _DistHeadStub(ModelValidationMixin):
    """A heteroscedastic strategy: the model emits ``[mean, logvar]``, not real/imag.

    ``predicts_distribution_params`` is the existing declaration (``base.py``;
    ``heteroscedastic_ulf_strategy.py`` sets it True) that the two channel-collapse
    fallbacks never consulted.
    """

    def __init__(self, dist_head: bool) -> None:
        self.predicts_distribution_params = dist_head
        self.generator_model = types.SimpleNamespace(eval=lambda: None)
        self.state = types.SimpleNamespace(
            config=types.SimpleNamespace(
                training=None,
                validation=ValidationConfigStub(scoring={"enable_image_metrics": True}),
            )
        )
        self.validation_metrics_computer = None

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        b, _, h, w = input_batch.shape
        mean = torch.full((b, 1, h, w), 0.5)
        logvar = torch.full((b, 1, h, w), -4.0)  # near-constant, as logvar is
        return torch.cat([mean, logvar], dim=1)


def _val_loss(dist_head: bool) -> float:
    s = _DistHeadStub(dist_head)
    target = torch.full((1, 1, 4, 4), 0.5)
    out = s.validation_step(torch.zeros(1, 1, 4, 4), target)
    return float(out["val_loss"])


def test_distribution_head_is_not_rss_combined() -> None:
    """[mean, logvar] must collapse to the MEAN, never sqrt(mean^2 + logvar^2).

    RSS-ing a distribution head mixes an intensity with a log-variance. Because
    logvar is near-constant across the image, the result is dragged toward an
    input-independent field — the b29 wrong-channel metric, which reported a
    large val_loss for a prediction that exactly matched its target.
    """
    # mean == target exactly, so a correct collapse gives ~0 loss.
    assert _val_loss(dist_head=True) < 1e-6, (
        "distribution head was RSS-combined: sqrt(0.5^2 + (-4)^2) ~= 4.03 is graded "
        "against a target of 0.5, so a perfect prediction scores as a large error"
    )


def test_complex_two_channel_output_still_rss_combines() -> None:
    """The guard must not disturb genuine real/imag outputs."""
    got = _val_loss(dist_head=False)
    expected = abs((0.5**2 + 4.0**2) ** 0.5 - 0.5)
    assert got > 1.0, f"real/imag path must still RSS; got {got}"
    assert abs(got - expected) < 0.5, f"expected ~{expected}, got {got}"


# ---------------------------------------------------------------------------
# One `val_psnr`, not two (issue #179 / #180 follow-through).
#
# When the metrics computer raised, this mixin recomputed `val_psnr` from
# `target.max() - target.min()` -- a PER-IMAGE extent -- and wrote it under the
# SAME key the primary path uses. So one key carried two incomparable definitions,
# selected by an exception nobody observes, and early stopping plus
# `final_metrics.best` consumed it as if it were one quantity.
# ---------------------------------------------------------------------------


def test_fallback_no_longer_derives_its_own_data_range():
    """Source guard: the per-image `max - min` derivation must not come back.

    Pinned at the source because the defect is a *definition* divergence — two
    numeric paths that agree on the fixtures you happen to pick and diverge on a
    dark slice. A value assertion alone would not have caught the original bug.
    """
    import inspect

    from spectramr.infrastructure.training.strategies.mixins import model_validation

    src = inspect.getsource(model_validation)
    assert "resolve_image_data_range" in src, (
        "the fallback must share the registered psnr's data_range resolver"
    )
    assert ".max().item() - " not in src, (
        "a per-image `max - min` data_range reappeared in the fallback; it makes "
        "val_psnr mean something different depending on which branch produced it"
    )


def test_fallback_and_canonical_psnr_agree_on_a_dark_slice():
    """The case that separates the two definitions.

    A slice spanning [0, 0.3] scored ~10 dB better under the per-image extent than
    under the normalization contract — the exact "flattered dark image" failure.
    """
    import math

    from spectramr.core.metrics.evaluation_metrics import (
        PSNR,
        resolve_image_data_range,
    )

    gen = torch.Generator().manual_seed(0)
    target = torch.rand(1, 1, 32, 32, generator=gen) * 0.3
    pred = (target + 0.02 * torch.randn(1, 1, 32, 32, generator=gen)).clamp(0, 0.3)

    canonical = float(PSNR()(pred, target))
    mse = torch.nn.functional.mse_loss(pred, target).item()
    fallback = 10.0 * math.log10(
        resolve_image_data_range(target, None, metric_name="val_psnr") ** 2 / mse
    )
    assert abs(canonical - fallback) < 1e-3, (
        f"fallback {fallback:.3f} dB disagrees with canonical {canonical:.3f} dB"
    )

    # And the old derivation is what it must NOT reproduce.
    old = 10.0 * math.log10(
        (target.max().item() - target.min().item()) ** 2 / mse
    )
    assert abs(old - canonical) > 5.0, (
        "fixture no longer separates the two definitions — pick a darker slice"
    )


def test_fallback_declines_rather_than_inventing_on_unnormalized_data():
    from spectramr.core.metrics.evaluation_metrics import resolve_image_data_range
    from spectramr.core.metrics.outcome import MetricNotApplicableError

    import pytest

    unnormalized = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(0)) * 2479
    with pytest.raises(MetricNotApplicableError):
        resolve_image_data_range(unnormalized, None, metric_name="val_psnr")


# ---------------------------------------------------------------------------
# The metric-context pass and the coil-manifold projection (NuCert / Rep-BV).
#
# Driven through the real ``validation_step`` with a spy model and the real
# ``ValidationMetricsComputer``: the defects these guard are wiring defects --
# a batch field that reaches the step and stops, a projection declared and never
# applied -- which a test of the metric alone cannot see.
# ---------------------------------------------------------------------------

_C, _N = 4, 16


def _complex(*shape: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=gen), torch.randn(*shape, generator=gen))


def _coil_batch(*, with_maps: bool = True, with_alt: bool = False):
    """A validation ``TrainingBatch`` as the loop delivers it: 5-D extras, 4-D tensors."""
    from spectramr.core.metrics.coil_manifold import as_interleaved
    from spectramr.data.batch_types import BatchAdapter

    maps = _complex(2, _C, _N, _N, 1, seed=0)
    maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    collated = {
        "input": as_interleaved(_complex(2, _C, _N, _N, seed=1)),
        "target": as_interleaved(_complex(2, _C, _N, _N, seed=2)),
    }
    if with_maps:
        collated["sensitivity_complex"] = maps
        collated["sensitivity"] = maps.abs()  # the magnitude spelling, bound to coil_maps
    if with_alt:
        collated["input_alt"] = as_interleaved(_complex(2, _C, _N, _N, seed=3))
    return BatchAdapter.from_dict(collated), maps[..., 0]


class _SpyModel:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x)
        return 0.5 * x


class _CoilStub(ModelValidationMixin):
    """Real validation_step, real computer; the metric transform is a recorder."""

    def __init__(
        self,
        compute: list[str],
        *,
        project: bool = False,
        recorrupted: dict | None = None,
        target_mode: str = "phase_aligned_mean",
    ) -> None:
        from spectramr.core.metrics.computer import ValidationMetricsComputer
        from spectramr.core.metrics.types import ValidationMetricsConfig

        self.generator_model = types.SimpleNamespace(eval=lambda: None)
        self.model = _SpyModel()
        self.logging_service = None
        self.graded: list[torch.Tensor] = []
        scoring = {"compute": compute, "coil_manifold_projection": project}
        if recorrupted is not None:
            scoring["recorrupted_inference"] = recorrupted
        self.state = types.SimpleNamespace(
            config=types.SimpleNamespace(
                training=None,
                model=types.SimpleNamespace(
                    model_type="complex_unet", input_type="kspace", output_type="kspace"
                ),
                validation=ValidationConfigStub(scoring=scoring),
                data=types.SimpleNamespace(
                    target_mode=target_mode,
                    r2r_alpha=1.0,
                    r2r_covariance_source="committed",
                    r2r_split_angle_deg=45.0,
                    r2r_whitening="committed",
                ),
                run=types.SimpleNamespace(seed=3),
            )
        )
        self.validation_metrics_computer = ValidationMetricsComputer(
            ValidationMetricsConfig.from_dict({"metrics": compute})
        )

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        return self.model(input_batch)

    def _metric_transform_views(self, pred, target, cfg, *, batch_data=None):
        self.graded.append(pred)
        return MetricViews(pred, target)


class TestTheLoopForwardsTheBatch:
    def test_the_real_selector_hands_the_mixin_its_batch(self) -> None:
        """``select_validation_extra_fields`` keys on the declared parameter names."""
        import inspect

        from spectramr.pipelines.train import select_validation_extra_fields

        stub = _CoilStub(["mse"])
        params = list(inspect.signature(stub.validation_step).parameters)
        assert params[:2] == ["input_batch", "target_batch"]  # still Pattern A
        batch, _ = _coil_batch()
        assert select_validation_extra_fields(batch, params)["batch_data"] is batch


class TestTheProjectionIsApplied:
    def test_on_the_graded_prediction_is_the_projected_one(self) -> None:
        from spectramr.core.metrics.coil_manifold import project_kspace_onto_manifold

        batch, maps = _coil_batch()
        stub = _CoilStub(["mse"], project=True)
        stub.validation_step(batch.input, batch.target, batch_data=batch)
        raw = 0.5 * batch.input
        assert torch.allclose(stub.graded[0], project_kspace_onto_manifold(raw, maps), atol=1e-5)
        assert not torch.allclose(stub.graded[0], raw, atol=1e-3)

    def test_off_the_graded_prediction_is_the_raw_one(self) -> None:
        batch, _ = _coil_batch()
        stub = _CoilStub(["mse"], project=False)
        stub.validation_step(batch.input, batch.target, batch_data=batch)
        assert torch.equal(stub.graded[0], 0.5 * batch.input)

    def test_on_without_maps_raises(self) -> None:
        """Planted: a projection silently skipped would report the raw PSNR as projected."""
        import pytest

        batch, _ = _coil_batch(with_maps=False)
        with pytest.raises(ValueError, match="carries no coil maps"):
            _CoilStub(["mse"], project=True).validation_step(
                batch.input, batch.target, batch_data=batch
            )


class TestTheContextPass:
    def test_a_context_metric_reads_the_raw_prediction_with_the_complex_maps(self) -> None:
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry

        batch, maps = _coil_batch()
        out = _CoilStub(["null_space_residual", "mse"], project=True).validation_step(
            batch.input, batch.target, batch_data=batch
        )
        expected = MetricsRegistry.get("null_space_residual")(
            0.5 * batch.input, context=MetricContext(coil_maps=maps)
        )
        assert out["val_null_space_residual"] == pytest_approx(expected.item())
        assert "val_mse" in out  # the plain pass still ran, without the context metric

    def test_a_missing_map_fails_the_batch_instead_of_degrading(self) -> None:
        """Planted: outside the fallback `try`, so no psnr-only row stands in for it."""
        import pytest

        batch, _ = _coil_batch(with_maps=False)
        with pytest.raises(ValueError, match="null_space_residual"):
            _CoilStub(["null_space_residual", "mse"]).validation_step(
                batch.input, batch.target, batch_data=batch
            )

    def test_pred_alt_is_the_model_run_on_input_alt(self) -> None:
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry

        batch, _ = _coil_batch(with_alt=True)
        stub = _CoilStub(["rep_variance"])
        out = stub.validation_step(batch.input, batch.target, batch_data=batch)
        alt = batch.metadata["input_alt"]
        assert any(torch.equal(x, alt) for x in stub.model.inputs), "the model never saw input_alt"
        expected = MetricsRegistry.get("rep_variance")(
            0.5 * batch.input, context=MetricContext(pred_alt=0.5 * alt)
        )
        assert out["val_rep_variance"] == pytest_approx(expected.item())

    def test_no_second_forward_when_nothing_reads_it(self) -> None:
        batch, _ = _coil_batch(with_alt=True)
        stub = _CoilStub(["null_space_residual"])
        stub.validation_step(batch.input, batch.target, batch_data=batch)
        assert len(stub.model.inputs) == 1

    def test_the_sense_columns_sit_beside_the_rss_ones(self) -> None:
        """``psnr`` keeps its transform; ``sense_psnr`` is scored on the batch's maps."""
        from spectramr.core.metrics.computer import ValidationMetricsComputer
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry
        from spectramr.core.metrics.types import ValidationMetricsConfig

        batch, maps = _coil_batch()
        stub = _CoilStub(["psnr", "sense_psnr", "sense_ssim"])
        stub.validation_metrics_computer = ValidationMetricsComputer(
            ValidationMetricsConfig.from_dict({"metrics": ["psnr", "sense_psnr", "sense_ssim"]}),
            data_range=1.0,
        )
        out = stub.validation_step(batch.input, batch.target, batch_data=batch)
        expected = MetricsRegistry.get("sense_psnr")(
            0.5 * batch.input, batch.target, context=MetricContext(coil_maps=maps), data_range=1.0
        )
        assert {"val_psnr", "val_sense_psnr", "val_sense_ssim"} <= set(out)
        assert out["val_sense_psnr"] == pytest_approx(expected.item())
        assert out["val_sense_psnr"] != pytest_approx(out["val_psnr"])

    def test_a_sense_column_without_maps_fails_the_batch(self) -> None:
        """Planted: a one-virtual-coil arm declaring the column has no maps to read."""
        import pytest

        batch, _ = _coil_batch(with_maps=False)
        # The computer rewraps a metric's ValueError with its name; it still raises.
        with pytest.raises(ValueError, match="sense_psnr needs coil sensitivity maps"):
            _CoilStub(["psnr", "sense_psnr"]).validation_step(
                batch.input, batch.target, batch_data=batch
            )

    def test_a_variance_metric_without_input_alt_raises(self) -> None:
        import pytest

        batch, _ = _coil_batch(with_alt=False)
        with pytest.raises(ValueError, match="emit_alternate_input"):
            _CoilStub(["rep_variance"]).validation_step(batch.input, batch.target, batch_data=batch)


class _SenseTransformStub(_CoilStub):
    """``_CoilStub`` with the REAL metric transform, declared ``sense_magnitude``."""

    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
        MetricsMixin as _Mixin,
    )

    _metric_transform_views = _Mixin._metric_transform_views
    _apply_metric_transforms = _Mixin._apply_metric_transforms
    _log_metric_transform_once = _Mixin._log_metric_transform_once
    del _Mixin

    def __init__(self) -> None:
        from spectramr.core.metrics.computer import ValidationMetricsComputer
        from spectramr.core.metrics.types import ValidationMetricsConfig

        super().__init__(["psnr"])
        self.state.config.validation = ValidationConfigStub(
            scoring={"compute": ["psnr"], "output_transform": "sense_magnitude"}
        )
        self.validation_metrics_computer = ValidationMetricsComputer(
            ValidationMetricsConfig.from_dict({"metrics": ["psnr"]}), data_range=1.0
        )


class TestTheSenseTransformOnTheProductionStep:
    """``output_transform: sense_magnitude`` through ``validation_step`` itself."""

    def test_val_psnr_is_the_sense_psnr_and_not_the_rss_one(self) -> None:
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry
        from spectramr.infrastructure.training.utils.metric_transform import (
            kspace_rss_magnitude,
        )

        batch, maps = _coil_batch()
        out = _SenseTransformStub().validation_step(batch.input, batch.target, batch_data=batch)
        prediction = 0.5 * batch.input
        sense = MetricsRegistry.get("sense_psnr")(
            prediction, batch.target, context=MetricContext(coil_maps=maps), data_range=1.0
        )
        rss = MetricsRegistry.get("psnr")(
            kspace_rss_magnitude(prediction), kspace_rss_magnitude(batch.target), data_range=1.0
        )
        assert out["val_psnr"] == pytest_approx(sense.item())
        assert out["val_psnr"] != pytest_approx(rss.item())
        assert "val_zf_psnr" in out  # the baseline is handed the batch's maps too

    def test_without_maps_it_raises_past_the_fallback(self) -> None:
        """Planted: the fallback would otherwise leave a NaN val_psnr row."""
        import pytest

        from spectramr.core.metrics.sense_image_metrics import SenseMapsMissingError

        batch, _ = _coil_batch(with_maps=False)
        with pytest.raises(SenseMapsMissingError):
            _SenseTransformStub().validation_step(batch.input, batch.target, batch_data=batch)


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, rel=1e-5)


def _r2r_batch(*, with_alt: bool = False, with_maps: bool = False):
    """``_coil_batch`` plus the per-sample ``kspace_scale`` the loader publishes."""
    from spectramr.data.batch_types import BatchAdapter

    batch, maps = _coil_batch(with_alt=with_alt, with_maps=with_maps)
    fields = {"input": batch.input, "target": batch.target, **batch.metadata}
    fields["kspace_scale"] = torch.full((batch.input.shape[0],), 25.0)
    return BatchAdapter.from_dict(fields), maps


class TestRecorruptedInferenceIsTheScoredEstimator:
    """``validation.scoring.recorrupted_inference`` on the production ``validation_step``."""

    def test_the_graded_output_is_the_mean_over_j_forwards(self) -> None:
        batch, _ = _r2r_batch()
        stub = _CoilStub(["mse"], recorrupted={"samples": 5}, target_mode="r2r")
        stub.validation_step(batch.input, batch.target, batch_data=batch)
        fed = stub.model.inputs
        assert len(fed) == 5 and all(not torch.equal(x, batch.input) for x in fed)
        torch.testing.assert_close(stub.graded[0], 0.5 * torch.stack(fed).mean(0))

    def test_off_the_model_runs_once_on_the_raw_input(self) -> None:
        batch, _ = _r2r_batch()
        stub = _CoilStub(["mse"], target_mode="r2r")
        stub.validation_step(batch.input, batch.target, batch_data=batch)
        assert len(stub.model.inputs) == 1 and torch.equal(stub.model.inputs[0], batch.input)

    def test_pred_alt_is_the_same_estimator_on_input_alt(self) -> None:
        """rep_variance must measure the estimator that is reported, not the bare model."""
        batch, _ = _r2r_batch(with_alt=True)
        stub = _CoilStub(["rep_variance"], recorrupted={"samples": 3}, target_mode="r2r")
        stub.validation_step(batch.input, batch.target, batch_data=batch)
        assert len(stub.model.inputs) == 6
        alt = batch.metadata["input_alt"]
        assert all(not torch.equal(x, alt) for x in stub.model.inputs[3:])

    def test_the_average_is_taken_before_the_projection(self) -> None:
        from spectramr.core.metrics.coil_manifold import project_kspace_onto_manifold

        batch, maps = _r2r_batch(with_maps=True)
        stub = _CoilStub(["mse"], project=True, recorrupted={"samples": 2}, target_mode="r2r")
        stub.validation_step(batch.input, batch.target, batch_data=batch)
        mean = 0.5 * torch.stack(stub.model.inputs).mean(0)
        torch.testing.assert_close(
            stub.graded[0], project_kspace_onto_manifold(mean, maps), atol=1e-5, rtol=1e-5
        )

    def test_refused_on_an_arm_that_never_trained_on_recorruptions(self) -> None:
        import pytest

        batch, _ = _r2r_batch()
        stub = _CoilStub(["mse"], recorrupted={"samples": 5}, target_mode="rep_pair")
        with pytest.raises(ValueError, match="only an R2R-trained network"):
            stub.validation_step(batch.input, batch.target, batch_data=batch)

    def test_every_epoch_scores_a_batch_under_the_same_draws(self) -> None:
        batch, _ = _r2r_batch()
        stub = _CoilStub(["mse"], recorrupted={"samples": 2}, target_mode="r2r")
        for _ in range(2):
            stub.validation_step(batch.input, batch.target, batch_data=batch, batch_idx=0)
        epoch_1, epoch_2 = stub.model.inputs[:2], stub.model.inputs[2:]
        assert all(torch.equal(a, b) for a, b in zip(epoch_1, epoch_2, strict=True))

    def test_it_announces_itself_once(self, caplog) -> None:
        import logging

        batch, _ = _r2r_batch()
        stub = _CoilStub(["mse"], recorrupted={"samples": 2}, target_mode="r2r")
        with caplog.at_level(logging.INFO):
            for _ in range(3):
                stub.validation_step(batch.input, batch.target, batch_data=batch)
        assert sum("recorrupted_inference: ON" in r.message for r in caplog.records) == 1


def test_a_config_without_a_validation_block_scores_the_bare_model() -> None:
    """Planted: resolving the knob off ``config.validation is None`` raised
    AttributeError in every smoke arm without a validation block."""
    batch, _ = _coil_batch()
    stub = _CoilStub(["mse"])
    stub.state.config.validation = None
    stub.validation_step(batch.input, batch.target, batch_data=batch)
    assert len(stub.model.inputs) == 1 and torch.equal(stub.model.inputs[0], batch.input)


class TestTheProjectionReachesTheRepBVMetrics:
    """Arm F's question is read on the decomposition of the PROJECTED output: the
    certificate stays on the raw output, everything else moves with the knob."""

    _METRICS = ("null_space_residual", "rep_variance", "nex_equivalent")

    def _scores(self, project: bool) -> dict:
        batch, _ = _r2r_batch(with_maps=True, with_alt=True)
        stub = _CoilStub(list(self._METRICS), project=project, target_mode="r2r")
        return stub.validation_step(batch.input, batch.target, batch_data=batch)

    def test_the_certificate_reads_the_raw_output_either_way(self) -> None:
        off, on = self._scores(False), self._scores(True)
        assert on["val_null_space_residual"] == off["val_null_space_residual"] > 0

    def test_the_decomposition_and_pred_alt_read_the_projected_output(self) -> None:
        off, on = self._scores(False), self._scores(True)
        assert on["val_nex_equivalent"] != off["val_nex_equivalent"]
        # rep_variance compares pred to pred_alt: a projected pred against an
        # unprojected pred_alt would move by the null-space energy of only one side.
        assert on["val_rep_variance"] < off["val_rep_variance"]


def test_the_zero_filled_baseline_reports_ssim_beside_psnr() -> None:
    """The self-supervised arms select on val_ssim and read it against the
    zero-filled adjoint, so the baseline must carry that column too."""
    from spectramr.core.metrics.computer import ValidationMetricsComputer
    from spectramr.core.metrics.types import ValidationMetricsConfig

    computer = ValidationMetricsComputer(ValidationMetricsConfig.from_dict({"metrics": ["psnr", "ssim"]}))
    stub = _CoilStub(["psnr", "ssim"])
    torch.manual_seed(0)
    target = torch.rand(1, 1, 32, 32)
    baseline = stub._zero_filled_baseline_metrics(
        target + 0.1 * torch.randn_like(target), target, computer, None, None
    )
    assert {"val_zf_psnr", "val_zf_ssim"} <= set(baseline)


# ---------------------------------------------------------------------------
# The fallback PSNR reads the declared range and scale contract the computer was
# built with (#2241). It used to pass ``None``, so on a declared-range arm the
# fallback re-ran the sign test the declaration exists to replace.
# ---------------------------------------------------------------------------


class _FallbackStub(_CoilStub):
    """The computer fails after the (identity) transform, so ``val_psnr`` comes from the fallback."""

    def __init__(self, prediction: torch.Tensor, **computer_kwargs) -> None:
        from spectramr.core.metrics.computer import create_validation_metrics_computer

        super().__init__(["psnr"])
        computer = create_validation_metrics_computer(["psnr"], **computer_kwargs)

        def _fail(*args, **kwargs):
            raise RuntimeError("computer failed; the fallback scores the batch")

        computer.compute = _fail
        self.validation_metrics_computer = computer
        self._prediction = prediction

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        return self._prediction


def _p99_slice(scale: float = 1.0) -> torch.Tensor:
    """p99 = 1.0 with a 2.05 peak: the M4Raw shape the peak test reports as N/A."""
    image = torch.zeros(1, 1, 64, 64)
    image[..., 8:56, 8:56] = 1.0
    image[..., 20:23, 30:33] = 2.05
    return image * scale


class TestTheFallbackReadsTheComputersRange:
    def _fallback(self, target: torch.Tensor, **computer_kwargs) -> dict:
        prediction = target * 0.98
        return _FallbackStub(prediction, **computer_kwargs).validation_step(prediction, target)

    def _expected(self, target: torch.Tensor, data_range: float) -> float:
        import math

        mse = torch.nn.functional.mse_loss(target * 0.98, target).item()
        return 10.0 * math.log10(data_range**2 / mse)

    def test_without_a_declaration_it_declines_as_before(self) -> None:
        assert "val_psnr" not in self._fallback(_p99_slice())

    def test_the_declared_range_reaches_it(self) -> None:
        target = _p99_slice()
        got = self._fallback(target, data_range=1.0)["val_psnr"]
        assert abs(got - self._expected(target, 1.0)) < 1e-4

    def test_the_scale_contract_reaches_it(self) -> None:
        from spectramr.core.metrics.data_range_contract import QuantileScaleContract

        target = _p99_slice()
        contract = QuantileScaleContract(quantile=0.99, source="test")
        got = self._fallback(target, scale_contract=contract)["val_psnr"]
        assert abs(got - self._expected(target, 1.0)) < 1e-4

    def test_a_contradicted_contract_is_raised_not_logged(self) -> None:
        import pytest

        from spectramr.core.metrics.data_range_contract import (
            DataRangeContractError,
            QuantileScaleContract,
        )

        contract = QuantileScaleContract(quantile=0.99, source="test")
        with pytest.raises(DataRangeContractError):
            self._fallback(_p99_slice(scale=2400.0), scale_contract=contract)

# ---------------------------------------------------------------------------
# The context-metric adapter and the pred_alt guard (#2509).
#
# An image network (EI, SSDU) emits one SENSE-combined image; Rep-BV reads coil
# k-space. Non-square slices, so a transposed expansion would not line up.
# ---------------------------------------------------------------------------

_H2, _W2 = 12, 20


def _image_arm_batch(*, with_alt: bool = False):
    from spectramr.core.metrics.coil_manifold import as_interleaved
    from spectramr.data.batch_types import BatchAdapter

    maps = _complex(2, _C, _H2, _W2, 1, seed=0)
    maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    collated = {
        "input": as_interleaved(_complex(2, _C, _H2, _W2, seed=1)),
        "target": as_interleaved(_complex(2, _C, _H2, _W2, seed=2)),
        "sensitivity_complex": maps,
    }
    if with_alt:
        collated["input_alt"] = as_interleaved(_complex(2, _C, _H2, _W2, seed=3))
    return BatchAdapter.from_dict(collated), maps[..., 0]


def _sense_kspace(image: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
    from spectramr.core.metrics.coil_manifold import as_complex_coils, as_interleaved
    from spectramr.infrastructure.physics.fft_ops import sense_forward

    return as_interleaved(sense_forward(as_complex_coils(image), maps, None))


class _ImageArmStub(_CoilStub):
    """A network that emits one complex image (2 channels) from 2C coil channels."""

    def __init__(self, compute: list[str], maps=None, *, ignores_input: bool = False) -> None:
        super().__init__(compute)
        self.state.config.model.output_type = "image"
        self.maps = maps
        self.ignores_input = ignores_input
        self.fixed = None

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        self.model.inputs.append(input_batch)
        if self.ignores_input:  # the EI/SSDU shape: y is read from the context
            self.fixed = input_batch[:, :2] * 0.5 if self.fixed is None else self.fixed
            return self.fixed
        return input_batch[:, :2] * 0.5

    def _context_metric_prediction(self, raw, batch_context):
        if self.maps is None:
            return super()._context_metric_prediction(raw, batch_context)
        return _sense_kspace(raw, self.maps), "kspace"


class TestTheContextMetricAdapter:
    def test_without_an_override_an_image_output_is_refused(self) -> None:
        """Planted: the image graded as k-space would be a number describing nothing."""
        import pytest

        batch, _ = _image_arm_batch()
        with pytest.raises(ValueError, match="tensors are 'image'"):
            _ImageArmStub(["reference_noise_corrected_mse"]).validation_step(
                batch.input, batch.target, batch_data=batch
            )

    def test_with_the_override_the_metric_scores_f_s_x_hat(self) -> None:
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry

        batch, maps = _image_arm_batch()
        stub = _ImageArmStub(["reference_noise_corrected_mse"], maps)
        out = stub.validation_step(batch.input, batch.target, batch_data=batch)
        expected = MetricsRegistry.get("reference_noise_corrected_mse")(
            _sense_kspace(0.5 * batch.input[:, :2], maps),
            batch.target,
            context=MetricContext(coil_maps=maps, signal_domain="kspace"),
        )
        assert out["val_reference_noise_corrected_mse"] == pytest_approx(expected.item())

    def test_the_override_reaches_pred_alt_too(self) -> None:
        """rep_variance compares like with like: both outputs through the adapter."""
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry

        batch, maps = _image_arm_batch(with_alt=True)
        out = _ImageArmStub(["rep_variance"], maps).validation_step(
            batch.input, batch.target, batch_data=batch
        )
        alt = batch.metadata["input_alt"]
        expected = MetricsRegistry.get("rep_variance")(
            _sense_kspace(0.5 * batch.input[:, :2], maps),
            context=MetricContext(pred_alt=_sense_kspace(0.5 * alt[:, :2], maps)),
        )
        assert out["val_rep_variance"] == pytest_approx(expected.item())


class TestThePredAltGuard:
    def test_a_forward_that_ignores_its_input_is_refused(self) -> None:
        """Planted: the EI/SSDU forward reads y from the context, so pred_alt == pred
        and rep_variance would read 0 while bias_squared equals the corrected MSE."""
        import pytest

        batch, maps = _image_arm_batch(with_alt=True)
        stub = _ImageArmStub(["rep_variance"], maps, ignores_input=True)
        with pytest.raises(ValueError, match="equals its output on `input`"):
            stub.validation_step(batch.input, batch.target, batch_data=batch)

    def test_a_forward_that_reads_its_input_passes(self) -> None:
        batch, maps = _image_arm_batch(with_alt=True)
        out = _ImageArmStub(["rep_variance"], maps).validation_step(
            batch.input, batch.target, batch_data=batch
        )
        assert out["val_rep_variance"] > 0


class _ContextReadingStub(_ImageArmStub):
    """The EI/SSDU shape: the forward reads y from the context, never its argument."""

    validation_reads_context_measurement = True

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        self.model.inputs.append(input_batch)
        return batch_context["measured_kspace"][:, :2] * 0.5


class TestTheAlternateContext:
    """#2593: the second-repetition forward runs on a context carrying ``input_alt``."""

    def _run(self, stub):
        batch, _ = _image_arm_batch(with_alt=True)
        context = {"measured_kspace": batch.input, "use_dc": False}
        return batch, stub.validation_step(
            batch.input, batch.target, batch_data=batch, batch_context=context
        )

    def test_rep_variance_reads_the_alternate_repetition(self) -> None:
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry

        _, maps = _image_arm_batch(with_alt=True)
        batch, out = self._run(_ContextReadingStub(["rep_variance"], maps))
        alt = batch.metadata["input_alt"]
        expected = MetricsRegistry.get("rep_variance")(
            _sense_kspace(0.5 * batch.input[:, :2], maps),
            context=MetricContext(pred_alt=_sense_kspace(0.5 * alt[:, :2], maps)),
        )
        assert out["val_rep_variance"] == pytest_approx(expected.item())
        assert out["val_rep_variance"] > 0

    def test_planted_without_the_flag_the_guard_refuses(self) -> None:
        """The same forward, with the context left as the primary input's."""
        import pytest

        _, maps = _image_arm_batch(with_alt=True)
        stub = _ContextReadingStub(["rep_variance"], maps)
        stub.validation_reads_context_measurement = False
        with pytest.raises(ValueError, match="equals its output on `input`"):
            self._run(stub)

    def test_ei_and_ssdu_declare_that_they_read_the_context(self) -> None:
        from spectramr.infrastructure.training.strategies.equivariant_imaging_strategy import (
            EquivariantImagingStrategy,
        )
        from spectramr.infrastructure.training.strategies.ssdu_strategy import (
            SSDUReconstructionStrategy,
        )

        assert EquivariantImagingStrategy.validation_reads_context_measurement
        assert SSDUReconstructionStrategy.validation_reads_context_measurement


class TestTheEstimatorRefusesAContextReadingForward:
    """The J-sample estimator recorrupts the tensor it hands the forward; a strategy
    that reads y from the context would average J copies of one output."""

    def _run(self, stub):
        batch, _ = _image_arm_batch(with_alt=True)
        context = {"measured_kspace": batch.input, "use_dc": False}
        return stub.validation_step(
            batch.input, batch.target, batch_data=batch, batch_context=context
        )

    def test_the_combination_raises(self, monkeypatch) -> None:
        import pytest

        _, maps = _image_arm_batch(with_alt=True)
        stub = _ContextReadingStub(["rep_variance"], maps)
        monkeypatch.setattr(stub, "_scoring_estimator", lambda config: object())
        with pytest.raises(ValueError, match="reads its measurement from the batch context"):
            self._run(stub)

    def test_without_an_estimator_the_same_strategy_validates(self, monkeypatch) -> None:
        _, maps = _image_arm_batch(with_alt=True)
        stub = _ContextReadingStub(["rep_variance"], maps)
        monkeypatch.setattr(stub, "_scoring_estimator", lambda config: None)
        assert self._run(stub)["val_rep_variance"] > 0


def test_val_loss_stays_on_the_device() -> None:
    """Planted: ``.item()`` on the per-batch val_loss is a host sync on every validation batch."""
    import inspect

    from spectramr.infrastructure.training.strategies.mixins import model_validation

    assert "l1_loss(pred_l, tgt_l).item()" not in inspect.getsource(model_validation)



class _PreModelStub(_Stub):
    """A declared ``adapters.pre_model`` chain; the forward records the tensor it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.adapter_chains = {"pre_model": [lambda x: x + 100.0]}
        self.seen_input: torch.Tensor | None = None

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        self.seen_input = input_batch
        return torch.zeros_like(input_batch)


def test_validation_applies_pre_model_before_the_forward() -> None:
    # train_step applies the pre_model chain; validation must hand the model the same
    # tensor, or it grades a model on an input it never trained on (cohort5 §2a).
    s = _PreModelStub()
    s.validation_step(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    assert s.seen_input is not None
    assert torch.equal(s.seen_input, torch.full((1, 1, 4, 4), 100.0))


class _SliceRecorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def observe(self, **kw) -> None:
        self.calls.append(kw)


def test_the_test_pass_writer_sees_every_batch_with_its_index() -> None:
    s = _Stub()
    s._metric_transform_views = lambda p, t, cfg, batch_data=None: MetricViews(p, t, None, None)
    s._test_slice_writer = _SliceRecorder()
    s.validation_step(torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4), batch_idx=7)
    (call,) = s._test_slice_writer.calls
    assert call["batch_index"] == 7
    assert torch.equal(call["target_view"], torch.ones(1, 1, 4, 4))


def test_no_writer_outside_the_test_pass() -> None:
    s = _Stub()
    s._metric_transform_views = lambda *a, **k: (_ for _ in ()).throw(AssertionError("called"))
    s.validation_step(torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
