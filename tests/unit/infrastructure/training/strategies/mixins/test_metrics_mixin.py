"""Regression tests for ``MetricsMixin._apply_metric_transforms``.

2026-07-11 cohort triage. The ``magnitude`` transform is selected on the
PREDICTION's channel count but used to index ``target[:, 1]`` unconditionally.
A 1-channel magnitude target raised ``IndexError``, which ``ModelValidationMixin``
swallows in a bare ``except Exception``: the run silently degraded to a
psnr/mse-only metric set, so the configured early-stopping monitor key was never
emitted and training burned its FULL iteration budget
(``mrixfields_b29_heteroscedastic_ulf``: 150,000 iters / ~23 h GPU while
``val_psnr`` diverged from -9.1 dB to -21.5 dB with nothing to stop it).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from spectramr.config.schemas.data import DataProcessingConfigSchema
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin


class _Mixin(MetricsMixin):
    """Bare host — ``_apply_metric_transforms`` touches no other strategy state."""


def _cfg(transform: str = "magnitude"):
    return SimpleNamespace(transform=transform)


def test_magnitude_transform_with_one_channel_target_does_not_raise():
    """The bug: 2-ch real-stacked pred + 1-ch magnitude target -> IndexError."""
    pred = torch.randn(2, 2, 8, 8)  # real-stacked complex
    target = torch.rand(2, 1, 8, 8)  # already a magnitude reference

    out_pred, out_target = _Mixin()._apply_metric_transforms(pred, target, _cfg())

    assert out_pred.shape == (2, 1, 8, 8)
    assert out_target.shape == (2, 1, 8, 8)
    expected = torch.complex(pred[:, 0], pred[:, 1]).abs().unsqueeze(1)
    assert torch.allclose(out_pred, expected)
    assert torch.allclose(out_target, target)


def test_magnitude_transform_with_two_channel_target_still_folds_to_magnitude():
    """The pre-existing happy path must stay byte-identical."""
    pred = torch.randn(2, 2, 8, 8)
    target = torch.randn(2, 2, 8, 8)

    out_pred, out_target = _Mixin()._apply_metric_transforms(pred, target, _cfg())

    assert out_pred.shape == (2, 1, 8, 8)
    assert out_target.shape == (2, 1, 8, 8)
    assert torch.allclose(out_target, torch.complex(target[:, 0], target[:, 1]).abs().unsqueeze(1))


def test_magnitude_transform_with_complex_target():
    pred = torch.randn(2, 2, 8, 8)
    target = torch.complex(torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8))

    _, out_target = _Mixin()._apply_metric_transforms(pred, target, _cfg())

    assert not torch.is_complex(out_target)
    assert torch.allclose(out_target, target.abs())


def test_magnitude_transform_rejects_unmappable_target():
    """#9 — never silently compare |pred| against a target in another domain."""
    pred = torch.randn(2, 2, 8, 8)
    target = torch.rand(2, 3, 8, 8)  # neither real-stacked, complex, nor magnitude

    with pytest.raises(ValueError, match="magnitude"):
        _Mixin()._apply_metric_transforms(pred, target, _cfg())


def test_domain_none_short_circuits():
    pred = torch.randn(2, 2, 8, 8)
    target = torch.rand(2, 1, 8, 8)
    cfg = SimpleNamespace(domain="none", transform="magnitude")

    out_pred, out_target = _Mixin()._apply_metric_transforms(pred, target, cfg)

    assert out_pred is pred
    assert out_target is target


def test_prediction_for_visualization_is_identity_for_an_ordinary_head():
    """Unchanged for everything that is not a distribution head, and that is the
    load-bearing half: a 2-channel COMPLEX prediction must pass through so
    ``to_magnitude`` can RSS it into a modulus. Slicing channel 0 would show the
    real part and call it the image.

    The default is no longer *literally* identity — it delegates to
    ``VisualizationReducer`` (#709), which returns the input unchanged here.
    Asserted by equality rather than identity for that reason.
    """
    pred = torch.randn(2, 2, 8, 8)
    assert torch.equal(_Mixin()._prediction_for_visualization(pred), pred)


def test_prediction_for_visualization_reduces_a_distribution_head():
    """The #390 half the old default could not do.

    `evidential_unet` emits ``[mean, var, alpha, beta]`` per 1-channel target.
    With an identity default those four reached ``to_magnitude`` and rendered as
    ``sqrt(Σ params²)`` — a picture of no quantity at all. Only
    ``HeteroscedasticULFStrategy`` overrode the hook, so only it was correct.
    """
    import types

    mixin = _Mixin()
    mixin.config = types.SimpleNamespace(
        model=types.SimpleNamespace(model_type="evidential_unet"),
        data=types.SimpleNamespace(processing=DataProcessingConfigSchema()),
    )
    pred = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1)

    out = mixin._prediction_for_visualization(pred)

    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == 0.0, "channel 0 is the mean"


class TestReportCaseId:
    """The recorded case label must carry its acceleration rung.

    Cascading validation feeds ``feed_report_case_recorder`` once per level with
    the same training iteration. Labelling every rung ``val_step<N>`` made
    ``cases_index.json`` carry indistinguishable rows, so per-case analysis
    (``scripts/diagnostics/spectral_sharpness.py``) silently compared an R=2 case
    against an R=32 one -- and the tie is what crashed the recorder's eviction
    path (#617).
    """

    def test_level_is_encoded_and_rungs_are_distinct(self) -> None:
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            report_case_id,
        )

        ids = {report_case_id(12000, level) for level in (2, 8, 32)}
        assert ids == {
            "val_step12000_R2x",
            "val_step12000_R8x",
            "val_step12000_R32x",
        }

    def test_absent_level_keeps_the_legacy_label(self) -> None:
        """Strategies without a cascade (the base mixin) must not change."""
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            report_case_id,
        )

        assert report_case_id(7) == "val_step7"

    def test_fractional_level_does_not_collide_with_its_floor(self) -> None:
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            report_case_id,
        )

        assert report_case_id(1, 2.5) != report_case_id(1, 2)
        assert report_case_id(1, 2.5) == "val_step1_R2.5x"

    def test_feeder_threads_the_level_into_recorder_and_sink(self) -> None:
        """The label must reach BOTH sinks, not just the image recorder.

        ``_Sink.observe`` declares ``context`` **without a default** on purpose.
        The feeder passes it unconditionally, so a required parameter here pins
        that: were the call made conditional -- forwarded only when the mapping
        is non-empty -- this double would raise instead of silently dropping the
        columns the CSV promises. The two ``observe`` signatures below are also
        deliberately different shapes; the recorder protocol (``arrays`` /
        ``domain``) never took a context and is unchanged by it.
        """
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            feed_report_case_recorder,
        )

        seen: dict[str, list] = {"rec": [], "sink": [], "ctx": []}

        class _Rec:
            enabled = True
            record_volumes = False

            def observe(self, *, case_id, arrays, metrics, domain):
                seen["rec"].append(case_id)

        class _Sink:
            enabled = True

            def observe(self, *, case_id, metrics, split, step, context):
                seen["sink"].append(case_id)
                seen["ctx"].append(context)

        for level in (2, 32):
            feed_report_case_recorder(
                _Rec(),
                predictions=torch.rand(1, 1, 4, 4),
                targets=torch.rand(1, 1, 4, 4),
                inputs=torch.rand(1, 1, 4, 4),
                metrics={"psnr": 30.0},
                step=500,
                sink=_Sink(),
                cascade_level=level,
                context={"acceleration_level": float(level)},
            )
        assert seen["rec"] == ["val_step500_R2x", "val_step500_R32x"]
        assert seen["sink"] == ["val_step500_R2x", "val_step500_R32x"]
        assert seen["ctx"] == [
            {"acceleration_level": 2.0},
            {"acceleration_level": 32.0},
        ], "the feeder must forward the caller's context verbatim, per rung"

    def test_every_metric_sink_in_src_accepts_the_context_kwarg(self) -> None:
        """``feed_report_case_recorder`` passes ``context=`` to whatever it is handed.

        That makes "accepts ``context``" a requirement on *every* metric sink, and
        nothing in the tree states it: ``sink`` is an untyped parameter defaulting
        to ``None``, so a sink that omits the kwarg type-checks, imports, lints
        clean, and raises ``TypeError`` only once validation runs -- hours into a
        training job. This is the ratchet for that.

        Discriminating by *shape* matters. Three unrelated ``observe`` protocols
        live in this tree -- the metric sink (``split`` / ``step``), the image
        recorder (``arrays`` / ``domain``) and the inference-artifact writer
        (``prediction`` / ``target``) -- and only the first is passed a context.
        Matching on the bare name ``observe`` would drag the other two in and make
        this test red for implementers it has no claim over.
        """
        import ast
        import pathlib

        import spectramr

        src_root = pathlib.Path(spectramr.__file__).resolve().parent
        offenders, found = [], []
        for path in sorted(src_root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name != "observe":
                    continue
                kwonly = {a.arg for a in node.args.kwonlyargs}
                if not {"case_id", "split", "step"} <= kwonly:
                    continue  # a different observe protocol -- not our claim
                where = f"{path.relative_to(src_root.parent)}:{node.lineno}"
                found.append(where)
                if "context" not in kwonly and node.args.kwarg is None:
                    offenders.append(where)

        assert found, (
            f"found no metric sinks at all under {src_root} -- the scan is vacuous, not passing"
        )
        assert not offenders, (
            f"these metric sinks would raise TypeError when the feeder passes context=: {offenders}"
        )


# ---------------------------------------------------------------------------
# #585: saved validation images carried a call counter, not the training step.
#
# `step` came from `validation_step_count` — bumped once per validation call (and
# once per cascade level) — with a fallback to `self.env.current_step` that can
# NEVER fire: `current_step` exists on neither `TrainingEnvironment`. So images
# were numbered 0,1,2,... while the run reported iterations 500,1000,1500, and
# nothing tied a picture to the checkpoint that produced it. `diffusion.py`
# overrode the whole method to reach the right seam; every other strategy did not.
# ---------------------------------------------------------------------------


def test_the_dead_env_current_step_fallback_is_gone():
    """`hasattr(self.env, "current_step")` was always False — pin that it left.

    Source-level, because the defect was an unreachable branch: it never raised,
    never logged, and produced a plausible-looking small integer.
    """
    import inspect

    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
        MetricsMixin,
    )

    src = inspect.getsource(MetricsMixin._log_validation_images_to_tensorboard)
    # Comments stripped: the fix's own explanation names the dead attribute, and a
    # guard that its own rationale trips is a guard nobody can document.
    code = "\n".join(line for line in src.splitlines() if not line.strip().startswith("#"))
    assert "current_step" not in code, "the unreachable env.current_step fallback reappeared"
    assert "resolve_loop_iteration" in code, (
        "the image step label must come from the loop-state SSOT"
    )


def test_neither_environment_class_ever_had_current_step():
    """The evidence that the fallback was unreachable, not merely unused."""
    from spectramr.infrastructure.training.builders.environment import (
        TrainingEnvironment as BuilderEnv,
    )
    from spectramr.infrastructure.training.contexts import TrainingEnvironment

    for cls in (TrainingEnvironment, BuilderEnv):
        names = set(getattr(cls, "__annotations__", {})) | set(dir(cls))
        assert "current_step" not in names, (
            f"{cls.__module__}.{cls.__name__} grew a `current_step`; if this is now "
            "a real seam, the image-label fix should be revisited"
        )


def test_image_label_tracks_the_training_iteration():
    import types

    from spectramr.infrastructure.training.loop_state import (
        LoopState,
        resolve_loop_iteration,
    )

    strategy = types.SimpleNamespace(loop_state=LoopState(iteration=1500, epoch=3))
    assert resolve_loop_iteration(strategy) == 1500
    # A mixin built standalone in a unit test must still be safe.
    assert resolve_loop_iteration(types.SimpleNamespace()) == 0


# ---------------------------------------------------------------------------
# #173 / #660: metric selection is gated before the first batch.
#
# Two surfaces, two policies, deliberately:
#   metrics.compute   an EXPLICIT name -> raise. A typo used to be logged once
#                     inside the computer and then produce a silently missing CSV
#                     column, indistinguishable from "that metric was never any
#                     good on this data".
#   compute_* flags   LEGACY booleans -> skip with one log. Six shipped flags name
#                     an unregistered metric and `compute_advanced_metrics` alone
#                     is set by 734 arms, so raising would fail the corpus at load
#                     and help nobody.
# ---------------------------------------------------------------------------


class TestMetricSelectionIsGated:
    @staticmethod
    def _mixin():
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            MetricsMixin,
        )

        class _Stub(MetricsMixin):
            pass

        return _Stub()

    def test_an_unregistered_name_in_compute_raises(self):
        import pytest

        from spectramr.config.schemas.metrics import MetricsConfigSchema

        cfg = MetricsConfigSchema(compute=["psnr", "totally_bogus_metric"])
        with pytest.raises(ValueError, match="not registered"):
            self._mixin()._extract_metrics_from_config(cfg)

    def test_a_valid_explicit_list_is_returned_unchanged(self):
        from spectramr.config.schemas.metrics import MetricsConfigSchema

        got = self._mixin()._extract_metrics_from_config(
            MetricsConfigSchema(compute=["psnr", "ssim"])
        )
        assert got == ["psnr", "ssim"]

    @pytest.mark.parametrize(
        "flag",
        [
            "compute_precision_recall",
        ],
    )
    def test_dangling_legacy_flags_never_reach_the_computer(self, flag):
        """The one dangling flag that still EXISTS to be selected.

        This parametrised over six. Five of them (`compute_blur`,
        `compute_dvars`, `compute_fd`, `compute_gcor`, `compute_pe_cross_corr`)
        were deleted from the schema — with ten more found by the same
        alias-aware measurement — because they named metrics registered under no
        name and no arm declared them. A flag that cannot be written cannot
        reach the computer, so there is nothing left here to assert about them;
        `test_metrics.py` pins that they are gone.

        `compute_precision_recall` survives: 29 arms declare it (all False) and
        the block is `extra="forbid"`, so the field cannot go. It is now
        refused at the schema rather than allowed through to fail as an
        `UnknownMetricDirectionError` whose message is about DIRECTIONS and says
        nothing about the real cause.
        """
        import pytest
        from pydantic import ValidationError

        from spectramr.config.schemas.metrics import MetricsConfigSchema
        from spectramr.core.metrics.flag_map import metric_for_flag
        from spectramr.core.metrics.registry import MetricsRegistry

        target = metric_for_flag(flag)
        assert not MetricsRegistry.is_registered(target), (
            f"{flag} -> '{target}' is now registered; drop it from this list"
        )

        # The GATE MOVED UPSTREAM. This used to build the config and assert the
        # mixin dropped the metric while keeping the valid ones. It can no
        # longer build the config at all: the flag is refused by the schema, so
        # a run is told at load rather than discovering a missing column later.
        with pytest.raises(ValidationError, match="not registered"):
            MetricsConfigSchema(**{flag: True, "compute_psnr": True})

        # And the valid flags are unaffected by its presence at its only legal
        # value — the half of the original assertion that still has meaning.
        got = self._mixin()._extract_metrics_from_config(
            MetricsConfigSchema(**{flag: False, "compute_psnr": True})
        )
        assert target not in got
        assert "psnr" in got

    def test_a_legacy_flag_does_not_fail_the_corpus(self):
        """`compute_advanced_metrics` defaults True and is set by 734 arms.

        Raising on a dangling FLAG (as opposed to an explicit name) would fail
        every one of them at load.
        """
        from spectramr.config.schemas.metrics import MetricsConfigSchema

        got = self._mixin()._extract_metrics_from_config(
            MetricsConfigSchema(compute_advanced_metrics=True, compute_psnr=True)
        )
        assert "psnr" in got and "advanced_metrics" not in got

    def test_the_dead_gradient_snr_branch_is_gone(self):
        """`compute_gradient_snr` is not a schema field, so the branch was doubly
        unreachable: an unregistered target behind a `hasattr` that never passes."""
        import inspect

        from spectramr.config.schemas.metrics import MetricsConfigSchema
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            MetricsMixin,
        )

        assert "compute_gradient_snr" not in MetricsConfigSchema.model_fields
        src = inspect.getsource(MetricsMixin._extract_metrics_from_config)
        code = "\n".join(line for line in src.splitlines() if not line.strip().startswith("#"))
        assert '"gradient_snr"' not in code


class TestFlagCoverageDerivesFromTheSchema:
    """#340's coverage half: 22 flags bought a CSV column and no measurement.

    ``_extract_metrics_from_config``'s flag map was hand-written with 43 entries;
    ``training_loop._CSV_METRIC_NAME_MAP`` was hand-written with 78. Both spelled
    their names correctly, so the existing name-drift ratchet passed — but a flag
    present in the second and absent from the first produced a ``losses.csv``
    header over a column nothing could fill. An empty column under a header is
    read as "we measured it and got nothing", not "we never selected it".
    """

    @staticmethod
    def _mixin():
        class _M(MetricsMixin):
            pass

        return _M()

    @pytest.mark.parametrize("flag", ["compute_wm2max", "compute_frd", "compute_pdm"])
    def test_a_formerly_header_only_flag_now_selects(self, flag: str):
        from spectramr.config.schemas.metrics import MetricsConfigSchema
        from spectramr.core.metrics.flag_map import metric_for_flag

        got = self._mixin()._extract_metrics_from_config(MetricsConfigSchema(**{flag: True}))
        assert metric_for_flag(flag) in got

    def test_the_default_arm_emits_no_dangling_flag_warning(self, caplog):
        """The corpus-wide guard. ``compute_advanced_metrics`` defaults True, so if
        it were treated as an ordinary flag naming an unregistered metric, this
        warning would fire on EVERY run — and ``audit --strict`` exits 2 on it."""
        import logging

        from spectramr.config.schemas.metrics import MetricsConfigSchema

        with caplog.at_level(logging.WARNING):
            got = self._mixin()._extract_metrics_from_config(MetricsConfigSchema())
        assert sorted(got) == ["mae", "mse", "psnr", "ssim"]
        assert not [r for r in caplog.records if "name no registered" in r.getMessage()]

    # ``test_a_genuinely_unregistered_flag_still_warns`` was deleted here, not
    # silenced. It was the anti-vacuity partner of the test above and used
    # ``compute_blur`` as its specimen of a flag naming an unregistered metric.
    # That flag no longer exists — it was one of fifteen deleted once the
    # alias-aware measurement showed none of them named a registered metric and
    # no arm declared any of them — and the one dangling flag that remains
    # (``compute_precision_recall``) is refused by the schema, so it cannot
    # reach this code path either.
    #
    # There is no longer a way to construct the input it needed, which makes it
    # a test of an unreachable branch. The invariant it protected now lives at
    # the source, in ``test_metrics.py::TestNoFlagAdvertisesAnUnregisteredMetric``:
    # exactly one dangling flag, and that one cannot be enabled.

    def test_selection_and_csv_columns_cover_the_same_flags(self):
        """The two maps are one map now. A column can no longer outlive a selector."""
        from spectramr.core.metrics.flag_map import schema_flag_to_metric
        from spectramr.pipelines.training_loop import _CSV_METRIC_NAME_MAP

        assert dict(_CSV_METRIC_NAME_MAP) == schema_flag_to_metric()


class TestFusedTransferIsSharedWithCore:
    """``_convert_metrics_to_floats`` keeps its REDUCTION and delegates its SYNC.

    The fused-transfer idiom had grown at three sites; the mechanism now lives in
    ``core.metrics.scalar_transfer`` so ``core.metrics.computer`` can use it too
    (it cannot import from ``infrastructure`` — non-negotiable #5).

    What must NOT be shared is the reduction: this mixin converts LOSSES, where
    mean-over-batch is right, while the computer converts METRICS, where a
    non-scalar is a defect. Folding both into the helper would answer two
    different questions the same way (pitfall #13b).
    """

    def test_the_core_helper_is_the_one_doing_the_transfer(self):
        from unittest import mock

        from spectramr.core.metrics import scalar_transfer

        with mock.patch.object(scalar_transfer.torch, "stack", wraps=torch.stack) as spy:
            out = MetricsMixin._convert_metrics_to_floats(
                {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
            )
        assert out == {"a": 1.0, "b": 2.0}
        assert spy.call_count == 1

    def test_batch_reduction_still_happens_here_not_in_the_helper(self):
        """A per-sample loss vector must still mean-reduce — the helper refuses to."""
        out = MetricsMixin._convert_metrics_to_floats({"l1": torch.tensor([1.0, 3.0])})
        assert out["l1"] == pytest.approx(2.0)

    def test_complex_still_takes_the_real_part_here(self):
        """Also caller policy: the shared helper rejects complex outright."""
        out = MetricsMixin._convert_metrics_to_floats({"z": torch.tensor(1 + 2j)})
        assert out["z"] == pytest.approx(1.0)

    def test_mixed_amp_dtypes_convert(self):
        out = MetricsMixin._convert_metrics_to_floats(
            {"half": torch.tensor(1.0, dtype=torch.float16), "full": torch.tensor(2.0)}
        )
        assert out == {"half": 1.0, "full": 2.0}

    def test_unconvertible_type_still_raises_typeerror(self):
        """The helper raises ValueError; this seam's contract is TypeError."""
        with pytest.raises(TypeError, match="Cannot convert"):
            MetricsMixin._convert_metrics_to_floats({"s": "not a number"})


# ---------------------------------------------------------------------------
# The transform leaves moved into `validation.scoring` and the resolver did not
# ---------------------------------------------------------------------------
#
# `_apply_metric_transforms` reads `domain` / `transform` / `output_transform`
# off whatever it is handed. Those leaves live at `validation.scoring.domain`
# and `validation.scoring.output_transform` (RENAMES records both as folds), so
# every lookup returned None and NO configured transform ever fired -- only the
# auto-detected `magnitude` for complex / 2-channel predictions.
#
# Measured: 145 arms declare `output_transform` (120 of them `ifft_magnitude`)
# and the tree has exactly ONE reader, `diffusion.py`, which then passed the
# resolved STRING into this function's config parameter -- where every key
# lookup on a str is None, so it fell through to `return pred, target`.
#
# Consequence on experiment_11_attention_none (2026-08-08): metrics computed on
# 8-channel k-space. val_psnr 58 dB at iteration 1000, robust_mri_psnr NaN.
#
# These drive a REAL TrainingSettings. A SimpleNamespace spelling the leaves
# flat is what hid the same class of bug in test_domain_inference.py -- test and
# code agreed on a spelling the schema had abandoned.

ARM = (
    Path(__file__).resolve().parents[6]
    / "experiments/inprogress/kspace_filling/attention_shootout"
    / "experiment_11_attention_none.yaml"
)


def _real_validation_config(transform: str | None = "ifft_magnitude"):
    """The arm's validation block; ``transform=None`` keeps the arm's own declaration.

    The cohort grades with ``ifft_sense_adjoint`` now, so the tests that exercise the
    map-free ``ifft_magnitude`` path pin it here instead of inheriting it from the arm.
    """
    if not ARM.is_file():
        pytest.skip(f"arm not present: {ARM}")
    from spectramr.config.settings import TrainingSettings

    val = TrainingSettings.from_yaml(str(ARM)).validation
    if transform is None:
        return val
    scoring = val.scoring.model_copy(update={"output_transform": transform})
    return val.model_copy(update={"scoring": scoring})


class _KspaceMixin(MetricsMixin):
    """Host with the coil knobs `ifft_magnitude` consults."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            data=SimpleNamespace(coil_processing_mode="none", num_virtual_coils=4),
            validation=None,
        )


@pytest.mark.unit
def test_output_transform_is_read_from_validation_scoring() -> None:
    """The declared `ifft_magnitude` must actually fire.

    8 interleaved real/imag channels (4 coils) is the exp_11 layout. A fired
    `ifft_magnitude` pairs them into complex, IFFTs and RSS-combines, so the
    channel axis collapses to 1. Unchanged 8 channels means the transform never
    ran and the metrics are being computed on k-space.
    """
    val_config = _real_validation_config()
    assert val_config.scoring.output_transform == "ifft_magnitude", (
        "fixture arm no longer declares the transform this test is about"
    )

    kspace = torch.randn(1, 8, 32, 32)
    out, target = _KspaceMixin()._apply_metric_transforms(kspace, kspace.clone(), val_config)

    assert out.shape[1] == 1, f"transform did not fire: {tuple(out.shape)}"
    assert target.shape[1] == 1
    assert not torch.is_complex(out)


@pytest.mark.unit
def test_domain_none_under_scoring_suppresses_the_transform() -> None:
    """`validation.scoring.domain: none` is the documented opt-out.

    Read flat it was always None, so the opt-out could not be expressed -- the
    mirror of the bug above, and the reason a naive fix that only chases
    `output_transform` is incomplete.
    """
    val_config = _real_validation_config().model_copy(deep=True)
    object.__setattr__(
        val_config,
        "scoring",
        val_config.scoring.model_copy(update={"domain": "none"}),
    )

    kspace = torch.randn(1, 8, 32, 32)
    out, _ = _KspaceMixin()._apply_metric_transforms(kspace, kspace.clone(), val_config)

    assert out.shape[1] == 8, "domain: none must suppress the transform"


@pytest.mark.unit
def test_a_bare_string_transform_name_is_still_honoured() -> None:
    """`diffusion.py` hands this function a resolved name, not a config.

    Every key lookup on a str is None, so the name was silently dropped. The
    call site now passes the config object, but accepting a plain name keeps
    the two spellings from diverging again.
    """
    kspace = torch.randn(1, 8, 32, 32)
    out, _ = _KspaceMixin()._apply_metric_transforms(kspace, kspace.clone(), "ifft_magnitude")

    assert out.shape[1] == 1, f"string transform name dropped: {tuple(out.shape)}"


@pytest.mark.unit
def test_the_none_sentinel_string_does_not_become_a_transform_name() -> None:
    """`diffusion.py` used `output_transform or "none"`; "none" is not a lookup key."""
    kspace = torch.randn(1, 8, 32, 32)
    out, _ = _KspaceMixin()._apply_metric_transforms(kspace, kspace.clone(), "none")

    assert out.shape[1] == 8


# ---------------------------------------------------------------------------
# #931 — the dispatcher swallowed unknown names; 146 arms declare one
# ---------------------------------------------------------------------------


class _KspaceMixinWithMetricsBlock(_KspaceMixin):
    """Host whose `config.metrics` block exists, as a real strategy's does."""

    def __init__(self, transform=None, domain=None, validation=None) -> None:
        super().__init__()
        self.config.metrics = SimpleNamespace(transform=transform, domain=domain)
        self.config.validation = validation


@pytest.mark.unit
def test_an_undispatchable_name_raises_instead_of_grading_raw_kspace() -> None:
    """Pitfall #9, and the actual defect behind #931.

    The if/elif chain has no `else`, so `ifft_mag_combine` — declared on 143
    arms — fell out of the bottom and the tensors were returned untouched. The
    run then reported a PSNR computed on k-space with nothing in the log to say
    the requested transform had not happened.

    Passed in the FIRST slot because that is how `_compute_training_metrics`
    hands the metrics block in; that path has always read this key.
    """
    kspace = torch.randn(1, 8, 32, 32)

    with pytest.raises(ValueError, match="Unknown metric transform"):
        _KspaceMixin()._apply_metric_transforms(
            kspace, kspace.clone(), SimpleNamespace(transform="ifft_mag_combine")
        )


@pytest.mark.unit
def test_metrics_transform_does_not_leak_onto_the_validation_path() -> None:
    """`metrics.transform` must not start firing where it never fired.

    Wiring it into the validation cascade would activate 146 arms, and 112 of
    them have `losses.output_domain` and `infer_output_domain` both `image` —
    an IFFT there yields a Fourier magnitude, not the coil combine intended.
    8 channels surviving is the evidence it stayed on its own path.
    """
    host = _KspaceMixinWithMetricsBlock(transform="ifft_sense_adjoint")
    kspace = torch.randn(1, 8, 32, 32)

    out, _ = host._apply_metric_transforms(kspace, kspace.clone(), None)

    assert out.shape[1] == 8


@pytest.mark.unit
def test_scoring_output_transform_outranks_a_metrics_block_declaration() -> None:
    """The 67 arms declaring both, disagreeing, must not change what they measure.

    `experiment_11_attention_none` is one: `metrics.transform: ifft_sense_adjoint`
    (SENSE, needs sensitivity maps) against `validation.scoring.output_transform:
    ifft_magnitude` (map-free RSS). This host has no smaps, so an RSS-shaped
    result is the evidence that precedence holds.
    """
    val_config = _real_validation_config()
    host = _KspaceMixinWithMetricsBlock(transform="ifft_sense_adjoint", validation=val_config)
    kspace = torch.randn(1, 8, 32, 32)

    out, _ = host._apply_metric_transforms(kspace, kspace.clone(), val_config)

    assert out.shape[1] == 1
    assert not torch.is_complex(out)


@pytest.mark.unit
def test_the_arm_grades_validation_with_sense_adjoint() -> None:
    """Validation and training metrics share one coil combine (SENSE adjoint)."""
    val_config = _real_validation_config(transform=None)

    assert val_config.scoring.output_transform == "ifft_sense_adjoint"


@pytest.mark.unit
def test_sense_adjoint_with_maps_combines_the_coils_into_one_image() -> None:
    host = _KspaceMixinWithMetricsBlock(transform="ifft_sense_adjoint")
    host._current_smaps = torch.complex(torch.ones(1, 4, 32, 32), torch.zeros(1, 4, 32, 32)) / 2
    kspace = torch.randn(1, 8, 32, 32)

    out, target = host._apply_metric_transforms(kspace, kspace.clone(), "ifft_sense_adjoint")

    assert out.shape == (1, 1, 32, 32)
    assert target.shape == (1, 1, 32, 32)


@pytest.mark.unit
def test_explicit_maps_override_the_strategys_own_for_sense_adjoint() -> None:
    """The measured-map scoring family hands its own maps; the strategy's must not win."""
    host = _KspaceMixinWithMetricsBlock(transform="ifft_sense_adjoint")
    host._current_smaps = torch.complex(torch.ones(1, 4, 32, 32), torch.zeros(1, 4, 32, 32)) / 2
    override = torch.zeros(1, 4, 32, 32, dtype=torch.complex64)
    override[:, 0] = 1.0  # all weight on coil 0
    kspace = torch.randn(1, 8, 32, 32)

    from spectramr.infrastructure.physics.fft_ops import ifft2c

    own = host._metric_transform_views(kspace, kspace.clone(), "ifft_sense_adjoint")
    explicit = host._metric_transform_views(
        kspace, kspace.clone(), "ifft_sense_adjoint", smaps=override
    )
    coil0 = ifft2c(torch.complex(kspace[:, 0], kspace[:, 1])).abs().unsqueeze(1)

    assert not torch.allclose(own.pred, explicit.pred)
    assert torch.allclose(explicit.pred, coil0, atol=1e-5)


@pytest.mark.unit
def test_the_none_sentinel_survives_raise_on_unknown() -> None:
    """21 arms declare `metrics.transform: 'none'` — a *truthy* string.

    Raising on unknown names without normalising the sentinel first would turn
    every one of them into a crash. Two more declare `output_transform: 'none'`
    and would break on the raise alone.
    """
    kspace = torch.randn(1, 8, 32, 32)

    out, _ = _KspaceMixin()._apply_metric_transforms(
        kspace, kspace.clone(), SimpleNamespace(transform="none")
    )

    assert out.shape[1] == 8


@pytest.mark.unit
def test_the_two_stage_fallback_to_self_config_validation_is_preserved() -> None:
    """A caller passing a config that declares nothing still falls back.

    The dispatcher has always consulted `self.config.validation` second.
    Collapsing to a single source would silently drop that for any caller
    handing in a narrowed or per-level config.
    """
    host = _KspaceMixinWithMetricsBlock(validation=_real_validation_config())
    kspace = torch.randn(1, 8, 32, 32)

    out, _ = host._apply_metric_transforms(
        kspace, kspace.clone(), SimpleNamespace(scoring=None, output_transform=None)
    )

    assert out.shape[1] == 1, "fallback to self.config.validation was lost"


@pytest.mark.unit
def test_metrics_domain_none_suppresses_the_auto_magnitude_gate() -> None:
    """`domain: none` outranks the combine we would otherwise pick ourselves."""
    host = _KspaceMixinWithMetricsBlock(transform="ifft_magnitude", domain="none")
    pred = torch.randn(1, 2, 8, 8)  # would trip the auto-magnitude gate

    out, _ = host._apply_metric_transforms(pred, pred.clone(), None)

    assert out.shape[1] == 2


@pytest.mark.unit
def test_the_resolved_transform_reaches_the_run_log(caplog) -> None:
    """Pitfall #15 obligation (c): a wired knob must be visible in provenance.

    #927 and #931 were both "which knob fired?" questions that a run log could
    not answer. The line names the winning source, not just the transform.
    """
    host = _KspaceMixinWithMetricsBlock(validation=_real_validation_config())
    val_config = _real_validation_config()
    kspace = torch.randn(1, 8, 32, 32)

    with caplog.at_level(
        logging.INFO,
        logger="spectramr.infrastructure.training.strategies.mixins.metrics_mixin",
    ):
        host._apply_metric_transforms(kspace, kspace.clone(), val_config)
        host._apply_metric_transforms(kspace, kspace.clone(), val_config)

    lines = [r for r in caplog.records if "transform:" in r.getMessage()]
    assert len(lines) == 1, "provenance must be logged once, not per batch"
    assert "ifft_magnitude" in lines[0].getMessage()
    assert "validation.scoring.output_transform" in lines[0].getMessage()


def test_a_host_copy_that_fails_is_not_swallowed():
    """Planted violation: until 2026-09-03 ``_magnitude_np`` wrapped the detach /
    device move in ``except Exception: pass`` and returned whatever ``np.asarray``
    made of the tensor. The nested helper must contain no ``try`` at all."""
    import ast
    import inspect

    from spectramr.infrastructure.training.strategies.mixins import metrics_mixin

    tree = ast.parse(inspect.getsource(metrics_mixin))
    helpers = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_magnitude_np"
    ]
    assert helpers, "_magnitude_np is gone; move this guard to its replacement"
    assert not any(isinstance(n, ast.Try) for n in ast.walk(helpers[0]))


@pytest.mark.unit
def test_ifft_target_magnitude_crosses_only_the_target() -> None:
    """#2325: an image prediction against a k-space target.

    The prediction is left in its domain (its magnitude); the 8-channel target
    is paired, inverse-transformed and RSS-combined. ``ifft_magnitude`` would have
    Fourier-transformed the prediction too.
    """
    from spectramr.infrastructure.physics.fft_ops import ifft2c

    pred = torch.randn(1, 2, 32, 32)
    kspace = torch.randn(1, 8, 32, 32)
    out, target = _KspaceMixin()._apply_metric_transforms(pred, kspace, "ifft_target_magnitude")

    assert torch.allclose(out, torch.complex(pred[:, 0], pred[:, 1]).abs().unsqueeze(1))
    coils = ifft2c(torch.complex(kspace[:, 0::2], kspace[:, 1::2]))
    assert torch.allclose(target, coils.abs().pow(2).sum(1, keepdim=True).sqrt(), atol=1e-6)


def _sense_batch(coils: int = 4, size: int = 16):
    """Unit-norm complex maps, the anatomy, and its coil k-space interleaved."""
    from spectramr.core.metrics.coil_manifold import as_interleaved
    from spectramr.infrastructure.physics.fft_ops import fft2c

    torch.manual_seed(0)
    maps = torch.randn(1, coils, size, size, dtype=torch.complex64)
    maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    anatomy = torch.rand(1, 1, size, size).to(torch.complex64)
    return maps, anatomy, as_interleaved(fft2c(maps * anatomy))


@pytest.mark.unit
def test_sense_magnitude_combines_a_kspace_prediction_with_the_batch_maps() -> None:
    from spectramr.core.metrics.sense_image_metrics import sense_scoring_pair

    maps, anatomy, kspace = _sense_batch()
    views = _KspaceMixin()._metric_transform_views(
        kspace, kspace.clone(), "sense_magnitude", batch_data={"sensitivity_complex": maps}
    )
    expected, _ = sense_scoring_pair(kspace, kspace, maps)
    assert views.pred.shape == (1, 1, 16, 16)
    torch.testing.assert_close(views.pred, expected.abs())
    torch.testing.assert_close(views.target, anatomy.abs(), atol=1e-5, rtol=0)
    assert views.complex_pair is not None  # phase metrics see the SENSE image


@pytest.mark.unit
def test_sense_magnitude_takes_an_image_prediction_as_it_is() -> None:
    """EI/SSDU emit one S^H-combined image; a second combine would be wrong."""
    maps, anatomy, kspace = _sense_batch()
    pred = torch.cat([anatomy.real, anatomy.imag], dim=1)
    out, target = _KspaceMixin()._apply_metric_transforms(
        pred, kspace, "sense_magnitude", batch_data={"sensitivity_complex": maps}
    )
    torch.testing.assert_close(out, anatomy.abs())
    torch.testing.assert_close(out, target, atol=1e-5, rtol=0)


@pytest.mark.unit
def test_sense_magnitude_without_maps_raises_instead_of_scoring_rss() -> None:
    """Planted: the RSS image is the one fallback a SENSE column must never take."""
    from spectramr.core.metrics.sense_image_metrics import SenseMapsMissingError

    _, _, kspace = _sense_batch()
    with pytest.raises(SenseMapsMissingError):
        _KspaceMixin()._apply_metric_transforms(kspace, kspace, "sense_magnitude")
    with pytest.raises(SenseMapsMissingError):
        _KspaceMixin()._apply_metric_transforms(
            kspace, kspace, "sense_magnitude", batch_data={"input": kspace}
        )


@pytest.mark.unit
def test_training_metrics_hand_the_training_batch_maps_to_sense_magnitude(monkeypatch) -> None:
    """Step 1 of a fresh n2n arm: the metrics block names no transform, so the
    validation ``sense_magnitude`` reaches the training metrics and needs the batch."""
    maps, anatomy, kspace = _sense_batch()
    host = _KspaceMixin()
    host.config = SimpleNamespace(
        data=host.config.data,
        metrics=None,
        validation=SimpleNamespace(scoring=SimpleNamespace(output_transform="sense_magnitude")),
        training=SimpleNamespace(max_iterations=10),
    )
    seen: list[torch.Tensor] = []
    computer = SimpleNamespace(compute=lambda p, t, **_: seen.append(p) or {"psnr": 1.0})
    monkeypatch.setattr(type(host), "training_metrics_computer", property(lambda _: computer))

    from spectramr.data.batch_types import BatchAdapter

    # The loop's TrainingBatch: the real ``sensitivity`` binds to ``coil_maps``,
    # the complex maps ride in ``metadata``.
    batch = BatchAdapter.from_dict(
        {
            "input": kspace,
            "target": kspace,
            "sensitivity_complex": maps,
            "sensitivity": maps.abs(),
        }
    )
    out = host._compute_training_metrics(
        kspace, kspace.clone(), host.config, 1, batch_data=batch
    )
    assert out == {"train_psnr": 1.0}
    torch.testing.assert_close(seen[0], anatomy.abs(), atol=1e-5, rtol=0)


def test_convert_metrics_to_floats_reduces_through_the_shared_reducer(monkeypatch):
    """The reduction has one owner (``mixins.utils.reduce_on_device``), which the
    per-step ``_handle_anomalies`` also uses; this seam only adds the transfer."""
    from spectramr.infrastructure.training.strategies.mixins import metrics_mixin as mm

    seen: list[str] = []
    real = mm.reduce_on_device

    def _spy(key, value):
        seen.append(key)
        return real(key, value)

    monkeypatch.setattr(mm, "reduce_on_device", _spy)
    out = MetricsMixin._convert_metrics_to_floats({"a": torch.tensor([1.0, 3.0]), "b": 4})
    assert out == {"b": 4.0, "a": 2.0}
    assert seen == ["a", "b"]
