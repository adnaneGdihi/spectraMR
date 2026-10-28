"""F36 regression — validation must fail loud when every batch raises.

Before F36 (2026-05-22) a systematic validation crash (shape/channel
mismatch, OOM) was swallowed per-batch as a warning: ``_run_validation``
returned empty metrics, no images were saved, and the training run still
exited 0 = PASS. Smoke 20260521 had 36/41 "passing" experiments produce no
validation images this way (CLAUDE.md #10 silent-failure class).

These tests pin the new contract on the real ``_run_validation``:

* if validation was attempted and EVERY batch raised, the function raises
  RuntimeError (so the smoke gate records a FAIL); and
* a single failing batch among successful ones does NOT raise (transient
  failures are tolerated).
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.utils.config_block_stub import block_stub
from tests.utils.data_config_stub import DataConfigStub

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from spectramr.pipelines.train import _run_validation  # noqa: E402


def _logging_cfg() -> SimpleNamespace:
    # Disable image logging so the heavy capture path is skipped — this test
    # only exercises the per-batch failure accounting.
    # The shared stub: every one of these leaves folded under `logging.images.*`,
    # and a bare namespace leaves the reader without the sub-block at all
    # ("'SimpleNamespace' object has no attribute 'images'"). block_stub routes
    # them from RENAMES onto the real schema.
    return block_stub(
        "logging",
        log_validation_images=False,
        save_validation_images=False,
        validation_image_interval=1,
        max_images_per_batch=4,
        log_input_images=False,
        log_difference_images=True,
    )


def _make_pipeline(batches: list[dict]) -> SimpleNamespace:
    cfg = SimpleNamespace(
        validation=block_stub(
            "validation",
            num_validation_batches=len(batches),
            num_samples=None,
            effective_val_batch_size=1,
        ),
        logging=_logging_cfg(),
        # `sampling.patch_size` is non-optional now, so "no patch_size" is no
        # longer expressible — the empty namespace just made the read raise.
        # Match the 8x8 batches below instead, which is what the old absence
        # bought: `_preprocess_validation_tensor` only pads when H/W DIFFER
        # from patch_size[:2] (train.py:316), so an equal size is a no-op.
        data=DataConfigStub(patch_size=(8, 8)),
        # `model` is REQUIRED, not optional padding. #370 removed the
        # `getattr(getattr(config, "model", None), "spatial_dims", 2)` fallback in
        # `_preprocess_validation_tensor` (train.py:229) — it silently defaulted a
        # 3-D arm to spatial_dims=2 and flattened depth into the batch axis. A real
        # TrainingSettings always carries `.model`, so a stub without it is not
        # modelling anything reachable; it just re-hid the fallback #370 deleted.
        model=SimpleNamespace(spatial_dims=2),
    )
    return SimpleNamespace(
        ema=None,
        device=torch.device("cpu"),
        config=cfg,
        data_loaders={"val": batches},
        generator=torch.nn.Identity(),
        models={"generator": torch.nn.Identity()},
    )


def _batch() -> dict:
    return {
        "input": torch.randn(1, 1, 8, 8),
        "target": torch.randn(1, 1, 8, 8),
    }


class _AlwaysFailStrategy:
    """validation_step raises on every batch (Pattern A signature)."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(logging=_logging_cfg())

    def validation_step(self, input_batch, target_batch, batch_idx=0):
        raise RuntimeError("synthetic validation failure")


class _AlwaysOkStrategy:
    """validation_step returns a metric dict on every batch."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(logging=_logging_cfg())

    def validation_step(self, input_batch, target_batch, batch_idx=0):
        return {"val_loss": 0.5, "psnr": 25.0}

    def finalize_validation(self) -> dict:
        return {}


class _SometimesFailStrategy:
    """First batch raises, the rest succeed."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(logging=_logging_cfg())

    def validation_step(self, input_batch, target_batch, batch_idx=0):
        if batch_idx == 0:
            raise RuntimeError("transient batch failure")
        return {"val_loss": 0.5, "psnr": 25.0}

    def finalize_validation(self) -> dict:
        return {}


def test_all_batches_failing_raises() -> None:
    pipeline = _make_pipeline([_batch(), _batch()])
    with pytest.raises(RuntimeError, match="zero successful batches"):
        _run_validation(
            pipeline, _AlwaysFailStrategy(), iteration=10, epoch=0, logging_service=None
        )


def test_all_batches_failing_embeds_first_failure_traceback() -> None:
    """The raised error must carry the FIRST batch failure's root-cause
    traceback, not just point at a "logged above" line that a different log
    handler may have dropped (2026-06-21: ib_infonce / b17 validation crashes
    whose real shape-mismatch traceback was never persisted to the per-arm log).
    """
    pipeline = _make_pipeline([_batch(), _batch()])
    with pytest.raises(RuntimeError) as excinfo:
        _run_validation(
            pipeline, _AlwaysFailStrategy(), iteration=10, epoch=0, logging_service=None
        )
    msg = str(excinfo.value)
    assert "synthetic validation failure" in msg  # the underlying cause is embedded
    assert "root cause" in msg
    assert "Traceback" in msg  # the actual stack, in the persisted exception


def test_all_batches_succeeding_does_not_raise() -> None:
    pipeline = _make_pipeline([_batch(), _batch()])
    metrics = _run_validation(
        pipeline, _AlwaysOkStrategy(), iteration=10, epoch=0, logging_service=None
    )
    assert "val_loss" in metrics


def test_single_batch_failure_fails_the_run() -> None:
    """One batch fails and one succeeds: the survivors would select the checkpoint, so it raises."""
    pipeline = _make_pipeline([_batch(), _batch()])
    with pytest.raises(RuntimeError, match=r"lost 1 of 2 batch\(es\)") as excinfo:
        _run_validation(
            pipeline, _SometimesFailStrategy(), iteration=10, epoch=0, logging_service=None
        )
    assert "transient batch failure" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2026-06-21 — field-conditioned visual-capture context regression.
#
# The mrixfields field_* strategies (field_fno, field_bridge, koopman_field,
# confluence, …) read their conditioning (field_strength, field_strength_target,
# contrast_id, sources, input) from the ``batch_context`` handed to
# ``_validation_forward`` and RAISE when a required key is absent. The
# visual-capture path used to build a bare ``_batch_ctx = {use_dc,
# measured_kspace}``, so those strategies raised, the exception was swallowed
# (preds=None), and the validation IMAGE was silently dropped — even though the
# metrics path (which sees the full batch) computed val_psnr fine. That is the
# "metrics yes, image no" mrixfields symptom the forensics surfaced. The fix
# threads the batch's conditioning into the capture ctx.
# ---------------------------------------------------------------------------


def _image_logging_cfg() -> SimpleNamespace:
    # Image logging ON so the heavy visual-capture path runs; save-to-disk OFF
    # so the test stays filesystem-free.
    return block_stub(
        "logging",
        log_validation_images=True,
        save_validation_images=False,
        validation_image_interval=1,
        max_images_per_batch=4,
        log_input_images=False,
        log_difference_images=True,
    )


class _FieldConditionedStub:
    """Mirror of the FieldFNO-style contract: ``_validation_forward`` reads its
    conditioning from ``batch_context`` and raises when
    ``field_strength_target`` is missing; ``validation_step`` (metrics) succeeds
    on the full batch."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(logging=_image_logging_cfg())
        self.env = SimpleNamespace(generator=torch.nn.Identity())
        self.seen_ctx: dict | None = None

    def validation_step(self, input_batch, target_batch, batch_idx=0):
        return {"val_psnr": 25.0}

    def finalize_validation(self) -> dict:
        return {}

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        self.seen_ctx = dict(batch_context)
        if batch_context.get("field_strength_target") is None:
            raise ValueError(
                "field_* validation requires per-sample 'field_strength_target'"
            )
        return input_batch  # a valid prediction → visual_samples captured


def _field_batch() -> dict:
    return {
        "input": torch.randn(1, 1, 8, 8),
        "target": torch.randn(1, 1, 8, 8),
        "field_strength": torch.tensor([3.0]),
        "field_strength_target": torch.tensor([7.0]),
        "contrast_id": torch.tensor([0]),
    }


def test_field_conditioning_threaded_into_visual_capture_ctx() -> None:
    """The visual-capture ``_batch_ctx`` must carry the batch's conditioning so a
    field-conditioned ``_validation_forward`` renders instead of raising →
    preds=None → no validation image (the mrixfields cohort symptom)."""
    from unittest.mock import MagicMock

    pipeline = _make_pipeline([_field_batch()])
    strat = _FieldConditionedStub()
    metrics = _run_validation(
        pipeline,
        strat,
        iteration=0,
        epoch=0,
        logging_service=None,
        metrics_service=MagicMock(),
        tb_writer=MagicMock(),
    )
    # The metrics path succeeded (val_psnr present).
    assert "val_psnr" in metrics
    # The capture path actually reached _validation_forward …
    assert strat.seen_ctx is not None, "_validation_forward was never invoked"
    # … with the conditioning threaded through (the regression assertion).
    assert strat.seen_ctx.get("field_strength_target") is not None, (
        "field_strength_target missing from visual-capture batch_context — the "
        "conditioning was not threaded into _batch_ctx"
    )
    # The prepared input/target are passed too (other field_* strategies read
    # batch_context['input']).
    assert strat.seen_ctx.get("input") is not None


def _save_images_cfg() -> SimpleNamespace:
    # Image logging + save-to-disk ON so the capture AND save path both run; the
    # metrics_service is a capturing stub (MagicMock) so no files are written.
    return block_stub(
        "logging",
        log_validation_images=True,
        save_validation_images=True,
        validation_image_interval=1,
        max_images_per_batch=4,
        log_input_images=False,
        log_difference_images=True,
    )


class _DistributionHeadStub:
    """Mirror of HeteroscedasticULF: ``_validation_forward`` returns a 2-channel
    ``[mean, logvar]`` head for a 1-channel target, and
    ``_prediction_for_visualization`` slices channel 0 (the mean). Pins that
    ``_run_validation`` routes the captured prediction through the viz reducer
    before ``_to_magnitude`` (issue #371) — otherwise the saved fake is the
    ``sqrt(mean^2 + logvar^2)`` blend, not the restored image."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            logging=_save_images_cfg(), data=DataConfigStub(patch_size=(8, 8))
        )
        self.env = SimpleNamespace(generator=torch.nn.Identity())
        self.viz_reduced_from: tuple | None = None

    def validation_step(self, input_batch, target_batch, batch_idx=0):
        return {"val_gaussian_nll": 1.0}

    def finalize_validation(self) -> dict:
        return {}

    def _validation_forward(self, input_batch, batch_context, **kwargs):
        b, _, h, w = input_batch.shape
        mean = torch.full((b, 1, h, w), 0.4)
        logvar = torch.full((b, 1, h, w), -2.0)
        return torch.cat([mean, logvar], dim=1)  # [mean, logvar]

    def _prediction_for_visualization(self, predictions):
        self.viz_reduced_from = tuple(predictions.shape)
        if predictions.dim() >= 2 and predictions.shape[1] >= 2:
            return predictions[:, 0:1]
        return predictions


def test_distribution_head_prediction_reduced_before_save() -> None:
    """``_run_validation`` must route the captured 2-channel [mean, logvar] head
    through the strategy's viz reducer before ``_to_magnitude``, so the saved
    fake_images are the mean channel — not the sqrt(mean^2 + logvar^2) RSS blend
    (issue #371). Without the train.py hook the reducer is never called here."""
    from unittest.mock import MagicMock

    pipeline = _make_pipeline([_batch()])
    strat = _DistributionHeadStub()
    metrics_service = MagicMock()
    _run_validation(
        pipeline,
        strat,
        iteration=0,
        epoch=0,
        logging_service=None,
        metrics_service=metrics_service,
        tb_writer=MagicMock(),
    )
    # The regression: the viz reducer was invoked on the raw 2-channel head.
    assert strat.viz_reduced_from == (
        1,
        2,
        8,
        8,
    ), f"viz reducer not called on the 2-channel head; got {strat.viz_reduced_from}"
    # The save path ran and received a single-channel (mean) fake image.
    assert metrics_service.save_images_batch.called, "fake images were never saved"
    saved = metrics_service.save_images_batch.call_args.kwargs["fake_images"]
    assert (
        saved.shape[1] == 1
    ), f"saved fake is not single-channel: {tuple(saved.shape)}"


@pytest.mark.parametrize(
    "arm", ["ei_multicoil_m4raw_r4", "robust_ssdu_m4raw_r4", "robust_ssdu_selfcal_m4raw_r4"]
)
def test_a_batch_context_arm_saves_its_validation_image_through_the_loop(arm) -> None:
    """#2431: these arms logged metrics and never an image. The capture's bare context
    lacked the maps and measurement, and preparing the input folded its four coils
    across the axes; a builder-only test passed while the loop still dropped the image."""
    from unittest.mock import MagicMock

    from spectramr.pipelines.train import _visual_capture_state
    from tests.fixtures.self_supervised_batches import build_strategy, load_arm, real_batches

    config = load_arm(arm)
    _, val = real_batches(config, coils=4)
    strategy = build_strategy(config)
    pipeline = SimpleNamespace(
        ema=None,
        device=torch.device("cpu"),
        config=config,
        data_loaders={"val": [val]},
        generator=strategy.env.generator,
        models={"generator": strategy.env.generator},
    )
    tb_writer = MagicMock()
    _run_validation(
        pipeline, strategy, iteration=0, epoch=0, logging_service=None,
        metrics_service=MagicMock(), tb_writer=tb_writer,
    )
    assert _visual_capture_state(strategy).get("last_error") is None
    assert any("image" in name for name, _, _ in tb_writer.mock_calls), "no image was logged"



class _SentinelOnFirstBatchStrategy:
    """Returns the failure sentinel for batch 0 and real metrics after, as a strategy does
    when one batch yields no prediction (``DiffusionTrainingStrategy.validation_step``)."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(logging=_logging_cfg())

    def validation_step(self, input_batch, target_batch, batch_idx=0):
        if batch_idx == 0:
            return {"validation_error": 1.0}
        return {"val_loss": 0.5, "psnr": 25.0}

    def finalize_validation(self) -> dict:
        return {}


def test_a_sentinel_batch_counts_as_failed_not_scored() -> None:
    """Planted: before, the sentinel batch counted as a success and the run selected on the rest."""
    pipeline = _make_pipeline([_batch(), _batch()])
    with pytest.raises(RuntimeError, match=r"lost 1 of 2 batch\(es\)") as excinfo:
        _run_validation(
            pipeline, _SentinelOnFirstBatchStrategy(), iteration=10, epoch=0, logging_service=None
        )
    assert "failure sentinel" in str(excinfo.value)
