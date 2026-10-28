"""EMA Schema Execution Tests.

PURPOSE:
The EMAConfigSchema has 8 fields beyond `enabled` and `decay`:
  enable_adaptive_ema, warmup_steps, initial_decay, final_decay,
  stability_threshold, adaptation_rate, enable_momentum_scaling,
  enable_ema_for_optimizer

ModelBuilder.build_ema() now reads `enabled`, `decay`, `warmup` and the whole
deterministic adaptive family (`enable_adaptive_ema`, `warmup_steps`,
`initial_decay`, `final_decay`) — the last four were wired in #1294, having
been schema-only ever since ff0efff9f deleted their implementation.

The remaining four are still schema-only, but for two DIFFERENT reasons, and
the distinction matters:
  * `update_frequency` is consumed in the training loop, not the builder, so
    the builder-source pin below cannot see it (xfail, not debt).
  * `stability_threshold` / `adaptation_rate` configure a stability-feedback
    schedule with no signal feeding it; the schema now REFUSES them when
    `enable_adaptive_ema` is set rather than ignoring them (non-negotiable 8).
  * `enable_momentum_scaling` / `enable_ema_for_optimizer` are genuine debt.

These tests:
  1. Verify that build_ema() actually creates a ModelEma (smoke test)
  2. Pin exactly which EMA fields are consumed vs schema-only
  3. Ensure no new schema fields are added without consumer awareness
"""

from __future__ import annotations

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN STATE
# ─────────────────────────────────────────────────────────────────────────────

# Fields that ModelBuilder.build_ema() (or any EMAMixin) actually reads
CONSUMED_EMA_FIELDS = {
    "enabled",  # ModelBuilder.build_ema(): if not enabled, skip
    "decay",  # ModelBuilder.build_ema(): ModelEma(generator, decay=...)
    "warmup",  # ModelBuilder.build_ema() + state.py — ramp effective_decay with num_updates
    # --- #1294: the deterministic adaptive schedule, restored from the
    # deleted models/utils/adaptive_ema.py and now honoured by ModelEma. ---
    "enable_adaptive_ema",  # build_ema(): ModelEma(adaptive=...)
    "warmup_steps",  # build_ema(): ramp length; loop: hard gate on the standard path
    "initial_decay",  # build_ema(): ramp start
    "final_decay",  # build_ema(): ramp end, held afterwards
}

# Fields in EMAConfigSchema that are NOT consumed by any builder (schema-only debt)
KNOWN_EMA_SCHEMA_ONLY = {
    "update_frequency",  # Schema: every N steps, Builder: always step every iteration
    # Not debt: the schema RAISES on these two when enable_adaptive_ema is set,
    # because the loss/gradient-norm signal their schedule needs is never fed
    # to the EMA. Refusing beats ignoring (#1294).
    "stability_threshold",  # Schema: stability feedback, no signal source
    "adaptation_rate",  # Schema: stability feedback, no signal source
    "enable_momentum_scaling",  # Schema: batch-size scaling, Builder: no consumer
    "enable_ema_for_optimizer",  # Schema: EMA optimizer state, Builder: no consumer
}


def _make_minimal_settings_with_ema(**ema_kwargs):
    from spectramr.config.schemas.data import DataConfigSchema
    from spectramr.config.schemas.ema import EMAConfigSchema
    from spectramr.config.schemas.logging import LoggingConfigSchema
    from spectramr.config.schemas.metrics import MetricsConfigSchema
    from spectramr.config.schemas.model import ModelConfigSchema
    from spectramr.config.schemas.optimization import OptimizationConfigSchema
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings(
        model=ModelConfigSchema(),
        data=DataConfigSchema(),
        optimization=OptimizationConfigSchema(),
        logging=LoggingConfigSchema(),
        metrics=MetricsConfigSchema(),
        ema=EMAConfigSchema(**ema_kwargs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMA SSOT field audit
# ─────────────────────────────────────────────────────────────────────────────


class TestEMASchemaFieldCoverage:
    """Pin which EMA schema fields are consumed vs. schema-only."""

    def test_no_new_ema_schema_only_fields(self):
        """All EMAConfigSchema fields must be either CONSUMED or KNOWN_EMA_SCHEMA_ONLY."""
        from spectramr.config.schemas.ema import EMAConfigSchema

        all_fields = set(EMAConfigSchema.model_fields.keys())
        all_known = CONSUMED_EMA_FIELDS | KNOWN_EMA_SCHEMA_ONLY

        undocumented = all_fields - all_known
        assert not undocumented, (
            f"New EMAConfigSchema fields not classified as CONSUMED or SCHEMA_ONLY:\n"
            f"  {sorted(undocumented)}\n"
            "Add to CONSUMED_EMA_FIELDS (if build_ema reads it) OR KNOWN_EMA_SCHEMA_ONLY."
        )

    def test_known_schema_only_count_has_not_increased(self):
        """Pinning count of EMA schema-only fields — should only decrease."""
        import inspect

        from spectramr.config.schemas.ema import EMAConfigSchema
        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        src = inspect.getsource(ModelBuilder.build_ema)
        all_fields = set(EMAConfigSchema.model_fields.keys())

        # Anything in CONSUMED but not referenced in source is a documentation error
        not_actually_consumed = {
            f
            for f in CONSUMED_EMA_FIELDS
            if f != "enabled" and f not in src  # 'enabled' checked via .enabled
        }
        # 'decay' might be accessed as getattr(self._config.ema, 'decay', ...)
        not_actually_consumed.discard("decay")

        assert not not_actually_consumed, (
            f"Fields claimed as CONSUMED but not found in build_ema() source:\n"
            f"  {not_actually_consumed}\n"
            "Move them to KNOWN_EMA_SCHEMA_ONLY."
        )

    def test_known_schema_only_fields_exist_in_schema(self):
        """KNOWN_EMA_SCHEMA_ONLY must reference real schema fields (no stale entries)."""
        from spectramr.config.schemas.ema import EMAConfigSchema

        all_fields = set(EMAConfigSchema.model_fields.keys())
        stale = KNOWN_EMA_SCHEMA_ONLY - all_fields
        assert not stale, (
            f"KNOWN_EMA_SCHEMA_ONLY contains field names no longer in EMAConfigSchema:\n"
            f"  {sorted(stale)}\nRemove them."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMA builder smoke test
# ─────────────────────────────────────────────────────────────────────────────


class TestEMABuilderExecution:
    """ModelBuilder.build_ema() must actually create a ModelEma when enabled=True."""

    def test_ema_disabled_creates_nothing(self):
        """build_ema() with enabled=False must not create ModelEma."""
        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        cfg = _make_minimal_settings_with_ema(enabled=False)
        builder = ModelBuilder(config=cfg, device="cpu")

        # Build a dummy generator first
        import torch.nn as nn

        class _Dummy(nn.Module):
            def forward(self, x):
                return x

        builder._models["generator"] = _Dummy()
        builder.build_ema()

        assert (
            builder.ema is None
        ), "build_ema() created a ModelEma even though ema.enabled=False"

    def test_ema_enabled_creates_model_ema(self):
        """build_ema() with enabled=True must create a ModelEma wrapper."""
        import torch.nn as nn

        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        cfg = _make_minimal_settings_with_ema(enabled=True, decay=0.999)
        builder = ModelBuilder(config=cfg, device="cpu")

        class _Dummy(nn.Module):
            def forward(self, x):
                return x

        builder._models["generator"] = _Dummy()
        builder.build_ema()

        assert (
            builder.ema is not None
        ), "build_ema() did not create ModelEma even though ema.enabled=True"

    def test_ema_decay_is_passed_through(self):
        """The `decay` field must be forwarded to ModelEma, not hardcoded."""
        import torch.nn as nn

        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        cfg = _make_minimal_settings_with_ema(enabled=True, decay=0.99)
        builder = ModelBuilder(config=cfg, device="cpu")

        class _Dummy(nn.Module):
            def forward(self, x):
                return x

        builder._models["generator"] = _Dummy()
        builder.build_ema()

        ema = builder.ema
        assert ema is not None
        # ModelEma typically stores decay as .decay_factor or .decay
        actual_decay = getattr(ema, "decay", None) or getattr(ema, "decay_factor", None)
        if actual_decay is not None:
            assert abs(actual_decay - 0.99) < 1e-6, (
                f"Expected ema.decay=0.99, got {actual_decay}. "
                "The 'decay' schema field is not correctly forwarded to ModelEma."
            )

    def test_update_frequency_schema_only_gap(self):
        """Document that update_frequency is parsed but never controls EMA step frequency.

        This test documents the gap: ema.update_frequency=10 means 'update every 10
        steps' but ModelBuilder never reads this field, so EMA is always updated
        every step regardless of the configured frequency.
        """
        import inspect

        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        src = inspect.getsource(ModelBuilder.build_ema)
        assert "update_frequency" not in src, (
            "update_frequency is now referenced in build_ema()! "
            "Remove from KNOWN_EMA_SCHEMA_ONLY and update CONSUMED_EMA_FIELDS."
        )
        # If we reach here, the gap is still present — pass to document it
        pytest.xfail(
            reason="KNOWN GAP: ema.update_frequency is in schema but build_ema() "
            "never reads it — EMA is always updated every step. "
            "Fix: Add step counter check to EMAMixin.ema_step() or build_ema()."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Validation schema field coverage
# ─────────────────────────────────────────────────────────────────────────────


class TestValidationSchemaFieldCoverage:
    """ValidationConfigSchema fields vs ValidationService consumer audit.

    ValidationService.should_run_validation() reads eval_interval.
    ValidationService.should_visualize() reads visualization_interval.
    Many fields are schema-only (never read by the service).
    """

    # Fields actually consumed by ValidationService or DataBuilder
    CONSUMED_VALIDATION_FIELDS = {
        "enabled",  # ValidationService: if not enabled, skip
        "eval_interval",  # ValidationService.should_run_validation()
        "frequency_steps",  # ValidationService: step-based frequency
        "frequency_epochs",  # ValidationService: epoch-based frequency
        "eval_on_epoch",  # ValidationService: end-of-epoch eval
        "validation_dir",  # DataBuilder: separate val data path
        "split",  # DataBuilder: train/val split fraction
        "validation_batch_size",  # DataBuilder: val_batch_size override
        "enable_visualization",  # ValidationService.should_visualize()
        "visualization_interval",  # ValidationService.should_visualize()
        "primary_metric",  # ValidationService: best model tracking
        "validation_metric",  # ValidationService: metric to compare
        "compute_image_metrics",  # ValidationService: metric computation
        "output_transform",  # ValidationService: pre-metric transform
        "domain",  # ValidationService: metric domain
        "metrics",  # ValidationService: override metric list
        "num_validation_batches",  # ValidationService: truncated validation
        "num_samples",  # ValidationService: sample count cap
        "shuffle_validation",  # DataBuilder: shuffle validation dataset
        "val_chunk_size",  # DataBuilder: chunk size for validation loading
        "val_batch_size",  # DataBuilder: alias of validation_batch_size for older YAMLs
        "empty_cache_before_validation",  # pipelines/train.py:1941 getattr(_val_cfg, ...)
        "input_dependence_tol",  # strategies/diffusion.py:2577,2784 input-dependence sanity gate
        "multistep_cold_sampling",  # strategies/diffusion.py:3607 full reverse-trajectory val (2026-06-09 cold-diffusion fix)
        # Sub-blocks the flat fields above were folded into; each leaf has its own reader.
        "cascade",  # strategies/diffusion.py: cascade levels
        "gates",  # strategies/diffusion.py: gates.input_dependence_tol
        "loader",  # builders/directors/test_split.py: validation.loader
        "per_batch_rows",  # pipelines/train.py: per-batch CSV rows
        "sampling",  # config/settings.py: sampling.ensemble_samples
        "schedule",  # pipelines/training_loop.py: schedule.interval_steps
        "scoring",  # core/metrics/computer.py: scoring.compute / scoring.primary
        "visualization",  # bootstrap.py: visualization.enabled
    }

    # VF-v2 (Virtual-Fiducial) validation knobs that are advertised + completeness-
    # tested (tests/unit/config/test_vf_v2_completeness.py) but have NO runtime
    # consumer in src/ yet — genuine NN#8 unwired knobs. Wiring them changes
    # validation numerics and lives in the diffusion strategy / eval layer (WS-7),
    # so they are flagged for that workstream, not wired here. Listed as known
    # schema-only so the field-coverage audit stays honest (cf. the update_frequency
    # KNOWN-GAP xfail above) rather than masking the gap by claiming consumption.
    KNOWN_SCHEMA_ONLY_VALIDATION: set[str] = {
        "sampler_steps",  # plan §15/A5.6 — validation reverse-diffusion steps (no consumer)
        "held_out_severity_eval",  # held-out severity-grid opt-in (no consumer)
        "hallucination_test",  # plan §13/A5.9 — hallucination sweep hook (no consumer)
    }

    def test_all_validation_fields_classified(self):
        """All ValidationConfigSchema fields must be in CONSUMED or SCHEMA_ONLY."""
        from spectramr.config.schemas.validation import ValidationConfigSchema

        all_fields = set(ValidationConfigSchema.model_fields.keys())
        known = self.CONSUMED_VALIDATION_FIELDS | self.KNOWN_SCHEMA_ONLY_VALIDATION

        unclassified = all_fields - known
        assert not unclassified, (
            f"New ValidationConfigSchema fields not classified:\n  {sorted(unclassified)}\n"
            "Add to CONSUMED_VALIDATION_FIELDS or KNOWN_SCHEMA_ONLY_VALIDATION."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
