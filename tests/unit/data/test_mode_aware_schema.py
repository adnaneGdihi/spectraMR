"""Tests for the Phase-3 mode-aware data schema.

Coverage areas (per ``TODO/audit/data_layer_unification_plan.md`` Phase 3):

1. **Backward-compat shim** — ``DataConfigSchema.derive_modes_from_legacy``
   populates all 4 modes from legacy top-level fields when ``data.modes``
   is omitted.
2. **User overrides win** — a user-supplied ``data.modes.<mode>`` block
   completely replaces the derived value for that mode.
3. **Partial overrides** — declaring ``modes.train`` without
   ``modes.val`` leaves the other modes derived from legacy.
4. **Builder dispatch** — ``TorchIOQueueConfig.from_training_config``
   reads from ``modes.train`` / ``modes.val`` when present, falls back
   to legacy fields otherwise.
5. **Graph early-exit** — ``DataConfigSchema.is_graph_paradigm`` flips
   true when ``enable_graph_encoding`` is set.
6. **Frozen invariants** — the resolved ``modes`` is immutable; the
   ``resolve_mode`` helper never returns None.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.augmentation import AugmentationConfigSchema
from spectramr.config.schemas.data import (
    DataConfigSchema,
    DataModesSchema,
    ModeConfigSchema,
    ModeSamplerSchema,
)
from spectramr.data.builders.torchio_queue_builder import TorchIOQueueConfig

# ── Section 1: legacy → modes derivation ────────────────────────────────────


def test_legacy_yaml_with_no_modes_block_derives_train_uniform() -> None:
    """A legacy YAML (no ``data.modes``) → train mode is uniform with
    samples_per_volume / queue_length from top-level fields."""
    c = DataConfigSchema(samples_per_volume=8, queue_length=200)
    train = c.resolve_mode("train")
    assert train.sampler.type == "uniform"
    assert train.sampler.samples_per_volume == 8
    assert train.sampler.queue_length == 200
    assert train.sampler.shuffle is True


def test_legacy_yaml_derives_val_full_volume_by_default() -> None:
    """Default val behavior preserves current code: full volumes, no aug."""
    c = DataConfigSchema(samples_per_volume=8, queue_length=200)
    val = c.resolve_mode("val")
    assert val.sampler.type == "full"
    assert val.sampler.shuffle is False


def test_legacy_yaml_derives_infer_and_eval_full_no_aug() -> None:
    c = DataConfigSchema()
    for mode in ("infer", "eval"):
        cfg = c.resolve_mode(mode)
        assert cfg.sampler.type == "full"
        assert cfg.sampler.shuffle is False


def test_mode_block_has_no_augmentation_switch() -> None:
    """Nothing read ``modes.<mode>.augmentation_enabled``; declaring it raises.

    ``data.augmentation.enabled`` is the one switch: it gates the train chain, and
    the val chain builds no augmentation step.
    """
    with pytest.raises(ValidationError, match="augmentation_enabled"):
        ModeConfigSchema(augmentation_enabled=True)
    c = DataConfigSchema(augmentation=AugmentationConfigSchema(enabled=True))
    assert not hasattr(c.resolve_mode("train"), "augmentation_enabled")


# ── Section 2: user overrides ───────────────────────────────────────────────


def test_user_supplied_mode_block_overrides_legacy_fields() -> None:
    """When the user declares ``data.modes.train``, it wins entirely —
    no merging with legacy fields. CLAUDE.md #9 (no silent fallbacks)."""
    c = DataConfigSchema(
        samples_per_volume=8,  # legacy
        queue_length=100,
        modes=DataModesSchema(
            train=ModeConfigSchema(
                sampler=ModeSamplerSchema(
                    samples_per_volume=64, queue_length=512
                ),
            )
        ),
    )
    train = c.resolve_mode("train")
    assert train.sampler.samples_per_volume == 64
    assert train.sampler.queue_length == 512


def test_partial_overrides_leave_other_modes_legacy_derived() -> None:
    """Declaring only ``modes.train`` doesn't affect val/infer/eval —
    those still derive from legacy fields."""
    c = DataConfigSchema(
        samples_per_volume=8,
        modes=DataModesSchema(
            train=ModeConfigSchema(
                sampler=ModeSamplerSchema(samples_per_volume=32),
            )
        ),
    )
    assert c.resolve_mode("train").sampler.samples_per_volume == 32
    # Val derived from legacy → still 8 samples_per_volume in the sampler
    # config (even though sampler.type=full means it's unused).
    assert c.resolve_mode("val").sampler.samples_per_volume == 8
    assert c.resolve_mode("val").sampler.type == "full"


def test_user_can_request_uniform_val_for_patch_parity() -> None:
    """Phase-3 use case: surface patch-edge bias by sampling val
    patches identically to train."""
    c = DataConfigSchema(
        samples_per_volume=8,
        modes=DataModesSchema(
            val=ModeConfigSchema(
                sampler=ModeSamplerSchema(
                    type="uniform", samples_per_volume=4, shuffle=False,
                )
            )
        ),
    )
    val = c.resolve_mode("val")
    assert val.sampler.type == "uniform"
    assert val.sampler.samples_per_volume == 4
    assert val.sampler.shuffle is False


# ── Section 3: builder dispatch ─────────────────────────────────────────────


def test_queue_builder_reads_from_legacy_when_no_modes_override() -> None:
    """``TorchIOQueueConfig.from_training_config`` must still honor legacy
    YAMLs that don't carry a ``modes`` block (most of the repo today)."""
    c = DataConfigSchema(samples_per_volume=12, queue_length=200)
    qc = TorchIOQueueConfig.from_training_config(c)
    assert qc.samples_per_volume == 12
    assert qc.queue_length == 200
    assert qc.use_queue_for_validation is False  # default


def test_queue_builder_prefers_modes_train_when_present() -> None:
    c = DataConfigSchema(
        samples_per_volume=8,  # legacy — should be overridden
        queue_length=100,
        modes=DataModesSchema(
            train=ModeConfigSchema(
                sampler=ModeSamplerSchema(
                    samples_per_volume=64, queue_length=512
                )
            ),
        ),
    )
    qc = TorchIOQueueConfig.from_training_config(c)
    assert qc.samples_per_volume == 64
    assert qc.queue_length == 512


def test_queue_builder_dispatches_val_uniform_via_modes_block() -> None:
    """``modes.val.sampler.type == "uniform"`` is the mode-aware
    equivalent of the legacy ``use_queue_for_validation: true``."""
    c = DataConfigSchema(
        samples_per_volume=8,
        modes=DataModesSchema(
            val=ModeConfigSchema(
                sampler=ModeSamplerSchema(type="uniform", samples_per_volume=4)
            ),
        ),
    )
    qc = TorchIOQueueConfig.from_training_config(c)
    assert qc.use_queue_for_validation is True


# ── Section 4: graph early-exit ─────────────────────────────────────────────


def test_is_graph_paradigm_false_by_default() -> None:
    assert DataConfigSchema().is_graph_paradigm() is False


def test_is_graph_paradigm_true_when_graph_encoding_enabled() -> None:
    """Amendment A: graph datasets bypass the patch/resample pipeline."""
    c = DataConfigSchema(domain={"enable_graph_encoding": True})
    assert c.is_graph_paradigm() is True


def test_graph_paradigm_with_modes_still_dispatches_modes() -> None:
    """Graph paradigm CAN carry ``modes`` (e.g., for a sampler or
    transforms_override); ``is_graph_paradigm`` is purely an
    early-exit signal for the spatial-sampling layer.

    The downstream builder (Phase 3.5 / Phase 2) consults
    ``is_graph_paradigm`` to skip GridSampler / Resample only — other
    mode-aware behavior remains in effect."""
    c = DataConfigSchema(
        domain={"enable_graph_encoding": True},
        modes=DataModesSchema(
            train=ModeConfigSchema(transforms_override=[{"name": "noop"}])
        ),
    )
    assert c.is_graph_paradigm() is True
    assert c.resolve_mode("train").transforms_override == [{"name": "noop"}]


# ── Section 5: invariants ───────────────────────────────────────────────────


def test_resolve_mode_returns_non_none_for_all_modes() -> None:
    """Post-validation invariant: every mode resolves to a real
    ``ModeConfigSchema``. The internal ``modes.<x>`` may have been None
    in the user input, but ``derive_modes_from_legacy`` fills them all in."""
    c = DataConfigSchema()
    for mode in ("train", "val", "infer", "eval"):
        cfg = c.resolve_mode(mode)
        assert cfg is not None
        assert isinstance(cfg, ModeConfigSchema)


def test_resolve_mode_raises_on_invalid_mode_name() -> None:
    c = DataConfigSchema()
    # Literal["train","val","infer","eval"] guarantees this at the type
    # level, but the runtime fallback should still surface unknown modes
    # rather than silently return None.
    with pytest.raises(RuntimeError, match="invariant broken"):
        c.resolve_mode("nope")  # type: ignore[arg-type]


def test_schema_is_frozen_no_post_init_mutation() -> None:
    """The whole DataConfigSchema is frozen — even after the validator
    populates ``modes``, callers can't mutate it."""
    c = DataConfigSchema()
    with pytest.raises((TypeError, ValueError)):  # pydantic frozen → ValidationError
        c.samples_per_volume = 99  # type: ignore[misc]


def test_modes_schema_rejects_unknown_keys() -> None:
    """``DataModesSchema`` and ``ModeConfigSchema`` use
    ``extra='forbid'`` — typos in mode names raise at config load."""
    with pytest.raises(Exception):  # pydantic ValidationError
        DataModesSchema(traain=ModeConfigSchema())  # type: ignore[call-arg]


# ── Section 6: edge case — eval role default ────────────────────────────────


def test_derived_eval_mode_has_clean_reference_role() -> None:
    """Legacy → derived ``eval`` mode defaults role='clean_reference'."""
    c = DataConfigSchema()
    eval_cfg = c.resolve_mode("eval")
    assert eval_cfg.role == "clean_reference"


def test_user_can_override_eval_role_to_uncertainty() -> None:
    c = DataConfigSchema(
        modes=DataModesSchema(
            eval=ModeConfigSchema(role="uncertainty"),
        ),
    )
    assert c.resolve_mode("eval").role == "uncertainty"
