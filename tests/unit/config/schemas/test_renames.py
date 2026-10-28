"""Tests for the config-key rename SSOT.

Two halves. The *mechanism* is exercised against a fixture table, so it is
covered even when the production table is small; the *table* is checked against
the live schema, so an entry cannot rot into the defect class it exists to
remove.
"""

from __future__ import annotations

import pathlib
from typing import ClassVar

import pytest
from pydantic import BaseModel, ValidationError, model_validator

from spectramr.config.schemas.renames import (
    _DRAINED,
    RENAMES,
    ROOT,
    RenameRecord,
    fold_renamed_keys,
    reject_renamed_keys,
    renames_for_block,
)
from tests.utils.corpus import repo_root, tracked_yamls

_FIXTURE = {
    "demo.old_name": RenameRecord(
        legacy="demo.old_name",
        canonical="demo.new_name",
        since="2026-01-01",
        reason="Two spellings of one knob.",
    ),
    "demo.disable_thing": RenameRecord(
        legacy="demo.disable_thing",
        canonical="demo.enable_thing",
        since="2026-01-01",
        reason="Negated booleans are forbidden by the naming rule.",
        value_transform="negate",
    ),
    "other.moved": RenameRecord(
        legacy="other.moved",
        canonical="other.arrived",
        since="2026-01-01",
        reason="Unrelated block; must not leak into demo's validator.",
    ),
}


class _Demo(BaseModel):
    model_config = {"extra": "ignore"}
    new_name: int = 1
    enable_thing: bool = False

    _reject = model_validator(mode="before")(classmethod(reject_renamed_keys("demo", _FIXTURE)))


class TestRejectionShim:
    def test_retired_key_raises(self) -> None:
        with pytest.raises(ValueError, match=r"demo\.old_name"):
            _Demo(old_name=3)

    def test_message_names_both_spellings_and_the_fixer(self) -> None:
        """A rejection that does not say what to write instead is a worse error
        than the silent swallow it replaced."""
        with pytest.raises(ValueError) as exc:
            _Demo(old_name=3)
        msg = str(exc.value)
        assert "demo.old_name" in msg and "demo.new_name" in msg
        assert "migrate_config_keys.py" in msg

    def test_negate_record_says_to_invert_the_value(self) -> None:
        """A sense inversion is not a mechanical key rewrite: `disable: true` and
        `enable: true` are opposite behaviours."""
        with pytest.raises(ValueError) as exc:
            _Demo(disable_thing=True)
        assert "inverted" in str(exc.value)

    def test_canonical_key_still_loads(self) -> None:
        assert _Demo(new_name=7).new_name == 7

    def test_other_blocks_entries_do_not_leak(self) -> None:
        """The validator is built per block; `other.moved` must not fire here."""
        _Demo(moved=1)  # extra="ignore" swallows it; the point is it does not raise

    def test_shim_is_inert_when_the_block_has_no_entries(self) -> None:
        validator = reject_renamed_keys("nonexistent_block", _FIXTURE)
        assert validator(_Demo, {"anything": 1}) == {"anything": 1}


class TestRenamesForBlock:
    def test_groups_by_owning_block(self) -> None:
        assert set(renames_for_block("demo", _FIXTURE)) == {
            "old_name",
            "disable_thing",
        }
        assert set(renames_for_block("other", _FIXTURE)) == {"moved"}


class TestRootLevelKeys:
    """A legacy key with no dot sat at the document ROOT.

    ``RenameRecord.block`` used to be ``legacy.split(".", 1)[0]``, which for a
    bare ``seed`` returns ``"seed"`` — so the record would have been grouped
    under a block named after itself, and ``reject_renamed_keys`` on
    ``TrainingSettings`` would never have seen it. The shim would have been
    silently inert, which is the failure mode this whole table exists to
    prevent.
    """

    _ROOT_FIXTURE: ClassVar[dict] = {
        "bare": RenameRecord(
            legacy="bare",
            canonical="run.bare",
            since="2026-01-01",
            reason="Root scalar.",
        ),
        "nested.key": RenameRecord(
            legacy="nested.key",
            canonical="run.key",
            since="2026-01-01",
            reason="Nested.",
        ),
    }

    def test_a_dotless_legacy_path_belongs_to_the_root(self) -> None:
        assert self._ROOT_FIXTURE["bare"].block == ROOT
        assert self._ROOT_FIXTURE["nested.key"].block == "nested"

    def test_the_root_validator_sees_only_root_records(self) -> None:
        assert set(renames_for_block(ROOT, self._ROOT_FIXTURE)) == {"bare"}
        assert set(renames_for_block("nested", self._ROOT_FIXTURE)) == {"key"}

    def test_the_production_table_has_root_entries_and_they_are_grouped(self) -> None:
        """Anti-vacuity: if the table ever holds no root records, the grouping
        above proves nothing about production."""
        root_records = renames_for_block(ROOT)
        assert root_records, "no root-level renames in the production table"
        for key in root_records:
            assert "." not in key


class TestCreateDestinationBlock:
    def test_it_is_off_by_default(self) -> None:
        """A missing destination block usually means the arm relies on that
        block's defaults; conjuring it invents structure the author never
        wrote. Only a move into a genuinely NEW block opts in."""
        rec = RenameRecord(legacy="a.b", canonical="c.d", since="2026-01-01", reason="x")
        assert rec.create_destination_block is False

    def test_every_record_targeting_a_new_block_opts_in(self) -> None:
        """``run:`` exists in no file, so a record moving into it MUST opt in or
        the migrator refuses every single arm."""
        for rec in RENAMES.values():
            if rec.canonical.startswith("run."):
                assert rec.create_destination_block, (
                    f"{rec.legacy} -> {rec.canonical} targets the `run:` block, "
                    "which no config has; without create_destination_block the "
                    "fixer skips every file and the gate never reaches zero."
                )


class TestProductionTableDoesNotRot:
    """Anti-rot: the table must describe the schema as it actually is.

    A rename record pointing at a field that does not exist is the same defect
    the rename exists to remove, one level up — the error message would send an
    author to a key that is itself undeclared.
    """

    @staticmethod
    def _resolve(dotted: str) -> bool:
        """Whether ``dotted`` resolves to a real field from ``TrainingSettings``.

        Explores EVERY model in a union annotation, not just the first.

        The original took ``nested[0]``. Against ``A | B`` that silently tested
        only ``A`` and reported the whole path unresolvable if the field lived on
        ``B``. The discriminated union at ``training.diffusion`` (phase 2b) makes
        that the normal case, so the walk branches instead of guessing.

        Model discovery is delegated to ``schemas.introspection.nested_models``
        rather than done here. My first attempt at this fix hand-rolled a
        one-level ``get_args`` walk and STILL failed on the union it was written
        for: the real annotation is
        ``Optional[Annotated[A | B | ..., FieldInfo(discriminator=...)]]``, so the
        models sit two wrappers down. Six introspection sites in this repo made
        the same one-level assumption; they now share one implementation.
        """
        from spectramr.config.schemas.introspection import nested_models as _models
        from spectramr.config.settings import TrainingSettings

        def _walk(model: type[BaseModel], parts: list[str]) -> bool:
            field = model.model_fields.get(parts[0])
            if field is None:
                return False
            if len(parts) == 1:
                return True
            # Any branch that resolves the remainder is enough: the discriminator
            # picks one variant at runtime, and a path real on that variant is a
            # real path.
            return any(_walk(nested, parts[1:]) for nested in _models(field.annotation))

        return _walk(TrainingSettings, dotted.split("."))

    @pytest.mark.parametrize("legacy", sorted(RENAMES))
    def test_canonical_path_resolves_to_a_real_field(self, legacy: str) -> None:
        rec = RENAMES[legacy]
        assert self._resolve(rec.canonical), (
            f"{rec.legacy} points at {rec.canonical}, which is not a field on "
            "TrainingSettings. The error message would send an author to a key "
            "that does not exist."
        )

    @pytest.mark.parametrize("legacy", sorted(RENAMES))
    def test_legacy_path_is_actually_gone(self, legacy: str) -> None:
        """An entry means the key is retired. If it still resolves, the shim and
        the field contradict each other and the field wins."""
        rec = RENAMES[legacy]
        assert not self._resolve(rec.legacy), (
            f"{rec.legacy} is in the rename table but still declared as a field; "
            "delete the field or drop the entry."
        )

    def test_keys_match_their_records(self) -> None:
        for key, rec in RENAMES.items():
            assert key == rec.legacy


class TestFoldPosture:
    """``fold`` stages a rename the corpus is too large to land atomically.

    The distinction that matters: a fold keeps the legacy spelling working in
    YAML while removing it from Python. If it ever leaked onto the model it
    would be the forwarding property the design forbids, so the first test
    below is the load-bearing one.
    """

    @staticmethod
    def _table(**kw):
        rec = RenameRecord(
            legacy=kw.pop("legacy", "blk.old"),
            canonical=kw.pop("canonical", "blk.sub.new"),
            since="2026-07-31",
            reason="test",
            posture="fold",
            **kw,
        )
        return {rec.legacy: rec}

    def test_a_fold_moves_the_value_instead_of_raising(self) -> None:
        table = self._table()
        out = fold_renamed_keys("blk", table)(None, {"old": 7})
        assert out == {"sub": {"new": 7}}

    def test_the_legacy_key_does_not_survive(self) -> None:
        """The whole point. A fold that left the key behind would hand
        ``extra="forbid"`` an unknown field, and a fold that added a property
        would recreate two read paths."""
        out = fold_renamed_keys("blk", self._table())(None, {"old": 7})
        assert "old" not in out

    def test_it_does_not_mutate_the_caller_dict(self) -> None:
        table = self._table()
        data = {"old": 7, "sub": {"other": 1}}
        fold_renamed_keys("blk", table)(None, data)
        assert data == {"old": 7, "sub": {"other": 1}}

    def test_an_existing_sibling_in_the_destination_survives(self) -> None:
        out = fold_renamed_keys("blk", self._table())(None, {"old": 7, "sub": {"k": 1}})
        assert out == {"sub": {"k": 1, "new": 7}}

    def test_agreeing_duplicates_collapse(self) -> None:
        out = fold_renamed_keys("blk", self._table())(None, {"old": 7, "sub": {"new": 7}})
        assert out == {"sub": {"new": 7}}

    def test_disagreeing_duplicates_raise(self) -> None:
        """Picking one would silently decide what the arm trains."""
        with pytest.raises(ValueError, match="disagree"):
            fold_renamed_keys("blk", self._table())(None, {"old": 7, "sub": {"new": 9}})

    def test_a_negating_fold_inverts_the_value(self) -> None:
        table = self._table(value_transform="negate")
        assert fold_renamed_keys("blk", table)(None, {"old": True}) == {"sub": {"new": False}}

    def test_a_non_bool_under_negate_raises(self) -> None:
        table = self._table(value_transform="negate")
        with pytest.raises(ValueError, match="must be a bool"):
            fold_renamed_keys("blk", table)(None, {"old": "yes"})

    def test_reject_and_fold_do_not_serve_each_others_records(self) -> None:
        """Two validators, one table. If ``reject`` picked up a fold record the
        staging would be undone silently and 826 arms would stop loading."""
        table = self._table()
        assert reject_renamed_keys("blk", table)(None, {"old": 7}) == {"old": 7}
        raising = {
            "blk.gone": RenameRecord(
                legacy="blk.gone", canonical="blk.sub.new", since="x", reason="y"
            )
        }
        assert fold_renamed_keys("blk", raising)(None, {"gone": 1}) == {"gone": 1}

    @pytest.mark.parametrize("legacy", sorted(k for k, r in RENAMES.items() if r.posture == "fold"))
    def test_every_fold_stays_inside_its_own_block(self, legacy: str) -> None:
        """A per-block fold validator is mounted on ONE class and cannot reach a
        sibling, so a cross-block fold would drop the value on the floor.

        ROOT records are exempt, and not by widening the predicate to make a
        failure go away: the root validator is mounted on `TrainingSettings`,
        which genuinely CAN reach every top-level block, and pydantic runs it
        before any sub-model is constructed. That is the whole reason a block
        rename (`acceleration:` -> `undersampling:`) has to live at the root.
        """
        rec = RENAMES[legacy]
        if rec.block == ROOT:
            return
        assert rec.canonical.split(".", 1)[0] == rec.block, (
            f"{rec.legacy} folds into {rec.canonical}, a different top-level "
            "block. Use posture='raise' for a cross-block move."
        )


class TestPhase8FoldTableIsTotal:
    """Every field that ``optimization:`` used to declare must be accounted for.

    ``optimization`` is ``extra="forbid"``. Before phase 8 all 43 names below
    were declared fields, so every one of them loaded. The moment a field moved,
    any name this table forgot went from "works" to hard ValidationError on
    every arm that sets it -- and the popular keys are exactly the ones you
    check by hand, so the miss would be in the tail.
    """

    #: The pre-phase-8 field list, read off the class before the split.
    FIELDS_BEFORE_PHASE_8: frozenset[str] = frozenset(
        {
            "learning_rate",
            "generator_learning_rate",
            "discriminator_learning_rate",
            "optimizer_type",
            "weight_decay",
            "beta1",
            "beta2",
            "betas",
            "eps",
            "momentum",
            "nesterov",
            "amsgrad",
            "optimizer_kwargs",
            "lookahead",
            "use_amp",
            "amp_dtype",
            "use_gradient_checkpointing",
            "gradient_checkpointing",
            "enable_memory_monitoring",
            "memory_monitoring_interval",
            "enable_memory_fragmentation_mitigation",
            "memory_cleanup_interval",
            "optimize_batch_size_for_memory",
            "memory_safety_margin",
            "enable_gradient_clipping",
            "gradient_clip_method",
            "gradient_clip_value",
            "detect_anomalies",
            "lr_scheduler_strategy",
            "warmup_steps",
            "lr_scheduler_kwargs",
            "scheduler",
            "param_group_overrides",
            "scheduler_type",
            "T_max",
            "eta_min",
            "compile_model",
            "compile_mode",
            "compile_backend",
            "compile_fullgraph",
            "compile_dynamic",
            "gradient_accumulation_steps",
            "num_steps",
        }
    )

    def test_the_pinned_list_is_the_size_it_was(self) -> None:
        assert len(self.FIELDS_BEFORE_PHASE_8) == 43

    @pytest.mark.parametrize("name", sorted(FIELDS_BEFORE_PHASE_8))
    def test_every_old_field_still_has_somewhere_to_go(self, name: str) -> None:
        from spectramr.config.schemas.optimization import OptimizationConfigSchema

        still_declared = name in OptimizationConfigSchema.model_fields
        has_record = f"optimization.{name}" in RENAMES
        assert still_declared or has_record, (
            f"`optimization.{name}` is neither a field nor a rename record. "
            "Under extra='forbid' every arm that sets it now fails to load with "
            "'extra fields not permitted' and no hint about where it went."
        )


class TestPhase9FoldTableIsTotal:
    """Every field ``data:`` used to declare must be accounted for.

    This matters MORE than its phase-8 twin, because the two blocks have
    opposite strictness. ``optimization`` is ``extra="forbid"``: a fold-table
    gap there produced a hard ValidationError on the next load, so the totality
    test was belt-and-braces. ``data`` is ``extra="ignore"`` -- a gap here is
    SILENCE. The key vanishes from the resolved config and the arm trains on the
    schema default, which is the issue #550 mechanism exactly.

    Nor do the two parametrized tests above cover it: they check that every
    RECORD is well-formed, and a field moved-and-forgotten has no record.
    """

    #: The pre-phase-9 scalar field list, read off the live class before the
    #: split. Nested sub-blocks (`augmentation`, `caching`, `multi_domain`, ...)
    #: are excluded -- they were already grouped and do not move.
    SCALARS_BEFORE_PHASE_9: frozenset[str] = frozenset(
        {
            "allow_unpaired",
            "batch_size",
            "bidirectional_mode",
            "coil_processing_mode",
            "contrasts",
            "data_layout",
            "data_range",
            "data_root",
            "dataset_type",
            "enable_graph_encoding",
            "enable_slab_mode",
            "expose_acquisition_params",
            "expose_conformal_jacobian",
            "expose_cortex_flatten_grid",
            "expose_field_strength",
            "expose_field_strength_target",
            "expose_glm_design_matrix",
            "expose_scanner_id",
            "expose_site_id",
            "extra_kwargs",
            "graph_config",
            "graph_type",
            "hf_resolution",
            "holdout_site",
            "holdout_subject",
            "image_undersampling",
            "index_path",
            "input_artifact",
            "known_dataset",
            "kspace_percentile",
            "kspace_scale_domain",
            "log_scaling",
            "log_scaling_center_fraction",
            "loso_fold",
            "max_prefetch",
            "max_train_subjects",
            "max_val_subjects",
            "mrixfields_max_resident_volumes",
            "mrixfields_output_contrast",
            "mrixfields_pairing_policy",
            "mrixfields_rescale_per_image",
            "mrixfields_slice_mode",
            "mrixfields_target_field",
            "multislice_enabled",
            "nex_target_exclude_input",
            "normalization_kwargs",
            "normalization_type",
            "normalize_images",
            "normalize_kspace",
            "num_synthetic_samples",
            "num_virtual_coils",
            "num_workers",
            "output_domain",
            "paired_manifest_path",
            "patch_size",
            "pde_problem",
            "persistent_workers",
            "phase_encode_axis",
            "pin_memory",
            "prefetch_factor",
            "preprocessing_dir",
            "queue_length",
            "rescale_images",
            "rescale_percentiles",
            "rescale_range",
            "return_image_domain",
            "samples_per_volume",
            "sessions",
            "single_contrast",
            "slice_2d",
            "slice_cache_size",
            "split_strategy",
            "svd_calibration_lines",
            "target_artifact",
            "target_channels",
            "target_contrasts",
            "target_mode",
            "target_sessions",
            "test_split",
            "train_sites",
            "trajectory",
            "transforms",
            "use_async_dataloader",
            "use_repetitions",
            "validation_index_path",
            "validation_split",
        }
    )

    def test_the_pinned_list_is_the_size_it_was(self) -> None:
        assert len(self.SCALARS_BEFORE_PHASE_9) == 86

    def test_the_flat_reserved_cores_knob_is_folded_like_its_loader_siblings(self) -> None:
        # Added after the phase-9 list was pinned, so it is outside the 86 above.
        assert RENAMES["data.reserved_cores"].canonical == "data.loader.reserved_cores"
        assert RENAMES["data.reserved_cores"].posture == RENAMES["data.num_workers"].posture

    @pytest.mark.parametrize("name", sorted(SCALARS_BEFORE_PHASE_9))
    def test_every_old_scalar_still_has_somewhere_to_go(self, name: str) -> None:
        from spectramr.config.schemas.data import DataConfigSchema

        still_declared = name in DataConfigSchema.model_fields
        has_record = f"data.{name}" in RENAMES
        assert still_declared or has_record, (
            f"`data.{name}` is neither a field nor a rename record. `data` is "
            "extra='ignore', so every arm that sets it now has the value "
            "SILENTLY DISCARDED and trains on the default -- no error, no "
            "warning, a different experiment."
        )


class TestNoDataKeyIsSilentlyDropped:
    """The end-to-end net for phase 9, on real corpus files.

    The totality test above is a unit check against `model_fields`. This one
    asks the question that actually matters: when a real arm loads, does any
    `data.*` key it declared vanish? `data` is ``extra="ignore"``, so the
    failure is silent by construction -- the execution ledger's
    ``EXTRA_IGNORE_DROPPED`` record is the only thing that can see it, which is
    exactly what it was built for (#550).

    The two nets fail differently: a fold-table gap that the unit test misses
    (say, a key whose field was never in the pinned list) still shows up here.
    """

    #: Dropped ``data.*`` declarations across the whole ``inprogress/`` cohort,
    #: measured 2026-08-13: 620 across 244 arms and 29 distinct paths, with 647
    #: of 647 loading and none refused.
    #:
    #: A SHRINK-ONLY ratchet, the pattern ``UNDISPATCHABLE_BASELINE`` already
    #: uses in ``test_metric_transform.py``. It exists because both honest
    #: alternatives were worse: asserting zero keeps ``dev`` red for debt no
    #: single PR can clear, and deleting the offending keys from the six sampled
    #: arms would turn the gate green while 244 arms kept losing keys -- the
    #: facade pitfall #16 names. This decouples "stop the bleeding" from "decide
    #: wire / fold / delete for 29 paths", which is an owner call (#1012, and the
    #: same call #675 and #681 need for their blocks).
    #:
    #: Baselined on the COUNT, not a list of arm paths: ``witness_baseline.txt``
    #: is path-keyed and #658 records what that cost -- promoting one arm
    #: ``inprogress/`` -> ``active/`` read as 388 NEW drops.
    DATA_DROP_BASELINE = 620

    #: The leaf paths those drops land on. A count-only ratchet would let a brand
    #: new phantom key hide behind any decrease elsewhere, so the NAMES are
    #: pinned too: an unlisted key fails immediately even while the total falls.
    KNOWN_DROPPED_DATA_PATHS = frozenset(
        {
            "data.caching.dataset_type",
            "data.caching.path",
            "data.channels",
            "data.coil_sensitivity_source",
            "data.data_type",
            "data.dataset_name",
            "data.enable_geometric_standardization",
            "data.ground_truth_folder",
            "data.image_size",
            "data.img_size",
            "data.input_domain",
            "data.input_hr_dir",
            "data.input_lr_dir",
            "data.limit_samples",
            "data.multicoil",
            "data.normalization",
            "data.phase",
            "data.preprocessing",
            "data.shape_spec",
            "data.site_partition",
            "data.slice_aware",
            "data.synthetic_samples",
            "data.target_is_complex",
            "data.train_manifest",
            "data.transform_strategy",
            "data.val_manifest",
            "data.volume_format",
        }
    )

    @staticmethod
    def _census():
        """Every dropped ``data.*`` declaration in the cohort, counted by path.

        Whole-cohort rather than a six-arm sample: the sample reported six
        offenders while 244 arms were losing keys, a misleading floor for the
        number this gate exists to hold down. 647 arms scan in ~16 s.
        """
        import collections

        from spectramr.config.settings import TrainingSettings
        from spectramr.core.execution_ledger import ExecutionLedger, SubstitutionClass

        counts: collections.Counter = collections.Counter()
        checked = 0
        for path in tracked_yamls("experiments/inprogress"):
            ledger = ExecutionLedger.begin_run(source="test")
            try:
                TrainingSettings.from_yaml(str(path))
            except Exception:
                continue  # a broken arm is a different test's problem
            checked += 1
            for sub in ledger.substitutions:
                if sub.class_id is SubstitutionClass.EXTRA_IGNORE_DROPPED and sub.path.startswith(
                    "data."
                ):
                    counts[sub.path] += 1
        return checked, counts

    def test_dropped_data_keys_do_not_increase(self) -> None:
        """``data`` is extra='ignore', so a lost key is silent by construction.

        The run does not fail -- it trains on the default. The execution
        ledger's ``EXTRA_IGNORE_DROPPED`` record is the only thing that can see
        it, which is what it was built for (#550).
        """
        pytest.importorskip("torch")
        checked, counts = self._census()
        assert checked, "no arm loaded -- this assertion would be vacuous"

        total = sum(counts.values())
        assert total <= self.DATA_DROP_BASELINE, (
            f"{total} dropped `data.*` declarations, up from the "
            f"{self.DATA_DROP_BASELINE} baseline. A key declared under `data:` "
            "that the schema does not define is discarded SILENTLY -- the arm "
            "trains on the default. Add it to the fold table in renames.py, or "
            "delete it from the arm.\n  by path: "
            + ", ".join(f"{p}={n}" for p, n in counts.most_common(8))
        )

    def test_no_new_dropped_data_path_appears(self) -> None:
        """Teeth the count alone would not keep.

        A new phantom key could hide behind a decrease elsewhere and leave the
        total under baseline, so the leaf NAMES are ratcheted separately.
        """
        pytest.importorskip("torch")
        checked, counts = self._census()
        assert checked, "no arm loaded -- this assertion would be vacuous"

        unknown = sorted(set(counts) - self.KNOWN_DROPPED_DATA_PATHS)
        assert not unknown, (
            "these `data.*` paths are newly discarded and are not in the known "
            f"set: {unknown}\n`data` is extra='ignore', so the run does NOT "
            "fail -- it trains on the default. Add the key to the fold table in "
            "renames.py, or delete it from the arm."
        )


class TestFoldMergesIntoAPartiallyAuthoredTarget:
    """A fold must MERGE into an existing target block, never replace it.

    ``TestFoldPosture`` covers the empty target and the disagreeing duplicate.
    The case between them is the dangerous one: the author wrote a sibling of
    the canonical leaf, so the target block already exists as a dict. Replacing
    it would drop that sibling silently -- and silently is the whole problem,
    since these blocks are ``extra="ignore"``.
    """

    @staticmethod
    def _table():
        rec = RenameRecord(
            legacy="blk.old",
            canonical="blk.sub.new",
            since="2026-08-01",
            reason="test",
            posture="fold",
        )
        return {rec.legacy: rec}

    def test_a_sibling_in_the_target_block_survives(self) -> None:
        out = fold_renamed_keys("blk", self._table())(
            None, {"old": 7, "sub": {"untouched": "keep me"}}
        )
        assert out == {"sub": {"untouched": "keep me", "new": 7}}

    def test_an_agreeing_duplicate_is_dropped_not_doubled(self) -> None:
        out = fold_renamed_keys("blk", self._table())(None, {"old": 7, "sub": {"new": 7}})
        assert out == {"sub": {"new": 7}}

    def test_the_callers_nested_dict_is_not_mutated(self) -> None:
        """Copy-on-write: the input document may be shared."""
        sub = {"untouched": "keep me"}
        data = {"old": 7, "sub": sub}
        fold_renamed_keys("blk", self._table())(None, data)
        assert sub == {"untouched": "keep me"}, "caller's dict was mutated"


class TestFoldValidatorRunsLast:
    """The fold must be the LAST ``mode="before"`` validator on its block.

    Every other before-validator on these classes reads and writes the *legacy*
    flat namespace -- ``DataConfigSchema.migrate_legacy_sizes`` injects the
    M4Raw preset (``coil_processing_mode``, ``use_repetitions``, ...) and
    rewrites ``known_dataset`` into ``dataset_type``. If the fold ran first,
    those writes would land at the old level *after* it, and ``extra="ignore"``
    would discard them without a word (#550).

    Pydantic runs ``mode="before"`` model validators in REVERSE definition
    order, so "runs last" means "mounted FIRST in the class body" -- the exact
    opposite of how the file reads top-to-bottom. That inversion is why this is
    pinned rather than left to a comment: a new validator added above the mount
    would break it, and nothing else would notice.
    """

    @pytest.mark.parametrize("module", ["data", "optimization"])
    def test_the_fold_mount_precedes_every_other_before_validator(self, module: str) -> None:
        import importlib
        import inspect
        import re

        src = inspect.getsource(importlib.import_module(f"spectramr.config.schemas.{module}"))
        lines = src.splitlines()
        mount = next(i for i, ln in enumerate(lines) if "_fold_renamed = model_validator" in ln)
        later = [
            i
            for i, ln in enumerate(lines)
            if re.search(r'model_validator\(mode="before"\)', ln)
            and i > mount
            and "_reject_renamed" not in ln
        ]
        # A validator DEFINED after the mount runs BEFORE it. That is required.
        # One defined before the mount would run after the fold -- the bug.
        earlier = [
            i
            for i, ln in enumerate(lines)
            if re.search(r'model_validator\(mode="before"\)', ln)
            and i < mount
            and "_reject_renamed" not in ln
            # Nested helper classes declared above the block have their own
            # namespace; only same-class validators can race the fold.
            and _enclosing_class(lines, i) == _enclosing_class(lines, mount)
        ]
        assert not earlier, (
            f'{module}.py: `mode="before"` validator(s) at line(s) '
            f"{[i + 1 for i in earlier]} are mounted ABOVE the fold, so they run "
            "AFTER it. Any legacy key they write lands at the old level and is "
            "silently dropped. Move the fold mount above them."
        )
        # `later` is informational per-module: a block may legitimately have no
        # other before-validator (optimization.py does not). The anti-vacuity
        # guard is suite-level, below -- asserting it here would fail a class
        # for being simple.
        assert isinstance(later, list)

    def test_the_check_is_not_vacuous_across_the_suite(self) -> None:
        """At least one folded block must actually have a validator to order.

        Without this, both blocks could quietly lose their other
        before-validators and the parametrised test above would keep passing
        while guarding nothing.
        """
        import inspect
        import re

        from spectramr.config.schemas import data as data_mod

        lines = inspect.getsource(data_mod).splitlines()
        mount = next(i for i, ln in enumerate(lines) if "_fold_renamed = model_validator" in ln)
        later_same_class = [
            i
            for i, ln in enumerate(lines)
            if re.search(r'model_validator\(mode="before"\)', ln)
            and i > mount
            and _enclosing_class(lines, i) == _enclosing_class(lines, mount)
        ]
        assert later_same_class, (
            "DataConfigSchema no longer has a before-validator after the fold "
            "mount, so the ordering rule is untested. It had "
            "`migrate_legacy_sizes`, which injects the M4Raw preset."
        )


def _enclosing_class(lines: list[str], idx: int) -> str | None:
    """Name of the ``class`` whose body contains ``lines[idx]``."""
    import re

    for i in range(idx, -1, -1):
        m = re.match(r"class (\w+)", lines[i])
        if m:
            return m.group(1)
    return None


class TestM4RawPresetSurvivesTheFold:
    """The concrete instance of the ordering rule above.

    ``migrate_legacy_sizes`` injects four M4Raw defaults using the FLAT key
    names. Phase 9a moved one of them (``coil_processing_mode`` ->
    ``coils.processing_mode``), so this pairing is live today: get the ordering
    wrong and every M4Raw arm silently reverts to ``processing_mode='none'``
    while still reporting a clean load.
    """

    def test_injected_preset_reaches_the_sub_block(self) -> None:
        from spectramr.config.schemas.data import DataConfigSchema

        cfg = DataConfigSchema(dataset_type="m4raw", data_root="/tmp")
        default = type(cfg.coils).model_fields["processing_mode"].default
        assert default != "svd", (
            "the schema default now equals the preset, so this test can no "
            "longer tell injection from default -- pick another witness"
        )
        assert cfg.coils.processing_mode == "svd"

    def test_an_explicit_value_still_wins_over_the_preset(self) -> None:
        from spectramr.config.schemas.data import DataConfigSchema

        cfg = DataConfigSchema(dataset_type="m4raw", data_root="/tmp", coil_processing_mode="rss")
        assert cfg.coils.processing_mode == "rss"

    # The case above passes the LEGACY spelling, so it could never see the
    # collision: the preset wrote the legacy key too, and two legacy writes
    # merge. A MIGRATED arm -- the output of `migrate_config_keys.py` -- passes
    # the canonical one, and got `svd` written beside it and a "two spellings
    # disagree" ValidationError. Three of the four preset knobs are folded, so
    # all three are pinned here.
    @pytest.mark.parametrize(
        ("kwargs", "read", "expected"),
        [
            (
                {"coils": {"processing_mode": "rss"}},
                ("coils", "processing_mode"),
                "rss",
            ),
            (
                {"processing": {"enable_kspace_normalization": False}},
                ("processing", "enable_kspace_normalization"),
                False,
            ),
            (
                {"processing": {"kspace_percentile": 0.5}},
                ("processing", "kspace_percentile"),
                0.5,
            ),
        ],
    )
    def test_a_canonical_declaration_wins_without_colliding(
        self, kwargs: dict, read: tuple[str, str], expected: object
    ) -> None:
        from spectramr.config.schemas.data import DataConfigSchema

        cfg = DataConfigSchema(dataset_type="m4raw", data_root="/tmp", **kwargs)
        block, leaf = read
        assert getattr(getattr(cfg, block), leaf) == expected

    def test_the_preset_still_fires_when_nothing_is_declared(self) -> None:
        """Anti-vacuity: the fix must not have simply stopped injecting."""
        from spectramr.config.schemas.data import DataConfigSchema

        cfg = DataConfigSchema(dataset_type="m4raw", data_root="/tmp")
        assert cfg.coils.processing_mode == "svd"
        assert cfg.processing.enable_kspace_normalization is True
        assert cfg.processing.kspace_percentile == 0.99
        assert cfg.use_repetitions is True

    def test_a_non_m4raw_arm_is_untouched(self) -> None:
        from spectramr.config.schemas.data import DataConfigSchema

        cfg = DataConfigSchema(dataset_type="nifti", data_root="/tmp")
        assert cfg.coils.processing_mode != "svd"
        assert cfg.processing.enable_kspace_normalization is False


class TestDefaultKnob:
    """``default_knob`` is the fifth consumer of the rename table.

    A schema that injects its own presets needs to ask the table where a knob
    LIVES, not where it used to live -- otherwise it writes the legacy spelling
    beside a migrated arm's canonical one and the fold validator, correctly,
    rejects the pair.
    """

    def test_it_writes_the_canonical_path_not_the_legacy_leaf(self) -> None:
        from spectramr.config.schemas.renames import default_knob

        raw: dict = {}
        default_knob(raw, "data", "coil_processing_mode", "svd")
        assert raw == {"coils": {"processing_mode": "svd"}}
        assert "coil_processing_mode" not in raw

    def test_a_legacy_declaration_suppresses_the_write(self) -> None:
        from spectramr.config.schemas.renames import default_knob

        raw: dict = {"coil_processing_mode": "rss"}
        default_knob(raw, "data", "coil_processing_mode", "svd")
        assert raw == {"coil_processing_mode": "rss"}, (
            "writing the canonical path here would present the fold validator "
            "with two spellings that disagree"
        )

    def test_a_canonical_declaration_suppresses_the_write(self) -> None:
        from spectramr.config.schemas.renames import default_knob

        raw: dict = {"coils": {"processing_mode": "rss"}}
        default_knob(raw, "data", "coil_processing_mode", "svd")
        assert raw == {"coils": {"processing_mode": "rss"}}

    def test_it_merges_into_a_partially_authored_sub_block(self) -> None:
        from spectramr.config.schemas.renames import default_knob

        raw: dict = {"coils": {"num_virtual_coils": 8}}
        default_knob(raw, "data", "coil_processing_mode", "svd")
        assert raw == {"coils": {"num_virtual_coils": 8, "processing_mode": "svd"}}

    def test_an_unfolded_leaf_stays_flat(self) -> None:
        """No record -> the knob has no canonical home; write it where it is."""
        from spectramr.config.schemas.renames import default_knob

        raw: dict = {}
        default_knob(raw, "data", "use_repetitions", True)
        assert raw == {"use_repetitions": True}

    def test_a_raise_posture_record_still_writes_to_its_canonical_path(self) -> None:
        """Replaces `test_a_raise_posture_record_is_not_a_fold_destination` (2026-08-18).

        That pin asserted the OPPOSITE -- that a promoted record's canonical path
        is not a write target -- and contradicted itself doing so: its message
        read "the rejection validator owns that spelling now" while asserting the
        seeder writes that very spelling.

        A promotion retires a SPELLING, not a destination. `data.slice_2d` raising
        does not move the live field, which is still `data.sampling.enable_slice_2d`,
        and this class exists to ask the table where a knob LIVES rather than where
        it used to. The old behaviour made a preset seeder emit an UNLOADABLE arm:
        it wrote the retired flat key, on which `reject_renamed_keys` then raised,
        naming a key the user never authored. `_canonical_tail` records the two
        further contracts it inverted, and the promotion that exposed it.
        """
        from spectramr.config.schemas.renames import (
            RenameRecord,
            _canonical_tail,
            default_knob,
        )

        table = {
            "demo.retired": RenameRecord(
                legacy="demo.retired",
                canonical="demo.sub.arrived",
                since="2026-01-01",
                reason="Gone.",
                posture="raise",
            )
        }
        assert _canonical_tail("demo", "retired", table) == ["sub", "arrived"]
        raw: dict = {}
        default_knob(raw, "demo", "retired", 1, table)
        assert raw == {"sub": {"arrived": 1}}, (
            "a promoted record's canonical field is still the live one -- seeding "
            "the retired spelling instead is what made presets unloadable"
        )

    def test_seeding_canonical_is_not_a_back_door_around_the_rejection(self) -> None:
        """Writing canonical must not rescue an arm that authored the retired key.

        This is the half the replaced pin was reaching for, and it holds without
        posture: `declared_knob` consults the legacy spelling FIRST, so an arm
        that authored it short-circuits and the seeder returns having written
        nothing -- leaving the key exactly where `reject_renamed_keys` will find
        it. The rejection is the raw block's business, never this function's.
        """
        from spectramr.config.schemas.renames import (
            RenameRecord,
            declared_knob,
            default_knob,
        )

        table = {
            "demo.retired": RenameRecord(
                legacy="demo.retired",
                canonical="demo.sub.arrived",
                since="2026-01-01",
                reason="Gone.",
                posture="raise",
            )
        }
        raw: dict = {"retired": 7}
        assert declared_knob(raw, "demo", "retired", table) == 7
        default_knob(raw, "demo", "retired", 1, table)
        assert raw == {"retired": 7}, (
            "the retired key must survive untouched, and no canonical sub-block "
            "may appear beside it -- either would change what the validator sees"
        )

    def test_a_cross_block_move_is_not_this_blocks_to_write(self) -> None:
        """`training.seed` -> `run.seed` lands in a DIFFERENT top-level block.

        The only guard that survived dropping the posture check, and the reason
        it is a same-block test rather than a bare `rec is None` one: a tail
        computed against `run:` is meaningless under `training:`, so the flat
        leaf is kept and the rejection validator speaks instead.

        Real record, not a fixture -- a synthetic table would not notice if the
        production one grew a second cross-block move with a live caller.
        """
        from spectramr.config.schemas.renames import RENAMES, _canonical_tail

        assert RENAMES["training.seed"].canonical == "run.seed"
        assert _canonical_tail("training", "seed") == ["seed"]

    def test_a_non_dict_sub_block_is_never_overwritten(self) -> None:
        from spectramr.config.schemas.renames import default_knob

        raw: dict = {"coils": "not-a-mapping"}
        default_knob(raw, "data", "coil_processing_mode", "svd")
        assert raw == {"coils": "not-a-mapping"}

    def test_undeclared_is_distinct_from_a_declared_none(self) -> None:
        """``svd_calibration_lines: null`` means "full FoV" — a value, not absence."""
        from spectramr.config.schemas.renames import UNDECLARED, declared_knob

        assert declared_knob({}, "data", "svd_calibration_lines") is UNDECLARED
        raw = {"coils": {"svd_calibration_lines": None}}
        assert declared_knob(raw, "data", "svd_calibration_lines") is None

    def test_declared_knob_sees_either_spelling(self) -> None:
        from spectramr.config.schemas.renames import declared_knob

        assert (
            declared_knob({"coil_processing_mode": "rss"}, "data", "coil_processing_mode") == "rss"
        )
        assert (
            declared_knob({"coils": {"processing_mode": "rss"}}, "data", "coil_processing_mode")
            == "rss"
        )


class TestOverrideKnob:
    """A DERIVATION has the opposite contract to a preset.

    ``_sync_coil_processing_to_legacy`` documents that "an explicit new block
    wins over a conflicting legacy mode". Using :func:`default_knob` there made
    the unified physics block INERT for every arm derived from the v6.1
    reference template, which authors ``data.coils.processing_mode: 'none'``.
    """

    def test_it_displaces_a_canonical_declaration(self) -> None:
        from spectramr.config.schemas.renames import override_knob

        raw: dict = {"coils": {"processing_mode": "rss"}}
        previous = override_knob(raw, "data", "coil_processing_mode", "svd")
        assert previous == "rss"
        assert raw == {"coils": {"processing_mode": "svd"}}

    def test_it_clears_the_legacy_spelling_it_displaces(self) -> None:
        """The load-bearing half: leaving it beside the canonical write is
        exactly the "two spellings disagree" failure."""
        from spectramr.config.schemas.renames import override_knob

        raw: dict = {"coil_processing_mode": "rss"}
        assert override_knob(raw, "data", "coil_processing_mode", "svd") == "rss"
        assert raw == {"coils": {"processing_mode": "svd"}}
        assert "coil_processing_mode" not in raw

    def test_it_reports_undeclared_when_nothing_was_there(self) -> None:
        from spectramr.config.schemas.renames import UNDECLARED, override_knob

        raw: dict = {}
        assert override_knob(raw, "data", "coil_processing_mode", "svd") is UNDECLARED
        assert raw == {"coils": {"processing_mode": "svd"}}


class TestCoilProcessingBridgeIsNotInert:
    """End-to-end: the unified physics block must reach ``data.coils``.

    Pinned through ``TrainingSettings`` rather than the helper, because the
    regression was a composition — the bridge wrote one spelling, the fold
    validator reconciled another, and the template authored a third value.
    """

    @staticmethod
    def _load(tmp_path, mutate):
        import yaml

        from spectramr.config.settings import TrainingSettings

        template = pathlib.Path("src/spectramr/config/schemas/templates/v1.0_reference.yaml")
        base = yaml.safe_load(template.read_text())
        mutate(base)
        target = tmp_path / "arm.yaml"
        target.write_text(yaml.safe_dump(base))
        return TrainingSettings.from_yaml(str(target))

    @staticmethod
    def _svd(doc: dict) -> None:
        doc.setdefault("physics", {}).setdefault("coil_processing", {})["compression"] = {
            "method": "svd",
            "num_virtual_coils": 8,
            "calibration_lines": 24,
        }

    def test_it_wins_over_the_templates_own_coils_block(self, tmp_path) -> None:
        cfg = self._load(tmp_path, self._svd)
        assert cfg.data.coils.processing_mode == "svd"
        assert cfg.data.coils.num_virtual_coils == 8
        assert cfg.data.coils.svd_calibration_lines == 24

    @pytest.mark.parametrize(
        "conflicting",
        [
            {"coils": {"processing_mode": "rss"}},
            {"coil_processing_mode": "rss"},
        ],
        ids=["canonical", "legacy"],
    )
    def test_it_wins_over_either_conflicting_spelling(self, tmp_path, conflicting: dict) -> None:
        def mutate(doc: dict) -> None:
            self._svd(doc)
            doc["data"].pop("coils", None)
            doc["data"].update(conflicting)

        assert self._load(tmp_path, mutate).data.coils.processing_mode == "svd"

    def test_it_is_a_no_op_without_the_physics_block(self, tmp_path) -> None:
        """Anti-vacuity: the bridge must not just force svd unconditionally."""
        cfg = self._load(tmp_path, lambda doc: None)
        assert cfg.data.coils.processing_mode == "none"

    def test_a_legacy_arm_still_loads_unchanged(self, tmp_path) -> None:
        def mutate(doc: dict) -> None:
            doc["data"].pop("coils", None)
            doc["data"].update(coil_processing_mode="svd", num_virtual_coils=6)

        cfg = self._load(tmp_path, mutate)
        assert cfg.data.coils.processing_mode == "svd"
        assert cfg.data.coils.num_virtual_coils == 6


class TestCanonicalOverridePath:
    """``--override`` is the fourth consumer of the table.

    Without it a staged rename makes YAML and the CLI disagree about what the
    config language is: the old spelling loads from a file and raises from the
    command line. See ``tests/unit/config/test_overrides.py`` for the
    end-to-end regression.
    """

    def test_a_fold_record_translates(self) -> None:
        from spectramr.config.schemas.renames import canonical_override_path

        table = {
            "blk.old": RenameRecord(
                legacy="blk.old",
                canonical="blk.sub.new",
                since="2026-08-01",
                reason="test",
                posture="fold",
            )
        }
        assert canonical_override_path("blk.old", table) == "blk.sub.new"

    def test_a_raise_record_falls_through_untranslated(self) -> None:
        from spectramr.config.schemas.renames import canonical_override_path

        assert canonical_override_path("demo.old_name", _FIXTURE) == "demo.old_name"

    def test_an_unknown_path_is_returned_unchanged(self) -> None:
        from spectramr.config.schemas.renames import canonical_override_path

        assert canonical_override_path("blk.never_renamed") == "blk.never_renamed"

    def test_a_negating_fold_refuses_rather_than_mistranslate(self) -> None:
        """Rewriting the key while leaving the literal alone inverts intent."""
        from spectramr.config.schemas.renames import canonical_override_path

        table = {
            "blk.disable_x": RenameRecord(
                legacy="blk.disable_x",
                canonical="blk.enable_x",
                since="2026-08-01",
                reason="test",
                posture="fold",
                value_transform="negate",
            )
        }
        with pytest.raises(ValueError, match="cannot be used with --override"):
            canonical_override_path("blk.disable_x", table)


# --------------------------------------------------------------------------- #
# Source-side anti-rot: no string-keyed read of a folded legacy leaf name
# --------------------------------------------------------------------------- #
# Receivers that carry one of these names LEGITIMATELY -- they are not config
# schema objects, so the identical spelling means something else. Keyed on
# (path, receiver-source, field) rather than line number so the entries survive
# edits above them. Each entry must still MATCH something (see the stale check).
_NOT_CONFIG_RECEIVERS: frozenset[tuple[str, str, str]] = frozenset(
    {
        # `simulator.acceleration` is the DigitalTwinSimulator's own float
        # acceleration FACTOR (`self.acceleration = acceleration`), not the
        # renamed config block. 28 of the 75 repo-wide `.acceleration` reads are
        # of that kind -- on transforms, simulators and argparse namespaces.
        (
            "src/spectramr/infrastructure/training/strategies/virtual_fiducial_strategy.py",
            "simulator",
            "acceleration",
        ),
        # --- phase 10 leaf names that collide with an unrelated object ------
        # `config.metrics` is the TOP-LEVEL `metrics:` block (MetricsConfigSchema),
        # not the retired `validation.metrics`. Same name, different block.
        ("src/spectramr/bootstrap.py", "config", "metrics"),
        ("src/spectramr/core/metrics/computer.py", "config", "metrics"),
        # (disentangled_strategy.py, "self.config", "metrics") was here until
        # 30fa0ea0a converted that site to direct `self.config.metrics.output_dir`
        # access. The entry then matched nothing, which is exactly what the
        # stale-entry check below exists to catch.
        (
            "src/spectramr/infrastructure/training/strategies/mixins/metrics_mixin.py",
            "config",
            "metrics",
        ),
        (
            "src/spectramr/infrastructure/validation/config_health_checker.py",
            "config",
            "metrics",
        ),
        ("src/spectramr/pipelines/training_loop.py", "config", "metrics"),
        # `reporting.metrics`, a third block with the same leaf.
        ("src/spectramr/cli/app.py", "rep", "metrics"),
        ("src/spectramr/infrastructure/reporting/run_hook.py", "reporting", "metrics"),
        # top-level `metrics.best_metric_name`, the checkpoint selector.
        (
            "src/spectramr/infrastructure/validation/witness/checks/validation_metric_checks.py",
            "settings",
            "metrics",
        ),
        # top-level `metrics.domain`, not `validation.scoring.domain`.
        (
            "src/spectramr/infrastructure/training/strategies/mixins/metrics_mixin.py",
            "metrics_s",
            "domain",
        ),
        (
            "src/spectramr/infrastructure/validation/config_health_checker.py",
            "metrics",
            "domain",
        ),
        # ExperimentMetadataSchema, not validation/logging.
        ("src/spectramr/core/metrics/computer.py", "config", "primary_metric"),
        (
            "src/spectramr/infrastructure/validation/config_health_checker.py",
            "meta",
            "primary_metric",
        ),
        (
            "src/spectramr/infrastructure/validation/config_health_checker.py",
            "meta",
            "tags",
        ),
        ("src/spectramr/infrastructure/validation/inference.py", "metadata", "tags"),
        # a stdlib `logging.Handler`, not `logging.sinks.level`.
        ("src/spectramr/infrastructure/services/logging_service.py", "handler", "level"),
        # physics coil-processing output domain, not losses.policy.output_domain.
        (
            "src/spectramr/infrastructure/validation/spec_card.py",
            "getattr(coil_proc, 'output', None)",
            "domain",
        ),
        # --- `num_timesteps` is BOTH a retired config key and a live runtime
        # attribute, so the same spelling has two owners. The config key folds
        # (`training.diffusion.num_timesteps` -> `.timesteps`) and no schema
        # carries it any more; the runtime attribute is set by
        # `DiffusionMixin.__init__`, `BlurringDiffusion.__init__`,
        # `DiffusionScheduler`, the cold samplers and the k-space mask builders,
        # and is read off `self`/`model` -- never off a config object.
        #
        # `DiffusionTrainingStrategy(BaseTrainingStrategy, DiffusionStrategyMixin)`
        # inherits the attribute, so these reads are live, not vestigial.
        ("src/spectramr/infrastructure/training/strategies/diffusion.py", "self", "num_timesteps"),
        # an nn.Module (e.g. BlurringDiffusion, which sets it in __init__).
        ("src/spectramr/infrastructure/validation/phase3_probes.py", "model", "num_timesteps"),
        # argparse Namespaces
        ("src/spectramr/cli/app.py", "args", "batch_size"),
        ("src/spectramr/main.py", "args", "batch_size"),
        # physics.coil_processing.compression, a different block
        (
            "src/spectramr/infrastructure/validation/config_health_checker.py",
            "compression",
            "num_virtual_coils",
        ),
        # the strategy's own uncertainty block, not optimization.optimizer.eps
        (
            "src/spectramr/infrastructure/training/strategies/geomamba_ulf_strategy.py",
            "unc_cfg",
            "eps",
        ),
        # a diffusion model's registered `betas` buffer
        ("src/spectramr/models/diffusion/cold_diffusion.py", "self", "betas"),
        # AMPPolicy / StepExecutor attributes that happen to share a config name
        (
            "src/spectramr/pipelines/training_loop.py",
            "current",
            "enable_gradient_clipping",
        ),
        # The receiver was `getattr(strategy, 'step_executor', None)` until the
        # cadence resolver hoisted it into a local `executor`, because that site
        # now reads TWO attributes off it -- `requested_gradient_accumulation_steps`
        # first (the configured value, which is what cadence wants) and this
        # negotiated one only as a fallback. Only the negotiated name is a folded
        # leaf, so only it reaches this scan.
        (
            "src/spectramr/pipelines/training_loop.py",
            "executor",
            "gradient_accumulation_steps",
        ),
        # `output_domain` also exists on `losses` and on ModelCapabilities, and
        # those did NOT move -- only `data.output_domain` folded (to
        # `data.domain.output`). The gate keys on the leaf name alone, so it
        # cannot tell the three apart; the receiver can. If `losses.output_domain`
        # ever moves too, its entry here must be deleted, and the stale-entry
        # check below will not let it linger once the site changes.
        (
            "src/spectramr/infrastructure/training/utils/domain_inference.py",
            "caps",
            "output_domain",
        ),
        # `transforms` is TorchIO's own attribute -- a `Compose` holds its
        # children there -- and `data_range` is also a field on the metrics
        # block. Neither moved; only `data.transforms` / `data.data_range` did.
        # This is the cost of a genuinely generic leaf name, and the receiver is
        # what separates them.
        ("src/spectramr/data/transforms/signature.py", "t", "transforms"),
        ("src/spectramr/data/transforms/signature.py", "transforms", "transforms"),
        (
            "src/spectramr/infrastructure/training/strategies/mixins/metrics_mixin.py",
            "metrics_s",
            "data_range",
        ),
        # Receivers that are not config blocks: argparse namespaces, model and loss
        # capabilities, the twin spec, and `metadata.tags`, which already is the
        # canonical path. The gate keys on the leaf name alone.
        ("src/spectramr/cli/app.py", "args", "acceleration"),
        ("src/spectramr/cli/app.py", "args", "metrics"),
        ("src/spectramr/cli/app.py", "args", "contrasts"),
        ("src/spectramr/data/transforms/normalization.py", "transform", "transforms"),
        (
            "src/spectramr/infrastructure/inference/cold_diffusion_inference_strategy.py",
            "unwrap_model(self.model)",
            "num_timesteps",
        ),
        (
            "src/spectramr/infrastructure/inference/predict_data_consistency.py",
            "caps",
            "output_domain",
        ),
        ("src/spectramr/infrastructure/training/builders/loss_builder.py", "caps", "domain"),
        ("src/spectramr/infrastructure/training/snapshot_provenance.py", "transform", "transforms"),
        (
            "src/spectramr/infrastructure/validation/witness/checks/ood_range_checks.py",
            "twin",
            "acceleration",
        ),
        (
            "src/spectramr/infrastructure/validation/witness/checks/strategy_override_checks.py",
            "metadata",
            "tags",
        ),
        # PRE-EXISTING flat-root reads that never worked on the nested schema
        # (pitfall #1, predates the renames). Tracked separately; listed here so
        # this gate reports only NEW breakage.
        ("src/spectramr/models/model_cards.py", "config", "batch_size"),
        ("src/spectramr/models/model_cards.py", "config", "learning_rate"),
    }
)


def _string_keyed_reads_of_folded_names() -> list[tuple[str, int, str, str, str]]:
    """Every ``getattr``/``hasattr`` in ``src/`` keyed on a folded legacy leaf."""
    import ast
    import pathlib as _pl

    from spectramr.config.schemas.renames import RENAMES

    leaves: dict[str, set[str]] = {}
    for legacy, rec in RENAMES.items():
        if rec.posture == "fold":
            leaves.setdefault(legacy.split(".")[-1], set()).add(rec.canonical)
    subblocks = {c.split(".")[1] for cs in leaves.values() for c in cs if c.count(".") >= 2}

    found: list[tuple[str, int, str, str, str]] = []

    class _V(ast.NodeVisitor):
        def __init__(self, path: str) -> None:
            self.path = path

        def visit_Call(self, node: ast.Call) -> None:
            fn = getattr(node.func, "id", None)
            if fn in ("getattr", "hasattr") and len(node.args) >= 2:
                key = node.args[1]
                if isinstance(key, ast.Constant) and key.value in leaves:
                    recv = ast.unparse(node.args[0])
                    # A receiver already naming the new sub-block is correct.
                    if not any(sb in recv for sb in subblocks):
                        found.append((self.path, node.lineno, fn, key.value, recv))
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            # `SomeSchema.model_fields["legacy_leaf"]` -- a fourth string-keyed
            # form, and the one that bites hardest: it raises KeyError rather
            # than returning a default, but only on the path that reads it. This
            # is how three mrixfields default lookups in dataset_instantiator
            # survived phase 9a as live crashes.
            value = node.value
            if isinstance(value, ast.Attribute) and value.attr == "model_fields":
                key = node.slice
                if isinstance(key, ast.Constant) and key.value in leaves:
                    recv = ast.unparse(value)
                    if not any(sb in recv.lower() for sb in subblocks):
                        found.append((self.path, node.lineno, "model_fields[]", key.value, recv))
            self.generic_visit(node)

    root = _pl.Path(__file__).resolve().parents[4]
    for f in sorted((root / "src").rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        _V(rel).visit(ast.parse(f.read_text()))
    return found


class TestNoStringKeyedReadsOfFoldedNames:
    """A rename that moves a field cannot be found by attribute-grep.

    ``getattr(cfg, "batch_size", 4)`` and ``hasattr(cfg, "batch_size")`` hold the
    field name as a STRING, so a sweep over ``cfg.batch_size`` misses them
    entirely -- and both fail SILENTLY once the attribute moves: ``getattr``
    returns its default and ``hasattr`` returns False. Phase 9a shipped with the
    guards left on the legacy names while the value reads had been moved:

        data_config.loader.batch_size if hasattr(data_config, "batch_size") else 1

    which pinned the training data builder to batch_size=1 for every arm, turned
    off gradient accumulation, gradient checkpointing and memory monitoring,
    stamped a null batch size into provenance, and computed the federated-DP
    sample rate from a fallback of 4.

    Phase 9b moves far more generic names (``patch_size``, ``transforms``,
    ``trajectory``), so this must hold BEFORE that lands rather than after.

    Covers three forms: ``getattr``, ``hasattr``, and
    ``<Schema>.model_fields["legacy_leaf"]``. The last was added after 9b
    found three of them in ``dataset_instantiator`` reading mrixfields
    defaults off the schema by their pre-9a flat names -- a KeyError on the
    mrixfields path that the other two forms could not have surfaced.
    """

    def test_no_unallowlisted_string_keyed_reads(self) -> None:
        offenders = [
            (p, ln, fn, field, recv)
            for p, ln, fn, field, recv in _string_keyed_reads_of_folded_names()
            if (p, recv, field) not in _NOT_CONFIG_RECEIVERS
        ]
        assert not offenders, "string-keyed read(s) of a folded legacy field name:\n" + "\n".join(
            f"  {p}:{ln}  {fn}({recv}, {field!r}) -- the attribute moved; "
            "read the canonical path directly"
            for p, ln, fn, field, recv in offenders
        )

    def test_the_allowlist_has_no_stale_entries(self) -> None:
        """An entry that matches nothing is a comment pretending to be a gate."""
        live = {
            (p, recv, field) for p, _ln, _fn, field, recv in _string_keyed_reads_of_folded_names()
        }
        stale = sorted(_NOT_CONFIG_RECEIVERS - live)
        assert not stale, (
            "allowlist entries no longer match any source site -- delete them:\n"
            + "\n".join(f"  {e}" for e in stale)
        )

    def test_the_scan_finds_something(self) -> None:
        """Anti-vacuity: a scan that returns nothing would pass silently."""
        assert _string_keyed_reads_of_folded_names(), (
            "the AST scan found no string-keyed reads at all -- it has stopped "
            "working (check RENAMES has fold records and src/ is reachable)"
        )


class TestPhase10FoldTableIsTotal:
    """Every scalar ``validation:`` used to declare must be accounted for.

    ``validation`` is ``extra="forbid"``, so unlike ``data`` a gap here is a
    hard ValidationError rather than silence -- loud, but on 633 arms at once.
    """

    #: The pre-phase-10 scalar list, read off the live class before the split.
    SCALARS_BEFORE_PHASE_10: frozenset[str] = frozenset(
        {
            "compute_image_metrics",
            "domain",
            "empty_cache_before_validation",
            "enable_validation_augmentation",
            "enable_visualization",
            "enabled",
            "eval_interval",
            "eval_on_epoch",
            "frequency_epochs",
            "frequency_steps",
            "held_out_severity_eval",
            "input_dependence_tol",
            "metrics",
            "multistep_cold_sampling",
            "num_samples",
            "num_validation_batches",
            "num_visualizations",
            "output_transform",
            "primary_metric",
            "sampler_steps",
            "shuffle_validation",
            "split",
            "use_training_loss",
            "val_batch_size",
            "val_chunk_size",
            "validation_batch_size",
            "validation_dir",
            "validation_metric",
            "visualization_dir",
            "visualization_interval",
        }
    )

    def test_the_pinned_list_is_the_size_it_was(self) -> None:
        assert len(self.SCALARS_BEFORE_PHASE_10) == 30

    @pytest.mark.parametrize("name", sorted(SCALARS_BEFORE_PHASE_10))
    def test_every_old_scalar_still_has_somewhere_to_go(self, name: str) -> None:
        from spectramr.config.schemas.removals import REMOVALS
        from spectramr.config.schemas.validation import ValidationConfigSchema

        still_declared = name in ValidationConfigSchema.model_fields
        has_record = f"validation.{name}" in RENAMES or f"validation.{name}" in REMOVALS
        assert still_declared or has_record, (
            f"`validation.{name}` is neither a field nor a rename/removal record, so "
            "every arm that sets it now fails to load (the block is "
            "extra='forbid')."
        )

    def test_the_two_batch_size_spellings_share_one_destination(self) -> None:
        """Both spellings must land on one field, or the duplicate survives."""
        short = RENAMES["validation.val_batch_size"].canonical
        long = RENAMES["validation.validation_batch_size"].canonical
        assert short == long == "validation.loader.batch_size"

    def test_declaring_both_batch_size_spellings_now_raises(self) -> None:
        """Replaces `test_the_short_batch_size_spelling_wins` (2026-08-18).

        `validation.val_batch_size` drained to 0 and was promoted to `raise`;
        its sibling `validation.validation_batch_size` did not (26 live
        declarations). `_resolve_batch_size_duplicate` still runs first and
        still picks the short form as the winner -- but the winner now raises,
        so the pair no longer resolves to a value.

        Nothing regresses: the rescue existed for arms declaring BOTH, and an
        arm declaring both is now told, accurately, which spelling is retired
        and what replaced it. The long form alone is unaffected -- pinned by
        `test_the_long_batch_size_spelling_still_works_alone` just below, which
        is what the 26 arms actually rely on.
        """
        from spectramr.config.schemas.validation import ValidationConfigSchema

        with pytest.raises(ValidationError, match=r"validation\.val_batch_size"):
            ValidationConfigSchema(val_batch_size=2, validation_batch_size=1)

        # The error must name the destination, or a migrating author is stuck.
        with pytest.raises(ValidationError) as exc:
            ValidationConfigSchema(val_batch_size=2, validation_batch_size=1)
        assert "validation.loader.batch_size" in str(exc.value)

    def test_the_long_batch_size_spelling_still_works_alone(self) -> None:
        from spectramr.config.schemas.validation import ValidationConfigSchema

        assert ValidationConfigSchema(validation_batch_size=3).loader.batch_size == 3


class TestNoLegacyLeafNamesADestinationBlock:
    """A retired leaf must not share its name with a sub-block records fold into.

    The fold matches on the key ALONE, so had ``validation.metrics: [psnr]``
    folded into ``validation.metrics.compute`` the key ``metrics`` would mean two
    things at once, and neither reading is recoverable from the key:

    * a legacy-only arm breaks on **declaration order** -- a sibling written
      above ``metrics`` descends into a *list* and the arm is refused (that was
      58/58 of the kspace_filling cohort);
    * a MIGRATED arm writing ``metrics: {compute: [...]}`` has the whole block
      folded into itself, so the migration's own output stops loading.

    The second is why sorting the fold order cannot fix this and why the phase-10
    block is named ``scoring``. The rule is cheap and applies to every phase after.
    """

    def test_no_fold_record_collides(self) -> None:
        collisions = sorted(
            f"{rec.legacy} -> {rec.canonical}"
            for rec in RENAMES.values()
            if rec.posture == "fold"
            and rec.canonical.count(".") >= 2
            and any(
                other.posture == "fold"
                and other.block == rec.block
                and other.legacy_key == rec.canonical.split(".")[1]
                for other in RENAMES.values()
            )
        )
        assert not collisions, (
            "these records fold into a sub-block whose name is ALSO a retired "
            "leaf, so the key is ambiguous and a migrated arm will not load:\n  "
            + "\n  ".join(collisions)
            + "\nRename the destination block."
        )

    def test_the_check_can_actually_fire(self) -> None:
        """Anti-vacuity: the predicate must detect the clash it forbids."""
        table = {
            "blk.metrics": RenameRecord(
                legacy="blk.metrics",
                canonical="blk.metrics.compute",
                since="x",
                reason="y",
                posture="fold",
            ),
            "blk.primary_metric": RenameRecord(
                legacy="blk.primary_metric",
                canonical="blk.metrics.primary",
                since="x",
                reason="y",
                posture="fold",
            ),
        }
        clashing = [
            rec
            for rec in table.values()
            if rec.canonical.count(".") >= 2
            and any(o.legacy_key == rec.canonical.split(".")[1] for o in table.values())
        ]
        assert clashing, "the collision predicate no longer detects a known clash"

    def test_a_migrated_arm_round_trips(self) -> None:
        """The shape the collision destroyed: canonical in, canonical out."""
        from spectramr.config.schemas.validation import ValidationConfigSchema

        cfg = ValidationConfigSchema(
            scoring={"compute": ["psnr", "ssim"], "primary": "ssim", "domain": "image"}
        )
        assert cfg.scoring.compute == ["psnr", "ssim"]
        assert cfg.scoring.primary == "ssim"
        assert cfg.scoring.domain == "image"

    def test_declaration_order_does_not_change_the_result(self) -> None:
        """With the collision gone, any rotation of the legacy keys agrees."""
        from spectramr.config.schemas.validation import ValidationConfigSchema

        items = list(
            {
                "metrics": ["psnr", "ssim"],
                "primary_metric": "ssim",
                "domain": "image",
                "output_transform": "fft",
                "compute_image_metrics": False,
            }.items()
        )
        builds = [ValidationConfigSchema(**dict(items[i:] + items[:i])) for i in range(len(items))]
        assert all(b == builds[0] for b in builds)
        assert builds[0].scoring.compute == ["psnr", "ssim"]

    def test_it_still_refuses_a_genuine_disagreement(self) -> None:
        """The fix must not have cost the two-spellings-disagree guard."""
        from spectramr.config.schemas.validation import ValidationConfigSchema

        with pytest.raises(ValidationError, match="disagree"):
            ValidationConfigSchema(eval_interval=1000, schedule={"interval_steps": 500})


class TestPhase10bFoldTableIsTotal:
    """Every scalar ``logging:`` used to declare must be accounted for.

    This matters MORE than its phase-10a twin, for the same reason phase 9's
    mattered more than phase 8's: ``validation:`` is ``extra="forbid"``, so a
    gap in its table raised on 633 arms at once -- loud and unmissable. This
    block is ``extra="ignore"``. A key this table forgets does not raise, it
    **vanishes**: it is dropped from the resolved config and the arm runs on the
    schema default, with no error and no warning. That is the #550 mechanism,
    and this pin is the only thing standing between a typo'd record and ~600
    arms silently reverting.
    """

    #: The pre-phase-10b field list, read off the live class before the split.
    SCALARS_BEFORE_PHASE_10B: frozenset[str] = frozenset(
        {
            "anomaly_check_interval",
            "debug_log_steps",
            "debug_snapshot_interval_steps",
            "debug_snapshot_max_calls",
            "debug_snapshot_save_images",
            "debug_snapshot_save_json",
            "debug_snapshots",
            "enable_experiment_tracking",
            "enable_tensorboard",
            "experiment_name",
            "level",
            "log_activations",
            "log_difference_images",
            "log_dir",
            "log_gradients",
            "log_input_images",
            "log_interval",
            "log_to_console",
            "log_to_file",
            "log_validation_graphs",
            "log_validation_images",
            "log_weights",
            "max_images_per_batch",
            "notes",
            "progress_bar_enabled",
            "progress_bar_no_progress",
            "progress_bar_on_warning",
            "report_cases_subdir",
            "run_name",
            "save_images_per_epoch",
            "save_interval",
            "save_report_cases",
            "save_validation_images",
            "silent",
            "tags",
            "tensorboard_dir",
            "tracking_service",
            "validation_image_interval",
            "wandb_entity",
            "wandb_project",
        }
    )

    def test_the_pinned_list_is_the_size_it_was(self) -> None:
        assert len(self.SCALARS_BEFORE_PHASE_10B) == 40

    @pytest.mark.parametrize("name", sorted(SCALARS_BEFORE_PHASE_10B))
    def test_every_old_scalar_still_has_somewhere_to_go(self, name: str) -> None:
        from spectramr.config.schemas.logging import LoggingConfigSchema

        still_declared = name in LoggingConfigSchema.model_fields
        has_record = f"logging.{name}" in RENAMES
        assert still_declared or has_record, (
            f"`logging.{name}` is neither a field nor a rename record. "
            "`logging` is extra='ignore', so every arm that sets it now has the "
            "value SILENTLY DISCARDED and runs on the default -- no error, no "
            "warning, a different run."
        )

    def test_the_nine_blocks_are_mounted(self) -> None:
        from spectramr.config.schemas.logging import LoggingConfigSchema as L

        blocks = {k for k, v in L.model_fields.items() if hasattr(v.annotation, "model_fields")}
        assert blocks == {
            "identity",
            "sinks",
            "intervals",
            "images",
            "tracking",
            "snapshots",
            "report_cases",
            "distributed",
            "progress",
        }

    def test_only_inert_scalars_stayed_flat(self) -> None:
        """A LIVE knob left flat would be a grouping the phase simply missed.

        `log_gradients` is a deliberate exception: it IS read, but both of its
        group-mates (`log_weights`, `log_activations`) are inert, so there is no
        group left to put it in. `progress_bar_enabled` and `progress_bar_no_progress`
        are the next two (as is `progress_bar_on_warning`): they ARE read now (the progress mode resolver), but 980
        arms declare the flat spelling, so moving them into `logging.progress` needs
        a fold record and its pins. That is a follow-up, not this change.
        """
        import sys

        sys.path.insert(0, "tests/unit/config")
        from test_schema_key_consumption import KNOWN_UNCONSUMED

        from spectramr.config.schemas.logging import LoggingConfigSchema as L

        live_but_flat = {
            "log_gradients",
            "progress_bar_enabled",
            "progress_bar_no_progress",
            "progress_bar_on_warning",
        }
        flat = {k for k, v in L.model_fields.items() if not hasattr(v.annotation, "model_fields")}
        unexplained = {
            k for k in flat if f"logging.{k}" not in KNOWN_UNCONSUMED and k not in live_but_flat
        }
        assert not unexplained, (
            f"{sorted(unexplained)} stayed flat but are not tracked as inert -- "
            "either group them or record why they cannot be grouped"
        )


class TestPhase10dFoldTableIsTotal:
    """Every loose scalar `losses:` used to declare must be accounted for.

    Like `logging:`, this block is `extra="ignore"`, so a forgotten record makes
    the key VANISH rather than raise.
    """

    SCALARS_BEFORE_PHASE_10D: frozenset[str] = frozenset(
        {
            "clip_loss_value",
            "disable_default_losses",
            "lambda_deep_supervision",
            "loss_scaling",
            "normalize_losses",
            "output_domain",
        }
    )

    #: Deleted outright in #676 (`c5a31ec97`) because each had ZERO readers and
    #: ZERO corpus declarations -- there is no successor to point a rename record
    #: at, so "somewhere to go" is genuinely "nowhere". They are enumerated here
    #: rather than dropped from the pinned list above, because deleting them from
    #: the list would shrink the totality claim instead of qualifying it.
    #:
    #: The escape hatch is deliberately narrower than the fix: a size cap, and
    #: disjointness from the records, so it cannot quietly absorb a key that DOES
    #: have a home. Same shape as the `DELIBERATE_TRUE` set the default-hygiene
    #: ratchet uses.
    #:
    #: Cost of the deletion, accepted knowingly: the retired spelling now fails
    #: with a bare `AttributeError` instead of a message naming a replacement,
    #: and `losses:` is extra="ignore", so a YAML still declaring one is silently
    #: dropped. Zero arms do -- that was measured before the deletion and is
    #: re-measured by `test_the_retired_scalars_are_absent_from_the_corpus`.
    RETIRED_WITHOUT_REPLACEMENT: frozenset[str] = frozenset(
        {"clip_loss_value", "loss_scaling", "normalize_losses"}
    )

    def test_the_pinned_list_is_the_size_it_was(self) -> None:
        assert len(self.SCALARS_BEFORE_PHASE_10D) == 6

    def test_the_retirement_hatch_stays_small_and_disjoint(self) -> None:
        """An escape hatch easier to reach than the fix inverts the ratchet."""
        assert len(self.RETIRED_WITHOUT_REPLACEMENT) == 3
        assert self.RETIRED_WITHOUT_REPLACEMENT <= self.SCALARS_BEFORE_PHASE_10D
        overlap = {n for n in self.RETIRED_WITHOUT_REPLACEMENT if f"losses.{n}" in RENAMES}
        assert not overlap, f"{overlap} now HAVE a rename record -- take them out of the hatch"

    @pytest.mark.parametrize("name", sorted(SCALARS_BEFORE_PHASE_10D))
    def test_every_old_scalar_still_has_somewhere_to_go(self, name: str) -> None:
        from spectramr.config.schemas.loss import LossConfigSchema

        if name in self.RETIRED_WITHOUT_REPLACEMENT:
            pytest.skip(f"`losses.{name}` was retired outright (#676), not moved")
        still_declared = name in LossConfigSchema.model_fields
        has_record = f"losses.{name}" in RENAMES
        assert still_declared or has_record, (
            f"`losses.{name}` is neither a field nor a rename record; the block "
            "is extra='ignore', so it now vanishes silently."
        )

    def test_the_retired_scalars_are_absent_from_the_corpus(self) -> None:
        """The deletion's premise, re-measured rather than trusted.

        `losses:` is extra="ignore": a surviving declaration is dropped in
        silence, so "zero arms declare it" is the ONLY thing that makes deleting
        these safe. If an arm reappears, this fails before anyone debugs a knob
        that vanished.
        """
        import pathlib as _pl

        import yaml

        # Same anchor the string-keyed-read scan uses (line ~1237).
        root = _pl.Path(__file__).resolve().parents[4]
        offenders = []
        for arm in tracked_yamls(root / "experiments"):
            try:
                doc = yaml.safe_load(arm.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            losses = doc.get("losses") if isinstance(doc, dict) else None
            if not isinstance(losses, dict):
                continue
            for name in sorted(self.RETIRED_WITHOUT_REPLACEMENT):
                if name in losses:
                    offenders.append(f"{arm.name}: losses.{name}")
        assert not offenders, "\n  ".join(offenders)

    def test_exclude_defaults_is_not_a_negation(self) -> None:
        """The plan and this module's own docstring called
        `disable_default_losses` a negated boolean needing `value_transform=
        'negate'`. It is a `list[str]` of loss NAMES: negating it is meaningless,
        and `enable_default_losses: ['mse']` would read as "enable ONLY mse" --
        the opposite of a filter. The record must carry `identity`."""
        rec = RENAMES["losses.disable_default_losses"]
        assert rec.canonical == "losses.policy.exclude_defaults"
        assert rec.value_transform == "identity"

    def test_the_losses_list_guard_defers_to_the_rename_table(self) -> None:
        """`disable_default_losses` ends in `_losses` but is a filter, not a
        domain list. The `*_losses` guard computes its allow-set from
        `model_fields`, so without an exemption it reads as an undeclared loss
        list and reports "move the entries into the matching domain" -- which is
        not what happened to the key.

        The guard consults the rename table in EVERY posture, not just `fold`.
        Consulting `folded_input_keys` alone was correct only while this record
        was staged: the moment its corpus count reached zero and it was promoted
        to `raise`, it dropped out of that set and the guard resumed shadowing
        the rename message. Now retired, so the assertion is that the author is
        told the replacement rather than sent to fix a loss list."""
        from spectramr.config.schemas.loss import LossConfigSchema

        with pytest.raises(ValidationError) as exc:
            LossConfigSchema(disable_default_losses=["mse"])
        message = str(exc.value)
        assert "losses.policy.exclude_defaults" in message
        assert "is not a declared loss list" not in message

    def test_the_guard_still_rejects_a_genuinely_unknown_list(self) -> None:
        """Both directions: making the guard fold-aware must not blunt it."""
        from spectramr.config.schemas.loss import LossConfigSchema

        with pytest.raises(ValidationError, match="not a declared loss list"):
            LossConfigSchema(custom_losses=[])


class TestRootFoldRenamesAWholeBlock:
    """The ROOT fold is the only one that may cross top-level blocks.

    Every other fold validator is mounted on a single schema class and can only
    move a key within it. This one is mounted on ``TrainingSettings``, so the
    value it moves is an entire block mapping and the destination is the full
    canonical path -- not a path relative to a block. Getting that wrong strips
    the destination name and folds the block into nothing.
    """

    TABLE: ClassVar[dict] = {
        "acceleration": RenameRecord(
            legacy="acceleration",
            canonical="undersampling",
            since="2026-08-02",
            reason="The block name meant two unrelated things.",
            posture="fold",
        )
    }

    def _fold(self, doc):
        return fold_renamed_keys(ROOT, self.TABLE)(None, doc)

    def test_the_whole_mapping_moves_intact(self) -> None:
        out = self._fold({"acceleration": {"center_fraction": 0.08, "base_acceleration": 4.0}})
        assert out == {"undersampling": {"center_fraction": 0.08, "base_acceleration": 4.0}}

    def test_sibling_blocks_are_untouched(self) -> None:
        out = self._fold({"acceleration": {"a": 1}, "seed": 7, "data": {"b": 2}})
        assert out["seed"] == 7 and out["data"] == {"b": 2}
        assert "acceleration" not in out

    def test_the_canonical_spelling_passes_through(self) -> None:
        doc = {"undersampling": {"center_fraction": 0.04}}
        assert self._fold(doc) == doc

    def test_two_agreeing_spellings_drop_the_legacy_one(self) -> None:
        out = self._fold({"acceleration": {"a": 1}, "undersampling": {"a": 1}})
        assert out == {"undersampling": {"a": 1}}

    def test_two_disagreeing_spellings_are_refused(self) -> None:
        """A block-level fold must refuse rather than merge: silently unioning
        two blocks would invent a configuration the author never wrote."""
        with pytest.raises(ValueError, match="disagree"):
            self._fold({"acceleration": {"a": 1}, "undersampling": {"a": 2}})

    def test_the_source_document_is_not_mutated(self) -> None:
        doc = {"acceleration": {"a": 1}}
        self._fold(doc)
        assert doc == {"acceleration": {"a": 1}}


class TestRootFoldRunsBeforeAnySubModel:
    """Load-bearing ordering: the root fold must see the RAW document.

    If a nested `mode="before"` ran first, the retired block would already have
    been handed to its own validator -- and after the rename that block no
    longer exists, so the value would be built from a field that is about to be
    renamed away. Pydantic runs the root validator first; this pins it rather
    than trusting the recollection.
    """

    def test_root_before_precedes_nested_before(self) -> None:
        from pydantic import BaseModel, Field, model_validator

        order: list[str] = []

        class Inner(BaseModel):
            x: int = 0

            @model_validator(mode="before")
            @classmethod
            def _inner(cls, d):
                order.append("inner")
                return d

        class Outer(BaseModel):
            inner: Inner = Field(default_factory=Inner)

            @model_validator(mode="before")
            @classmethod
            def _root(cls, d):
                order.append("root")
                return d

        Outer(inner={"x": 1})
        assert order[0] == "root", (
            "a nested before-validator ran first; a root block rename would "
            f"then be too late to matter (order={order})"
        )

    def test_a_root_rename_reaches_the_nested_model(self) -> None:
        """End to end: the renamed block is CONSTRUCTED, not just moved."""
        from pydantic import BaseModel, Field, model_validator

        table = {
            "legacy_block": RenameRecord(
                legacy="legacy_block",
                canonical="new_block",
                since="x",
                reason="y",
                posture="fold",
            )
        }

        class Inner(BaseModel):
            x: int = 0

        class Outer(BaseModel):
            model_config = {"extra": "forbid"}
            new_block: Inner = Field(default_factory=Inner)

            _fold = model_validator(mode="before")(classmethod(fold_renamed_keys(ROOT, table)))

        assert Outer(legacy_block={"x": 7}).new_block.x == 7
        assert Outer(new_block={"x": 7}).new_block.x == 7

    def test_the_root_validator_is_mounted_on_the_real_settings(self) -> None:
        """Anti-vacuity: the fixture above proves the mechanism, not the wiring."""
        from spectramr.config.settings import TrainingSettings

        assert any("fold" in name for name in vars(TrainingSettings) if name.startswith("_")), (
            "TrainingSettings has no root fold validator mounted"
        )


class TestAccelerationBlockRename:
    """`acceleration:` -> `undersampling:` on the REAL settings class.

    The fixture-table tests above prove the ROOT mechanism; this proves it is
    wired to production, which is the half a fixture cannot show.
    """

    @staticmethod
    def _minimal(**extra):
        return {
            "logging": {"identity": {"experiment": "t"}},
            "data": {"dataset_type": "kspace"},
            "model": {"model_type": "unet"},
            "optimization": {},
            **extra,
        }

    def test_the_legacy_block_still_loads(self) -> None:
        from spectramr.config.settings import TrainingSettings

        cfg = TrainingSettings(**self._minimal(acceleration={"base_acceleration": 6.0}))
        assert cfg.undersampling is not None
        assert cfg.undersampling.base_acceleration == 6.0

    def test_the_legacy_name_is_gone_from_python(self) -> None:
        """One read path: the whole point of a fold is that only the YAML
        surface keeps two spellings."""
        from spectramr.config.settings import TrainingSettings

        assert "acceleration" not in TrainingSettings.model_fields
        cfg = TrainingSettings(**self._minimal(acceleration={"base_acceleration": 6.0}))
        assert not hasattr(cfg, "acceleration")

    def test_the_canonical_block_loads(self) -> None:
        from spectramr.config.settings import TrainingSettings

        cfg = TrainingSettings(**self._minimal(undersampling={"base_acceleration": 3.0}))
        assert cfg.undersampling.base_acceleration == 3.0

    def test_declaring_both_with_different_values_is_refused(self) -> None:
        """A block-level fold must refuse rather than merge -- silently unioning
        two blocks would invent a configuration nobody wrote."""
        from spectramr.config.settings import TrainingSettings

        with pytest.raises(ValidationError, match="disagree"):
            TrainingSettings(
                **self._minimal(
                    acceleration={"base_acceleration": 6.0},
                    undersampling={"base_acceleration": 3.0},
                )
            )

    def test_the_record_is_a_root_fold(self) -> None:
        rec = RENAMES["acceleration"]
        assert rec.block == ROOT and rec.posture == "fold"
        assert rec.canonical == "undersampling"


class TestFoldedInputPaths:
    """The companion to ``folded_input_keys``: WHERE a folded key lands.

    The set alone tells the execution ledger a key was moved rather than
    dropped, which is enough to stop mis-reporting it. It is not enough to keep
    auditing what is INSIDE it -- and ``acceleration:`` is a whole top-level
    block, so with only the set every ``extra="ignore"`` drop beneath it was
    invisible.
    """

    def test_root_fold_resolves_acceleration_to_undersampling(self) -> None:
        from spectramr.config.schemas.renames import ROOT, folded_input_paths

        assert folded_input_paths(ROOT)["acceleration"] == ("undersampling",)

    def test_a_block_fold_is_relative_to_its_block(self) -> None:
        """`data.batch_size` -> `data.loader.batch_size` yields ('loader','batch_size').

        Relative, because the ledger is already positioned on the block's model
        when it descends; an absolute chain would walk `data.data.loader`.
        """
        from spectramr.config.schemas.renames import folded_input_paths

        assert folded_input_paths("data")["batch_size"] == ("loader", "batch_size")

    def test_depth_is_not_capped(self) -> None:
        """A truncated chain does not fail loudly, it walks to the wrong object.

        Same reason ``flat_to_canonical`` refuses a cap: the three
        ``optimization.gradient.clip.*`` records are three levels deep, and a
        two-level cap silently returned the enclosing block.
        """
        from spectramr.config.schemas.renames import folded_input_paths

        assert folded_input_paths("optimization")["gradient_clip_value"] == (
            "gradient",
            "clip",
            "value",
        )

    def test_keys_agree_with_folded_input_keys(self) -> None:
        """The two publications must describe the same set, or the ledger would
        accept a key it cannot then descend into."""
        from spectramr.config.schemas.renames import (
            ROOT,
            folded_input_keys,
            folded_input_paths,
        )

        for block in (ROOT, "data", "logging", "losses", "optimization", "validation"):
            assert set(folded_input_paths(block)) == set(folded_input_keys(block)), block


class TestDrainedRecordsArePromoted:
    """The fold records whose corpus count reached zero now RAISE.

    61 of them as of 2026-08-18 (was 41). Read that as a reading, not a
    constant -- the cases below are generated from `_DRAINED` precisely so the
    count never has to be maintained here.

    This is the promotion rule from the module docstring, applied: a ``fold``
    record's legacy spelling keeps working only until no arm declares it, at
    which point the shim is debt rather than compatibility.

    The tests below pin the BEHAVIOUR (each promoted spelling raises, naming its
    replacement), not the bookkeeping. Asserting only "the table has fewer
    folds" would pass just as happily if `_folds` stopped building records at
    all.
    """

    def test_every_drained_name_is_a_real_legacy_path(self) -> None:
        """The set cannot rot into naming keys that no longer exist."""
        unknown = sorted(n for n in _DRAINED if n not in RENAMES)
        assert not unknown, f"_DRAINED names non-existent record(s): {unknown}"

    def test_every_drained_record_now_raises(self) -> None:
        still_folding = sorted(n for n in _DRAINED if RENAMES[n].posture != "raise")
        assert not still_folding, (
            f"drained but still folding: {still_folding} — the whole point of "
            "draining a record is that its shim goes away"
        )

    @pytest.mark.parametrize("legacy", sorted(_DRAINED))
    def test_a_promoted_spelling_raises_naming_its_replacement(self, legacy: str) -> None:
        """One case per promoted record, so a failure names the offender.

        The message must carry the canonical path: a bare "extra fields not
        permitted" tells an author their config is wrong but not what to write
        instead, which is the reason `reject_renamed_keys` exists at all.
        """
        rec = RENAMES[legacy]
        validator = reject_renamed_keys(rec.block)
        with pytest.raises(ValueError) as exc:
            validator(type(None), {rec.legacy_key: "anything"})
        assert rec.canonical in str(exc.value)

    def test_a_promoted_spelling_is_no_longer_silently_folded(self) -> None:
        """It must not ALSO still fold — one posture, not two behaviours."""
        legacy = sorted(_DRAINED)[0]
        rec = RENAMES[legacy]
        folded = fold_renamed_keys(rec.block)(type(None), {rec.legacy_key: "anything"})
        assert rec.legacy_key in folded, (
            "the fold validator must leave a raise-posture key alone, so the "
            "reject validator is what the author actually hits"
        )


class TestTheFixtureCorpusStaysSpellingCurrent:
    """``tests/`` is a second config corpus the drain gate never looks at.

    ``check_no_legacy_config_keys.py`` scans ``experiments/`` plus the two
    reference templates. Nothing counts the 65 tracked YAMLs under ``tests/``,
    and ``migrate_config_keys.py`` skips every one of them unless you pass
    ``--all-versions`` — so the repo's own fixer answers "no retired keys found"
    on a file that has one. On 2026-08-09 that hid 26 declarations of two records
    retired months earlier (``workflow.name``, ``seed``).

    **This is hygiene, not a promotion gate, and the distinction is load-bearing.**
    The obvious reading — "a fixture declaring a retired key would break when the
    record is promoted" — is false here: all 65 fixtures declare ``6.0`` (42) or
    ``5.0`` (23), so none reaches the loader and none can break anything. What
    actually blocks a promotion is the **Python** population (dict literals and
    constructor kwargs), which hits the schema with no version gate in front of
    it; 23 of the 30 records at count zero were blocked by that, not by YAML.

    What this class buys is that the fixtures stay usable: whoever bumps them to
    ``1.0`` should not also have to discover that their keys were retired.
    Parsed by PATH, not grepped by leaf — half these renames keep the leaf
    (``logging.tensorboard_dir`` -> ``logging.tracking.tensorboard_dir``), so a
    name match proves nothing about which spelling is in use.
    """

    @staticmethod
    def _declares(doc: object, dotted: str) -> bool:
        node = doc
        parts = dotted.split(".")
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return isinstance(node, dict) and parts[-1] in node

    def _fixtures(self) -> list[pathlib.Path]:
        return tracked_yamls(repo_root() / "tests")

    def test_the_fixture_sweep_is_non_empty(self) -> None:
        """Anti-vacuity: an empty sweep would make the guard below always pass."""
        assert len(self._fixtures()) >= 50, (
            f"expected the tracked tests/ YAML corpus, found "
            f"{len(self._fixtures())} — a narrowed sweep silently disarms this"
        )

    def test_no_fixture_declares_a_retired_spelling(self) -> None:
        import yaml

        promoted = [r for r in RENAMES.values() if r.posture == "raise"]
        offenders = []
        for path in self._fixtures():
            try:
                doc = yaml.safe_load(path.read_text())
            except yaml.YAMLError:
                continue
            if not isinstance(doc, dict):
                continue
            offenders.extend(
                f"{path.relative_to(repo_root())}: {rec.legacy} -> {rec.canonical}"
                for rec in promoted
                if self._declares(doc, rec.legacy)
            )
        assert not offenders, (
            "these fixtures declare a spelling that has been retired. They may "
            "already be failing on `config_version` too — the fix is the same "
            "either way: `migrate_config_keys.py --apply --all-versions "
            "--record <legacy> tests`:\n  " + "\n  ".join(offenders)
        )

    def test_the_guard_can_fire(self) -> None:
        """A synthetic offender must be detected, or the walk above proves nothing."""
        rec = next(r for r in RENAMES.values() if r.posture == "raise")
        parts = rec.legacy.split(".")
        doc: dict = {}
        node = doc
        for part in parts[:-1]:
            node[part] = {}
            node = node[part]
        node[parts[-1]] = "anything"
        assert self._declares(doc, rec.legacy)
        assert not self._declares({}, rec.legacy)


# `TestEveryRaiseRecordCanActuallyBeRefused` lived here until 2026-08-22. Its
# invariant -- a retired spelling must FAIL rather than vanish -- now has ONE
# owner, `tests/unit/config/schemas/test_rename_mounts.py`, and this one was
# deleted rather than kept beside it (non-negotiable 17: never keep the weaker
# checker as defence in depth, because neither is then audited as the sole line
# of defence).
#
# It lost on three counts, all of which it passed green:
#
#  * It asserted a DISJUNCTION, `forbids or mounts_reject`, so a block that
#    mounted nothing passed on `extra="forbid"` alone. `reporting` was in that
#    state and the docstring named it as acceptable -- pydantic refused the key
#    but never said what replaced it, so the record's guided message (the
#    replacement, the reason, #503, and the one-line fixer) had never once
#    reached a user. A characterisation test that pins the defect it should
#    report.
#  * It detected a mount with `vars(schema)`, which sees only the class's OWN
#    dict: an INHERITED mount is invisible (`training/base.py` mounts one that
#    seven diffusion `*Params` subclasses inherit), as is any mount not bound
#    to the name `_reject_renamed`.
#  * `_block_schema` returned `None` for a nested block and the test SKIPPED,
#    so `training.diffusion` was never covered and said so in a skip nobody read.
#
# The history it carried is worth keeping, and now lives in
# `rename_mounts.py`'s module docstring: 57 of ~163 mounted schema classes are
# `extra="ignore"`, and when the first 31 records were promoted to `raise`, 28
# landed in blocks (`data`, `logging`, `losses`, `validation`) that mounted
# `fold_renamed_keys` and no `reject_renamed_keys` -- so the retired key was
# accepted and discarded, and `exclude_defaults` came back `[]` from a config
# that declared `["mse"]`. That is why the promotion precondition exists.


class TestAFoldedKeyLeavesEvidence:
    """A fold is a silent substitution; the run must be able to prove it happened.

    `_bind_config_version` records a ledger substitution when it folds a legacy
    schema VERSION, precisely because the fold is invisible otherwise: the key
    is popped from the raw document and re-emerges at a path the document never
    named, so `diff_declared_vs_resolved`'s walker cannot find it.

    A folded KEY has exactly the same shape and recorded nothing — across a
    corpus that folds 29,250 declarations. `losses.output_domain` arriving at
    `losses.policy.output_domain` was indistinguishable from an arm that
    declared the canonical path directly, so nothing downstream could tell a
    migrated arm from an unmigrated one.
    """

    @staticmethod
    def _fold(block: str, data: dict):
        from spectramr.core.execution_ledger import ExecutionLedger

        ledger = ExecutionLedger.begin_run(source="test")
        out = fold_renamed_keys(block)(type(None), data)
        return out, ledger

    def test_a_folded_key_is_recorded_with_both_spellings(self) -> None:
        from spectramr.core.execution_ledger import SubstitutionClass

        rec = next(r for r in RENAMES.values() if r.posture == "fold" and r.block == "losses")
        out, ledger = self._fold("losses", {rec.legacy_key: "kspace"})

        subs = [
            s
            for s in ledger.substitutions
            if s.class_id is SubstitutionClass.VALUE_CHANGED_ON_FINALIZE and s.path == rec.legacy
        ]
        assert subs, f"folding {rec.legacy} left no evidence"
        assert subs[0].resolved == rec.canonical, (
            "the record must name where the value WENT, not just that it moved"
        )
        assert rec.canonical.split(".")[-1] in str(out)

    def test_an_unfolded_document_records_nothing(self) -> None:
        """No fold, no record -- the ledger must not accrue noise per load."""
        _, ledger = self._fold("losses", {"policy": {"output_domain": "kspace"}})
        assert not ledger.substitutions

    def test_folding_works_with_no_ledger_armed(self) -> None:
        """Recording is best-effort: config loading must never depend on it."""
        from spectramr.core.execution_ledger import ExecutionLedger

        ExecutionLedger.reset()
        assert ExecutionLedger.current() is None
        rec = next(r for r in RENAMES.values() if r.posture == "fold" and r.block == "losses")
        out = fold_renamed_keys("losses")(type(None), {rec.legacy_key: "kspace"})
        assert rec.legacy_key not in out


class TestNestedMount:
    """`mount` lets a record name a schema deeper than a top-level block.

    Before it, `renames_for_block` derived the block as `legacy.split(".", 1)[0]`,
    so a record for `training.diffusion.num_timesteps` was handed to the
    validator mounted on `training` and looked for `num_timesteps` as a direct
    child of `training:`. It did not fail — it simply never fired, which is the
    worst of the three outcomes.
    """

    def test_existing_records_are_unaffected(self) -> None:
        """The default must reproduce the pre-`mount` behaviour exactly.

        If this drifts, every record that predates the field changes which
        validator serves it — a silent, corpus-wide behaviour change.
        """
        from spectramr.config.schemas.renames import RENAMES

        undeclared = [r for r in RENAMES.values() if r.mount is None]
        assert undeclared, "no records left with mount unset; this test is vacuous"
        assert all(r.mount_path == r.block for r in undeclared)

    def test_a_nested_record_reaches_its_own_validator_only(self) -> None:
        from spectramr.config.schemas.renames import renames_for_block

        assert "num_timesteps" in renames_for_block("training.diffusion")
        assert "num_timesteps" not in renames_for_block("training"), (
            "the nested record is also being served by the parent block's "
            "validator, which would look for the key in the wrong place"
        )

    def test_the_fold_fires_and_lands_on_the_field(self) -> None:
        """Registered is not enough — the mechanism has to do work."""
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        config = DiffusionTrainingConfigSchema(num_timesteps=250)

        assert config.timesteps == 250, (
            "num_timesteps did not fold onto timesteps; without the fold this is "
            "1000 (the default) and the declared value is silently discarded"
        )
        assert config.model_extra == {}, "num_timesteps is still being absorbed as an untyped extra"

    def test_the_canonical_spelling_still_works(self) -> None:
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        assert DiffusionTrainingConfigSchema(timesteps=250).timesteps == 250

    def test_two_spellings_that_disagree_raise(self) -> None:
        """A fold must not silently pick a winner."""
        import pytest

        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        with pytest.raises(Exception, match="disagree"):
            DiffusionTrainingConfigSchema(timesteps=250, num_timesteps=500)

    def test_two_spellings_that_agree_are_accepted(self) -> None:
        from spectramr.config.schemas.training.base import DiffusionTrainingConfigSchema

        config = DiffusionTrainingConfigSchema(timesteps=250, num_timesteps=250)
        assert config.timesteps == 250

    def test_a_mount_that_does_not_own_its_legacy_key_raises(self) -> None:
        """The constraint is enforced, not documented.

        A mount that does not contain the legacy key produces a record that is
        registered and never fires. Making it a construction error means that
        state is unreachable rather than merely discouraged.
        """
        import pytest

        from spectramr.config.schemas.renames import RenameRecord

        with pytest.raises(ValueError, match="does not own legacy"):
            RenameRecord(
                legacy="training.diffusion.foo",
                canonical="training.diffusion.bar",
                since="2026-08-12",
                reason="mount is a sibling, not an ancestor",
                mount="training.latent",
                posture="fold",
            )

    def test_a_fold_whose_destination_escapes_its_mount_raises(self) -> None:
        """The fold writes relative to its mount, so an outside destination
        would silently land in the wrong place."""
        import pytest

        from spectramr.config.schemas.renames import RenameRecord

        with pytest.raises(ValueError, match="does not own canonical"):
            RenameRecord(
                legacy="training.diffusion.foo",
                canonical="losses.diffusion.foo",
                since="2026-08-12",
                reason="cross-block move dressed up as a fold",
                mount="training.diffusion",
                posture="fold",
            )

    def test_a_raise_record_may_point_outside_its_mount(self) -> None:
        """`raise` only ever formats a message, so it has no such constraint."""
        from spectramr.config.schemas.renames import RenameRecord

        record = RenameRecord(
            legacy="training.diffusion.foo",
            canonical="losses.diffusion.lambda_foo",
            since="2026-08-12",
            reason="a genuine cross-block move, announced rather than folded",
            mount="training.diffusion",
            posture="raise",
        )
        assert record.mount_path == "training.diffusion"

    def test_the_destination_chain_strips_the_whole_mount(self) -> None:
        """Stripping one component regardless of depth was the original bug.

        With a two-component mount, a one-component strip would have written the
        value to `training.diffusion.diffusion.timesteps`.
        """
        from spectramr.config.schemas.renames import folded_input_paths

        assert folded_input_paths("training.diffusion")["num_timesteps"] == ("timesteps",)


def test_enable_validation_augmentation_is_refused_at_load() -> None:
    """The validation chain never augmented; the retired flag must say so, not vanish."""
    from spectramr.config.schemas.validation import ValidationConfigSchema

    with pytest.raises(ValueError, match="enable_validation_augmentation"):
        ValidationConfigSchema(enable_validation_augmentation=False)
