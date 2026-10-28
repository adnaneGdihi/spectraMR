"""Config-key rename SSOT — one table, consumed by four things.

A key that moves must fail **loudly** at the old spelling. That is not automatic
here: 57 of ~163 mounted schema classes are ``extra="ignore"``, including
``data:``, ``model:``, ``physics:``, ``losses:``, ``logging:`` and
``acceleration:``. Inside those, simply deleting a field is *worse* than leaving
the ambiguity in place — the key stops working **and** stops being visible, so a
config keeps loading while silently training something else.

So every rename is declared here once and consumed by:

1. :func:`reject_renamed_keys` — a ``mode="before"`` validator mixed into the
   owning block, which raises naming the replacement.
2. ``scripts/ci/migrate_config_keys.py`` — rewrites YAML from the same table, so
   the fixer and the check can never disagree about what the rename *is*.
3. ``scripts/ci/check_no_legacy_config_keys.py`` — the corpus gate that drives
   each entry's usage to zero, after which the entry (and the shim) is deleted.
4. :func:`canonical_override_path`, used by ``config/overrides.py`` so a
   ``--override`` written in the old spelling lands on the new key. A staged
   rename that the CLI does not know about turns every legacy ``--override``
   into a hard error; see that function for why.

Two record kinds, because they are not mechanically interchangeable
--------------------------------------------------------------------

A **move** (``optimization.accumulate_grad_steps`` ->
``optimization.gradient_accumulation_steps``) is a pure key rewrite: the value
carries over untouched.

A **sense inversion** — a boolean whose meaning flips, so that
``disable_x: true`` and ``enable_x: true`` are opposite behaviours — is *not*: a
migrator that rewrote the key alone would silently invert every affected arm.
Those carry ``value_transform="negate"`` so the migrator negates the literal and
the error message tells a human to do the same.

**No production record uses it.** The example this docstring used to give,
``losses.disable_default_losses``, is a ``list[str]`` of loss NAMES rather than a
boolean — negating it is meaningless, and it was retired in phase 10d as
``losses.policy.exclude_defaults`` with ``identity``. It was also the repo's only
claimed negated boolean, so the naming rule's "negation is forbidden" clause
currently has no violations to drain. Keep the machinery: it is cheap, tested
against a fixture table, and the first genuine ``enable_``/``disable_`` pair will
need it.

Two postures, because some renames are too big to land atomically
------------------------------------------------------------------

A ``raise`` record (the default) means the legacy spelling is **gone**. Land it
in the same PR as the corpus migration that drives its usage to zero — otherwise
the next load of an unmigrated arm is a hard failure with no fixer run yet.

That rule is only affordable while the affected corpus is small. It is not
affordable for a key like ``optimization.learning_rate``, which **826 loadable
arms declare**: retiring it atomically means one PR that rewrites the entire
corpus, and no reviewer can check that diff against a schema change buried in
the same commit.

A ``fold`` record is the staged form. The ``mode="before"`` validator *moves* the
value to its canonical path instead of raising, so unmigrated YAML keeps loading
while the code above it reads one path only. Critically this is **not** the
forwarding property the design forbids: nothing is added to the model, so
``config.optimization.learning_rate`` raises ``AttributeError`` the moment the
field moves. There is one read path in Python and, temporarily, two accepted
spellings in YAML.

The promotion rule, which is what stops "staged" from meaning "permanent":

1. ``scripts/ci/check_no_legacy_config_keys.py`` counts fold hits every run and
   prints the remaining total. Fold hits do **not** fail the gate; raise hits do.
2. When a fold record's count reaches zero, flip ``posture`` to ``raise``.
3. When every record for a block is ``raise``, delete the fold validator from it.

A fold record's canonical path must stay **inside its own block** — the
validator is mounted on that block and cannot reach a sibling. Pinned by
``tests/unit/config/test_renames.py``.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Literal

ValueTransform = Literal["identity", "negate"]

#: ``raise`` retires a spelling outright; ``fold`` stages the retirement by
#: rewriting the key at parse time. See the module docstring for the promotion
#: rule that drives every ``fold`` back to ``raise``.
Posture = Literal["raise", "fold"]

#: ``block`` for a legacy key that sat at the document ROOT (``seed:``, not
#: ``optimization.seed:``). Spelled as a constant rather than ``""`` so a call
#: site reads as a deliberate choice; ``reject_renamed_keys(ROOT)`` mounts the
#: root validator on :class:`TrainingSettings`.
ROOT = ""


@dataclass(frozen=True, slots=True)
class RenameRecord:
    """One retired config key and where its value now lives."""

    #: Dotted path of the retired spelling, e.g. ``optimization.foo``.
    legacy: str
    #: Dotted path that replaces it. Must resolve to a real field — pinned by
    #: ``tests/unit/config/test_renames.py``, so the table cannot rot into the
    #: very defect class it exists to remove.
    canonical: str
    #: ISO date the rename landed, so an error message can date itself.
    since: str
    #: One line: why the old spelling went away.
    reason: str
    #: ``negate`` for a boolean whose sense flipped; see the module docstring.
    value_transform: ValueTransform = "identity"
    #: ``raise`` (default) retires the spelling now; ``fold`` stages it. Use
    #: ``fold`` only when the corpus is too large to migrate in the same PR, and
    #: only for a canonical path inside the same block.
    posture: Posture = "raise"
    #: The retired key this one LOSES to when both are declared and disagree.
    #:
    #: Normally two spellings with different values is a refusal: only a human
    #: can say which the arm meant. But some pairs are already adjudicated at
    #: parse time -- `validation.val_batch_size` beats
    #: `validation.validation_batch_size`, per both fields' own descriptions and
    #: the retired `effective_val_batch_size` property. Without this the migrator
    #: SKIPs all 74 arms that declare the pair, so the record's corpus count
    #: never reaches zero and its fold shim can never be promoted to `raise`.
    #:
    #: The schema's own resolver reads this field too, so there is ONE rule
    #: rather than a validator and a fixer that agree until they don't.
    superseded_by: str | None = None
    #: Let the migrator CREATE the destination's top-level block when the file
    #: lacks it. Off by default: for a move between two existing schema blocks,
    #: a missing destination usually means the arm relies on that block's
    #: defaults, and conjuring it is inventing structure the author never wrote.
    #: A move into a genuinely NEW block (root ``seed:`` -> ``run.seed``) is the
    #: exception — there the block cannot pre-exist in any file, so refusing
    #: would refuse every file. Opt in per record, never globally.
    create_destination_block: bool = False
    #: Dotted path of the schema class whose validator serves this record.
    #:
    #: ``None`` means "the top-level block of ``legacy``" — what every record
    #: meant before this field existed, so all pre-existing records are
    #: byte-identical with it unset.
    #:
    #: Set it only for a **nested** mount, which is otherwise inexpressible.
    #: :func:`renames_for_block` used to derive the block as
    #: ``legacy.split(".", 1)[0]``, so a record for
    #: ``training.diffusion.num_timesteps`` was served by the validator mounted
    #: on ``training`` and hunted for ``num_timesteps`` as a direct child of
    #: ``training:`` — where it never appears. The record did not fail loudly;
    #: it simply never fired.
    #:
    #: The mount must own the key: ``legacy`` has to sit under it, and for a
    #: ``fold`` so does ``canonical``, because the validator strips the mount
    #: prefix to find the destination chain. Enforced in :meth:`__post_init__`
    #: rather than documented, so a wrong mount cannot be a silent no-op.
    mount: str | None = None

    def __post_init__(self) -> None:
        if self.mount is None:
            return
        prefix = f"{self.mount}."
        if not self.legacy.startswith(prefix):
            raise ValueError(
                f"mount={self.mount!r} does not own legacy={self.legacy!r}: the "
                f"validator is mounted on {self.mount!r} and matches on the leaf "
                f"name, so it would never see this key."
            )
        if self.posture == "fold" and not self.canonical.startswith(prefix):
            raise ValueError(
                f"mount={self.mount!r} does not own canonical={self.canonical!r}. "
                f"A fold writes the value relative to its mount, so a destination "
                f"outside it would be written to the wrong place. Use posture="
                f"'raise' for a cross-block move, or mount the record higher."
            )

    @property
    def block(self) -> str:
        """Top-level block the legacy key lived under; :data:`ROOT` if none."""
        return self.legacy.split(".", 1)[0] if "." in self.legacy else ROOT

    @property
    def mount_path(self) -> str:
        """The schema this record's validator is mounted on.

        Defaults to :attr:`block`, which is what every record meant before
        ``mount`` existed.
        """
        return self.block if self.mount is None else self.mount

    @property
    def legacy_key(self) -> str:
        """Leaf name of the legacy key, as it appears in YAML."""
        return self.legacy.rsplit(".", 1)[-1]

    def message(self) -> str:
        """The error a user sees. Names both spellings, the why, and the fixer."""
        head = f"`{self.legacy}` was renamed to `{self.canonical}` ({self.since}). {self.reason}"
        if self.value_transform == "negate":
            head += (
                " NOTE: the sense is inverted — the replacement means the "
                "opposite, so negate the value as well as the key."
            )
        return head + "\nFix automatically:  python scripts/ci/migrate_config_keys.py <file>"


def _index(records: list[RenameRecord]) -> dict[str, RenameRecord]:
    by_legacy: dict[str, RenameRecord] = {}
    for rec in records:
        if rec.legacy in by_legacy:
            raise ValueError(f"duplicate rename entry for {rec.legacy!r}")
        by_legacy[rec.legacy] = rec
    return by_legacy


#: Fold records whose corpus declaration count has reached **zero**, and which
#: are therefore retired outright: :func:`_folds` gives these ``posture="raise"``
#: instead of ``"fold"``.
#:
#: This is the promotion rule in the module docstring, applied. Keeping the
#: promoted names in one set rather than lifting 31 pairs out of their
#: ``_folds`` calls means the ratchet's progress is legible in a single place,
#: and a record stays next to the siblings that share its rationale.
#:
#: **Never hand-edit this.** Regenerate it from the gate, which derives the
#: count from the fixer itself::
#:
#:     python scripts/ci/check_no_legacy_config_keys.py
#:
#: and copy the records it lists as DRAINED and promotable (0 staged, 0
#: unreachable). Promoting a record the corpus still declares hard-fails those
#: arms at load -- which is the whole reason the count gates the flip.
#:
#: Measured 2026-08-09 on `dev` after PR #906 drained `experiments/inprogress/`:
#: 41 of 161 fold records drained; 120 still declared.
#:
#: Re-measured 2026-08-18 on `dev` @ 30d216a74, immediately before the promotion
#: below: 118 fold records, 17 drained, 101 still declared. **Run the gate over
#: BOTH root sets before promoting** -- the default roots are `experiments/` plus
#: the two reference templates, and a record drained there can still be declared
#: under `tests/`, `src/spectramr/config/presets/` or `scripts/`, which the default
#: invocation never opens::
#:
#:     python scripts/ci/check_no_legacy_config_keys.py
#:     python scripts/ci/check_no_legacy_config_keys.py tests src/spectramr/config/presets scripts
#:
#: (roots are POSITIONAL; `--roots` exits 2 on argparse, which reads as a gate
#: failure in a repo whose convention is that warnings exit 2.) Those 17 were
#: drained in both -- 17 of 118 in the first, inside the 84 of 118 in the second.
#:
#: A leaf grep is not a third opinion, it is a WORSE instrument, and it fails in
#: both directions. Resolving each raw-YAML hit's PARENT BLOCK for those 17 found
#: one over-count (`svd_calibration_lines` at `config_version: '1.0'` -- but under
#: `data.coils:`, i.e. already canonical, because that rename preserves the leaf)
#: and one key that is not this record at all (`val_batch_size` under `data:`,
#: where `extra="ignore"` silently drops it -- filed as #1205). Neither is a
#: declaration of the record being promoted.
#:
#: The gate's countdown is necessary but NOT sufficient for a flip. It scans
#: `experiments/` plus the two reference templates and counts YAML declarations.
#: The population it cannot see is **Python**: a dict literal or constructor kwarg
#: in a test reaches the schema with no corpus and no version gate in front of it,
#: so it breaks the moment the record raises. Of the 30 records sitting at 0 on
#: 2026-08-09, **23 were declared in Python**; 7 were free outright and 3 more
#: were promoted here after repointing the 8 tests that declared them. The
#: remaining 20 need their declarations migrated first.
#:
#: **`data.output_domain` was listed here as "216 sites" and it was promoted on
#: 2026-08-13 with zero migration.** That number did not survive re-measurement:
#: an AST walk for the three declaration shapes below found **0**, and running
#: the 60 `tests/unit` files that mention the leaf produced a failure set
#: IDENTICAL to `dev` (11 failed / 2172 passed vs 11 failed / 2173 passed -- the
#: extra pass is this record's own new case). 216 was almost certainly a
#: leaf-name count: `output_domain` is spelled by three unrelated owners --
#: `losses.output_domain` (a separate fold record, 101 live corpus
#: declarations), the `@register_model` `output_domain` capability (51 files
#: under `models/generators/`), and this record. A grep for the leaf returns
#: 2926 hits corpus-wide and **every one of them belongs to one of the other
#: two**. Re-measure a per-record count through the loader before believing it.
#:
#: A static scan of that population is a LOWER bound, never an upper one. An AST
#: walk found 20; running the tests found 3 more (`data.loso_fold`,
#: `data.max_train_subjects`, `validation.input_dependence_tol`), declared as
#: bare kwargs to the BLOCK schema -- `DataConfigSchema(loso_fold=3)` -- where
#: there is no enclosing `data:` key to reconstruct a path from. Promote, run the
#: tests, and diff against the same selection on `dev`; the scan only tells you
#: where to look.
#:
#: Promotion also turns a legacy kwarg into a `ValidationError`, so any
#: `pytest.raises(ValidationError)` over that kwarg goes VACUOUS rather than red
#: -- it starts catching the rename instead of the constraint it was written for.
#: Repoint those to the canonical spelling in the same change; they will not
#: appear in the failure diff.
#:
#: A third population WRITES config keys rather than declaring them: `src/` code
#: seeding a preset or deriving a value through `default_knob` / `override_knob`
#: (`loader._sync_coil_processing_to_legacy`, and `DataConfigSchema`'s M4Raw
#: block). Neither instrument above can see it -- the YAML gate reads no Python,
#: and an AST walk looking for declaration shapes sees only a string argument.
#: Until 2026-08-18 this was not a risk but a GUARANTEED break: `_canonical_tail`
#: consulted posture, so every such writer silently reverted to the flat spelling
#: the moment its record was promoted, and the arm then failed to load on a key
#: its author never wrote. `data.svd_calibration_lines` below is the record that
#: surfaced it, and `_canonical_tail`'s docstring carries the post-mortem. The
#: helpers no longer consult posture, so this population is now safe by
#: construction; what still needs checking before a promotion is any caller that
#: hard-codes a legacy spelling WITHOUT going through them.
#:
#: Test YAML fixtures are NOT that population, contrary to the obvious guess:
#: all 65 tracked YAMLs under `tests/` declare `6.0` (42) or `5.0` (23), so none
#: of them reaches the loader and none can break a promotion. That is also why
#: `migrate_config_keys.py` reports "no retired keys found" over `tests/` unless
#: you pass `--all-versions` -- it skips every one of them.
#:
#: So: measure the Python population before adding a name here. The countdown
#: alone will tell you a record is free when it is not.
_DRAINED: frozenset[str] = frozenset(
    {
        "data.data_layout",
        "data.enable_graph_encoding",
        "data.expose_acquisition_params",
        "data.expose_conformal_jacobian",
        "data.expose_cortex_flatten_grid",
        "data.expose_field_strength",
        "data.expose_field_strength_target",
        "data.expose_glm_design_matrix",
        "data.expose_scanner_id",
        "data.expose_site_id",
        "data.graph_config",
        "data.graph_type",
        "data.hf_resolution",
        "data.holdout_subject",
        "data.input_artifact",
        "data.kspace_scale_domain",
        "data.loso_fold",
        "data.max_train_subjects",
        "data.max_val_subjects",
        "data.mrixfields_max_resident_volumes",
        "data.mrixfields_output_contrast",
        "data.mrixfields_pairing_policy",
        "data.mrixfields_rescale_per_image",
        "data.mrixfields_slice_mode",
        "data.mrixfields_target_field",
        "data.num_synthetic_samples",
        # NOT its sibling `losses.output_domain`, which shares the LEAF but is a
        # different record with a different owner and **101 live corpus
        # declarations**. The leaf has a third owner too: the `@register_model`
        # `output_domain` capability (51 files under models/generators). A
        # leaf-name measurement of this record reads 2926 raw hits and every one
        # of them belongs to one of the other two -- `data.output_domain` itself
        # sits at 0. Measure this one through the loader, never through a grep.
        #
        # The loader's 0 is exact but NOT the whole corpus: 8 files under
        # `experiments/training/` still spell the flat key in raw YAML. All 8
        # declare `config_version: 5.0`, which `base.py` refuses outright, so the
        # countdown never attempts them and they are already unloadable on the
        # version alone -- this promotion breaks nothing that loads today. It does
        # add a SECOND blocker they will hit whenever the deferred 5.0 corpus is
        # revived (TODO/backlog_config_version_5_0_corpus.md): they will need
        # `data.domain.output`, not just a version bump. Same shape as the 90
        # configs `check_experiment_configs_load.py` stopped counting -- the debt
        # changed buckets, it was not paid.
        "data.output_domain",
        "data.preprocessing_dir",
        "data.rescale_images",
        "data.rescale_percentiles",
        "data.sessions",
        "data.slice_2d",
        # The rename PRESERVES the leaf (`data.svd_calibration_lines` ->
        # `data.coils.svd_calibration_lines`), so a leaf grep cannot tell the two
        # apart. `exp_coil_svd_espirit_sense.yaml` spells the leaf at
        # `config_version: '1.0'` under `data.coils:` -- already canonical, and the
        # reason the loader's 0 is right where a grep's 1 is not. Resolve the PARENT
        # BLOCK before reading a leaf count as a declaration count.
        "data.svd_calibration_lines",
        "data.target_artifact",
        "data.target_contrasts",
        "data.target_sessions",
        "logging.anomaly_check_interval",
        "logging.debug_log_steps",
        "logging.debug_snapshot_interval_steps",
        "logging.debug_snapshot_max_calls",
        "logging.debug_snapshot_save_images",
        "logging.debug_snapshot_save_json",
        "logging.debug_snapshots",
        "logging.report_cases_subdir",
        "logging.save_report_cases",
        "logging.tensorboard_dir",
        "losses.disable_default_losses",
        "optimization.amp_dtype",
        "optimization.amsgrad",
        "optimization.detect_anomalies",
        # NOT its sibling `optimization.use_gradient_checkpointing`, which folds
        # to the same canonical field and still has 35 corpus declarations. Two
        # spellings of one knob drain independently; promoting the pair together
        # would break those 35 arms at load.
        "optimization.gradient_checkpointing",
        "optimization.lookahead",
        "optimization.nesterov",
        "optimization.param_group_overrides",
        "validation.hallucination_test",
        "validation.held_out_severity_eval",
        "validation.input_dependence_tol",
        "validation.multistep_cold_sampling",
        "validation.sampler_steps",
        "validation.shuffle_validation",
        # NOT its sibling `validation.validation_batch_size`, which folds to the same
        # canonical field and still has 26 live corpus declarations -- same shape as
        # `optimization.gradient_checkpointing` above. Two consequences the countdown
        # does not show:
        #
        # (1) That sibling's `superseded_by` now points at a spelling that RAISES.
        #     Nothing breaks -- `_resolve_batch_size_duplicate` only consults it when
        #     BOTH are declared, and in that case raising is the correct answer -- but
        #     the pair mechanism (resolver + `superseded_by`) is vestigial from here
        #     and retires with the sibling, not before it.
        # (2) 20 arms under `experiments/validated/sota/` still spell the flat key in
        #     raw YAML. All 20 declare `config_version: '6.0'`, which `base.py`
        #     refuses outright, so the countdown never attempts them -- they are
        #     already unloadable on the version alone. Same as `data.output_domain`
        #     above: this adds a SECOND blocker for whenever that corpus is revived.
        "validation.val_batch_size",
    }
)


def _folds(pairs: list[tuple[str, str]], reason: str) -> list[RenameRecord]:
    """Staged records for one decomposition, sharing a rationale.

    Written as a helper rather than 35 near-identical literals so the *reason*
    is stated once per sub-block, where it is actually a different reason each
    time. Legacy names stay literal strings, so grepping for a retired key still
    lands here.
    """
    return [
        RenameRecord(
            legacy=legacy,
            canonical=canonical,
            since="2026-07-31",
            reason=reason,
            posture="raise" if legacy in _DRAINED else "fold",
        )
        for legacy, canonical in pairs
    ]


#: Every retired config key. Keyed by dotted legacy path.
RENAMES: dict[str, RenameRecord] = _index(
    [
        # Naming convention (tests/unit/config/test_naming_convention.py).
        RenameRecord(
            legacy="training.ulf_map.data_weight_ratio",
            canonical="training.ulf_map.relative_data_weight",
            since="2026-10-25",
            reason="It is w / rho, an unbounded scale rather than a fraction in [0, 1].",
            mount="training.ulf_map",
        ),
        RenameRecord(
            legacy="training.privileged.n_min",
            canonical="training.privileged.min_keep_fraction",
            since="2026-10-25",
            reason="It is the floor of the marker keep-fraction, not a count.",
            mount="training.privileged",
        ),
        RenameRecord(
            legacy="training.privileged.n_max",
            canonical="training.privileged.initial_keep_fraction",
            since="2026-10-25",
            reason="It is the marker keep-fraction at epoch 0, not a count.",
            mount="training.privileged",
        ),
        RenameRecord(
            legacy="training.dispersion_bloch_ae.n_pools",
            canonical="training.dispersion_bloch_ae.num_pools",
            since="2026-10-25",
            reason="Counts are `num_<thing>`.",
            mount="training.dispersion_bloch_ae",
        ),
        RenameRecord(
            legacy="training.bloch_equivariant_translation.use_bloch_checkpoint",
            canonical="training.bloch_equivariant_translation.enable_bloch_checkpoint",
            since="2026-10-25",
            reason="Boolean switches are `enable_<thing>`.",
            mount="training.bloch_equivariant_translation",
        ),
        RenameRecord(
            legacy="training.riemannian_bloch_diffusion.use_bloch_checkpoint",
            canonical="training.riemannian_bloch_diffusion.enable_bloch_checkpoint",
            since="2026-10-25",
            reason="Boolean switches are `enable_<thing>`.",
            mount="training.riemannian_bloch_diffusion",
        ),
        # Never read: `data` is extra='ignore', so the list was dropped and the
        # synthetic arms ran without their simulator. Registered names only.
        RenameRecord(
            legacy="data.custom_transforms",
            canonical="data.processing.transforms",
            since="2026-10-22",
            reason=(
                "Nothing read `data.custom_transforms`, so its entries were discarded "
                "without error. Declare the transform under `data.processing.transforms` "
                "by its registered name; a dotted `module:` path is never resolved."
            ),
        ),
        RenameRecord(
            legacy="data.augmentation.custom_transforms",
            canonical="data.processing.transforms",
            since="2026-10-22",
            reason=(
                "Nothing read `data.augmentation.custom_transforms`, so its entries were "
                "discarded without error. Declare the transform under "
                "`data.processing.transforms` by its registered name."
            ),
            mount="data.augmentation",
        ),
        RenameRecord(
            legacy="training.scattering_besov.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the scattering_besov strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.steerable_synthesis.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the steerable_synthesis strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.brenier_synthesis.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the brenier_synthesis strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.ulf_redegrad_tta.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the ulf_redegrad_tta strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.cartoon_texture_safe.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the cartoon_texture_safe strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.monotone_field.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the monotone_field strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.field_conditioned_inr.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the field_conditioned_inr strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.field_fno.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the field_fno strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.bloch_field.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the bloch_field strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.field_wiener.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the field_wiener strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.koopman_field.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the koopman_field strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.fisher_rao_geodesic.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the fisher_rao_geodesic strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.lora_modulation.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the lora_modulation strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.mccann_field_path.lambda_l1",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the mccann_field_path strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="training.recoverability_vib.lambda_recon",
            canonical="losses.image_losses",
            since="2026-09-03",
            reason=(
                "The inline L1 weight of the recoverability_vib strategy was read from its own "
                "training sub-block while losses.image_losses documented another value "
                "(16 mrixfields arms said 10.0 and ran 1.0). The loss-weight table is the "
                "one owner: declare `- {name: l1, weight: <w>}` on losses.image_losses "
                "(an author-written losses.reconstruction.lambda_l1 still wins there). "
                "The 43 arms that declared this key were drained, so it retires at raise."
            ),
        ),
        RenameRecord(
            legacy="reporting.per_case_metrics",
            canonical="reporting.per_call_metrics",
            since="2026-08-05",
            reason=(
                "The name promised a distribution the artifact never carried. "
                "The sink writes one row per validation CALL holding that call's "
                "BATCH-AGGREGATE metrics -- there is no per-sample value to "
                "record, because the run computes metrics batch-wise. A plotter "
                "believed the name and rendered eight copies of one scalar as a "
                "box-and-whisker (#503). Zero corpus arms declared the key, so "
                "this retires at `raise` rather than folding."
            ),
        ),
        RenameRecord(
            legacy="optimization.accumulate_grad_steps",
            canonical="optimization.gradient.accumulation_steps",
            since="2026-07-31",
            reason=(
                "Two spellings of one number. Only the canonical one is read by "
                "the training loop; the legacy name was folded in by a "
                "hand-written validator, so which one won depended on "
                "declaration order rather than intent."
            ),
        ),
        # --- the `run:` block (phase 4b) ------------------------------------
        # ORDER MATTERS for the two seed records. `seed` moves first, creating
        # `run:`; `training.seed` then finds `run.seed` already present and, when
        # the values agree, is dropped as redundant rather than moved twice. All
        # 95 arms that set both agree, so none is refused.
        RenameRecord(
            legacy="seed",
            canonical="run.seed",
            since="2026-07-31",
            reason=(
                "Two spellings of one seed. `training.seed` WON at runtime "
                "(pipelines/train.py prefers it), so 96 arms setting the root "
                "key were writing the loser. Both now name `run.seed`."
            ),
            create_destination_block=True,
        ),
        RenameRecord(
            legacy="training.seed",
            canonical="run.seed",
            since="2026-07-31",
            reason=(
                "The seed is a fact about the RUN, not about the training "
                "paradigm — it also drives data shuffling and augmentation, "
                "neither of which lives under `training:`."
            ),
            create_destination_block=True,
        ),
        RenameRecord(
            legacy="device",
            canonical="run.device",
            since="2026-07-31",
            reason=(
                "A bare scalar at the document root, where a reader looks first "
                "and learns least. Resolution is unchanged: still only through "
                "`spectramr.core.compute_device` (non-negotiable 9b)."
            ),
            create_destination_block=True,
        ),
        RenameRecord(
            legacy="model_domain",
            canonical="model.model_domain",
            since="2026-07-31",
            reason=(
                "The root field was a write-only alias: NOTHING read it. A "
                "`mode='before'` validator copied it into `model.model_domain` / "
                "`model.target_domain`, which are the fields consumers actually "
                "read. Write the nested key directly."
            ),
        ),
        RenameRecord(
            legacy="deep_supervision_weight",
            canonical="losses.lambda_deep_supervision",
            since="2026-07-31",
            reason=(
                "A loss weight sitting at the config root. Every other lambda "
                "lives under `losses.` and resolves through the loss-weight SSOT "
                "(pitfall #13b)."
            ),
        ),
        RenameRecord(
            legacy="workflow.name",
            canonical="workflow.regime",
            since="2026-07-31",
            reason=(
                "`name` is the most overloaded key in the corpus — every "
                "loss-list entry and every `metadata:` block has one — so it "
                "read as a label for the block rather than the load-bearing "
                "physical claim it is. `regime` says what the value means."
            ),
        ),
        # --- phase 8: `optimization:` decomposed into sub-blocks -------------
        # All STAGED (`fold`), because 826 loadable arms declare
        # `optimization.learning_rate` alone. Retiring these atomically would
        # mean one PR rewriting the whole corpus with the schema change buried
        # in it. The fold keeps that YAML loading while Python reads one path.
        *_folds(
            [
                ("optimization.optimizer_type", "optimization.optimizer.type"),
                ("optimization.learning_rate", "optimization.optimizer.learning_rate"),
                (
                    "optimization.generator_learning_rate",
                    "optimization.optimizer.generator_learning_rate",
                ),
                (
                    "optimization.discriminator_learning_rate",
                    "optimization.optimizer.discriminator_learning_rate",
                ),
                ("optimization.weight_decay", "optimization.optimizer.weight_decay"),
                ("optimization.beta1", "optimization.optimizer.beta1"),
                ("optimization.beta2", "optimization.optimizer.beta2"),
                ("optimization.betas", "optimization.optimizer.betas"),
                ("optimization.eps", "optimization.optimizer.eps"),
                ("optimization.momentum", "optimization.optimizer.momentum"),
                ("optimization.nesterov", "optimization.optimizer.nesterov"),
                ("optimization.amsgrad", "optimization.optimizer.amsgrad"),
                ("optimization.optimizer_kwargs", "optimization.optimizer.kwargs"),
                ("optimization.lookahead", "optimization.optimizer.lookahead"),
                (
                    "optimization.param_group_overrides",
                    "optimization.optimizer.param_groups",
                ),
            ],
            "Which optimizer, at what rate, with which hyper-parameters is one "
            "decision; it was spread across 15 keys interleaved with memory and "
            "compile knobs. `eps` declared next to `memory_safety_margin` is how "
            "#624 (eps twice, momentum on adamw) stayed invisible.",
        ),
        *_folds(
            [
                (
                    "optimization.gradient_accumulation_steps",
                    "optimization.gradient.accumulation_steps",
                ),
                (
                    "optimization.use_gradient_checkpointing",
                    "optimization.gradient.enable_checkpointing",
                ),
                (
                    "optimization.gradient_checkpointing",
                    "optimization.gradient.enable_checkpointing",
                ),
                (
                    "optimization.detect_anomalies",
                    "optimization.gradient.detect_anomalies",
                ),
                (
                    "optimization.enable_gradient_clipping",
                    "optimization.gradient.clip.enabled",
                ),
                (
                    "optimization.gradient_clip_method",
                    "optimization.gradient.clip.method",
                ),
                (
                    "optimization.gradient_clip_value",
                    "optimization.gradient.clip.value",
                ),
            ],
            "Everything between `loss.backward()` and `optimizer.step()`. Note "
            "the two checkpointing spellings collapse to ONE field here — "
            "`gradient_checkpointing` was an alias declared only so it would "
            "validate under extra=forbid, which is one knob too many.",
        ),
        *_folds(
            [
                ("optimization.use_amp", "optimization.precision.enabled"),
                ("optimization.amp_dtype", "optimization.precision.dtype"),
            ],
            "AMP is one decision with two parts. Split across a `use_*` boolean "
            "and a dtype 40 lines apart, the third state — dtype 'float32' "
            "disables AMP even when the flag is true — was unreadable.",
        ),
        *_folds(
            [
                ("optimization.compile_model", "optimization.compile.enabled"),
                ("optimization.compile_mode", "optimization.compile.mode"),
                ("optimization.compile_backend", "optimization.compile.backend"),
                ("optimization.compile_fullgraph", "optimization.compile.fullgraph"),
                ("optimization.compile_dynamic", "optimization.compile.dynamic"),
            ],
            "A `compile_` prefix repeated five times is a sub-block spelled the long way.",
        ),
        *_folds(
            [
                (
                    "optimization.enable_memory_monitoring",
                    "optimization.memory.enable_monitoring",
                ),
                (
                    "optimization.memory_monitoring_interval",
                    "optimization.memory.monitoring_interval",
                ),
                (
                    "optimization.enable_memory_fragmentation_mitigation",
                    "optimization.memory.enable_fragmentation_mitigation",
                ),
                (
                    "optimization.memory_cleanup_interval",
                    "optimization.memory.cleanup_interval",
                ),
                (
                    "optimization.optimize_batch_size_for_memory",
                    "optimization.memory.enable_batch_size_optimization",
                ),
                (
                    "optimization.memory_safety_margin",
                    "optimization.memory.safety_margin",
                ),
            ],
            "Diagnostics, not a training decision — and the `memory_` prefix "
            "repeated six times was already naming the block.",
        ),
        # --- phase 9: `data:` decomposition, first tranche -------------------
        # STAGED like phase 8, but the stakes differ: `data` is extra="ignore",
        # so a key this table forgets is not a loud error -- it is silently
        # discarded and the arm trains on the default (#550). The 86-name
        # totality test in tests/unit/config/schemas/test_renames.py is what
        # stands between that and a green run.
        *_folds(
            [
                ("data.batch_size", "data.loader.batch_size"),
                ("data.num_workers", "data.loader.num_workers"),
                ("data.persistent_workers", "data.loader.persistent_workers"),
                ("data.prefetch_factor", "data.loader.prefetch_factor"),
                ("data.reserved_cores", "data.loader.reserved_cores"),
                ("data.pin_memory", "data.loader.pin_memory"),
                ("data.max_prefetch", "data.loader.prefetch_factor"),
            ],
            "How samples are batched and moved to the device -- mechanical "
            "knobs that change throughput, never what the model sees. "
            "`max_prefetch` was a deprecated alias folded in by a hand-written "
            "validator; it lands on the same field here, so shim, fixer and "
            "gate finally read one table. No corpus arm declares both.",
        ),
        *_folds(
            [
                ("data.expose_acquisition_params", "data.expose.acquisition_params"),
                ("data.expose_conformal_jacobian", "data.expose.conformal_jacobian"),
                (
                    "data.expose_cortex_flatten_grid",
                    "data.expose.cortex_flatten_grid",
                ),
                ("data.expose_field_strength", "data.expose.field_strength"),
                (
                    "data.expose_field_strength_target",
                    "data.expose.field_strength_target",
                ),
                ("data.expose_glm_design_matrix", "data.expose.glm_design_matrix"),
                ("data.expose_scanner_id", "data.expose.scanner_id"),
                ("data.expose_site_id", "data.expose.site_id"),
            ],
            "Eight flags whose shared `expose_` prefix was already naming the "
            "block. `expose.scanner_id: true` reads as the sentence it is.",
        ),
        *_folds(
            [
                (
                    "data.mrixfields_max_resident_volumes",
                    "data.mrixfields.max_resident_volumes",
                ),
                ("data.mrixfields_output_contrast", "data.mrixfields.output_contrast"),
                ("data.mrixfields_pairing_policy", "data.mrixfields.pairing_policy"),
                (
                    "data.mrixfields_rescale_per_image",
                    "data.mrixfields.rescale_per_image",
                ),
                ("data.mrixfields_slice_mode", "data.mrixfields.slice_mode"),
                ("data.mrixfields_target_field", "data.mrixfields.target_field"),
            ],
            "Cohort-specific options for the paired ULF/HF source. Six fields "
            "sharing one prefix are a sub-block spelled the long way.",
        ),
        *_folds(
            [
                ("data.coil_processing_mode", "data.coils.processing_mode"),
                ("data.num_virtual_coils", "data.coils.num_virtual_coils"),
                ("data.svd_calibration_lines", "data.coils.svd_calibration_lines"),
            ],
            "How multi-coil data is reduced before the model sees it. Note "
            "`num_virtual_coils` keeps its full name inside `coils:` -- the "
            "naming rule is `num_<thing>` and the thing is virtual coils.",
        ),
        *_folds(
            [
                ("data.split_strategy", "data.split.type"),
                ("data.validation_split", "data.split.validation_fraction"),
                ("data.holdout_site", "data.split.holdout_site"),
                ("data.train_sites", "data.split.train_sites"),
                ("data.holdout_subject", "data.split.holdout_subject"),
                ("data.loso_fold", "data.split.loso_fold"),
                ("data.max_train_subjects", "data.split.max_train_subjects"),
                ("data.max_val_subjects", "data.split.max_val_subjects"),
            ],
            "What decides which records train and which validate. The strategy "
            "now sits next to the companions that make it valid -- `loso` needs "
            "`holdout_site`, `loso_subject` needs exactly one of "
            "`holdout_subject`/`loso_fold` -- which is precisely what "
            "`validate_split_strategy` checks. `validation_split` becomes "
            "`validation_fraction` because it is a fraction in [0, 1], not a "
            "split. `test_split` stays FLAT: nothing reads it (#665), and a "
            "tidy home for an inert knob implies it works.",
        ),
        *_folds(
            [
                ("data.output_domain", "data.domain.output"),
                ("data.target_channels", "data.domain.target_channels"),
                ("data.input_artifact", "data.domain.input_artifact"),
                ("data.target_artifact", "data.domain.target_artifact"),
                ("data.graph_type", "data.domain.graph_type"),
                ("data.enable_graph_encoding", "data.domain.enable_graph_encoding"),
                ("data.graph_config", "data.domain.graph_config"),
            ],
            "What representation the loader hands the model -- image or "
            "k-space, how many target channels, which artifact variants, and "
            "whether the sample arrives as a graph. `output_domain` drops its "
            "suffix inside `domain:` (the block supplies the noun, as with "
            "`expose:`), which also disambiguates it from "
            "`losses.output_domain`. `return_image_domain` stays FLAT: no "
            "reader in src/spectramr, already carried in KNOWN_UNCONSUMED.",
        ),
        *_folds(
            [
                # k-space scaling
                (
                    "data.normalize_kspace",
                    "data.processing.enable_kspace_normalization",
                ),
                ("data.kspace_percentile", "data.processing.kspace_percentile"),
                ("data.kspace_scale_domain", "data.processing.kspace_scale_domain"),
                ("data.log_scaling", "data.processing.enable_log_scaling"),
                (
                    "data.log_scaling_center_fraction",
                    "data.processing.log_scaling_center_fraction",
                ),
                # image-domain normalisation
                ("data.normalization_type", "data.processing.normalization_type"),
                ("data.normalization_kwargs", "data.processing.normalization_kwargs"),
                (
                    "data.normalize_images",
                    "data.processing.enable_image_normalization",
                ),
                ("data.rescale_images", "data.processing.enable_image_rescale"),
                ("data.rescale_range", "data.processing.rescale_range"),
                ("data.rescale_percentiles", "data.processing.rescale_percentiles"),
                # downstream
                ("data.data_range", "data.processing.data_range"),
                ("data.transforms", "data.processing.transforms"),
            ],
            "What is done to the VALUES between disk and the model, in the order "
            "it happens: k-space scaling, then image-domain normalisation and "
            "rescale, then config-driven transforms. The k-space group sat ~200 "
            "lines from the image group, so nothing showed they compose. The four "
            "booleans take the ratified `enable_<thing>` spelling; each now sits "
            "directly above the nouns it gates. `resample`/`crop_or_pad` stay "
            "put -- they were already nested blocks, and this phase groups "
            "scalars rather than re-parenting blocks.",
        ),
        *_folds(
            [
                ("data.patch_size", "data.sampling.patch_size"),
                ("data.samples_per_volume", "data.sampling.samples_per_volume"),
                ("data.queue_length", "data.sampling.queue_length"),
                ("data.enable_slab_mode", "data.sampling.enable_slab_mode"),
                ("data.slice_2d", "data.sampling.enable_slice_2d"),
                (
                    "data.num_synthetic_samples",
                    "data.sampling.num_synthetic_samples",
                ),
            ],
            "How samples are DRAWN from a volume, before anything is done to "
            "them. The TorchIO queue's three knobs finally sit together: "
            "`patch_size` is what a sample IS, `samples_per_volume` how many "
            "are taken per subject per epoch, `queue_length` the RAM buffer "
            "holding them -- tuning one blind to the others is how a queue "
            "build comes to dominate a smoke run. `patch_size` is also the "
            "most overloaded name in the repo (~134 model attributes mean "
            "the ViT/MAE patch embedding, not this), and `sampling:` "
            "disambiguates it. `phase_encode_axis` stays FLAT: nothing reads "
            "it from config -- FMRIVolumeDataset takes it as a constructor "
            "default and no dataset_type selects that class.",
        ),
        *_folds(
            [
                # input side
                ("data.contrasts", "data.pairing.contrasts"),
                ("data.sessions", "data.pairing.sessions"),
                # target side (each defaults to its input counterpart)
                ("data.target_contrasts", "data.pairing.target_contrasts"),
                ("data.target_sessions", "data.pairing.target_sessions"),
                # which end of a pair is which, and what counts as a pair
                ("data.single_contrast", "data.pairing.single_contrast"),
                ("data.bidirectional_mode", "data.pairing.bidirectional_mode"),
                ("data.hf_resolution", "data.pairing.hf_resolution"),
                ("data.allow_unpaired", "data.pairing.allow_unpaired"),
            ],
            "What counts as the INPUT and what counts as the TARGET. The plan "
            "called this block `contrast:`, which is wrong twice over: three of "
            "the eight fields are ULF/HF field-strength pairing with no "
            "contrast content, and `data.contrast` would sit beside the "
            "pre-existing `data.input_contrast`/`data.target_contrast` -- which "
            "are ContrastConfigSchema NORMALISATION specs (percentile, "
            "out_range, clamp), not selection filters. Leaf names are carried "
            "across unchanged for the same reason: `contrasts` is NOT renamed "
            "to `input_contrasts`, because singular-vs-plural is the worst "
            "available disambiguator against `data.input_contrast` and the "
            "block prefix already does that work. `bidirectional_mode` keeps "
            "its `_mode` spelling rather than becoming `direction`: it is a "
            "genuine mode (the `hf_to_hf` value DROPS the opposite arm), and "
            "132 arms declare it.",
        ),
        *_folds(
            [
                # the tree, and how it is arranged
                ("data.data_root", "data.source.root"),
                ("data.data_layout", "data.source.layout"),
                # the files that enumerate what to read from it
                ("data.index_path", "data.source.index_path"),
                (
                    "data.validation_index_path",
                    "data.source.validation_index_path",
                ),
                (
                    "data.paired_manifest_path",
                    "data.source.paired_manifest_path",
                ),
                ("data.preprocessing_dir", "data.source.preprocessing_dir"),
            ],
            "WHERE the bytes come from. Six answers to one question, previously "
            "scattered across ~300 lines of `data:`. `dataset_type` is "
            "deliberately NOT folded: it is a DISPATCH key rather than a "
            "location -- dataset_instantiator, config_health_checker and "
            "axis_exposure.DATASET_TYPE_SIGNAL_DOMAINS branch on it, the plan "
            "binds it to `workflow.regime`, and 636 arms declare it as the "
            "first line of their `data:` block where nothing about it is "
            "ambiguous. `datasets` and `manifest_roles` stay for a different "
            "reason: both are already nested blocks, and this phase groups "
            "scalars rather than re-parenting blocks. `data_root`/`data_layout` "
            "drop the `data_` prefix the outer block supplies; the `_path` "
            "leaves keep their full names, since under the naming rule "
            "`<thing>_path` is a file and index/validation_index/paired_manifest "
            "are the things.",
        ),
        # --- phase 10a: `validation:` decomposition ---------------------------
        # Two of these groups map TWO legacy spellings onto ONE canonical leaf.
        # That is only safe because it was measured, not assumed:
        #   * eval_interval / frequency_steps -- 58 arms declare both, 0 disagree.
        #   * val_batch_size / validation_batch_size -- 86 declare both and 74
        #     DISAGREE, so the fold alone would refuse them. The parent's
        #     `_resolve_batch_size_duplicate` collapses that pair first, with the
        #     short form winning, before either record is applied.
        *_folds(
            [
                ("validation.eval_interval", "validation.schedule.interval_steps"),
                ("validation.frequency_steps", "validation.schedule.interval_steps"),
                ("validation.eval_on_epoch", "validation.schedule.on_epoch"),
                (
                    "validation.frequency_epochs",
                    "validation.schedule.interval_epochs",
                ),
            ],
            "HOW OFTEN validation runs. Four spellings of one cadence, two of "
            "them the same number: `eval_interval` and `frequency_steps` share a "
            "default of 1000, 58 arms declare both, and not one disagrees -- so "
            "merging them is a measurement, not a guess. Only `eval_interval` "
            "was ever read (pipelines/training_loop.py), which means the merge "
            "also carries 58 arms' previously-dead `frequency_steps` into the "
            "loop. `on_epoch` and `interval_epochs` are NOT duplicates and both "
            "survive: the first selects epoch-based mode, the second is its N.",
        ),
        *_folds(
            [
                ("validation.val_batch_size", "validation.loader.batch_size"),
                ("validation.val_chunk_size", "validation.loader.chunk_size"),
                (
                    "validation.num_validation_batches",
                    "validation.loader.num_batches",
                ),
                ("validation.num_samples", "validation.loader.num_samples"),
                ("validation.shuffle_validation", "validation.loader.shuffle"),
            ],
            "HOW validation batches are drawn. The two batch-size spellings were "
            "not interchangeable: `effective_val_batch_size` and data_builder.py "
            "preferred the short `val_batch_size`, while "
            "training_pipeline_director.py read only the long "
            "`validation_batch_size` -- and 74 arms declare the two with "
            "different values, so the batch size a run used depended on which "
            "builder path it took. One field ends that. The `val_`/`validation_` "
            "prefixes go: the block already says which pass this is.",
        ),
        # The loser of the batch-size pair, declared on its own so it can carry
        # `superseded_by`. 74 arms declare both with DIFFERENT values; without
        # this the migrator SKIPs every one of them, the record's count never
        # reaches zero, and its fold shim can never be promoted to `raise`.
        RenameRecord(
            legacy="validation.validation_batch_size",
            canonical="validation.loader.batch_size",
            since="2026-07-31",
            reason=(
                "The long spelling of the validation batch size. It LOSES to "
                "`val_batch_size`, which both field descriptions and the retired "
                "`effective_val_batch_size` property already named as the winner "
                "-- and which `ValidationConfigSchema._resolve_batch_size_duplicate` "
                "already enforces at parse time. The migrator reads the same "
                "`superseded_by` field, so the fixer and the validator cannot "
                "disagree about which spelling wins."
            ),
            posture="fold",
            superseded_by="validation.val_batch_size",
        ),
        *_folds(
            [
                ("validation.metrics", "validation.scoring.compute"),
                ("validation.primary_metric", "validation.scoring.primary"),
                ("validation.domain", "validation.scoring.domain"),
                (
                    "validation.output_transform",
                    "validation.scoring.output_transform",
                ),
                (
                    "validation.compute_image_metrics",
                    "validation.scoring.enable_image_metrics",
                ),
            ],
            "WHAT validation measures. The block is `scoring` rather than the "
            "obvious `metrics` because `validation.metrics` is one of the keys "
            "being retired here, and the fold matches on the key ALONE: a block "
            "named `metrics` would make that key mean both the old scalar and "
            "its own destination, so a legacy arm would break on declaration "
            "order and a MIGRATED arm writing `metrics: {compute: [...]}` would "
            "have the whole block folded into itself. The list leaf is `compute` "
            "for a plainer reason -- it matches the standing `metrics.compute_*` "
            "-> `metrics.compute: [psnr, ssim]` migration, so the two surfaces "
            "read the same way -- and `compute_image_metrics` becomes "
            "`enable_image_metrics` because it is a switch, which `compute_` "
            "beside a `compute` list would disguise. `validation_metric` is "
            "deliberately NOT folded onto `primary` despite naming the same "
            "idea: the defaults differ ('loss' vs 'psnr'), so folding would "
            "silently repoint the 53 arms that set it.",
        ),
        *_folds(
            [
                (
                    "validation.enable_visualization",
                    "validation.visualization.enabled",
                ),
                (
                    "validation.visualization_interval",
                    "validation.visualization.interval",
                ),
            ],
            "Validation image dumps: the gate and its cadence. "
            "`enable_visualization` becomes the block's bare `enabled` -- the "
            "naming rule reserves that spelling for exactly this. "
            "`num_visualizations` and `visualization_dir` are NOT folded in: "
            "nothing reads either (both sit in KNOWN_UNCONSUMED), and giving an "
            "inert knob a tidy home implies it works.",
        ),
        *_folds(
            [
                ("validation.sampler_steps", "validation.sampling.steps"),
                (
                    "validation.multistep_cold_sampling",
                    "validation.sampling.enable_multistep_cold",
                ),
            ],
            "Reverse-diffusion sampling AT VALIDATION TIME, where it differs "
            "from training. Both leaves shed the `sampl*` stem the block now "
            "supplies.",
        ),
        *_folds(
            [
                (
                    "validation.input_dependence_tol",
                    "validation.gates.input_dependence_tol",
                ),
                (
                    "validation.held_out_severity_eval",
                    "validation.gates.held_out_severity_eval",
                ),
                (
                    "validation.hallucination_test",
                    "validation.gates.hallucination_test",
                ),
            ],
            "The checks that can FAIL a run, as opposed to `metrics`, which only "
            "measures. Each exists to catch a way a run can score well while "
            "meaning nothing: a measurement-independent DC blob (#20), an "
            "in-distribution-only severity range, hallucinated structure. "
            "Grouping them puts the anti-facade gates where a reader looks for "
            "them instead of scattered among cadence and batch-size knobs.",
        ),
        # --- phase 10b: `logging:` decomposition ------------------------------
        # `logging:` is extra="ignore", so unlike `validation:` a key this table
        # forgets does not raise -- it VANISHES and the arm runs on the default.
        # `TestPhase10bFoldTableIsTotal` pins all 40 pre-split scalars against that.
        *_folds(
            [
                ("logging.experiment_name", "logging.identity.experiment"),
                ("logging.run_name", "logging.identity.run"),
                ("logging.notes", "logging.identity.notes"),
                ("logging.tags", "logging.identity.tags"),
            ],
            "WHAT this run is called, for a human and for the tracking backend. "
            "The `_name` suffixes go: inside `identity:` there is nothing else "
            "`experiment` or `run` could denote.",
        ),
        *_folds(
            [
                ("logging.level", "logging.sinks.level"),
                ("logging.silent", "logging.sinks.silent"),
                ("logging.log_to_console", "logging.sinks.to_console"),
                ("logging.log_to_file", "logging.sinks.to_file"),
                ("logging.log_dir", "logging.sinks.dir"),
            ],
            "WHERE log lines go and how much of them. `level` and `silent` are "
            "not destinations but they decide what reaches one, so they read "
            "better here than three blocks away. The `log_` prefix goes -- the "
            "outer block already supplies it, which is why 41 arms wrote "
            "`log_level` and had it silently discarded (issue #675).",
        ),
        *_folds(
            [
                ("logging.log_interval", "logging.intervals.log"),
                ("logging.save_interval", "logging.intervals.save"),
                (
                    "logging.validation_image_interval",
                    "logging.intervals.validation_images",
                ),
                (
                    "logging.anomaly_check_interval",
                    "logging.intervals.anomaly_check",
                ),
            ],
            "Every-N cadences. The block supplies the word `interval`, so the "
            "leaves are the things being timed rather than four repetitions of "
            "the suffix.",
        ),
        *_folds(
            [
                ("logging.log_input_images", "logging.images.log_input"),
                ("logging.log_validation_images", "logging.images.log_validation"),
                ("logging.log_difference_images", "logging.images.log_difference"),
                ("logging.save_validation_images", "logging.images.save_validation"),
                ("logging.max_images_per_batch", "logging.images.max_per_batch"),
            ],
            "Which image panels are written, and how many. The log/save "
            "distinction is kept on the leaf because the two are genuinely "
            "different sinks (TensorBoard vs disk) and bootstrap.py ORs them. "
            "`save_images_per_epoch` is NOT folded in: nothing reads it and 882 "
            "arms set it, so a tidy home would imply it works.",
        ),
        *_folds(
            [
                (
                    "logging.enable_experiment_tracking",
                    "logging.tracking.enabled",
                ),
                ("logging.tracking_service", "logging.tracking.service"),
                ("logging.enable_tensorboard", "logging.tracking.enable_tensorboard"),
                ("logging.tensorboard_dir", "logging.tracking.tensorboard_dir"),
            ],
            "The experiment-tracking backend. `enable_experiment_tracking` "
            "becomes the block's bare `enabled` gate. The two `wandb_*` fields "
            "are NOT folded in: W&B is deferred, so a non-null value now RAISES "
            "rather than being silently accepted and ignored (issue #675, "
            "`LoggingConfigSchema._refuse_deferred_wandb`).",
        ),
        *_folds(
            [
                ("logging.debug_snapshots", "logging.snapshots.enabled"),
                (
                    "logging.debug_snapshot_interval_steps",
                    "logging.snapshots.interval_steps",
                ),
                ("logging.debug_snapshot_max_calls", "logging.snapshots.max_calls"),
                (
                    "logging.debug_snapshot_save_images",
                    "logging.snapshots.save_images",
                ),
                ("logging.debug_snapshot_save_json", "logging.snapshots.save_json"),
                ("logging.debug_log_steps", "logging.snapshots.log_steps"),
            ],
            "The per-step tensor/JSON debug dumps. The block is `snapshots`, "
            "NOT the plan's `debug_snapshots`: that is one of the retired "
            "scalars' own names, and a destination block may not share a legacy "
            "leaf name -- a migrated arm writing `debug_snapshots: {enabled: "
            "true}` would have the whole block folded into itself. The `debug_` "
            "prefix is redundant besides: a snapshot is a debug artifact.",
        ),
        *_folds(
            [
                ("logging.save_report_cases", "logging.report_cases.enabled"),
                ("logging.report_cases_subdir", "logging.report_cases.subdir"),
            ],
            "Per-case artifacts kept for the end-of-training report.",
        ),
        # --- phase 10d: `losses.policy` ---------------------------------------
        *_folds(
            [
                ("losses.output_domain", "losses.policy.output_domain"),
                ("losses.disable_default_losses", "losses.policy.exclude_defaults"),
            ],
            "HOW the objective is assembled, as opposed to WHICH terms are in "
            "it. `losses:` is otherwise sixteen loss-family blocks, so a loose "
            "scalar beside them reads as a seventeenth; neither of these is a "
            "family. `disable_default_losses` becomes `exclude_defaults` and "
            "does NOT use the `negate` transform: the naming rule forbids a "
            "negated boolean, but this field is a `list[str]` of loss NAMES, so "
            "inverting it is meaningless -- `enable_default_losses: ['mse']` "
            "would mean 'enable ONLY mse', the opposite of a filter. "
            "`normalize_losses`, `clip_loss_value` and `loss_scaling` are NOT "
            "folded in: all three have zero readers and zero corpus "
            "declarations (issue #676), and `lambda_deep_supervision` is "
            "already correctly placed as a loss WEIGHT under the naming rule.",
        ),
        # --- phase 11: the `acceleration:` block becomes `undersampling:` -------
        # The only fold that crosses top-level blocks, and the reason
        # `fold_renamed_keys` grew a ROOT mount: this moves the BLOCK, not a key
        # inside it, so the validator has to be one that can reach every
        # sibling. Mounted on `TrainingSettings`.
        RenameRecord(
            legacy="acceleration",
            canonical="undersampling",
            since="2026-08-02",
            reason=(
                "The block name meant two unrelated things: the MRI k-space "
                "ACCELERATION FACTOR (base_acceleration, center_fraction, mask "
                "types, the schedule ladder -- 26 fields, what the block is "
                "actually for) and COMPUTE acceleration (mixed_precision, "
                "use_compile, use_distributed, use_gradient_checkpointing, "
                "gradient_accumulation_steps). A reader could not tell which "
                "sense a given key belonged to from the block name. The five "
                "compute knobs are all inert -- zero readers on an acceleration "
                "receiver -- and four of them duplicate a live "
                "`optimization.*` field; they stay flat inside the renamed "
                "block rather than being folded, so their inertness stays "
                "visible (issue #680)."
            ),
            posture="fold",
        ),
        # The first record to use `mount`, and the reason the field exists.
        RenameRecord(
            legacy="training.diffusion.num_timesteps",
            canonical="training.diffusion.timesteps",
            since="2026-08-12",
            mount="training.diffusion",
            reason=(
                "Two spellings of the timestep count, and the corpus was told to "
                "use the one nothing reads. `DiffusionTrainingConfigSchema` "
                "declares `timesteps`; `num_timesteps` was absorbed by that "
                "class's `extra='allow'` into an untyped attribute, while THREE "
                "places advertised it as canonical -- "
                "`validation_constants.py`, `base.py`'s own path list, and a "
                "`loss.py` field description saying 'DEPRECATED: use "
                "training.diffusion.num_timesteps'. The reader takes the other "
                "spelling (`cold_diffusion_inference_strategy.py` reads "
                "`diffusion_config.get('timesteps', 1000)`) and one site pops "
                "`num_timesteps` outright. 28 arms declare it, and 3 of them "
                "were RUNNING WRONG: they asked for 100 timesteps and got 1000. "
                "The other 25 asked for 1000, which is also the default, so they "
                "were unaffected by luck rather than by design (issue #980)."
            ),
            posture="fold",
        ),
        # ── #421: `losses.reconstruction` duplicated three `losses.physics` pairs ──
        # Both spellings canonicalise to the SAME loss, so the moment a config is
        # materialised -- `Model(**cfg.model_dump())`, which marks every defaulted
        # key as author-set -- `build_loss_weight_table` saw two declarations of
        # one loss at two different defaults and raised. That is not theoretical:
        # `presets/cli.py` and `pipeline_strategy.py` both round-trip a complete
        # dump today.
        #
        # `losses.physics` wins the election on every axis measured on 2026-08-23:
        # it holds the only readers (`physics_driven_strategy.py:362`,
        # `loss_builder.py:473`), it is the only spelling the v1.0 reference
        # template declares, and it is where all 56 corpus arms that set these
        # lambdas already put them -- ZERO arms used the `reconstruction` spelling.
        # A `raise` is therefore free of corpus migration (non-negotiable 17: the
        # loser's declaration is deleted, not kept in sync). Deleting the fields
        # WITHOUT these records would be silent: every loss sub-schema is
        # `extra="ignore"`, so the key would keep loading and stop working.
        #
        # The `enable_*` gates ride with their `lambda_*` partners deliberately --
        # retiring a weight while leaving its gate behind produces a knob that
        # validates, stamps into provenance and controls nothing (non-negotiable 8).
        # All four retired gates had zero readers anywhere in `src/` and zero
        # corpus declarations.
        RenameRecord(
            legacy="losses.reconstruction.lambda_snr_preserving",
            canonical="losses.physics.lambda_snr_preserving",
            since="2026-08-23",
            mount="losses.reconstruction",
            posture="raise",
            reason=(
                "Duplicated `losses.physics.lambda_snr_preserving`; both "
                "canonicalise to `snr_preserving`, which made a materialised "
                "config raise a false weight conflict (#421). Zero corpus arms "
                "declared this spelling. NOTE that the elected winner is itself "
                "unread -- it stays in the `test_schema_key_consumption.py` "
                "allowlist as pre-existing non-negotiable 8 debt, tracked "
                "separately; this record only elects the owner."
            ),
        ),
        RenameRecord(
            legacy="losses.reconstruction.enable_snr_preserving",
            canonical="losses.physics.enable_snr_preserving",
            since="2026-08-23",
            mount="losses.reconstruction",
            posture="raise",
            reason=(
                "Gate for the retired `lambda_snr_preserving` above. Zero "
                "readers, zero corpus declarations (#421)."
            ),
        ),
        RenameRecord(
            legacy="losses.reconstruction.lambda_bloch_residual",
            canonical="losses.physics.lambda_bloch_residual",
            since="2026-08-23",
            mount="losses.reconstruction",
            posture="raise",
            reason=(
                "Duplicated `losses.physics.lambda_bloch_residual`, which is the "
                "spelling `physics_driven_strategy.py:362` actually reads. Both "
                "canonicalise to `bloch_residual`, at defaults 0.0 and 1.0, so a "
                "materialised config raised a false conflict (#421). Zero corpus "
                "arms declared this spelling."
            ),
        ),
        RenameRecord(
            legacy="losses.reconstruction.enable_bloch_residual",
            canonical="losses.physics.enable_bloch_residual",
            since="2026-08-23",
            mount="losses.reconstruction",
            posture="raise",
            reason=(
                "Gate for the retired `lambda_bloch_residual` above. Zero "
                "readers, zero corpus declarations (#421)."
            ),
        ),
        RenameRecord(
            legacy="losses.reconstruction.lambda_physics_constraint",
            canonical="losses.physics.lambda_physics_constraint",
            since="2026-08-23",
            mount="losses.reconstruction",
            posture="raise",
            reason=(
                "Duplicated `losses.physics.lambda_physics_constraint`, which is "
                "the spelling `loss_builder.py:473` reads as `lambda_dc`. Both "
                "canonicalise to `physics_constraint`, at defaults 0.0 and 0.1, "
                "so a materialised config raised a false conflict (#421). Zero "
                "corpus arms declared this spelling."
            ),
        ),
        RenameRecord(
            legacy="losses.reconstruction.enable_physics_constraint",
            canonical="losses.physics.enable_physics_constraint",
            since="2026-08-23",
            mount="losses.reconstruction",
            posture="raise",
            reason=(
                "Gate for the retired `lambda_physics_constraint` above. Zero "
                "readers, zero corpus declarations (#421)."
            ),
        ),
        # ── #421: `lambda_content` was ONE field serving TWO different losses ──
        # This schema described it as the CONTENT CONSISTENCY weight, and two
        # consumers read it that way by raw field name -- `unified_disentangled.py`
        # (where `KNOWN_LOSS_COMPONENTS` lists `content_consistency` and
        # `perceptual` as SEPARATE components) and `disentangled_strategy.py`.
        # But `LossRegistry._aliases` maps `content -> perceptual`, so the weight
        # table treated it and `lambda_perceptual` as one loss. A disentangled
        # author could not express content != perceptual by any route: declaring
        # both crashed, and declaring only `content` leaked their content weight
        # into `perceptual`.
        #
        # This is the ONE collision of the eight that fired unconditionally --
        # `reconstruction` is materialised on every config and both defaults live
        # there (0.0 vs 10.0), so it did not need a partner section to be declared.
        #
        # Posture is `raise` even though a same-mount `fold` is expressible, and
        # that is deliberate: a fold would silently write an ambiguous key to ONE
        # of its two successors, which is exactly the wrong-place write the fold
        # rule exists to prevent. An author must say which loss they meant.
        RenameRecord(
            legacy="losses.reconstruction.lambda_content",
            canonical="losses.reconstruction.lambda_perceptual",
            since="2026-08-23",
            mount="losses.reconstruction",
            posture="raise",
            reason=(
                "Overloaded key with TWO successors -- pick by meaning. For the "
                "VGG content/perceptual weight use "
                "`losses.reconstruction.lambda_perceptual` (the registry aliases "
                "`content` -> `perceptual`, and this is what 100% of measured "
                "corpus usage meant). For the DISENTANGLEMENT content-consistency "
                "term use `losses.reconstruction.lambda_content_consistency`, "
                "which canonicalises to itself and so no longer collides. The one "
                "corpus arm that declared this "
                "(`kspace_filling/experiment_11_kspace_cold_diffusion_perceptual"
                ".yaml`) already carried `lambda_perceptual: 0.1` at the identical "
                "value, so its migration was a pure line deletion with a provably "
                "unchanged weight table (#421)."
            ),
        ),
        RenameRecord(
            legacy="losses.reconstruction.enable_content",
            canonical="losses.reconstruction.enable_content_consistency",
            since="2026-08-23",
            mount="losses.reconstruction",
            posture="raise",
            reason=(
                "Gate renamed alongside `lambda_content` -> "
                "`lambda_content_consistency`, so the pair states which of the "
                "two losses it gates. Zero readers, zero corpus declarations "
                "(#421)."
            ),
        ),
        RenameRecord(
            legacy="training.quality_matching.use_acquisition_prior",
            canonical="training.quality_matching.enable_acquisition_prior",
            since="2026-10-25",
            reason="Boolean switches are `enable_<thing>` (test_naming_convention).",
            mount="training.quality_matching",
        ),
    ]
)


def renames_for_block(
    block: str,
    table: dict[str, RenameRecord] | None = None,
    posture: Posture | None = None,
) -> dict[str, RenameRecord]:
    """Every record whose legacy key lived under ``block``.

    ``posture`` narrows to one kind; the two validators below each take only
    the records they know how to handle, so a record can never be silently
    served by the wrong one.
    """
    src = RENAMES if table is None else table
    return {
        rec.legacy_key: rec
        for rec in src.values()
        if rec.mount_path == block and (posture is None or rec.posture == posture)
    }


def folded_input_keys(block: str, table: dict[str, RenameRecord] | None = None) -> frozenset[str]:
    """Legacy leaf names ``block`` still accepts as INPUT and folds into place.

    Published by the owning schema as ``__folded_input_keys__`` so the execution
    ledger can tell a *moved* key from a *dropped* one without importing this
    module: ``core/`` and ``config/`` are peers at the rightmost layer, and the
    ledger scanning a rename table would couple them for a fact the class can
    simply carry.

    Semantically these are input-only aliases, which is why the ledger folds
    them into the same set it already builds from ``FieldInfo.alias``.
    """
    return frozenset(renames_for_block(block, table, posture="fold"))


def folded_input_paths(
    block: str, table: dict[str, RenameRecord] | None = None
) -> dict[str, tuple[str, ...]]:
    """Legacy leaf name -> the canonical attribute chain it folds into.

    The companion to :func:`folded_input_keys`, published alongside it as
    ``__folded_input_paths__`` for the same reason: ``core/`` and ``config/`` are
    peers, so the class carries the fact rather than the ledger importing this
    module.

    Knowing a key was *moved* is enough to stop reporting it as dropped, which is
    all the set was ever needed for. It is NOT enough to keep auditing what is
    inside it. ``acceleration:`` is a whole top-level block folded to
    ``undersampling:``, so with only the set the ledger accepted the key and then
    never descended into it, and every ``extra="ignore"`` drop beneath became
    invisible -- including a British-spelling ``min_centre_fraction`` that four
    independent tests were written to catch.

    The chain is returned relative to ``block`` and may be any depth, for the
    same reason ``flat_to_canonical`` refuses a depth cap: a truncated path does
    not fail loudly, it silently walks to the wrong object.
    """
    prefix = f"{block}." if block else ""
    out: dict[str, tuple[str, ...]] = {}
    for legacy, rec in renames_for_block(block, table, posture="fold").items():
        canonical = rec.canonical
        if prefix and canonical.startswith(prefix):
            canonical = canonical[len(prefix) :]
        out[legacy] = tuple(canonical.split("."))
    return out


def canonical_override_path(dotted: str, table: dict[str, RenameRecord] | None = None) -> str:
    """Translate a legacy ``--override`` path to its canonical spelling.

    The fourth consumer of the table, and the one whose absence made the whole
    fold posture unsound. :func:`apply_overrides` re-validates a **complete**
    ``model_dump()`` (deliberately not ``exclude_unset`` — pitfall #15c), so the
    dict it hands back already carries the canonical path with its default in
    place. Writing the legacy spelling next to it therefore presented the fold
    validator with two spellings that disagree, and it did the correct thing for
    a YAML document and the wrong thing here: it raised. That took out
    ``--override optimization.learning_rate=1e-4`` — this module's own docstring
    example — and ``data.max_train_subjects``, which every cluster smoke run
    passes (``scripts/ci/smoke_test_vf_configs.sh``).

    Translating the path *before* the write means the override lands on the
    canonical key and simply replaces the dumped default. A genuine
    double-declaration in YAML still raises, because there both spellings are in
    the source document.

    Only ``fold`` records translate. A ``raise`` record falls through unchanged
    so the owning block's :func:`reject_renamed_keys` validator produces its
    message, which names the replacement — a silent rewrite would rob the user
    of the one thing that tells them the key is gone.
    """
    src = RENAMES if table is None else table
    rec = src.get(dotted)
    if rec is None or rec.posture != "fold":
        return dotted
    if rec.value_transform != "identity":
        # No such record exists today. If one is ever added, translating the key
        # while leaving the literal alone would invert the user's intent in
        # silence, which is precisely the failure this table exists to prevent.
        raise ValueError(f"`{dotted}` cannot be used with --override: {rec.message()}")
    return rec.canonical


UNDECLARED = object()
"""Sentinel: neither spelling of a knob is present in the raw block.

Distinct from a declared ``None`` -- ``svd_calibration_lines: null`` means "use
the full FoV", which is a value, not an absence.
"""


def declared_knob(
    raw_block: Any,
    block: str,
    legacy_leaf: str,
    table: dict[str, RenameRecord] | None = None,
) -> Any:
    """The value a raw block declares for a knob, under EITHER spelling.

    Returns :data:`UNDECLARED` when neither spelling is present — distinct from
    a declared ``None``, which several of these knobs use as a real value
    (``svd_calibration_lines: null`` means "full FoV", not "unset").

    A raw block is the parsed YAML, so anything present in it was *authored*;
    schema defaults are not here yet. That is what lets a caller tell a genuine
    user contradiction from a knob it may freely seed.
    """
    if not isinstance(raw_block, MutableMapping):
        return UNDECLARED
    if legacy_leaf in raw_block:
        return raw_block[legacy_leaf]
    node: Any = raw_block
    for part in _canonical_tail(block, legacy_leaf, table):
        if not isinstance(node, MutableMapping) or part not in node:
            return UNDECLARED
        node = node[part]
    return node


def _canonical_tail(
    block: str, legacy_leaf: str, table: dict[str, RenameRecord] | None = None
) -> list[str]:
    """Path segments BELOW ``block`` where ``legacy_leaf`` now lives.

    Posture is deliberately NOT consulted. A promotion retires a *spelling*, not
    a destination: ``data.slice_2d`` raising does not move the live field, which
    is still ``data.sampling.enable_slice_2d``. All three callers want where the
    knob LIVES, and for a ``raise`` record that is still ``canonical``.

    This used to read ``rec.posture != "fold"``, which reverted all three to the
    flat spelling the moment a record was promoted -- inverting each of their
    documented contracts at once:

    * :func:`default_knob` promises to "seed a preset default at the CANONICAL
      path"; it went back to writing the legacy one. That is verbatim the
      collision its own docstring records being written to fix, except the
      legacy key now *raises* rather than merely disagreeing -- so a preset
      would hand the user an unloadable arm naming a key the user never wrote.
    * :func:`override_knob` calls clearing the legacy spelling "the load-bearing
      half"; it popped the legacy key and then wrote it straight back.
    * :func:`declared_knob` stopped being able to SEE a canonical declaration,
      returning ``UNDECLARED`` for a value sitting right there -- a silent wrong
      answer, and the one of the three with no fail-loud story at all.

    Found on 2026-08-18 by the first ``_DRAINED`` promotion big enough to include
    a leaf a production caller writes (``data.svd_calibration_lines``, via
    ``loader._sync_coil_processing_to_legacy``). Every later promotion of a
    knob-helper leaf -- ``coil_processing_mode``, ``num_virtual_coils``,
    ``normalize_kspace``, ``kspace_percentile`` -- would have hit the same wall.

    A user's own retired declaration is unaffected: :func:`reject_renamed_keys`
    reads the raw block directly, and :func:`declared_knob` checks the legacy key
    *before* consulting this function, so the rejection still fires.

    The guard that remains is a same-block one. ``training.seed -> run.seed``
    moves to a different top-level block, so its tail is not this block's to
    write; such records keep the flat leaf and let the rejection validator speak.
    A canonical with no tail at all is treated the same way, rather than handing
    the writers an empty path to ``IndexError`` on.
    """
    rec = (RENAMES if table is None else table).get(f"{block}.{legacy_leaf}")
    if rec is None:
        return [legacy_leaf]
    head, _, tail = rec.canonical.partition(".")
    if head != block or not tail:
        return [legacy_leaf]
    return tail.split(".")


def default_knob(
    raw_block: Any,
    block: str,
    legacy_leaf: str,
    value: Any,
    table: dict[str, RenameRecord] | None = None,
) -> None:
    """Seed a preset default at the CANONICAL path, unless already declared.

    The fifth consumer, and the one a schema needs when it injects presets of
    its own. ``DataConfigSchema`` seeds four M4Raw defaults and
    ``_drive_legacy_coil_knobs`` bridges ``physics.coil_processing`` into
    ``data:``; both wrote the LEGACY spelling and both guarded with
    ``if key not in data``, which cannot see the canonical one. So an arm that
    had been migrated to ``data.coils.processing_mode: rss`` got ``svd`` written
    beside it and failed to load with "two spellings disagree" — the migration
    script's own output, rejected by the schema. Same for
    ``processing.enable_kspace_normalization`` and ``processing.kspace_percentile``.

    Writing canonical removes the collision class rather than dodging it: the
    fold validator then has nothing to reconcile, whichever spelling the user
    chose. The declared-check still consults BOTH, because a user's legacy
    declaration must keep winning over a preset.
    """
    if not isinstance(raw_block, MutableMapping):
        return
    if declared_knob(raw_block, block, legacy_leaf, table) is not UNDECLARED:
        return  # authored under either spelling; the fold validator sorts it out
    path = _canonical_tail(block, legacy_leaf, table)
    node: Any = raw_block
    for part in path[:-1]:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        if not isinstance(child, MutableMapping):
            return  # a non-dict sub-block is the user's; never overwrite it
        node = child
    node.setdefault(path[-1], value)


def override_knob(
    raw_block: Any,
    block: str,
    legacy_leaf: str,
    value: Any,
    table: dict[str, RenameRecord] | None = None,
) -> Any:
    """Force a DERIVED value onto the canonical path; return what it displaced.

    :func:`default_knob` defers to whatever the user authored, which is right for
    a preset. A *derivation* is the opposite contract: ``_sync_coil_processing_to_legacy``
    documents that "an explicit new block wins over a conflicting legacy mode",
    so ``physics.coil_processing.compression.method: svd`` must reach
    ``data.coils.processing_mode`` even when ``data:`` says otherwise.

    Deferring instead would be a facade (pitfall #16): the v6.1 reference
    template authors ``data.coils.processing_mode: 'none'``, so an arm adopting
    the unified physics block on top of it would have that block silently
    ignored.

    Clearing the legacy spelling is the load-bearing half. Writing canonical
    while a legacy key remains beside it hands the fold validator two spellings
    that disagree, which is exactly how this bridge broke.
    """
    previous = declared_knob(raw_block, block, legacy_leaf, table)
    if not isinstance(raw_block, MutableMapping):
        return previous
    raw_block.pop(legacy_leaf, None)
    path = _canonical_tail(block, legacy_leaf, table)
    node: Any = raw_block
    for part in path[:-1]:
        child = node.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            node[part] = child
        node = child
    node[path[-1]] = value
    return previous


def reject_renamed_keys(
    block: str,
    table: dict[str, RenameRecord] | None = None,
    removals: dict[str, Any] | None = None,
) -> Any:
    """Build a ``mode="before"`` validator that raises on any retired key.

    ``table`` is injectable so the mechanism can be unit-tested without adding a
    real rename to the production surface.

    Raising here rather than relying on ``extra="forbid"`` is the whole point:
    most blocks are ``extra="ignore"``, where a removed key is silently
    swallowed, and even under ``forbid`` the stock message ("extra fields not
    permitted") does not tell the author what to write instead.

    Only ``posture="raise"`` records are handled; see :func:`fold_renamed_keys`.
    Keys removed outright (:mod:`removals`) are refused by the same validator, so
    one mount serves both. An injected ``table`` stands alone unless ``removals``
    is injected too.
    """
    from .removals import REMOVALS, removals_for_block

    if removals is None:
        removals = REMOVALS if table is None else {}
    retired: dict[str, Any] = {
        **renames_for_block(block, table, posture="raise"),
        **removals_for_block(block, removals),
    }

    def _validator(_cls: type, data: Any) -> Any:
        if not isinstance(data, dict) or not retired:
            return data
        hits = [retired[k] for k in data if k in retired]
        if hits:
            raise ValueError(
                "\n".join(rec.message() for rec in sorted(hits, key=lambda r: r.legacy))
            )
        return data

    # The block is NOT recoverable from the closure: `_validator` captures only
    # the RESOLVED record dict, so a mount serving zero records is
    # indistinguishable at runtime from no mount at all -- which is exactly the
    # defect `rename_mounts.audit_mounts` exists to find. Stamp it.
    _validator.__dict__["__rename_block__"] = block
    _validator.__dict__["__rename_posture__"] = "raise"
    return _validator


def fold_renamed_keys(block: str, table: dict[str, RenameRecord] | None = None) -> Any:
    """Build a ``mode="before"`` validator that MOVES staged keys into place.

    The staged counterpart of :func:`reject_renamed_keys`. A ``fold`` record's
    legacy spelling keeps working: the value is lifted to its canonical path
    inside the same block, creating intermediate mappings as needed, and the
    legacy key is dropped before the model is built. Nothing is added to the
    model, so the legacy spelling is unreachable from Python — the point of the
    exercise is one read path, not one write path.

    Refuses rather than guesses, matching the fixer
    (``scripts/ci/migrate_config_keys.py``): if both spellings are present and
    **disagree**, the arm records two intentions and only a human can say which
    it meant. Agreeing duplicates drop the legacy key silently.

    Mounted at :data:`ROOT` it renames a whole top-level BLOCK: the value it
    moves is the block's entire mapping, and the destination is the full
    canonical path rather than a path relative to a block. That is the only
    place a fold may cross blocks — a per-block validator is mounted on one
    class and cannot reach a sibling, but the root one is mounted on
    ``TrainingSettings`` and can reach every top-level block. Pydantic runs the
    root ``mode="before"`` validator before any sub-model is constructed, so the
    renamed block is built from the destination field, never the retired one.

    A legacy leaf may **not** share its name with a destination sub-block. The
    fold matches on the key alone, so if ``validation.metrics: [psnr]`` folded
    into ``validation.metrics.compute``, the key ``metrics`` would mean both the
    retired scalar and the block four siblings fold into — and the two are
    indistinguishable. Legacy-only arms would break on declaration order (a
    sibling declared above ``metrics`` descends into a *list*), and a MIGRATED
    arm writing ``metrics: {compute: [...]}`` would have that whole block folded
    into itself. Pinned by
    ``test_renames.py::TestNoLegacyLeafNamesADestinationBlock``, which is why
    the phase-10 block is ``scoring`` rather than ``metrics``.
    """
    foldable = renames_for_block(block, table, posture="fold")

    def _validator(_cls: type, data: Any) -> Any:
        if not isinstance(data, dict) or not foldable:
            return data
        hits = [k for k in data if k in foldable]
        if not hits:
            return data

        out = dict(data)
        for key in hits:
            rec = foldable[key]
            value = out.pop(key)
            if rec.value_transform == "negate":
                if not isinstance(value, bool):
                    raise ValueError(
                        f"{rec.legacy} inverts to {rec.canonical}, so its value "
                        f"must be a bool; got {value!r}"
                    )
                value = not value

            # Copy-on-write down to the leaf: the caller's nested dicts are
            # shared with whatever else holds this document and must not be
            # mutated underneath it.
            #
            # `canonical` is absolute, but the cursor starts INSIDE the mounted
            # block, so the mount's own components are already consumed and must
            # be stripped -- EXCEPT at the ROOT, where the validator is mounted
            # on `TrainingSettings` and the whole path is the destination.
            # Dropping a component there would strip the destination block name
            # and fold `acceleration` -> `undersampling` into nothing.
            #
            # Strip by the mount's DEPTH, not one component: a nested mount like
            # `training.diffusion` consumes two. Hardcoding 1 here is what made a
            # nested record inexpressible -- it would have written the value to
            # `training.diffusion.diffusion.<leaf>`.
            cursor = out
            depth = 0 if block == ROOT else len(block.split("."))
            parts = rec.canonical.split(".")[depth:]
            for part in parts[:-1]:
                existing = cursor.get(part)
                if existing is None:
                    cursor[part] = {}
                elif isinstance(existing, dict):
                    cursor[part] = dict(existing)
                else:
                    raise ValueError(
                        f"cannot fold `{rec.legacy}` into `{rec.canonical}`: "
                        f"`{part}` is already {type(existing).__name__}, not a block"
                    )
                cursor = cursor[part]

            leaf = parts[-1]
            if leaf in cursor and cursor[leaf] != value:
                raise ValueError(
                    f"`{rec.legacy}`={value!r} and `{rec.canonical}`="
                    f"{cursor[leaf]!r} disagree. {rec.reason} Declare one of "
                    "them.\nFix automatically:  python "
                    "scripts/ci/migrate_config_keys.py <file>"
                )
            cursor[leaf] = value
            _record_fold(rec, value)
        return out

    # Mirrors :func:`reject_renamed_keys`: the closure carries `block`, but
    # relying on a freevar NAME is exactly the source-text coupling this
    # stamp removes. Both builders advertise themselves the same way.
    _validator.__dict__["__rename_block__"] = block
    _validator.__dict__["__rename_posture__"] = "fold"
    return _validator


def _record_fold(rec: RenameRecord, value: Any) -> None:
    """Note in the execution ledger that ``rec.legacy`` was moved.

    A fold is a silent substitution, and uniquely one the ledger's own walker
    cannot find on its own: ``diff_declared_vs_resolved`` descends into
    ``raw_keys & resolved_keys``, but the legacy key is popped here and its
    value re-emerges at a path the document never named. Without this record,
    ``losses.output_domain: kspace`` and ``losses: {policy: {output_domain:
    kspace}}`` resolve to byte-identical documents — so nothing downstream can
    tell a migrated arm from an unmigrated one.

    ``config/settings.py::_bind_config_version`` already does exactly this for
    the schema VERSION, for exactly this reason. The key fold has the same shape
    and recorded nothing, across a corpus that folds ~29,250 declarations.

    Best-effort and lazily imported: config loading must not depend on a ledger
    being armed, and ``config/`` must not gain an import-time edge to ``core/``
    (they are peers at the rightmost layer — see :func:`folded_input_keys`).
    """
    try:
        from spectramr.core.execution_ledger import ExecutionLedger, SubstitutionClass

        ledger = ExecutionLedger.current()
        if ledger is None:
            return
        ledger.record(
            class_id=SubstitutionClass.VALUE_CHANGED_ON_FINALIZE,
            site="config.schemas.renames.fold_renamed_keys",
            stage="parse",
            path=rec.legacy,
            requested=value,
            # The destination PATH, not the value: the value is unchanged (that
            # is what makes a fold invisible), so recording it on both sides
            # would say nothing. Where it went is the whole content.
            resolved=rec.canonical,
            reason=(
                f"`{rec.legacy}` is a retired spelling folded to "
                f"`{rec.canonical}` at parse time ({rec.since}). The resolved "
                "document is identical either way, so only this record "
                "distinguishes a migrated arm from an unmigrated one."
            ),
            severity="warning",
        )
    except Exception:  # pragma: no cover - recording must never break a load
        return


__all__ = [
    "RENAMES",
    "ROOT",
    "Posture",
    "RenameRecord",
    "ValueTransform",
    "canonical_override_path",
    "fold_renamed_keys",
    "folded_input_keys",
    "reject_renamed_keys",
    "renames_for_block",
]
