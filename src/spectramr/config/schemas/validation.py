"""Validation configuration schema."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .renames import (
    RENAMES,
    fold_renamed_keys,
    folded_input_keys,
    folded_input_paths,
    reject_renamed_keys,
)

__all__ = [
    "HallucinationTestConfig",
    "ValidationCascadeConfigSchema",
    "ValidationConfigSchema",
    "ValidationGatesConfigSchema",
    "ValidationLoaderConfigSchema",
    "ValidationSamplingConfigSchema",
    "ValidationScheduleConfigSchema",
    "ValidationScoringConfigSchema",
    "ValidationVisualizationConfigSchema",
]

#: New sub-blocks are born strict. The parent stays `forbid` too, so an
#: unmigrated key that the fold table forgot RAISES rather than vanishing —
#: the opposite of `data:`, where `extra="ignore"` silently ate one (#550).
_VAL_SUBBLOCK = ConfigDict(extra="forbid", frozen=True)


_UNREAD_NUM_VIS = (
    "NOT READ — nothing in src/ consumes it (KNOWN_UNCONSUMED ledger, "
    "tests/unit/config/test_schema_key_consumption.py). 209 arms declare it "
    'and this block is extra="forbid", so deleting the field would fail every one of them at load. '
    "Wire it or retire it with a corpus migration; do not trust it today."
)
_UNREAD_VIS_DIR = (
    "NOT READ — nothing in src/ consumes it (KNOWN_UNCONSUMED ledger, "
    "tests/unit/config/test_schema_key_consumption.py). 60 arms declare it "
    'and this block is extra="forbid", so deleting the field would fail every one of them at load. '
    "Wire it or retire it with a corpus migration; do not trust it today."
)
_UNREAD_USE_TRAIN_LOSS = (
    "NOT READ — nothing in src/ consumes it (KNOWN_UNCONSUMED ledger, "
    "tests/unit/config/test_schema_key_consumption.py). 136 arms declare it "
    'and this block is extra="forbid", so deleting the field would fail every one of them at load. '
    "Wire it or retire it with a corpus migration; do not trust it today."
)


class HallucinationTestConfig(BaseModel):
    """Hallucination test stub (CLAUDE.md #10 — clinical defensibility).

    Used by the cold-diffusion locus-ablation campaign and any other
    image-prior paradigm that risks hallucinated structure. The
    feature-insertion sweep adds N synthetic features to the input
    and checks whether the reconstruction faithfully preserves them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(default=False)
    method: Literal["feature_insertion", "lesion_swap", "edge_jitter"] = Field(
        default="feature_insertion"
    )
    n_features: int = Field(default=5, ge=1)
    interval_validations: int = Field(default=4, ge=1)


class ValidationScheduleConfigSchema(BaseModel):
    """*How often* validation runs.

    Four spellings arrived here for one cadence. ``eval_interval`` and
    ``frequency_steps`` are the same number with the same default (1000): 58
    arms declare both and **not one of them disagrees**, which is what makes
    merging them safe rather than a guess. Only ``eval_interval`` was ever read
    (`pipelines/training_loop.py`), so the merge also turns 58 arms' dead
    ``frequency_steps`` into a knob that finally reaches the loop.

    ``on_epoch`` and ``interval_epochs`` are NOT duplicates and both survive:
    the first picks the epoch-based mode, the second is its N.
    """

    model_config = _VAL_SUBBLOCK

    interval_steps: int = Field(
        default=1000,
        ge=1,
        description=(
            "Run validation every N training steps (step-based mode). Must not "
            "exceed `training.max_iterations`: the loop gate is a bare "
            "`iteration % interval_steps == 0` with no first/last-iteration "
            "force, so an interval above the budget yields ZERO validation "
            "events -- early stopping never evaluates, no `checkpoint_best.pt` "
            "is written, and the run still exits reporting success. "
            "`_execute_training_loop` rejects that combination at startup "
            "(`ConfigurationError`) unless `on_epoch` supplies events instead. "
            "`ge=1` cannot express this: the budget is a field in another block "
            "and can be epoch-derived at runtime, so the check is a runtime "
            "guard rather than a validator. Watch it when shortening a run with "
            "`-O training.max_iterations=...` -- the override moves the "
            "budget under a fixed interval."
        ),
    )
    on_epoch: bool = Field(
        default=False,
        description=(
            "ADDITIONALLY run validation at each epoch boundary, every "
            "`interval_epochs` epochs. Additive: it does not replace "
            "`interval_steps`, so enabling it can only add validation events, "
            "never remove the one an arm selects its checkpoint from."
        ),
    )
    interval_epochs: int = Field(
        default=1,
        ge=1,
        description=(
            "Run validation every N epochs. Only consulted in epoch-based mode; "
            "`IterationCounterService.should_validate` takes it as its "
            "`frequency_epochs` argument."
        ),
    )


class ValidationLoaderConfigSchema(BaseModel):
    """How validation batches are drawn.

    ``batch_size`` absorbs BOTH prior spellings. They were not interchangeable:
    ``effective_val_batch_size`` and `data_builder.py` preferred the short
    ``val_batch_size``, while `training_pipeline_director.py:690` read only the
    long ``validation_batch_size`` — and **74 arms declare the two with
    different values**, so which batch size a run used depended on which builder
    path it took. One field ends that by construction.

    The short form wins, matching the precedence both field descriptions
    already documented; `_resolve_batch_size_duplicate` logs whenever it
    displaces a disagreeing long form.
    """

    model_config = _VAL_SUBBLOCK

    batch_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Validation batch size; None inherits the training batch size. "
            "Set to 1 or 2 to prevent validation OOM when the training batch is "
            "large or validation uses 5D->4D slice flattening (see "
            "docs/superpowers/specs/2026-05-04-smoke-test-postmortem-fix-spec.md)."
        ),
    )
    chunk_size: int = Field(
        default=2,
        ge=1,
        description=(
            "Micro-batch size for chunked validation inference. Reduces peak GPU "
            "memory when validation batches are large (e.g. 18 slices from a "
            "5D->4D flatten)."
        ),
    )
    num_batches: int | None = Field(
        default=None,
        description="Number of batches to validate on (None = all), for speed.",
    )
    num_samples: int | None = Field(
        default=None,
        description="Number of validation samples (None = all).",
    )
    num_workers: int = Field(
        default=0,
        ge=0,
        description=(
            "Worker processes for the VALIDATION loader only; training keeps its "
            "own `data.loader.num_workers` fan-out. The default 0 (load in the "
            "main process) is not a placeholder -- it is the fix for a "
            "cohort-wide cgroup host-RAM oom_kill at the first validation "
            "(2026-07). Where a validation sample is a whole-volume decode "
            "(~226 MB), N workers decode N SEPARATE volumes in parallel at "
            "spawn time, on top of the still-resident persistent TRAIN workers, "
            "so peak host RAM jumps by N whole volumes before a single metric is "
            "written. `num_batches` caps the loop, not this spawn-time fan-out, "
            "which is why capping it did not help. Raising this is only safe "
            "where validation samples do NOT share an expensive decode (e.g. "
            "`npy_slice`, whose samples are already independent 2-D files); the "
            "loader builder RAISES rather than risking the oom_kill when it is "
            "nonzero on a volume-backed validation set."
        ),
    )
    shuffle: bool = Field(
        default=False,
        description="Randomly shuffle validation data during evaluation.",
    )


class ValidationScoringConfigSchema(BaseModel):
    """*What* validation measures.

    The block is ``scoring``, not ``metrics``, and that is load-bearing rather
    than taste. The retired scalar is ``validation.metrics``, and the fold
    matches on the key alone — so a block named ``metrics`` would make the key
    mean both the old scalar and its own destination. A legacy-only arm would
    then break on declaration order, and a MIGRATED arm writing
    ``metrics: {compute: [...]}`` would have that block folded into itself.
    ``TestNoLegacyLeafNamesADestinationBlock`` pins the rule.

    The list leaf is ``compute`` for a plainer reason: it is the spelling the
    standing ``metrics.compute_*`` -> ``metrics.compute: [psnr, ssim]``
    migration already uses, so the two surfaces read the same way.
    """

    model_config = _VAL_SUBBLOCK

    compute: list[str] | dict[str, bool] | None = Field(
        default=None,
        description=(
            "Metrics to compute during validation; None inherits "
            "`training.metrics`. A list of registry names — ['psnr', 'ssim', "
            "'lpips'] — is the only shape any arm uses; the dict form "
            "{'psnr': True} is accepted for back compatibility."
        ),
    )
    primary: str = Field(
        default="psnr",
        description="Primary metric for early stopping and model selection.",
    )
    domain: str | None = Field(
        default=None,
        description=(
            "Domain for validation metrics: 'image', 'kspace', or None to "
            "auto-detect from the physics config / inherit `training.metrics.domain`."
        ),
    )
    output_transform: str | None = Field(
        default=None,
        description=(
            "Transform applied before validation metrics: 'ifft_magnitude', "
            "'ifft_sense_adjoint', 'magnitude', or None. Any other name RAISES "
            "— 'fft' and 'ifft_mag_combine' were advertised here and dispatched "
            "by nothing. This is the knob the VALIDATION path reads; "
            "`metrics.transform` is read by the training-metrics path only. "
            "SSOT: infrastructure/training/utils/metric_transform.py."
        ),
    )
    enable_image_metrics: bool = Field(
        default=True,
        description=(
            "Compute image-quality metrics (PSNR, SSIM, MAE) in addition to the validation loss."
        ),
    )


class ValidationVisualizationConfigSchema(BaseModel):
    """Validation image dumps.

    Only the gate and its cadence live here. ``num_visualizations`` and
    ``visualization_dir`` stay flat on the parent because nothing reads them —
    see the parent's inert-knob note.
    """

    model_config = _VAL_SUBBLOCK

    enabled: bool = Field(
        default=False,
        description="Save validation visualizations.",
    )
    interval: int = Field(
        default=1000,
        ge=1,
        description="Save visualizations every N steps.",
    )


class ValidationSamplingConfigSchema(BaseModel):
    """Reverse-diffusion sampling *at validation time*, where it differs from training."""

    model_config = _VAL_SUBBLOCK

    steps: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Reverse-diffusion steps used at validation. Set equal to the "
            "deployment sampler steps so the validation curve reflects "
            "deployment-time performance (plan §15 / A5.6). None falls back to "
            "`training.diffusion.sampling_steps`."
        ),
    )
    enable_multistep_cold: bool = Field(
        default=False,
        description=(
            "Cold-diffusion validation/inference: reconstruct via the genuine "
            "multi-step reverse restoration (the registered `cold_mri` sampler / "
            "`PhysicsInformedColdDiffusion.sample`: x_T=measurement -> ... -> "
            "x_0, enforcing data consistency every step) instead of a single "
            "deterministic forward pass. The single forward is a one-shot x0 "
            "regressor that converges to the posterior mean (blur/blank) at heavy "
            "undersampling; iterative sampling builds structure progressively. "
            "WARNING (pitfall #18): this changes what the validation metric "
            "MEASURES, so early-stopping/best-metric thresholds must be "
            "re-baselined, and it is ~steps x slower. Only the "
            "`kspace_cold_diffusion` cold branch honours it."
        ),
    )
    ensemble_samples: int = Field(
        default=1,
        ge=1,
        le=4,
        description=(
            "Stochastic cold-diffusion reverse samples drawn per validation input "
            "when `enable_multistep_cold` is on. 1 (default) is the single "
            "reconstruction every arm validated with before this knob. N > 1 runs "
            "the multi-step sampler N times with distinct C6 noise streams (seed "
            "offsets 0..N-1 on the sampler_seed), grades the pixelwise mean, and "
            "emits `val_ensemble_std_mean` and `val_empirical_coverage` per rung. "
            "Needs `model.model_kwargs.sampler_sigma > 0`: at sigma 0 every member "
            "is identical and the load is refused. Capped at 4 because validation "
            "costs N x the multistep pass, per rung."
        ),
    )
    coverage_k: float = Field(
        default=2.0,
        gt=0,
        description=(
            "Half-width, in ensemble standard deviations, of the per-pixel "
            "interval `mean +/- k * std` that `val_empirical_coverage` counts "
            "target pixels inside. Read only when `ensemble_samples > 1`."
        ),
    )

    reveal_attribution: bool = Field(
        default=False,
        description=(
            "Attribute the reconstruction error to the reverse step that wrote "
            "each coefficient, and emit one `val_band_*` column per reveal band "
            "(`models/diffusion/reveal_attribution.py`). The freeze loops write "
            "every unobserved coefficient exactly once, so the final k-space "
            "partitions by writing step; each band reports its complex gain, "
            "whose MODULUS is amplitude shrinkage and whose ARGUMENT is the "
            "systematic phase error that displaces the band along the "
            "phase-encode axis. An ORACLE diagnostic -- it reads the target, so "
            "it measures and never steers. Needs `enable_multistep_cold` (a "
            "single forward has no reveal schedule to partition by) and a "
            "line-structured mask; a point pattern or non-Cartesian trajectory "
            "RAISES rather than reporting a band that is not one."
        ),
    )

    @model_validator(mode="after")
    def _reveal_attribution_needs_the_multistep_sampler(self) -> "ValidationSamplingConfigSchema":
        """The partition is over reveal steps, which a single forward does not have.

        Refused at load rather than emitting empty columns: an all-``nan``
        `val_band_*` block reads as "the bands are unmeasurable", not as "the
        arm asked for a diagnostic that cannot apply to it".
        """
        if self.reveal_attribution and not self.enable_multistep_cold:
            raise ValueError(
                "validation.sampling.reveal_attribution partitions the reconstruction "
                "by the reverse step that wrote each coefficient, which exists only "
                "under the multi-step cold sampler: set enable_multistep_cold: true, "
                "or leave reveal_attribution at false."
            )
        return self

    @model_validator(mode="after")
    def _ensemble_needs_the_multistep_sampler(self) -> "ValidationSamplingConfigSchema":
        """N > 1 without the multi-step sampler would be N identical forwards.

        Only `PhysicsInformedColdDiffusion.sample` carries the C6 noise stream;
        the single-step forward is deterministic, so an ensemble on it is a
        facade. Refused at load rather than silently reported as std 0.
        """
        if self.ensemble_samples > 1 and not self.enable_multistep_cold:
            raise ValueError(
                f"validation.sampling.ensemble_samples={self.ensemble_samples} draws "
                "stochastic reverse samples, which only the multi-step cold sampler "
                "produces: set enable_multistep_cold: true, or leave ensemble_samples at 1."
            )
        return self


class ValidationGatesConfigSchema(BaseModel):
    """Degeneracy and defensibility gates — the checks that can FAIL a run.

    Distinct from `metrics`, which only measures: each of these exists to catch
    a specific way a run can score well while meaning nothing (pitfalls #16/#20).
    """

    model_config = _VAL_SUBBLOCK

    input_dependence_tol: float | None = Field(
        default=None,
        gt=0,
        description=(
            "L4 measurement-independence gate. During cascaded validation the "
            "strategy measures the spread of the prediction across the "
            "acceleration cascade (per-pixel structural std, or the per-level "
            "val_pred_mean_<R>x scalars as a fallback) via the "
            "`input_dependence_spread` metric. When that spread falls below this "
            "tolerance the run is flagged as collapsed "
            "(`val_measurement_collapse=1.0` + a warning the strict smoke audit "
            "catches), pinpointing the Experiment-11 'DC blob' where the output "
            "is a measurement-independent constant. None disables the gate; a "
            "positive float enables it (e.g. 0.01)."
        ),
    )
    held_out_severity_eval: bool = Field(
        default=False,
        description=(
            "Evaluate at the "
            "model.model_kwargs.digital_twin_kwargs.held_out_severity_grid points "
            "in addition to standard in-distribution validation. Cold-diffusion "
            "paradigm pitfall guard (plan §12 / A5.8)."
        ),
    )
    hallucination_test: HallucinationTestConfig = Field(
        default_factory=HallucinationTestConfig,
        description=(
            "Hallucination test sweep — clinical-defensibility hook "
            "(plan §13 / A5.9). Disabled by default."
        ),
    )


class ValidationCascadeConfigSchema(BaseModel):
    """Which acceleration rungs the validation cascade evaluates.

    The ladder used to be a module constant read straight out of
    ``core.cascading_validation`` by the strategy — so an arm could set
    ``undersampling.acceleration_range`` freely for *training* while
    *validation* stayed pinned at 2/8/32 with no way to say otherwise. Training
    already draws its severities from config; this is the same knob for the
    other side of the loop.
    """

    model_config = _VAL_SUBBLOCK

    levels: list[float] | None = Field(
        default=None,
        description=(
            "In-distribution acceleration rungs for the cascading validation "
            "sweep, e.g. [2, 4, 8]. Each rung is evaluated in its own pass and "
            "written as one row of validation_metrics.csv (columns "
            "`acceleration_level` / `timestep`) plus the flat "
            "`val_<metric>_<R>x` names the L4 gate indexes by. Omit the key — "
            "or leave it null — for the default ladder (2, 8, 32), which is "
            "what every arm evaluated before this knob existed. Values are "
            "deduplicated and sorted ascending; the first rung must be the "
            "mildest because the accel-gap readout subtracts the last from the "
            "first. Under `undersampling.schedule_type: step` a rung that is "
            "not in `undersampling.acceleration_range` cannot be inverted to a "
            "timestep and is SKIPPED at runtime — `spectramr audit` warns about "
            "that before the launch (`validation_cascade_levels_in_range`)."
        ),
    )

    @field_validator("levels", mode="before")
    @classmethod
    def _reject_bool_levels(cls, value: Any) -> Any:
        """Refuse a boolean rung BEFORE Pydantic coerces it to a number.

        ``bool`` is a subclass of ``int``, so a ``list[float]`` field turns
        ``levels: [true]`` into ``[1.0]`` — a perfectly legal ladder that
        nobody asked for, evaluated and reported as though it were declared.
        That is the silent substitution non-negotiable 3 forbids, and it is
        also the one case where this field and
        ``normalize_cascade_levels`` (which rejects bools) would otherwise
        disagree about the same input. Numeric strings are still coerced, as
        on every sibling float field — the divergence being closed here is the
        one that changes the *value*, not the one that changes the spelling.
        """
        if isinstance(value, list) and any(isinstance(v, bool) for v in value):
            raise ValueError(
                "validation.cascade.levels contains a boolean. Accelerations "
                "are numbers; `true` would silently become R=1.0."
            )
        return value

    @field_validator("levels")
    @classmethod
    def _validate_levels(cls, value: list[float] | None) -> list[float] | None:
        """Reject an illegal ladder at load time, via the ONE owner of the rule.

        The check lives in ``core.cascading_validation.normalize_cascade_levels``
        and is merely *called* here (non-negotiable 17) — re-implementing "is
        this ladder legal" beside the runtime resolver is how the two come to
        disagree, and a disagreeing validator admits a ladder the strategy then
        silently reinterprets. Imported inside the function because every other
        ``config/`` → ``core/`` edge in this package is lazy for the same
        reason: schemas are imported early enough that a module-level edge
        risks a cycle.
        """
        if value is None:
            return None
        from spectramr.core.cascading_validation import normalize_cascade_levels

        return [float(level) for level in normalize_cascade_levels(value)]


class ValidationConfigSchema(BaseModel):
    """Validation dataset and evaluation configuration.

    Reads in the order a reader asks: *how often* (`schedule`), *on what*
    (`loader`), *measuring what* (`metrics`), *failing on what* (`gates`), plus
    `visualization` and diffusion-only `sampling`.

    Example:
        >>> config = ValidationConfigSchema(
        ...     enabled=True,
        ...     schedule={"interval_steps": 1000},
        ... )

    The flat spellings still LOAD — `fold_renamed_keys` moves each into its
    sub-block before validation — but they are gone from Python, so there is one
    read path and, temporarily, two accepted spellings in YAML.

    Nine scalars stay flat on purpose. **Nothing reads them**, and giving an
    inert knob a tidy home implies it works (the call `data.test_split` and
    `optimization.num_steps` already got). They stay visibly odd beside the
    grouped blocks until each is wired or deleted:

    * `enabled` — 1006 arms set it, 8 to `false`, and validation runs anyway.
      A comment in `metrics_report_generator.py` even attributes a missing CSV
      to `validation.enabled=false`, a cause that cannot occur. Issue #673.
    * `split` — 388 arms set it; the live validation fraction is
      `data.split.validation_fraction` (10+ readers). Issue #673.
    * `use_training_loss`, `enable_validation_augmentation`, `validation_dir`,
      `validation_metric`, `num_visualizations`, `visualization_dir` — already
      carried by `KNOWN_UNCONSUMED` in `test_schema_key_consumption.py`.
    * `empty_cache_before_validation` — the one LIVE ungrouped scalar; a
      memory-policy knob that belongs to no group here.

    `validation_metric` is deliberately NOT merged into `metrics.primary`
    despite naming the same idea: the defaults differ (`'loss'` vs `'psnr'`), so
    folding would silently repoint the 53 arms that say `validation_metric: loss`.
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    # ---- phase 10: grouped sub-blocks ------------------------------------
    schedule: ValidationScheduleConfigSchema = Field(default_factory=ValidationScheduleConfigSchema)
    loader: ValidationLoaderConfigSchema = Field(default_factory=ValidationLoaderConfigSchema)
    scoring: ValidationScoringConfigSchema = Field(default_factory=ValidationScoringConfigSchema)
    visualization: ValidationVisualizationConfigSchema = Field(
        default_factory=ValidationVisualizationConfigSchema
    )
    sampling: ValidationSamplingConfigSchema = Field(default_factory=ValidationSamplingConfigSchema)
    gates: ValidationGatesConfigSchema = Field(default_factory=ValidationGatesConfigSchema)
    cascade: ValidationCascadeConfigSchema = Field(default_factory=ValidationCascadeConfigSchema)

    __folded_input_keys__ = folded_input_keys("validation")
    __folded_input_paths__ = folded_input_paths("validation")

    # Retired flat spellings, BOTH postures. `reject_renamed_keys` raises on
    # the ones already driven to zero; `fold_renamed_keys` moves the rest.
    #
    # The reject half is not optional here even while the raise set is small.
    # This block is `extra="ignore"`, so a raise-posture record with nothing
    # to refuse it is not "retired" -- it is SILENTLY DROPPED, which the
    # renames module docstring calls strictly worse than leaving the fold in
    # place: the key stops working AND stops being visible. Promoting a
    # drained record in a block without this validator converts a working
    # fold into exactly that.
    _reject_renamed = model_validator(mode="before")(classmethod(reject_renamed_keys("validation")))
    _fold_renamed = model_validator(mode="before")(classmethod(fold_renamed_keys("validation")))

    # ---- ungrouped scalars ------------------------------------------------
    # See the class docstring: every one of these but `empty_cache_before_
    # validation` is inert, and flatness is the tell.
    enabled: bool = Field(
        default=True,
        description=(
            "Enable validation during training. NOT READ — validation runs "
            "regardless; the training loop gates only on the block's presence "
            "(issue #673)."
        ),
    )
    split: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description=(
            "Fraction of training data to use for validation. NOT READ — the "
            "live knob is `data.split.validation_fraction` (issue #673)."
        ),
    )
    validation_dir: str | None = Field(
        default=None,
        description="Separate directory for validation data (if None, use split).",
    )
    enable_validation_augmentation: bool = Field(
        default=False,
        description="Apply augmentation during validation.",
    )
    validation_metric: str = Field(
        default="loss",
        description=(
            "Metric to track during validation. NOT READ; near-duplicate of "
            "`metrics.primary`, but with a different default, so it was not "
            "merged into it."
        ),
    )
    use_training_loss: bool = Field(
        default=True,
        description=(
            "Use the same loss computation as training (includes L1, perceptual, "
            "LPIPS). If False, only compute MSE. " + _UNREAD_USE_TRAIN_LOSS
        ),
    )
    num_visualizations: int = Field(
        default=4,
        ge=1,
        description=("Number of samples to visualize. " + _UNREAD_NUM_VIS),
    )
    visualization_dir: str = Field(
        default="./visualizations",
        description=("Directory to save validation visualizations. " + _UNREAD_VIS_DIR),
    )
    empty_cache_before_validation: bool = Field(
        default=True,
        description=(
            "Call torch.cuda.empty_cache() before each validation pass to free "
            "the training allocator pool (training usually holds most of VRAM; "
            "the EMA weight-swap transiently doubles parameter memory). Default "
            "True preserves the OOM-safe behavior. Set False on memory-headroom "
            "runs to avoid the allocator re-grow cost on the next train step "
            "(backlog_wasted_compute_audit_2026_05_29 PIPE-2)."
        ),
    )

    # Defined AFTER `_fold_renamed` so it runs BEFORE it: pydantic runs
    # `mode="before"` validators in reverse definition order. The fold would
    # otherwise see two spellings of one leaf and refuse the arm.
    @model_validator(mode="before")
    @classmethod
    def _resolve_batch_size_duplicate(cls, data: Any) -> Any:
        """Collapse the two batch-size spellings before either is folded.

        `val_batch_size` and `validation_batch_size` both mean the validation
        batch size and 74 arms declare them with DIFFERENT values, so folding
        both onto `loader.batch_size` would raise "two spellings disagree" on
        every one of them. The short form wins — the precedence the retired
        `effective_val_batch_size` property and both field descriptions already
        documented — and the displaced value is logged, not swallowed.
        """
        if not isinstance(data, dict):
            return data
        # The precedence lives in the rename SSOT (`superseded_by`), not here.
        # The migrator reads the same field, so the fixer and this validator
        # cannot drift into disagreeing about which spelling wins -- which is the
        # two-resolvers defect the whole campaign exists to remove.
        rec = RENAMES.get("validation.validation_batch_size")
        loser = rec.legacy_key if rec is not None else "validation_batch_size"
        winner = (
            rec.superseded_by.rsplit(".", 1)[-1]
            if rec is not None and rec.superseded_by
            else "val_batch_size"
        )
        if data.get(winner) is None or loser not in data:
            return data

        out = dict(data)
        displaced = out.pop(loser)
        if displaced is not None and displaced != out[winner]:
            import logging

            logging.getLogger(__name__).info(
                "[VALIDATION] val_batch_size=%r wins over validation_batch_size=%r "
                "(the short form is the documented winner); both now fold to "
                "validation.loader.batch_size",
                out[winner],
                displaced,
            )
        return out
