"""Base schemas and shared components for configuration.

This module contains base classes, metadata schemas, and shared objective configurations
that are used across the configuration system.
"""

# v6.0: No longer need warnings import - objectives section raises errors not warnings
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .renames import reject_renamed_keys

#: The one version a config should declare. Everything that *writes* a version —
#: the migrator, the defaults provider, the reference template — writes this.
#:
#: The number restarts at 1.0 deliberately. The 5.x/6.x lineage numbered a schema
#: that no longer exists: the 15-phase decomposition (PR #644) rebuilt the block
#: layout underneath those names, and nothing ever branched on the version anyway
#: — it is validated, bound onto ``run.config_version``, and read by no consumer.
#: A version that gates nothing but keeps incrementing is a changelog pretending
#: to be a contract. 1.0 names the schema as it now is, once.
CANONICAL_CONFIG_VERSION: str = "1.0"

#: **Empty. The retirement trigger fired on 2026-08-08.**
#:
#: This held ``{"6.0", "6.1"}``: versions accepted at the door but *folded* to
#: :data:`CANONICAL_CONFIG_VERSION` before anything downstream saw them. The
#: fold made the migration gradual — a hard flip to 1.0-only would have broken
#: hundreds of arms in one commit — and its documented end condition was the
#: corpus reaching zero declarations, at which point this set empties and the
#: fold in ``config/settings.py::_bind_config_version`` is deleted.
#:
#: What actually happened at the end is worth recording, because "zero" was not
#: reached the way the plan assumed. Of 191 legacy declarations, 101 migrated
#: cleanly. The other 90 were refused by the migrator's own precondition: it
#: will not touch a file it cannot resolve *before* the bump, because it then
#: cannot prove the bump was a no-op. Those 90 are exactly the 90 known
#: failures in the config-load baseline.
#:
#: So the trigger fired on the same reasoning this file already gives for v5.0
#: — the gate is not what was keeping them alive. Every one of the 90 was
#: re-tested with its version rewritten to ``1.0`` in memory, and **all 90 still
#: failed**, on retired keys and schema errors that have nothing to do with the
#: version. Emptying this set therefore changes the outcome for zero files; it
#: only changes which error a broken file reports.
#:
#: Kept as an empty frozenset rather than deleted so the gates that import it
#: keep one honest answer to "is this version legacy?" — and so re-introducing
#: a legacy version is a one-line, reviewable change rather than a re-import.
LEGACY_CONFIG_VERSIONS: frozenset[str] = frozenset()

#: The schema versions the loader accepts. **The SSOT** — every version gate imports
#: this rather than restating the set.
#:
#: It lives here, in the lowest config module, because the three gates that enforce it
#: sit on opposite sides of an import edge: ``config/settings.py`` imports
#: ``config/schemas/*``, so a constant in ``settings`` could not be reached from
#: ``config/schemas/training/base.py`` without a cycle. Both already import this module.
#:
#: It is a constant rather than three literals because it was three literals
#: (``settings.py`` twice, ``schemas/training/base.py`` once) and nothing tied them
#: together: adding a version meant finding all three by grep, and a copy that drifts
#: agrees with itself while the others disagree. That is the same failure that made 11
#: of 17 workflow axis-table keys dead (#353) and the sim2rank axis families dead
#: (#317). ``tests/unit/config/schemas/test_base.py`` pins that every gate agrees with
#: this set and rejects what it excludes.
#:
#: Since 2026-08-08 this is exactly ``{CANONICAL_CONFIG_VERSION}``: the legacy set
#: above is empty, so "accepted" and "current" finally name the same thing.
#:
#: v5.x and below are not loadable, and not because of this gate — lifting it
#: entirely still fails every 5.0 file on retired keys (``validation.test_split``
#: in 102 of a 120-file sample). That is the same argument that let 6.0/6.1 be
#: dropped: all 90 files still declaring one were re-tested with the version
#: rewritten to ``1.0`` and all 90 still failed. 613 *committed* 5.0 configs
#: remain, a deliberately deferred backlog
#: (``TODO/backlog_config_version_5_0_corpus.md``), not a bug to fix in passing.
ACCEPTED_CONFIG_VERSIONS: frozenset[str] = (
    frozenset({CANONICAL_CONFIG_VERSION}) | LEGACY_CONFIG_VERSIONS
)


__all__ = [
    "ACCEPTED_CONFIG_VERSIONS",
    "CANONICAL_CONFIG_VERSION",
    "EXPERIMENT_STATUSES",
    "LAUNCH_REFUSED_STATUSES",
    "LEGACY_CONFIG_VERSIONS",
    "BiophysicalFlowObjectiveConfig",
    "DeepSpeedConfigSchema",
    "DiffusionObjectiveConfig",
    "DistBackend",
    "DomainAdaptationObjectiveConfig",
    "EncoderDecoderDefinitionSchema",
    "ExperimentMetadataSchema",
    "ExperimentStatus",
    "FSDPAutoWrapPolicy",
    "FSDPConfigSchema",
    "FSDPShardingStrategy",
    "GANObjectiveConfig",
    "LatentObjectiveConfig",
    "ModuleDefinitionSchema",
    "MultiContrastContrastiveConfigSchema",
    "PEFTConfigSchema",
    "ParallelStrategy",
    "ParallelismConfigSchema",
    "R1RegularizationConfig",
    "ReconstructionObjectiveConfig",
    "SSLConfigSchema",
    "SSLObjectiveConfig",
    "ServicesConfigSchema",
    "TrainingTaskConfigSchema",
    "ZeROOffload",
]


class ModuleDefinitionSchema(BaseModel):
    """Declarative component definition for modular architectures."""

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    name: str = Field(default="")
    pretrained_source: str | None = Field(default=None)
    checkpoint_path: str | None = Field(default=None)
    freeze: bool = Field(default=False)
    params: dict[str, Any] = Field(default_factory=dict)


class EncoderDecoderDefinitionSchema(BaseModel):
    """Composite definition for encoder/decoder style networks."""

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    encoder: ModuleDefinitionSchema = Field(
        default_factory=ModuleDefinitionSchema,
    )
    decoder: ModuleDefinitionSchema = Field(
        default_factory=ModuleDefinitionSchema,
    )
    bridge: ModuleDefinitionSchema | None = Field(default=None)
    shared_weights: bool = Field(default=False)
    notes: dict[str, Any] = Field(default_factory=dict)


#: The closed ``metadata.status`` vocabulary (cohort review 2026-09-02, T0.4).
#: ``LAUNCH_REFUSED_STATUSES`` are the ones ``train`` refuses to launch: an arm
#: whose own metadata says its mechanism is unimplemented, inert or blocked
#: would otherwise run, report a number, and have it read as the mechanism's
#: result (the facade shape, CLAUDE.md non-negotiable 16).
EXPERIMENT_STATUSES: tuple[str, ...] = (
    "ready",
    "needs_implementation",
    "inert",
    "blocked",
    "needs_data",
    "method_demonstration",
    "wiring_demonstrator",
    "baseline_control",
)
LAUNCH_REFUSED_STATUSES: frozenset[str] = frozenset({"needs_implementation", "inert", "blocked"})

ExperimentStatus = Literal[
    "ready",
    "needs_implementation",
    "inert",
    "blocked",
    "needs_data",
    "method_demonstration",
    "wiring_demonstrator",
    "baseline_control",
]

_METADATA_STRING_FIELDS = ("version", "created_at", "timestamp")


class ExperimentMetadataSchema(BaseModel):
    """Experiment metadata harmonised across configuration versions.

    Mounted on ``TrainingSettings.metadata`` since the 2026-09-02 cohort review
    (it used to be declared here and never mounted, so ``train``, the audit and
    ``paired_arms`` each read a bare dict their own way). ``extra="allow"``
    rather than ``forbid``: the corpus carries ~60 free-form keys
    (``note``, ``negative_result_plan``, ``marker_source``, ``group``, ...)
    that are prose for humans, and refusing them would fail the config-load gate
    on most arms for no gain. What IS closed is ``status``: a free-text status
    is how ~190 arms came to admit an unimplemented mechanism in words nothing
    read.
    """

    model_config = {"protected_namespaces": (), "extra": "allow", "frozen": True}

    name: str = Field(default="comprehensive_experiment_template")
    description: str | None = Field(default=None)
    author: str | None = Field(default=None)
    # `None` == "the author did not say", matching every free-form neighbour in
    # this block (`description`, `author`, `created_at`, `timestamp`).
    #
    # This held the string "6.0" -- a *retired, unloadable* config tier (6.0 left
    # ACCEPTED_CONFIG_VERSIONS in PR #891, so declaring it at the root now raises)
    # standing as the default for an author's experiment revision. It is the same
    # defect already fixed in `defaults_provider.py`, in its second location:
    # `metadata.version` versions the ARM, `config_version` versions the SCHEMA,
    # and the v5/v6 -> 1.0 restart smeared the second into the first.
    #
    # Latent rather than live, and it stays fixed for that reason: the production
    # load path types `TrainingSettings.metadata` as a raw `dict[str, Any]`, so
    # this schema reached no arm and is NOT the mechanism that put 6.x into the
    # corpus (those 468 `inprogress/` arms declared it in their own YAML, from a
    # template; drained in this change). Fixing the literal keeps this schema from
    # re-teaching the retired spelling the day it IS wired -- an unwired surface
    # is capability to wire, not dead code to delete.
    version: str | None = Field(default=None)
    created_at: str | None = Field(default=None)
    tags: Any = Field(default_factory=lambda: {})
    timestamp: str | None = Field(default=None)
    # `author`, `source_experiment`, `generated_from` and `roadmap_phase` were declared
    # here with no reader in src/ (2026-09-02 review). They load as free-form extras;
    # a declared field advertises a knob something reads (non-negotiable 8).
    # First-class scientific-coherence metadata (2026-06 scientific-validation campaign):
    # the falsifiable claim, the control arm it is compared against, and the metric that
    # adjudicates it. Previously these had to be smuggled into ``tags`` because the schema
    # was ``extra="forbid"``; making them first-class lets the audit cross-check that a
    # named baseline exists and that primary_metric is actually emitted. All optional so
    # no existing config breaks.
    hypothesis: str | None = Field(default=None)
    baseline: str | None = Field(default=None)
    primary_metric: str | None = Field(default=None)
    # What the arm is DESIGNED to score, so triage stops re-flagging deliberate
    # controls. Several mrixfields ablations exist precisely to die — b110_ablate_pin
    # (h_scale=0, SSIM 0.0013), b22_ablate_likelihood and b27_wiener_refiner_ablate
    # (0.2069) — yet each sweep re-reports them as failures next to arms at 0.75.
    # `floor` says "a near-zero score here is the result, not a bug"; `ceiling` marks
    # an oracle/upper-bound arm. A Literal, not free text, so an unknown value raises
    # at config load rather than silently meaning nothing (pitfall #15).
    expected_outcome: Literal["comparable", "floor", "ceiling"] | None = Field(
        default=None,
        description="Designed outcome relative to the cohort: 'floor' = a degenerate "
        "control expected to score near zero, 'ceiling' = an oracle upper bound, "
        "'comparable' = expected within noise of metadata.baseline. Read by the "
        "diagnostics compiler to annotate the validation-quality table.",
    )
    #: Closed vocabulary. ``train`` refuses ``needs_implementation`` / ``inert`` /
    #: ``blocked`` unless ``--allow-status`` names the status, and stamps the
    #: status into provenance. ``status_reason`` carries the free text that used
    #: to BE the status (``testable_on_real_b0``, ``operational_stub``, ...).
    status: ExperimentStatus | None = Field(
        default=None,
        description=(
            "What the arm can honestly produce today. ready: runs and its number is "
            "the mechanism's result. needs_implementation: the mechanism is not in "
            "the code. inert: the mechanism is in the code but cannot fire as "
            "configured (identity Jacobian, unread knob). blocked: cannot run "
            "(missing component). needs_data: runs on simulated / proxy data only; "
            "the claim needs data not on the cluster. method_demonstration / "
            "wiring_demonstrator: exercises a code path, makes no scientific claim. "
            "baseline_control: a control arm. Refused at launch: needs_implementation, "
            "inert, blocked."
        ),
    )
    status_reason: str | None = Field(
        default=None,
        description="Why the status is what it is (the old free-text status, or a note).",
    )
    #: A property of the arm that a reader of its numbers must know, printed as a
    #: § footnote on its row of the ``report --compare`` main-results table.
    comparison_note: str | None = Field(
        default=None,
        description=(
            "One sentence the cross-run comparison table prints as a footnote on this "
            "arm's row (read from the run's resolved_config.json)."
        ),
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: Any) -> Any:
        """Convert list of tags to dict for backward compatibility."""
        if isinstance(value, list):
            return {tag: tag for tag in value}
        return value

    @model_validator(mode="before")
    @classmethod
    def _stringify_scalar_dates_and_versions(cls, data: Any) -> Any:
        """``version: 6.0`` and ``created_at: 2026-04-10`` arrive as float / date.

        YAML parses them that way; the fields are strings by contract. Stringify
        the scalar rather than refuse the arm -- the value is not wrong, only its
        YAML type is. Only these three fields; a dict or list where a string is
        expected still raises.
        """
        if isinstance(data, dict):
            for key in _METADATA_STRING_FIELDS:
                value = data.get(key)
                if value is not None and not isinstance(value, str | dict | list):
                    data = {**data, key: str(value)}
        return data

    @model_validator(mode="before")
    @classmethod
    def _status_has_one_spelling(cls, data: Any) -> Any:
        """``metadata.tags.status`` is retired: ``metadata.status`` is the owner.

        111 arms carried the status inside ``tags`` where nothing typed could
        read it (and two carried both spellings, disagreeing). The migration
        moved every one; a new arm that writes the tag spelling is refused here
        rather than silently ignored.
        """
        if isinstance(data, dict):
            tags = data.get("tags")
            if isinstance(tags, dict) and "status" in tags:
                raise ValueError(
                    "metadata.tags.status is retired; declare metadata.status "
                    f"(one of {list(EXPERIMENT_STATUSES)}) and put any free text in "
                    "metadata.status_reason. Migrate with "
                    "scripts/migrations/normalise_metadata_status.py."
                )
        return data


class R1RegularizationConfig(BaseModel):
    """Compact configuration for GAN R1 regularisation."""

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    enabled: bool = Field(default=True)
    weight: float = Field(default=10.0, ge=0)
    probability: float = Field(default=1.0, ge=0, le=1)
    interval: int = Field(default=16, gt=0)


class GANObjectiveConfig(BaseModel):
    """GAN-specific training hyperparameters (deprecated, use losses.gan instead).

    Note: All lambda weights and loss-related configs have been moved to LossConfigSchema.gan
    This class is retained for backward compatibility but should only contain paradigm-specific
    hyperparameters that are NOT loss-related.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    # Retained for backward compatibility - these are now in losses.gan
    # but kept here to avoid breaking existing configs
    r1: R1RegularizationConfig = Field(default_factory=R1RegularizationConfig)

    @model_validator(mode="after")
    def _emit_deprecation_warning(self) -> "GANObjectiveConfig":
        """v6.0: objectives.gan is NO LONGER supported."""
        from spectramr.domain.exceptions import ConfigurationError

        raise ConfigurationError(
            "objectives.gan is no longer supported in v6.0. "
            "Migrate loss weights to losses.gan.* in your YAML config. "
            "Example: objectives.gan.lambda_adv → losses.gan.lambda_adv"
        )


class ReconstructionObjectiveConfig(BaseModel):
    """Reconstruction objective configuration (deprecated, use losses.reconstruction instead).

    All lambda weights and loss-related configurations have been moved to LossConfigSchema.reconstruction.
    This class is retained for backward compatibility only.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    # Kept for backward compatibility - see losses.reconstruction for current SSOT
    loss_type: str = Field(default="l1", description="Default loss type (metadata only)")

    @model_validator(mode="after")
    def _emit_deprecation_warning(self) -> "ReconstructionObjectiveConfig":
        """v6.0: objectives.reconstruction is NO LONGER supported."""
        from spectramr.domain.exceptions import ConfigurationError

        raise ConfigurationError(
            "objectives.reconstruction is no longer supported in v6.0. "
            "Migrate loss weights to losses.reconstruction.* in your YAML config. "
            "Example: objectives.reconstruction.lambda_l1 → losses.reconstruction.lambda_l1"
        )


class DiffusionObjectiveConfig(BaseModel):
    """Diffusion-specific hyperparameters (DEPRECATED - use training.diffusion instead).

    NOTE: This entire objectives schema is deprecated. All training configs have been moved to:
    - training.diffusion.timesteps (was objectives.diffusion.timesteps)
      NB: this line said `num_timesteps` until 2026-08-12. That spelling is not a
      declared field — it was absorbed by `extra="allow"` and read by nothing, so
      following this list sent authors to a key that did nothing (issue #980).
    - training.diffusion.noise_schedule (was objectives.diffusion.noise_schedule)
    - training.diffusion.guidance_scale (was objectives.diffusion.guidance_scale)
    - training.diffusion.prediction_type, beta_start, beta_end

    SSOT: Always reference training.diffusion for diffusion training configuration.
    This class is kept only for backward compatibility.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    # DEPRECATED: Use training.diffusion instead
    cond_drop_prob: float = Field(default=0.1, ge=0, le=1)
    enable_diffusion_amp: bool = Field(default=False)
    enforce_output_range: bool = Field(default=True)
    sampler: str = Field(
        default="ddpm", description="Sampling method: ddpm, ddim, predictor_corrector"
    )

    # Cold Diffusion / Advanced Configs
    type: str | None = Field(default=None, description="Diffusion type (e.g. ColdDiffusion)")
    sampling_steps: int | None = Field(default=None, description="Number of sampling steps")
    degradation: dict[str, Any] | None = Field(
        default=None, description="Degradation configuration"
    )
    dynamics: dict[str, Any] | None = Field(default=None, description="Dynamics configuration")

    @model_validator(mode="after")
    def _emit_deprecation_warning(self) -> "DiffusionObjectiveConfig":
        """v6.0: objectives.diffusion is NO LONGER supported."""
        # from spectramr.domain.exceptions import ConfigurationError
        # raise ConfigurationError(
        #     "objectives.diffusion is no longer supported in v6.0. "
        #     "Migrate diffusion hyperparams to training.diffusion.* in your YAML config. "
        #     "Example: objectives.diffusion.timesteps → training.diffusion.timesteps"
        # )
        return self


class LatentObjectiveConfig(BaseModel):
    """Latent variable model hyperparameters (DEPRECATED - use training.latent_dim instead).

    NOTE: The latent_dim parameter has been moved to training.latent_dim for SSOT.
    This class is kept only for backward compatibility with existing configs.
    All VAE/VQVAE training now references training.latent_dim.

    SSOT: Always use config.training.latent_dim for latent dimensionality.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    latent_dim: int = Field(
        default=256, gt=0, description="DEPRECATED: Use training.latent_dim instead"
    )
    n_embeddings: int = Field(default=512, gt=0)
    embedding_dim: int = Field(default=64, gt=0)
    anneal_kl_beta: bool = Field(default=False)
    kl_beta_start: float = Field(default=0.0, ge=0)
    kl_beta_end: float = Field(default=1.0, ge=0)
    kl_anneal_steps: int = Field(default=10000, gt=0)
    latent_regularization_weight: float = Field(default=0.01, ge=0)
    use_latent_regularization: bool = Field(default=True)
    latent_loss_type: str = Field(default="kl_divergence")
    temperature: float = Field(default=0.1, gt=0)

    @model_validator(mode="after")
    def _emit_deprecation_warning(self) -> "LatentObjectiveConfig":
        """v6.0: objectives.latent is NO LONGER supported."""
        from spectramr.domain.exceptions import ConfigurationError

        raise ConfigurationError(
            "objectives.latent is no longer supported in v6.0. "
            "Migrate latent config to training.vae.* in your YAML config. "
            "Example: objectives.latent.latent_dim → training.vae.latent_dim"
        )


class DomainAdaptationObjectiveConfig(BaseModel):
    """Domain adaptation specific hyperparameters."""

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    lambda_adversarial: float = Field(default=0.1, ge=0)
    gradient_reversal_alpha: float = Field(default=1.0, ge=0)
    num_domains: int = Field(default=2, gt=1)


class BiophysicalFlowObjectiveConfig(BaseModel):
    """Biophysical Flow (LaNS/HoBS) Loss Configuration."""

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    lambda_dc: float = Field(default=1.0, ge=0)
    lambda_bloch: float = Field(default=0.1, ge=0)
    lambda_sparsity: float = Field(default=0.01, ge=0)
    lambda_smoothness: float = Field(default=0.1, ge=0)
    t1_range: tuple[float, float] = Field(default=(10.0, 3000.0))
    t2_range: tuple[float, float] = Field(default=(5.0, 500.0))


class SSLObjectiveConfig(BaseModel):
    """SSL-specific hyperparameters."""

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    type: str = Field(default="dinov2")
    student_temp: float = Field(default=0.1, gt=0)
    center_momentum: float = Field(default=0.9, ge=0, le=1)
    teacher_temp: float = Field(default=0.04, gt=0)
    warmup_teacher_temp: float = Field(default=0.04, gt=0)
    warmup_teacher_temp_epochs: int = Field(default=30, ge=0)
    weight_decay: float = Field(default=0.04, ge=0)
    weight_decay_end: float = Field(default=0.4, ge=0)
    clip_grad: float = Field(default=3.0, ge=0)
    freeze_last_layer: int = Field(default=1, ge=0)
    patch_size: int = Field(default=16, gt=0)
    out_dim: int = Field(default=65536, gt=0)
    norm_last_layer: bool = Field(default=True)
    use_bn_in_head: bool = Field(default=False)
    momentum_teacher: float = Field(default=0.996, ge=0, le=1)
    use_fp16: bool = Field(default=False)
    local_crops_number: int = Field(default=8, ge=0)
    global_crops_scale: list[float] = Field(default=[0.4, 1.0])
    local_crops_scale: list[float] = Field(default=[0.05, 0.4])
    warmup_epochs: int = Field(default=10, ge=0)
    min_lr: float = Field(default=1e-6, ge=0)
    lambda_ssl: float = Field(default=1.0, ge=0)


class MultiContrastContrastiveConfigSchema(BaseModel):
    """Knobs for ``MultiContrastContrastiveStrategy`` (InfoNCE / SimCLR-style).

    Declared as a nested sub-block under ``training.ssl`` so every knob the
    strategy reads is validated at load time and stamped into provenance
    (CLAUDE.md non-negotiable #8 / pitfall #15). The strategy reads these
    fields directly from ``config.training.ssl.multi_contrast_contrastive``
    and raises if the block is absent — no flat-config fallback, no
    ``getattr`` defaults that silently swallow YAML knobs.

    ``symmetrize`` and ``use_momentum`` are mutually exclusive: a momentum
    target encoder is gradient-frozen, so running the second (symmetric)
    InfoNCE direction through it has no gradient path and would silently
    no-op. The strategy raises on that combination (pitfall #9) — defaults
    here keep them non-conflicting (``symmetrize=False`` with the default
    ``use_momentum=True``).
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    temperature: float = Field(default=0.07, gt=0)
    symmetrize: bool = Field(default=False)
    use_momentum: bool = Field(default=True)
    momentum: float = Field(default=0.999, ge=0, le=1)
    warmup_steps: int = Field(default=1000, ge=0)


class SSLConfigSchema(BaseModel):
    """Self-supervised learning configuration."""

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    method: str = Field(default="byol")
    temperature: float = Field(default=0.1, gt=0)
    projection_dim: int = Field(default=256, gt=0)
    predictor_hidden_dim: int = Field(default=4096, gt=0)
    ema_decay: float = Field(default=0.996, ge=0, le=1)
    lambda_ssl: float = Field(default=1.0, ge=0)
    multi_contrast_contrastive: MultiContrastContrastiveConfigSchema | None = Field(default=None)


class ServicesConfigSchema(BaseModel):
    """Configuration for optional services."""

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    enable_logging: bool = Field(default=True)
    enable_checkpointing: bool = Field(default=True)
    enable_profiling: bool = Field(default=False)
    enable_metrics_tracking: bool = Field(default=True)
    enable_visualization: bool = Field(default=True)
    enable_early_stopping: bool = Field(default=False)
    enable_validation: bool = Field(default=True)
    enable_brain_extraction: bool = Field(default=True)
    enable_co_registration: bool = Field(default=True)
    enable_memory_optimization: bool = Field(default=True)


class TrainingTaskConfigSchema(BaseModel):
    """Training task specific configuration."""

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    sr_factor: int = Field(default=2, gt=0)
    synthesis_mode: str = Field(default="paired")
    reconstruction_target: str = Field(default="magnitude")
    domain_adaptation_mode: str | None = Field(default=None)


#: How a run is parallelised. Closed because the v6.1 reference template
#: advertised ``fsdp`` and ``deepspeed`` while ``apply_parallelism`` implemented
#: only ``none``/``dp``/``ddp`` and raised on the rest — so the value the template
#: told you to write was a hard error, and the one extra implemented value (``dp``)
#: was undocumented (#620 F1).
ParallelStrategy = Literal["none", "dp", "ddp", "fsdp", "deepspeed"]

#: ``torch.distributed`` process-group backend.
DistBackend = Literal["nccl", "gloo", "mpi"]

#: ZeRO offload target for optimizer state / parameters.
ZeROOffload = Literal["none", "cpu", "nvme"]

#: FSDP parameter-sharding policy.
FSDPShardingStrategy = Literal["full_shard", "shard_grad_op", "no_shard", "hybrid_shard"]

#: How FSDP decides which submodules become their own shard unit.
FSDPAutoWrapPolicy = Literal["size_based", "transformer_block", "none"]


class FSDPConfigSchema(BaseModel):
    """FSDP (Fully Sharded Data Parallel) configuration. PR-14 / L1.

    Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-14.

    Sharding policy and mixed-precision settings for ``torch.distributed.fsdp``.
    Default ``enabled=False`` preserves single-GPU semantics; ``enabled`` must
    agree with ``parallel.strategy`` (see
    :meth:`ParallelismConfigSchema._strategy_matches_subblocks`).
    """

    # forbid (was "ignore"): a typo'd FSDP knob must raise, not be swallowed
    # (pitfall #15). Zero experiment YAML uses an ``fsdp:`` block, so the flip
    # rejects nothing that runs today. WS-1 round-2 closure (2026-07-02).
    model_config = {"extra": "forbid", "frozen": True}

    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("parallel.fsdp"))  # type: ignore[arg-type]
    )

    enabled: bool = Field(default=False)
    sharding_strategy: FSDPShardingStrategy = Field(default="full_shard")
    auto_wrap_policy: FSDPAutoWrapPolicy = Field(
        default="size_based",
        description="size_based uses min_num_params; transformer_block requires "
        "transformer_layer_cls; none wraps only the root.",
    )
    transformer_layer_cls: list[str] = Field(
        default_factory=list,
        description="Class names (as they appear in the model's module tree, e.g. "
        "'TransformerBlock') that become FSDP wrap units under the "
        "transformer_block policy. REQUIRED for that policy: the code previously "
        "passed transformer_layer_cls=set(), which matches no module, so FSDP "
        "wrapped nothing while logging 'FSDP wrapped: sharding=full_shard' — "
        "sharding advertised, NO_SHARD delivered (#621 F1).",
    )
    min_num_params: int = Field(
        default=1_000_000,
        ge=0,
        description="Modules with ≥ this many params are wrapped (size_based policy).",
    )
    cpu_offload: bool = Field(default=False)
    mixed_precision: Literal["fp16", "bf16"] | None = Field(
        default=None,
        description="null | fp16 | bf16 — an override of FSDP's MixedPrecision policy. "
        "null follows optimization.precision, the arm's one precision decision: off is "
        "float32, enabled with no dtype is float16, and bfloat16 must be declared. fp16 "
        "runs under the sharded fp16 scaler, whose non-finite flag is all-reduced across "
        "ranks. Neither half type holds complex parameters, so a complex model is refused. "
        "V100 has no bf16, so fp16 is the half precision there.",
    )
    reduce_dtype: Literal["float32", "match_param"] = Field(
        default="float32",
        description="Precision of the gradient reduce-scatter. float32 (default) "
        "keeps the reduction exact while parameters and activations run in "
        "mixed_precision; match_param reduces in the parameter dtype, which halves "
        "the traffic and loses gradient precision. Ignored when mixed_precision "
        "is null.",
    )

    @model_validator(mode="after")
    def _transformer_policy_needs_layer_classes(self) -> "FSDPConfigSchema":
        if (
            self.enabled
            and self.auto_wrap_policy == "transformer_block"
            and not self.transformer_layer_cls
        ):
            raise ValueError(
                "fsdp.auto_wrap_policy='transformer_block' requires a non-empty "
                "fsdp.transformer_layer_cls. An empty class set matches no module, "
                "so FSDP would wrap nothing and silently behave as no_shard."
            )
        return self


#: DeepCompile graph passes. ``z1``/``z3`` insert the ZeRO-1 / ZeRO-3
#: collectives into the compiled graph; ``autosp`` is automatic sequence
#: parallelism. Closed here because DeepSpeed's own Literal is closed.
DeepCompilePass = Literal["z1", "z3", "autosp"]

#: ZenFlow's index-reselection cadence.
ZenFlowSelectStrategy = Literal["auto", "step", "epoch"]


class DeepSpeedCompileConfigSchema(BaseModel):
    """DeepCompile — compiler passes over the ENGINE graph, not the module.

    This is not ``torch.compile`` with extra steps. ``optimization.compile.enabled``
    compiles the bare module in Stage A, *before* ``deepspeed.initialize``, so
    the ZeRO collectives are opaque calls the compiler cannot see across.
    DeepCompile traces the engine's graph *including* the allgather/reduce-scatter
    and schedules them against compute -- which is the entire reason it exists,
    and why the two are mutually exclusive rather than complementary.

    **A ds_config key alone does nothing.** ``engine.compile()` must also be
    called; without it the engine emits one ``log_dist_once`` line on rank 0 and
    runs eagerly (``engine.py``: "DeepCompile is enabled but engine.compile()
    has not been called"). A once-only rank-0 log at startup is indistinguishable
    from silence on a cluster, so a config-only wiring would be a textbook
    pitfall-#16 facade: declared, accepted, stamped into provenance, inert.
    ``DeepSpeedStrategy.adopt`` makes the call.
    """

    model_config = {"extra": "forbid", "frozen": True}

    enabled: bool = Field(
        default=False,
        description="Renders as ds_config compile.deepcompile AND triggers the "
        "engine.compile() call. Both halves, or neither.",
    )
    passes: list[DeepCompilePass] | None = Field(
        default=None,
        description="Which optimisation passes to compose. Must agree with "
        "zero_stage: 'z1' targets stage 1/2, 'z3' targets stage 3.",
    )
    free_activation: bool = Field(default=False)
    offload_activation: bool = Field(default=False)
    offload_opt_states: bool = Field(default=False)
    offload_parameters: bool = Field(default=False)
    double_buffer: bool = Field(default=True)
    symmetric_memory: bool = Field(default=False)
    debug_log: bool = Field(default=False, description="Dump the compiled graphs. Very verbose.")

    @model_validator(mode="after")
    def _knobs_require_enabled(self) -> "DeepSpeedCompileConfigSchema":
        """Any knob declared with ``enabled: false`` is a silent no-op.

        The generator omits the whole ``compile`` block when DeepCompile is off,
        so ``passes: ['z3']`` or ``double_buffer: false`` alone never reaches
        DeepSpeed at all -- the config looks configured and nothing runs.

        Checked over *every* field rather than a hand-listed subset: a curated
        list is the shape that goes stale the moment a field is added, and the
        rule genuinely applies to all of them equally.
        """
        if self.enabled:
            return self
        declared = sorted(self.model_fields_set - {"enabled"})
        if declared:
            raise ValueError(
                f"parallel.deepspeed.compile declares {declared} but enabled is "
                "false, so the whole block is omitted from the ds_config and "
                "none of it reaches DeepSpeed. Set enabled: true, or remove them."
            )
        return self


class DeepSpeedZenFlowConfigSchema(BaseModel):
    """ZenFlow — importance-aware asynchronous updates on top of ZeRO-Offload.

    Ranks gradient columns by importance, applies the top-k every step and the
    remainder every ``update_interval`` steps, so the CPU-side optimizer step
    stops being the critical path. Supported at stage 1/2
    (``ZenFlowZeroOptimizer``) and at stage 3 (``engine_stage3`` prologue /
    epilogue hooks).

    **It requires CPU offload.** ``zenflow_stage_1_and_2.py`` raises
    ``ValueError("Zenflow must be used with cpu offload")`` -- ZenFlow's whole
    subject is the offloaded optimizer state, so without offload there is
    nothing for it to schedule.

    **It OVERWRITES gradient_accumulation_steps.** ``configure_zenflow`` ends
    with ``engine._config.gradient_accumulation_steps = engine.update_interval``.
    That is a direct collision with this repo's rule that
    ``optimization.gradient.accumulation_steps`` is the single home for that
    number: an arm declaring 4 and a ZenFlow ``update_interval`` of 16 trains at
    16 while its provenance says 4 -- a successful run of a different
    experiment, which is the exact failure the no-``config_path`` design exists
    to prevent. ``check_zenflow_coherent`` makes the disagreement an error.
    """

    model_config = {"extra": "forbid", "frozen": True}

    enabled: bool = Field(default=False)
    topk_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of gradient columns treated as important and updated every step.",
    )
    select_strategy: ZenFlowSelectStrategy = Field(default="auto")
    select_interval: int | Literal["auto"] = Field(
        default="auto",
        description="Reselection cadence. Must be an int unless select_strategy "
        "is 'auto' (DeepSpeed forces it to 1 and warns otherwise).",
    )
    update_interval: int | Literal["auto"] = Field(
        default="auto",
        description="How often the unimportant-gradient accumulation is applied. "
        "THIS BECOMES gradient_accumulation_steps -- see the class docstring.",
    )
    overlap_step: bool = Field(
        default=False,
        description="Overlap the CPU optimizer step with forward/backward. "
        "Selects ZenFlowCPUAdam, so it needs the offloaded optimizer.",
    )
    offload: bool = Field(
        default=False,
        description="Offload the SELECTIVE optimizer states. Distinct from "
        "deepspeed.offload_optimizer; DeepSpeed asserts overlap_comm with it.",
    )
    auto_ratio: float = Field(default=0.99, ge=0.0, le=1.0)
    full_warm_up_rounds: int = Field(
        default=0,
        ge=0,
        description="Rounds during which every gradient is updated (no selection).",
    )
    pt_reserved_cores_perc: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _knobs_require_enabled(self) -> "DeepSpeedZenFlowConfigSchema":
        """Same rule as the compile block, and for the same reason.

        ``_zenflow_block`` returns ``None`` when disabled -- it *must*, because
        ``ZenFlowConfig`` has no ``enabled`` field and DeepSpeed branches on
        ``zenflow_config == None``, so emitting ``{}`` would turn ZenFlow ON with
        all defaults. The consequence is that a disabled block's knobs are not
        merely ignored, they are never rendered.
        """
        if self.enabled:
            return self
        declared = sorted(self.model_fields_set - {"enabled"})
        if declared:
            raise ValueError(
                f"parallel.deepspeed.zenflow declares {declared} but enabled is "
                "false, so the whole block is omitted from the ds_config and "
                "none of it reaches DeepSpeed. Set enabled: true, or remove them."
            )
        return self

    @model_validator(mode="after")
    def _interval_shapes_agree_with_strategy(self) -> "DeepSpeedZenFlowConfigSchema":
        """Mirror the two shape rules DeepSpeed enforces, at load time.

        DeepSpeed raises a bare ``Warning`` (not a warning -- it *raises* a
        Warning instance) for the auto+int case, which is an odd control flow to
        discover on a cluster node after the environment is built.
        """
        if not self.enabled:
            return self
        if self.select_strategy == "auto" and isinstance(self.select_interval, int):
            raise ValueError(
                "parallel.deepspeed.zenflow: select_strategy='auto' forces "
                "select_interval to 1, so declaring select_interval="
                f"{self.select_interval} would be silently overwritten. Pick "
                "select_strategy 'step' or 'epoch' to set it yourself."
            )
        if self.select_strategy != "auto" and isinstance(self.select_interval, str):
            raise ValueError(
                f"parallel.deepspeed.zenflow: select_strategy="
                f"{self.select_strategy!r} requires an integer select_interval."
            )
        if (
            isinstance(self.select_interval, int)
            and isinstance(self.update_interval, int)
            and self.select_interval < self.update_interval
        ):
            raise ValueError(
                f"parallel.deepspeed.zenflow: select_interval="
                f"{self.select_interval} must be >= update_interval="
                f"{self.update_interval} (DeepSpeed raises on this)."
            )
        return self


class DeepSpeedConfigSchema(BaseModel):
    """DeepSpeed ZeRO configuration.

    **This schema is the SSOT; the ``ds_config`` dict is a derived artifact**
    rendered by
    ``spectramr.infrastructure.distributed.deepspeed_backend.config_builder.build_deepspeed_config``
    and written next to the run's provenance.

    There is deliberately no ``config_path`` field. A hand-edited ``ds_config.json``
    is exactly the object ``.claude/rules/data.md`` bans — a cluster-critical file
    with no committed generator — and it is worse than a manifest, because a
    divergence between the JSON and the YAML (``batch_size``,
    ``gradient_accumulation_steps``, ``amp_dtype``) is *silent*: DeepSpeed happily
    accepts a ``train_micro_batch_size_per_gpu`` that contradicts
    ``data.batch_size``.

    Consequently the following ds_config keys are **not declarable here** and are
    always sourced from their existing SSOT: ``train_micro_batch_size_per_gpu``
    (``data.batch_size``), ``gradient_accumulation_steps`` and
    ``gradient_clipping`` (``optimization.*``), ``fp16``/``bf16`` enablement
    (``resolve_amp_precision``), and ``optimizer``/``scheduler`` (the engine adopts
    the objects ``OptimizationBuilder`` already built).
    """

    model_config = {"extra": "forbid", "frozen": True}

    # Removed keys (removals.py) are refused with the reason, not pydantic's bare extra error.
    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("parallel.deepspeed"))  # type: ignore[arg-type]
    )

    enabled: bool = Field(default=False)
    zero_stage: Literal[0, 1, 2, 3] = Field(
        default=0,
        description="0 disables ZeRO (plain DDP through the engine); 1 shards "
        "optimizer state, 2 adds gradients, 3 adds parameters.",
    )
    offload_optimizer: ZeROOffload = Field(default="none")
    offload_param: ZeROOffload = Field(default="none")
    nvme_path: str | None = Field(
        default=None, description="Required when either offload target is 'nvme'."
    )
    contiguous_gradients: bool = Field(default=True)
    overlap_comm: bool = Field(default=True)
    reduce_bucket_size: int = Field(default=500_000_000, gt=0)
    allgather_bucket_size: int = Field(default=500_000_000, gt=0)
    round_robin_gradients: bool = Field(default=False)
    stage3_gather_16bit_weights_on_model_save: bool = Field(
        default=True,
        description="Lets save_16bit_model() emit a consolidated checkpoint under "
        "ZeRO-3.",
    )
    allow_multi_engine: bool = Field(
        default=False,
        description="Permit more than one DeepSpeed engine (e.g. a GAN's opt_g + "
        "opt_d). OFF by default: engine.step() issues collectives, so an arm whose "
        "discriminator steps every k iterations DEADLOCKS rather than erroring.",
    )
    steps_per_print: int = Field(default=1000, gt=0)
    compile: DeepSpeedCompileConfigSchema = Field(
        default_factory=DeepSpeedCompileConfigSchema,
        description="DeepCompile. Renders as the top-level ds_config 'compile' "
        "block AND drives the engine.compile() call.",
    )
    zenflow: DeepSpeedZenFlowConfigSchema = Field(
        default_factory=DeepSpeedZenFlowConfigSchema,
        description="ZenFlow. Renders under zero_optimization.zenflow. Requires "
        "CPU offload, and takes over gradient_accumulation_steps.",
    )

    @model_validator(mode="after")
    def _nvme_needs_a_path(self) -> "DeepSpeedConfigSchema":
        if "nvme" in (self.offload_optimizer, self.offload_param) and not self.nvme_path:
            raise ValueError(
                "deepspeed.nvme_path is required when offload_optimizer or offload_param is 'nvme'."
            )
        return self

    @model_validator(mode="after")
    def _zenflow_needs_cpu_offload(self) -> "DeepSpeedConfigSchema":
        """ZenFlow's subject is the offloaded optimizer state.

        Caught here rather than left to DeepSpeed's own
        ``ValueError("Zenflow must be used with cpu offload")``, which fires
        inside ``deepspeed.initialize`` -- i.e. after the model, optimizer,
        losses, physics and dataloaders have all been built on a cluster node.
        """
        if not self.zenflow.enabled:
            return self
        if self.offload_optimizer == "none":
            raise ValueError(
                "parallel.deepspeed.zenflow.enabled requires "
                "offload_optimizer: 'cpu' (or 'nvme'). ZenFlow schedules the "
                "OFFLOADED optimizer step; with the optimizer on GPU there is "
                "nothing for it to overlap, and DeepSpeed raises 'Zenflow must "
                "be used with cpu offload' from inside deepspeed.initialize."
            )
        if self.zenflow.offload and not self.overlap_comm:
            raise ValueError(
                "parallel.deepspeed.zenflow.offload requires overlap_comm: true "
                "(DeepSpeed asserts this in ZenFlowZeroOptimizer)."
            )
        return self

    @model_validator(mode="after")
    def _round_robin_is_a_stage2_feature(self) -> "DeepSpeedConfigSchema":
        """``round_robin_gradients`` only exists in the ZeRO-2 reduction path."""
        if self.round_robin_gradients and self.zero_stage != 2:
            raise ValueError(
                f"parallel.deepspeed.round_robin_gradients is a ZeRO stage-2 option; "
                f"zero_stage={self.zero_stage} never reads it, so it would be "
                "advertised and inert. Set zero_stage: 2, or remove it."
            )
        return self

    @model_validator(mode="after")
    def _deepcompile_passes_match_the_stage(self) -> "DeepSpeedConfigSchema":
        """A ``z3`` pass at stage 2 optimises collectives that are not there.

        The pass names are not free labels -- ``compile_zero_optimization_stage``
        dispatches on exactly these strings to decide whether to insert the
        ZeRO-1 or ZeRO-3 graph rewrites.
        """
        passes = self.compile.passes or []
        if not self.compile.enabled or not passes:
            return self
        if "z3" in passes and self.zero_stage != 3:
            raise ValueError(
                f"parallel.deepspeed.compile.passes includes 'z3' but "
                f"zero_stage={self.zero_stage}. The z3 pass rewrites the "
                "parameter-partitioning collectives, which only exist at stage 3."
            )
        if "z1" in passes and self.zero_stage not in (1, 2):
            raise ValueError(
                f"parallel.deepspeed.compile.passes includes 'z1' but "
                f"zero_stage={self.zero_stage}. The z1 pass targets the stage-1/2 "
                "optimizer-state partitioning."
            )
        return self


class PEFTConfigSchema(BaseModel):
    """Parameter-efficient fine-tuning (LoRA / adapters). PR-14 / L1."""

    # forbid (was "ignore"): see FSDPConfigSchema — typo'd PEFT knobs must raise.
    model_config = {"extra": "forbid", "frozen": True}

    enabled: bool = Field(default=False)
    method: str = Field(
        default="lora",
        description="lora | adapter | prefix_tuning | bitfit",
    )
    rank: int = Field(default=8, ge=1)
    alpha: float = Field(default=16.0, gt=0.0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["qkv", "proj"],
        description="Module-name substrings to inject LoRA into.",
    )
    bias_mode: str = Field(
        default="none",
        description="none | all | lora_only — controls bias trainability.",
    )


class ParallelismConfigSchema(BaseModel):
    """Configuration for distributed training and parallelism.

    ``strategy`` is the single switch. It used to be a bare string that only
    ``none``/``dp``/``ddp`` were dispatched for, while FSDP was reached through an
    *independent* ``fsdp.enabled`` flag read in ``ModelBuilder.build()``. So
    ``strategy: 'fsdp'`` — the spelling the reference template advertised — raised
    ``ValueError`` after the whole training environment had been built, whereas
    ``strategy: 'none'`` + ``fsdp.enabled: true`` silently sharded. Two switches for
    one mechanism, and the one that read like the switch was the broken one
    (#620 F2). :meth:`_strategy_matches_subblocks` now makes them one declaration.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    strategy: ParallelStrategy = Field(
        default="none",
        description="none (single process) | dp (DataParallel, single process, "
        "N GPUs) | ddp | fsdp | deepspeed. The last three require a process group "
        "(launch via torchrun -m spectramr.cli train-distributed).",
    )
    num_devices: int = Field(default=1, gt=0)
    num_nodes: int = Field(default=1, gt=0)
    backend: DistBackend = Field(
        default="nccl",
        description="Process-group backend. THIS is the SSOT; the "
        "`train-distributed --backend` flag overrides it only when explicitly "
        "passed. Previously read by nothing — setup_distributed took the CLI flag, "
        "whose own default made 'user asked for nccl' indistinguishable from "
        "'argparse filled it in' (#621 F3).",
    )
    find_unused_parameters: bool = Field(default=False)
    gradient_as_bucket_view: bool = Field(default=False)
    static_graph: bool = Field(default=False)
    sync_batch_norm: bool = Field(
        default=False,
        description="Convert BatchNorm to SyncBatchNorm. Requires a process group, "
        "so it is honoured for ddp/fsdp/deepspeed and REJECTED for dp — "
        "SyncBatchNorm cannot work under DataParallel.",
    )
    device_ids: list[int] | None = Field(default=None)
    output_device: int | None = Field(default=None)
    allow_idle_devices: bool = Field(
        default=False,
        description="Permit a launch that leaves scheduler-allocated GPUs unused. "
        "OFF by default: `train-distributed` refuses when this node's GPU grant "
        "exceeds the ranks launched on it, because that failure is otherwise "
        "SILENT -- a --gpus=4 job whose launcher hardcoded nproc_per_node=1 ran "
        "41 minutes on one card with three idle, and the run log, the health "
        "report (141/141 passed) and provenance all read as healthy (#1274). "
        "Turn it ON for a deliberate single-rank debug run on a multi-GPU "
        "allocation; prefer `-O parallel.allow_idle_devices=true` over editing "
        "the arm, so the acknowledgement stays with the launch and not with the "
        "experiment.",
    )

    # PR-14 — FSDP + PEFT extensions
    fsdp: FSDPConfigSchema = Field(default_factory=FSDPConfigSchema)
    peft: PEFTConfigSchema = Field(default_factory=PEFTConfigSchema)
    deepspeed: DeepSpeedConfigSchema = Field(default_factory=DeepSpeedConfigSchema)
    timeout_seconds: int | None = Field(
        default=None,
        gt=0,
        description="Process-group collective timeout in seconds, passed to "
        "init_process_group. null keeps torch's backend default. It bounds how "
        "long peers wait on a rank that died before a collective (a raise on one "
        "rank leaves the others blocked until this expires); raise it only if "
        "rank-0-only validation or checkpointing legitimately exceeds it.",
    )

    @model_validator(mode="after")
    def _strategy_matches_subblocks(self) -> "ParallelismConfigSchema":
        """One declaration, one meaning: the sub-block flag must agree with ``strategy``.

        Without this, the two ways of saying "shard this" could disagree, and the
        disagreement was silent in one direction and fatal in the other.
        """
        if (self.strategy == "fsdp") != self.fsdp.enabled:
            raise ValueError(
                f"parallel.strategy={self.strategy!r} disagrees with "
                f"parallel.fsdp.enabled={self.fsdp.enabled}. FSDP is requested by "
                "setting BOTH strategy: fsdp AND fsdp.enabled: true."
            )
        if (self.strategy == "deepspeed") != self.deepspeed.enabled:
            raise ValueError(
                f"parallel.strategy={self.strategy!r} disagrees with "
                f"parallel.deepspeed.enabled={self.deepspeed.enabled}. DeepSpeed is "
                "requested by setting BOTH strategy: deepspeed AND "
                "deepspeed.enabled: true."
            )
        if self.strategy == "dp" and self.sync_batch_norm:
            raise ValueError(
                "parallel.sync_batch_norm=true is incompatible with strategy='dp': "
                "SyncBatchNorm needs a process group, which DataParallel does not "
                "create. Use strategy='ddp' or drop sync_batch_norm."
            )
        return self
