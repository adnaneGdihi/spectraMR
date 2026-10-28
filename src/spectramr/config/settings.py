import logging
import warnings
from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from spectramr.config.io import load_yaml_strict
from spectramr.config.schemas.acceleration import AccelerationConfigSchema
from spectramr.config.schemas.acquisition import AcquisitionConfigSchema
from spectramr.config.schemas.adapters import AdaptersConfigSchema
from spectramr.config.schemas.audit import AuditConfigSchema
from spectramr.config.schemas.base import (
    ACCEPTED_CONFIG_VERSIONS,
    CANONICAL_CONFIG_VERSION,
    ExperimentMetadataSchema,
    ParallelismConfigSchema,
)
from spectramr.config.schemas.certification import CertificationConfigSchema
from spectramr.config.schemas.checkpoint import CheckpointConfigSchema
from spectramr.config.schemas.data import DataConfigSchema
from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
from spectramr.config.schemas.ema import EMAConfigSchema
from spectramr.config.schemas.logging import LoggingConfigSchema
from spectramr.config.schemas.loss import LossConfigSchema
from spectramr.config.schemas.loss_logging import LossLoggingConfigSchema
from spectramr.config.schemas.loss_schedule import LossScheduleConfigSchema
from spectramr.config.schemas.metrics import MetricsConfigSchema
from spectramr.config.schemas.model import ModelConfigSchema
from spectramr.config.schemas.mrf import MRFConfigSchema

# objectives import removed in v6.0 - deprecated section no longer supported
from spectramr.config.schemas.optimization import OptimizationConfigSchema
from spectramr.config.schemas.physics import PhysicsConfigSchema
from spectramr.config.schemas.plugins import PluginsConfigSchema
from spectramr.config.schemas.renames import (
    ROOT,
    fold_renamed_keys,
    folded_input_keys,
    folded_input_paths,
    reject_renamed_keys,
)
from spectramr.config.schemas.reporting import ReportingSettings
from spectramr.config.schemas.run import (
    CONFIG_VERSION_NOT_AUTHORED_HERE,
    RunConfigSchema,
)
from spectramr.config.schemas.spectra import BackendAccelerationConfigSchema
from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
from spectramr.config.schemas.training.privacy import PrivacyConfig
from spectramr.config.schemas.validation import ValidationConfigSchema
from spectramr.config.schemas.workflow import WorkflowConfigSchema
from spectramr.config.schemas.training.levy_denoising import reads_levy_denoising

logger = logging.getLogger(__name__)


def _bind_config_version(data: dict[str, Any], version: str | None) -> dict[str, Any]:
    """Move the root ``config_version`` key onto ``run.config_version``.

    Shared by BOTH entry points. ``from_yaml`` requires the key and
    ``settings_from_dict`` treats it as optional, but the *binding* must be
    identical: an in-memory config that silently lacked the version while a YAML
    one carried it would make provenance depend on which door the caller used.

    The root key is the only authored spelling — see
    :mod:`spectramr.config.schemas.run` for why the version cannot live inside a
    block whose layout the version itself determines.

    **The legacy fold was deleted on 2026-08-08**, its retirement trigger having
    fired. It rewrote a declared ``6.0``/``6.1`` to
    :data:`CANONICAL_CONFIG_VERSION` here, before binding, so ``run.config_version``
    read ``1.0`` whichever the file declared — and recorded a ledger substitution,
    because that rewrite was otherwise invisible.

    Nothing folds now: :data:`LEGACY_CONFIG_VERSIONS` is empty, so
    :data:`ACCEPTED_CONFIG_VERSIONS` is exactly ``{CANONICAL_CONFIG_VERSION}``
    and a declared version either is canonical or is refused by the gate. See
    that constant for why emptying it changed the outcome for zero files.
    """
    run_section = data.get("run")
    if isinstance(run_section, dict) and "config_version" in run_section:
        raise ValueError(CONFIG_VERSION_NOT_AUTHORED_HERE)
    data.pop("config_version", None)
    if version is not None:
        data["run"] = {**(run_section or {}), "config_version": version}
    return data


# The mypy hook env has no pydantic-settings, so BaseSettings is Any there (#2659).
class TrainingSettings(BaseSettings):  # type: ignore[misc]
    """
    Main configuration class for the application.
    Uses Pydantic V2 and pydantic-settings for robust validation and loading.

    **Configuration Schema Version**: declare
    :data:`~spectramr.config.schemas.base.CANONICAL_CONFIG_VERSION`. It is now the
    **only** accepted version —
    :data:`~spectramr.config.schemas.base.LEGACY_CONFIG_VERSIONS` is empty and the
    fold that rewrote 6.0/6.1 was deleted on 2026-08-08, so a declared version
    either is canonical or is refused by the gate. See ``from_yaml``.

    All nested configurations must use their proper schema objects.
    There are NO legacy compatibility properties or fallback mechanisms.
    All access must go through nested configuration objects (e.g., config.training.seed).

    **Strict Requirements**:
    - All required fields must be present in YAML/config dict
    - No fallbacks to defaults outside Pydantic schema
    - All field access must be type-safe
    - No direct mutation of config after loading (frozen via Pydantic)
    """

    model_config = SettingsConfigDict(
        # Orphan top-level keys are silent fallbacks (CLAUDE.md #9).
        extra="forbid",
        protected_namespaces=(),
        yaml_file=None,
        env_nested_delimiter="__",
        frozen=True,  # Enforce immutability - config must not be mutated after loading
        # Sourced from the constant, never spelled out. These three strings said
        # "6.0" -- the tier this class's own docstring (15 lines above) records
        # as deleted on 2026-08-08 -- so the schema self-described as the one
        # version it now REFUSES. A literal here cannot track
        # CANONICAL_CONFIG_VERSION, and this is the published JSON Schema: every
        # downstream consumer that reads the version off it was told 6.0.
        json_schema_extra={
            "version": CANONICAL_CONFIG_VERSION,
            "title": f"Training Configuration Schema v{CANONICAL_CONFIG_VERSION}",
            "description": (
                f"Strict v{CANONICAL_CONFIG_VERSION} schema - no legacy "
                f"support. All deprecated fields removed. Accepted versions: "
                f"{sorted(ACCEPTED_CONFIG_VERSIONS)}."
            ),
        },
    )

    # Which dotted paths ``apply_overrides`` wrote onto THIS object, in the
    # order they were applied. Read it through
    # :func:`~spectramr.config.overrides.applied_override_paths`, never directly.
    #
    # A PRIVATE attr, not a field, on purpose. This is not configuration -- it
    # is a record of how the configuration was assembled, and making it a field
    # would put it in ``model_dump()``, in the published JSON Schema, in the
    # provenance snapshot, and (with ``extra="forbid"``) make ``override_paths:``
    # a spellable YAML key that an arm could assert about itself. A private attr
    # appears in none of those. It also sidesteps ``frozen=True``: Pydantic v2
    # guards FIELD assignment on a frozen model but not private-attr
    # assignment, so ``apply_overrides`` can stamp the record on the object it
    # just built without the class becoming mutable in any sense that
    # CLAUDE.md #1 is about. Nothing here is ever read back as config.
    #
    # It is not carried across a ``model_dump()``/re-validate round trip --
    # private attrs do not survive one -- which is why ``apply_overrides``
    # merges the incoming object's record into the outgoing one explicitly.
    _override_paths: tuple[str, ...] = PrivateAttr(default=())

    # Top-level fields
    run: RunConfigSchema = Field(
        default_factory=RunConfigSchema,
        description=(
            "Execution-level facts: seed, device, and the schema version this "
            "run was built from. `seed`, `device`, `model_domain` and "
            "`deep_supervision_weight` were bare scalars here until 2026-07-31; "
            "the first two moved into this block and the other two moved to "
            "where they are actually read (see schemas/renames.py)."
        ),
    )

    # Nested configurations
    model: ModelConfigSchema
    data: DataConfigSchema
    optimization: OptimizationConfigSchema
    # objectives field removed in v6.0 - use losses section instead
    losses: LossConfigSchema | None = Field(
        default=None,
        description="Loss configuration (SSOT for all lambda weights and enable flags)",
    )
    logging: LoggingConfigSchema
    metrics: MetricsConfigSchema | None = Field(default_factory=MetricsConfigSchema)
    # Optional block: a config that omits ``checkpoint:`` still gets a usable
    # default (every sub-field is defaulted) so the DI container build does not
    # abort. Mirrors the ``metrics`` field above. The whole mrixfields2026
    # cohort regressed when this defaulted to ``None`` (2026-06-20).
    checkpoint: CheckpointConfigSchema | None = Field(default_factory=CheckpointConfigSchema)

    # Central training strategy configuration
    training: TrainingStrategyConfigSchema | None = Field(
        default=None,
        description="Central training strategy configuration. Specifies strategy_class for dispatch.",
    )
    # Declared imaging regime × task (the workflow contract). Optional today so no
    # existing config breaks. ``config_health_checker.check_workflow_declared`` reports
    # an ABSENT block as advisory (#283) — no arm on dev predates the feature with one —
    # but hard-errors a declaration that is wrong (STUB regime / unsupported task).
    workflow: WorkflowConfigSchema | None = Field(
        default=None,
        description=(
            "Imaging regime × task declaration. Optional in Pydantic; the audit "
            "reports a missing block as advisory and hard-errors a wrong one. "
            "See spectramr.domain.workflows."
        ),
    )

    # Top-level configuration sections
    # These provide flexible schema support for various configuration subsystems
    # One owner (cohort review 2026-09-02, T0.4). This was ``dict[str, Any]`` while
    # ``ExperimentMetadataSchema`` sat unmounted, so `train`, the audit and
    # `paired_arms` each read the block their own way and a free-text `status`
    # was read by nothing. The schema keeps ``extra="allow"`` for the prose keys.
    metadata: ExperimentMetadataSchema | None = Field(default=None)
    loss_logging: LossLoggingConfigSchema | None = Field(default=None)
    early_stopping: EarlyStoppingConfigSchema | None = Field(default=None)
    loss_schedule: LossScheduleConfigSchema | None = Field(
        default=None,
        description="Dynamic loss-term scheduling: clock curriculum + plateau "
        "triggers + post-change metric monitoring (see loss_schedule.py).",
    )
    ema: EMAConfigSchema | None = Field(default=None)
    validation: ValidationConfigSchema | None = Field(default=None)
    # Deprecated: top-level ``diffusion:`` is silently absorbed.  All real
    # diffusion configuration lives at ``training.diffusion:`` in v6.0.  Kept
    # as ``dict[str, Any]`` so legacy YAMLs that never made it through the
    # migration still load under ``extra="forbid"``.
    diffusion: dict[str, Any] | None = Field(
        default=None,
        deprecated=(
            "Top-level config.diffusion is unused — set training.diffusion instead. "
            "This field is kept solely to avoid breaking legacy YAMLs under "
            "extra='forbid'."
        ),
    )
    # Renamed from `acceleration:` in phase 11: the old name meant two unrelated
    # things -- the MRI k-space acceleration factor AND compute acceleration --
    # and only the first is what this block configures. The five compute knobs
    # that motivated the second reading are inert; see the class docstring.
    # The class is still `AccelerationConfigSchema`: renaming the FIELD is the
    # user-visible deliverable, and the class name touches 121 call sites for no
    # reader benefit.
    undersampling: AccelerationConfigSchema | None = Field(default=None)
    # SPECTRA hardware inference backend (distinct from ``acceleration``, which
    # is k-space undersampling). Declaring the typed field here is what makes a
    # YAML ``backend_acceleration:`` block readable under ``extra="forbid"``;
    # selecting the spectra backend without it is a hard audit failure, never a
    # silent default. See spectramr.config.schemas.spectra and the
    # ``check_spectra_*`` Tier-1 guards.
    backend_acceleration: BackendAccelerationConfigSchema | None = Field(
        default=None,
        description=(
            "SPECTRA hardware inference-backend configuration (WCSR systolic "
            "array + NMAE physics-remap over the RoCC contract). Optional; when "
            "omitted the run uses the default torch/device backend. See "
            "spectramr.config.schemas.spectra.BackendAccelerationConfigSchema."
        ),
    )
    physics: PhysicsConfigSchema | None = Field(default=None)
    # `artifacts` was removed (#698) and is refused at load; see removals.py.
    # Declarative adapter chains. Per CLAUDE.md #9 adapters NEVER fire
    # silently — they're the only way to bridge a capability mismatch
    # between data, model, loss, and metric layers. See
    # docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md
    adapters: AdaptersConfigSchema | None = Field(default=None)
    reporting: ReportingSettings | None = Field(
        default=None,
        description=(
            "End-of-training reporting hook (canonical figures / tables, "
            "see spectramr.infrastructure.reporting). When omitted, no report is generated."
        ),
    )
    # v6.1 top-level blocks. Each schema is opt-in (sub-blocks default to
    # ``enabled=False``); declaring the field here is what makes the block
    # readable from YAML — without it, ``extra="forbid"`` would reject the
    # whole config.
    acquisition: AcquisitionConfigSchema | None = Field(
        default=None,
        description=(
            "Acquisition / sampling design (PILOT-style co-design, BALD active "
            "acquisition). See spectramr.config.schemas.acquisition."
        ),
    )
    certification: CertificationConfigSchema | None = Field(
        default=None,
        description=(
            "Regulatory certification block (R1 CDFC, R2 CHD, R4 PRC, R5 PAC-Bayes, "
            "V.3 validation badge). See spectramr.config.schemas.certification."
        ),
    )
    parallel: ParallelismConfigSchema | None = Field(
        default=None,
        description=(
            "Distributed-training / parallelism configuration (DDP, FSDP, PEFT). "
            "See spectramr.config.schemas.base.ParallelismConfigSchema."
        ),
    )
    audit: AuditConfigSchema | None = Field(
        default=None,
        description=(
            "Optional Tier-3 audit hooks (audit_plan_novel.md). Currently "
            "exposes ``audit.ksd`` for the kernelised Stein discrepancy "
            "defensibility check. See spectramr.config.schemas.audit."
        ),
    )
    mrf: MRFConfigSchema | None = Field(
        default=None,
        description=(
            "MR-fingerprinting metadata block (audit_plan_novel_fmri.md "
            "MRF §§1, 4, 5). Hosts spiral rotation schedules, SAR "
            "ceilings, and phantom-calibration references consumed by "
            "the MRF Tier-1/Tier-2 audit validators."
        ),
    )
    privacy: PrivacyConfig | None = Field(
        default=None,
        description=(
            "Differential-privacy (DP-SGD) knobs consumed by federated DP "
            "training strategies (noise_multiplier, max_grad_norm, "
            "target_delta, dataset_size). Declaring the typed field here is "
            "what makes a YAML `privacy:` block readable under extra='forbid' "
            "and lets a YAML-set noise_multiplier actually reach the strategy "
            "(pitfall #15). See spectramr.config.schemas.training.privacy."
        ),
    )
    plugins: PluginsConfigSchema | None = Field(
        default=None,
        description=(
            "Out-of-tree plugin discovery. Lists dotted module paths imported "
            "at registry population (peer of the SPECTRAMR_PLUGINS env var) so a "
            "user's @register_model / @register_loss / @register_metric "
            "decorators defined OUTSIDE the spectramr tree fire and resolve by "
            "name. Declaring the typed field here is what makes a YAML "
            "`plugins:` block readable under extra='forbid'. See "
            "spectramr.config.schemas.plugins and spectramr.plugins.discover_plugins."
        ),
    )

    # v6.0: No legacy validation needed - strict schema enforcement only

    # `model_domain` used to sit at the root as a write-only alias: NOTHING read
    # it, and a `mode="before"` validator here copied it into
    # `model.model_domain` / `model.target_domain` -- the fields consumers
    # actually read -- raising when the two disagreed. Retired 2026-07-31: an
    # alias whose only job is to be copied somewhere else is a second spelling,
    # and `renames.py` now rejects it by name pointing at `model.model_domain`.

    _reject_renamed_root = model_validator(mode="before")(
        classmethod(reject_renamed_keys(ROOT))  # type: ignore[arg-type]
    )

    # The only fold that may cross top-level blocks. A per-block fold validator
    # is mounted on one class and cannot reach a sibling; this one is mounted
    # here, so it can rename a whole block (`acceleration:` -> `undersampling:`)
    # before any sub-model is constructed from it. Pydantic runs a root
    # `mode="before"` ahead of every nested one -- pinned by
    # `test_renames.py::TestRootFoldRunsBeforeAnySubModel`.
    # Published for the execution ledger, which compares the RAW declaration
    # against the resolved model. Without this, `acceleration:` -- folded to
    # `undersampling:` above -- reads as EXTRA_IGNORE_DROPPED at severity
    # "error" on every arm that has not migrated yet: "the run never sees it",
    # about a value the run does see. `core/execution_ledger.py` states the cost
    # directly ("a ledger that cries wolf 35 times stops being read"), and the
    # per-block schemas already publish theirs; the ROOT fold added in phase 11
    # simply forgot to.
    __folded_input_keys__ = folded_input_keys(ROOT)
    __folded_input_paths__ = folded_input_paths(ROOT)

    _fold_renamed_root = model_validator(mode="before")(
        classmethod(fold_renamed_keys(ROOT))  # type: ignore[arg-type]
    )

    @model_validator(mode="after")
    def _warn_on_deprecated_top_level_blocks(self) -> "TrainingSettings":
        """Emit a runtime ``DeprecationWarning`` for deprecated top-level blocks.

        Pydantic's ``Field(deprecated=...)`` marker emits its warning on every
        attribute *access* (via the field descriptor's ``__get__``), not on
        *set* — so a plain ``if self.diffusion is not None:`` guard would fire
        the warning on EVERY load, even for configs that never carried the
        legacy block (a false migration signal, and under CLAUDE.md #10 an
        error in the test suite). We therefore read the stored values straight
        from ``self.__dict__`` (bypassing the descriptor) and only warn when the
        block was actually present in the parsed YAML. ``mode="after"`` is safe
        despite ``frozen=True`` because we only read ``self`` (no mutation).
        Messages mirror the ``deprecated=`` strings on the fields.
        """
        if self.__dict__.get("diffusion") is not None:
            warnings.warn(
                "Top-level config.diffusion is unused — set training.diffusion "
                "instead. This field is kept solely to avoid breaking legacy "
                "YAMLs under extra='forbid'.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _recorrupted_inference_needs_an_r2r_target(self) -> "TrainingSettings":
        """Refuse ``validation.scoring.recorrupted_inference`` where nothing recorrupts.

        The Monte-Carlo estimator (#2329) redraws the training recorruption at
        validation, so it is defined only for ``data.target_mode`` ``r2r`` (a
        drawn ``alpha``) and ``nullspace_r2r`` (a split ``angle_deg``). The
        validation mixin re-checks the same facts at the first batch; refusing
        here is what lets ``spectramr audit`` reject the YAML at load time
        (non-negotiable 3) instead of the first validation step.
        """
        scoring = getattr(self.validation, "scoring", None)
        knob = getattr(scoring, "recorrupted_inference", None)
        if knob is None:
            return self
        mode = getattr(self.data, "target_mode", None)
        strengths = {"r2r": "alpha", "nullspace_r2r": "angle_deg"}
        if mode not in strengths:
            raise ValueError(
                "validation.scoring.recorrupted_inference is read only under "
                f"data.target_mode 'r2r' or 'nullspace_r2r'; this arm declares {mode!r}, "
                "which draws no recorruption to average over."
            )
        other = "angle_deg" if strengths[mode] == "alpha" else "alpha"
        if getattr(knob, other, None) is not None:
            raise ValueError(
                f"validation.scoring.recorrupted_inference.{other} is the strength of the "
                f"other target mode; under data.target_mode={mode!r} declare "
                f"'{strengths[mode]}' or leave the strength to the data block."
            )
        return self

    @model_validator(mode="after")
    def _ensemble_needs_a_stochastic_sampler(self) -> "TrainingSettings":
        """Refuse a validation ensemble whose members would all be identical.

        ``validation.sampling.ensemble_samples > 1`` draws N reverse samples and
        reports their spread. The C6 sampler is deterministic at
        ``model.model_kwargs.sampler_sigma == 0`` (the value an absent key
        resolves to), so all N members would coincide, the std would be exactly
        0 and the coverage number a facade (non-negotiable 16). This checks the
        DECLARED sigma; ``DiffusionTrainingStrategy._resolve_validation_ensemble``
        re-checks the RESOLVED generator, which is a different fact.
        """
        sampling = getattr(self.validation, "sampling", None)
        n = int(getattr(sampling, "ensemble_samples", 1) or 1)
        if n <= 1:
            return self
        model_kwargs = getattr(self.model, "model_kwargs", None) or {}
        raw_sigma = model_kwargs.get("sampler_sigma", 0.0)
        try:
            sigma = float(raw_sigma)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"model.model_kwargs.sampler_sigma={raw_sigma!r} is not a number, so "
                f"validation.sampling.ensemble_samples={n} cannot be honoured."
            ) from exc
        if not sigma > 0:
            raise ValueError(
                f"validation.sampling.ensemble_samples={n} needs a stochastic reverse "
                f"sampler, but model.model_kwargs.sampler_sigma resolves to {sigma!r}: "
                "every ensemble member would be identical. Set "
                "model.model_kwargs.sampler_sigma > 0 (and refit any trust calibration "
                "at that sigma), or leave ensemble_samples at 1."
            )
        return self

    @model_validator(mode="after")
    def _forward_noise_excludes_sampler_sigma(self) -> "TrainingSettings":
        """Refuse ``forward_noise_sigma > 0`` with ``sampler_sigma > 0`` (one owner of reverse noise).

        Both knobs inject noise into the reverse trajectory, so declaring both would
        make the reverse noise a sum of two undeclared scales. Mirrors the build-time
        check in ``KSpaceColdDiffusionGenerator``.
        """
        model_kwargs = getattr(self.model, "model_kwargs", None) or {}
        forward = float(model_kwargs.get("forward_noise_sigma", 0.0) or 0.0)
        sampler = float(model_kwargs.get("sampler_sigma", 0.0) or 0.0)
        if forward > 0 and sampler > 0:
            raise ValueError(
                "model.model_kwargs.forward_noise_sigma > 0 and model.model_kwargs.sampler_sigma > 0 "
                "both inject reverse-trajectory noise; declare one."
            )
        return self

    @model_validator(mode="after")
    def _tall_locality_needs_a_tall_model(self) -> "TrainingSettings":
        """Refuse ``training.tall_sr.lambda_locality > 0`` on any model but ``tall_sr``.

        The locality term reads the TALL traversal of the ``tall_sr`` generator; any
        other model has none to regularise. Refused here (non-negotiable 3) rather
        than at the first training step.
        """
        tall_cfg = getattr(self.training, "tall_sr", None)
        weight = getattr(tall_cfg, "lambda_locality", 0.0)
        model_type = getattr(self.model, "model_type", None)
        if weight > 0 and model_type != "tall_sr":
            raise ValueError(
                f"training.tall_sr.lambda_locality={weight} requires model.model_type "
                f"'tall_sr'; this arm declares {model_type!r}, which has no TALL traversal."
            )
        return self

    @model_validator(mode="after")
    def _levy_denoising_needs_its_strategy(self) -> "TrainingSettings":
        """Refuse ``training.levy_denoising`` on a strategy that never reads it."""
        if getattr(self.training, "levy_denoising", None) is None:
            return self
        strategy = getattr(self.training, "strategy_class", None)
        mode = getattr(self.training, "training_mode", None)
        if not reads_levy_denoising(strategy, mode):
            raise ValueError(
                "training.levy_denoising is read only by LevyDenoisingStrategy; this arm "
                f"routes to strategy_class={strategy!r}, training_mode={mode!r}."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingSettings":
        """Load settings from a YAML file with strict version enforcement.

        **Version Policy**: Schema versions 6.0 and 6.1 are both accepted.
        Older versions raise ValueError — there is no auto-migration path.

        Args:
            path: Path to YAML configuration file

        Returns:
            TrainingSettings instance

        Raises:
            ValueError: If config_version is missing or not in ACCEPTED_CONFIG_VERSIONS
            FileNotFoundError: If YAML file not found
        """
        # Performance: validate path exists before expensive YAML parsing
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # Performance: single file read operation with context manager
        with path_obj.open("r", encoding="utf-8") as f:
            data = load_yaml_strict(f)

        # Fail-Fast: Validate we got dict-like structure
        if not isinstance(data, dict):
            raise ValueError(
                f"YAML file {path} must be a dictionary at root level, got {type(data).__name__}"
            )

        # Fail-Fast: version validation with minimal string operations
        config_version = data.get("config_version")
        if config_version is None:
            raise ValueError(
                f"config_version is required in {path}. "
                f"Must be one of {sorted(ACCEPTED_CONFIG_VERSIONS)}."
            )
        if config_version not in ACCEPTED_CONFIG_VERSIONS:
            raise ValueError(
                f"Config version {config_version} not supported in {path}. "
                f"Accepted values: {sorted(ACCEPTED_CONFIG_VERSIONS)}. "
                f"Please update your configuration."
            )

        # Previously a bare `del`, which meant the version never existed on the
        # object and survived only into provenance -- an inspectable fact
        # reduced to a parse-time side effect.
        data = _bind_config_version(data, config_version)

        return cls._finalize_from_dict(data, source=str(path))

    @model_validator(mode="after")
    def _domain_grl_needs_cyclegan(self) -> "TrainingSettings":
        """Refuse ``losses.gan.domain_grl.enabled`` on any strategy but ``cyclegan``.

        Only CycleGANTrainingStrategy builds the bottleneck head; another strategy would
        read the block as an unread knob (non-negotiable 8).
        """
        gan_cfg = getattr(self.losses, "gan", None)
        enabled = bool(getattr(getattr(gan_cfg, "domain_grl", None), "enabled", False))
        strategy = getattr(self.training, "strategy_class", None)
        if enabled and strategy != "cyclegan":
            raise ValueError(
                "losses.gan.domain_grl.enabled requires training.strategy_class 'cyclegan'; "
                f"this arm declares {strategy!r}, which builds no domain head."
            )
        return self

    @classmethod
    def _finalize_from_dict(cls, data: dict[str, Any], *, source: str) -> "TrainingSettings":
        """Apply the dict-level transforms shared by ``from_yaml`` and
        ``settings_from_dict``, then construct the frozen settings.

        1. Validate ``model.model_type`` against the populated registry.
        2. Apply the coil-processing legacy↔new bridges — the unified
           ``physics.coil_processing`` block is the SSOT the data-load reads.
           These live HERE (not in a Pydantic validator), so any in-memory
           constructor MUST route through this method or the legacy
           ``coil_processing_mode`` knob is silently un-migrated (pitfall #15).

        When an :class:`~spectramr.core.execution_ledger.ExecutionLedger` is armed,
        the declared dict is snapshotted here and diffed against the constructed
        settings, so every key the schema dropped, defaulted, or rewrote lands in
        the run's ``resolved_config.json``. This is the only point where both
        sides are in scope, and it is shared by ``from_yaml`` and
        ``settings_from_dict``. Unarmed callers (audit, tests, notebooks) pay one
        ``ContextVar`` read.
        """
        from spectramr.core.execution_ledger import ExecutionLedger

        # deepcopy, not dict(): the coil-processing bridges below rewrite nested
        # sub-dicts in place, and a shallow copy would compare the post-bridge
        # tree against itself and report every bridge rewrite as a no-op.
        _declared: dict[str, Any] | None = None
        if ExecutionLedger.recording():
            import copy

            _declared = copy.deepcopy(data)

        # 0. Out-of-tree plugins declared in the config (config.plugins.paths).
        #    Populate the IN-TREE registry FIRST, THEN import the plugins, so a
        #    plugin re-registering an in-tree component name collides LOUDLY
        #    (register_* raises → PluginImportError) instead of silently
        #    shadowing it (spec §6.1; pitfall #9 — no silent fallback). The
        #    plugin import still runs BEFORE the model_type check below, so a
        #    config may legitimately name a model_type a plugin provides.
        #    populate_model_registry is idempotent (modules are cached in
        #    sys.modules), so the later call in step 1 is a cheap no-op.
        plugins_block = data.get("plugins")
        if not isinstance(plugins_block, dict):
            plugins_block = {}
        plugin_paths = plugins_block.get("paths") or []
        if plugins_block.get("enabled") and plugin_paths:
            from spectramr.models.init_registry import populate_model_registry
            from spectramr.plugins import import_plugin_paths

            populate_model_registry()  # in-tree names registered BEFORE plugins
            import_plugin_paths(plugin_paths)
        elif plugin_paths:
            # Paths declared but the master switch is off → surface the no-op
            # rather than silently skipping the import (pitfall #15c — a knob
            # that is set must visibly do something or say why it didn't).
            logger.info(
                "plugins.paths lists %d module(s) but plugins.enabled is false "
                "— not importing them; set plugins.enabled: true to load.",
                len(plugin_paths),
            )

        # 1. Validate model_type against registry
        model_type = data.get("model", {}).get("model_type")
        if model_type:
            from spectramr.models.init_registry import populate_model_registry
            from spectramr.models.registry import list_models

            # Ensure the registry is fully populated. ``from_yaml`` runs on
            # every config load (and the same config is loaded multiple times
            # across audit/probe/train); ``populate_model_registry`` is now
            # internally idempotent (see step 0), so the expensive
            # ``pkgutil.walk_packages`` discovery walk runs once and is a cheap
            # no-op on later calls.
            # NB: do NOT gate this on ``MODEL_REGISTRY`` being empty — the
            # registry is partially populated by eager ``@register_model``
            # import side-effects before the walk runs, so an emptiness check
            # would skip the walk and drop ~200 walk-only models plus all the
            # compatibility aliases.
            populate_model_registry()

            registry = list_models()
            if model_type not in registry:
                # A model can be absent for two very different reasons: the name is
                # genuinely wrong, or its module failed to import during discovery
                # (a missing optional dep drops every @register_model inside it). The
                # second case used to masquerade as the first — the CI audit rejected
                # `neural_complex_sum`, which is registered and correct, purely because
                # torchvision was absent and vf_reconstruction_generators never loaded.
                # Say which one it is.
                from spectramr.models.init_registry import get_registry_import_failures

                failures = get_registry_import_failures()
                detail = ""
                if failures:
                    lines = "\n".join(f"  - {mod}: {err}" for mod, err in sorted(failures.items()))
                    detail = (
                        f"\n\nNOTE: the model catalog is INCOMPLETE — {len(failures)} module(s) "
                        f"failed to import during discovery, so any model they register is "
                        f"missing from the list below. Fix the import first; '{model_type}' may "
                        f"well be one of them:\n{lines}"
                    )
                raise ValueError(
                    f"Invalid model_type: '{model_type}' in {source}. "
                    f"Must be one of the registered models. "
                    f"Available models: {list(registry.keys())}{detail}"
                )

        # 2. Coil-processing config bridge (before schema construction).
        #   - derive: a legacy data.coil_processing_mode synthesizes the full
        #     4-axis block (merged — user-set sub-blocks win). No-op + silent
        #     when there is no legacy mode (~200 legacy YAMLs load unchanged).
        #   - sync: a user-authored svd compression block back-fills the legacy
        #     data.coil_processing_mode / num_virtual_coils for any code still
        #     reading those during Phase 1.
        from spectramr.config.schemas.loader import (
            _derive_coil_processing_from_legacy,
            _sync_coil_processing_to_legacy,
        )

        data = _derive_coil_processing_from_legacy(data)
        data = _sync_coil_processing_to_legacy(data)

        settings = cls(**data)

        # 3. Record what the declaration became. Never let a diagnostic break a
        #    real config load: an exception here would turn "your run is
        #    instrumented" into "your run does not start".
        if _declared is not None:
            from spectramr.core.execution_ledger import diff_declared_vs_resolved

            ledger = ExecutionLedger.current()
            if ledger is not None:
                try:
                    diff_declared_vs_resolved(
                        _declared, settings, ledger=ledger, stage="config_finalize"
                    )
                except Exception:
                    logger.exception(
                        "declared-vs-resolved diff failed for %s; the ledger in "
                        "this run's artifacts is incomplete",
                        source,
                    )
                    ledger.notes.append(f"declared-vs-resolved diff failed for {source}")

        return settings

    @classmethod
    def settings_from_dict(cls, data: dict[str, Any]) -> "TrainingSettings":
        """Construct from an in-memory dict (no YAML file), validated + frozen.

        The public Python-scripting peer of :meth:`from_yaml` — the path a
        ``spectramr.api.fit`` / ``settings_from_dict`` caller uses to build the
        SSOT config in code. It applies the SAME dict-level transforms
        (model_type registry validation + the coil-processing bridges), with
        two deliberate differences from ``from_yaml``:

        * ``config_version`` is OPTIONAL — absent → the schema default is
          assumed; present → validated against ``ACCEPTED_CONFIG_VERSIONS`` and
          stripped (an unsupported version still raises, never silently accepted).
        * the caller's dict is never mutated (a deep copy is taken first), so a
          dict reused across calls is safe.
        """
        import copy as _copy

        if not isinstance(data, dict):
            raise ValueError(f"settings_from_dict expects a dict, got {type(data).__name__}")

        data = _copy.deepcopy(data)

        config_version = data.get("config_version")
        if config_version is not None and config_version not in ACCEPTED_CONFIG_VERSIONS:
            raise ValueError(
                f"Config version {config_version} not supported. "
                f"Accepted values: {sorted(ACCEPTED_CONFIG_VERSIONS)}."
            )
        data = _bind_config_version(data, config_version)

        return cls._finalize_from_dict(data, source="<in-memory dict>")

    def get_validated_snapshot(self) -> dict[str, Any]:
        """Return a dictionary representation of the validated configuration.

        This snapshot is the run's provenance record (serialized to JSON/YAML
        for experiment tracking). It dumps ALL resolved fields, including those
        left at their default — a knob at its default value still *drove* the
        run, and pitfall #15c requires provenance to prove which value was used.
        ``exclude_unset`` / ``exclude_defaults`` would silently omit exactly the
        defaulted knobs whose value we need to certify, so they are NOT used.
        """
        return dict(self.model_dump(mode="json"))


#: Module-level alias of :meth:`TrainingSettings.settings_from_dict` so the
#: public scripting surface (``spectramr.settings_from_dict`` / ``spectramr.api``)
#: can re-export an in-memory config builder from this lightweight config module
#: without importing the heavier ``spectramr.api`` facade.
settings_from_dict = TrainingSettings.settings_from_dict
