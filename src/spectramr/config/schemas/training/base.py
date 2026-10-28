"""Base training configuration schema for shared fields across all training paradigms.

This module contains BaseTrainingConfigSchema which defines fields common to all
training modes, plus infrastructure-level configurations that are paradigm-agnostic.

Paradigm-specific training configs (TrainingConfigGAN, TrainingConfigDiffusion, etc.)
inherit from this base class and add paradigm-specific fields and validators.
"""

import logging
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ..acceleration import AccelerationConfigSchema
from ..augmentation import AugmentationConfigSchema
from ..base import (
    ACCEPTED_CONFIG_VERSIONS,
    CANONICAL_CONFIG_VERSION,
    LEGACY_CONFIG_VERSIONS,
    ServicesConfigSchema,
)
from ..enums import DIFFUSION_PARADIGM_ALIASES, NoiseSchedule
from ..loss import LossConfigSchema
from ..objectives import ObjectiveConfigSchema
from ..renames import fold_renamed_keys, reject_renamed_keys
from ..strictness import CompatSchema, StrategySchema
from .ambient import AmbientTrainingConfigSchema
from .bloch_field import BlochFieldConfig, BlochSynthConfig
from .brenier import BrenierSynthesisConfig
from .cartoon_texture_safe import CartoonTextureSafeConfig
from .certified_robustness import CertifiedRobustnessConfig
from .confluence import ConfluenceConfig
from .cross_field import CrossFieldConfig, FieldCocycleConfig, FieldFlowConfig
from .cycle_bloch import CycleBlochConfig
from .distillation import DistillationTrainingConfigSchema
from .doob_bridge import DoobBridgeConfig
from .equivariant_imaging import TrainingConfigEquivariantImaging
from .field_bridge import FieldBridgeConfig
from .field_cold_diffusion import FieldColdDiffusionConfig
from .field_conditioned_inr import FieldConditionedINRConfig
from .field_fno import FieldFNOConfig
from .field_guided_diffusion import FieldGuidedDiffusionConfig
from .field_wiener import FieldWienerConfig
from .fisher_rao_geodesic import FisherRaoGeodesicConfig
from .generative_refiner import GenerativeRefinerConfig
from .geomamba_ulf import GeoMambaULFTrainingConfigSchema
from .hamiltonian_acquisition import HamiltonianAcquisitionConfig
from .heteroscedastic_ulf import HeteroscedasticULFConfig
from .koopman_field import KoopmanFieldConfig
from .lora_modulation import LoRAModulationConfig
from .low_rank_sparse import LowRankSparseTrainingConfigSchema
from .mccann_field_path import McCannFieldPathConfig
from .monotone_field import MonotoneFieldConfig
from .multi_echo_b0_fit import MultiEchoB0FitConfig
from .phase_contrast import (
    MRSQuantificationConfig,
    PerfusionKineticConfig,
    PhaseContrastFlowConfig,
)
from .pipeline import MultiTrainingConfigSchema, StageEnvironmentSchema
from .quality_matching import QualityMatchingConfig
from .recoverability_vib import RecoverabilityVIBConfig
from .scattering_besov import ScatteringBesovConfig
from .sparse_frame import SparseFrameConfig
from .ssdu import SSDUTrainingConfigSchema
from .steerable_synthesis import SteerableSynthesisConfig
from .strategy_knobs_2026_06 import (
    CorticalConformalFMRIReconTrainingConfigSchema,
    HRFManifoldDiffusionTrainingConfigSchema,
    InverseBlochPhaseTrainingConfigSchema,
    MAEPretrainingTrainingConfigSchema,
    MRISLAMTrainingConfigSchema,
    QSMPipelineTrainingConfigSchema,
    QSpaceDiffusionTrainingConfigSchema,
    RiemannianDFCDiffusionTrainingConfigSchema,
    RiemannianMRFDiffusionTrainingConfigSchema,
    SchrodingerBridgeTrainingConfigSchema,
    ScoreFieldTomographyTrainingConfigSchema,
    SliceToVolumeTrainingConfigSchema,
    SpatiotemporalMRFReconTrainingConfigSchema,
    SpinSDETrainingConfigSchema,
    SyntheticPathologyAugTrainingConfigSchema,
    TrainingConfigTrajectoryRecon,
    TTTTrainingConfigSchema,
    VFConsistencyDistillationTrainingConfigSchema,
)
from .strategy_knobs_2026_08 import (
    AdaptiveSFCHSSCTrainingConfigSchema,
    BeltramiEPIDistortionTrainingConfigSchema,
    BlochEquivariantTranslationTrainingConfigSchema,
    ConformalDiffusionReconTrainingConfigSchema,
    ConformalMRFDictlessReconTrainingConfigSchema,
    CRLBMRFPulseDesignTrainingConfigSchema,
    CrossScannerMRFHarmonisationTrainingConfigSchema,
    DTN2STrainingConfigSchema,
    IBActiveAcquisitionTrainingConfigSchema,
    PrivilegedLearningTrainingConfigSchema,
    RiemannianBlochDiffusionTrainingConfigSchema,
    SpatiotemporalAdaptiveSFCReconTrainingConfigSchema,
)
from .levy_denoising import LevyDenoisingTrainingConfigSchema
from .tall_sr import TALLSRTrainingConfigSchema
from .teichmuller import TeichmullerTrainingConfigSchema
from .ulf_dps import UlfDpsConfig
from .ulf_map import UlfMapConfig
from .ulf_lowpass import UlfLowpassConfig
from .ulf_redegrad_tta import UlfReDegradationTTAConfig

logger = logging.getLogger(__name__)


class DiffusionTrainingConfigSchema(StrategySchema):
    """Diffusion-specific training parameters (sub-section of training).

    Located at: training.diffusion
    Used when training_mode = 'diffusion' or similar.

    StrategySchema (extra='allow', frozen) — forwards diffusion-specific knobs
    read downstream via getattr; now immutable (was silently mutable).
    """

    # Core diffusion parameters.
    #
    # The bound and the enum below are HARVESTED from `TrainingConfigDiffusion`
    # (`config/schemas/training/diffusion.py`), the better-designed schema for
    # this same block that no arm has ever mounted -- its only constructor,
    # `create_training_config()`, has zero production callers. It was strict
    # where the class arms actually use was loose, so the strictness sat in
    # dead code for as long as the dead code did.
    #
    # Both were verified free before landing: across all 725 constructible arms,
    # `noise_schedule` takes only `cosine` (167) and `linear` (33) -- both enum
    # members -- and no arm declares `timesteps < 1`.
    timesteps: int | None = Field(
        default=1000,
        ge=1,
        description="Number of diffusion timesteps",
    )
    #: A CLOSED vocabulary, not a free string. As `str | None` this accepted any
    #: value, so `noise_schedule: cosnie` parsed happily and the scheduler fell
    #: through to whatever its own default was -- a silent fallback on a knob
    #: that decides the forward process (non-negotiable #3). `NoiseSchedule` is
    #: a `str` Enum, so downstream `== "cosine"` comparisons are unaffected.
    noise_schedule: NoiseSchedule | None = Field(
        default=NoiseSchedule.COSINE,
        description="Noise schedule: linear, cosine, sqrt, quadratic",
    )
    # #799: `DiffusionTrainingStrategy.__init__` forwards these two to the noise
    # scheduler unconditionally, but this -- the class `training.diffusion`
    # actually mounts -- did not declare them. `extra='allow'` meant an arm that
    # spelled them in YAML worked and the other 30-odd raised AttributeError at
    # construction. Values and bounds mirror `TrainingConfigDiffusion`, the
    # schema that always carried them, so a declaring arm is unaffected.
    beta_start: float = Field(
        default=0.0001, ge=0, le=1, description="Starting value for noise schedule"
    )
    beta_end: float = Field(default=0.02, ge=0, le=1, description="Ending value for noise schedule")

    # Sampling
    sampler: str | None = Field(
        default="ddpm",
        description="Sampler: ddpm, ddim, predictor_corrector, dpm_solver",
    )
    sampling_steps: int | None = Field(
        default=100, description="Number of sampling steps for inference"
    )

    # Conditioning
    cond_drop_prob: float | None = Field(
        default=0.1, description="Conditioning dropout probability"
    )
    guidance_scale: float | None = Field(
        default=7.5,
        description="Classifier-free guidance scale (SSOT in training schema)",
    )
    condition_on_input: bool = Field(
        default=False,
        description=(
            "Concatenate the (low-res / ULF) input batch onto the noised target "
            "along the channel axis before the denoiser, turning standard "
            "image-domain diffusion into measurement-conditioned reconstruction. "
            "The model's in_channels must account for the extra channels (noisy "
            "target + condition). Default False keeps the historical "
            "unconditional behaviour; skipped for cold/latent diffusion and when "
            "smaps were already concatenated."
        ),
    )

    # Model specifics
    type: str | None = Field(
        default=None,
        description="Diffusion type: ColdDiffusion, GaussianDiffusion, LatentDiffusion",
    )
    prediction_type: str | None = Field(
        default="epsilon", description="Prediction type: epsilon, sample, v_prediction"
    )

    # Degradation
    degradation: str | None = Field(
        default=None,
        description=(
            "Degradation type: physics, gaussian, kspace_mask, blur, rician. "
            "GraphColdDiffusionStrategy reads it (None = kspace_mask; it accepts "
            "kspace_mask and rician and raises on anything else)."
        ),
    )
    rician_sigma_max: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Rician cold-diffusion forward process: noise std at t = T-1, as a "
            "fraction of each image's p99 magnitude; sigma_t = rician_sigma_max * "
            "t / (T-1). Required by, and only accepted with, degradation: rician."
        ),
    )
    validation_timestep: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Degradation level GraphColdDiffusionStrategy tells the model the "
            "validation input sits at (None = 0, the clean end, where a cold model "
            "learns the identity). Must be < timesteps."
        ),
    )
    degradation_source: str = Field(
        default="input",
        description=(
            "Which tensor the cold-diffusion forward process degrades: 'input' "
            "(the measurement, e.g. M4Raw rep[0]) or 'target' (the clean "
            "NEX-averaged supervision signal). Must match what VALIDATION "
            "degrades, which is always the input -- re-masking the target at "
            "validation leaks ground truth. Training used to degrade the TARGET "
            "unconditionally (issue #536), so the model learned to inpaint "
            "noise-free k-space and was then graded on inpainting AND denoising "
            "single-rep k-space, a distribution it never saw. Default 'input' is "
            "the validation-consistent choice; 'target' reproduces pre-fix runs."
        ),
    )

    # Numerical stability
    enable_diffusion_amp: bool | None = Field(default=False, description="Enable AMP for diffusion")
    enforce_output_range: bool | None = Field(default=True, description="Enforce output range")

    # Loss weighting
    lambda_mse: float | None = Field(default=1.0, description="MSE loss weight")

    @field_validator("degradation_source")
    @classmethod
    def _validate_degradation_source(cls, value: str) -> str:
        """Reject an unadvertised source at load time (pitfall #9, no fallback)."""
        allowed = {"input", "target"}
        if value not in allowed:
            raise ValueError(
                f"training.diffusion.degradation_source must be one of "
                f"{sorted(allowed)}, got {value!r}."
            )
        return value

    @model_validator(mode="after")
    def _rician_sigma_needs_rician(self) -> "DiffusionTrainingConfigSchema":
        """A Rician noise ceiling is read only by the Rician forward process."""
        rician = self.degradation == "rician"
        if rician and self.rician_sigma_max is None:
            raise ValueError("training.diffusion.degradation: rician needs rician_sigma_max.")
        if not rician and self.rician_sigma_max is not None:
            raise ValueError(
                f"training.diffusion.rician_sigma_max is read only with degradation: "
                f"rician, not {self.degradation!r}."
            )
        level, horizon = self.validation_timestep, self.timesteps
        if level is not None and horizon is not None and level >= horizon:
            raise ValueError(
                f"training.diffusion.validation_timestep={level} must be < timesteps={horizon}."
            )
        return self

    # The first NESTED rename mount in the codebase (`training.diffusion`, not
    # `training`). Mounting it is not optional bookkeeping: `renames_for_block`
    # serves a record to the validator whose mount path matches, so a record
    # declared with `mount="training.diffusion"` and no validator here is
    # registered, reachable, and silently never fires -- the exact three-claim
    # gap the reachability contract is about.
    # pydantic accepts the classmethod at runtime; its protocol types cannot describe one.
    _fold_renamed = model_validator(mode="before")(
        classmethod(fold_renamed_keys("training.diffusion"))  # type: ignore[arg-type]
    )
    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("training.diffusion"))  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# The paradigm selector: a discriminated union over `training.diffusion.type`
# --------------------------------------------------------------------------- #
#
# `training.diffusion` is meant to be the interface between the user and the
# creation patterns -- it should say WHICH generative paradigm trains this arm
# and select a construction path. As one `extra="allow"` class it did neither: a
# typo became a live untyped attribute that downstream `getattr` read, and `type`
# was a free string nothing branched on.
#
# The union IS the factory. An unknown tag raises at parse with every valid tag
# named, which satisfies non-negotiable #3 and `restruct.md`'s "factory over
# if/elif chains" without a single `if`.
#
# WHY THESE LIVE HERE rather than in the `config/schemas/training/diffusion_paradigm.py`
# the plan suggested: every variant must inherit `DiffusionTrainingConfigSchema`
# (for its 17 shared fields AND its rename validators), and `base.py` already
# imports its siblings -- so a separate module importing the base back would be a
# cycle. Defining them beside the base is the cycle-free option and keeps code
# that changes together in one place.
#
# MEMBERSHIP FOLLOWS WHAT THE FRAMEWORK IMPLEMENTS, NOT WHAT THE CORPUS USES.
# `ddim` and `chi_square` have zero arms and are members anyway: the `ddim`
# sampler is registered and chi-square has three implementations, so the reverse
# processes exist and only the config vocabulary was missing. Withholding a tag
# makes `type: ddim` raise for a user exploring it, which is the opposite of what
# a research framework is for. This is NOT in tension with phase 1.3, which
# deleted names that duplicated a canonical key or bound to the wrong class:
# those made the framework ambiguous, an unused paradigm makes it larger.
#
# STRICTNESS IS PER VARIANT, AND IT IS MEASURED, NOT ASSUMED. Corpus census of
# keys each paradigm's arms declare that the schema does not (2026-08-12, after
# the `num_timesteps` fold removed the largest one):
#
#   score_based (11 arms)      0 undeclared keys  -> extra="forbid", free
#   ddpm (4)                   0                  -> extra="forbid", free
#   rectified_flow (1)         0                  -> extra="forbid", free
#   ddim (0), chi_square (0)   0 by construction  -> extra="forbid"
#   cold (108)                11 undeclared keys  -> extra="allow", see below
#   latent_diffusion (1)       3                  -> extra="allow"
#   untagged (75)              4                  -> extra="allow"
#
# A zero-arm variant starts strict deliberately: there is no legacy YAML to
# accommodate, so the first arm to use it gets typed validation from day one
# rather than inheriting a permissive block it never needed.
#
# `cold` and `latent` stay permissive TRANSITIONALLY because three of their keys
# -- `clip_sample`, `clip_sample_range`, `dynamic_thresholding_func` -- have no
# reader anywhere in `src/`. None of the three available homes is honest for
# them: a typed field advertises an unread knob (#8), a `raise` rename record
# makes live `inprogress/` arms unloadable, and silently dropping them is the
# defect this whole cleanup exists to remove. They need an owner decision, and
# until then the variant says so rather than pretending.


class UnspecifiedParams(DiffusionTrainingConfigSchema):
    """Transitional variant for the 75 arms that declare no ``type``.

    Reproduces today's behaviour exactly, so landing the union is
    behaviour-neutral. An arm gains strictness the moment it declares a tag.

    The discriminator CANNOT simply be made required: many of these are
    non-diffusion arms carrying a vestigial block, and requiring the tag would
    make them unloadable. Drain this variant via the corpus gate, then delete it
    and make ``type`` required -- the same ratchet shape as ``fold`` -> ``raise``.
    """

    type: Literal[None] = None


class ColdParams(DiffusionTrainingConfigSchema):
    """Cold diffusion: deterministic degradation instead of Gaussian noise."""

    type: Literal["cold"] = "cold"


class DDPMParams(DiffusionTrainingConfigSchema):
    """Ancestral DDPM sampling."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())
    type: Literal["ddpm"] = "ddpm"


class ScoreBasedParams(DiffusionTrainingConfigSchema):
    """Score-based / SDE diffusion."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())
    type: Literal["score_based"] = "score_based"


class DDIMParams(DiffusionTrainingConfigSchema):
    """Deterministic implicit sampling.

    Zero arms declare this today. That is not a reason to withhold it: the
    ``ddim`` sampler IS registered, so the reverse process exists and only the
    config vocabulary was missing -- without this variant `type: ddim` raises,
    which closes the framework against exactly the exploration it is for.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())
    type: Literal["ddim"] = "ddim"


class RectifiedFlowParams(DiffusionTrainingConfigSchema):
    """Rectified flow -- straight-line transport rather than a noising chain."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())
    type: Literal["rectified_flow"] = "rectified_flow"


class ChiSquareParams(DiffusionTrainingConfigSchema):
    """Chi-square degradation, for magnitude/multi-coil noise statistics.

    Zero arms today, three implementations behind it: ``ChiSquareColdDiffusion``,
    and the ``chi_square_diffusion`` / ``simple_chi_square_diffusion`` models.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())
    type: Literal["chi_square"] = "chi_square"


class LatentDiffusionParams(DiffusionTrainingConfigSchema):
    """Diffusion in a learned latent space."""

    type: Literal["latent_diffusion"] = "latent_diffusion"


#: The paradigm selector. Discrimination requires the tag to be PRESENT, so
#: `TrainingStrategyConfigSchema._tag_diffusion_paradigm` injects the untagged default
#: before validation -- without it, the 75 untagged arms raise "Unable to extract
#: tag using discriminator 'type'".
DiffusionParadigmParams = Annotated[
    UnspecifiedParams
    | ColdParams
    | DDPMParams
    | DDIMParams
    | ScoreBasedParams
    | RectifiedFlowParams
    | LatentDiffusionParams
    | ChiSquareParams,
    Field(discriminator="type"),
]


class LatentTrainingConfigSchema(StrategySchema):
    """Latent/VAE-specific training parameters (sub-section of training).

    Located at: training.latent
    Used when training_mode = 'vae', 'vqvae', etc.

    NOTE: latent_dim is SSOT in training schema (objectives is deprecated).

    StrategySchema (extra='allow', frozen) — forwards latent-specific knobs
    read downstream via getattr; now immutable (was silently mutable).

    KL-annealing knobs accept two spellings (audit 2026-07-18). Experiment
    YAMLs were authored with ``enable_kl_annealing`` / ``kl_anneal_start`` /
    ``kl_anneal_end`` while the resolver in
    ``models/losses/computers/unified_vae.py`` reads ``anneal_kl_beta`` /
    ``kl_beta_start`` / ``kl_beta_end``. The two never met, so every arm that
    asked for a KL warm-up got beta pinned at ``kl_beta_end`` from step 0 —
    the classic posterior-collapse setup that warm-up exists to prevent.
    ``AliasChoices`` folds both spellings onto one canonical field so the
    weight has a single resolver (pitfall #13b applied to the KL term).
    """

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        protected_namespaces=(),
        # Both the canonical field name and its aliases populate the field.
        populate_by_name=True,
    )

    # Core latent parameters
    latent_dim: int | None = Field(default=256, description="Latent dimension (SSOT)")
    beta_kl: float | None = Field(default=1.0, description="KL divergence weight")
    commitment_weight: float | None = Field(default=0.25, description="VQ commitment weight")
    codebook_weight: float | None = Field(default=1.0, description="Codebook loss weight")
    n_embeddings: int | None = Field(default=512, description="Number of VQ embeddings")
    embedding_dim: int | None = Field(default=64, description="Embedding dimension")

    # KL annealing. Aliases keep the YAML spelling (`enable_kl_annealing`,
    # `kl_anneal_start`, `kl_anneal_end`) and the resolver spelling on ONE field.
    anneal_kl_beta: bool | None = Field(
        default=False,
        validation_alias=AliasChoices("anneal_kl_beta", "enable_kl_annealing"),
        description="Anneal KL beta during training (alias: enable_kl_annealing)",
    )
    kl_beta_start: float | None = Field(
        default=0.0,
        validation_alias=AliasChoices("kl_beta_start", "kl_anneal_start"),
        description="Starting KL beta value (alias: kl_anneal_start)",
    )
    kl_beta_end: float | None = Field(
        default=1.0,
        validation_alias=AliasChoices("kl_beta_end", "kl_anneal_end"),
        description="Ending KL beta value (alias: kl_anneal_end)",
    )
    kl_anneal_steps: int | None = Field(
        default=10000, description="Number of steps for KL annealing"
    )

    # Regularization
    use_latent_regularization: bool | None = Field(
        default=True, description="Use latent space regularization"
    )
    latent_regularization_weight: float | None = Field(
        default=0.01, description="Latent regularization weight"
    )

    # Loss type
    latent_loss_type: str | None = Field(
        default="kl_divergence",
        description="Latent loss type: kl_divergence, mmd, wasserstein",
    )
    lambda_kl: float | None = Field(default=0.0, description="KL loss weight")
    lambda_vq: float | None = Field(default=0.0, description="VQ loss weight")
    lambda_commit: float | None = Field(default=0.0, description="Commit loss weight")
    temperature: float | None = Field(default=0.1, description="Temperature for softmax")


class MultiParameterTrainingConfigSchema(BaseModel):
    """One-shot multi-parameter mapping (idea 10).

    Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §10.

    Wires the MultiParameterHead + UncertaintyWeightedMultiTaskLoss +
    Bloch-self-consistency regulariser. Used by
    ``OneShotMultiParameterStrategy``.
    """

    model_config = {"extra": "ignore", "frozen": True}

    parameters: list[str] = Field(
        default_factory=lambda: ["t1", "t2", "pd", "segmentation"],
        description="Heads to estimate from a single multi-contrast acquisition.",
    )
    uncertainty_weighting: bool = Field(
        default=True,
        description="Use Kendall-style learnable uncertainty weighting across heads.",
    )
    bloch_consistency_lambda: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight on the Bloch signal-synthesis consistency loss.",
    )
    n_contrasts: int = Field(
        default=3,
        ge=1,
        description="Number of input contrasts (must match data.multi_contrast.acquisition_params).",
    )


class PnPTrainingConfigSchema(BaseModel):
    """Plug-and-Play / RED training configuration (PR-9)."""

    model_config = {"extra": "ignore", "frozen": True}

    enabled: bool = Field(default=False)
    iterations: int = Field(default=50, ge=1)
    rho: float = Field(default=1.0, gt=0.0, description="ADMM penalty parameter.")
    spectral_norm_iters: int = Field(
        default=1,
        ge=1,
        description="Spectral-norm power-iteration steps per forward (Lipschitz control).",
    )
    denoiser_lipschitz_constant: float = Field(default=1.0, gt=0.0)


class GANSubConfigSchema(CompatSchema):
    """GAN training sub-parameters nested under ``training.gan``.

    Currently carries the progressive-growing GAN controls read by
    ``ProgressiveGANStrategy``. Defined in this module (not ``gan.py``) to
    avoid a circular import: ``gan.py`` already imports from here.

    CompatSchema (extra='ignore', frozen) — now immutable (was silently mutable).
    """

    phase_schedule: list[int] | None = Field(
        default=None,
        description=(
            "Per-phase step counts for progressive-growing GAN fade-in "
            "(e.g. [10000, 20000, 40000]). Entries must be positive; their "
            "sum should approximate training.max_iterations. When omitted, "
            "ProgressiveGANStrategy spreads phases uniformly over max_iterations."
        ),
    )
    progan_gradient_penalty_lambda: float = Field(
        default=10.0,
        ge=0.0,
        description="WGAN-GP gradient-penalty weight for progressive GAN training.",
    )
    beta_tc_weight: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Outer mixing weight for the β-TC-VAE disentanglement term added "
            "to the generator loss by BetaVAEGANStrategy (the total-correlation "
            "strength itself is set by the model's `beta`)."
        ),
    )


class TrainingStrategyConfigSchema(BaseModel):
    """Central configuration for training strategy selection.

    This is the NEW way to configure training - the `training:` section
    in experiment YAML becomes the single source of truth for strategy dispatch.

    Example:
        training:
          strategy_class: "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
          lambda_recon: 10.0

        # Multi-stage pipeline example:
        training:
          strategy_class: "pipeline"
          pipeline:
            stages:
              - name: encoder
                model_type: standard_unet
                training_state: { freeze_all: true }
              - name: decoder
                model_type: swin_unet
            routing:
              - from_stage: encoder
                to_stage: decoder
    """

    model_config = {
        "extra": "allow",
        "frozen": True,
        # Four of the paradigm sub-blocks below live in modules that import
        # `BaseTrainingConfigSchema` / `TrainingConfigDiffusion` from THIS file
        # (for a different class in the same file), so importing them at the top
        # cycles. `defer_build` postpones schema construction until the explicit
        # `model_rebuild()` at the bottom of this module, by which point those
        # modules can import cleanly. Without it pydantic raises
        # `PydanticUndefinedAnnotation` at class creation.
        "defer_build": True,
    }  # strategy-specific fields; immutable (CLAUDE.md #4)

    # `training.seed` was one of TWO spellings of one seed, and the one that
    # WON at runtime. Both it and the root `seed:` now name `run.seed`. The
    # shim is load-bearing here rather than cosmetic: this class is
    # `extra="allow"`, so simply deleting the field would let `training.seed`
    # keep parsing, silently, while nothing read it.
    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("training"))  # type: ignore[arg-type]
    )

    @model_validator(mode="before")
    @classmethod
    def _tag_diffusion_paradigm(cls, data: Any) -> Any:
        """Normalise the paradigm tag before the union discriminates on it.

        Two jobs, and neither may swallow a bad value:

        1. **Inject the untagged default.** A discriminated union needs the tag
           PRESENT -- an absent one raises "Unable to extract tag using
           discriminator 'type'", which would make the 75 untagged arms
           unloadable. `setdefault` selects `UnspecifiedParams` for them.
        2. **Normalise retired spellings** via the closed
           `DIFFUSION_PARADIGM_ALIASES` map, so `type: cold_diffusion` reaches
           `ColdParams` rather than dying on an unknown tag.

        Deliberately a closed map, never a fuzzy match: an unrecognised tag must
        still raise with the valid set named (non-negotiable #3). A normaliser
        that coerced anything unfamiliar to a default would reintroduce the
        silent fallback this union exists to remove.

        NOTE this does NOT affect inference routing. `strategy_detector` accepts
        both spellings directly (#991), so normalisation here cannot silently
        re-route an arm -- that coupling was measured and severed first.
        """
        if not isinstance(data, dict):
            return data
        block = data.get("diffusion")
        if not isinstance(block, dict):
            return data

        tag = block.get("type")
        normalised = DIFFUSION_PARADIGM_ALIASES.get(tag) if isinstance(tag, str) else None
        if normalised is None and "type" in block:
            return data  # tag present and not an alias -- let the union judge it

        updated = dict(block)
        if normalised is not None:
            updated["type"] = normalised
        else:
            updated["type"] = None
        return {**data, "diffusion": updated}

    # REQUIRED: Fully qualified path to strategy class
    strategy_class: str | None = Field(
        default=None,  # Optional for backward compatibility
        description="Fully qualified path to strategy class, e.g. "
        "'spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy'",
    )

    # Universal keys that arrive on nearly every arm and ARE read. They were
    # reaching the model through `extra="allow"`, so they carried no type and
    # the ledger classified them as untyped extras -- for `training_mode`, the
    # key 49 call sites dispatch the whole paradigm on.
    #
    # Only the three with readers are declared. `enable_gradient_checkpointing`
    # and `num_workers` had zero readers and are refused at load (removals.py,
    # #1135 / #557). Three more are equally universal -- `batch_size` (401 arms),
    # `enable_mixed_precision` (321), `max_steps` (85) -- and have **zero**
    # readers in `src/`. Declaring those would advertise a
    # knob nothing consumes, which non-negotiable #8 forbids outright; it would
    # also make the audit go quiet about 20 arms that run at a batch size they
    # never asked for and 84 that run at an AMP setting they never asked for.
    # Their fix is a fold onto the canonical path, which changes what those arms
    # compute and so is an owner decision -- issue #887.
    #
    # `enable_mixed_precision` (2026-09-03): the `inprogress/` corpus is drained
    # (238 arms; nine hilbert_mamba arms had declared `false` against a live
    # `optimization.precision.enabled: true` and had been running with AMP on),
    # and the `no_dead_precision_flag` witness reports the spelling on the audit
    # surface. A `raise` rename record is corpus-wide and 84 files under
    # `active/`, `validated/`, `experiments/training/`, `experiments/templates/`
    # and test fixtures still carry the line; a fold cannot cross blocks. Those
    # trees are the owner's to drain, after which the record replaces the witness.
    #
    # Typed `str` rather than an enum on purpose: `SignalDomain` does not
    # contain `feature`, and 471 corpus arms declare `output_domain: feature`.
    # A closed enum here would reject them at load.
    training_mode: str | None = Field(
        default=None,
        description="Paradigm key resolved through "
        "`TrainingStrategyFactory.STRATEGY_CLASS_PATHS`; the alternative to "
        "naming `strategy_class` outright.",
    )
    input_domain: str | None = Field(
        default=None,
        description="Domain the model consumes ('image', 'kspace').",
    )
    output_domain: str | None = Field(
        default=None,
        description="Domain the model emits ('image', 'kspace', 'feature').",
    )

    @model_validator(mode="after")
    def _graph_cold_knobs_need_their_reader(self) -> "TrainingStrategyConfigSchema":
        """Refuse the GraphCold-only diffusion knobs on any other strategy (NN8)."""
        block = self.diffusion
        if block is None:
            return self
        declared = [
            name
            for name, on in (
                ("degradation: rician", block.degradation == "rician"),
                ("validation_timestep", block.validation_timestep is not None),
            )
            if on
        ]
        owner = (self.strategy_class or "").rsplit(".", 1)[-1]
        if declared and owner != "GraphColdDiffusionStrategy":
            raise ValueError(
                f"training.diffusion {declared} is read only by GraphColdDiffusionStrategy; "
                f"declare it as training.strategy_class (got {self.strategy_class!r})."
            )
        return self

    # [REMOVED] Flat training keys (v5.0 strictness)
    # legacy fields like diffusion_type, timesteps, etc. are now strictly nested
    # under the appropriate subsection (diffusion, latent, ssl).

    @model_validator(mode="before")
    @classmethod
    def reject_flat_keys(cls, data: Any) -> Any:
        """Reject legacy flat keys and guide user to nested sections."""
        if not isinstance(data, dict):
            return data

        legacy_map = {
            "diffusion_type": "diffusion.type",
            "timesteps": "diffusion.timesteps",
            "degradation_type": "diffusion.degradation",
            "lambda_recon": "losses.reconstruction.lambda_l1",
            "forward_operator": "physics.forward_model",
            "trajectory_type": "data.trajectory",
        }

        errors = []
        for key, replacement in legacy_map.items():
            if key in data:
                errors.append(f"- {key}: Use {replacement} instead")

        if errors:
            raise ValueError(
                "Legacy flat keys found in training config (DEPRECATED):\n"
                + "\n".join(errors)
                + "\nPlease use nested sections instead (e.g., diffusion.timesteps)."
            )
        return data

    # Migrated Top-Level Fields
    epochs: int | None = Field(None, description="Number of training epochs")
    max_iterations: int | None = Field(None, description="Maximum total iterations")
    iteration_budget_scope: Literal["per_rank", "global"] = Field(
        "per_rank",
        description=(
            "Whether max_iterations is a PER-RANK loop bound (default, today's "
            "behaviour) or a GLOBAL budget to be divided by world_size. Declared "
            "because this schema is extra='allow': an undeclared "
            "'iteration_budget_scope: gloabl' is silently swallowed, whereas a "
            "Literal turns the typo into a load-time error. 'global' currently "
            "RAISES -- see pipelines/training_loop.py for why (issue #1163)."
        ),
    )
    max_steps_per_epoch: int | None = Field(None, description="Max steps per epoch")
    task: str | None = Field(None, description="Task type (reconstruction, etc)")
    device: str | None = Field(None, description="Device (cuda/cpu)")
    output_dir: str | None = Field(None, description="Output directory")

    # Diffusion-curriculum learning knobs (Phase 8 promotion from the
    # diffusion sub-schema). Real experiment YAMLs set these at the
    # training top level (see e.g. experiment_11_kspace_cold_diffusion),
    # so they belong here for validation rather than being silently
    # accepted via ``extra="allow"``. The diffusion strategy reads them
    # via ``self.config.training.curriculum_*`` -- no code change needed.
    curriculum_start_timestep: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Starting maximum timestep for diffusion curriculum (light "
            "degradation). When None the strategy defaults to no curriculum."
        ),
    )
    curriculum_ramp_rate: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Timesteps added per iteration for diffusion curriculum "
            "(e.g. 0.01 means 10 steps per 1000 iterations). When None "
            "the strategy defaults to no curriculum."
        ),
    )

    # ✅ NEW: Paradigm-specific sub-sections (moved from top-level)
    # These are now nested under training to organize strategy-specific parameters
    diffusion: DiffusionParadigmParams | None = Field(
        default=None,
        description=(
            "Which generative paradigm trains this arm, and its parameters. "
            "Discriminated on `type`; an unknown tag raises with the valid set "
            "named. Omitting `type` selects the transitional UnspecifiedParams."
        ),
    )
    latent: LatentTrainingConfigSchema | None = Field(
        default=None,
        description="Latent/VAE-specific training parameters (for vae/vqvae training modes)",
    )
    # WIRED 2026-08-12 (issue #996). These three schemas existed, fully written
    # and bounded, but were reachable only through `create_training_config()` --
    # which has zero production callers, so no YAML could ever select them. They
    # are not dead code; they are the pre-decomposition hierarchy's description
    # of blocks the live class simply never declared, so an arm writing
    # `training.flow:` had it absorbed by `extra="allow"` as a raw dict and every
    # `getattr(cfg.flow, knob, default)` in the strategy resolved to the default.
    #
    # Zero arms declare any of the three today, which is exactly why they go
    # first: wiring them is provably behaviour-neutral (the golden gate stays
    # byte-identical) while making the paradigms configurable for the first
    # time. The blocks that ARE declared by arms are wired per-block afterwards,
    # each with its own delta table, because those DO change behaviour.
    flow: "FlowConfig | None" = Field(
        default=None,
        description="Flow-matching / rectified-flow parameters (training_mode: flow)",
    )
    motion: "MotionConfig | None" = Field(
        default=None,
        description="Motion-correction parameters (training_mode: motion)",
    )
    federated_config: "FederatedConfig | None" = Field(
        default=None,
        description="Federated-learning parameters (training_mode: federated)",
    )
    # `training.vae` was read by unified_vae.py but never DECLARED, so
    # extra='allow' admitted it as a raw dict and every `getattr` against it
    # returned None (audit 2026-07-18). Declaring it makes the block validate
    # and turns attribute access into a real read (pitfall #15).
    vae: LatentTrainingConfigSchema | None = Field(
        default=None,
        description=(
            "VAE-specific training parameters. Same schema as `latent`; "
            "`training.vae` takes precedence where both are present."
        ),
    )
    multi: MultiTrainingConfigSchema | None = Field(
        default=None,
        description="Multi-stage training (cascaded/stitched models with DAG flow)",
    )
    geomamba_ulf: GeoMambaULFTrainingConfigSchema | None = Field(
        default=None,
        description=(
            "GeoMamba-ULF paradigm parameters (metric SFC, FiLM contrast embedding, "
            "physical unwarp composition, topology losses). Used when "
            "training.strategy_class resolves to GeoMambaULFStrategy."
        ),
    )
    levy_denoising: LevyDenoisingTrainingConfigSchema | None = Field(
        default=None,
        description=(
            "Conditional alpha-stable denoising parameters (noise law, noise range, "
            "validation sampler). Required by LevyDenoisingStrategy."
        ),
    )
    tall_sr: TALLSRTrainingConfigSchema | None = Field(
        default=None,
        description=(
            "TALL-SR paradigm parameters (differentiable locality regulariser). "
            "Read by ReconstructionTrainingStrategy when model_type is tall_sr."
        ),
    )
    ssdu: SSDUTrainingConfigSchema | None = Field(
        default=None,
        description=(
            "Self-supervised SSDU reconstruction strategy sub-block (Lambda/Theta "
            "k-space split). Used when training_mode is "
            "self_supervised_reconstruction."
        ),
    )
    equivariant_imaging: TrainingConfigEquivariantImaging | None = Field(
        default=None,
        description=(
            "Equivariant Imaging (EI) self-supervised reconstruction sub-block "
            "(group, alpha_equivariance, robust_correction). Used when the "
            "strategy_class resolves to EquivariantImagingStrategy "
            "(equivariant_imaging / robust_ei)."
        ),
    )
    ambient: AmbientTrainingConfigSchema | None = Field(
        default=None,
        description=(
            "Ambient Diffusion sub-block (theta_fraction, ambient_weight, "
            "denoise_weight). Used when the strategy_class resolves to "
            "AmbientDiffusionStrategy (ambient_diffusion); the SSDU Λ/Θ split "
            "lifted onto a diffusion prior."
        ),
    )
    low_rank_sparse: LowRankSparseTrainingConfigSchema | None = Field(
        default=None,
        description=(
            "Low-rank + sparse (RPCA) decomposition hyperparameters "
            "(tau_low_rank, tau_sparse, lambda_nuclear, lambda_sparse, "
            "lambda_consistency). Read by LowRankSparseStrategy at "
            "construction. When omitted, the strategy uses the documented "
            "defaults that equal its historical hard-coded values."
        ),
    )
    teichmuller: TeichmullerTrainingConfigSchema | None = Field(
        default=None,
        description=(
            "Teichmüller cold-diffusion schedule-head + Beltrami-endpoint "
            "parameters (r_max, spatial_size, n_steps, lambda_teichmuller, "
            "mu_t_max, lambda_degrade). Read by "
            "TeichmullerColdDiffusionStrategy at construction time, which "
            "RAISES when the block is absent (no silent fallback)."
        ),
    )

    # Phase 3 (2026-05-22) — progressive-growing GAN controls. Read by
    # ProgressiveGANStrategy via ``config.training.gan.phase_schedule``.
    gan: GANSubConfigSchema | None = Field(
        default=None,
        description=(
            "GAN training sub-parameters (progressive-growing phase schedule, "
            "GP weight). Used when training.strategy_class resolves to "
            "ProgressiveGANStrategy."
        ),
    )

    # v6.1 — multi-parameter mapping (integration plan §10)
    multi_parameter: MultiParameterTrainingConfigSchema | None = Field(
        default=None,
        description="One-shot multi-parameter mapping (T1/T2/PD/segmentation) configuration.",
    )

    # v6.1 — Plug-and-Play / RED (PR-9)
    pnp: PnPTrainingConfigSchema | None = Field(
        default=None,
        description="Plug-and-Play / RED ADMM iterates with spectral-norm-bounded denoiser.",
    )
    distillation: DistillationTrainingConfigSchema | None = Field(
        default=None,
        description="Teacher-student distillation: the frozen teacher's checkpoint.",
    )

    # MICCAI MRIxFields2026 — encode-once/render-anywhere cross-field translation
    # (B-3.8 / B-1.9). Read by CrossFieldTranslationStrategy via
    # ``config.training.cross_field``.
    cross_field: CrossFieldConfig | None = Field(
        default=None,
        description=(
            "Cross-field translation knobs (latent-cycle weight, contrast "
            "conditioning) for CrossFieldTranslationStrategy."
        ),
    )
    field_cocycle: FieldCocycleConfig | None = Field(
        default=None,
        description=(
            "Cocycle-consistent unified cross-field operator knobs (reference field, "
            "cocycle/identity/adversarial weights, triple sampling) for "
            "FieldCocycleTranslationStrategy (idea 4.2, Task 3)."
        ),
    )
    field_flow: FieldFlowConfig | None = Field(
        default=None,
        description=(
            "Neural-ODE field-flow knobs (integration steps/solver, straightness "
            "weight) for FieldFlowStrategy (B-3.1)."
        ),
    )
    field_bridge: FieldBridgeConfig | None = Field(
        default=None,
        description=(
            "Field-conditioned Schrödinger-bridge knobs (entropic noise scale, "
            "sampler steps) for FieldBridgeStrategy (B-3.3)."
        ),
    )
    cycle_bloch: CycleBlochConfig | None = Field(
        default=None,
        description=(
            "Cycle-Bloch backward operator (bloch_fixed | ulf_forward) and its ULF "
            "sequence for CycleBlochStrategy."
        ),
    )
    ulf_map: UlfMapConfig | None = Field(
        default=None,
        description=(
            "Ultra-low-field physics-noise MAP knobs (source field, Hoult sigma "
            "floor) for UlfMapStrategy (B-2.1); ADMM iters/rho/denoiser come from "
            "training.pnp."
        ),
    )
    heteroscedastic_ulf: HeteroscedasticULFConfig | None = Field(
        default=None,
        description=(
            "Heteroscedastic ULF restoration knobs (Hoult sigma floor + variance-"
            "prior weight) for HeteroscedasticULFStrategy (B-2.9)."
        ),
    )
    field_cold_diffusion: FieldColdDiffusionConfig | None = Field(
        default=None,
        description=(
            "Field-degradation cold-diffusion knobs (steps + low-pass cutoff range) "
            "for FieldColdDiffusionStrategy (B-2.4)."
        ),
    )
    field_guided_diffusion: FieldGuidedDiffusionConfig | None = Field(
        default=None,
        description=(
            "Continuous-field classifier-free-guidance diffusion knobs (timesteps, "
            "schedule, guidance_prob/scale) for FieldGuidedDiffusionStrategy (B-3.5)."
        ),
    )
    ulf_dps: UlfDpsConfig | None = Field(
        default=None,
        description=(
            "ULF-consistent diffusion-posterior-sampling knobs (likelihood_weight, "
            "sampling_steps) for UlfDpsStrategy (B-2.2)."
        ),
    )
    generative_refiner: GenerativeRefinerConfig | None = Field(
        default=None,
        description=(
            "Generative-iterative refiner knobs (timesteps, sampling_steps, "
            "guidance_weight, forward_operator) for GenerativeRefinerStrategy — the "
            "shared Track-A chassis that generalizes the b22 ulf_dps ceiling-break "
            "(MRIxFields2026 saturation work). guidance_weight=0 is the one-knob ablation."
        ),
    )
    field_conditioned_inr: FieldConditionedINRConfig | None = Field(
        default=None,
        description=(
            "Field-conditioned SIREN-INR super-resolution knobs "
            "(use_field_conditioning, lambda_l1) for FieldConditionedINRStrategy (B-2.8)."
        ),
    )
    monotone_field: MonotoneFieldConfig | None = Field(
        default=None,
        description=(
            "Monotone field-ordered generator knobs (lambda_l1, lambda_monotone) for "
            "MonotoneFieldStrategy (B-2.6); enforce_monotone lives on model_kwargs."
        ),
    )
    ulf_redegrad_tta: UlfReDegradationTTAConfig | None = Field(
        default=None,
        description=(
            "Re-degradation-consistency test-time-adaptation knobs (adaptation_steps/lr, "
            "consistency_weight) for UlfReDegradationTTAStrategy (B-2.10)."
        ),
    )
    ulf_lowpass: UlfLowpassConfig | None = Field(
        default=None,
        description=(
            "Cutoff of the ULF low-pass forward operator A shared by ulf_map, ulf_dps, "
            "ulf_redegrad_tta and generative_refiner (field_lowpass); those strategies "
            "raise when it is absent."
        ),
    )
    quality_matching: QualityMatchingConfig | None = Field(
        default=None,
        description=(
            "HQ->LQ quality-matched degradation synthesis (axes, target, fit budget) "
            "for QualityMatchingStrategy. Declared optional, so consumers MUST test "
            "`is not None` -- hasattr on a declared optional is always True."
        ),
    )
    field_fno: FieldFNOConfig | None = Field(
        default=None,
        description=(
            "Field-conditioned spectral transfer operator (FNO) knobs (lambda_l1) for "
            "FieldFNOStrategy (B-1.6/3.7); field_token_enable lives on model_kwargs."
        ),
    )
    bloch_field: BlochFieldConfig | None = Field(
        default=None,
        description=(
            "Bloch quantitative-parameter-bottleneck knobs (lambda_l1) for "
            "BlochFieldStrategy (B-1.8); use_field_dispersion lives on model_kwargs."
        ),
    )
    phase_contrast: PhaseContrastFlowConfig | None = Field(
        default=None,
        description=(
            "Mechanics for PhaseContrastFlowStrategy (regime mri_flow). The "
            "physics (venc, encoding scheme, flux masks) lives on "
            "data.phase_contrast; the weights on losses.physics.lambda_*. NOT "
            "flow matching — that is `flow` / FlowConfig."
        ),
    )
    perfusion_kinetic: PerfusionKineticConfig | None = Field(
        default=None,
        description=(
            "Mechanics for PerfusionKineticMappingStrategy (regime "
            "mri_perfusion). The physics (time axis, AIF source, kinetic model) "
            "lives on data.perfusion; the weights on losses.physics.lambda_*."
        ),
    )
    mrs_quantification: MRSQuantificationConfig | None = Field(
        default=None,
        description=(
            "Mechanics for MRSQuantificationStrategy (regime mri_spectroscopy). "
            "The physics (FID length, dwell time, resonance count, signal model) "
            "lives on data.spectroscopy; the weights on losses.physics.lambda_*."
        ),
    )
    bloch_synth: BlochSynthConfig | None = Field(
        default=None,
        description=(
            "Cross-field relaxometry inversion + Bloch resynthesis knobs (source "
            "contrasts, dispersion bounds, seg/source/dispersion weights, segmenter "
            "backend) for BlochSynthesisStrategy (idea 2.1, Task 1)."
        ),
    )
    steerable_synthesis: SteerableSynthesisConfig | None = Field(
        default=None,
        description=(
            "Steerable C_4-equivariant synthesis knobs (lambda_l1) for "
            "SteerableSynthesisStrategy (B-1.4); use_equivariance lives on model_kwargs."
        ),
    )
    doob_bridge: DoobBridgeConfig | None = Field(
        default=None,
        description=(
            "Doob h-transform 7T diffusion-bridge knobs (timesteps/beta_schedule/"
            "sampling_steps/strength/h_scale/eta) for DoobBridgeStrategy (B-1.10)."
        ),
    )
    confluence: ConfluenceConfig | None = Field(
        default=None,
        description=(
            "Cross-source confluence (Fréchet-mean consensus) knobs (lambda_consensus, "
            "contrast_conditioning) for ConfluenceStrategy (B-1.1); needs "
            "data.mrixfields_pairing_policy='multi_source'."
        ),
    )
    brenier_synthesis: BrenierSynthesisConfig | None = Field(
        default=None,
        description=(
            "Source-conditioned Brenier OT-map knobs (lambda_l1) for BrenierSynthesisStrategy "
            "(B-1.5); enforce_convexity lives on model_kwargs."
        ),
    )
    mccann_field_path: McCannFieldPathConfig | None = Field(
        default=None,
        description=(
            "McCann single-potential field-path knobs (lambda_l1) for McCannFieldPathStrategy "
            "(B-3.9); enforce_convexity lives on model_kwargs."
        ),
    )
    fisher_rao_geodesic: FisherRaoGeodesicConfig | None = Field(
        default=None,
        description=(
            "Fisher-Rao geodesic translation knobs (lambda_l1) for FisherRaoGeodesicStrategy "
            "(B-3.4); use_fisher_rao_geometry lives on model_kwargs."
        ),
    )
    lora_modulation: LoRAModulationConfig | None = Field(
        default=None,
        description=(
            "Continuous low-rank modulation knobs (lambda_l1) for LoRAModulationStrategy "
            "(B-3.6); lora_rank (the compliance bound) lives on model_kwargs."
        ),
    )
    koopman_field: KoopmanFieldConfig | None = Field(
        default=None,
        description=(
            "Koopman cross-field propagator knobs (lambda_l1, lambda_recon) for "
            "KoopmanFieldStrategy (B-3.10); use_koopman_semigroup + koopman_dim live on "
            "model_kwargs."
        ),
    )
    scattering_besov: ScatteringBesovConfig | None = Field(
        default=None,
        description=(
            "Scattering-Besov detail-prior knobs (lambda_scatter + filterbank shape) for "
            "ScatteringBesovStrategy (B-1.3); lambda_scatter=0 is the L1-only control."
        ),
    )
    recoverability_vib: RecoverabilityVIBConfig | None = Field(
        default=None,
        description=(
            "Deep-VIB recoverability-ceiling knobs (beta = IB rate budget) for "
            "RecoverabilityVIBStrategy (B-2.3); latent_channels lives on model_kwargs."
        ),
    )
    cartoon_texture_safe: CartoonTextureSafeConfig | None = Field(
        default=None,
        description=(
            "Cartoon-texture (BV/G) hallucination-confinement knobs (lambda_ct + ROF/texture "
            "shape) for CartoonTextureSafeStrategy (B-2.5); lambda_ct=0 is the L1-only control."
        ),
    )
    field_wiener: FieldWienerConfig | None = Field(
        default=None,
        description=(
            "Field-dependent spectral Wiener knobs (lambda_l1) for FieldWienerStrategy (B-2.7); "
            "use_field_noise (the field-dependent-noise one-knob) lives on model_kwargs."
        ),
    )
    multi_echo_b0_fit: MultiEchoB0FitConfig | None = Field(
        default=None,
        description=(
            "Physics-in-the-loop multi-echo B0 fit knobs (t_shift, gamma, lambda_field) for "
            "MultiEchoB0FitStrategy (audit 2026-07 I1); the analytic fit is the SSOT "
            "infrastructure.physics.b0_mapping.map_b0, graded in Hz with B0FieldRMSE. This is the "
            "real B0 estimator; B0MappingStrategy is deformable registration, not a field fit."
        ),
    )
    certified_robustness: CertifiedRobustnessConfig | None = Field(
        default=None,
        description=(
            "Certified-robust reconstruction knobs (A-8.3 CertRob) for "
            "CertifiedRobustnessStrategy: lambda_lipschitz (the one-knob), target_sigma, "
            "n_power_iterations + adversarial knobs (epsilon, attack_steps, ...)."
        ),
    )
    sparse_frame: SparseFrameConfig | None = Field(
        default=None,
        description=(
            "Sparsifying-frame coherence-optimisation knobs (A-6.5 SCO-Frame) for "
            "SparseFrameStrategy: lambda_coherence (the one-knob) + lambda_parseval (tightness)."
        ),
    )

    # 2026-06 knob-wiring batch: typed sub-blocks for strategies whose loss
    # weights / physics constants were previously hardcoded class attributes or
    # read via getattr against extra="allow" (silent no-op, pitfall #15).
    qsm_pipeline: QSMPipelineTrainingConfigSchema | None = Field(
        default=None,
        description="QSM differentiable-pipeline loss weights (lambda_field/chi/tv/...).",
    )

    # 2026-08 knob-wiring batch: the unfinished tail of the above. These four
    # blocks had NO schema at all, so `extra="allow"` stored them as raw dicts
    # and every `getattr(cfg, knob, default)` in the strategy returned the
    # DEFAULT -- discarding what the arm declared. Two of them were an
    # ablation's only axis; see strategy_knobs_2026_08 for the measurement.
    spatiotemporal_adaptive_sfc_recon: SpatiotemporalAdaptiveSFCReconTrainingConfigSchema | None = (
        Field(
            default=None,
            description="4-D Beltrami weights for spatiotemporal fMRI recon (lambda_mu/lambda_t).",
        )
    )
    beltrami_epi_distortion: BeltramiEPIDistortionTrainingConfigSchema | None = Field(
        default=None,
        description="EPI distortion knobs (t_esp echo spacing, lambda_mu).",
    )
    adaptive_sfc_hssc: AdaptiveSFCHSSCTrainingConfigSchema | None = Field(
        default=None,
        description="Beltrami-SFC block geometry and loss weights (grid_size/lambda_*).",
    )
    conformal_diffusion_recon: ConformalDiffusionReconTrainingConfigSchema | None = Field(
        default=None,
        description="Conformal diffusion recon noise range and DC strength.",
    )
    conformal_mrf_dictless_recon: ConformalMRFDictlessReconTrainingConfigSchema | None = Field(
        default=None,
        description="Dictionary-free MRF conformality weight (lambda_conformality).",
    )
    crlb_mrf_pulse_design: CRLBMRFPulseDesignTrainingConfigSchema | None = Field(
        default=None,
        description="CRLB MRF pulse-design Beltrami weight (lambda_beltrami).",
    )
    cross_scanner_mrf_harmonisation: CrossScannerMRFHarmonisationTrainingConfigSchema | None = (
        Field(
            default=None,
            description="Cross-scanner MRF time-reparam anchor weight (lambda_anchor).",
        )
    )
    bloch_equivariant_translation: BlochEquivariantTranslationTrainingConfigSchema | None = Field(
        default=None,
        description="ULF->HF Bloch sequence constants and equivariance weights.",
    )
    ib_active_acquisition: IBActiveAcquisitionTrainingConfigSchema | None = Field(
        default=None,
        description="Information-bottleneck active-acquisition beta and mode.",
    )
    riemannian_bloch_diffusion: RiemannianBlochDiffusionTrainingConfigSchema | None = Field(
        default=None,
        description="Riemannian Bloch manifold-diffusion knobs (t_min/t_max/cache).",
    )
    privileged: PrivilegedLearningTrainingConfigSchema | None = Field(
        default=None,
        description="Privileged-distillation loss weights and curriculum.",
    )
    dtn2s: DTN2STrainingConfigSchema | None = Field(
        default=None, description="DTN2S mask receptive window."
    )

    # Specs that already EXISTED but were never mounted -- `BYPASSED_BLOCKS` in
    # tests/unit/config/test_knob_behaviour.py was their inventory. Each was
    # written, reviewed and correct, and zero percent of it executed. Annotated
    # as strings because their modules import from this one; resolved by the
    # `model_rebuild()` at the bottom.
    se3_equivariant_navigator: "SE3NavigatorConfig | None" = Field(
        default=None, description="SE(3)-equivariant navigator knobs."
    )
    phase_steganography: "PhaseSteganographyConfig | None" = Field(
        default=None, description="VF phase-domain steganographic marker."
    )
    bloch_ode: "BlochODEConfig | None" = Field(
        default=None, description="VF spin-history (through-plane Bloch ODE) corruption."
    )
    null_space_erasure: "NullSpaceErasureConfig | None" = Field(
        default=None, description="VF null-space marker erasure at inference."
    )
    twin_dps: "TwinDPSConfig | None" = Field(
        default=None, description="Twin diffusion-posterior-sampling knobs."
    )
    ib_vf: "IBVFConfig | None" = Field(
        default=None, description="Information-bottleneck virtual-fiducial knobs."
    )
    hamiltonian_acquisition: HamiltonianAcquisitionConfig | None = Field(
        default=None, description="Hamiltonian acquisition potential/integrator knobs."
    )
    bloch_manifold_dps: "BlochManifoldDPSConfig | None" = Field(
        default=None, description="Bloch-manifold DPS pulse/parameter-bound knobs."
    )
    equivariance_conformal: "EquivarianceConformalConfig | None" = Field(
        default=None, description="Equivariance conformal-calibration knobs."
    )
    phys_residual_conformal: "PhysResidualConformalConfig | None" = Field(
        default=None, description="Physics-residual conformal (PR-CC) calibration knobs."
    )
    acq_hypernetwork: "AcqHypernetworkConfig | None" = Field(
        default=None, description="LCAH hypernetwork width, depth and Lipschitz knobs."
    )
    dispersion_bloch_ae: "DispersionBlochAEConfig | None" = Field(
        default=None, description="DL-BAE pool count, fields and loss weights."
    )
    qspace_diffusion: QSpaceDiffusionTrainingConfigSchema | None = Field(
        default=None,
        description="q-space diffusion SH-regularisation knobs (sh_max_order/lambda_*).",
    )
    slice_to_volume: SliceToVolumeTrainingConfigSchema | None = Field(
        default=None,
        description="Slice-to-volume through-plane / orthogonal consistency weights.",
    )
    spatiotemporal_mrf_recon: SpatiotemporalMRFReconTrainingConfigSchema | None = Field(
        default=None,
        description="Spatiotemporal MRF recon Beltrami regularisation weights (lambda_mu/nu).",
    )
    trajectory_recon: TrainingConfigTrajectoryRecon | None = Field(
        default=None,
        description="Spiral trajectory-recovery knobs (exp_vf_35): supervised Δk vs "
        "measured deviation, GIRF param dim, no-grad sharpness diagnostic.",
    )
    riemannian_mrf_diffusion: RiemannianMRFDiffusionTrainingConfigSchema | None = Field(
        default=None,
        description="Riemannian MRF diffusion noise-scale bounds (sigma_min/max).",
    )
    score_field_tomography: ScoreFieldTomographyTrainingConfigSchema | None = Field(
        default=None,
        description="Score-field-tomography noise-scale + conditioning knobs.",
    )
    synthetic_pathology_aug: SyntheticPathologyAugTrainingConfigSchema | None = Field(
        default=None,
        description="Synthetic-pathology augmentation probability + lesion-loss weights.",
    )
    inverse_bloch_phase: InverseBlochPhaseTrainingConfigSchema | None = Field(
        default=None,
        description="Inverse-Bloch phase-residual smoothness weight.",
    )
    spin_sde: SpinSDETrainingConfigSchema | None = Field(
        default=None,
        description="Continuous-time Spin SDE integration + loss-weight knobs.",
    )
    cortical_conformal_fmri_recon: CorticalConformalFMRIReconTrainingConfigSchema | None = Field(
        default=None,
        description="Cortical-conformal fMRI recon curvature-regularisation weight.",
    )
    riemannian_dfc_diffusion: RiemannianDFCDiffusionTrainingConfigSchema | None = Field(
        default=None,
        description="Riemannian dynamic-FC diffusion noise-scale bounds.",
    )
    hrf_manifold_diffusion: HRFManifoldDiffusionTrainingConfigSchema | None = Field(
        default=None,
        description="HRF-manifold diffusion noise-scale bounds.",
    )
    mri_slam: MRISLAMTrainingConfigSchema | None = Field(
        default=None,
        description="MRI-SLAM loss weights + trajectory-delta optimizer LR.",
    )
    mae: MAEPretrainingTrainingConfigSchema | None = Field(
        default=None,
        description="Masked-autoencoder pretraining knobs (mask_ratio/mask_domain).",
    )
    schrodinger_bridge: SchrodingerBridgeTrainingConfigSchema | None = Field(
        default=None,
        description="Schrodinger-bridge Bloch-manifold penalty weight + cache resolution.",
    )
    ttt: TTTTrainingConfigSchema | None = Field(
        default=None,
        description="Test-time-training adaptation knobs (steps/lr/consistency_weight).",
    )
    vf_consistency_distillation: VFConsistencyDistillationTrainingConfigSchema | None = Field(
        default=None,
        description="VF consistency-distillation knobs (beta0/ema_decay/...).",
    )

    # Test-Time Optimization (TTO) — read by TTOTrainingStrategy.
    # Typed as ``Any`` here to avoid an import cycle: ``tto.py`` already
    # imports ``BaseTrainingConfigSchema`` from this module.
    # ``TTOTrainingStrategy.setup()`` coerces the raw dict to a typed
    # ``TTOConfig`` so attribute access (``config.training.tto.lambda_tv``)
    # works downstream. Audit-2026-05-14 E19.
    tto: Any = Field(
        default=None,
        description=(
            "Test-Time Optimization parameters (TV / DC weights, "
            "translation / rotation bounds, inner-loop LR + num_steps). "
            "Required when ``training_mode`` is ``tto`` or "
            "``test_time_optimization``. Coerced to ``TTOConfig`` by "
            "``TTOTrainingStrategy.setup()``."
        ),
    )

    spectra_tta: Any = Field(
        default=None,
        description=(
            "SPECTRA test-time-adaptation parameters (consistency weight, "
            "inner-loop steps + LR, freeze_backbone). Read by "
            "``SpectraTestTimeAdaptationStrategy`` at construction; when omitted "
            "the strategy uses documented defaults. Mirrors the ``tto`` block's "
            "extra='allow' loading path (plan §D)."
        ),
    )

    # Operator-ID (Proposal 1) — Lie-algebraic BCH effective-generator
    # identification. Typed ``Any`` to avoid an import cycle (operator_id.py
    # imports ``BaseTrainingConfigSchema`` from this module). The strategy
    # coerces the raw dict to a typed ``OperatorIDConfig`` at setup time, so
    # ``config.training.operator_id.bch_order`` works downstream. Mirrors the
    # ``tto`` field convention above.
    operator_id: Any = Field(
        default=None,
        description=(
            "Operator-identification parameters (mode_dictionary, bch_order, "
            "krylov_dim, covariance_rank, mmd_weight, …). Required when "
            "``training_mode`` is ``operator_id``. Coerced to ``OperatorIDConfig`` "
            "by ``OperatorIdBCHTrainingStrategy``."
        ),
    )

    # [CONFIG FIX] Detection thresholds and debug settings (strategy-agnostic, used by strategies)
    mask_saturation_threshold: float | None = Field(
        default=None,
        description="Mask saturation threshold (default 0.95): warn if mean(mask) > this value",
    )
    identity_collapse_threshold: float | None = Field(
        default=None,
        description="Identity collapse threshold (default 0.01): warn if model≈target from early training",
    )
    early_training_steps: int | None = Field(
        default=None,
        description="Number of initial steps for strict identity collapse check (default 500)",
    )
    mask_coverage_tolerance: float | None = Field(
        default=None,
        description="Mask coverage tolerance (default 0.1) for k-space diagnostics",
    )
    # Note: debug_log_steps and anomaly_check_interval are in logging schema


class BaseTrainingConfigSchema(BaseModel):
    """Base schema with fields common to all training paradigms.

    This schema contains:
    - Shared infrastructure (metadata, optimization, services, parallelism)
    - Feature-specific configs (checkpoint, logging, metrics, EMA, etc.)
    - Core training hyperparameters (epochs, batch_size, etc.)

    Subclasses should override/add paradigm-specific fields and validators.

    Example:
        >>> from spectramr.config.schemas.training import (
        ...     BaseTrainingConfigSchema
        ... )
        >>> base = BaseTrainingConfigSchema(
        ...     epochs=100,
        ...     batch_size=32,
        ...     learning_rate=0.0002,
        ... )
    """

    # The 17 duplicated root blocks (`data`, `model`, `optimization`, `logging`,
    # `validation`, ...) were deleted 2026-08-02. This class is UNREACHABLE from
    # `TrainingSettings` -- a recursive annotation walk reaches 268 model classes
    # and neither it nor any of its 20 subclasses is among them -- so those fields
    # made `training.gan.data.batch_size` look like legal, meaningful YAML while
    # nothing could ever read it. Zero corpus arms declared one.
    #
    # What actually makes such a key ACCEPTED is unrelated to this class:
    # `TrainingStrategyConfigSchema` is `extra="allow"` and `GANSubConfigSchema`
    # is `extra="ignore"`, so the key is swallowed rather than validated. Deleting
    # these fields does not close that hole; declaring the paradigm blocks
    # (strategy_knobs_2026_08) is what narrows it.
    #
    # Pinned by tests/unit/config/training/test_second_root_is_unreachable.py.

    model_config = {
        "protected_namespaces": (),
        # TEMPORARY RELAXATION: Allow extra fields for backward compatibility during v6.0 migration
        "extra": "ignore",
        "frozen": True,
    }

    # === Configuration Metadata ===
    config_version: str = Field(
        default=CANONICAL_CONFIG_VERSION,
        description=(
            f"Configuration schema version. Declare {CANONICAL_CONFIG_VERSION!r}. "
            f"Legacy {sorted(LEGACY_CONFIG_VERSIONS)} still load but are folded to "
            f"{CANONICAL_CONFIG_VERSION!r} at the door and are draining to zero — "
            "see spectramr.config.schemas.base for the promotion rule."
        ),
    )

    # === Core Training Hyperparameters ===
    epochs: int = Field(
        default=100,
        ge=1,
        description="Number of training epochs",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for training (canonical source)",
    )
    max_iterations: int = Field(
        default=-1,
        description="Maximum total iterations across all epochs (-1 = unlimited)",
    )
    max_steps_per_epoch: int = Field(
        default=-1,
        ge=-1,
        description="Maximum steps per epoch (-1 = unlimited/all batches)",
    )
    max_steps: int | None = Field(
        default=1,
        description="Maximum total steps for dry-run or quick testing (-1 = unlimited)",
    )
    task: str = Field(
        default="reconstruction",
        description="Task type: reconstruction, super_resolution, denoising, etc.",
    )
    # [REMOVED] training_mode (deprecated) - use training.strategy_class

    services: ServicesConfigSchema = Field(
        default_factory=ServicesConfigSchema,
        description="Infrastructure services (logging, checkpointing, profiling)",
    )

    augmentation: AugmentationConfigSchema = Field(
        default_factory=AugmentationConfigSchema,
        description="Data augmentation configuration",
    )
    acceleration: AccelerationConfigSchema = Field(
        default_factory=AccelerationConfigSchema,
        description="K-space acceleration and undersampling configuration",
    )

    objectives: ObjectiveConfigSchema | None = Field(
        default=None,
        description="[DEPRECATED] Training objectives and loss configuration. Use 'loss' instead.",
    )
    loss: LossConfigSchema = Field(
        default_factory=LossConfigSchema,
        description="Loss computation (losses, weights, FID, LPIPS, etc.)",
    )

    # === Device & Runtime ===
    device: Any = Field(
        default="cuda",
        description="Device to use: cuda, cpu",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility (numpy, torch, python random).",
    )
    deterministic: bool = Field(
        # Default True preserves the historical behaviour: the train path had
        # always FORCED full determinism via a hardcoded cudnn override in
        # main.py (while this knob, then defaulting False, was read by nobody
        # — pitfall #15). The knob is now the wired SSOT consumed by
        # ``initialize_accelerator`` and ``set_global_seed``; set False to
        # opt INTO the cuDNN autotuner for speed at the cost of run-to-run
        # reproducibility.
        default=True,
        description=(
            "If True (default), set torch.backends.cudnn.deterministic=True, "
            "cudnn.benchmark=False and torch.use_deterministic_algorithms"
            "(True, warn_only=True): bit-for-bit reproducible, slightly "
            "slower. Set False to enable the cuDNN autotuner (benchmark "
            "mode) for speed on fixed-shape workloads."
        ),
    )
    output_dir: str = Field(
        default="./training_output",
        description="Directory for training outputs. Authoritative (#698: artifacts.persistent_root is a no-op).",
    )
    output_base_dir: str = Field(
        default="./gan_training_output",
        description="[DEPRECATED] Base directory for outputs. Use output_dir instead.",
    )
    task_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific settings",
    )
    metrics_output_dir: str | None = Field(
        default=None,
        description="Directory for metrics output. Unset means the run derives it from output_dir.",
    )

    # `resolve_output_dirs` lived here and was deleted with #698. It was a
    # `model_validator(mode="after")` guarded on `hasattr(self, "artifacts")`,
    # but `artifacts` is a field of `TrainingSettings`, not of this class, so
    # the guard was False on every construction and the body never ran. Had it
    # run it would have raised: it did `self.artifacts.persistent_root` on a
    # field typed `dict[str, Any] | None`. `output_dir` is authoritative.

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: Any) -> Any:
        """Extract device string from dict for backward compatibility."""
        if isinstance(value, dict):
            return value.get("device_type", "cuda")
        return value


    @model_validator(mode="before")
    @classmethod
    def reject_deprecated_fields(cls, data: Any, info: ValidationInfo) -> Any:
        """Reject deprecated configuration fields with clear error messages.

        DEPRECATED FIELDS (REMOVED IN v5.0):
        - training_mode: Use training.strategy_class instead
        - num_epochs: Use epochs instead
        - lambda_l1: Use objectives.reconstruction.lambda_l1 instead
        - lambda_l2: Use objectives.reconstruction.lambda_l2 instead
        - lambda_adv: Use objectives.gan.lambda_adv instead

        This validator ensures v5.0-strict configuration with no backward compatibility.

        Args:
            data: Raw configuration dictionary
            info: Validation context

        Returns:
            Cleaned data without deprecated fields

        Raises:
            ValueError: If deprecated fields found
        """
        if not isinstance(data, dict):
            return data

        deprecated_fields = {
            "training_mode": (
                "training_mode field is deprecated. "
                "Use training.strategy_class instead.\n"
                "Example: training:\n  strategy_class: 'spectramr.infrastructure.training.strategies.gan.GANTrainingStrategy'"
            ),
            "num_epochs": ("num_epochs field is deprecated. Use epochs instead."),
            "lambda_l1": (
                "lambda_l1 field is deprecated. Use objectives.reconstruction.lambda_l1 instead."
            ),
            "lambda_l2": (
                "lambda_l2 field is deprecated. Use objectives.reconstruction.lambda_l2 instead."
            ),
            "lambda_adv": (
                "lambda_adv field is deprecated. Use objectives.gan.lambda_adv instead."
            ),
            "lambda_perc": (
                "lambda_perc field is deprecated. "
                "Use objectives.reconstruction.lambda_perceptual instead."
            ),
            "validation_check_interval": (
                "validation_check_interval field is deprecated. "
                "Use early_stopping.check_every_n_epochs instead."
            ),
        }

        found_deprecated = []
        for field, message in deprecated_fields.items():
            if field in data:
                found_deprecated.append((field, message))

        if found_deprecated:
            warning_msg = "DEPRECATED CONFIGURATION FIELDS FOUND:\n\n"
            for field, message in found_deprecated:
                warning_msg += f"⚠️ {field}:\n   {message}\n\n"
            warning_msg += "Please update your configuration to use the new field names.\n"
            warning_msg += "See docs/CONFIG_MIGRATION_V3_TO_V5_GUIDE.md for migration instructions."
            raise ValueError(warning_msg)

        return data

    @field_validator("config_version", mode="after")
    @classmethod
    def validate_config_version(cls, value: str) -> str:
        """Accept only documented schema versions; reject older / unknown.

        ``CANONICAL_CONFIG_VERSION`` is what a config should declare; the legacy
        6.x spellings are accepted only because ``config/settings.py`` folds them
        before binding. Earlier versions (v5.x and below) are not loadable — and
        not merely because of this gate: lifting it still fails every 5.0 file on
        retired keys.

        The accepted set is ``ACCEPTED_CONFIG_VERSIONS`` — imported, never
        restated, so this gate cannot drift from the two in ``config/settings.py``.
        """
        if value not in ACCEPTED_CONFIG_VERSIONS:
            raise ValueError(
                f"Unsupported config_version {value!r}. "
                f"Declare {CANONICAL_CONFIG_VERSION!r} "
                f"(legacy, still folded: {sorted(LEGACY_CONFIG_VERSIONS)}). "
                f"Earlier schemas (v5.x and below) are not loadable; migrate with "
                f"scripts/migrations/migrate_config_version_to_v1.py."
            )
        return value

    @model_validator(mode="after")
    def validate_critical_infrastructure_fields(self) -> "BaseTrainingConfigSchema":
        """Enforce critical infrastructure fields that must be present.

        CRITICAL FIELDS REQUIREMENT (Architectural Mandate):
        ===================================================

        The following fields are MANDATORY and must be present in all training configs.
        These fields enable the complete training infrastructure:

        1. logging: Experiment tracking, logging levels, console/file output
        2. loss_logging: CSV logging of training losses for analysis
        3. metrics: Metric computation and best model tracking
        4. early_stopping: Early stopping to prevent overfitting
        5. ema: Exponential Moving Average for model smoothing
        6. validation: Validation dataset and evaluation configuration

        CRITICAL ARCHITECTURAL PRINCIPLE:
        - These fields are NOT OPTIONAL - they define the training infrastructure
        - Every training config MUST explicitly include these sections
        - This enforces observability, monitoring, and stability across all experiments
        - Default values are provided by schemas, but fields must be present

        This validator logs when configs properly include all infrastructure fields,
        ensuring transparent and fully-monitored training workflows.
        """
        critical_fields = {
            "logging": "Experiment tracking and logging configuration",
            "loss_logging": "CSV loss logging for loss history and analysis",
            "metrics": "Metric computation and best model tracking",
            "early_stopping": "Early stopping to prevent overfitting",
            "ema": "Exponential Moving Average for model smoothing",
            "validation": "Validation dataset and evaluation",
            "checkpoint": "Checkpoint saving and loading configuration",
            "services": "Infrastructure services configuration",
        }

        # Only fields this class still DECLARES can be checked. Phase 13 deleted
        # 17 duplicated blocks from this schema (they are owned by
        # `TrainingSettings`) but left this list naming seven of them. The class
        # is `extra="ignore"`, so a caller supplying `logging={}` had it dropped
        # before this ran and `hasattr` was False no matter what the input said:
        # the validator rejected EVERY input, including the tests written to
        # drive it. Intersecting with `model_fields` restores it to the thing it
        # can actually enforce, and makes it self-maintaining if more blocks move.
        declared = set(type(self).model_fields)
        missing = [
            field_name
            for field_name in critical_fields
            if field_name in declared and not hasattr(self, field_name)
        ]

        if missing:
            raise ValueError(
                f"CRITICAL INFRASTRUCTURE FIELDS MISSING: {missing}\n\n"
                f"All training configs MUST include these infrastructure fields:\n"
                f"{chr(10).join(f'  - {k}: {v}' for k, v in critical_fields.items())}\n\n"
                f"These fields enable comprehensive training monitoring, logging, metrics tracking, "
                f"and stability mechanisms. Please add them to your config."
            )

        return self

    @model_validator(mode="after")
    def validate_shape_contracts(self) -> "BaseTrainingConfigSchema":
        """Validate shape consistency between Data and Model configurations.

        Enforces geometric contracts:
        1. Spatial Dimensions: data.patch_size[2]==1 => model.spatial_dims=2
        2. Channel Consistency: Warn if model.in_channels != data.target_channels (heuristic)
        """
        if not hasattr(self, "data") or not hasattr(self, "model"):
            return self

        # 1. Spatial Dimension Validation
        # `data.patch_size` folded to `data.sampling.patch_size`. The guard above
        # checks the BLOCK exists, never the leaf, so after the fold this line
        # raised AttributeError and every construction of a schema reaching it
        # died -- including every `TrainingConfigMetaLearning(...)`.
        # Read directly rather than behind a `hasattr(self.data, "sampling")`:
        # a subclass without it should fail loudly, not skip the contract check.
        patch = self.data.sampling.patch_size
        patch_d = patch[2] if len(patch) >= 3 else 1
        model_dims = self.model.spatial_dims

        if patch_d == 1 and model_dims == 3:
            raise ValueError(
                "DIMENSION MISMATCH: data.patch_size indicates 2D data (depth=1), "
                "but model.spatial_dims is set to 3.\n"
                "Fix: Set model.spatial_dims=2 OR increase patch_size depth."
            )
        elif patch_d > 1 and model_dims == 2:
            raise ValueError(
                f"DIMENSION MISMATCH: data.patch_size indicates 3D data (depth={patch_d}), "
                f"but model.spatial_dims is set to 2.\n"
                f"Fix: Set model.spatial_dims=3 OR set patch_size depth to 1."
            )

        # 2. Channel Hint (Soft validation)
        # Note: Input channels are complex (depends on domain), so we only warn on obvious mismatches
        # e.g. if dataset is 'image' (1 ch) but model expects 2 ch
        dataset_type = self.data.dataset_type
        if dataset_type == "image" and self.model.in_channels == 2:
            # This might be intended (real/imag split) but is suspicious for image domain
            pass

        return self

    # === Experimental Features ===
    # (To be added as needed; kept minimal here)
    enable_mixed_precision: bool = Field(
        default=False,
        description="Enable automatic mixed precision (AMP) training",
    )

    # === Runtime Capabilities (Auto-detected) ===
    has_timm: bool = Field(default=False, exclude=True)
    has_torchvision: bool = Field(default=False, exclude=True)
    has_nibabel: bool = Field(default=False, exclude=True)
    has_lpips: bool = Field(default=False, exclude=True)
    has_safetensors: bool = Field(default=False, exclude=True)
    train_transforms: Any = Field(
        default=None, exclude=False, description="Runtime training transforms override"
    )
    val_transforms: Any = Field(
        default=None,
        exclude=False,
        description="Runtime validation transforms override",
    )

    @classmethod
    def from_yaml(cls, path: str) -> "BaseTrainingConfigSchema":
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Deferred sub-block references (see `defer_build` on TrainingStrategyConfigSchema).
#
# `se3_navigator`, `vf_advanced`, `bloch_manifold_dps`, `acq_hypernetwork`,
# `dispersion_bloch_ae` and
# `equivariance_conformal` each import from this module for a DIFFERENT class in
# the same file, so a top-level import would cycle. The knob classes themselves
# are cycle-free: import them here, once every class in this module exists, and
# resolve the string annotations.
#
# This MUST precede `StageEnvironmentSchema.model_rebuild()` below: that call
# references `TrainingStrategyConfigSchema` and so forces its schema to build,
# which `defer_build` would otherwise have postponed past this point.
from .acq_hypernetwork import AcqHypernetworkConfig  # noqa: E402
from .bloch_manifold_dps import BlochManifoldDPSConfig  # noqa: E402
from .bloch_ode import BlochODEConfig  # noqa: E402
from .dispersion_bloch_ae import DispersionBlochAEConfig  # noqa: E402
from .equivariance_conformal import EquivarianceConformalConfig  # noqa: E402
from .federated import FederatedConfig  # noqa: E402
from .flow import FlowConfig  # noqa: E402
from .motion import MotionConfig  # noqa: E402
from .null_space_erasure import NullSpaceErasureConfig  # noqa: E402
from .phase_steganography import PhaseSteganographyConfig  # noqa: E402
from .phys_residual_conformal import PhysResidualConformalConfig  # noqa: E402
from .se3_navigator import SE3NavigatorConfig  # noqa: E402
from .vf_advanced import IBVFConfig, TwinDPSConfig  # noqa: E402

TrainingStrategyConfigSchema.model_rebuild()


# Resolve forward references for Multi-Stage components that depend back on base.py
StageEnvironmentSchema.model_rebuild(
    _types_namespace={"TrainingStrategyConfigSchema": TrainingStrategyConfigSchema}
)

__all__ = ["BaseTrainingConfigSchema"]
