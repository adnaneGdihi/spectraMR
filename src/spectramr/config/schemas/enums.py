"""Enumeration types for strict type-safe configuration validation.

This module defines all valid enumeration values for configuration fields,
ensuring type safety and preventing typos in configuration files.

Example:
    >>> from spectramr.config.schemas.enums import TrainingMode, GANLossType
    >>> mode = TrainingMode.GAN
    >>> loss = GANLossType.WGAN_GP
"""

from enum import Enum, StrEnum


class TrainingMode(str, Enum):
    """Valid training paradigms."""

    GAN = "gan"
    DIFFUSION = "diffusion"
    VAE = "vae"
    RECONSTRUCTION = "reconstruction"
    SSL = "ssl"
    MAE = "mae"
    PRETRAINING = "pretraining"
    FLOW_MATCHING = "flow_matching"
    KAN = "kan"
    KSpace_COLD_DIFFUSION = "kspace_cold_diffusion"
    DOMAIN_ADAPTATION = "domain_adaptation"
    THREE_D_GENERATION = "3d_generation"
    DISENTANGLED = "disentangled"
    OPERATOR_ID = "operator_id"


class GANLossType(str, Enum):
    """Valid GAN loss functions."""

    WGAN_GP = "wgan-gp"
    HINGE = "hinge"
    BCE = "bce"
    LSGAN = "lsgan"
    WGAN = "wgan"
    WGAN_SOFTPLUS = "wgan_softplus"
    VANILLA = "vanilla"


class LRScheduler(str, Enum):
    """Valid learning rate scheduler strategies."""

    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    WARMUP = "warmup"
    POLYNOMIAL = "polynomial"
    CYCLIC = "cyclic"
    ONE_CYCLE = "one_cycle"


class NoiseSchedule(str, Enum):
    """Valid noise schedules for diffusion models."""

    LINEAR = "linear"
    COSINE = "cosine"
    SQRT = "sqrt"
    QUADRATIC = "quadratic"


class AccelerationSchedule(str, Enum):
    """Valid acceleration factor schedules for k-space sampling."""

    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    STEP = "step"
    POWER_LAW = "power_law"


class OptimizerType(str, Enum):
    """The advertised optimizer vocabulary — the SSOT ``OptimizerRegistry`` must cover.

    Three tiers, annotated because they fail differently:

    * **torch** — always constructible; ``torch.optim`` ships them.
    * **in-repo** — implemented under ``infrastructure/training/optimizers/``,
      dependency-free.
    * **extra-backed** — registered unconditionally so an unknown-name error can
      never be confused with a missing extra, but construction RAISES with the
      install command when the extra is absent. Never a silent fall back to
      Adam: the 8-bit state (or the schedule-free iterate) IS the arm's claim.

    ``optimizer_registry`` asserts at import that every member here is registered,
    so a ``@register_optimizer`` in a module nothing imports is an ImportError
    rather than a name that silently vanishes from ``list_available()``.

    LARS and LAMB were members of this enum with no implementation anywhere from
    its introduction until 2026-07-30; ``tests/unit/config/test_optimizer_type_coverage.py``
    xfailed them as "needs apex / custom". They are now real.
    """

    # --- torch.optim (always available) ---
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAMAX = "adamax"
    NADAM = "nadam"
    RADAM = "radam"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    ASGD = "asgd"
    RPROP = "rprop"
    ADAFACTOR = "adafactor"
    #: Neither accepts ``weight_decay``. Named deliberately rather than left to
    #: the reflective ``getattr(torch.optim, ...)`` path that used to reach them:
    #: they are precisely the two that ``TypeError``'d on the unconditionally
    #: forwarded ``weight_decay`` default, so naming them makes that a designed,
    #: tested drop instead of a crash.
    LBFGS = "lbfgs"
    SPARSEADAM = "sparseadam"

    # --- in-repo, dependency-free ---
    LARS = "lars"
    LAMB = "lamb"
    LION = "lion"
    #: SAM is deliberately ABSENT until its stepper lands.
    #:
    #: It needs two forward+backward passes per step (ascend to w + rho*g/||g||,
    #: re-evaluate, restore, then step the base optimizer), which the current
    #: single-backward seam cannot express: ``AMPPolicy.backward_and_step``
    #: receives an already-computed loss tensor, not a re-invocable closure.
    #: Registering it anyway would be worse than omitting it — a single
    #: ``step()`` would perturb the weights and never restore them, so the run
    #: would train on w + rho*g and report success.
    #:
    #: Adding the name back is one line, and belongs with
    #: ``optimizers/sharpness_aware_stepper.py``, not before it.

    # --- optional-extra backed ---
    ADAM8BIT = "adam8bit"  # [bnb]
    ADAMW8BIT = "adamw8bit"  # [bnb]
    SCHEDULEFREE_ADAMW = "schedulefree_adamw"  # [schedulefree]
    SCHEDULEFREE_SGD = "schedulefree_sgd"  # [schedulefree]


#: Accepted spellings that normalise onto an :class:`OptimizerType` value. Kept
#: beside the enum (not in the registry) because the config-layer validator needs
#: it and ``config/`` cannot import ``infrastructure/`` — the layer rule points
#: inward only, so the enum can be checked *against* the registry but never
#: derived *from* it.
OPTIMIZER_ALIASES: dict[str, str] = {
    "adam_w": "adamw",
    "adamw_8bit": "adamw8bit",
    "adam_8bit": "adam8bit",
    "bnb_adamw": "adamw8bit",
    "bnb_adam": "adam8bit",
    "rms_prop": "rmsprop",
    "n_adam": "nadam",
    "r_adam": "radam",
    "schedule_free_adamw": "schedulefree_adamw",
    "schedule_free_sgd": "schedulefree_sgd",
}

#: Every canonical optimizer name, as a frozenset for O(1) membership.
OPTIMIZER_NAMES: frozenset[str] = frozenset(m.value for m in OptimizerType)


# ---------------------------------------------------------------------------
# Workflow vocabulary — the imaging-regime x task contract.
#
# These six closed enums declare *what a signal is* (Modality/Regime), *what
# is being done to it* (Task), *what non-spatial axes it carries* (Axis),
# *where an artifact comes from* (ArtifactOrigin), and *how far the framework
# can actually honour a regime* (Maturity). A typo in a ``workflow:`` block
# therefore fails at Tier-0 (Pydantic ``ValidationError``) rather than
# silently mislabelling an arm. The frozen ``WorkflowProfile`` facts that back
# each ``Regime`` live in ``spectramr.domain.workflows``.
#
# 4D/5D acquisitions are *axis compositions*, not enum members: "4D cine" is
# 3 spatial ranks + ``Axis.TEMPORAL``; "5D 4D-flow" adds ``Axis.VELOCITY_ENCODING``.
# ---------------------------------------------------------------------------


class Modality(str, Enum):
    """Physical imaging modality — what kind of signal is acquired."""

    MR = "mr"
    XRAY_TRANSMISSION = "xray_transmission"
    ULTRASOUND = "ultrasound"
    OPTICAL = "optical"


class Regime(str, Enum):
    """Imaging regime — the physical nature of the MR (or other) signal.

    ``diffusion_weighted`` is spelled out in full: bare ``diffusion`` already
    means DDPM/score generative models elsewhere in this codebase.
    """

    STRUCTURAL = "mri_structural"
    QUANTITATIVE = "mri_quantitative"
    FINGERPRINTING = "mri_fingerprinting"
    FUNCTIONAL = "mri_functional"
    DIFFUSION_WEIGHTED = "mri_diffusion_weighted"
    PERFUSION = "mri_perfusion"
    SPECTROSCOPIC = "mri_spectroscopy"
    FLOW = "mri_flow"
    DYNAMIC = "mri_dynamic"
    NMR_SPECTROSCOPY = "nmr_spectroscopy"
    CT = "ct"
    XRAY = "xray"
    ULTRASOUND = "ultrasound"
    OPTICAL = "optical"


class Task(str, Enum):
    """What the arm is *doing* to the signal (orthogonal to the regime).

    Every member must be supported by at least one
    :class:`~spectramr.domain.workflows.profiles.WorkflowProfile`, pinned by
    ``tests/unit/config/schemas/test_workflow.py``. A member no regime supports
    is an advertised option that hard-errors on use — the enum promises a task
    the framework cannot run.

    ``SEGMENTATION`` was removed on 2026-07-31 for exactly that reason. Across
    152 strategy classes none tagged it, no loss carried it, no profile listed
    it, and no YAML in the corpus declared it; the only segmentation the
    framework has is *metrics* (``schemas/metrics.py``) plus SynthSeg region
    caching, and a task you can score but not train is not a task an arm can
    declare. Note ``loss_recommender.TaskType.SEGMENTATION`` and
    ``adaptive_hyperparameter_config.SEGMENTATION`` are unrelated enums with
    their own vocabularies — they were never this one.
    """

    RECONSTRUCTION = "reconstruction"
    DENOISING = "denoising"
    SUPER_RESOLUTION = "super_resolution"
    SYNTHESIS = "synthesis"
    FIELD_TRANSLATION = "field_translation"
    CONTRAST_TRANSLATION = "contrast_translation"
    PARAMETER_MAPPING = "parameter_mapping"
    MOTION_CORRECTION = "motion_correction"
    ACQUISITION_DESIGN = "acquisition_design"


class Axis(str, Enum):
    """Non-spatial axes a dataset may expose / a regime may require."""

    TEMPORAL = "temporal"
    TRANSIENT = "transient"
    ECHO = "echo"
    # Flip-angle encoding: the axis a B1+ transmit map is measured along
    # (double-angle, AFI, Bloch-Siegert). Added with the rule that reads it --
    # ``mri_quantitative`` accepts it as an alternative to ECHO -- so it is not
    # a member whose only reader is a gate that cannot fire, the argument that
    # deleted ``optional_axes`` from WorkflowProfile.
    FLIP_ANGLE = "flip_angle"
    SPECTRAL = "spectral"
    DIFFUSION_ENCODING = "diffusion_encoding"
    VELOCITY_ENCODING = "velocity_encoding"
    CARDIAC_PHASE = "cardiac_phase"
    COIL = "coil"


class ArtifactOrigin(str, Enum):
    """Where a degradation/artifact originates in the imaging chain."""

    SCANNER = "scanner"
    ACQUISITION = "acquisition"
    PATIENT = "patient"


class Maturity(str, Enum):
    """How far the framework can honestly run a regime.

    Each value states the rule ``tests/unit/domain/workflows/test_maturity_ledger.py``
    actually enforces against the live registries — not an aspiration. Keep the two
    in step.

    - ``LIVE``: a **registered forward model** (a linear operator OR a signal
      model) AND a regime-tagged strategy AND regime-tagged metrics — runs.
      Together: the physics is expressible, trainable, and gradeable on its own
      terms.
    - ``PARTIAL``: either a regime-tagged strategy, or BOTH tagged losses and tagged
      metrics — runs; the audit reports the gap. (Metrics *alone* is EVAL_ONLY, not
      PARTIAL: the distinction decides whether ``train`` is allowed to run.)
    - ``EVAL_ONLY``: metrics exist, zero losses/strategies — ``evaluate``/``predict`` run, ``train`` raises.
    - ``STUB``: nothing exists — every pipeline raises ``WorkflowNotImplementedError``.

    LIVE deliberately does NOT require tagged **losses**. A regime may honestly
    train on agnostic objectives — ``mri_structural`` trains on L1/SSIM, which is
    correct rather than a gap — and requiring a tag would force mis-tagging L1 as
    structural-only. Metrics are required because the asymmetry is real: a regime
    can train on a generic objective, but it cannot be *graded* generically once
    its output stops being an image (PSNR on a Ktrans map is pitfall #18).

    The forward-model clause accepts either kind because an ``OperatorRegistry``
    entry promises ``adjoint()`` while a signal model is nonlinear and has none.
    Before it was widened it demanded an operator, so ``mri_perfusion`` could only
    reach LIVE by registering a fake adjoint — the rule rewarded the facade it
    existed to prevent.
    """

    LIVE = "live"
    PARTIAL = "partial"
    EVAL_ONLY = "eval_only"
    STUB = "stub"


class DatasetType(str, Enum):
    """Valid dataset types."""

    TWO_D = "2d"
    THREE_D = "3d"
    VOLUMETRIC = "volumetric"
    SLICE = "slice"
    RECONSTRUCTION = "reconstruction"
    GRAPH_MRI = "graph_mri"
    SYNTHETIC = "synthetic"


class VolumeFormat(str, Enum):
    """Valid volume data formats."""

    NII = "nii"
    NII_GZ = "nii.gz"
    H5 = "h5"
    DICOM = "dicom"
    RAW = "raw"
    NUMPY = "numpy"


class PredictionType(str, Enum):
    """Valid prediction types for diffusion models."""

    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"


class KANType(str, Enum):
    """Valid Kolmogorov-Arnold Network types."""

    RBFKAN = "rbfkan"
    BSPLINEKAN = "bsplinekan"
    WAVELETKAN = "waveletkan"


class InputType(str, Enum):
    """Valid input types for models."""

    IMAGE = "image"
    KSPACE = "kspace"
    SINOGRAM = "sinogram"
    FREQUENCY = "frequency"


class TrackingService(str, Enum):  # noqa: UP042 — matches the other 17
    # enums in this module; `StrEnum` here alone would make the file
    # inconsistent for no behavioural gain under Pydantic v2.
    """Experiment-tracking backend.

    Closed on purpose. `service` was a bare `str` compared against the literal
    `"tensorboard"` in `pipelines/train.py`, so ANY other value -- the `wandb`
    its own field description advertised, or a typo -- fell off the branch and
    the run trained to completion with no tracking, no warning and a zero exit
    (pitfall #9, non-negotiable #3).

    `NONE` is a member rather than an omission because 6 committed arms declare
    it and mean it: those are evaluation arms that deliberately want no writer.
    Retiring the spelling would break them for no gain.

    W&B is deliberately absent. `logging.wandb_project` / `wandb_entity` are
    inert (KNOWN_UNCONSUMED, issue #675), so a W&B run configured through this
    framework is configured through nothing -- accepting the value would be the
    facade this enum exists to remove. Zero committed arms declare it, so
    refusing it costs no migration.
    """

    TENSORBOARD = "tensorboard"
    NONE = "none"


class LogLevel(str, Enum):
    """Valid logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ConsoleStream(str, Enum):  # noqa: UP042 — matches the other 17
    """Which stream the console log writes to.

    ``split`` sends DEBUG/INFO to stdout and WARNING and above to stderr, so a
    SLURM ``.out`` reads as the run log and ``.err`` holds only problems.
    """

    SPLIT = "split"
    STDOUT = "stdout"
    STDERR = "stderr"


class RankScope(str, Enum):  # noqa: UP042 — matches the other 17
    """Which ranks of a distributed launch write to a sink."""

    RANK0 = "rank0"
    ALL = "all"


class ProgressMode(str, Enum):  # noqa: UP042 — matches the other 17
    """How progress is shown: a live bar, periodic log lines, or not at all.

    ``auto`` is a bar on a TTY and periodic lines otherwise.
    """

    AUTO = "auto"
    BAR = "bar"
    LINES = "lines"
    OFF = "off"


class GradientClipMethod(str, Enum):
    """Valid gradient clipping methods."""

    NORM = "norm"
    VALUE = "value"
    NORM_TYPE = "norm_type"


class ActivationFunction(str, Enum):
    """Valid activation functions."""

    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SILU = "silu"
    TANH = "tanh"
    SIGMOID = "sigmoid"


class NormalizationType(str, Enum):
    """Valid normalization types."""

    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    INSTANCE_NORM = "instance_norm"
    GROUP_NORM = "group_norm"
    NONE = "none"


class SelectionRule(StrEnum):
    """Checkpoint-selection rules over the cascade's stamped metrics (``validation.selection``)."""

    NONE = "none"
    GUARDED_RING_FIDELITY = "guarded_ring_fidelity"


class MetricMode(str, Enum):
    """Valid metric optimization modes."""

    MIN = "min"
    MAX = "max"


class TriggerType(str, Enum):
    """When a loss-schedule rule fires (see ``loss_schedule.py``)."""

    ITERATION = "iteration"  # at a fixed global iteration (or every N)
    EPOCH = "epoch"  # at a fixed epoch (or every N)
    METRIC_PLATEAU = "metric_plateau"  # when a validation metric stops improving
    LOSS_PLATEAU = "loss_plateau"  # when a training loss stops improving
    METRIC_THRESHOLD = "metric_threshold"  # when a metric crosses a value


class ActionType(str, Enum):
    """What a loss-schedule rule does to its target loss term's weight."""

    SET_WEIGHT = "set_weight"  # set the weight to an absolute value
    SCALE = "scale"  # multiply the current weight by a factor
    RAMP = "ramp"  # interpolate from current to a target over N steps
    ENABLE = "enable"  # restore the config lambda (or an explicit value)
    DISABLE = "disable"  # set the weight to 0.0


class RampShape(str, Enum):
    """Interpolation shape for a ``ramp`` loss-schedule action."""

    LINEAR = "linear"
    SIGMOID = "sigmoid"


class ExpectKind(str, Enum):
    """Post-change expectation for a loss-schedule ``monitor_after`` block."""

    IMPROVE = "improve"  # the metric must strictly improve within the window
    NOT_WORSE = "not_worse"  # the metric must not degrade beyond min_delta


class OnFailureKind(str, Enum):
    """Action when a ``monitor_after`` expectation is not met."""

    ROLLBACK = "rollback"  # restore the pre-change weight override
    HOLD = "hold"  # keep the changed weight, stop monitoring
    CONTINUE = "continue"  # keep the changed weight and take no action


class PrecisionPolicy(str, Enum):
    """Valid per-layer arithmetic precision policies for the SPECTRA backend.

    Maps to the morphable-PE precision modes (SPECTRA Workstream A7). ``MIXED``
    lets the compiler choose FP32/BF16/FP8 per layer against a fidelity floor;
    the others pin a single output precision. ``FP8`` alone is an
    inner-loop/preconditioner precision, never a diagnostic output precision —
    the ``check_spectra_precision_envelope`` Tier-1 guard enforces this.
    """

    MIXED = "mixed"
    FP32 = "fp32"
    BF16 = "bf16"
    FP8 = "fp8"


class SpectraRuntime(str, Enum):
    """Valid SPECTRA execution runtimes (Workstream B4).

    ``COSIM`` drives the real RTL through the Verilator co-simulation TCP bridge
    (sandbox-portable, correctness vehicle — not a performance path). ``ROCC``
    issues the RoCC instruction stream directly on silicon after Chipyard
    elaboration.
    """

    COSIM = "cosim"
    ROCC = "rocc"


# =============================================================================
# Legacy Enums (Consolidated from domain/entities/enums/)
# These maintain backward compatibility with existing code
# =============================================================================


class LossTypes(str, Enum):
    """Valid loss function types."""

    L1 = "l1"
    L2 = "l2"
    HUBER = "huber"
    PERCEPTUAL = "perceptual"
    SSIM = "ssim"
    GAN = "gan"
    MSE = "mse"
    FREQUENCY = "frequency"


class SchedulerTypes(str, Enum):
    """Valid scheduler types for diffusion models."""

    DDPM = "ddpm"
    DDIM = "ddim"
    PNDM = "pndm"
    LMS = "lms"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_a"
    LINEAR = "linear"
    COSINE = "cosine"
    CYCLIC = "cyclic"
    COLD_MRI = "cold_mri"


class GeneratorTypes(str, Enum):
    """Valid generator architecture types."""

    STANDARD_UNET = "standard_unet"
    ENHANCED_UNET = "enhanced_unet"
    VAE = "vae"
    DIFFUSION = "diffusion"
    LATENT_DIFFUSION = "latent_diffusion"
    COLD_DIFFUSION = "cold_diffusion"
    SCORE_BASED = "score_based"
    FNO = "fno"
    PINN_NERF = "pinn_nerf"
    VIT = "vit"
    SWIN = "swin"
    TRELLIS_3D = "trellis_3d"
    MAMBA_UNET = "mamba_unet"
    MAMBA_RECONSTRUCTION = "mamba_reconstruction"
    UNET_2_5D = "unet_2_5d"
    UNET_3D = "unet_3d"
    VNET = "vnet"
    UNETR = "unetr"
    VOLUMETRIC_UNET = "volumetric_unet"
    VOLUMETRIC_DIFFUSION = "volumetric_diffusion"
    VOLUMETRIC_VAE = "volumetric_vae"


class TrainingModeTypes(str, Enum):
    """Valid training mode types."""

    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    SELF_SUPERVISED = "self_supervised"
    CONDITIONAL = "conditional"
    ADVERSARIAL = "adversarial"
    GAN = "gan"
    DIFFUSION = "diffusion"
    VAE = "vae"
    SSL = "ssl"
    RECONSTRUCTION = "reconstruction"
    DISENTANGLED = "disentangled"
    DOMAIN_ADAPTATION = "domain_adaptation"
    LATENT_GAN = "latent_gan"
    LATENT_DIFFUSION = "latent_diffusion"
    MAE = "mae"
    GEOMAMBA_ULF = "geomamba_ulf"


class DataFormatTypes(str, Enum):
    """Valid data format types."""

    H5 = "h5"
    NIFTI = "nifti"
    DICOM = "dicom"
    PNG = "png"
    NUMPY = "numpy"
    TORCH = "torch"


class OptimizationTypes(str, Enum):
    """Valid optimization algorithm types."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class ReportTask(str, Enum):
    """Report task preset selecting the default figure/table set."""

    DEFAULT = "default"
    RECONSTRUCTION = "reconstruction"
    SYNTHESIS = "synthesis"
    SUPER_RESOLUTION = "super_resolution"
    GAN = "gan"
    DIFFUSION = "diffusion"
    VAE = "vae"
    # Coverage/certification reporting for the conformal-calibration arms
    # (run_calibration strategies). Without this member every VF
    # `reporting.task: calibration` config (exp_p1, exp_vf_conformal_calib,
    # eval_c5) raised ValidationError at load time and could not run at all.
    CALIBRATION = "calibration"


class SignalDomain(str, Enum):
    """What a tensor physically IS — the SSOT for every domain vocabulary.

    This vocabulary previously existed as bare strings in four places that had
    drifted apart: ``domain/workflows/profiles.py`` knew ``spectrum`` but not
    ``latent``; ``config/schemas/loss.py`` hardcoded the opposite pair;
    ``data/datasets/axis_exposure.py`` knew only three; and
    ``models/capabilities.Domain`` — the most complete of the four, and therefore
    the one these members are taken from — knew all seven.

    The practical cost of that split: ``mri_spectroscopy`` is a LIVE regime whose
    only signal domain is ``spectrum``, and the loss layer had no way to name it.

    ``LATENT``, ``PDE_GRID`` and ``MESH`` are *representations* rather than
    acquired signals, so they are legal for a model or a loss to operate in while
    belonging to no imaging regime's ``signal_domains``. That asymmetry is
    deliberate; see ``WorkflowProfile``.
    """

    IMAGE = "image"  # real-valued magnitude image, [B, C, H, W(, D)]
    KSPACE = "kspace"  # k-space, real/imag interleaved as 2C channels
    COMPLEX_IMAGE = "complex_image"  # complex tensor or 2C interleaved
    LATENT = "latent"  # learned latent (post-VAE encoder)
    PDE_GRID = "pde_grid"  # PDE benchmark grid (Burgers, Darcy, ...)
    MESH = "mesh"  # graph / mesh data (graph_mri)
    SPECTRUM = "spectrum"  # MRS free induction decay, real/imag as 2T channels


class ReportStyle(str, Enum):
    """House style for report figures."""

    NATURE = "nature"
    IEEE = "ieee"
    POSTER = "poster"
    SLIDES = "slides"


class DiffusionParadigm(str, Enum):
    """Which generative paradigm a ``training.diffusion`` block selects.

    The closed set of tags for the discriminated union at
    ``training.diffusion`` (``config/schemas/training/base.py``).

    **Membership follows what the framework IMPLEMENTS, not what the corpus
    happens to declare.** This is a research framework whose purpose is
    exploring combinations, so a paradigm with zero arms today is capability,
    not dead surface — refusing to name it is what makes
    ``type: ddim`` raise for a user who wants to try it.

    That is the opposite of the phase-1.3 rule, and deliberately so: there the
    retired names DUPLICATED a canonical key or bound to the wrong class, so
    each one made the framework ambiguous. A paradigm nobody has used yet makes
    it *larger*. Delete names that mislead; do not withhold names that work.

    Corpus usage, measured 2026-08-12 over the 725 constructible configs, is
    therefore documentation rather than a membership test:

    ==================  =====  ================================================
    tag                  arms  implementation backing
    ==================  =====  ================================================
    ``cold``              108  incl. 4 spelling it ``cold_diffusion``
    ``score_based``        11
    ``ddpm``                4  no registered sampler yet — see plan phase 3a
    ``rectified_flow``      1  ``rectified_flow`` model
    ``latent_diffusion``    1
    ``ddim``                0  ``ddim`` sampler (registered)
    ``chi_square``          0  ``ChiSquareColdDiffusion``,
                               ``chi_square_diffusion``,
                               ``simple_chi_square_diffusion``
    *(untagged)*           75  -> ``UnspecifiedParams``, transitional
    ==================  =====  ================================================

    Adding a member is a four-line change (enum entry + variant class), which is
    the point of the union — extension should be cheap.

    Kept in agreement with the union's actual tags by
    ``test_the_enum_and_the_union_agree``, so this cannot rot into documentation.
    """

    COLD = "cold"
    SCORE_BASED = "score_based"
    DDPM = "ddpm"
    DDIM = "ddim"
    RECTIFIED_FLOW = "rectified_flow"
    LATENT_DIFFUSION = "latent_diffusion"
    CHI_SQUARE = "chi_square"


#: Retired spellings of a paradigm tag, normalised before discrimination.
#: Deliberately a CLOSED map rather than a fuzzy match: an unrecognised tag must
#: still raise with the valid set named (non-negotiable #3), so this may only
#: rewrite spellings that are known-equivalent, never coerce nonsense.
DIFFUSION_PARADIGM_ALIASES: dict[str, str] = {
    "cold_diffusion": DiffusionParadigm.COLD.value,
    "ColdDiffusion": DiffusionParadigm.COLD.value,
}

__all__ = [
    # Core enums
    "TrainingMode",
    "GANLossType",
    "LRScheduler",
    "NoiseSchedule",
    "AccelerationSchedule",
    "OptimizerType",
    "OPTIMIZER_ALIASES",
    "OPTIMIZER_NAMES",
    # Workflow vocabulary (imaging-regime x task contract)
    "Modality",
    "Regime",
    "Task",
    "Axis",
    "ArtifactOrigin",
    "Maturity",
    "DatasetType",
    "VolumeFormat",
    "PredictionType",
    "KANType",
    "InputType",
    "LogLevel",
    "ConsoleStream",
    "RankScope",
    "ProgressMode",
    "TrackingService",
    "GradientClipMethod",
    "ActivationFunction",
    "NormalizationType",
    "MetricMode",
    "PrecisionPolicy",
    "SpectraRuntime",
    "ReportTask",
    "ReportStyle",
    "SignalDomain",
    "DiffusionParadigm",
    "DIFFUSION_PARADIGM_ALIASES",
    # Legacy enums (from domain/entities/enums)
    "LossTypes",
    "SchedulerTypes",
    "GeneratorTypes",
    "TrainingModeTypes",
    "DataFormatTypes",
    "OptimizationTypes",
]
