"""Configuration schema modules organised by domain.

Each YAML section under a ``TrainingSettings`` config corresponds to one
schema in this package:

| YAML section          | Schema module                    | Class                       |
|-----------------------|----------------------------------|-----------------------------|
| ``model:``            | ``model.py``                     | ``ModelConfigSchema``       |
| ``data:``             | ``data.py``                      | ``DataConfigSchema``        |
| ``losses:``           | ``loss.py``                      | ``LossConfigSchema``        |
| ``physics:``          | ``physics.py``                   | ``PhysicsConfigSchema``     |
| ``optimization:``     | ``optimization.py``              | ``OptimizationConfigSchema``|
| ``training:``         | ``training/<paradigm>.py``       | ``TrainingConfig*``         |
| ``acceleration:``     | ``acceleration.py``              | ``AccelerationConfigSchema``|
| ``augmentation:``     | ``augmentation.py``              | ``AugmentationConfigSchema``|
| ``validation:``       | ``validation.py``                | ``ValidationConfigSchema``  |
| ``checkpoint:``       | ``checkpoint.py``                | ``CheckpointConfigSchema``  |
| ``early_stopping:``   | ``early_stopping.py``            | ``EarlyStoppingConfigSchema``|
| ``ema:``              | ``ema.py``                       | ``EMAConfigSchema``         |
| ``logging:``          | ``logging.py``                   | ``LoggingConfigSchema``     |
| ``loss_logging:``     | ``loss_logging.py``              | ``LossLoggingConfigSchema`` |
| ``metrics:``          | ``metrics.py``                   | ``MetricsConfigSchema``     |
| ``objectives:``       | ``objectives.py``                | ``ObjectivesConfigSchema``  |
| ``collation:``        | ``collation.py``                 | ``CollationConfigSchema``   |
| (campaign)            | ``campaign.py``                  | ``CampaignConfigSchema``    |
| (manifest)            | ``manifest_schema.py``           | ``ManifestSchema``          |

Cross-cutting modules:

- ``base.py``                  — generic ``ModuleDefinitionSchema``, objectives, services.
- ``enums.py``                 — ``Literal`` whitelists (NoiseSchedule, AccelerationSchedule, ...).
- ``defaults_provider.py``     — environment-aware default factories.
- ``loader.py``                — YAML → ``TrainingSettings`` entry point.
- ``validator_registry.py``    — health-check registry.
- ``multi_stage_config_validator.py`` — pipeline-stage cross-validation.
- ``training/``                — paradigm-specific subschemas (gan/diffusion/vae/...).
- ``strictness.py``            — the three shared ``model_config`` bases (below).

Strictness policy (one SSOT, three tiers — see ``strictness.py``)
----------------------------------------------------------------
Every schema is ``frozen=True`` (immutable after load; non-negotiable #1/#4) and
disables the protected ``model_`` namespace. They differ only in how unknown keys
are handled:

- **StrictSchema**   (``extra='forbid'``) — the default; an unknown key is a
  load-time ``ValidationError`` (no silent-swallow; pitfall #9/#16). 149 classes.
- **CompatSchema**   (``extra='ignore'``) — v6.0<->v6.1 forward-compat blocks
  that must accept additive sub-fields from newer YAML. 66 classes.
- **StrategySchema** (``extra='allow'``) — strategy/adapter/audit blocks that
  forward strategy-specific knobs read downstream via ``getattr``. 14 classes,
  each enumerated in the test allowlist.

New schemas SHOULD inherit one of these bases. The invariant is enforced for
*all* classes — inherited or inline — by
``tests/unit/config/test_schema_strictness_policy.py`` (frozen everywhere,
``extra='allow'`` only on the reviewed allowlist, ``protected_namespaces=()``
wherever a ``model_*`` field exists), so it can no longer silently drift.
"""

from .acceleration import AccelerationConfigSchema
from .augmentation import AugmentationConfigSchema
from .base import (
    DiffusionObjectiveConfig,
    EncoderDecoderDefinitionSchema,
    EXPERIMENT_STATUSES,
    LAUNCH_REFUSED_STATUSES,
    ExperimentMetadataSchema,
    ExperimentStatus,
    GANObjectiveConfig,
    LatentObjectiveConfig,
    ModuleDefinitionSchema,
    ParallelismConfigSchema,
    R1RegularizationConfig,
    ReconstructionObjectiveConfig,
    ServicesConfigSchema,
    SSLConfigSchema,
    TrainingTaskConfigSchema,
)
from .checkpoint import CheckpointConfigSchema
from .data import DataConfigSchema, DatasetSourceSchema
from .early_stopping import EarlyStoppingConfigSchema
from .ema import EMAConfigSchema
from .enums import (
    AccelerationSchedule,
    ActivationFunction,
    ArtifactOrigin,
    Axis,
    DatasetType,
    GANLossType,
    GradientClipMethod,
    InputType,
    KANType,
    LogLevel,
    ConsoleStream,
    ProgressMode,
    RankScope,
    LRScheduler,
    Maturity,
    MetricMode,
    Modality,
    NoiseSchedule,
    NormalizationType,
    OptimizerType,
    PredictionType,
    Regime,
    Task,
    TrainingMode,
    VolumeFormat,
)
from .loader import (
    Config,
    ConfigFactory,
    ConfigValidator,
    JSONConfigLoader,
    YAMLConfigLoader,
    get_config_factory,
)
from .logging import LoggingConfigSchema
from .loss import (
    DiffusionLossesConfig,
    GANLossesConfig,
    LatentLossesConfig,
    LossConfigSchema,
    MetricsConfig,
    ReconstructionLossesConfig,
)
from .loss_logging import LossLoggingConfigSchema
from .loss_schedule import (
    ActionConfig,
    LossScheduleConfigSchema,
    LossScheduleRule,
    MonitorAfterConfig,
    TriggerConfig,
)
from .metrics import MetricsConfigSchema, ProfilingConfigSchema
from .model import ModelComponentSchema, ModelConfigSchema
from .objectives import ObjectiveConfigSchema
from .optimization import OptimizationConfigSchema
from .physics import (
    B0CorrectionConfig,
    CoilSensitivityConfig,
    CompressedSensingConfig,
    DataConsistencyConfig,
    IterativeRefinementConfig,
    KSpaceConfig,
    MotionCorrectionConfig,
    PhaseCorrectionConfig,
    PhaseCorrectionnConfig,  # deprecated alias (doubled-n); remove 2026-08
    PhysicsConfigSchema,
    RegularizationConfig,
)
from .strictness import CompatSchema, StrategySchema, StrictSchema

# Paradigm-specific training subschemas + the ``training_mode``
# dispatch factory. Restored 2026-05-15 (round-6 zero-consumer
# restoration pass — see TODO/audit/CHANGES_RATIONALE.md §7). The
# schemas are dispatch fuel for ``training_mode`` (R4): each carries
# kwargs and conditions that the matching strategy consumes via
# ``hasattr`` / duck-typing. Downstream forks may also import them
# directly to construct typed configs.
from .training import (
    TrainingConfigDiffusion,
    TrainingConfigFederated,
    TrainingConfigFlow,
    TrainingConfigGAN,
    TrainingConfigMetaLearning,
    TrainingConfigMotion,
    TrainingConfigReconstruction,
    TrainingConfigSSL,
    TrainingConfigTTO,
    TrainingConfigVAE,
    create_training_config,
)
from .trellis import (
    TrellisConfigSchema,
    TrellisDecoderConfigSchema,
    TrellisEncoderConfigSchema,
    TrellisTrainerConfigSchema,
    TrellisVAEConfigSchema,
)
from .validation import ValidationConfigSchema

__all__ = [
    # ==================== Strictness bases (policy SSOT) ====================
    "StrictSchema",
    "CompatSchema",
    "StrategySchema",
    # ==================== Enumerations (17 types) ====================
    "TrainingMode",
    "GANLossType",
    "LRScheduler",
    "NoiseSchedule",
    "AccelerationSchedule",
    "OptimizerType",
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
    "GradientClipMethod",
    "ActivationFunction",
    "NormalizationType",
    "MetricMode",
    # ==================== Base Schemas ====================
    "EXPERIMENT_STATUSES",
    "ExperimentMetadataSchema",
    "ExperimentStatus",
    "LAUNCH_REFUSED_STATUSES",
    "EncoderDecoderDefinitionSchema",
    "ModuleDefinitionSchema",
    "R1RegularizationConfig",
    # ==================== Objective Configurations ====================
    "ObjectiveConfigSchema",
    "GANObjectiveConfig",
    "ReconstructionObjectiveConfig",
    "DiffusionObjectiveConfig",
    "LatentObjectiveConfig",
    # ==================== Data Configuration ====================
    "DataConfigSchema",
    "DatasetSourceSchema",
    # ==================== Model Configuration ====================
    "ModelConfigSchema",
    "ModelComponentSchema",
    # ==================== Optimization Configuration ====================
    "OptimizationConfigSchema",
    # ==================== Feature-Specific Configs (8 types) ====================
    "CheckpointConfigSchema",
    "LoggingConfigSchema",
    "LossLoggingConfigSchema",
    "MetricsConfigSchema",
    "ProfilingConfigSchema",
    "EMAConfigSchema",
    "AugmentationConfigSchema",
    "EarlyStoppingConfigSchema",
    "LossScheduleConfigSchema",
    "LossScheduleRule",
    "TriggerConfig",
    "ActionConfig",
    "MonitorAfterConfig",
    "AccelerationConfigSchema",
    "ValidationConfigSchema",
    # ==================== Physics Configurations (10 types) ====================
    "PhysicsConfigSchema",
    "DataConsistencyConfig",
    "CoilSensitivityConfig",
    "KSpaceConfig",
    "RegularizationConfig",
    "IterativeRefinementConfig",
    "PhaseCorrectionConfig",
    "PhaseCorrectionnConfig",
    "B0CorrectionConfig",
    "MotionCorrectionConfig",
    "CompressedSensingConfig",
    # ==================== Loss Configurations (5 types) ====================
    "LossConfigSchema",
    "ReconstructionLossesConfig",
    "GANLossesConfig",
    "DiffusionLossesConfig",
    "LatentLossesConfig",
    "MetricsConfig",
    # ==================== Services & Infrastructure ====================
    "ServicesConfigSchema",
    "TrainingTaskConfigSchema",
    "ParallelismConfigSchema",
    "SSLConfigSchema",
    # ==================== TRELLIS 3D Configs (5 types) ====================
    "TrellisConfigSchema",
    "TrellisVAEConfigSchema",
    "TrellisEncoderConfigSchema",
    "TrellisDecoderConfigSchema",
    "TrellisTrainerConfigSchema",
    # ==================== Paradigm-Specific Training Configs ====================
    "TrainingConfigGAN",
    "TrainingConfigDiffusion",
    "TrainingConfigVAE",
    "TrainingConfigReconstruction",
    "TrainingConfigSSL",
    "TrainingConfigFlow",
    "TrainingConfigFederated",
    "TrainingConfigMetaLearning",
    "TrainingConfigMotion",
    "TrainingConfigTTO",
    "create_training_config",
    # ==================== Loaders ====================
    "Config",
    "ConfigFactory",
    "ConfigValidator",
    "JSONConfigLoader",
    "YAMLConfigLoader",
    "get_config_factory",
]
