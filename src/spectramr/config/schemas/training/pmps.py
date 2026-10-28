"""PMPS training schemas — bundled for the four PMPS phases.

One file per phase would be slop; the four schemas share the same
structural pattern (paradigm-specific block on top of
``BaseTrainingConfigSchema``) and can co-exist here without coupling.

Phases covered:

* :class:`TrainingConfigTissueDiffusionPretrain` — Phase 1, tissue
  parameter diffusion prior pretraining.
* :class:`TrainingConfigCorruptionCalibration` — Phase 2,
  :math:`\\psi` calibration via MMD.
* :class:`TrainingConfigPairedSynthesis` — Phase 3, frozen-model paired
  synthesis pipeline.
* :class:`TrainingConfigDataEfficiencyHarness` — Phase 4, downstream
  utility data-efficiency curve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from spectramr.config.schemas.data import AcquisitionParamsSchema

from .base import BaseTrainingConfigSchema

# ----------------------------------------------------------------------
# Phase 1 — TissueDiffusionPretrain.
# ----------------------------------------------------------------------


class SiteConditioningConfig(BaseModel):
    """Per-site conditioning for the tissue-parameter prior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(default=False)
    n_sites: int = Field(default=1, ge=1)
    site_embed_dim: int = Field(default=32, ge=1)


class PseudoQMRIBackrecoveryConfig(BaseModel):
    """Pseudo-qMRI back-recovery from multi-contrast HF (Phase 1).

    When the heterogeneous corpus carries multi-contrast HF images but
    no qMRI maps, the BlochSignalEncoder is used in reverse to recover
    approximate tissue parameters. ``pseudo_qmri_noise_inflation``
    widens the diffusion-prior variance on back-recovered samples to
    reflect the additional uncertainty.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(default=True)
    pseudo_qmri_noise_inflation: float = Field(default=2.0, gt=0.0)


class TrainingConfigTissueDiffusionPretrain(BaseTrainingConfigSchema):
    """Tissue-parameter diffusion prior pretraining (PMPS Phase 1)."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    timesteps: int = Field(default=1000, ge=1)
    noise_schedule: Literal["cosine", "linear", "edm"] = Field(default="cosine")
    prediction_type: Literal["epsilon", "v", "x0"] = Field(default="v")
    site_conditioning: SiteConditioningConfig = Field(default_factory=SiteConditioningConfig)
    field_strength_conditioning: bool = Field(default=True)
    pseudo_qmri_backrecovery: PseudoQMRIBackrecoveryConfig = Field(
        default_factory=PseudoQMRIBackrecoveryConfig
    )


# ----------------------------------------------------------------------
# Phase 2 — CorruptionCalibration.
# ----------------------------------------------------------------------


class ULFCorruptionConfig(BaseModel):
    """Calibratable ULF corruption operator parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snr_scaling_law: Literal["hoult", "learned"] = Field(default="hoult")
    concomitant_enabled: bool = Field(default=True)
    coil_geometry: Literal["hyperfine_swoop", "siemens_head_8ch", "generic_birdcage"] = Field(
        default="hyperfine_swoop"
    )
    motion_severity: float = Field(default=0.1, ge=0.0, le=1.0)
    reconstruction_artifact_severity: float = Field(default=0.05, ge=0.0, le=1.0)


class HFCorruptionConfig(BaseModel):
    """Calibratable HF corruption operator parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rician_noise_sigma: float = Field(default=0.02, ge=0.0, le=1.0)
    bias_field_severity: float = Field(default=0.05, ge=0.0, le=1.0)
    motion_severity: float = Field(default=0.01, ge=0.0, le=1.0)


class MMDConfig(BaseModel):
    """MMD configuration for paired-distribution calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_extractor: Literal["pixel", "vgg16_imagenet", "vgg16_medical", "dino"] = Field(
        default="pixel"
    )
    kernel_bandwidths: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    per_contrast_weights: dict[str, float] = Field(
        default_factory=lambda: {"T1w": 1.0, "T2w": 1.0, "FLAIR": 1.0, "PDw": 1.0}
    )

    @field_validator("kernel_bandwidths")
    @classmethod
    def _check_bandwidths(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("kernel_bandwidths must be a non-empty list.")
        if any(b <= 0 for b in v):
            raise ValueError("kernel_bandwidths must all be > 0.")
        return v


class TrainingConfigCorruptionCalibration(BaseTrainingConfigSchema):
    """Corruption-operator calibration training config (PMPS Phase 2)."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    ulf_corruption: ULFCorruptionConfig = Field(default_factory=ULFCorruptionConfig)
    hf_corruption: HFCorruptionConfig = Field(default_factory=HFCorruptionConfig)
    mmd: MMDConfig = Field(default_factory=MMDConfig)
    calibration_iters: int = Field(default=5000, ge=1)


# ----------------------------------------------------------------------
# Phase 3 — PairedSynthesis.
# ----------------------------------------------------------------------


# ONE definition of an acquisition tuple, not two. This was a re-declaration of
# `AcquisitionParamsSchema` (config/schemas/data.py) with the same eight fields, the same
# types and the same defaults -- except `contrast_type`, whose Literal omitted
# `diffusion_weighted`. So the identical acquisition dict validated as a `data:` entry and
# was REJECTED as a PMPS protocol: two spellings of one concept that agreed until they did
# not (issue #828, the shape pitfall #13b describes for loss-weight resolvers).
#
# The data-layer schema is elected because it is the wider of the two (narrowing would
# invalidate the 4 configs using `diffusion_weighted`) and because this module's own test,
# `tests/unit/config/test_pmps_schema_dispatch.py`, already imports IT rather than the local
# copy. Aliasing only widens what PMPS accepts; nothing that validated before stops.
#
# The name is kept so `fixed_protocols` and `__all__` are unchanged for callers.
AcquisitionParam = AcquisitionParamsSchema


class ProtocolSamplingConfig(BaseModel):
    """How synthesis chooses (P_ULF, P_HF) per emitted pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["fixed", "uniform_over_corpus", "site_balanced"] = Field(default="site_balanced")
    fixed_protocols: list[AcquisitionParam] = Field(default_factory=list)
    site_weights: dict[str, float] = Field(default_factory=dict)


class TrainingConfigPairedSynthesis(BaseTrainingConfigSchema):
    """Paired-synthesis pipeline config (PMPS Phase 3)."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    n_synthetic_pairs: int = Field(default=10_000, ge=1)
    protocol_sampling: ProtocolSamplingConfig = Field(default_factory=ProtocolSamplingConfig)
    output_format: Literal["nifti", "h5", "npz"] = Field(default="nifti")
    output_root: Path
    defensibility_threshold: float = Field(default=1.5, ge=1.0, le=5.0)
    write_rejected: bool = Field(default=True)
    parallel_shards: int = Field(default=8, ge=1)


# ----------------------------------------------------------------------
# Phase 4 — DataEfficiencyHarness.
# ----------------------------------------------------------------------


class DataEfficiencySettingConfig(BaseModel):
    """One cell of the harness experiment grid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    n_real: int = Field(default=10, ge=0)
    n_synthetic: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=100, ge=1)


class TrainingConfigDataEfficiencyHarness(BaseTrainingConfigSchema):
    """Downstream-utility data-efficiency harness (PMPS Phase 4)."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    downstream_translator_model: str = Field(
        default="enhanced_unet",
        description="Registry name of the translator trained per cell.",
    )
    downstream_translator_kwargs: dict[str, Any] = Field(default_factory=dict)
    real_paired_manifest: Path
    synthetic_paired_manifest: Path
    settings: list[DataEfficiencySettingConfig] = Field(default_factory=list)
    n_seeds_per_setting: int = Field(default=5, ge=1)
    loso_n_folds: int = Field(default=10, ge=1)
    primary_metric: Literal["lpips", "ssim", "psnr"] = Field(default="lpips")
    secondary_metrics: list[str] = Field(default_factory=lambda: ["ssim", "psnr", "haarpsi"])
    bootstrap_n_resamples: int = Field(default=1000, ge=1)
    multiple_comparison_correction: Literal["holm_sidak", "bonferroni", "fdr_bh"] = Field(
        default="holm_sidak"
    )

    @field_validator("settings")
    @classmethod
    def _at_least_one_setting(
        cls, v: list[DataEfficiencySettingConfig]
    ) -> list[DataEfficiencySettingConfig]:
        if not v:
            raise ValueError("data_efficiency_harness.settings must contain at least one cell.")
        return v


__all__ = [
    "AcquisitionParam",
    "DataEfficiencySettingConfig",
    "HFCorruptionConfig",
    "MMDConfig",
    "ProtocolSamplingConfig",
    "PseudoQMRIBackrecoveryConfig",
    "SiteConditioningConfig",
    "TrainingConfigCorruptionCalibration",
    "TrainingConfigDataEfficiencyHarness",
    "TrainingConfigPairedSynthesis",
    "TrainingConfigTissueDiffusionPretrain",
    "ULFCorruptionConfig",
]
