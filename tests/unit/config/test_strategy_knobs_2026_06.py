"""Tests for the 2026-06 typed strategy-knob sub-blocks.

Proves each block (a) validates its fields (extra="forbid" + ranges) and
(b) is mounted on TrainingStrategyConfigSchema so a YAML mapping is coerced to
the *typed* block — not stored as a plain dict (which is what made
``getattr(self.config.training.<block>, "x", default)`` a silent no-op).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
from spectramr.config.schemas.training.strategy_knobs_2026_06 import (
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
    TTTTrainingConfigSchema,
    VFConsistencyDistillationTrainingConfigSchema,
)


def test_qsm_pipeline_schema_validates_fields() -> None:
    cfg = QSMPipelineTrainingConfigSchema(lambda_chi=2.0, lambda_tv=0.0)
    assert cfg.lambda_chi == 2.0
    assert cfg.lambda_field == 0.1  # default

    # Negative weight is rejected (ge=0).
    with pytest.raises(ValidationError):
        QSMPipelineTrainingConfigSchema(lambda_chi=-1.0)

    # A typo'd knob is rejected (extra="forbid") instead of silently ignored.
    with pytest.raises(ValidationError):
        QSMPipelineTrainingConfigSchema(lambda_chii=2.0)


def test_qsm_pipeline_block_mounted_and_coerced_to_typed() -> None:
    """A YAML mapping at training.qsm_pipeline must become the typed block,
    not a plain dict — so the strategy reads validated typed attributes."""
    schema = TrainingStrategyConfigSchema(qsm_pipeline={"lambda_chi": 3.0})
    assert isinstance(schema.qsm_pipeline, QSMPipelineTrainingConfigSchema)
    assert schema.qsm_pipeline.lambda_chi == 3.0
    # Absent block defaults to None (strategy then uses its documented defaults).
    assert TrainingStrategyConfigSchema().qsm_pipeline is None


def test_qspace_diffusion_schema_and_mount() -> None:
    cfg = QSpaceDiffusionTrainingConfigSchema(sh_max_order=6, lambda_b0=0.2)
    assert cfg.sh_max_order == 6 and cfg.lambda_b0 == 0.2

    with pytest.raises(ValidationError):
        QSpaceDiffusionTrainingConfigSchema(sh_max_order=99)  # le=12
    with pytest.raises(ValidationError):
        QSpaceDiffusionTrainingConfigSchema(lambda_smooth=-1.0)  # ge=0

    schema = TrainingStrategyConfigSchema(qspace_diffusion={"sh_max_order": 8})
    assert isinstance(schema.qspace_diffusion, QSpaceDiffusionTrainingConfigSchema)
    assert schema.qspace_diffusion.sh_max_order == 8


# Each tranche-2 block: (attr on TrainingStrategyConfigSchema, schema class,
# a valid kwarg, an invalid kwarg that must raise).
_TRANCHE2 = [
    ("slice_to_volume", SliceToVolumeTrainingConfigSchema,
     {"lambda_ortho": 0.2}, {"lambda_ortho": -1.0}),
    ("spatiotemporal_mrf_recon", SpatiotemporalMRFReconTrainingConfigSchema,
     {"lambda_mu": 0.5}, {"lambda_mu": -0.1}),
    ("riemannian_mrf_diffusion", RiemannianMRFDiffusionTrainingConfigSchema,
     {"sigma_min": 0.2}, {"sigma_min": 0.0}),  # gt=0
    ("score_field_tomography", ScoreFieldTomographyTrainingConfigSchema,
     {"cond_dim": 7}, {"cond_dim": 0}),  # ge=1
    ("score_field_tomography", ScoreFieldTomographyTrainingConfigSchema,
     {"val_num_levels": 4, "val_step_size": 0.5}, {"val_step_size": 2.0}),  # lt=2
    ("synthetic_pathology_aug", SyntheticPathologyAugTrainingConfigSchema,
     {"p_augment": 0.3}, {"p_augment": 1.5}),  # le=1
    ("inverse_bloch_phase", InverseBlochPhaseTrainingConfigSchema,
     {"lambda_phase_smooth": 0.05}, {"lambda_phase_smooth": -1.0}),
    ("spin_sde", SpinSDETrainingConfigSchema,
     {"n_steps": 16}, {"n_steps": 0}),  # ge=1
    ("cortical_conformal_fmri_recon", CorticalConformalFMRIReconTrainingConfigSchema,
     {"lambda_curvature": 0.2}, {"lambda_curvature": -1.0}),
    ("riemannian_dfc_diffusion", RiemannianDFCDiffusionTrainingConfigSchema,
     {"sigma_max": 2.0}, {"sigma_max": 0.0}),  # gt=0
    ("hrf_manifold_diffusion", HRFManifoldDiffusionTrainingConfigSchema,
     {"sigma_min": 0.05}, {"sigma_min": 0.0}),  # gt=0
    ("mri_slam", MRISLAMTrainingConfigSchema,
     {"lambda_photometric": 0.25}, {"trajectory_lr": 0.0}),  # gt=0
    ("mae", MAEPretrainingTrainingConfigSchema,
     {"mask_ratio": 0.5}, {"mask_ratio": 1.0}),  # lt=1
    ("schrodinger_bridge", SchrodingerBridgeTrainingConfigSchema,
     {"lambda_bloch": 0.2}, {"bloch_cache_resolution": -1}),  # ge=0
    ("ttt", TTTTrainingConfigSchema,
     {"adaptation_steps": 5}, {"adaptation_lr": 0.0}),  # gt=0
    ("vf_consistency_distillation", VFConsistencyDistillationTrainingConfigSchema,
     {"beta0": 0.6, "vf_im_size": 128}, {"ema_decay": 1.5}),  # le=1
]


def test_mae_mask_domain_rejects_typo() -> None:
    # Literal["image","kspace"] — a typo'd domain must raise, not silently
    # fall back to image-domain masking (pitfall #9).
    MAEPretrainingTrainingConfigSchema(mask_domain="kspace")
    with pytest.raises(ValidationError):
        MAEPretrainingTrainingConfigSchema(mask_domain="k-space")


def test_spin_sde_echo_times_optional_tuple() -> None:
    # echo_times is an optional per-contrast readout schedule (seconds).
    assert SpinSDETrainingConfigSchema().echo_times is None
    block = SpinSDETrainingConfigSchema(echo_times=[0.005, 0.016, 0.032])
    assert block.echo_times == (0.005, 0.016, 0.032)
    # A YAML list is coerced through the mounted block too.
    schema = TrainingStrategyConfigSchema(spin_sde={"echo_times": [0.01, 0.02]})
    assert schema.spin_sde.echo_times == (0.01, 0.02)


@pytest.mark.parametrize("attr,cls,ok,bad", _TRANCHE2)
def test_tranche2_block_validates_and_mounts(attr, cls, ok, bad) -> None:
    # Valid kwargs construct; an out-of-range kwarg raises; an unknown key raises.
    inst = cls(**ok)
    assert isinstance(inst, cls)
    with pytest.raises(ValidationError):
        cls(**bad)
    with pytest.raises(ValidationError):
        cls(definitely_not_a_knob=1)

    # The block is mounted and a YAML mapping is coerced to the typed block.
    schema = TrainingStrategyConfigSchema(**{attr: ok})
    assert isinstance(getattr(schema, attr), cls)
    assert getattr(TrainingStrategyConfigSchema(), attr) is None
