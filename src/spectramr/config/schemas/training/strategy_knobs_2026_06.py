"""Typed strategy hyperparameter sub-blocks (2026-06 knob-wiring batch).

The 2026-05-31 strategy audit found a cluster of paradigm strategies whose
loss weights / physics constants were hardcoded as class attributes (or read
with bare ``getattr(self.config.training, "<block>", None)`` against the
``extra="allow"`` :class:`TrainingStrategyConfigSchema`, where a supplied YAML
block is stored as a plain ``dict`` and ``getattr(dict, ...)`` silently returns
the default). Both are pitfall #15 (an advertised knob that is a silent no-op).

This module defines typed, range-validated sub-blocks for those strategies so
the knobs are READ (as typed attributes, not dict ``getattr``), VALIDATED
(``extra="forbid"`` + Field constraints raise on a typo / out-of-range value),
and STAMPED into provenance by the strategy at construction. Each block is
mounted ``Optional`` on :class:`TrainingStrategyConfigSchema`; when absent the
strategy falls back to the documented defaults (and logs the resolved values),
so existing YAMLs keep working while a supplied block is now honored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QSMPipelineTrainingConfigSchema(BaseModel):
    """Loss weights for the differentiable QSM pipeline strategy.

    Read by
    :class:`~spectramr.infrastructure.training.strategies.qsm_pipeline_strategy.QSMPipelineStrategy`.
    Previously these were hardcoded class attributes; a YAML
    ``training.qsm_pipeline.lambda_chi: 2.0`` had no effect.
    """

    model_config = {"extra": "forbid", "frozen": True}

    lambda_field: float = Field(
        default=0.1, ge=0.0, description="Weight of the field-map fidelity term."
    )
    lambda_unwrap: float = Field(
        default=0.1, ge=0.0, description="Weight of the phase-unwrapping term."
    )
    lambda_background: float = Field(
        default=0.1, ge=0.0, description="Weight of the background-field-removal term."
    )
    lambda_chi: float = Field(
        default=1.0, ge=0.0, description="Weight of the susceptibility (chi) data term."
    )
    lambda_tv: float = Field(
        default=1e-3, ge=0.0, description="Weight of the isotropic-TV regulariser on chi."
    )


class QSpaceDiffusionTrainingConfigSchema(BaseModel):
    """Spherical-harmonic regularisation knobs for q-space diffusion recon.

    Read by
    :class:`~spectramr.infrastructure.training.strategies.qspace_diffusion_strategy.QSpaceDiffusionStrategy`.
    Previously hardcoded class attributes — a YAML override was a silent no-op.
    """

    model_config = {"extra": "forbid", "frozen": True}

    sh_max_order: int = Field(
        default=4,
        ge=0,
        le=12,
        description="Maximum (even) spherical-harmonic order for the SH basis.",
    )
    lambda_smooth: float = Field(
        default=1e-3, ge=0.0, description="Laplace-Beltrami smoothness weight on SH coefs."
    )
    lambda_b0: float = Field(
        default=0.1, ge=0.0, description="Weight anchoring the b0 SH coef to the M0 map."
    )
    lambda_sparse_high: float = Field(
        default=1e-4, ge=0.0, description="Sparsity weight on high-order SH coefficients."
    )


class SliceToVolumeTrainingConfigSchema(BaseModel):
    """Through-plane / orthogonal consistency weights for slice-to-volume SR."""

    model_config = {"extra": "forbid", "frozen": True}

    lambda_through_plane: float = Field(
        default=0.1, ge=0.0, description="Weight of the through-plane consistency term."
    )
    lambda_ortho: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Weight of the gradient-isotropy prior: matches the through-plane "
            "gradient texture-energy to the in-plane energy so the synthesised "
            "volume is isotropically sharp (SVR isotropy prior). Replaces the "
            "old degenerate mean-projection cross-MSE; audit 2026-06 D3."
        ),
    )


class SpatiotemporalMRFReconTrainingConfigSchema(BaseModel):
    """SFC Beltrami regularisation weights for spatiotemporal MRF recon."""

    model_config = {"extra": "forbid", "frozen": True}

    lambda_mu: float = Field(
        default=0.01, ge=0.0, description="Weight of the mu Beltrami regulariser."
    )
    lambda_nu: float = Field(
        default=0.01, ge=0.0, description="Weight of the nu Beltrami regulariser."
    )


class RiemannianMRFDiffusionTrainingConfigSchema(BaseModel):
    """Noise-scale bounds for the Riemannian MRF diffusion strategy."""

    model_config = {"extra": "forbid", "frozen": True}

    sigma_min: float = Field(
        default=1e-2, gt=0.0, description="Minimum diffusion noise scale sigma_min."
    )
    sigma_max: float = Field(
        default=1.0, gt=0.0, description="Maximum diffusion noise scale sigma_max."
    )


class ScoreFieldTomographyTrainingConfigSchema(BaseModel):
    """Noise-scale + conditioning knobs for score-field tomography."""

    model_config = {"extra": "forbid", "frozen": True}

    sigma_min: float = Field(
        default=1e-3, gt=0.0, description="Minimum score-matching noise scale sigma_min."
    )
    sigma_max: float = Field(
        default=1.0, gt=0.0, description="Maximum score-matching noise scale sigma_max."
    )
    cond_dim: int = Field(default=5, ge=1, description="Conditioning vector dimensionality.")
    val_num_levels: int = Field(
        default=10, ge=2, description="Annealed-Langevin noise levels (geometric sigma_max→sigma_min)."
    )
    val_steps_per_level: int = Field(
        default=5, ge=1, description="Langevin steps per noise level in validation sampling."
    )
    val_step_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=2.0,
        description="Langevin epsilon; the step at level i is epsilon * sigma_i**2 (stable below 2).",
    )


class SyntheticPathologyAugTrainingConfigSchema(BaseModel):
    """Augmentation probability + lesion-loss weights for synthetic-pathology aug."""

    model_config = {"extra": "forbid", "frozen": True}

    p_augment: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Probability of applying lesion augmentation."
    )
    lambda_lesion_l1: float = Field(
        default=0.5, ge=0.0, description="Weight of the lesion-region L1 term."
    )
    lambda_lesion_cls: float = Field(
        default=0.1, ge=0.0, description="Weight of the lesion-classification term."
    )


class InverseBlochPhaseTrainingConfigSchema(BaseModel):
    """Phase-smoothness weight for the inverse-Bloch phase strategy."""

    model_config = {"extra": "forbid", "frozen": True}

    lambda_phase_smooth: float = Field(
        default=0.01, ge=0.0, description="Weight of the phase-residual smoothness term."
    )


class SpinSDETrainingConfigSchema(BaseModel):
    """Euler-Maruyama SDE knobs for the continuous-time Spin SDE strategy.

    ``diffusion_init`` initialises the *learnable* diffusion coefficient
    (promoted to an nn.Parameter by the strategy); the rest are fixed
    integration / loss-weight hyperparameters.
    """

    model_config = {"extra": "forbid", "frozen": True}

    n_steps: int = Field(
        default=32, ge=1, description="Number of Euler-Maruyama integration steps."
    )
    dt: float = Field(default=1e-3, gt=0.0, description="Integration time step dt (seconds).")
    diffusion_init: float = Field(
        default=0.05, ge=0.0, description="Initial value of the learnable diffusion coefficient."
    )
    lambda_consistency: float = Field(
        default=1.0, ge=0.0, description="Weight of the Bloch-consistency term."
    )
    lambda_diffusion_prior: float = Field(
        default=1e-3, ge=0.0, description="Weight of the diffusion-coefficient ELBO prior."
    )
    echo_times: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Optional per-contrast readout schedule for the Bloch-consistency "
            "loss, in SECONDS (same units as ``dt``). Each value is mapped to "
            "the nearest Euler-Maruyama step (``round(t / dt)``, clamped to "
            "``[1, n_steps]``) and the SDE's transverse magnitude is read out "
            "there, then matched to the corresponding channel of "
            "``multi_contrast_signal``. Its length MUST equal the observed "
            "contrast count. When omitted, readouts are evenly spaced over the "
            "trajectory and end at the final step (so a single-contrast target "
            "reduces exactly to the legacy final-state readout)."
        ),
    )


class CorticalConformalFMRIReconTrainingConfigSchema(BaseModel):
    """Curvature-regularisation weight for cortical-conformal fMRI recon."""

    model_config = {"extra": "forbid", "frozen": True}

    lambda_curvature: float = Field(
        default=0.1, ge=0.0, description="Weight of the cortical-curvature regulariser."
    )


class RiemannianDFCDiffusionTrainingConfigSchema(BaseModel):
    """Noise-scale bounds for the Riemannian dynamic-FC diffusion strategy."""

    model_config = {"extra": "forbid", "frozen": True}

    sigma_min: float = Field(
        default=1e-2, gt=0.0, description="Minimum diffusion noise scale sigma_min."
    )
    sigma_max: float = Field(
        default=1.0, gt=0.0, description="Maximum diffusion noise scale sigma_max."
    )


class HRFManifoldDiffusionTrainingConfigSchema(BaseModel):
    """Noise-scale bounds for the HRF-manifold diffusion strategy."""

    model_config = {"extra": "forbid", "frozen": True}

    sigma_min: float = Field(
        default=1e-2, gt=0.0, description="Minimum diffusion noise scale sigma_min."
    )
    sigma_max: float = Field(
        default=1.0, gt=0.0, description="Maximum diffusion noise scale sigma_max."
    )


class MRISLAMTrainingConfigSchema(BaseModel):
    """Loss weights + trajectory-delta LR for the MRI-SLAM strategy.

    Previously the smoothness / drift / photometric weights and the per-subject
    trajectory-delta optimizer LR were hardcoded inline constants.
    """

    model_config = {"extra": "forbid", "frozen": True}

    lambda_smooth: float = Field(
        default=1e-3, ge=0.0, description="Weight of the trajectory-smoothness term."
    )
    lambda_drift: float = Field(
        default=1e-4, ge=0.0, description="Weight of the running-mean drift regulariser."
    )
    lambda_photometric: float = Field(
        default=0.5, ge=0.0, description="Weight of the photometric NUFFT/proxy residual."
    )
    trajectory_lr: float = Field(
        default=1e-4, gt=0.0, description="Learning rate for the per-subject trajectory deltas."
    )


class MAEPretrainingTrainingConfigSchema(BaseModel):
    """Masked-autoencoder pretraining knobs.

    ``mask_ratio`` was previously read only from the untyped ``task_settings``
    dict (silent default); ``mask_domain`` is constrained so a typo (e.g.
    "k-space") raises instead of silently falling back to image-domain masking.
    """

    model_config = {"extra": "forbid", "frozen": True}

    mask_ratio: float = Field(
        default=0.75,
        ge=0.0,
        lt=1.0,
        description="Fraction of patches/lines masked (0.75 = 75%).",
    )
    mask_domain: Literal["image", "kspace"] = Field(
        default="image", description="Domain in which masking is applied."
    )
    patch_size: int = Field(
        default=16,
        ge=1,
        description="Side in pixels of the square patch masked when mask_domain is 'image'.",
    )


class SchrodingerBridgeTrainingConfigSchema(BaseModel):
    """Bloch-manifold penalty weight + cache resolution for the Schrodinger bridge.

    ``lambda_bloch`` was a constructor-only kwarg the strategy factory never
    passed (it forwards services only), so a YAML value was unreachable.
    """

    model_config = {"extra": "forbid", "frozen": True}

    lambda_bloch: float = Field(
        default=0.1, ge=0.0, description="Weight on the Bloch-manifold-consistency penalty."
    )
    bloch_cache_resolution: int = Field(
        default=16,
        ge=0,
        description="Per-axis resolution of the relaxation-manifold cache (0 = off).",
    )


class TTTTrainingConfigSchema(BaseModel):
    """Test-time-training adaptation knobs.

    Previously the strategy read these off ``model.adaptation_config`` (a
    meta-learning block), conflating two paradigms. This typed
    ``training.ttt`` block is the canonical home.
    """

    model_config = {"extra": "forbid", "frozen": True}

    adaptation_steps: int = Field(
        default=10, ge=1, description="Number of inner test-time adaptation steps."
    )
    adaptation_lr: float = Field(
        default=1e-3, gt=0.0, description="Inner-loop adaptation learning rate."
    )
    consistency_weight: float = Field(
        default=1.0, ge=0.0, description="Weight of the data-consistency adaptation loss."
    )


class VFConsistencyDistillationTrainingConfigSchema(BaseModel):
    """Virtual-fiducial consistency-distillation knobs (were a _DEFAULTS dict)."""

    model_config = {"extra": "forbid", "frozen": True}

    beta0: float = Field(default=0.5, ge=0.0, description="Base consistency-distillation weight.")
    ema_decay: float = Field(default=0.999, ge=0.0, le=1.0, description="EMA-teacher decay.")
    sigma_data: float = Field(
        default=0.5, gt=0.0, description="Data standard deviation for preconditioning."
    )
    num_consistency_steps: int = Field(
        default=4, ge=1, description="Number of consistency-distillation steps."
    )
    vf_grid_spacing: int = Field(
        default=16, ge=1, description="Virtual-fiducial marker grid spacing (pixels)."
    )
    vf_im_size: int | None = Field(
        default=None, description="Override image size for the VF marker grid (None = infer)."
    )


class TrainingConfigTrajectoryRecon(BaseModel):
    """Spiral trajectory-recovery knobs (exp_vf_35, read by TrajectoryReconstructionStrategy).

    The model estimates a GIRF/delay parametrisation theta and predicts the
    spiral trajectory deviation Delta_k(t); the strategy SUPERVISES it against
    the measured deviation Delta_k_true = trajectory_measured - trajectory_nominal
    (the only differentiable path — torchkbnufft does not pass gradients to the
    trajectory, so a through-NUFFT sharpness objective cannot train theta). The
    sharpness knobs drive a no-grad VALIDATION diagnostic only.
    """

    model_config = {"extra": "forbid", "frozen": True}

    read_measured_trajectory: bool = Field(
        default=True,
        description="Read trajectory_measured/trajectory_nominal from the batch and "
        "supervise Delta_k against Delta_k_true. Requires the dataset to emit them "
        "(ismrmrd_kspace.emit_paired_trajectory=true).",
    )
    trajectory_units: Literal["cycles_per_fov", "rad_per_pixel"] = Field(
        default="cycles_per_fov",
        description="Units the trajectory metric grades in (must match the dataset).",
    )
    girf_param_dim: int = Field(
        default=256,
        ge=1,
        description="GIRF FIR kernel length per axis (the theta dimension).",
    )
    self_supervised_sharpness: bool = Field(
        default=False,
        description="Report a no-grad image-sharpness gain diagnostic of the "
        "estimated trajectory vs the nominal one (gradient_entropy). NOT a training "
        "objective — the NUFFT does not backprop to theta (pitfall #16).",
    )
    sharpness_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight of the sharpness DIAGNOSTIC in reporting (0 = report only). "
        "Has no effect on the supervised trajectory training loss.",
    )
