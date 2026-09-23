"""Loss configuration schema — SSOT for loss management.

This file is paradigm-organised and intentionally large (one schema per
paradigm). Use the table of contents below to navigate.

╭────────────────────── Table of Contents ───────────────────────╮
│  Line  Class                          Domain / Paradigm        │
├────────────────────────────────────────────────────────────────┤
│   40   ReconstructionLossesConfig     image / recon            │
│  593   GANLossesConfig                adversarial              │
│  658   DiffusionLossesConfig          diffusion                │
│  716   LatentLossesConfig             VAE / VQ-VAE / latent    │
│  792   PhysicsLossesConfig            kspace / physics priors  │
│  880   SSLLossesConfig                self-supervised          │
│  913   EvidentialLossesConfig         uncertainty              │
│  930   SpatialLossesConfig            spatial regularisers     │
│  943   RegistrationLossesConfig       deformable registration  │
│  957   PINNLossesConfig               PDE-residual losses      │
│  989   LossComponentConfig            entry of *_losses lists  │
│ 1026   ComposedLossConfig             multi-term composition   │
│ 1069   MetricsConfig                  PSNR/SSIM/LPIPS toggles  │
│ 1134   LossConfigSchema               top-level container      │
╰────────────────────────────────────────────────────────────────╯

Layout reasoning:

- Each paradigm has its own ``*LossesConfig`` so a YAML editor can
  reference only the section relevant to their experiment without
  scrolling the whole file.
- ``LossComponentConfig`` is the *entry* type used inside
  ``image_losses: [...]`` / ``kspace_losses: [...]`` / ``complex_losses: [...]``
  lists in YAML — each entry has ``name`` (string, looked up in
  ``LossRegistry``) plus ``weight`` and free ``kwargs``.
- ``LossConfigSchema`` at the bottom is what consumers import.

This is the SINGLE SOURCE OF TRUTH for all loss configuration.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from spectramr.config.schemas.enums import SignalDomain
from spectramr.config.schemas.strictness import CompatSchema

from .renames import (
    fold_renamed_keys,
    folded_input_keys,
    folded_input_paths,
    reject_renamed_keys,
    renames_for_block,
)

#: Which :class:`SignalDomain` each ``losses.*_losses`` list grades in.
#:
#: The SSOT for that mapping. ``check_loss_domain_block_match`` used to carry
#: its own ``{"complex_losses": "complex"}`` literal, which agreed with the
#: registry's adapter only by coincidence -- both happened to spell it
#: ``complex`` on the raw side. Two tables for one fact, currently coincident,
#: is the defect this phase exists to remove; the checker reads this one.
#:
#: Note the list name is NOT mechanically ``f"{domain.value}_losses"``:
#: ``complex_image`` is spelled ``complex_losses`` in YAML. That irregularity is
#: exactly why the mapping has to be written down once rather than derived
#: twice.
LOSS_LIST_DOMAINS: dict[str, SignalDomain] = {
    "kspace_losses": SignalDomain.KSPACE,
    "image_losses": SignalDomain.IMAGE,
    "complex_losses": SignalDomain.COMPLEX_IMAGE,
    "latent_losses": SignalDomain.LATENT,
}

__all__ = [
    "ComposedLossConfig",
    "DiffusionLossesConfig",
    "EvidentialLossesConfig",
    "GANLossesConfig",
    "LatentLossesConfig",
    "LossComponentConfig",
    "LossConfigSchema",
    "LossPolicyConfigSchema",
    "MetricsConfig",
    "PhysicsLossesConfig",
    "ReconstructionLossesConfig",
    "SSLLossesConfig",
]


class ReconstructionLossesConfig(CompatSchema):
    """Reconstruction loss configuration (SSOT for all lambda weights).

    Contains both enable flags and lambda weights for all reconstruction losses.
    Lambda weights control the contribution of each loss to the total objective.
    Enable flags determine whether to compute a loss at all.

    Inherits CompatSchema (extra='ignore', frozen, protected_namespaces=()) to
    match its sibling ``*LossesConfig`` blocks. Previously declared *no*
    ``model_config`` at all, so it was silently mutable and non-strict.
    """

    # ==================== SCHEDULING & WARMUP (Stabilization) ====================
    warmup_iterations: int = Field(
        default=1000,
        ge=0,
        description="Number of initial iterations during which spatial/mixed-domain losses are masked (set to 0.0) to stabilize k-space mapping.",
    )
    warmup_losses: list[str] | None = Field(
        default=None,
        description=(
            "Losses masked to 0.0 while iteration < warmup_iterations. None => the "
            "legacy set (LEGACY_WARMUP_LOSSES in models/losses/weights.py). Named here "
            "so the set is configurable and stamped into provenance rather than being a "
            "hardcoded literal duplicated across resolvers. Resolved by "
            "``build_loss_weight_table``; an empty list disables the gate."
        ),
    )

    # ==================== CORE RECONSTRUCTION LOSSES ====================
    lambda_reconstruction: float = Field(
        default=1.0,
        ge=0,
        description=(
            "Umbrella weight for the strategy-computed reconstruction term (the one in "
            "STRATEGY_MANAGED_LOSSES, probed by UnifiedReconstructionLossComputer). It "
            "had no schema field, so it fell through to the hardcoded default tables. "
            "Declared here so the knob has ONE visible, auditable home (pitfall #15); "
            "1.0 preserves the legacy value."
        ),
    )

    enable_l1: bool = Field(default=False, description="Enable L1 (MAE) loss")
    lambda_l1: float = Field(default=10.0, ge=0, description="L1 (MAE) loss weight")

    enable_l2: bool = Field(default=False, description="Enable L2 (MSE) loss")
    lambda_l2: float = Field(default=0.0, ge=0, description="L2 (MSE) loss weight")

    enable_smooth_l1: bool = Field(default=False, description="Enable Smooth L1 (Huber) loss")
    lambda_smooth_l1: float = Field(default=0.0, ge=0, description="Smooth L1 weight")

    enable_complex_l1: bool = Field(default=False, description="Enable complex L1 loss")
    lambda_complex_l1: float = Field(default=0.0, ge=0, description="Complex L1 weight")
    # OPT-IN pre-DC fidelity supervision (Experiment-11 DC-blob L1+). When > 0
    # and the generator exposes its pre-DC prediction (kspace_cold_diffusion),
    # DiffusionTrainingStrategy._add_pre_dc_fidelity adds a k-space L1 between
    # that PRE-DC prediction and the target, forcing the net to produce
    # measurement-dependent HF itself rather than leaning on the soft-DC-
    # injected ACS centre. Default 0.0 -> no extra term (existing behaviour).
    lambda_pre_dc_kspace: float = Field(
        default=0.0,
        ge=0,
        description="Pre-DC k-space L1 fidelity weight (DC-blob L1+; 0 disables)",
    )
    # Its sibling, on the ACQUIRED bins. The plainer name above is the
    # null-band term -- `_add_pre_dc_fidelity` masks it with `_unsampled_weight`
    # -- so under `dc_method: hard` NOTHING scores the bins the scanner
    # measured: hard DC makes d(output)/d(prediction) exactly (1 - M), which
    # zeroes every post-DC term there too. The acquired bins are the only place
    # the target is knowable from the input, so they are where the output scale
    # and the inter-coil phase relationship are learnable at all; in the null
    # band the L1 optimum is ~0 and phase is undefined. Default 0.0 keeps every
    # arm that does not declare it bit-for-bit unchanged.
    lambda_pre_dc_acquired: float = Field(
        default=0.0,
        ge=0,
        description="Pre-DC k-space L1 on the ACQUIRED bins (calibration anchor; 0 disables)",
    )

    # ==================== PERCEPTUAL LOSSES ====================
    enable_perceptual: bool = Field(default=False, description="Enable perceptual (VGG) loss")
    lambda_perceptual: float = Field(default=10.0, ge=0, description="Perceptual loss weight")

    enable_ssim: bool = Field(default=False, description="Enable SSIM loss")
    lambda_ssim: float = Field(default=0.0, ge=0, description="SSIM loss weight")

    enable_ms_ssim: bool = Field(default=False, description="Enable Multi-Scale SSIM loss")
    lambda_ms_ssim: float = Field(default=0.0, ge=0, description="Multi-Scale SSIM weight")

    enable_lpips: bool = Field(default=False, description="Enable LPIPS loss")
    lambda_lpips: float = Field(default=0.0, ge=0, description="LPIPS loss weight")

    enable_dino_perceptual: bool = Field(default=False, description="Enable DINOv2 perceptual loss")
    lambda_dino_perceptual: float = Field(default=0.0, ge=0, description="DINOv2 weight")

    enable_hfen: bool = Field(default=False, description="Enable HFEN loss")
    lambda_hfen: float = Field(default=0.0, ge=0, description="HFEN weight")

    enable_dists: bool = Field(
        default=False, description="Enable DISTS (Deep Image Structure/Texture) loss"
    )
    lambda_dists: float = Field(default=0.0, ge=0, description="DISTS loss weight")

    # ==================== CURRICULUM SCHEDULING ====================
    use_curriculum_scheduling: bool = Field(
        default=False,
        description="Enable epoch-based curriculum scheduling of loss weights (L1 decay, HFEN/Grad increase)",
    )
    curriculum_warmup_epochs: float = Field(
        default=20.0,
        description="Epochs to keep weights static before starting transition",
    )
    curriculum_max_epochs: float = Field(
        default=100.0, description="Maximum epoch for the transition schedule to end"
    )
    curriculum_l1_end_weight: float = Field(
        default=1.0, description="Final weight for L1 at end of curriculum"
    )
    curriculum_hfen_end_weight: float = Field(
        default=5.0, description="Final weight for HFEN at end of curriculum"
    )

    # ==================== EXPLICIT GRADIENT ====================
    enable_explicit_gradient: bool = Field(
        default=False,
        description="Enable explicit first-order edge penalty using Sobel gradient L1",
    )
    lambda_explicit_gradient: float = Field(
        default=0.0, ge=0, description="Explicit gradient loss weight"
    )

    # ==================== K-SPACE / FREQUENCY LOSSES ====================
    enable_kspace: bool = Field(default=False, description="Enable k-space consistency loss")
    lambda_kspace: float = Field(default=0.0, ge=0, description="K-space consistency weight")

    enable_frequency: bool = Field(default=False, description="Enable frequency-domain loss")
    lambda_frequency: float = Field(default=0.0, ge=0, description="Frequency-domain weight")

    enable_log_spectral: bool = Field(default=False, description="Enable log-spectral loss")
    lambda_log_spectral: float = Field(default=0.0, ge=0, description="Log-spectral weight")
    log_spectral_skip_fft: bool = Field(
        default=False, description="Skip FFT in log-spectral loss (already in k-space)"
    )

    enable_spectral_kspace: bool = Field(default=False, description="Enable spectral k-space loss")
    lambda_spectral_kspace: float = Field(default=0.0, ge=0, description="Spectral k-space weight")

    enable_weighted_kspace_l1: bool = Field(
        default=False, description="Enable frequency-weighted k-space L1"
    )
    lambda_weighted_kspace_l1: float = Field(
        default=0.0, ge=0, description="Frequency-weighted k-space L1"
    )
    weighted_kspace_exponent: float = Field(
        default=1.0, ge=0.0, description="Exponent for frequency-weighted k-space L1"
    )

    enable_frequency_domain: bool = Field(
        default=False, description="Enable frequency domain consistency"
    )
    lambda_frequency_domain: float = Field(
        default=0.0, ge=0, description="Frequency domain consistency"
    )

    enable_ffl: bool = Field(default=False, description="Enable focal frequency loss (FFL)")
    lambda_ffl: float = Field(default=0.0, ge=0, description="Focal frequency loss weight")
    ffl_alpha: float = Field(
        default=1.0,
        ge=0,
        description="Focal frequency loss alpha parameter (focal weighting exponent)",
    )

    # ==================== FREQUENCY-WEIGHTED K-SPACE L1 (Phase 5) ====================
    enable_frequency_weighted_l1_kspace: bool = Field(
        default=False,
        description="Enable physics-informed frequency-weighted L1 loss for k-space (prevents blur via radial frequency weighting)",
    )
    lambda_frequency_weighted_l1_kspace: float = Field(
        default=0.0,
        ge=0,
        description="Frequency-weighted k-space L1 loss weight",
    )
    frequency_weighted_l1_kspace_alpha: float = Field(
        default=2.0,
        ge=0,
        description="Alpha parameter for radial frequency weighting W(r)=1+alpha*(r/r_max). Higher values upweight high-freq components more aggressively.",
    )

    # ==================== SOBOLEV K-SPACE LOSS ====================
    enable_sobolev_kspace: bool = Field(
        default=False,
        description="Enable Fourier-domain Sobolev loss",
    )
    lambda_sobolev_kspace: float = Field(
        default=1.0,
        ge=0,
        description="Sobolev k-space loss weight (1+kx^2+ky^2 weighting)",
    )

    # ==================== SENSE ADJOINT LOSS ====================
    enable_sense_adjoint_l1: bool = Field(
        default=False,
        description="Enable SENSE image-domain complex L1 loss",
    )
    lambda_sense_adjoint_l1: float = Field(
        default=1.0,
        ge=0,
        description="SENSE adjoint loss weight",
    )

    # ==================== BACKGROUND SUPPRESSION LOSS (Phase 5) ====================
    # NEW: Suppresses artifacts in background/noise regions by forcing predictions to zero
    # where the ground truth is near the noise floor. Addresses "gray fog" in MRI reconstruction.
    enable_background_suppression: bool = Field(
        default=False,
        description="Enable background (noise) suppression loss - forces predictions to zero in background regions to eliminate gray fog artifacts",
    )
    lambda_background_suppression: float = Field(
        default=0.0,
        ge=0,
        description="Background suppression loss weight",
    )
    background_suppression_threshold_ratio: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Threshold ratio for background detection: pixels < threshold_ratio * noise_floor are considered background",
    )
    background_suppression_use_fourier_bridge: bool = Field(
        default=True,
        description=(
            "Whether to use DifferentiableFourierBridge when computing BackgroundSuppressionLoss. "
            "True (default): inputs are k-space tensors — bridge applies iFFT (norm='ortho') before "
            "spatial masking, preserving Parseval's theorem for correct gradient scaling. "
            "False: inputs are already in image domain — bridge is skipped."
        ),
    )

    # ==================== UNIVERSAL SPATIAL LOSS BRIDGING (Phase 4) ====================
    spatial_losses_use_fourier_bridge: bool = Field(
        default=False,
        description=(
            "Whether to automatically wrap all standard image-domain losses (L1, Perceptual, SSIM, DISTS, Edge) "
            "inside the DifferentiableFourierBridge. Set to True if training in k-space domain to allow these "
            "losses to receive complex-valued k-space directly while internally validating on image magnitudes."
        ),
    )

    # ==================== RICIAN CONSISTENCY LOSS (Phase 5 - Noise Debiasing) ====================
    enable_rician_consistency: bool = Field(
        default=False,
        description=(
            "Enable Rician consistency loss. Computes the Rician-unbiased target "
            "T = sqrt(max(y² - 2σ², 0)) and penalizes |pred - T|. "
            "This removes the noise floor bias present in MRI background regions (grey fog artifact) "
            "without requiring an explicit denoising pre-processing step."
        ),
    )
    lambda_rician_consistency: float = Field(
        default=0.3,
        ge=0.0,
        description="Weight for Rician consistency loss term.",
    )
    rician_noise_sigma: float | None = Field(
        default=None,
        description=(
            "Fixed noise standard deviation σ for Rician bias correction. "
            "If None (default), σ is estimated per-batch from background voxels via MAD."
        ),
    )
    rician_use_fourier_bridge: bool = Field(
        default=True,
        description=(
            "If True (default), wraps RicianConsistencyLoss with DifferentiableFourierBridge "
            "to process k-space inputs (iFFT → magnitude before loss). "
            "Set False if inputs are already image-domain magnitudes."
        ),
    )

    # ==================== DISTRIBUTION / HISTOGRAM LOSSES ====================
    enable_hist: bool = Field(default=False, description="Enable histogram consistency loss")
    lambda_hist: float = Field(default=0.0, ge=0, description="Histogram consistency weight")
    histogram_bins: int = Field(
        default=100,
        ge=10,
        le=512,
        description="Number of bins for histogram consistency loss",
    )

    # ==================== EDGE / GRADIENT LOSSES ====================
    enable_edge: bool = Field(default=False, description="Enable edge preservation loss")
    lambda_edge: float = Field(default=0.0, ge=0, description="Edge preservation weight")

    enable_sobel: bool = Field(default=False, description="Enable Sobel edge detection loss")
    lambda_sobel: float = Field(default=0.0, ge=0, description="Sobel edge detection weight")

    # ==================== COMPLEX-VALUED LOSSES ====================
    enable_complex_mse: bool = Field(default=False, description="Enable complex MSE loss")
    lambda_complex_mse: float = Field(default=0.0, ge=0, description="Complex MSE weight")

    enable_log_spectral_phase: bool = Field(
        default=False,
        description="Enable log-spectral phase loss for phase-sensitive reconstruction",
    )
    lambda_log_spectral_phase: float = Field(
        default=0.0, ge=0, description="Log-spectral phase loss weight"
    )

    enable_local_cross_correlation: bool = Field(
        default=False,
        description="Enable local normalized cross-correlation loss",
    )
    lambda_local_cross_correlation: float = Field(
        default=0.0, ge=0, description="Local cross-correlation weight"
    )

    # ==================== REGULARIZATION & TOPOLOGY ====================
    enable_persistent_homology: bool = Field(default=False, description="Enable topological loss")
    lambda_persistent_homology: float = Field(
        default=0.0, ge=0, description="Topological loss weight"
    )

    enable_tv: bool = Field(default=False, description="Enable total variation loss")
    lambda_tv: float = Field(default=0.0, ge=0, description="Total variation weight")
    # `enable_snr_preserving` / `lambda_snr_preserving` were RETIRED here on
    # 2026-08-23: `PhysicsLossesConfig` declares the same pair and is the only
    # side anything reads. Both spellings canonicalise to `snr_preserving`, so a
    # materialised config presented them as two conflicting declarations and
    # `build_loss_weight_table` raised (#421). See RENAMES.

    # ==================== PHYSICS LOSSES ====================
    # `enable_bloch_residual` / `lambda_bloch_residual` and
    # `enable_physics_constraint` / `lambda_physics_constraint` were RETIRED here
    # on 2026-08-23 for the same reason as the SNR pair above: duplicated in
    # `PhysicsLossesConfig`, which is the side the readers use
    # (`physics_driven_strategy.py`, `loss_builder.py`). See RENAMES.
    enable_physics_informed: bool = Field(default=False, description="Enable physics-informed loss")
    lambda_physics_informed: float = Field(
        default=0.0, ge=0, description="Physics-informed (PINN) weight"
    )
    enable_parallel_imaging_kspace: bool = Field(
        default=False, description="Enable parallel imaging k-space loss"
    )
    lambda_parallel_imaging_kspace: float = Field(
        default=0.0, ge=0, description="Parallel imaging k-space weight"
    )
    enable_graph_consistency: bool = Field(
        default=False, description="Enable graph consistency loss"
    )
    lambda_graph_consistency: float = Field(
        default=0.0, ge=0, description="Graph consistency weight"
    )
    enable_biophysical_flow: bool = Field(default=False, description="Enable biophysical flow loss")
    lambda_biophysical_flow: float = Field(default=0.0, ge=0, description="Biophysical flow weight")
    enable_spectral_graph: bool = Field(default=False, description="Enable spectral graph loss")
    lambda_spectral_graph: float = Field(default=0.0, ge=0, description="Spectral graph weight")
    spectral_graph_k: int = Field(
        default=8, ge=1, description="Number of neighbors for spectral graph loss"
    )
    spectral_graph_patch_size: int | None = Field(
        default=None,
        description="Patch size for local graph construction in spectral loss",
    )

    # Energy conservation loss (Parseval's theorem enforcement)
    enable_energy_conservation: bool = Field(
        default=False, description="Enable Parseval's theorem energy conservation loss"
    )
    lambda_energy_conservation: float = Field(
        default=0.0,
        ge=0,
        description="Energy conservation loss weight (Parseval's theorem)",
    )

    # ==================== DISENTANGLED VAE LOSSES ====================
    enable_recon: bool = Field(default=False, description="Enable general reconstruction loss")
    lambda_recon: float = Field(default=0.0, ge=0, description="General reconstruction weight")
    enable_style: bool = Field(default=False, description="Enable style consistency loss")
    lambda_style: float = Field(default=0.0, ge=0, description="Style consistency weight")
    # RENAMED from `enable_content` / `lambda_content` on 2026-08-23. The old
    # spelling was overloaded: this schema described it as the CONTENT
    # CONSISTENCY weight (a disentanglement term, `KNOWN_LOSS_COMPONENTS`), while
    # `LossRegistry._aliases` maps `content -> perceptual`, so
    # `build_loss_weight_table` saw it and `lambda_perceptual` as two
    # declarations of ONE loss and raised (#421). A disentangled author could
    # therefore not set content != perceptual at all: setting both crashed, and
    # setting only content leaked the weight into `perceptual`. The two meanings
    # now have two spellings -- `content_consistency` is not a registry alias, so
    # it canonicalises to itself.
    enable_content_consistency: bool = Field(
        default=False, description="Enable content consistency loss"
    )
    lambda_content_consistency: float = Field(
        default=0.0, ge=0, description="Content consistency weight"
    )
    # Physics-Informed Disentanglement
    enable_bloch: bool = Field(default=False, description="Enable Bloch regression loss")
    lambda_bloch: float = Field(
        default=0.0,
        ge=0,
        description="Bloch regression loss weight (Physics consistency)",
    )
    enable_anat: bool = Field(default=False, description="Enable anatomy locking loss")
    lambda_anat: float = Field(
        default=0.0,
        ge=0,
        description="Anatomy locking loss weight (Paired consistency)",
    )

    # MIND-SSC Anatomical Structure Lock
    enable_mind_ssc: bool = Field(
        default=False, description="Enable MIND-SSC anatomical structure loss"
    )
    lambda_mind_ssc: float = Field(
        default=0.5,
        ge=0,
        description="MIND-SSC loss weight (intensity-invariant anatomy lock)",
    )

    # Latent Consistency (Disentanglement Enforcement)
    enable_latent_consistency: bool = Field(
        default=False, description="Enable latent consistency loss via cross-swapping"
    )
    lambda_latent_consistency: float = Field(
        default=0.1,
        ge=0,
        description="Latent consistency loss weight (content code preservation)",
    )

    # Tissue Parameter Bounds Regularization
    enable_tissue_bounds: bool = Field(
        default=False, description="Enable tissue parameter bounds regularization"
    )
    lambda_tissue_bounds: float = Field(
        default=0.01,
        ge=0,
        description="Tissue parameter bounds regularization weight",
    )

    # ==================== MAGNITUDE-AWARE SCALING ====================
    # Reference: Chen et al. "GradNorm" (2018) - Gradient balancing for multi-task learning
    use_magnitude_scaling: bool = Field(
        default=False,
        description="Enable magnitude-aware loss scaling for gradient balancing",
    )
    magnitude_scale_recon: float = Field(
        default=10.0,
        ge=0,
        description="Magnitude scale for reconstruction loss (primary signal)",
    )
    magnitude_scale_structural: float = Field(
        default=100.0,
        ge=0,
        description="Magnitude scale for structural priors (MIND-SSC, etc.) - boosts small-magnitude losses",
    )
    magnitude_scale_physics: float = Field(
        default=5.0,
        ge=0,
        description="Magnitude scale for physics-informed losses (Bloch, etc.)",
    )

    # ==================== RECONSTRUCTION OPTIONS ====================
    loss_type: str = Field(default="l1", description="Default loss type")
    density_weighting: bool = Field(default=False, description="Enable density weighting")
    enable_persistent_homology_option: bool = Field(
        default=False, description="Enable topological loss option"
    )

    # ==================== VIRTUAL FIDUCIAL (VF) LOSSES ====================
    # Marker-anchored losses used by ConcreteVFADMMStrategy and
    # ConcreteVirtualFiducialStrategy.  Both lambdas must be set here (SSOT)
    # so that individual experiment YAMLs can tune them without touching code.
    enable_marker_loss: bool = Field(
        default=False,
        description="Enable marker-corruption loss (penalises deviation from ideal marker)",
    )
    lambda_marker: float = Field(
        default=2.0,
        ge=0,
        description="Weight for the marker-corruption loss (VF strategies)",
    )
    enable_prior_loss: bool = Field(
        default=False,
        description="Enable marker-prior projection loss (ADMM prior anchoring)",
    )
    lambda_prior: float = Field(
        default=10.0,
        ge=0,
        description="Weight for the marker-prior projection loss (VF-ADMM strategy)",
    )
    enable_distill: bool = Field(
        default=False, description="Enable teacher-student distillation loss"
    )
    lambda_distill: float = Field(
        default=1.0,
        ge=0,
        description=(
            "Weight for the teacher→student knowledge distillation loss "
            "(DistillationStrategy). Controls how strongly the student "
            "mimics the teacher's output distribution."
        ),
    )

    # ==================== PADNET PHYSICS-INFORMED LOSSES ====================
    enable_padnet_l2: bool = Field(default=False, description="Enable PaDNet L2 loss")
    lambda_padnet_l2: float = Field(
        default=1.0,
        ge=0,
        description="Weight for L2 image-domain loss in PaDNet physics-informed training",
    )
    enable_padnet_dc: bool = Field(default=False, description="Enable PaDNet data consistency loss")
    lambda_padnet_dc: float = Field(
        default=0.0,
        ge=0,
        description="Weight for data consistency loss in PaDNet (0 = DC handled separately)",
    )
    enable_padnet_reg: bool = Field(default=False, description="Enable PaDNet regularization loss")
    lambda_padnet_reg: float = Field(
        default=0.1,
        ge=0,
        description="Weight for regularization in PaDNet physics-informed loss",
    )

    # Retired spellings of THIS block (#421). The mount is on this class rather
    # than on `LossConfigSchema` because `renames_for_block` selects by
    # `mount_path`, and these records' legacy keys live one level down, at
    # `losses.reconstruction.*` -- a `reject_renamed_keys("losses")` validator
    # matches leaf names against `LossConfigSchema`'s OWN children and would
    # never see them.
    #
    # This class is `extra="ignore"`, so without this validator the eight
    # retired keys would not raise: they would be swallowed and discarded, and
    # an author who kept writing `lambda_bloch_residual` under `reconstruction`
    # would train at the schema default with no diagnostic. That is the exact
    # "stops working AND stops being visible" failure the renames module
    # docstring names, and `rename_mounts.audit_mounts` reports an unmounted
    # record set as `missing_mount` -- it caught this one before it shipped.
    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("losses.reconstruction"))
    )


class GANLossesConfig(BaseModel):
    """GAN-specific loss configuration (both enable flags and lambda weights).

    Consolidates GAN objective settings from objectives.gan into a single SSOT.
    """

    model_config = {"protected_namespaces": (), "extra": "ignore", "frozen": True}

    enable_adversarial: bool = Field(default=False, description="Enable adversarial loss")
    lambda_adv: float = Field(default=1.0, ge=0, description="Adversarial loss weight")
    lambda_discriminator: float = Field(
        default=1.0,
        ge=0,
        description=(
            "Weight of the discriminator's own loss component (the term "
            "UnifiedGANLossComputer.compute_discriminator_loss stacks under the name "
            "'discriminator'). It had no schema field, so it fell through to the "
            "hardcoded default tables. Declared here so the knob has one visible home "
            "(pitfall #15); 1.0 preserves the legacy value."
        ),
    )

    enable_gradient_penalty: bool = Field(default=False, description="Enable gradient penalty")
    lambda_gp: float = Field(default=10.0, ge=0, description="Gradient penalty weight")

    enable_feature_matching: bool = Field(default=False, description="Enable feature matching")
    feature_matching: float = Field(default=0.0, ge=0, description="Feature matching weight")

    enable_r1: bool = Field(default=False, description="Enable R1 regularization")
    lambda_r1: float = Field(default=10.0, ge=0, description="R1 regularization weight")
    r1_interval: int = Field(
        default=16,
        ge=0,
        description="Interval for R1 regularization. If 0, uses r1_probability.",
    )
    r1_probability: float = Field(
        default=1.0, ge=0, le=1, description="R1 regularization probability"
    )

    # GAN-specific hyperparameters
    gan_loss_type: str = Field(default="wgan-gp", description="GAN loss type")
    disc_updates: int = Field(
        default=1, gt=0, description="Discriminator updates per generator update"
    )
    label_smoothing: float = Field(
        default=0.0, ge=0, le=1, description="Label smoothing for GAN loss"
    )

    # ==================== CYCLE-BLOCH STRATEGY ====================
    lambda_cycle_bloch: float = Field(
        default=10.0,
        ge=0,
        description="Weight for Bloch cycle-consistency loss (CycleBlochStrategy)",
    )
    lambda_cycle_adv: float = Field(
        default=1.0,
        ge=0,
        description="Weight for adversarial loss in cycle-Bloch training",
    )

    # ==================== DOMAIN ADAPTATION ====================
    lambda_domain: float = Field(
        default=1.0,
        ge=0,
        description="Weight for domain adversarial alignment loss (DomainAdaptationStrategy)",
    )

    # ==================== TRANSLATION / STARGAN ====================
    lambda_cycle: float = Field(
        default=10.0,
        ge=0.0,
        description="CycleGAN cycle-consistency weight.",
    )
    lambda_identity: float = Field(
        default=0.5,
        ge=0.0,
        description="CycleGAN identity weight.",
    )
    lambda_nce: float = Field(
        default=1.0,
        ge=0.0,
        description="CUT PatchNCE weight.",
    )
    lambda_style: float = Field(
        default=1.0,
        ge=0.0,
        description="StarGAN v2 style-reconstruction weight.",
    )
    lambda_diversity: float = Field(
        default=1.0,
        ge=0.0,
        description="StarGAN v2 style-diversification weight.",
    )


class DiffusionLossesConfig(BaseModel):
    """Diffusion-specific loss configuration (enable flags and hyperparameters).

    NOTE: Diffusion training hyperparameters are now at training.diffusion (SSOT):
    - num_timesteps (was objectives.diffusion.timesteps)
    - noise_schedule (was objectives.diffusion.noise_schedule)
    - guidance_scale (was objectives.diffusion.guidance_scale)
    - prediction_type, beta_start, beta_end

    This config handles only loss weights and flags for diffusion training.
    """

    model_config = {"protected_namespaces": (), "extra": "ignore", "frozen": True}

    enable_diffusion: bool = Field(default=False, description="Enable diffusion loss")
    lambda_mse: float = Field(default=1.0, ge=0, description="MSE loss weight for diffusion")

    enable_reconstruction_auxiliary: bool = Field(
        default=False, description="Enable aux reconstruction"
    )
    enable_data_consistency: bool = Field(default=False, description="Enable data consistency")

    prediction_type: str = Field(default="epsilon", description="epsilon, sample, v_prediction")

    # DEPRECATED: These fields should now be read from training.diffusion (SSOT)
    # Kept for backward compatibility only
    timesteps: int = Field(
        default=1000,
        gt=0,
        description="Number of diffusion timesteps (DEPRECATED: use training.diffusion.timesteps)",
    )
    noise_schedule: str = Field(
        default="cosine",
        description="Noise schedule type (DEPRECATED: use training.diffusion.noise_schedule)",
    )
    cond_drop_prob: float = Field(
        default=0.1, ge=0, le=1, description="Conditional drop probability"
    )
    guidance_scale: float = Field(default=7.5, ge=0, description="Classifier-free guidance scale")
    enable_diffusion_amp: bool = Field(default=False, description="Enable AMP for diffusion")
    enforce_output_range: bool = Field(default=True, description="Enforce output in valid range")
    sampler: str = Field(
        default="ddpm", description="Sampling method: ddpm, ddim, predictor_corrector"
    )


class LatentLossesConfig(BaseModel):
    """Latent/VAE loss configuration (enable flags and lambda weights).

    NOTE: Latent dimensionality parameter is now at training.latent_dim (SSOT).
    This config handles only loss weights for latent-based training strategies.
    See training.latent_dim for VAE/VQVAE dimensionality configuration.
    """

    model_config = {"protected_namespaces": (), "extra": "ignore", "frozen": True}

    enable_reconstruction: bool = Field(default=False, description="Enable reconstruction loss")
    lambda_recon: float = Field(default=0.0, ge=0, description="Reconstruction loss weight")
    lambda_l1: float = Field(default=0.0, ge=0, description="L1 reconstruction weight")
    lambda_ssim: float = Field(default=0.0, ge=0, description="SSIM loss weight")

    enable_kl: bool = Field(default=False, description="Enable KL divergence loss")
    lambda_kl: float = Field(default=0.0, ge=0, description="KL divergence weight")
    beta_kl: float = Field(default=1.0, ge=0, description="KL annealing beta")

    enable_commitment: bool = Field(default=False, description="Enable VQ commitment loss")
    lambda_commit: float = Field(default=0.0, ge=0, description="Commitment loss weight")
    commitment_weight: float = Field(default=0.25, ge=0, description="VQ commitment weight")

    enable_codebook: bool = Field(default=False, description="Enable codebook loss")
    lambda_codebook: float = Field(
        default=1.0,
        ge=0,
        description=(
            "Codebook loss weight. `enable_codebook` existed but the matching weight "
            "field did not, so UnifiedVAELossComputer's probe for 'codebook' fell "
            "through to a hardcoded default table. Declared here so the knob has one "
            "visible home (pitfall #15); 1.0 preserves the legacy value."
        ),
    )
    lambda_vq: float = Field(default=0.0, ge=0, description="VQ loss weight")
    codebook_weight: float = Field(default=1.0, ge=0, description="Codebook weight")

    # NOTE: latent_dim is now EXCLUSIVELY in training.latent_dim (SSOT)
    # Use config.training.latent_dim, not config.losses.latent.latent_dim
    n_embeddings: int = Field(default=512, gt=0, description="Number of codebook embeddings")
    embedding_dim: int = Field(default=64, gt=0, description="Embedding dimension")
    anneal_kl_beta: bool = Field(default=False, description="Anneal KL beta")
    kl_beta_start: float = Field(default=0.0, ge=0, description="Initial KL beta")
    kl_beta_end: float = Field(default=1.0, ge=0, description="Final KL beta")
    kl_anneal_steps: int = Field(default=10000, gt=0, description="KL annealing steps")
    latent_regularization_weight: float = Field(default=0.01, ge=0, description="Latent reg weight")
    use_latent_regularization: bool = Field(default=False, description="Use latent regularization")
    latent_loss_type: str = Field(
        default="kl_divergence",
        description=(
            "Latent regulariser selected via the loss registry (dispatched in "
            "UnifiedVAELossComputer). One of: kl_divergence (default), "
            "latent_regularization, mmd, vq_kl, beta_tc_vae."
        ),
    )

    @field_validator("latent_loss_type")
    @classmethod
    def _validate_latent_loss_type(cls, v: str) -> str:
        # NN#3: reject an unsupported latent loss at load time rather than at the
        # first training step (it is dispatched through the loss registry).
        supported = {
            "kl_divergence",
            "kl",
            "latent_regularization",
            "mmd",
            "maximum_mean_discrepancy",
            "vq_kl",
            "beta_tc_vae",
        }
        if v.lower() not in supported:
            raise ValueError(
                f"latent_loss_type={v!r} is not supported. Choose one of: {sorted(supported)}."
            )
        return v

    temperature: float = Field(default=0.1, gt=0, description="Temperature for softmax")

    # ==================== CONTROLLED CAPACITY (C-TARGET) ====================
    # Reference: Burgess et al. "Understanding disentangling in β-VAE" (2018)
    use_capacity_scheduling: bool = Field(
        default=False,
        description="Enable controlled capacity increase (C-target approach) to prevent posterior collapse",
    )
    kl_capacity_target: float = Field(
        default=20.0,
        ge=0,
        description="Target KL capacity in nats (e.g., 20 nats ≈ 29 bits of information)",
    )
    kl_capacity_warmup_steps: int = Field(
        default=10000,
        gt=0,
        description="Number of steps to linearly increase capacity from 0 to target",
    )


class PhysicsLossesConfig(BaseModel):
    """Physics-specific loss configuration for MRI reconstruction."""

    model_config = {"protected_namespaces": (), "extra": "ignore", "frozen": True}

    # Bloch equation residual
    enable_bloch_residual: bool = Field(
        default=False,
        description="Enable Bloch equation residual loss (PINN for MRI physics)",
    )
    lambda_bloch_residual: float = Field(
        default=1.0,
        ge=0,
        description="Weight for Bloch residual loss",
    )

    # Physics constraint (smoothness + energy + k-space)
    enable_physics_constraint: bool = Field(
        default=False,
        description="Enable physics constraint loss (TV + energy conservation)",
    )
    lambda_physics_constraint: float = Field(
        default=0.1,
        ge=0,
        description="Weight for physics constraint loss",
    )

    # Parallel imaging k-space loss
    enable_parallel_imaging: bool = Field(
        default=False,
        description="Enable parallel imaging k-space consistency loss (multi-coil)",
    )
    lambda_parallel_imaging: float = Field(
        default=1.0,
        ge=0,
        description="Weight for parallel imaging loss",
    )

    # SNR preserving
    enable_snr_preserving: bool = Field(
        default=False,
        description="Enable SNR preservation loss",
    )
    lambda_snr_preserving: float = Field(
        default=1.0,
        ge=0,
        description="Weight for SNR preserving loss",
    )
    target_snr: float = Field(
        default=20.0,
        ge=0,
        description="Target SNR value for SNR preserving loss",
    )

    # Complex Spatial Gradient Loss (Phase 5 - Phase & Sharpness coherence)
    enable_complex_spatial_gradient: bool = Field(
        default=False,
        description="Enable complex spatial gradient loss. Penalizes first-order derivatives of the complex image to enforce both magnitude sharpness and local phase coherence.",
    )
    lambda_complex_spatial_gradient: float = Field(
        default=1.0,
        ge=0,
        description="Weight for complex spatial gradient loss",
    )
    complex_spatial_gradient_use_fourier_bridge: bool = Field(
        default=True,
        description="Whether to use DifferentiableFourierBridge to map k-space inputs to complex image-space before computing the gradient penalty.",
    )

    # Helmholtz PDE residual loss (physics-informed)
    enable_helmholtz_pde: bool = Field(
        default=False,
        description="Enable Helmholtz PDE residual loss for physics-informed reconstruction",
    )
    lambda_helmholtz_pde: float = Field(
        default=0.0, ge=0, description="Helmholtz PDE residual loss weight"
    )

    # Jacobian determinant loss (deformation regularization)
    enable_jacobian_determinant: bool = Field(
        default=False,
        description="Enable Jacobian determinant loss for deformation field regularity",
    )
    lambda_jacobian_determinant: float = Field(
        default=0.0, ge=0, description="Jacobian determinant loss weight"
    )

    # --- Phase-contrast / 4D-flow (regime: mri_flow) -----------------------
    # Every weight defaults to 0.0 = "not requested". resolve_loss_weight()
    # RAISES on a name with neither a declaration nor a schema default (it never
    # invents 1.0), so these fields are what make the FLOW losses selectable at
    # all. PhaseContrastFlowStrategy raises if ALL of them are 0 — an arm that
    # advertises phase-contrast physics and weights none of it would train as a
    # plain regressor (pitfall #16).
    lambda_phase_contrast_velocity: float = Field(
        default=0.0,
        ge=0,
        description="Weight for the phase-contrast velocity loss (v_pred vs "
        "v_target through the phi = pi*v/venc encoding).",
    )
    lambda_through_plane_flux_conservation: float = Field(
        default=0.0,
        ge=0,
        description="Weight for the inlet/outlet through-plane flux-conservation "
        "loss. Requires data.phase_contrast.flux_conservation and the "
        "inlet/outlet mask batch keys.",
    )
    lambda_velocity_unwrap_consistency: float = Field(
        default=0.0,
        ge=0,
        description="Weight for the velocity unwrap-consistency hinge, which "
        "penalises |v| > venc (where the phase aliases).",
    )

    # --- Tracer kinetics / DCE perfusion (regime: mri_perfusion) -----------
    lambda_tofts_residual: float = Field(
        default=0.0,
        ge=0,
        description="Weight for the extended-Tofts residual — the primary, "
        "SELF-SUPERVISED perfusion objective (refits the measured "
        "concentration-time curve; needs no ground-truth parameter maps).",
    )
    lambda_aif_consistency: float = Field(
        default=0.0,
        ge=0,
        description="Weight for grading a LEARNED AIF against the Parker "
        "population reference. Inert today: no registered model exposes an AIF "
        "head, so data.perfusion.aif_source has no 'learned' option and the "
        "strategy raises if this is weighted > 0.",
    )
    lambda_perfusion_physiological_box: float = Field(
        default=0.0,
        ge=0,
        description="Weight for the soft physiological box constraint "
        "(Ktrans, ve, vp >= 0 and ve + vp <= 1).",
    )
    lambda_perfusion_map_smoothness: float = Field(
        default=0.0,
        ge=0,
        description="Weight for spatial smoothness of the kinetic parameter maps.",
    )

    # --- Relaxometry (regime: mri_quantitative) ---------------------------
    lambda_bloch_signal_synthesis_consistency: float = Field(
        default=0.0,
        ge=0,
        description="Weight for re-synthesising the observed multi-contrast "
        "magnitudes from the predicted (T1, T2, PD) through the SPGR signal "
        "equation. Needs per-contrast TR_ms/TE_ms/FA_deg; the loss raises "
        "without them rather than assuming defaults.",
    )
    lambda_contrast_consistency: float = Field(
        default=0.0,
        ge=0,
        description="Weight for grading the predicted T1 map (ms) against the "
        "Bottomley field-strength-dependent relaxometry prior, weighted by "
        "tissue-class probabilities.",
    )

    # --- Fingerprinting (regime: mri_fingerprinting) ----------------------
    lambda_mrf_dictionary_match: float = Field(
        default=0.0,
        ge=0,
        description="Weight for the soft Bloch-dictionary matching consistency — "
        "MRF's defining operation, and the regime's primary SELF-SUPERVISED "
        "objective (the dictionary supplies the supervision, so no ground-truth "
        "parameter maps are needed). Requires a dictionary + its paired "
        "parameters; the loss raises without them.",
    )

    # --- Diffusion (regime: mri_diffusion_weighted) -----------------------
    lambda_dwi_adc_monoexp: float = Field(
        default=0.0,
        ge=0,
        description="Weight for the mono-exponential diffusion consistency "
        "ln S = ln S0 - b*ADC. The real consumer of the 'b_values' batch key.",
    )

    # --- Dynamic (regime: mri_dynamic) ------------------------------------
    lambda_storm: float = Field(
        default=0.0,
        ge=0,
        description="Weight for the SToRM manifold-smoothness regulariser "
        "Tr(X L X^T) over the temporal frame axis. With learn_laplacian=True it "
        "requires navigators (or a precomputed L) and raises without them.",
    )

    # --- Spectroscopy (regime: mri_spectroscopy) --------------------------
    lambda_mrs_fid_residual: float = Field(
        default=0.0,
        ge=0,
        description="Weight for the AMARES-style FID fit residual — the primary, "
        "SELF-SUPERVISED spectroscopy objective (refits the measured free "
        "induction decay; needs no ground-truth concentrations, which real MRS "
        "acquisitions do not have). Fitted in the TIME domain.",
    )
    lambda_mrs_prior_knowledge: float = Field(
        default=0.0,
        ge=0,
        description="Weight for the soft prior-knowledge hinges (amplitudes >= 0, "
        "linewidth within a plausible shim range). Under the default "
        "parameter_activation='softplus' the non-negativity term is identically "
        "zero with zero gradient — softplus already enforces it — so only the "
        "linewidth bounds are live. Do not read a healthy value as evidence "
        "non-negativity is being learned.",
    )

    # --- EPI readout (regimes: mri_functional, mri_diffusion_weighted) ----
    lambda_beltrami_epi_residual: float = Field(
        default=0.0,
        ge=0,
        description="Weight for the EPI susceptibility-distortion data "
        "consistency. Requires an estimated delta_b0 field; the loss raises "
        "without one rather than degrading to a plain L1 under this name.",
    )


class SSLLossesConfig(BaseModel):
    """Self-supervised / contrastive loss configuration."""

    model_config = {"protected_namespaces": (), "extra": "ignore", "frozen": True}

    # Contrastive loss (InfoNCE)
    enable_contrastive: bool = Field(
        default=False,
        description="Enable InfoNCE contrastive loss",
    )
    lambda_contrastive: float = Field(
        default=1.0,
        ge=0,
        description="Weight for contrastive loss",
    )
    temperature: float = Field(
        default=0.07,
        gt=0,
        description="Temperature for contrastive softmax",
    )

    # Anatomy-Contrast disentanglement
    enable_ca_contrastive: bool = Field(
        default=False,
        description="Enable contrastive anatomy-contrast disentanglement loss",
    )
    lambda_ca_contrastive: float = Field(
        default=1.0,
        ge=0,
        description="Weight for CA contrastive loss",
    )


class EvidentialLossesConfig(BaseModel):
    """Evidential Deep Learning loss configuration."""

    model_config = {"protected_namespaces": (), "extra": "ignore", "frozen": True}

    enable_evidential: bool = Field(
        default=False,
        description="Enable evidential deep learning loss (NLL + Evidence Regularizer)",
    )

    lambda_evidential: float = Field(
        default=1.0,
        ge=0,
        description="Regularization coefficient (lambda) for the evidence penalty",
    )


class SpatialLossesConfig(BaseModel):
    """Spatial mapping and deformation loss configuration."""

    model_config = {"protected_namespaces": (), "extra": "ignore", "frozen": True}

    enable_hyperelastic_jacobian: bool = Field(
        default=False, description="Enable hyperelastic Jacobian regularization"
    )
    lambda_hyperelastic_jacobian: float = Field(
        default=0.0, ge=0, description="Jacobian regularization weight"
    )


class RegistrationLossesConfig(BaseModel):
    """Registration-specific loss configuration (LNCC, Smoothness)."""

    model_config = {"protected_namespaces": (), "extra": "ignore", "frozen": True}

    enable_lncc: bool = Field(default=False, description="Enable LNCC loss")
    lambda_sim: float = Field(default=1.0, ge=0, description="LNCC similarity weight")

    enable_smoothness: bool = Field(default=False, description="Enable smoothness loss")
    lambda_smooth: float = Field(default=1.0, ge=0, description="Smoothness regularization weight")


class PINNLossesConfig(BaseModel):
    """PINN-specific loss configuration (PDE, Norm, TV, DC)."""

    model_config = {"protected_namespaces": (), "extra": "ignore", "frozen": True}

    enable_pde: bool = Field(default=False, description="Enable PINN PDE loss (Helmholtz)")
    lambda_pde: float = Field(default=0.1, ge=0, description="PDE constraint weight")

    enable_unit_norm_coil: bool = Field(
        default=False, description="Enable coil sensitivity unit norm constraint"
    )
    lambda_unit_norm_coil: float = Field(default=10.0, ge=0, description="Coil unit norm weight")

    enable_magnitude_tv: bool = Field(default=False, description="Enable PINN sensitivity TV")
    lambda_magnitude_tv: float = Field(default=0.001, ge=0, description="Magnitude TV weight")

    enable_pinn_dc: bool = Field(default=False, description="Enable PINN data consistency loss")
    lambda_pinn_dc: float = Field(default=1.0, ge=0, description="Data consistency weight")


class LossComponentConfig(BaseModel):
    """Configuration for a single loss component in a composite loss.

    Used when composing multiple losses together with flexible weighting.

    Example:
        ```python
        component = LossComponentConfig(
            name="l1",
            weight=10.0,
            enabled=True,
            kwargs={"reduction": "mean"}
        )
        ```
    """

    name: str = Field(
        ...,
        description="Name of the loss component (e.g., 'l1', 'perceptual', 'ssim')",
    )
    weight: float = Field(
        default=1.0,
        description="Weight multiplier for this loss component",
        ge=0.0,
    )
    enabled: bool = Field(
        default=True,
        description="Whether this component is enabled",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments passed to loss constructor",
    )

    model_config = {"extra": "ignore", "frozen": True}


class ComposedLossConfig(BaseModel):
    """Configuration for composing multiple loss components.

    Allows flexible combination of different loss types with independent weighting.

    Example:
        ```yaml
        composed:
          components:
            - name: l1
              weight: 10.0
              enabled: true
            - name: perceptual
              weight: 0.1
              enabled: true
            - name: ssim
              weight: 0.1
              enabled: false
          normalize_total: false
        ```
    """

    components: list[LossComponentConfig] = Field(
        default_factory=list,
        description="List of loss components to combine",
    )
    normalize_total: bool = Field(
        default=False,
        description="Whether to normalize by number of enabled components",
    )

    @field_validator("components")
    @classmethod
    def validate_unique_names(cls, components):
        """Ensure all component names are unique."""
        names = [c.name for c in components]
        if len(names) != len(set(names)):
            raise ValueError("Loss component names must be unique")
        return components

    model_config = {"extra": "ignore", "frozen": True}


class MetricsConfig(BaseModel):
    """Metrics computation configuration (not losses, but evaluation metrics)."""

    model_config = {"protected_namespaces": (), "extra": "ignore", "frozen": True}

    # Image quality metrics
    compute_psnr: bool = Field(
        default=True,
        description="Compute PSNR (Peak Signal-to-Noise Ratio)",
    )
    compute_ssim: bool = Field(
        default=True,
        description="Compute SSIM (Structural Similarity Index)",
    )
    compute_mse: bool = Field(
        default=True,
        description="Compute MSE (Mean Squared Error)",
    )
    compute_mae: bool = Field(
        default=True,
        description="Compute MAE (Mean Absolute Error)",
    )

    # Perceptual metrics
    compute_lpips: bool = Field(
        default=False,
        description="Compute LPIPS (Learned Perceptual Image Patch Similarity)",
    )
    compute_fid: bool = Field(
        default=False,
        description="Compute FID (Fréchet Inception Distance)",
    )
    compute_inception_score: bool = Field(
        default=False,
        description="Compute Inception Score (IS)",
    )

    # MRI-specific metrics
    compute_nmse: bool = Field(
        default=False,
        description="Compute NMSE (Normalized Mean Squared Error)",
    )
    compute_nrmse: bool = Field(
        default=False,
        description="Compute NRMSE (Normalized Root Mean Squared Error)",
    )

    # Medical imaging metrics
    compute_dice: bool = Field(
        default=False,
        description="Compute Dice coefficient (for segmentation)",
    )
    compute_hausdorff: bool = Field(
        default=False,
        description="Compute Hausdorff distance (for segmentation)",
    )

    # Metric computation interval
    metric_interval: int = Field(
        default=1000,
        ge=1,
        description="Compute metrics every N steps",
    )


class LossPolicyConfigSchema(BaseModel):
    """How the objective is ASSEMBLED, as opposed to which terms are in it.

    ``losses:`` is otherwise sixteen loss-family blocks, so a loose scalar beside
    them reads as if it might be a seventeenth. These two are not families:
    ``output_domain`` decides where every family's tensor is bridged to, and
    ``exclude_defaults`` filters the paradigm's implicit terms.

    ``disable_default_losses`` becomes ``exclude_defaults``, and the ``negate``
    machinery is deliberately NOT used. The naming rule forbids a negated
    boolean, but this field is a ``list[str]`` of loss NAMES -- inverting it is
    meaningless (``enable_default_losses: ['mse']`` would mean "enable only
    mse", the opposite of a filter). ``exclude_`` states the filter's sense
    without a negation to invert.

    Three more scalars stayed flat on the parent because nothing reads them --
    see the parent's note.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_domain: str = Field(
        default="",
        description=(
            "Domain of the model output for automatic loss bridging: 'kspace', "
            "'image', or 'complex_image'. With list-based losses "
            "(by-domain lists below) the LossBuilder inserts a "
            "DifferentiableFourierBridge where needed. Leave empty to use the "
            "legacy per-loss fourier_bridge flags."
        ),
    )
    exclude_defaults: list[str] = Field(
        default_factory=list,
        description=(
            "Default losses the paradigm would otherwise add, to leave out. "
            "E.g. ['mse'] drops MSE for diffusion, ['adversarial'] drops the GAN "
            "term. Supported: mse, adversarial, kl, reconstruction, commitment, "
            "codebook."
        ),
    )


class LossConfigSchema(BaseModel):
    """Unified loss configuration schema.

    Allows users to declaratively specify:
    1. Which losses to compute per paradigm
    2. Lambda (weight) for each loss
    3. Which metrics to track
    4. Which default losses to disable

    Example:
        >>> loss_config = LossConfigSchema(
        ...     reconstruction=ReconstructionLossesConfig(
        ...         lambda_l1=10.0,
        ...         lambda_perceptual=10.0,
        ...         lambda_ssim=1.0,
        ...     ),
        ...     metrics=MetricsConfig(
        ...         compute_fid=True,
        ...         compute_lpips=True,
        ...     ),
        ...     disable_default_losses=['mse'],  # Disable MSE for diffusion
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    # ---- phase 10d: the policy sub-block ---------------------------------
    # The flat spellings still LOAD -- `fold_renamed_keys` moves them -- but they
    # are gone from Python. This block is `extra="ignore"`, so a forgotten record
    # would make the key VANISH rather than raise (#550);
    # `TestPhase10dFoldTableIsTotal` pins that.
    policy: LossPolicyConfigSchema = Field(default_factory=LossPolicyConfigSchema)

    __folded_input_keys__ = folded_input_keys("losses")
    __folded_input_paths__ = folded_input_paths("losses")

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
    _reject_renamed = model_validator(mode="before")(classmethod(reject_renamed_keys("losses")))
    _fold_renamed = model_validator(mode="before")(classmethod(fold_renamed_keys("losses")))

    # ---- ungrouped: inert, and flatness is the tell ----------------------
    # `normalize_losses` and `clip_loss_value` have ZERO readers and ZERO corpus
    # declarations; `loss_scaling` likewise -- its apparent references are
    # `MixedPrecisionConfig.loss_scaling`, a different class (the leaf-name
    # blind spot of issue #674). Grouping them would imply they work; issue #676.
    # `lambda_deep_supervision` is live and correctly placed already: it is a
    # loss WEIGHT, and `lambda_<term>` on `losses:` is the ratified spelling.

    # Paradigm-specific loss configurations (optional - can be None if paradigm not used)
    reconstruction: ReconstructionLossesConfig | None = Field(
        default_factory=ReconstructionLossesConfig,
        description="Reconstruction loss configuration (L1, Perceptual, SSIM, LPIPS, Frequency)",
    )

    gan: GANLossesConfig | None = Field(
        default=None,
        description="GAN-specific loss configuration",
    )

    diffusion: DiffusionLossesConfig | None = Field(
        default=None,
        description="Diffusion-specific loss configuration",
    )

    latent: LatentLossesConfig | None = Field(
        default=None,
        description="Latent/VAE loss configuration",
    )

    physics: PhysicsLossesConfig | None = Field(
        default=None,
        description="Physics-specific loss configuration (Bloch, parallel imaging, SNR)",
    )

    registration: RegistrationLossesConfig | None = Field(
        default=None,
        description="Registration-specific loss configuration (LNCC, Smoothness)",
    )

    pinn: PINNLossesConfig | None = Field(
        default=None,
        description="Physics-Informed Neural Network loss configuration",
    )

    ssl: SSLLossesConfig | None = Field(
        default=None,
        description="Self-supervised / contrastive loss configuration",
    )

    evidential: EvidentialLossesConfig | None = Field(
        default=None,
        description="Evidential deep learning loss configuration",
    )

    spatial: SpatialLossesConfig | None = Field(
        default=None,
        description="Spatial/Deformation loss configuration (Jacobian, etc)",
    )

    # Composite loss configuration
    composed: ComposedLossConfig | None = Field(
        default=None,
        description="Flexible loss composition allowing arbitrary component combinations",
    )

    # Metrics configuration
    metrics: MetricsConfig | None = Field(
        default=None,
        description="Metrics computation configuration",
    )

    # ==================== DOMAIN-AWARE LIST-BASED LOSSES ====================
    # When these lists are populated, the LossBuilder auto-wraps losses with
    # DifferentiableFourierBridge instead of requiring per-loss flags.
    # Old configs without these lists continue to work unchanged.
    #
    # output_domain: declares what the model produces (kspace | image | complex_image).
    # kspace_losses:  losses that expect k-space input (no bridge when output_domain=kspace).
    # image_losses:   losses that expect real-valued image magnitude (auto iFFT bridge).
    # complex_losses: losses that expect complex image-domain tensors (auto iFFT, keep complex).

    kspace_losses: list[LossComponentConfig] = Field(
        default_factory=list,
        description=(
            "Losses that operate in k-space domain. When output_domain='kspace', "
            "these receive the raw model output. When output_domain='image', "
            "an FFT bridge is inserted automatically."
        ),
    )

    image_losses: list[LossComponentConfig] = Field(
        default_factory=list,
        description=(
            "Losses that operate on real-valued image magnitudes (e.g., LPIPS, SSIM, L1). "
            "When output_domain='kspace', an iFFT bridge (magnitude) is inserted automatically. "
            "When output_domain='image', these receive the raw model output."
        ),
    )

    complex_losses: list[LossComponentConfig] = Field(
        default_factory=list,
        description=(
            "Losses that operate on complex-valued images (e.g., complex_spatial_gradient). "
            "When output_domain='kspace', an iFFT bridge (return_complex=True) is inserted. "
            "When output_domain='image', these receive the raw output cast to complex."
        ),
    )

    latent_losses: list[LossComponentConfig] = Field(
        default_factory=list,
        description=(
            "Losses that operate on a learned latent (post-encoder), e.g. "
            "physics_equivariance, domain_adversarial_grl, teichmuller_geodesic. "
            "Valid only with output_domain='latent': there is NO bridge from any "
            "other domain into a latent, because the encoder that would produce "
            "it is a learned map, not a transform. Declaring these alongside a "
            "kspace/image output raises rather than silently grading the wrong "
            "tensor."
        ),
    )

    @property
    def uses_list_based_losses(self) -> bool:
        """Check if the new list-based loss configuration is active.

        When True, the LossBuilder uses domain-aware auto-bridging
        instead of per-loss fourier_bridge flags.
        """
        return bool(
            self.kspace_losses or self.image_losses or self.complex_losses or self.latent_losses
        )

    @model_validator(mode="before")
    @classmethod
    def _reject_undeclared_loss_lists(cls, data: Any) -> Any:
        """A ``*_losses`` key the schema does not declare must RAISE.

        This class is ``extra="ignore"``, so an invented or misspelled list key
        was dropped in silence: the arm loaded clean, ``uses_list_based_losses``
        stayed False, the "output_domain required" validator never fired, and
        the run fell through to default losses. The arm advertises a method it
        never trains (pitfall #16).

        Three corpus arms did exactly this (issue #655) -- one
        ``latent_losses`` (now a real field) and two ``custom_losses``, which is
        not a domain at all. ``check_witness_corpus`` recorded all three drops
        as grandfathered debt; this converts them from recorded to rejected.

        Scoped to the ``*_losses`` suffix rather than flipping the whole class
        to ``extra="forbid"``: this block still carries legacy keys that other
        migrations own, and the loss LISTS are the ones whose silent loss costs
        an entire objective.
        """
        if not isinstance(data, dict):
            return data
        # A key the RENAME TABLE owns is NOT unknown -- it is a retired spelling
        # that `fold_renamed_keys` moves or `reject_renamed_keys` refuses by
        # name. `disable_default_losses` ends in `_losses` but is a filter list,
        # not a domain list, and after phase 10d it is no longer a field;
        # without this it is rejected here with a message telling the author to
        # "move the entries into the list matching the domain", which is not
        # what happened to their key.
        #
        # Every posture, not just `fold`. Consulting `folded_input_keys` alone
        # was correct only while this record was staged: the moment its corpus
        # count reached zero and it was promoted to `raise`, it dropped out of
        # that set and this guard resumed misclassifying it -- shadowing the
        # rename message that names the replacement. A guard that policies
        # extras has to consult the table, not one posture of it.
        declared = {n for n in cls.model_fields if n.endswith("_losses")}
        declared |= {k for k in renames_for_block("losses") if k.endswith("_losses")}
        unknown = sorted(k for k in data if isinstance(k, str) and k.endswith("_losses"))
        unknown = [k for k in unknown if k not in declared]
        if unknown:
            raise ValueError(
                f"losses.{unknown[0]} is not a declared loss list, so every entry "
                f"in it would be silently discarded (extra='ignore'). Declared "
                f"lists: {sorted(LOSS_LIST_DOMAINS)}. Move the entries into the "
                "list matching the domain the loss is registered for, or "
                "register the loss and add its domain."
            )
        return data

    @model_validator(mode="after")
    def _validate_list_losses_require_output_domain(self) -> "LossConfigSchema":
        """Ensure output_domain is set when list-based losses are used."""
        if self.uses_list_based_losses and not self.policy.output_domain:
            raise ValueError(
                "losses.output_domain must be set ('kspace', 'image', or 'complex_image') "
                "when using list-based loss configuration (kspace_losses, image_losses, complex_losses)."
            )
        # Legal set = the domains a loss LIST can actually grade in, i.e. the
        # values of `LOSS_LIST_DOMAINS`. Deriving it from the whole
        # `SignalDomain` enum instead was a trap: the enum has seven members and
        # the schema accepted all seven, but `loss_builder.py` only knows how to
        # bridge four and raises for `spectrum` / `pde_grid` / `mesh`. So an arm
        # could declare a domain that validates at load and then fails at build
        # -- acceptance without buildability, which is the shape this campaign
        # exists to remove.
        #
        # Supporting spectroscopy needs a `spectrum_losses` list AND a bridge for
        # it, not a wider enum; zero corpus arms declare any of the three today.
        _legal = {d.value for d in LOSS_LIST_DOMAINS.values()}
        if self.policy.output_domain and self.policy.output_domain not in _legal:
            raise ValueError(
                f"losses.output_domain='{self.policy.output_domain}' is invalid. "
                f"Must be one of: {sorted(_legal)}."
            )
        # A latent is produced by a LEARNED encoder, not by a transform, so
        # unlike kspace<->image there is no bridge that can manufacture one from
        # another domain's tensor. Grading a latent loss against a k-space or
        # image output would compare tensors that share no space. Raise rather
        # than degrade (NN#3).
        if self.latent_losses and self.policy.output_domain != SignalDomain.LATENT.value:
            raise ValueError(
                f"losses.latent_losses is declared with "
                f"losses.output_domain='{self.policy.output_domain}'. Latent losses "
                "grade the encoder's latent, and no bridge exists from "
                f"'{self.policy.output_domain}' into a latent -- the encoder is a "
                "learned map, not a transform. Set output_domain='latent' (the "
                "model must emit the latent), or move these entries to the list "
                "matching the domain the model actually outputs."
            )
        return self

    # Default losses per paradigm that can be disabled
    # Example: disable_default_losses=['mse'] disables MSE for diffusion
    # Supported values: 'mse', 'adversarial', 'kl', 'reconstruction'

    # `normalize_losses`, `clip_loss_value` and `loss_scaling` were deleted
    # 2026-08-03 (issue #676). All three had ZERO readers and ZERO corpus
    # declarations, so removal changes nothing that runs -- and leaving them
    # advertised three global loss controls that did nothing (pitfall #15).
    # `loss_scaling`'s apparent references were `MixedPrecisionConfig.loss_scaling`,
    # a different class: the leaf-name blind spot of issue #674.
    lambda_deep_supervision: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Weight for deep-supervision loss components. Was the root-level "
            "`deep_supervision_weight` until 2026-07-31 — a loss weight sitting "
            "at the config root, where nothing else that resolves through the "
            "loss-weight SSOT lives (pitfall #13b). Default 0.0 is unchanged, so "
            "the term stays off unless an arm asks for it."
        ),
    )

    def reconstruction_managed_losses(self) -> set[str]:
        """Loss names computed by the reconstruction loss computer.

        These resolve their optimized weight from ``reconstruction.lambda_*``
        (via ``resolve_static_loss_weight``) and are computed independently of
        the declarative ``kspace/image/complex`` lists. LossBuilder's
        unmigrated-key guard must therefore NOT flag them when they are
        deliberately kept OUT of those lists — the pma_02 / direct_ulf_to_hf_sr
        pattern, where a declarative entry's per-term weight is ignored for the
        total, so l1/ssim are routed through ``reconstruction.lambda_*``
        instead. Mirrors the ``reconstruction`` branch of ``get_enabled_losses``.
        """
        recon = self.reconstruction
        if recon is None:
            return set()
        cfg = recon.model_dump() if hasattr(recon, "model_dump") else {}
        managed: set[str] = set()
        for field_name, value in cfg.items():
            if not field_name.startswith("lambda_"):
                continue
            name = field_name[7:]
            if name == "recon":
                enable_field = "enable_recon"
            elif name == "marker":
                enable_field = "enable_marker_loss"
            elif name == "prior":
                enable_field = "enable_prior_loss"
            else:
                enable_field = f"enable_{name}"
            if cfg.get(enable_field, False) and isinstance(value, (int, float)) and value > 0:
                managed.add(name)
        return managed

    def get_enabled_losses(self) -> dict[str, float]:
        """Get all enabled losses with their weights across all paradigms.

        Returns a dictionary mapping loss names to their weights (lambda values).
        Only includes losses that are explicitly enabled.

        Returns:
            dict[str, float]: Mapping of loss name to weight.
                Example: {'l1': 10.0, 'perceptual': 10.0, 'ssim': 1.0, 'adversarial': 1.0}

        Note:
            - Returns only enabled losses (enable_* field must be True)
            - Respects disable_default_losses list
            - Only includes paradigm losses if that paradigm config is not None
            - Global loss_scaling is NOT applied (caller should apply if needed)
        """
        enabled_losses = {}

        # Helper function to extract enabled losses from a config
        def extract_from_config(config: BaseModel | None, paradigm: str) -> dict[str, float]:
            """Extract enabled losses from a paradigm-specific config.

            Only includes losses that are enabled AND have weight > 0.
            """
            losses = {}
            if config is None:
                return losses

            # Use model_dump to safely iterate over configuration fields (SSOT compliant)
            config_dict = config.model_dump() if hasattr(config, "model_dump") else {}

            # Iterate through all fields in the config dictionary
            for field_name, field_value in config_dict.items():
                # Special cases for GAN
                if paradigm == "gan":
                    if field_name == "lambda_adv":
                        loss_name = "adversarial"
                        enable_field = "enable_adversarial"
                    elif field_name == "lambda_gp":
                        loss_name = "gradient_penalty"
                        enable_field = "enable_gradient_penalty"
                    elif field_name == "feature_matching":
                        loss_name = "feature_matching"
                        enable_field = "enable_feature_matching"
                    elif field_name == "lambda_r1":
                        loss_name = "r1"
                        enable_field = "enable_r1"
                    elif field_name.startswith("lambda_"):
                        loss_name = field_name[7:]
                        enable_field = f"enable_{loss_name}"
                    else:
                        continue
                # Special cases for Diffusion
                elif paradigm == "diffusion":
                    if field_name == "lambda_mse":
                        loss_name = "mse"
                        enable_field = "enable_diffusion"
                    elif field_name.startswith("lambda_"):
                        loss_name = field_name[7:]
                        enable_field = f"enable_{loss_name}"
                    else:
                        continue
                # Special cases for Latent
                elif paradigm == "latent":
                    if field_name == "lambda_recon":
                        loss_name = "reconstruction"
                        enable_field = "enable_reconstruction"
                    elif field_name == "lambda_kl":
                        loss_name = "kl"
                        enable_field = "enable_kl"
                    elif field_name == "lambda_commit":
                        loss_name = "commitment"
                        enable_field = "enable_commitment"
                    elif field_name.startswith("lambda_"):
                        loss_name = field_name[7:]
                        enable_field = f"enable_{loss_name}"
                    else:
                        continue
                elif paradigm == "registration":
                    if field_name == "lambda_sim":
                        loss_name = "sim"
                        enable_field = "enable_lncc"
                    elif field_name == "lambda_smooth":
                        loss_name = "smooth"
                        enable_field = "enable_smoothness"
                    elif field_name.startswith("lambda_"):
                        loss_name = field_name[7:]
                        enable_field = f"enable_{loss_name}"
                    else:
                        continue
                else:
                    # Generic case for all paradigms
                    if not field_name.startswith("lambda_"):
                        continue

                    # Extract loss name: lambda_l1 → l1, lambda_ms_ssim → ms_ssim
                    loss_name = field_name[7:]  # Remove "lambda_" prefix

                    # Special cases for names that don't match enable flags directly
                    if paradigm == "reconstruction" and loss_name == "recon":
                        # lambda_recon pairs with enable_recon (the disentangled-VAE
                        # "general reconstruction loss"); enable_l1 is its own pair.
                        enable_field = "enable_recon"
                    elif paradigm == "reconstruction" and loss_name == "marker":
                        enable_field = "enable_marker_loss"
                    elif paradigm == "reconstruction" and loss_name == "prior":
                        enable_field = "enable_prior_loss"
                    else:
                        enable_field = f"enable_{loss_name}"

                # Check if this loss is enabled AND has non-zero weight
                is_enabled = config_dict.get(enable_field, False)

                if is_enabled and field_value > 0 and not is_disabled(f"{paradigm}_{loss_name}"):
                    losses[loss_name] = field_value

            return losses

        def is_disabled(loss_identifier: str) -> bool:
            """Check if a loss is in the disable_default_losses list."""
            # Check both full identifier (e.g., "gan_adversarial") and short name
            short_name = loss_identifier.rsplit("_", maxsplit=1)[-1]
            return (
                loss_identifier in self.policy.exclude_defaults
                or short_name in self.policy.exclude_defaults
            )

        def is_active(component: "LossComponentConfig") -> bool:
            """Single owner for "is this declared component active?".

            The ``composed`` block and every list in ``LOSS_LIST_DOMAINS`` hold
            the same type (``LossComponentConfig``), so they must answer this
            question identically. They did not: the ``composed`` loop honoured
            neither ``enabled`` nor ``weight > 0``, so ``enabled: false`` --
            the spelling ``ComposedLossConfig``'s own docstring advertises as
            the way to switch a component off -- was silently ignored and the
            component stayed in the returned dict. Non-negotiable 17: one owner
            per invariant, and the loser's copy is deleted rather than kept in
            sync.
            """
            return component.enabled and component.weight > 0 and not is_disabled(component.name)

        # Extract losses from each paradigm
        if self.reconstruction is not None:
            # If using list-based losses, only extract legacy reconstruction if explicitly defined
            # This prevents Pydantic's default_factory from triggering a ConfigurationError
            if not self.uses_list_based_losses or bool(self.reconstruction.model_fields_set):
                enabled_losses.update(extract_from_config(self.reconstruction, "reconstruction"))

        if self.gan is not None:
            # Similarly, don't extract default GAN if using list-based (though default is None)
            if not self.uses_list_based_losses or bool(self.gan.model_fields_set):
                enabled_losses.update(extract_from_config(self.gan, "gan"))

        if self.diffusion is not None:
            if not self.uses_list_based_losses or bool(self.diffusion.model_fields_set):
                enabled_losses.update(extract_from_config(self.diffusion, "diffusion"))

        if self.latent is not None:
            if not self.uses_list_based_losses or bool(self.latent.model_fields_set):
                enabled_losses.update(extract_from_config(self.latent, "latent"))

        if self.physics is not None:
            if not self.uses_list_based_losses or bool(self.physics.model_fields_set):
                enabled_losses.update(extract_from_config(self.physics, "physics"))

        if self.registration is not None:
            if not self.uses_list_based_losses or bool(self.registration.model_fields_set):
                enabled_losses.update(extract_from_config(self.registration, "registration"))

        if self.pinn is not None:
            if not self.uses_list_based_losses or bool(self.pinn.model_fields_set):
                enabled_losses.update(extract_from_config(self.pinn, "pinn"))

        if self.ssl is not None:
            if not self.uses_list_based_losses or bool(self.ssl.model_fields_set):
                enabled_losses.update(extract_from_config(self.ssl, "ssl"))

        if self.evidential is not None:
            if not self.uses_list_based_losses or bool(self.evidential.model_fields_set):
                enabled_losses.update(extract_from_config(self.evidential, "evidential"))

        if self.spatial is not None:
            if not self.uses_list_based_losses or bool(self.spatial.model_fields_set):
                enabled_losses.update(extract_from_config(self.spatial, "spatial"))

        # Handle composed losses if present
        if self.composed is not None:
            for component in self.composed.components:
                if is_active(component):
                    enabled_losses[component.name] = component.weight

        # Handle domain-aware list-based losses
        # These are included in the same dict so strategies can iterate uniformly.
        # The LossBuilder handles the bridging logic separately.
        #
        # EVERY list must appear here, from LOSS_LIST_DOMAINS rather than a
        # hand-written tuple. `LossBuilder._build_all_dynamic` early-returns
        # when this dict is empty, and that return sits BEFORE the branch that
        # calls `_build_list_based_losses` -- so a list missing from this loop
        # is not merely under-reported, it disables the whole list-based path
        # for any arm that declares nothing else. `latent_losses` was added to
        # the builder and to the schema without being added here, which made it
        # exactly the inert mechanism it was introduced to remove.
        for name in LOSS_LIST_DOMAINS:
            loss_list = getattr(self, name)
            for component in loss_list:
                if is_active(component):
                    enabled_losses[component.name] = component.weight

        return enabled_losses
