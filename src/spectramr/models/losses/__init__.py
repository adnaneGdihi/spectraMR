"""Loss functions and factory - Single Source of Truth for loss creation.

This module provides unified access to all loss functions via the registry pattern:
    from spectramr.models.losses import create_loss, create_composite_loss, register_loss

Phase 5 composition classes (recommended for composite losses):
    from spectramr.models.losses import ComposedLoss, WeightedLoss, ConditionalComposedLoss
"""

# Every submodule is imported for its @register_loss side effect: a module missing
# here leaves its losses unregistered, and create_loss raises on the name.
from . import (  # noqa: F401  (imported for @register_loss side effects)
    acq_flow_lie_loss,
    acquisition_losses,
    ambient_consistency_loss,
    beltrami_diagnostic_loss,  # GeoMamba-ULF: quasi-conformal regularity diagnostic
    beta_tc_vae_loss,
    biophysical_flow_loss,
    bloch_consistency_loss,  # Multi-contrast: parameter-map physics anchor
    bloch_signal_synthesis_consistency_loss,
    bloch_synth_losses,  # MRIxFields2026 2.1: dispersion_prior + bloch_source_consistency
    calibrated_gain_bloch_loss,  # exp 32: gain-calibrated q-maps from noise-unit M4Raw
    cartoon_texture_loss,  # B-2.5 MRIxFields2026 — BV/G decomposition, hallucination confinement
    charbonnier_loss,  # Losses guide §3
    complex_losses,
    composed_loss,
    concomitant_phase_residual_loss,
    conformal_geometry,  # audit_plan_novel — SFC §§3, 4, 6 (Beltrami / modulus / Teichmüller)
    contrast_consistency_loss,
    contrastive_losses,
    cross_contrast_losses,  # LNCCLoss registration
    cross_field_losses,  # MRIxFields2026: latent_cycle + field_flow_velocity + cocycle
    cubical_ph_w2_loss,  # GeoMamba-ULF: cubical persistent homology + Wasserstein-2
    cut_contrastive_loss,  # Losses guide §25 — patch-NCE for unpaired translation
    deep_supervision_loss,
    dice_anatomy_loss,  # Losses guide §69 — segmentation-Dice anti-hallucination
    diffusion_losses,
    dino_perceptual_loss,
    disentanglement_losses,
    dispersion_monotonicity_loss,
    distillation_losses,  # Losses guide §62/§63 — KD + EMA self-distillation
    domain_adaptation_loss,
    domain_adversarial_grl,
    dps_loss,  # Losses guide §66/§67 — DPS data-fit + manifold-constrained
    dwi_adc_monoexp_loss,  # monoexp ADC consumer of b_values
    edge_detection,
    equivariant_recon_loss,  # Losses guide §73 — equivariant SSL reconstruction
    evidential_loss,
    fisher_rao_geodesic_loss,  # fisher_rao_flow (information geometry)
    flow_losses,  # mri_flow regime: phase-contrast velocity losses
    flow_matching_losses,  # Losses guide §36/§37/§38 — CFM + OT-CFM + rectified flow
    fmri_mrf_losses,  # audit_plan_novel_fmri — fMRI §§2, 5 + MRF §3
    focal_frequency_loss,  # Focal Frequency Loss for Gibbs/SR
    frame_coherence_loss,  # A-6.5 SCO-Frame: learnable-frame mutual-coherence penalty
    frequency_loss,
    frontdoor_criterion_loss,  # frontdoor_federated / frontdoor_scanner
    gan_loss_library,
    gradient_domain_consistency_loss,  # Multi-contrast: edge preservation under translation
    gradient_entropy_loss,  # vf_35: image-sharpness autofocus diagnostic
    gw_cross_field_loss,
    heteroscedastic_losses,  # MRIxFields2026 B-2.9: heteroscedastic_ulf NLL
    hfen_loss,
    high_frequency_residual_loss,  # Multi-contrast Item 2 — HF residual for latent-diffusion residual heads
    histogram_loss,
    hjb_residual_loss,  # hjb_trajectory / hjb_waveform
    hodge_decomposition_loss,  # hodge_motion (Helmholtz-Hodge)
    hyperelastic_jacobian_loss,  # Jacobian det(J)=1 for tissue incompressibility
    infonce_critic,
    jacobian_anatomy_loss,  # NC cohort folding / volume-distortion guard
    joint_multi_contrast_sparsity_loss,  # Multi-contrast: Bilgic 2011 coupled gradient sparsity
    kl_divergence,
    koopman_linearity_loss,  # koopman_fmri (DMD linearity residual)
    latent_orthogonality_loss,  # pillar_05: z_p / z_a cross-covariance penalty
    kspace_consistency_tta_loss,
    kspace_magnitude_phase_loss,  # magnitude L1 + weighted phase; counters HF shrinkage
    kspace_physics_losses,  # K-space cold diffusion physics losses
    latent_consistency_loss,  # Experiment 32a
    latent_losses,
    lesion_weighted_loss,  # Multi-contrast: anomaly-aware weighting (§7)
    lipschitz_penalty,  # A-8.3 CertRob: differentiable spectral Lipschitz penalty
    lpips_loss,
    m4_losses,  # PMA-VarNet global loss
    marker_corruption_loss,  # Marker-anchored corruption loss for VF
    masked_kspace_l2_loss,  # CoNEX: L2 on the second repetition's lines M2
    masked_uncertainty_multi_contrast_loss,
    mind_ssc_loss,  # Experiment 32a
    mmd_loss,  # Losses guide §59 — Maximum Mean Discrepancy
    mrf_losses,  # mri_fingerprinting regime: soft Bloch-dictionary matching
    ncchi_consistency_loss,
    nll_bits_per_dim,  # bits-per-dim NLL for normalising flows
    noise2noise_loss,  # Losses guide §26 — Noise2Noise paired-noisy supervision
    nufft_sample_consistency_loss,  # Non-Cartesian sample-domain DC
    null_space_loss,  # Cold-diffusion C5: fibre-aware null-space content supervision
    ood_gated_reconstruction_loss,  # Multi-contrast: OOD-gated refinement (§7)
    optimal_transport_losses,  # sinkhorn / dynamic_ot / kidot
    pde_losses,
    perceptual_loss,
    perfusion_losses,  # mri_perfusion regime: Tofts / AIF / physiological-box
    phase_stego_score,
    physics_anchor_loss,
    physics_equivariance,  # audit_plan_novel — Ideas 3 & 6
    physics_informed_integration_loss,
    physics_losses,
    pisco_self_consistency_loss,
    pmps_losses,
    primary_ideal_membership_loss,  # primary_ideal (vanishing-ideal of tissue varieties)
    red_diff_loss,  # Losses guide §74 — RED-diff re-noising regularisation
    registration,
    registration_extras,  # Losses guide §48/§50/§52 — MI + bending energy + inverse consistency
    regularizers,  # MRI-specific regularizers (wavelet, hessian, hermitian, hankel, low-rank, phase smoothness)
    rician_consistency_loss,  # Rician noise-aware consistency loss for MRI
    scattering_besov_loss,  # B-1.3 MRIxFields2026 — wavelet-scattering + Besov detail prior
    se3_equivariance_defect,
    segmentation_equivariance_loss,  # Multi-contrast: frozen-segmenter cycle anchor (§2)
    sensor_vf_anchor_loss,
    sheaf_consistency_loss,  # sheaf_kspace / sheaf_multi_contrast
    sliced_score_matching_loss,  # Losses guide §35 — sliced score matching
    soft_dtw_loss,  # Hilbert-Mamba alignment loss
    spectral_band_split_loss,
    spectral_graph_loss,
    spectral_loss,
    spectral_radial_prior_loss,  # Multi-contrast: contrast-conditional radial spectrum prior
    spectral_triple_loss,  # Breakthrough 2026: Connes Morita-morphism ULF→HF
    spectroscopy_losses,  # mri_spectroscopy regime: AMARES-style FID fitting
    ssdu_loss,  # Losses guide §30/§31 — SSDU + multi-mask SSDU
    ssdu_normalized_loss,  # SSDU's normalised L1-L2 (Yaman et al. 2020)
    ssim_loss,
    standard_losses,
    stein_discrepancy_loss,  # stein_federated (KSD)
    storm_loss,  # manifold smoothness
    style_loss,  # Losses guide §16 — Gram/style loss
    sure_n2self_losses,  # Losses guide §29/§28 — SURE + Noise2Self
    topological_loss,
    translation_losses,  # Losses guide §23/§24 — cycle + identity
    tropical_semiring_loss,  # tropical_mrf / tropical_quantitative_maps
    tta_consistency_loss,  # Losses guide §72 — test-time consistency
    uncertainty_loss,
    vf_correction_losses,  # Dual-marker correction losses
    vf_losses,  # Virtual Fiducial marker-anchored losses
    vq_losses,
)

# Operator-ID (Proposal 1): structured-Gaussian paired likelihood (complex)
# and unpaired pushforward-MMD (image), registered via @register_loss.
# coil_subspace_residual penalises energy off the per-pixel coil manifold.
from .complex import (  # noqa: F401
    coil_subspace_residual,
    structured_gaussian_nll,
)

# Composition classes, re-exported below.
from .composed_loss import ComposedLoss, ConditionalComposedLoss, WeightedLoss
from .image import (  # noqa: F401
    multifield_data_consistency,
    pushforward_mmd,
)
from .registry import (
    LossRegistry,
    compatible_domains,
    create_composite_loss,
    create_loss,
    get_loss_capabilities,
    list_available,
    register_loss,
)

__all__ = [
    "ComposedLoss",
    "ConditionalComposedLoss",
    "LossRegistry",
    "WeightedLoss",
    "compatible_domains",
    "create_composite_loss",
    "create_loss",
    "get_loss_capabilities",
    "list_available",
    "register_loss",
]

# Out-of-tree losses: fire @register_loss decorators from entry-points /
# SPECTRAMR_PLUGINS modules. Placed at module end (late import, like the metrics
# subpackage import) so it never reorders the load-bearing ``from . import
# <loss>`` registration block above. spectramr.plugins is stdlib-only — no layer
# violation. A bad SPECTRAMR_PLUGINS token raises (fail-fast, pitfall #15).
from spectramr.plugins import discover_plugins as _discover_plugins

_discover_plugins("spectramr.losses")
