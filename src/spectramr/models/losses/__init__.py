"""Loss functions and factory - Single Source of Truth for loss creation.

This module provides unified access to all loss functions via the registry pattern:
    from spectramr.models.losses import create_loss, create_composite_loss, register_loss

Phase 5 composition classes (recommended for composite losses):
    from spectramr.models.losses import ComposedLoss, WeightedLoss, ConditionalComposedLoss
"""

# New registry API (preferred)
# Import all loss modules to trigger registration. Without these imports
# the @register_loss decorators inside each module never fire, and the
# registry would silently advertise an incomplete surface — exactly the
# CLAUDE.md #9 silent-fallback class. See TODO/audit/12_losses.md F8.
# PMPS (2026-05-19): tissue-parameter constraint / smoothness, MMD,
# per-pair consistency residual — registered via @register_loss.
# IB-VF / InfoNCE (Phase 2): symmetric bilinear-critic mutual-information
# lower bound used by the IBVFTrainingStrategy.
# Twin-DPS (Phase 3): phase-stego marker-score for diffusion-posterior sampling.
# VF campaign Phase 2 breakthrough-method losses (B-3 navigator defect, B-4 acquisition).
# SPECTRA TTA (plan §D): self-supervised k-space consistency objective for
# single-subject test-time adaptation; used by SpectraTestTimeAdaptationStrategy.
# DL-BAE (contrast/field-agnostic bundle M4): field-summed reconstruction
# fidelity (image) + the dT1/dB0 >= 0 hinge (physics), registered via
# @register_loss. Explicit imports so registration fires at registry load.
# SOTA plan: nc-chi Bessel-ratio consistency (T4) + Ambient held-out k-space
# consistency (Phase B) + PISCO calibrationless PI self-consistency (P3.3).
# Explicit import so @register_loss fires at registry load.
# Contrast/field-agnostic bundle: GW correspondence-free cross-field alignment
# (GW-CFA) and the acquisition-flow Lie-derivative equivariance penalty (AFE).
# NOSE physics-anchoring term (operator output vs analytic Bloch render).
from . import (
    acq_flow_lie_loss,  # noqa: F401
    acquisition_losses,  # noqa: F401
    ambient_consistency_loss,  # noqa: F401
    beltrami_diagnostic_loss,  # GeoMamba-ULF: quasi-conformal regularity diagnostic
    beta_tc_vae_loss,  # audit 12 F8: was decorated but not imported
    biophysical_flow_loss,
    bloch_consistency_loss,  # Multi-contrast: parameter-map physics anchor
    bloch_signal_synthesis_consistency_loss,  # audit 12 F8
    bloch_synth_losses,  # MRIxFields2026 2.1: dispersion_prior + bloch_source_consistency
    cartoon_texture_loss,  # B-2.5 MRIxFields2026 — BV/G decomposition, hallucination confinement
    charbonnier_loss,  # Losses guide §3
    complex_losses,
    composed_loss,  # NEW: Phase 5 composition
    concomitant_phase_residual_loss,  # audit 12 F8
    conformal_geometry,  # audit_plan_novel — SFC §§3, 4, 6 (Beltrami / modulus / Teichmüller)
    contrast_consistency_loss,  # audit 12 F8
    contrastive_losses,
    cross_contrast_losses,  # LNCCLoss registration
    cross_field_losses,  # MRIxFields2026: latent_cycle + field_flow_velocity + cocycle
    cubical_ph_w2_loss,  # GeoMamba-ULF: cubical persistent homology + Wasserstein-2
    cut_contrastive_loss,  # Losses guide §25 — patch-NCE for unpaired translation
    deep_supervision_loss,  # Extracted from composite.py in Phase 5
    dice_anatomy_loss,  # Losses guide §69 — segmentation-Dice anti-hallucination
    diffusion_losses,
    dino_perceptual_loss,
    disentanglement_losses,
    dispersion_monotonicity_loss,  # noqa: F401
    distillation_losses,  # Losses guide §62/§63 — KD + EMA self-distillation
    domain_adaptation_loss,
    domain_adversarial_grl,  # @register_loss decorator triggers on import
    dps_loss,  # Losses guide §66/§67 — DPS data-fit + manifold-constrained
    dwi_adc_monoexp_loss,  # audit 2026-07 I2: monoexp ADC consumer of b_values
    edge_detection,
    equivariant_recon_loss,  # Losses guide §73 — equivariant SSL reconstruction
    evidential_loss,
    fisher_rao_geodesic_loss,  # fisher_rao_flow (information geometry)
    flow_losses,  # mri_flow regime: phase-contrast velocity losses
    flow_matching_losses,  # Losses guide §36/§37/§38 — CFM + OT-CFM + rectified flow
    fmri_mrf_losses,  # audit_plan_novel_fmri — fMRI §§2, 5 + MRF §3
    focal_frequency_loss,  # NEW: Focal Frequency Loss for Gibbs/SR
    frame_coherence_loss,  # A-6.5 SCO-Frame: learnable-frame mutual-coherence penalty
    frequency_loss,
    frontdoor_criterion_loss,  # frontdoor_federated / frontdoor_scanner
    gan_loss_library,
    gradient_domain_consistency_loss,  # Multi-contrast: edge preservation under translation
    gradient_entropy_loss,  # vf_35: image-sharpness autofocus diagnostic
    gw_cross_field_loss,  # noqa: F401
    heteroscedastic_losses,  # MRIxFields2026 B-2.9: heteroscedastic_ulf NLL
    hfen_loss,  # audit 12 F1+F8: canonical "hfen" registration
    high_frequency_residual_loss,  # Multi-contrast Item 2 — HF residual for latent-diffusion residual heads
    histogram_loss,
    hjb_residual_loss,  # hjb_trajectory / hjb_waveform
    hodge_decomposition_loss,  # hodge_motion (Helmholtz-Hodge)
    hyperelastic_jacobian_loss,  # NEW: Jacobian det(J)=1 for tissue incompressibility
    infonce_critic,  # noqa: F401
    joint_multi_contrast_sparsity_loss,  # Multi-contrast: Bilgic 2011 coupled gradient sparsity
    jacobian_anatomy_loss,  # noqa: F401 — NC cohort folding / volume-distortion guard
    kl_divergence,
    koopman_linearity_loss,  # koopman_fmri (DMD linearity residual)
    kspace_consistency_tta_loss,  # noqa: F401
    kspace_physics_losses,  # NEW: K-space cold diffusion physics losses
    latent_consistency_loss,  # NEW: Experiment 32a Phase 2
    latent_losses,
    lesion_weighted_loss,  # Multi-contrast: anomaly-aware weighting (§7)
    levy_score_consistency_loss,  # levy_diffusion (alpha-stable fractional score)
    lipschitz_penalty,  # A-8.3 CertRob: differentiable spectral Lipschitz penalty
    lpips_loss,
    m4_losses,  # NEW: PMA-VarNet global loss
    marker_corruption_loss,  # NEW: Marker-anchored corruption loss for VF
    masked_uncertainty_multi_contrast_loss,  # audit 12 F8
    mind_ssc_loss,  # NEW: Experiment 32a Phase 2
    mmd_loss,  # Losses guide §59 — Maximum Mean Discrepancy
    mrf_losses,  # mri_fingerprinting regime: soft Bloch-dictionary matching
    ncchi_consistency_loss,  # noqa: F401
    nll_bits_per_dim,  # Phase 3: bits-per-dim NLL for normalising flows
    noise2noise_loss,  # Losses guide §26 — Noise2Noise paired-noisy supervision
    nufft_sample_consistency_loss,  # noqa: F401  # Non-Cartesian sample-domain DC
    null_space_loss,  # Cold-diffusion C5: fibre-aware null-space content supervision
    ood_gated_reconstruction_loss,  # Multi-contrast: OOD-gated refinement (§7)
    optimal_transport_losses,  # Phase 5 canonical-home migration (sinkhorn/dynamic_ot/kidot)
    pde_losses,
    perceptual_loss,
    perfusion_losses,  # mri_perfusion regime: Tofts / AIF / physiological-box
    phase_stego_score,  # noqa: F401
    physics_anchor_loss,  # noqa: F401
    physics_equivariance,  # audit_plan_novel — Ideas 3 & 6
    physics_informed_integration_loss,  # Phase 5 canonical-home migration
    physics_losses,
    pisco_self_consistency_loss,  # noqa: F401
    pmps_losses,  # noqa: F401
    primary_ideal_membership_loss,  # primary_ideal (vanishing-ideal of tissue varieties)
    red_diff_loss,  # Losses guide §74 — RED-diff re-noising regularisation
    registration,
    registration_extras,  # Losses guide §48/§50/§52 — MI + bending energy + inverse consistency
    regularizers,  # MRI-specific regularizers (wavelet, hessian, hermitian, hankel, low-rank, phase smoothness)
    resetting_consistency_loss,  # resetting_diffusion (Evans-Majumdar criticality)
    rician_consistency_loss,  # NEW: Rician noise-aware consistency loss for MRI
    scattering_besov_loss,  # B-1.3 MRIxFields2026 — wavelet-scattering + Besov detail prior
    se3_equivariance_defect,  # noqa: F401
    segmentation_equivariance_loss,  # Multi-contrast: frozen-segmenter cycle anchor (§2)
    sensor_vf_anchor_loss,  # audit 12 F8
    sheaf_consistency_loss,  # sheaf_kspace / sheaf_multi_contrast
    sliced_score_matching_loss,  # Losses guide §35 — sliced score matching
    soft_dtw_loss,  # NEW: Hilbert-Mamba alignment loss
    spectral_band_split_loss,  # @register_loss decorator triggers on import
    spectral_graph_loss,
    spectral_loss,
    spectral_radial_prior_loss,  # Multi-contrast: contrast-conditional radial spectrum prior
    spectral_triple_loss,  # Breakthrough 2026: Connes Morita-morphism ULF→HF
    spectroscopy_losses,  # mri_spectroscopy regime: AMARES-style FID fitting
    ssdu_loss,  # Losses guide §30/§31 — SSDU + multi-mask SSDU
    ssim_loss,
    standard_losses,
    stein_discrepancy_loss,  # stein_federated (KSD)
    storm_loss,  # Phase 5 canonical-home migration (manifold smoothness)
    style_loss,  # Losses guide §16 — Gram/style loss
    sure_n2self_losses,  # Losses guide §29/§28 — SURE + Noise2Self
    topological_loss,
    translation_losses,  # Losses guide §23/§24 — cycle + identity
    # PR-0 (2026-05-28): façade-strategy completion — each wraps a real
    # orphaned physics primitive so the themed key stops silently running
    # as a vanilla baseline (see TODO/pr0_facade_strategy_triage_2026_05_28.md).
    tropical_semiring_loss,  # tropical_mrf / tropical_quantitative_maps
    tta_consistency_loss,  # Losses guide §72 — test-time consistency
    uncertainty_loss,
    vf_correction_losses,  # NEW: Dual-marker correction losses
    vf_losses,  # NEW: Virtual Fiducial marker-anchored losses
    vq_losses,
)

# Operator-ID (Proposal 1): structured-Gaussian paired likelihood (complex)
# and unpaired pushforward-MMD (image), registered via @register_loss.
# coil_subspace_residual penalises energy off the per-pixel coil manifold.
from .complex import (  # noqa: F401
    coil_subspace_residual,
    structured_gaussian_nll,
)

# Phase 5: Composed loss classes
from .composed_loss import ComposedLoss, ConditionalComposedLoss, WeightedLoss
from .image import (
    multifield_data_consistency,  # noqa: F401
    pushforward_mmd,  # noqa: F401
)
from .registry import (
    ILoss,
    LossRegistry,
    compatible_domains,
    create_composite_loss,
    create_loss,
    get_loss_capabilities,
    list_available,
    register_loss,
)

__all__ = [
    # Registry API
    "LossRegistry",
    "create_loss",
    "create_composite_loss",
    "register_loss",
    "compatible_domains",
    "get_loss_capabilities",
    "list_available",
    # Phase 5: Composition classes
    "ComposedLoss",
    "ConditionalComposedLoss",
    "WeightedLoss",
]

# Out-of-tree losses: fire @register_loss decorators from entry-points /
# SPECTRAMR_PLUGINS modules. Placed at module end (late import, like the metrics
# subpackage import) so it never reorders the load-bearing ``from . import
# <loss>`` registration block above. spectramr.plugins is stdlib-only — no layer
# violation. A bad SPECTRAMR_PLUGINS token raises (fail-fast, pitfall #15).
from spectramr.plugins import discover_plugins as _discover_plugins

_discover_plugins("spectramr.losses")
