#!/usr/bin/env python3
"""Configuration Validation Constants

Centralized constants for configuration validation across the application.
This module contains all validation-related constants to ensure consistency
and maintainability.
"""

# Training Modes
VALID_TRAINING_MODES = [
    "gan",
    "diffusion",
    "reconstruction",
    "vae",
    "vqvae",  # discrete-latent VQ-VAE family (hierarchical_vq_vae, ...)
    "generative",  # maximum-likelihood density models (glow, equivariant_flow)
    "disentangled",  # Multi-contrast disentangled VAE
    "3d_generation",
    "test_time_adaptation",
    "swarm_learning",
    "domain_adaptation",
    "kspace_cold_diffusion",
    "ssl",
    "federated",
    "b0_mapping",
    "flow_matching",
    # [REMOVED 2026-07-19] "pretraining" -- never had a STRATEGY_CLASS_PATHS
    #   entry; the resolvable names are "mae" / "ssl" -> MAEPretrainingStrategy.
    # [REMOVED 2026-07-19] "kan" -- a category error: KAN is an architecture
    #   axis (model_type / attention_type / gate_type), never a paradigm. All
    #   40+ KAN arms already dispatch via diffusion / gan / reconstruction; the
    #   matching model_type was purged 2026-05-22.
    # [REMOVED 2026-07-19] "sensitivity_estimation" -- never had a strategy
    #   entry; "pinn" -> ConcretePINNSensitivityStrategy is the real name and
    #   the constraints entries were byte-identical. siren_sens_net retagged.
    "mae",
    "pinn",
    "test_time_optimization",
    "virtual_fiducial",
    "vf_admm",
    "multi_acquisition",  # faithful AFI/DAM/dual-echo field mapping
    "trajectory_recon",  # exp_vf_35 spiral trajectory recovery (GIRF/delay estimate)
    "pma_varnet",
    # Registered strategies that were dispatchable via strategy_class but absent
    # from the training_mode allow-list (2026-05-31): teacher-student privileged
    # distillation + the multi-stage pipeline orchestrator.
    "privileged_learning",
    "privileged",
    "multi",
    # audit_plan_novel.md — original 7 novel arms
    "physics_equivariant_ssl",
    "phys_eq_ssl",
    "ib_active_acquisition",
    "ib_bald",
    "score_field_tomography",
    "bloch_equivariant_translation",
    "riemannian_bloch_diffusion",
    # audit_plan_novel.md §§194-344 — SFC / conformal arms
    "adaptive_sfc_hssc",
    "conformal_diffusion_recon",
    "beltrami_motion_correction",
    "cortical_conformal_recon",
    "teichmuller_cold_diffusion",
    # audit_plan_novel_fmri.md — fMRI arms
    "spatiotemporal_adaptive_sfc_recon",
    "beltrami_epi_distortion",
    "cortical_conformal_fmri_recon",
    "riemannian_dfc_diffusion",
    "hrf_manifold_diffusion",
    # audit_plan_novel_fmri.md — MRF arms
    "spatiotemporal_mrf_recon",
    "riemannian_mrf_diffusion",
    "conformal_mrf_dictless_recon",
    "crlb_mrf_pulse_design",
    "cross_scanner_mrf_harmonisation",
    # Breakthrough 2026 (TODO/I'll run the full deep-ideation pipeline.md)
    # — both are thin aliases of `reconstruction` with paradigm-specific
    # data/loss configuration via YAML; the strategy_class resolves to
    # `ReconstructionTrainingStrategy` (see STRATEGY_CLASS_PATHS).
    "sle_compressed_sensing",
    "spectral_triple",
    # Breakthrough 2026 Batches 1 + 2 — thin aliases for the 20 deep-ideation
    # proposals. Paradigm-specific math lives in src/infrastructure/physics/,
    # src/models/blocks/, src/models/diffusion/, src/core/metrics/,
    # src/infrastructure/services/. Strategy mapping in
    # src/infrastructure/training/strategy_factory.py.
    "tropical_mrf",
    "fisher_rao_flow",
    "sheaf_kspace",
    "resetting_diffusion",
    "hyperbolic_cardiac",
    "heisenberg_phase",
    "hjb_trajectory",
    "primary_ideal",
    "symplectic_bloch",
    "frontdoor_federated",
    "hodge_motion",
    "levy_diffusion",
    "koopman_fmri",
    "sheaf_multi_contrast",
    "hjb_waveform",
    "stein_federated",
    "tropical_quantitative_maps",
    "frontdoor_scanner",
    # VF "P-arm" breakthrough strategies (B-1..B-4).
    "equivariance_conformal",
    "bloch_manifold_dps",
    "se3_equivariant_navigator",
    "hamiltonian_acquisition",
    # Frontier gap audit PR-4/5/7/9 (2026-05-29) — concrete strategy classes.
    "bloch_schrodinger_bridge",
    "i2sb",
    "vf_consistency_distillation",
    "universal_reconstruction",
    "flow_matching_pfode",
    # VF advanced arms (2026-06-04) — registered in strategy_factory.py
    # (IBVFTrainingStrategy / TwinLikelihoodDPSStrategy) with config schemas in
    # config/schemas/training/vf_advanced.py, but missing here so main rejected
    # exp_vf_ib_infonce_v2 / exp_vf_twin_dps_v2 at startup with "Unknown
    # training mode" despite a green Tier-0/1 audit (dispatch 20260603_200125,
    # tasks 49/50).
    "ib_vf",
    "twin_dps",
    # MICCAI MRIxFields2026 leads (2026-07-05): idea 4.2 cocycle-consistent unified
    # operator (adversarial, GAN-family) and idea 2.1 relaxometry+Bloch synthesis.
    # Registered in strategy_factory.py; field_cocycle rides the dual-optimizer path
    # (see optimization_builder), bloch_synth is a reconstruction-family arm.
    "field_cocycle",
    "bloch_synth",
]

# Model Types
VALID_MODEL_TYPES = [
    "unet",
    "unet3d",
    "enhanced_unet",
    "enhanced_deep_unet",
    # VF campaign Phase 2 breakthrough-method model types (B-3, B-4).
    "se3_equivariant_dc_navigator",
    "hamiltonian_trajectory_generator",
    # moe_kan → duplicate of moe_kan_generator (purged 2026-05-22)
    "hypernetwork_generator",
    "vit_kan",
    "kan_gan",
    # swin_transformer_kan → duplicate of swin_kan_transformer (purged 2026-05-22)
    "attention_unet",
    "edsr",
    "rcan",
    "swinir",
    "swin_ir",
    # nafnets → duplicate of nafnet (purged 2026-05-22)
    "nafnet",
    "diffusion_sr",
    "laplace_diffusion",
    "standard_unet",
    "diffusion",
    # stylegan → duplicate of stylegan2 (purged 2026-05-22)
    "sagan",
    # kan → no standalone impl; experiments use kan_unet (purged 2026-05-22)
    "wavelet_kan_vae",
    "fourier_kan_vae",
    "vae",
    "vqvae",
    "vqgan",
    "latent_diffusion",
    # latent_unet → no @register_model; LatentUNet is internal utility (purged 2026-05-22)
    # latent_diffusion_sr → no implementing class; config-filter entry only (purged 2026-05-22)
    "kspace_cold_diffusion",
    "kspace_cold_diffusion_generator",
    "image_cold_diffusion",
    # vit_with_kan → duplicate of vit_kan (purged 2026-05-22)
    "vit_hybrid_unet",
    # trellis_volume → no impl class (purged 2026-05-22)
    "trellis",
    # dicom_series → dataset type, not a model (purged 2026-05-22)
    # 2d_stack → dataset type, not a model (purged 2026-05-22)
    # trellis_diffusion → no impl class (purged 2026-05-22)
    # trellis_gaussian → duplicate of trellis_gaussian_vae; experiments use trellis_gaussian_vae (purged 2026-05-22)
    "trellis_gaussian_vae",
    # trellis_mesh → duplicate of trellis_mesh_vae; experiments use trellis_mesh_vae (purged 2026-05-22)
    # trellis_image_large → no impl class (purged 2026-05-22)
    "trellis_mesh_vae",
    # trellis_structured_vae → base class only; subclasses registered as trellis_gaussian_vae/trellis_mesh_vae (purged 2026-05-22)
    "sparse_vae",
    # unet_gan → no impl class (purged 2026-05-22)
    # gan → training_mode value, not a model_type (purged 2026-05-22)
    "chi_square_diffusion",
    "enhanced_kan_unet",
    "multicontrast_fusion",
    "variational_network",
    "unrolled_reconstruction",
    "conditional_diffusion",
    "diffusion_unet",
    "fusion_unet",
    "mae",
    # multimodal_conditional_diffusion → alias for conditional_diffusion; wired in stubs.py (purged from VALID 2026-05-22)
    # standard_vit → alias for stable_vit; wired in stubs.py (purged from VALID 2026-05-22)
    "unet_recon",
    # vit → alias for vision_transformer; wired in stubs.py (purged from VALID 2026-05-22)
    # vq_vae → duplicate of vqvae (purged 2026-05-22)
    # Frontier Models
    "moe_kan_generator",
    "kspace_gpt",
    "rl_acquisition_agent",
    "graph_unet",
    "split_client_encoder",
    "split_server_decoder",
    "adversarial_purification",
    "deep_image_prior",
    "continual_learning_unet",
    "mamba",
    "d2_mamba",
    "nca_generator",
    "gflownet_generator",
    "equivariant_diffusion_generator",
    "neural_ode_generator",
    "reflexion_reconstruction",
    # test_time_adaptation_strategy → strategy class, not a model; no implementing nn.Module (purged 2026-05-22)
    "symbolic_regression_wrapper",
    "foundation_model",
    "scanner_invariant_embedding",
    "digital_twin_simulator",
    "spiking_unet",
    "hyperspectral_gan",
    "surgery_loop_agent",
    # Paradigm Shift Models (Phase 16)
    "mk_recon",
    "clifford_gan",
    "world_model",
    "quantum_unet",
    "geometric_mri",
    "gaussian_splatting",
    "rl_scanner",
    # photonic_fft → hardware layer/block (PhotonicFFT2d), not a registered generator model (purged 2026-05-22)
    "implicit_kan",
    "implicit_mri_field",
    # mesh_mri → no implementing class anywhere (purged 2026-05-22)
    # Frontier Models - Continued
    "continual_learning",
    "moe_sparse_routing",
    "hierarchical_window_transformer",
    "lottery_ticket_network",
    # New Experimental Models
    "dino_trellis",
    "structure_tensor_transformer",
    "evidential_unet",
    "diffeomorphic_synthesis_net",
    # universal_adapter → no implementing class (purged 2026-05-22)
    "ensemble_disagreement",
    "attention_cross_modality",
    "fourier_augmented",
    "semantic_adaptive_norm",
    "probabilistic_graphical",
    "neural_cache",
    "isotonic_constrained",
    "riemannian_geometric",
    "hierarchical_softmax_attention",
    "pinn",
    "pinn_mri",
    "pinn_nerf",
    "fno",
    "uno",
    # MNO family — see TODO/Mamba_NO.md and docs/cs_mno.rst
    "cs_mno_operator",  # CS-MNO: Continuous-SFC + Spectral + Hilbert (recommended)
    "hs_mno_operator",  # MNO-1: Hilbert-Scan only (discrete Δ, no spectral)
    "c_mno_operator",  # MNO-2: Continuous-SFC only (no spectral)
    "s_mno_operator",  # MNO-4: Spectral + discrete-Δ Mamba
    "me_mno_operator",  # MNO-3: Multi-curve Equivariant (D4 averaging)
    "g_mno_operator",  # MNO-5: Geodesic-SFC Mamba (uses MetricSFCLinearizer)
    "geo_mamba_unet",  # GeoMamba-ULF generator (foreground-restricted SFC + FiLM-Mamba)
    "slice_to_volume",
    "cascaded_diffusion",
    "vision_transformer",
    "equivariant_unet",
    "neural_ode",
    "swin_transformer",
    "coil_sensitivity_network",
    "fpn_unet",
    "linformer",
    "vision_mamba",
    "perceiver",
    "convnext",
    "densenet_plus",
    "batch_harmonization",
    "generative_prior",
    "deep_unfolding",
    "graph_cuts_neural",
    "energy_based_gan",
    "cascade_residual",
    "contrastive_divergence",
    "differentiable_rendering",
    "symbolic_regression",
    # pareto_optimization → no implementing class (purged 2026-05-22)
    # capsule_networks → duplicate of capsule_network (purged 2026-05-22)
    # kspace_gpt_foundation → duplicate of kspace_gpt (purged 2026-05-22)
    "llm_mri_foundation",
    "federated_learning",
    "temporal_coherence_4d",
    "conditional_continual_vae",
    # adversarial_robustness → training strategy, not a model_type (purged 2026-05-22)
    # mixture_density → duplicate of mixture_density_network (purged 2026-05-22)
    "hierarchical_bayesian",
    "causal_discovery",
    "spin_glass_inspired",
    "universal_multitask",
    "physics_aware_gaussian_splatter",
    "pggs_pipeline",
    "lans_generator",
    "hobs_generator",
    "pimn",
    "latent_gaussian_diffusion",
    # MambaReconstruction (CamelCase) → duplicate; lowercase "mamba_reconstruction" is registered (purged 2026-05-22)
    # Missing Legacy Models
    # advanced_vae_latent → no implementing class (purged 2026-05-22)
    # attention_dense → no implementing class (purged 2026-05-22)
    # autoregressive_decoder → no implementing class (purged 2026-05-22)
    # max_pool → no implementing class (purged 2026-05-22)
    "disentangled_mri",
    # beta_vae_gan → no implementing class; experiment_52 uses kan_gan (purged 2026-05-22)
    "bloch_aware_unet",
    "bloch_cycle_network",
    # blurring_diffusion → no implementing class (purged 2026-05-22)
    "capsule_network",
    "causal_autoregressive",
    "causal_transformer",
    "cin_unet",
    "conditional_batch_norm",
    "consistency_model",
    "consistency_regularized",
    # contrastive_network → no implementing class (purged 2026-05-22)
    "coupling_flow_diffusion",
    "cross_attention_fusion",
    # dense_prediction_unet → no implementing class (purged 2026-05-22)
    "differentiable_convex",
    "disentangled_vae",
    # divergence_free_flow → no implementing class (purged 2026-05-22)
    "domain_adversarial",
    # dynamic_unet → no implementing class (purged 2026-05-22)
    "ensemble_unet",
    # epistemic_aleatoric → no implementing class (purged 2026-05-22)
    # equivariant_flow → no implementing class (purged 2026-05-22)
    "foundation_model_adapter",
    # gated_gnn → no implementing class (purged 2026-05-22)
    # glow → no implementing class; normalizing_flow is the registered flow model (purged 2026-05-22)
    "gnn_coil_fusion",
    "guided_diffusion",
    # hierarchical_vae_ladder → no implementing class (purged 2026-05-22)
    # hierarchical_vq_vae → no implementing class (purged 2026-05-22)
    "hypernetwork",
    "hypernetwork_unet",
    # hyperspherical_network → no implementing class (purged 2026-05-22)
    # hyperspherical_vae → no implementing class (purged 2026-05-22)
    # implicit_neural_representation → helper class in diffdeur.py, not a top-level registered model (purged 2026-05-22)
    "implicit_representation",
    "invariance_equivariance",
    "invertible_network",
    "knowledge_distillation",
    "latent_diffusion_hierarchical",
    "learned_dictionary",
    "llm_guided_reconstruction",
    "lottery_ticket_unet",
    "maml_meta_learning",
    "mixture_density_network",
    "mixture_of_experts",
    # moe_vae → no implementing class (purged 2026-05-22)
    "multi_branch_residual",
    "multi_task_shared",
    "neural_ca",
    # neural_ode_adaptive → alias for neural_ode; wired in stubs.py (purged from VALID 2026-05-22)
    "neural_process",
    "normalizing_flow",
    # normalizing_flow_image → duplicate of normalizing_flow (purged 2026-05-22)
    # octave_conv → no implementing class (purged 2026-05-22)
    "padnet",
    "parametric_instance_norm",
    # progressive_gan → no implementing class (purged 2026-05-22)
    "prompt_based_adaptation",
    "quantum_inspired",
    # rdn → no implementing class (purged 2026-05-22)
    "rectified_flow",
    # recursive_cascade → no implementing class (purged 2026-05-22)
    # recursive_residual → no implementing class (purged 2026-05-22)
    # reversible_network → duplicate of invertible_network (purged 2026-05-22)
    "robust_certified",
    "rotation_prediction_unet",
    "score_based_diffusion",
    # score_based_gan → no implementing class (purged 2026-05-22)
    # sde_diffusion → duplicate name for score_based_diffusion (purged 2026-08-12)
    # sn_gan → no implementing class (purged 2026-05-22)
    # sparse_hybrid_attention → no implementing class (purged 2026-05-22)
    # sparse_transformer → no implementing class (purged 2026-05-22)
    "spiking_neural_network",
    "structured_vae",
    # student_unet → no implementing class (purged 2026-05-22)
    "tda_informed",
    "tensor_decomposition_net",
    # topographic_vae → no implementing class (purged 2026-05-22)
    # uncertain_pyramid → no implementing class (purged 2026-05-22)
    # unrolled_list → duplicate of unrolled_reconstruction (purged 2026-05-22)
    "vision_mamba_local_global",
    "vision_transformer_ssl",
    # vit_adapter → no implementing class (purged 2026-05-22)
    "vrnn_reconstruction",
    # wasserstein_vae → no implementing class (purged 2026-05-22)
    # zero_shot_transfer → one name bound to two classes; ZeroRFReconstructor in
    #   model_factory, RectifiedFlowGenerator in init_registry (purged 2026-08-12).
    #   Use ``zerorf`` or ``rectified_flow`` — they say which model you mean.
    "x_diffusion",
    "conditional_diffusion_generator",
    "uncertainty_wrapper",
    "swinir_generator",
    "nafnet_generator",
    "rcan_generator",
    "elan_generator",
    "edsr_generator",
    "anisotropic_voxel_generator",
    "unetr_generator",
    "monai_swin_unetr",  # Alias for unetr_generator (registered in stubs.py)
    "vnet_generator",
    "nesvor",
    "slice_to_volume_generator",
    "vae_3d",
    "score_based_diffusion_generator",
    "laplace_diffusion_generator",
    "rician_diffusion_generator",
    "x_diffusion_latent",
    "medical_dino",
    "mcgi_encoder",
    "lcah_encoder",
    "nose_operator",
    "fpsp_score",
    "ac_ivae",
    "dcsrn",
    "han_generator",
    "efficient_transformer",
    "stable_vit",
    "stable_diffusion_adapter",
    "score_based_unet",
    "refined_kan_unet",
    "diffdeur",
    "dummy",
    "hybrid_qmap_generator",
    "zerorf",
    # New Experimental Models (Phase 17)
    "geo_fno",
    "aftnet",
    "inr_cristal",
    "tenf_inr",
    "simmlm",
    "css_diff",
    "vqvae_generator",
    "b0_estimator",
    # SIREN / PINN models
    "siren",
    "siren_sens_net",
    # Physics-guided diffusion models
    "tissue_diffusion_mri",
    # Virtual Fiducial motion-correction models
    "hyper_mamba_unet",
    "cross_attention_oracle_unet",
    "hyper_mamba_bridge",
    # Hilbert-Mamba ULF-to-HF generators
    "ct_mamba",
    "fe_mamba",
    "crm_mamba",
    "hw_mamba",
    "mm_mamba",
    "mdi_mamba",
    "two_half_d_mamba",
    "hld_mamba",
    "inr_mamba",
    "hilbert_mamba_unet",
    "hdsf_mamba_unet",
    "hdsf_hld_mamba",
    # PMA-VarNet Blueprint: Task 1 (Reconstruction)
    "modl_sr",
    "wiener_unet",
    "diff_nufft",
    "kalman_pll",
    "dirichlet_espirit",
    "vcc_varnet",
    "ls_ista",
    "hard_dc_forcing",
    "pdhg_recon",
    "neural_complex_sum",
    # PMA-VarNet Blueprint: Task 2 (Proof Blocks)
    "fourier_shift_op",
    "sh_fitter",
    "stn_warp",
    "eddy_predictor_1d",
    "richardson_lucy",
    "pinn_helmholtz",
    "pre_whitening_layer",
    "rigid_affine_kabsch",
    "dip_unet",
    "svd_projection",
    # PMA-VarNet Blueprint: Task 3 (Field Mapping)
    "resnet_unwrap",
    "bloch_siegert_algebraic",
    "afi_ratio_cnn",
    "dam_trigfit",
    "siamese_epi_tracker",
    "phase_tracking_lstm",
    "reciprocity_divisor",
    "mrf_dict_matcher",
    "bssfp_peak_finder",
    "graph_cut_unwrap",
    # PMA-VarNet Blueprint: Task 5 (Unified)
    "pma_varnet",
    # K-Space-Native Virtual Fiducial Operators
    "kspace_phase_ramp",
    "mtf_deapodization",
    "phase_navigator_lock",
    "marker_grappa_conv",
    "diff_trajectory_opt",
    # Grand Unified VF Pipeline Models
    "dynamic_mr_nerf",
    "neural_advection",
    # JFE-VarNet SOTA
    "jfe_varnet",
    # Legacy experiment model types
    "latent_flow",
    "varnet",
    "artifact_robust_generator",
    "vit_kan_generator",
    "mae_mri",
    "efficient_unet",
    "bloch_cycle",
    # SOTA Registry models (Beyond-SOTA Amorphous Encoding & Pathology Synthesis)
    "q_fno",
    "q_inr",
    "hnf",
    "gauss_mrf",
    "bloch_mamba_v2",
    "graph_kan",
    # clifford_gan already listed above (deduped 2026-05-22)
    # ── Auto-synced from runtime MODEL_REGISTRY (smoke test 2026-04-27) ──
    "base_3d",
    "bloch_mamba",
    "causal_gan",
    "coil_sensitivity",
    "cold_diffusion",
    "complex_unet",
    "conditional_patchgan_discriminator",
    "conditioning_encoder",
    "configurable_unet",
    "configurable_vae",
    "cpt_4dmr",
    "cv_ssd",
    "cycle_physio_diff",
    "cyclegan_generator",
    "cyclesr",
    "deep_ensemble",
    "deeponet",
    "diff_mamba",
    "diff_seq",
    "diff_varnet",
    "diff_varnet_kan",
    "differentiable_slicer",
    "diffusion_reconstruction",
    "disentangled_encoder",
    "dual_motion_inr",
    "dynamic_3dgs",
    "elan",
    "eq_deq",
    "equivariant_diffusion",
    "fourier_kan_swin",
    "frequency_domain_discriminator",
    "gen_3dgs",
    "gen_mrf",
    "generative_world_model",
    "gflownet",
    "grafmri",
    "graph_unet_diffusion",
    "han",
    "hobs",
    "homotopic_transport",
    "hybrid_pinn",
    "hybrid_transformer_diffusion",
    "imdn",
    "inr_hypernetwork",
    "k_hash",
    "kan_discriminator",
    "kan_inr",
    "kan_unet",
    "kspace_aware_discriminator",
    "kspace_discriminator",
    "lagrange_euler",
    "lans",
    "latent_discriminator",
    "latent_gan_generator",
    "ldm_latent_discriminator",
    "ldm_multiscale_latent_discriminator",
    "ldm_patch_latent_discriminator",
    "lie_diffusion",
    "local_inr",
    "mamba_reconstruction",
    "mamba_unet",
    "mamba_unet_v2",
    "mc_cm",
    "mc_dropout",
    "medgs",
    "meta_tta",
    # metal_diffusion → facade alias of cold_diffusion_process; the name promised
    #   metal-artefact degradation the alias never set (purged 2026-08-12)
    "multi_contrast_generator",
    "multiscale_latent_discriminator",
    "nca",
    "neuro_mamba",
    "nsp_tta",
    "pa_pcfm",
    "patch_gan",
    "patch_latent_discriminator",
    "patchgan_3d_discriminator",
    "patchgan_discriminator",
    "physics_driven",
    "physics_vae",
    "pin_inr",
    "q_kan",
    "qmap_generator",
    "qspace_translation",
    "realesrgan_discriminator",
    "res_mocodiff",
    "restormer",
    "rician_diffusion",
    "rl_mri",
    "robust_vae",
    "scanner_adaptive_generator",
    "score_disent",
    "simae",
    "simple_chi_square_diffusion",
    "simple_enhanced_diffusion",
    "simple_gaussian_diffusion",
    "simple_gaussian_kan_diffusion",
    "sota_adapter",
    "sparse_vae_generator",
    "spectral_graph",
    "swin_diff_rec",
    "swin_diff_rec_kan",
    "swin_hybrid_unet",
    "swin_mamba_kan",
    "template_deformer",
    "therm_pinn",
    "tissue_diffusion_generator",
    "transformer_unet",
    "ttt_mamba",
    "umr_fm",
    "unet_2_5d",
    "vae_fourier_kan_vae",
    "vae_generator",
    "vae_wavelet_kan_vae",
    "varnet_kan",
    "vnet",
    "vqgan_generator",
    "wasserstein_discriminator",
    # ---------------------------------------------------------------
    # Wave-2 ledger restoration (2026-05-22): 20 name-variant aliases
    # re-added to VALID_MODEL_TYPES after being wired as aliases in
    # models/stubs.py (TODO/deleted_model_types_reimplementation_plan.md
    # Phase 1). Each resolves to an already-registered canonical model,
    # so the Tier-0 audit promise (advertised ⇒ resolvable) holds.
    # ---------------------------------------------------------------
    "MambaReconstruction",
    "capsule_networks",
    "kspace_gpt_foundation",
    "mixture_density",
    "moe_kan",
    "nafnets",
    "normalizing_flow_image",
    "reversible_network",
    "standard_vit",
    "stylegan",
    "swin_transformer_kan",
    "trellis_gaussian",
    "trellis_mesh",
    "unrolled_list",
    "vit",
    "vit_with_kan",
    "vq_vae",
    # conditional_diffusion_cfg, multimodal_conditional_diffusion → duplicate names
    #   for conditional_diffusion; CFG and multimodal conditioning are knobs on it,
    #   not separate models (purged 2026-08-12)
    "neural_ode_adaptive",
    # ---------------------------------------------------------------
    # Phase 3 new model implementations (2026-05-22): 12 freshly-built
    # models registered under models/{generative,vae,vq,diffusion,gans,
    # generators}. See TODO/phase3_model_implementations_detailed.md.
    # ---------------------------------------------------------------
    "glow",
    "equivariant_flow",
    "divergence_free_flow",
    "blurring_diffusion",
    "hierarchical_vae_ladder",
    "hierarchical_vq_vae",
    "hyperspherical_vae",
    "moe_vae",
    "beta_vae_gan",
    "progressive_gan",
    "octave_conv",
    "gated_gnn",
    # MICCAI MRIxFields2026 leads (2026-07-05) — registered via @register_model in
    # models/generators/{field_cocycle_generator,relaxometry_encoder}.py. The opaque-
    # band residual refiner is an internal submodule of relaxometry_encoder (one
    # generator the pipeline builds), not a separately-registered model.
    "field_cocycle_generator",
    "relaxometry_encoder",
]

# KAN Types
VALID_KAN_TYPES = [
    "BSpline",
    "Fourier",
    "Polynomial",
    "Chebyshev",
    "Jacobi",
    "RBF",
    "WavKAN",
    "FastKAN",
]

# Input Types
VALID_INPUT_TYPES = ["image", "latent", "embedding", "kspace"]

# GAN Loss Types
VALID_GAN_LOSS_TYPES = ["wgan-gp", "vanilla", "lsgan", "hinge"]

# Device Types
VALID_DEVICE_PREFIXES = ["cpu", "cuda"]

# Discriminator Types
VALID_DISCRIMINATOR_TYPES = [
    "patchgan",
    "standard_discriminator",
]

# Dataset Types
VALID_DATASET_TYPES = ["2d", "3d", "3d_volumetric", "kspace"]

# Performance Limits
MAX_RECOMMENDED_WORKERS = 8
MIN_Z_DIM_FOR_GAN = 1

# ============================================================================
# VALIDATION RANGES & BOUNDS
# ============================================================================

# Range: (min, max) - inclusive on both ends
VALID_AUGMENTATION_PROBABILITY_RANGE = (0.0, 1.0)
VALID_SPLIT_RATIO_RANGE = (0.0, 1.0)
VALID_LEARNING_RATE_RANGE = (1e-6, 1.0)  # Reasonable bounds for LR
VALID_BETA_RANGE = (0.0, 1.0)

# Optimizer-specific ranges
VALID_ADAM_BETA1_RANGE = (0.0, 1.0)
VALID_ADAM_BETA2_RANGE = (0.0, 1.0)
VALID_ADAM_EPSILON_RANGE = (1e-10, 1e-6)
VALID_MOMENTUM_RANGE = (0.0, 1.0)

# Batch size bounds
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 10000
RECOMMENDED_BATCH_SIZE_RANGE = (8, 512)

# Data loading bounds
MIN_NUM_WORKERS = 0
MAX_NUM_WORKERS = 32
RECOMMENDED_NUM_WORKERS_RANGE = (0, 8)

# Training bounds
MIN_EPOCHS = 1
MAX_EPOCHS = 10000
MIN_VALIDATION_INTERVAL = 1
MAX_VALIDATION_INTERVAL = 100

# Model parameter bounds
MIN_IN_CHANNELS = 1
MAX_IN_CHANNELS = 256
MIN_OUT_CHANNELS = 1
MAX_OUT_CHANNELS = 256

# GAN-specific bounds
MIN_GAN_DISCRIMINATOR_LAYERS = 1
MAX_GAN_DISCRIMINATOR_LAYERS = 10
MIN_LAMBDA_ADVERSARIAL = 0.0
MAX_LAMBDA_ADVERSARIAL = 10.0
MIN_LAMBDA_L1 = 0.0
MAX_LAMBDA_L1 = 100.0

# Diffusion-specific bounds
MIN_DIFFUSION_STEPS = 10
MAX_DIFFUSION_STEPS = 10000
MIN_NOISE_SCALE = 0.0
MAX_NOISE_SCALE = 1.0

# VAE-specific bounds
MIN_LATENT_DIM = 1
MAX_LATENT_DIM = 2048
MIN_KL_WEIGHT = 0.0
MAX_KL_WEIGHT = 1.0

# Acceleration bounds
MIN_GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRADIENT_ACCUMULATION_STEPS = 1000

# Field validation specifications
FIELD_VALIDATION_SPECS = {
    # Data configuration
    "data.batch_size": {
        "type": "int",
        "min": MIN_BATCH_SIZE,
        "max": MAX_BATCH_SIZE,
        "default": 32,
        "recommended_range": RECOMMENDED_BATCH_SIZE_RANGE,
    },
    "data.num_workers": {
        "type": "int",
        "min": MIN_NUM_WORKERS,
        "max": MAX_NUM_WORKERS,
        "default": 4,
        "recommended_range": RECOMMENDED_NUM_WORKERS_RANGE,
    },
    "data.center_fraction": {
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.08,
    },
    "data.acceleration": {
        "type": "int",
        "min": 1,
        "max": 32,
        "default": 4,
    },
    # Optimization configuration
    "optimization.optimizer.learning_rate": {
        "type": "float",
        "min": 1e-6,
        "max": 1.0,
        "default": 0.0001,
    },
    "optimization.epochs": {
        "type": "int",
        "min": MIN_EPOCHS,
        "max": MAX_EPOCHS,
        "default": 100,
    },
    "optimization.optimizer.beta1": {
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.9,
    },
    "optimization.optimizer.beta2": {
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.999,
    },
    # Model configuration
    "model.in_channels": {
        "type": "int",
        "min": MIN_IN_CHANNELS,
        "max": MAX_IN_CHANNELS,
        "default": 2,
    },
    "model.out_channels": {
        "type": "int",
        "min": MIN_OUT_CHANNELS,
        "max": MAX_OUT_CHANNELS,
        "default": 2,
    },
    # Training configuration
    "training.epochs": {
        "type": "int",
        "min": MIN_EPOCHS,
        "max": MAX_EPOCHS,
        "default": 100,
    },
    "training.validation_interval": {
        "type": "int",
        "min": MIN_VALIDATION_INTERVAL,
        "max": MAX_VALIDATION_INTERVAL,
        "default": 5,
    },
    "training.checkpoint_interval": {
        "type": "int",
        "min": 1,
        "max": 100,
        "default": 10,
    },
    # Loss configuration (v6.0)
    "losses.gan.lambda_adv": {
        "type": "float",
        "min": MIN_LAMBDA_ADVERSARIAL,
        "max": MAX_LAMBDA_ADVERSARIAL,
        "default": 0.1,
    },
    "losses.reconstruction.lambda_l1": {
        "type": "float",
        "min": MIN_LAMBDA_L1,
        "max": MAX_LAMBDA_L1,
        "default": 1.0,
    },
    # Spelled `num_timesteps` until 2026-08-12, which is not the field the schema
    # declares or the code reads (`cold_diffusion_inference_strategy` asks for
    # `timesteps`). `training.diffusion` is extra="allow", so the advertised
    # spelling parsed into an untyped attribute and the arm ran on the default.
    # A fold record now moves it; this entry names the real field (issue #980).
    "training.diffusion.timesteps": {
        "type": "int",
        "min": MIN_DIFFUSION_STEPS,
        "max": MAX_DIFFUSION_STEPS,
        "default": 1000,
    },
    "training.vae.latent_dim": {
        "type": "int",
        "min": MIN_LATENT_DIM,
        "max": MAX_LATENT_DIM,
        "default": 256,
    },
}

# ============================================================================
# OPTIMIZER-SPECIFIC CONFIGURATIONS
# ============================================================================

OPTIMIZER_DEFAULTS = {
    "adam": {
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.0,
    },
    "sgd": {
        "momentum": 0.9,
        "weight_decay": 0.0,
    },
    "adamw": {
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.01,
    },
}

# ============================================================================
# SCHEDULER-SPECIFIC CONFIGURATIONS
# ============================================================================

SCHEDULER_DEFAULTS = {
    "cosine": {
        "T_max": 100,
        "eta_min": 0.0,
    },
    "exponential": {
        "gamma": 0.9,
    },
    "linear": {
        "start_factor": 1.0,
        "total_iters": 5,
    },
    "step": {
        "step_size": 30,
        "gamma": 0.1,
    },
}

# ============================================================================
# LOSS CONFIGURATION RANGES
# ============================================================================

# Keyed by the CANONICAL v1.0 `losses.*` field name (issue #933). The
# pre-#933 table was keyed `lambda_adversarial`, a v4.0 `objectives.*`
# spelling that reads a block `extra="forbid"` rejects and 0/647 arms
# declare -- so the range for it could never be consulted. Only
# `lambda_adversarial` had a rename (-> `losses.gan.lambda_adv`);
# `lambda_l1`/`lambda_l2`/`lambda_ssim`/`lambda_perceptual`/`lambda_kl`
# already matched their `losses.reconstruction.*` / `losses.latent.*`
# field spellings, so those keys are unchanged. `lambda_vae` and
# `lambda_consistency` are dropped: neither names a field under `losses.*`
# in the v1.0 schema -- `lambda_consistency` lives under strategy-specific
# `training.*` knobs (e.g. low_rank_sparse, spectra_tta), a different SSOT
# this table does not cover.
LOSS_WEIGHT_RANGES = {
    "lambda_adv": (0.0, 10.0),
    "lambda_l1": (0.0, 100.0),
    "lambda_l2": (0.0, 100.0),
    "lambda_ssim": (0.0, 10.0),
    "lambda_perceptual": (0.0, 10.0),
    "lambda_kl": (0.0, 10.0),
}

# ============================================================================
# METRIC CONFIGURATION
# ============================================================================

METRIC_DEFAULTS = {
    "psnr": {"data_range": 1.0},
    "ssim": {"data_range": 1.0},
    "mae": {},
    "lpips": {},
    "fid": {"num_samples": 10000},
}

# ─────────────────────────────────────────────────────────────────────────────
# Contrast-conditioning strategy allow-list (SSOT).
#
# The MRIxFields2026 dataset ALWAYS emits ``batch['contrast_id']``. A model
# widened with ``model_kwargs.use_contrast_conditioning: true`` guards on that
# key (#15, no silent mode-averaging) and RAISES "batch carries no
# 'contrast_id'" at step 0 if its training strategy drops the key on the way to
# the model forward. This frozenset is the single source of truth for the
# strategies VERIFIED (grep of every ``contrast_id`` seam + the mechanism-fires
# test) to thread it through BOTH training and validation.
#
# Consumed by ``ConfigHealthChecker.check_contrast_conditioning_strategy_threaded``
# (the cluster-side audit gate) AND ``tests/unit/experiments/
# test_mrixfields_arms_audit.py`` (the repo-side per-arm gate). Enabling contrast
# on an arm whose strategy is NOT here shipped the 2026-07-07 regression (26 arms
# enabled, only 8 strategies threaded) — adding a new contrast-enabled strategy
# requires threading contrast_id AND listing it here.
# ─────────────────────────────────────────────────────────────────────────────
CONTRAST_THREADED_STRATEGIES: frozenset[str] = frozenset(
    {
        "field_flow",
        "field_bridge",
        "field_guided_diffusion",
        "fisher_rao_geodesic",
        "scattering_besov",
        "steerable_synthesis",
        "brenier_synthesis",
        "field_conditioned_inr",
        "ulf_redegrad_tta",
        "ulf_dps",
        "generative_refiner",
        "field_cold_diffusion",
    }
)
