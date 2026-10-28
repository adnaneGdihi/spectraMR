"""Generator Implementations Package
================================

This package contains all concrete generator implementations.
Each generator follows SOLID principles and implements the IGenerator interface.
"""

import logging

logger = logging.getLogger(__name__)

from spectramr.models.reconstruction.unet import (
    EnhancedUNetGenerator,
    StandardUNetGenerator,
    UNet,
    create_configurable_unet_generator,
    create_enhanced_unet_generator,
    create_standard_unet_generator,
)

# Mixins (Composition over Inheritance)
from .mixins import DiffusionMixin, LatentMixin, PhysicsMixin

# Protocols (Static Polymorphism)
from .protocols import (
    IDiffusionGeneratorProtocol,
    IGeneratorProtocol,
    ILatentGeneratorProtocol,
    IPhysicsGeneratorProtocol,
)

from ..gans.kan_gan import KANGenerator
from ..gans.vit_kan_gan import ViTKANGenerator

# Automatically scanned and verified generators
from .adversarial_purification import AdversarialPurificationDiffusion
from .anisotropic_voxel_generator import AnisotropicVoxelGenerator
from .artifact_aware_augmentation import ArtifactAwareAugmentation
from .autoencoder_pretrain_generator import AutoencoderPretrainGenerator
from .base_3d_generator import Base3DGenerator
from .bloch_cycle_network import BlochCycleNetwork
from .bloch_parameter_bottleneck import BlochParameterBottleneckGenerator
from .breakthrough_attention_generators import (  # Breakthrough 2026 (P6–P10)
    BlochLRSAttentionUNet,
    LanczosAttentionUNet,
    MPSAttentionUNet,
    TJLMRFEstimator,
    ToeplitzAttentionUNet,
)
from .breakthrough_geometric_generators import (  # Breakthrough 2026 (Batch 1)
    HeisenbergReconUNet,
    HyperbolicCineUNet,
)
from .cascaded_diffusion_generator import CascadedDiffusionGenerator
from .chi_square_diffusion_generator import ChiSquareDiffusionGenerator
from .coil_sensitivity_network import CoilSensitivityNetwork
from .cold_diffusion_generator import ColdDiffusionGenerator
from .conditional_diffusion_generator import ConditionalDiffusionGenerator
from .continual_learning_unet import ContinualLearningUNet
from .counterfactual_healthy_generator import CounterfactualHealthyGenerator
from .cross_contrast_cross_domain_generator import CrossContrastCrossDomainGenerator
from .cs_mno_operator import CSMNOOperator  # noqa: F401
from .cycle_physio_diff import CyclePhysioDiffGenerator
from .cyclesr_generator import CycleSRGenerator
from .dcsrn_generator import DCSRNGenerator
from .deep_image_prior import DeepImagePriorGenerator
from .deeponet_generator import DeepONetGenerator
from .diffdeur import DiffDeuR
from .diffusion_sr_generator import DiffusionSRGenerator
from .digital_twin_simulator import DigitalTwinSimulator
from .disentangled_mri import DisentangledMRI
from .edsr_generator import EDSRGenerator
from .efficient_transformer import EfficientTransformerGenerator
from .elan_generator import ELANGenerator
from .equivariant_diffusion_generator import EquivariantDiffusionGenerator
from .fno_generator import FNOGenerator
from .foundation_model import FoundationModel
from .gflownet_generator import GFlowNetGenerator
from .gnn_coil_fusion import GNNCoilFusion
from .graph_unet_generator import GraphUNetGenerator
from .guided_diffusion_generator import GuidedDiffusionGenerator
from .han_generator import HANGenerator
from .hobs_generator import HoBSGenerator
from .hybrid_transformer_diffusion_generator import HybridTransformerDiffusionGenerator
from .hypernetwork_contrast_generator import HypernetworkContrastGenerator
from .hypernetwork_generator import HypernetworkGenerator
from .hyperspectral_gan import HyperspectralGAN
from .imdn_generator import IMDNGenerator
from .kan_vae_generator import KANVAEGenerator
from .kspace_cold_diffusion_generator import KSpaceColdDiffusionGenerator
from .kspace_gpt import KSpaceGPT
from .lans_generator import LaNSGenerator
from .laplace_diffusion_generator import LaplaceDiffusionGenerator
from .latent_diffusion_generator import LatentDiffusionGenerator
from .latent_diffusion_with_residual_head import LatentDiffusionWithResidualHead
from .latent_flow_generator import LatentFlowGenerator
from .latent_gaussian_diffusion import LatentGaussianDiffusion
from .mno_family import (
    CMNOOperator,  # noqa: F401
    GMNOOperator,  # noqa: F401
    HSMNOOperator,  # noqa: F401
    MEMNOOperator,  # noqa: F401
    SMNOOperator,  # noqa: F401
)
from .multi_contrast_conditioning import MultiContrastConditionalGenerator
from .multicontrast_fusion_generator import MultiContrastFusionGenerator
from .nafnet_generator import NAFNetGenerator
from .nca_generator import NCAGenerator
from .nesvor import NeSVoR
from .neural_ode_generator import NeuralODEGenerator
from .normalizing_flow_generator import NormalizingFlowGenerator
from .physics_driven_network import PhysicsDrivenNetwork
from .pimn import PIMN
from .pin_inr import PININR
from .pinn_nerf_generator import PINNNeRFGenerator
from .siren_universal import SIRENUniversalGenerator

PINNeRF = PINNNeRFGenerator
# PMA-VarNet Blueprint Task 3b: K-space operators (fires @register_model decorators)
# audit_plan_novel SFC §3: conformal multi-scale ULF→HF
# (registers "conformal_geomamba_ulf")
import spectramr.models.generators.b0_hypernetwork  # noqa: E402  # B0-conditioned generator
import spectramr.models.generators.conformal_geomamba_ulf

# audit_plan_novel_fmri — fMRI + MRF 2026 models
# (registers "b0_beltrami_estimator", "hrf_diffusion_prior",
#  "tangent_spd_diffusion", "mrf_tangent_score", "conformal_fp_embedding",
#  "scanner_time_reparam")
import spectramr.models.generators.fmri_models

# geomamba_ulf — foreground-restricted, contrast-conditioned Mamba SR
# (registers "geo_mamba_unet"). The @register_model decorator only fires on
# import; without this line the 14 GeoMambaULFStrategy arms crashed at
# model-build with "Generator type 'geo_mamba_unet' not registered" even
# though the module existed (the conformal sibling above is a DISTINCT model).
import spectramr.models.generators.geo_mamba_unet

# audit_plan_novel — Idea 1: Hermitian-equivariant operators
# (registers "hermitian_fno" and "hermitian_kspace_unet")
import spectramr.models.generators.hermitian_fno_unet
import spectramr.models.generators.mrf_models
import spectramr.models.generators.vf_kspace_operators  # noqa: F401  registers "kspace_phase_ramp"

from ..experimental.implicit_kan import ImplicitMRIField  # noqa: F401
from ..gaussian_splatting.medgs import MedGS  # noqa: F401
from ..geometric.b0_estimator import B0Estimator  # noqa: F401

# MAE generator is in a separate package
from ..mae.mae_generator import MAEGenerator  # noqa: F401
from ..reconstruction.zerorf import ZeroRFReconstructor  # noqa: F401
from .aftnet_generator import AFTNetGenerator

# Phase 1: Newly implemented experimental generators
from .capsule_network import CapsuleNetworkGenerator
from .causal_discovery import CausalDiscoveryGenerator
from .conditional_continual_vae import ConditionalContinualVAEGenerator
from .consistency_model_generator import ConsistencyModelGenerator  # noqa: F401

# Virtual Fiducial generators
from .cross_attention_oracle import CrossAttentionOracleUNet, MarkerCrossAttention

# New experimental generators
from .css_diff_generator import CSSdiffGenerator
from .diffeomorphic_synthesis_net import DiffeomorphicSynthesisNet
from .differentiable_rendering import DifferentiableRenderingGenerator
from .dino_trellis import DINOTrellisGenerator

# Phase 3: Cross-package and experimental registrations
from .dummy_generator import DummyGenerator  # noqa: F401
from .dynamic_mr_nerf import DynamicMRNeRF
from .efficient_unet import EfficientUNetGenerator
from .ensemble_disagreement import EnsembleDisagreementGenerator
from .evidential_unet import EvidentialUNet
from .geo_fno_generator import GeoFNOGenerator
from .hdsf_mamba_generators import (
    HDSFHLDMamba,
    HDSFMambaUNet,
)
from .crm_mamba_generator import CRMMamba  # noqa: E402
from .fe_mamba import FEMamba  # noqa: E402
from .ct_mamba_generator import CTMamba  # noqa: E402
from .hw_mamba import HWMamba  # noqa: E402
from .inr_mamba import INRMamba  # noqa: E402
from .hilbert_mamba_generators import (
    HilbertMambaUNet,
    HLDMamba,
    MDIMamba,
    MMMamba,
    TwoHalfDMamba,
)
from .hybrid_pinn import HybridQMapGenerator  # noqa: F401
from .hyper_mamba_bridge import HyperMambaBridge, SSMParameters
from .hyper_mamba_unet import HyperMambaUNet
from .image_cold_diffusion_generator import ImageColdDiffusionGenerator
from .image_fno_operator import ImageFNOOperator  # noqa: E402, F401
from .windowed_sequence_denoiser import WindowedSequenceDenoiser  # noqa: E402, F401
from .inr_cristal_generator import INRCRISTALGenerator
from .isotonic_constrained import IsotonicConstrainedGenerator
from .mamba_unet import MambaReconstruction, MambaUNet, MambaUNet_v2
from .masked_contrast_unet import MaskedContrastUNet  # noqa: E402  (this block follows code)
from .neural_advection_generator import NeuralAdvectionGenerator
from .neural_cache import NeuralCacheGenerator

# PMA-VarNet Blueprint Task 5: Unified architecture
from .padnet import PaDNet
from .paired_synthesis_generator import PairedSynthesisGenerator
from .pma_varnet import PMAVarNet

# PMPS Phase 2/3 — corruption operators + paired synthesis generator.
from .pmps_corruption_operators import HFCorruptionOperator, ULFCorruptionOperator
from .probabilistic_graphical import ProbabilisticGraphicalGenerator
from .qspace_translation import qSpaceTranslation
from .rcan_generator import RCANGenerator

# Phase 2: Missing registrations (P0 gap analysis fixes)
from .rectified_flow_generator import RectifiedFlowGenerator  # noqa: F401
from .reflexion_reconstruction import ReflexionReconstruction
from .restormer_generator import RestormerGenerator
from .rician_diffusion_generator import RicianDiffusionGenerator

# Relocated 2026-05-15 from src/application/agents/ per
# TODO/audit/00_implementation_tracker.md D14. They are nn.Module +
# IGenerator, so canonical home is here (CLAUDE.md pitfall #12).
from .rl_acquisition_agent import RLAcquisitionAgent
from .robust_vae_generator import RobustVAEGenerator
from .scanner_invariant_embedding import ScannerInvariantEmbedding
from .scanner_shift_adaptation import ScannerAdaptiveGenerator
from .score_based_diffusion_generator import ScoreBasedDiffusionGenerator
from .siren_generator import SIRENGenerator
from .siren_pinn import SirenSensNet
from .slat_vae_slab_to_volume import SLATVAESlabToVolumeGenerator
from .slice_to_volume_generator import SliceToVolumeGenerator
from .sparse_vae_generator import SparseVAEGenerator
from .sparse_vae_slab_to_volume import SparseVAESlabToVolumeGenerator
from .spiking_unet import SpikingUNet
from .spiral_trajectory_estimator import SpiralTrajectoryEstimator
from .stable_diffusion_adapter_generator import StableDiffusionAdapterGenerator
from .stargan_v2 import (
    MappingNetwork,
    StarGANv2Generator,
    StyleEncoder,
)
from .structure_tensor_transformer import StructureTensorTransformer
from .surgery_loop_agent import SurgeryLoopAgent
from .swin_transformer_generator import SwinTransformerGenerator
from .swinir_generator import SwinIRGenerator
from .symbolic_regression import SymbolicRegressionGenerator
from .symbolic_regression_wrapper import SymbolicRegressionWrapper
from .tall_sr import TALLSuperResolution  # noqa: F401
from .temporal_coherence_4d import TemporalCoherence4DGenerator
from .tissue_diffusion_mri import TissueDiffusionMRIGenerator
from .tissue_parameter_diffusion import TissueParameterDiffusion
from .trellis_generator import TRELLISGenerator
from .trellis_structured_vae import TrellisStructuredGaussianVAEGenerator
from .uncertainty_wrappers import DeepEnsembleGenerator
from .unet3d_generator import UNet3DGenerator
from .unet_2_5d_generator import UNet2_5DGenerator

# Core verified generators
from .unet_generator import AttentionUNetGenerator, UNetGenerator
from .unetr_generator import UNETRGenerator
from .universal_multitask import (
    UniversalMultitaskDualContrastGenerator,
    UniversalMultitaskGenerator,
)
from .unrolled_reconstruction_generator import UnrolledReconstructionGenerator
from .unrolled_tissue_net import UnrolledTissueNet  # noqa: E402  (this block follows code)
from .vae_3d_generator import VAE3DGenerator
from .vae_generator import VAEGenerator
from .vf_architectures import (
    BlochAwareUNet,
    CascadeResidualGenerator,
    CINUNet,
    CrossAttentionFusionGenerator,
    DeepUnfoldingGenerator,
    LearnedDictionaryGenerator,
    PINNMRIGenerator,
    VariationalNetworkGenerator,
)

# PMA-VarNet Blueprint Task 3: Field mapping generators
from .vf_field_generators import (
    AFIRatioCNN,
    BlochSiegertAlgebraicGenerator,
    BSSFPB0Regressor,
    BSSFPPeakFinder,
    DAMTrigFitGenerator,
    GraphCutUnwrapGenerator,
    MRFDictMatcher,
    PhaseTrackingLSTM,
    ReciprocityDivisor,
    ResNetUnwrapGenerator,
    SiameseEPITracker,
)
from .vf_kspace_operators import (
    DifferentiableTrajectoryOptimizer,
    KSpacePhaseRampConjugator,
    MarkerGRAPPAConvolution,
    MTFDeApodizationOperator,
    PhaseNavigatorLock,
)

# PMA-VarNet Blueprint Task 2: Proof blocks
from .vf_proof_blocks import (
    DIPUNet,
    EddyPredictor1D,
    FourierShiftOperator,
    PINNHelmholtzBlock,
    PreWhiteningLayer,
    RichardsonLucyFilter,
    RigidAffineKabsch,
    SphericalHarmonicFitter,
    STNWarpBlock,
    SVDProjectionLayer,
)

# PMA-VarNet Blueprint Task 1: Reconstruction generators
from .vf_reconstruction_generators import (
    DiffNUFFTGenerator,
    DirichletESPIRiTGenerator,
    HardDCForcingGenerator,
    KalmanPLLGenerator,
    LSISTAGenerator,
    MoDLSRGenerator,
    NeuralComplexSumGenerator,
    PDHGReconGenerator,
    VCCVarNetGenerator,
    WienerUNetGenerator,
)
from .vision_transformer import VisionTransformer
from .vnet_generator import VNetGenerator
from .vqgan_generator import VQGANGenerator
from .vqvae_generator import VQVAEGenerator
from .x_diffusion_generator import XDiffusionGenerator
from .x_diffusion_latent import XDiffusionLatentGenerator
from .xdta import XDTA  # IMPLEMENTATION_SPEC.md §11 — Cross-Domain Traversal Attention

__all__ = [
    # Protocols (Static Polymorphism)
    "IGeneratorProtocol",
    "IDiffusionGeneratorProtocol",
    "ILatentGeneratorProtocol",
    "IPhysicsGeneratorProtocol",
    # Mixins (Composition over Inheritance)
    "DiffusionMixin",
    "PhysicsMixin",
    "LatentMixin",
    # Generators
    "UNetGenerator",
    "StandardUNetGenerator",
    "UNet",
    "EnhancedUNetGenerator",
    "create_standard_unet_generator",
    "create_enhanced_unet_generator",
    "create_configurable_unet_generator",
    "AdversarialPurificationDiffusion",
    "AnisotropicVoxelGenerator",
    "ArtifactAwareAugmentation",
    "Base3DGenerator",
    "BlochCycleNetwork",
    "CascadedDiffusionGenerator",
    "ChiSquareDiffusionGenerator",
    "CoilSensitivityNetwork",
    "ColdDiffusionGenerator",
    "ConditionalDiffusionGenerator",
    "ImageColdDiffusionGenerator",
    "ContinualLearningUNet",
    "CyclePhysioDiffGenerator",
    "CycleSRGenerator",
    "DCSRNGenerator",
    "DeepImagePriorGenerator",
    "DeepONetGenerator",
    "DiffDeuR",
    "DiffusionSRGenerator",
    "DigitalTwinSimulator",
    "DisentangledMRI",
    "EDSRGenerator",
    "EfficientTransformerGenerator",
    "ELANGenerator",
    "EquivariantDiffusionGenerator",
    "FNOGenerator",
    "FoundationModel",
    "GFlowNetGenerator",
    "GNNCoilFusion",
    "GraphUNetGenerator",
    "GuidedDiffusionGenerator",
    "HANGenerator",
    "HoBSGenerator",
    "HybridTransformerDiffusionGenerator",
    "HypernetworkGenerator",
    "HyperspectralGAN",
    "IMDNGenerator",
    "KANGenerator",
    "KANVAEGenerator",
    "KSpaceColdDiffusionGenerator",
    "KSpaceGPT",
    "LaNSGenerator",
    "LaplaceDiffusionGenerator",
    "LatentDiffusionGenerator",
    "LatentGaussianDiffusion",
    "LatentFlowGenerator",
    "MultiContrastConditionalGenerator",
    "MultiContrastFusionGenerator",
    "NAFNetGenerator",
    "NCAGenerator",
    "NeSVoR",
    "NeuralODEGenerator",
    "NormalizingFlowGenerator",
    "PhysicsDrivenNetwork",
    "PIMN",
    "PININR",
    "PINNNeRFGenerator",
    "qSpaceTranslation",
    "RCANGenerator",
    "ReflexionReconstruction",
    "RestormerGenerator",
    "RicianDiffusionGenerator",
    "RobustVAEGenerator",
    "ScannerInvariantEmbedding",
    "ScannerAdaptiveGenerator",
    "ScoreBasedDiffusionGenerator",
    "SliceToVolumeGenerator",
    "SparseVAEGenerator",
    "SpikingUNet",
    "StableDiffusionAdapterGenerator",
    "StarGANv2Generator",
    "MappingNetwork",
    "StyleEncoder",
    "SwinIRGenerator",
    "SymbolicRegressionWrapper",
    "TRELLISGenerator",
    "TrellisStructuredGaussianVAEGenerator",
    "DeepEnsembleGenerator",
    "UNet3DGenerator",
    "UNet2_5DGenerator",
    "UnrolledTissueNet",
    "MaskedContrastUNet",
    "AttentionUNetGenerator",
    "UNETRGenerator",
    "UnrolledReconstructionGenerator",
    "VAE3DGenerator",
    "VAEGenerator",
    "SwinTransformerGenerator",
    "VisionTransformer",
    "VNetGenerator",
    "VQGANGenerator",
    "VQVAEGenerator",
    "XDiffusionGenerator",
    "ViTKANGenerator",
    # New experimental generators
    "CSSdiffGenerator",
    "GeoFNOGenerator",
    "INRCRISTALGenerator",
    "AFTNetGenerator",
    "SirenSensNet",
    # Virtual Fiducial generators
    "CrossAttentionOracleUNet",
    "MarkerCrossAttention",
    # VF Architectures (legacy)
    "BlochAwareUNet",
    "CascadeResidualGenerator",
    "CINUNet",
    "CrossAttentionFusionGenerator",
    "DeepUnfoldingGenerator",
    "LearnedDictionaryGenerator",
    "PINNMRIGenerator",
    "VariationalNetworkGenerator",
    # PMA-VarNet Task 1 (Reconstruction)
    "MoDLSRGenerator",
    "WienerUNetGenerator",
    "DiffNUFFTGenerator",
    "KalmanPLLGenerator",
    "DirichletESPIRiTGenerator",
    "VCCVarNetGenerator",
    "LSISTAGenerator",
    "HardDCForcingGenerator",
    "PDHGReconGenerator",
    "NeuralComplexSumGenerator",
    # PMA-VarNet Task 2 (Proof Blocks)
    "FourierShiftOperator",
    "SphericalHarmonicFitter",
    "STNWarpBlock",
    "EddyPredictor1D",
    "RichardsonLucyFilter",
    "PINNHelmholtzBlock",
    "PreWhiteningLayer",
    "RigidAffineKabsch",
    "DIPUNet",
    "SVDProjectionLayer",
    # PMA-VarNet Task 3 (Field Mapping)
    "ResNetUnwrapGenerator",
    "BlochSiegertAlgebraicGenerator",
    "AFIRatioCNN",
    "DAMTrigFitGenerator",
    "SiameseEPITracker",
    "PhaseTrackingLSTM",
    "ReciprocityDivisor",
    "MRFDictMatcher",
    "BSSFPPeakFinder",
    "BSSFPB0Regressor",
    "GraphCutUnwrapGenerator",
    "KSpacePhaseRampConjugator",
    "MTFDeApodizationOperator",
    "PhaseNavigatorLock",
    "MarkerGRAPPAConvolution",
    "DifferentiableTrajectoryOptimizer",
    "SpiralTrajectoryEstimator",
    # PMA-VarNet Task 5 (Unified)
    "PMAVarNet",
    "PaDNet",
    # Other
    "HyperMambaBridge",
    "SSMParameters",
    "HyperMambaUNet",
    # Mamba backbone
    "MambaUNet",
    "MambaUNet_v2",
    "MambaReconstruction",
    # Grand Unified VF pipeline generators
    "DynamicMRNeRF",
    "NeuralAdvectionGenerator",
    # Hilbert-Mamba generators
    "CTMamba",
    "FEMamba",
    "CRMMamba",
    "HWMamba",
    "MMMamba",
    "MDIMamba",
    "TwoHalfDMamba",
    "HLDMamba",
    "INRMamba",
    "StructureTensorTransformer",
    "HilbertMambaUNet",
    "HDSFMambaUNet",
    "HDSFHLDMamba",
    "EvidentialUNet",
    "DiffeomorphicSynthesisNet",
    # Phase 1: Newly implemented generators
    "XDiffusionLatentGenerator",
    "EfficientUNetGenerator",
    "TissueDiffusionMRIGenerator",
    "TissueParameterDiffusion",
    "ULFCorruptionOperator",
    "HFCorruptionOperator",
    "PairedSynthesisGenerator",
    "SIRENGenerator",
    "ProbabilisticGraphicalGenerator",
    "NeuralCacheGenerator",
    "IsotonicConstrainedGenerator",
    "DifferentiableRenderingGenerator",
    "SymbolicRegressionGenerator",
    "CapsuleNetworkGenerator",
    "CausalDiscoveryGenerator",
    "UniversalMultitaskGenerator",
    "TemporalCoherence4DGenerator",
    "EnsembleDisagreementGenerator",
    "DINOTrellisGenerator",
    "ConditionalContinualVAEGenerator",
    "RLAcquisitionAgent",
    "SurgeryLoopAgent",
]

# =========================================================================
# DISABLED GENERATORS - Need import path repairs
# =========================================================================
# - physics_aware_gaussian_splatter: attempted relative import beyond top-level package
# - mamba_generator: No module named 'spectramr.models.implementations'
# - latent_gaussian_diffusion: No module named 'spectramr.models.generators.swin_kan_generator'
