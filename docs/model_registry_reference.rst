Model Registry Reference
========================

This page is a reference for models registered in the spectraMR framework.
Models are instantiated via the ``ModelFactory`` using the ``@register_model`` decorator.

.. note::

   **This page is hand-written and not exhaustive.** It documents 168 of the
   586 models the registry holds, so a name's absence here is **not** evidence
   that it does not exist -- check the registry before concluding one is
   unavailable. Counts are deliberately not restated in the prose above: a frozen
   number in a hand-maintained page drifts silently, and this one had -- it claimed to cover "all" of them.

   .. code-block:: python

      from spectramr.models.init_registry import populate_model_registry
      from spectramr.models.registry import MODEL_REGISTRY

      populate_model_registry()   # REQUIRED -- empty without it
      sorted(MODEL_REGISTRY)

.. contents:: Table of Contents
   :local:
   :depth: 2

Reconstruction Models
---------------------
Models with ``training_mode="reconstruction"``. These models typically map an input (e.g., zero-filled reconstruction or k-space) to a target image.

**Standard UNet**
   - **Registry Name:** ``standard_unet``
   - **Class:** ``spectramr.models.reconstruction.unet.StandardUNetGenerator``
   - **Description:** A robust U-Net implementation with configurable depth, attention block types, and normalization.

   .. code-block:: python

      def __init__(
          self,
          config: Optional[ModelConfigSchema] = None,
          in_channels: int = 1,
          out_channels: int = 1,
          features: tuple[int, ...] = (64, 128, 256, 512),
          depth: int = 4,
          bilinear: bool = False,
          attention_type: AttentionType = AttentionType.NONE,
          norm_type: NormalizationType = NormalizationType.INSTANCE,
          block_type: BlockType = BlockType.RESIDUAL,
          use_attention: bool = False,
          use_residual: bool = True,
          dropout: float = 0.0,
          deep_supervision: bool = False,
      ):

**Mamba (Foundation Model)**
   - **Registry Name:** ``mamba``
   - **Class:** ``spectramr.models.generators.foundation_model.FoundationModel``
   - **Description:** A state-space model (SSM) based architecture using Selective SSM (Mamba) blocks for efficient long-range dependency modeling.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          d_model: int = 64,
          num_layers: int = 4,
          patch_size: int = 4,
          text_embedding_dim: Optional[int] = None,
      ):

**NAFNet**
   - **Registry Name:** ``nafnet``
   - **Class:** ``spectramr.models.generators.nafnet_generator.NAFNetGenerator``
   - **Description:** Nonlinear Activation Free Network for Image Restoration. efficient and simpler than standard transformers.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          width: int = 32,
          enc_blk_nums: list[int] = [1, 1, 1, 28],
          middle_blk_num: int = 1,
          dec_blk_nums: list[int] = [1, 1, 1, 1],
          scale_factor: int = 1,
      ):

**Unrolled Reconstruction**
   - **Registry Name:** ``unrolled_reconstruction``
   - **Class:** ``spectramr.models.generators.unrolled_reconstruction_generator.UnrolledReconstructionGenerator``
   - **Description:** Physics-informed unrolled network alternating between data consistency and regularization.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 2,
          out_channels: int = 2,
          img_size: tuple[int, int] = (256, 256),
          num_unrolls: int = 5,
          features: int = 64,
          kernel_size: int = 3,
          lambda_dc: float = 0.1,
          num_coils: int = 1,
          activation: str = "relu",
      ):

**VarNet (Variational Network)**
   - **Registry Name:** ``varnet``
   - **Class:** ``spectramr.models.generators.unrolled_reconstruction_generator.VariationalNetworkGenerator``
   - **Description:** End-to-end variational network learning optimal gradient descent steps.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 2,
          out_channels: int = 2,
          img_size: tuple[int, int] = (256, 256),
          num_stages: int = 5,
          features: int = 64,
          step_size: float = 0.1,
          num_coils: int = 1,
      ):

**TRELLIS**
   - **Registry Name:** ``trellis``
   - **Class:** ``spectramr.models.generators.trellis_generator.TRELLISGenerator``
   - **Description:** Transformer-based Reconstruction and Learning with Linear Invariance and Sparsity.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          resolution: tuple[int, ...] = (64, 128, 128),
          coarse_resolution: tuple[int, ...] = (8, 16, 16),
          input_resolution: tuple[int, ...] = (64, 128, 128),
          patch_size: tuple[int, ...] = (2, 4, 4),
          stride: int = 4,
          num_views: int = 2,
          feature_dim: int = 256,
          num_layers: int = 6,
          num_heads: int = 8,
          mlp_ratio: int = 4,
          num_stages: int = 4,
      ):

**SwinIR**
   - **Registry Name:** ``swinir``
   - **Class:** ``spectramr.models.generators.swinir_generator.SwinIRGenerator``
   - **Description:** Image Restoration using Swin Transformer.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          upscale: int = 2,
          window_size: int = 7,
          img_size: int = 128,
          patch_size: int = 4,
          in_chans: int = 1,
          embed_dim: int = 60,
          depths: tuple[int, ...] = (6, 6, 6, 6),
          num_heads: tuple[int, ...] = (6, 6, 6, 6),
          mlp_ratio: float = 2.0,
          qkv_bias: bool = True,
          qk_scale: Optional[float] = None,
          drop_rate: float = 0.0,
          attn_drop_rate: float = 0.0,
          drop_path_rate: float = 0.1,
          norm_layer: nn.Module = nn.LayerNorm,
          patch_norm: bool = True,
      ):

**Restormer**
   - **Registry Name:** ``restormer``
   - **Class:** ``spectramr.models.generators.restormer_generator.RestormerGenerator``
   - **Description:** Efficient Transformer for High-Resolution Image Restoration using Multi-Dconv Head Transposed Attention.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          dim: int = 48,
          num_blocks: list[int] = [4, 6, 6, 8],
          num_refinement_blocks: int = 4,
          heads: list[int] = [1, 2, 4, 8],
          ffn_expansion_factor: float = 2.66,
          bias: bool = False,
          scale: int = 1,
      ):

**UNETR**
   - **Registry Name:** ``unetr``
   - **Class:** ``spectramr.models.generators.unetr_generator.UNETRGenerator``
   - **Description:** UNet Transformers for 3D Medical Image Segmentation (adapted for reconstruction).

   .. code-block:: python

      def __init__(
          self,
          in_channels: int,
          out_channels: int,
          img_size: tuple[int, int, int],
          feature_size: int = 16,
          hidden_size: int = 768,
          mlp_dim: int = 3072,
          num_heads: int = 12,
          proj_type: str = "conv",
          norm_name: Union[tuple, str] = "instance",
          conv_block: bool = True,
          res_block: bool = True,
          dropout_rate: float = 0.0,
          spatial_dims: int = 3,
          qkv_bias: bool = False,
          save_attn: bool = False,
      ):

**Slice-to-Volume**
   - **Registry Name:** ``slice_to_volume``
   - **Class:** ``spectramr.models.generators.slice_to_volume_generator.SliceToVolumeGenerator``
   - **Description:** Reconstructs a 3D volume from a stack of 2D slices.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          num_slices: int = 64,
          features: tuple[int, ...] = (64, 128, 256, 512),
      ):

**Bloch Cycle**
   - **Registry Name:** ``bloch_cycle``
   - **Class:** ``spectramr.models.generators.bloch_cycle_network.BlochCycleNetwork``
   - **Description:** Cycle-consistent network enforcing Bloch equation constraints.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          features: tuple[int, ...] = (64, 128, 256, 512),
          depth: int = 4,
          bilinear: bool = True,
          dropout: float = 0.0,
          normalization: str = "instance",
          activation: str = "relu",
      ):

**Coil Sensitivity Network**
   - **Registry Name:** ``coil_sensitivity``
   - **Class:** ``spectramr.models.generators.coil_sensitivity_network.CoilSensitivityNetwork``
   - **Description:** Estimates coil sensitivity maps from MRI data.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 2,
          out_channels: int = 2,
          features: tuple[int, ...] = (64, 128, 256, 512),
          depth: int = 4,
          bilinear: bool = True,
          dropout: float = 0.0,
          normalization: str = "instance",
          activation: str = "leaky",
          **kwargs,
      ):

**Vision Transformer (ViT)**
   - **Registry Name:** ``vision_transformer``
   - **Class:** ``spectramr.models.generators.vision_transformer.VisionTransformer``
   - **Description:** Standard Vision Transformer adapted for image reconstruction tasks.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          img_size: int = 256,
          patch_size: int = 16,
          embed_dim: int = 768,
          depth: int = 12,
          num_heads: int = 12,
          mlp_ratio: float = 4.0,
          dropout: float = 0.0,
          **kwargs,
      ):

**EDSR (Enhanced Deep Super-Resolution)**
   - **Registry Name:** ``edsr``
   - **Class:** ``spectramr.models.generators.edsr_generator.EDSRGenerator``
   - **Description:** Deep residual network optimized for super-resolution, removing batch normalization.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          num_features: int = 64,
          num_blocks: int = 16,
          scale_factor: int = 2,
          skip_scale: float = 1.0,
          output_activation: Optional[nn.Module] = None,
      ):

**UNet 2.5D**
   - **Registry Name:** ``unet_2_5d``
   - **Class:** ``spectramr.models.generators.unet_2_5d_generator.UNet2_5DGenerator``
   - **Description:** Processes stacked 2D slices to capture 3D spatial context efficiently.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          num_slices: int = 5,
          features: tuple[int, ...] = (64, 128, 256, 512, 1024),
          bilinear: bool = True,
          scale_factor: int = 4,
      ):

**Multi-Contrast Fusion**
   - **Registry Name:** ``multicontrast_fusion``
   - **Class:** ``spectramr.models.generators.multicontrast_fusion_generator.MultiContrastFusionGenerator``
   - **Description:** Fuses features from multiple MRI contrasts (e.g., T1, T2) using attention.

   .. code-block:: python

      def __init__(
          self,
          n_contrasts: int = 3,
          in_channels: int = 1,
          out_channels: int = 1,
          base_channels: int = 64,
          num_layers: int = 4,
          attention_heads: int = 8,
          dropout_rate: float = 0.1,
          use_residual: bool = True,
          use_cross_attention: bool = True,
          use_adaptive_fusion: bool = True,
          output_activation: Optional[str] = None,
      ):


**Deep Image Prior (DIP)**
   - **Registry Name:** ``deep_image_prior``
   - **Class:** ``spectramr.models.generators.deep_image_prior.DeepImagePriorGenerator``
   - **Description:** Reconstruction by optimizing network weights to map fixed noise to the observed k-space.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          input_noise_shape: tuple = (1, 32, 256, 256),
      ):

**Disentangled MRI**
   - **Registry Name:** ``disentangled_mri``
   - **Class:** ``spectramr.models.generators.disentangled_mri.DisentangledMRI``
   - **Description:** Separates anatomy (content) from contrast (style) for unpaired synthesis.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          dim: int = 64,
          style_dim: int = 8,
          n_downsample: int = 2,
          n_res: int = 4,
          anatomy_dim: int | None = None,
      ):

**PaD-Net (Physics-Informed Latent Volume)**
   - **Registry Name:** ``padnet``
   - **Class:** ``spectramr.models.generators.padnet.PaDNet``
   - **Description:** Combines multimodal encoding, conditional latent diffusion, and physics-based decoding.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          latent_dim: int = 32,
          diffusion_timesteps: int = 1000,
          q_map_channels: int = 3,
      ):

**HoBS (Holographic Bloch-Splatting)**
   - **Registry Name:** ``hobs``
   - **Class:** ``spectramr.models.generators.hobs_generator.HoBSGenerator``
   - **Description:** Generates k-space via biophysical Gaussian splatting and Bloch simulation.

   .. code-block:: python

      def __init__(
          self,
          num_gaussians: int = 1000,
          temp_dim: int = 1,
          in_channels: int = 2,
          out_channels: int = 2,
          _learnable_biophysics: bool = True,
      ):

**Physics-Driven Network**
   - **Registry Name:** ``physics_driven``
   - **Class:** ``spectramr.models.generators.physics_driven_network.PhysicsDrivenNetwork``
   - **Description:** Simultaneous reconstruction and quantitative mapping using differentiable physics.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 2,
          out_channels: int = 1,
          q_map_channels: int = 3,
          operator_type: str = "fft2d",
          te: float = 10.0,
          tr: float = 500.0,
      ):

**Fourier Neural Operator (FNO)**
   - **Registry Name:** ``fno``
   - **Class:** ``spectramr.models.generators.fno_generator.FNOGenerator``
   - **Description:** Resolution-invariant operator learning in the frequency domain.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 2,
          out_channels: int = 2,
          width: int = 64,
          modes1: int = 64,
          modes2: int = 64,
          depth: int = 4,
          output_domain: str = "image",
      ):

**PIMN (Physics-Informed Momentum Net)**
   - **Registry Name:** ``pimn``
   - **Class:** ``spectramr.models.generators.pimn.PIMN``
   - **Description:** Unrolled network treating reconstruction as a momentum-accelerated dynamic system.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 2,
          out_channels: int = 1,
          num_iterations: int = 10,
          operator_type: str = "fft2d",
      ):

**UNet 3D**
   - **Registry Name:** ``unet3d``
   - **Class:** ``spectramr.models.generators.unet3d_generator.UNet3DGenerator``
   - **Description:** Full 3D U-Net for volumetric reconstruction.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          features: tuple[int, ...] = (32, 64, 128, 256),
          trilinear: bool = True,
          output_activation: Optional[nn.Module] = None,
      ):

**Graph U-Net**
   - **Registry Name:** ``graph_unet``
   - **Class:** ``spectramr.models.generators.graph_unet_generator.GraphUNetGenerator``
   - **Description:** Graph CNN for non-Cartesian MRI, treating k-space samples as nodes.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 5,
          out_channels: int = 2,
          hidden_dim: int = 64,
          depth: int = 3,
          trajectory_points: torch.Tensor | None = None,
          k_neighbors: int = 8,
          im_size: tuple[int, int] = (256, 256),
      ):

**LaNS (Lagrangian Neuro-Splatting)**
   - **Registry Name:** ``lans``
   - **Class:** ``spectramr.models.generators.lans_generator.LaNSGenerator``
   - **Description:** Lagrangian dynamics for temporal MRI reconstruction using moving Gaussians.

   .. code-block:: python

      def __init__(
          self,
          num_gaussians: int = 1000,
          temp_dim: int = 20,
          in_channels: int = 2,
          out_channels: int = 2,
          siren_hidden_dim: int = 64,
          siren_layers: int = 3,
          dt: float = 0.1,
      ):

**RCAN (Residual Channel Attention)**
   - **Registry Name:** ``rcan``
   - **Class:** ``spectramr.models.generators.rcan_generator.RCANGenerator``
   - **Description:** Very deep residual channel attention network for super-resolution.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          num_features: int = 64,
          num_groups: int = 10,
          num_blocks: int = 20,
          reduction: int = 16,
          scale_factor: int = 2,
          skip_scale: float = 1.0,
      ):

**Physics-Informed UNet**
   - **Registry Name:** ``physics_informed_unet``
   - **Class:** ``spectramr.models.gans.standard_gan.PhysicsInformedUNet``
   - **Description:** Cascaded unrolled network alternating between U-Net refinement and data consistency.

   .. code-block:: python

      def __init__(
          self,
          base_unet: nn.Module,
          num_cascades: int = 5,
      ):

**K-Space GPT**
   - **Registry Name:** ``kspace_gpt``
   - **Class:** ``spectramr.models.generators.kspace_gpt.KSpaceGPT``
   - **Description:** Generative Pre-trained Transformer modeling k-space as a sequence of patches.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 2,
          out_channels: int = 2,
          d_model: int = 256,
          nhead: int = 8,
          num_layers: int = 6,
          dim_feedforward: int = 1024,
          img_size: int = 256,
          patch_size: int = 16,
      ):

**PINNeRF (Physics-Informed NeRF)**
   - **Registry Name:** ``pinn_nerf``
   - **Class:** ``spectramr.models.generators.pinn_nerf_generator.PINNNeRFGenerator``
   - **Description:** Generator using PINN-NeRF for continuous MRI reconstruction with dilated encoder.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          features: list[int] = [64, 128, 256],
          hidden_dim: int = 256,
          num_layers: int = 4,
          num_freqs: int = 10,
          output_domain: str = "image",
      ):

**DCSRN (Dense Channel Squeeze-and-Excitation)**
   - **Registry Name:** ``dcsrn``
   - **Class:** ``spectramr.models.generators.dcsrn_generator.DCSRNGenerator``
   - **Description:** 3D Super-Resolution network with dense connections and SE blocks.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          num_blocks: int = 16,
          growth_rate: int = 32,
          num_dense_layers: int = 4,
          scale_factor: int = 2,
          output_activation: Optional[nn.Module] = None,
      ):

**Efficient Transformer**
   - **Registry Name:** ``efficient_transformer``
   - **Class:** ``spectramr.models.generators.efficient_transformer.EfficientTransformerGeneratorWrapper``
   - **Description:** Memory-efficient transformer with LoRA, gradient checkpointing, and linear attention.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          embed_dim: int = 768,
          depth: int = 12,
          num_heads: int = 12,
          use_lora: bool = True,
          use_linear_attn: bool = False,
          **kwargs,
      ):

**ELAN (Efficient Long-range Attention Network)**
   - **Registry Name:** ``elan``
   - **Class:** ``spectramr.models.generators.elan_generator.ELANGenerator``
   - **Description:** Efficient Long-range Attention Network for image super-resolution.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 3,
          out_channels: int = 3,
          n_feat: int = 64,
          n_blocks: int = 8,
          scale: int = 4,
          bias: bool = False,
      ):

**MK-Recon (Mamba-KAN Hybrid)**
   - **Registry Name:** ``mk_recon``
   - **Class:** ``spectramr.models.experimental.mk_recon.MKRecon``
   - **Description:** Hybrid architecture combining K-Space Mamba (global) and Image-Space KAN (physics) for MRI reconstruction.

   .. code-block:: python

      def __init__(
          self,
          in_channels=2,
          out_channels=1,
          seq_len=256 * 256,
          embed_dim=64,
          mamba_depth=4,
          kan_depth=4,
          img_size=256,
      ):

**ZeroRF Reconstructor**
   - **Registry Name:** ``zerorf``
   - **Class:** ``spectramr.models.reconstruction.zerorf.ZeroRFReconstructor``
   - **Description:** Zero-Shot Neural Radiance Fields approach using TensorVM decomposition.

   .. code-block:: python

      def __init__(
          self,
          volume_shape: Tuple[int, int, int],
          rank: int = 16,
          latent_dim: int = 64
      ):

**Swin Transformer KAN**
   - **Registry Name:** ``swin_kan_generator``
   - **Class:** ``spectramr.models.gans.swin_kan_gan.SwinKANGenerator``
   - **Description:** Swin Transformer variant using KAN layers instead of MLPs.

   .. code-block:: python

      def __init__(
          self,
          img_size=240,
          patch_size=4,
          in_chans=1,
          num_classes=1,
          embed_dim=96,
          depths=None,
          num_heads=None,
          window_size=7,
          mlp_ratio=4.0,
          qkv_bias=True,
          qk_scale=None,
          drop_rate=0.0,
          attn_drop_rate=0.0,
          norm_layer=nn.LayerNorm,
          **kwargs,
      ):

**Swin Transformer U-Net**
   - **Registry Name:** ``transformer_unet``
   - **Class:** ``spectramr.models.generators.swin_transformer_generator.SwinTransformerGenerator``
   - **Description:** Standard Swin Transformer U-Net for image reconstruction.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          img_size: int = 256,
          patch_size: int = 4,
          embed_dim: int = 96,
          depths: tuple[int, ...] = (2, 2, 6, 2),
          num_heads: tuple[int, ...] = (3, 6, 12, 24),
          window_size: int = 7,
          mlp_ratio: float = 4.0,
          qkv_bias: bool = True,
          drop_rate: float = 0.0,
          attn_drop_rate: float = 0.0,
          **kwargs,
      ):

**CycleSR Generator**
   - **Registry Name:** ``cyclesr``
   - **Class:** ``spectramr.models.generators.cyclesr_generator.CycleSRGenerator``
   - **Description:** Cycle-consistent Super-Resolution generator.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 3,
          out_channels: int = 3,
          n_feat: int = 64,
          n_resblocks: int = 16,
          scale: int = 4,
          bias: bool = False,
      ):

      ):

**Anisotropic Voxel Generator**
   - **Registry Name:** ``anisotropic_voxel_generator``
   - **Class:** ``spectramr.models.generators.anisotropic_voxel_generator.AnisotropicVoxelGenerator``
   - **Description:** Hybrid 2D/3D generator for anisotropic voxel handling.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          base_channels: int = 64,
          num_levels: int = 4,
          anisotropic_factor: float = 5.0,
          dropout: float = 0.1,
      ):

**Digital Twin Simulator**
   - **Registry Name:** ``digital_twin_simulator``
   - **Class:** ``spectramr.models.generators.digital_twin_simulator.DigitalTwinSimulator``
   - **Description:** Digital Twin for patient-specific MRI simulation.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          patient_embedding_dim: int = 128,
          protocol_dim: int = 16,
          hidden_dim: int = 256,
      ):

**GNN Coil Fusion**
   - **Registry Name:** ``gnn_coil_fusion``
   - **Class:** ``spectramr.models.generators.gnn_coil_fusion.GNNCoilFusion``
   - **Description:** Graph Neural Network for Coil Fusion in MRI.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 2,
          out_channels: int = 2,
          hidden_dim: int = 64,
          depth: int = 3,
          **kwargs,
      ):

**Split Learning (Client)**
   - **Registry Name:** ``split_client_encoder``
   - **Class:** ``spectramr.models.generators.split_learning_unet.SplitClientEncoder``
   - **Description:** Client-side Encoder for Federated Split Learning.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          base_channels: int = 64
      ):

**Split Learning (Server)**
   - **Registry Name:** ``split_server_decoder``
   - **Class:** ``spectramr.models.generators.split_learning_unet.SplitServerDecoder``
   - **Description:** Server-side Decoder for Federated Split Learning.

   .. code-block:: python

      def __init__(
          self,
          out_channels=1,
          base_channels=64
      ):

      ):

**Multi-Contrast Generator**
   - **Registry Name:** ``multi_contrast_generator``
   - **Class:** ``spectramr.models.generators.multi_contrast_conditioning.MultiContrastGenerator``
   - **Description:** Multi-contrast conditional generator.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          num_aux_contrasts: int = 2,
          base_channels: int = 64,
          num_residual_blocks: int = 16,
          **kwargs,
      ):

**Reflexion Reconstruction**
   - **Registry Name:** ``reflexion_reconstruction``
   - **Class:** ``spectramr.models.generators.reflexion_reconstruction.ReflexionReconstruction``
   - **Description:** "System 2" Reflexion Reconstruction with iterative refinement.

   .. code-block:: python

      def __init__(
          self,
          base_model_type: str = "standard_unet",
          base_model_kwargs: Dict[str, Any] = None,
          critique_threshold: float = 0.05,
          max_refinement_steps: int = 5,
          step_size: float = 0.1,
          in_channels: int = 1,
          out_channels: int = 1,
      ):

**Scanner Invariant Embedding**
   - **Registry Name:** ``scanner_invariant_embedding``
   - **Class:** ``spectramr.models.generators.scanner_invariant_embedding.ScannerInvariantEmbedding``
   - **Description:** Generator with scanner-invariant feature learning.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          base_features: int = 64,
          embedding_dim: int = 256,
          num_domains: int = 3,
      ):

      ):

**Scanner Adaptive Generator**
   - **Registry Name:** ``scanner_adaptive_generator``
   - **Class:** ``spectramr.models.generators.scanner_shift_adaptation.ScannerAdaptiveGenerator``
   - **Description:** Generator with test-time scanner adaptation capabilities.

   .. code-block:: python

      def __init__(
          self,
          base_generator: IGenerator,
          adaptation_config: Optional[dict[str, Any]] = None,
      ):

**Continual Learning U-Net**
   - **Registry Name:** ``continual_learning_unet``
   - **Class:** ``spectramr.models.generators.continual_learning_unet.ContinualLearningUNet``
   - **Description:** U-Net with Elastic Weight Consolidation (EWC) support.

   .. code-block:: python

      def __init__(self, in_channels: int = 1, out_channels: int = 1):

**q-Space Translation**
   - **Registry Name:** ``qspace_translation``
   - **Class:** ``spectramr.models.generators.qspace_translation.qSpaceTranslation``
   - **Description:** q-Space Translation Model for physics-faithful upscaling.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          q_map_channels: int = 2,
          b0_low: float = 0.064,
          b0_high: float = 3.0,
          beta: float = 0.4,
      ):

      ):

**Symbolic Regression Wrapper**
   - **Registry Name:** ``symbolic_regression_wrapper``
   - **Class:** ``spectramr.models.generators.symbolic_regression_wrapper.SymbolicRegressionWrapper``
   - **Description:** Wrapper for analyzing residuals using Symbolic Regression.

   .. code-block:: python

      def __init__(
          self,
          base_model_type: str = "standard_unet",
          base_model_kwargs: Dict[str, Any] = None,
          in_channels: int = 1,
          out_channels: int = 1,
      ):

**CPT-4DMR Generator**
   - **Registry Name:** ``cpt_4dmr``
   - **Class:** ``spectramr.models.generators.cpt_4dmr.cpt_4dmr_generator.CPT4DMRGenerator``
   - **Description:** Continuous Spatio-Temporal 4D MRI Generator using SAN and TMN.

   .. code-block:: python

      def __init__(
          self,
          spatial_dim: int = 2,
          san_hidden_dim: int = 256,
          san_num_layers: int = 4,
          tmn_hidden_dim: int = 128,
          tmn_num_layers: int = 4,
          out_channels: int = 1,
          use_respiratory: bool = True,
          max_displacement: float = 0.3,
          jacobian_weight: float = 0.1,
          **kwargs,
      ):

**Gaussian Splatting Reconstructor**
   - **Registry Name:** ``gs_mri``
   - **Class:** ``spectramr.models.gaussian_splatting.gs_mri_reconstructor.GSMRReconstructor``
   - **Description:** End-to-End Reconstructor using 3D Gaussian Splatting.

   .. code-block:: python

      def __init__(
          self,
          volume_shape: Tuple[int, int, int] = (128, 128, 128),
          num_gaussians: int = 1000,
          init_scale: float = 0.1,
          adaptive_hparams: Optional[Dict] = None,
      ):

**Latent Flow Generator**
   - **Registry Name:** ``latent_flow``
   - **Class:** ``spectramr.models.generators.latent_flow_generator.LatentFlowGenerator``
   - **Description:** Normalizing flow model for latent space transformations.

   .. code-block:: python

      def __init__(
          self,
          latent_dim: int = 256,
          in_channels: int = 1,
          flow_layers: int = 8,
          hidden_dim: int = 512,
          conditioning_dim: Optional[int] = None,
      ):


Diffusion Models
----------------
Models with ``training_mode="diffusion"``. These are typically denoising networks or wrappers.

**Conditional Diffusion**
   - **Registry Name:** ``conditional_diffusion``
   - **Class:** ``spectramr.models.generators.conditional_diffusion_generator.ConditionalDiffusionGenerator``
   - **Description:** DDPM/DDIM implementation with flexible conditioning support.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          base_channels: int = 64,
          channel_mults: tuple[int, ...] = (1, 2, 4, 8),
          time_emb_dim: int = 256,
          conditioning_dim: Optional[int] = None,
          num_res_blocks: int = 2,
          noise_schedule: str = "linear",
          **kwargs,
      ):

**Cold Diffusion**
   - **Registry Name:** ``cold_diffusion``
   - **Class:** ``spectramr.models.generators.cold_diffusion_generator.ColdDiffusionGenerator``
   - **Description:** Deterministic diffusion-like process operating on arbitrary degradations (blur, downsampling, etc.).

   .. code-block:: python

      def __init__(
          self,
          denoising_model: nn.Module,
          timesteps: int = 100,
          beta_schedule: str = 'linear',
          degradation_type: str = 'deblurring',
          cold_schedule: str = 'linear',
          restoration_steps: int = 10,
      ):

**K-Space Cold Diffusion**
   - **Registry Name:** ``kspace_cold_diffusion``
   - **Class:** ``spectramr.models.generators.kspace_cold_diffusion_generator.KSpaceColdDiffusionGenerator``
   - **Description:** Cold diffusion applied directly in the frequency domain (k-space).

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 2,
          out_channels: int = 2,
          base_channels: int = 64,
          num_layers: int = 4,
          attention_type: str = "none",
          num_timesteps: int = 1000,
          time_embedding_dim: int = 128,
      ):

**Chi-Square Diffusion**
   - **Registry Name:** ``chi_square_diffusion``
   - **Class:** ``spectramr.models.generators.chi_square_diffusion_generator.ChiSquareDiffusionGenerator``
   - **Description:** Diffusion model using Chi-Square noise distributions, suitable for MRI magnitude data.

   .. code-block:: python

      def __init__(
          self,
          denoising_model: nn.Module,
          in_channels: int = 1,
          hidden_channels: int = 64,
          image_size: int = 256,
          num_inference_steps: int = 1000,
          timesteps: int = 1000,
          beta_schedule: str = "linear",
          device: str = "cuda",
          min_dof: float = 2.0,
          max_dof: float = 10.0,
          dof_schedule: str = "linear",
          gaussian_approximation_threshold: int = 30,
      ):

**Hybrid Transformer Diffusion**
   - **Registry Name:** ``hybrid_transformer_diffusion``
   - **Class:** ``spectramr.models.generators.hybrid_transformer_diffusion_generator.HybridTransformerDiffusionGenerator``
   - **Description:** Combines global (ViT) and local (Swin) attention mechanisms within a diffusion framework.

   .. code-block:: python

      def __init__(
          self,
          in_channels=1,
          out_channels=1,
          time_dim=256,
          image_size=256,
          patch_size=16,
          dim=512,
          depth=6,
          heads=8,
          mlp_dim=1024,
          window_size=8,
          dropout=0.1,
          bilinear=False,
          layer_norm_strategy="pre",
          layer_norm_eps=1e-6,
          T=1000,
          beta_start=1e-4,
          beta_end=2e-2,
      ):


      ):

**Diffusion Reconstruction (PnP/RED)**
   - **Registry Name:** ``diffusion_reconstruction``
   - **Class:** ``spectramr.models.diffusion.diffusion_reconstruction.ReconstructionWithDiffusionPrior``
   - **Description:** High-level wrapper for PnP/RED/Posterior Sampling using a diffusion prior.

   .. code-block:: python

      def __init__(
          self,
          method: str = "pnp",
          diffusion_model: Optional[nn.Module] = None,
          noise_scheduler: Optional[Callable] = None,
          forward_operator: Optional[Callable] = None,
          **kwargs,
      ):

**Rician Diffusion**
   - **Registry Name:** ``rician_diffusion``
   - **Class:** ``spectramr.models.generators.rician_diffusion_generator.RicianDiffusionGenerator``
   - **Description:** Diffusion model tailored for Rician noise distributions in magnitude MRI.

   .. code-block:: python

      def __init__(
          self,
          denoising_model: nn.Module,
          timesteps: int = 1000,
          beta_schedule: str = "linear",
          device: Optional[str] = None,
          v_min: float = 0.1,
          v_max: float = 10.0,
          v_schedule: str = "linear",
      ):

**Consistency Model**
   - **Registry Name:** ``consistency_model``
   - **Class:** ``spectramr.models.generators.consistency_model_generator.ConsistencyModelGenerator``
   - **Description:** Fast 1-2 step generation by mapping trajectory points to origin.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 2,
          out_channels: int = 2,
          features: list[int] = [64, 128, 256, 512],
          time_dim: int = 256,
          sigma_min: float = 0.002,
          sigma_max: float = 80.0,
          rho: float = 7.0,
          distillation_steps: int = 2,
      ):

**Latent Diffusion**
   - **Registry Name:** ``latent_diffusion``
   - **Class:** ``spectramr.models.generators.latent_diffusion_generator.LatentDiffusionGenerator``
   - **Description:** Diffusion in compressed latent space for high-resolution synthesis.

   .. code-block:: python

      def __init__(
          self,
          config: Optional[LatentDiffusionGeneratorConfig] = None,
          in_channels: Optional[int] = None,
          out_channels: Optional[int] = None,
          latent_channels: Optional[int] = None,
          base_channels: Optional[int] = None,
          timesteps: Optional[int] = None,
          beta_schedule: Optional[str] = None,
          device: Optional[str] = None,
          spatial_dims: Optional[int] = None,
          **kwargs,
      ):


**Enhanced Deep Diffusion U-Net**
   - **Registry Name:** ``enhanced_deep_unet``
   - **Class:** ``spectramr.models.diffusion.architectures.enhanced_deep_unet.EnhancedDeepDiffusionUNet``
   - **Description:** U-Net variant with time embedding and optional complex convolutions for diffusion.

   .. code-block:: python

      def __init__(
          self,
          in_channels,
          out_channels,
          time_dim,
          bilinear=False,
          use_complex_conv=False,
          **kwargs,
      ):

**Rectified Flow**
   - **Registry Name:** ``rectified_flow``
   - **Class:** ``spectramr.models.generators.rectified_flow_generator.RectifiedFlowGenerator``
   - **Description:** Measurement-conditional flow matching for straight-line trajectory generation.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 2,
          out_channels: int = 2,
          features: list[int] = [64, 128, 256, 512],
          context_dim: int = 512,
          flow_steps: int = 10,
      ):


**KAN U-Net**
   - **Registry Name:** ``kan_unet``
   - **Class:** ``spectramr.models.diffusion.architectures.kan_unet.RefinedKANUNet``
   - **Description:** U-Net using KAN blocks in the bottleneck for enhanced semantic feature processing.

   .. code-block:: python

      def __init__(
          self,
          in_channels,
          out_channels,
          base_dim=32,
          time_dim=128,
          small_kan=False,
          cond_channels=0,
          bilinear=False,
      ):

**Swin Hybrid U-Net**
   - **Registry Name:** ``swin_hybrid_unet``
   - **Class:** ``spectramr.models.diffusion.architectures.transformer_hybrid_unets.SwinHybridUNet``
   - **Description:** Hybrid U-Net with Swin Transformer path for time-conditional score prediction.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          time_dim: int = 256,
          cond_channels: int = 0,
          image_size: int = 256,
          patch_size: int = 4,
          embed_dim: int = 96,
          depths: list[int] = None,
          num_heads: list[int] = None,
          window_size: int = 7,
          mlp_ratio: float = 4.0,
          drop_rate: float = 0.0,
          attn_drop_rate: float = 0.0,
          drop_path_rate: float = 0.1,
          layer_norm_strategy: str = "pre",
          layer_norm_eps: float = 1e-6,
          use_enhanced_layer_norm: bool = True,
          bilinear: bool = False,
          **kwargs,
      ):

**ViT Hybrid U-Net**
   - **Registry Name:** ``vit_hybrid_unet``
   - **Class:** ``spectramr.models.diffusion.architectures.transformer_hybrid_unets.ViTHybridUNet``
   - **Description:** Hybrid U-Net with Vision Transformer (ViT) path.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          time_dim: int = 256,
          cond_channels: int = 0,
          image_size: int = 256,
          patch_size: int = 16,
          embed_dim: int = 768,
          depth: int = 12,
          num_heads: int = 12,
          mlp_ratio: float = 4.0,
          drop_rate: float = 0.1,
          bilinear: bool = False,
          **kwargs,
      ):


**Laplace Diffusion**
   - **Registry Name:** ``laplace_diffusion``
   - **Class:** ``spectramr.models.diffusion.laplace_diffusion.LaplaceDiffusion``
   - **Description:** Diffusion process using Laplace noise distributions.

   .. code-block:: python

      def __init__(
          self,
          timesteps: int = 1000,
          beta_schedule: str = "linear",
          device: str = None,
      ):

**Score-Based Diffusion**
   - **Registry Name:** ``score_based_diffusion``
   - **Class:** ``spectramr.models.generators.score_based_diffusion_generator.ScoreBasedDiffusionGenerator``
   - **Description:** Continuous-time score-based generative modeling (SDE).

   .. code-block:: python

      def __init__(
          self,
          denoising_model: nn.Module,
          timesteps: int = 1000,
          beta_schedule: str = "linear",
          device: Optional[str] = None,
          gradient_lambda: float = 1.0,
          entropy_lambda: float = 0.1,
          ssim_lambda: float = 0.5,
      ):

**X-Diffusion (Cross-Modal)**
   - **Registry Name:** ``x_diffusion``
   - **Class:** ``spectramr.models.generators.x_diffusion_generator.XDiffusionGenerator``
   - **Description:** Cross-modal generation (2D <-> 3D) using diffusion.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          resolution_2d: tuple[int, int] = (256, 256),
          resolution_3d: tuple[int, int, int] = (64, 128, 128),
          diffusion_steps: int = 1000,
          cross_attention_dim: int = 512,
          conditioning_dim: int = 768,
          num_layers: int = 8,
          num_heads: int = 8,
          diffusion_process: Any = None,
      ):

**Stable Diffusion Adapter**
   - **Registry Name:** ``stable_diffusion_adapter``
   - **Class:** ``spectramr.models.generators.stable_diffusion_adapter_generator.StableDiffusionAdapterGenerator``
   - **Description:** Adapter wrapper for leveraging pre-trained Stable Diffusion models.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 4,
          out_channels: int = 4,
          _base_model_path: str = None,
          adapter_type: str = None,
          ckpt_gradients: bool = False,
          image_size: int = 256,
          hidden_dim: int = 320,
      ):

**Laplace Diffusion**

**Cascaded Diffusion**
   - **Registry Name:** ``cascaded_diffusion``
   - **Class:** ``spectramr.models.generators.cascaded_diffusion_generator.CascadedDiffusionGenerator``
   - **Description:** Wrapper for standard U-Net to support cascaded diffusion (e.g., low-res conditioning).

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          features: tuple[int, ...] = (64, 128, 256, 512),
          low_res_conditioning: bool = False,
          **kwargs,
      ):


VAE Models
----------
Models with ``training_mode="vae"``.

**Standard VAE**
   - **Registry Name:** ``vae``
   - **Class:** ``spectramr.models.vae.vae.VAE``
   - **Description:** Standard Convolutional Variational Autoencoder.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int,
          out_channels: int,
          latent_dim: int,
          hidden_dims: list = None,
          use_batch_norm: bool = True,
          dropout_rate: float = 0.0,
          **kwargs,
      ):

**VQ-VAE (Vector Quantized VAE)**
   - **Registry Name:** ``vqvae``
   - **Class:** ``spectramr.models.vq.vqvae.VQVAE``
   - **Description:** VAE with discrete latent codes using vector quantization.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int,
          out_channels: int,
          hiddein_channels: int,
          num_embeddings: int,
          embedding_dim: int,
          commitment_cost: float = 0.25,
          num_layers: int = 2,
          downsample_factor: int = 4,
          **kwargs,
      ):

**KAN-VAE**
   - **Registry Name:** ``kan_vae``
   - **Class:** ``spectramr.models.generators.kan_vae_generator.KANVAEGenerator``
   - **Description:** VAE utilizing Kolmogorov-Arnold Networks (KAN) for dense symbolic feature learning.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          latent_dim: int = 128,
          kl_weight: float = 0.001,
          hidden_dims: tuple[int, ...] = (32, 64, 128),
          kan_hidden_dim: int = 64,
          **kwargs,
      ):

**Robust VAE**
   - **Registry Name:** ``robust_vae``
   - **Class:** ``spectramr.models.generators.robust_vae_generator.RobustVAEGenerator``
   - **Description:** VAE with enhanced stability features (batch norm, dropout, KL annealing).

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          latent_dim: int = 256,
          hidden_dims: Optional[list] = None,
          beta: float = 1.0,
      ):

**Sparse VAE**
   - **Registry Name:** ``sparse_vae``
   - **Class:** ``spectramr.models.generators.sparse_vae_generator.SparseVAEGenerator``
   - **Description:** VAE with learned sparsity masks for compact latent representations.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          latent_dim: int = 256,
          hidden_dims: tuple[int, ...] = (32, 64, 128, 256),
          beta_kl: float = 0.001,
          sparsity_lambda: float = 0.001,
          sparsity_target: float = 0.05,
          use_batch_norm: bool = True,
          dropout_rate: float = 0.0,
          **kwargs,
      ):


**Fourier-KAN VAE**
   - **Registry Name:** ``fourier_kan_vae``
   - **Class:** ``spectramr.models.vae.fourier_kan_vae.FourierKANVAE``
   - **Description:** VAE using Fourier-Kolmogorov-Arnold Networks for frequency-domain latent modeling.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          latent_dim: int = 256,
          num_bases: int = 8,
          max_frequency: float = 1.0,
          basis_trainable: bool = True,
          input_size: int = 64,
          use_complex_conv: bool = False,
      ):

**Wavelet-KAN VAE**
   - **Registry Name:** ``wavelet_kan_vae``
   - **Class:** ``spectramr.models.vae.wavelet_kan_vae.WaveletKANVAE``
   - **Description:** VAE using Wavelet-KAN layers for multi-scale feature learning.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          latent_dim: int = 256,
          num_bases: int = 8,
          wavelet_family: str = "haar",
          basis_trainable: bool = True,
          input_size: int = 64,
      ):

**Configurable VAE**
   - **Registry Name:** ``configurable_vae``
   - **Class:** ``spectramr.models.latent.configurable_vae.ConfigurableVAE``
   - **Description:** Highly modular VAE with presets (simple, resnet, attention, deep).

   .. code-block:: python

      def __init__(
          self,
          encoder_config: Optional[EncoderConfig] = None,
          decoder_config: Optional[DecoderConfig] = None,
          latent_config: Optional[LatentConfig] = None,
          beta: float = 1.0,
      ):

**TRELLIS Gaussian VAE**
   - **Registry Name:** ``trellis_gaussian_vae``
   - **Class:** ``spectramr.models.generators.trellis_structured_vae.TrellisStructuredGaussianVAEGenerator``
   - **Description:** Weighted Gaussian structured latent VAE from TRELLIS.

**TRELLIS Mesh VAE**
   - **Registry Name:** ``trellis_mesh_vae``
   - **Class:** ``spectramr.models.generators.trellis_structured_vae.TrellisStructuredMeshVAEGenerator``
   - **Description:** Mesh structured latent VAE from TRELLIS.

**3D VAE**
   - **Registry Name:** ``vae_3d``
   - **Class:** ``spectramr.models.generators.vae_3d_generator.VAE3DGenerator``
   - **Description:** 3D Variational Autoencoder optimized for volumetric MRI data.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          latent_dim: int = 512,
          hidden_dims: Optional[list] = None,
          beta: float = 1.0,
          input_shape: tuple[int, int, int] = (64, 64, 64),
          **kwargs,
      ):

GAN Models
----------
Models with ``training_mode="gan"``.


**Latent Discriminator**
   - **Registry Name:** ``latent_discriminator``
   - **Class:** ``spectramr.models.latent_diffusion.latent_discriminator.LatentDiscriminator``
   - **Description:** Discriminator for latent representations (vector or image).

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 4,
          base_channels: int = 64,
          channel_mult: tuple[int, ...] = (1, 2, 4, 8),
          num_layers: int = 4,
          use_spectral_norm: bool = True,
          dropout: float = 0.1,
          input_type: str = "image",
      ):

**Patch Latent Discriminator**
   - **Registry Name:** ``patch_latent_discriminator``
   - **Class:** ``spectramr.models.latent_diffusion.latent_discriminator.PatchLatentDiscriminator``
   - **Description:** Patch-based discriminator for latent space.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 4,
          base_channels: int = 64,
          use_spectral_norm: bool = True,
          dropout: float = 0.1,
      ):

**Multi-Scale Latent Discriminator**
   - **Registry Name:** ``multiscale_latent_discriminator``
   - **Class:** ``spectramr.models.latent_diffusion.latent_discriminator.MultiScaleLatentDiscriminator``
   - **Description:** Multi-scale discriminator for latent space.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 4,
          base_channels: int = 64,
          num_scales: int = 3,
          use_spectral_norm: bool = True,
      ):
**CycleGAN Generator (ResNet)**
   - **Registry Name:** ``cyclegan_generator``
   - **Class:** ``spectramr.models.generators.cycle_gan.ResNetGenerator``
   - **Description:** ResNet-based generator used in CycleGAN for unpaired translation.

   .. code-block:: python

      def __init__(
          self,
          input_nc: int = 1,
          output_nc: int = 1,
          ngf: int = 64,
          n_blocks: int = 6,
      ):

**VQ-GAN**
   - **Registry Name:** ``vqgan``
   - **Class:** ``spectramr.models.vq.vqvae.VQGAN``
   - **Description:** VQ-VAE coupled with a discriminator and perceptual loss for high-fidelity generation.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int,
          out_channels: int,
          hiddein_channels: int,
          num_embeddings: int,
          embedding_dim: int,
          commitment_cost: float = 0.25,
          num_layers: int = 2,
          downsample_factor: int = 4,
          disc_channels: int = 64,
          disc_layers: int = 3,
          use_spectral_norm: bool = True,
          **kwargs,
      ):

**Latent GAN**
   - **Registry Name:** ``latent_gan``
   - **Class:** ``spectramr.models.latent_gan.generator.LatentGANGenerator``
   - **Description:** Generator operating in the latent space of a pretrained VAE/VQ-VAE.

   .. code-block:: python

      def __init__(
          self,
          out_channels: int,
          latent_dim: int,
          hidden_dims: list = [256, 512, 1024],
          use_batch_norm: bool = True,
          dropout_rate: float = 0.0,
          **kwargs,
      ):

**Wasserstein Discriminator**
   - **Registry Name:** ``wasserstein_discriminator``
   - **Class:** ``spectramr.models.adaptation.transport.WassersteinDiscriminator``
   - **Description:** Critic network for WGAN-GP (Wasserstein GAN with Gradient Penalty).

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          ndf: int = 64
      ):

**Clifford GAN**
   - **Registry Name:** ``clifford_gan``
   - **Class:** ``spectramr.models.experimental.clifford_gan.CliffordGANGenerator``
   - **Description:** Geometric Algebra based GAN preserving phase/spin physics using Clifford convolutions.

   .. code-block:: python

      def __init__(
          self,
          in_channels=2,
          out_channels=2,
          base_dim=64
      ):

**Hyperspectral GAN**
   - **Registry Name:** ``hyperspectral_gan``
   - **Class:** ``spectramr.models.generators.hyperspectral_gan.HyperspectralGAN``
   - **Description:** Multi-contrast GAN generating T1/T2/FLAIR simultaneously with spectral attention.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          num_contrasts: int = 3,
          base_features: int = 64,
      ):

**KAN GAN Generator**
   - **Registry Name:** ``kan_gan``
   - **Class:** ``spectramr.models.gans.kan_gan.KANGenerator``
   - **Description:** Generator based on Kolmogorov-Arnold Networks (KAN) U-Net.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          bilinear: bool = False,
          norm_layer: type[nn.Module] = nn.InstanceNorm2d,
          opt: dict | None = None,
          **kwargs,
      ):

**KAN Discriminator**
   - **Registry Name:** ``kan_discriminator``
   - **Class:** ``spectramr.models.gans.kan_gan.KANDiscriminator``
   - **Description:** PatchGAN discriminator wrapper for KAN-GAN.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          **kwargs,
      ):

**Latent GAN Encoder**
   - **Registry Name:** ``latent_gan_encoder``
   - **Class:** ``spectramr.models.latent_gan.encoder.LatentGANEncoder``
   - **Description:** Deterministic encoder maps low-res inputs to latent space for Latent GAN.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          latent_dim: int = 128,
          hidden_dims: tuple[int, ...] = (64, 128, 256, 512),
          use_batch_norm: bool = True,
          dropout_rate: float = 0.0,
      ):

**StyleGAN2 Generator**
   - **Registry Name:** ``stylegan2``
   - **Class:** ``spectramr.models.gans.stylegan_variants.StyleGAN2Generator``
   - **Description:** StyleGAN2 implementation following SOLID principles.

   .. code-block:: python

      def __init__(
          self,
          z_dim: int,
          style_dim: int,
          n_mlp: int,
          out_size: int,
          out_channels: int = 1
      ):

**StyleGAN KAN Generator**
   - **Registry Name:** ``stylegan_kan_gen``
   - **Class:** ``spectramr.models.gans.stylegan_variants.StyleGANKANGenerator``
   - **Description:** StyleGAN2 using KAN layers in the mapping network.

   .. code-block:: python

      def __init__(
          self,
          z_dim: int,
          style_dim: int,
          n_mlp: int,
          out_size: int,
          out_channels: int = 1
      ):

**StyleGAN Discriminator**
   - **Registry Name:** ``stylegan_discriminator``
   - **Class:** ``spectramr.models.gans.stylegan_variants.StyleGANDiscriminator``
   - **Description:** Discriminator tailored for StyleGAN generation.

   .. code-block:: python

      def __init__(
          self,
          in_size: int,
          in_channels: int = 1
      ):


Self-Supervised Learning
------------------------
Models with ``training_mode="ssl"``.

**Masked Autoencoder (MAE)**
   - **Registry Name:** ``mae_mri``
   - **Class:** ``spectramr.models.mae.mae_generator.MAEGenerator``
   - **Description:** Masked Autoencoder for MRI with 3D patch embedding.

   .. code-block:: python

      def __init__(
          self,
          img_size: tuple[int, int, int] = (64, 64, 64),
          patch_size: tuple[int, int, int] = (16, 16, 16),
          in_channels: int = 1,
          embed_dim: int = 768,
          encoder_depth: int = 12,
          decoder_depth: int = 8,
          num_heads: int = 12,
          mlp_ratio: float = 4.0,
          mask_ratio: float = 0.75,
      ):

Domain Adaptation
-----------------
Models with ``training_mode="ssl"``.

**Domain Discriminator**
   - **Registry Name:** ``domain_discriminator``
   - **Class:** ``spectramr.models.domain_adaptation.DomainDiscriminator``
   - **Description:** Simple discriminator for distinguishing between domains (e.g., scanner vendors).

   .. code-block:: python

      def __init__(
          self,
          in_channels: int,
          num_domains: int = 2,
          hiddein_channels: int = 512,
      ):

Uncertainty Wrappers
--------------------
Models/Wrappers for uncertainty quantification.

**MC Dropout Generator**
   - **Registry Name:** ``mc_dropout``
   - **Class:** ``spectramr.models.generators.uncertainty_wrappers.MCDropoutGenerator``
   - **Description:** Wrapper that adds Monte Carlo Dropout to any generator for epistemic uncertainty estimation.

   .. code-block:: python

      def __init__(
          self,
          base_generator: IGenerator,
          dropout_p: float = 0.1,
      ):

**Deep Ensemble Generator**
   - **Registry Name:** ``deep_ensemble``
   - **Class:** ``spectramr.models.generators.uncertainty_wrappers.DeepEnsembleGenerator``
   - **Description:** Ensemble of multiple generator instances for robust uncertainty estimation.

   .. code-block:: python

      def __init__(
          self,
          generator_factory: Callable[..., nn.Module],
          n_ensemble: int = 5,
          generator_params: Optional[dict[str, Any]] = None,
      ):


Encoders
--------
Models with ``training_mode="encoder"``. These are used for feature extraction, perceptual losses, or as backbones.

**Medical DINOv2**
   - **Registry Name:** ``medical_dino``
   - **Class:** ``spectramr.models.encoders.medical_dino_encoder.MedicalDINOv2Encoder``
   - **Description:** DINOv2 Vision Transformer adapted for single-channel medical imaging.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          model_name: str = "vit_base_patch14_dinov2",
          img_size: int = 224,
          patch_size: int = 14,
          embed_dim: int = 768,
          use_pretrained: bool = True,
          freeze_backbone: bool = True,
          num_register_tokens: int = 4,
      ):


**KAN Encoder**
   - **Registry Name:** ``kan_encoder``
   - **Class:** ``spectramr.models.encoders.kan_encoder.KANEncoder``
   - **Description:** Encoder using Kolmogorov-Arnold Networks (KAN) for feature extraction.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          features: List[int] = [32, 64, 128, 256],
          kan_type: str = "BSpline",
          norm_layer: Optional[nn.Module] = nn.InstanceNorm2d,
          dropout: float = 0.0,
      ):

**Disentangled Encoder**
   - **Registry Name:** ``disentangled_encoder``
   - **Class:** ``spectramr.models.inr.disentangled_encoder.DisentangledEncoder``
   - **Description:** Encoder that separates anatomical structure from physics/contrast information.

   .. code-block:: python

      def __init__(
          self,
          anatomy_dim: int = 256,
          physics_dim: int = 64,
          in_channels: int = 1,
      ):


Meta-Learning Models
--------------------
Models with ``training_mode="meta_learning"``.

**Meta-Learning Friendly Generator**
   - **Registry Name:** ``meta_learning_friendly``
   - **Class:** ``spectramr.models.specialized.meta_learning_wrapper.MetaLearningFriendlyGenerator``
   - **Description:** Generator with fast adaptation capabilities for meta-learning.

   .. code-block:: python

      def __init__(
          self,
          generator_factory: Callable[...,nn.Module],
          adaptation_config: Optional[MetaAdaptationConfig] = None,
          **kwargs,
      ):

**MetaVarNet**
   - **Registry Name:** ``meta_varnet``
   - **Class:** ``spectramr.models.meta_learning.meta_varnet.MetaVarNet``
   - **Description:** Variational Network compatible with functional MAML.

   .. code-block:: python

      def __init__(
          self,
          num_cascades: int = 12,
          chans: int = 18,
          in_chans: int = 2,
      ):


Implicit Neural Representations (INR)
-------------------------------------
Models with ``training_mode="reconstruction"`` or ``"experimental"``.

**Velocity Network**
   - **Registry Name:** ``velocity_network``
   - **Class:** ``spectramr.models.transport.velocity_network.VelocityNetwork``
   - **Description:** Predicts velocity fields for continuous normalizing flow transport.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          hidden_channels: int = 64,
          num_layers: int = 4,
          time_embed_dim: int = 64,
      ):

**Hypernetwork**
   - **Registry Name:** ``hypernetwork``
   - **Class:** ``spectramr.models.inr.hypernetwork.Hypernetwork``
   - **Description:** Generates weights for HyperSIREN from latent codes.

   .. code-block:: python

      def __init__(
          self,
          z_geo_dim: int,
          z_phy_dim: int,
          hyper_siren: HyperSIREN,
      ):

**Local INR**
   - **Registry Name:** ``local_inr``
   - **Class:** ``spectramr.models.inr.local_inr.LocalINR``
   - **Description:** Single Local INR module (Encoder + Hypernet + SIREN).

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          z_geo_dim: int = 256,
          z_phy_dim: int = 64,
          hidden_features: int = 64,
          hidden_layers: int = 3,
      ):

**Dual Motion INR**
   - **Registry Name:** ``dual_motion_inr``
   - **Class:** ``spectramr.models.inr.motion.DualMotionINR``
   - **Description:** Combines Spatial Canonical INR with Temporal Flow INR.

   .. code-block:: python

      def __init__(
          self,
          spatial_inr: LocalINR,
          temporal_features: int = 64,
      ):

**Implicit KAN**
   - **Registry Name:** ``implicit_kan``
   - **Class:** ``spectramr.models.experimental.implicit_kan.ImplicitKAN``
   - **Description:** KAN-based Implicit Neural Representation.

   .. code-block:: python

      def __init__(
          self,
          spatial_dim=3,
          hidden_dim=64,
          num_layers=4,
          out_channels=1
      ):

**Implicit MRI Field**
   - **Registry Name:** ``implicit_mri_field``
   - **Class:** ``spectramr.models.experimental.implicit_kan.ImplicitMRIField``
   - **Description:** Full INR model for MRI with positional encoding.

   .. code-block:: python

      def __init__(
          self,
          spatial_dim=2,
          hidden_dim=128,
          num_layers=4,
          num_frequencies=6
      ):


Gaussian Splatting Models
-------------------------
Models with ``training_mode="reconstruction"`` or ``"experimental"``.

**MedGS**
   - **Registry Name:** ``medgs``
   - **Class:** ``spectramr.models.gaussian_splatting.medgs.MedGS``
   - **Description:** Gaussian Cloud with Polynomial Trajectory (Folded Gaussians).

   .. code-block:: python

      def __init__(
          self,
          num_gaussians: int = 10000,
          sh_degree: int = 3,
          polynomial_degree: int = 2,
          feature_dim: int = 32,
          spatial_extent: float = 1.0,
      ):

**SuGaR SDF**
   - **Registry Name:** ``sugar_sdf``
   - **Class:** ``spectramr.models.gaussian_splatting.sugar.SuGaRSDF``
   - **Description:** A learnable SDF implicit function for Surface-Aligned Gaussian Splatting.

   .. code-block:: python

      def __init__(
          self,
          in_features=3,
          hidden_features=64,
          num_layers=4
      ):


Geometric & Mesh Models
-----------------------
Models with ``training_mode="reconstruction"``.

**Template Deformer**
   - **Registry Name:** ``template_deformer``
   - **Class:** ``spectramr.models.geometric.template_deformation.TemplateDeformer``
   - **Description:** Deforms a template mesh based on image features using Graph Conv.

   .. code-block:: python

      def __init__(
          self,
          template_vertices: Float[Tensor, "V 3"],
          template_faces: Int[Tensor, "F 3"],
          image_encoder_dims: List[int] = [16, 32, 64],
          hidden_dim: int = 128
      ):

**Differentiable Slicer**
   - **Registry Name:** ``differentiable_slicer``
   - **Class:** ``spectramr.models.geometric.differentiable_slicer.DifferentiableSlicer``
   - **Description:** Computes soft occupancy of a mesh on a query grid (slice).

   .. code-block:: python

      def __init__(
          self,
          sigma: float = 1.0
      ):


Additional Diffusion Variants
-----------------------------
Models with ``training_mode="diffusion"``.

**Simple Enhanced Diffusion**
   - **Registry Name:** ``simple_enhanced_diffusion``
   - **Class:** ``spectramr.models.diffusion.simple_enhanced_diffusion.SimpleEnhancedDiffusionUNet``
   - **Description:** Simple but robust enhanced diffusion U-Net (Base Class).

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          out_channels: int = 1,
          hidden_channels: int = 64,
          num_layers: int = 4,
          time_embed_dim: int = 128,
          use_attention: bool = True,
      ):

**Simple Gaussian Diffusion**
   - **Registry Name:** ``simple_gaussian_diffusion``
   - **Class:** ``spectramr.models.diffusion.simple_enhanced_diffusion.SimpleGaussianDiffusionUNet``
   - **Description:** Gaussian noise variant of Simple Enhanced Diffusion.

**Simple Chi-Square Diffusion**
   - **Registry Name:** ``simple_chi_square_diffusion``
   - **Class:** ``spectramr.models.diffusion.simple_enhanced_diffusion.SimpleChiSquareDiffusionUNet``
   - **Description:** Chi-Square noise variant of Simple Enhanced Diffusion.

**Simple Gaussian KAN Diffusion**
   - **Registry Name:** ``simple_gaussian_kan_diffusion``
   - **Class:** ``spectramr.models.diffusion.simple_enhanced_diffusion.SimpleGaussianKANDiffusionUNet``
   - **Description:** KAN-embedding variant of Simple Enhanced Diffusion.

**Adversarial Purification Diffusion**
   - **Registry Name:** ``adversarial_purification``
   - **Class:** ``spectramr.models.generators.adversarial_purification.AdversarialPurificationDiffusion``
   - **Description:** Adversarial Purification using Diffusion Models.

   .. code-block:: python

      def __init__(
          self,
          diffusion_model: nn.Module,
          num_steps: int = 10,
          step_size: float = 0.01,
          epsilon: float = 0.1,
      ):


Other Specialized Models
------------------------

**Low-Field Augmented Generator**
   - **Registry Name:** ``low_field_augmented``
   - **Class:** ``spectramr.models.specialized.low_field_augmented_generator.LowFieldStyleAugmentedGenerator``
   - **Description:** Generator with low-field style augmentation.

   .. code-block:: python

      def __init__(
          self,
          generator: IGenerator,
          style_augmentation: IStyleAugmentation,
          augmentation_probability: float = 0.5,
      ):

**Generative World Model**
   - **Registry Name:** ``generative_world_model``
   - **Class:** ``spectramr.models.experimental.world_model.GenerativeWorldModel``
   - **Description:** Generative World Model using Swin-Transformer Encoder and FNO Decoder.

   .. code-block:: python

      def __init__(
          self,
          img_size=256,
          patch_size=4,
          in_chans=1,
          embed_dim=96,
          depths=[2, 2, 6, 2],
          num_heads=[3, 6, 12, 24],
          window_size=7,
          mlp_ratio=4.0,
          params_dim=10,
      ):

**SoTa Adapter**
   - **Registry Name:** ``sota_adapter``
   - **Class:** ``spectramr.models.adaptation.sota_adapter.SoTaAdapter``
   - **Description:** Test-Time Adapter for Score-Based Diffusion Models.

   .. code-block:: python

      def __init__(
          self,
          score_net: ScoreNetwork,
          learning_rate: float = 1e-4,
          num_steps: int = 10,
      ):

**Latent Aligner**
   - **Registry Name:** ``latent_aligner``
   - **Class:** ``spectramr.models.adaptation.federated.LatentAligner``
   - **Description:** Computes statistical alignment (MMD) between local latent batches and global statistics.

   .. code-block:: python

      def __init__(self, kernel_type='rbf'):

Additional Registered Models (Audit Findings)
---------------------------------------------
Models identified and registered during codebase audit.

**Configurable UNet**
   - **Registry Name:** ``configurable_unet``
   - **Class:** ``spectramr.models.reconstruction.unet.UNet``
   - **Description:** A highly configurable UNet generator that consolidates multiple UNet variants.

   .. code-block:: python

      def __init__(self, config: UNetConfig | None = None, **kwargs):

**KIDOT Transport**
   - **Registry Name:** ``kidot_transport``
   - **Class:** ``spectramr.models.transport.kidot_transport.KIDOTTransport``
   - **Description:** Knowledge-Informed Dynamic Optimal Transport model.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 1,
          hidden_channels: int = 64,
          num_layers: int = 4,
          num_steps: int = 10,
          lambda_physics: float = 1.0,
          degradation_type: str = "combined",
          **kwargs,
      ):

**ReconFormer**
   - **Registry Name:** ``recon_former``
   - **Class:** ``spectramr.models.transformer.recon_former.ReconFormer``
   - **Description:** K-Space / Image Domain Transformer for Reconstruction.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int = 2,
          embed_dim: int = 96,
          depths: Tuple[int, ...] = (2, 2, 6, 2),
          num_heads: Tuple[int, ...] = (3, 6, 12, 24),
          window_size: int = 7
      ):

**Latent Input Wrapper**
   - **Registry Name:** ``latent_input_wrapper``
   - **Class:** ``spectramr.models.latent_input_wrapper.LatentInputWrapper``
   - **Description:** Wrapper that allows VAE/VQ-VAE models to accept latent vectors as input.

   .. code-block:: python

      def __init__(self, model: nn.Module, latent_dim: int):

Additional Models (Audit Batch 12)
----------------------------------
Models registered for Domain Adaptation, Uncertainty, and Compression.

**Cross-Scanner Adaptation Network**
   - **Registry Name:** ``cross_scanner_adaptation``
   - **Class:** ``spectramr.models.domain_adaptation.CrossScannerAdaptationNetwork``
   - **Description:** Complete cross-scanner adaptation network with reconstruction heads.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int,
          num_scanners: int,
          feature_channels: int = 256,
      ):

**Multi-Domain Feature Extractor**
   - **Registry Name:** ``multi_domain_extractor``
   - **Class:** ``spectramr.models.domain_adaptation.MultiDomainFeatureExtractor``
   - **Description:** Feature extractor that handles multiple domains/scanners with ensemble capabilities.

   .. code-block:: python

      def __init__(
          self,
          in_channels: int,
          feature_channels: int = 256,
          num_domains: int = 3,
      ):

**Ensemble Model**
   - **Registry Name:** ``ensemble_model``
   - **Class:** ``spectramr.models.uncertainty.EnsembleModel``
   - **Description:** Ensemble of models for uncertainty quantification.

   .. code-block:: python

      def __init__(self, models: list[nn.Module]):

**Efficient UNet**
   - **Registry Name:** ``efficient_unet``
   - **Class:** ``spectramr.models.model_compression.EfficientUNet``
   - **Description:** Optimized UNet architecture supporting depthwise separable convolutions.

   .. code-block:: python

      def __init__(
          self,
          input_channels: int = 1,
          output_channels: int = 1,
          base_channels: int = 64,
          use_depthwise: bool = False,
      ):


Contract Testing
----------------

The model registry is covered by ``tests/contracts/test_model_registry.py``,
which includes a ``@pytest.mark.slow`` parametric forward-pass probe for every
registered model key.  The probe uses a capabilities-driven builder rather than
a naïve zero-arg attempt so that models with required constructor parameters are
actually exercised.

Minimal Builder Utility
~~~~~~~~~~~~~~~~~~~~~~~

``tests/utils/minimal_builders.py`` provides three public helpers:

``build_minimal_model(name)``
   Three-step build ladder:

   1. Zero-arg construction (``cls()``).
   2. Signature introspection — required params filled from declared
      :class:`~spectramr.models.capabilities.ModelCapabilities` and a name-based
      fallback table (``in_channels``, ``spatial_shape``, ``denoising_model``,
      etc.).
   3. Explicit per-model overrides in ``_OVERRIDES`` dict (e.g. ``cs_mno_operator``
      needs a ``spatial_shape`` tuple).

   Returns ``(instance, "ok")`` on success or ``(None, reason)`` on failure.

``phantom_for(caps)``
   Synthesises a domain-correct ``torch.Tensor`` input: 2-D vs 3-D spatial
   layout from ``caps.spatial_dims``; ``complex64`` if ``accepts_complex=True``;
   two channels if ``expects_real_imag_interleaved=True``.

``expected_output_ok(out, caps)``
   Loose forward-output sanity check: finds the first Tensor in any
   ``Tensor | tuple | list | dict`` nesting, verifies ``ndim >= 2`` and all
   dimensions positive.

Two-Mode Gate
~~~~~~~~~~~~~

The slow contract test has two modes controlled by the ``CONTRACT_STRICT``
environment variable:

``CONTRACT_STRICT=0`` (default)
   Build or forward failures are *xfailed* (first-run discovery mode).
   Run the suite once on the cluster to discover the real set of
   unbuildable models.

``CONTRACT_STRICT=1``
   Models listed in ``tests/contracts/_known_unbuildable.json`` are xfailed
   (tracked debt).  Models **not** in the list that fail are hard-failed
   (regression gate).  Shrink the allowlist by adding per-model overrides to
   ``tests/utils/minimal_builders._OVERRIDES``.

Models built from published formulations
========================================

Twelve further models, each built from its published formulation, registered via
``@register_model`` and constructible with no kwargs (hyperparameters flow
through ``model.model_kwargs`` in YAML).

Generative density models (``training_mode="generative"``)
----------------------------------------------------------

**Glow**
   - **Registry Name:** ``glow`` — ``spectramr.models.generative.glow.Glow``
   - Multi-scale normalising flow (Kingma & Dhariwal, NeurIPS 2018):
     actnorm + invertible 1×1 conv (LU) + affine coupling. Exact
     log-likelihood; trained by the ``generative`` strategy (NLL).

**Equivariant Flow**
   - **Registry Name:** ``equivariant_flow`` —
     ``spectramr.models.generative.equivariant_flow.EquivariantFlow``
   - :math:`C_n`-equivariant normalising flow (Köhler et al., ICML 2020;
     Cohen & Welling group convs). Rotation-invariant learned density.

Flow-matching
-------------

**Divergence-Free Flow**
   - **Registry Name:** ``divergence_free_flow`` —
     ``spectramr.models.generative.divergence_free_flow.DivergenceFreeFlow``
   - Streamfunction (2D) / curl (3D) parameterisation enforcing
     :math:`\nabla\cdot v \equiv 0` structurally (``training_mode="flow_matching"``).

Diffusion
---------

**Blurring Diffusion**
   - **Registry Name:** ``blurring_diffusion`` —
     ``spectramr.models.diffusion.blurring_diffusion.BlurringDiffusion``
   - Heat-equation (DCT-diagonalised) forward process with a
     v-parameterised denoiser (Hoogeboom & Salimans, ICLR 2023). DCT ops
     live in ``spectramr.infrastructure.physics.dct_ops``.

VAE / VQ family
---------------

**Ladder VAE** — ``hierarchical_vae_ladder``
   (``spectramr.models.vae.hierarchical_vae_ladder.LadderVAE``) — top-down
   inference with precision-weighted posterior merge (Sønderby et al., 2016).

**Hyperspherical VAE** — ``hyperspherical_vae``
   (``spectramr.models.vae.hyperspherical_vae.HypersphericalVAE``) — vMF latent
   prior on :math:`S^{d-1}`, avoids KL collapse (Davidson et al., UAI 2018).

**MoE VAE** — ``moe_vae`` (``spectramr.models.vae.moe_vae.MoEVAE``) — top-1
   gated expert routing with load-balancing loss (Shazeer et al., 2017).

**Hierarchical VQ-VAE (VQ-VAE-2)** — ``hierarchical_vq_vae``
   (``spectramr.models.vq.hierarchical_vq_vae.HierarchicalVQVAE``) — two-level
   discrete latent hierarchy (Razavi et al., 2019); reuses ``VectorQuantizer``.

**β-VAE/GAN** — ``beta_vae_gan``
   (``spectramr.models.generative.beta_vae_gan.BetaVAEGAN``) — composes the
   ``beta_tc_vae`` objective with a hinge adversarial head (``training_mode="gan"``).

GAN / reconstruction
--------------------

**Progressive GAN** — ``progressive_gan``
   (``spectramr.models.gans.progressive_gan.ProgressiveGAN``) — phase-grown
   generator with fade-in, pixel-norm, minibatch-stddev (Karras et al., 2018).
   Driven by ``ProgressiveGANStrategy`` (``training_mode="progressive_gan"``).

**Octave U-Net** — ``octave_conv``
   (``spectramr.models.generators.octave_unet.OctaveUNet``) — high/low-frequency
   octave convolution (Chen et al., ICCV 2019), a learned k-space band split.

**Gated GNN Reconstructor** — ``gated_gnn``
   (``spectramr.models.generators.gated_gnn_reconstructor.GatedGNNReconstructor``)
   — GRU message-passing over a k-space graph (Li et al., ICLR 2016).

Name aliases
------------

Twenty name-variants are registered as aliases of their canonical
registrations in ``spectramr.models.stubs.register_aliases`` (e.g.
``vq_vae`` → ``vqvae``, ``stylegan`` → ``stylegan2``,
``standard_vit`` → ``vision_transformer``). The frozen mapping and its
regression test live in ``tests/unit/models/test_registry_aliases.py``.

Rejected names
--------------

Twenty-four names with no implementation and no anchoring
specification are recorded in ``spectramr.models.registry.REJECTED_NAMES``
with a per-name rationale. They are absent from ``VALID_MODEL_TYPES``; the
``namespace_axis`` audit check surfaces the rationale as a fix hint if a
YAML references one.

Audit checks on model_type
--------------------------

Two Tier-1 checks in ``ConfigHealthChecker`` make an advertised-but-unresolvable
``model_type`` a load-time error rather than a runtime one:

``check_registered_model_resolves``
   A registered ``model_type`` must resolve to a concrete class (not an
   abstract base / missing ``forward``), else it is a Tier-1 error.

``check_namespace_axis``
   Rejects training-mode / dataset-type / strategy / block tokens (and
   ``REJECTED_NAMES`` entries) misfiled under ``model.model_type``, with a
   precise fix hint naming the correct config axis.
