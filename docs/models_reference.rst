.. _models_reference:

====================================
Model Architectures — Reference
====================================

.. sectionauthor:: spectraMR Research

spectraMR's generators are organized by reconstruction paradigm. All follow the
``IGenerator`` interface: ``__init__(in_channels, out_channels, **kwargs)`` and
``forward(x)``.

.. note::

   **This page states no model count, deliberately.** A frozen count in a
   hand-written page drifts silently, and is worse than no count. The registry is
   the only answer that cannot go stale. Ask it:

   .. code-block:: python

      from spectramr.models.init_registry import populate_model_registry
      from spectramr.models.registry import MODEL_REGISTRY

      populate_model_registry()   # REQUIRED -- the registry is empty without it
      len(MODEL_REGISTRY)

   This page is also **not exhaustive**: it documents a subset of what is
   registered, so a name's absence here is not evidence it does not exist. Check
   ``MODEL_REGISTRY`` before concluding a model is unavailable.

.. contents:: Table of Contents
   :depth: 2
   :local:


Registration & Usage
====================

.. code-block:: python

   from spectramr.models.factories.model_factory import ModelFactory

   factory = ModelFactory()
   model = factory.create_generator(
       model_type="standard_unet",
       in_channels=2, out_channels=2,
   )

.. code-block:: yaml

   model:
     model_type: standard_unet
     in_channels: 2
     out_channels: 2
     model_kwargs:
       features: [32, 64, 128, 256]


---

U-Net Family
============

Standard U-Net
--------------

**Registry:** ``standard_unet`` — **File:** ``unet_generator.py``

The workhorse encoder-decoder with skip connections:

.. math::

   \hat{x} = D\bigl(E(y) \oplus \text{skip}\bigr)

where :math:`E` is the contracting path, :math:`D` the expanding path,
and :math:`\oplus` denotes concatenation along the channel axis. Each
level halves spatial resolution while doubling feature channels.

**Architecture:**

- Encoder: ``[Conv → BN → ReLU → Conv → BN → ReLU → MaxPool]`` × depth
- Decoder: ``[UpConv → Cat(skip) → Conv → BN → ReLU → Conv → BN → ReLU]`` × depth
- Final: ``1×1 Conv`` to output channels


Attention U-Net
---------------

**Registry:** ``attention_unet`` — **File:** ``attention_unet.py``

Augments skip connections with additive attention gates that learn to
suppress irrelevant regions:

.. math::

   \alpha_i = \sigma_2\bigl(\psi^T \cdot \sigma_1(W_x x_i + W_g g + b)\bigr)

where :math:`g` is the gating signal from the decoder, :math:`x_i` is
the skip feature, and :math:`\alpha_i` is the spatial attention weight.


Enhanced Deep U-Net
-------------------

**Registry:** ``enhanced_deep_unet`` — **File:** ``enhanced_deep_unet.py``

U-Net with residual dense blocks, squeeze-and-excitation channel attention,
and deep supervision at multiple decoder levels.


Complex U-Net
-------------

**Registry:** (via ``complex_unet``) — **File:** ``complex_unet.py``

Enforces complex algebra throughout the network by using complex-valued
convolutions. Input is a 2-channel (real, imaginary) tensor treated
as a single complex feature map:

.. math::

   W * z = (W_{re} * z_{re} - W_{im} * z_{im}) + i(W_{re} * z_{im} + W_{im} * z_{re})


U-Net 3D Variants
------------------

**Registry:** ``unet3d``, ``vnet``, ``unetr`` — **Files:** ``3d_generators/``

- **VNet**: 3D U-Net with volumetric convolutions and residual connections
- **UNETR**: Vision Transformer encoder with U-Net decoder for 3D volumes
- **AnisotropicVoxelGenerator**: Hybrid 2D/3D for anisotropic voxel spacing


FPN U-Net
---------

**Registry:** ``fpn_unet`` — **File:** ``fpn_unet.py``

Feature Pyramid Network with top-down pathway and lateral connections,
producing multi-scale feature maps.


---

Diffusion & Flow Models
=======================

Rectified Flow Generator
------------------------

**Registry:** ``rectified_flow`` — **File:** ``rectified_flow_generator.py``

Learns a straight-line deterministic ODE trajectory from a base distribution to the data distribution:

.. math::

   v_\theta(z_t, t, c) \approx \frac{d z_t}{d t}

This allows 1-step or few-step sampling over highly complex manifolds without stochastic Langevin variance.

Neural ODE Continuous-Depth
---------------------------

**Registry:** ``neural_ode`` — **File:** ``neural_ode_generator.py``

Replaces discrete residual connections with continuous differential equations integrated via RK4 or Dopri5:

.. math::

   h(L) = h(0) + \int_0^L f_\theta(h(t), t) dt

Enables memory-efficient, depth-adaptive inference.

Cold Diffusion Generator
------------------------

**Registry:** ``kspace_cold_diffusion`` — **File:** ``cold_diffusion_generator.py``

Deterministic degradation in k-space instead of Gaussian noise:

.. math::

   \tilde{x}_t = (1-t) \cdot x_0 + t \cdot D(x_0)

where :math:`D` is a degradation operator (undersampling, Rician noise,
chi-square noise). The model learns to invert:

.. math::

   \hat{x}_0 = R_\theta(\tilde{x}_t, t)


K-Space Cold Diffusion Generator
---------------------------------

**Registry:** ``kspace_cold_diffusion_generator`` — **File:** ``kspace_cold_diffusion_generator.py``

Specialized cold diffusion operating directly in k-space with structured
degradations: variable-density undersampling masks and field-strength-dependent
noise models.


Chi-Square Diffusion Generator
------------------------------

**Registry:** ``chi_square_diffusion`` — **File:** ``chi_square_diffusion_generator.py``

Uses the chi-square noise distribution native to multi-coil magnitude MRI:

.. math::

   M = \sqrt{\sum_{c=1}^C |x_c + n_c|^2}, \quad n_c \sim \mathcal{CN}(0, \sigma^2)

where :math:`M` follows a non-central chi distribution with :math:`2C`
degrees of freedom.


Consistency Model Generator
----------------------------

**Registry:** ``consistency_model`` — **File:** ``consistency_model_generator.py``

Maps any point on an ODE trajectory directly to the origin in a single step:

.. math::

   f_\theta(x_t, t) = f_\theta(x_{t'}, t') \quad \forall\, t, t' \in [0, T]

This self-consistency property enables one-step generation.


Score-Based Diffusion
---------------------

**Registry:** ``score_based_diffusion`` — **File:** ``score_based_diffusion_generator.py``

Learns the score function :math:`\nabla_x \log p(x)` via denoising
score matching:

.. math::

   \mathcal{L} = \mathbb{E}\left[\left\|\mathbf{s}_\theta(x + \sigma\epsilon) - \frac{-\epsilon}{\sigma}\right\|^2\right]


Conditional Diffusion Generator
-------------------------------

**Registry:** ``conditional_diffusion`` — **File:** ``conditional_diffusion_generator.py``

Physics-conditioned diffusion with acceleration-aware timestep injection:

.. math::

   \epsilon_\theta(x_t, t, c) \quad \text{where } c = (\text{mask}, \text{acceleration}, \text{kspace})


Cascaded Diffusion Generator
-----------------------------

**Registry:** ``cascaded_diffusion`` — **File:** ``cascaded_diffusion_generator.py``

Multi-stage coarse-to-fine diffusion: each stage operates at increasing
spatial resolution with the previous stage's output as conditioning.


---

Variational & Quantized Models
================================

VAE
---

**Registry:** ``vae`` — **File:** ``vae/vae.py``

Standard variational autoencoder:

.. math::

   q(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x)), \quad
   p(x|z) = \mathcal{N}(\mu_\theta(z), \sigma^2 I)


VQ-VAE
------

**Registry:** ``vqvae`` — **File:** ``vq/vqvae.py``

Replaces continuous latent with discrete codebook entries via nearest-neighbor
lookup:

.. math::

   z_q = e_k, \quad k = \arg\min_j \|z_e - e_j\|_2

Gradients bypass the non-differentiable quantization via straight-through
estimator.


VQ-GAN
------

**Registry:** ``vqgan`` — **File:** ``vq/vqgan.py``

VQ-VAE with a PatchGAN discriminator to improve perceptual quality of
decoded images.


Sparse VAE
----------

**Registry:** ``sparse_vae`` — **File:** ``vae/sparse_vae.py``

VAE with sparsity-inducing prior on the latent space for disentangled
representations.

Masked Autoencoder (MAE)
-------------------------

**Registry:** ``mae_mri`` — **File:** ``mae_generator.py``

Self-supervised pretraining model for MRI (Pillar 6). It masks a high percentage (e.g., 75%) of volumetric patches in image space or k-space lines, forcing the asymmetric encoder-decoder to reconstruct missing contextual details.


---

Transformer Architectures
==========================

Swin Transformer U-Net
-----------------------

**Registry:** ``swin_transformer`` — **File:** ``swin_unet_generator.py``

Replaces convolutional blocks with Shifted Window (Swin) Transformer blocks:

.. math::

   \text{Attn}(Q, K, V) = \text{SoftMax}\left(\frac{QK^T}{\sqrt{d}} + B\right)V

where :math:`B` is the relative position bias. Window-based attention reduces
complexity from :math:`O(N^2)` to :math:`O(N \cdot w^2)` for window size :math:`w`.

Structure Tensor Transformer
----------------------------

**Registry:** ``structure_tensor_transformer`` — **File:** ``structure_tensor_transformer.py``

Injects physical eigen-decomposition (tissue anisotropy) directly into the self-attention weights (Pillar 8).
The attention mechanism favors structurally coherent tissue trajectories over isotropic noise:

.. math::

   \text{Attn}(Q,K,V)_{ij} = \text{Softmax}\bigl(QK^T - \gamma \cdot D_{ST}(i,j)\bigr)V


Vision Transformer (ViT)
-------------------------

**Registry:** ``vit``, ``standard_vit`` — **File:** ``vit_generator.py``

Divides image into patches, embeds them, and processes with self-attention:

.. math::

   z_0 = [x_{cls}; x_1 E; x_2 E; \ldots; x_N E] + E_{pos}

.. math::

   z_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}


Vision Mamba
------------

**Registry:** ``vision_mamba`` — **File:** ``vision_mamba_generator.py``

State-space model replacing self-attention with linear-time sequence modeling:

.. math::

   h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t

where :math:`\bar{A}, \bar{B}` are discretized state-space parameters. Achieves
:math:`O(N)` complexity versus :math:`O(N^2)` for attention.


Linformer
---------

**Registry:** ``linformer`` — **File:** ``linformer_generator.py``

Projects keys and values to lower dimension for linear-complexity attention:

.. math::

   \text{Attn}(Q, K, V) = \text{SoftMax}\left(\frac{Q (E_K K)^T}{\sqrt{d}}\right) (E_V V)


Perceiver
---------

**Registry:** ``perceiver`` — **File:** ``perceiver_generator.py``

Cross-attention from a learned latent array to the input, enabling
arbitrary input modalities with constant computation.


---

Operator & Neural Field Models
================================

Fourier Neural Operator (FNO)
-----------------------------

**Registry:** ``fno`` — **File:** ``fno_generator.py``

Learns operators in Fourier space via spectral convolutions:

.. math::

   v_{l+1}(x) = \sigma\left(W v_l(x) + \mathcal{F}^{-1}(R_l \cdot \mathcal{F}(v_l))(x)\right)

where :math:`R_l` is a learnable filter in frequency domain. Achieves
resolution-independent operator learning.


Geo-FNO
-------

**Registry:** ``geo_fno`` — **File:** ``geo_fno_generator.py``

Geometry-aware FNO that learns a diffeomorphic mapping from irregular
geometries to a uniform latent grid before applying spectral convolutions.


DeepONet
--------

**File:** ``deeponet_generator.py``

Learns nonlinear operators via a branch-trunk decomposition:

.. math::

   G(u)(y) = \sum_{k=1}^{p} b_k(u) \cdot t_k(y)

where the branch net :math:`b_k` encodes the input function and the trunk
net :math:`t_k` encodes the query coordinates.


SIREN / SirenSensNet
---------------------

**Registry:** ``siren_sens_net`` — **File:** ``siren_pinn.py``

SIREN (Sinusoidal Representation Network) uses periodic activations:

.. math::

   \phi_i(x) = \sin(\omega_i \cdot (W_i x + b_i))

The initial frequency :math:`\omega_0` controls the bandwidth. Higher
:math:`\omega_0` captures finer spatial detail. ``SirenSensNet`` maps
``(x,y)`` coordinates to complex-valued coil sensitivity maps.


Implicit Neural Representation (INR)
-------------------------------------

**Registry:** ``implicit_mri_field`` — **File:** ``implicit_representation.py``

Coordinate-based MLP that represents MRI volumes as continuous functions:

.. math::

   I(x, y, z) = f_\theta(x, y, z) \in \mathbb{R}^C


DiffDeuR
--------

**Registry:** ``diffdeur`` — **File:** ``diffdeur.py``

Zero-shot enhancement via coordinate-based INR fitted to a single scan.


---

Super-Resolution Models
========================

EDSR
----

**Registry:** ``edsr`` — **File:** ``edsr_generator.py``

Enhanced Deep Super-Resolution with residual blocks and no batch
normalization:

.. math::

   \hat{x} = x + \sum_{i=1}^{B} R_i(x) \quad \text{(global residual)}


RCAN
----

**Registry:** ``rcan`` — **File:** ``rcan_generator.py``

Residual Channel Attention Network with squeeze-and-excitation:

.. math::

   s = \sigma(W_2 \cdot \delta(W_1 \cdot \text{GAP}(F)))

where :math:`\text{GAP}` is global average pooling and :math:`s` is the
channel attention vector.


SwinIR
------

**Registry:** ``swinir`` — **File:** ``swinir_generator.py``

Swin Transformer for image restoration with residual Swin blocks.


NAFNet
------

**Registry:** ``nafnet`` — **File:** ``nafnet_generator.py``

Nonlinear Activation Free Network replacing ReLU/GELU with Simple Gate:

.. math::

   \text{SimpleGate}(x) = x_1 \odot x_2 \quad \text{(channel split)}

Achieves state-of-the-art denoising with extremely simple operations.


HAN
---

**Registry:** ``han_generator`` — **File:** ``han_generator.py``

Holistic Attention Network with layer attention modules for adaptive
feature aggregation across residual groups.


Diffeomorphic Synthesis Net
---------------------------

**Registry:** ``diffeomorphic_synthesis_net`` — **File:** ``diffeomorphic_synthesis_net.py``

Outputs velocity/deformation fields to warp ultra-low-field features into high-field representations, constrained by Hyperelastic Jacobian penalties (Pillar 10). It ensures topological preservation and prevents hallucinated artifacts via deformation flows.

Evidential U-Net
----------------

**Registry:** ``evidential_unet`` — **File:** ``evidential_unet.py``

Reconstructs standard deterministic images alongside parameter maps for a Normal-Inverse-Gamma distribution (Pillar 9). Provides mapped aleatoric and epistemic uncertainties globally across the reconstructed matrix.


---

Unrolled Optimization
======================

Variational Network (VarNet)
----------------------------

**Registry:** ``variational_network`` — **File:** ``variational_network.py``

Unrolled gradient descent with learned regularizer:

.. math::

   x^{(k+1)} = x^{(k)} - \eta_k \left[A^H(Ax^{(k)} - y) + \lambda_k R_\theta(x^{(k)})\right]

Each cascade :math:`k` has its own learned regularizer :math:`R_\theta`.


Deep Unfolding
--------------

**Registry:** ``deep_unfolding`` — **File:** ``deep_unfolding.py``

ADMM/ISTA unrolled with learnable parameters:

.. math::

   z^{(k)} = \text{prox}_{\lambda R}(x^{(k)} + u^{(k)}), \quad
   x^{(k+1)} = (A^HA + \rho I)^{-1}(A^Hy + \rho(z^{(k)} - u^{(k)}))


DiffVarNet
----------

**File:** ``diff_varnet.py``

Differentiable VarNet with complex-valued convolutions and sensitivity-weighted
data consistency in each cascade.


---

Bayesian & Uncertainty Models
==============================

Bayesian U-Net
--------------

**Registry:** ``bayesian_unet`` — **File:** ``bayesian_generators.py``

Dropout-based approximate Bayesian inference for epistemic uncertainty
estimation. At inference time, dropout is kept active and T forward passes
produce a distribution over predictions:

.. math::

   \hat{x} = \frac{1}{T}\sum_{t=1}^{T} G_{\theta}^{(t)}(y), \quad
   \text{Var}[\hat{x}] = \frac{1}{T}\sum_{t=1}^{T} (G^{(t)} - \hat{x})^2

``T`` is the number of Monte Carlo samples (default 20). Higher variance
voxels correspond to regions where the model is uncertain — typically at
tissue boundaries or in highly undersampled regions.

**Configuration:**

.. code-block:: yaml

   model:
     model_type: bayesian_unet
     in_channels: 2
     out_channels: 2
     model_kwargs:
       mc_dropout_rate: 0.1
       mc_samples: 20


MC-Dropout Ensemble
-------------------

**Registry:** ``mc_dropout_ensemble`` — **File:** ``bayesian_generators.py``

Explicit ensemble wrapper that runs any generator with ``model.train()`` mode
to activate stochastic dropout during inference. Outputs a mean prediction and
a pixel-wise uncertainty map.


---

CPT-4DMR (4D Cardiac Reconstruction)
======================================

**Registry:** ``cpt_4dmr`` — **Files:** ``cpt_4dmr/``

Decomposed spatial-temporal model for 4D cardiac MRI reconstruction
(free-breathing, ungated). The generator is split into:

1. **SpatialAnatomyNet**: Reconstructs the reference-phase anatomy
   :math:`x_{ref}` from undersampled k-space.
2. **TemporalMotionNet**: Estimates deformation fields
   :math:`\phi_t` from the reference to each temporal phase.
3. **Composition**: :math:`x_t = x_{ref} \circ \phi_t^{-1}` warps the
   reference to synthesize any phase.

.. math::

   \mathcal{L} = \mathcal{L}_{recon}(x_0, \hat{x}_0) +
   \lambda_{\phi} \sum_{t} \|\phi_t\|_{TV} +
   \lambda_{cyc} \|x_0 - x_0 \circ \phi_t \circ \phi_t^{-1}\|_1

The topology-preserving constraint prevents non-physical folds in the
deformation field.

**Configuration:**

.. code-block:: yaml

   model:
     model_type: cpt_4dmr
     in_channels: 2
     out_channels: 2
     model_kwargs:
       num_phases: 10          # Cardiac phases
       spatial_features: [32, 64, 128, 256]
       motion_features: [16, 32, 64]
       warp_mode: bilinear


---

B0 Hypernetwork
================

**Registry:** ``b0_hypernetwork`` — **File:** ``b0_hypernetwork.py``

A hypernetwork conditioned on estimated B0 field maps that generates
personalized weight corrections for the reconstruction backbone:

.. math::

   \Delta W = H_\psi(\hat{B}_0), \quad
   \hat{x} = G_{\theta + \Delta W}(y)

This approach allows a single reconstruction model to generalize
across different field strengths and B0 inhomogeneity patterns without
fine-tuning from scratch. The hypernetwork :math:`H_\psi` is a
lightweight convolutional network.

**Use case:** Low-field MRI (0.3T) where B0 inhomogeneity is significant
but coil corrections are unavailable.


---

AFTNet (Axial-Frequency Transformer)
======================================

**Registry:** ``aftnet`` — **File:** ``aftnet_generator.py``

Combines axial (row-wise + column-wise) attention with Fourier-space
operations in alternating blocks:

1. **Spatial block**: Axial self-attention with :math:`O(N)` complexity
2. **Frequency block**: Element-wise multiplication in FFT domain
   for global frequency modulation

.. math::

   z^{(l+1)} = \text{AxialAttn}(z^l) + \mathcal{F}^{-1}\bigl(R_l \odot \mathcal{F}(z^l)\bigr)

Designed for k-space reconstruction where both spatial and frequency
inductive biases are important.


---

Continual Learning U-Net
=========================

**Registry:** ``continual_learning_unet`` — **File:** ``continual_learning_unet.py``

Extends the standard U-Net with Elastic Weight Consolidation (EWC) or
PackNet-style progressive masking to prevent catastrophic forgetting when
sequentially trained on data from new scanner sites or protocols.

**EWC Regularization:**

.. math::

   \mathcal{L} = \mathcal{L}_{\text{new}} + \lambda \sum_i F_i (\theta_i - \theta^*_i)^2

where :math:`F_i` is the diagonal Fisher information for parameter :math:`i`
computed on the previous task's data, and :math:`\theta^*_i` are the
parameter values after the previous task.

**Use case:** Sequential training across M4Raw → FastMRI → CMRxRecon
protocols without revisiting earlier datasets.


---

GAN Architectures
==================

CycleGAN
--------

**File:** ``cycle_gan.py``

Image-to-image translation with cycle consistency:

.. math::

   \mathcal{L}_{cyc} = \|G_{BA}(G_{AB}(x_A)) - x_A\|_1 + \|G_{AB}(G_{BA}(x_B)) - x_B\|_1


CycleSR Generator
------------------

**Registry:** ``cyclesr`` — **File:** ``cyclesr_generator.py``

Super-resolution CycleGAN with learned upsampling blocks.


StarGAN v2
----------

**Registry:** ``stargan_v2_generator`` (``training_mode: stargan_v2``) — **File:**
``models/generators/stargan_v2.py``

Multi-domain any-to-any FIELD translation for MRIxFields Task 3. The five field-strength
domains are ``{0.1, 1.5, 3, 5, 7} T``. Three components work together: an
AdaIN-modulated generator (``StarGANv2Generator``) that translates an image conditioned
on a style vector; a ``MappingNetwork`` (shared MLP with per-domain heads) that maps a
latent code ``z`` + domain label ``y`` to a style vector; and a ``StyleEncoder`` (shared
convolutional trunk with per-domain heads) that extracts a style vector from a reference
image. Style genuinely drives the AdaIN affine transforms — the generator output is
verified to differ under two independent styles. The companion
discriminator (``stargan_v2_discriminator``, ``models/discriminators/stargan_v2_discriminator.py``)
shares the same domain-selection contract: ``forward(x, y) -> [B]`` logits gathered
from per-domain output heads of a shared convolutional trunk (Task 9).


---

Physics-Informed Models
========================

Bloch Cycle Network
-------------------

**Registry:** ``bloch_cycle_network`` — **File:** ``bloch_cycle_network.py``

Integrates Bloch equation simulation into a reconstruction cycle:

.. math::

   S_{pred} = M_0 (1 - e^{-TR/\hat{T}_1}) e^{-TE/\hat{T}_2}

Enforces that reconstruction outputs produce physically consistent MR signals.


Coil Sensitivity Network
-------------------------

**Registry:** ``coil_sensitivity_network`` — **File:** ``coil_sensitivity_network.py``

Directly regresses coil sensitivity maps from multi-coil k-space data using
a U-Net backbone.


---

Discriminators
==============

PatchGANDiscriminator
---------------------

**File:** ``patchgan_discriminator.py``

Classifies overlapping :math:`70 \times 70` patches as real/fake:

.. math::

   D(x) \in \mathbb{R}^{H/s \times W/s}

where each output element corresponds to a local receptive field. This
encourages high-frequency structure fidelity.


PatchGAN3DDiscriminator
-----------------------

**File:** ``patchgan_3d_discriminator.py``

Volumetric extension for 3D data with 3D convolutions.


K-Space Discriminator
---------------------

**File:** ``kspace_discriminator.py``

Operates in dual domains:

- **FrequencyDomainDiscriminator**: Pure k-space classification
- **KSpaceAwareDiscriminator**: Combines spatial + frequency features

Ensures both image and k-space quality.


Latent Discriminator
--------------------

**File:** ``latent_discriminator.py``

Operates on latent space embeddings for Latent GAN training.


---

Quick Reference Table
======================

.. list-table::
   :header-rows: 1
   :widths: 22 22 15 41

   * - Registry Name
     - Class
     - Paradigm
     - Key Innovation
   * - ``standard_unet``
     - ``UNet``
     - Reconstruction
     - Skip connections, multi-scale features
   * - ``attention_unet``
     - ``AttentionUNet``
     - Reconstruction
     - Attention gates on skip connections
   * - ``swin_transformer``
     - ``SwinTransformerUNet``
     - Reconstruction
     - Shifted window self-attention
   * - ``vision_mamba``
     - ``VisionMambaUNet``
     - Reconstruction
     - O(N) state-space sequence modeling
   * - ``fno``
     - ``FNOGenerator``
     - Reconstruction
     - Spectral convolutions (resolution-free)
   * - ``kspace_cold_diffusion``
     - ``ColdDiffusionGenerator``
     - Diffusion
     - Deterministic k-space degradation
   * - ``consistency_model``
     - ``ConsistencyModelGenerator``
     - Diffusion
     - One-step generation via self-consistency
   * - ``score_based_diffusion``
     - ``ScoreBasedDiffusion``
     - Diffusion
     - Score matching + SDE sampling
   * - ``vae``
     - ``VAE``
     - VAE
     - ELBO with reparameterization
   * - ``vqvae``
     - ``VQVAE``
     - VQ-VAE
     - Discrete codebook + straight-through
   * - ``siren_sens_net``
     - ``SirenSensNet``
     - PINN
     - Sinusoidal activations for CSM
   * - ``variational_network``
     - ``VarNet``
     - Unrolled
     - Learned regularizer per cascade
   * - ``nafnet``
     - ``NAFNetGenerator``
     - SR
     - Activation-free simple gate
   * - ``edsr``
     - ``EDSRGenerator``
     - SR
     - Deep residual, no BN
   * - ``swinir``
     - ``SwinIRGenerator``
     - SR
     - Swin blocks for restoration
   * - ``bloch_cycle_network``
     - ``BlochCycleNetwork``
     - Physics
     - Bloch equation cycle consistency
   * - ``deep_image_prior``
     - ``DeepImagePrior``
     - Zero-shot
     - Network structure as implicit prior
   * - ``rectified_flow``
     - ``RectifiedFlowGenerator``
     - ODE Flow
     - Straight-trajectory deterministic ODE
   * - ``neural_ode``
     - ``NeuralODEGenerator``
     - ODE Flow
     - Continuous depth integration
   * - ``mae_mri``
     - ``MAEGenerator``
     - Self-Supervised
     - Patch-based asymmetrical masking
   * - ``bayesian_unet``
     - ``BayesianUNet``
     - Uncertainty
     - MC-Dropout epistemic uncertainty
   * - ``cpt_4dmr``
     - ``CPT4DMRGenerator``
     - 4D/Temporal
     - Spatial-temporal decomposition for cardiac
   * - ``b0_hypernetwork``
     - ``B0Hypernetwork``
     - Physics
     - B0-conditioned weight generation
   * - ``aftnet``
     - ``AFTNetGenerator``
     - Transformer
     - Axial + Fourier alternating blocks
   * - ``continual_learning_unet``
     - ``ContinualLearningUNet``
     - Reconstruction
     - EWC / PackNet anti-forgetting
   * - ``structure_tensor_transformer``
     - ``StructureTensorTransformer``
     - Transformer
     - Physics-gated structure attention
   * - ``diffeomorphic_synthesis_net``
     - ``DiffeomorphicSynthesisNet``
     - Synthesis
     - Topological mapping via Jacobian penalty
   * - ``evidential_unet``
     - ``EvidentialUNet``
     - Synthesis
     - Epistemic uncertainty mapping (NIG)


WavKAN norm-kwarg filtering
===========================

:py:class:`spectramr.models.layers.kan.kan_convs.wav_kan.WavKANConvNDLayer`
accepts ``**norm_kwargs``, which must not be forwarded unfiltered to
``norm_class(output_dim, **norm_kwargs)``. The caller
chain ``RefinedKANUNet → DoubleConvWithKAN → WavKANConv2DLayer``
forwards the full YAML ``model_kwargs`` dict, which can include
KAN-specific options (``grid_size``, ``spline_order``, ``scale_noise``,
``scale_base``, ``grid_update_freq``, ``grid_range``,
``grid_update_decay``, ``base_filters``, ``features``,
``bottleneck_only``). Unfiltered, those land on ``BatchNorm2d.__init__`` and
raise ``TypeError: got an unexpected keyword argument 'grid_size'``.

The layer therefore introspects ``norm_class`` with ``inspect.signature`` and
keeps only kwargs the norm constructor's signature accepts.
Mirrors the filter pattern already present in
:py:mod:`spectramr.models.layers.kan.kan_convs.fast_kan_conv`. Pinned by
:py:mod:`tests.unit.models.blocks.test_wav_kan_norm_kwarg_filter` (14
tests: filter behaviour with both ``BatchNorm2d`` and
``InstanceNorm2d``, forward pass after filter, sanity check that
unfiltered kwargs would still break ``nn.BatchNorm2d``).


Utilities and Support Modules
==============================

Shape Validator (``models.utils.shape_validator``)
---------------------------------------------------

``validate_tensor_shape``, ``validate_5d_or_4d_tensor``,
``validate_channel_count``, ``validate_spatial_dims``,
``validate_batch_size``, and ``assert_shape_equals`` provide early
dimension-mismatch detection.  All raise ``ValueError`` (never fall back
silently) so out-of-spec tensors are caught at module boundaries rather than
deep inside forward passes.

Parameter Normalization (``models.parameter_normalization``)
-------------------------------------------------------------

:class:`ParameterNormalizer` maintains a per-model-type
:class:`ModelParameterSchema` registry.  Call
:func:`normalize_model_parameters` to remap legacy alias keys (e.g.
``in_channels`` → canonical target) and
:func:`validate_model_parameters` to obtain a list of validation errors
before constructing a module.


Stability Utilities
===================

GANBalanceManager (``models.stability.gan_balance_manager``)
-------------------------------------------------------------

Prevents discriminator dominance via a GPU-resident ring-buffer of D/G
loss history.  Key API:

* ``update_losses(d_loss, g_loss)`` — update ring buffer (no CPU sync except
  for the emergency-mode scalar check).
* ``should_skip_discriminator_update()`` — probabilistic skip.
* ``get_discriminator_lr_factor()`` — adaptive LR reduction factor.
* ``emergency_mode`` attribute — set when ``d_loss < emergency_threshold``
  for more than 10 consecutive steps.

RuntimeErrorHandler (``models.stability.runtime_error_handler``)
-----------------------------------------------------------------

Static helpers for safe model calls, device migration, and gradient
clearing.  Used by strategies that need robust fallback on device
mismatches without silencing genuine logic errors.

StabilityAnalyzer (``models.stability.stability_linter``)
----------------------------------------------------------

Architecture + parameter + gradient + training-stability analysis.
``run_full_analysis()`` returns a weighted composite score in [0, 100].
Penalises models with no normalisation layers; rewards residual
connections and sound gradient norms.


Infrastructure: Training Helpers
=================================

GANLossHelper / ReconstructionLossHelper
(``infrastructure.training.loss_computation_helpers``)
---------------------------------------------------------

:class:`GANLossHelper` wraps R1 regularisation (disabled → zero tensor)
and extracts per-component loss dicts.  :class:`ReconstructionLossHelper`
wraps k-space consistency loss and deep-supervision loss with fall-through
defaults for the no-FFT-transformer case.


Infrastructure: Validation
===========================

DatasetComplianceChecker (``infrastructure.validation.dataset_compliance``)
---------------------------------------------------------------------------

Checks filesystem layout, file counts, contrast subdirectories, and
variant status.  Returns :class:`DatasetComplianceReport` with a tiered
:class:`ComplianceSeverity` (``OK`` < ``INFO`` < ``WARNING`` < ``ERROR``
< ``CRITICAL``).  Serialises to JSON via ``.to_dict()``.


Data Layer: Image Volume Utilities
====================================

``data.datasets.utils.image_volume_utils`` provides:

* ``load_grayscale_png`` — normalised float32 ``(1, H, W)`` tensor.
* ``load_grayscale_stack`` — stacks slice PNGs to ``(C, H, W, D)``.
* ``list_png_files`` — sorted glob.
* ``group_slices_by_prefix`` — groups slices into volumes by filename prefix.
* ``find_common_png_stems`` — intersection of PNG stems across directories.

These helpers centralise the logic shared across TRELLIS dataset variants.


Infrastructure: Reporting Tables
==================================

CertificateTable (``infrastructure.reporting.tables.certificate_table``)
------------------------------------------------------------------------

Builds a per-certificate summary table from a ``ValidationBadge`` JSON
payload (``upstream_certificates`` key).  ``render_markdown`` and
``render_latex`` produce ``.md`` / ``.tex`` side-by-side outputs.
``write_certificate_table(badge_payload, out_dir)`` writes both files.


Diffusion-Transformer Backbones (k-space adapted)
=================================================

Four ``backbone_type`` values reachable from the k-space cold-diffusion stack:
``dit``, ``uvit`` (alias ``u_vit``), ``hat`` and ``diffit``. Unlike the rest of
this page they are *backbones*, not whole models — ``backbone_type`` selects the
entire learnable network inside ``KSpaceColdDiffusionGenerator``, whose other
children (``kspace_process``, ``dc_layer``, ``sense_projector``) hold no
parameters at all.

.. code-block:: yaml

   model:
     model_type: kspace_cold_diffusion
     in_channels: 8            # 4 coils, interleaved real/imag
     out_channels: 8
     model_kwargs:
       backbone_type: dit
       base_channels: 384      # the trunk's embedding width
       patch_size: 8
       depth: 12
       heads: 6                # omit and it is derived from base_channels
       force_pure_kspace: true
       attention_type: none    # required — see below

Both domains, one stem
----------------------

These architectures were published on natural images: one tensor, one domain, a
patch is a picture. MRI arrives k-space-native, and a patch of k-space is a
frequency band whose neighbours share no spatial locality — so a stem that
patch-embeds whichever tensor it is handed means two different things depending
on ``force_pure_kspace``, with nothing recording which.

``DomainStem`` (``models/generators/diffusion_transformer_stem.py``) is the
single owner of that difference for all four. It embeds **both** views of the
field — deriving the second through ``ifft2c``/``fft2c`` from the declared
domain — and sums them, with a separate projection per view because a k-space
patch is dominated by its distance from the centre and an image patch by local
structure. ``DomainHead`` returns the prediction in the domain the caller passed
in. ``feature_domain`` is therefore derived from ``force_pure_kspace`` and
load-bearing, not recorded and ignored.

The trunks that consume those tokens are the published architectures.

What distinguishes each
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Name
     - Mechanism
     - Notes
   * - ``dit``
     - adaLN-Zero conditioning
     - Patch tokens, plain pre-norm transformer. Zero-initialised gates make
       every block the identity at init.
   * - ``uvit``
     - Long skip connections
     - Encoder-half outputs concatenated onto the decoder half and projected
       back down. Time rides as a token, not through adaLN. **Depth must be
       odd** so the halves pair around one middle block.
   * - ``hat``
     - Window attention + conv channel attention
     - Never patchifies — full resolution throughout, which suits k-space where
       a patch stride discards the outer frequencies. ``window_size`` must tile
       the field. ``cab_weight`` is a small correction, not an equal partner.
   * - ``diffit``
     - Time-dependent self-attention
     - The timestep enters q, k and v additively rather than gating the block
       output, so *what* a token attends to changes with the step. Paired with
       ``dit``, which differs only in that mechanism.

Two consequences worth knowing before declaring one
---------------------------------------------------

**They have no pluggable attention seam.** They are attention end to end, but
none implements this framework's ``attention_type`` block dispatch — only
``complex_unet`` does. Declaring anything but ``attention_type: none`` therefore
**raises at build**, rather than being validated and dropped.

**The output head is not zero-initialised**, though DiT's published one is. A
forward whose output does not move when its input does is the DC-blob facade the
Tier-2 probe rejects, and the probe cannot tell that apart from "untrained". The
block-level adaLN-Zero gates — where the paper's stability argument actually
lives — are kept.

New backbones register in ``models/generators/backbone_builders.py`` rather than
extending the generator's ``if/elif`` chain, which is already a baselined
dispatch offender.


References
==========

1. Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical
   Image Segmentation." MICCAI, 2015.

2. Ho, J., et al. "Denoising Diffusion Probabilistic Models."
   NeurIPS, 2020.

3. Song, Y., et al. "Score-Based Generative Modeling through Stochastic
   Differential Equations." ICLR, 2021.

4. Song, Y., et al. "Consistency Models." ICML, 2023.

5. Van den Oord, A., et al. "Neural Discrete Representation Learning."
   NeurIPS, 2017.

6. Sitzmann, V., et al. "Implicit Neural Representations with Periodic
   Activation Functions." NeurIPS, 2020.

7. Li, Z., et al. "Fourier Neural Operator for Parametric Partial
   Differential Equations." ICLR, 2021.

8. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer
   using Shifted Windows." ICCV, 2021.

9. Zhu, A., et al. "Vision Mamba: Efficient Visual Representation Learning
   with Bidirectional State Space Model." ICML, 2024.

10. Sriram, A., et al. "End-to-End Variational Networks for Accelerated
    MRI Reconstruction." MICCAI, 2020.

11. Chen, X., et al. "Simple Baselines for Image Restoration (NAFNet)."
    ECCV, 2022.

12. Gal, Y. and Ghahramani, Z. "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning." ICML, 2016.
    (Bayesian U-Net)

13. Kirchler, M., et al. "CPT-4DMR: Compact and Periodic Temporal Model
    for Free-Breathing Cardiac MRI." MICCAI, 2024.

14. Kiranyaz, S., et al. "1D Convolutional Neural Networks and Applications."
    Mechanical Systems and Signal Processing, 2021. (AFTNet axial attention)

15. Kirkpatrick, J., et al. "Overcoming Catastrophic Forgetting in Neural
    Networks." PNAS, 2017. (Continual Learning, EWC)

16. Peebles, W. and Xie, S. "Scalable Diffusion Models with Transformers
    (DiT)." ICCV, 2023.

17. Bao, F., et al. "All are Worth Words: A ViT Backbone for Diffusion
    Models (U-ViT)." CVPR, 2023.

18. Chen, X., et al. "Activating More Pixels in Image Super-Resolution
    Transformer (HAT)." CVPR, 2023.

19. Hatamizadeh, A., et al. "DiffiT: Diffusion Vision Transformers for
    Image Generation." ECCV, 2024.
