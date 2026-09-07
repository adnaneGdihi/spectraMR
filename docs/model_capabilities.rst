.. _model_capabilities:

=============================
Model Capabilities Reference
=============================

This reference provides a comprehensive matrix of all models in the framework, showing their capabilities, supported tasks, training modes, and constraints.

.. contents:: Quick Navigation
   :local:
   :depth: 2

=======================
Model Capability Matrix
=======================

Core Reconstruction Models
===========================

.. list-table:: Reconstruction Models
   :header-rows: 1
   :widths: 20 25 25 30

   * - Model
     - Supported Tasks
     - Training Modes
     - Constraints & Requirements
   * - **UNet**
     - Image reconstruction, super-resolution
     - Reconstruction, supervised
     - None - general purpose
   * - **Enhanced Deep UNet**
     - High-quality reconstruction, complex anatomy
     - Reconstruction, supervised
     - Requires >16GB VRAM for full resolution
   * - **Mamba SSM** (Exp 30)
     - Long-range reconstruction, O(N) complexity
     - Reconstruction, supervised
     - Requires mamba-ssm package, GPU-only
   * - **Transformer UNet** (Exp 14)
     - Global context reconstruction
     - Reconstruction, supervised
     - High memory usage, requires batch size ≤4
   * - **Physics-Informed UNet**
     - Data-consistent reconstruction
     - Reconstruction + physics
     - Requires k-space data and sampling mask

---

Diffusion Models
================

.. list-table:: Diffusion Model Variants
   :header-rows: 1
   :widths: 20 25 25 30

   * - Model
     - Supported Tasks
     - Training Modes
     - Constraints & Requirements
   * - **DDPM** (Standard)
     - Unconditional/conditional generation
     - Diffusion (1000 steps)
     - Slow inference (50s/image), high quality
   * - **Cold Diffusion** (Exp 40, 11, 23)
     - Zero-shot reconstruction, various accelerations
     - Diffusion (k-space degradation)
     - Requires physics operators, k-space masks
   * - **Latent Diffusion** (Exp 17, 47)
     - Efficient generation, reconstruction
     - Diffusion (latent space)
     - Requires pre-trained VAE encoder/decoder
   * - **Consistency Model** (Exp 31)
     - 1-2 step generation, real-time
     - Distillation from DDPM
     - Requires teacher DDPM model first
   * - **Rectified Flow / InstaFlow** (Exp 48)
     - 1-step generation (after reflow)
     - Flow matching → reflow
     - Requires multiple training iterations
   * - **Chi-Square Diffusion** (Exp 13, 19, 28, 29)
     - Robust to outliers/artifacts
     - Diffusion (heavy-tailed noise)
     - Good for motion-corrupted data
   * - **Complex-Valued Latent Diffusion** (Exp 47)
     - Phase-preserving reconstruction
     - Diffusion (complex latent)
     - Requires 2-channel (real/imag) input
   * - **Latent ODE** (Exp 52)
     - Temporal/cine MRI, continuous dynamics
     - ODE integration in latent space
     - Requires temporal sequences (4D data)

---

Generative Adversarial Networks (GANs)
=======================================

.. list-table:: GAN Models
   :header-rows: 1
   :widths: 20 25 25 30

   * - Model
     - Supported Tasks
     - Training Modes
     - Constraints & Requirements
   * - **E2E GAN Super-Resolution**
     - 2×, 4× super-resolution
     - GAN (adversarial)
     - Requires paired low/high-res data
   * - **Composite GAN** (Exp 12)
     - Multi-objective reconstruction
     - GAN (adversarial + perceptual + style)
     - Requires VGG for perceptual loss
   * - **Hybrid Transformer-Laplacian GAN** (Exp 20)
     - Multi-scale generation
     - GAN (ViT-KAN generator)
     - High computational cost, requires 24GB+ VRAM
   * - **PatchGAN Discriminator**
     - Local texture discrimination
     - Used with any GAN generator
     - Works on image patches (70×70, 286×286)
   * - **Multi-Scale Discriminator**
     - Multi-resolution discrimination
     - Used with any GAN generator
     - 3 discriminators at different scales

---

Variational Autoencoders (VAEs)
================================

.. list-table:: VAE Models
   :header-rows: 1
   :widths: 20 25 25 30

   * - Model
     - Supported Tasks
     - Training Modes
     - Constraints & Requirements
   * - **E2E VAE Multi-Contrast**
     - Multi-contrast generation (T1, T2, FLAIR)
     - VAE (ELBO optimization)
     - Requires multi-contrast training data
   * - **Wavelet-Fourier KAN VAE** (Exp 24)
     - Multi-domain encoding, disentanglement
     - VAE (multi-domain)
     - Complex architecture, slow training
   * - **β-VAE**
     - Disentangled representation learning
     - VAE (weighted KL)
     - Set β ∈ [1, 10] for disentanglement vs quality trade-off
   * - **VQ-VAE**
     - Discrete latent codes
     - VAE (vector quantization)
     - Requires codebook size tuning

---

Specialized Models
==================

.. list-table:: Specialized Architectures
   :header-rows: 1
   :widths: 20 25 25 30

   * - Model
     - Supported Tasks
     - Training Modes
     - Constraints & Requirements
   * - **NeSVoR** (Exp 69)
     - Fetal MRI, extreme motion correction
     - Implicit neural representation
     - Requires 3D volume from 2D slices, GPU-intensive
   * - **PIMN - Physics-Informed Motion** (Exp 41)
     - Motion-compensated reconstruction
     - Reconstruction + motion estimation
     - Requires k-space data with motion artifacts
   * - **Bayesian KAN** (Exp 10)
     - Uncertainty quantification
     - Ensemble training (Bayesian)
     - Requires multiple forward passes (10-20)
   * - **TRELLIS** (Exp 26)
     - 3D volumetric generation with Gaussian splatting
     - Generative (atlas-based)
     - Requires pre-computed atlas
   * - **xDiffusion** (Exp 15, 27, 29)
     - Cross-domain transfer (2D↔3D, low↔high field)
     - Diffusion (cross-domain)
     - Requires paired cross-domain data
   * - **DINO Transfer** (Exp 21)
     - Transfer learning with minimal data
     - Fine-tuning (DINOv2/v3)
     - Requires pre-trained DINO weights
   * - **Multimodal Super-Resolution** (Exp 22)
     - Multi-contrast fusion for SR
     - Supervised (cross-attention)
     - Requires aligned multi-contrast volumes

---

=====================
Training Mode Details
=====================

Reconstruction
==============

**Purpose:** Direct supervised image reconstruction from undersampled k-space

**Requirements:**
- Fully-sampled reference images
- Paired undersampled/fully-sampled data
- Ground truth for loss computation

**Loss Functions:** L1, L2, SSIM, perceptual

**Typical Training Time:** 10-50 hours (50-100 epochs)

**Best For:** Standard MRI reconstruction tasks

---

Diffusion
=========

**Purpose:** Generative modeling via noise denoising process

**Requirements:**
- Clean training images (unconditional) OR
- Paired data (conditional)
- GPU with 16GB+ VRAM

**Variants:**
- Standard DDPM: 1000-step Gaussian noise
- Cold Diffusion: Structured degradation (k-space)
- Latent Diffusion: Diffusion in compressed space

**Typical Training Time:** 100-500 hours (500-1000 epochs)

**Best For:** High-quality generation, when inference speed is not critical

---

GAN (Adversarial)
=================

**Purpose:** Adversarial training for sharp, realistic images

**Requirements:**
- Paired or unpaired data
- Discriminator + generator training
- Careful hyperparameter tuning

**Training Challenges:**
- Mode collapse risk
- Training instability
- Requires balancing G/D updates

**Typical Training Time:** 50-200 hours (100-300 epochs)

**Best For:** Perceptual quality, super-resolution, when PSNR is secondary

---

VAE (Latent Encoding)
=====================

**Purpose:** Probabilistic latent representation learning

**Requirements:**
- Training images (no pairing needed)
- Latent dimension tuning

**Outputs:**
- Encoder: image → latent code
- Decoder: latent code → image
- Sampling: random latent → new image

**Typical Training Time:** 20-100 hours (50-200 epochs)

**Best For:** Data augmentation, missing modality synthesis, generation

---

Physics-Informed
================

**Purpose:** Enforce MRI physics constraints during training

**Requirements:**
- K-space data
- Sampling masks
- Coil sensitivity maps (multi-coil)

**Constraints Applied:**
- Data consistency: :math:`\mathcal{F}(\hat{x}) \odot M = y`
- Compressed sensing priors
- Bloch equation constraints (advanced)

**Typical Training Time:** 20-80 hours (similar to reconstruction)

**Best For:** High acceleration factors (8×, 16×), strict fidelity required

---

====================
Task Type Categories
====================

Reconstruction
==============

**Goal:** Recover fully-sampled image from undersampled k-space

**Acceleration Factors:**
- 2×: Easy (35-38 dB PSNR)
- 4×: Moderate (30-34 dB)
- 8×: Challenging (25-30 dB)
- 16×: Very hard (20-25 dB, research)

**Recommended Models:**
1. UNet (baseline)
2. Mamba (efficient, high-res)
3. Cold Diffusion (zero-shot)
4. Physics-Informed UNet (high acceleration)

---

Super-Resolution
================

**Goal:** Upsample low-resolution MRI to higher resolution

**Upsampling Factors:** 2×, 4×

**Recommended Models:**
1. GAN Super-Resolution (best perceptual)
2. Diffusion Models (best PSNR)
3. Transformer UNet (global context)

---

Generation
==========

**Goal:** Synthesize new MRI images or augment dataset

**Recommended Models:**
1. DDPM (unconditional, high quality)
2. Latent Diffusion (efficient)
3. VAE (controllable, fast)

---

Multi-Contrast Synthesis
=========================

**Goal:** Generate missing MRI contrasts (e.g., T1 → T2)

**Recommended Models:**
1. VAE Multi-Contrast
2. Multimodal Super-Resolution (Exp 22)
3. xDiffusion Cross-Domain

---

Motion Correction
=================

**Goal:** Reconstruct clean images from motion-corrupted acquisitions

**Recommended Models:**
1. PIMN (Exp 41) - explicit motion modeling
2. NeSVoR (Exp 69) - fetal MRI
3. Chi-Square Diffusion - robust to outliers

---

Temporal / Cine MRI
===================

**Goal:** Reconstruct dynamic sequences (cardiac, respiratory)

**Recommended Models:**
1. Latent ODE (Exp 52) - continuous dynamics
2. Standard UNet with 3D convolutions
3. Temporal attention models

---

====================
Constraint Types
====================

Data Consistency (DC)
=====================

**Formulation:**

.. math::

   x_{dc} = \arg\min_x \| \mathcal{F}(x) \odot M - y \|^2 + \lambda \| x - x_{nn} \|^2

**When to Use:**
- All physics-informed models
- High acceleration factors
- When k-space fidelity is critical

**Implementation:** CG-DC, Proximal DC, Unrolled iterations

---

Coil Sensitivity
================

**Formulation:**

.. math::

   y_c = M \odot \mathcal{F}(S_c \cdot x)

**When to Use:**
- Multi-coil MRI data
- Parallel imaging (SENSE, GRAPPA)

**Requirements:** Coil sensitivity maps (ESPIRiT, Walsh method)

---

Motion Parameters
=================

**Formulation:** Rigid or non-rigid motion fields

**When to Use:**
- Motion artifacts visible
- Pediatric/fetal MRI
- Long acquisition times

**Models:** PIMN (Exp 41), NeSVoR (Exp 69)

---

Compressed Sensing
==================

**Priors:** Sparsity in transform domain (wavelet, total variation)

**When to Use:**
- Extreme undersampling (10×, 20×)
- Combine with deep learning

**Regularization:** TV, :math:`\ell_1` wavelet

---

=====================
Quick Selection Guide
=====================

By Use Case
===========

**"I need real-time reconstruction for clinical deployment"**
→ Consistency Model (Exp 31) or Rectified Flow (Exp 48)
**Speed:** 50-100ms, **Quality:** 28-32 dB

**"I need maximum reconstruction quality, speed doesn't matter"**
→ DDPM or Latent Diffusion
**Speed:** 10-60s, **Quality:** 33-36 dB

**"I have limited training data (<1000 samples)"**
→ DINO Transfer (Exp 21) or Unsupervised (Exp 9)
**Data Required:** 500-1000 samples vs 5000+

**"I have severe motion artifacts"**
→ PIMN (Exp 41) or NeSVoR (Exp 69) or Chi-Square Diffusion
**Improvement:** 6+ dB over standard methods

**"I need uncertainty estimates"**
→ Bayesian KAN (Exp 10) or MC Dropout ensembles
**Output:** Prediction + confidence intervals

**"I need to synthesize missing T2 from T1"**
→ VAE Multi-Contrast or Multimodal SR (Exp 22)
**Quality:** 30-34 dB cross-modality

---

By Computational Budget
========================

**Low: 1 GPU, 8-12GB VRAM**
- Standard UNet
- Mamba (Exp 30)
- Small VAE

**Medium: 1 GPU, 16-24GB VRAM**
- Diffusion models (with gradient checkpointing)
- GANs
- Physics-Informed models

**High: 4+ GPUs, 32GB+ each**
- 3D volumetric models
- Large transformer models
- Ensemble methods

---

**See Also:**

- :doc:`getting_started` - Quick start guide
- :doc:`getting_started` - Install and first run
- :doc:`tutorials/index` - Step-by-step tutorials
