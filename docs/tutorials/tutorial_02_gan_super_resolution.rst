.. _tutorial_gan_super_resolution:

==========================================
Tutorial 2: GAN Super-Resolution Training
==========================================

This tutorial trains a GAN for 4× MRI super-resolution, producing sharper,
perceptually richer images than a supervised U-Net alone.

**What You'll Learn:**

- Configuring a GAN training strategy
- Balancing adversarial, pixel, and perceptual losses using the ``losses:`` schema
- Reading discriminator-specific training metrics
- Common GAN instability fixes

**Prerequisites:**

- Completed :doc:`tutorial_01_basic_reconstruction`
- GPU with ≥ 12 GB VRAM

.. contents:: Tutorial Steps
   :local:
   :depth: 2

==================
Background
==================

Standard supervised training minimises a pixel-level loss (L1/MSE) that
promotes PSNR but tends to over-smooth textures. A GAN adds an adversarial
discriminator *D* that learns to distinguish real fully-sampled images from
generator outputs, forcing *G* to produce realistic high-frequency detail.

The total generator objective is:

.. math::

   \mathcal{L}_G = \lambda_{adv} \mathcal{L}_{adv}(G)
                 + \lambda_1 \| x - G(y) \|_1
                 + \lambda_p \mathcal{L}_{perceptual}

The discriminator is updated separately with its own loss.

==================
Step 1: Create Configuration
==================

.. code-block:: bash

   mkdir -p experiments/tutorials
   cp experiments/active/dummy_gan.yaml \
      experiments/tutorials/tutorial_02_gan_sr.yaml

Then edit the file:

.. code-block:: yaml

   # Tutorial 02: GAN Super-Resolution
   config_version: '1.0'

   metadata:
     name: "Tutorial 02 - GAN Super-Resolution"
     description: "4× accelerated MRI super-resolution with adversarial training"
     tags: ["tutorial", "gan", "super-resolution"]
     version: '1.0'

   model:
     model_type: standard_unet
     in_channels: 1      # Magnitude input
     out_channels: 1
     model_kwargs:
       features: [64, 128, 256, 512]
     discriminator_component:
       name: patch_gan
       kwargs:
         in_channels: 1
         ndf: 64
         n_layers: 3

   training:
     output_dir: experiments/results/tutorial_02_gan_sr
     training_mode: gan
     strategy_class: spectramr.infrastructure.training.strategies.gan.GANTrainingStrategy
     epochs: 100

   data:
     dataset_type: image

     loader:
       batch_size: 4
       num_workers: 4
     source:
       root: databases/fastmri/datasets
   optimization:
     lr_scheduler_strategy: none

     optimizer:
       type: adam
       learning_rate: 0.0002
       weight_decay: 0.0
       kwargs:
         betas: [0.5, 0.999]   # Standard GAN betas
   losses:
     image_losses:
       - name: l1
         weight: 10.0         # Strong pixel anchor prevents mode collapse
         enabled: true
       - name: perceptual
         weight: 1.0
         enabled: true
     kspace_losses: []
     complex_losses: []
     gan:
       enable_adversarial: true
       lambda_adv: 1.0
       gan_loss_type: lsgan   # LSGAN more stable than vanilla BCE
       disc_updates: 1

     policy:
       output_domain: image
   validation:

     schedule:
       interval_steps: 500
     scoring:
       compute: [psnr, ssim, lpips]
   logging:
     intervals:
       log: 50
     tracking:
       enable_tensorboard: true
   run:
     seed: 42

**Key Choices Explained:**

- ``betas: [0.5, 0.999]`` — Lower β₁ dampens gradient oscillations during GAN training.
- ``l1 weight: 10.0`` — High pixel anchor keeps the generator from collapsing to a noise pattern.
- ``lsgan`` — Least-squares GAN loss provides smoother gradients than BCE near saturation.

==================
Step 2: Train
==================

.. code-block:: bash

   spectramr train --config experiments/tutorials/tutorial_02_gan_sr.yaml

**Expected Training Output:**

.. code-block:: text

   [Step  100] g_total_loss=2.31  l1=0.183  perceptual=0.044  g_adv=0.512
   [Step  100] d_total_loss=0.421  d_real=0.198  d_fake=0.223
   [Step  500] Val: PSNR=29.2 dB | SSIM=0.841 | LPIPS=0.124
   ...
   [Step 5000] Val: PSNR=30.1 dB | SSIM=0.869 | LPIPS=0.091

.. note::

   PSNR and SSIM may be slightly *lower* than the supervised U-Net, but LPIPS
   (lower is better) should improve significantly — indicating perceptually
   sharper outputs.

==================
Step 3: Monitor GAN Health
==================

A well-behaved GAN training shows these patterns in TensorBoard:

**Healthy signs:**

- ``d_total_loss`` ≈ 0.3–0.7 (discriminator is neither perfect nor failing)
- ``g_adv_loss`` decreases steadily
- ``l1`` loss decreases over time
- No loss spikes or NaN values

**Warning signs:**

- ``d_total_loss`` → 0.0 : Discriminator always wins → generator collapses
- ``g_adv_loss`` explodes : Learning rate too high
- Loss NaN : Reduce ``lr`` or enable gradient clipping

==================
Step 4: Troubleshooting GAN Instability
==================

**Problem: Mode Collapse (all outputs look similar)**

.. code-block:: yaml

   losses:
     image_losses:
       - name: l1
         weight: 20.0    # Double the pixel anchor

**Problem: Discriminator too strong (G never improves)**

.. code-block:: yaml

   optimization:
     learning_rate: 0.0001  # Halve generator lr

**Problem: Checkerboard artifacts**

Switch discriminator architecture:

.. code-block:: yaml

   discriminator:
     discriminator_type: patch_gan
     model_kwargs:
       ndf: 64
       n_layers: 4      # Increase receptive field

==================
Step 5: Compare to Supervised Baseline
==================

.. code-block:: python

   # Load both checkpoints and evaluate on test set
   import json

   with open('experiments/tutorials/tutorial_01_basic_unet/inference/metrics.json') as f:
       unet_metrics = json.load(f)

   with open('experiments/results/tutorial_02_gan_sr/inference/metrics.json') as f:
       gan_metrics = json.load(f)

   print("Model       | PSNR (dB) | SSIM  | LPIPS")
   print("------------|-----------|-------|------")
   print(f"U-Net (L1)  | {unet_metrics['psnr_mean']:.2f}     | "
         f"{unet_metrics['ssim_mean']:.3f} | {unet_metrics.get('lpips_mean', 'N/A')}")
   print(f"GAN         | {gan_metrics['psnr_mean']:.2f}     | "
         f"{gan_metrics['ssim_mean']:.3f} | {gan_metrics['lpips_mean']:.3f}")

==================
Next Steps
==================

- :doc:`tutorial_03_diffusion_training` — Generative diffusion for MRI
- :doc:`tutorial_04_physics_constraints` — Add data consistency to GAN training
- :doc:`tutorial_05_custom_loss` — Write your own loss and register it

==================
Summary
==================

✅ GAN strategy selected via ``training_mode: gan``

✅ Adversarial, pixel, and perceptual losses composed via ``losses.image_losses`` list

✅ LSGAN chosen for training stability

✅ Generator/discriminator learning rates independently tunable
