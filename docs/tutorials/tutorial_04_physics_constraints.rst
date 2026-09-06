.. _tutorial_physics_constraints:

==========================================
Tutorial 4: Physics-Informed Constraints
==========================================

This tutorial shows how to add **data consistency** and **k-space losses** to any
reconstruction model, ensuring that predicted images do not contradict the
acquired measurements.

**What You'll Learn:**

- Enabling k-space data consistency layers
- Using ``kspace_losses`` alongside ``image_losses``
- Configuring the Cycle-Bloch physics strategy for quantitative synthesis
- Understanding the domain-aware loss routing system

**Prerequisites:**

- Completed :doc:`tutorial_01_basic_reconstruction`
- Basic understanding of k-space and the MRI forward model

.. contents:: Tutorial Steps
   :local:
   :depth: 2

==================
Step 1: Add Data Consistency to Reconstruction
==================

The simplest physics constraint is **hard data consistency**: after each
forward pass, replace the model's k-space predictions in measured locations
with the actual measurements.

.. code-block:: yaml

   # experiments/tutorials/tutorial_04a_physics_dc.yaml
   config_version: '1.0'

   metadata:
     name: "Tutorial 04a - Physics Data Consistency"
     version: '1.0'

   model:
     model_type: standard_unet
     in_channels: 2
     out_channels: 2

   training:
     training_mode: reconstruction
     output_dir: experiments/results/tutorial_04a_physics_dc
     strategy_class: spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy
     epochs: 50

   data:
     dataset_type: kspace
     volume_format: h5
     loader:
       batch_size: 4
     source:
       root: databases/fastmri/datasets
       index_path: data/manifests/fastmri_brain_multicoil_train.json

   # k-space data into an image-domain U-Net: bridge explicitly (NN#9, no silent rescue).
   adapters:
     pre_model:
       - name: ifft_kspace_to_image
       - name: complex_to_real_imag_interleave

   optimization:
     optimizer:
       type: adam
       learning_rate: 0.0001

   logging:
     identity:
       experiment: tutorial_04a_physics_dc
     intervals:
       log: 50

   physics:
     data_consistency:
       enabled: true
       method: hard       # Replace measured k-space exactly
       weight: 1.0        # Only used for 'soft' method
     kspace:
       enable_kspace_recon: true
       enforce_hermitian_symmetry: true

   undersampling:
     base_acceleration: 4.0
     center_fraction: 0.08

   losses:
     image_losses:
       - name: l1
         weight: 1.0
         enabled: true
       - name: ssim
         weight: 1.0
         enabled: true
     # NOTE: hard DC is enforced by the physics: block above (it replaces measured
     # k-space after the forward pass). The registered `data_consistency` LOSS is the
     # *soft* alternative — declaring both gives one invariant two owners.
     kspace_losses: []
     policy:
       output_domain: image
   run:
     seed: 42

**Expected Result:** +0.5–1.0 dB PSNR compared to Tutorial 01 baseline,
and significantly reduced aliasing artefacts near the k-space centre.

==================
Step 2: Add K-Space Frequency Losses
==================

Frequency-domain losses penalise errors in specific k-space bands. This is
particularly useful for preserving fine anatomical detail (high frequencies)
or low-contrast tissue boundaries (low frequencies).

.. code-block:: yaml

   losses:
     output_domain: image
     image_losses:
       - name: l1
         weight: 1.0
         enabled: true
     kspace_losses:
       - name: data_consistency
         weight: 1.0
         enabled: true
       - name: sobolev_kspace          # Penalises high-freq errors more
         weight: 0.1
         enabled: true
         kwargs:
           sobolev_order: 2           # s=2: quadratic frequency penalty
       - name: log_spectral            # Preserves spectral shape
         weight: 0.05
         enabled: true

The full k-space loss suite available in the framework:

.. list-table:: Available K-Space Losses
   :header-rows: 1
   :widths: 30 20 50

   * - Registry Name
     - Class
     - When to Use
   * - ``data_consistency``
     - ``DataConsistencyLoss``
     - Always; fundamental k-space fidelity constraint
   * - ``sobolev_kspace``
     - ``SobolevKSpaceLoss``
     - Sharper edges (penalises high-freq errors)
   * - ``log_spectral``
     - ``LogSpectralLoss``
     - Preserve overall spectral envelope shape
   * - ``focal_frequency``
     - ``InvertedFocalFrequencyLoss``
     - Adaptively focus on poorly-reconstructed frequencies
   * - ``parallel_imaging_kspace``
     - ``ParallelImagingKSpaceLoss``
     - Multi-coil SENSE-aware data consistency
   * - ``spectral_kspace``
     - ``SpectralKSpaceLoss``
     - Frequency-band–weighted loss with custom :math:`w(u,v)`

==================
Step 3: Cycle-Bloch Physics Strategy (Advanced)
==================

The :class:`CycleBlochStrategy` enforces physical self-consistency for
ultra-low-field (ULF) → high-field (HF) synthesis. It contains a full
differentiable Bloch simulation loop:

.. code-block:: text

   ULF Image
      │
      ▼
   Generator ──────────────────────────────► HF Estimate
      │                                          │
      │                                 ParameterEstimator
      │                                    (T1, T2, PD)
      │                                          │
      │                                  BlochSimulator
      │                                 (spin-echo model)
      │                                          │
      └──────────────── Cycle Loss (L1) ◄────────┘
                   L_cycle = ‖ULF_sim − ULF_real‖₁

The cycle loss ensures the synthesised HF image is *physically plausible*:
if you estimated tissue parameters from it and forward-simulated an MRI
acquisition, you would recover the original ULF measurement.

Configuration for Cycle-Bloch:

.. code-block:: yaml

   # experiments/tutorials/tutorial_04b_cycle_bloch.yaml
   config_version: '1.0'

   metadata:
     name: "Tutorial 04b - Cycle-Bloch ULF-to-HF"
     version: '1.0'

   model:
     model_type: standard_unet
     in_channels: 1        # ULF magnitude input
     out_channels: 1
     # The discriminator is a component OF the model, not a top-level block.
     discriminator_component:
       name: patch_gan
       kwargs:
         n_layers: 3
         ndf: 64

   data:
     dataset_type: kspace
     volume_format: h5
     loader:
       batch_size: 2
     coils:
       processing_mode: rss
     domain:
       target_channels: 1
     source:
       root: databases/m4raw/data
       index_path: data/manifests/m4raw_train.json

   adapters:
     pre_model:
       - name: ifft_kspace_to_image
       - name: magnitude_from_complex

   training:
     training_mode: cycle_bloch
     output_dir: experiments/results/tutorial_04b_cycle_bloch
     strategy_class: spectramr.infrastructure.training.strategies.cycle_bloch_strategy.CycleBlochStrategy
     epochs: 100

   physics:
     bloch:
       enabled: true
       ulf_tr: 500.0    # ms – repetition time for ULF sequence
       ulf_te: 14.0     # ms – echo time for ULF sequence

   losses:
     image_losses:
       - name: l1
         weight: 1.0
         enabled: true
     gan:
       enable_adversarial: true   # required whenever a discriminator is declared
       lambda_adv: 1.0
       gan_loss_type: lsgan
       disc_updates: 1
       lambda_cycle_bloch: 10.0   # Bloch cycle loss weight
       lambda_cycle_adv: 1.0      # GAN adversarial weight
     kspace_losses: []
     complex_losses: []

     policy:
       output_domain: image
   optimization:
     optimizer:
       type: adam
       learning_rate: 0.0002
       kwargs:
         betas: [0.5, 0.999]
   run:
     seed: 42

   logging:
     identity:
       experiment: tutorial_04b_cycle_bloch
     intervals:
       log: 50

**Loss Terms (auto-logged to TensorBoard):**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Formula
     - Description
   * - ``loss_g``
     - :math:`\lambda_{adv}\mathcal{L}_{adv} + \lambda_{cyc}\mathcal{L}_{cyc}`
     - Total generator loss
   * - ``loss_cycle``
     - :math:`\|ULF_{sim} - ULF_{real}\|_1`
     - Bloch cycle-consistency fidelity
   * - ``loss_d``
     - :math:`\frac{1}{2}(\mathcal{L}_{D,real} + \mathcal{L}_{D,fake})`
     - Discriminator loss

==================
Step 4: Complex-Valued Losses (K-Space Domain Models)
==================

When ``output_domain: complex_image``, losses are computed in k-space and
routed through the automatic ``DifferentiableFourierBridge``:

.. code-block:: yaml

   losses:
     output_domain: complex_image   # Enables Fourier bridge routing
     image_losses: []
     kspace_losses:
       - name: data_consistency
         weight: 1.0
         enabled: true
     complex_losses:
       - name: complex_l1
         weight: 1.0
         enabled: true
       - name: helmholtz_pde          # PINN: Helmholtz on sensitivity maps
         weight: 0.1
         enabled: true

.. note::

   The ``DifferentiableFourierBridge`` is inserted automatically by
   :class:`LossConfigSchema` when ``output_domain`` is ``complex_image`` or
   ``kspace``. You do not need to manually convert tensors.

==================
Step 5: Run & Verify
==================

.. code-block:: bash

   # Data consistency reconstruction
   spectramr train --config experiments/tutorials/tutorial_04a_physics_dc.yaml

   # Cycle-Bloch ULF↔HF synthesis
   spectramr train --config experiments/tutorials/tutorial_04b_cycle_bloch.yaml

After training, compare results:

.. code-block:: bash

   # Quick PSNR comparison
   python -c "
   import json
   for exp in ['tutorial_01_basic_unet', 'tutorial_04a']:
       m = json.load(open(f'experiments/tutorials/{exp}/inference/metrics.json'))
       print(f'{exp}: PSNR={m[\"psnr_mean\"]:.2f} dB  SSIM={m[\"ssim_mean\"]:.3f}')
   "

==================
Summary
==================

✅ Hard data consistency enforced via ``physics.data_consistency.enabled: true``

✅ K-space frequency losses added under ``losses.kspace_losses``

✅ Cycle-Bloch strategy for physics-consistent ULF→HF synthesis

✅ Complex-valued domain routing handled by the ``DifferentiableFourierBridge``

**Next:** :doc:`tutorial_05_custom_loss` — Writing and registering a custom loss function.
