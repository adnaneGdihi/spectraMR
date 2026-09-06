.. _tutorial_diffusion_training:

==========================================
Tutorial 3: Cold Diffusion Training
==========================================

This tutorial trains a **k-space Cold Diffusion** model — the same architecture
used in ``experiment_11_timestep_accelerated_cold_diffusion``. Unlike stochastic
diffusion (DDPM/DDIM), Cold Diffusion uses **deterministic, physics-grounded
degradations** (k-space masking and Rician noise) instead of Gaussian noise.

**What You'll Learn:**

- Configuring a Cold Diffusion experiment from scratch
- Understanding the ``DiffusionTrainingStrategy`` and its curriculum scheduler
- Reading k-space-specific training metrics (``kspace_error``, ``phase_mse``)
- Tuning the timestep importance sampler for faster convergence
- Running validation in image-domain while training in k-space

**Prerequisites:**

- Completed :doc:`tutorial_01_basic_reconstruction`
- Basic familiarity with k-space and the FFT
- GPU with ≥ 16 GB VRAM (the k-space cold diffusion generator is memory-intensive)

.. contents:: Tutorial Steps
   :local:
   :depth: 2

==================
Background
==================

Why Cold Diffusion for MRI?
----------------------------

Standard DDPM adds **random Gaussian noise** at each timestep. MRI artefacts
from undersampling are not random — they are **deterministic aliases** created
by a known k-space mask. Cold Diffusion replaces the noise process with
physically-motivated, invertible degradations:

.. list-table:: DDPM vs Cold Diffusion for MRI
   :header-rows: 1
   :widths: 20 40 40

   * - Property
     - DDPM / DDIM
     - k-Space Cold Diffusion
   * - Degradation
     - Gaussian noise :math:`\epsilon \sim \mathcal{N}(0, I)`
     - Structured k-space masking
   * - Inversion
     - Stochastic denoising score
     - Deterministic restoration
   * - Physics
     - No MRI prior
     - SENSE-aware, Hermitian symmetry
   * - Sampling steps
     - 1000 (DDPM) / 50 (DDIM)
     - 28 — one per acceleration rung
   * - Inference speed
     - Slow
     - Fast (28-step cold sampler)

The training objective at timestep :math:`t` is:

.. math::

   \mathcal{L}_{cold} = \mathbb{E}_{t \sim p(t)}\left[ \| \hat{x}_0 - x_0 \|^2 \right]

where :math:`\hat{x}_0 = G(x_t, t)` is the direct prediction of the clean
k-space (``prediction_type: sample``), and :math:`x_t` is the partially-masked
k-space at step :math:`t`.

==================
Step 1: Create Configuration
==================

The configuration closely mirrors the experiment_11 YAML. Key choices are
annotated.

.. code-block:: yaml

   # experiments/tutorials/tutorial_03_cold_diffusion.yaml
   config_version: '1.0'

   metadata:
     name: "Tutorial 03 - k-Space Cold Diffusion"
     description: "Cold Diffusion for MRI k-space reconstruction"
     tags:
       type: reconstruction
       paradigm: diffusion
       domain: kspace
     version: '1.0'

   model:
     model_type: kspace_cold_diffusion  # Registered generator
     in_channels: 8                     # 2 × 4 virtual coils (SVD coil compression)
     out_channels: 8
     spatial_dims: 2
     model_kwargs:
       base_channels: 64
       num_res_blocks: 4
       timesteps: 28                    # one timestep per acceleration rung
       time_embedding_dim: 256
       use_complex_conv: true           # ComplexConv2d layers throughout
       activation: complex
       force_pure_kspace: true
       backbone_type: unet
       attention_type: none    # PureKSpaceUNet has no attention seam; the
                               # constructor default 'self' would be dropped
       process_type: cold_diffusion
       num_contrasts: 4                 # M4Raw has 4 contrasts

   data:
     dataset_type: kspace
     volume_format: h5

     loader:
       batch_size: 2                      # Reduce to 1 if <16 GB VRAM
       num_workers: 4
     coils:
       processing_mode: svd         # SVD virtual coil compression
     split:
       validation_fraction: 0.1
     domain:
       target_channels: 8
     processing:
       enable_kspace_normalization: true
       normalization_type: robust_percentile
       normalization_kwargs:
         percentile: 0.99
         clamp: true
         num_virtual_coils: 4            # → in_channels = 2×4 = 8
     sampling:
       patch_size: [256, 256, 1]
       samples_per_volume: 8
       queue_length: 100
     source:
       root: databases/m4raw/data
       index_path: data/manifests/m4raw_train.json
       validation_index_path: data/manifests/m4raw_multicoil_val.json
   training:
     training_mode: diffusion
     strategy_class: spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy
     input_domain: kspace
     output_domain: kspace
     epochs: 200                        # 500 for full training (exp_11)
     max_iterations: 300000             # 700k for full training
     device: cuda
     output_dir: experiments/results/tutorial_03_cold_diffusion
     enable_gradient_checkpointing: true

     # ── Diffusion Schedule ──────────────────────────────────────────
     diffusion:
       timesteps: 28                   # one timestep per acceleration rung
       noise_schedule: linear
       sampler: cold_mri               # Physics-aware cold sampler
       sampling_steps: 28              # full reverse process (one step per rung)
       cond_drop_prob: 0.1             # Classifier-free guidance dropout
       guidance_scale: 1.0
       type: cold
       prediction_type: sample         # Predict x_0 directly (not noise)
       degradation: kspace_mask        # Structured masking degradation
       enforce_output_range: true

     # ── Curriculum ──────────────────────────────────────────────────
     # Starts training on easy timesteps (small masks), gradually hard
     curriculum_start_timestep: 4
     curriculum_ramp_rate: 0.005

     # ── Importance Sampling ─────────────────────────────────────────
     # Focuses training budget on hard timesteps (higher loss variance)
     timestep_sampling_strategy: importance
     identity_collapse_threshold: 0.0005  # Stop if prediction collapses
     early_training_steps: 1620

   undersampling:
     # The cold-diffusion cascade must NEST: every rung's kept lines must be a
     # subset of the rung below it. `random` ranks k-space once and truncates by
     # budget, so it nests by construction; `equispaced` cannot. One rung per
     # timestep means no timestep leaves the mask unchanged. `spectramr audit`
     # checks both (schedule.nesting_leakfree, schedule.no_inert_steps).
     acceleration_type: random
     base_acceleration: 2.0
     max_acceleration: 32.0
     center_fraction: 0.08
     min_center_fraction: 0.02
     acceleration_range:
     - 2.0
     - 2.226
     - 2.485
     - 2.753
     - 3.048
     - 3.413
     - 3.765
     - 4.197
     - 4.655
     - 5.224
     - 5.818
     - 6.4
     - 7.111
     - 8.0
     - 8.828
     - 9.846
     - 10.667
     - 11.636
     - 12.8
     - 14.222
     - 16.0
     - 18.286
     - 19.692
     - 21.333
     - 23.273
     - 25.6
     - 28.444
     - 32.0
     mask_direction: phase
     schedule_type: step
     schedule_steps: 28
     enable_dynamic_mask: true         # Vary mask across training steps
     enforce_nested: true
     mask_seed: 42

   physics:
     kspace:
       enable_kspace_recon: true
       enforce_hermitian_symmetry: false  # Complex-valued multi-contrast
     data_consistency:
       enabled: true

   optimization:
     precision:
       enabled: false        # AMP off: complex FFT can NaN under FP16
     lr_scheduler_strategy: cosine_annealing_warm_restarts
     scheduler:
       warmup_steps: 1000
       T_0: 20000
       T_mult: 2
       eta_min: 1.0e-6

     optimizer:
       type: adamw
       learning_rate: 2.0e-6             # Very low LR — diffusion is sensitive
       weight_decay: 0.0001
       beta1: 0.9
       beta2: 0.999
     gradient:
       accumulation_steps: 4   # Effective batch = 2×4 = 8
       clip:
         method: norm
         value: 1.0
   losses:
     image_losses:
       - name: mse
         weight: 1.0
         enabled: true
     kspace_losses: []
     complex_losses: []

     policy:
       output_domain: kspace   # the model emits k-space; validation converts (below)
   metrics:
     domain: kspace
     compute_kspace_error: true        # RMSE in k-space
     compute_phase_mse: true           # Phase fidelity
     compute_psnr: false               # Computed at validation (image domain)
     enable_tracking: true
     track_best_metric: true
     best_metric_name: lpips
     best_metric_mode: min

   validation:
     enabled: true

     schedule:
       interval_steps: 15000             # Validate every 15k iterations
     scoring:
       compute: [lpips, psnr]
       domain: image
       output_transform: ifft_magnitude  # k-space → magnitude image for PSNR/SSIM
       enable_image_metrics: true
   ema:
     enabled: true
     decay: 0.9999                    # EMA essential for stable diffusion sampling
     update_frequency: 1

   checkpoint:
     enabled: true
     save_interval: 10000
     keep_last_n: 5
     keep_best_n: 3

   early_stopping:
     enabled: true
     patience: 20000
     metric: val_psnr
     mode: max
   run:
     seed: 42

   logging:
     identity:
       experiment: tutorial_03_cold_diffusion
     intervals:
       log: 100
       save: 5
     tracking:
       enable_tensorboard: true

==================
Step 2: Understand the Key Concepts
==================

Timestep Curriculum
-------------------

The ``curriculum_start_timestep`` parameter implements a paced training strategy:

.. code-block:: text

   Iteration 0     → Only easy timesteps t ∈ [0, 4]
   Iteration 200   → Unlocked t ∈ [0, 5]
   ...
   Iteration 4800  → Full range t ∈ [0, 28]

The ramp rate is ``curriculum_ramp_rate = 0.005`` timesteps/iteration, so each
additional timestep unlocks after 200 iterations.

**Why this helps:** Early in training, the model learns to denoise only slightly
corrupted k-space (small mask ratio). As it becomes confident, harder
degradations (large masks, 32× acceleration) are gradually introduced.

Importance Timestep Sampling
------------------------------

``timestep_sampling_strategy: importance`` overrides uniform sampling with a
loss-weighted distribution. Timesteps with high loss variance receive more
training budget:

.. math::

   p(t) \propto \sqrt{ \mathbb{E}\left[ \left( L(t) - \bar{L} \right)^2 \right] }

This typically doubles convergence speed for the hardest timesteps (very
coarsely sampled k-space at high acceleration).

SVD Coil Compression
--------------------

With ``coil_processing_mode: svd`` and ``num_virtual_coils: 4``, the
multi-coil k-space is compressed to 4 virtual coils via truncated SVD. Each
coil has 2 channels (real + imaginary), so:

.. code-block:: text

   in_channels = 2 × num_virtual_coils = 2 × 4 = 8

This is verified at startup by the
:class:`~spectramr.infrastructure.validation.config_health_checker.ConfigHealthChecker`.

==================
Step 3: Run Training
==================

.. code-block:: bash

   # Validate config first (no GPU needed)
   spectramr train --config experiments/tutorials/tutorial_03_cold_diffusion.yaml --dry-run

   # Start training
   spectramr train --config experiments/tutorials/tutorial_03_cold_diffusion.yaml

**Expected output (first 1000 iterations):**

.. code-block:: text

   [Step    100] g_total_loss=0.0431  mse=0.0431  kspace_error=0.182  phase_mse=0.031
   [Step    200] g_total_loss=0.0318  mse=0.0318  kspace_error=0.141  phase_mse=0.021
   ...
   [Step   1000] g_total_loss=0.0091  kspace_error=0.088  phase_mse=0.009
   [Val  Step 15000] val_psnr=26.4 dB  val_lpips=0.241  (ifft_magnitude domain)

.. note::

   ``mse`` here is computed on k-space, not image pixels. A loss of ``0.01``
   in normalised k-space corresponds to roughly 25–28 dB PSNR in image space
   (validated every ``eval_interval=15000`` steps using ``ifft_magnitude``
   transform).

==================
Step 4: Monitor Diffusion-Specific Metrics
==================

TensorBoard will show:

.. list-table:: Key Training Metrics
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Interpretation
   * - ``g_total_loss``
     - Weighted sum of image_losses (MSE on predicted x_0)
   * - ``kspace_error``
     - RMSE between predicted and ground-truth k-space (lower is better)
   * - ``phase_mse``
     - MSE on the phase of the k-space (phase fidelity)
   * - ``val_psnr``
     - Evaluated after IFFT; tracks reconstruction quality in image space
   * - ``val_lpips``
     - Perceptual quality; used as ``best_metric`` for checkpointing

**Healthy convergence looks like:**

- ``g_total_loss`` decreasing monotonically for first 10k steps
- ``kspace_error`` below 0.05 after 50k iterations
- ``val_psnr`` steadily climbing; plateau around 200k iterations

==================
Step 5: Sample From the Trained Model
==================

After training, run the cold diffusion reverse process:

.. code-block:: bash

   spectramr infer \
       --checkpoint experiments/results/tutorial_03_cold_diffusion/checkpoints/best_ema.pt \
       --input databases/m4raw/data/multicoil_val \
       --output experiments/results/tutorial_03_cold_diffusion/inference

The sampler's step count is not a CLI flag — it is read from the config's
diffusion block, so change it there (or in a copy of the YAML) rather than at
the command line.

**Expected metrics (200 epochs, M4Raw 4× acceleration):**

.. code-block:: text

   PSNR:  28.1 ± 1.9 dB
   SSIM:  0.841 ± 0.032
   LPIPS: 0.187 ± 0.041

==================
Troubleshooting
==================

**Loss NaN after 500 steps**

- Complex FFT + FP16 can produce NaN. Ensure ``enable_mixed_precision: false``
  (already set above).

**Identity collapse (``loss → 0`` immediately)**

- Check ``identity_collapse_threshold: 0.0005``. If triggered, training stops.
- Reduce ``learning_rate`` to ``1e-6`` and restart.

**OOM with batch_size=2**

.. code-block:: yaml

   data:
     batch_size: 1
   optimization:
     gradient_accumulation_steps: 8   # Keep effective batch = 8

**Slow validation (eval_interval too small)**

.. code-block:: yaml

   validation:
     eval_interval: 30000  # Double the interval

==================
Summary
==================

✅ ``kspace_cold_diffusion`` generator handles complex-valued multi-coil k-space

✅ ``prediction_type: sample`` predicts :math:`x_0` directly at each timestep

✅ Curriculum + importance sampling accelerate convergence on hard timesteps

✅ ``ifft_magnitude`` output transform allows image-domain PSNR validation
   while training remains entirely in k-space

✅ EMA (``ema.decay: 0.9999``) is essential — always load EMA checkpoint for inference

**Next:** :doc:`tutorial_04_physics_constraints` — Add data consistency and Bloch physics.
