.. _tutorial_basic_reconstruction:

==========================================
Tutorial 1: Basic MRI Reconstruction
==========================================

This tutorial walks you through training a basic U-Net model for MRI reconstruction from undersampled k-space data.

**What You'll Learn:**

- Setting up a reconstruction experiment
- Understanding the configuration file
- Training a model from scratch
- Evaluating reconstruction quality
- Visualizing results

**Prerequisites:**

- Completed :doc:`../getting_started` (framework installed, dataset ready)
- Basic understanding of MRI physics
- ~2 hours of GPU time (RTX 3090 or equivalent)

.. contents:: Tutorial Steps
   :local:
   :depth: 2

==================
Step 1: Create Configuration
==================

Create a new configuration file for your experiment:

.. code-block:: bash

   cd /home/<user>/work/spectramr
   mkdir -p experiments/tutorials
   nano experiments/tutorials/tutorial_01_basic_unet.yaml

Configuration File
------------------

.. code-block:: yaml

   # Tutorial 01: Basic U-Net Reconstruction
   config_version: '1.0'

   metadata:
     name: "Tutorial 01 - Basic U-Net Reconstruction"
     description: "Baseline U-Net for 4× accelerated MRI reconstruction"
     tags: ["tutorial", "reconstruction", "unet", "baseline"]
     version: '1.0'

   model:
     model_type: standard_unet
     in_channels: 2  # Real + Imaginary k-space
     out_channels: 2
     model_kwargs:
       features: [64, 128, 256, 512]  # Encoder/decoder widths
       num_res_blocks: 2  # Residual blocks per level
       attention_levels: [2, 3]  # Add attention at 256 and 512 levels

   training:
     task: reconstruction
     input_domain: image
     output_domain: image
     strategy_class: spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy
     num_epochs: 50
     device: cuda
     output_dir: experiments/results/tutorial_01_basic_unet

   data:
     dataset_type: kspace
     datasets:
       - name: fastmri_train
         path: databases/fastmri/datasets/multicoil_train

     loader:
       batch_size: 4
       num_workers: 4  # CPU workers for data loading
     processing:
       enable_kspace_normalization: false
     source:
       root: databases/fastmri/datasets
       index_path: data/manifests/fastmri_brain_multicoil_train.json
       validation_index_path: data/manifests/fastmri_brain_multicoil_val.json
   physics:
     compressed_sensing:
       enabled: true
     data_consistency:
       enabled: true
     kspace:
       enable_kspace_recon: false
       enforce_hermitian_symmetry: true

   undersampling:
     base_acceleration: 4.0  # 4× undersampling
     center_fraction: 0.08  # 8% center fully sampled (calibration)
     acceleration_type: cartesian_vd

   optimization:
     lr_scheduler_strategy: cosine
     precision:
       enabled: true  # AMP (FP16) -- the live switch

     optimizer:
       type: adam
       learning_rate: 0.0001
       weight_decay: 0.0
   losses:
     image_losses:
       - name: l1
         weight: 1.0
         enabled: true
       - name: ssim
         weight: 1.0
         enabled: true
     kspace_losses: []
     complex_losses: []

     policy:
       output_domain: image
   logging:
     enable_wandb: false  # Set true to use Weights & Biases
     intervals:
       log: 50
       save: 5
     tracking:
       enable_tensorboard: true
   run:
     seed: 42

   adapters:
     pre_model:
       - name: ifft_kspace_to_image
       - name: complex_to_real_imag_interleave

**Key Configuration Choices:**

1. **Model**: ``standard_unet`` - Proven architecture for reconstruction
2. **Features**: ``[64, 128, 256, 512]`` - Standard U-Net depth
3. **Acceleration**: ``4.0`` - Moderate undersampling (good starting point)
4. **Loss**: L1 + SSIM - Balance pixel accuracy with perceptual quality
5. **Learning Rate**: ``1e-4`` - Safe default for Adam

==================
Step 2: Verify Dataset
==================

Before training, ensure your dataset is properly set up:

.. code-block:: bash

   # Check if manifest exists
   ls -lh data/manifests/fastmri_brain_multicoil_train.json

   # Verify dataset path
   ls /path/to/fastmri/brain_multicoil_train/ | head -5

If manifests don't exist, generate them:

.. code-block:: bash

   python scripts/data/regenerate_cluster_manifests.py \
       --data-base databases \
       --datasets fastmri

**Expected Output:**

.. code-block:: text

   Generated manifest: data/manifests/fastmri_brain_multicoil_train.json
   Total samples: 34,742
   Scans: 973
   Average slices per scan: 35.7

==================
Step 3: Train the Model
==================

Start training:

.. code-block:: bash

   spectramr train --config experiments/tutorials/tutorial_01_basic_unet.yaml

The run directory is not a command-line flag — it comes from
``training.output_dir`` in the YAML above
(``experiments/results/tutorial_01_basic_unet``). Override it for a one-off run
with ``-O training.output_dir=...`` if you need to.

**Expected Output (first few epochs):**

.. code-block:: text

   [Epoch 1/50] Step 100: loss=0.1523 | l1=0.0892 | ssim=0.0631 | lr=1.0e-04
   [Epoch 1/50] Step 200: loss=0.1245 | l1=0.0734 | ssim=0.0511 | lr=1.0e-04
   [Epoch 1/50] Val Metrics: PSNR=24.3 dB | SSIM=0.732

   [Epoch 5/50] Val Metrics: PSNR=28.1 dB | SSIM=0.823
   [Epoch 10/50] Val Metrics: PSNR=30.2 dB | SSIM=0.867
   [Epoch 20/50] Val Metrics: PSNR=31.8 dB | SSIM=0.891
   [Epoch 50/50] Val Metrics: PSNR=32.4 dB | SSIM=0.902

**Training Time:** ~2 hours on RTX 3090 (50 epochs, 34k samples)

Monitoring Training
-------------------

**TensorBoard (recommended):**

.. code-block:: bash

   # In a separate terminal
   tensorboard --logdir experiments/results/tutorial_01_basic_unet/logs

   # Open browser to: http://localhost:6006

**Console Monitoring:**

.. code-block:: bash

   # Watch training progress
   tail -f experiments/results/tutorial_01_basic_unet/logs/train.log

==================
Step 4: Run Inference
==================

Test the trained model on validation data:

.. code-block:: bash

   spectramr infer \
       --checkpoint experiments/results/tutorial_01_basic_unet/checkpoints/best.pt \
       --input databases/fastmri/datasets/multicoil_val \
       --output experiments/results/tutorial_01_basic_unet/inference

``--config`` is optional here: the run directory beside the checkpoint holds
``resolved_config.json``, which wins unless you pass ``--from-yaml``. Every
input under ``--input`` is reconstructed — there is no per-run sample cap.

**Output Structure:**

.. code-block:: text

   experiments/results/tutorial_01_basic_unet/inference/
   ├── predictions/
   │   ├── slice_0000.npy  # Reconstructed image
   │   ├── slice_0001.npy
   │   └── ...
   ├── ground_truth/
   │   ├── slice_0000.npy  # Original fully-sampled
   │   └── ...
   ├── undersampled/
   │   ├── slice_0000.npy  # Zero-filled reconstruction (baseline)
   │   └── ...
   └── metrics.json  # Quantitative results

==================
Step 5: Evaluate Results
==================

Compute Metrics
---------------

.. code-block:: python

   import json
   import numpy as np

   # Load metrics
   with open('experiments/results/tutorial_01_basic_unet/inference/metrics.json') as f:
       metrics = json.load(f)

   print(f"Average PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
   print(f"Average SSIM: {metrics['ssim_mean']:.3f} ± {metrics['ssim_std']:.3f}")
   print(f"Average NMSE: {metrics['nmse_mean']:.4f}")

**Expected Results (4× acceleration):**

.. code-block:: text

   Average PSNR: 32.4 ± 2.1 dB
   Average SSIM: 0.902 ± 0.031
   Average NMSE: 0.0023

Visualize Reconstructions
--------------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   def visualize_reconstruction(slice_idx=0):
       # Load data
       pred = np.load(f'experiments/results/tutorial_01_basic_unet/inference/predictions/slice_{slice_idx:04d}.npy')
       gt = np.load(f'experiments/results/tutorial_01_basic_unet/inference/ground_truth/slice_{slice_idx:04d}.npy')
       zf = np.load(f'experiments/results/tutorial_01_basic_unet/inference/undersampled/slice_{slice_idx:04d}.npy')

       # Create visualization
       fig, axes = plt.subplots(1, 4, figsize=(16, 4))

       axes[0].imshow(np.abs(zf), cmap='gray')
       axes[0].set_title('Zero-Filled (Input)')
       axes[0].axis('off')

       axes[1].imshow(np.abs(pred), cmap='gray')
       axes[1].set_title('U-Net Reconstruction')
       axes[1].axis('off')

       axes[2].imshow(np.abs(gt), cmap='gray')
       axes[2].set_title('Ground Truth (Fully Sampled)')
       axes[2].axis('off')

       # Error map
       error = np.abs(pred - gt)
       axes[3].imshow(error, cmap='hot', vmin=0, vmax=0.1)
       axes[3].set_title('Absolute Error')
       axes[3].axis('off')

       plt.tight_layout()
       plt.savefig(f'reconstruction_slice_{slice_idx}.png', dpi=150)
       plt.show()

   # Visualize first 5 slices
   for i in range(5):
       visualize_reconstruction(i)

==================
Step 6: Experiment Variations
==================

Try Different Configurations
----------------------------

**1. Increase acceleration (more challenging):**

.. code-block:: yaml

   acceleration:
     base_acceleration: 8.0  # Change from 4.0

**Expected:** PSNR drops to ~28-30 dB

**2. Add perceptual loss (better visual quality):**

.. code-block:: yaml

   losses:
     image_losses:
       - name: l1
         weight: 1.0
         enabled: true
       - name: ssim
         weight: 1.0
         enabled: true
       - name: perceptual  # NEW
         weight: 0.1
         enabled: true

**Expected:** Sharper edges, slightly lower PSNR but better perceptual quality

**3. Deeper network (more capacity):**

.. code-block:: yaml

   model:
     model_kwargs:
       features: [64, 128, 256, 512, 1024]  # Add 5th level

**Expected:** +0.5-1.0 dB PSNR, but slower training

==================
Troubleshooting
==================

**Issue: CUDA Out of Memory**

.. code-block:: yaml

   training:
     batch_size: 2  # Reduce from 4
     gradient_accumulation_steps: 2  # Maintain effective batch size of 4

**Issue: Training Not Converging**

- Check learning rate: Try ``5e-5`` instead of ``1e-4``
- Verify data normalization: Ensure images are in [0, 1] range
- Check loss weights: Ensure SSIM weight isn't too high (try 0.5)

**Issue: Poor Reconstruction Quality**

- Increase training epochs: Try 100 instead of 50
- Check acceleration factor: 4× should be reasonable; 8× is challenging
- Verify k-space masking: Ensure center fraction includes low frequencies

==================
Next Steps
==================

**More Advanced Tutorials:**

1. :doc:`tutorial_02_gan_super_resolution` - Adversarial training for sharper images
2. Tutorial 03 (Diffusion Training) - Coming soon
3. :doc:`tutorial_04_physics_constraints` - Data consistency and Cycle-Bloch physics

**Experiment Ideas:**

- Compare different loss functions (L1 vs L2 vs perceptual)
- Try different network architectures (ResNet, Transformer)
- Implement multi-coil reconstruction
- Add data augmentation

**Resources:**

- :doc:`../user_guide` - Detailed framework reference

==================
Summary
==================

**You've learned:**

✅ How to create a reconstruction experiment configuration
✅ Training a U-Net model from scratch
✅ Evaluating reconstruction quality
✅ Visualizing and comparing results
✅ Common troubleshooting techniques

**Expected Results:**

- **PSNR:** 32-34 dB (4× acceleration)
- **SSIM:** 0.89-0.91
- **Training Time:** ~2 hours (50 epochs)
- **Inference Speed:** ~50ms per slice

**Congratulations!** You've successfully trained your first MRI reconstruction model. 🎉
