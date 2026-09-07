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
- A CUDA GPU. Wall-clock depends on the corpus, the patch size and the
  batch size; measure one epoch before committing to a schedule.

.. contents:: Tutorial Steps
   :local:
   :depth: 2

============================
Step 1: Create Configuration
============================

Save the following as ``experiments/tutorials/tutorial_01_basic_unet.yaml``.

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
     epochs: 50
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

======================
Step 2: Verify Dataset
======================

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

The script reports the manifest it wrote and how many volumes it indexed. Use
``--dry-run`` first to see what it would regenerate without writing.

=======================
Step 3: Train the Model
=======================

Start training:

.. code-block:: bash

   spectramr train --config experiments/tutorials/tutorial_01_basic_unet.yaml

The run directory is not a command-line flag — it comes from
``training.output_dir`` in the YAML above
(``experiments/results/tutorial_01_basic_unet``). Override it for a one-off run
with ``-O training.output_dir=...`` if you need to.

Training prints a progress bar carrying the live loss terms, and a
``[VAL] Results:`` line at each validation event listing every metric the arm
declared. The run ends with a summary naming the final loss, the best value of
each tracked metric, the iteration count and the elapsed time.

Watch the validation line rather than the training loss: it is the number that
says whether the model generalises, and ``val_zf_psnr`` — the zero-filled
baseline scored on the same batch — is what the reconstruction has to beat.

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

=====================
Step 4: Run Inference
=====================

Test the trained model on validation data:

.. code-block:: bash

   spectramr infer \
       --checkpoint experiments/results/tutorial_01_basic_unet/checkpoints/checkpoint_epoch_<E>_step_<S>.pt \
       --input databases/fastmri/datasets/multicoil_val \
       --output experiments/results/tutorial_01_basic_unet/inference

``ls`` the ``checkpoints/`` directory and substitute the file you want.
A ``checkpoint_best.pt`` is written there instead once more than one
validation event has been scored.

``--config`` is optional here: the run directory beside the checkpoint holds
``resolved_config.json``, which wins unless you pass ``--from-yaml``. Every
input under ``--input`` is reconstructed — there is no per-run sample cap.

**Output Structure:**

.. code-block:: text

   experiments/results/tutorial_01_basic_unet/inference/
   ├── <input-stem>_output.npy    # one reconstruction per input file
   ├── final_eval.json            # aggregated metric values
   ├── final_eval_manifest.json   # which metrics were computed, and why any were skipped
   ├── inference_metrics.csv      # one row per input
   └── run_summary.json           # checkpoint, config source, seed, duration, counts

The output filename follows ``data.modes.infer.output.filename_template``,
which defaults to ``{file_id}_output``; the format defaults to ``npy`` and can
be set to ``nifti`` or ``h5`` in the same block. Read
``final_eval_manifest.json`` before quoting a number from ``final_eval.json``:
a full-reference metric that had no reference to score against is recorded
there as skipped, with the reason, rather than silently omitted.

========================
Step 5: Evaluate Results
========================

Compute Metrics
---------------

The run writes its metrics twice: ``final_metrics.json`` in the run directory
holds the best value each tracked metric reached, and
``logs/validation_metrics.csv`` holds one row per validation event.

.. code-block:: python

   import json

   run = "experiments/results/tutorial_01_basic_unet"

   with open(f"{run}/final_metrics.json") as handle:
       final = json.load(handle)

   for name, value in sorted(final["best"].items()):
       print(f"{name}: {value}")

The ``best`` block is keyed by metric name with a ``_best`` suffix — the
validation metrics appear as ``val_psnr_best``, ``val_ssim_best`` and so on,
beside the losses. For the per-event curve, read the CSV: its columns are
``iteration``, ``epoch`` and one column per validation metric.

Visualize Reconstructions
--------------------------

Inference writes the reconstruction only — the reference and the zero-filled
baseline are not re-emitted, so a side-by-side panel comes from ``spectramr
report`` (below) rather than from hand-loading three arrays.

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   pred = np.load("experiments/results/tutorial_01_basic_unet/inference/"
                  "<input-stem>_output.npy")

   # (C, H, W) for a single-file write; take the magnitude channel.
   image = np.abs(pred[0]) if pred.ndim == 3 else np.abs(pred)

   plt.imshow(image, cmap="gray")
   plt.axis("off")
   plt.savefig("reconstruction.png", dpi=150, bbox_inches="tight")

For the comparison panels, run the report verb against the run directory:

.. code-block:: bash

   spectramr report --exp-dir experiments/results/tutorial_01_basic_unet

It writes ``report/qc_report.html`` and ``report/report_summary.md`` beside
``report/figures/`` (learning curves, loss decomposition, a computational
profile, a run-summary card and a contact sheet — each rendered as both
``.png`` and ``.pdf``, the ``.pdf`` carrying a sidecar ``.meta.json``) and
``report/tables/run_summary.{csv,md}``. Every figure is drawn from artifacts
already on disk, so a figure whose inputs are missing is recorded in
``report/report_manifest.json`` with ``"status": "skipped"`` rather than
being drawn from nothing.

=============================
Step 6: Experiment Variations
=============================

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

- :doc:`../config_schema_reference` - Every configuration key, with defaults

==================
Summary
==================

**You've learned:**

✅ How to create a reconstruction experiment configuration
✅ Training a U-Net model from scratch
✅ Evaluating reconstruction quality
✅ Visualizing and comparing results
✅ Common troubleshooting techniques

**Where the numbers are:** ``final_metrics.json`` for the best value each
tracked metric reached, ``logs/validation_metrics.csv`` for the per-event
curve, and ``report/`` for the rendered figures. Quality depends on the
corpus, the acceleration factor and the schedule — read your own run rather
than a quoted range, and compare against ``val_zf_psnr`` in the same row,
which is the zero-filled baseline scored on the same batch.
