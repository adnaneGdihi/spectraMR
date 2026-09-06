.. _getting_started:

===============
Getting Started
===============

Welcome to the spectraMR framework! This guide will help you set up your environment and run your first MRI reconstruction experiment.

.. contents:: Table of Contents
   :local:
   :depth: 2

System Requirements
===================

Hardware
--------

**Minimum:**

- CPU: 4+ cores (Intel i5 or AMD equivalent)
- RAM: 16 GB
- Storage: 50 GB free space
- GPU: NVIDIA GPU with 8GB+ VRAM (GTX 1080 or better)

**Recommended:**

- CPU: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- RAM: 32+ GB
- Storage: 200+ GB SSD
- GPU: NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A100, or V100)

Software
--------

- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: 3.12 or newer (``requires-python = ">=3.12"``)
- **CUDA**: 12.6 (for GPU acceleration). The project pins the
  ``pytorch-cu126`` wheel index deliberately: cu126 is the last lane that
  still ships ``sm_70``, which the V100 above needs. A cu129 (or newer)
  wheel fails every kernel launch on a V100 with
  ``cudaErrorNoKernelImageForDevice``.
- **Git**: For cloning the repository

Installation
============

There are two routes. Install from PyPI if you only want to *run* the framework;
clone if you want the experiment corpus, the tutorials' companion files, or to
modify the source.

From PyPI
---------

.. code-block:: bash

   pip install spectraMR              # core
   pip install "spectraMR[mri]"       # + TorchIO / MONAI / NiBabel / torchkbnufft
   pip install "spectraMR[all]"       # everything that resolves in one shot

This installs the ``spectramr`` console script, which is what every command in
these pages uses. It does **not** bring the ``experiments/`` tree with it, so the
tutorials below have you write their configuration files yourself.

The remaining steps are the clone route.

Step 1: Clone the Repository
-----------------------------

.. code-block:: bash

   git clone https://github.com/adnaneGdihi/spectramr.git
   cd spectramr

Step 2: Create Python Environment
----------------------------------

Using **conda** (recommended):

.. code-block:: bash

   # Create environment
   conda create -n spectramr python=3.12
   conda activate spectramr

Using **venv**:

.. code-block:: bash

   # Create environment
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

Step 3: Install Dependencies
-----------------------------

**Basic Installation** (CPU-only or existing CUDA):

.. code-block:: bash

   pip install -e .

**With Medical Imaging Tools** (TorchIO, MONAI, NiBabel, torchkbnufft):

.. code-block:: bash

   pip install -e ".[mri]"

**With Development Tools** (everything in ``all`` below, plus the
config-migration toolchain):

.. code-block:: bash

   pip install -e ".[dev]"

**Complete Installation** (everything that resolves in one ``pip install`` —
every feature group *and* every role group, so ``docs``, ``test``, ``qa`` and
``profile`` come along too). Only ``mamba``, ``attention`` and ``radiomics`` are
excluded, each because it cannot build under isolation:

.. code-block:: bash

   pip install -e ".[all]"

**Mamba / SSM models** (``hilbert_mamba``, ``geomamba``, ``bloch_mamba``, …)
require the official CUDA selective-scan kernel. It compiles from source, so it
is a separate extra installed with build isolation **off** on an ``nvcc``-equipped
machine (it is intentionally excluded from ``all`` because it cannot resolve in a
one-shot install):

.. code-block:: bash

   pip install -e ".[all]"
   pip install -e ".[mamba]" --no-build-isolation

You can confirm every declared dependency for a chosen extra set is installed,
version-correct, and importable with the SSOT checker:

.. code-block:: bash

   python scripts/verify/verify_dependencies.py --all --import-check

Step 4: Verify Installation
----------------------------

.. code-block:: bash

   # Check PyTorch and CUDA
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

   # Check framework installation
   python -c "from spectramr.config.settings import TrainingSettings; print('spectraMR installed successfully!')"

Expected output::

   PyTorch: 2.8.0+cu126
   CUDA Available: True
   spectraMR installed successfully!

Dataset Setup
=============

The framework supports multiple MRI datasets. We'll demonstrate with **FastMRI** (publicly available).

Option 1: FastMRI Dataset (Recommended for Beginners)
------------------------------------------------------

1. **Register and Download**:

   - Visit `fastMRI <https://fastmri.med.nyu.edu/>`_
   - Create an account and accept the data usage agreement
   - Download the **Brain** dataset (multicoil training data)

2. **Organize Data**:

   .. code-block:: bash

      # Create directories
      mkdir -p databases/fastmri/datasets

      # Extract downloaded data
      # Assuming you downloaded to ~/Downloads/
      tar -xzf ~/Downloads/brain_multicoil_train.tar.gz -C databases/fastmri/datasets/
      tar -xzf ~/Downloads/brain_multicoil_val.tar.gz -C databases/fastmri/datasets/

   Expected structure::

      databases/
      └── fastmri/
          └── datasets/
              ├── multicoil_brain_train/
              │   ├── file_brain_AXT1_200_2000001.h5
              │   ├── file_brain_AXT1_200_2000002.h5
              │   └── ...
              └── multicoil_brain_val/
                  └── ...

3. **Generate Dataset Index**:

   .. code-block:: bash

      # Create manifests directory
      mkdir -p data/manifests

      # Run preprocessing to generate index
      python scripts/data/regenerate_cluster_manifests.py \
          --data-base databases \
          --datasets fastmri_brain

Option 2: M4Raw Dataset
------------------------

M4Raw is another excellent dataset for rapid prototyping.

.. code-block:: bash

   # Fetch M4Raw from its own release (CC-BY-4.0) and unpack it under
   # databases/m4raw/ -- see https://doi.org/10.5281/zenodo.8056074
   #
   # Then build the manifests:
   python scripts/data/regenerate_cluster_manifests.py \
       --data-base databases \
       --datasets m4raw

Option 3: Using Sample Data (Quick Start)
------------------------------------------

For testing without downloading large datasets. To *train* on a synthetic set
rather than merely probe one, :doc:`tutorials/first_reconstruction` builds a
phantom dataset from scratch in about twenty lines.

.. code-block:: bash

   # Exercise a config end to end without any dataset at all: the Tier-2
   # probe synthesises its own batch, builds the model, and runs a forward
   # and backward pass through it.
   spectramr audit experiments/templates/comprehensive_config_template.yaml --probe

Your First Experiment
======================

We'll train a basic U-Net for MRI reconstruction with 4× acceleration.

Step 1: Understand the Configuration
-------------------------------------

Experiments are defined using YAML configuration files. Let's examine a simple configuration:

.. code-block:: yaml
   config_version: '1.0'
   model:
     model_type: standard_unet
     in_channels: 2
     out_channels: 2

   training:
     task: reconstruction
     input_domain: image
     output_domain: image
     strategy_class: spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy
     max_iterations: 10000
     epochs: 10
     device: cuda
     output_dir: experiments/results/quickstart_basic_reconstruction

   data:
     dataset_type: kspace
     datasets:
       - name: fastmri_train
         path: databases/fastmri/datasets/multicoil_train

     loader:
       batch_size: 4
     source:
       root: databases/fastmri/datasets
       index_path: data/manifests/fastmri_brain_multicoil_train.json
   adapters:
     pre_model:
       - name: ifft_kspace_to_image          # kspace -> complex_image
       - name: complex_to_real_imag_interleave  # complex_image -> image (2 channels)

   undersampling:
     base_acceleration: 4
     center_fraction: 0.08
     acceleration_type: cartesian_vd

   optimization:
     optimizer:
       type: adam
       learning_rate: 0.0001

   logging:
     identity:
       experiment: quickstart_basic_reconstruction
     intervals:
       log: 50
       save: 5

**Key Parameters:**

- ``model_type: standard_unet`` - Using U-Net architecture
- ``in_channels: 2`` - Real and imaginary k-space components
- ``training_mode: reconstruction`` - Direct image reconstruction (not generative)
- ``acceleration: 4`` - 4× undersampling (keep 25% of k-space data)
- ``center_fraction: 0.08`` - Keep 8% of central k-space (important low-frequency data)

Step 2: Run Your First Training
--------------------------------

.. code-block:: bash

   # Activate environment
   conda activate spectramr

   # Train the model
   python -m spectramr.cli train --config experiments/templates/comprehensive_config_template.yaml

**What to expect:**

.. code-block:: text

   [2024-12-26 10:30:00] INFO - Loading configuration from experiments/templates/comprehensive_config_template.yaml
   [2024-12-26 10:30:01] INFO - Initializing DI container...
   [2024-12-26 10:30:02] INFO - Loading dataset: fastmri_brain
   [2024-12-26 10:30:05] INFO - Dataset loaded: 5000 training samples, 500 validation samples
   [2024-12-26 10:30:06] INFO - Model initialized: Mamba (2.4M parameters)
   [2024-12-26 10:30:07] INFO - Starting training...

   Epoch 1/100:
   [=====>                    ] 20% | Loss: 0.0245 | PSNR: 28.3 dB

**Training will output:**

- Checkpoints: ``experiments/results/comprehensive_experiment_template/checkpoints/``
- Logs: ``experiments/results/comprehensive_experiment_template/logs/``
- Visualizations: ``experiments/results/comprehensive_experiment_template/visualizations/``

Step 3: Monitor Training
-------------------------

**Option A: Real-time with Weights & Biases**

If you have W&B enabled in your config (``logging.enable_wandb: true``):

1. Create a free account at `wandb.ai <https://wandb.ai/>`_
2. Log in:

   .. code-block:: bash

      wandb login

3. View training at: ``https://wandb.ai/<your-username>/spectramr_research``

**Option B: TensorBoard**

.. code-block:: bash

   # In a separate terminal
   tensorboard --logdir experiments/results/comprehensive_experiment_template/logs

   # Open browser to: http://localhost:6006

**Option C: Console Output**

Training metrics are printed to console every ``log_interval`` iterations (default: 25).

Step 4: Run Inference
----------------------

Once training completes, run inference on test data:

.. code-block:: bash

   python -m spectramr.cli infer \
       --config experiments/templates/comprehensive_config_template.yaml \
       --checkpoint experiments/results/comprehensive_experiment_template/checkpoints/best.pt \
       --input databases/fastmri/datasets/multicoil_brain_val \
       --output output/inference_results

Step 5: Evaluate Results
-------------------------

.. code-block:: bash

   spectramr report --exp-dir experiments/results/comprehensive_experiment_template

**Expected output:**

.. code-block:: text

   Evaluation Results:
   ├── PSNR: 32.5 ± 2.1 dB
   ├── SSIM: 0.89 ± 0.04
   └── MSE: 0.0012 ± 0.0003

Next Steps
==========

Now that you've run your first experiment, explore:

1. **Different Model Architectures**:

   - Try a GAN: ``experiments/configs/e2e_gan_super_resolution_mri.yaml``
   - Try diffusion: ``experiments/active/experiment_31_consistency_distillation.yaml``

2. **Advanced Topics**:

   - :doc:`user_guide` - Deep dive into the framework
   - :doc:`tutorials/index` - Step-by-step tutorials

3. **Custom Development**:

   - Tutorial 05 (Custom Loss) — coming soon in the next release

Common Issues
=============

Issue: CUDA Out of Memory
--------------------------

**Symptom:**

.. code-block:: text

   RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB

**Solutions:**

1. Reduce batch size in config:

   .. code-block:: yaml

      training:
        batch_size: 2  # Reduce from 16

2. Reduce image size:

   .. code-block:: yaml

      data:
        img_size: [256, 256]  # Reduce from [320, 320]

3. Enable gradient checkpointing (trades compute for memory):

   .. code-block:: yaml

      optimization:
        use_gradient_checkpointing: true

4. Use mixed precision training (already enabled in most configs):

   .. code-block:: yaml

      optimization:
        use_amp: true  # Automatic Mixed Precision

Issue: Dataset Not Found
-------------------------

**Symptom:**

.. code-block:: text

   FileNotFoundError: Manifest not found: data/manifests/fastmri_brain_multicoil_train.pkl

**Solution:**

Generate the dataset index:

.. code-block:: bash

   python scripts/data/regenerate_cluster_manifests.py \
       --data-base databases \
       --datasets fastmri_brain

Issue: Slow Training
---------------------

**Symptom:**

Training takes > 1 minute per iteration.

**Solutions:**

1. **Enable data caching** (loads full dataset into RAM):

   .. code-block:: yaml

      data:
        caching:
          strategy: full  # Options: none, partial, full

2. **Increase num_workers** (parallel data loading):

   .. code-block:: yaml

      data:
        num_workers: 8  # Set to number of CPU cores

3. **Pin memory** (faster GPU transfer):

   .. code-block:: yaml

      data:
        pin_memory: true

Issue: NaN Loss
----------------

**Symptom:**

.. code-block:: text

   Epoch 3, Iteration 150: Loss = nan

**Solutions:**

1. Reduce learning rate:

   .. code-block:: yaml

      optimization:
        learning_rate: 0.00001  # 10× smaller

2. Enable gradient clipping:

   .. code-block:: yaml

      optimization:
        clip_grad_norm: 1.0

3. Check data normalization:

   .. code-block:: yaml

      data:
        normalize_images: true
        normalization: znorm  # or minmax

Config Validation (--dry-run)
==============================

Before committing a long GPU run, validate your YAML configuration in
seconds using the ``--dry-run`` flag. This runs the full
:class:`~spectramr.infrastructure.validation.config_health_checker.ConfigHealthChecker`
pipeline without instantiating models or loading data:

.. code-block:: bash

   python -m spectramr.cli train --config <your-arm>.yaml --dry-run

**Example output:**

.. code-block:: text

   ✅ [required_section] Section 'data' present
   ✅ [required_section] Section 'model' present
   ✅ [model_registry] model_type='kspace_cold_diffusion' registered
   ✅ [strategy_registry] strategy='.DiffusionTrainingStrategy' → valid strategy
   ✅ [domain_alignment] model.in_channels=8 matches expected=8
        (coil_processing_mode='svd' with num_virtual_coils=4 → 2×4)
   ⚠️  [physics_config] physics config is inert for k-space strategy='DiffusionTrainingStrategy'
        (dataset_type='kspace'): physics.data_consistency.enabled is False and no
        undersampling: block is declared. Nothing constrains the reconstruction to the
        acquired measurements. This may be intentional for a denoising/restoration arm
        whose degradation is not k-space undersampling — verify before treating this as a bug.
   Config Health: 6/7 checks passed

The checks run are:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Check Name
     - Severity
     - Description
   * - ``required_section``
     - Error
     - Ensures data, model, training, optimization, logging are all present
   * - ``model_registry``
     - Error
     - Verifies ``model.model_type`` is registered in the ``ModelFactory``
   * - ``strategy_registry``
     - Error
     - Verifies ``training.strategy_class`` or ``training.training_mode`` resolves
   * - ``domain_alignment``
     - Error
     - Pre-flight channel count check: derives expected ``in_channels`` from
       ``coil_processing_mode`` × ``num_virtual_coils`` and compares to
       ``model.in_channels`` / ``model.out_channels``
   * - ``loss_weights``
     - Warning
     - Warns if all reconstruction loss weights are 0.0
   * - ``physics_config``
     - Info
     - Flags a k-space diffusion/reconstruction arm whose ``physics:`` block is
       present but inert (``physics.data_consistency.enabled`` is ``False`` and no
       ``undersampling:`` block is declared) — advisory, since this is sometimes a
       deliberate no-physics denoising/restoration control

.. admonition:: Domain Alignment is a Hard Failure
   :class: warning

   If ``domain_alignment`` emits an **error**, the training pipeline aborts
   immediately — before any GPU memory is allocated. This prevents silent
   dimension mismatch crashes after hours of training.

   Fix by aligning ``model.in_channels`` with
   ``data.coil_processing_mode`` × ``data.num_virtual_coils``.

Additional Resources
====================

- **User guide**: :doc:`user_guide`
- **API reference**: :doc:`scripting_api`
- **GitHub Issues**: `Report bugs <https://github.com/adnaneGdihi/spectramr/issues>`_

Quick Reference Commands
=========================

.. code-block:: bash

   # Training
   python -m spectramr.cli train --config <path-to-yaml>

   # Inference
   python -m spectramr.cli infer --config <config> --checkpoint <checkpoint> --input <dir> --output <dir>

   # Evaluation
   spectramr report --exp-dir <experiment-output-dir>

   # Hyperparameter optimization
   python -m spectramr.tools.tune --config <config> --trials 50

   # Dry run (config validation)
   python -m spectramr.cli train --config <config> --dry-run

   # View logs
   tensorboard --logdir experiments/<experiment-name>/logs

   # Run tests
   pytest tests/

   # Build documentation
   cd docs && make html

What's Next?
============

You're ready to explore! Here are suggested learning paths:

**Path 1: Reconstruction Specialist**

1. Complete this getting started guide ✓
2. Try :doc:`tutorials/tutorial_01_basic_reconstruction`
3. Experiment with different acceleration factors (2×, 4×, 8×)
4. Compare U-Net vs Transformer vs Mamba architectures

**Path 2: Generative Models Researcher**

1. Complete this getting started guide ✓
2. Understand diffusion models: :doc:`tutorials/tutorial_03_diffusion_training`
3. Train a GAN: :doc:`tutorials/tutorial_02_gan_super_resolution`
4. Advanced: Try Rectified Flow (InstaFlow) for 1-step generation

**Path 3: Physics-Informed ML**

1. Complete this getting started guide ✓
2. Tutorial: :doc:`tutorials/tutorial_04_physics_constraints`
3. Experiment: ``experiments/active/experiment_41_physics_informed_motion_networks_pimn.yaml``

Welcome to spectraMR! 🚀
