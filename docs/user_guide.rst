.. _user_guide:

==========
User Guide
==========

This comprehensive guide explains how to use the spectraMR framework effectively, from understanding its architecture to developing custom models.

.. contents:: Table of Contents
   :local:
   :depth: 3

Framework Overview
==================

Architecture Philosophy
-----------------------

The spectraMR framework follows **Clean Architecture** principles with clear separation of concerns:

.. code-block:: text

   ┌─────────────────────────────────────────┐
   │         Application Layer               │
   │    (Use Cases, Orchestration)           │
   └──────────────┬──────────────────────────┘
                  │
   ┌──────────────▼──────────────────────────┐
   │          Domain Layer                    │
   │    (Models, Physics, Losses)             │
   └──────────────┬──────────────────────────┘
                  │
   ┌──────────────▼──────────────────────────┐
   │      Infrastructure Layer                │
   │  (Training, Data, Services, Config)      │
   └──────────────────────────────────────────┘

**Key Components:**

1. **Domain Layer** (``src/spectramr/models/``, ``src/spectramr/models/losses/``): Core ML logic
2. **Infrastructure Layer** (``src/spectramr/infrastructure/``): Training engines, data loaders
3. **Application Layer** (``src/spectramr/application/``): Use cases and pipelines
4. **Services Layer** (``src/spectramr/infrastructure/services/``): Logging, checkpointing, metrics

Dependency Injection
--------------------

The framework uses a DI container to manage dependencies:

.. code-block:: python

   from spectramr.infrastructure.di_container import DIContainer

   # Automatically initialized from config
   container = DIContainer.get_instance()

   # Services are registered and injected
   logger = container.resolve("logger")
   checkpoint_service = container.resolve("checkpoint_service")

**Benefits:**

- Decoupled components
- Easy testing with mocks
- Configuration-driven initialization

Configuration System
====================

Configuration Schema (v5.0)
----------------------------

All experiments use YAML configurations with schema version ``5.0``. The configuration is strictly typed and validated using Pydantic schemas.

Top-Level Structure
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   config_version: '1.0'

   metadata:            # Metadata (name, tags, description)
     name: <str>
     tags: <dict>
     description: <str>
     version: '1.0'

   model:               # Architecture definition
     model_type: <str>
     ...

   data:                # Dataset and loading
     dataset_type: <str>
     ...

   training:            # Training loop and strategy
     task: <str>
     strategy_class: <str>
     ...

   optimization:        # Optimizer and scheduler
     optimizer_type: <str>
     ...

   physics:             # MRI physics constraints
     data_consistency: {}
     ...

   logging:             # Logging, tracking, checkpoints
     enable_logging: true
     ...

Model Configuration (`model`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configures the neural network architecture. For a comprehensive list of all registered models and their initialization parameters, see :doc:`model_registry_reference`.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Type
     - Description
   * - ``model_type``
     - ``str``
     - Architecture identifier (e.g., ``unet``, ``mamba``, ``diffusion_unet``).
   * - ``in_channels``
     - ``int``
     - Number of input channels (e.g., 2 for complex, 1 for magnitude).
   * - ``out_channels``
     - ``int``
     - Number of output channels.
   * - ``model_kwargs``
     - ``dict``
     - Architecture-specific parameters (e.g., ``features``, ``num_layers``, ``dropout``).
   * - ``kan_type``
     - ``str``
     - Efficiency layer type: ``BSpline``, ``Chebyshev`` (for KAN models).
   * - ``input_type``
     - ``str``
     - Domain: ``image`` or ``kspace``.

Data Configuration (`data`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unified configuration for data loading and preprocessing.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Type
     - Description
   * - ``dataset_type``
     - ``str``
     - Dataset engine: ``fastmri_kspace``, ``m4raw``, ``image_folder``, ``volume_h5``.
   * - ``data_root``
     - ``str``
     - Root directory for raw data files.
   * - ``index_path``
     - ``str``
     - Path to pre-computed ``.pkl`` manifest (required for H5/TorchIO).
   * - ``manifest_roles``
     - ``object``
     - **[New in v5.0]** Assigns multiple manifests to roles (``inputs``, ``targets``, ``auxiliary``).
   * - ``patch_size``
     - ``list[int]``
     - Spatial dimensions `[H, W, D]` (D=1 for 2D).
   * - ``batch_size``
     - ``int``
     - Batch size per GPU.
   * - ``num_workers``
     - ``int``
     - specific number of CPU workers.
   * - ``acceleration``
     - ``int``
     - Undersampling factor (e.g., 4, 8).
   * - ``center_fraction``
     - ``float``
     - Fraction of fully sampled center frequencies.
   * - ``trajectory``
     - ``str``
     - Non-Cartesian trajectory (e.g., ``spiral``, ``radial``).

**Manifest Roles Example:**

.. code-block:: yaml

   data:
     manifest_roles:
       inputs:
         - manifest: "data/manifests/fastmri_brain_multicoil_kspace.pkl"
           key: "kspace"
       targets:
         - manifest: "data/manifests/fastmri_brain_multicoil_image_gt.pkl"
           key: "target"

Training Configuration (`training`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Polymorphic configuration based on ``training_mode``.

**Common Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Type
     - Description
   * - ``task``
     - ``str``
     - **REQUIRED**. ``reconstruction``, ``diffusion``, ``gan``, or ``vae``.
   * - ``strategy_class``
     - ``str``
     - Full Python path to strategy class (e.g., ``spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy``).
   * - ``input_domain``
     - ``str``
     - ``image`` or ``kspace``
   * - ``output_domain``
     - ``str``
     - ``image`` or ``kspace``
   * - ``epochs``
     - ``int``
     - Total epochs.
   * - ``max_iterations``
     - ``int``
     - Total steps (overrides epochs if set).
   * - ``seed``
     - ``int``
     - Random seed for reproducibility.

**Reconstruction Mode Specifics:**

*   ``enable_data_consistency`` (bool): Apply DC in loss.
*   ``dc_weight`` (float): Weight of DC term.

**Diffusion Mode Specifics:**

*   ``num_timesteps`` (int): Diffusion steps (default: 1000).
*   ``noise_schedule`` (str): ``linear``, ``cosine``, or ``sigmoid``.
*   ``prediction_type`` (str): ``epsilon``, ``sample``, or ``v_prediction``.
*   ``guidance_scale`` (float): Classifier-free guidance scale.

Optimization Configuration (`optimization`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Controls the training dynamics.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Type
     - Description
   * - ``optimizer_type``
     - ``str``
     - ``adam``, ``adamw``, ``sgd``, or ``rmsprop``.
   * - ``learning_rate``
     - ``float``
     - Base learning rate.
   * - ``weight_decay``
     - ``float``
     - L2 regularization factor.
   * - ``use_amp``
     - ``bool``
     - Enable Automatic Mixed Precision (FP16/BF16).
   * - ``lr_scheduler_strategy``
     - ``str``
     - ``cosine``, ``step``, ``plateau``, or ``linear_warmup``.
   * - ``gradient_accumulation_steps``
     - ``int``
     - Simulate larger batches.

Physics Configuration (`physics`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Defines MRI-specific constraints and simulations.

.. code-block:: yaml

   physics:
     data_consistency:
       enabled: true
       method: "projection_2d_consistency"  # or "adaptive_soft_dc"
       weight: 1.0

     coil_sensitivity:
       enable_estimation: false
       precomputed_path: "path/to/maps.h5"

     compressed_sensing:
       enabled: true
       acceleration_factor: 4.0
       reconstruction_algorithm: "iterative_soft_thresh"

Model Types
-----------

Available ``model_type`` values:

**Reconstruction:**

- ``standard_unet``: Classic U-Net architecture
- ``mamba``: Mamba SSM for long-range dependencies
- ``transformer_recon``: Transformer-based reconstruction
- ``neural_ode``: Neural ODE dynamics modeling

**Diffusion:**

- ``diffusion_unet``: Standard DDPM
- ``kspace_cold_diffusion``: Cold diffusion with k-space degradation
- ``latent_diffusion``: Diffusion in latent space
- ``consistency_model``: Fast 1-2 step inference
- ``rectified_flow``: Straight ODE trajectories (InstaFlow)

**GANs:**

- ``gan``: Standard GAN with U-Net generator
- ``wgan_gp``: Wasserstein GAN with gradient penalty

**VAE:**

- ``vae``: Variational autoencoder
- ``wavelet_fourier_kan_vae``: Multi-domain VAE

Training Modes
--------------

The ``training_mode`` determines the training strategy:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Use Case
     - Loss Functions
   * - ``reconstruction``
     - Direct image reconstruction
     - L1, L2, SSIM, Perceptual
   * - ``diffusion``
     - Generative modeling via diffusion
     - Denoising score matching
   * - ``gan``
     - Adversarial super-resolution
     - Adversarial + Content
   * - ``vae``
     - Latent representation learning
     - ELBO (reconstruction + KL)

Dataset Configuration
---------------------

**Single Dataset:**

.. code-block:: yaml

   data:
     dataset_type: kspace
     data_root: databases/fastmri/datasets
     datasets:
       - name: fastmri_train
         path: databases/fastmri/datasets/multicoil_train
     index_path: data/manifests/fastmri_brain_multicoil_train.json
     batch_size: 4

**Multi-Dataset (Multi-Contrast):**

.. code-block:: yaml

   data:
     dataset_type: kspace
     data_root: databases/mri
     datasets:
       - name: t1_weighted
         path: databases/mri/t1/train
       - name: t2_weighted
         path: databases/mri/t2/train
     index_path: data/manifests/multicontrast_train.json

**Caching Strategies:**

.. code-block:: yaml

   data:
     caching:
       strategy: none       # No caching (disk I/O each batch)
       # strategy: partial  # Cache preprocessed data
       # strategy: full     # Load entire dataset into RAM

Training Pipeline
=================

Execution Flow
--------------

.. mermaid::

   sequenceDiagram
       participant CLI as CLI Interface
       participant Config as Config Loader
       participant DI as DI Container
       participant Data as DataLoader
       participant Model as Model
       participant Strategy as Training Strategy
       participant Logger as Logger/W&B

       CLI->>Config: Load YAML
       Config->>DI: Initialize services
       DI->>Data: Create DataLoader
       DI->>Model: Initialize model
       DI->>Strategy: Create strategy

       loop Training Loop
           Data->>Strategy: Fetch batch
           Strategy->>Model: Forward pass
           Model->>Strategy: Predictions
           Strategy->>Strategy: Compute loss
           Strategy->>Model: Backward + update
           Strategy->>Logger: Log metrics
       end

       Strategy->>DI: Save checkpoint
       DI->>CLI: Training complete

Training Strategies
-------------------

**Reconstruction Strategy:**

.. code-block:: python

   # src/spectramr/infrastructure/training/strategies/reconstruction.py

   def train_step(self, batch):
       undersampled, ground_truth = batch

       # Forward pass
       prediction = self.model(undersampled)

       # Compute losses
       l1_loss = F.l1_loss(prediction, ground_truth)
       ssim_loss = 1 - ssim(prediction, ground_truth)
       perceptual = self.perceptual_loss(prediction, ground_truth)

       # Combined loss (weights loaded dynamically from list-based schema)
       loss = (self.loss_weights.get('l1', 1.0) * l1_loss +
               self.loss_weights.get('ssim', 1.0) * ssim_loss +
               self.loss_weights.get('perceptual', 0.5) * perceptual)

       # Backward
       loss.backward()
       self.optimizer.step()

       return {"loss": loss.item(), "psnr": compute_psnr(prediction, ground_truth)}

**Diffusion Strategy:**

Implements the denoising score matching objective:

.. math::

   \\mathcal{L} = \\mathbb{E}_{t, x_0, \\epsilon} [\\|\\epsilon_\\theta(x_t, t) - \\epsilon\\|^2]

**GAN Strategy:**

Alternates between generator and discriminator updates with WGAN-GP:

.. math::

   \\mathcal{L}_D = \\mathbb{E}[D(G(z))] - \\mathbb{E}[D(x)] + \\lambda_{gp} \\mathbb{E}[(\\|\\nabla_{\\hat{x}} D(\\hat{x})\\|_2 - 1)^2]

Checkpointing
-------------

**Automatic Checkpointing:**

.. code-block:: yaml

   checkpoint:
     save_interval: 10000      # Save every 10k iterations
     save_best: true           # Keep best validation checkpoint
     save_ema: true            # Exponential moving average weights

**Checkpoint Contents:**

.. code-block:: python

   checkpoint = {
       "epoch": current_epoch,
       "iteration": global_step,
       "model_state_dict": model.state_dict(),
       "optimizer_state_dict": optimizer.state_dict(),
       "scheduler_state_dict": scheduler.state_dict(),
       "best_metric": best_val_psnr,
       "config": config.to_dict(),
       "random_state": get_random_state(),
   }

**Resume Training:**

.. code-block:: bash

   python -m spectramr.cli train \
       --config experiments/active/experiment_30_mamba_mri_reconstruction.yaml \
       --resume experiments/active/experiment_30_mamba_mri_reconstruction/checkpoints/latest.pt

Logging and Monitoring
======================

Weights & Biases (deferred)
----------------------------

W&B is **not implemented** and is deferred by owner decision (2026-08-12). TensorBoard is
the only tracking backend (``TrackingService``, see below) -- ``logging:`` has no working
path to a W&B run.

``logging.wandb_project`` and ``logging.wandb_entity`` still exist as fields, but only so
that declaring either one **raises** at config load instead of being silently accepted and
discarded (issue #675, ``LoggingConfigSchema._refuse_deferred_wandb``). Do not set them to a
value:

.. code-block:: yaml

   logging:
     log_interval: 25
     run_name: experiment_30_mamba_mri
     # wandb_project / wandb_entity: DEFERRED -- setting either raises a ValidationError.
     # Leave both unset (or explicit `null`, which is equivalent) and use TensorBoard.

**Logged Metrics:**

- Training loss (every ``log_interval`` steps)
- Validation metrics (PSNR, SSIM, MSE, MAE)
- Learning rate
- GPU memory usage
- Training time per iteration

**Custom Logging:**

.. code-block:: python

   from spectramr.infrastructure.services.logging_service import LoggingService

   logger = LoggingService.get_instance()
   logger.log_metrics({"custom_metric": value}, step=iteration)
   logger.log_images({"reconstruction": pred_image}, step=iteration)

TensorBoard
-----------

.. code-block:: bash

   # Start TensorBoard
   tensorboard --logdir experiments/active/

Inference Workflow
==================

Running Inference
-----------------

.. code-block:: bash

   python -m spectramr.cli infer \
       --config experiments/active/experiment_30_mamba_mri_reconstruction.yaml \
       --checkpoint experiments/active/experiment_30_mamba_mri_reconstruction/checkpoints/best.pt \
       --input-dir databases/fastmri/datasets/multicoil_brain_val \
       --output-dir output/experiment_30_inference \
       --batch-size 8

**Output Structure:**

.. code-block:: text

   output/experiment_30_inference/
   ├── predictions/
   │   ├── slice_0000.npy
   │   ├── slice_0001.npy
   │   └── ...
   ├── visualizations/
   │   ├── slice_0000_comparison.png
   │   └── ...
   └── metrics.json

Evaluation
----------

.. code-block:: bash

   spectramr report --exp-dir output/experiment_30_inference

**Metrics Output:**

.. code-block:: json

   {
     "psnr": {"mean": 32.5, "std": 2.1, "median": 32.8},
     "ssim": {"mean": 0.89, "std": 0.04, "median": 0.90},
     "mse": {"mean": 0.0012, "std": 0.0003},
     "mae": {"mean": 0.015, "std": 0.004},
     "hfen": {"mean": 0.0235, "std": 0.0067}
   }

Hyperparameter Optimization
============================

Using Optuna
------------

**HPO Configuration:**

.. code-block:: yaml

   # experiments/hpo/hpo_config.yaml
   hpo:
     study_name: mamba_reconstruction_hpo
     n_trials: 50
     optimization_direction: maximize  # or minimize
     metric: val_psnr

     search_space:
       learning_rate: [1e-5, 1e-3]  # log scale
       batch_size: [4, 8, 16, 32]
       model.model_kwargs.base_dim: [32, 64, 128, 256]
       losses.image_losses.ssim.weight: [0.1, 2.0]

**Run HPO:**

.. code-block:: bash

   spectramr hpo \
       --config experiments/hpo/hpo_config.yaml \
       --model-type standard_unet \
       --n-trials 50

``--model-type`` is required and repeatable — one Optuna study per type.
There is no ``--jobs``: run trials in parallel by pointing several
``spectramr hpo`` processes at one shared study with
``--storage sqlite:///experiments/hpo/study.db``.

**Multi-Objective Optimization:**

.. code-block:: yaml

   hpo:
     optimization_direction: maximize
     objectives:
       - name: val_psnr
         direction: maximize
         weight: 0.7
       - name: inference_speed
         direction: maximize
         weight: 0.3

Custom Model Development
========================

Extending Base Models
---------------------

**Creating a Custom Generator:**

.. code-block:: python

   # src/spectramr/models/generators/custom_generator.py

   from spectramr.models.base.base_unet_generator import BaseUnetGenerator
   import torch.nn as nn

   class CustomGenerator(BaseUnetGenerator):
       \"\"\"Custom generator with attention mechanism.\"\"\"

       def __init__(self, in_channels: int, out_channels: int, features: list[int] = [64, 128, 256, 512]):
           super().__init__(in_channels, out_channels, features)

           # Add custom layers
           self.self_attention = nn.MultiheadAttention(embed_dim=features[-1], num_heads=8)

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           # Encoder
           enc_outputs = []
           for encoder in self.encoders:
               x = encoder(x)
               enc_outputs.append(x)

           # Bottleneck with attention
           x = self.bottleneck(x)
           B, C, H, W = x.shape
           x_flat = x.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
           x_attn, _ = self.self_attention(x_flat, x_flat, x_flat)
           x = x_attn.permute(1, 2, 0).reshape(B, C, H, W)

           # Decoder
           for decoder, skip in zip(self.decoders, reversed(enc_outputs[:-1])):
               x = torch.cat([x, skip], dim=1)
               x = decoder(x)

           return self.final_conv(x)

**Register the Model:**

.. code-block:: python

   # src/spectramr/models/registry.py

   from spectramr.models.generators.custom_generator import CustomGenerator

   MODEL_REGISTRY = {
       # ... existing models
       "custom_generator": CustomGenerator,
   }

**Use in Config:**

.. code-block:: yaml

   model:
     model_type: custom_generator
     in_channels: 2
     out_channels: 2
     model_kwargs:
       features: [64, 128, 256, 512]

Creating Custom Loss Functions
-------------------------------

.. code-block:: python

   # src/spectramr/models/losses/custom_loss.py

   import torch
   import torch.nn as nn
   from typing import Dict

   class CustomPerceptualLoss(nn.Module):
       \"\"\"Custom perceptual loss using medical-specific features.\"\"\"

       def __init__(self, feature_extractor: nn.Module, layers: list[str]):
           super().__init__()
           self.feature_extractor = feature_extractor
           self.layers = layers

       def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
           pred_features = self.feature_extractor(pred, self.layers)
           target_features = self.feature_extractor(target, self.layers)

           loss = 0.0
           for layer in self.layers:
               loss += F.mse_loss(pred_features[layer], target_features[layer])

           return {
               "perceptual_loss": loss,
               "num_layers": len(self.layers)
           }

Advanced Topics
===============

Physics-Informed Training
--------------------------

**Data Consistency Layer:**

Enforces k-space consistency during training:

.. math::

   \\hat{x}_{dc} = \\mathcal{F}^{-1}(M \\odot y + (1 - M) \\odot \\mathcal{F}(\\hat{x}))

where :math:`M` is the undersampling mask, :math:`y` is measured k-space, and :math:`\\hat{x}` is the network prediction.

.. code-block:: yaml

   physics:
     data_consistency:
       enabled: true
       weight: 1.0
       method: projection_2d_consistency

**Compressed Sensing:**

.. code-block:: yaml

   physics:
     compressed_sensing:
       enabled: true
       sampling_pattern: cartesian
       acceleration_factor: 4.0
       reconstruction_algorithm: iterative_soft_thresh
       enforce_data_consistency: true

Multi-Stage Training
---------------------

Train in multiple stages with different objectives:

.. code-block:: yaml

   training:
     stages:
       - name: warmup
         epochs: 10
         learning_rate: 1e-4
         losses:
           image_losses:
             - name: l1
               weight: 1.0
               enabled: true

       - name: refinement
         epochs: 50
         learning_rate: 1e-5
         losses:
           image_losses:
             - name: l1
               weight: 0.5
               enabled: true
             - name: perceptual
               weight: 0.5
               enabled: true
             - name: ssim
               weight: 1.0
               enabled: true

Uncertainty Quantification
---------------------------

Enable Bayesian uncertainty estimation:

.. code-block:: yaml

   model:
     model_type: bayesian_kan
     model_kwargs:
       use_dropout: true
       dropout_rate: 0.1
       num_mc_samples: 10  # Monte Carlo samples

Best Practices
==============

Performance Optimization
------------------------

1. **Enable Mixed Precision:**

   .. code-block:: yaml

      optimization:
        use_amp: true

2. **Optimize Data Loading:**

   .. code-block:: yaml

      data:
        num_workers: 8  # Match CPU cores
        pin_memory: true
        prefetch_factor: 2
        caching:
          strategy: partial  # If RAM allows

3. **Gradient Accumulation** (for large batches on limited VRAM):

   .. code-block:: yaml

      optimization:
        gradient_accumulation_steps: 4
        effective_batch_size: 64  # = batch_size * accumulation_steps

4. **Learning Rate Scheduling:**

   .. code-block:: yaml

      optimization:
        lr_scheduler_strategy: cosine
        warmup_steps: 1000
        min_lr: 1e-7

Reproducibility
---------------

.. code-block:: yaml

   training:
     seed: 42
     deterministic: true
     benchmark: false  # Disable cuDNN benchmarking for determinism

Debug Mode
----------

.. code-block:: yaml

   training:
     debug: true
     max_iterations: 100  # Quick test
     val_interval: 10

   data:
     num_workers: 0  # Single-threaded for debugging

Next Steps
==========

- **Tutorials**: :doc:`tutorials/index` for hands-on examples
- **Config Schema**: :doc:`config_schema_reference` for all YAML keys and defaults
- **Troubleshooting**: :doc:`troubleshooting` for common errors and fixes
- **API Reference**: :doc:`scripting_api` for driving the framework from Python


---

Adding a Custom Training Strategy
===================================

The framework uses a registry-dispatcher pattern for training strategies.
Adding a new one requires 5 steps and zero changes to orchestration code.

**Step 1 — Create the strategy class:**

.. code-block:: python

   # src/spectramr/infrastructure/training/strategies/my_strategy.py
   from spectramr.infrastructure.training.base import BaseTrainingStrategy

   class MyCustomStrategy(BaseTrainingStrategy):
       """Custom training strategy."""

       def training_step(self, batch, step: int) -> dict[str, float]:
           input_batch = batch["input"]
           target_batch = batch["target"]

           # Your training logic here
           pred = self.generator(input_batch)
           loss = self.loss_computer.compute(pred, target_batch)

           self.optimizer_g.zero_grad()
           loss["total"].backward()
           self.optimizer_g.step()

           return {k: v.item() for k, v in loss.items()}

       def validation_step(self, batch, step: int) -> dict[str, float]:
           with torch.no_grad():
               pred = self.generator(batch["input"])
           return self.metrics_computer.compute(pred, batch["target"])

**Step 2 — Register in the strategy registry:**

.. code-block:: python

   # src/spectramr/pipelines/train.py  — add to STRATEGY_REGISTRY
   STRATEGY_REGISTRY = {
       # ... existing strategies ...
       "my_custom": MyCustomStrategy,    # ← add this line
   }

**Step 3 — Add training mode enum** (``src/spectramr/config/schemas/enums.py``):

.. code-block:: python

   class TrainingMode(str, Enum):
       # ... existing modes ...
       MY_CUSTOM = "my_custom"

**Step 4 — Create experiment config.** The excerpt below shows only the blocks
this step adds; a loadable file also needs ``model:``, ``data:``, ``optimization:``
and ``logging:`` (see :doc:`config_schema_reference`).

.. code-block:: yaml

   config_version: "1.0"

   logging:
     identity:
       experiment: my_custom_experiment

   training:
     training_mode: my_custom   # ← dispatches to MyCustomStrategy
     max_iterations: 50000

   model:
     model_type: standard_unet
     in_channels: 1
     out_channels: 1

   losses:
     image_losses:
       - name: l1
         weight: 10.0
     policy:
       output_domain: image

**Step 5 — Run:**

.. code-block:: bash

   python -m spectramr.cli train --config <your-arm>.yaml


---

Adding a Custom Loss Function
================================

Losses are resolved via the loss registry. Adding one takes 3 steps.

**Step 1 — Implement the loss:**

.. code-block:: python

   # src/spectramr/models/losses/my_loss.py
   import torch
   import torch.nn as nn

   class MyLoss(nn.Module):
       """Custom frequency-weighted MSE loss."""

       def __init__(self, weight_high_freq: float = 2.0):
           super().__init__()
           self.weight_high_freq = weight_high_freq

       def forward(
           self,
           pred: torch.Tensor,
           target: torch.Tensor,
       ) -> torch.Tensor:
           # Frequency-weighted MSE (emphasize high-frequency errors)
           diff = pred - target
           freq = torch.fft.fft2(diff, norm="ortho")
           freq_abs = freq.abs()
           # Weight high-frequency errors more
           h, w = freq_abs.shape[-2:]
           ky = torch.fft.fftfreq(h, device=diff.device).abs()
           kx = torch.fft.fftfreq(w, device=diff.device).abs()
           freq_weight = 1.0 + self.weight_high_freq * (ky[:, None] + kx[None, :])
           return (freq_weight * freq_abs**2).mean()

**Step 2 — Register in the loss registry:**

.. code-block:: python

   # src/spectramr/models/losses/__init__.py or registry.py
   from spectramr.models.losses.my_loss import MyLoss

   LOSS_REGISTRY["my_freq_mse"] = MyLoss

**Step 3 — Use in experiment config:**

.. code-block:: yaml

   losses:
     output_domain: image
     image_losses:
       - name: my_freq_mse
         weight: 5.0
         enabled: true
         kwargs:
           weight_high_freq: 3.0   # Forwarded to MyLoss.__init__


---

Multi-Site Training (Continual Learning)
==========================================

For training across multiple scanners/sites without catastrophic forgetting,
use the ``continual_learning`` strategy with EWC (Elastic Weight Consolidation).

**YAML config.** An excerpt — only the continual-learning blocks are shown;
a loadable file also needs ``data:``, ``optimization:`` and ``logging:``.

.. code-block:: yaml

   config_version: "1.0"

   logging:
     identity:
       experiment: multi_site_ewc

   training:
     training_mode: reconstruction
     curriculum:
       phases:
         - name: site_a
           max_iterations: 30000
           data_root: /data/site_a/
         - name: site_b
           max_iterations: 30000
           data_root: /data/site_b/
           ewc_lambda: 5000        # EWC regularization weight

   model:
     model_type: continual_learning_unet
     model_kwargs:
       ewc_lambda: 5000
       fisher_estimation_samples: 200

**EWC loss formulation:** Adds a parameter importance penalty to prevent
forgetting site A while learning site B:

.. math::

   \mathcal{L}_{EWC} = \mathcal{L}_{CE}(\theta) + \frac{\lambda}{2}
   \sum_i F_i (\theta_i - \theta_i^*)^2

where :math:`F_i` is the Fisher information (importance) for parameter :math:`i`
and :math:`\theta^*` are the parameters after site A training.


---

Cluster & Distributed Training
================================

SLURM Job Submission
----------------------

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=spectramr_exp01
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=4
   #SBATCH --gres=gpu:4
   #SBATCH --time=48:00:00
   #SBATCH --mem=128G
   #SBATCH --partition=gpu

   source ~/.bashrc
   conda activate spectramr

   # Required: set cluster data root
   export SPECTRAMR_DATA_ROOT=/project/<allocation>/<user>/spectramr/databases/

   torchrun \
       --nproc_per_node=4 \
       --nnodes=1 \
       spectramr train \
       --config experiments/training/experiment_01_baseline_gan.yaml \
       --override "data.num_workers=8" \
       --override "data.batch_size=8"

DistributedDataParallel (DDP)
------------------------------

Multi-GPU is declared in the ``parallel:`` block and launched explicitly --
there is no auto-detection, and a config that merely names a strategy will not
start a process group on its own.

.. code-block:: yaml

   parallel:
     strategy: ddp            # none | dp | ddp | fsdp | deepspeed
     backend: nccl            # nccl | gloo | mpi
     find_unused_parameters: false
     gradient_as_bucket_view: true

Then launch with ``torchrun`` (``ddp``, ``fsdp`` and ``deepspeed`` all require a
process group; ``dp`` is single-process and does not):

.. code-block:: bash

   torchrun --nproc_per_node=4 -m spectramr.cli train-distributed --config <arm>.yaml

``fsdp`` and ``deepspeed`` additionally require their sub-block flag to agree
with ``strategy`` (``fsdp.enabled: true`` / ``deepspeed.enabled: true``);
declaring only one half raises at config-load time.

.. note::

   This section previously documented an ``acceleration:`` block with a
   ``strategy`` key. No such key exists -- ``AccelerationConfigSchema`` governs
   k-space undersampling and is ``extra="ignore"``, so that YAML validated
   cleanly and was silently discarded, and the run stayed single-GPU.

.. admonition:: DDP + AMP + Diffusion
   :class: warning

   When using DDP + AMP with diffusion models, set ``amp_dtype: bfloat16``.
   Float16 AMP causes NaN gradients in some multi-GPU configurations due
   to gradient norm scaling differences across ranks.

Manifest Path Alignment (Cluster)
-----------------------------------

Manifests contain absolute paths generated on the local machine.
On cluster nodes, use the ``PathResolver`` prefix map:

.. code-block:: yaml

   data:
     train_manifest: data/manifests/train_knee.pkl
     # Prefix rewriting: local /home/<user>/... -> cluster /project/<allocation>/...
     path_prefix_map:
       "/home/<user>/work/spectramr": "/project/<allocation>/<user>/spectramr"

Or regenerate cluster-native manifests:

.. code-block:: bash

   # On cluster login node
   python scripts/data/regenerate_cluster_manifests.py \
       --data-base /project/<allocation>/<user>/spectramr/databases/

Multi-Node Training
--------------------

.. code-block:: bash

   # Node 0 (master)
   torchrun \
       --nproc_per_node=4 \
       --nnodes=2 \
       --node_rank=0 \
       --master_addr=node0.cluster.local \
       --master_port=29500 \
       spectramr train --config my.yaml

   # Node 1
   torchrun \
       --nproc_per_node=4 \
       --nnodes=2 \
       --node_rank=1 \
       --master_addr=node0.cluster.local \
       --master_port=29500 \
       spectramr train --config my.yaml
