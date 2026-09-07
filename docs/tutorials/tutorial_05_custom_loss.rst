.. _tutorial_05_custom_loss:

================================================
Tutorial 5: Building a Custom Loss Function
================================================

In this tutorial you will build a production-grade custom loss function
from scratch, register it with the framework's loss registry, and use it
in a training experiment — without modifying any orchestration code.

.. contents:: Table of Contents
   :local:
   :depth: 2


Prerequisites
=============

- Completed :doc:`tutorial_01_basic_reconstruction`
- Familiarity with PyTorch ``nn.Module``
- Basic understanding of k-space and MRI reconstruction


Overview
========

The spectraMR loss registry enables any loss to be:

1. **Registered** with a string key
2. **Configured** via YAML ``kwargs``
3. **Composed** with other losses via ``output_domain`` routing
4. **Weighted** with a scalar ``weight`` per experiment

We will build a **Frequency-Weighted Reconstruction Loss** — one that
penalises high-frequency reconstruction errors more than low-frequency ones,
improving edge sharpness in the reconstructed MRI.

.. math::

   \mathcal{L}_{FreqW} = \frac{1}{HW}
   \sum_{k_x, k_y} w(k_x, k_y) \cdot |\hat{X}(k_x,k_y) - X(k_x,k_y)|^2

where the frequency weight is:

.. math::

   w(k_x, k_y) = 1 + \gamma \cdot \frac{\sqrt{k_x^2 + k_y^2}}{k_{max}}

with :math:`\gamma` controlling the high-frequency emphasis.


Step 1 — Implement the Loss Module
====================================

Create the file ``src/spectramr/models/losses/freq_weighted_loss.py``:

.. code-block:: python

   """Frequency-Weighted Reconstruction Loss.

   Penalises high-frequency errors more than low-frequency ones,
   improving edge sharpness in MRI reconstruction.
   """
   from __future__ import annotations

   import torch
   import torch.nn as nn
   from torch import Tensor


   class FreqWeightedLoss(nn.Module):
       """Frequency-weighted MSE in k-space.

       Args:
           gamma: High-frequency emphasis factor (0.0 = pure MSE).
           norm:  FFT normalisation — must be ``"ortho"``.
           reduction: ``"mean"`` or ``"sum"``.

       Example::

           loss_fn = FreqWeightedLoss(gamma=2.0)
           loss = loss_fn(pred, target)
       """

       def __init__(
           self,
           gamma: float = 2.0,
           norm: str = "ortho",
           reduction: str = "mean",
       ) -> None:
           super().__init__()
           if gamma < 0:
               raise ValueError(f"gamma must be >= 0, got {gamma}")
           self.gamma = gamma
           self.norm = norm
           self.reduction = reduction

       def _frequency_weight(self, h: int, w: int, device: torch.device) -> Tensor:
           """Build (H, W) frequency weight map centred at DC."""
           ky = torch.fft.fftfreq(h, device=device)   # [-0.5, 0.5)
           kx = torch.fft.fftfreq(w, device=device)
           # Shifted so DC is at centre
           ky = torch.fft.fftshift(ky)
           kx = torch.fft.fftshift(kx)
           grid_ky, grid_kx = torch.meshgrid(ky, kx, indexing="ij")
           radius = torch.sqrt(grid_ky**2 + grid_kx**2) * 2.0   # normalise to [0,1]
           return 1.0 + self.gamma * radius

       def forward(self, pred: Tensor, target: Tensor) -> Tensor:
           """Compute frequency-weighted MSE.

           Args:
               pred:   Predicted image ``(B, C, H, W)``.
               target: Ground-truth image ``(B, C, H, W)``.

           Returns:
               Scalar loss tensor.
           """
           if pred.shape != target.shape:
               raise ValueError(
                   f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
               )

           diff = pred - target                           # (B, C, H, W)
           h, w = diff.shape[-2], diff.shape[-1]

           # Transform residual to k-space
           kspace_diff = torch.fft.fft2(diff, norm=self.norm)   # complex
           kspace_diff_shift = torch.fft.fftshift(kspace_diff, dim=(-2, -1))

           # Squared magnitude
           sq_mag = kspace_diff_shift.abs() ** 2           # (B, C, H, W)

           # Apply frequency weight
           weight = self._frequency_weight(h, w, diff.device)   # (H, W)
           weighted = sq_mag * weight.unsqueeze(0).unsqueeze(0)

           if self.reduction == "mean":
               return weighted.mean()
           return weighted.sum()


Step 2 — Register the Loss
===========================

Open ``src/spectramr/models/losses/__init__.py`` and add the registration:

.. code-block:: python

   # src/spectramr/models/losses/__init__.py
   from spectramr.models.losses.freq_weighted_loss import FreqWeightedLoss

   # Register with the loss registry
   # (Assuming LOSS_REGISTRY is a dict defined in registry.py)
   from spectramr.models.losses.registry import LOSS_REGISTRY

   LOSS_REGISTRY["freq_weighted"] = FreqWeightedLoss

Alternatively, use the ``@register_loss`` decorator pattern if your
codebase uses it (check ``src/spectramr/models/losses/registry.py``):

.. code-block:: python

   from spectramr.models.losses.registry import register_loss

   @register_loss("freq_weighted", aliases=["FreqWeightedLoss"])
   class FreqWeightedLoss(nn.Module):
       ...


Step 3 — Write Unit Tests
==========================

**Rule:** Always write tests before using in training (TDD principle).

Create ``tests/unit/losses/test_freq_weighted_loss.py``:

.. code-block:: python

   """Unit tests for FreqWeightedLoss."""
   import pytest
   import torch

   from spectramr.models.losses.freq_weighted_loss import FreqWeightedLoss


   @pytest.fixture
   def loss_fn():
       return FreqWeightedLoss(gamma=2.0)


   def test_perfect_reconstruction_zero_loss(loss_fn):
       """Identical pred and target → zero loss."""
       x = torch.randn(2, 1, 64, 64)
       assert loss_fn(x, x).item() == pytest.approx(0.0, abs=1e-6)


   def test_output_is_scalar(loss_fn):
       """Loss must return a scalar tensor."""
       pred = torch.randn(2, 1, 64, 64)
       target = torch.randn(2, 1, 64, 64)
       out = loss_fn(pred, target)
       assert out.shape == torch.Size([])


   def test_gamma_increases_loss():
       """Higher gamma → larger loss (for non-zero residuals)."""
       pred = torch.randn(2, 1, 64, 64)
       target = torch.zeros_like(pred)
       loss_low = FreqWeightedLoss(gamma=0.0)(pred, target)
       loss_high = FreqWeightedLoss(gamma=5.0)(pred, target)
       assert loss_high.item() > loss_low.item()


   def test_gradient_flows():
       """Loss must be differentiable end-to-end."""
       pred = torch.randn(2, 1, 64, 64, requires_grad=True)
       target = torch.randn(2, 1, 64, 64)
       loss = FreqWeightedLoss(gamma=2.0)(pred, target)
       loss.backward()
       assert pred.grad is not None
       assert not pred.grad.isnan().any()


   def test_shape_mismatch_raises(loss_fn):
       """Shape mismatch → ValueError."""
       pred = torch.randn(2, 1, 64, 64)
       target = torch.randn(2, 1, 32, 32)
       with pytest.raises(ValueError, match="Shape mismatch"):
           loss_fn(pred, target)

Run tests:

.. code-block:: bash

   pytest tests/unit/losses/test_freq_weighted_loss.py -v


Step 4 — Create an Experiment Config
======================================

Create ``experiments/training/tutorial_05_freq_weighted.yaml``:

.. code-block:: yaml

   config_version: "1.0"

   data:
     dataset_type: fastmri_knee
     in_channels: 2   # rss produces real+imag
     out_channels: 2

     loader:
       batch_size: 8
       num_workers: 4
     coils:
       processing_mode: rss
     source:
       root: databases/fastmri/datasets/knee_singlecoil_train/

   # k-space in, image-domain U-Net out: bridge explicitly (NN#9, no silent rescue).
   adapters:
     pre_model:
       - name: ifft_kspace_to_image
       - name: complex_to_real_imag_interleave

   undersampling:
     acceleration_type: cartesian_vd
     base_acceleration: 4.0
     center_fraction: 0.08
   model:
     model_type: standard_unet
     in_channels: 4   # rss(2ch) -> ifft -> real/imag interleave = 4
     out_channels: 4

   training:
     training_mode: reconstruction
     output_dir: experiments/results/tutorial_05_freq_weighted_loss
     max_iterations: 30000

   losses:
     image_losses:
       # Combine with L1 for stability
       - name: l1
         weight: 5.0
         enabled: true
       # Our custom loss
       - name: freq_weighted
         weight: 2.0
         enabled: true
         kwargs:
           gamma: 3.0          # High-frequency emphasis
           norm: ortho
           reduction: mean
       # SSIM for perceptual quality
       - name: ssim
         weight: 1.0
         enabled: true

     policy:
       output_domain: image
   optimization:
     lr_scheduler_strategy: cosine
     warmup_steps: 1000
     gradient:
       clip:
         enabled: true
         method: norm
         value: 1.0

     optimizer:
       type: adamw
       learning_rate: 1e-4
       weight_decay: 1e-4
     precision:
       enabled: true
       dtype: bfloat16
   checkpoint:
     checkpoint_dir: checkpoints/tutorial_05
     save_interval: 5000
     best_metric_name: val_psnr
     best_metric_mode: max
     format: safetensors

   validation:
     enabled: true

     schedule:
       interval_steps: 2000
   physics:
     data_consistency:
       enabled: true
       method: hard

   metrics:
     compute_psnr: true
     compute_ssim: true
     compute_hfen: true     # High-frequency error norm (matches our loss goal)
     compute_lpips: false   # Skip for speed
   run:
     seed: 42
     device: cuda

   logging:
     identity:
       experiment: tutorial_05_freq_weighted_loss
     images:
       save_validation: true
     intervals:
       log: 50


Step 5 — Launch Training
==========================

.. code-block:: bash

   # Dry run first — validates config without GPU
   spectramr train --config experiments/training/tutorial_05_freq_weighted.yaml --dry-run

   # Full training
   spectramr train --config experiments/training/tutorial_05_freq_weighted.yaml

   # Monitor with TensorBoard
   tensorboard --logdir logs/tutorial_05_freq_weighted_loss/


Step 6 — Evaluate Edge Sharpness
===================================

After training, compare HFEN between the baseline L1-only model and
our frequency-weighted version:

.. code-block:: python

   import torch
   from spectramr.core.metrics.hfen import HFENMetric
   from spectramr.core.metrics.registry import compute_metric

   # Load predictions (B, 1, H, W)
   pred_baseline = torch.load("results/baseline_preds.pt")
   pred_freqw    = torch.load("results/freqw_preds.pt")
   targets       = torch.load("results/targets.pt")

   hfen = HFENMetric()
   print(f"Baseline HFEN: {hfen(pred_baseline, targets):.4f}")
   print(f"FreqW HFEN:    {hfen(pred_freqw, targets):.4f}")
   # Lower HFEN = better edge preservation

To measure what the weighting buys you, train the same arm three times —
changing only ``gamma`` and keeping ``seed`` fixed — and compare the three
runs' ``final_metrics.json``. Declare ``hfen`` in ``metrics.compute``
alongside ``psnr`` and ``ssim``, because it is the metric that responds to
edge preservation; PSNR alone will not show the effect you are looking for.

Expect the weighting to trade the three against each other rather than
improve all of them: emphasising high frequencies sharpens edges (HFEN
falls) at some cost in PSNR. The size of that trade depends on your corpus
and acceleration factor, so read it off your own runs.


Key Takeaways
=============

1. **Registry pattern** — add a loss without changing orchestration code
2. **YAML ``kwargs``** — all constructor args are configurable from config
3. **TDD first** — write tests before use in training
4. **Loss composition** — combine ``freq_weighted`` with ``l1`` for stability
5. **Metric alignment** — use ``compute_hfen: true`` to validate what your
   loss targets (high-frequency fidelity)
6. **Trade-offs** — higher ``gamma`` improves HFEN but slightly reduces PSNR;
   tune based on clinical priority (edge detection vs. quantitative accuracy)


Extensions
==========

- Apply ``gamma`` scheduling (increase during training)
- Use a **learnable** frequency weight (parameterize ``gamma``)
- Extend to 3D: ``torch.fft.fftn`` over ``(D, H, W)`` dimensions
- Combine with ``data_consistency`` for k-space-grounded edge preservation


See Also
========

- :doc:`tutorial_04_physics_constraints` — physics-constrained losses
- :doc:`../losses_reference` — all registered losses
- :doc:`../config_schema_reference` — ``losses:`` schema documentation
- :doc:`../troubleshooting` — NaN gradient debugging
