.. _strategies_reference:

================================================
Training Strategies — Architectural Reference
================================================

.. sectionauthor:: spectraMR Research

spectraMR's training strategies implement the **Template Method** design pattern
through ``BaseTrainingStrategy``. Each strategy encapsulates paradigm-specific
training logic while inheriting common infrastructure (checkpointing, EMA,
gradient clipping, AMP, metrics).

.. note::

   **This page is hand-written and not exhaustive.** It documents a subset of
   the strategy classes the registry holds, so a name's absence here is **not**
   evidence that it does not exist — ask the registry before concluding a
   strategy is unavailable:

   .. code-block:: python

      from spectramr.infrastructure.training.strategy_factory import (
          TrainingStrategyFactory,
      )

      paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS  # a CLASS attribute
      len(paths), len(set(paths.values()))

.. contents:: Table of Contents
   :depth: 2
   :local:


Loss composition and the SSOT seam
==================================

Most reconstruction-derived strategies get their objective from the declarative
``losses:`` block: ``ReconstructionTrainingStrategy._compute_losses_impl`` reads
``env.losses`` (composed by the ``LossBuilder``) and folds it through the
``UnifiedReconstructionLossComputer``. A number of paradigm strategies instead
override ``_compute_losses_impl`` and compute their objective **inline** (e.g.
``F.l1_loss`` plus a themed term whose weights live under ``training.<strategy>.*``).
For those strategies the declarative ``losses:`` block is, by itself, an inert
**decoy** — adding a term there changes nothing.

To let an inline strategy also honour the declarative block, the base class exposes
a seam, ``ReconstructionTrainingStrategy._apply_builder_image_losses``. The strategy
calls it with its grad-carrying ``(pred, target)`` image pair; every module the
``LossBuilder`` placed on ``env.losses`` is applied and folded onto the strategy's
total with the weight declared in ``config.losses.*_losses`` (resolved by
``_declared_loss_weights``). It returns ``None`` when no builder losses are
configured, so inline-only arms stay byte-identical. ``ScatteringBesovStrategy``
(MRIxFields2026 B-1.3) is the reference consumer: it keeps its inline
``L1 + scattering-Besov`` objective and folds declarative ``hfen`` / ``ms_ssim``
sharpness terms on top. Do **not** also list the strategy's inline term (e.g.
``l1``) in the declarative block, or it double-counts.


The lifecycle contract
======================

``BaseTrainingStrategy`` declares four lifecycle hooks — ``on_epoch_start``,
``on_epoch_end``, ``on_validation_start`` and ``on_validation_end``. They are
driven by
:class:`~spectramr.infrastructure.training.strategies.lifecycle.StrategyLifecycleDriver`,
which the training loop constructs once above the iteration loop and polls at
each boundary.

When each hook fires
--------------------

======================== ====================================================
Hook                     Fired
======================== ====================================================
``on_epoch_start``       the first iteration whose ``epoch`` index differs
                         from the previously-opened one, **before**
                         ``train_step`` — so an unfreeze taken here applies to
                         the epoch's first gradient step
``on_epoch_end``         at the epoch boundary, **after** the validation
                         block, for the epoch that actually *completed*
``on_validation_start``  immediately before every validation pass
``on_validation_end``    immediately after it, with the aggregated metrics
======================== ====================================================

Within the boundary iteration ``on_epoch_start(N + 1)`` precedes
``on_epoch_end(N)``. That order is forced, not accidental: the loop computes
``epoch = iteration // train_loader_len``, so at the boundary the index has
already advanced, its ``train_step`` is the first step of the new epoch, and the
boundary validation — the only source of fresh end-of-epoch metrics — runs after
that step. Firing ``on_epoch_end`` an iteration earlier would restore the
intuitive order at the cost of scoring each epoch on a measurement taken before
it finished. ``on_epoch_end`` is passed the index of the epoch that finished
(``N``), matching its own docstring, not the loop's current ``epoch``.

The **epoch** pair is gated exactly as epoch-based validation is: it needs a
non-empty train loader (``train_loader_len`` falls back to ``1`` for a missing
one, which would make ``epoch == iteration`` and fire ``on_epoch_start`` every
step), and it is skipped under ``--sanity-check``, because the hooks mutate
persistent strategy state and an overfit-one-batch pass must not spend an arm's
early-stopping patience. The **validation** pair is not gated: a sanity run
really does validate, and ``TrainingLoop.evaluate()`` drives the same pair so a
standalone evaluation and an in-training one stay indistinguishable, as that
method's contract promises.

Nothing is gated on the main rank. The hooks flip ``requires_grad`` on whole
stages, so a rank-0-only dispatch would desynchronise DDP parameter groups; both
inputs (the epoch index, the all-reduced ``val_metrics``) are already
rank-identical.

A hook a strategy does not implement is reported once and skipped. A hook that
raises propagates — an exception inside ``on_epoch_end`` means early stopping
did not evaluate, and a run continuing past that reports a guarantee it no
longer has.

Features that live only inside a hook
-------------------------------------

``MultiTrainingStrategy`` implements two schema-declared knobs entirely inside these
hooks, which is why they were the visible casualties of the missing driver:

``training.pipeline.end_to_end_finetune_epoch``
   Unfreezes every stage of a multi-stage pipeline once the epoch index reaches
   the threshold, and re-registers the newly-trainable parameters with the
   global optimizer. Implemented in ``on_epoch_start``.

per-stage ``early_stopping``
   Freezes a stage whose monitored validation metric has not improved for
   ``patience`` epochs. Implemented in ``on_epoch_end``.

The per-stage monitor resolves through
:func:`~spectramr.infrastructure.services.metric_keys.resolve_metric_key`, the
framework's single owner of monitor-key aliasing (the training loop's own early
stopping already routes through it). Its default is ``val_<stage>_l1`` — the key
``MultiTrainingStrategy.validation_step`` actually emits, and only when
``training.pipeline.evaluate_intermediates`` is ``true``. When no alias of the
configured monitor is present, the stage is warned about once and skipped rather
than silently never freezing.


Architecture Overview
=====================

Class Hierarchy
---------------

.. mermaid::

   classDiagram
       class TrainingStepStrategy {
           <<abstract>>
           +generator_model
           +discriminator_model
       }

       class BaseTrainingStrategy {
           +train_step()
           +on_epoch_start()
           +on_epoch_end()
           +on_validation_start()
           #_setup_strategy_specific_components()
           #_compute_generator_loss()
       }

       class ReconstructionTrainingStrategy
       class GANTrainingStrategy
       class DiffusionTrainingStrategy
       class VAETrainingStrategy
       class PhysicsDrivenTrainingStrategy

       TrainingStepStrategy <|-- BaseTrainingStrategy

       BaseTrainingStrategy <|-- ReconstructionTrainingStrategy
       BaseTrainingStrategy <|-- GANTrainingStrategy
       BaseTrainingStrategy <|-- DiffusionTrainingStrategy
       BaseTrainingStrategy <|-- VAETrainingStrategy

       ReconstructionTrainingStrategy <|-- PhysicsDrivenTrainingStrategy
       ReconstructionTrainingStrategy <|-- MaskedPretrainingStrategy

       DiffusionTrainingStrategy <|-- PaDNetTrainingStrategy


Mixin Composition
-----------------

All strategies compose shared behavior via ISP-compliant mixins:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mixin
     - Responsibility
   * - ``BatchPreparationMixin``
     - Extracts inputs/targets from TorchIO batches, handles k-space vs image
   * - ``EMAMixin``
     - Empty marker class. The actual EMA shadow update runs once per
       training step in ``src/spectramr/pipelines/train.py``
       (``pipeline.ema.update(pipeline.generator)``).
   * - ``KspaceMixin``
     - FFT/IFFT transforms, k-space mask generation, data consistency
   * - ``MetricsMixin``
     - Delegates metric computation to ``spectramr.core.metrics`` (SSOT)
   * - ``OptimizerMixin``
     - Builds the optimizer stepper (``_build_optimizer_stepper``). Runtime
       gradient ops (``_zero_gradients`` / ``_clip_and_log_gradients`` /
       ``_backward_and_step``) live on ``BaseTrainingStrategy``, not on the
       mixin.
   * - ``AdversarialMixin``
     - Discriminator forward/backward, gradient penalty computation
   * - ``ReconstructionMixin``
     - Supervised image loss computation (L1, SSIM, perceptual, etc.)
   * - ``ValidationMixin``
     - Validation step orchestration, image saving
   * - ``ModelValidationMixin``
     - Input/output shape validation, channel consistency checks


Training Environment
--------------------

Every strategy receives a frozen ``TrainingEnvironment`` dataclass (never raw
config dicts):

.. code-block:: python

   @dataclass(frozen=True)
   class TrainingEnvironment:
       config: TrainingSettings        # Immutable Pydantic config
       models: dict[str, nn.Module]    # {"generator": G, "discriminator": D}
       optimizers: dict[str, Optimizer] # {"generator": opt_g, ...}
       schedulers: dict[str, LRScheduler]
       device: torch.device
       services: dict[str, Any]        # DI-resolved services


Loss Results
------------

All training steps return a ``LossResult`` dataclass:

.. code-block:: python

   @dataclass
   class LossResult:
       losses: dict[str, torch.Tensor]  # Named loss tensors
       metrics: dict[str, float] | None  # Scalar metrics for logging
       g_total_loss: torch.Tensor | None # Total generator loss for backward()


Strategy Dispatch
-----------------

Strategy selection is performed by ``TrainingStrategyFactory``:

1. **Explicit**: ``config.training.strategy_class`` (fully qualified path)
2. **Schema inference**: Typed schema fields (``.gan``, ``.diffusion``, ``.vae``)
3. **Legacy**: ``config.training.training_mode`` fallback

.. code-block:: yaml

   # Method 1: Explicit strategy class
   training:
     strategy_class: spectramr.infrastructure.training.strategies.gan.GANTrainingStrategy

   # Method 2: Training mode dispatch
   training:
     training_mode: diffusion


---

Core Strategies
===============

ReconstructionTrainingStrategy
------------------------------

**File:** ``reconstruction.py`` — **Bases:** ``ModelValidationMixin``, ``ReconstructionMixin``, ``BaseTrainingStrategy``

**Purpose:** Supervised MRI image reconstruction from undersampled k-space.

**Training Objective:**

.. math::

   \mathcal{L} = \lambda_1 \|x - G(y)\|_1 + \lambda_{ssim}(1 - \text{SSIM}(x, G(y))) + \lambda_p \sum_l \|\phi_l(x) - \phi_l(G(y))\|_1

where :math:`x` is the fully-sampled target, :math:`y` is the undersampled
input, and :math:`G` is the reconstruction network.

**Training Loop:**

1. Extract ``(input, target)`` from batch via ``BatchPreparationMixin``
2. Forward pass: ``prediction = G(input)``
3. Compute composite reconstruction loss via ``ReconstructionMixin``
4. Backward pass with AMP scaling via the ``Trainer`` (the optimizer stepper
   built by ``OptimizerMixin._build_optimizer_stepper`` drives it)
5. Optional data consistency projection in k-space

**Key Configuration:**

.. code-block:: yaml

   training:
     training_mode: reconstruction
   losses:
     output_domain: image
     image_losses:
       - name: l1
         weight: 10.0
         enabled: true
       - name: ssim
         weight: 1.0
         enabled: true
       - name: perceptual
         weight: 0.5
         enabled: true


GANTrainingStrategy
-------------------

**File:** ``gan.py`` — **Bases:** ``BaseTrainingStrategy``, ``AdversarialMixin``

**Purpose:** Adversarial training with generator-discriminator alternation.

**Training Objective:**

.. math::

   \mathcal{L}_G = \lambda_{adv} \mathcal{L}_{adv}(G) + \lambda_1 \|x - G(y)\|_1 + \lambda_p \mathcal{L}_{perc}

.. math::

   \mathcal{L}_D = \mathcal{L}_{adv}(D) + \frac{\gamma}{2} \|\nabla_x D(x)\|^2

The generator loss combines adversarial, pixel-wise, and perceptual terms. The
discriminator loss includes the adversarial objective plus R1 gradient penalty.

**Training Loop (one step):**

1. **Discriminator step** (``AdversarialMixin``):

   a. Forward real: ``d_real = D(x)``
   b. Forward fake: ``d_fake = D(G(y).detach())``
   c. Compute :math:`\mathcal{L}_D` + R1 penalty
   d. Backward + optimizer step for D

2. **Generator step:**

   a. Forward: ``pred = G(y)``
   b. :math:`d_{fake} = D(pred)` (no detach)
   c. Compute :math:`\mathcal{L}_G`
   d. Backward + optimizer step for G

**Dynamic Balancing:** The strategy monitors ``d_loss / g_loss`` ratio and can
skip discriminator or generator steps to prevent mode collapse.

**Key Configuration:**

.. code-block:: yaml

   training:
     training_mode: gan
   losses:
     output_domain: image
     image_losses:
       - name: adversarial
         weight: 0.1
         enabled: true
         kwargs:
           loss_type: hinge      # vanilla | lsgan | hinge | wgan
     gan:
       disc_updates: 2           # canonical home for "D updates per G step"
       gan_loss_type: wgan-gp    # see GANLossesConfig in src/spectramr/config/schemas/loss.py


DiffusionTrainingStrategy
-------------------------

**File:** ``diffusion.py`` — **Bases:** ``BaseTrainingStrategy``, ``DiffusionStrategyMixin``, ``StrategyLoggingMixin``

**Purpose:** Train denoising diffusion models for MRI reconstruction.

**Forward Process (Fixed):**

.. math::

   q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t) I)

where :math:`\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)` and
:math:`\beta_t` is the noise schedule.

**Training Objective (noise prediction):**

.. math::

   \mathcal{L} = \mathbb{E}_{t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|_2^2 \right]

**Cold Diffusion Variant** (for k-space):

.. math::

   \mathcal{L}_{cold} = \| x_0 - D_\theta(\tilde{x}_t, t) \|_2^2

where :math:`\tilde{x}_t` is a deterministic degradation (e.g., k-space
undersampling mask) instead of Gaussian noise. The model predicts
:math:`x_0` directly (``prediction_type: sample``).

**Training Loop:**

1. Sample random timestep :math:`t \sim \text{Uniform}(1, T)`
2. Sample noise :math:`\epsilon \sim \mathcal{N}(0, I)` (or apply cold degradation)
3. Create noisy/degraded input :math:`x_t`
4. Predict: :math:`\hat{\epsilon} = \epsilon_\theta(x_t, t)` or :math:`\hat{x}_0 = D_\theta(x_t, t)`
5. Compute :math:`\mathcal{L} = MSE(\epsilon, \hat{\epsilon})` or :math:`MSE(x_0, \hat{x}_0)`
6. Backward + optimizer step

**Curriculum Timestep Scheduling** (Experiment 11):

When ``training.curriculum_start_timestep > 0``, the strategy progressively
unlocks harder timesteps as training progresses:

.. math::

   t_{max}(n) = t_{start} + \rho \cdot n

where :math:`n` is the current iteration, :math:`t_{start}` is
``curriculum_start_timestep``, and :math:`\rho` is ``curriculum_ramp_rate``.

This prevents the model from seeing near-fully-masked k-space too early,
before it has learned to handle moderate undersampling.

**Importance-Based Timestep Sampling** (Experiment 11):

When ``timestep_sampling_strategy: importance``, the timestep distribution
is re-weighted by current loss variance:

.. math::

   p(t) \propto \sqrt{\mathbb{E}\left[(\mathcal{L}(t) - \bar{\mathcal{L}})^2\right]}

This concentrates training budget on difficult timesteps (large
undersampling ratios) that have the highest loss variance.

**Identity Collapse Guard:**

``identity_collapse_threshold: 0.0005`` — if the total loss falls below
this value in the first ``early_training_steps`` iterations, training
aborts immediately. This catches silent failures where the model learns the
identity map (e.g., from incorrect data normalization).

**Cascading Validation:**

The strategy validates at multiple acceleration levels simultaneously
(``_CASCADING_LEVELS = [2, 8, 32]``), emitting metrics like
``val_psnr_2x``, ``val_psnr_8x``, ``val_psnr_32x``. This gives a
complete picture of zero-shot generalization without separate inference
runs.

**Key Methods:**

- ``sample_timesteps(batch_size)`` — Uniform or importance-weighted sampling
- ``q_sample(x_0, t, noise)`` — Forward diffusion process
- ``get_noise_prediction(model, x_t, t)`` — Reverse prediction
- ``_apply_curriculum(iteration)`` — Computes current ``t_max`` unlock
- ``_update_importance_weights(t, loss)`` — Updates per-timestep EMA loss

**Key Configuration:**

.. code-block:: yaml

   training:
     training_mode: diffusion
     diffusion:
       timesteps: 1000
       noise_schedule: cosine           # linear | cosine | sqrt
       prediction_type: sample          # sample (cold) or noise (DDPM)
       degradation: kspace_mask         # cold diffusion only
       sampler: cold_mri
       sampling_steps: 50

     # Curriculum (optional)
     curriculum_start_timestep: 100
     curriculum_ramp_rate: 0.005

     # Importance sampling (optional)
     timestep_sampling_strategy: importance
     identity_collapse_threshold: 0.0005


VAETrainingStrategy
-------------------

**File:** ``vae.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** Variational Autoencoder training with KL regularization.

**Training Objective (ELBO):**

.. math::

   \mathcal{L} = \underbrace{-\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}} + \underbrace{\beta \cdot D_{KL}(q(z|x) \| p(z))}_{\text{KL Regularization}}

For Gaussian encoder :math:`q(z|x) = \mathcal{N}(\mu, \sigma^2)` and prior
:math:`p(z) = \mathcal{N}(0, I)`:

.. math::

   D_{KL} = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)

**Reparameterization Trick:**

.. math::

   z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)

**Training Loop:**

1. Encode: :math:`\mu, \log\sigma^2 = E(x)`
2. Reparameterize: :math:`z = \mu + \sigma \odot \epsilon`
3. Decode: :math:`\hat{x} = D(z)`
4. Compute :math:`\mathcal{L} = \mathcal{L}_{recon} + \beta \cdot D_{KL}`

**Key Configuration:**

.. code-block:: yaml

   training:
     training_mode: vae
   losses:
     output_domain: image
     image_losses:
       - name: l1
         weight: 1.0
         enabled: true
       - name: kl
         weight: 0.001           # β-VAE weight
         enabled: true


VQVAETrainingStrategy
---------------------

**File:** ``vae.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** Vector-Quantized VAE with discrete latent codebook.

**Training Objective:**

.. math::

   \mathcal{L} = \|x - D(e_k)\|_2^2 + \|sg[z_e] - e_k\|_2^2 + \beta\|z_e - sg[e_k]\|_2^2

where :math:`e_k = \arg\min_{e \in \mathcal{C}} \|z_e - e\|` is the nearest
codebook entry, and :math:`sg[\cdot]` is the stop-gradient operator.


---

Physics-Informed Strategies
============================

PhysicsDrivenTrainingStrategy
-----------------------------

**File:** ``physics_driven_strategy.py`` — **Bases:** ``ReconstructionTrainingStrategy``

**Purpose:** PINN-based training with PDE constraints and data consistency.
Used for coil sensitivity estimation (Helmholtz PDE) and physics-regularized
reconstruction.

**Training Objective:**

.. math::

   \mathcal{L} = \underbrace{\| M \odot (\mathcal{F}(\hat{x}) - y) \|_2^2}_{\text{Data Consistency}} + \lambda_{PDE} \underbrace{\mathcal{R}_{PDE}(\hat{x})}_{\text{PDE Residual}} + \lambda_{TV} TV(\hat{x})

For PINN coil sensitivity estimation with the Helmholtz equation:

.. math::

   \mathcal{R}_{PDE} = \sum_c \left\| \nabla^2 S_c + k^2 S_c \right\|^2 \quad (k^2 \approx 0 \text{ at 0.3T})

**Additional Features:**

- Synthesizes B0 field maps for augmentation via ``synthesize_b0_map()``
- Supports collocation point sub-sampling for PDE loss efficiency
- Integrates with ``HelmholtzPDELoss`` (autograd-based PDE residual)

**Key Configuration:**

.. code-block:: yaml

   training:
     training_mode: pinn
     strategy_class: ...strategies.physics_driven_strategy.PhysicsDrivenTrainingStrategy
   physics:
     pinn:
       enabled: true
       pde_type: wave_equation
       collocation_points: 2048
   losses:
     output_domain: complex_image
     complex_losses:
       - name: pde
         weight: 0.1
         enabled: true


CycleBlochStrategy
------------------

**File:** ``cycle_bloch_strategy.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** Cycle-consistent physics training using Bloch equation simulation
for anatomically-faithful multi-contrast MRI synthesis.

**Training Objective:**

.. math::

   \mathcal{L} = \underbrace{\|x - G(y)\|_1}_{\text{Reconstruction}} + \lambda_{cyc} \underbrace{\|y - \text{Bloch}(G(y), \hat{\theta})\|_1}_{\text{Cycle-Bloch}} + \lambda_{param} \underbrace{\|\hat{\theta} - \theta_{ref}\|_2^2}_{\text{Parameter}}

The cycle-consistency term passes the reconstruction through the Bloch equation
simulator with estimated tissue parameters :math:`\hat{\theta} = (PD, T1, T2)`,
enforcing that the output is physically realizable.

**Architecture:**

- ``ParameterEstimator``: Lightweight CNN predicting :math:`(PD, T1, T2)` maps
- Bloch forward model: :math:`S = PD \cdot (1 - e^{-TR/T1}) \cdot e^{-TE/T2}`

**Key Configuration:**

.. code-block:: yaml

   training:
     training_mode: reconstruction
     strategy_class: ...strategies.cycle_bloch_strategy.CycleBlochStrategy

This strategy **requires a discriminator**: declare ``model.discriminator``, or
use ``CycleConsistencyStrategy`` if you do not want an adversarial term.


CycleGANTrainingStrategy
------------------------

**File:** ``cyclegan_strategy.py`` — **Bases:** ``BaseTrainingStrategy``, ``AdversarialMixin``
— **Key:** ``cyclegan`` (alias ``cycle_gan``)

**Purpose:** Unpaired cross-field / cross-contrast translation (MRIxFields2026
baseline). Two generators (``gen_ab`` A→B, ``gen_ba`` B→A) and two PatchGAN
discriminators (``disc_a``, ``disc_b``) are trained **without paired
supervision**: the batch ``target`` is a *real B-domain sample* for ``disc_b`` and
for B-side cycle/identity self-consistency, **never** a pixel-L1 target for
``gen_ab(input)`` (a paired term would silently collapse the arm to a supervised
denoiser).

**Training Objective:**

.. math::

   \mathcal{L}_G = \mathcal{L}^{LSGAN}_{adv} + \lambda_{cyc}\big(\|G_{BA}(G_{AB}(a)) - a\|_1 + \|G_{AB}(G_{BA}(b)) - b\|_1\big) + \lambda_{id}\big(\|G_{BA}(a) - a\|_1 + \|G_{AB}(b) - b\|_1\big)

The LSGAN adversarial term and the cycle/identity math are lifted (``.item()``-free)
from ``CycleGAN.forward``, but ``lambda_cycle`` / ``lambda_identity`` are read from
the config SSOT ``config.losses.gan`` (Task 5) rather than hard-coded. The
discriminators minimise ``0.5(\|D(real)-1\|_2^2 + \|D(fake)\|_2^2)`` on **detached**
fakes over both domains. ``_compute_losses_impl`` returns
``{g_total_loss, adv_g, adv_d, cycle, identity}``; validation forward is
``gen_ab(input)``.

**Key Configuration:**

.. code-block:: yaml

   training:
     training_mode: gan
     strategy_class: ...strategies.cyclegan_strategy.CycleGANTrainingStrategy
   model:
     model_type: cyclegan_generator   # gen_ab / gen_ba (ResNet); disc = patch_gan
   losses:
     gan:
       lambda_cycle: 10.0
       lambda_identity: 0.5


CUTTrainingStrategy
-------------------

**File:** ``cut_strategy.py`` — **Bases:** ``BaseTrainingStrategy``, ``AdversarialMixin``
— **Key:** ``cut``

**Purpose:** Single-generator Contrastive Unpaired Translation (CUT, Park et al.,
ECCV 2020) — the MRIxFields2026 baseline that replaces CycleGAN's second generator +
cycle-consistency with a patch-wise InfoNCE (``cut_patch_nce``). ONE generator ``G``
(``cyclegan_generator``) + ONE PatchGAN discriminator (``patch_gan``). The batch
``target`` is used **only** as a real B-domain sample for ``D``; there is **no**
pixel-L1 between ``G(input)`` and ``target`` (a paired term would collapse the arm to
a supervised denoiser).

**Training Objective:**

.. math::

   \mathcal{L}_G = \mathcal{L}^{LSGAN}_{adv}\big(D(G(x))\big) + \lambda_{nce}\,\mathcal{L}_{PatchNCE}\big(E(x),\, E(G(x))\big)

where :math:`E` is the encoder half of :math:`G`. ``_encode_features`` registers
forward hooks on the encoder submodules (the post-conv ReLUs at the 64/128/256-channel
scales plus the first ResnetBlocks) to capture ``feat_source = E(x)`` and
``feat_translated = E(G(x))`` as equal-length lists for the NCE ``context`` dict.
The PatchNCE projection head (``cut_patch_nce``'s ``_PatchSampleMLP``) builds its
per-channel MLPs **lazily**, so ``setup_models`` runs a no-grad warm-up to materialise
them and folds their parameters into ``opt_g`` (CUT's ``optimizer_F`` merged into the
generator optimizer) — otherwise the NCE would fire but never learn its projection.
``lambda_nce`` is read from the config SSOT ``config.losses.gan``.
``_compute_losses_impl`` returns ``{g_total_loss, adv_g, nce}`` (the discriminator loss
is owned by the D-closure); validation forward is ``generator(input)``.

**Key Configuration:**

.. code-block:: yaml

   training:
     training_mode: gan
     strategy_class: ...strategies.cut_strategy.CUTTrainingStrategy
   model:
     model_type: cyclegan_generator   # generator (ResNet); disc = patch_gan
   losses:
     gan:
       lambda_nce: 1.0


StarGANv2TrainingStrategy
-------------------------

**File:** ``stargan_v2_strategy.py`` — **Bases:** ``BaseTrainingStrategy``, ``AdversarialMixin``
— **Key:** ``stargan_v2``

**Purpose:** Multi-domain *any-to-any* FIELD translation (MRIxFields2026 baseline,
StarGAN v2 — Choi et al., CVPR 2020). The domains are the five discrete field
levels ``{0.1, 1.5, 3, 5, 7} T``; a continuous ``field_strength`` is snapped to the
nearest level by :meth:`_field_to_domain` (vectorised nearest-neighbour →
``long`` domain index 0..4). Four networks are trained: the style-conditioned
generator ``G(x, s)`` (``stargan_v2_generator``), the mapping network ``F(z, y)``,
the style encoder ``E(x, y)``, and the multi-domain discriminator ``D(x, y)``
(``stargan_v2_discriminator``). There is **no** paired pixel-L1 between the fake and
the real target — the only reconstruction anchor is the CYCLE back to the source.

**Training Objective (generator side):**

.. math::

   \mathcal{L}_G = \mathcal{L}^{LSGAN}_{adv}\big(D(G(x, s_1), y_t)\big)
   + \lambda_{sty}\,\|E(G(x, s_1), y_t) - s_1\|_1
   - \lambda_{div}\,\|G(x, s_1) - G(x, s_2)\|_1
   + \lambda_{cyc}\,\|G(G(x, s_1), E(x, y_s)) - x\|_1

with :math:`s_i = F(z_i, y_t)` from two sampled latents (the *diverse* branch); the
diversification term is **maximised** (hence the negative sign — a constant
:math:`\lambda_{div}` for this baseline; the paper's decay schedule is intentionally
not built). The discriminator minimises the ``gan_lsgan`` term on the real
``D(target, y_t)`` + the detached fake ``D(G(x, s_1)^{-}, y_t)`` plus an ``r1``
gradient penalty on the real image.

**Optimiser wiring (anti-facade):** StarGAN v2 trains ``{G, F, E}`` together and
``D`` separately. ``setup_models`` reuses the env ``stargan_v2_generator`` /
``stargan_v2_discriminator`` when present, always builds the (unregistered)
mapping-network + style-encoder, and folds their parameters into ``opt_g`` — so an
unoptimised mapping net can never silently collapse the arm to a plain AdaIN denoiser.
``_compute_losses_impl`` returns
``{g_total_loss, adv_g, style, diversity, cycle}`` (each already lambda-weighted);
validation forward renders at the sample's target field — reference-guided
``G(x, E(target, y_t))`` when a reference is available, else latent-guided
``G(x, F(0, y_t))``.

**Key Configuration:**

.. code-block:: yaml

   training:
     training_mode: gan
     strategy_class: ...strategies.stargan_v2_strategy.StarGANv2TrainingStrategy
   model:
     model_type: stargan_v2_generator   # G(x,s); disc = stargan_v2_discriminator
   losses:
     gan:
       lambda_style: 1.0
       lambda_diversity: 1.0
       lambda_cycle: 1.0
       lambda_r1: 1.0


B0MappingStrategy
-----------------

**File:** ``b0_mapping_strategy.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** Neural B0 field estimation for geometric distortion correction.

**Training Objective:**

.. math::

   \mathcal{L} = \|B_0^{pred} - B_0^{ref}\|_2^2 + \lambda_{smooth} \|\nabla B_0^{pred}\|_2^2

where the smoothness term enforces physical plausibility of the field map.


---

Self-Supervised Strategies
===========================

MAEPretrainingStrategy
----------------------

**File:** ``pretraining.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** Masked Autoencoder pretraining for self-supervised MRI
feature learning.

**Training Objective:**

.. math::

   \mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} (x_i - \hat{x}_i)^2

where :math:`\mathcal{M}` is the set of masked patch indices. Only the
reconstruction of masked regions contributes to the loss (visible patches are
ignored).

**Pipeline:**

1. Divide image into :math:`P \times P` patches
2. Randomly mask 75% of patches
3. Encode visible patches with ViT encoder
4. Decode all patches (visible + masked) with lightweight decoder
5. Compute MSE on masked patches only


MaskedPretrainingStrategy
-------------------------

**File:** ``masked_strategy.py`` — **Bases:** ``ReconstructionTrainingStrategy``

**Purpose:** Masked Image Modeling (MIM) with reconstruction loss.
Extends ``ReconstructionTrainingStrategy`` with a masking step before
the forward pass.


NoiseToNoiseStrategy
--------------------

**File:** ``n2n_strategy.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** Self-supervised denoising via the Noise2Noise paradigm.

**Key Insight:** If noise is zero-mean, training on noisy-to-noisy pairs
produces the same optimal solution as training on noisy-to-clean:

.. math::

   \hat{f} = \arg\min_f \mathbb{E}[\|f(y_1) - y_2\|^2] = \arg\min_f \mathbb{E}[\|f(y_1) - x\|^2]

where :math:`y_1, y_2` are independent noisy observations of clean signal :math:`x`.


---

Domain Adaptation Strategies
==============================

DomainAdaptationTrainingStrategy
--------------------------------

**File:** ``domain_adaptation.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** Cross-field-strength domain adaptation for MRI (e.g., 0.3T → 3T).

**Training Objective (DANN-style):**

.. math::

   \mathcal{L} = \mathcal{L}_{task}(f \circ g; x_s, y_s) - \lambda_{da} \mathcal{L}_{domain}(d \circ g; x_s, x_t)

where :math:`g` is a shared feature extractor, :math:`f` is the task head,
:math:`d` is the domain discriminator, and the negative sign implements
gradient reversal.

**Key Methods:**

- ``initialize_domain_adaptation()`` — Sets up domain discriminator
- ``compute_domain_loss()`` — GRL-based domain confusion


DisentangledTrainingStrategy
----------------------------

**File:** ``disentangled_strategy.py`` — **Bases:** ``AdversarialMixin``, ``ReconstructionMixin``, ``BaseTrainingStrategy``

**Purpose:** Content-style disentanglement for multi-contrast MRI synthesis.

**Architecture:**

.. math::

   z_{content} = E_c(x), \quad z_{style} = E_s(x), \quad \hat{x} = G(z_{content}, z_{style})

**Training Objective:**

.. math::

   \mathcal{L} = \mathcal{L}_{recon} + \lambda_{adv}\mathcal{L}_{adv} + \lambda_{content}\|z_c^{(1)} - z_c^{(2)}\|_2^2 + \lambda_{style}\mathcal{L}_{style}

Content codes from the same anatomy should match; style codes from different
contrasts should diverge.


DisentangledVAETrainingStrategy
-------------------------------

**File:** ``disentangled_vae_strategy.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** VAE-based cross-physics anatomy disentanglement with
KL regularization on both content and style latent spaces.


---

Specialized Strategies
=======================

XDiffusionTrainingStrategy
--------------------------

**File:** ``diffusion.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** X-Diffusion for cross-modal multi-contrast synthesis.

The denoising step is the generic cosine-schedule ``x_0``-prediction shared with
``DiffusionTrainingStrategy``. **Source-modality conditioning** is opt-in via an
optional ``training.diffusion.cross_modal`` block:

.. code-block:: yaml

   training:
     diffusion:
       timesteps: 1000
       cross_modal:
         enabled: true
         condition_encoder: conv        # only "conv" — any other value RAISES (#9)
         condition_embedding_dim: 64     # must be > 0
         source_modality: t1
         target_modalities: [t2, flair]
         num_contrasts: 2

When the block is present, every key is **read, validated, and stamped** into
``XDiffusionTrainingStrategy.cross_modal_provenance``. A
strategy-owned condition encoder (a plain ``nn.Module`` attribute — the strategy
itself is not an ``nn.Module``) encodes ``input_batch`` to a spatial ``z_cond``
embedding, has its parameters registered on the generator optimizer via
``opt_g.add_param_group`` (so it actually trains), and the embedding is fed to
the generator's ``forward`` through its ``cond`` / ``context`` / ``condition``
kwarg. If cross-modal is enabled but the generator accepts none of those kwargs,
construction/step **raises** rather than silently dropping the condition.
When the block is absent the strategy degrades to a plain
single-modality denoiser and advertises no unread knob.

Conditions the diffusion reverse process on a reference contrast:

.. math::

   \epsilon_\theta(x_t^{(B)}, t, x_0^{(A)})

where :math:`x_0^{(A)}` is the source contrast (e.g., T1) used as
conditioning for generating target contrast :math:`x^{(B)}` (e.g., T2).

**Dispatch keys.** Set one of these as ``training.training_mode`` in
the YAML (or use ``training.strategy_class`` for fully explicit dispatch):

* ``x_diffusion`` — short name that matches the
  ``@register_model(name="x_diffusion", ...)`` model_type.
* ``cross_modal_diffusion`` — descriptive alias matching the
  strategy's own ``expected_modes`` advertisement.

Both keys are listed in
``TrainingStrategyFactory.STRATEGY_CLASS_PATHS``. Without them the
dispatcher's legacy ``training_mode`` lookup silently falls back to
``DiffusionTrainingStrategy``.


GraphColdDiffusionStrategy
---------------------------

**File:** ``graph_cold_diffusion_strategy.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** Cold diffusion on graph-structured non-Cartesian k-space data.

Uses graph neural networks to operate on irregular k-space sampling
trajectories (spiral, radial) with deterministic degradation operators.


PaDNetTrainingStrategy
----------------------

**File:** ``padnet_strategy.py`` — **Bases:** ``DiffusionTrainingStrategy``

**Purpose:** Physics-Driven Parameter Mapping Network combining diffusion
denoising with quantitative parameter estimation (T1, T2, PD mapping).


MetaLearningTrainingStrategy
-----------------------------

**File:** ``meta_learning_strategy.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** MAML/Reptile-based meta-learning for anatomy-agnostic adaptation.

**MAML Inner Loop:**

.. math::

   \theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}(\mathcal{T}_i; \theta)

**Meta Update:**

.. math::

   \theta \leftarrow \theta - \beta \sum_i \nabla_\theta \mathcal{L}(\mathcal{T}_i; \theta'_i)


TttAdaptationStrategy
---------------------

**File:** ``test_time_adaptation_strategy.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** Test-Time Training (TTT) for online adaptation. Adapts model
parameters at inference time using self-supervised objectives on the test
sample itself (e.g., data consistency in k-space).


DIFFSirenStrategy
-----------------

**File:** ``diff_siren.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** Distributed Implicit Fiducial Framework (DIFF) with SIREN
curriculum. Trains coordinate-based networks with progressive frequency
scheduling.


TRELLISTrainingStrategy
-----------------------

**File:** ``volumetric.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** 3D asset generation from 2D MRI slices using regression-based
volumetric reconstruction.


StandardTrainingStrategy
------------------------

**File:** ``standard_strategy.py`` — **Bases:** ``BaseTrainingStrategy``

**Purpose:** Minimal supervised training with user-defined loss. Useful as a
fallback or for quick prototyping.


---

Frontier & Specialized Strategies
====================================

NoiseToNoiseStrategy
----------------------

**File:** ``n2n_strategy.py`` — **Mode:** ``n2n``

**Purpose:** Self-supervised denoising from multi-repetition MRI data
(M4Raw dataset). Requires no clean ground truth — exploits the
**Noise2Noise** principle: since noise is uncorrelated between repetitions,
the network learns to predict one noisy repetition from another.

**Batch unpacking:** Expects ``image_reps`` tensor of shape ``(B, N, C, H, W)``
where ``N`` is the number of repetitions. Randomly selects rep ``i`` as input
and rep ``j ≠ i`` as target per training step.

.. math::

   \mathcal{L}_{N2N} = \mathbb{E}_{i \neq j}[\mathcal{L}(G(x^{(i)}), x^{(j)})]

**Key config:**

.. code-block:: yaml

   training:
     training_mode: n2n
   data:
     dataset_type: m4raw_multi_rep
     num_repetitions: 3


MetaLearningTrainingStrategy
------------------------------

**File:** ``meta_learning_strategy.py`` — **Mode:** ``meta_learning``

**Purpose:** MAML-style outer-loop meta-learning for anatomy-agnostic
rapid adaptation. Enables a single model to specialize to new anatomies
or contrasts in 5–10 inner gradient steps.

**Algorithm:**

.. math::

   \theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{query}
   \bigl(\theta - \alpha \nabla_\theta \mathcal{L}_{support}(\theta)\bigr)

- **Inner loop**: ``adaptation_steps`` SGD steps on the per-task support
  set, at ``adaptation_lr``
- **Outer loop**: meta-gradient on the held-out query set

**Implementation.** Model-agnostic **second-order MAML** via
:py:func:`torch.func.functional_call` against the model's *real* ``forward``.
There is **no** ``adapt_to_domain`` interface — no registered model implements
one. The loop is wrapped in :py:func:`torch.enable_grad` because it
differentiates *through* the forward pass. Set ``first_order: true`` for
the cheaper FOMAML. Hyperparameters are scanned across **every** config
home so the knobs are never a silent no-op:
``model.adaptation_config`` (``AdaptationConfig`` — ``adaptation_lr`` /
``adaptation_steps``), the ``training.meta`` block (``first_order``), and
``model.model_kwargs`` (``meta_lr_inner`` / ``inner_steps``).

**Key config:**

.. code-block:: yaml

   training:
     training_mode: meta_learning
   model:
     model_type: meta_learning_friendly
     adaptation_config:
       adaptation_lr: 0.01
       adaptation_steps: 5



ConcreteDistillationStrategy
------------------------------

**File:** ``distillation_strategy.py`` — **Mode:** ``distillation``

**Purpose:** Teacher-Student latent knowledge distillation for the
Virtual Fiducial pipeline. The frozen **Teacher** (CrossAttentionOracleUNet)
receives ground-truth degradation markers; the **Student** learns to
produce the same latent ``Z`` from anatomy alone (no markers at test time).

**Loss:**

.. math::

   \mathcal{L} = \mathcal{L}_{recon}(\hat{x}_{student}, x_{clean})
   + \lambda_{distill} \| Z_{student} - Z_{teacher} \|^2
   + \lambda_{marker} \mathcal{L}_{marker}(\hat{x}_{student})

All simulator parameters come from ``config.physics.digital_twin`` (SSOT).


GuidedSuperResolutionStrategy
-------------------------------

**File:** ``guided_sr_strategy.py`` — **Mode:** ``guided_sr``

**Purpose:** Reference-guided ULF→HF super-resolution using content-style
disentanglement. A subset of HF slices act as a style reference:

1. ``ContentEncoder(ULF_full)`` → anatomical content embedding
2. ``StyleEncoder(HF_reference)`` → style embedding from :math:`K` HF slices
3. ``Generator(content, style)`` → full HF volume

**Loss:**

.. math::

   \mathcal{L} = \mathcal{L}_{recon}(\hat{x}, x_{HF})
   + \mathcal{L}_{adv} + \lambda_{ref} \mathcal{L}_{ref\_consistency}

At inference time, style comes from the patient's actual partial HF scan,
enabling zero-shot adaptation to scanner-specific characteristics.


ConcreteTTOStrategy
---------------------

**File:** ``tto_strategy.py`` — **Mode:** ``test_time_optimization``

**Purpose:** Stage-2 of the Virtual Fiducial pipeline. Freezes ALL network
weights and optimizes **only** the motion trajectory :math:`\hat{\theta}(t)`
to minimize data consistency against raw M4Raw scanner measurements.
A zero-shot approach — no paired ground truth required.

**Optimization loop (test-time, not training):**

.. math::

   \min_{\hat{\theta}} \| y_{real} - H_{\hat{\theta}}(\hat{x}) \|^2
   + \lambda_{TV} \| \nabla \hat{x} \|_1

where :math:`\hat{x} = G(y_{real}, P_{\hat{\theta}})` and
:math:`H_{\hat{\theta}}` is the ``KinematicForwardOperator``.

.. important::

   Despite being called a "strategy," TTO is invoked during **inference**,
   not training. Use ``--resume`` with a pre-trained ``HyperMambaUNet``
   checkpoint and ``training_mode: test_time_optimization``.


ConcreteVFADMMStrategy
------------------------

**File:** ``vf_admm_strategy.py`` — **Mode:** ``vf_admm``

**Purpose:** ADMM-based reconstruction for the Marker-Anchored Digital Twin
pipeline. Solves the joint reconstruction-regularization problem via operator
splitting:

.. math::

   x^{k+1} &= \arg\min_x \| Ax - y \|^2 + \rho \| x - (z^k - u^k) \|^2 \\
   z^{k+1} &= G_\theta(x^{k+1} + u^k)  \quad \text{(neural proximal)} \\
   u^{k+1} &= u^k + x^{k+1} - z^{k+1}

After each ADMM step, ``MarkerPriorProjection`` anchors marker voxels to
known physical values, preventing hallucination at calibration regions.

All simulator parameters from ``config.physics.digital_twin`` (SSOT).


GraphColdDiffusionStrategy
----------------------------

**File:** ``graph_cold_diffusion_strategy.py`` — **Mode:** ``graph_cold_diffusion``

**Purpose:** Cold Diffusion for non-Cartesian MRI using Graph Neural Networks.
Degradation is defined by progressively removing k-space spokes/arms
(not Gaussian noise):

.. math::

   x_t = D_t(x_0) \quad \text{(deterministic undersampling, t=0 fully-sampled)}

Backward process learns :math:`x_{t-1} = R_\theta(x_t, t)` in GNN-compatible
coordinate representations (nodes = k-space samples, edges = trajectory
neighborhood).

**Key distinction from** ``DiffusionTrainingStrategy``:
no Gaussian noise schedule; degradation is purely deterministic k-space removal.


GeoMambaULFStrategy
---------------------

**File:** ``geomamba_ulf_strategy.py`` — **Mode:** ``geomamba_ulf``

**Purpose:** Paired ULF→HF MRI super-resolution using the GeoMamba-ULF
architecture. The full forward path:

.. math::

   \hat{I}_{HF}^{(c)} = R\!\left(
       \pi^{-1} \circ f_\theta^{(c)} \circ \pi \circ \varphi^{-1}(I_{LF}^{(c)})
   \right)\!\Big|_F

where :math:`\varphi^{-1}` is ``PhysicalUnwarp`` (B0 + gradient non-linearity),
:math:`\pi` is the space-filling curve reordering, :math:`f_\theta^{(c)}`
is the contrast-conditioned Mamba, and :math:`R` is background re-embedding.

**Composite loss:**

.. code-block:: yaml

   losses:
     image_losses:
       - name: l1
         weight: 10.0     # foreground-masked
         enabled: true
     complex_losses:
       - name: cubical_ph_w2
         weight: 0.1
         enabled: true
       - name: beltrami_diagnostic
         weight: 0.05
         enabled: true


PaDNetTrainingStrategy
------------------------

**File:** ``padnet_strategy.py`` — **Bases:** ``DiffusionTrainingStrategy``

**Mode:** ``padnet``

**Purpose:** Physics-Driven Parameter Mapping Network. Extends diffusion
training to generate quantitative tissue parameter maps (T1, T2, T2*).

**Pipeline:**

1. Condition: Input T1w image → encoder → condition embedding
2. Latent diffusion: generates parameter latent codes
3. Decoder: latent → T1/T2/T2* maps
4. Bloch simulator: parameters → synthetic MR signal
5. Loss on synthetic vs acquired signal

**Physics loss** enforces tissue parameter ranges:

.. math::

   \mathcal{L}_{phys} = \max(0, T_1 - T_{1,max}) + \max(0, T_{2,min} - T_2)

Expected ranges: Brain T1=500–2000ms, T2=20–100ms.


DisentangledDiffusionStrategy
-------------------------------

**File:** ``disentangled_diffusion_strategy.py`` — **Mode:** ``disentangled_diffusion``

**Purpose:** Combines disentangled content-style representation with diffusion
denoising. The generator predicts noise in a factorized latent space where
anatomy and modality/contrast codes are separated:

.. math::

   \mathcal{L} = \mathcal{L}_{diff}(z_{anat} + z_{mod})
   + \lambda_{CA} \mathcal{L}_{contrastive}(z_{anat})
   + \lambda_{cyc} \mathcal{L}_{cycle}(z_{mod})


PMAVarNetStrategy
------------------

**File:** ``pma_varnet_strategy.py`` — **Mode:** ``pma_varnet``

**Purpose:** Physics-Motivated Accelerated VarNet. Unrolled gradient descent
with sensitivity-weighted data consistency at each cascade:

.. math::

   x^{(k+1)} = x^{(k)} - \eta_k \left[
       \underbrace{S^H \mathcal{F}^H M^H(\mathcal{F} S x^{(k)} - y)}_{\text{k-space DC}}
       + \lambda_k \mathcal{R}_\theta(x^{(k)})
   \right]

Extends ``VarNet`` by accepting pre-estimated coil maps from
``EspiritCalibration`` or ``SirenSensNet``.


MotionMetaStrategy
-------------------

**File:** ``motion_meta_strategy.py`` — **Mode:** ``motion_meta``

**Purpose:** Meta-learning strategy specialized for motion-corrupted MRI.
Maintains a **motion trajectory memory buffer** that tracks acquisition-time
motion patterns. The inner-loop adapts the reconstruction network using
motion priors extracted from navigator signals or self-navigating data.


OperatorIdBCHTrainingStrategy
-----------------------------

**File:** ``operator_id_bch_strategy.py`` — **Mode:** ``operator_id`` —
**Short name:** ``operator_id_bch`` — **Bases:** ``BaseTrainingStrategy``,
``KspaceMixin``

**Purpose:** Lie-algebraic effective-generator identification of the composite
degradation operator of an MRI scanner from data (Proposal 1). Rather than
fitting per-mode severities independently, every catalogued degradation mode is
embedded as a flow in **one affine Lie algebra**, so the Baker-Campbell-Hausdorff
(BCH) series governs the whole composite operator.

**Identified operator:**

.. math::

   \mathcal{D}(\cdot\,;\theta) = \mathcal{N}_{\Sigma(u)} \circ
       \exp\!\big(\Omega(s(u))\big),\qquad
   \Omega(s) = \sum_k s_k L_k
             + \tfrac12 \sum_{j<k} s_j s_k [L_j, L_k] + \cdots,

with :math:`u=(c,\beta)` the contrast/field coordinate, :math:`L_k` the
generator of mode :math:`k`, and
:math:`\Sigma(u) = \mathcal{F}^{-1}\operatorname{diag}(\rho(u))\mathcal{F} +
U(u)U(u)^{H}` the structured (k-space-diagonal plus low-rank) noise covariance.

**Training objective:** exact Gaussian likelihood on paired data plus
pushforward maximum-mean-discrepancy on unpaired data,

.. math::

   \mathcal{L} = \underbrace{\tfrac1{N_p}\sum_n\big[r_n^{H}\Sigma^{-1} r_n
       + \log\det\Sigma\big]}_{\text{structured\_gaussian\_nll}}
       + \lambda\,\underbrace{\mathrm{MMD}^2_{\mathcal H}}_{\text{pushforward\_mmd}},
   \quad r_n = y_n - \exp(\Omega(s(u_n)))\,x_n.

**Single sources of truth (under** ``infrastructure/physics/`` **):**

* ``magnus_exponential.py`` — matrix-free Krylov action :math:`\exp(\Omega)x`
  (Arnoldi/Lanczos), BCH folding to order 1/2/3, autograd-differentiable.
* ``degradation_generators.py`` — the operator basis: each mode as a
  matrix-free generator :math:`L_k` with a verified Hermitian adjoint
  (``motion_rigid``, ``bias_field_multiplicative``, ``b0_phase``,
  ``gibbs_truncation`` enter :math:`\exp(\Omega)`; ``rician_noise`` /
  ``gaussian_noise`` enter :math:`\Sigma`).
* ``structured_covariance.py`` — ``apply`` / ``solve`` / ``logdet`` for
  :math:`\Sigma` via FFT and the Woodbury / determinant lemma.

**Model:** ``bch_operator_conditioner`` maps :math:`u=(c,\beta)` to per-mode
severities :math:`s` (softplus), the radial noise PSD :math:`\rho` (softplus),
and complex low-rank covariance gains.

**Audit (Tier-1):** ``operator_basis_registered``, ``covariance_rank_bound``,
``bch_order_supported``, ``krylov_dim_valid`` fire whenever
``training.operator_id`` is present.

**Audit (Tier-2 probe):** because the conditioner returns a *parameter
dictionary* (not an image), the generic ``synthetic_forward_probe`` does not
apply; ``python -m spectramr.cli audit <yaml> --probe`` dispatches instead to
:func:`spectramr.infrastructure.validation.operator_id_probe.operator_id_forward_probe`.
On a Shepp-Logan phantom with the configured ``mode_dictionary`` it asserts,
at config-load time, the four correctness properties from the design: (i)
:math:`\exp(\Omega(s))x` has matching shape and only finite values; (ii) the
gradient of the structured-Gaussian NLL w.r.t. the conditioner is non-zero
(the operator is actually fitted); (iii) every generator satisfies the
adjoint identity :math:`\langle L_k u, v\rangle = \langle u, L_k^{H} v\rangle`
to ``1e-4`` (the gating risk for ``gradcheck``); and (iv) at :math:`s\equiv 0`
the operator reduces to the identity. A failing arm is rejected before
training starts.

**Key Configuration:**

.. code-block:: yaml

   training:
     strategy_class: operator_id_bch
     operator_id:
       mode_dictionary:
         - motion_rigid
         - bias_field_multiplicative
         - b0_phase
         - gibbs_truncation
         - rician_noise
       bch_order: 2
       krylov_dim: 30
       covariance_rank: 8
       mmd_weight: 1.0
   losses:
     complex:
       - structured_gaussian_nll
     image:
       - pushforward_mmd

**Reporting figures:** the recovered severity field :math:`s(c,\beta)` across
contrasts and field, alongside a commutator-interaction heatmap
:math:`\|[L_j, L_k]\|_F` showing which mode pairs interact most strongly (i.e.
where the Baker-Campbell-Hausdorff correction beyond first order actually
matters). Both are produced by
:mod:`spectramr.models.analysis.operator_id_report`:

* :func:`~spectramr.models.analysis.operator_id_report.commutator_interaction_matrix`
  -- pure physics, reproducible from the ``mode_dictionary`` with no trained
  model (a static diagnostic of how non-commutative a catalog is).
* :func:`~spectramr.models.analysis.operator_id_report.severity_field` -- reads
  a trained conditioner on a ``(contrast, field)`` grid.

**Calibrated operator posterior (SOTA plan T3).** Setting
``operator_id.operator_posterior: laplace`` equips the point-estimate operator
with a Laplace uncertainty posterior over the per-mode severities,
:math:`\Sigma = (H + \lambda_0 I)^{-1}` with ``H`` the Hessian of the
structured-Gaussian NLL at the MAP
(:mod:`spectramr.models.operator_id.operator_posterior`). The strategy stamps the
mean posterior std into the metrics (``operator_posterior_std``) — the
uncertainty that distinguishes T3 from the REJECTed blind operator-image
diffusion (JSMoCo), whose image prior silently absorbs operator error. **Corner
case:** ``posterior_prior_precision`` :math:`\lambda_0 \to \infty` ⇒
:math:`\Sigma \to 0` ⇒ the BCH point estimate (the incumbent). **Identifiability
(honest):** a meaningful posterior needs complex multi-coil k-space
(Ahmed–Recht–Romberg) — the ``operator_posterior_requires_complex`` audit rejects
it on magnitude-only data (so there is, by design, no M4Raw posterior arm).

The rendering helpers below are importable directly; the maintainers' figure
driver that calls them is not distributed, so build the report from the module:

.. code-block:: python

   from spectramr.models.analysis.operator_id_report import (
       commutator_interaction_matrix,
       severity_field,
   )

.. automodule:: spectramr.models.analysis.operator_id_report
   :members: commutator_interaction_matrix, severity_field
   :no-index:


AcquisitionHypernetworkStrategy (LCAH)
---------------------------------------

**File:** ``hypernetwork_strategy.py`` — **Mode:** ``acq_hypernetwork`` —
**Short name:** ``acq_hypernetwork`` — **Bases:**
``ReconstructionTrainingStrategy``

**Purpose:** Train :class:`~spectramr.models.encoders.lcah_encoder.LCAHEncoder`, a
spectral-normalised hypernetwork :math:`h_\psi` that maps the continuous
acquisition vector :math:`\varphi=(\mathrm{TE},\mathrm{TR},\mathrm{TI},\alpha,B_0)`
to FiLM modulation for a target :math:`f_\theta`.

**What is actually new** is the *certificate*, not the conditioning — HyperMorph
and neural-CDE acquisition-independent estimators already condition on
acquisition parameters. With both networks spectral-normalised,

.. math::

   \bigl\|f_{h_\psi(\varphi)}(\mathbf x) - f_{h_\psi(\varphi^\ast)}(\mathbf x)\bigr\|
   \;\le\; L_w L_h\,\lVert\varphi-\varphi^\ast\rVert ,

so the certified extrapolation radius at an unseen protocol is reportable at
inference at no training cost, via
:meth:`AcquisitionHypernetworkStrategy.certified_radius_for`.

**Config:** ``training.acq_hypernetwork`` (``TrainingConfigAcqHypernetwork``).
``spectral_norm: false`` is a permitted ablation but **voids the certificate** —
Tier-1 ``spectral_norm_enabled`` warns. The arm must also actually receive the
acquisition vector, via ``data.acquisition_metadata.enabled`` (per-sample) or
``data.multi_contrast.acquisition_params`` (fixed per contrast); Tier-1
``acq_vector_present`` **errors** otherwise, because a hypernetwork with no
conditioning vector trains a constant FiLM while still carrying the claim.


DispersionBlochAEStrategy (DL-BAE)
-----------------------------------

**File:** ``dispersion_bloch_ae_strategy.py`` — **Mode:**
``dispersion_bloch_ae`` — **Short name:** ``dispersion_bloch_ae`` — **Bases:**
``ReconstructionTrainingStrategy``

**Purpose:** Train the dispersion-latent Bloch autoencoder (M4), whose *decoder
is a physical law*.

**Objective:**

.. math::

   \mathcal L = \lambda_{\mathrm{DC}}\,\mathcal L_{\mathrm{DC}}
              + \lambda_{\mathrm{mono}}\,\mathcal L_{\mathrm{mono}} ,

the ``multifield_data_consistency`` term forcing one field-invariant latent to
explain **every** observed field, plus the ``dispersion_monotonicity`` hinge
keeping :math:`\partial T_1/\partial B_0 \ge 0`.

**Config:** ``training.dispersion_bloch_ae``
(``TrainingConfigDispersionBlochAE``). Identifiability is a hard constraint:
Tier-1 ``dispersion_identifiability`` **errors** when
``2*n_pools + 1 > len(fields_present)``, and the model constructor raises on the
same condition, because an under-determined fit converges to a meaningless
latent rather than failing visibly.


Pipeline Execution Flow
=======================

The training pipeline orchestrates all strategies through a shared execution
flow in ``src/spectramr/pipelines/train.py``.

**High-Level Stages:**

1. **Config health check** → abort on domain-alignment errors
2. **DI container bootstrap** → services registered and wired
3. **TrainingEnvironmentDirector** → builds frozen ``TrainingEnvironment``
4. **Parallelism** → DataParallel / DDP wrapping
5. **Strategy creation** → three-level dispatch via ``TrainingStrategyFactory``
6. **Iteration loop** → ``strategy.train_step()`` called per iteration
7. **CSV + TensorBoard logging** → every ``log_interval`` steps
8. **Checkpointing** → every ``checkpoint.save_interval`` steps
9. **Validation** → every ``validation.eval_interval`` steps

All strategies receive the same ``TrainingEnvironment`` and return
``LossResult`` dicts that the pipeline handles uniformly. This makes
it trivial to swap strategies without touching the orchestration logic.


---

Strategy Selection Quick Reference
====================================

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Training Mode
     - Strategy Class
     - Use Case
   * - ``reconstruction``
     - ``ReconstructionTrainingStrategy``
     - Supervised MRI reconstruction (L1, SSIM, perceptual)
   * - ``gan``
     - ``GANTrainingStrategy``
     - Adversarial training with dynamic G/D balancing
   * - ``diffusion``
     - ``DiffusionTrainingStrategy``
     - Denoising diffusion (DDPM, DDIM, cold diffusion)
   * - ``vae``
     - ``VAETrainingStrategy``
     - Variational Autoencoder with ELBO objective
   * - ``vqvae``
     - ``VQVAETrainingStrategy``
     - Vector-quantized discrete latent space
   * - ``pinn``
     - ``PhysicsDrivenTrainingStrategy``
     - PINN with PDE constraints + data consistency
   * - ``domain_adaptation``
     - ``DomainAdaptationTrainingStrategy``
     - Cross-domain transfer (GRL + domain discriminator)
   * - ``mae``
     - ``MAEPretrainingStrategy``
     - Masked Autoencoder self-supervised pretraining
   * - ``disentangled``
     - ``DisentangledTrainingStrategy``
     - Content-style disentanglement for multi-contrast
   * - ``b0_mapping``
     - ``B0MappingStrategy``
     - B0 field map estimation
   * - ``pinn``
     - ``ConcretePINNSensitivityStrategy``
     - PINN-based coil sensitivity maps (Helmholtz)
   * - ``n2n``
     - ``NoiseToNoiseStrategy``
     - Self-supervised N2N on multi-repetition data (M4Raw)
   * - ``meta_learning``
     - ``MetaLearningTrainingStrategy``
     - MAML outer-loop for anatomy-agnostic adaptation
   * - ``distillation``
     - ``ConcreteDistillationStrategy``
     - Teacher-student latent distillation (Virtual Fiducial)
   * - ``guided_sr``
     - ``GuidedSuperResolutionStrategy``
     - Reference-guided ULF→HF with content-style split
   * - ``test_time_optimization``
     - ``ConcreteTTOStrategy``
     - Zero-shot motion correction at inference time
   * - ``vf_admm``
     - ``ConcreteVFADMMStrategy``
     - ADMM + marker prior projection (Digital Twin)
   * - ``graph_cold_diffusion``
     - ``GraphColdDiffusionStrategy``
     - Cold diffusion for non-Cartesian GNN reconstruction
   * - ``geomamba_ulf``
     - ``GeoMambaULFStrategy``
     - ULF→HF super-resolution with GeoMamba-ULF
   * - ``padnet``
     - ``PaDNetTrainingStrategy``
     - Quantitative T1/T2 parameter map generation
   * - ``disentangled_diffusion``
     - ``DisentangledDiffusionStrategy``
     - Diffusion in factorized anatomy+modality latent space
   * - ``pma_varnet``
     - ``PMAVarNetStrategy``
     - Physics-motivated unrolled VarNet with SENSE DC
   * - ``motion_meta``
     - ``MotionMetaStrategy``
     - Meta-learning with motion trajectory memory buffer
   * - ``cycle_bloch``
     - ``CycleBlochStrategy``
     - Bloch-equation cycle consistency (quantitative MRI)
   * - ``virtual_fiducial``
     - ``VirtualFiducialStrategy``
     - Digital Twin marker-based calibration pipeline


References
==========

1. Gamma, E., et al. "Design Patterns: Template Method."
   Addison-Wesley, 1994.

2. Ho, J., et al. "Denoising Diffusion Probabilistic Models."
   NeurIPS, 2020.

3. Kingma, D.P. and Welling, M. "Auto-Encoding Variational Bayes."
   ICLR, 2014.

4. Goodfellow, I., et al. "Generative Adversarial Nets."
   NeurIPS, 2014.

5. Finn, C., et al. "Model-Agnostic Meta-Learning for Fast Adaptation
   of Deep Networks." ICML, 2017. (MetaLearningTrainingStrategy)

6. He, K., et al. "Masked Autoencoders Are Scalable Vision Learners."
   CVPR, 2022.

7. Mescheder, L., et al. "Which Training Methods for GANs Do Actually
   Converge?" ICML, 2018.

8. Lehtinen, J., et al. "Noise2Noise: Learning Image Restoration
   without Clean Data." ICML, 2018. (NoiseToNoiseStrategy)

9. Hinton, G., et al. "Distilling the Knowledge in a Neural Network."
   NIPS Deep Learning Workshop, 2015. (ConcreteDistillationStrategy)

10. Boyd, S., et al. "Distributed Optimization and Statistical Learning
    via the Alternating Direction Method of Multipliers."
    Foundations and Trends in Machine Learning, 2011. (ConcreteVFADMMStrategy)

11. Batchelor, P.G., et al. "Matrix Description of General Motion Correction
    Applied to Multishot Images." MRM, 2005. (ConcreteTTOStrategy)

12. Bansal, A., et al. "Cold Diffusion: Inverting Arbitrary Image Transforms
    Without Noise." NeurIPS, 2022. (GraphColdDiffusionStrategy)

13. Park, T., et al. "Contrastive Learning for Unpaired Image-to-Image
    Translation." ECCV, 2020. (GuidedSuperResolutionStrategy/PatchNCE)

14. Sriram, A., et al. "End-to-End Variational Networks for Accelerated
    MRI Reconstruction." MICCAI, 2020. (PMAVarNetStrategy)


Self-Supervised Reconstruction (SSDU)
=====================================

The ``self_supervised_reconstruction`` strategy (alias ``ssdu``) trains an
unrolled reconstructor from undersampled k-space alone -- no fully-sampled
reference. The acquired sample :math:`\mathbf{y}` (support :math:`\Omega`) is
split into a network-input partition :math:`\Lambda` and a held-out loss
partition :math:`\Theta`, with :math:`\Lambda \cup \Theta = \Omega` and
:math:`\Lambda \cap \Theta = \emptyset`. Training minimises the held-out
k-space residual

.. math::

   \mathcal{L}_{\text{SSDU}}
     = \big\|\, \mathbf{M}_\Theta \mathbf{F}\mathbf{S}\,
       f_{\boldsymbol{\theta}}(\mathbf{y}_\Lambda) - \mathbf{y}_\Theta \,\big\|_2^2 .

Configure via the ``training.ssdu`` sub-block (``theta_fraction``,
``num_masks``, ``split_seed``) and add an ``ssdu`` loss to the k-space loss
list. The strategy raises if no ``ssdu`` / ``multi_mask_ssdu`` loss is wired
(no silent fallback).

.. automodule:: spectramr.infrastructure.training.strategies.ssdu_strategy
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   SSDU needs the acquisition mask and the original measured k-space exposed in
   the batch (no fully-sampled target). Reference: Yaman *et al.*,
   "Self-supervised learning of physics-guided reconstruction neural networks
   without fully sampled reference data," *MRM* 84(6), 2020.

**Robust SSDU (Noisier2Noise).** The ``robust_ssdu`` key selects the same
strategy with ``ssdu.noisier2noise_correction = true``. For NOISY sub-sampled
training data (Millard & Chiew 2024), the network is fed a *noisier* measurement
:math:`\mathbf{z} = \mathbf{y} + \tilde{\mathbf{n}}`, with
:math:`\tilde{\mathbf{n}} \sim \mathcal{CN}(0, \sigma^2)` injected on the
acquired support (``ssdu_strategy.inject_noisier_kspace``), while the held-out
:math:`\Theta` loss target stays the original (less-noisy) :math:`\mathbf{y}` —
the Noisier2Noise principle (train noisier → less-noisy). The noise
:math:`\sigma` is wired via ``ssdu.noise_std_estimate`` (fail-closed: the schema
validator and the strategy raise if the correction is on without it), and the
dataset-driven ``ssdu.noise_model`` (``gaussian_kspace`` / ``ncchi_magnitude``)
is shared with the T1 keystone. **Corner case:** at :math:`\sigma \to 0` the
injected noise vanishes and Robust SSDU reduces exactly to vanilla SSDU, so the
repair provably generalizes the base method. The ``robust_ssdu_key_matches_correction``
audit refuses ``training_mode: robust_ssdu`` with the correction off, and
``ssdu_selection_density_range`` warns when ``theta_fraction`` leaves
``[0.2, 0.5]``.


Equivariant Imaging (EI / Robust EI)
====================================

The ``equivariant_imaging`` strategy (and its ``robust_ei`` sibling) trains a
reconstructor from undersampled k-space alone by exploiting a *symmetry* of the
signal set that the forward operator :math:`A = M F` does **not** share. For a
group element :math:`T_g` drawn from a brain-appropriate group (dihedral
:math:`D_4` or a small continuous rotation, **not** full ``SO(2)``), the network
:math:`f` must satisfy

.. math::

   f\big(A\, T_g\, f(A^{*}\mathbf{y})\big) \;=\; T_g\, f(A^{*}\mathbf{y}),

scored by the registered ``EquivariantSSLReconLoss``. The strategy emits both
branches via ``context["transformed_recon"]`` (the reference branch
:math:`T_g f(A^{*}\mathbf{y})`) — without this the loss is an **inert facade**.
A measurement-consistency
anchor :math:`\lVert A f(A^{*}\mathbf{y}) - \mathbf{y}\rVert^2` (or the nc-χ
GSURE term of the T1 keystone when ``robust_correction`` is set) prevents the
trivial :math:`f \equiv \text{const}` solution.

Identifiability (Tachella *et al.* sensing theorems) requires :math:`A` to break
the group symmetry; ``infrastructure/physics/group_actions.sensing_margin``
quantifies this and the ``ei_sensing_margin`` audit refuses an arm whose
operator is (near-)equivariant to the group (e.g. a fully-sampled arm). The
``ei_robust_key_matches_correction`` audit refuses ``training_mode: robust_ei``
with ``robust_correction: false`` (a key that silently does nothing). The
Tier-2 ``ei_forward_probe`` asserts, on the configured model + mask, that
``sensing_margin > 0`` and the equivariance term is non-zero with a gradient
reaching the network — the mechanism-fires guard.

Configure via the ``training.equivariant_imaging`` sub-block (``group``,
``alpha_equivariance``, ``n_group_samples``, ``robust_correction``,
``noise_std_estimate``, ``noise_model``, ``n_coils``). Corner case:
``alpha_equivariance: 0`` reduces EI to a measurement-consistency
reconstruction.

.. automodule:: spectramr.infrastructure.training.strategies.equivariant_imaging_strategy
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   EI assumes a single (coil-combined) complex forward operator :math:`A = M F`;
   multi-coil EI (:math:`A = M F S` via ``sense_forward``) is a future
   extension. References: Chen *et al.*, "Equivariant imaging: Learning beyond
   the range space," *ICCV* 2021; Chen *et al.*, "Robust equivariant imaging,"
   *CVPR* 2022; Tachella *et al.*, sensing theorems, *JMLR* 2023.


Ambient Diffusion (A-DPS)
=========================

The ``ambient_diffusion`` strategy learns a *clean image prior from undersampled
k-space alone* by lifting the SSDU Λ/Θ split onto a score-based diffusion model
(Daras *et al.*, NeurIPS 2023; Aali *et al.*, MRI Ambient Diffusion Posterior
Sampling). Per step it noises the Λ-undersampled measurement, predicts
:math:`\hat{x}_0` from the eps estimate, and supervises it on the **held-out** Θ
k-space:

.. math::

   \mathcal{L} = \lambda_{\text{d}}\,\big\| \boldsymbol{\epsilon}_\theta(x_t, t)
   - \boldsymbol{\epsilon} \big\|^2
   + \lambda_{\text{a}}\,\big\| M_\Theta (F\,\hat{x}_0 - \mathbf{y}) \big\|^2 ,
   \quad x_t = q(A^{*}_\Lambda \mathbf{y}, t, \boldsymbol{\epsilon}).

The held-out term is the registered ``ambient_consistency`` loss. **Corner case:**
at Θ-fraction / noise → 0 the objective reduces to the SSDU held-out residual, so
Ambient generalizes SSDU onto a diffusion prior. Configure via the
``training.ambient`` sub-block (``theta_fraction``, ``ambient_weight``,
``denoise_weight``). The ``ambient_requires_score_model`` audit refuses a
non-score model or image-only input.

**A-DPS inference reuses DDS.** Because the strategy subclasses
``DiffusionTrainingStrategy`` and the score generator exposes ``sample()``, setting
``inference_sampler: dds`` runs the conditioned posterior reverse loop
(``DDSReconSampler``, CG data-consistency via the T1 ``cg_data_consistency``
primitive) — i.e. A-DPS is DDS applied to the ambient-trained prior, not a
redundant sampler. The score prior's forward is intentionally unconditional, so
``ScoreBasedDiffusionGenerator`` opts out of the Tier-2 ``input_invariant`` probe
(measurement-dependence lives in the sampler + the held-out consistency).

.. automodule:: spectramr.infrastructure.training.strategies.ambient_diffusion_strategy
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   Ambient needs the dataloader to expose the complex measured k-space
   (``batch["measured_kspace"]``) + acquisition mask (the SSDU contract); it
   raises if absent. **The R=8 superiority reported on fastMRI is untested at
   0.3T — not advertised.**


Strategy Authoring Contract
===========================

These four invariants are enforced by the base orchestrator and the DI
environment. Violating any of them produces a *crash on the first training
step* or a *silent no-op* — both of which escape import-time smoke checks. New
strategies (and edits to existing ones) must follow them.

1. ``_compute_losses_impl`` signature is keyword-only
-----------------------------------------------------

``BaseTrainingStrategy._compute_losses`` invokes the hook with **named**
arguments only::

    losses = self._compute_losses_impl(
        input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
    )

Therefore every override **must** use the canonical signature — a legacy
``(self, batch, *args, **kwargs)`` form leaves ``batch`` unbound and raises
``TypeError`` on every step::

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)  # dict, if any
        ...

When delegating to the parent, forward the **named** arguments (never a
positional ``batch`` dict — the parent treats ``input_batch`` as a tensor).
To hand the parent a *modified* batch, inject it through ``kwargs["batch"]``::

    base = super()._compute_losses_impl(
        input_batch=input_batch, target_batch=target_batch, epoch=epoch,
        **{**kwargs, "batch": patched},
    )

Do **not** write ``super()._compute_losses_impl(batch, *args, **kwargs)`` —
``*args`` is undefined under the canonical signature (``NameError``).

2. The generator lives on ``self.env``, not ``self.context``
------------------------------------------------------------

``self.context`` is a :class:`StrategyContext` dataclass that carries device /
config / utilities but **has no** ``generator`` field, so
``getattr(self.context, "generator", None)`` is *always* ``None``. Read the
model from the environment (or the ``generator_model`` property)::

    gen = getattr(self.env, "generator", None)

3. Register strategy-owned learnable modules with ``opt_g``
-----------------------------------------------------------

If a strategy builds its own learnable module (an auxiliary head, a learned
loss weight, an estimator), its parameters must be added to the generator
optimizer or they never train. The environment exposes the optimizer as
``opt_g`` (``env.optimizer_g`` / ``env.optimizer`` do **not** exist)::

    opt = getattr(self.env, "opt_g", None)
    if opt is not None:
        existing = {p for g in opt.param_groups for p in g["params"]}
        new_params = [p for p in module.parameters() if p not in existing]
        if new_params:
            opt.add_param_group({"params": new_params})

Also move the module to ``self.device`` before its first forward.

4. Never silently coalesce tensors or swallow failures
------------------------------------------------------

Use :func:`spectramr.infrastructure.training.strategies.mixins.utils.pick_present`
instead of ``a or b`` whenever an operand may be a tensor — ``bool()`` on a
multi-element tensor raises *"Boolean value of Tensor … is ambiguous"*
(``a or b`` semantics are fine only for dict/int operands such as
``kwargs.get("batch") or {}``). And do not wrap correctness-critical work
(normalization, domain/shape guards) in a bare ``except`` that returns a
default — surface the failure.

**Two shapes this takes.**

- Wrapping a strategy's *distinctive* terms — a cross-contrast LNCC, a SIREN PDE
  Laplacian — in an ``except Exception:`` that zeroes or drops them. Any
  shape / NaN / AMP error then collapses the arm to a plain reconstruction while
  smoke still PASSes. A genuine failure must raise. Reading the loop iteration
  from ``kwargs.get("step", 0)`` has the same effect on step-gated schedules: the
  loop never sets that key, so the schedule freezes at 0. Use
  ``resolve_loop_iteration(self)``.
- Catching a teacher-checkpoint load failure and setting ``self._teacher = None``,
  which trains the student as a plain reconstruction even though
  ``checkpoint.resume_from`` was configured. A configured-but-unloadable teacher
  must raise; the genuinely-optional no-teacher path returns early, before the
  load is attempted.
