.. _losses_reference:

=======================================
Loss Functions — Mathematical Reference
=======================================

.. sectionauthor:: spectraMR Research

Losses are registered with the ``@register_loss`` decorator and resolved through
the ``LossRegistry`` singleton. All implement
``forward(prediction, target, **kwargs) → Tensor``.

.. note::

   **This page is hand-written and not exhaustive.** It documents 106 of the
   217 losses the registry holds, so a name's absence here is **not** evidence
   that it does not exist -- check the registry before concluding one is
   unavailable. Counts are deliberately not restated in the prose above: a frozen
   number in a hand-maintained page drifts silently, and this one had -- it read 147.

   .. code-block:: python

      from spectramr.models.losses.registry import LossRegistry

      LossRegistry.list_available()          # the canonical names
      len(LossRegistry.list_available())

.. contents:: Table of Contents
   :depth: 2
   :local:

Strict-duplicate registration
=============================

Because silent fallbacks are forbidden, ``LossRegistry`` refuses to silently
overwrite a registration:

* Re-registering the **same** class under the **same** name is idempotent
  (test reloads / repeated imports stay green).
* Registering a **different** class under the **same** canonical name
  raises :py:class:`ValueError` immediately at import time.
* Re-binding an alias to a **different** canonical name also raises.

Practical consequence: a colliding ``@register_loss`` decoration breaks
the import that triggers it, so the audit ladder
(``python -m spectramr.cli audit``) catches the bug before training starts —
rather than letting the registry silently flip between two classes
depending on import order.

The regression tests are under
``tests/unit/models/test_registry_strict_duplicates.py``.

Usage
=====

.. code-block:: python

   from spectramr.models.losses.registry import create_loss

   l1 = create_loss("l1")
   value = l1(prediction, target)

.. code-block:: yaml

   # In experiment YAML
   losses:
     image_losses:
       - name: l1
         weight: 10.0
         enabled: true
       - name: ssim
         weight: 1.0
         enabled: true

Declarative-list loss contract
==============================

The declarative loss form (``losses.image_losses``,
``losses.kspace_losses``, ``losses.complex_losses``) is the canonical
way to wire losses into a reconstruction experiment. The LossBuilder
turns each list entry into a callable and pushes it into
``env.losses``, which the strategy passes to
:py:meth:`spectramr.models.losses.computers.unified_diffusion_reconstruction.UnifiedReconstructionLossComputer.compute`
as ``losses_dict``.

Two collaborating paths inside the computer add components to the
total:

1. **Explicit blocks** for ``l1`` / ``l2`` / ``perceptual`` / ``pinn``
   gated on ``losses.reconstruction.lambda_*`` (schema defaults:
   ``lambda_l1=10.0``, ``lambda_l2=0.0``).
2. **Dynamic loop** that iterates ``losses_dict`` and adds anything not
   already in ``components``.

The two paths must not double-count, and must not cancel each other
out. The dynamic-loop skip is therefore conditional: ``mse`` / ``l1``
/ ``l2`` are only dropped from the loop when the **canonical key**
(``l2`` for ``mse``, otherwise the loss name itself) is already in
``components``. ``gradient_penalty`` / ``r1`` / ``r1_regularization``
remain unconditionally skipped because they need discriminator
context owned by the GAN strategy.

Pinned by
:py:func:`tests.unit.test_loss_computers.TestUnifiedReconstructionLossComputer.test_declarative_mse_only_yaml_still_computes_loss_pre_warmup`.

One weight, one declaration surface
------------------------------------

A loss weight can be written in two places: ``losses.<section>.lambda_<name>``
and an entry in one of the domain lists. When both are present,
:py:func:`~spectramr.models.losses.weights.build_loss_weight_table` resolves them
to a single number, and raises when the two values differ, so an arm declaring
both trains correctly today. The redundancy still costs something, because only the list entry constructs
the loss module and selects the FFT bridge. The matching lambda adds no term to
the objective, and editing it changes nothing.

**The domain list is the single source of truth.** Delete the redundant
``lambda_``, not the list entry.

Two detectors cover the two directions, at different times:

``check_loss_weight_declaration_ssot``
   Reports a name declared on both surfaces. Runs under ``spectramr audit``, at
   ``severity="warning"``, category ``duplication``. It does not filter on
   ``enabled``: a disabled list entry with a live lambda is the one an author
   re-enables later, at the weight the list holds. Two entries in *different*
   lists are not reported: the list a term appears in selects its domain bridge,
   so two lists declare a two-domain objective.

``LossBuilder``'s ``unmigrated`` guard
   Raises when a weight resolves on the lambda surface alone, which would
   otherwise train without the supervision the YAML advertises (non-negotiable 9).
   It runs at build time, and only for arms that already populate a domain list —
   ``_build_all_dynamic`` returns earlier when
   ``uses_list_based_losses`` is false. An arm that has not begun migrating is
   therefore outside its scope; the duplicate shape above is what ``audit``
   covers corpus-wide.

Planted violations for both live in
``tests/unit/infrastructure/validation/test_config_health_checker_loss_ssot_2026_09.py``
and ``tests/unit/infrastructure/training/builders/test_loss_builder_unmigrated_guard_2026_09.py``.

Two exemptions from the pairing rule
-------------------------------------

The weight table aliases ``lambda_mse`` onto the key ``l2``, so
``losses.diffusion.lambda_mse`` shares a table entry with an ``mse`` entry in
``image_losses``. They weight different terms. The lambda is step 3 of
``_resolve_diffusion_weight``
(``models/losses/computers/unified_diffusion_reconstruction.py``), which weights
the diffusion term; the list entry builds an MSE module on the image branch.
Deleting the lambda would drop the diffusion weight to the fallback and leave
the image term where it was.

``COMPUTER_RESOLVED_LAMBDA_SOURCES``
(``infrastructure/training/builders/loss_builder.py``) names the lambdas a
computer reads directly under a meaning of their own. The audit does not report
them and the migration does not delete them.

The second exemption has an unrelated cause. ``losses.reconstruction.lambda_l2``
defaults to 0.0 and ``losses.diffusion.lambda_mse`` defaults to 1.0, and both
alias the table key ``l2``. A config load stamps every schema default as a value,
so an arm that trains an MSE term at 1.0 has to write the first field for the two
to agree. On the written surface that field reads as a duplicate of the ``mse``
list entry, and deleting it leaves the two declarations 0.0 apart from 1.0.

:py:func:`~spectramr.models.losses.weights.deleting_lambda_would_conflict` answers
this by re-reading the config with the field at its schema default and comparing
the result. Two ``kspace_filling`` arms carry the shape and keep the field. The
divergent per-section defaults are issue #421; closing it retires the exemption.

The pairing rule has one owner (non-negotiable 17):
:py:func:`~spectramr.infrastructure.validation.config_health_checker.dual_surface_loss_declarations`
returns ``(name, weight, lambda_sites, list_sites)`` for each weight written on
both surfaces, with the exemption applied. ``check_loss_weight_declaration_ssot``
renders it as a finding, and the corpus migration script gates every deletion on
it, so the script cannot remove a field the audit would not have reported.

Migrating a cohort
------------------

The migration is driven by ``migrate_loss_lambdas_to_domain_lists.py``, kept in the
research tree's ``scripts/migrations`` directory. The distribution does not carry
it: it rewrites the ``experiments/`` corpus, which is not published either. Pointed
at a cohort directory it dry-runs by default and writes only under ``--apply``:

.. code-block:: bash

   python migrate_loss_lambdas_to_domain_lists.py <cohort-dir>          # dry run, the default
   python migrate_loss_lambdas_to_domain_lists.py <cohort-dir> --apply

A field is deleted only if three independent surfaces agree. A **domain list**
names it — the script scans the YAML text, so a ``lambda_`` outside a category
block under top-level ``losses:`` is never a candidate at all. The **deny-list**
lets it go. And ``dual_surface_loss_declarations`` **reports that exact
block/name pair**. A candidate the audit does not report is kept and named in the
verdict as ``UNREPORTED``, never folded into "already migrated": a divergence
between the text scan and the audit is a finding, not a no-op. It is the one
non-fatal verdict that makes the **run exit 1**, so a cohort sweep cannot report
success over it — including on an arm that migrated *and* diverged, where the
verdict still carries the ``MIGRATED`` head. ``DENIED`` exits 0: it is a recorded
exemption that a fully migrated cohort reports on every run.

The edit is then written, the arm reloaded, and rolled back unless two censuses
come back identical. The first is the **resolved weight table** — weight, enabled
*and* warmup gating per term, so a migration that changed a term's gating without
changing its number cannot pass as inert. The second is a **raw-reader census**,
which exists because the weight table cannot see every consumer: some readers
resolve a ``lambda_`` off the config field directly and bypass the SSOT entirely.
The script also refuses outright rather than emptying a category block.

The deny-list is for the removals **neither census can catch**, and both its
entries are latent rather than live — which is exactly why a before/after
comparison is not enough on its own:

* ``lambda_perceptual`` is read raw to construct ``gan_composite``, inside a
  branch taken only when the arm declares a critic. No arm in the residue does, so
  deleting an explicit ``0.1`` changes nothing measurable today and becomes a 100x
  jump against the ``10.0`` schema default the moment one is added.
* ``losses.diffusion.lambda_mse`` is resolved by its literal dotted path, while the
  SSOT canonicalizes ``lambda_mse`` onto ``l2`` — a different term. The audit is
  silent here **on purpose**: ``REINTERPRETED_LAMBDA_SOURCES`` exempts it precisely
  because a lambda a computer reads under a different meaning is a second knob, not
  a duplicate. So the audit gate would already keep the field, and the deny-list
  entry does two other jobs — it makes the verdict say ``DENIED`` (a recorded,
  reasoned exemption) rather than ``UNREPORTED`` (a divergence to go look at), and
  it is what still stands between the field and a deletion if that exemption is
  ever narrowed.

Scope: ``experiments/inprogress/`` only. Done: ``kspace_filling``, 2026-09-07.
What survives there is pinned per arm by
``tests/audit/test_kspace_filling_loss_ssot_2026_09.py`` rather than counted here —
that test fails if a dual-surface declaration returns to the cohort, and its
per-arm exemption list shrinks as each pin is retired.

Per-entry constructor arguments: ``kwargs:``, never ``config:``
---------------------------------------------------------------

A list entry passes constructor arguments through ``kwargs:``:

.. code-block:: yaml

   losses:
     kspace_losses:
       - name: rician_consistency
         weight: 0.1
         kwargs:
           sigma: 0.02

:class:`~spectramr.config.schemas.loss.LossComponentConfig` declares exactly four
fields — ``name``, ``weight``, ``enabled``, ``kwargs`` — and is
``extra="ignore"``. A ``config:`` mapping in that position therefore **loads
without error and is discarded at parse time**, so the loss is constructed with
its own defaults while the YAML still advertises the setting. No gate catches
this: it is not a ``RENAMES`` record, so
``scripts/ci/check_no_legacy_config_keys.py`` cannot see it, and
``scripts/ci/report_discarded_config_keys.py`` stops at mappings and does not
descend into list entries.

**Do not migrate ``config:`` to ``kwargs:`` blindly.** ``LossBuilder`` merges
the entry's kwargs over the schema-derived ones and splats the result into the
loss constructor, so a key the constructor does not accept becomes a
``TypeError`` at build time rather than a no-op. The 56 committed occurrences
were all ``sobolev_kspace`` carrying ``sobolev_order``, and
:class:`~spectramr.models.losses.kspace_physics_losses.SobolevKSpaceLoss` is
parameterless by design — its ``1 + k_x^2 + k_y^2`` weighting is fixed, and
``loss_builder.py`` maps it to ``{}`` explicitly. The declaration was never
true, so the correct migration was deletion.

Pinned by
:py:class:`tests.unit.config.test_dead_legacy_key_spellings.TestLossEntryConfigSpelling`,
which checks the field names, that a rename would raise against the real
constructor signature, and that no committed arm under ``experiments/``
declares ``config:`` on a loss entry.

``pre_model`` adapter wiring for m4raw cross-contrast
=====================================================

The adapter chains
(:py:mod:`spectramr.infrastructure.builders.leaf.adapter_builders`) have five
hooks: ``pre_model``, ``post_model``, ``pre_loss_pred``,
``pre_loss_target``, ``pre_metric``. The audit
(:py:mod:`spectramr.infrastructure.validation`) validates every declared chain,
and the training strategy applies it at forward time.

The m4raw dataset's cross-contrast pipeline (lines 818–852 of
``src/spectramr/data/datasets/m4raw_dataset.py``) emits a 4-channel tensor even
when the YAML declares ``coil_processing_mode: rss_image``: per-side
RSS reduces to 1 ch, then source||target ``cat`` doubles it, then
real/imag interleave doubles it again. The audit's
``_derive_expected_channels`` does not model that doubling, so it
reports 1 ch and the YAML passes — but
:py:meth:`spectramr.infrastructure.training.strategies.base.BaseTrainingStrategy.train_step`'s
strict DomainMismatch check at lines 633–663 (and a downstream
``Conv2d`` channel-mismatch crash) catches the discrepancy at runtime
with the same opaque error every time.

Declare the adapter to close the gap:

.. code-block:: yaml

   adapters:
     pre_model:
       - name: rss_coils_to_magnitude

``rss_coils_to_magnitude`` registers ``pre_model`` in its ``insertion_points``
tuple (:py:mod:`spectramr.data.adapters.channels`), and
:py:meth:`BaseTrainingStrategy.train_step` applies the chain to **both**
``input_batch_prepared`` and ``target_batch`` immediately after the
complex→real guard and before the DomainMismatch check. The adapter is
idempotent (identity on 1-ch input), so re-application in the loss path is
safe.

Pinned by
:py:mod:`tests.unit.data.test_adapters_pre_model_rss` (4 tests:
insertion-point registration, builder acceptance, 4-ch → 1-ch
collapse, 1-ch idempotency).

``pre_loss_pred`` hook
----------------------

``rss_coils_to_magnitude`` is also allowed at ``pre_loss_pred`` (model output
side, before per-loss comparison). The RSS reduction is a side-effect-free
squashing of an arbitrary-channel real tensor to a 1-channel magnitude image, so
it is equally valid there. An adapter declared at a hook it does not advertise
fails loud: ``Adapter '<name>' is not allowed at hook '<hook>'``. Pinned by three
``test_adapters_pre_model_rss`` cases (``test_rss_coils_to_magnitude_advertises_pre_loss_pred``,
``test_adapter_chain_builder_accepts_pre_loss_pred_chain``,
``test_rss_coils_to_magnitude_lists_all_four_hooks``).

Dict-output unwrap in unified loss computers
============================================

Several generators (latent / cold diffusion, multi-head VAEs) return a
``dict`` like ``{"pred": tensor, "aux": ...}`` from ``forward``. The loss
computers unwrap it rather than calling ``.shape`` / ``.device`` on the dict:
``_unwrap_tensor_arg(name, value)`` probes a
fixed list of common prediction keys (``pred`` / ``prediction`` /
``image`` / ``output`` / ``x`` / ``x0`` / ``sample`` /
``reconstruction`` / ``denoised``, etc.) and falls back to the first
tensor-valued entry. Dict-of-non-tensors raises a typed ``TypeError``
rather than failing silently. Tuples and lists unwrap to their first
element.

Pinned by :py:mod:`tests.unit.losses.test_unified_diffusion_unwrap` (24
tests: passthrough, every probed key, fallback to first tensor, dict
of non-tensors raises, tuple/list unwrap, ``_complex_safe_mse`` and
``_complex_safe_l1`` end-to-end with dict inputs).

K-space RSS phase-strip pitfall
===============================

The legacy :py:meth:`spectramr.data.transforms.kspace_coil_transforms.CoilCombineTransform._rss_combine`
chains IFFT → RSS-magnitude → FFT-back-to-k-space. The third step
``fft2c(|x|)`` is phase-stripping: the returned k-space is
Hermitian-symmetric, and any downstream IFFT-then-magnitude produces a
*centro-symmetric* image — a "doubled brain" reconstruction.

Use ``method="rss_image"`` instead: it stops at step 2 and returns the real
magnitude image directly. The TorchIO transform
builder (:py:mod:`spectramr.data.builders.torchio_transform_builder`) routes
``coil_processing_mode: rss_image`` to this new branch at both
training and eval sites.

Pinned by :py:mod:`tests.unit.data.transforms.test_kspace_coil_transforms`
(4 tests: real image-domain output, Hermitian-magnitude regression on
legacy ``rss`` mode AND non-Hermitian on new ``rss_image``, end-to-end
through ``tio.Subject``, unknown-method raise).

---

Noise-model-aware k-space fidelity (SOTA plan T1)
=================================================

``gsure_kspace`` (alias ``ncchi_gsure``, ``domain="kspace"``) is the shared
data-fidelity term consumed by the self-supervised reconstruction arms
(Robust-SSDU, Equivariant Imaging, Ambient/A-DPS). Because the framework spans
many datasets — complex multi-coil raw k-space (``calgary_campinas``,
``fastmri``, ``ocmr``) as well as magnitude-only low-field — the noise model is a
**config knob**, never hardcoded:

* ``noise_model="gaussian_kspace"`` (default): complex k-space Gaussian NLL
  :math:`\tfrac{1}{2\sigma^2}\lVert\hat{y}-y\rVert_2^2`, correct where the per-coil
  k-space noise is approximately complex-Gaussian (the regime where SURE /
  Noisier2Noise / Tweedie are unbiased).
* ``noise_model="ncchi_magnitude"`` (``n_coils=L``): moment-corrected
  non-central-:math:`\chi` magnitude fidelity using the exact second-moment bias
  :math:`\mathbb{E}[m^2]=\nu^2+2L\sigma^2` to debias the measured magnitude,
  :math:`\hat{\nu}=\sqrt{\max(m^2-2L\sigma^2,0)}`. This is a standard low-SNR
  correction, not the full arbitrary-order Bessel nc-:math:`\chi` NLL.

**Corner-case:** as :math:`\sigma\to 0` the noise floor vanishes,
:math:`\hat{\nu}\to m`, and the nc-:math:`\chi` branch reduces to the Gaussian
magnitude fidelity — so the repair provably generalises the published objective.
The k-space data-consistency solve it pairs with lives in the physics layer as
``spectramr.infrastructure.physics.tangent_cg.cg_data_consistency``.

Ambient held-out consistency (SOTA plan Phase B)
================================================

``ambient_consistency`` (alias ``ambient_dps_consistency``, ``domain="kspace"``)
is the held-out Θ k-space residual for Ambient Diffusion,
:math:`w\,\lVert M_\Theta (F\,\hat{x}_0 - y)\rVert_2^2`, evaluated on the
diffusion model's clean-image estimate :math:`\hat{x}_0`. It reads
``context["target_kspace"]`` and ``context["theta_mask"]`` (the SSDU split lifted
onto the diffusion prior). The scalar ``ambient_weight`` is the ambient
reweighting hook; at ``weight = 1`` it is the plain held-out residual, and
combined with the diffusion denoising term the objective reduces to SSDU as the
noise level → 0. Consumed by ``AmbientDiffusionStrategy``.

nc-χ Bessel-ratio data consistency (SOTA plan T4)
=================================================

``ncchi_consistency`` (alias ``ncchi_dc``, ``domain="image"``) pulls the
reconstruction toward the **non-central-χ ML magnitude** of the noisy
measurement — the Bessel-ratio fixed point :math:`\nu = m\,R_L(m\nu/\sigma^2)`,
:math:`R_L=I_L/I_{L-1}` — computed by
``spectramr.infrastructure.physics.ncchi_data_consistency`` (the ratio via a stable
backward continued fraction, any coil count ``L``). It is the **exact** ``L``-coil
generalisation of the moment-based :class:`rician_consistency` unbiasing
(:math:`\sqrt{\max(m^2-2\sigma^2,0)}`, the high-SNR approximation at ``L=1``) and
the image-domain analogue of the T1 GSURE k-space term. ``sigma`` and ``n_coils``
are dataset-driven config knobs (fail-closed; never hardcoded to 0.3T).

**Corner cases:** ``L=1`` ⇒ Rician (``I_1/I_0``); high SNR (``z→∞``) ⇒
``R_L→1`` ⇒ target = the measurement (Gaussian data-consistency); below the noise
floor ``√(2Lσ²)`` ⇒ target → 0 (signal suppressed). Consumed by reconstruction
arms via ``losses.image_losses`` (``kwargs: {sigma, n_coils}``); wiring the same
Bessel-ratio DC into the EDM strategy's bespoke Karras loss is a noted follow-up.

PISCO calibrationless PI self-consistency (SOTA plan P3.3)
==========================================================

``pisco_self_consistency`` (alias ``pisco``, ``domain="kspace"``) regularises
multi-coil reconstruction **without a calibration region** (Spieker *et al.*,
2024). A genuine multi-coil image (image × smooth sensitivities) satisfies a
shift-invariant *linear self-consistency* in k-space (the SPIRiT relation: each
point is predictable from its cross-coil neighbourhood). PISCO fits that kernel
on **disjoint subsets** of the acquired k-space and penalises their disagreement
:math:`\lVert G_A - G_B\rVert_F^2` (ridge-regularised per-subset least squares) —
≈0 for parallel-imaging-consistent data, large for noise/inconsistency, with no
ACS region. Requires ``≥ 2`` coils (the cross-coil relation is the signal).
Consumed by the IMJENSE-style scan-specific INR arm (registered ``siren``
reconstruction → multi-coil k-space) via ``losses.kspace_losses``; supersedes the
de-faced ``kspace_inr`` stub.

Image-Space Losses
==================

The registered losses group into six categories.

**Pixel and structural** — L1, L2, Smooth-L1 (Huber), Charbonnier, Log-Cosh, SSIM, MS-SSIM.


**Complex and k-space** — Complex L1/MSE, Spectral, Sobolev, Focal Frequency, Log-Spectral, Data Consistency.


**Perceptual and edge** — VGG (conv1–conv4), Sobel, HFEN, Gradient Magnitude, Structure.


**Regularisation and VQ** — Total Variation, KL Divergence, VQ Commitment, Deep Supervision at 4 scales.


**Diffusion, uncertainty and SNR-preserving** — Diffusion MSE at 4 timesteps, Heteroscedastic, R1 Penalty, SNR-Preserving.


**Physics and domain** — Data Consistency at R=2/4/8/16, Energy Conservation, Rician, Cycle Consistency, Histogram.


**GAN Loss Functions** — Vanilla, LSGAN, Hinge, WGAN discriminator objectives:


L1 Loss (Mean Absolute Error)
-----------------------------

**Registry name:** ``l1`` — **Class:** ``L1Loss``

Measures the mean absolute difference between prediction and target:

.. math::

   \mathcal{L}_{L1} = \frac{1}{N} \sum_{i=1}^{N} |x_i - \hat{x}_i|

Promotes sharp edges but can produce blurry results when used alone. Often
combined with perceptual losses for reconstruction.


L2 / MSE Loss
-------------

**Registry name:** ``l2`` — **Class:** ``MSELoss``

Mean squared error penalizes large errors quadratically:

.. math::

   \mathcal{L}_{L2} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2

Sensitive to outliers; tends to produce over-smoothed outputs.


Smooth L1 Loss
--------------

**Registry name:** ``smooth_l1`` — **Class:** ``SmoothL1Loss``

Acts as L2 for small errors and L1 for large errors, combining the
benefits of both:

.. math::

   \mathcal{L}_{\text{smooth}} = \begin{cases}
   \frac{1}{2}(x - \hat{x})^2 / \beta & \text{if } |x - \hat{x}| < \beta \\
   |x - \hat{x}| - \frac{\beta}{2} & \text{otherwise}
   \end{cases}


Huber Loss
----------

**Registry name:** ``huber`` — **Class:** ``HuberLoss``

Identical formulation to Smooth L1 with configurable :math:`\delta`:

.. math::

   \mathcal{L}_H = \begin{cases}
   \frac{1}{2}(x - \hat{x})^2 & \text{if } |x - \hat{x}| \leq \delta \\
   \delta(|x - \hat{x}| - \frac{\delta}{2}) & \text{otherwise}
   \end{cases}


SSIM Loss
---------

**Registry name:** ``ssim`` — **Class:** ``SSIMLoss``

The Structural Similarity Index measures luminance, contrast, and structure:

.. math::

   \text{SSIM}(x, \hat{x}) = \frac{(2\mu_x \mu_{\hat{x}} + C_1)(2\sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}

where :math:`C_1 = (k_1 L)^2`, :math:`C_2 = (k_2 L)^2` are stabilization
constants, :math:`L` is the dynamic range, and statistics are computed over
local windows.

The loss is :math:`\mathcal{L} = 1 - \text{SSIM}`.


Multi-Scale SSIM Loss
---------------------

**Registry name:** ``ms_ssim`` — **Class:** ``MSSSIMLoss``

Computes SSIM at multiple resolutions and combines them:

.. math::

   \text{MS-SSIM}(x, \hat{x}) = l_M^{\alpha_M} \cdot \prod_{j=1}^{M} cs_j^{\beta_j}

where :math:`l_M` is luminance at the coarsest scale and :math:`cs_j` is the
contrast-structure component at scale :math:`j`. Exponents :math:`\alpha_j, \beta_j`
are calibrated to human visual perception.


LPIPS Loss
----------

**Registry name:** ``lpips`` — **Class:** ``LPIPSLoss``

Learned Perceptual Image Patch Similarity extracts deep features from a
pre-trained VGG network:

.. math::

   \mathcal{L}_{\text{LPIPS}} = \sum_{l} \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\phi_l(x) - \phi_l(\hat{x})) \|_2^2

where :math:`\phi_l` extracts features from VGG layer :math:`l` and :math:`w_l`
are learned linear weights.


Perceptual Loss (VGG)
---------------------

**Registry name:** ``perceptual`` — **Class:** ``PerceptualLoss``

Feature matching in VGG feature space (unweighted):

.. math::

   \mathcal{L}_{\text{perc}} = \sum_{l \in \mathcal{S}} \| \phi_l(x) - \phi_l(\hat{x}) \|_1

where :math:`\mathcal{S}` is a set of VGG layers (typically ``relu1_2``,
``relu2_2``, ``relu3_4``, ``relu4_4``).


DINOv2 Perceptual Loss
-----------------------

**Registry name:** ``dino_perceptual`` — **Class:** ``DINOv2PerceptualLoss``

Uses DINOv2 vision transformer features instead of VGG. Self-supervised
features capture semantic structure better than supervised ImageNet features.


HFEN Loss
---------

**Registry name:** ``hfen`` — **Class:** ``HFENLoss``

High-Frequency Error Norm applies a Laplacian of Gaussian (LoG) filter
before computing the error, emphasizing edge preservation:

.. math::

   \mathcal{L}_{\text{HFEN}} = \| \text{LoG}(x) - \text{LoG}(\hat{x}) \|_2^2

where :math:`\text{LoG}(f) = \nabla^2 (G_\sigma * f)` is the Laplacian of Gaussian.

.. note::

   **Volumetric (5-D) input.** The LoG kernel is 2-D, so ``hfen`` and
   ``soft_dtw`` are applied **slice-wise** to a ``[B, C, H, W, D]`` volume
   (TorchIO depth-last): the trailing depth axis is folded into the batch
   dimension and every in-plane ``(H, W)`` slice is scored independently,
   then averaged. Before 2026-06 both losses hard-raised on a 5-D tensor
   (``ValueError: too many values to unpack`` / ``expects 3D or 4D``),
   which crashed the volumetric Hilbert-Mamba arms ``exp_hm_07_25d_mamba``
   and ``exp_hm_08_sdtw_mamba`` (both emit 5-D ``[B,C,H,W,D]`` predictions)
   at the first train step. A single-slice volume (``D=1``) reduces exactly
   to the 4-D path.


Sobel Edge Loss
---------------

**Registry name:** ``sobel_edge`` — **Class:** ``SobelLoss``

Computes horizontal and vertical Sobel gradients and penalizes edge differences:

.. math::

   \mathcal{L}_{\text{Sobel}} = \| G_x(x) - G_x(\hat{x}) \|_1 + \| G_y(x) - G_y(\hat{x}) \|_1

where :math:`G_x, G_y` are the :math:`3 \times 3` Sobel kernels.


---

Complex / K-Space Losses
=========================

Complex L1 Loss
---------------

**Registry name:** ``complex_l1`` — **Class:** ``ComplexL1Loss``

L1 on the magnitude of complex-valued differences:

.. math::

   \mathcal{L} = \frac{1}{N} \sum_i \sqrt{(\text{Re}(x_i - \hat{x}_i))^2 + (\text{Im}(x_i - \hat{x}_i))^2}


Complex MSE Loss
----------------

**Registry name:** ``complex_mse`` — **Class:** ``ComplexMSELoss``

MSE on both real and imaginary channels:

.. math::

   \mathcal{L} = \frac{1}{N} \sum_i \left[ (\text{Re}(x_i) - \text{Re}(\hat{x}_i))^2 + (\text{Im}(x_i) - \text{Im}(\hat{x}_i))^2 \right]


Spectral K-Space Loss
---------------------

**Registry name:** ``spectral_kspace`` — **Class:** ``SpectralKSpaceLoss``

Compares outputs in the Fourier domain with frequency-dependent weighting:

.. math::

   \mathcal{L}_{k} = \sum_{u,v} w(u,v) \cdot |\mathcal{F}(x)(u,v) - \mathcal{F}(\hat{x})(u,v)|^2

Higher-frequency components can receive larger weights :math:`w(u,v)` to preserve
fine detail.


Frequency-Weighted K-Space L1
-----------------------------

**Registry name:** ``frequency_weighted_l1_kspace`` — **Class:** ``FrequencyWeightedL1Loss``

Variable-density weighting in k-space to compensate for non-uniform sampling:

.. math::

   \mathcal{L} = \sum_{u,v} \frac{1}{P(u,v)} |\hat{k}(u,v) - k(u,v)|


Log-Spectral Loss
-----------------

**Registry name:** ``log_spectral`` — **Class:** ``LogSpectralLoss``

Compares log-magnitude spectra:

.. math::

   \mathcal{L}_{\text{log}} = \| \log(1 + |\mathcal{F}(x)|) - \log(1 + |\mathcal{F}(\hat{x})|) \|_2^2


Focal Frequency Loss
--------------------

**Registry name:** ``focal_frequency`` / ``focal_frequency_parabolic`` — **Class:** ``InvertedFocalFrequencyLoss``

Down-weights easy (well-reconstructed) frequency components and focuses on
hard (poorly-reconstructed) ones. Can be configured for radial or parabolic weighting modes:

**Radial Mode:**

.. math::

   W(k) = \|k\|^\alpha \implies \mathcal{L}_{\text{focal}} = \sum_{u,v} W(u,v) \cdot |e(u,v)|^2

**Parabolic Mode:**

.. math::

   W(k) = 1 + \alpha(k_x^2 + k_y^2) \implies \mathcal{L}_{\text{focal}} = \sum_{u,v} W(u,v) \cdot |e(u,v)|^2

where :math:`e(u,v) = |\mathcal{F}(x)(u,v) - \mathcal{F}(\hat{x})(u,v)|`.




Sobolev K-Space Loss
--------------------

**Registry name:** ``sobolev_kspace`` — **Class:** ``SobolevKSpaceLoss``

Applies Sobolev norm weighting :math:`(1 + |k|^2)^{s/2}` in k-space:

.. math::

   \mathcal{L}_{\text{Sob}} = \sum_{u,v} (1 + u^2 + v^2)^s \cdot |k(u,v) - \hat{k}(u,v)|^2

The Sobolev exponent :math:`s > 0` penalizes high-frequency errors more heavily.


---

Physics-Informed Losses
========================

Data Consistency Loss
---------------------

**Registry name:** ``data_consistency`` — **Class:** ``DataConsistencyLoss``

Enforces fidelity to acquired k-space measurements:

.. math::

   \mathcal{L}_{DC} = \| M \odot (\mathcal{F}(\hat{x}) - y) \|_2^2

where :math:`M` is the sampling mask, :math:`\mathcal{F}` is the FFT, and
:math:`y` is the measured k-space. This is the fundamental constraint in
all MRI reconstruction.


Parallel Imaging K-Space Loss
-----------------------------

**Registry name:** ``parallel_imaging_kspace`` — **Class:** ``ParallelImagingKSpaceLoss``

Multi-coil SENSE-aware data consistency:

.. math::

   \mathcal{L}_{SENSE} = \sum_{c=1}^{C} \| M \odot (\mathcal{F}(S_c \cdot \hat{x}) - y_c) \|_2^2

where :math:`S_c` is the sensitivity map for coil :math:`c`.


SENSE Adjoint L1 Loss
---------------------

**Registry name:** ``sense_adjoint_l1`` — **Class:** ``SENSEAdjointL1Loss``

Computes L1 error in the SENSE-adjoint image domain:

.. math::

   \mathcal{L} = \left\| \sum_c S_c^* \cdot \mathcal{F}^{-1}(y_c) - \hat{x} \right\|_1


Helmholtz PDE Loss (PINN CSM)
------------------------------

**Registry name:** ``helmholtz_pde`` — **Class:** ``HelmholtzPDELoss``

Enforces the Helmholtz equation on learned sensitivity maps.
At ultra-low field (0.3T), :math:`k^2 \approx 0`, reducing to the
Laplace equation:

.. math::

   \mathcal{L}_{PDE} = \sum_{c=1}^{C} \sum_{j=1}^{N_{coll}} \left| \frac{\partial^2 S_c}{\partial x^2}\bigg|_{r_j} + \frac{\partial^2 S_c}{\partial y^2}\bigg|_{r_j} + k^2 S_c(r_j) \right|^2

where :math:`\{r_j\}` are collocation points and second derivatives are
computed exactly via ``torch.autograd.grad``.


Helmholtz-Hodge Decomposition Loss (Motion Fields)
--------------------------------------------------

**Registry name:** ``hodge_decomposition`` — **Aliases:** ``hodge_motion``,
``helmholtz_hodge`` — **Class:** ``HodgeDecompositionLoss`` — **Domain:** ``image``

Encodes the Helmholtz-Hodge structure of a 2D motion / displacement field,
backing the ``hodge_motion`` strategy with real physics instead of vanilla
pixel losses. The first two channels of the ``[B, C, H, W]`` tensor are
treated as the planar :math:`(u, v)` vector field. Each field splits
:math:`L^2`-orthogonally into a curl-free (irrotational) and a
divergence-free (solenoidal) part:

.. math::

   \mathbf{v} = \nabla\phi + \mathbf{v}_{\rm sol},
   \qquad \langle\nabla\phi,\mathbf{v}_{\rm sol}\rangle_{L^2}=0.

The loss combines a divergence-discrepancy (incompressibility) term with a
decomposed-component matching term:

.. math::

   \mathcal{L} = \lambda_{\rm div}\,\| \nabla\!\cdot\mathbf{v}_{\rm pred}
                                - \nabla\!\cdot\mathbf{v}_{\rm tgt} \|
               + \lambda_{\rm comp}\,\big(
                   \| \mathbf{v}^{\rm pred}_{\rm irrot} - \mathbf{v}^{\rm tgt}_{\rm irrot}\|
                 + \| \mathbf{v}^{\rm pred}_{\rm sol}   - \mathbf{v}^{\rm tgt}_{\rm sol}\|
                 \big).

The decomposition reuses
:func:`spectramr.infrastructure.physics.helmholtz_hodge.helmholtz_hodge_decompose`
and :func:`spectramr.infrastructure.physics.helmholtz_hodge.divergence`; the
3D primitive is invoked with a degenerate depth axis (``D=1``) so every
:math:`\partial/\partial z` term vanishes and the operators collapse to
their 2D forms. Requires :math:`\geq 2` channels; shape mismatches raise
``ValueError``.


Hamilton-Jacobi-Bellman Residual Loss (Optimal Control)
-------------------------------------------------------

**Registry name:** ``hjb_residual`` — **Aliases:** ``hjb_trajectory``,
``hjb_waveform`` — **Class:** ``HJBResidualLoss`` — **Domain:** ``image``

Backs the ``hjb_trajectory`` and ``hjb_waveform`` strategy keys with the
optimal-control mathematics they advertise, instead of routing a vanilla
reconstruction objective. The value function :math:`V(\mathbf k, t)` of the
trajectory/waveform design problem satisfies the Hamilton-Jacobi-Bellman PDE
[Fleming-Soner 2006, §III.5], whose Hamiltonian min-control term has the
closed form

.. math::

   \partial_t V + \min_{\|\mathbf u\|\le G_{\max}}
       \bigl\{\nabla_{\mathbf k} V\cdot\mathbf u + L(\mathbf k,t)\bigr\} = 0,
   \qquad
   \min_{\|\mathbf u\|\le G_{\max}}\nabla_{\mathbf k}V\cdot\mathbf u
       = -G_{\max}\,\|\nabla_{\mathbf k}V\|,

under the hardware gradient-amplitude bound :math:`G_{\max}`. The network
output is read as a sampled value field — the **channel axis is the control
time axis** and the trailing spatial axes are the (2-D) state coordinates,
i.e. ``[B, T, H, W]``. A bilinear ``grid_sample`` interpolant makes the field
differentiable at continuous collocation points so the deep-Galerkin residual
operator :math:`\mathcal R[V] = \partial_t V - G_{\max}\|\nabla_{\mathbf k}V\|`
can be formed by autograd [Sirignano-Spiliopoulos 2018, JCP].

Because the absolute running cost :math:`L` of a reconstruction task is
unknown, the loss is a **discrepancy** formulation: it penalises
:math:`\|\mathcal R[V_{\rm pred}] - \mathcal R[V_{\rm tgt}]\|^2` over the
collocation set, using the target's residual operator as the running cost. A
prediction that satisfies the same HJB relation as the target scores ~0; any
departure — in particular one changing the Hamiltonian
:math:`G_{\max}\|\nabla V\|` term — is penalised. The implementation reuses
:func:`spectramr.infrastructure.physics.hjb_optimal_control.hjb_residual_loss`.
Shape mismatches raise ``ValueError``; reduction is ``"mean"`` or ``"sum"``.


Fisher-Rao Geodesic Loss (Information Geometry)
-----------------------------------------------

**Registry name:** ``fisher_rao_geodesic`` — **Aliases:** ``fisher_rao_flow``,
``fr_geodesic``, ``information_geometry_geodesic`` — **Class:**
``FisherRaoGeodesicLoss`` — **Domain:** ``image``

Backs the ``fisher_rao_flow`` strategy with the information-geometry it
advertises, replacing a Euclidean residual that ignores the curvature of
distribution space. Each :math:`[B, C, H, W]` field is summarised per sample
as a Gaussian acquisition likelihood :math:`\mathcal{N}(\mu,\sigma^2)` with
natural coordinates :math:`\theta=(\mu,\sigma)`. The Fisher-Rao distance
between the prediction's and target's distributions is the metric arc-length
of the displacement :math:`\Delta\theta=\theta_q-\theta_p` under the Gaussian
Fisher metric evaluated at the geodesic midpoint:

.. math::

   \mathrm{FR}(p,q) \;\approx\; \big\|\theta_q-\theta_p\big\|_{g(\bar\theta)},
   \qquad
   g(\theta) = \mathrm{diag}\!\left(\tfrac{1}{\sigma^2},\,\tfrac{2}{\sigma^2}\right).

The metric is built through the physics SSOT
:func:`spectramr.infrastructure.physics.fisher_rao_geometry.fisher_information_from_jacobian`
(fed the Gaussian score Jacobian) and the arc-length is taken with
:func:`spectramr.infrastructure.physics.fisher_rao_geometry.fisher_norm`, rather
than hand-coding the metric. Because :math:`g\propto 1/\sigma^2`, the same
:math:`(\mu,\sigma)` gap costs *more* between tight distributions than broad
ones — the curvature signature a plain :math:`L_2` cannot reproduce. The loss
is zero iff the distributions coincide, strictly positive otherwise, smooth and
differentiable; ``reduction`` is ``"mean"`` or ``"sum"`` and shape mismatches
raise ``ValueError``.


Bloch Residual Loss
-------------------

**Registry name:** ``bloch_residual`` — **Class:** ``BlochResidualLoss``

Penalizes deviations from the Bloch equations during training:

.. math::

   \mathcal{L}_{Bloch} = \left\| \frac{dM}{dt} - \gamma M \times B + \frac{M_{xy}}{T_2} + \frac{M_z - M_0}{T_1}\hat{z} \right\|_2^2


Physics-Informed Loss
---------------------

**Registry name:** ``physics_informed`` — **Class:** ``PhysicsInformedLoss``

General-purpose physics constraint registered in
``infrastructure/physics/integration.py``:

.. math::

   \mathcal{L}_{phys} = \mathcal{L}_{data} + \lambda_{PDE}\,\mathcal{L}_{PDE} + \lambda_{BC}\,\mathcal{L}_{BC}


Energy Conservation Loss
-------------------------

**Registry name:** ``energy_conservation`` — **Class:** ``EnergyConservationLoss``

Enforces Parseval's theorem — total energy must be preserved between
image and k-space:

.. math::

   \mathcal{L}_{E} = \left| \sum_{i} |x_i|^2 - \sum_{u,v} |k(u,v)|^2 \right|


Rician Consistency Loss
-----------------------

**Registry name:** ``rician_consistency`` — **Class:** ``RicianConsistencyLoss``

Enforces that magnitude images follow the Rician noise distribution.
The expected value of a Rician-distributed signal is:

.. math::

   E[M] = \sigma \sqrt{\frac{\pi}{2}} L_{1/2}\left(-\frac{A^2}{2\sigma^2}\right)

where :math:`L_{1/2}` is the Laguerre polynomial and :math:`A` is the
true signal amplitude.


---

GAN Losses
==========

Standard GAN Loss
-----------------

**Registry name:** ``gan_standard`` — **Class:** ``StandardGANLoss``

Binary cross-entropy game:

.. math::

   \mathcal{L}_D = -\mathbb{E}[\log D(x)] - \mathbb{E}[\log(1 - D(G(z)))]

.. math::

   \mathcal{L}_G = -\mathbb{E}[\log D(G(z))]


Least-Squares GAN
-----------------

**Registry name:** ``gan_lsgan`` — **Class:** ``LSGANLoss``

Replaces log-likelihood with least-squares for more stable gradients:

.. math::

   \mathcal{L}_D = \mathbb{E}[(D(x) - 1)^2] + \mathbb{E}[D(G(z))^2]

.. math::

   \mathcal{L}_G = \mathbb{E}[(D(G(z)) - 1)^2]


Hinge Loss
----------

**Registry name:** ``gan_hinge`` — **Class:** ``HingeLoss``

Spectral normalization-compatible hinge formulation:

.. math::

   \mathcal{L}_D = \mathbb{E}[\max(0, 1 - D(x))] + \mathbb{E}[\max(0, 1 + D(G(z)))]

.. math::

   \mathcal{L}_G = -\mathbb{E}[D(G(z))]


Wasserstein GAN
---------------

**Registry name:** ``gan_wgan`` — **Class:** ``WGANLoss``

Earth-mover distance with Lipschitz constraint:

.. math::

   \mathcal{L}_D = \mathbb{E}[D(G(z))] - \mathbb{E}[D(x)]

.. math::

   \mathcal{L}_G = -\mathbb{E}[D(G(z))]

Requires gradient penalty or spectral normalization for Lipschitz continuity.


R1 Regularization
-----------------

**Registry name:** ``r1_regularization`` — **Class:** ``R1RegularizationLoss``

Gradient penalty on real samples (Mescheder et al., 2018):

.. math::

   \mathcal{R}_1 = \frac{\gamma}{2} \mathbb{E}\left[\|\nabla D(x)\|^2\right]


---

Regularization Losses
=====================

Strategy-Managed Losses & Regularization Options
------------------------------------------------

Several critical regularization terms are computed inline by their respective training strategies rather than being instantiated as standalone modules via the ``LossBuilder``. These losses require tight integration with the training graph (e.g., accessing discriminator intermediate features, computing gradients across the model, or relying on complex spatial transformations).

When these keys are enabled in the YAML configuration, they are safely skipped by the ``LossBuilder`` to prevent instantiation crashes, and are instead parsed and applied natively by the underlying strategy logic:

* **GAN & Adversarial Regularization**:
  - **r1** (``R1RegularizationLoss``): R1 gradient penalty for GAN training.
  - **patch_nce**: Patch-based Noise Contrastive Estimation for unpaired translation.
* **Vector Quantization & Latent**:
  - **commitment** / **codebook**: Vector quantization regularizations.
  - **reconstruction** / **recon**: High-level meta-keys for VAE/Latent base reconstructions.
  - **kl**: KL-divergence for VAE latent space smoothing.
* **Physics & Biophysical Constraints**:
  - **bloch** / **bloch_residual** (``BlochResidualLoss``): Physics consistency enforcing Bloch equations during the forward pass.
  - **biophysical_flow**: Constraints enforcing computational fluid dynamics and Navier-Stokes equations for flow-MRI.
  - **physics_informed** / **physics_constraint**: PINN (Physics-Informed Neural Network) loss logic managed directly in coordinate-MLP paradigms.
* **Virtual Fiducials & ADMM**:
  - **marker**: Penalises deviation from an ideal simulated marker corruption.
  - **prior**: ADMM prior anchoring loss for virtual fiducial tracking.
* **Registration & Spatial Deformation**:
  - **sim** / **lncc**: Local Normalized Cross-Correlation similarity metric for B0 map registration.
  - **smooth**: Spatial smoothness constraint on displacement fields during deformable registration.
* **Unrolled Optimization (PadNet)**:
  - **padnet_reg** / **padnet_l2** / **padnet_dc**: Iterative data-consistency and L2 regularization applied per-cascade in unrolled optimization networks.
* **Knowledge Distillation**:
  - **distill**: Feature-matching and response-based knowledge distillation constraints from a teacher network.
* **Perceptual Ensembles**:
  - **anat**, **style**, **content**: Domain-specific perceptual constraints managed by deep feature extraction hooks.

Total Variation Loss
--------------------

**Registry name:** ``tv`` — **Class:** ``TotalVariationLoss``

Promotes piecewise-smooth images:

.. math::

   \mathcal{L}_{TV} = \sum_{i,j} \sqrt{(x_{i+1,j} - x_{i,j})^2 + (x_{i,j+1} - x_{i,j})^2 + \epsilon}


KL Divergence Loss
------------------

**Registry name:** ``kl`` — **Class:** ``KLDivergenceLoss``

Standard VAE regularization term:

.. math::

   D_{KL}(q(z|x) \| p(z)) = -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)


Vector Quantization Loss
------------------------

**Registry name:** ``vq`` — **Class:** ``VQLoss``

Commitment loss for VQ-VAE codebook learning:

.. math::

   \mathcal{L}_{VQ} = \| z_e - \text{sg}[e] \|_2^2 + \beta \| \text{sg}[z_e] - e \|_2^2

where :math:`z_e` is the encoder output, :math:`e` is the nearest codebook
entry, and :math:`\text{sg}[\cdot]` denotes stop-gradient.


Deep Supervision Loss
---------------------

**Registry name:** ``deep_supervision`` — **Class:** ``DeepSupervisionLoss``

Applies a base loss at multiple resolution levels of the decoder:

.. math::

   \mathcal{L}_{DS} = \sum_{s=1}^{S} w_s \cdot \mathcal{L}_{\text{base}}(x^{(s)}, \hat{x}^{(s)})

where :math:`x^{(s)}` is downsampled by factor :math:`2^s` and weights
:math:`w_s` decrease with depth.


SNR-Preserving Loss
-------------------

**Registry name:** ``snr_preserving`` — **Class:** ``SNRPreservingLoss``

Penalizes changes in local SNR between input and output, critical for
clinical MRI where noise characteristics must be preserved:

.. math::

   \mathcal{L}_{SNR} = \sum_p \left| \frac{\mu_p(x)}{\sigma_p(x)} - \frac{\mu_p(\hat{x})}{\sigma_p(\hat{x})} \right|

where patches :math:`p` slide over the image.


Persistent Homology Loss
-------------------------

**Registry name:** ``topological`` — **Class:** ``PersistentHomologyLoss``

Compares topological features (connected components, loops, voids) between
predicted and target images using persistence diagrams from TDA.


Cubical Persistent-Homology Wasserstein-2 Loss
----------------------------------------------

**Registry name:** ``cubical_ph_w2`` — **Class:** ``CubicalPHWassersteinLoss``

Cubical persistent homology + Wasserstein-2 matching between the prediction and
target persistence diagrams. The differentiable path routes the gradient back
into pred through the birth/death voxels of each matched feature. Requires the
``[topology]`` extra (``gudhi`` + ``POT``) — the class raises ``ImportError`` at
construction without them, and the ``geomamba_ulf`` strategy re-raises rather
than silently degrading to L1.

**Diagonal-projection terms.** The W2 matching pads each diagram with the
diagonal projections of the other, so a pred feature the target lacks is matched
to the diagonal. This is what gives hallucinated / spurious topology a gradient:
each diagonal-matched pred feature :math:`(b, d)` contributes its
:math:`L_\infty` distance to the
diagonal, :math:`\left(\tfrac{d-b}{2}\right)^p`, differentiable through both pred
voxels — pushing the spurious feature's persistence toward zero. The loss still
vanishes at ``pred == target`` (each feature matches its identical partner) and
remains bounded by the Cohen-Steiner stability inequality.


Evidential Regularization Loss (NIG)
------------------------------------

**Registry name:** ``evidential`` — **Class:** ``EvidentialLoss``

Replaces deterministic point estimates with Normal-Inverse-Gamma (NIG)
distributions for aleatoric and epistemic uncertainty quantification (Pillar 9).
The generator predicts four parameters :math:`\gamma, \nu, \alpha, \beta`, and the loss maximizes model evidence:

.. math::

   \mathcal{L}_{evidential} = \mathcal{L}_{NLL}(\gamma, \nu, \alpha, \beta, y) + \lambda \mathcal{L}_{R}(\gamma, \nu, \alpha)

where :math:`\mathcal{L}_{R}` is the evidence regularizer pulling predictions to zero uncertainty when the prediction contains high error.



Hyperelastic Jacobian Loss
--------------------------

**Registry name:** ``hyperelastic_jacobian`` — **Class:** ``HyperelasticJacobianLoss``

Applies topological and diffeomorphism constraints directly on intermediate deformation fields.
It penalizes non-volume-preserving deformations (Pillar 10):

.. math::

   \mathcal{L}_{Je} = \frac{1}{|V|}\sum_{x \in V} \frac{(det(J_\phi(x)) - 1)^2}{det(J_\phi(x))}

where :math:`J_\phi` is the spatial Jacobian matrix of the deformation grid.



Koopman Linearity (DMD-residual) Loss
-------------------------------------

**Registry name:** ``koopman_linearity`` —
**Aliases:** ``koopman_dmd_residual``, ``koopman_linearity_residual`` —
**Domain:** ``image`` — **Class:** ``KoopmanLinearityLoss``

Backs the ``koopman_fmri`` paradigm with genuine Koopman-operator
mathematics instead of a vanilla reconstruction term. For a temporal fMRI
series (input ``[B, T, H, W]``, or ``[B, C, H, W]`` with the channel axis read
as time), each spatial snapshot is lifted into a learned observable space
:math:`\Phi` by the encoder of the existing primitive
:class:`spectramr.models.temporal.koopman_operator.KoopmanMotionFilter`. The
best-fit *constant* linear Koopman operator :math:`K` is fitted to the
prediction's own snapshots by ridge-stabilised least squares (the exact-DMD
estimator :math:`K = Z_{+} Z_{-}^{+}`, then detached), and the loss penalizes
the linear-evolution residual:

.. math::

   \mathcal{L}_{\text{koopman}}
     = \big\lVert \Phi(x_{t+1}) - K\,\Phi(x_t) \big\rVert^2 .

A sequence that genuinely evolves linearly in :math:`\Phi` (a constant series,
or one propagated through a fixed operator) yields a near-zero residual; an
unstructured or nonlinear series yields a positive one. Detaching :math:`K`
makes the term measure *how far the prediction is from admitting a single
linear propagator* rather than the fit capacity of a free matrix.



---

Diffusion Process Losses
=========================

Diffusion MSE Loss
-------------------

**Registry name:** ``diffusion_mse`` — **Class:** ``DiffusionMSELoss``

The standard denoising diffusion training objective. Predicts noise
:math:`\epsilon` added to a clean sample at timestep :math:`t`:

.. math::

   \mathcal{L}_{\text{diff}} = \mathbb{E}_{t, x_0, \epsilon} \left[
   \| \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\, t) \|^2
   \right]

For cold diffusion (``prediction_type: sample``), predicts :math:`x_0` directly:

.. math::

   \mathcal{L}_{\text{cold}} = \mathbb{E}_{t, x_0} \left[
   \| x_0 - D_\theta(\tilde{x}_t, t) \|^2
   \right]

**Key configuration:**

.. code-block:: yaml

   losses:
     kspace_losses:
       - name: diffusion_mse
         weight: 1.0
         enabled: true
         kwargs:
           prediction_type: sample    # or 'noise'
           reduction: mean


Velocity Prediction Loss
-------------------------

**Registry name:** ``diffusion_velocity`` — **Class:** ``VelocityPredictionLoss``

Flow-matching objective that predicts the velocity field :math:`v_t` pointing
from noise to data along a straight path:

.. math::

   \mathcal{L}_v = \mathbb{E}_{t, x_0, x_1} \left[
   \| v_\theta(x_t, t) - (x_1 - x_0) \|^2
   \right]

where :math:`x_t = (1-t)x_0 + t x_1` is the linear interpolation and
:math:`v_\theta` is the model's velocity prediction. Used by
``RectifiedFlowGenerator``.


Latent Diffusion Loss
---------------------

**Registry name:** ``latent_diffusion`` — **Class:** ``LatentDiffusionLoss``

Extends ``diffusion_mse`` to latent space encoded by a VAE:

.. math::

   \mathcal{L}_{LDM} = \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2

where :math:`z = \mathcal{E}(x)` is the VAE-encoded latent and
:math:`c` is the conditioning signal (e.g., acceleration factor, mask).
The VAE encoder is frozen during LDM training.


---

Registration & Deformation Losses
===================================

Local Cross-Correlation Loss
-----------------------------

**Registry name:** ``local_cross_correlation_loss`` — **Class:** ``LocalCrossCorrelationLoss``

Measures local image similarity as the normalized cross-correlation within
sliding windows, invariant to local intensity offsets:

.. math::

   \text{LNCC}(x, \hat{x}) = -\sum_p
   \frac{\bigl(\sum_q (x_q - \bar{x}_p)(\hat{x}_q - \bar{\hat{x}}_p)\bigr)^2}
   {\bigl(\sum_q (x_q - \bar{x}_p)^2\bigr)\bigl(\sum_q (\hat{x}_q - \bar{\hat{x}}_p)^2\bigr)}

Primary similarity metric for B0 field-map registration and motion-corrected
reconstruction. Robust to intensity scaling between scans.


Jacobian Determinant Loss
--------------------------

**Registry name:** ``jacobian_determinant`` — **Class:** ``JacobianDeterminantLoss``

Penalizes spatial folds (negative Jacobian determinants) in deformation fields
to ensure diffeomorphism:

.. math::

   \mathcal{L}_{J} = \sum_{x} \max\bigl(0, -\det(J_\phi(x))\bigr)

Differs from ``hyperelastic_jacobian`` in that it only penalizes negative
determinants (folding), not deviation from volume-preservation.


MIND-SSC (Modality Independent Neighbourhood Descriptor)
---------------------------------------------------------

**Registry name:** ``mind_ssc`` — **Class:** ``MINDSSCLoss``

Descriptor-based similarity metric for multimodal image registration
(e.g., T1 → T2 registration without intensity normalization):

.. math::

   \text{MIND}(x, p) = \frac{1}{n} \exp\left(-\frac{\text{SSC}(x, p)}{V(x)}\right)

where :math:`\text{SSC}` is the self-similarity in a patch neighbourhood.
Compared across modalities as an L1 distance of descriptors.


---

Contrastive & Self-Supervised Losses
======================================

Contrastive Loss (SimCLR-style)
--------------------------------

**Registry name:** ``contrastive`` — **Class:** ``ContrastiveLoss``

Normalised Temperature-scaled Cross Entropy (NT-Xent):

.. math::

   \mathcal{L}_{\text{NCE}} = -\log
   \frac{\exp(z_i \cdot z_j / \tau)}
   {\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(z_i \cdot z_k / \tau)}

where :math:`z_i, z_j` are normalized projections of two augmented views of
the same image. Temperature :math:`\tau` controls hardness of negatives.
Used for self-supervised pre-training (MAE / ``ssl`` training mode).


Contrastive Anatomy Disentanglement
------------------------------------

**Registry name:** ``contrastive_anatomy_disentanglement`` — **Class:** ``CAContrastiveLoss``

Extends NT-Xent with anatomy/modality disentanglement for multi-contrast
MRI. Pulls same-anatomy pairs together and pushes different-anatomy pairs
apart in a factorized latent space:

.. math::

   \mathcal{L}_{CA} = \mathcal{L}_{NCE}(z_{anat}) + \lambda \mathcal{L}_{NCE}(z_{mod})

where :math:`z_{anat}` and :math:`z_{mod}` are the disentangled anatomy and
modality codes.


PatchNCE Loss
--------------

**Registry name:** ``patch_nce`` — **Class:** ``PatchNCELoss``

Patch-level noise contrastive estimation for CUT (Contrastive Unpaired
Translation). Maximizes mutual information between same-position patches
in input and output domains:

.. math::

   \mathcal{L}_{\text{PatchNCE}} = \sum_{l} \sum_s -\log
   \frac{\exp(z_l^s \cdot z_l^{+,s} / \tau)}
   {\exp(z_l^s \cdot z_l^{+,s} / \tau) + \sum_{n=1}^{N} \exp(z_l^s \cdot z_l^{-,n} / \tau)}

Supersedes CycleGAN cycle-consistency for unpaired domain adaptation.


---

beta-TCVAE Loss
================

**Registry name:** ``beta_tc_vae`` — **Class:** ``BetaTCVAELoss``
**File:** ``src/spectramr/models/losses/beta_tc_vae_loss.py``

Decomposes the ELBO into three interpretable terms for better
disentanglement:

.. math::

   \mathcal{L} = \mathbb{E}[\log p(x|z)] - \underbrace{\alpha I(z;x)}_{\text{MI}}
   + \underbrace{\beta \text{TC}(z)}_{\text{total correlation}}
   + \underbrace{\gamma D_{KL}(q(z)\|p(z))}_{\text{dim-wise KL}}

where :math:`\text{TC}(z) = D_{KL}(q(z) \| \prod_j q(z_j))` measures
statistical dependence between latent dimensions. Higher :math:`\beta`
forces the encoder to find independent factors of variation.

**Use case:** Multi-contrast reconstruction where anatomy and pathology
should be disentangled into separate latent dimensions.


---

QSM / Quantitative Losses
==========================

Self-Supervised Physics Loss (QMap)
-----------------------------------

**Registry name:** ``qmap_physics`` — **Class:** ``SelfSupervisedPhysicsLoss``

Enforces quantitative MRI signal models during training:

.. math::

   \mathcal{L}_{QMap} = \| S(\hat{\theta}) - y \|_2^2

where :math:`S(\hat{\theta})` is the forward signal model evaluated at
predicted quantitative parameters :math:`\hat{\theta}` (e.g., T1, T2, PD).

Typical YAML configuration (3-parameter mode):

.. code-block:: yaml

   losses:
     image_losses:
       - name: qmap_physics
         weight: 1.0
         enabled: true


Advanced Physics Loss with B0/B1 Field Correction (QMap)
---------------------------------------------------------

**Registry name:** ``qmap_physics_advanced`` — **Class:** ``AdvancedSelfSupervisedPhysicsLoss``

Extends the 3-parameter QMap loss (T1, T2, PD) to the full **5-parameter** model
(T1, T2, PD, B0, B1), adding:

1. **B0 off-resonance phase** — complex-valued signal simulation:

   .. math::

      S_{complex}(\hat{\theta}, B_0) = |S(\hat{\theta})| \cdot e^{i 2\pi f_{B_0} TE}

2. **B1 transmit field correction** — scales nominal flip angle:

   .. math::

      \theta_{eff} = \theta_{nom} \cdot B_1

3. **B1 Smoothness Regularization** — penalises sharp spatial gradients:

   .. math::

      \mathcal{L}_{smooth} = \lambda_{B_1} \| \nabla B_1 \|_2^2

The total loss is:

.. math::

   \mathcal{L}_{adv} = \lambda_{data} \| S_{sim}(\hat{\theta}) - y_{meas} \|_1
                     + \lambda_{B_1} \| \nabla B_1 \|_2^2

**When to use B0/B1 correction:**

- Phase-banding artefacts in k-space data
- Spatial T1/T2 bias near tissue boundaries or sinuses
- High-field (3T+) or low-field systems with strong B0 inhomogeneity

**Configuration (5-parameter mode):**

.. code-block:: yaml

   model:
     model_type: qmap_generator   # or hybrid_qmap_generator

   losses:
     image_losses:
       - name: qmap_physics_advanced
         weight: 1.0
         enabled: true
         kwargs:
           b1_smoothness_weight: 10.0   # Tune: 5.0–20.0

   physics:
     b0_range_hz: 200.0     # ±200 Hz (increase to 400 for near-sinus)
     b1_min: 0.5
     b1_max: 1.5



---

The ``domain=`` vocabulary
==========================

``@register_loss(..., domain=...)`` accepts exactly the values in
:data:`~spectramr.models.losses.registry.REGISTRABLE_DOMAINS`. **Anything else
raises at import.**

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - value
     - block check
     - meaning
   * - ``image``, ``kspace``, ``complex``, ``complex_image``, ``latent``
     - **enforced**
     - Signal domains. The loss must be placed under the matching
       ``losses.<domain>_losses`` block.
   * - ``agnostic``
     - exempt
     - Operates on any tensor, so it is legal under every block.
   * - ``physics``, ``bloch_synthesis``
     - exempt
     - :data:`~spectramr.models.losses.registry.NON_SIGNAL_DOMAINS` — grades
       gradient waveforms, Hamiltonians or Bloch simulations rather than the
       reconstruction signal, so no block corresponds to it.
   * - omitted (``None``)
     - skipped
     - Unannotated. Legal, but the loss is invisible to the domain guards.

.. admonition:: Why an unknown value must raise
   :class: warning

   ``domain=`` is a closed vocabulary, not a free string. Were it looked up in a
   dict, an unrecognised value would fall out as ``None`` — *unannotated* — and a
   typo (``domain="imagee"``) would be indistinguishable from a deliberate
   non-signal domain: both silently skipping every check they were meant to face.

   A loss that genuinely faces no domain check declares ``domain_agnostic=True``,
   so the pass is stated rather than inferred. Adding a value to
   ``REGISTRABLE_DOMAINS`` means deciding which column above it belongs in.

Domain-Aware Loss Routing
=========================

The framework automatically routes losses to the correct tensor domain based
on the ``losses.output_domain`` field in your configuration. This is handled
by the ``DifferentiableFourierBridge`` inside :class:`LossConfigSchema`.

.. list-table:: Domain Routing Rules
   :header-rows: 1
   :widths: 25 25 50

   * - ``output_domain``
     - Active Loss Lists
     - Behaviour
   * - ``image``
     - ``image_losses``
     - Losses computed directly on image-domain tensors.
   * - ``complex_image``
     - ``image_losses``, ``complex_losses``
     - FFT bridge inserted; losses can mix image and complex domains.
   * - ``kspace``
     - ``kspace_losses``, ``complex_losses``
     - All losses computed in Fourier domain. Image losses skipped.

.. admonition:: Common Misconfiguration
   :class: warning

   Setting ``output_domain: image`` while listing only ``kspace_losses``
   will silently skip all losses. Always match ``output_domain`` to the
   lists you populate.

Example — mixed domain (k-space cold diffusion):

.. code-block:: yaml

   losses:
     output_domain: complex_image
     image_losses:
       - name: ssim
         weight: 1.0
         enabled: true
     kspace_losses:
       - name: data_consistency
         weight: 1.0
         enabled: true
     complex_losses:
       - name: complex_l1
         weight: 0.5
         enabled: true


.. rubric:: Full Registry (79 losses)

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Registry Name
     - Class
     - Category
   * - ``background_suppression``
     - ``BackgroundSuppressionLoss``
     - Regularization
   * - ``bce``
     - ``BCELoss``
     - Classification
   * - ``bce_with_logits``
     - ``BCEWithLogitsLoss``
     - Classification
   * - ``biophysical_flow``
     - ``BiophysicalFlowLoss``
     - Physics
   * - ``bloch_residual``
     - ``BlochResidualLoss``
     - Physics
   * - ``complex_l1``
     - ``ComplexL1Loss``
     - K-Space
   * - ``complex_mse``
     - ``ComplexMSELoss``
     - K-Space
   * - ``complex_spatial_gradient``
     - ``ComplexGradientLoss``
     - K-Space
   * - ``contrastive``
     - ``ContrastiveLoss``
     - SSL
   * - ``contrastive_anatomy_disentanglement``
     - ``CAContrastiveLoss``
     - SSL
   * - ``cross_entropy``
     - ``CrossEntropyLoss``
     - Classification
   * - ``data_consistency``
     - ``DataConsistencyLoss``
     - Physics
   * - ``deep_supervision``
     - ``DeepSupervisionLoss``
     - Regularization
   * - ``diffusion_mse``
     - ``DiffusionMSELoss``
     - Diffusion
   * - ``diffusion_velocity``
     - ``VelocityPredictionLoss``
     - Diffusion
   * - ``dino_perceptual``
     - ``DINOv2PerceptualLoss``
     - Image
   * - ``dists``
     - ``DISTS``
     - Image
   * - ``divergence_free``
     - ``DivergenceFreeLoss``
     - Physics
   * - ``domain_adversarial``
     - ``DomainAdversarialLoss``
     - Domain Adaptation
   * - ``edge_consistency``
     - ``EdgePreservationLoss``
     - Image
   * - ``energy_conservation``
     - ``EnergyConservationLoss``
     - Physics
   * - ``evidential``
     - ``EvidentialLoss``
     - Regularization
   * - ``explicit_gradient``
     - ``GradientLoss``
     - Image
   * - ``focal_frequency``
     - ``InvertedFocalFrequencyLoss``
     - K-Space
   * - ``frequency``
     - ``FrequencyLoss``
     - K-Space
   * - ``frequency_domain_consistency``
     - ``FrequencyDomainLoss``
     - K-Space
   * - ``frequency_weighted_l1_kspace``
     - ``FrequencyWeightedL1Loss``
     - K-Space
   * - ``gan_composite``
     - ``CompositeGANLoss``
     - GAN
   * - ``gan_hinge``
     - ``HingeLoss``
     - GAN
   * - ``gan_lsgan``
     - ``LSGANLoss``
     - GAN
   * - ``gan_ralsgan``
     - ``RALSGANLoss``
     - GAN
   * - ``gan_standard``
     - ``StandardGANLoss``
     - GAN
   * - ``gan_wgan``
     - ``WGANLoss``
     - GAN
   * - ``graph_consistency``
     - ``GraphConsistencyLoss``
     - Graph
   * - ``helmholtz_pde``
     - ``HelmholtzPDELoss``
     - Physics/PINN
   * - ``hfen``
     - ``HFENLoss``
     - Image
   * - ``histogram_consistency``
     - ``HistogramConsistencyLoss``
     - Domain Adaptation
   * - ``huber``
     - ``HuberLoss``
     - Image
   * - ``hyperelastic_jacobian``
     - ``HyperelasticJacobianLoss``
     - Regularization
   * - ``jacobian_determinant``
     - ``JacobianDeterminantLoss``
     - Registration
   * - ``kl``
     - ``KLDivergenceLoss``
     - VAE
   * - ``l1``
     - ``L1Loss``
     - Image
   * - ``l2``
     - ``MSELoss``
     - Image
   * - ``latent_consistency``
     - ``LatentConsistencyLoss``
     - VAE
   * - ``latent_diffusion``
     - ``LatentDiffusionLoss``
     - Diffusion
   * - ``latent_regularization``
     - ``LatentRegularizationLoss``
     - VAE
   * - ``local_cross_correlation_loss``
     - ``LocalCrossCorrelationLoss``
     - Registration
   * - ``log_spectral``
     - ``LogSpectralLoss``
     - K-Space
   * - ``log_spectral_phase``
     - ``LogSpectralPhaseLoss``
     - K-Space
   * - ``lpips``
     - ``LPIPSLoss``
     - Image
   * - ``mind_ssc``
     - ``MINDSSCLoss``
     - Registration
   * - ``modality_swap``
     - ``ModalitySwapLoss``
     - Domain Adaptation
   * - ``ms_ssim``
     - ``MSSSIMLoss``
     - Image
   * - ``non_cartesian_graph``
     - ``NonCartesianGraphLoss``
     - Graph
   * - ``parallel_imaging_kspace``
     - ``ParallelImagingKSpaceLoss``
     - Physics
   * - ``patch_nce``
     - ``PatchNCELoss``
     - SSL/Other
   * - ``perceptual``
     - ``PerceptualLoss``
     - Image
   * - ``perceptual_latent``
     - ``PerceptualLatentLoss``
     - VAE
   * - ``physics_constraint``
     - ``PhysicsConstraintLoss``
     - Physics
   * - ``physics_informed``
     - ``PhysicsInformedLoss``
     - Physics
   * - ``qmap_physics``
     - ``SelfSupervisedPhysicsLoss``
     - QSM
   * - ``qmap_physics_advanced``
     - ``AdvancedSelfSupervisedPhysicsLoss``
     - QSM
   * - ``r1_regularization``
     - ``R1RegularizationLoss``
     - GAN
   * - ``rician_consistency``
     - ``RicianConsistencyLoss``
     - Physics
   * - ``sense_adjoint_l1``
     - ``SENSEAdjointL1Loss``
     - Physics
   * - ``sense_adjoint_phase``
     - ``SENSEAdjointPhaseLoss``
     - Physics
   * - ``smooth_l1``
     - ``SmoothL1Loss``
     - Image
   * - ``smoothness_loss``
     - ``SmoothnessLoss``
     - Regularization
   * - ``snr_preserving``
     - ``SNRPreservingLoss``
     - Regularization
   * - ``sobel_edge``
     - ``SobelLoss``
     - Image
   * - ``sobolev_frequency``
     - ``SobolevFrequencyLoss``
     - K-Space
   * - ``sobolev_kspace``
     - ``SobolevKSpaceLoss``
     - K-Space
   * - ``spectral_graph``
     - ``SpectralGraphLoss``
     - Graph
   * - ``spectral_kspace``
     - ``SpectralKSpaceLoss``
     - K-Space
   * - ``ssim``
     - ``SSIMLoss``
     - Image
   * - ``topological``
     - ``PersistentHomologyLoss``
     - Graph
   * - ``tv``
     - ``TotalVariationLoss``
     - Regularization
   * - ``uncertainty``
     - ``UncertaintyAwareLoss``
     - Regularization
   * - ``vq``
     - ``VQLoss``
     - VAE
   * - ``vq_kl``
     - ``VQKLLoss``
     - VAE
   * - ``vqgan``
     - ``VQGANLoss``
     - VAE
   * - ``weighted_kspace_l1``
     - ``WeightedKSpaceL1Loss``
     - K-Space


Distributional / Federated Losses
=================================

Kernelised Stein Discrepancy Loss
---------------------------------

**Registry name:** ``stein_discrepancy`` — **Class:** ``SteinDiscrepancyLoss``
(domain ``image``; aliases ``ksd``, ``kernelized_stein_discrepancy``,
``stein_federated_consistency``).

Backs the ``stein_federated`` strategy, which advertises
kernelised-Stein-discrepancy (KSD) mathematics for federated posterior
aggregation. This loss penalises the KSD between the (aggregated) prediction's sample
distribution and the target distribution.

KSD measures how far a sample set :math:`\{x_i\}` is from a target distribution
:math:`p` via the Stein operator and an inverse-multi-quadric kernel, **without**
the normalising constant — only the score :math:`s_p(x)=\nabla_x\log p(x)`:

.. math::

   \mathrm{KSD}^2(P\,\|\,Q_n)
     = \frac{1}{n(n-1)}\sum_{i\neq j} k_p(x_i,x_j),
   \qquad
   k_p(x,y)=s_p(x)^\top k(x,y)\,s_p(y)+\dots

The target batch defines an empirical Gaussian
:math:`p=\mathcal N(\mu_t,\Sigma_t)`, so the score is
:math:`s_p(x)=-\Sigma_t^{-1}(x-\mu_t)` (diagonal shrinkage on
:math:`\Sigma_t` keeps it invertible on degenerate batches). The prediction
samples are then scored against this target distribution; the discrepancy is
near zero when prediction and target share a distribution and grows as the
aggregated prediction drifts (mode collapse, biased aggregation, DP-noise
corruption). The U-statistic estimator is reused verbatim from
:func:`spectramr.core.metrics.stein.kernelised_stein_discrepancy`; the loss adds a
differentiable in-graph twin (identical IMQ Stein-kernel math) so gradients flow
to the prediction. ``batch_chunks`` splits the batch into federated
mini-cohorts; ``reduction`` is ``"mean"`` or ``"sum"`` over chunks.

.. autoclass:: spectramr.models.losses.stein_discrepancy_loss.SteinDiscrepancyLoss
   :members:
   :no-index:


Front-Door Criterion Loss (Pearl Causal Adjustment)
---------------------------------------------------

**Registry name:** ``frontdoor_criterion`` — **Aliases:**
``front_door_adjustment``, ``pearl_frontdoor``, ``frontdoor_federated_loss``,
``frontdoor_scanner_loss`` — **Class:** ``FrontdoorCriterionLoss`` —
**Domain:** ``image``

Concretises the ``frontdoor_federated`` / ``frontdoor_scanner`` strategy keys,
which advertise Pearl front-door causal-adjustment mathematics. When scanner
identity :math:`S`
confounds anatomy :math:`A` and image :math:`I` but is unmeasured at inference
(a new site / vendor), back-door adjustment fails; the front-door criterion
[Pearl 2009, Th. 3.3.4] still identifies :math:`P(I\mid\mathrm{do}(A))` through
a site-invariant mediator :math:`M` (here the spectral / k-space energy profile)
that intercepts every directed path :math:`A\to I`:

.. math::

   P(I\mid\mathrm{do}(A=a))
       = \sum_m P(M=m\mid A=a)\sum_{a'} P(I\mid M=m, A=a')\,P(A=a').

Each :math:`[B, C, H, W]` batch is summarised per sample into anatomy,
mediator, and image proxies, soft-binned into empirical conditionals, and the
front-door-adjusted estimate is computed through the physics SSOT primitive
:func:`spectramr.infrastructure.physics.causal_frontdoor.frontdoor_adjusted_prediction`.
The loss penalises the discrepancy between the prediction's and target's
front-door-adjusted quantities (so the reconstruction respects the
confounder-robust relationship), plus a mediator-scanner mutual-information
term from
:func:`spectramr.infrastructure.physics.causal_frontdoor.donsker_varadhan_mi`
that enforces the site-invariance making the estimate valid. The scanner/site
label is supplied via ``kwargs["scanner"]`` (defaulting to a per-sample
intensity proxy). A bare :math:`L_1`/:math:`L_2` residual that ignores the
causal primitive is not confounder-robust. The loss is near-zero when
prediction and target coincide, strictly positive otherwise; ``reduction`` is
``"mean"`` or ``"sum"`` and shape mismatches raise ``ValueError``.

.. autoclass:: spectramr.models.losses.frontdoor_criterion_loss.FrontdoorCriterionLoss
   :members:
   :no-index:


Stochastic-Resetting Consistency Loss
-------------------------------------

**Registry name:** ``resetting_consistency`` — **Aliases:**
``stochastic_resetting_consistency``, ``mfpt_consistency`` — **Class:**
``ResettingConsistencyLoss`` — **Domain:** ``image``

Concretises the ``resetting_diffusion`` strategy key, which advertises
stochastic-resetting diffusion mathematics (Evans & Majumdar 2011; Evans,
Majumdar & Schehr 2020). Stochastic resetting intermittently resets the reverse-SDE
sample toward a data-consistent anchor :math:`\mathcal{R}(\hat y)`; this
strictly accelerates first-passage to the data manifold whenever the
coefficient of variation of the unreset passage-time distribution exceeds 1,
and at the optimal resetting rate the distribution sits at *criticality*

.. math::

   \mathrm{CoV}(\tau)=\frac{\sqrt{\mathrm{Var}(\tau)}}{\mathbb{E}[\tau]}=1 .

The loss treats the residual magnitude :math:`|\mathrm{prediction}-\mathrm{target}|`
as an empirical sample of the passage / residual distribution and combines three
terms, all routed through the genuine primitives in
:mod:`spectramr.models.diffusion.stochastic_resetting`: (1) a **criticality** term
:math:`(\mathrm{CoV}(\tau)-1)^2` via
:func:`~spectramr.models.diffusion.stochastic_resetting.coefficient_of_variation`,
(2) a **reset-consistency** term that applies one Bernoulli
:func:`~spectramr.models.diffusion.stochastic_resetting.apply_resetting` step
toward the anchor (= target) and penalises the leftover discrepancy (exactly
zero when prediction equals the anchor), and (3) an **MFPT** term penalising
:func:`~spectramr.models.diffusion.stochastic_resetting.mfpt_with_resetting` at the
configured resetting rate so training shortens the expected passage to the data
manifold. A bare :math:`L_1`/:math:`L_2` residual ignoring the resetting
primitive is not the advertised objective. ``reduction`` is ``"mean"`` or
``"sum"`` and shape mismatches raise ``ValueError``.

.. autoclass:: spectramr.models.losses.resetting_consistency_loss.ResettingConsistencyLoss
   :members:
   :no-index:


Lévy Score-Consistency Loss (α-stable Diffusion)
------------------------------------------------

**Registry name:** ``levy_score_consistency`` — **Aliases:**
``alpha_stable_score``, ``fractional_levy_score`` — **Class:**
``LevyScoreConsistencyLoss`` — **Domain:** ``image``

Backs the ``levy_diffusion`` strategy with genuine heavy-tailed
:math:`\alpha`-stable Lévy mathematics instead of a vanilla Gaussian
denoising objective. Standard score-based diffusion fits the
denoising-score-matching residual to a light-tailed (Brownian) model; at
ultra-low-field SNR the residuals are heavy-tailed (Rician / non-central
chi), which a Gaussian objective under-weights. The loss treats the
reconstruction residual :math:`r = \text{prediction} - \text{target}` as
the network's score estimate and matches it against the fractional
:math:`\alpha`-stable score target:

.. math::

   \mathcal{L} = \mathbb{E}\!\left[\big\| r - (-\,\text{noise}/\sigma^{\alpha}) \big\|^2\right],
   \qquad \text{noise} \sim \text{Stable}(\alpha,\sigma),

where the noise draw and the :math:`\sigma^{\alpha}` scaling both depend on
the stability index :math:`\alpha\in(0,2]` (:math:`\alpha=2` recovers the
Gaussian VP-SDE limit). The objective is computed by genuinely calling
:func:`spectramr.models.diffusion.alpha_stable_levy.fractional_score_matching_loss`
(Chambers-Mallows-Stuck α-stable sampler); a bare :math:`L_1`/:math:`L_2`
residual ignoring the primitive is not the advertised objective. The
``alpha`` and ``sigma`` dispersion are constructor knobs; shape mismatches
raise ``ValueError``.

.. autoclass:: spectramr.models.losses.levy_score_consistency_loss.LevyScoreConsistencyLoss
   :members:
   :no-index:


Sparsifying-Frame Coherence Penalty (A-6.5 SCO-Frame)
-----------------------------------------------------

**Registry name:** ``frame_coherence`` — **Aliases:** ``mutual_coherence_penalty``,
``frame_incoherence`` — **Class:** ``FrameCoherenceLoss`` — **Domain:** ``agnostic``

The coherence-optimisation half of A-6.5 SCO-Frame. The framework had *fixed*-transform
sparsity priors (``wavelet_sparsity_l1``, ``kspace_l1_sparsity``) but no **learnable**
sparsifying transform whose atoms are optimised for **mutual incoherence**. Mutual
incoherence is a classical sparse-recovery principle — for a *fixed* dictionary, a low
coherence :math:`\mu` certifies unique :math:`\ell_1` recovery (Donoho-Elad). That
theorem does **not** transfer to a *learned*, task-specific conv frame trained
end-to-end with soft-thresholding (not :math:`\ell_1` minimisation); here incoherence is
an **empirical** regulariser and the claim is the testable ablation below, not a recovery
guarantee. This loss is that lever. For the atom matrix :math:`W\in\mathbb{R}^{m\times n}`
(``m`` unit-normalised atoms), it is the Gram off-diagonal Frobenius energy

.. math::

   \mathcal{L}_{\mathrm{coh}} = \sum_{i\neq j}\langle\hat d_i,\hat d_j\rangle^2
       = \lVert\hat W\hat W^\top - I_m\rVert_F^2 ,

which is 0 iff the atoms are mutually orthogonal and rises with coherence. It is a
**parameter** penalty (not a ``(pred, target)`` loss): invoked without a frame /
atom tensor it **raises** so it cannot silently no-op in a YAML ``losses``
block. It is wired through
:class:`~spectramr.infrastructure.training.strategies.sparse_frame_strategy.SparseFrameStrategy`
(``training_mode: sparse_frame``), which trains a
:class:`~spectramr.models.generators.tight_frame_learner.TightFrameLearner` (a learnable
convolutional analysis/synthesis frame, ``synthesize`` = the exact ``conv_transpose2d``
adjoint of ``analyze``) with reconstruction + an always-on **Parseval-tightness** term
(``lambda_parseval``) + this **coherence** term (the one-knob ``lambda_coherence``).
Tightness and incoherence are distinct objectives — a tight frame can still be coherent —
so the one-knob ablation isolates coherence. The strategy emits ``val_mutual_coherence``
(the classical :math:`\mu = \max_{i\neq j}|\langle\hat d_i,\hat d_j\rangle|`) as a runtime
witness; the oracle is that ``lambda_coherence>0`` yields lower learned :math:`\mu` than
the control at matched reconstruction.

.. autoclass:: spectramr.models.losses.frame_coherence_loss.FrameCoherenceLoss
   :members:
   :no-index:


Dispersion-latent Bloch autoencoder terms (DL-BAE, bundle M4)
=============================================================

``multifield_data_consistency`` (domain: ``image``)
---------------------------------------------------

.. math::

   \mathcal L_{\mathrm{DC}} = \frac1M\sum_{m=1}^{M} w_m
       \bigl\|\hat x(B_0^{(m)}) - x(B_0^{(m)})\bigr\|_1

Field-summed reconstruction fidelity. Summing over fields is the whole
mechanism, not a convenience: it forces a **single** field-invariant latent to
explain every observed field at once. Fitting each field independently would
minimise a per-field loss just as well while learning nothing transferable
across field — which is what training :math:`M` separate models already does.

``field_weights`` compensates the SNR imbalance across fields; unweighted L1 lets
the high-field terms dominate the gradient, because a 0.055 T acquisition is far
noisier than 7 T.

.. autoclass:: spectramr.models.losses.image.multifield_data_consistency.MultiFieldDataConsistency
   :members:
   :no-index:

``dispersion_monotonicity`` (domain: ``physics``)
-------------------------------------------------

.. math::

   \mathcal L_{\mathrm{mono}} = \bigl\langle
       \mathrm{ReLU}\bigl(-(\Delta_{B_0} T_1 + \epsilon)\bigr)\bigr\rangle

A one-sided hinge enforcing :math:`\partial T_1/\partial B_0 \ge 0`. Exactly zero
on a physical solution, so it adds no gradient there — it only acts when the fit
strays into a region BPP theory forbids.

.. autoclass:: spectramr.models.losses.dispersion_monotonicity_loss.DispersionMonotonicity
   :members:
   :no-index:


References
==========

1. Wang, Z., et al. "Image Quality Assessment: From Error Visibility to
   Structural Similarity." IEEE TIP, 2004.

2. Zhang, R., et al. "The Unreasonable Effectiveness of Deep Features as
   a Perceptual Metric." CVPR, 2018.

3. Mescheder, L., et al. "Which Training Methods for GANs Do Actually
   Converge?" ICML, 2018.

4. Sitzmann, V., et al. "Implicit Neural Representations with Periodic
   Activation Functions." NeurIPS, 2020.

5. Jiang, L., et al. "Focal Frequency Loss for Image Reconstruction and
   Synthesis." ICCV, 2021.

6. Bloch, F. "Nuclear Induction." Physical Review, 1946.

7. Knoll, F., et al. "Deep-Learning Methods for Parallel Magnetic Resonance
   Imaging Reconstruction." IEEE Signal Processing Magazine, 2020.

8. Tancik, M., et al. "Fourier Features Let Networks Learn High Frequency
   Functions in Low Dimensional Domains." NeurIPS, 2020.

9. Amari, S. "Natural Gradient Works Efficiently in Learning." Neural
   Computation, 1998. (Background for evidential regularization)

10. Milletari, F., et al. "V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation." 3DV, 2016.
    (Background for deep supervision loss)

11. Liu, Q., Lee, J., Jordan, M. "A Kernelized Stein Discrepancy for
    Goodness-of-fit Tests." ICML, 2016. (Background for the Stein discrepancy
    loss; with Gorham & Mackey, NeurIPS 2017, and Chwialkowski et al., 2016.)
