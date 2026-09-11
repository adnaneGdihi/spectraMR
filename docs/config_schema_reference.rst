.. _config_schema_reference:

=================================================
Configuration Schema Reference — v1.0
=================================================

.. sectionauthor:: spectraMR Research

Every training run is driven by a single YAML file loaded once via
``TrainingSettings.from_yaml(path)`` and validated immediately. After
loading the object is **frozen** (Pydantic ``frozen=True``) — no
downstream code may mutate it.

.. contents:: Table of Contents
   :depth: 2
   :local:


Top-Level Structure
===================

.. code-block:: yaml

   config_version: '1.0'        # required — validated on load; the ONLY accepted value
   device: cuda                 # cuda | cpu | mps | auto
   seed: 42
   model_domain: kspace         # OPTIONAL convenience knob — image | kspace | (omit)
   deep_supervision_weight: 0.0

   # ---- required nested blocks ----
   data:           { ... }      # DataConfigSchema
   model:          { ... }      # ModelConfigSchema
   optimization:   { ... }      # OptimizationConfigSchema
   logging:                     # LoggingConfigSchema
     experiment_name: my_experiment   # NOT a top-level key
     ...

   # ---- optional nested blocks (default None or default_factory) ----
   training:       { ... }      # TrainingStrategyConfigSchema
   losses:         { ... }      # LossConfigSchema
   metrics:        { ... }      # MetricsConfigSchema (default_factory)
   checkpoint:     { ... }      # CheckpointConfigSchema
   validation:     { ... }      # ValidationConfigSchema
   early_stopping: { ... }      # EarlyStoppingConfigSchema
   ema:            { ... }      # EMAConfigSchema
   loss_logging:   { ... }      # LossLoggingConfigSchema
   undersampling:  { ... }      # AccelerationConfigSchema (was `acceleration:`)
   physics:        { ... }      # PhysicsConfigSchema
   adapters:       { ... }      # AdaptersConfigSchema (declarative chains)
   reporting:      { ... }      # ReportingSettings (end-of-training report)
   parallel:       { ... }      # ParallelismConfigSchema (DDP / FSDP / PEFT)
   metadata:       { ... }      # ExperimentMetadataSchema (name/description/tags/...
                                #   + first-class hypothesis / baseline / primary_metric
                                #   + status (closed: ready | needs_implementation | inert |
                                #     blocked | needs_data | method_demonstration |
                                #     wiring_demonstrator | baseline_control) and
                                #     status_reason; `train` refuses needs_implementation /
                                #     inert / blocked unless --allow-status names it)
   workflow:                    # WorkflowConfigSchema — declared imaging regime × task
     regime: mri_structural     #   Regime enum (closed) — what the signal IS
     task: reconstruction       #   Task enum (closed, optional) — what the arm DOES to it
     signal_domain: image       #   SignalDomain (optional) — which domain the arm CONSUMES
     spatial_rank: 2            #   int (optional) — 2 = slices, 3 = volumes

   # ---- additive blocks ----
   acquisition:    { ... }      # AcquisitionConfigSchema (PILOT codesign, BALD)
   certification:  { ... }      # CertificationConfigSchema (conformal / CHD / PRC / PAC-Bayes)
   audit:          { ... }      # AuditConfigSchema (Tier-3 KSD defensibility)
   mrf:            { ... }      # MRFConfigSchema (MR-fingerprinting metadata)

.. note::

   Compute settings — mixed precision, compilation, gradient checkpointing and
   gradient accumulation — are configured under ``optimization:``, not here.

   **``enforce_nested``.** Cold diffusion's forward process is
   ``x_t = M_t * x_0`` and assumes the masks are *nested* — k-space is only ever
   removed as ``t`` grows, never added. Several families break that by re-drawing
   their pattern per timestep instead of truncating one fixed ranking, and the
   reverse loop has no mechanism to undo an addition. Setting
   ``undersampling.enforce_nested: true`` makes the guarantee structural: the
   cascade is intersected, so ``M_t`` becomes ``M_0 and ... and M_t``.

   Two consequences worth knowing before enabling it:

   * Enforcement can only **remove** samples. A family that re-draws heavily
     collapses towards the ACS, so the accelerator raises when the enforced mask
     retains less than ``nested_tolerance`` (default ``0.5``) of what that
     timestep's **own raw draw** kept, rather than training on a degenerate
     cascade. The denominator is the raw draw, deliberately, and not the
     continuous ``1 / declared_R``: Cartesian families quantise in whole k-space
     lines, so no realised fraction can ever equal a continuous target, and a
     continuous denominator makes the guard fire on sub-line rounding -- which
     leaves ``nested_tolerance: 1.0`` unsatisfiable even for families that nest
     perfectly. Against the raw draw, ``1.0`` is the meaningful strict setting:
     *enforcement must be a no-op*. Whether a family's raw draw honours its declared ``R`` is a
     separate question, answered by ``declared_ladder_defects``.
     Measured at 256² with the default ``0.5``:
     ``radial``, ``spiral`` and ``multi_mask`` raise (``equispaced`` raises only
     at the strict ``1.0``);
     ``variable_density``, ``variable_density_cava``,
     ``fractional_variable_density``, ``random_cartesian`` and ``nested`` pass
     through byte-identical because they already nest.
   * It applies to the **fixed-seed cascade only**. ``enable_dynamic_mask``
     deliberately varies the pattern per sample at training time, where each
     sample sees exactly one ``t`` and nesting is irrelevant; that path is left
     unenforced. Nesting has to hold along the fixed-seed path the reverse
     trajectory and validation walk, which is what this flag covers.

   Default ``false``, so every existing run is byte-identical until it opts in.

.. note::

   **Default hygiene: a disabled block must not carry sub-flags that default ON.**
   When a block's own gate is ``enabled: false``, a sub-flag defaulting ``true``
   misdescribes the run in the resolved config, and enabling the parent silently
   buys every such flag. This holds for every schema class.

The full list of accepted top-level keys lives on
:class:`spectramr.config.settings.TrainingSettings`. See
``src/spectramr/config/schemas/templates/v1.0_reference.yaml`` — the single
canonical, round-trip-tested reference template, with inline
``# options:`` comments for every constrained field. It is the only reference
template: two references would be two SSOTs.

.. note::

   **Top-level** ``model_domain`` is an *optional* convenience knob, not the
   field consumers read. It defaults to ``None`` ("unspecified — defer to the
   nested value") and, when supplied, propagates into ``model.model_domain`` /
   ``model.target_domain`` (the fields the strategies actually consume) and
   **raises** on a genuine conflict with an explicit nested value. Prefer
   setting only ``model.model_domain`` (or ``model.target_domain``). The default is
   ``None`` ("unspecified"), which keeps ``--override`` round-trips idempotent.

.. note::

   The experiment name lives **under** ``logging:``, not at the top level.
   ``TrainingSettings`` is ``extra="forbid"``, so a top-level
   ``experiment_name`` makes the YAML unloadable. Its path is
   ``logging.identity.experiment``.

Python access:

.. code-block:: python

   from spectramr.config.settings import TrainingSettings

   cfg = TrainingSettings.from_yaml("experiments/<paradigm>/<your-arm>.yaml")

   # ✅ CORRECT: nested access, at the depth the schema actually has
   cfg.training.training_mode
   cfg.optimization.optimizer.learning_rate
   cfg.data.loader.batch_size
   cfg.losses.image_losses          # list[LossComponentConfig]

   # ❌ FORBIDDEN: flat aliases, never on the schema
   cfg.lr                           # AttributeError
   cfg.lambda_l1                    # AttributeError


---

``data:`` — DataConfigSchema
==============================

``data:`` is decomposed into named sub-blocks (phases 9a-9g).  The order below
is the order a reader needs the answers in: *where is the data, what pairs with
what, how are samples drawn, how are they loaded, what is done to the values,
what representation comes out.*

.. code-block:: yaml

   data:
     # `dataset_type` is deliberately NOT inside `source:` -- it is a DISPATCH
     # key, not a location, and is bound to `workflow.regime` by the plan.
     dataset_type: m4raw

     source:                     # WHERE the bytes come from
       root: /path/to/databases
       layout: flat              # flat | bids
       index_path: null          # enumerates the TRAIN split
       validation_index_path: null   # null = carve from index_path
       paired_manifest_path: null    # v4 paired JSON (ULF/HF)
       preprocessing_dir: null       # a *_image/ preprocessing output tree

     pairing:                    # what counts as INPUT and what as TARGET
       contrasts: null           # input-side filter; null = all available
       sessions: null
       target_contrasts: null    # null = mirror the input counterpart
       target_sessions: null
       single_contrast: false    # true = no cross-contrast pairing at all
       bidirectional_mode: ulf_to_hf   # <input>_to_<target>
       hf_resolution: null       # highres | lowres | unknown
       allow_unpaired: false     # admit records with no partner

     split:                      # how the corpus is divided
       type: auto
       validation_fraction: 0.1
       holdout_site: null
       train_sites: null
       holdout_subject: null
       loso_fold: null
       max_train_subjects: null
       max_val_subjects: null

     sampling:                   # how samples are DRAWN from a volume
       patch_size: [320, 320, 1]
       samples_per_volume: 8
       queue_length: 200
       enable_slab_mode: false
       enable_slice_2d: false
       num_synthetic_samples: 10

     loader:                     # how they reach the GPU
       batch_size: 4
       num_workers: 4
       persistent_workers: false
       prefetch_factor: 2
       pin_memory: true

     processing:                 # what is done to the VALUES
       enable_kspace_normalization: false
       kspace_percentile: 0.99   # a FRACTION, not a percent
       kspace_scale_domain: kspace
       enable_log_scaling: false
       log_scaling_center_fraction: 0.25
       normalization_type: none
       normalization_kwargs: {}
       enable_image_normalization: false
       enable_image_rescale: false
       rescale_range: [-1.0, 1.0]
       rescale_percentiles: [0.0, 100.0]
       data_range: null
       transforms: []

     domain:                     # what representation comes out
       output: image             # image | kspace | complex_image | ...
       target_channels: 1
       input_artifact: null
       target_artifact: null
       graph_type: null
       enable_graph_encoding: false

     coils:                      # coil combination
       processing_mode: none     # rss | flatten | sense | svd | none
       num_virtual_coils: 4
       svd_calibration_lines: null

     expose:                     # opt-in extra batch keys (all default false)
       acquisition_params: false
       scanner_id: false
       site_id: false
       field_strength: false
       field_strength_target: true    # the one expose_* that defaults ON
       conformal_jacobian: false
       cortex_flatten_grid: false
       glm_design_matrix: false

Naming inside ``pairing:`` is deliberately un-prefixed.  ``contrasts`` is **not**
renamed to ``input_contrasts``, because singular-vs-plural would be the only
thing distinguishing it from the adjacent ``data.input_contrast`` normalization
block.  The block prefix already carries that meaning.

**Dataset types** (``dataset_type``) — the complete accepted set, from the
field validator:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Value
     - Description
   * - ``kspace``
     - Raw k-space H5 (FastMRI-style single/multi-coil)
   * - ``m4raw``
     - M4Raw 0.3T multi-contrast (repetition-aware)
   * - ``nifti``
     - NIfTI volumes (3D or 2D+time)
   * - ``nifti_paired``
     - Paired NIfTI (input + target directories)
   * - ``contrast_aware_paired``
     - Paired records carrying per-contrast metadata. **Also requires**
       ``data.input_contrast`` and ``data.target_contrast`` (each at least a
       ``name:``) — these are ``ContrastConfigSchema`` normalization specs,
       not the ``pairing.contrasts`` filter.
   * - ``npy_slice``
     - Pre-extracted ``.npy`` slices
   * - ``image``
     - Generic 2D image directory
   * - ``dicom``
     - Clinical DICOM series
   * - ``synthetic``
     - Physics-simulated synthetic MRI
   * - ``graph_mri``
     - Graph-encoded MRI (see ``domain.graph_type``)
   * - ``preprocessed``
     - A ``*_image/`` preprocessing output tree
   * - ``pde_synthetic``
     - PDE-generated fields (PINN arms)
   * - ``quantitative``
     - Quantitative maps (T1/T2/PD)
   * - ``cine``
     - Cardiac cine (2D+t)
   * - ``bart_kspace``
     - BART-format k-space
   * - ``bids_paired``
     - BIDS-layout paired low/high field
   * - ``png_paired``
     - Paired PNG (input + target)
   * - ``field_ref``
     - Field-strength reference volumes
   * - ``ismrmrd_kspace``
     - ISMRMRD-format raw k-space
   * - ``oracle_bssfp``
     - Oracle bSSFP simulation
   * - ``mrixfields``
     - MRIxFields multi-field cohort

**Coil channel arithmetic:**

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - ``data.coils.processing_mode``
     - ``in_channels``
     - Notes
   * - ``rss``
     - 1
     - Root-sum-of-squares → magnitude
   * - ``flatten``
     - ``2 × num_coils``
     - Real + imaginary per coil
   * - ``sense``
     - 2
     - SENSE-combined complex (real + imag)
   * - ``none``
     - ``num_coils``
     - Raw coil images


---

``model:`` — ModelConfigSchema
================================

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Key
     - Default
     - Description
   * - ``model_type``
     - ``"standard_unet"``
     - Generator registry key (see :doc:`models_reference`)
   * - ``in_channels``
     - ``1``
     - Must match ``data.in_channels``
   * - ``out_channels``
     - ``1``
     - Must match ``data.out_channels``
   * - ``discriminator_type``
     - ``"patch_gan"``
     - Discriminator registry key (GAN mode)
   * - ``model_kwargs``
     - ``{}``
     - Extra kwargs to ``ModelFactory.create_generator()``
   * - ``pretrained_encoder``
     - ``null``
     - Path to encoder checkpoint
   * - ``freeze_encoder``
     - ``false``
     - Freeze encoder weights during training

Example:

.. code-block:: yaml

   model:
     model_type: kspace_cold_diffusion_generator
     in_channels: 2
     out_channels: 2
     model_kwargs:
       num_timesteps: 1000
       rician_noise_std: 0.05


---

``optimization:`` — OptimizationConfigSchema
==============================================

Five named sub-blocks, plus the scheduler keys that are still flat:

.. code-block:: yaml

   optimization:
     optimizer:
       type: adamw
       learning_rate: 2.0e-4
       weight_decay: 1.0e-4
       betas: [0.9, 0.999]
     gradient:
       accumulation_steps: 2
       enable_checkpointing: false
       clip: {enabled: true, method: norm, value: 1.0}
     precision:
       enabled: true
       dtype: bfloat16
     compile:
       enabled: false
     memory:
       enable_monitoring: false

     # still flat -- see the scheduler note below
     lr_scheduler_strategy: cosine
     scheduler: {T_max: 50000, eta_min: 1.0e-6, warmup_steps: 500}

``optimizer:``

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Key
     - Default
     - Description
   * - ``type``
     - ``"adamw"``
     - Closed vocabulary (``OptimizerType``); an unknown name raises at load.
   * - ``learning_rate``
     - ``1e-5``
     - Base LR. Multiplied per role for GAN TTUR.
   * - ``generator_learning_rate`` / ``discriminator_learning_rate``
     - ``None``
     - Explicit per-role LR; overrides the base × multiplier.
   * - ``weight_decay``
     - ``1e-4``
     - Forwarded only to optimizers whose signature accepts it.
   * - ``beta1`` / ``beta2`` / ``betas``
     - ``0.5`` / ``0.999`` / ``None``
     - ``betas`` wins; declaring both with different values raises.
   * - ``eps``, ``momentum``, ``nesterov``, ``amsgrad``
     - see schema
     - Dropped silently if left at default and unaccepted; **raises** if declared.
   * - ``kwargs``
     - ``{}``
     - Escape hatch, validated against the optimizer's signature.
   * - ``param_groups``
     - ``None``
     - Per-prefix overrides. A key matching zero parameters raises.
   * - ``lookahead``
     - ``{enabled: false}``
     - Wrapper applied after the base optimizer is built.

``gradient:`` — ``accumulation_steps`` (1), ``enable_checkpointing`` (false),
``detect_anomalies`` (false), and ``clip: {enabled, method, value}``.

``precision:`` — ``enabled`` (false) and ``dtype``. Note the third state:
``dtype: float32`` disables AMP even when ``enabled: true``.

``compile:`` — ``enabled`` (false), ``mode``, ``backend``, ``fullgraph``,
``dynamic``. Compilation failure raises rather than falling back to eager.

``memory:`` — ``enable_monitoring``, ``monitoring_interval``,
``enable_fragmentation_mitigation``, ``cleanup_interval``,
``enable_batch_size_optimization``, ``safety_margin``. Diagnostics only.

.. admonition:: A declared scheduler family is honoured with or without a dict
   :class: note

   ``optimization.lr_scheduler_strategy: cosine`` resolves on its own, with the
   period defaulting to ``training.max_iterations``. A ``scheduler:`` dict is
   only needed to *parameterise* the family.

   The field carries ``default="cosine"``, so every resolved config presents a
   family name. Only a name you actually declared is honoured — declare neither
   key and the run resolves to no scheduler.

.. admonition:: A warmup-bearing name must declare a warmup length
   :class: warning

   ``warmup_cosine``, ``warmup`` and ``linear_warmup`` name two things: a decay
   family and a warmup. The alias table resolves only the decay half — the
   warmup comes from the wrapper that ``warmup_steps`` selects — so one of
   these names without a warmup length **raises**. Declare
   ``optimization.warmup_steps`` (or ``scheduler.warmup_steps``) above zero,
   or name the decay family (``cosine``) directly.

.. admonition:: AMP + NaN Gradients
   :class: warning

   On V100 GPUs, ``float16`` AMP can produce NaN gradients in complex
   loss compositions. Use ``precision.dtype: bfloat16`` or
   ``precision.enabled: false`` if training is unstable. K-space losses are
   most susceptible.


---

``training:`` — TrainingStrategyConfigSchema
=============================================

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Key
     - Default
     - Description
   * - ``training_mode``
     - ``"reconstruction"``
     - Strategy dispatch key (see :doc:`strategies_reference`)
   * - ``seed``
     - ``42``
     - Global random seed
   * - ``max_iterations``
     - ``100000``
     - Total training iterations. **Three different mechanisms can produce this
       number**, so the launch banner names which one did:
       ``[Pipeline] Starting training for N iterations (budget source: …)``.
       The sources are this key as the config file declares it, an
       ``--override``/``-O training.max_iterations=…`` on the command line, or
       -- when the key is absent or non-positive -- ``training.epochs`` x the
       train-loader length, derived at runtime. Sanity-check mode is a fourth:
       it *forces* 5000 **after** overrides are applied, so it reports itself
       along with the budget it replaced (``sanity-check mode (forced;
       overrides the 30000 from …)``). Without that attribution a log showing
       ``5000`` could not distinguish an operator's ``-O …=5000`` from the
       mode's hardcoded 5000. The banner deliberately does **not** claim the operator
       typed the override: ``main.py`` injects overrides of its own and the
       smoke dispatcher injects ``training.max_iterations=<cap>``, all by the
       same route. Its claim is the one that matters -- *this value did not
       come from the config file*.
   * - ``iteration_budget_scope``
     - ``"per_rank"``
     - How ``max_iterations`` is read under ``world_size > 1``. ``per_rank``
       (the default) means **every rank runs the
       full count** -- data parallelism buys effective batch, not a shorter
       run, so a 4-GPU launch costs ~4x the GPU-hours for ~1x the wall-clock.
       ``global`` would divide the bound by ``world_size`` and currently
       **raises**: nothing shards the stream, and dividing the
       bound silently reshapes every iteration-keyed schedule (the diffusion
       curriculum, the EMA horizon, the validation cadence).
   * - ``log_interval``
     - ``50``
     - Log every N steps
   * - ``save_images_interval``
     - ``1000``
     - Save validation images every N steps

**Paradigm sub-schemas** (nested under ``training:``):

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Sub-key
     - Schema file
   * - ``training.diffusion``
     - ``training/diffusion.py``
   * - ``training.gan``
     - ``training/gan.py``
   * - ``training.vae``
     - ``training/vae.py``
   * - ``training.reconstruction``
     - ``training/reconstruction.py``
   * - ``training.ssl``
     - ``training/ssl.py``
   * - ``training.meta_learning``
     - ``training/meta_learning.py``
   * - ``training.tto``
     - ``training/tto.py``
   * - ``training.geomamba_ulf``
     - ``training/geomamba_ulf.py``


---

``losses:`` — LossConfigSchema
================================

Replaces all legacy ``objectives:`` + ``lambda_*`` flat keys.

.. code-block:: yaml

   losses:
     policy:                       # how the objective is ASSEMBLED, not which terms
       output_domain: image        # image | complex_image | kspace | latent
       exclude_defaults: []        # e.g. ['mse'] drops the paradigm's implicit MSE
     image_losses:
       - name: l1
         weight: 10.0
         enabled: true
     kspace_losses:
       - name: data_consistency
         weight: 1.0
         enabled: true
     complex_losses:
       - name: complex_l1
         weight: 0.5
         enabled: true

**Domain routing:**

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - ``output_domain``
     - Active Lists
     - Notes
   * - ``image``
     - ``image_losses``
     - Direct image-space
   * - ``complex_image``
     - ``image_losses`` + ``complex_losses``
     - FFT bridge inserted
   * - ``kspace``
     - ``kspace_losses`` + ``complex_losses``
     - Image losses skipped
   * - ``latent``
     - ``latent_losses``
     - Post-encoder losses. **No bridge exists into a latent** — the encoder that
       would produce one is a learned map, not a transform — so this row is the
       only one whose Active Lists column is exclusive. Declaring
       ``latent_losses`` alongside a kspace/image output raises.

**LossComponentConfig fields:**

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - Field
     - Default
     - Description
   * - ``name``
     - *required*
     - Registry key (see :doc:`losses_reference`)
   * - ``weight``
     - ``1.0``
     - Loss weight λ
   * - ``enabled``
     - ``true``
     - Set ``false`` to skip without removing
   * - ``kwargs``
     - ``{}``
     - Extra kwargs to loss constructor

.. admonition:: Common Misconfiguration
   :class: warning

   ``output_domain: image`` + only ``kspace_losses`` = **all losses silently skipped**.


.. note::

   ``output_domain`` and ``disable_default_losses`` live in ``policy:`` — read
   ``config.losses.policy.output_domain``.

   ``disable_default_losses`` becomes ``exclude_defaults`` rather than being
   inverted to ``enable_default_losses``. The naming rule forbids a negated
   boolean, but this field is a ``list[str]`` of loss **names** — inverting it is
   meaningless, since ``enable_default_losses: ['mse']`` would read as "enable
   ONLY mse", the opposite of a filter. ``exclude_`` states the sense with
   nothing to invert.

---

``physics:`` — PhysicsConfigSchema
=====================================

.. list-table::
   :header-rows: 1
   :widths: 38 15 47

   * - Key
     - Default
     - Description
   * - ``data_consistency.enabled``
     - ``false``
     - Enable k-space data consistency
   * - ``data_consistency.method``
     - ``"projection_2d_consistency"``
     - Which DC layer to build. The schema default is **not** one of the names
       the generator accepts -- declare ``hard`` (projection) or one of the
       soft families explicitly.
   * - ``data_consistency.weight``
     - ``1.0``
     - Soft-DC trust parameter only (``lambda_init`` / ``beta`` / ``hf_lambda``
       depending on family). **Inert under** ``method: hard``, whose blend is
       weight 1.0 by construction.
   * - ``data_consistency.train_noise_level``
     - ``0.01``
     - Noise added to measured k-space during training. Read by ``hard`` and
       the ``SimpleDataConsistency`` fallback; inert under the soft families.
   * - ``data_consistency.eval_noise_level``
     - ``0.005``
     - Same, at inference. Same readership as ``train_noise_level``.
   * - ``data_consistency.noise_type``
     - ``"gaussian"``
     - Only ``gaussian`` is implemented; anything else raises at construction
       rather than degrading silently.
   * - ``data_consistency.apply_at_predict``
     - ``false``
     - Hard projection onto the measured k-space as the last step of
       ``infer``. Read by the inference strategies only, so independent of
       ``enabled`` and ``method``. Needs the ``mask`` dataset of an HDF5 input
       and a k-space input route; raises when either is missing. See
       :doc:`running_pipelines`.
   * - ``data_consistency.acs_mask_center_fraction``
     - ``0.08``
     - ACS center fraction
   * - ``data_consistency.enable_acs_replacement``
     - ``true``
     - Hard-replace ACS lines at validation
   * - ``kspace.enforce_hermitian_symmetry``
     - ``true``
     - Conjugate symmetry for real output
   * - ``b0_range_hz``
     - ``200.0``
     - ±B0 range in Hz
   * - ``b1_min``
     - ``0.5``
     - Minimum B1 scale
   * - ``b1_max``
     - ``1.5``
     - Maximum B1 scale

.. important::

   ``data_consistency.enabled: true`` is REQUIRED for all reconstruction
   experiments to prevent identity-mapping collapse.

.. note::

   **DC configuration SSOT (single source of truth).**
   ``physics.data_consistency`` is the only authoritative location for DC
   behaviour. Legacy YAML keys ``model.model_kwargs.dc_method`` and
   ``model.model_kwargs.dc_weight`` are still tolerated, but the
   ``ModelBuilder`` reconciles them against ``physics.data_consistency``
   on every build and raises :class:`ValueError` on disagreement
   (see :mod:`spectramr.infrastructure.training.builders.model_builder`,
   ``_reconcile`` helper). Generators that consume ``dc_method`` via
   ``**kwargs`` (e.g. :class:`~spectramr.models.generators.kspace_cold_diffusion_generator.KSpaceColdDiffusionGenerator`)
   are also reconciled — the contract inspector adds a ``"**kwargs"``
   sentinel for generators that accept arbitrary keyword arguments.


---

``checkpoint:`` — CheckpointConfigSchema
==========================================

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Key
     - Default
     - Description
   * - ``enabled``
     - ``true``
     - Enable checkpoint saving
   * - ``checkpoint_dir``
     - ``"./checkpoints"``
     - Save directory
   * - ``save_interval``
     - ``1000``
     - Save every N steps
   * - ``keep_last_n``
     - ``5``
     - Retain last N checkpoints
   * - ``keep_best_n``
     - ``3``
     - Retain best N by metric
   * - ``best_metric_name``
     - ``"val_psnr"``
     - Metric for best-checkpoint selection
   * - ``best_metric_mode``
     - ``"max"``
     - ``max`` (PSNR/SSIM) or ``min`` (LPIPS/MSE)
   * - ``pretrained_path``
     - ``null``
     - Load weights at start
   * - ``resume_training``
     - ``false``
     - Resume iteration counter
   * - ``format``
     - ``"safetensors"``
     - ``safetensors`` (preferred) or ``pth``
   * - ``produced_by_arm``
     - ``null``
     - Name of the upstream campaign arm that builds this arm's declared
       checkpoint. When set, the ``checkpoint_existence`` audit defers a
       still-absent, campaign-artefact-rooted checkpoint (info-pass under
       ``--strict``) instead of hard-failing — see
       :doc:`audit_ladder_user_guide`.

.. tip::

   For perceptual quality, use ``best_metric_name: val_lpips`` +
   ``best_metric_mode: min``.

.. note::

   **The whole** ``checkpoint:`` **block is optional.** A config that omits it
   receives a default ``CheckpointConfigSchema`` (every sub-field above is
   defaulted), so :func:`spectramr.bootstrap.build_container` builds a working
   checkpoint service rather than aborting.
   Declare the block only to override the defaults (e.g. a per-arm
   ``checkpoint_dir`` or ``best_metric_name``).


---

``validation:`` — ValidationConfigSchema
==========================================

Seven sub-blocks, in the order a reader asks: *how often* (``schedule``), *on
what* (``loader``), *measuring what* (``metrics``), *failing on what*
(``gates``), *at which severities* (``cascade``), plus ``visualization`` and
diffusion-only ``sampling``. Every value below is the live schema default.

.. code-block:: yaml

   validation:
     enabled: true                       # see the inert-knob warning below
     split: 0.2                          # see the inert-knob warning below

     schedule:
       interval_steps: 1000              # validate every N steps
       on_epoch: true                    # also validate at each epoch end
       interval_epochs: 1                # ...every N epochs, in epoch mode

     loader:
       batch_size: null                  # null inherits the training batch size
       chunk_size: 2                     # micro-batch for chunked val inference
       num_batches: null                 # null = all batches
       num_samples: null                 # null = all samples
       shuffle: false

     scoring:                            # named `scoring`, not `metrics` -- see the note
       compute: null                     # e.g. [psnr, ssim]; null inherits training.metrics
       primary: psnr                     # early stopping / model selection
       domain: null                      # 'image' | 'kspace' | null to auto-detect
       output_transform: null            # 'ifft_magnitude' | 'ifft_mag_combine' | 'fft'
       enable_image_metrics: true

     visualization:
       enabled: false
       interval: 1000

     sampling:                           # diffusion paradigms only
       steps: null                       # null falls back to training.diffusion.sampling_steps
       enable_multistep_cold: false

     cascade:                            # diffusion paradigms only
       levels: null                      # null = framework default (2, 8, 32)

     gates:                              # checks that can FAIL a run, not just measure it
       input_dependence_tol: null        # L4 measurement-independence (DC-blob) gate
       held_out_severity_eval: false
       hallucination_test:
         enabled: false
         method: feature_insertion       # | lesion_swap | edge_jitter
         n_features: 5
         interval_validations: 4

     empty_cache_before_validation: true

``validation.cascade.levels`` is the acceleration ladder the cascading
validation sweep evaluates — one pass per rung, each written as its own row of
``validation_metrics.csv`` (``acceleration_level`` / ``timestep`` as values)
and as flat ``val_<metric>_<R>x`` columns. Leave it ``null`` for the default
ladder.

Levels are deduplicated and sorted ascending, and an integral rung stays an
``int`` so ``val_psnr_2x`` does not become ``val_psnr_2.0x`` — the L4 gate and
the accel-gap stamp look those names up and do not raise on a miss. An empty
ladder, a rung below 1x, a non-finite value and a boolean are all **refused at
load time**. Under ``undersampling.schedule_type: step`` a rung outside
``undersampling.acceleration_range`` has no timestep inverse and is skipped at
runtime; ``spectramr audit`` warns before the launch.

.. warning::

   ``schedule.interval_steps`` must not exceed ``training.max_iterations``.

   The loop gate is a bare ``iteration % interval_steps == 0``. Unlike its two
   sibling intervals (``logging.intervals.log`` and
   ``metrics.train_metric_interval``), it has **no** unconditional
   first/last-iteration force, so an interval above the budget produces *zero*
   validation events: early stopping never evaluates, no ``checkpoint_best.pt``
   is written, and the run still exits reporting success.
   ``_execute_training_loop`` therefore rejects that combination at startup with
   a ``ConfigurationError``, unless ``on_epoch`` supplies events instead.

   ``interval_steps == max_iterations`` is legal but degenerate — validation
   runs exactly once, on the final iteration — and logs a warning: early
   stopping cannot act on a single event, and any validation-time failure costs
   the whole budget before it is seen.

   This is not a Pydantic validator because the budget lives in another block
   and can be derived from ``training.epochs`` × loader length at runtime. Watch
   it whenever you shorten a run with ``-O training.max_iterations=…``: the
   override moves the budget while the interval stays put. Sanity-check mode is
   the one exemption — it *forces* ``max_iterations`` to 5000 after overrides are
   applied, so it warns instead of raising rather than veto a budget the arm
   never chose. That exemption is about the *mode's* budget only: if the
   interval also exceeds the budget the arm **declared**, sanity mode still says
   so and names it a defect, because that arm is fatal on a full-length run and
   an all-clear there would waste the early warning.

.. note::

   Read the canonical paths: ``config.validation.schedule.interval_steps`` for
   the cadence and ``config.validation.loader.batch_size`` for the batch size.
   The metric block is named ``scoring``.

.. note::

   ``empty_cache_before_validation`` is the one live ungrouped scalar. It calls
   ``torch.cuda.empty_cache()`` before each validation pass to free the training
   allocator pool (training usually holds most of VRAM; the EMA weight-swap
   transiently doubles parameter memory). Default ``true`` preserves the OOM-safe
   behavior; set ``false`` on memory-headroom runs to avoid the allocator re-grow
   cost on the next train step.


---

``logging:`` — LoggingConfigSchema
====================================

Seven sub-blocks: *what the run is called* (``identity``), *where lines go*
(``sinks``), *how often* (``intervals``), plus ``images``, ``tracking``,
``snapshots`` and ``report_cases``. Every value below is the live schema default.

.. code-block:: yaml

   logging:
     identity:
       experiment: default_experiment
       run: null                       # null derives a run name
       notes: null
       tags: {}

     sinks:                            # where lines go, and how much of them
       level: info                     # debug | info | warning | error | critical
       silent: false
       to_console: true
       to_file: true
       dir: ./logs

     intervals:                        # every-N-steps cadences
       log: 100
       save: 100
       validation_images: 1            # every N validations
       anomaly_check: 100

     images:
       log_input: false
       log_validation: true
       log_difference: true            # |prediction - target|
       save_validation: true           # to disk, as opposed to the tracker
       max_per_batch: 4

     tracking:
       enabled: true
       service: tensorboard            # tensorboard | wandb
       enable_tensorboard: true
       tensorboard_dir: null

     snapshots:                        # per-step debug tensor/JSON dumps
       enabled: true
       interval_steps: 0               # 0 = every step, bounded by max_calls
       max_calls: 8                    # CALL budget per (run, tag) -- not a step bound
       save_images: true
       save_json: true
       log_steps: [0, 1, 2, 10]        # forces the diffusion anomaly LOG, not snapshots

     report_cases:
       enabled: true
       subdir: report_cases

     # wandb_project / wandb_entity: a non-null value RAISES; `null` (shown
     # here) is the only accepted declaration of either.
     wandb_project: null
     wandb_entity: null
     log_gradients: false

.. note::

   Read the canonical path ``config.logging.intervals.log``.

   The debug block is named ``snapshots``.

.. warning::

   **There is no Weights & Biases integration.** ``wandb_project`` and
   ``wandb_entity`` are the only fields that could configure one, and a
   non-null value **raises** — leave them unset or explicitly ``null``.
   Runs log to TensorBoard.

   ``logging:`` is ``extra="ignore"``, so a key it does not declare is
   **silently discarded** rather than rejected. Check a logging key against
   the block above before relying on it: ``sinks.level``, ``sinks.to_console``,
   ``sinks.to_file`` and ``intervals.log`` are the real spellings of the four
   most commonly guessed wrong.


---

``ema:`` — EMAConfigSchema
============================

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Key
     - Default
     - Description
   * - ``enabled``
     - ``false``
     - Enable EMA of generator weights
   * - ``decay``
     - ``0.999``
     - EMA decay coefficient
   * - ``update_frequency``
     - ``1``
     - Update EMA every N steps
   * - ``warmup_steps``
     - ``0``
     - Steps before EMA starts
   * - ``enable_adaptive_ema``
     - ``false``
     - Ramp from ``initial_decay`` → ``final_decay``
   * - ``initial_decay``
     - ``0.0``
     - Starting decay for adaptive EMA
   * - ``final_decay``
     - ``0.999``
     - Final decay

**Recommended for diffusion:**

.. code-block:: yaml

   ema:
     enabled: true
     decay: 0.9999
     warmup_steps: 5000
     enable_adaptive_ema: true
     initial_decay: 0.99
     final_decay: 0.9999


---

``early_stopping:`` — EarlyStoppingConfigSchema
=================================================

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Key
     - Default
     - Description
   * - ``enabled``
     - ``false``
     - Enable early stopping
   * - ``metric``
     - ``"val_psnr"``
     - Metric name to monitor
   * - ``mode``
     - ``"max"``
     - ``max`` or ``min``
   * - ``patience``
     - ``10``
     - Checks without improvement before stop
   * - ``patience_min_iterations``
     - ``5000``
     - Don't stop before this iteration count
   * - ``min_delta``
     - ``1e-4``
     - Minimum improvement threshold


---

Enum Catalog
============

All enums from ``src/spectramr/config/schemas/enums.py``:

**TrainingMode values** → see :ref:`strategies_reference` for strategy dispatch.

.. list-table:: Optimizer Types
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Notes
   * - ``adam``
     - Standard Adam
   * - ``adamw``
     - **Recommended** — decoupled weight decay
   * - ``sgd``
     - Use with ``lr_scheduler: cyclic``
   * - ``rmsprop``
     - Useful for RNN-based decoders

.. list-table:: LR Schedulers
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Behaviour
   * - ``cosine``
     - Cosine annealing — recommended for diffusion
   * - ``linear``
     - Linear decay to 0
   * - ``cyclic``
     - Cyclic LR (Smith 2015)
   * - ``constant``
     - No decay
   * - ``cold_mri``
     - MRI-specific curriculum schedule

.. list-table:: Noise Schedules
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``linear``
     - β linearly spaced
   * - ``cosine``
     - Cosine — avoids collapse at t→T
   * - ``cold_mri``
     - Deterministic undersampling-based

.. list-table:: Prediction Types
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Loss Target
   * - ``noise``
     - Predict added noise ε (DDPM)
   * - ``sample``
     - Predict clean x₀ (cold diffusion)
   * - ``velocity``
     - Predict velocity v (rectified flow)


---

Config Validation & Health Checks
====================================

Dry-run before allocating GPU:

.. code-block:: bash

   spectramr train --config experiments/<paradigm>/<your-arm>.yaml --dry-run

This resolves your config and runs the health checks without allocating a GPU or
reading the dataset. The checks cover:

- **Required sections are present.**
- **The model resolves.** ``model.model_type`` names a registered model, and
  that model can actually be constructed from the ``model_kwargs`` you gave it.
- **The strategy resolves.** ``training.training_mode`` names a registered
  training strategy.
- **Domains line up.** ``model.in_channels`` is consistent with
  ``data.coils.processing_mode`` and the dataset type, and
  ``losses.policy.output_domain`` matches the loss lists you populated.
- **Loss weights are sane** — no all-zero weighting, no weight on a loss list
  that is empty.
- **Physics is coherent.** In reconstruction modes, this is where an arm that
  never enforces ``physics.data_consistency.enabled`` is flagged.

Warnings are errors here: ``audit`` is ``--strict`` by default and a warning
exits non-zero. Fix the warning rather than suppressing it — see
:doc:`audit_ladder_user_guide`.

**CLI overrides** (without editing YAML):

.. code-block:: bash

   spectramr train --config experiments/<paradigm>/<your-arm>.yaml \
       --override "optimization.optimizer.learning_rate=5e-5" \
       --override "data.batch_size=16"

An override takes the same dotted path the key has in YAML, and is applied
before validation — so an override that produces an invalid config fails the
same way an invalid YAML file would, rather than at first use.


---

References
==========

1. Pydantic V2 Documentation — https://docs.pydantic.dev/latest/


.. _naming-convention:

Naming convention (enforced)
----------------------------

A reader cannot skim a config whose keys follow no rule. These ratify the
plurality already present in the schema rather than imposing a new style.

=====================  ========================================================
Kind                   Rule
=====================  ========================================================
Boolean switch         ``enable_<thing>``
A block's own gate     bare ``enabled`` — never a feature flag
Count                  ``num_<thing>``
Loss weight            ``lambda_<term>``
Registry selector      ``<thing>_type``
Fraction in [0, 1]     ``<thing>_fraction``
Path                   ``_path`` file, ``_dir`` directory, ``_root`` tree root
Negation               forbidden — invert the sense instead
=====================  ========================================================

Two of these cannot be decided from a name alone, so follow them by hand: a
registry selector is not always distinguishable from a genuine mode
(``bidirectional_mode`` really is a mode), and whether a path names a file, a
directory or a tree root depends on what it points at.

One acquisition tuple, one definition
=====================================

An acquisition tuple — ``{name, TE, TR, TI, FA, B0, contrast_type,
include_concomitant}`` — is declared **once**, as
:class:`spectramr.config.schemas.data.AcquisitionParamsSchema`.
``spectramr.config.schemas.training.pmps.AcquisitionParam`` is an alias of it, kept so
``fixed_protocols`` and the module's ``__all__`` are unchanged for callers.

.. note::

   Ten acquisition-named constructs exist across the codebase, five of them config
   schemas (``AcquisitionConfigSchema`` is acquisition *design* — trajectory codesign
   and active acquisition — and is unrelated to this tuple;
   ``AcquisitionMetadataConfigSchema`` configures the metadata *loader*;
   ``AcquisitionParamsConfig`` in ``physics.py`` is a tighter 4-field SPGR descriptor
   using ``tr_ms``/``field_strength_t`` rather than ``TR``/``B0``). Before adding an
   eleventh, check whether one of these is the construct you want.
