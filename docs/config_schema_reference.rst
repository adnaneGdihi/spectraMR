.. _config_schema_reference:

=================================================
Configuration Schema Reference — v1.0
=================================================

.. sectionauthor:: spectraMR Research

Every training run is driven by a single YAML file loaded once via
``TrainingSettings.from_yaml(path)`` and validated immediately. After
loading the object is **frozen** (Pydantic ``frozen=True``) — no
downstream code may mutate it.

.. note::

   **Why ``1.0`` comes after ``6.x``.** The number restarts at ``1.0``
   deliberately. The 5.x/6.x lineage numbered a schema that no longer exists —
   the 15-phase block decomposition rebuilt the layout underneath those names,
   and nothing ever branched on the version anyway: it is validated, bound onto
   ``run.config_version``, and read by no consumer. A version that gates nothing
   but keeps incrementing is a changelog pretending to be a contract. ``1.0``
   names the schema as it now is, once.

   ``6.0`` and ``6.1`` were briefly *accepted at the door and folded* to ``1.0``
   so the corpus could migrate gradually. That fold was removed on 2026-08-08:
   ``LEGACY_CONFIG_VERSIONS`` is now an empty frozenset and a legacy declaration
   **raises**. If you are holding an older file, migrate the *keys* first, then
   set the version:

   .. code-block:: bash

      python scripts/ci/migrate_config_keys.py <file> --apply --all-versions
      # then edit the file: config_version: '1.0'

   ``--all-versions`` is not optional here: without it the key migrator skips
   any file the loader rejects — which is every file that still declares a
   legacy version — and reports "no retired keys found" while exiting 0.

   Do the two in that order. The key migration is what makes the file loadable;
   the version line on its own only changes which error you get.

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

   # ---- deprecated (accepted to keep legacy YAMLs loading) ----
   diffusion:      { ... }      # legacy top-level — set training.diffusion instead
   artifacts:      { ... }      # legacy top-level — set training.output_dir instead

.. note::

   **``acceleration:`` is now ``undersampling:``** (phase 11). The old name meant
   two unrelated things — the MRI k-space *acceleration factor* (what the block's
   26 real fields configure) and *compute* acceleration — so a reader could not
   tell which sense a key belonged to. The legacy spelling still loads: a ROOT
   fold moves the whole block before any sub-model is built. It is gone from
   Python, so read ``config.undersampling``.

   Five compute knobs still sit in the block and are all **inert** —
   ``mixed_precision``, ``use_compile``, ``use_gradient_checkpointing``,
   ``gradient_accumulation_steps`` (each duplicating a live ``optimization.*``
   field) and ``use_distributed`` (which has no equivalent — ``parallel:`` has no
   boolean gate). They stay flat so their inertness stays visible; issue #680.

   **``enforce_nested`` (added 2026-08).** Cold diffusion's forward process is
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
     lines, so no realised fraction can ever equal a continuous target and the
     guard used to fire on sub-line rounding -- which made ``nested_tolerance:
     1.0`` unsatisfiable even for families that nest perfectly. Against the raw
     draw, ``1.0`` is the meaningful strict setting: *enforcement must be a
     no-op*. Whether a family's raw draw honours its declared ``R`` is a
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
   buys every such flag. ``tests/unit/config/test_default_hygiene.py`` enforces
   this for every schema class, walked live.

   Five existing fields are recorded as ratcheted exceptions rather than flipped,
   because flipping them is a behaviour change and not a tidy-up:
   ``DataConsistencyConfig.enable_acs_replacement`` (594 arms enable the parent
   without it), ``DigitalTwinConfig.enable_motion`` / ``enable_b0`` /
   ``enable_b1`` (9 arms), and ``FSDPConfigSchema.use_orig_params`` (``True`` is
   torch's own recommended FSDP setting). Shrink that list; never grow it.

The full list of accepted top-level keys lives on
:class:`spectramr.config.settings.TrainingSettings`. See
``src/spectramr/config/schemas/templates/v1.0_reference.yaml`` — the single
canonical, round-trip-tested reference template, with inline
``# options:`` comments for every constrained field. It replaced the
``v6.0``/``v6.1`` pair: two references are two SSOTs, and those two had
diverged by 404 documented paths.

.. note::

   **Top-level** ``model_domain`` is an *optional* convenience knob, not the
   field consumers read. It defaults to ``None`` ("unspecified — defer to the
   nested value") and, when supplied, propagates into ``model.model_domain`` /
   ``model.target_domain`` (the fields the strategies actually consume) and
   **raises** on a genuine conflict with an explicit nested value. Prefer
   setting only ``model.model_domain`` (or ``model.target_domain``). The default
   is deliberately ``None`` rather than ``"image"``: ``main.apply_overrides``
   round-trips the config through ``model_dump()`` → reconstruct, and a
   materialized ``"image"`` default used to be read back and wrongly flagged as
   conflicting with a nested ``"kspace"`` — which broke every non-image
   experiment under ``--override`` (smoke-test regression, 2026-05-29). Keeping
   the default ``None`` makes that round-trip idempotent.

.. note::

   **Schema currency for ``experiments/inprogress/``.** ``1.0`` is the only
   accepted version, and the corpus is migrating to it *plus* a ``workflow:``
   block, opportunistically: any ``inprogress/`` YAML opened during a task is
   brought to ``config_version: '1.0'`` with a declared regime × task before that
   task ends. Note the audit's deliberate asymmetry — an **absent** ``workflow:``
   is advisory, a **wrong** one is a hard error (a ``STUB``-maturity regime, or a
   task the regime does not support), so never guess: bump the version alone and
   leave ``workflow:`` off if the regime is unclear. The bump by itself is only
   bookkeeping — ``config_version`` is validated then deleted in
   ``TrainingSettings.from_yaml`` and nothing branches on it; the ``workflow:``
   block is what activates the axis / spatial-rank / signal-domain /
   component-regime checks. ``active/``, ``validated/``, ``campaigns/`` and the
   deferred ``config_version: '5.0'`` corpus are out of scope. See ``CLAUDE.md``,
   "Keep ``inprogress/`` YAMLs at current schema".

.. note::

   The experiment name lives **under** ``logging:``, not at the top level.
   Earlier revisions of the reference templates surfaced it at the top level,
   but ``TrainingSettings`` is ``extra="forbid"``, so a top-level
   ``experiment_name`` makes the YAML unloadable. Since phase 10b its canonical
   path is ``logging.identity.experiment``; the flat ``logging.experiment_name``
   still loads and is folded into place.

Python access:

.. code-block:: python

   from spectramr.config.settings import TrainingSettings

   cfg = TrainingSettings.from_yaml("experiments/training/my.yaml")

   # ✅ CORRECT: nested access, at the depth the schema actually has
   cfg.training.training_mode
   cfg.optimization.optimizer.learning_rate
   cfg.data.loader.batch_size
   cfg.losses.image_losses          # list[LossComponentConfig]

   # ❌ FORBIDDEN: flat aliases, never on the schema
   cfg.lr                           # AttributeError
   cfg.lambda_l1                    # AttributeError

   # ❌ ALSO WRONG: a `fold` rename is a YAML/override courtesy, not an attribute.
   # These two spellings are accepted in a YAML file and by ``-O``, and are
   # relocated during validation -- but nothing binds them on the object.
   cfg.optimization.learning_rate   # AttributeError
   cfg.data.batch_size              # AttributeError


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

.. note::

   **The old flat spellings still load.**  ``data.batch_size``,
   ``data.contrasts``, ``data.patch_size`` and the rest are ``fold`` records in
   :mod:`spectramr.config.schemas.renames`: a ``mode="before"`` validator MOVES
   the key to its canonical home at parse time rather than raising, so an
   unmigrated arm is unaffected.  There is no forwarding property — the read
   side is canonical only.  ``--override data.batch_size=8`` is translated the
   same way.

.. warning::

   This section previously documented ``in_channels``, ``out_channels``,
   ``contrast``, ``train_manifest`` and ``val_manifest`` as ``data:`` keys.
   **None of them are fields on ``DataConfigSchema``** and none ever fold to
   one — they were doc-only inventions.  ``data.contrast`` in particular is not
   the singular of ``data.pairing.contrasts``; the nearest real fields are
   ``data.input_contrast`` / ``data.target_contrast``, which are
   ``ContrastConfigSchema`` *normalization* specs, not filters.

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

   * - ``coil_processing_mode``
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

.. admonition:: The flat spellings still load, and are being drained
   :class: note

   ``optimization.learning_rate`` and its 34 siblings are ``fold`` records in
   :mod:`spectramr.config.schemas.renames`: a ``mode="before"`` validator moves
   each into its sub-block, so an unmigrated arm keeps loading. They are gone
   from Python — ``config.optimization.learning_rate`` raises ``AttributeError``.
   ``scripts/ci/check_no_legacy_config_keys.py`` prints how many remain; when a
   record reaches zero its posture flips to ``raise`` and the shim is deleted.

Verifying a drain
-----------------

The countdown above decides when a record is promoted, so a drain has to be
verified before it is believed. ``scripts/ci/verify_config_migration.py`` runs
three legs against a baseline ref (``HEAD`` by default, ``--ref origin/dev`` for a
whole PR):

**(i) it still constructs.** Every touched config loads. Reuses
``check_experiment_configs_load.load_failure``.

**(ii) the resolved document is unchanged.** ``TrainingSettings.from_yaml(p)
.model_dump(mode="json")`` at the ref vs the working tree must deep-diff empty.

   This is a *total* oracle for a fold-posture rename rather than a sample: the
   fold validator already performs the migration at parse time, so a correct text
   migration is by definition a no-op on the resolved document. It also subsumes
   the ``metrics.compute`` check, since identical resolved dumps imply identical
   extracted metric sets.

**(iii) the diff is in scope.** ``git diff --unified=0`` may only add or remove
lines belonging to the records actually run. Lines whose content appears on both
sides are *moves* and always pass — a whole-block rename reindents every
descendant, and a multi-line value travels with its key.

The three legs do not overlap, and each has a change only it can catch:

.. list-table::
   :header-rows: 1
   :widths: 46 22 32

   * - change
     - loads?
     - caught by
   * - a moved value was altered
     - yes
     - legs (ii) and (iii)
   * - key renamed under an ``extra="forbid"`` block
     - **no**
     - leg (i)
   * - key smuggled into an ``extra="ignore"`` block
     - yes
     - **leg (iii) only**

The last row is why leg (iii) is not redundant: an ignored key is dropped before
the model exists, so the resolved document is identical and leg (ii) is blind to
it by construction.

.. admonition:: A clean tree passes leg (ii) vacuously
   :class: warning

   With nothing to compare, the deep-diff is empty and the script would report
   success having checked nothing. So it prints how many files it compared and
   how many it skipped with the reason, and ``--require-changes`` turns
   "compared nothing" into an error. Pass it whenever verifying an actual drain.

   The ordering this forces is easy to get wrong: verify on the **dirty working
   tree, before committing**. Commit first and the default ``--ref HEAD``
   compares the tree to itself, reports green, and has compared zero files.

Drained cohorts
---------------

``kspace_filling`` — 58 arms, drained 2026-08-02 (3999 keys; 0 skipped, 0
unparseable, 0 unsupported). Verified with all three legs against the pre-drain
commit: 58/58 construct, 58 resolved documents deep-diff empty, 1892 lines moved
unchanged. The cohort's ``metrics.compute`` drain landed separately on
2026-07-31, and no rename record targets ``metrics.*``, so the two migrations
cannot fold into one another.

Two things a drain does that the legs cannot all see, worth checking by hand
first:

* a ``superseded_by`` record *deletes* a declared key
  (``validation.validation_batch_size``). Leg (i) still loads and leg (iii) sees
  a legal key name, so only leg (ii) can catch it. Before draining, confirm
  every arm declaring the superseded key also declares the winner — otherwise
  the drop reverts that arm to the schema default and
  ``_resolve_batch_size_duplicate`` never had a short form to adjudicate.
* a record whose canonical path lands inside a block an *earlier* migration
  produced folds the migration into its own output. Grep the canonical paths
  against the previous drain's destination before running.

Staying drained is pinned by
``tests/unit/config/test_kspace_filling_cohort_drained.py``. Its third check —
keys authored under a paradigm block that the block's schema does not declare —
compares against ``model_fields``, **not** against the resolved document.
``TrainingStrategyConfigSchema`` and its paradigm sub-blocks are
``extra="allow"``, so a misspelt knob is accepted, carried into the resolved
dump, and therefore looks authored in provenance while no strategy reads it.
"Authored but absent from the resolved document" — the formulation that looks
right — returns nothing for exactly that defect. Each check ships with a control
that injects the defect it exists to catch.

.. admonition:: Never guard a moved field with the OLD name
   :class: warning

   A rename is invisible to attribute-grep wherever the field name is held as a
   **string**. ``getattr(cfg, "batch_size", 4)`` and ``hasattr(cfg,
   "batch_size")`` keep compiling and keep running; the first silently returns
   its default and the second silently returns ``False``. Phase 9a shipped with
   the value reads moved and the guards left behind —

   .. code-block:: python

      data_config.loader.batch_size if hasattr(data_config, "batch_size") else 1

   — which pinned training to ``batch_size=1`` for every arm, disabled gradient
   accumulation, gradient checkpointing and memory monitoring, and computed the
   federated-DP sample rate from a fallback. Nothing went red.

   These fields are **declared**, so a defensive wrapper can never help: the
   ``AttributeError`` is the signal that names the problem. Read the canonical
   path directly.
   ``tests/unit/config/schemas/test_renames.py::TestNoStringKeyedReadsOfFoldedNames``
   ASTs ``src/`` for any ``getattr``/``hasattr`` keyed on a folded legacy leaf,
   with a self-checking allowlist for receivers that carry the name legitimately.

.. admonition:: A declared scheduler family is honoured with or without a dict (#662)
   :class: note

   **Fixed 2026-08-08.** ``resolve_scheduler_spec`` used to return ``None`` —
   *no scheduler at all* — whenever ``optimization.scheduler`` was absent,
   before it ever read ``lr_scheduler_strategy``. 531 arms declared a strategy
   with no ``scheduler:`` dict and therefore trained at a constant LR while
   their config said ``cosine``; nothing warned, because ``None`` is also how
   "no scheduler wanted" is spelled.

   ``optimization.lr_scheduler_strategy: cosine`` alone now resolves, with the
   period defaulting to ``training.max_iterations``. A ``scheduler:`` dict is
   only needed to *parameterise* the family.

   The distinction that keeps this safe is **declared vs defaulted**: the field
   carries ``default="cosine"``, so every resolved config presents a family
   name. Only a name in ``model_fields_set`` is honoured — 305 arms declare
   neither key and must keep resolving to no scheduler.

   The scheduler group stays flat for now, but the fold is no longer *blocked*:
   declaring a strategy and folding it into ``scheduler.type`` finally mean the
   same thing.

.. admonition:: A warmup-bearing name must declare a warmup length
   :class: warning

   ``warmup_cosine``, ``warmup`` and ``linear_warmup`` name two things: a decay
   family and a warmup. The alias table resolves only the decay half — the
   warmup comes from the wrapper that ``warmup_steps`` selects — so declaring
   one of these without ``warmup_steps`` used to drop the warmup silently. 16
   arms asked for ``warmup_cosine`` and got plain cosine annealing. It now
   raises: declare ``optimization.warmup_steps`` (or ``scheduler.warmup_steps``)
   above zero, or name the decay family (``cosine``) directly.

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
       mode's hardcoded 5000 -- which is not a hypothetical: it is how a 4-GPU
       run of ``experiment_11_attention_none`` became unreadable after the fact
       on 2026-08-21. The banner deliberately does **not** claim the operator
       typed the override: ``main.py`` injects overrides of its own and the
       smoke dispatcher injects ``training.max_iterations=<cap>``, all by the
       same route. Its claim is the one that matters -- *this value did not
       come from the config file*.
   * - ``iteration_budget_scope``
     - ``"per_rank"``
     - How ``max_iterations`` is read under ``world_size > 1``. ``per_rank``
       (the default, and the historical behaviour) means **every rank runs the
       full count** -- data parallelism buys effective batch, not a shorter
       run, so a 4-GPU launch costs ~4x the GPU-hours for ~1x the wall-clock.
       ``global`` would divide the bound by ``world_size`` and currently
       **raises**: nothing shards the stream (issue #1163), and dividing the
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
       output_domain: image        # image | complex_image | kspace
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

   ``output_domain`` and ``disable_default_losses`` moved into ``policy:``
   (phase 10d). Both flat spellings still load and are folded into place, but
   they are gone from Python — read ``config.losses.policy.output_domain``.
   ``losses:`` is otherwise sixteen loss-family blocks, so a loose scalar beside
   them read as if it might be a seventeenth; neither of these is a family.

   ``disable_default_losses`` becomes ``exclude_defaults`` rather than being
   inverted to ``enable_default_losses``. The naming rule forbids a negated
   boolean, but this field is a ``list[str]`` of loss **names** — inverting it is
   meaningless, since ``enable_default_losses: ['mse']`` would read as "enable
   ONLY mse", the opposite of a filter. ``exclude_`` states the sense with
   nothing to invert.

.. warning::

   Three further loose scalars stay flat because nothing reads them and no arm
   sets them: ``normalize_losses``, ``clip_loss_value`` and ``loss_scaling``
   (issue #676). ``loss_scaling`` looks referenced only because
   ``MixedPrecisionConfig`` has an unrelated field of the same name.

   ``lambda_deep_supervision`` also stays flat, but for the opposite reason: it
   is live and already correctly placed — a loss **weight**, and
   ``lambda_<term>`` directly on ``losses:`` is the ratified spelling.

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
   Regression: ``tests/unit/infrastructure/training/builders/test_dc_config_unification.py``.
   See ``TODO/backlog_unify_dc_config.md`` for migration plan.


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
   checkpoint service rather than aborting. Prior to 2026-06-20 the root field
   defaulted to ``None`` and ``build_container`` raised
   ``config.checkpoint is required but not provided`` — which silently blocked
   the entire 28-arm ``mrixfields2026`` cohort (none of those configs declare a
   ``checkpoint:`` block, while kspace_filling 14/14 and the VF arms all do).
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

     # inert — declared, read by nothing (see below)
     validation_dir: null
     enable_validation_augmentation: false
     validation_metric: loss
     use_training_loss: true
     num_visualizations: 4
     visualization_dir: ./visualizations

     empty_cache_before_validation: true

``validation.cascade.levels`` is the acceleration ladder the cascading
validation sweep evaluates — one pass per rung, each written as its own row of
``validation_metrics.csv`` (``acceleration_level`` / ``timestep`` as values)
and as flat ``val_<metric>_<R>x`` columns. It was a module constant until
#1394, so an arm could widen ``undersampling.acceleration_range`` for training
while validation stayed pinned at 2/8/32. Leave it ``null`` for the default
ladder; every arm in the corpus does, so nothing moves unless it is declared.

Levels are deduplicated and sorted ascending, and an integral rung stays an
``int`` so ``val_psnr_2x`` does not become ``val_psnr_2.0x`` — the L4 gate and
the accel-gap stamp look those names up and do not raise on a miss. An empty
ladder, a rung below 1x, a non-finite value and a boolean are all **refused at
load time**. Under ``undersampling.schedule_type: step`` a rung outside
``undersampling.acceleration_range`` has no timestep inverse and is skipped at
runtime; ``spectramr audit`` warns before the launch. See
:ref:`validation-cascade-levels`.

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

   The flat spellings still load. ``eval_interval``, ``val_batch_size``,
   ``compute_image_metrics`` and the rest are folded onto their canonical paths
   at parse time, so an unmigrated arm is unaffected — but they are gone from
   Python, so read ``config.validation.schedule.interval_steps``, never
   ``config.validation.eval_interval``.

   The metric block is called ``scoring`` rather than the obvious ``metrics``
   because ``validation.metrics`` is itself one of the retired keys, and the fold
   matches on the key name alone. A block named ``metrics`` would make that key
   mean both the old scalar and its own destination — so a legacy arm would break
   depending on where the author happened to write it, and a migrated arm writing
   ``metrics: {compute: [...]}`` would have the whole block folded into itself.

   Two pairs were merged rather than moved:

   * ``eval_interval`` and ``frequency_steps`` were one cadence with one default.
     58 arms declared both and none disagreed, so both fold to
     ``schedule.interval_steps`` — which also carries 58 arms' previously-unread
     ``frequency_steps`` into the training loop for the first time.
   * ``val_batch_size`` and ``validation_batch_size`` were one number read by two
     builders that preferred *different* spellings, and 86 arms declared both
     with 74 disagreeing. Both now fold to ``loader.batch_size``, with the short
     form winning as both field descriptions always documented.

.. warning::

   **Nine keys are declared and read by nothing.** They stay flat instead of
   being given a tidy home, because a tidy home implies a knob works:

   * ``enabled`` — 1006 arms set it, 8 to ``false``, and validation runs
     regardless: the training loop gates on the *presence* of the block, never on
     this flag (issue #673).
   * ``split`` — 388 arms set it; the fraction that actually partitions the
     corpus is ``data.split.validation_fraction`` (issue #673).
   * ``validation_dir``, ``enable_validation_augmentation``, ``validation_metric``,
     ``use_training_loss``, ``num_visualizations``, ``visualization_dir`` — tracked
     by ``KNOWN_UNCONSUMED`` in ``tests/unit/config/test_schema_key_consumption.py``.

   ``validation_metric`` names the same idea as ``scoring.primary`` but was
   deliberately **not** merged into it: the defaults differ (``'loss'`` vs
   ``'psnr'``), so folding would silently repoint the 53 arms that set it.

   ``empty_cache_before_validation`` is the one live ungrouped scalar. It calls
   ``torch.cuda.empty_cache()`` before each validation pass to free the training
   allocator pool (training usually holds most of VRAM; the EMA weight-swap
   transiently doubles parameter memory). Default ``true`` preserves the OOM-safe
   behavior; set ``false`` on memory-headroom runs to avoid the allocator re-grow
   cost on the next train step (wasted-compute audit PIPE-2).


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

     # wandb_project / wandb_entity: DEFERRED -- read by nothing, and a non-null
     # value now RAISES (issue #675). `null` (shown here) is the only accepted
     # declaration of either.
     wandb_project: null
     wandb_entity: null
     # inert — declared, read by nothing (see below)
     log_weights: false
     log_activations: false
     log_validation_graphs: true
     save_images_per_epoch: 4
     progress_bar_enabled: true
     progress_bar_on_warning: true
     progress_bar_no_progress: false

     log_gradients: false              # live, but its group-mates are not

.. note::

   The flat spellings still load and are folded into place, but they are gone
   from Python: read ``config.logging.intervals.log``, never
   ``config.logging.log_interval``.

   The debug block is ``snapshots``, not the flat key's own name
   ``debug_snapshots``: a destination sub-block may not share a retired leaf
   name, or a migrated arm writing ``debug_snapshots: {enabled: true}`` would
   have that block folded into itself.

.. warning::

   **Ten keys are declared and read by nothing.** Nine are tracked by
   ``KNOWN_UNCONSUMED`` in ``tests/unit/config/test_schema_key_consumption.py``
   and stay flat rather than being given a tidy home that would imply they work:
   ``wandb_project`` (96 arms, measured after the 41-arm ``inprogress`` drain --
   issue #675), ``wandb_entity`` (32, same measurement), ``log_weights`` (112),
   ``log_activations`` (112), ``log_validation_graphs`` (1059),
   ``save_images_per_epoch`` (882), and the three ``progress_bar_*`` flags (752 /
   31 / 31). ``log_gradients`` is the tenth entry in the flat list but is the
   opposite case — it **is** read; it stays flat only because both of its
   group-mates are inert, leaving no group to join.

   Note the consequence for W&B: ``wandb_project`` and ``wandb_entity`` are the
   only fields that could configure it, and neither is read. Unlike the other
   eight, though, they are not silently accepted: a non-null value now RAISES
   (issue #675, ``LoggingConfigSchema._refuse_deferred_wandb``) — only an
   unset field or an explicit ``null`` constructs.

.. warning::

   **``logging:`` is ``extra="ignore"``, and 26 phantom keys are silently
   discarded across 1154 declarations** — see issue #675. The two largest are
   ``project_name`` (419 arms) and ``enable_wandb`` (417 arms); neither exists,
   so those arms get TensorBoard regardless. Several others are one edit away
   from a real field — ``log_level`` (62 arms) for ``sinks.level``,
   ``console_logging`` / ``file_logging`` (20 each) for ``sinks.to_console`` /
   ``to_file``, ``log_frequency`` (63) for ``intervals.log``. They were **not**
   absorbed by phase 10b: wiring ``enable_wandb`` would switch ~416 arms to a
   backend that needs credentials the cluster jobs do not have, which is an
   owner decision rather than a side effect of a readability refactor.


---

``ema:`` — EMAConfigSchema
============================

.. admonition:: Breaking Change (v5.1)
   :class: warning

   Legacy aliases (``enable_ema``, ``ema_decay``, ``ema_update_frequency``)
   removed. Use primary field names below.

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

   spectramr train --config experiments/training/my.yaml --dry-run

The ``ConfigHealthChecker`` validates:

1. ``model.in_channels`` matches ``data.coil_processing_mode``
2. ``losses.policy.output_domain`` matches populated loss lists
3. ``config_version`` present and in ``ACCEPTED_CONFIG_VERSIONS`` — which is
   exactly ``{'1.0'}``; ``LEGACY_CONFIG_VERSIONS`` emptied on 2026-08-08, so
   "accepted" and "current" now name the same thing (see the schema-currency
   note above)
4. All manifest paths exist on disk
5. ``physics.data_consistency.enabled`` in reconstruction modes

**CLI overrides** (without editing YAML):

.. code-block:: bash

   spectramr train --config my.yaml \
       --override "optimization.learning_rate=5e-5" \
       --override "data.batch_size=16"

Both spellings work. ``optimization.learning_rate`` above is a ``fold`` record
whose canonical path is ``optimization.optimizer.learning_rate``;
:func:`spectramr.config.schemas.renames.canonical_override_path` translates the
legacy path before the write, so the two forms are interchangeable exactly as
they are in YAML.

.. admonition:: Why the translation is needed at all
   :class: note

   ``apply_overrides`` re-validates a **complete** ``model_dump()`` — not
   ``exclude_unset``, because provenance must stamp defaulted knobs (pitfall
   #15c). So the canonical key is *always* present with its default, and writing
   the legacy spelling beside it looked to the fold validator like two spellings
   that disagree, which it rejects. That rejection is right for a YAML document,
   where both keys were authored and only a human can say which was meant, and
   wrong here, where one of the two is an artefact of the dump.

   ``raise``-posture records are deliberately **not** translated: they fall
   through untranslated so the owning block still produces the error naming the
   replacement, rather than being silently rewritten into a key that works.


---

References
==========

1. Pydantic V2 Documentation — https://docs.pydantic.dev/latest/

2. Schema source: ``src/spectramr/config/schemas/`` (45 files)

3. Settings entry: ``src/spectramr/config/settings.py::TrainingSettings.from_yaml``

.. _naming-convention:

Naming convention (enforced)
----------------------------

A reader cannot skim a config whose keys follow no rule. These ratify the
plurality already present in the schema rather than imposing a new style — the
census behind them covered 1,888 unique field names across 295 classes.

=====================  =========================  ================================
Kind                   Rule                       Basis
=====================  =========================  ================================
Boolean switch         ``enable_<thing>``         165, vs 20 ``use_*`` / 4 ``*_enabled``
A block's own gate     bare ``enabled``           reserved; never a feature flag
Count                  ``num_<thing>``            36, vs 30 ``n_*``
Loss weight            ``lambda_<term>``          148, vs 40 ``*_weight``; matches the papers
Registry selector      ``<thing>_type``           30, vs 16 ``_mode`` / 8 ``_strategy``
Fraction in [0, 1]     ``<thing>_fraction``       12, vs 5 ``*_ratio``
Path                   ``_path`` file, ``_dir`` directory, ``_root`` tree root  16-16 split; decide by what it points at
Negation               forbidden                  invert the sense instead
=====================  =========================  ================================

``tests/unit/config/test_naming_convention.py`` enforces the mechanically
decidable rules. Today's violators are grandfathered in
``KNOWN_NAMING_EXCEPTIONS`` so the gate is green immediately and can only shrink;
**new fields get no such grace, and the list must not be added to.**

Two rules are documented but deliberately not gated. *Registry selector* cannot
be separated from a genuine mode by name alone (``bidirectional_mode`` really is
a mode), and *path* is a 16-16 split that depends on what the value points at.
Gating either on the name would produce false failures, which is how a ratchet
gets switched off.

One acquisition tuple, one definition
=====================================

An acquisition tuple — ``{name, TE, TR, TI, FA, B0, contrast_type,
include_concomitant}`` — is declared **once**, as
:class:`spectramr.config.schemas.data.AcquisitionParamsSchema`.
``spectramr.config.schemas.training.pmps.AcquisitionParam`` is an alias of it, kept so
``fixed_protocols`` and the module's ``__all__`` are unchanged for callers.

It was not always one. ``pmps.py`` redeclared the class with the same eight fields,
the same types and the same defaults — except ``contrast_type``:

=========================  ===================================================
``data.py``                ``spin_echo | inversion_recovery | gradient_echo |
                           diffusion_weighted | ssfp | mprage``
``training/pmps.py``       ``spin_echo | gradient_echo | inversion_recovery |
                           ssfp | mprage``
=========================  ===================================================

So the *identical* acquisition dict validated as a ``data:`` entry and was **rejected**
as a PMPS protocol. Nothing announced the divergence; the two agreed on every field a
reader would think to compare. This is pitfall #13b's shape — one concept, two
resolvers, silently disagreeing — expressed in schemas rather than loss weights.

The data-layer schema was elected because it is the wider of the two (narrowing would
invalidate the four configs declaring ``diffusion_weighted``) and because ``pmps.py``'s
own dispatch test already imported it rather than the local copy.

``tests/unit/config/schemas/training/test_pmps.py`` pins the **election, not the field
list**: it asserts ``AcquisitionParam is AcquisitionParamsSchema``. A faithful
re-declaration would satisfy any field-by-field comparison on the day it is written and
drift on the day either side changes, which is precisely what happened — only identity
catches it.

.. note::

   Ten acquisition-named constructs exist across the codebase, five of them config
   schemas (``AcquisitionConfigSchema`` is acquisition *design* — trajectory codesign
   and active acquisition — and is unrelated to this tuple;
   ``AcquisitionMetadataConfigSchema`` configures the metadata *loader*;
   ``AcquisitionParamsConfig`` in ``physics.py`` is a tighter 4-field SPGR descriptor
   using ``tr_ms``/``field_strength_t`` rather than ``TR``/``B0``). Before adding an
   eleventh, check whether one of these is the construct you want — issue #828 asked for
   a new shared ``AcquisitionVector`` and the shared schema turned out to already exist,
   twice.
