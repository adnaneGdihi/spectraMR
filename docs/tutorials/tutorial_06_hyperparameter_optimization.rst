.. _tutorial_06_hyperparameter_optimization:

================================================
Tutorial 06: Hyperparameter Optimization (HPO)
================================================

**Difficulty:** Advanced | **Time:** ~3 hours | **GPU:** 8GB+ VRAM

This tutorial covers the HPO pipeline built on **Optuna**, accessed
through the framework's ``HPOUseCase``. You will run a Bayesian
optimization search over learning rate, model architecture, and loss
weights — and automatically select the best configuration.

.. contents:: Table of Contents
   :local:
   :depth: 2


Prerequisites
=============

- Completed :doc:`tutorial_01_basic_reconstruction`
- Familiarity with experiment YAML configs (:doc:`../config_schema_reference`)
- ``optuna`` installed: ``pip install optuna optuna-dashboard``


HPO Architecture
================

.. mermaid::

   flowchart TD
       CLI["spectramr hpo"] --> UC["HPOUseCase"]
       UC --> HC["HPOCoordinator"]
       HC --> OS["Optuna Study"]
       OS --> T1["Trial 1\n(sampled config)"]
       OS --> T2["Trial 2\n(sampled config)"]
       OS --> TN["Trial N\n..."]
       T1 --> TS["Training Loop\n(truncated)"]
       TS --> OBJ["Objective Value\n(val_psnr @ 5k steps)"]
       OBJ --> OS
       OS --> BEST["Best Trial\n→ Full Training"]

       style UC fill:#4a90d9,color:white
       style OS fill:#7bc47f,color:white
       style BEST fill:#e67e22,color:white


Step 1 — Define the HPO Search Space
=======================================

Create ``experiments/hpo/hpo_reconstruction.yaml``:

.. code-block:: yaml

   config_version: "1.0"

   # Base config. The search space is a SEPARATE file (--search-space); the
   # trial knobs below are what it overrides.
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
     output_dir: experiments/results/hpo_reconstruction_tutorial
     max_iterations: 50000        # Full training (used for best trial only)

   losses:
     image_losses:
       - name: l1
         weight: 10.0
         enabled: true
       - name: ssim
         weight: 1.0
         enabled: true

     policy:
       output_domain: image
   optimization:
     lr_scheduler_strategy: cosine
     warmup_steps: 500
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
   checkpoint:
     checkpoint_dir: checkpoints/hpo_trials
     save_interval: 999999     # Don't save during trials (space)
     format: safetensors

   validation:
     enabled: true

     schedule:
       interval_steps: 1000
   physics:
     data_consistency:
       enabled: true
       method: hard
   run:
     seed: 42
     device: cuda

   logging:
     identity:
       experiment: hpo_reconstruction_tutorial
     intervals:
       log: 100


Step 1b — Define the Search Space (a Separate File)
=====================================================

The YAML above is what every trial *starts* from. What HPO is allowed to vary
lives in its own file, passed with ``--search-space``. Each top-level key is a
dotted config path; each value names a distribution.

Create ``experiments/hpo/search_space_reconstruction.yaml``:

.. code-block:: yaml

   # Optimizer
   optimization.optimizer.learning_rate:
     dist: loguniform
     low: 1.0e-5
     high: 2.0e-3

   optimization.optimizer.weight_decay:
     dist: loguniform
     low: 1.0e-7
     high: 1.0e-3

   # Architecture — `depth` is a real `standard_unet` knob (UNetConfig.depth);
   # at in/out 4 channels it moves the model from 9.0M to 36.3M parameters.
   model.model_kwargs.depth:
     dist: int_uniform
     low: 3
     high: 5

   # Loss weight. The `[name=l1]` selector picks one entry out of the
   # `losses.image_losses` list declared above.
   losses.image_losses[name=l1].weight:
     dist: uniform
     low: 1.0
     high: 20.0

Five distribution kinds ship: ``uniform``, ``loguniform``, ``int_uniform``,
``int_loguniform`` and ``categorical``. The first four take ``low:``/``high:``;
``categorical`` takes ``choices:``.

A ``[name=...]`` selector must name the list the entry actually lives in — it
raises on trial 1 if the arm declares that loss somewhere else. Plain dotted
paths are the opposite: they auto-create missing intermediate dicts, which is
why ``model.model_kwargs.depth`` works even though the base config declares no
``model_kwargs`` block at all.

If you would rather not write a file, ``--search-preset <name>`` selects a
built-in space instead. The two flags are mutually exclusive, and **you must
pass one of them**: with neither, every trial runs the base config unchanged,
the coordinator only logs a warning, and the run reports ``best_params: {}``.


Step 2 — Run the HPO Search
==============================

.. code-block:: bash

   # One study, 50 trials, each trained for 5k iterations.
   spectramr hpo \
       --config experiments/hpo/hpo_reconstruction.yaml \
       --model-type standard_unet \
       --search-space experiments/hpo/search_space_reconstruction.yaml \
       --n-trials 50 \
       --max-iter 5000 \
       --objective-metric val_psnr \
       --storage sqlite:///experiments/hpo/tutorial_06.db

``--storage`` is optional but you want it: without it Optuna keeps the study in
memory, so it dies with the process and neither the dashboard nor Step 3 can
reach it.

There is **no worker flag**. Parallelism is the shared storage URL — run the
same command in several processes and Optuna's SQLite backend hands each one a
different trial (the coordinator opens the study with ``load_if_exists=True``).
``--n-trials`` is counted **per process**, so four workers at ``--n-trials 50``
run 200 trials between them, not 50.

.. code-block:: bash

   # Four workers, one GPU each, sharing one study
   for GPU in 0 1 2 3; do
       CUDA_VISIBLE_DEVICES=$GPU spectramr hpo \
           --config experiments/hpo/hpo_reconstruction.yaml \
           --model-type standard_unet \
           --search-space experiments/hpo/search_space_reconstruction.yaml \
           --n-trials 13 \
           --max-iter 5000 \
           --objective-metric val_psnr \
           --storage sqlite:///experiments/hpo/tutorial_06.db &
   done
   wait

   # Monitor live
   optuna-dashboard sqlite:///experiments/hpo/tutorial_06.db

The dashboard shows trial history, parameter importances, and
Pareto fronts at ``http://localhost:8080``.

``--objective-metric`` names a **column in each trial's**
``logs/loss_log.csv``, matched case-insensitively and then by substring.
Validation metrics are logged as ``val_<metric>`` and ``psnr`` is computed by
default, so ``val_psnr`` resolves for this config. If you change the metrics
block, let trial ``0000`` write a few rows and read its header before
committing to a long search — an unresolvable name does not fail loudly, it
just never reports a score.


Step 3 — Inspect Results Programmatically
==========================================

.. code-block:: python

   import optuna

   # The study name is derived, not settable: it is always hpo_<model_type>.
   study = optuna.load_study(
       study_name="hpo_standard_unet",
       storage="sqlite:///experiments/hpo/tutorial_06.db",
   )

   # Best trial
   best = study.best_trial
   print(f"Best val_psnr: {best.value:.2f} dB")
   print(f"Best params:   {best.params}")

   # Parameter importance (requires 20+ trials)
   importances = optuna.importance.get_param_importances(study)
   for param, importance in sorted(importances.items(), key=lambda x: -x[1]):
       print(f"  {param:<45} {importance:.3f}")

Expected output:

.. code-block:: text

   Best val_psnr: 36.8 dB
   Best params:   {
     'optimization.optimizer.learning_rate': 0.000312,
     'optimization.optimizer.weight_decay': 1.7e-05,
     'model.model_kwargs.depth': 5,
     'losses.image_losses[name=l1].weight': 6.4,
   }

   optimization.optimizer.learning_rate       0.487
   model.model_kwargs.depth                   0.264
   losses.image_losses[name=l1].weight        0.170
   optimization.optimizer.weight_decay        0.079


Step 4 — Train Best Configuration to Convergence
==================================================

After HPO, apply the best params to a full training run:

.. code-block:: bash

   # There is no export step. HPO already wrote the winner:
   OUT=experiments/results/hpo_reconstruction_tutorial/hpo/hpo_standard_unet

   cat $OUT/best_params.json     # raw Optuna params + objective value
   cat $OUT/best_config.yaml     # the base YAML with every winning param applied

   # Audit it like any other config, then train it to convergence.
   spectramr audit $OUT/best_config.yaml
   spectramr train \
       --config $OUT/best_config.yaml \
       --override "training.max_iterations=100000" \
       --override "checkpoint.save_interval=5000"

The output directory is ``<--output-dir>/hpo_<model_type>``, and ``--output-dir``
itself defaults to ``<training.output_dir>/hpo`` — hence the path above.
``best_config.yaml`` carries a ``metadata`` block recording ``hpo_source``,
``hpo_objective_metric`` and ``hpo_best_value``, and it repoints
``training.output_dir`` to ``<--output-dir>/standard_unet_hpo_winner`` so the
convergence run cannot overwrite the trial directories it came from.

You can also apply the winning values to the base config by hand:

.. code-block:: bash

   spectramr train \
       --config experiments/hpo/hpo_reconstruction.yaml \
       --override "optimization.optimizer.learning_rate=0.000312" \
       --override "optimization.optimizer.weight_decay=1.7e-05" \
       --override "model.model_kwargs.depth=5" \
       --override "training.max_iterations=100000"

.. warning::

   The loss weight is missing from that list on purpose. ``--override`` cannot
   express a ``[name=...]`` selector: it splits on the first ``=``, which lands
   inside the selector, and then **silently discards** the result while logging
   ``Overrides applied (1)``. The selector
   works in a search space, which uses a different resolver. Until that is
   fixed, edit sampled loss weights into the YAML — or just run
   ``best_config.yaml``, which already has them.


Step 5 — Advanced: Multi-Objective HPO
========================================

Optimize for reconstruction quality *and* training cost. The second objective
is fixed — ``training_time_seconds``, minimized. You do not get to choose it,
and there is no YAML block for any of this; multi-objective HPO is entirely
CLI-driven:

.. code-block:: bash

   spectramr hpo \
       --config experiments/hpo/hpo_reconstruction.yaml \
       --model-type standard_unet \
       --search-space experiments/hpo/search_space_reconstruction.yaml \
       --objective-metric val_psnr \
       --multi-objective \
       --cost-weight 0.5 \
       --n-trials 50 \
       --storage sqlite:///experiments/hpo/tutorial_06_mo.db

``--multi-objective`` **alone does nothing**. The coordinator builds the second
objective only when ``--cost-weight`` is also greater than zero, and
``--cost-weight`` defaults to ``0.0`` — so ``--multi-objective`` on its own
gives you an ordinary single-objective run with no warning.

The result is a Pareto front over ``(val_psnr, training_time_seconds)``.
``best_config.yaml`` records the highest-quality point on it; ``best_params.json``
carries the whole front for further analysis.


HPO Sampler Guide
==================

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Sampler
     - Best For
     - Notes
   * - ``tpe``
     - Default — continuous params
     - Tree-structured Parzen Estimator; Bayesian
   * - ``cmaes``
     - Continuous params, smooth landscapes
     - Covariance Matrix Adaptation Evolution Strategy
   * - ``nsga2``
     - Multi-objective optimization
     - Pareto-front aware. **Only takes effect together with
       ``--multi-objective --cost-weight >0``** — see the warning below.

Those three are the whole ``--sampler`` vocabulary; anything else is rejected
by argparse.

HPO Pruner Guide
=================

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Pruner
     - Best For
     - Notes
   * - ``hyperband``
     - **Default** — large-scale sweeps
     - Successive halving with bracket scheduling. Milestones are fixed at
       iters 4k / 8k / 16k / 32k and do **not** rescale to ``--max-iter``, so a
       5k-iteration budget gets pruned once, at 4k.
   * - ``median``
     - Prune underperforming trials early
     - Stops a trial if it is below the running median at any checkpoint
   * - ``successive_halving``
     - Fixed-budget sweeps
     - Hyperband's inner loop without the bracket schedule
   * - ``threshold``
     - Absolute quality floor
     - **Unusable as shipped** — see the warning below
   * - ``none``
     - Short runs where pruning would kill trials before they stabilize
     - **Does not currently work** — see the warning below

.. warning::

   Three of these choices do not reach the code that implements them, because
   the CLI's ``choices=`` list and the factory that consumes the string are two
   separate, unsynced vocabularies:

   * ``--pruner none`` is translated to ``nop``, for which the pruner factory
     has no branch, so it falls through to **MedianPruner** — the opposite of
     what you asked for. Trials you expected to run to completion get pruned.
   * ``--pruner threshold`` raises ``TypeError: Either lower or upper must be
     specified.`` before the first trial; nothing plumbs those bounds.
   * ``--sampler nsga2`` is checked as ``nsgaii`` in the factory, so on a
     single-objective run it silently becomes **TPESampler**. It works under
     ``--multi-objective --cost-weight >0`` only because a separate
     ``n_objectives > 1`` branch catches it.

   Until that is fixed, the choices that behave as documented are
   ``--sampler {tpe,cmaes}`` and ``--pruner {hyperband,median,successive_halving}``.
   To genuinely disable pruning, raise ``--max-iter`` past the first Hyperband
   milestone instead.


Key Takeaways
=============

1. **Objective metric at partial training** — 5k steps proxy for 100k
2. **Log-uniform LR** — always sample in log space for learning rates
3. **Parallel workers** — each worker loads the same Optuna storage
4. **Dashboard** — ``optuna-dashboard`` gives live visualization
5. **No export step** — HPO writes ``best_config.yaml`` itself, ready to run
6. **Pruning is on by default** — Hyperband, not median, and ``--pruner none``
   does not turn it off


See Also
========

- :doc:`../config_schema_reference` — all YAML keys
- :doc:`../strategies_reference` — choose the right training mode
- :doc:`../models_reference` — model types for the search space
- :doc:`../troubleshooting` — OOM during HPO (reduce ``objective_steps``)
