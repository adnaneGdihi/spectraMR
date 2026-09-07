Running Pipelines: Modes & Lifecycle
====================================

spectraMR runs every paradigm (GAN, diffusion, VAE/VQ-VAE, SSL/MAE,
reconstruction-only, domain-adaptation, physics-driven, virtual-fiducial, …)
through **one config and one entry point**. You do not pick the paradigm on the
command line — it is selected by ``training.training_mode`` in the YAML and
resolved to a strategy by ``TrainingStrategyFactory``. What you *do* pick on the
command line is the **run mode**: validate, smoke-test, train, resume, ablate,
sweep, infer, or report.

Three orthogonal axes
---------------------

Keep these separate in your head — they compose freely:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Axis
     - Decided by
     - Covered in
   * - **WHAT** (paradigm)
     - ``training.training_mode`` in the config
     - :doc:`config_schema_reference`
   * - **WHICH** (run mode)
     - the CLI verb + flags
     - **this page**
   * - **WHERE / HOW-MANY**
     - the launcher backend + single/campaign
     - :doc:`execution_modes`

The same config flows, frozen and loaded once, through whichever mode you run —
so a config that trains is the same object that gets audited, dry-run, smoke-
tested, swept, and served.

Two equivalent entry points
---------------------------

Every example below works with either spelling:

.. code-block:: bash

   spectramr <verb> [options]            # the installed console script
   python -m spectramr.cli <verb> [options]   # module form (identical parser)

``spectramr --help`` lists every verb; ``spectramr <verb> --help`` shows that verb's
flags. (Heavy verbs print a one-line ``⏳ importing PyTorch + model registry…``
notice on first use so the unavoidable cold import is not a silent wait — see
:doc:`cli_reference`.)

The recommended lifecycle
-------------------------

A config travels from *idea* to *result* through a sequence of progressively
heavier modes. Each step is cheap insurance against the next: catch a schema typo
in 100 ms (``audit``) before you discover it 40 minutes into a cluster job.

.. code-block:: text

   doctor → audit → train --dry-run → sanity_check → train → report → infer
   (env)    (config)  (wiring)         (does it learn?) (real)  (figures) (serve)

**0. Is my environment sane?** ``doctor`` — torch/CUDA, visible devices, cache
and data roots, env knobs. The cluster pre-flight gate.

.. code-block:: bash

   spectramr doctor --require-cuda            # exit non-zero if no GPU is visible
   spectramr doctor --config exp.yaml --json  # also confirm the YAML loads

**1. Is my config valid?** ``audit`` — the audit ladder. Tier 0 (Pydantic
schema) + Tier 1 (static cross-validation) in ~100 ms; add ``--probe`` for the
Tier-2 synthetic forward pass (~30 s, instantiates the model, catches AMP / shape
/ OOM). Note the config is a **positional** argument here, not ``--config``.

.. code-block:: bash

   spectramr audit experiments/inprogress/workflow_baselines/b1_structural_recon_m4raw.yaml          # Tier 0+1
   spectramr audit experiments/inprogress/workflow_baselines/b1_structural_recon_m4raw.yaml --probe   # + Tier 2
   spectramr audit experiments/inprogress/workflow_baselines/            # bulk

``--strict`` is **on by default**: every warning is an error and the exit code is
2. Pass ``--no-strict`` to accept warnings with exit 1 — a per-arm opt-out belongs
in the config (``synthetic_forward_probe_skip``), not on the command line. A
directory argument audits every YAML beneath it and prints an aggregate summary.
See :doc:`audit_ladder_user_guide`.

**2. Does the whole thing wire up?** ``train --dry-run`` — loads the config,
builds the full DI container (model + losses + data + strategy), then stops
*before* the training loop. Proves every component resolves and the dataloader
constructs, without spending a single gradient step.

.. code-block:: bash

   spectramr train --config exp.yaml --dry-run     # --dry_run also accepted

**3. Does the model actually learn?** ``sanity_check`` — overfits a **single
batch**. It injects a fixed set of overrides (``data.batch_size=1``, EMA off,
``learning_rate=1e-4``, no warmup, constant LR) so the loss *must* collapse toward
zero if the model, losses, and gradients are wired correctly. A sanity check that
*cannot* overfit one batch is a bug you want to find before a full run.

.. code-block:: bash

   spectramr sanity_check --config exp.yaml --device cuda

**4. The real run.** ``train`` — the canonical training pipeline.

.. code-block:: bash

   spectramr train --config exp.yaml --device cuda --seed 42

   # tweak config values inline without editing the YAML (repeatable, nested keys):
   spectramr train -c exp.yaml -O optimization.optimizer.learning_rate=1e-4 \
                             -O validation.schedule.interval_steps=100

   # resume from a checkpoint (explicit path, or 'auto' for the latest in output_dir):
   spectramr train -c exp.yaml --resume experiments/results/exp/checkpoints/best.pt
   spectramr train -c exp.yaml --resume auto

``--override / -O`` uses **dotted nested paths** because the config is nested
(``optimization.optimizer.learning_rate``, never ``lr``); each ``-O`` is one
key. Overrides are re-validated against the schema, so an illegal value still
fails loudly.

**5. Figures and tables.** ``report`` — runs the same reporting pipeline the
end-of-training hook uses, against an existing output directory. Idempotent: the
output is identical whether training triggered it or you invoke it by hand.

.. code-block:: bash

   spectramr report --exp-dir experiments/results/exp --task reconstruction \
                  --method "my-method"

**6. Serve a trained checkpoint.** ``infer`` — the SSOT inference pipeline. The
settings come from the run's own ``resolved_config.json`` when it sits beside
the checkpoint (its directory or the parent): its ``_declared`` block
re-validates to exactly the settings the run resolved, overrides included, so
the checkpoint is scored under the config it trained under. ``--config`` is
then optional; it is read when no artifact exists, or under ``--from-yaml``. A
YAML that disagrees with the artifact is reported (the differing top-level
blocks, at WARNING and in the result's ``config_source``), not used.

.. code-block:: bash

   spectramr infer --checkpoint experiments/results/exp/checkpoints/best.pt \
                 --input data/test/ --output predictions/ --device cuda
   # the training YAML, even when the artifact exists beside the checkpoint:
   spectramr infer --config exp.yaml --from-yaml --checkpoint ... --input ... --output ...

.. note::

   ``predict`` (``--model``) runs the same pipeline through the same preamble
   (ledger, seed, determinism policy, accelerator resolution) and takes the
   same ``--from-yaml``; ``infer-dataset`` is a back-compat alias for ``infer``.
   An artifact carrying no ``_declared`` block falls back to the YAML, with a
   warning; without a YAML the run refuses.

**Hard data consistency at predict.** With
``physics.data_consistency.apply_at_predict: true`` the prediction is projected
onto the measured k-space as the last inference step: every sampled bin takes
the measured value and the unsampled bins keep the predicted one. This is the
predict-side counterpart of the training-time layer, which is model-integrated
and inactive for a plain reconstruction or GAN network at inference. The
setting is independent of ``enabled`` and ``method``. The projection needs
three tensors, so it is not an adapter: ``infer`` reads the mask from the
``mask`` dataset of an HDF5 input (the fastMRI test-set layout, ``(H, W)``,
``(N, H, W)`` or ``(N, 1, H, W)`` with ``N`` equal to one or to the number of
slices), takes the k-space input as the measurement, and raises when either is
missing. NIfTI and NPY inputs have no mask, an image-route arm has no
measurement, and a model with no registered ``output_domain`` has no projection
domain; each of the three raises at load or at the first file. A sampler that
already replaces the measured bins in its own loop (diffusion under
``method: hard``, cold diffusion given a mask, physics-driven) is projected once,
by that loop. The result dict and ``run_summary.json`` record a
``data_consistency_at_predict`` block with the setting, the domain, the number
of batches the predict step projected, and the loops that projected instead.
The measurement is used as acquired; ``eval_noise_level`` is a validation-time
simulation parameter and is not added.

Variant modes — same config, different question
-----------------------------------------------

These reuse the training pipeline but ask a comparative or search question.

**Ablation** — train the baseline plus one variant per ``--vary`` override, then
emit ``ablation_results.json`` with the per-metric baseline→variant delta. Runs
the arms **sequentially in-process**; for many arms or long runs use ``campaign``.

.. code-block:: bash

   spectramr ablation -c base.yaml \
       --vary model.model_kwargs.force_pure_kspace=false \
       --vary objectives.reconstruction.lambda_l1=0.5 \
       --max-iterations 2000 --output-dir experiments/results/exp_ablation --device cuda

**Hyperparameter optimization** — Optuna-backed search over a base config. Each
trial spawns a **separate trainer subprocess** so one trial crash (NaN / OOM)
can't poison the study. Define what may vary with a built-in preset or a search-
space YAML; with neither, every trial runs the base config unchanged.

.. code-block:: bash

   spectramr hpo --list-presets                       # discover built-in search spaces
   spectramr hpo --list-schema-paths                  # every tunable dotted-path key
   spectramr hpo --print-template > my_space.yaml     # starter search-space YAML

   spectramr hpo -c base.yaml -m unet --n-trials 50 \
       --search-preset lr_wd_curriculum --sampler tpe --pruner hyperband \
       --objective-metric val_loss --max-iter 30000 --device cuda

**Distributed (DDP)** — launched through ``torchrun``; the verb itself just wires
the DDP pipeline. The import-time env setup (cache root, thread isolation,
``PYTORCH_CUDA_ALLOC_CONF``) is preserved.

.. code-block:: bash

   torchrun --nproc_per_node=4 -m spectramr.cli train-distributed \
       --config exp.yaml --backend nccl --resume auto

Utility modes
-------------

.. code-block:: bash

   spectramr benchmark --suite all          # quality / throughput / memory micro-benchmarks
   spectramr export --model best.pt --config exp.yaml --format onnx   # ONNX / TorchScript
   spectramr list-features --module models --format markdown          # what's registered
   spectramr meta-evaluate ...              # rank a metric set (see meta-evaluation docs)

Scaling out & choosing where to run
-----------------------------------

The verbs above answer *which mode*. **Where** they run (this process, Docker,
Apptainer, or a SLURM job) and **how many** (one run or a whole campaign) are the
other two axes, handled by the unified launcher and campaign manifests:

.. code-block:: bash

   spectramr launch exp.yaml --where slurm --gpus 2          # train as a SLURM job
   spectramr launch exp.yaml --pipeline infer --where local -- --checkpoint best.pt --input d/
   spectramr launch campaign.yaml --fanout campaign --where slurm   # a whole sweep

``launch`` is an additive front door over the same machinery — every dedicated
verb still works on its own. ``--dry-run`` on ``launch`` prints the exact command
/ sbatch script that *would* run. See :doc:`execution_modes` for the full
WHAT × WHERE × HOW-MANY cube, including campaign fan-out.

Going config-free: the imperative API
-------------------------------------

When you'd rather write Python than a YAML — notebooks, quick experiments,
embedding training in a larger script — the imperative surface removes the config
axis entirely:

.. code-block:: python

   from spectramr import fit, make_model, make_dataloader, make_optimizer

   model = make_model("unet", in_channels=1, out_channels=1)
   ...

See :doc:`scripting_api` for ``fit`` / ``Trainer`` and the ``make_*`` builders.

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - Mode
     - Verb
     - Use it to…
   * - Environment
     - ``doctor``
     - confirm torch/CUDA, devices, roots
   * - Validate (static)
     - ``audit`` *(+ ``--probe``)*
     - catch schema / wiring / shape bugs fast
   * - Validate (wiring)
     - ``train --dry-run``
     - build the full container without training
   * - Smoke
     - ``sanity_check``
     - overfit one batch — does it learn?
   * - Train
     - ``train``
     - the real run (``-O``, ``--resume``)
   * - Resume
     - ``train --resume <path|auto>``
     - continue from a checkpoint
   * - Ablate
     - ``ablation --vary k=v``
     - baseline→variant deltas (sequential)
   * - Sweep
     - ``hpo``
     - Optuna search (subprocess-isolated)
   * - Distributed
     - ``torchrun … train-distributed``
     - multi-GPU DDP
   * - Infer
     - ``infer``
     - apply a checkpoint to new data
   * - Report
     - ``report``
     - figures + tables from an output dir
   * - Scale out
     - ``launch`` / ``campaign``
     - choose where / run many
   * - Config-free
     - ``spectramr.fit`` / ``Trainer``
     - imperative Python API

.. seealso::

   * :doc:`cli_reference` — full per-command flag reference and dispatch internals.
   * :doc:`execution_modes` — WHAT × WHERE × HOW-MANY launch cube (local / Docker / Apptainer / SLURM).
   * :doc:`audit_ladder_user_guide` — the Tier 0/1/2 audit ladder in depth.
   * :doc:`scripting_api` — the imperative ``fit`` / ``Trainer`` surface.
   * :doc:`config_schema_reference` — every config key, with defaults.
