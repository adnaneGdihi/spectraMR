CLI Reference & Entry-Point Scheme
==================================

spectraMR exposes a single installed entry point, the ``spectramr`` console
script, which maps to :func:`spectramr.cli.app.main` (``pyproject.toml``
``[project.scripts]``). Every operation is a subcommand of that one parser.

.. code-block:: bash

   spectramr <command> [options]
   # equivalently:
   python -m spectramr.cli <command> [options]

Command index
-------------

The **24** verbs, read from the live ``argparse`` tree. ``--config?`` is per
subparser: a blank cell means the verb takes its input another way (``audit``
takes the YAML as a positional). ``launch?`` marks the verbs reachable through
``spectramr launch --pipeline``.

.. list-table::
   :header-rows: 1
   :widths: 20 8 8 64

   * - Command
     - ``--config``
     - ``launch``
     - Purpose
   * - ``spectramr doctor``
     - yes
     - —
     - Print environment diagnostics (torch/CUDA, devices, cache/data roots, env knobs).
   * - ``spectramr train``
     - yes
     - yes
     - Train a model.
   * - ``spectramr sanity_check``
     - yes
     - yes
     - Sanity check — overfit a single batch.
   * - ``spectramr ablation``
     - yes
     - yes
     - Train the baseline plus one variant per ``--vary`` override; report per-metric deltas. Sequential/local.
   * - ``spectramr infer``
     - yes
     - yes
     - Run inference using a trained model.
   * - ``spectramr infer-dataset``
     - yes
     - —
     - **Deprecated** — alias for ``infer``.
   * - ``spectramr experiment``
     - yes
     - yes
     - Run a complete experiment with directory management.
   * - ``spectramr train-distributed``
     - yes
     - —
     - DDP training. Not launched directly — see :doc:`distributed_training`.
   * - ``spectramr predict``
     - yes
     - —
     - Inference via the SSOT pipeline.
   * - ``spectramr profile``
     - yes
     - —
     - Profile a training step and write a line-level report.
   * - ``spectramr benchmark``
     - —
     - —
     - Run benchmarks.
   * - ``spectramr export``
     - yes
     - —
     - Export a model.
   * - ``spectramr list-features``
     - —
     - —
     - List available models, losses, metrics, strategies.
   * - ``spectramr audit``
     - —
     - —
     - Audit an experiment YAML: Tier 0 schema, Tier 1 health checker, Tier 2 probe with ``--probe``. Takes the YAML as a **positional**, not ``--config``. Exit 0 pass / 1 warnings / 2 errors.
   * - ``spectramr campaign``
     - —
     - —
     - Manage campaigns (``submit`` / ``status`` / ``evaluate`` / ``cancel``).
   * - ``spectramr hpo``
     - yes
     - yes
     - Optuna-backed HPO over a base training YAML; each trial is a separate trainer subprocess.
   * - ``spectramr report``
     - yes
     - —
     - Canonical figures + tables for an output dir (same pipeline as the end-of-training hook).
   * - ``spectramr meta-evaluate``
     - —
     - —
     - Rank a metric set with the meta-evaluation framework.
   * - ``spectramr audit-ksd``
     - —
     - —
     - Tier-3 KSD defensibility audit over a generative-prior model.
   * - ``spectramr infer-protocol``
     - —
     - —
     - Posterior-mode inference over acquisition parameters.
   * - ``spectramr simulate-acquisition``
     - —
     - —
     - Synthetic IB-acquisition trajectory with per-step metrics; writes CSV.
   * - ``spectramr design-mrf-sequence``
     - —
     - —
     - Beltrami-CRLB-optimal MRF pulse-sequence design.
   * - ``spectramr regulatory``
     - —
     - —
     - Regulatory bundle CLI (``bundle`` / ``verify`` / ``status``).
   * - ``spectramr launch``
     - —
     - —
     - Unified launcher — any pipeline, anywhere. See :doc:`environment_variables`.

Dispatch architecture
---------------------

There is **one** argparse parser (in :mod:`spectramr.cli.app`). Each subcommand
sets a handler via ``set_defaults(func=...)`` and ``main()`` calls
``args.func(args)``.

The training commands (``train`` / ``sanity_check``) call
:func:`spectramr.main.train_command` / :func:`spectramr.main.sanity_check_command`
**directly** with the parsed args. They do *not* rebuild ``sys.argv`` and
re-invoke a second parser.

Common commands
---------------

.. code-block:: bash

   # Train (config is the SSOT; loaded once into a frozen TrainingSettings)
   spectramr train --config experiments/inprogress/<paradigm>/<arm>.yaml
   spectramr train -c <arm>.yaml --device cpu --seed 7 --dry-run
   spectramr train -c <arm>.yaml -O optimization.optimizer.learning_rate=1e-4

   # Cap validation for a smoke run (see "Capping validation" below)
   spectramr train -c <arm>.yaml --val-batches 1

   # Overfit a single batch (collapse-vs-bug diagnostic)
   spectramr sanity_check -c <arm>.yaml

   # Pre-flight a config (Tier 0+1 ~100 ms; --probe adds Tier 2 forward pass)
   spectramr audit <arm>.yaml [--probe]

   # Bulk-audit a whole cohort (positional is a directory → recurse all YAMLs)
   spectramr audit experiments/inprogress/<cohort>

   # ...skipping ablation arms to focus the audit on training arms.
   # --exclude PATTERN is fnmatch glob over the path relative to the directory
   # (a bare substring is wrapped as *PATTERN*); repeatable. Use -path-style
   # '*ablation*' so it catches BOTH ablations/ subdirs and *ablation* filenames.
   spectramr audit experiments/inprogress/<cohort> --exclude '*ablation*'
   spectramr audit experiments/inprogress/<cohort> --exclude '*ablation*' --exclude '*baseline*'

Capping validation (``--val-batches``)
--------------------------------------

A full validation pass walks every volume in the validation split, which is what
a real run wants and what makes a smoke run slow: the training side is already
capped at a handful of iterations while validation still grades the whole split.

``--val-batches N`` caps it, on ``train`` and ``sanity_check``:

.. code-block:: bash

   spectramr train -c <arm>.yaml --val-batches 1        # grade one batch
   spectramr sanity_check -c <arm>.yaml --val-batches 2

The flag is sugar for ``-O validation.loader.num_batches=N`` and is resolved in
:func:`spectramr.cli.val_batches.resolve_val_batches_overrides`, so it goes
through the config SSOT rather than a second code path. Passing it *and* an
explicit ``-O`` for the same key raises rather than letting one silently win.

Two things are worth knowing about what the cap buys:

* **It saves the decode, not just the progress bar.**
  :mod:`spectramr.infrastructure.builders.directors.data_pipeline_director`
  strides the validation dataset down to the capped budget when the loader is
  built, so the volumes outside the budget are never read;
  :mod:`spectramr.pipelines.train` caps the loop as well, which is what covers a
  dataset that has no ``__len__``.
* **The kept samples are spread across the split**, endpoint-inclusive, rather
  than taken from the front. Validation splits are commonly ordered by slice
  position, so a head-truncated cap grades a run almost entirely on background
  slices. ``--val-batches 1`` is therefore not necessarily the first sample.

On a volume-backed arm the direct validation loader emits one volume per batch,
so ``N`` is also the number of volumes graded.

``--val-batches`` caps validation; it cannot switch it off. ``validation.enabled``
is declared by many arms and read by none (issue #673), so a value below 1 is
refused rather than quietly meaning "no cap".

Generating reports (``report``)
-------------------------------

``spectramr report`` builds the figures + tables (and the quality-control QC
report) from an already-downloaded run directory — the same
:func:`spectramr.infrastructure.reporting.generate_report` orchestrator the
end-of-training hook calls, so the output is identical whether training triggers
it or you invoke it by hand. **Every registered figure is attempted** (data-less
ones soft-skip), so the report includes all figures the reporting module
supports; set ``reporting.figures`` in a ``--config`` YAML to restrict. It reads ``logs/training_metrics.csv`` /
``final_metrics.json`` (via the aggregator) plus recorded ``report_cases`` images
— or, when those npz cases are absent, the downloaded ``real_images`` /
``fake_images`` PNG pairs.

.. code-block:: bash

   # Report from a downloaded run (figures + tables + qc_report.html)
   spectramr report --exp-dir experiments/results/<run>/

   # Reuse a config's reporting: block verbatim (parity with the training hook)
   spectramr report -e experiments/results/<run>/ -c experiments/inprogress/<paradigm>/<arm>.yaml

   # Toggle the QC figures / HTML wrapper / interactive layer (override the config)
   spectramr report -e <run>/ --no-html            # figures only, skip the HTML
   spectramr report -e <run>/ --qc --task reconstruction
   spectramr report -e <run>/ --no-interactive     # static-only HTML (no plotly layer)

   # Batch: report EVERY run under a cohort root + a linking report_index.html
   spectramr report --exp-dir experiments/results/ --recursive
   spectramr report -e experiments/results/vf/ --all -c <arm>.yaml   # config parity per run

Outputs land under ``<exp-dir>/<out-subdir>/`` (default ``report/``): the vector
+ raster figures, ``report_summary.md``, ``report_manifest.json``, and the
self-contained ``qc_report.html``. When :mod:`plotly` is installed
(``pip install spectramr[viz]``) and ``--interactive`` is on (the default) the HTML
carries an interactive layer — hoverable group IQMs, interactive learning curves /
metric distributions / Bland-Altman, and **2-D + 3-D MRIQC-style slice viewers**
(scrub subjects/slices, flick Prediction/Target/\|Error\|). plotly.js is inlined
once so the report works **offline** (no CDN); ``--no-interactive`` or a missing
plotly falls back to static PNGs. The 3-D viewer appears only when the run
recorded volumes (``reporting.record_volumes``); see :doc:`reporting`.
Both ``--interactive`` and the config's ``reporting.interactive`` are forwarded
(CLI wins). With ``--recursive`` (alias ``--all``/``-r``) ``--exp-dir`` is treated
as a *cohort root*: every run beneath it (any dir carrying
``logs/training_metrics.csv`` / ``final_metrics.json`` / ...) is reported and a
top-level ``report_index.html`` links them all; a failing run is logged and
skipped rather than aborting the batch. To fire the single-run report
automatically at the end of training, add a ``reporting:`` block
(``enabled: true``) to the run's YAML; see :doc:`reporting`.

Cluster diagnostics & global flags
----------------------------------

``spectramr doctor`` prints an environment snapshot — no job launched — so you can
confirm a node is set up before submitting:

.. code-block:: bash

   spectramr doctor                       # version, torch/CUDA, devices+memory,
                                        # cudnn flags, cache/data roots, SPECTRAMR_* knobs, SLURM
   spectramr doctor --json                # machine-readable (for SLURM prologue checks)
   spectramr doctor -c <arm>.yaml         # also validate the config loads against the schema
   spectramr doctor --require-cuda        # exit non-zero if no GPU is visible (pre-flight GATE)

``doctor`` is import-safe: torch is probed defensively, so it still runs (and
reports the failure) on a broken / CPU-only environment — exactly when you need
it. ``--require-cuda`` / ``-c`` make it a gate you can chain before ``train`` in
a SLURM script (``spectramr doctor --require-cuda -c arm.yaml && spectramr train -c arm.yaml``).

Global flags (before the subcommand):

* ``--version`` — print the installed version and exit.
* ``-v`` / ``--verbose`` — print full tracebacks on error (otherwise a concise
  one-line error is logged; the env var ``SPECTRAMR_DEBUG=1`` does the same).

**Error boundary.** ``main()`` wraps the dispatch: ``Ctrl-C`` exits ``130`` with
a clean message (no raw ``KeyboardInterrupt`` traceback in the SLURM log), a
handler's own ``sys.exit`` code is preserved, and any other exception becomes a
concise error + exit ``1`` (full traceback only under ``-v`` / ``SPECTRAMR_DEBUG``).

Ablation studies
----------------

``spectramr ablation`` trains the baseline config plus one variant per
``--vary`` override, then writes ``ablation_results.json`` with the
baseline→variant delta and percent change per validation metric.

.. code-block:: bash

   spectramr ablation \
       --config <your-config>.yaml \
       --vary model.model_kwargs.force_pure_kspace=false \
       --output-dir experiments/results/fpk_ablation \
       --device cuda

* ``--vary DOTTED.PATH=VALUE`` (repeatable) — each defines one variant. The
  value is type-coerced exactly like ``--override`` (``false`` → ``bool``,
  ``1e-4`` → ``float``, ``none`` → ``None``). A spec without ``=`` raises at
  parse time (no silent fallback).
* ``--max-iterations N`` — caps iterations per arm for quick local sweeps
  (folded into every variant *and* the baseline for a fair comparison).
* ``--output-dir`` — defaults to ``<config_stem>_ablation/``.

Under the hood the command calls
:func:`spectramr.pipelines.ablation.run_ablation_study` with the default
training-backed evaluator
:func:`spectramr.pipelines.ablation.train_and_score`, which trains each config
via :func:`spectramr.pipelines.train.run_training_pipeline` and reads the best
validation metrics back from ``validation_metrics.csv``.

When to use ``ablation`` vs ``campaign`` vs ``hpo``
---------------------------------------------------

All three run more than one configuration; they differ in scale and venue:

.. list-table::
   :header-rows: 1
   :widths: 14 30 30 26

   * - Command
     - What it does
     - Where / how
     - Use when
   * - ``ablation``
     - Baseline + N single-knob ``--vary`` variants; automated delta report
     - **Local, sequential**, one process
     - A few variants, interactive, single GPU
   * - ``campaign``
     - Many declared arms (``CampaignConfigSchema``); leaderboard + reports
     - **Cluster (SLURM)**, parallel
     - Large studies, many arms
   * - ``hpo``
     - Optuna search over a base YAML; each trial a subprocess
     - Local or cluster, sampler/pruner-driven
     - Tuning continuous/categorical hyperparameters

.. warning::

   ``ablation`` trains the baseline and every variant **sequentially in one
   process**. For many arms or long runs, prefer ``campaign`` so arms run in
   parallel on the cluster.

Related verbs
-------------

``predict`` is the ``--model`` spelling of ``infer``: the same
``run_inference_pipeline`` behind the same preamble. Both take their settings
from the ``resolved_config.json`` beside the checkpoint when it exists
(``--config`` optional; the YAML is used when the artifact is absent or predates
its ``_declared`` block, and under ``--from-yaml``); see
:doc:`running_pipelines`.

``infer-dataset`` is a **separate verb**, not an alias: it requires ``--config``,
``--input`` and ``--output``, and adds ``--batch-size``, which ``infer`` does not
take. Use ``infer`` for one input and ``infer-dataset`` for a directory.

``experiment`` is a thin experiment-named wrapper over ``train``: it translates
``--experiment`` / ``--max-epochs`` / ``--checkpoint-interval`` into config
overrides (``training.output_dir`` / ``training.epochs`` /
``checkpoint.save_interval``) and runs the canonical ``run_training_pipeline``.

The first call in a fresh process is slow
-----------------------------------------

The first heavy verb (``train`` / ``audit`` / ``infer`` / …) in a fresh process
imports PyTorch and the model registry (transitively monai and torchio). Cold,
that is tens of seconds. ``import spectramr``, ``spectramr --help`` and the
lightweight verbs (``doctor``, ``campaign``, ``regulatory``, ``launch``) never
import torch and stay fast.

Before dispatching a heavy verb, one line goes to **stderr** — never stdout, so
``audit --json | jq`` stays parseable::

   ⏳ spectramr audit: importing PyTorch + model registry (first call in a fresh process is slow, ~30–60 s)…

Silence it in batch jobs with ``SPECTRAMR_QUIET=1`` or
``SPECTRAMR_SUPPRESS_CLINICAL_WARNING=1`` (the switch that silences the clinical
banner silences this too).
