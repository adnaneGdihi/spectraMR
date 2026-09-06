.. _campaigns_user_guide:

==============================================================================
Experiment Campaigns — User Guide
==============================================================================

.. contents:: Table of Contents
   :depth: 3
   :local:

What is a campaign?
===================

A **campaign** is a manifest that groups related experiments under a single
research question, with shared evaluation protocol and SLURM defaults.
The orchestrator submits every arm in parallel (one ``sbatch`` each), as a
single SLURM **job array** (``execution: array`` — one tracked job id for the
whole cohort, see :ref:`campaigns_array_mode`), or sequentially with
``--dependency=afterok`` chaining; it then polls SLURM for status, gathers
per-arm metrics, and produces a leaderboard + dashboard.

The five things a campaign manifest does:

1. **Names** every arm and points at its trainer YAML.
2. **Tags** each arm so the leaderboard can group / filter by role,
   architecture, ablated component, etc.
3. **Sets SLURM defaults** for the whole campaign and lets each arm
   override the defaults (more time, more memory, multi-GPU).
4. **Declares the evaluation protocol** (metrics, bootstrap,
   significance correction, plots, LaTeX export).
5. **Optionally chains stages** so one arm's checkpoint warm-starts
   the next arm.

The orchestrator is the existing
:class:`spectramr.infrastructure.orchestration.campaign_orchestrator.CampaignOrchestrator`,
the schema is
:class:`spectramr.config.schemas.campaign.CampaignConfigSchema`, and the
SLURM backend is
:class:`spectramr.infrastructure.orchestration.slurm_backend.SLURMBackend`.

.. note::

   The ``#SBATCH`` directive header of each per-arm job script is rendered by
   the shared :class:`spectramr.infrastructure.execution.SlurmBackend` (the same
   generator the unified ``spectramr launch`` uses), and both the launcher and the
   campaign submit through one ``sbatch`` primitive, so the directive format and
   submission mechanics live in one place. A golden-file test pins the generated
   script byte-for-byte. See :doc:`execution_modes`.

Where arms run (``--where``)
----------------------------

``spectramr campaign submit C.yaml --where {slurm,docker,apptainer}`` selects where
each arm executes. ``slurm`` (default) submits an sbatch job per arm with full
status polling and ``afterok`` chaining. ``docker`` / ``apptainer`` instead run
each arm in a container (``spectramr train --config <arm> -O training.output_dir=…``)
**synchronously** — arms run one at a time, so this suits a small parallel-mode
campaign on a single host; sequential (stage-group) campaigns still require
``--where slurm``.

Anatomy of a campaign manifest
==============================

Every campaign YAML lives at :file:`experiments/campaigns/<name>.yaml`
and conforms to ``CampaignConfigSchema``. Minimal valid example:

.. code-block:: yaml

   name: my_campaign
   description: |
     One-paragraph statement of what this campaign tests and what
     "success" looks like.

   task: experiments/training/tasks/task_field_translation.yaml
   test_manifest: data/manifests/ulf_paired_brain_v5.json

   execution: parallel       # 'parallel' | 'array' | 'sequential'

   slurm_defaults:
     partition: gpu
     time: "48:00:00"
     mem: "64GB"
     cpus_per_task: 8
     gpus: 1
     nodes: 1
     ntasks: 1
     mail_type: "END,FAIL"

   evaluation:
     metrics: [psnr, ssim, lpips]
     n_bootstrap: 10000
     confidence: 0.95
     correction_method: fdr
     compute_uncertainty: true
     generate_plots: true
     export_latex: true
     per_sample_inference: true
     batch_size: 4

   experiments:
     - name: my_baseline
       config: experiments/active/my_baseline.yaml
       role: baseline
       tags:
         arch: unet_recon
     - name: my_variant
       config: experiments/active/my_variant.yaml
       role: variant
       slurm_overrides:
         time: "96:00:00"
         mem: "96GB"
       tags:
         arch: my_new_model

Required fields
---------------

* ``name`` — campaign identifier; doubles as the auto-derived
  ``output_dir = experiments/results/campaigns/<name>``.
* ``experiments`` (parallel mode) or ``stage_groups`` (sequential).
* For each arm: ``name``, ``config``, ``role`` (one of ``baseline``,
  ``variant``, ``ablation``).

Optional but recommended
------------------------

* ``description`` — a paragraph that records *why* the campaign exists
  and the success criterion. The orchestrator prints this when you
  submit, and it travels into the dashboard.
* ``test_manifest`` — a single shared test split. Per-arm dataset
  configurations override the train / val splits but the test
  inference uses this manifest so leaderboard numbers are comparable.
* ``slurm_defaults`` and per-arm ``slurm_overrides`` — see
  :ref:`campaigns_slurm`.
* ``evaluation`` — see :ref:`campaigns_evaluation`.
* ``ablation_axes`` — see :ref:`campaigns_ablation_axes`.
* ``tags`` (per arm) — must be ``dict[str, str]``; values that look
  like booleans or numbers should be quoted (``"true"``, ``"10"``)
  to satisfy the schema validator.

Forbidden fields
----------------

The schema is ``extra="forbid"``. The orchestrator will reject any
top-level field outside the documented set. If you tried to put
``data:``, ``model:`` or ``losses:`` blocks into a campaign manifest
those belong inside the **per-arm trainer YAML**, not the campaign.

Roles and the "semantic role" tag
=================================

The schema's ``role`` field is a strict ``Literal["baseline",
"variant", "ablation"]`` because the leaderboard uses it to drive
the per-row colouring and the ablation-table baseline column. If
you want a finer grouping ("reference", "specialist",
"unified_multicontrast", "partial_promotion"), put it in
``tags.semantic_role`` instead:

.. code-block:: yaml

   - name: geomamba_ulf_v1_p2
     config: experiments/inprogress/geomamba_ulf/geomamba_ulf_v1_p2.yaml
     role: variant                       # schema literal
     tags:
       semantic_role: unified_multicontrast   # free-form
       contrasts: T1w_T2w_FLAIR

The orchestrator and evaluator carry tags through unchanged, so
``tag.semantic_role`` is filterable and groupable downstream.

.. _campaigns_slurm:

Per-arm SLURM overrides
=======================

Use ``slurm_overrides`` on heavy arms to bump time / memory / GPU
count *without* touching the campaign-level defaults:

.. code-block:: yaml

   - name: baseline_latent_diffusion_3d
     config: experiments/inprogress/geomamba_ulf/baselines/baseline_latent_diffusion_3d.yaml
     role: baseline
     slurm_overrides:
       mem: "96GB"
       time: "96:00:00"
     tags:
       arch: latent_diffusion_hierarchical

The merge order is
``slurm_defaults -> slurm_overrides``, so an override key (``mem``)
replaces the default; unspecified keys inherit.

.. _campaigns_array_mode:

Array campaigns (one SLURM job array for the whole cohort)
==========================================================

``execution: parallel`` submits **one ``sbatch`` per arm**. For a large
cohort (dozens of arms) that is a serial loop of blocking ``sbatch`` calls
on the login node — slow, opaque (no single job id to watch), and prone to
stalling against the scheduler. ``execution: array`` collapses that into a
**single ``sbatch --array`` call**:

.. code-block:: yaml

   execution: array
   array_concurrency: 8      # the '%N' throttle on --array=0-(M-1)%N

   experiments:              # the SAME flat list as parallel mode
     - name: arm_a
       config: experiments/inprogress/x/arm_a.yaml
       role: variant
     - name: arm_b
       config: experiments/inprogress/x/arm_b.yaml
       role: variant

On submit the orchestrator:

1. Freezes :file:`<campaign_dir>/array_manifest.txt` — one resolved config
   path per surviving arm, in array-index order.
2. Makes **one** ``sbatch --array=0-(M-1)%array_concurrency`` call, so the
   whole cohort is a single tracked **array job id** (``%N`` caps how many
   tasks run at once).
3. Each array task resolves *its* config (manifest line =
   ``$SLURM_ARRAY_TASK_ID``) via
   :mod:`spectramr.cli.manifest_dispatch`, runs the Tier-0/1 audit pre-flight,
   then trains it as the real experiment (``--prod`` = full config
   ``max_iterations``), routing output into
   :file:`<campaign_dir>/<config-stem>` so ``_discover_results`` finds the
   checkpoints.

Tracking is intentionally minimal (matching the "keep one job id" intent):
every arm in ``campaign_state.json`` shares the single array job id and is
distinguished by ``array_task_id`` (the sacct element is
``<job_id>_<array_task_id>``). ``campaign status`` / ``watch`` poll that one
job; ``campaign evaluate`` discovers each arm's results from its recorded
``output_dir``.

Constraints (all fail loud, never silently):

* ``--where`` must be ``slurm`` — a job array is a scheduler construct with
  no container equivalent (use ``execution: parallel`` for container fan-out).
* No two arms may share a config **file stem** — they would write the same
  :file:`<campaign_dir>/<stem>` output dir. The orchestrator raises at submit
  (rename one config).
* Missing configs are recorded ``failed`` and excluded from the array, so the
  task indices stay contiguous; ``--only`` / ``--include`` / ``--exclude``
  filters and campaign ``--resume`` apply exactly as in parallel mode.

The standalone array dispatcher
(:file:`scripts/training/submit_experiment_array.sh`) is the lower-level,
campaign-free path over a directory of YAMLs; ``execution: array`` is the same
mechanism wired into the campaign lifecycle (state, roles, evaluation).

Sequential campaigns and checkpoint chaining
============================================

For multi-stage workflows (e.g. pretrain VAE → train LDM → finetune)
use ``execution: sequential`` and group the arms into ordered
``stage_groups``:

.. code-block:: yaml

   execution: sequential

   stage_groups:
     - name: stage_p2
       experiments:
         - name: geomamba_ulf_v1_p2
           config: experiments/inprogress/geomamba_ulf/geomamba_ulf_v1_p2.yaml
           role: variant

     - name: stage_p3_promote
       depends_on: [stage_p2]
       experiments:
         - name: geomamba_ulf_v1_p3
           config: experiments/inprogress/geomamba_ulf/geomamba_ulf_v1_p3.yaml
           role: variant
           checkpoint_from:
             stage: stage_p2
             experiment: geomamba_ulf_v1_p2
             inject_as: model.checkpoint_path
           tags:
             promotion_step: "5_all_active"

The orchestrator submits every arm in a stage group in parallel,
then chains the next group via SLURM ``--dependency=afterok``.
``checkpoint_from`` resolves the parent arm's best checkpoint at
submission time and injects the path via ``-O <inject_as>=<path>``.
``inject_as`` defaults to ``model.checkpoint_path`` but you can
target any nested config key (e.g.
``training.multi.stages.0.checkpoint_path``).

.. _campaigns_ablation_axes:

Ablation axes (parameter sweeps)
================================

For HPO grids, declare ``ablation_axes`` on the campaign instead of
hand-writing one trainer YAML per grid point. Each axis is a
``config_path`` + a list of ``values``; the cross-product is
materialised into per-trial configs by
:class:`spectramr.infrastructure.orchestration.ablation_config_generator.AblationConfigGenerator`.

.. code-block:: yaml

   experiments:
     - name: geomamba_ulf_v0_reference
       config: experiments/inprogress/geomamba_ulf/geomamba_ulf_v0.yaml
       role: baseline                        # acts as the base config

   ablation_axes:
     - config_path: training.geomamba_ulf.metric_sfc.beta
       values: [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]
       label: metric_sfc_beta

     # Adding a second axis creates a 6 x 4 = 24-trial grid.
     - config_path: losses.image_losses[1].weight
       values: [0.0, 0.05, 0.10, 0.20]
       label: ph_w2_weight

The reference arm above acts as the base config the axes mutate.
For real examples in the repo see
:file:`experiments/campaigns/geomamba_ulf_hpo_metric_sfc_beta.yaml`
(single 6-value axis) and
:file:`experiments/campaigns/geomamba_ulf_hpo_topology_weights.yaml`
(joint 4 x 4 grid).

.. _campaigns_evaluation:

Evaluation protocol
===================

The ``evaluation`` block configures the post-training comparison
that runs when you call ``python -m spectramr.cli campaign evaluate``:

.. code-block:: yaml

   evaluation:
     metrics: [psnr, ssim, lpips]
     n_bootstrap: 10000           # bootstrap samples for CI
     confidence: 0.95             # 95% confidence interval
     significance_level: 0.05
     correction_method: fdr       # 'fdr' or 'bonferroni' for multiple-comparison
     compute_uncertainty: true
     generate_plots: true         # writes PNG dashboard
     export_latex: true           # writes leaderboard.tex
     per_sample_inference: true   # re-run each arm on test_manifest for matched-sample stats
     batch_size: 4                # inference batch size

The evaluator computes per-metric leaderboards with bootstrap
confidence intervals, runs pairwise significance tests with the
chosen correction, and (when ``generate_plots: true``) renders a
small dashboard via
:class:`spectramr.infrastructure.orchestration.campaign_report_generator.CampaignReportGenerator`.

Submitting and monitoring a campaign
====================================

The CLI surface is :program:`python -m spectramr.cli campaign <action>`:

.. code-block:: bash

   source .venv/bin/activate

   # 1. Validate without submitting (recommended first step).
   python -m spectramr.cli campaign submit \
       experiments/campaigns/geomamba_ulf_super_resolution.yaml \
       --dry-run

   # 2. Submit every arm in parallel.
   python -m spectramr.cli campaign submit \
       experiments/campaigns/geomamba_ulf_super_resolution.yaml

   # 3. Poll SLURM and update the per-arm status table.
   python -m spectramr.cli campaign status \
       experiments/results/campaigns/geomamba_ulf_super_resolution

   # 4. Watch every 60 s until everything is in a terminal state,
   #    then auto-evaluate.
   python -m spectramr.cli campaign watch \
       experiments/results/campaigns/geomamba_ulf_super_resolution

   # 5. Manually trigger evaluation (idempotent).
   python -m spectramr.cli campaign evaluate \
       experiments/results/campaigns/geomamba_ulf_super_resolution

   # 6. Cancel every active SLURM job for the campaign.
   python -m spectramr.cli campaign cancel \
       experiments/results/campaigns/geomamba_ulf_super_resolution

The state file ``campaign_state.json`` lives at the campaign root
and tracks SLURM job IDs, statuses, exit codes, and discovered
checkpoints for every arm.

Filtering: only / include / exclude
===================================

.. _campaigns_filtering:

The ``submit`` action accepts three filter flags so you can re-run
a subset without rewriting the manifest:

.. code-block:: bash

   # Re-submit only one arm by name:
   python -m spectramr.cli campaign submit campaign.yaml \
       --only baseline_swinir

   # Comma-separated list:
   python -m spectramr.cli campaign submit campaign.yaml \
       --only geomamba_ulf_v0,geomamba_ulf_v0_multicontrast

   # Restrict by tag (all ablation arms with tag.arch=geo_mamba):
   python -m spectramr.cli campaign submit campaign.yaml \
       --include role=ablation \
       --include tag.arch=geo_mamba

   # Drop the most expensive arm:
   python -m spectramr.cli campaign submit campaign.yaml \
       --exclude name=baseline_latent_diffusion_3d

The selector grammar is ``key=value`` with three recognised keys:

* ``name=<arm-name>``     — exact match against ``CampaignExperimentSchema.name``
* ``role=<role>``         — ``baseline``, ``variant``, or ``ablation``
* ``tag.<key>=<value>``   — exact match against ``tags[<key>]``

Combination rules:

* ``--only`` is an arm-name allowlist (comma-separated; flag
  repeatable). Combines with ``--include`` and ``--exclude`` via
  logical AND.
* Multiple ``--include`` selectors combine via OR: an arm is kept
  if **any** include matches.
* Multiple ``--exclude`` selectors combine via OR: an arm is dropped
  if **any** exclude matches.
* The composite is ``(only) AND (any include) AND NOT (any exclude)``.

Filters apply to both ``parallel`` campaigns and to every stage
group of a ``sequential`` campaign — handy for re-running a single
stage after a partial failure.

Per-arm artifact contract
=========================

When training finishes, every arm writes a
``final_metrics.json`` to its output directory with this schema:

.. code-block:: json

   {
     "schema_version": "1",
     "experiment_name": "geomamba_ulf_v0",
     "best": {
       "val_psnr_best": 31.42,
       "val_ssim_best":  0.918,
       "val_lpips_best": 0.083,
       "g_total_loss_best": 0.214
     },
     "final_loss": 0.219,
     "early_stopping_best_value": 31.42,
     "early_stopping_best_iteration": 78500,
     "csv_log_file": "experiments/results/campaigns/<name>/<arm>/metrics_history.csv"
   }

"Best" classification:

* **Higher-is-better** keys (key contains ``psnr``, ``ssim``,
  ``accuracy``, ``dice``, ``f1``, or ``iou``) take the maximum.
* All other keys (loss, lpips, mse, mae, rmse, ...) take the minimum.

The campaign aggregator and report generator read this file
directly — adding a new arm to a leaderboard requires no plumbing
beyond the file appearing in the right place.

Artifact layout
===============

For ``execution: parallel``:

.. code-block:: text

   experiments/results/campaigns/<campaign-name>/
   ├── campaign_state.json          # CampaignState snapshot (SLURM IDs, statuses, ...)
   ├── <arm-name>/
   │   ├── slurm_<jobid>.out / .err
   │   ├── checkpoints/
   │   │   ├── best.pt
   │   │   └── model_iter_*.pt
   │   ├── logs/
   │   ├── metrics_history.csv
   │   ├── final_metrics.json       # per-arm aggregator contract
   │   └── inference_test_split/    # auto-inference outputs (if test_manifest set)
   ├── leaderboard.csv              # written by `campaign evaluate`
   ├── leaderboard.tex              # if export_latex is true
   ├── dashboard/                   # PNG plots (if generate_plots is true)
   │   ├── psnr_by_role.png
   │   └── pareto_frontier.png
   └── reports/

For ``execution: sequential`` the layout is identical except that
each stage group's job-IDs in ``campaign_state.json`` carry a
``depends_on`` field pointing at the prior stage.

Cookbook
========

Re-run a single ablation after fixing a bug
-------------------------------------------

.. code-block:: bash

   python -m spectramr.cli campaign submit \
       experiments/campaigns/geomamba_ulf_super_resolution.yaml \
       --only abl_no_metric_sfc \
       --resume

Skip the most expensive baselines while iterating
-------------------------------------------------

.. code-block:: bash

   python -m spectramr.cli campaign submit \
       experiments/campaigns/geomamba_ulf_super_resolution.yaml \
       --exclude name=baseline_latent_diffusion_3d \
       --exclude name=baseline_swin_unetr_3d

Re-run only the partial-promotion arms of a sequential P3 campaign
------------------------------------------------------------------

.. code-block:: bash

   python -m spectramr.cli campaign submit \
       experiments/campaigns/geomamba_ulf_p3_promotion.yaml \
       --include role=variant \
       --exclude name=geomamba_ulf_v1_p3

Sweep one knob without writing per-trial YAMLs
----------------------------------------------

Add an ``ablation_axes:`` block to the campaign and submit:

.. code-block:: yaml

   ablation_axes:
     - config_path: training.geomamba_ulf.metric_sfc.beta
       values: [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]

Worked example: GeoMamba-ULF
============================

The repo ships five campaign manifests for the GeoMamba-ULF paradigm:

.. list-table::
   :header-rows: 1
   :widths: 35 12 53

   * - Campaign
     - Mode
     - What it tests
   * - :file:`geomamba_ulf_super_resolution.yaml`
     - parallel
     - Full v0 leaderboard: 3 reference + 6 ablation + 7 baseline arms
       (vanilla 3D U-Net, vanilla Mamba-UNet, Hilbert-Mamba-UNet,
       LF-SynthSR, SwinIR, Swin-UNETR-3D, latent diffusion).
   * - :file:`geomamba_ulf_p2_specialists.yaml`
     - parallel
     - P2 milestone: unified multi-contrast model vs T1w / T2w /
       FLAIR specialists. Validates the per-contrast log-σ
       uncertainty weighting.
   * - :file:`geomamba_ulf_p3_promotion.yaml`
     - sequential
     - P3 milestone: chains stage_p2 → stage_partial_promotion →
       stage_p3_promote, with the P3 model warm-starting from the
       P2 checkpoint via ``checkpoint_from``.
   * - :file:`geomamba_ulf_hpo_metric_sfc_beta.yaml`
     - parallel
     - 6-trial sweep over the metric-SFC β coefficient (axis-driven).
   * - :file:`geomamba_ulf_hpo_topology_weights.yaml`
     - parallel
     - Joint 4 × 4 sweep over (PH-W2 weight, Beltrami weight).

Recommended order:

1. Smoke-test one arm: ``python -m spectramr.cli train --config
   experiments/inprogress/geomamba_ulf/geomamba_ulf_v0.yaml --dry_run``.
2. ``--dry-run`` the full campaign:
   ``python -m spectramr.cli campaign submit
   experiments/campaigns/geomamba_ulf_super_resolution.yaml --dry-run``.
3. Submit a small subset to check the SLURM plumbing:
   ``--only geomamba_ulf_v0,baseline_unet_recon``.
4. Submit the full campaign once those land cleanly.
5. Watch and auto-evaluate:
   ``python -m spectramr.cli campaign watch
   experiments/results/campaigns/geomamba_ulf_super_resolution``.

Lightweight alternative: one SLURM array task per experiment
============================================================

When you just want to fan a *flat directory of experiment YAMLs* out
across the cluster — without authoring a campaign manifest, leaderboard
evaluation, or dependency staging — use the dispatch array instead of
the full orchestrator. One SLURM **array task trains exactly one
experiment**.

Two files under ``scripts/training/``:

* ``submit_experiment_array.sh`` — snapshots the experiment set into a
  timestamped manifest (so the array range can't drift from a
  re-globbed directory) and submits ``sbatch --array=0-(N-1)%C``.
* ``dispatch_experiments.sbatch`` — the array-task body: reads its
  config from line ``$SLURM_ARRAY_TASK_ID + 1`` of the manifest, runs
  the strict Tier-0/1 ``spectramr audit`` pre-flight (skips training with
  exit 2 if the config is rejected — no wasted GPU), then
  ``spectramr train``.

Usage::

    # All 40 Virtual-Fiducial arms, 6 running at a time:
    scripts/training/submit_experiment_array.sh

    # A different cohort (directory → recursive *.yaml, or a quoted glob):
    scripts/training/submit_experiment_array.sh experiments/inprogress/diffusion
    scripts/training/submit_experiment_array.sh 'experiments/inprogress/vf/exp_vf_2*.yaml'

    # Tune behaviour via env:
    CONCURRENCY=12 RESUME=1 scripts/training/submit_experiment_array.sh
    DRY_SUBMIT=1 scripts/training/submit_experiment_array.sh    # print plan, don't submit

    # Smoke-style short run: cap every arm at 100 iterations:
    TRAIN_ITERS=100 scripts/training/submit_experiment_array.sh

    # Focus a launch on training arms — drop ablation/baseline arms from
    # the manifest (fnmatch glob over each YAML's path AND basename; a bare
    # substring is wrapped as *substring*; space/comma-separated = repeatable).
    # Either as a LEADING env var, or as the --exclude flag (order-independent):
    EXCLUDE='*ablation*' scripts/training/submit_experiment_array.sh experiments/inprogress/kspace_filling
    EXCLUDE='ablation,baseline' scripts/training/submit_experiment_array.sh experiments/inprogress/<cohort>
    scripts/training/submit_experiment_array.sh experiments/inprogress/kspace_filling --exclude '*ablation*'

.. warning::

   **An ``EXCLUDE=…`` placed *after* the script name does nothing** —
   ``submit_experiment_array.sh <cohort> EXCLUDE=ablation`` ran the ablation
   arms anyway. The shell only treats ``KEY=value`` as an *environment
   assignment* when it **leads** the command; trailing, it is an inert
   positional argument. The wrapper now **rejects** such a misplaced knob loudly
   (exit 2, pointing at the two correct forms) instead of silently submitting
   the un-filtered cohort — the launcher-side guard against the silent-no-op
   trap (``.claude/rules/pitfalls.md`` #10/#15). Reach for the ``--exclude``
   flag when you want a position-independent form.

.. note::

   **The exclusion filter is the array analogue of**
   ``spectramr audit <dir> --exclude PATTERN``, exposed two equivalent ways: the
   leading ``EXCLUDE=PATTERN`` env var and the ``--exclude PATTERN`` CLI flag
   (``--exclude=PATTERN`` also accepted, repeatable, and **unioned** with the
   env var — neither silently overrides the other). It filters the snapshotted
   manifest *after* it is built and *before* the array range is computed, so
   the ``0-(N-1)%C`` range always matches the arms actually submitted. Each
   pattern is an ``fnmatch`` glob tested against both the YAML's
   path *and* its basename — matching the path (not just the filename) is what
   lets ``*ablation*`` catch arms nested under an ``ablations/`` subdirectory
   whose own filename has no "ablation" token, exactly like the bulk audit
   filter. A bare substring (no ``*``/``?``/``[``) is wrapped as ``*substring*``
   so ``ablation`` is intuitive; multiple env patterns are
   space- or comma-separated (the flag is repeatable). The skipped count is
   printed (non-silent), and if the filter empties the manifest the FATAL-empty
   guard still fires rather than submitting an empty array. This is the
   launcher-side complement to the campaign runner's semantic
   ``--exclude role=ablation`` selector.

.. note::

   **``TRAIN_ITERS=N`` caps the validation interval too, not just the
   iteration budget.** A smoke-style ``TRAIN_ITERS=100`` overrides
   ``training.max_iterations=100`` **and** ``validation.eval_interval`` down to
   the effective iteration count. Capping iterations alone is a silent trap:
   validation is interval-driven (``iteration % eval_interval == 0``) with no
   guaranteed end-of-run pass, and most VF arms set ``eval_interval: 2500`` — so
   a 100-iteration run would *never reach a validation step* and would silently
   skip the val path it exists to exercise (validation-time OOM, identity
   collapse — ``.claude/rules/pitfalls.md`` #10). With the eval-interval cap,
   validation fires at least once, on the final step.

   Both are **caps, not forced values**. A 1-shot calibration arm that already
   declares a smaller ``max_iterations`` (e.g. ``exp_vf_conformal_calib``:
   ``max_iterations: 1``, ``eval_interval: 1``) is left untouched — never
   inflated to ``N`` (the regression behind ``dispatch 20260603_200125`` task
   48, where a 1-shot arm was inflated into a 9 h run). An arm whose
   ``eval_interval`` already fires inside the capped run (e.g.
   ``experiment_vf_tto``'s ``25``) is also left untouched. The per-arm
   ``max_iterations:`` reader strips trailing ``# comments`` before parsing, so
   a numeric comment (``max_iterations: 1  # … CW-6``) can't be misread as
   ``116`` and defeat the cap-not-inflate guard. Inspect the resolved plan
   without submitting via ``DRYRUN=1`` on the dispatcher (the unit tests pin
   this).

Each task writes ``experiments/results/dispatch/<TS>/`` (manifest,
per-arm audit JSON, ``slurm_*_<jobid>_<taskid>.{out,err}``); training
checkpoints/metrics still go to each YAML's own ``training.output_dir``.
Resource directives match the orchestrator's per-arm jobs
(``--account=<your-slurm-account> --gres=gpu:1 --cpus-per-task=8 --mem=64GB``). GPUs
are allocated **via** ``--gres`` — this cluster has no dedicated ``gpu``
partition, so neither the dispatcher nor ``SLURMBackend.generate_job_script``
emits a ``--partition`` directive by default (SLURM uses the default
partition). Pin one only if your cluster requires it:
``SLURM_PARTITION=<name> scripts/training/submit_experiment_array.sh`` (or
``slurm_params={"partition": "<name>"}`` for the orchestrator). The
``--account=<your-slurm-account>`` allocation and ``--mail-user=${USER}@<your-institution>.edu``
match the cluster scheduling standard used by ``scripts/ci/*.sbatch`` and the
campaign orchestrator. Override the account for a different allocation with
``SLURM_ACCOUNT=<acct> scripts/training/submit_experiment_array.sh``
(forwarded as the ``sbatch --account`` flag, which wins over the ``#SBATCH``
directive).

.. note::

   **The project ``.venv`` is mandatory and fail-loud.** Each array task
   activates ``${REPO_ROOT}/.venv`` (override with ``VENV_PATH=/abs/path``)
   and then probes ``import torch`` *before* the audit. If the venv is
   missing the task aborts with exit 3 and a clear message — it never
   silently runs the audit against the system python. The dispatcher does
   **not** ``module load torchvision`` (unlike ``scripts/ci/*.sbatch``,
   which have no venv): the CUDA-pinned wheels in the venv are
   self-contained, and a module-provided torch on ``PYTHONPATH`` would
   shadow the venv's torch under the venv interpreter, producing the
   cryptic ``It appears that PyTorch has loaded the torch/_C folder of the
   PyTorch repository`` import error. If a task dies that way, the venv was
   not activated (wrong ``VENV_PATH``) or has no/partial torch — fix it
   with ``pip install -e '.[dev]'`` inside the venv.

.. note::

   **Repo-root anchoring (why ``FATAL: config file missing`` happened).**
   ``sbatch`` does not run the script from its on-disk location — SLURM
   copies the script body into its spool (e.g.
   ``/var/spool/slurmd/job*/slurm_script``) and the compute node executes
   *that copy*. So ``${BASH_SOURCE[0]}`` is **not**
   ``scripts/training/dispatch_experiments.sbatch`` at run time, and any
   ``cd "$(dirname "${BASH_SOURCE[0]}")/../.."`` lands in the wrong
   directory. Because the manifest stores **relative** config paths
   (``experiments/inprogress/vf/…``), every arm then looked missing even
   though the manifest itself (passed as an absolute path) read fine.

   The dispatcher now anchors ``REPO_ROOT`` on, in order: (1) the
   ``REPO_ROOT`` the wrapper exports via ``--export``, (2)
   ``SLURM_SUBMIT_DIR`` (the cwd ``sbatch`` ran from — the wrapper ``cd``\ s
   to the repo root first), and only then (3) the ``BASH_SOURCE``
   derivation (direct ``bash`` runs / unit tests). If you ever ``sbatch``
   the file by hand, submit **from the repo root** or pass
   ``--export=ALL,REPO_ROOT="$PWD",EXP_MANIFEST=…``. The header's
   ``GPU:`` field now probes ``SLURM_JOB_GPUS`` /
   ``SLURM_GPUS_ON_NODE`` / ``CUDA_VISIBLE_DEVICES`` (older SLURM leaves
   ``$SLURM_GPUS`` unset, which is why it printed ``GPU: ?`` — the GPU
   was always allocated by ``--gres=gpu:1 --partition=gpu``).

**When to prefer the campaign orchestrator instead:** cross-arm
leaderboards, shared test-set manifests, multi-stage dependencies
(teacher → student), or auto-evaluation. The array is purely "train
each of these N configs, one GPU each".

Regression coverage:
``tests/unit/scripts/test_dispatch_experiments.py``.

Known gaps and roadmap
======================

* **Per-arm job entrypoint.** The orchestrator's generated SLURM
  scripts call the canonical ``python -m spectramr.cli train`` (and
  ``predict`` for auto-inference). The pre-refactor ``python
  src/main.py`` form was removed in the src→spectramr migration; any
  campaign YAML or fork still expecting it will not match.
* **Stale shipped campaigns.** Some checked-in campaign manifests
  (e.g. ``experiments/campaigns/kspace_recon_shootout.yaml``) reference
  experiment YAMLs that were since renamed/removed —
  ``tests/smoke/test_campaign_smoke.py`` flags the missing configs.
  Re-point or prune those arms before submitting.
* **TensorRT-LLM cluster edge path.** The repo ships a TorchScript /
  ONNX / INT8 ``EdgeExporter`` but the cluster-side TensorRT
  custom-op path for Mamba is still external.
* **Real WMH segmenter.** The current
  :class:`spectramr.core.metrics.wmh_dice_evaluator.LazySegmenterAdapter`
  is a hyper-intensity threshold. Integrating LST-AI / SynthSeg-WMH
  is a one-evening job once the dependencies are in place.
* **Multi-seed / replicate handling.** The schema does not yet
  support ``seed_replicas: N`` to fan out N independent seeds per
  arm. Today this is done by listing N arms with different
  ``seed`` overrides.
