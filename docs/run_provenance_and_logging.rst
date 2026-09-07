Run Provenance & Traceability
=============================

Every training run started through :func:`spectramr.pipelines.train.run_training_pipeline`
(i.e. ``spectramr train`` / ``sanity_check`` / ``experiment`` / ``ablation``)
captures a **provenance record** — *what code, what machine, when, how long,
how big* — so a result on the cluster can always be tied back to the exact
commit and environment that produced it. A bundle is self-describing: "why
won't this reproduce?" is answerable from the artifacts alone.

What gets captured
------------------

The canonical capture lives in
:mod:`spectramr.infrastructure.logging.provenance` (the single source of truth,
also consumed by ``spectramr doctor``). Every helper is **fail-open** — a missing
``git`` binary, a broken torch, or an un-dumpable config degrades a single
field, never the run.

============================  ===================================================
Field                          Meaning
============================  ===================================================
``run_id``                     Correlation id ``<name>-<YYYYmmdd_HHMMSS>-<gitsha>``
``git``                        commit SHA (+ short), branch, **dirty flag**, subject, time
``env``                        python, platform, hostname, pid, user, torch/CUDA/devices
``env.node``                   **node hardware** — GPU count/type, cores, RAM (below)
``config_sha256``              12-char hash of the *resolved* config (post defaults/migration)
``model``                      total / trainable / **frozen** params + fp32 size (MB)
``data``                       train / val / test batch counts
``batch``                      effective batch = per-device × grad-accum × world-size
``slurm``                      ``SLURM_JOB_ID`` / nodelist / partition / **cores / memory / GPU grant**
``started_at`` / timing        ISO start, wall-clock duration, iterations/sec
============================  ===================================================

.. note::

   ``git.dirty`` is the most important traceability flag. A "clean" SHA alone
   does **not** reproduce a run whose working tree had uncommitted changes —
   ``dirty: true`` tells you the SHA is necessary but not sufficient.

Node hardware: allocated vs. physical
-------------------------------------

``env.node`` answers *what machine did this actually run on*. It has three
sub-records, each probed independently and each fail-open:

.. code-block:: json

   "node": {
     "cpu": {
       "model": "AMD EPYC 7763 64-Core Processor",
       "physical_cores": 64, "logical_cores": 128,
       "affinity_cores": 8, "allocated_cores": 8,
       "cgroup_quota_cores": null, "usable_cores": 8,
       "max_frequency_mhz": 3529.0
     },
     "memory": {
       "total_gb": 1007.5, "available_gb": 812.3, "swap_total_gb": 8.0,
       "cgroup_limit_gb": null, "allocated_gb": 64.0, "usable_gb": 64.0
     },
     "gpu": {
       "count": 2, "types": {"NVIDIA A100-SXM4-40GB": 2},
       "total_mem_gb": 80.0, "cuda_available": true,
       "visible_devices": "0,1", "allocated_count": 2,
       "driver_version": "550.54.15",
       "devices":      [{"index": 0, "name": "...", "capability": "8.0",
                         "multi_processor_count": 108, "uuid": "GPU-..."}],
       "node_count": 4, "node_types": {"NVIDIA A100-SXM4-40GB": 4},
       "node_devices": [{"index": 0, "name": "...", "uuid": "GPU-..."}]
     }
   }

Two distinctions carry the weight:

**Allocated ≠ physical.**
    ``logical_cores``/``total_gb`` describe the *node*; ``usable_cores``/
    ``usable_gb`` describe **this run**. The latter is a floor over every
    constraint that actually binds — the affinity mask (``os.sched_getaffinity``,
    which sees cpusets and ``taskset``), the scheduler grant
    (``SLURM_CPUS_PER_TASK`` / ``SLURM_MEM_PER_NODE``), the container cgroup
    quota, and the node total. Without it a job granted 8 of 128 cores produces
    a record indistinguishable from one that owns the whole machine, and
    "why did this OOM / run slow?" has no answer in the bundle.

**Visible ≠ present.**
    ``count``/``types`` are what torch can use (the ``CUDA_VISIBLE_DEVICES``
    subset); ``node_count``/``node_types`` come from ``nvidia-smi`` and describe
    every GPU on the box. The gap separates *this node has no GPU* from *this
    job was given none of its 4* — the distinction that matters when auditing
    the accelerated-run contract (:doc:`accelerated_run_contract`). When
    ``count == 0`` but
    ``node_count > 0``, the banner says so explicitly:

    .. code-block:: text

        gpu        : 0 visible / 4 on node · driver 550.54.15 · NOT USABLE BY TORCH

    which is a broken driver or a CPU-only torch wheel, **not** a CPU node.

.. note::

   The cgroup fields exist because ``psutil`` reports the *host's* CPUs and RAM
   from inside a container — a run that OOM'd under an 8 GB Docker limit (the
   ``spectramr launch`` docker backend) otherwise looks like it had 1 TB.
   ``cgroup_limit_gb`` / ``cgroup_quota_cores`` are ``null`` when unlimited or
   when no cgroup applies.

Device enumeration has exactly one site:
:func:`~spectramr.infrastructure.logging.provenance.torch_runtime`. The GPU census
consumes its result rather than re-probing, so the ``doctor`` view and the
provenance view cannot drift.

Where it lands
--------------

Self-describing artifacts are written into the run's ``output_dir`` — two
stamped early, and a completion footer (``run_summary.json``) documented after
the ``_ledger`` block below:

``provenance.json``
    The full record, stamped **early** (right after the output dirs are
    created, before the training loop). A crashed run still leaves this
    forensic trace.

``resolved_config.json``
    ``config.model_dump(mode="json")`` — the frozen ``TrainingSettings`` as it
    actually drove the run (after defaults + migrations), so post-hoc tooling
    needn't re-parse the source YAML. Carries a ``_ledger`` block (below) and a
    ``_declared`` block: the unset-excluded dump, the one
    shape that ``TrainingSettings.model_validate`` turns back into the resolved
    settings (the full dump does not re-validate, because cross-field
    validators read a materialised default as a declaration). ``predict`` and
    ``infer`` rebuild their settings from that block when the artifact sits
    beside the checkpoint.

``provenance_run_<run_id>.json`` / ``resolved_config_run_<run_id>.json``
    Run-id-qualified **copies** of the two above, written only when a ``run_id``
    is known — the ``spectramr audit`` path has none and so writes neither.
    They exist because the unqualified names are overwrite-on-launch:
    ``output_dir`` derives from the config, not from the run, so relaunching an
    arm into the same directory replaces the previous run's record while its
    checkpoints, images and debug snapshots all survive beside it. The result
    result is a directory whose config describes a *different* run than its
    images — self-consistent, parseable, and wrong. Build the name with
    :func:`~spectramr.infrastructure.validation.resolved_config_artifact.resolved_config_run_name`
    rather than re-deriving the spelling at the reading site.

The ``_ledger`` block
---------------------

``resolved_config.json`` shows what the run used; ``_ledger`` shows how that
differs from what the YAML *said*. It is written by
``spectramr.core.execution_ledger`` and produced by the same code path for
``train`` and for ``spectramr audit``, so the pre-flight and the run cannot
describe one config differently.

``substitutions``
    One record per divergence, each with a dotted path, the requested and
    resolved values, and a severity. The classes are ``EXTRA_IGNORE_DROPPED``
    (declared, not a field, silently discarded),
    ``EXTRA_ALLOW_UNTYPED`` (carried but never validated),
    ``RAW_DICT_UNVALIDATED`` (a ``dict[str, Any]`` field whose sub-keys bypass
    pydantic entirely) and ``VALUE_CHANGED_ON_FINALIZE``.

``defaults`` / ``defaults_injected``
    Every schema default the run took on trust, as dotted paths, plus its
    length. ``defaults_injected`` is a property over ``defaults`` — one
    derivation, so the two cannot disagree.

.. admonition:: ``defaults_injected`` is comparable only at equal schema depth
   :class: warning

   The walker descends only into keys present in **both** the raw YAML and the
   model, so a sub-block the YAML never mentions costs **1** and its own fields
   are never reached. Naming that block — which is exactly what a
   ``fold``-posture rename does — makes the walker descend and its remaining
   fields countable.

   Two configs that resolve to a byte-identical document can therefore report
   different totals purely because one names a block the other leaves implicit.
   Compare the **paths**, not the total. This is why the field is a list: the
   integer averages the effect away, and the per-key form is what makes it
   visible.

``run_summary.json``
    The footer, written on completion: success flag, best metrics, final loss,
    **wall-clock duration + iterations/sec**, and the provenance subset
    (``run_id``, ``git``, ``host``, ``config_sha256``, ``effective_batch``,
    ``model_params``, ``hardware``, ``slurm``).

    ``hardware`` is the flattened ``env.node`` census —
    ``gpu_count`` / ``gpu_types`` / ``gpu_node_count`` / ``gpu_driver_version``,
    ``cpu_model`` / ``cpu_cores_usable`` / ``cpu_cores_total``,
    ``memory_usable_gb`` / ``memory_total_gb``. It is what makes the footer's
    ``iterations_per_sec`` comparable across arms: 3 it/s on one RTX 3060 and
    3 it/s on eight A100s are opposite results. The key is ``null`` (not a dict
    of nulls) when nothing could be probed.

The console / log file also gets a scannable banner at startup::

    ─── run provenance ───
    run_id     : exp_gan-20260611_201500-61cf45551c5e
    git        : 61cf45551c5e @ dev (DIRTY)
    host       : node07 (pid 4242, <user>)
    torch      : 2.8.0+cu126 · 2x A100(40.0GB)
    gpu        : 2 visible / 4 on node · 2 allocated · driver 550.54.15 · CUDA_VISIBLE_DEVICES=0,1
    node       : 8/128 cores · 64.0/1007.5 GB RAM · AMD EPYC 7763 64-Core Processor
    python     : 3.12.12
    model      : unet · 12.00M (48.0MB) params
    data       : train[batches=100, samples=200, patients=25] / val[batches=20, samples=40]
    batch(eff) : 2×1accum×2gpu = 4
    config_sha : deadbeef0123
    seed       : 7  device: cuda
    slurm      : job=999 nodes=node07

Identical devices are grouped (``2x A100(40.0GB)``) so an 8-GPU node stays one
readable line. The ``gpu`` and ``node`` lines are omitted entirely when the
probes returned nothing, so a record missing them still renders.

On a *build* failure (before the loop), the banner is still emitted with the
git/env it managed to capture, so even a crash-at-startup is traceable to a
commit + host.

Cluster pre-flight: ``spectramr doctor``
----------------------------------------

The same environment probe powers the pre-flight gate (see
:doc:`cli_reference`)::

    spectramr doctor --require-cuda -c arm.yaml && spectramr train -c arm.yaml

``doctor`` and run-provenance share
:func:`spectramr.infrastructure.logging.provenance.torch_runtime`, so the
"is this node set up?" check and the "what ran here?" record never drift.

Programmatic access
-------------------

.. code-block:: python

    from spectramr.infrastructure.logging.provenance import (
        collect_run_provenance, format_provenance_lines, git_provenance,
        cpu_resources, memory_resources, gpu_resources, node_resources,
    )

    rec = collect_run_provenance(config, seed=7, device="cuda", run_name="myrun")
    for line in format_provenance_lines(rec):
        print(line)

    g = git_provenance()      # {"available": True, "sha": ..., "dirty": ...}

    # The hardware probes are usable standalone (e.g. to size dataloader
    # workers against what the run was actually granted, not the node total).
    n = node_resources()                    # {"cpu": ..., "memory": ..., "gpu": ...}
    workers = min(8, cpu_resources()["usable_cores"])
    gpus = gpu_resources()["count"]         # visible to torch, not node-total

API
---

.. automodule:: spectramr.infrastructure.logging.provenance
   :members:
   :undoc-members:
   :show-inheritance:

Where this run's log actually went
==================================

A run can write ``provenance.json``, ``resolved_config.json``, a TensorBoard
event file, its ``debug_snapshots/`` and every validation PNG — and not one log
line — and until the path is recorded, no artifact can say why.

Two mechanisms put a log somewhere other than beside the run:

**1. ``logging.sinks.dir`` outranks the run directory.**
:meth:`LoggingServiceFactory.create` passes ``log_dir=config.sinks.dir``
straight through, and ``ComprehensiveLoggingService`` resolves
``sinks.dir or log_dir``. That precedence is **correct** — a caller default
must not replace a declared value — so an arm declaring

.. code-block:: yaml

   logging:
     sinks:
       dir: experiments/results/my_arm/logs

writes its log there even when ``--output-dir`` puts the artifacts somewhere
else. Nothing is wrong, but the two halves cannot be connected unless the path
is recorded: *"the log is missing"* and *"the log is in the other tree"* look
identical from the run directory.

**2. An unwritable directory relocates the whole log, silently.**
``setup`` catches ``PermissionError``/``OSError`` around the file handler and
falls back to ``tempfile.mkdtemp(prefix="spectramr_logs_")``. Falling back is
defensible — a run should not die because its log sink is read-only — but the
``except`` branch touched neither ``self._logger`` nor ``warnings``, so it was
**mute by construction**. On a compute node that temp directory is wiped at job
teardown, so the log did not merely move: it ceased to exist, while the run
reported success.

Both are now recorded. ``LoggingService.setup`` sets two attributes
unconditionally — so a consumer never has to tell *"no attribute"* from *"no file
log"*:

.. code-block:: python

   service.resolved_log_path        # the file it actually opened, or None
   service.log_dir_relocated_from   # the intended dir, when relocation happened

and ``train.py`` stamps them into provenance beside the **declared** value, so
the two can be compared:

.. code-block:: json

   "logging": {
     "resolved_path": "/tmp/spectramr_logs_ab3f/kspace_cold_diffusion.log",
     "declared_sinks_dir": "experiments/results/my_arm/logs",
     "relocated_from": "experiments/results/my_arm/logs",
     "incomplete": ["the declared log directory was not writable; the log was
       moved to a temporary directory that a compute node wipes at job teardown"]
   }

A relocation is flagged ``incomplete`` rather than reported as a plain path,
because a path into a wiped temp directory reads as a log that exists. The
startup banner carries the same fact where it is still actionable — while the job
is running and the file can still be copied out::

   log        : /tmp/spectramr_logs_ab3f/kspace_cold_diffusion.log  [!] RELOCATED from experiments/results/.../logs (temp dir; wiped at teardown)

Relocation also raises a ``RuntimeWarning`` in addition to the log warning. That
is deliberate redundancy: a *log* warning about the log sink failing is precisely
the message most likely to be lost, and ``audit`` treats warnings as failures,
so the condition surfaces in a pre-flight too.

The stamp is **not** rank-gated. Non-zero ranks write ``provenance_rank{N}.json``
(see the parallel-topology section above), and each names its own log — an N-rank
run has N logs, and only rank 0's was ever discoverable.

Console-logging configuration
=============================

Run *provenance* (above) is the on-disk record of a run. Separately, the
**console logging** — the colored, badge-prefixed lines you see in a terminal —
is owned by :mod:`spectramr.infrastructure.services.logging_service`. Two entry
points configure it:

* :func:`~spectramr.infrastructure.services.logging_service.bootstrap_console_logging`
  is called once at the very top of ``spectramr.cli.app.main`` (before any
  subcommand), so ``logger.info(...)`` in *every* CLI flow renders through the
  :class:`~spectramr.infrastructure.services.logging_service.ColoredConsoleFormatter`
  rather than Python's plain ``lastResort`` handler.
* :meth:`LoggingService.setup` runs later, inside ``build_container``, applying
  the resolved :class:`~spectramr.config.schemas.logging.LoggingConfigSchema`
  (level, ``log_to_file`` / ``log_to_console`` / ``silent``, file handler).

The single source of truth
--------------------------

"Put exactly one colored console handler on this logger — upgrading any
pre-existing plain ``StreamHandler`` *in place* (so handler identity, stream and
filters survive) rather than stacking a duplicate" is the
:func:`_install_or_upgrade_colored_console` helper. Both entry points above
delegate to it. ``setup`` passes ``install_if_missing=log_to_console`` so silent
mode upgrades existing handlers but never *adds* a console sink.

Two behaviours follow from having one owner. ``ComprehensiveLoggingService.log``
injects default metadata and delegates; the base
:meth:`LoggingService.log` is the single throttle authority, and it rate-limits
only ``INFO`` / ``DEBUG`` — a ``WARNING`` or above is never throttled. The file
handler's format string is the single ``_FILE_LOG_FORMAT`` constant, shared by
the normal path and the permission-fallback path.

The bootstrap contract — idempotent, upgrade-in-place, ``force`` reset — is
pinned by ``tests/unit/infrastructure/logging/test_no_basicconfig_force.py``.

TensorBoard: one writer, and what it records
--------------------------------------------

:class:`~spectramr.infrastructure.services.tensorboard_writer.TensorBoardWriter`
is the single writer, owned by ``pipelines/train.py``. Per-run isolation is the
default and the knob is live: **the directory resolves relative to the run
directory**, so a relative declaration lands inside the run and an absolute one
overrides.

.. code-block:: yaml

   logging:
     tracking:
       enabled: true
       service: tensorboard      # closed enum: `tensorboard` | `none`
       enable_tensorboard: true
       tensorboard_dir: null     # -> <run_dir>/tensorboard
     intervals:
       histogram: 1000           # weight/grad histogram cadence

``service`` is a closed
:class:`~spectramr.config.schemas.enums.TrackingService`, so an unrecognised
value is refused rather than silently dropping tracking for the whole run. W&B
is refused rather than accepted-and-ignored, because ``logging.wandb_project``
and ``wandb_entity`` are inert. A missing ``tensorboard`` install raises rather
than warning and continuing.

What the writer records:

============================  ===================================================
Feature                       Why it is there
============================  ===================================================
``add_scalar``                per-metric ``train/`` and ``val/`` curves
``add_scalars``               loss terms on ONE axis — separate charts cannot
                              answer "is the adversarial term drowning L1"
``add_images``                validation panels, gated by ``images.log_validation``
``add_histogram``             weight/gradient **distributions**. A collapsed layer
                              and a saturated one can share a norm but never a
                              shape. Cadence-gated — see below.
``add_text``                  resolved-config dump, so provenance travels with
                              the event files
``add_hparams``               the run's hyper-parameters paired with its final
                              metrics. A confounded ablation — two knobs moved
                              in one run — is what the HParams dashboard makes
                              visible across runs.
                              Read off the resolved config, so it records what
                              the run USED, not what the arm claimed.
``purge_step``                set to the resume iteration, so a resumed chart
                              continues instead of folding back on itself
============================  ===================================================

``add_histogram`` copies every parameter to host memory, which is a GPU sync per
tensor. It is therefore gated on ``logging.intervals.histogram`` (default
**1000**), and the writer checks the cadence *before* touching a tensor, so the
sync happens only on the iterations that log. Lower it only for a short
debugging run.

Deliberately **not** wired, so the advertised surface stays equal to the wired
one: ``add_graph`` (tracing fails often on complex tensors and dict batches, and
the natural ``try/except`` around it would swallow the failure silently),
``add_embedding``, ``add_video``, ``add_figure``, and ``add_pr_curve`` (no
classification task exists here).

DDP is unchanged: only rank 0 owns a writer. A non-zero rank gets one that is
falsy and whose every method is a no-op, so the ``if tb_writer:`` guards keep
their exact meaning and nothing here is collective.

Artifacts must describe the run that wrote them
-----------------------------------------------

An arm's output directory is **reused across runs**, so an artifact that does
not bound itself to the run that wrote it attributes one run's evidence to
another. Each rule below closes one way for a bundle to stop being
self-describing.

``final_metrics.json`` is windowed to the current run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``logs/training_metrics.csv`` is **appended to** by every run writing into the
directory. A ``best`` block folded over the whole file would therefore report
whichever historical run scored best — and a run shorter than its own log
cadence, which writes zero rows, would reproduce a previous run's minimum
byte-identically while having measured nothing.

Two filters bound the window, in
:func:`spectramr.pipelines.train._summarise_best_metrics_from_csv`:

#. :func:`~spectramr.pipelines.train._select_current_run_rows` keeps the final
   non-decreasing ``iteration`` segment, since the column resets per run.
#. ``final_iteration`` — the last iteration this run actually reached — drops
   rows beyond it. This is what catches the zero-rows case, which segment
   detection alone cannot: the final segment still belongs to the previous run,
   but every one of its iterations exceeds this run's, so the window empties.

A run that logged nothing therefore reports ``best: {}`` rather than inheriting.

Every run writes rows, and the header promises only what it can fill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The section above bounds the *consequences* of an empty metrics CSV. The writer
prevents the cause.

**Every run yields at least two rows.** A row gate of
``iteration % log_interval == 0`` alone would be satisfied *zero* times by any
run shorter than ``logging.intervals.log`` — a header with no data rows, and,
through the same gate, no ``train`` TensorBoard scalars either, while the run
exits reporting success. The first and last iterations are therefore logged
unconditionally, in
:func:`spectramr.pipelines.training_loop._execute_training_loop`.

Two details matter for anyone changing that gate:

* The first iteration is ``start_iteration + 1``, **not** ``1``. On a resumed run
  the opening data point is at the resume offset, which is exactly the case a
  bare modulo loses.
* The gate is also the **only host transfer in the loop** — it exists so
  ``get_last_metrics`` can hand back on-device tensors rather than synchronising
  every step. The widening is therefore bounded to two extra iterations per
  *run*; a coarser modulo would put a ``.item()`` back into the hot path.

An arm whose whole budget is under its cadence still only gets a two-point curve,
so that condition now emits a ``WARNING`` at startup naming both numbers. It is a
warning rather than an info deliberately: ``LoggingService.setup`` clamps every
logger *and handler* to ``logging.sinks.level``, which is ``warning`` on the arms
this was found on, so an ``INFO`` would be discarded precisely where it is needed.

**The training CSV declares no ``val_*`` columns.** The row written is
``{"iteration", "epoch", **losses_scalar}`` and ``losses_scalar`` derives from the
training step's ``losses_history``; validation metrics are written to
``logs/validation_metrics.csv`` on their own cadence. A header for a column no
code path can fill is worse than an absent one, because a reader cannot
distinguish *"validation did not run"* from *"this file never carries this
column"*. Three downstream consumers therefore key off the header actually
present:

* :func:`~spectramr.pipelines.train._summarise_best_metrics_from_csv` folds the
  validation CSV as well (below);
* :func:`~spectramr.pipelines.train._select_current_run_rows` plus
  ``final_iteration`` empty the window instead of inheriting (above);
* ``_melt_metrics_csv`` in :mod:`spectramr.infrastructure.reporting.aggregator`
  drops all-empty columns so they cannot surface as phantom all-NaN series in the
  learning-curve figure.

So the set of columns is free to change without a consumer silently reading a
phantom series.

``best`` covers validation too, and ``run_summary`` receives it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validation and training are written to different files, so an aggregate that
reads one of them reports *"validation never ran"* for a healthy run — and hides
the case that matters most, a ``val_psnr`` far below its ``train_psnr``.

:func:`~spectramr.pipelines.train._summarise_best_metrics_from_csv` folds both
files, each windowed to the current run independently — they are written on
different cadences (``logging.log_interval`` vs
``validation.schedule.interval_steps``), so a run can produce rows in one and none
in the other. The pairing goes through
:func:`~spectramr.pipelines.train.validation_csv_for`, one derivation shared by the
writer and the reader; two copies of a path rule is how they end up pointing at
different files while both look correct.

``run_summary.json`` fills ``best_metrics`` from ``result["best_metrics"]``, and
**every** dict-returning path of ``_execute_training_loop`` carries that key —
pinned by a source-level test, because a behavioural test can only cover the
paths it happens to exercise while the requirement is that *no* path omits it.

Validation renders carry the training iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``step`` in a saved validation filename is the **training iteration**, the
same number on every cascade level of one validation event::

    validation_R2x_epoch000_step003000_...png
    validation_R8x_epoch000_step003000_...png
    validation_R32x_epoch000_step003000_...png

A per-level counter would label the later levels with small numbers, so any
consumer sorting by step would read them as the oldest files on disk.
:meth:`~spectramr.infrastructure.training.strategies.
diffusion.DiffusionTrainingStrategy._validation_image_step` returns the training
iteration unconditionally.

Debug snapshots state their scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fallback ``model_output`` snapshot pairs the generator's raw output
(network units) with ``target``/``input`` as they entered ``train_step`` —
before any normalization the strategy applies internally. Adjacent rows of one
stats table can therefore sit on different scales::

    model_output   min -3.799   max  2.644   std 0.1265
    target         min -2401    max  1279    std 9.083

Read at face value that says the model output is 630x too small. It is not:
``expm1(4.707) * 22.207 = 2435``. The target was displayed in physical units
while the output stayed log-compressed, and the loss compared like with like.

The snapshot now emits a scale context — which space each row lives in, both
peak magnitudes, their ratio, and a loud note when they differ by more than
10x, stating explicitly that a scale gap is *not* by itself evidence of a model
or loss defect.

Data counts carry their units (1024 files vs. 768 batches)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``provenance["data"]`` recorded ``{split: len(loader)}`` -- a bare integer whose
unit existed only in the banner's hardcoded ``" batches"`` suffix. A user who
compared ``train: 768`` against a training folder holding 1024 files read it as
25 % of the data having gone missing. Nothing was missing; four different units
were being compared::

    1024 files
      ->  384 groups                 one (patient, contrast) group per __getitem__
      x4  samples_per_volume    =   1536 patches
      /2  batch_size, drop_last =    768 batches

Every count is now named by its own unit, and the two that *any* loader can
answer are always present::

    data       : train[batches=768, samples=1536, groups=384, patients=128, files=1024]

``batches`` is ``len(loader)``; ``samples`` is ``len(loader.dataset)`` -- for a
``tio.Queue`` that is the patch count, not the subject count. Anything richer is
dataset vocabulary, so :func:`~spectramr.infrastructure.logging.provenance.describe_dataloader`
*asks* for it rather than guessing: a dataset, or anything it wraps (a ``Queue``'s
``subjects_dataset``, a wrapper's ``dataset``), may expose
``provenance_counts() -> dict`` and its keys are merged in. A dataset that cannot
answer in richer units simply gets the two universal keys -- nothing is invented,
which matters because this block serves every strategy while ``patients`` is
M4Raw vocabulary.

Two deliberate refusals:

* **No ``subjects`` key.** ``tio.Queue.num_subjects`` returns
  ``len(subjects_dataset)`` -- index entries, i.e. 384 ``(patient, contrast)``
  groups -- while the patient count is 128. Publishing either under a name a
  reader would take for the other would re-create, off by exactly the 3 x
  contrast factor, the ambiguity this change exists to remove. The M4Raw hook
  therefore names them ``groups`` and ``patients``, and the generic layer
  publishes neither.
* **No count that touches voxels.** ``len(tio.Queue)`` routes to the uncached
  ``iterations_per_epoch``, which walks ``dry_iter()``; that is affordable only
  because ``dry_iter`` returns metadata shells with a shared 1-voxel stub.
  ``provenance_counts`` likewise walks the in-memory index and opens no HDF5
  file.

Nested breakdowns stay in the JSON, and the banner filters them by *type* --
``_format_split_counts`` keeps only ``int`` values -- so a dataset may publish a
new one without touching the logging layer. M4Raw publishes two, and the pair is
what makes its repetition budget legible::

    per_contrast       {T1: 30, T2: 30, FLAIR: 30}     groups
    files_per_contrast {T1: 90, T2: 90, FLAIR: 60}     files

``per_contrast`` counts *groups*, so it is uniform for a corpus with equal
subject counts whatever the repetition budget -- it reads identically whether
FLAIR ships 2 repetitions or 3. ``files_per_contrast`` is the key that states the
difference. It matters beyond bookkeeping because the NEX target is an
average over a group's repetitions: a T1/T2 target averages 3 and a FLAIR target
averages 2, so the input repetition's own noise survives into its own target at
1/2 rather than 1/3, and leave-one-out (``data.nex_target_exclude_input``) is
available for T1/T2 but *structurally impossible* for FLAIR, which silently falls
back to the all-repetitions average. A validation number compared across
contrasts is therefore comparing against references of different quality, and the
record now says so.

Attribution is per *side*, not per record: a federated pair contributes its
``source`` files to ``source_contrast`` (T1 by construction) and its ``target``
files to ``target_contrast``, deduplicated by path -- one T1 group paired against
both T2 and FLAIR is counted once, under T1. A record naming neither contrast
key contributes nothing, so ``sum(files_per_contrast) < files`` is the signal
that something was unattributable rather than a bucket holding a guess.

The slice-level M4Raw route (``data.slice_level_records: true``) changes
the record unit, and the record states it. By default one record is one
(patient, contrast) group, and one ``[256, 256, 1]`` patch served from it costs
a full read of every repetition of the group: 18 slices per served slice, times
the repetition count, every epoch. With the option set, the index has one record
per (group, slice), each item reads ``f["kspace"][slice]`` of every repetition,
the subject is a depth-1 volume, and the NEX average is unchanged because it was
per slice already. ``groups`` still counts groups; the record count is published
as ``slice_records``. The queue is not bypassed: with
``data.modes.train.sampler.samples_per_volume: 1`` and a depth-1 patch, one
patch is sampled per record and an epoch is one pass over the slices. With any
other ``samples_per_volume``, that many patches of the same slice are sampled
per epoch, so the audit check ``slice_level_records_queue_shape`` reports it as
an error, as it does a patch depth other than 1. Under
``coils.processing_mode: rss`` the phase-reference coil is selected per record,
which on this route is per slice.

An empty loader records ``batches: 0``, not ``null``. A truthiness test would
write ``null``, because a ``DataLoader`` defines ``__len__`` and an empty one is
falsy -- indistinguishable from a split that was never built. ``0`` is a
finding; ``null`` is a shrug.

A count that could not be taken is named in ``incomplete`` rather than dropped,
and the banner renders that marker so it cannot be misread as a zero. Only the
parameter count is gated on a model having been built, so the in-process ``env=``
entry point keeps its data record either way.

These counts also make one *unread* knob visible.
``TorchIOQueueBuilder.build_val_queue`` has no production caller (the director
builds validation through ``DataLoaderBuilder``), so ``data.modes.val.sampler``
and a resolved ``use_queue_for_validation: True`` are never read and the val
loader hands out whole volumes rather than patches. The signature on the record
is ``samples == groups`` for that split, where a patch-sampled split would show
``samples == groups x samples_per_volume``. Wiring it would move every
validation metric on every arm carrying the block, so it is a science decision
rather than plumbing.

Parallel topology: nodes, ranks and which GPU each one got
==========================================================

"Provenance says the training used 1 GPU, but I asked for 4 with DDP" is a
question the record has to be able to answer. Three mechanisms have to hold for
it to:

#. **The declaration never reached provenance.** ``parallel.num_devices`` and
   ``num_nodes`` were written to ``resolved_config.json`` and stopped there, so
   comparing "asked for" against "got" meant opening a second file — and
   :mod:`spectramr.pipelines.distributed` *overwrites* ``num_devices`` from
   ``LOCAL_WORLD_SIZE``, so the resolved config does not reliably hold the
   authored value either.
#. **The runtime record was overwritten, not merged.** ``train.py`` replaced
   ``provenance["parallel"]`` with the parallel plugin's own thin record — often
   just ``{"strategy": ...}`` — discarding the ``rank``, ``local_rank``,
   ``launcher``, ``initialized`` and ``backend`` that
   :func:`~spectramr.infrastructure.logging.provenance.parallel_provenance` had
   already resolved. Those are exactly the fields the question turns on.
#. **There were no node or rank facts at all.** ``_SLURM_FIELDS`` carried the
   per-node allocation (CPUs, memory, GPUs) but no node *count*, so a 2-node run
   and a 1-node run stamped indistinguishable records.

What the record now carries
---------------------------

Under ``parallel``, declared and applied sit side by side, so a divergence is
the finding rather than a puzzle — the same declared-vs-applied discipline the
debug snapshots follow:

==============================  ===================================================
Key                             Meaning
==============================  ===================================================
``strategy``                    Declared strategy, or ``None`` for no block at all
``declared_num_devices``        What the YAML asked for; ``None`` when unauthored
``declared_num_nodes``          As above, for nodes
``declared_backend``            ``parallel.backend`` — the SSOT the CLI flag only
                                overrides when explicitly passed
``world_size``                  What the run actually got
``rank`` / ``local_rank``       This process's identity
``local_world_size``            Ranks per node
``node_count``                  Nodes; see *derived* below
``node_count_derived``          Present and ``True`` only when the count was
                                divided out of world/local-world rather than read
                                from the scheduler
``launcher``                    ``"torchrun"`` when ``TORCHELASTIC_RUN_ID`` is set
``initialized`` / ``backend``   Whether a process group existed, and which
==============================  ===================================================

``declared_*`` is ``None`` — not the schema default of ``1`` — when no
``parallel:`` block was authored, so "no block" and "``num_devices: 1``" stay
distinguishable.

Two new sibling blocks sit beside ``slurm``:

* ``launcher_env`` — ``MASTER_ADDR``, ``MASTER_PORT``, ``TORCHELASTIC_RUN_ID``,
  ``WORLD_SIZE``, ``LOCAL_WORLD_SIZE``, ``GROUP_WORLD_SIZE``, ``RANK``,
  ``LOCAL_RANK``. Deliberately **not** folded into ``slurm``, whose documented
  contract is *what the scheduler granted*: these say how the group was wired,
  and a run can have either without the other (torchrun outside Slurm, or a
  Slurm job launched with plain ``spectramr train``).
* ``rank_devices`` — one record per rank: ``hostname``, ``rank``,
  ``local_rank``, ``device_index``, ``device_name``, ``device_uuid``.

Every value in ``slurm`` and ``launcher_env`` is stamped as a **raw string**.
``SLURM_TASKS_PER_NODE`` is formatted ``4(x2)`` on a heterogeneous allocation, so
feeding it to ``_env_int`` would silently drop it.

Why the per-rank inventory is a gather, and where it has to live
----------------------------------------------------------------

:func:`~spectramr.infrastructure.logging.provenance.gpu_resources` shells out to
``nvidia-smi`` on the **local** host. On a multi-node run that alone would make
rank 0's record describe one node and silently imply that node is the whole job.
:func:`~spectramr.infrastructure.logging.provenance.rank_device_inventory` gathers
one small record per rank instead, which is the only way the artifact can show
that two ranks landed on the *same* physical device — a failure that presents as
a mysterious halving of throughput rather than as an error, and which is warned
about explicitly.

``all_gather_object`` is a **collective**: every rank must reach it or the job
hangs forever. That inverts this module's usual fail-open posture and makes the
call's *placement* load-bearing rather than stylistic:

* It sits **before** the pipeline build in
  :func:`~spectramr.pipelines.train.run_training_pipeline`. Every other provenance
  site is inside ``if provenance:`` *and* inside the build's ``try:``, and both
  guards are rank-divergent — ``collect_run_provenance`` fail-opens per rank, so
  ``provenance`` can be ``{}`` on one rank and populated on another, and a build
  failure returns early on one rank only. Either asymmetry strands the remaining
  ranks at the collective.
* Its only guard is ``is_available() and is_initialized()``, which is uniform
  across ranks by construction.
* Rank-gating applies to the **write**, never the gather. Putting a collective
  inside a rank-0 branch is a guaranteed hang, and
  ``tests/unit/pipelines/test_train.py::TestThePerRankGatherCannotDeadlock``
  walks the real AST to keep it out of one.
* The local record is built with a per-field ``try``, so a broken ``nvidia-smi``
  or an unset CUDA device degrades one string instead of stranding the job. If
  the gather itself fails, the reason lands in ``incomplete`` rather than being
  omitted: "no inventory" and "inventory says one node" must not look the
  same.

A rank whose ``device_index`` could not be resolved is **excluded** from
collision detection rather than grouped. ``device_index`` is ``None`` on every
rank of a CPU run, so keying on it would report the entire world as sharing one
device — a false alarm exactly where the record is least informative.

NCCL note: ``all_gather_object`` moves its pickled buffer to
``torch.cuda.current_device()``, so every rank must have called ``set_device``.
:func:`spectramr.pipelines.distributed.setup_distributed` does so *before*
``init_process_group``; ``infrastructure/distributed/launcher.py`` does so right
after. This is the same invariant ``RankUtility.broadcast_object`` already
relies on at three training-loop sites.

Rank 0 owns the canonical artifacts
-----------------------------------

``run_dir`` is rank-**invariant** — it derives from ``config.training.output_dir``
on a frozen config, not from a per-rank timestamp. So under DDP every rank wrote
the same ``provenance.json`` and ``resolved_config.json``: last-writer-wins, and
the surviving record was an arbitrary rank's view of the run. It *looked* fine
(the file exists, the JSON parses), which is what made it durable.

Rank 0 now writes ``provenance.json`` and ``resolved_config.json``; every other
rank writes ``provenance_rank{N}.json`` beside them. The non-zero files are kept
rather than dropped because the per-rank ``local_rank``/``device`` facts exist
nowhere else, and discarding them would make the new inventory unverifiable.
The naming rank comes from the same ``resolve_data_rank`` the data sharding uses
— a provenance file named for a different rank than the one that sharded the
data would be worse than no per-rank file at all.

``world_size`` prefers the live group
-------------------------------------

Reading the world size from the environment alone is not enough. Only
``torchrun`` exports ``WORLD_SIZE``: a ``torch.multiprocessing.spawn`` worker —
which ``launcher.launch_distributed`` uses for *every* single-node multi-GPU
run — receives its world size as an ``init_process_group`` **argument**, and the
environment is never set, so the record would read ``world_size: 1`` and an
``effective`` batch N times too small. Under torchrun inside a 1-task Slurm
allocation ``SLURM_NTASKS`` is likewise 1 while the true world size is the GPU
count.

:func:`~spectramr.infrastructure.logging.provenance.effective_batch_size`
therefore prefers the live process group, and ``world_size_source`` names which
of the two answered — otherwise a genuine single-process run cannot be told from one
whose group had not been initialised when the banner rendered.

The banner says it inline
-------------------------

A reader scanning a log will not diff two numbers in a JSON file they have not
opened, so the mismatch is stated where they are already looking::

    parallel   : ddp · world=1 rank=0 · single-process · [!] ddp declared on a
                 single process · [!] declared num_devices=4 but 1 rank(s) on
                 this node: the extra devices are NOT being used
    slurm      : job=12345 nodes=n[1-2] n=2 tasks/node=4(x2)
    ranks      : n1[r0→cuda:0, r1→cuda:1]; n2[r2→cuda:0, r3→cuda:1]

The ``ranks`` line is collapsed per host: a 32-rank job would otherwise push 32
lines into a banner whose whole value is being scannable, and the fact a reader
needs is *did every rank get its own GPU, on the host I expected*.

What the two banner warnings mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**"declared num_devices=N but M rank(s) on this node".** ``num_devices`` is
authored beside ``num_nodes`` and is **per-node**, so it is compared against the
ranks on *this* node, never against the global ``dist.get_world_size()`` — a
correct 2 × 4 run uses all eight devices and must not be warned about. A shape
that cannot be resolved, multi-node with no per-node count, is left alone rather
than divided: dividing would guess that the ranks were spread evenly, and a
banner warning that is wrong once is distrusted forever. The "extra devices are
NOT being used" clause is direction-gated, since a declaration *smaller* than
the rank count is a stale declaration, not idle hardware.

**"ddp declared on a single process" / "on an initialised 1-rank group".** These
are named apart because they are different situations: no process group at all,
versus a group that came up with one rank. Only the second silently wastes
hardware — the first raises out of ``_require_process_group`` during ``adopt``
anyway. The banner says it because that raise lands at Stage B, after the model
and the data are built, while the banner renders before anything is. A
deliberate single-rank DeepSpeed run — ZeRO-3 or CPU offload on one GPU is a
real memory strategy — gets a line it does not need; that is banner noise rather
than a false alarm, and the cheaper error.

Multi-node launches use ``train-distributed``
---------------------------------------------

``infrastructure/distributed/launcher.py`` emits ``-m spectramr.cli
train-distributed`` for its multi-node torchrun command, and a paired test pins
that subcommand against the real parser. ``train`` is the one spelling that
cannot work here: it never calls ``setup_distributed``, so no process group
exists and every group-requiring strategy raises out of
``_require_process_group`` during ``adopt`` — at Stage B, after the model and
data are already built, which reads like a DDP or DeepSpeed problem rather than
a wrong verb. If you write your own launch line, use ``train-distributed``.

.. _validation-cadence:

Validation cadence: when the loop validates
===========================================

Two independent triggers, both under ``validation.schedule``:

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Key
     - Default
     - Fires
   * - ``interval_steps``
     - ``1000``
     - Every N training steps. Always active (``ge=1``, so it cannot be
       switched off).
   * - ``on_epoch``
     - ``false``
     - **Additionally** at each epoch boundary, every ``interval_epochs``
       epochs. Skipped during sanity checks, which obey ``interval_steps``
       strictly.

``on_epoch`` is **additive**: it adds validation events, it never removes the
step-interval event an arm selects its checkpoint from. That is the safe
direction — the opposite reading could silently delete the only event feeding
``checkpoint_best.pt``.

.. rubric:: ``on_epoch`` defaults to ``false``

Epoch-boundary validation is opt-in. Turning it on adds validation passes and
wall-clock and changes the metric cadence, so an arm that wants it declares
``validation.schedule.on_epoch: true`` and sets ``interval_epochs``; that field
is consulted only in epoch mode.

.. _run-summary-checkpoints:

``run_summary.json`` says where the checkpoints went
====================================================

.. code-block:: json

    "checkpoints": {
      "dir": "/project/.../experiments/results/exp_x/checkpoints",
      "inside_run_dir": false,
      "files": [{"name": "checkpoint_best.pt", "bytes": 41231872}],
      "total_bytes": 41231872
    }

``inside_run_dir`` is the load-bearing field. When it is ``false`` the files do
not travel with a retrieved artifact bundle, and the bundle's ``checkpoints/``
directory arrives **empty** — which reads as "no checkpoint was saved".

An arm whose ``checkpoint.checkpoint_dir`` points outside the collected run
directory produces exactly that: the log names files it wrote, the weights exist
on the cluster, and the retrieved bundle contains none of them. Both statements
are true at once. No weights are then recoverable from a completed run, and
``early_stopping.restore_best_weights`` cannot function across a resume.

"Written elsewhere" and "never written" need opposite responses — fix the sync
versus fix the run — so the footer now records enough to tell them apart, and
the run logs a warning at the end in both cases.

The block never fails the run: a ``stat()`` hiccup on a network filesystem must
not cost a finished run its summary, so the helper swallows and reports
``dir: null``.
