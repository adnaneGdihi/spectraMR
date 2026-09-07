Distributed & parallel training
================================

There was no page for this. The only distributed document was
``docs/v6_2_distributed_and_orchestration.md``, which is markdown and sits
outside the toctree, so nothing linked to it.

The one switch
--------------

``parallel.strategy`` selects the backend. It is a closed vocabulary, validated
at config-load time:

.. code-block:: yaml

   parallel:
     strategy: 'none'    # none | dp | ddp | fsdp | deepspeed
     backend: 'nccl'     # nccl | gloo | mpi  -- THIS is the backend SSOT
     num_devices: 1
     num_nodes: 1

===============  ==========================================  =================
``strategy``     what it does                                needs a launcher?
===============  ==========================================  =================
``none``         single process, single device               no
``dp``           ``nn.DataParallel``, one process, N GPUs    no
``ddp``          ``DistributedDataParallel``, 1 proc/device  **yes**
``fsdp``         parameter/gradient/optimizer sharding       **yes**
``deepspeed``    ZeRO stages 0-3, optional CPU/NVMe offload  **yes**
===============  ==========================================  =================

``fsdp`` and ``deepspeed`` additionally require their sub-block flag to agree
with ``strategy``:

.. code-block:: yaml

   parallel:
     strategy: 'fsdp'
     fsdp:
       enabled: true          # must equal (strategy == 'fsdp')

Declaring only one half raises at load time. The agreement is checked when the
config loads rather than at dispatch, so a mismatch can never reach the point
where ``strategy: 'none'`` with ``fsdp.enabled: true`` would silently shard, or
where ``strategy: 'fsdp'`` alone would raise only after the whole training
environment had been built.

Launching
---------

.. code-block:: bash

   # none, dp -- no launcher
   spectramr train --config <arm>.yaml

   # ddp, fsdp, deepspeed
   torchrun --nproc_per_node=4 -m spectramr.cli train-distributed --config <arm>.yaml

The launcher does **not** rewrite ``parallel.strategy``. Forcing it to ``"ddp"``
on every distributed launch would make ``fsdp`` and ``deepspeed`` unreachable
from this entry point, overwriting the declaration before dispatch ever saw it.
``num_devices``/``num_nodes`` *are* overwritten, because those are observed
facts about the launcher rather than declarations.

There is no auto-detection. A config that names a strategy will not start a
process group on its own, and forgetting ``torchrun`` is an **error** for every
process-group-backed strategy -- not a fallback. That matters more than it
sounds: a warn-and-continue FSDP path returns the *unwrapped* model, so the run
completes, reports success, and stamps ``fsdp`` into its own provenance while
never having sharded anything.

``--nproc_per_node`` is checked against the allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A launch that leaves scheduler-allocated GPUs unused is **refused**, not warned
about. If this node's GPU grant exceeds the ranks torchrun started on it, the
launcher raises ``IdleDeviceError`` before any model is built.

The refusal exists because that failure is otherwise completely silent. On a
``--gpus=4`` allocation whose wrapper hardcoded ``GPUS=1``, the process group
initialised, DeepSpeed adopted, ``Config Health`` reported 141/141 checks passed,
and the arm trained correctly for 41 minutes -- on one card, with three idle. The
only thing that noticed was the cluster's own ``jobstats``, after the fact::

   This job did not use 3 of the 4 allocated GPUs.

Derive the rank count rather than typing it::

   GPUS="$(nvidia-smi -L | grep -c '^GPU ')"     # or: ${SLURM_GPUS_ON_NODE:?}
   torchrun --standalone --nproc_per_node="$GPUS" \
       -m spectramr.cli train-distributed --config <arm>.yaml

Note that ``--gpus=N`` populates only ``SLURM_GPUS``, which is a **job total**;
``SLURM_GPUS_ON_NODE`` and ``SLURM_GPUS_PER_NODE`` are per node. The check reads
all three and divides the job total by the node count, refusing to guess when it
does not divide (a heterogeneous allocation).

The check is deliberately narrow, so it never refuses a correct launch:

* **No scheduler grant, no finding.** Four idle cards on a workstation are nobody's
  allocation.
* **Visible bounds allocated.** Under ``srun --gpu-bind=single:1`` each task sees
  one device out of four; the other three belong to sibling ranks, not to nobody.
* **Equal counts pass.** Four ranks on four GPUs is the shape being protected.

For a deliberate single-rank debug run on a multi-GPU allocation, acknowledge it
at the launch rather than editing the arm::

   -O parallel.allow_idle_devices=true

Whether the guard was armed or waived is stamped into provenance as
``parallel.declared_allow_idle_devices``, so a 1-rank record on a 4-GPU
allocation can be told apart from a run made before the check existed.

Reading what a run actually got
-------------------------------

Because the declaration and the process group are independent, a run states
both at startup, on one line, before anything is built::

   parallel   : deepspeed · world=1 rank=0 · single-process ·
                [!] deepspeed declared on a single process
   knobs      : grad_ckpt=True · amp=off(fp32) · accum=2 · workers=8 ·
                max_iter=30000 · val_every=5000

The left half is the *declaration* (``parallel.strategy``, or
``none (no parallel: block)`` when the arm has none -- distinguished from
``strategy: none`` because a typo'd key otherwise renders as a plausible
opt-out nobody wrote). The right half is the *live* group, read from
``torch.distributed`` when it is up and from the launcher env otherwise. The
two ``[!]`` flags are the mismatches this page is about: a strategy declared
on a single process, and ``none`` under a multi-rank launcher -- the second
being *N* ranks each training the same un-sharded run.

``amp=off(fp32)`` is reported whenever ``precision.enabled`` is false, since a
block reading ``dtype: bfloat16`` under ``enabled: false`` runs fp32 and
printing the dtype alone would misreport it.

Emitted from ``log_startup_summary`` (``infrastructure/logging/provenance.py``)
**before** the container is built, because ``LoggingService.setup`` pushes
``logging.sinks.level`` onto the root logger, every existing logger and every
handler -- so on an arm setting ``level: warning`` this is the last point an
INFO line reaches the console. The same facts are stamped into the provenance
record under ``parallel``, so ``provenance.json`` carries them for runs read
after the fact.

Wrap ordering
-------------

Parallelisation happens at two hooks inside
:class:`~spectramr.infrastructure.training.builders.director.TrainingEnvironmentDirector`,
and which hook a strategy uses is a correctness constraint:

.. code-block:: text

   ModelBuilder:  build -> compile -> EMA
     Stage A   prepare_models()     SyncBatchNorm, FSDP
   OptimizationBuilder:  optimizers / schedulers
     Stage B   adopt()              DataParallel, DDP, DeepSpeed engine, samplers

**FSDP must be Stage A.** ``FSDP.__init__`` flattens parameters into a shard and
re-points their storage. An optimizer built first holds parameters whose storage
was swapped underneath it, so gradients arrive shard-shaped against full-shape
``exp_avg`` -- a shape error on the first step, or a silently wrong update on the
``foreach`` path -- and it allocates full-size state, defeating the point.

**DDP must be Stage B.** It does not change parameter identity, so the optimizer
stays valid; wrapping earlier would make ``count_parameters`` and the optimizer
builder's model selection see a ``DistributedDataParallel``.

**DeepSpeed must be Stage B.** ``deepspeed.initialize`` *consumes* an
already-built optimizer.

Adding a strategy means registering a plugin in
:mod:`spectramr.infrastructure.distributed.strategy_registry`, not editing a
dispatch chain. A test compares the schema's ``ParallelStrategy`` Literal against
the registry so the two cannot drift.

Gradient clipping under FSDP
----------------------------

``clip_grad_norm_(model.parameters(), n)`` computes the norm over the parameters
**this rank holds**. Under FSDP that is a shard, so each rank scales by a
different factor. Nothing errors -- every rank steps successfully -- and it
presents as training instability.

:class:`~spectramr.infrastructure.training.optimizers.FSDPStepPolicy` uses FSDP's
own ``model.clip_grad_norm_()``, which all-reduces the squared norms first. The
strategy supplies it, because the strategy is what knows how the model was
wrapped.

Installing the step policy
~~~~~~~~~~~~~~~~~~~~~~~~~~

The plugin *resolving* a step policy is only half the wiring. The strategy --
and therefore its ``StepExecutor`` -- is constructed **before** the parallel
runtime exists, so ``StepExecutor`` starts life holding the generic
``AMPPolicy``. ``_install_parallel_step_policy`` (``pipelines/training_loop.py``)
hands the resolved policy over and ``StepExecutor.adopt_step_policy`` re-runs the
capability negotiation.

Without that call the policy was produced, stored on ``ParallelRuntime``, and
read by nothing: ``DeepSpeedStepPolicy``'s ``owns_gradient_accumulation`` /
``owns_zero_grad`` never took effect, so the loop ran ``loss.backward()`` +
``optimizer.step()`` instead of ``engine.backward()`` / ``engine.step()`` -- and
divided the loss a second time on top of DeepSpeed's own division. The run
trains, every test passes, and only the numbers are wrong.

The registry deliberately accepts a **class or an instance**: FSDP hands back
``FSDPStepPolicy`` itself because only the loop layer knows the arm's
``gradient_clip_value``, while DeepSpeed hands back a configured instance because
only the engine knows itself. ``_install_parallel_step_policy`` resolves that;
``adopt_step_policy`` rejects a class outright, because ``getattr`` on a class
returns the descriptor rather than the value and would negotiate every capability
flag silently to ``False``.

DeepSpeed
---------

.. code-block:: yaml

   parallel:
     strategy: 'deepspeed'
     deepspeed:
       enabled: true
       zero_stage: 2              # 0 | 1 | 2 | 3
       offload_optimizer: 'none'  # none | cpu | nvme
       save_consolidated_best: true

Install the extra first (it is deliberately excluded from ``[all]``):

.. code-block:: bash

   pip install -e '.[deepspeed]'

.. warning::

   **ZeRO does nothing at ``world_size=1``.** ZeRO partitions optimizer state
   (and, from stage 2, gradients) *across data-parallel ranks*; with a single
   rank there is nothing to partition, so the run pays the engine's overhead
   for no saving. Raise the GPU count to make the sharding do any work.

**There is no ``config_path``.** The schema is the source of truth and the
``ds_config`` dict is derived from it by ``build_deepspeed_config``, written
beside the run's provenance. A hand-edited ``ds_config.json`` is a
cluster-critical file with no committed generator -- and worse than a data
manifest, because a manifest that fails to sync fails loudly, whereas a ds_config
that disagrees with the YAML produces a *successful run of the wrong
experiment*: DeepSpeed accepts a ``train_micro_batch_size_per_gpu`` that
contradicts ``data.batch_size`` and nothing compares them.

Consequently these are **not** declarable in the ``deepspeed:`` block and always
come from their existing home:

======================================  ==========================================
ds_config key                           source
======================================  ==========================================
``train_micro_batch_size_per_gpu``      ``data.batch_size``
``gradient_accumulation_steps``         ``optimization.gradient_accumulation_steps``
``gradient_clipping``                   ``optimization.gradient_clip_value``
``fp16`` / ``bf16``                     ``resolve_amp_precision(...)``
``optimizer`` / ``scheduler``           absent -- the engine adopts the built ones
======================================  ==========================================

ZeRO-3, DeepCompile and ZenFlow
-------------------------------

All three are wired and declarable. Each carries one non-obvious way to get a
*successful run of a different experiment*, so read the note that goes with it.

.. code-block:: yaml

   parallel:
     strategy: 'deepspeed'
     deepspeed:
       enabled: true
       zero_stage: 3
       offload_optimizer: 'cpu'
       stage3_gather_16bit_weights_on_model_save: true

       compile:                 # DeepCompile
         enabled: true
         passes: ['z3']

       zenflow:                 # importance-aware offloaded updates
         enabled: true
         topk_ratio: 0.1
         select_strategy: 'step'
         select_interval: 32
         update_interval: 4
         overlap_step: true
         offload: true

**DeepCompile is not torch.compile with extra steps.**
``optimization.compile_model`` compiles the bare module in Stage A, *before*
``deepspeed.initialize``, so the ZeRO collectives are opaque calls the compiler
cannot schedule across. DeepCompile traces the engine graph *including* the
allgather / reduce-scatter. They are alternatives, not layers: declaring both is
an audit **error**, because DeepCompile would be handed an already-compiled
module and could not do the one thing it exists for.

*The trap:* a ds_config key alone does nothing. ``engine.compile()`` must also
be called, and without it DeepSpeed emits a single ``log_dist_once`` line on
rank 0 — "DeepCompile is enabled but engine.compile() has not been called" —
then runs eagerly. A once-only, rank-0, startup-time log on a cluster is
indistinguishable from silence, so a config-only wiring would be declared,
accepted, stamped into provenance, and inert.
``initialize_deepspeed_engine`` makes the call and **raises** if the engine
cannot.

*The pass names are dispatch keys*, not labels: ``z3`` rewrites the
parameter-partitioning collectives and therefore requires ``zero_stage: 3``;
``z1`` targets stage 1/2. The schema rejects a mismatch.

**ZenFlow requires CPU offload, and takes over gradient accumulation.**
It ranks gradient columns by importance, applies the top ``topk_ratio`` every
step and the remainder every ``update_interval`` steps, so the offloaded
optimizer step stops being the critical path. With the optimizer on GPU there is
nothing to schedule, and DeepSpeed raises ``ValueError("Zenflow must be used
with cpu offload")`` from inside ``deepspeed.initialize``; the schema catches it
at load instead.

*The trap:* ``configure_zenflow`` ends with

.. code-block:: python

   engine._config.gradient_accumulation_steps = engine.update_interval

so ZenFlow **replaces** the number this repo treats as owned by
``optimization.gradient_accumulation_steps``. An arm declaring ``4`` with an
``update_interval`` of ``16`` trains at a 4x larger effective batch while
provenance, the run banner and ``effective_batch_size`` all report ``4``. That is
exactly the divergence the no-``config_path`` rule exists to prevent, so
``check_zenflow_accumulation_conflict`` makes a disagreement an **error** and
the generator renders the value the engine will actually use.

*Absent is not the same as disabled.* ``ZenFlowConfig`` has no ``enabled``
field, and DeepSpeed branches on ``zenflow_config == None``. Emitting
``"zenflow": {}`` for a disabled block would construct a full-default config and
turn ZenFlow **on**, so the generator omits the key entirely.

The full-stack reference arm is
``workflow_baselines/b4_..._zero3_deepcompile_zenflow.yaml``. Unlike b2/b3 it is
**not** a single-knob comparison against b1 — three mechanisms co-vary — so use
b3 (ZeRO-2 alone) as the intermediate when attributing a delta.

Two limitations to know before committing GPU time:

**GAN arms.** ``deepspeed.initialize`` returns one engine per optimizer, and
``engine.step()`` issues collectives. An arm whose discriminator steps on a
different cadence than the generator **deadlocks** rather than erroring. The
backend refuses more than one optimizer unless
``deepspeed.allow_multi_engine: true``.

**Complex / k-space arms with fp16.** ``get_autocast_context`` disables autocast
for complex+fp16 because there is no ``complex16``. DeepSpeed casts weights to
half from *inside* the engine, where that guard cannot see it. The audit makes
this an error; use ``bfloat16``.

Checkpoints
-----------

Every wrapper renames the keys ``state_dict()`` emits -- ``torch.compile`` adds
``_orig_mod.``, DP/DDP and ``ModelEma`` add ``module.``, FSDP adds
``_fsdp_wrapped_module.``. :mod:`spectramr.core.module_utils` is the single place
that strips them, applied at every save and load site.

This is not cosmetic. Every inference and evaluation path builds a **bare** model
and loads with ``strict=True``, so a wrapped checkpoint raises there -- and under
``strict=False`` (campaign evaluation, warm-start, distillation) it matches
nothing, loads nothing, and reports success, so a randomly-initialised model
produces metrics that read as a bad arm.

DeepSpeed writes a sharded tag *directory*. With
``save_consolidated_best: true`` (the default) rank 0 additionally writes a
single-file ``checkpoint_best.pt``, so ``discover_best_checkpoint``, campaign
evaluation and ``spectramr infer`` keep working without understanding ZeRO shards.
Turning it off makes the run resume-only, and the audit warns.

Reading a consolidated checkpoint back
--------------------------------------

That consolidated file is **not** the generic payload. ``save_best`` writes the
tag directory, calls ``save_16bit_model``, and then *returns* -- so
``checkpoint_best.pt`` is a bare parameter-keyed ``state_dict`` with no
``generator``, ``optimizer_g``, ``ema_state`` or ``counter_state`` key. Only
``save`` writes both artifacts, which is why periodic checkpoints carry that
metadata and the best checkpoint does not.

Restoring it therefore requires the strategy that wrote it.
:meth:`~spectramr.infrastructure.builders.directors.checkpoint_director.CheckpointDirector.load_from`
reads the sharded tag directory through the adapter and skips the generic parse
when the file has no ``generator`` key, and
:meth:`~spectramr.infrastructure.builders.directors.checkpoint_director.CheckpointDirector.with_parallel_runtime`
is what supplies that adapter. **A director built without it resolves**
``DefaultCheckpointAdapter`` **and cannot read any sharded strategy's
checkpoint** -- which is how ``early_stopping.restore_best_weights`` fails a
finished DeepSpeed run with ``KeyError('generator')``, discarding the best
weights while they sit on disk. Every director that saves *or* loads must be
handed the run's ``ParallelRuntime``.

Three consequences worth stating. The tag directory is the only source of ZeRO
optimizer state, so a missing tag is a failed restore and raises rather than
falling back to the consolidated weights. EMA shadow weights survive neither
artifact, because ``adopt`` wraps only the generator and discriminator, so
``ema_state`` reaches the generic payload that ``save_best`` never writes. And
the tag's ``client_state`` is the only source of the run POSITION -- both writers
record ``epoch`` and ``global_step`` on every save, so a tag missing them was
written by another tool or an older version, and ``load_from`` raises instead of
restoring at epoch 0. That absence cannot be detected by its value: 0 is also a
legitimate position, so a defaulted read would reset the LR schedule and the
early-stopping counter of a week-long run while reporting a successful restore.

Who participates vs who writes
------------------------------

These are **different questions**, and conflating them is the most expensive
bug in this subsystem.

``RankUtility.is_main_rank()`` gates every shared write (CSV,
``final_metrics.json``, TensorBoard) so non-zero ranks do not race on the output
directory. For ``none`` / ``dp`` / ``ddp`` that is also correct for
checkpoints: those strategies *replicate*, so rank 0's ``state_dict()`` is the
whole model.

Under FSDP and DeepSpeed it is a **deadlock**. Building the checkpoint is a
collective -- FSDP's ``state_dict()`` all-gathers the shards, DeepSpeed's
``save_checkpoint()`` synchronises across ranks -- so rank 0 enters a barrier
that ranks 1..N never reach. There is no exception and no log line; the job
hangs until SLURM kills it at walltime, and the last line in the log is a normal
training iteration.

.. code-block:: python

   may_checkpoint = is_main_process or checkpoints_need_all_ranks

One predicate, derived once from
:attr:`~spectramr.infrastructure.distributed.strategy_registry.ParallelRuntime.checkpoints_require_all_ranks`,
used at every checkpoint site. The strategy's
:class:`~spectramr.infrastructure.distributed.checkpoint_adapters.IParallelCheckpointAdapter`
then decides which rank touches the disk (with ``rank0_only=True`` the others
hold empty tensors, so letting them write would litter the run directory with
files that pass every existence check).

It is one variable rather than four edited conditions so that the *fifth*
checkpoint call site inherits the answer instead of silently reintroducing the
hang. A test walks the loop's AST and fails if any block containing a
``CheckpointDirector`` gates on ``is_main_process`` alone.

``broadcast_object`` synchronises decisions that must agree across ranks -- the
early-stopping verdict and the best-checkpoint path -- so ranks cannot diverge
on whether to stop.

Reference arms
--------------

``experiments/inprogress/workflow_baselines/`` carries one arm per sharding
strategy, each byte-identical to the ``b1`` control apart from its ``parallel:``
block:

===================================  =========================================
arm                                  exercises
===================================  =========================================
``b2_..._fsdp.yaml``                 Stage-A wrap, sharded clip, gathered save
``b3_..._deepspeed.yaml``            Stage-B adoption, engine-owned accumulation
===================================  =========================================

Both declare ``metadata.baseline: b1_structural_recon_m4raw``, so the pair is a
genuine single-knob comparison: sharding must not change what is optimised, and
a seed-matched b1/b2 run at ``world_size=1`` is the cheapest check that it did
not.

What is verified where
----------------------

Config validation, dispatch, wrap ordering, capability negotiation and the
generated ds_config are all covered by CPU unit tests; the DeepSpeed ones use a
fake engine on a single import site, plus real ``DeepSpeedConfig`` parsing when
the extra is installed.

Actual sharding, memory reduction, consolidated-checkpoint correctness and
loss parity at ``world_size > 1`` need a real multi-GPU run. In particular,
**seed-matched loss parity between ``world_size=1`` and ``world_size=4`` at
matched effective batch is the only test that catches double-scaled gradient
accumulation** -- every single-host test passes with that bug present.
