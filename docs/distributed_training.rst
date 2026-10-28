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

On SLURM, two wrappers in the research tree's ``scripts/training`` directory do
this, and the distribution does not carry them: each states one site's
allocation account and mail domain as data rather than taking them as
parameters. One stands up the allocation and the rendezvous for **one arm**
(single- or multi-node); the other fans a whole cohort out as an array, one task
per YAML, reading each arm's ``parallel.strategy`` and picking the verb for it —
see :ref:`array-dispatch-parallelism`. Neither needs to be told which arms are
distributed.

That pick belongs to ``train``. An array submitted for another pipeline
(``… spectramr infer <yamls>``) runs every arm single-process, because
``parallel.strategy`` states how the arm *trains*; inference under it is the
declaration honoured, not a downgrade.

.. note::

   Launching a process-group arm with plain ``spectramr train`` is not a
   degraded run, it is a refused one: the strategy raises out of
   ``_require_process_group`` after the environment is built. That is by
   design, and it is what the array dispatcher used to walk into — 36 of the
   45 tasks in job array 8589967 (2026-09-12), one per ``kspace_filling`` arm.

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

Derive the rank count rather than typing it — and prefer the committed
derivation over a hand-rolled one, because it is what the refusal above is
measured against::

   source scripts/common/gpu_count.sh
   derive_gpus_per_node        # sets GPUS_PER_NODE, or exits 1 rather than guessing
   torchrun --nnodes=1 --nproc_per_node="$GPUS_PER_NODE" \
       -m spectramr.cli train-distributed --config <arm>.yaml

Both SLURM launchers source that file, so a derivation that disagrees with the
guard cannot be introduced in one of them alone (non-negotiable 17).

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

Collective timeout and rank resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``parallel.timeout_seconds`` is passed to ``init_process_group`` (``null`` keeps
torch's backend default) and stamped into provenance. It is how long peers wait on
a rank that raised before a collective: the survivors block until it expires, so a
shorter value turns a silent hang into an error sooner. Raise it only if
rank-0-only validation or checkpointing legitimately takes longer.

When ``RANK`` is unset but ``WORLD_SIZE`` is declared (a hand-rolled ``srun``
without ``torchrun``), the rank is read from ``SLURM_PROCID``. Without a declared
``WORLD_SIZE`` it is not, because ``srun -n N`` then runs N independent jobs.

Under DDP/FSDP/DeepSpeed each loader is rebuilt around a ``DistributedSampler``.
The rewrap keeps the source loader's generator, timeout, multiprocessing context
and pin-memory device, refuses a loader built with a custom ``batch_sampler``, and
logs a warning when it replaces a custom sampler.

The one custom sampler it does not replace is the MRIxFields
``VolumeBlockedSliceSampler``, whose volume-blocked order is what keeps slice
expansion affordable. Each rank gets a **static share of the containers**: those
that share a volume stay together, so a volume is decoded on one rank rather than
on all of them, and the components are placed greedily to balance sample counts.
With fewer components than ranks it falls back to splitting single containers. Every
rank is padded to the longest share by repeating its own samples (as
``DistributedSampler`` does), because FSDP and ZeRO-3 forwards are collectives and
ranks that disagree on batch count would hang. More ranks than containers is refused.

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
**before** the container is built, so the summary is printed at the level the
entry point and the config resolved (see *Console-logging configuration* in
:doc:`run_provenance_and_logging`); on an arm setting ``level: warning`` it is
suppressed like every other INFO line. The same facts are stamped into the provenance
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

Under FSDP2 every sharded parameter, and so every gradient, is a DTensor, and
``torch.nn.utils.clip_grad_norm_`` over DTensor gradients all-reduces the squared
norms before scaling: the plain utility is already the global clip, so FSDP needs no
step policy of its own. The training loop clips through the strategy's
``_clip_and_log_gradients`` hook; ``_compute_gradient_norm`` returns the replicated
norm as a plain tensor so callers never mix DTensor and Tensor arithmetic.

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

The registry deliberately accepts a **class or an instance**: a class is built by
the loop layer, which alone knows the arm's ``gradient_clip_value``, while
DeepSpeed hands back a configured instance because only the engine knows itself. ``_install_parallel_step_policy`` resolves that;
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

The engine steps the scheduler it adopted, at every accumulation boundary.
``DeepSpeedStepPolicy.owns_lr_scheduler`` is ``True`` so the training loop does
not step it a second time; without it the schedule decays at twice the
configured rate and the run still looks healthy.

**Optimizers under ZeRO.** The engine adopts the optimizer this repo built, and it
refuses some from *inside* ``deepspeed.initialize``: at ``zero_stage >= 1`` anything
outside Adam, AdamW and Adagrad (and DeepSpeed's own fused variants) asserts, and
any ``offload_optimizer`` other than ``none`` raises unless the client optimizer is
``DeepSpeedCPUAdam`` or ``DeepSpeedCPULion``. Under offload this repo therefore rebuilds
the Adam or AdamW it built as ``DeepSpeedCPUAdam`` (same parameter groups and
hyperparameters; ``ZenFlowCPUAdam`` when ZenFlow is on), and refuses any other optimizer,
as well as ``amsgrad`` and ``maximize``. The audit reports these as
``deepspeed_optimizer_supported`` so they fail before cluster time is spent. ``round_robin_gradients`` is a stage-2 option and is
refused elsewhere rather than advertised and ignored.

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
``deepspeed.allow_multi_engine: true``. That knob stays the gate until the
two-engine GPU test passes (``test_deepspeed_multi_engine_gpu.py``, #2631); the
pair is refused at ``zero_stage: 3`` regardless.

**Complex / k-space arms with fp16.** ``get_autocast_context`` disables autocast
for complex+fp16 because there is no ``complex16``. DeepSpeed casts weights to
half from *inside* the engine, where that guard cannot see it. The audit makes
this an error; use ``bfloat16``.

Which training strategies run under which backend
-------------------------------------------------

The parallel wrapper only reaches what is in ``env.models``, and the training
strategy is built *after* it. A strategy that owns a module outside that dict
(CycleGAN's second generator, JEPA's predictor), adds a parameter group after the
optimizer was adopted, or calls a non-forward method on the generator is
therefore not parallelised the way the config says. Under DDP or ZeRO-0..2 the
ranks train different copies and nothing errors; under FSDP or ZeRO-3 the call
reads a shard.

``BaseTrainingStrategy.parallel_compat`` declares the answer, as a subset of two
tokens (:mod:`spectramr.infrastructure.distributed.strategy_compat`):

.. list-table::
   :header-rows: 1

   * - Token
     - Backends
   * - ``"replicated"``
     - ``dp``, ``ddp``, DeepSpeed ``zero_stage`` 0-2
   * - ``"sharded"``
     - ``fsdp``, DeepSpeed ``zero_stage`` 3

``None`` (the default) is *undeclared*. An undeclared strategy is refused on a
sharded backend and runs, with an info line, on a replicated one. ``frozenset()``
means single-process only. A restriction is inherited by subclasses; a
permission is not, so a subclass redeclares rather than borrowing its parent's
audit.

The gate runs twice: ``TrainingEnvironmentDirector`` raises before any model is
built, and ``spectramr audit`` reports it as ``strategy_parallel_compat`` so the
arm is caught before it is submitted. Currently refused for every parallel
backend: CycleGAN, CUT, StarGAN v2, JEPA, multi-contrast contrastive, distillation,
privileged learning, VF consistency distillation, and the strategies that train
parameters they create themselves and add to ``opt_g`` after the wrap (LOUPE, SCAS,
PILOT, MRI-SLAM, IB-VF, SpinSDE, XField-FM, Universal, TTO, Teichmuller, Adaptive
SFC-HSSC, Beltrami, CoordKSpaceGen, CycleBloch, FieldProbeCoupled, GIRFAware,
Disentangled, PhysicsEquivariantSSL and the multi-stage pipeline). Under DDP each rank
would train its own copy of those parameters, and DeepSpeed's ZeRO optimizers have no
``add_param_group`` at all, so the arm would crash on the first step. Reconstruction
declares both tokens; diffusion declares ``replicated``.

A strategy whose owned parameters exist only under some configuration
(``model.conditioning.enabled`` via the conditioning mixin, ``uncertainty.enabled`` in
GeoMamba-ULF, ``uncertainty_weighting`` in the one-shot multi-parameter strategy) is
left undeclared: the declaration is per class, so the audit cannot tell the two cases
apart yet.

Compilation is placed per strategy
----------------------------------

``torch.compile`` used to be applied at one fixed point -- director step 1,
before every wrap -- which is the right point for exactly one strategy. Where it
goes now is a property of the backend:

=====================  =============  =============================================
``strategy``           compiled at    why
=====================  =============  =============================================
``none``               after Stage B  last, and after the optimizer is built
``ddp``                after Stage B  dynamo must SEE the DDP wrapper to engage
                                      DDPOptimizer
``dp``                 after Stage B  allowed, advised against: DataParallel
                                      re-replicates every forward
``fsdp``               after Stage A  ``torch.compile(FSDP(m))``, the
                                      recommended order
``deepspeed`` z0/1/2   after Stage A  ``initialize()`` must receive an
                                      already-compiled module
``deepspeed`` z3       **refused**    measured crash; use DeepCompile instead
=====================  =============  =============================================

The measurements behind each row are in
:mod:`spectramr.infrastructure.training.builders.compile_placement`. Two are
worth repeating here.

**DDP.** ``torch/_dynamo/backends/distributed.py`` states that DDPOptimizer
"applies when dynamo compiles models wrapped in DistributedDataParallel". It
splits the graph at gradient-allreduce bucket boundaries so communication
overlaps with backward compute. The old order, ``DDP(torch.compile(m))``, hides
the wrapper from dynamo and forfeits that overlap entirely -- and the audit did
not cover ``ddp`` at all, so the combination passed review in silence.

**ZeRO-3.** Handing ``deepspeed.initialize`` an already-compiled module raises
``AttributeError: 'dict' object has no attribute '_in_forward'``, and calling
``engine.compile()`` afterwards raises inside dynamo. Both are refused at config
load rather than after the environment is built. DeepCompile
(``parallel.deepspeed.compile.enabled``) is the supported route for ZeRO-3; it
is torch.compile with the ZeRO collectives inserted as graph passes, which is
why it can do what pre-compiling cannot.

One consequence is worth knowing. For ``fsdp`` and ``deepspeed`` the wrap must
precede the optimizer, so the optimizer introspects a wrapped module -- which
breaks ``optimization.optimizer.param_groups``, whose keys are matched against
``named_parameters()`` prefixes. The placement records this as
``optimizer_sees_wrapper``; ``none``/``ddp``/``dp`` are clear of it because they
compile after the optimizer exists.

Regional compilation, and the complex opt-out
---------------------------------------------

Full compilation hands Inductor one large problem and pays the whole cost at
cold start. A model built from *n* copies of one block class is mostly the same
problem *n* times, so ``optimization.compile.regional`` compiles each child of a
uniform ``nn.ModuleList`` separately and hits the compiler cache after the
first. The parent is left uncompiled on purpose; wrapping the root as well would
reinstate the cost this avoids. If a model exposes no qualifying block list it
**raises** rather than quietly compiling the whole thing, because an arm that
asked for regional and silently got whole-model would be reporting a
configuration it did not run.

**Most models here do not qualify, and that is the expected outcome.** Of 877
model files, 187 mention ``nn.ModuleList`` but only 31 build one from a
``range`` comprehension -- the shape the detector matches -- while 319 stack
their blocks in ``nn.Sequential``, which it never matches by construction. Read
``regional`` as an opt-in for the minority that fits rather than a switch worth
trying on an arbitrary arm.

**A GAN needs ``apply_to``.** ``regional`` refuses an arm whose models do not
all expose a qualifying block list, and a patch critic is a plain
``nn.Sequential`` -- so without a selection the whole arm is refused on account
of a model that is discarded at inference anyway::

    optimization:
      compile:
        enabled: true
        regional: true
        apply_to: [generator]

``apply_to`` is a closed vocabulary (``generator``, ``discriminator``,
``encoder``, ``decoder``), so a misspelling raises at config load; naming a
model this arm does not build raises at build time. Omit it to compile
everything, which is what shipped.

**Only the outermost qualifying list is compiled.** Where a stage is itself a
stack of blocks, compiling on a parent-before-child walk descends into the
``OptimizedModule`` it has just created and compiles the inner blocks as well:
measured on a two-stage model whose stages each hold three layers, that is 8
compilations where 2 were intended, and every key doubly wrapped
(``stages.0._orig_mod.layers.0._orig_mod.conv.weight``). Nesting
``torch.compile`` that way destroys exactly the cache reuse the feature exists
for, so the walk stops descending once it has claimed a list.

Regional compilation changes the checkpoint keys. Wrapping a child rather than
the root puts the marker mid-path -- ``blocks.0._orig_mod.conv.weight`` -- and a
leading-prefix strip leaves it there. ``core.module_utils`` drops the synthetic
wrapper segments at any depth now. ``module`` is deliberately excluded from that:
it is an ordinary attribute name, so ``encoder.module.weight`` must survive, and
it is stripped only while it leads.

The complex opt-out
~~~~~~~~~~~~~~~~~~~

Inductor cannot generate code for complex operators. It does not fail on one --
``torch/_inductor/lowering.py`` routes it to an eager fallback and warns **once
per process** through ``@functools.cache``, which on a cluster is
indistinguishable from silence. So ``check_compile_with_complex_model`` is an
error for all 234 complex arms, and stays one by default.

``optimization.compile.allow_complex`` relaxes it, and is accepted **only**
alongside ``regional``. The opt-out rests on the physics SSOT being fenced out of
every dynamo graph with ``core.compile_fences.dynamo_disable``, so the compiled
regions provably contain no complex tensors rather than hopefully so. Measured
with ``torch._dynamo.explain`` on a real-valued backbone around a complex
``fft2c``/``ifft2c`` round trip:

==========  ========  ========  =======================================
variant     graphs    ops       what is in the graph
==========  ========  ========  =======================================
unfenced    1         19        the complex ops, where Inductor falls
                                back to eager per op
fenced      2         9         only the real-valued regions
==========  ========  ========  =======================================

The break reason is reported as *"Skip calling ``torch.compiler.disable()``d
function"*, which is the fence working as intended.

**One graph break per fence is not free**, and these numbers show the mechanism
works, not that it is faster. Whether the trade pays on a given arm is an
empirical question for that arm. ``allow_complex`` is also refused together with
``fullgraph``: a graph break under ``fullgraph`` raises, so the pair is a
guaranteed crash and is rejected at config load rather than at the first forward
pass.

**The fence is applied at import and unconditionally**, so every arm pays it
whether or not it compiles -- ``fft2c``/``ifft2c`` have around 480 call sites
across ``src/``. ``torch._dynamo.disable`` wraps the function, and that wrapper
costs something in eager mode too. Measured on this machine (T500), one
``fft2c`` call:

============================  ==========  ==========  ===============
shape                         bare        fenced      delta
============================  ==========  ==========  ===============
GPU 8x1x320x320               1608.6 us   1610.0 us   +1.4 us (+0.1%)
CPU 1x1x64x64                 53.4 us     59.6 us     +6.1 us (+11.4%)
============================  ==========  ==========  ===============

The overhead is a fixed per-call cost, so it is invisible against a realistic
transform and material only where the transform itself is trivial. Training runs
on the accelerator (non-negotiable 9b), so the first row is the one that governs
-- but the cost is real, unconditional, and stated here rather than assumed away.

Recompilation is a budget, and exhausting it used to be silent
--------------------------------------------------------------

Dynamo recompiles a frame when its guards fail -- a new shape, a new dtype. Each
frame has a budget (``torch._dynamo.config.recompile_limit``, 8 by default), and
on exhaustion torch **drops that frame to eager and carries on**:
``fail_on_recompile_limit_hit`` is ``False`` upstream. The run then reports a
compiled configuration while executing something else, which is the same lie
``apply_compile`` already refuses at build time (#619 F2) arriving later in the
run instead.

MRI is where this bites. Slice and coil counts vary between volumes, and
``dynamic: true`` is this block's default, so guard failures are expected rather
than exceptional.

``optimization.compile.fail_on_recompile_limit`` therefore defaults to **true**,
inverting torch's default: exhausting the budget raises. ``recompile_limit``
raises the budget when an arm legitimately needs more shapes. Declaring a budget
with ``fail_on_recompile_limit: false`` is refused at config load -- a budget you
decline to enforce is not a budget.

This is a real behaviour change for an arm that is silently degrading today: it
will start raising. That is the point, and the message names the two knobs.

Checkpoints
-----------

Every wrapper renames the keys ``state_dict()`` emits -- ``torch.compile`` adds
``_orig_mod.``, DP/DDP and ``ModelEma`` add ``module.``, and checkpoints written
under the former FSDP1 carry ``_fsdp_wrapped_module.`` (FSDP2 adds no prefix). :mod:`spectramr.core.module_utils` is the single place
that strips them, applied at every save and load site.

This is not cosmetic. Every inference and evaluation path builds a **bare** model
and loads with ``strict=True``, so a wrapped checkpoint raises there -- and under
``strict=False`` (campaign evaluation, warm-start, distillation) it matches
nothing, loads nothing, and reports success, so a randomly-initialised model
produces metrics that read as a bad arm.

Under FSDP the keys are the smaller problem. FSDP2 keeps the parameter names but
makes every parameter a DTensor shard, so ``generator.state_dict()`` is 1/N of each
weight and ``optimizer.state_dict()`` one rank's slice of the Adam moments. Both are
read through the checkpoint adapter's ``state_io``
(:mod:`spectramr.infrastructure.distributed.state_io`), which gathers with the
distributed-checkpoint API (``get_model_state_dict`` / ``get_optimizer_state_dict``,
full state dict, offloaded to rank 0) and scatters on resume with the ``set_*`` pair.
Every one of those calls is collective, so every rank enters them in the same order.
The EMA shadow is sharded like the generator (``prepare_ema``), so its blend and the
validation swap stay shard-local, and its checkpoint form is gathered the same way.

Every checkpoint records the topology it was written at (``world_size`` and
``parallel.strategy``), and a resume at a **different world size is refused**.
``training.max_iterations`` is a per-rank bound and the LR schedule is indexed by
iteration, so the same ``global_step`` at 4 ranks has seen 4x the samples it had at 1,
and the data epoch and shuffle seed (derived from the per-rank loader length) no longer
line up with what was saved. Set ``checkpoint.allow_world_size_change: true`` to accept
that deliberately; it is logged as a warning. A checkpoint without the record predates it
and is accepted, because its world size is unknown rather than different. For DeepSpeed
the check runs after the sharded restore, which has its own partition-count refusal.

The recorded ``parallel.strategy`` is checked too. ``none``, ``dp`` and ``ddp`` write
interchangeable checkpoints, but an FSDP checkpoint carries unsharded optimizer state keyed
by parameter name and a DeepSpeed one carries sharded per-rank state, so a resume that
swaps in or out of ``fsdp`` or ``deepspeed`` is refused unless
``checkpoint.allow_parallel_strategy_change: true`` (the weights load; the optimizer state
may not).

EMA under a wrapper
-------------------

The same renaming bites at *runtime*, not just at save/load. ``ModelEma`` holds a
shadow deep-copied from the **bare** module at ``build_ema()`` time, while the
live generator handed to ``update()`` has since been wrapped at Stage A/B. The
blend is key-matched::

    for k, ema_v in esd.items():
        if k in msd:

so a shadow whose keys are all absent from the live model blends **nothing** —
and raises nothing. ``num_updates`` still increments, the decay ramp still
advances, and a checkpoint is still written, so every observable stays healthy
while the shadow holds its random initialisation forever. Measured against a real
DeepSpeed engine: shadow ``0.weight`` against engine ``module.0.weight``, overlap
zero, on 75 arms (#2172).

Under FSDP the live generator holds DTensor shards, so the shadow is sharded the
same way at Stage A (the plugin's ``prepare_ema``, through ``shard_like``). The
blend then runs shard against shard with no gather, and validation installs the
shadow through ``state_io.swap_in`` the same way; an unsharded shadow beside a
sharded generator is refused rather than gathered on every swap. A failed update
raises: it used to be logged as a warning, which left EMA frozen while validation
went on grading it as EMA.

Three consequences worth knowing:

* **Callers unwrap.** ``ema.update(unwrap_model(generator))`` at every site. A
  total key mismatch now raises ``EMAKeyMismatchError`` rather than being
  skipped; a *partial* overlap still blends, because a rebuilt
  ``channel_adapter`` legitimately leaves keys unmatched.
* **The checkpoint form is** :meth:`~spectramr.infrastructure.optimization.ema.ModelEma.shadow_state_dict`.
  ``unwrap_model(ema)`` peels ``ModelEma``'s own ``.module``, which bypasses the
  ``state_dict`` override that persists the warmup counter — so the decay ramp
  restarted at 0 on every resume. Weight keys are bare either way, so existing
  checkpoints load unchanged.
* **Validation swaps in place.** It does not forward the EMA module; it copies
  the shadow weights into the live generator, forwards the generator, and copies
  the originals back. That swap is held in one context manager
  (:mod:`spectramr.infrastructure.optimization.ema_swap`) so the mutation cannot
  outlive its own restore, and a restore that cannot put a tensor back — its
  shape changed mid-forward — raises rather than leaving training on a blend of
  trained and shadow weights.

Checkpoints under DeepSpeed
---------------------------

Every save writes two things. The engine's sharded tag directory goes to
``checkpoints/deepspeed/<model>/step_NNNNNNNN`` (or ``.../best``), and resume reads
it, so ZeRO optimizer partitions survive. The RunState file beside it,
``checkpoints/step_NNNNNNNN.<format>`` or ``best.<format>``, carries gathered
weights: at ZeRO-3 they are consolidated on rank 0 in 16-bit precision, a
collective every rank enters. Inference, campaign evaluation and warm-starts read
that file and never need DeepSpeed. The best file carries the EMA shadow at stages
0-2; at stage 3 the shadow is partitioned and is not saved, and
``restore_best_weights`` refuses rather than pair the best weights with the last
EMA.

``--resume`` goes through :func:`spectramr.infrastructure.checkpointing.resume.resume`,
which reads both artifacts. It loads the tag directory named by the file's
``meta.native_tag`` first; the adapter reports which model keys it restored, and
the weights and optimizer state of exactly those keys are skipped in the RunState
file. Every other part is still restored from the file: the EMA shadow, the
scaler, early-stopping state, RNG streams, the LR schedulers and any model that is
not an engine, such as a plain discriminator beside an engine-wrapped generator.
Re-loading a handled key would hand the ZeRO stage 1/2 optimizer a dict where it
indexes a per-rank list, which raised
``KeyError: 0`` on every resume (#2482). At world size above 1 it would also
overwrite each rank's optimizer partition with the writer rank's (#1309). The
schedulers stay on the generic path because the engine restores only a
scheduler it owns, and a second load of replicated scheduler state changes
nothing.

At ZeRO stage 3 with EMA enabled, ``--resume`` refuses before loading anything: the
step file carries only rank 0's partition of the shadow, and loading it would give
every rank that partition. :doc:`checkpointing` ("ZeRO-3 with EMA") says what to do
instead.

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
checkpoint save gates on ``is_main_process`` alone.

``broadcast_object`` synchronises decisions that must agree across ranks -- the
early-stopping verdict and the best-checkpoint path -- so ranks cannot diverge
on whether to stop.

FSDP precision
--------------

``fsdp.mixed_precision`` is ``fp16``, ``bf16`` or ``null`` (follow
``optimization.precision``). ``optimization.precision`` with no dtype resolves to
``fp16``, so an arm that enables AMP runs fp16 under FSDP. ``bf16`` needs Ampere or
newer, and V100 has no bf16: fp16 is the half precision there.

fp16 runs under the same ``ComplexGradScaler`` as single-process arms, built with
``sharded=True``. Its non-finite flag is all-reduced (MAX) across ranks, so an inf
gradient on one rank makes every rank skip the same step and back off the scale
identically. Neither half type has a complex counterpart, so a model with complex
parameters is refused at wrap time under fp16 or bf16 and runs in ``float32``. The
scaler class is stamped into provenance as ``fsdp.grad_scaler``.

Gradients are reduced in ``float32`` whatever ``mixed_precision`` is, and module
buffers (BatchNorm running statistics) keep their own dtype: FSDP2 never casts them.
``fsdp.reduce_dtype: match_param`` reduces in the parameter dtype, which halves the
reduce-scatter traffic and loses gradient precision. It is stamped into the run's
provenance.

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
