Training throughput: which lever, in what order
================================================

Every acceleration mechanism in this framework is wired and reachable from YAML,
and a default run uses almost none of them:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Lever
     - Where it is declared
   * - Gradient accumulation
     - ``optimization.gradient.accumulation_steps``
   * - AMP (autocast)
     - ``optimization.precision.enabled`` / ``.dtype``
   * - Gradient checkpointing
     - ``optimization.gradient.enable_checkpointing``
   * - ``torch.compile``
     - ``optimization.compile.enabled``
   * - FSDP / DeepSpeed
     - ``parallel.strategy``
   * - Fused optimizer step
     - ``optimization.optimizer.fused``

So the common case is a single-GPU, eager, fp32 run. That is usually not a
deliberate choice — it is the default nobody revisited.

No in-repo speedup numbers exist
--------------------------------

This page orders the levers by cost-to-try and states what each one trades. It
deliberately quotes **no speedup figures**, because none have been measured on
this framework's models and data. The ordering below is a starting point for
your own measurement, not a result. If you benchmark a lever, land the numbers
here.

The ladder
----------

Work down it. Each rung costs more to adopt and is harder to attribute a result
to, so stopping early is usually right.

**1. Gradient accumulation** — ``optimization.gradient.accumulation_steps``

Buys effective batch size at no memory cost and no numerical change, and is the
cheapest of these to adopt. Not a throughput win on its own: it trades
wall-clock for batch size, so reach for it when the batch is the constraint, not
when the clock is.

**2. Fused optimizer step** — ``optimization.optimizer.fused: true``

The cheapest genuine throughput lever: one line, no numerical change to the
update, no interaction with anything else. It collapses the per-parameter
elementwise step into a single multi-tensor kernel, so the win scales with
parameter-tensor **count** rather than size — which suits the deep U-Nets and
unrolled cascades here. Requires CUDA parameters. Declaring it on an optimizer
whose constructor has no such argument **raises** rather than silently training
un-fused.

Left unset, ``adam`` and ``adamw`` default to fused when every trainable parameter
is a real floating CUDA tensor (the default is stamped in provenance as
``applied_defaults``). An explicit ``fused: false`` wins; a complex or CPU parameter
keeps the unfused path.

**3. AMP** — ``optimization.precision: {enabled: true, dtype: bfloat16}``

The largest single-GPU lever, and the first one that changes numerics.

Prefer ``bfloat16``: fp32 exponent range, so no ``GradScaler``, no overflow, no
scale-collapse, and it works under complex autocast. ``float16`` is viable but
carries the loss-scaler failure modes documented in :doc:`troubleshooting`.

.. important::

   **Diffusion arms are excluded.** They train in full fp32 —
   ``precision: {enabled: false, dtype: float32}`` — and
   ``check_diffusion_precision_policy`` makes anything else a hard audit error,
   for bf16 as well as fp16. Training a noise-prediction objective under
   autocast is exactly the failure it exists to catch. The policy and the
   reasoning are under "Mixed precision" in :doc:`troubleshooting`.

   (Referenced by page rather than by label so this page does not depend on the
   merge order of the PR that adds the label.)

**4. ``torch.compile``** — ``optimization.compile.enabled: true``

Real gains on stable-shape models, but the highest failure rate of anything
here: compilation failure **raises** by design, because silently training an
eager model while reporting compiled throughput is a lie about what ran. It is
mutually exclusive with DeepCompile (``check_compile_with_sharded_strategy``).

.. important::

   **Complex / k-space arms are excluded, and not because they crash.** Measured
   on torch 2.11 against this repo's ``fft2c``/``ifft2c`` round-trip,
   compilation *succeeds* under ``fullgraph=True`` and is numerically correct
   (max abs error 3.2e-07). Torchinductor emits one warning — *"does not support
   code generation for complex operators"* — and runs them eagerly. The arm then
   reports a compiled run while its complex regions execute eagerly, possibly
   slower than not compiling at all. ``check_compile_with_complex_model`` makes
   the combination a hard audit error. Details under
   "Complex / k-space arms do not torch.compile" in :doc:`troubleshooting`.

   A GAN that pairs a real-valued model with one that works on k-space compiles only
   the real one: ``optimization.compile.apply_to: [generator]`` (or ``[discriminator]``)
   leaves the other eager, and the check then passes with an info line instead of an
   error. Naming a subset asserts the compiled models never see complex tensors; the
   check still refuses ``generator`` when the registry marks the generator complex.

   Note this is about complex **dtype**, not complex arithmetic:
   ``ComplexConv2d`` stores real and imaginary parts as separate real tensors
   and compiles cleanly.

**4b. Channels-last** — ``optimization.compile.channels_last: [generator]``

Converts the named models to the NHWC memory format, which cuDNN and oneDNN run 2-D
convolutions fastest in. It uses the ``apply_to`` target names and is independent of
``compile.enabled``, so a GAN can convert its image-domain model and leave the k-space one
alone. A model with no 4-D real weight is refused. The converted models are recorded on
``TrainingEnvironment.memory_format``. Off by default and unmeasured: time it per arm on the
target GPU before adopting it.

**5. Gradient checkpointing** — ``optimization.gradient.enable_checkpointing``

Listed here to be clear that it is **not** a throughput lever: it trades compute
for memory, recomputing activations in the backward pass. Reach for it to fit a
model that will not otherwise fit, then use the headroom for a larger batch.

The knob reaches the model through the backbone's own ``set_grad_checkpointing``
hook, and ``KSpaceColdDiffusionGenerator`` **raises at build** when the selected
backbone has none — a memory claim that silently did not happen OOMs later with
nothing to point at (non-negotiable 3). Implemented for ``complex_unet``,
``swin_diff_rec``, ``swin_diff_rec_kan``, ``diff_varnet``, ``diff_varnet_kan``
and ``nafnet``; measured on a ``[2, 8, 128, 128]`` batch, peak allocation falls
by 49–61 % on the latter three and the forward output is unchanged. It is a
**training-only** lever: ``_checkpointing_active()`` requires ``self.training``
and ``torch.is_grad_enabled()``, so the reverse sampler used for validation
takes the plain path, where there is nothing saved for backward to shrink.

One consequence is easy to miss when adding a checkpointed block: PyTorch ends a
non-reentrant recompute by raising ``_StopRecomputationError`` **through** the
code it is re-running, so a ``try``/``except Exception`` anywhere inside a
checkpointed block catches it. The DC layer's shape diagnostic did, and turning
the knob on reported ten ``[DC LAYER CRASH]`` errors per ``diff_varnet`` run
that trained to completion. Catch the error class you are diagnosing --
``RuntimeError`` for a broadcast failure -- never ``Exception``.

**6. Sharding** — ``parallel.strategy: fsdp | deepspeed``

Only once a single GPU is genuinely the constraint. Needs a launcher
(``torchrun``); there is no auto-detection, and forgetting it is an error rather
than a fallback. ZeRO-2 partitions optimizer state and gradients; ZeRO-3 adds
parameters and admits CPU/NVMe offload. DeepSpeed also brings DeepCompile and
ZenFlow. All of it — including the ways each can be declared and stay inert — is
in :doc:`distributed_training`.

DeepSpeed is deliberately excluded from the ``[all]`` and ``[dev]`` extras::

    pip install -e '.[deepspeed]'

``check_deepspeed_extra_installed`` catches a missing extra at audit time
(~100 ms) rather than after the cluster node has built the whole training
environment.

Backend flags are not per-arm knobs
------------------------------------

TF32 and the cuDNN autotuner are process-level, set once by
:func:`spectramr.accelerator.seed_everything` and logged
(``[Accelerator] determinism=… cudnn.benchmark=… allow_tf32=…``).

* ``allow_tf32`` defaults **on**. It is reproducible, merely lower-precision, and
  is deliberately a *separate axis* from determinism — an arm can have
  bit-for-bit reproducibility and TF32 matmul throughput at once.
* ``cudnn.benchmark`` follows ``deterministic``. The autotuner is **not** a free
  win: it re-benchmarks on every new input shape, so variable patch sizes, coil
  counts and last-batch remainders can make it a net loss.

Start from a reference arm, not from scratch
---------------------------------------------

``experiments/inprogress/workflow_baselines/`` holds a seed-matched ladder whose
members differ from the control in exactly one respect:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Arm
     - What it adds over ``b1``
   * - ``b1_structural_recon_m4raw``
     - the control — plain single-GPU reconstruction
   * - ``b2_..._fsdp``
     - ``parallel:`` only, FSDP
   * - ``b3_..._deepspeed``
     - ``parallel:`` only, DeepSpeed ZeRO-2
   * - ``b4_..._zero3_deepcompile_zenflow``
     - ZeRO-3 + DeepCompile + ZenFlow

That single-knob discipline is the point: none of b2–b4 may change *what* is
optimised, only how it is scheduled and where it lives, so a seed-matched
``b1``/``bN`` comparison is the check that they did not. Copy the ``parallel:``
block from the closest one rather than writing it fresh — several of its fields
must agree with each other, and the audit enforces the agreement rather than
guessing.

Before committing GPU time
---------------------------

Run the audit. It rejects most misconfigurations of everything above in ~100 ms::

    spectramr audit experiments/inprogress/<paradigm>/<arm>.yaml

Add ``--probe`` for the Tier-2 synthetic forward pass, which is what catches AMP
and OOM problems — and note it must run on an accelerator, because a CPU probe
cannot see CUDA OOM or a GradScaler trap, the two things it exists for.

.. seealso::

   :doc:`distributed_training`
       FSDP / DeepSpeed / ZeRO-3 / DeepCompile / ZenFlow in full.

   :doc:`troubleshooting`
       Autocast dtypes, the diffusion fp32 policy, and the grep-under-reports trap.

   :doc:`accelerated_run_contract`
       Why a heavy pipeline raises instead of falling back to CPU.

Defaults that favour throughput
-------------------------------

``checkpoint.async_write`` defaults to ``true``: the checkpoint file write moves to a
background thread, and a failed write surfaces at the next save or at the end of the
loop. Set it to ``false`` to write on the training thread. ``data.loader.persistent_workers``
also defaults to ``true`` (workers, and their per-worker caches, stay alive between epochs).

Node-local data staging
-----------------------

A random slice read from the shared filesystem costs tens of milliseconds, and a loader that
does one per item per repetition can starve the GPU. ``data.staging.mode: node_local`` copies
the k-space files the train, validation and test indices name (and, for M4Raw, the coil-map
files they use) under ``data.staging.root`` (default ``/tmp``) once, in the main process before
any worker starts, and reads the copies. The default ``off`` reads in place.

Jobs that share a node serialise on one lock and reuse each other's copy; a copy interrupted by
a kill is discarded and made again. A set that does not leave ``min_free_gb`` free raises;
``when_full: read_in_place`` is the explicit, logged opt-out. Wired so far: M4Raw and the fastMRI-style HDF5 route (``kspace`` / ``image`` with an index); other dataset types read in place.
``scripts/diagnostics/simulate_staging.py`` models the mechanism on a synthetic tree; it does not
measure any cluster.
