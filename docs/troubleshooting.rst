.. _troubleshooting:

=========================================
Troubleshooting & FAQ
=========================================

.. sectionauthor:: spectraMR Research

This guide covers the most common errors encountered when developing,
running, and debugging experiments in the spectraMR framework.

.. contents:: Table of Contents
   :depth: 2
   :local:


Shape & Channel Errors
=======================

``Expected input channels X, got Y``
--------------------------------------

**Root cause**: ``model.in_channels`` doesn't match ``data.coil_processing_mode``
output channels.

**Fix**:

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - ``coil_processing_mode``
     - ``in_channels``
     - Fix
   * - ``rss``
     - 1
     - Set ``model.in_channels: 1``
   * - ``sense``
     - 2
     - Set ``model.in_channels: 2``
   * - ``flatten``
     - ``2 × num_coils``
     - Set ``model.in_channels: 2 * num_coils``

Run dry-run to validate before training:

.. code-block:: bash

   python -m spectramr.cli train --config my.yaml --dry_run

``RuntimeError: size mismatch for conv1.weight``
-------------------------------------------------

**Root cause**: Loading a checkpoint with different ``in_channels`` or
``out_channels`` than the current config.

**Fix**: Either match config to checkpoint architecture or use
``load_optimizer_state: false`` + ``resume_training: false`` and
fine-tune from scratch with the new channel count.


Config / Schema Errors
========================

``ValidationError: config_version field required``
----------------------------------------------------

Add to your YAML:

.. code-block:: yaml

   config_version: "1.0"

``AttributeError: 'TrainingSettings' object has no attribute 'lr'``
--------------------------------------------------------------------

Flat aliases were removed. Use the nested path — and note that
``optimization.learning_rate`` is **also** retired: the optimizer block was
decomposed and the field moved one level deeper, so the obvious first guess
raises the same ``AttributeError`` you are trying to fix.

.. code-block:: python

   # ❌ flat alias — never existed on the schema
   config.lr
   config.lambda_l1

   # ❌ one-level spelling — retired when the optimizer block was decomposed
   config.optimization.learning_rate
   # AttributeError: 'OptimizationConfigSchema' object has no attribute 'learning_rate'

   # ✅
   config.optimization.optimizer.learning_rate
   config.losses.image_losses[0].weight

When you are unsure where a field lives now, resolve it through
``spectramr/config/schemas/renames.py`` rather than guessing: it names the
canonical location for every retired spelling, and says whether the old one
still folds silently or raises.

``extra fields not permitted``
-------------------------------

``TrainingSettings`` itself sets ``extra='forbid'``, and so do 263 of the 353
config schema classes — an unknown key at those levels raises. It is **not**
universal, though: 23 classes set ``extra='allow'`` (``AdapterStepSchema``, for
one, because an adapter's kwargs are arbitrary by design) and 67 leave it
unset, which means Pydantic's default of ``ignore``. A typo inside one of those
sub-blocks is dropped in silence rather than reported, so a config that loads
is not proof that every key in it was read. Remove unknown
fields or check for typos. Common culprits: ``enable_ema`` (→ ``ema.enabled``),
``lambda_l1`` (→ ``losses.image_losses[].weight``).

``DomainMismatch Pre-Flight Failure``
--------------------------------------

The ``ConfigHealthChecker`` validates before GPU allocation. Common fixes:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Error Message
     - Fix
   * - ``output_domain=image but no image_losses``
     - Add at least one entry to ``losses.image_losses``
   * - ``in_channels mismatch``
     - Match ``model.in_channels`` to ``data.coil_processing_mode``
   * - ``Manifest not found``
     - Verify ``train_manifest`` path exists on current machine
   * - ``DC disabled in reconstruction mode``
     - Set ``physics.data_consistency.enabled: true``


AMP / NaN Gradient Errors
===========================

``loss is nan after N steps``
------------------------------

**Causes:**

1. **AMP float16 overflow** — k-space values can exceed float16 range
2. **Learning rate too high** — especially at start of training
3. **Loss weight imbalance** — one loss dominates and explodes

**Fixes (try in order):**

.. code-block:: yaml

   # Fix 1: switch to bfloat16
   optimization:
     amp_dtype: bfloat16

   # Fix 2: disable AMP entirely
   optimization:
     use_amp: false

   # Fix 3: use gradient clipping
   optimization:
     gradient_clip_val: 1.0

   # Fix 4: reduce warmup LR
   optimization:
     warmup_iterations: 5000
     learning_rate: 1e-5      # start lower

``GradScaler got inf/nan`` at step 1
--------------------------------------

The k-space tensor contains unnormalized values. Ensure
``kspace.enforce_hermitian_symmetry: true`` and that the
normalization transform is applied before the loss.


Out-of-Memory (OOM) Errors
============================

``CUDA out of memory at batch_size=N``
---------------------------------------

**Quick fixes** (in order of impact):

.. code-block:: yaml

   # 1. Reduce batch size
   data:
     batch_size: 2

   # 2. Gradient accumulation (equivalent effective batch)
   optimization:
     gradient_accumulation_steps: 4    # effective = 2 × 4 = 8

   # 3. Enable AMP (halves memory for activations)
   optimization:
     use_amp: true
     amp_dtype: bfloat16

   # 4. Patch-based inference for large volumes
   inference:
     tiling:
       enabled: true
       patch_size: [128, 128]

``OOM during validation but not training``
-------------------------------------------

Validation runs full-resolution volumes without gradient checkpointing.
Reduce validation batch size:

.. code-block:: yaml

   validation:
     validation_batch_size: 1


``OOM inside ComplexMHA / dual_domain_attention_kan``
------------------------------------------------------

The phase-aware complex self-attention in
:class:`spectramr.models.blocks.dual_domain_attention_kan.ComplexMHA` formed a
dense ``[B, h, N, N]`` score matrix over the full feature-map sequence
(``N = H*W``). On a 256² map this is tens of GiB and OOM'd every
``experiment_11`` KAN dual-domain arm on the 44 GiB cluster GPUs (2026-06
rerun, traceback ending in ``ComplexMHA.forward`` ``attn @ Vh``).

``forward`` now **chunks over the query dimension** above
``_COMPLEX_MHA_QUERY_CHUNK`` (default 2048): softmax / top-k act per query on
the key dimension, so the chunked result is *numerically identical* to the
dense one (pinned by
``tests/unit/models/blocks/test_dual_domain_attention_oom_caps.py::test_complex_mha_query_chunking_is_numerically_exact``)
while peak attention memory drops from ``O(N²)`` to ``O(chunk·N)``. This
mirrors the ``max_band_tokens`` cap already on ``RadialBandAttention`` and
``MultiScaleFreqBandAttention``. No YAML knob — the chunk size is an internal
memory optimisation; override per-instance via ``self._query_chunk`` if needed.

**Real-matmul scoring (follow-up to the chunking).** Chunking bounds the score
tensor to ``chunk·N`` but the 2026-06-22 cluster reruns still OOM'd at the
score line (``Tried to allocate 256.00 MiB`` at
``dual_domain_attention_kan.py:158``): the chunked score was still built in
**complex** before ``.real`` was taken, doubling its bytes. Because
``Re(q kᴴ) = qr·krᵀ + qi·kiᵀ``, the score is now computed from two **real**
matmuls via :func:`spectramr.models.blocks.dual_domain_attention_kan._phase_aware_real_scores`,
and the value aggregation keeps the weight tensor real via
:func:`spectramr.models.blocks.dual_domain_attention_kan._complex_weighted_sum`
(``ComplexMHA`` and ``CrossDomainAttention`` both use them). The result is
numerically identical (max abs diff ``~4e-7``, pinned by
``test_phase_aware_real_scores_matches_complex_reference``,
``test_complex_weighted_sum_matches_upcast_reference`` and
``test_complex_mha_forward_matches_complex_matmul_reference``); measured peak
at the failing ``[1, 4, 2048, 4096]`` frame drops **512 → 384 MiB (≈25 %,
128 MiB freed)**, which is the headroom the 32 GiB-card arms died for. The
44 GiB-card arms get the same 128 MiB relief but may still need a config lever
(``optimization.use_gradient_checkpointing: true``, a lower
``max_dense_attn_tokens``, or an 80 GiB GPU) since their peak is the aggregate
U-Net forward, not this single op.

**Cohort-wide token-budget reduction (2026-06-26).** ``max_dense_attn_tokens``
is the supported config lever for the score-tensor peak. When the feature-map
sequence ``N = H*W`` exceeds it, the dense image/cross branches adaptive-pool to
a ``round(sqrt(budget))²`` grid before attending, attend there, then interpolate
back (``dual_domain_attention_kan.py:891``); the **output k-space stays full
256²** regardless, so this only coarsens the *global attention mixing*, not the
data or the reconstruction target. The whole KAN dual-domain family (the
``attention_shootout`` arms, the ``attention_enhancements`` 5×2 matrix, and the
standalone ``experiment_11_kan_dual_domain``) was lowered from ``4096`` (64²) to
``2304`` (48²) to fit the 32 GiB cards. Because the budget sets the pooled
attention resolution, it must stay **uniform across the family**, or a
memory-vs-quality difference confounds the head-to-head deltas; that invariant
is pinned by
``tests/audit/test_kspace_filling_cohort_invariants.py::test_j_dense_attn_token_budget_uniform``.

This is the right knob precisely because ``data.patch_size`` is **not** safe to
shrink on these k-space arms. For ``dataset_type: m4raw`` the Subject's
``input`` / ``target`` keys hold *k-space* (``m4raw_dataset.py:797``), and
``patch_size`` equals M4Raw's native 256² acquisition matrix, so at ``256`` it
crops nothing (the ``UniformSampler`` returns the whole matrix). Reducing it
makes the sampler crop a window of **k-space**, which truncates high spatial
frequencies and permanently lowers the reconstruction resolution, i.e. it
changes the scientific task rather than approximating the attention. Prefer the
token budget (a benign attention approximation) over the patch size (a data
crop) for k-space memory relief.


Distributed (DDP) validation metrics are summed across ranks
=============================================================

Under DDP the validation loader is wrapped in a ``DistributedSampler``
(:func:`spectramr.pipelines.parallel._apply_distributed_samplers`) which *shards
and pads* the val set, so each rank validates only ``~1/world_size`` of it.
``_run_validation`` previously finalised ``v_sum / val_count`` on the **local**
shard and never reduced across ranks, so rank-0 reported (and early-stopped on)
a single padded shard's metric rather than the true full-set value.
:func:`spectramr.pipelines.train._all_reduce_val_metrics` now all-reduce-SUMs
both the per-metric running-sums and the sample count before dividing, giving
the correct sample-weighted global mean. It is a **no-op** when
``torch.distributed`` is not initialised (the default single-process
``spectramr train`` path), so single-GPU runs are unaffected. Pinned by
``tests/unit/pipelines/test_train.py::test_all_reduce_val_metrics_noop_without_process_group``
and ``::test_all_reduce_val_metrics_single_rank_identity``.


Mixed precision: ``precision.dtype`` selects the autocast dtype
================================================================

``optimization.precision.dtype`` chooses the autocast precision when
``optimization.precision.enabled: true``. It maps to the AMP policy in
:func:`spectramr.infrastructure.training.mixed_precision.resolve_amp_precision`:

.. note::

   The two older flat spellings differ, and the difference matters.
   ``optimization.use_amp`` is a ``RENAMES`` entry with posture ``fold``: it
   still loads and still lands on ``precision.enabled``. ``optimization.amp_dtype``
   has posture ``raise`` — declaring it is now a hard load error naming
   ``precision.dtype``. So a ``use_amp`` declaration is live, which has
   a practical consequence worth spelling out: **grepping for the canonical key
   under-reports AMP usage.** A sweep for
   ``optimization.precision.enabled: true`` across
   ``experiments/inprogress/`` returned zero while 46 arms were in fact running
   AMP through the legacy key. Ask ``resolve_amp_precision`` on the loaded
   settings, never the text. Declaring both spellings at disagreeing values
   raises (*"AMP is one decision"*), so migrate by **replacing** the legacy key,
   not by adding the block beside it.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - ``precision.dtype``
     - autocast
     - Notes
   * - *(unset)* / ``float16``
     - fp16
     - Historical default. Needs a ``GradScaler``; narrow dynamic range
       (max ≈ 65504) can overflow to ``inf`` on unstable models. **Disabled
       automatically for complex/k-space flows** (complex64 cannot mix with
       Half weights).
   * - ``bfloat16``
     - bf16
     - **Preferred.** Same exponent range as fp32 → no loss scaling, no
       overflow, no GradScaler scale-collapse. Halves the *autocast'd*
       activations (convs/projections). Works under complex autocast. Full
       Tensor-Core throughput on the cluster's Ada GPUs (and Ampere+).
   * - ``float32``
     - *(off)*
     - Full precision — AMP is disabled even if ``use_amp: true`` (the knob
       cannot silently no-op into fp16).

This knob was **inert before 2026-06-24** (the policy hardcoded fp16); it is
now threaded through ``BaseTrainingStrategy`` (pitfall #15) and the resolved
value is logged at startup (``[Mixed Precision] enabled=… precision=…``).

.. _diffusion-fp32-policy:

Diffusion arms train in fp32 (audit error)
-------------------------------------------

**Diffusion arms do not use AMP — neither fp16 nor bf16.** They declare::

    optimization:
      precision:
        enabled: false
        dtype: float32

``float32`` rather than a bare ``enabled: false`` because that is the third
state the table above documents: the choice is then validated at load time and
stamped into provenance instead of inherited from a default.

``check_diffusion_precision_policy`` (Tier-1, severity ``error``) enforces it.
When it landed it fired on **12 arms** under ``experiments/inprogress/`` that
were training a noise/score-prediction objective under autocast — 11 fp16, 1
bf16 — none of which a grep could see, for the reason in the note above.

bf16 is refused alongside fp16 even though it is the *safer* half-precision and
is the recommended fix elsewhere on this page. The policy is fp32 for diffusion;
a check that allowed one half-precision path would leave it half-enforced.

.. rubric:: How the check decides an arm is "diffusion"

It resolves the strategy class through
:meth:`~spectramr.infrastructure.training.strategy_factory.TrainingStrategyFactory.get_strategy_class`
(the SSOT dispatch) and unions two runtime signals, because measured across all
204 ``training_mode`` keys **neither is sufficient alone**:

* ``issubclass(DiffusionTrainingStrategy)`` misses 15 — the whole cold-diffusion
  family plus ``flow_matching``, ``rectified_flow``, ``x_diffusion`` and the
  riemannian variants inherit straight from ``BaseTrainingStrategy``.
* a ``"diffusion"`` substring on the strategy qualname misses 9 — ``edm``
  (Elucidated Diffusion Models), ``i2sb``, ``stochastic_interpolants``,
  ``twin_dps``, ``bloch_schrodinger_bridge`` and friends, whose class and module
  names say nothing.

A third candidate, ``training.diffusion is not None``, was tried and rejected:
it adds 14 arms, **all false positives** (PINN, FNO, Vision-Mamba and VAE
reconstruction arms that merely carry the sub-block) and zero true ones. In a
check that hard-errors, over-capture blocks AMP on arms legitimately entitled to
it, which is worse than under-capture.

Matching on ``model_type`` is *not* how this works, deliberately: that was a
second AMP resolver inside ``AMPPolicy.should_use_amp`` and was removed in #806.
The durable fix is a declaration rather than this inference —
``StrategyCapabilities.supported_paradigms`` is the seam and is currently
populated on 0 of 204 strategies (issue #810).

.. _complex-no-compile-policy:

Complex / k-space arms do not ``torch.compile`` (audit error)
---------------------------------------------------------------

``check_compile_with_complex_model`` (Tier-1, severity ``error``) refuses
``optimization.compile.enabled: true`` on an arm carrying ``complex64`` tensors.

**Not because it crashes.** Measured on torch 2.11 against this repo's own
``fft2c``/``ifft2c`` round-trip: compilation *succeeds*, under
``fullgraph=True``, and is numerically correct (max abs error 3.2e-07).
Torchinductor emits one line —

.. code-block:: text

   UserWarning: Torchinductor does not support code generation for complex
   operators. Performance may be worse than eager.

— and runs every complex operator eagerly.

That is the problem. The arm declares a compiled run, the audit accepts it,
provenance records it, and the complex regions execute eagerly, *possibly slower
than not compiling at all* because the graph-break machinery is not free. A
throughput claim that is untrue and undetectable downstream is the same defect
class as ``compile.deepcompile: true`` without ``engine.compile()``
(see :doc:`distributed_training`).

.. rubric:: What counts as "complex" — and what does not

Four unioned signals, none of them the layer you would guess: the
model's declared ``capabilities.accepts_complex`` and
``input``/``output_domain``, ``model.target_domain``, and
``physics.kspace.enable_kspace_recon``. The middle two are shared with
``check_deepspeed_precision_coherent`` so the two checks cannot disagree about
what "complex" means.

``ComplexConv2d`` is deliberately **not** a signal.
:class:`~spectramr.models.layers.complex_conv.ComplexConv2d` stores real and
imaginary parts as separate *real* tensors and performs a single fused real
``F.conv2d`` against a block weight matrix, returning ``float32`` — it compiles
cleanly under ``fullgraph=True``. "Uses complex arithmetic" and "carries a
complex dtype" are different properties, and only the second is what Inductor
cannot codegen. Keying on the layer would have blocked compilation on arms that
compile perfectly well.

Note ``capabilities.accepts_complex`` is the *declarative* signal and would be
the whole answer if it were populated; it is ``None`` on 484 of 589 registered
models, which is why four signals are needed instead of one lookup.

.. rubric:: Getting throughput on a complex arm instead

``bfloat16`` AMP (which *does* work under complex autocast),
``optimization.optimizer.fused``, and ZeRO sharding. See
:doc:`training_throughput`.

For OOM relief prefer ``bfloat16``: it does **not** need loss scaling and does
**not** aggravate the gradient-explosion → NaN instability (the hyper-mamba
``exp_hm_06``/``exp_hm_10`` failure mode), whereas fp16 can collapse the
``GradScaler`` scale and silently skip steps. The ``hilbert_mamba`` cohort sets
``use_amp: true`` + ``amp_dtype: bfloat16``.

**Mamba arms and fp16.** ``amp_dtype: float16`` *is* viable for the
``hilbert_mamba`` arms — they are ``domain=image`` (real fp16 autocast, the
``GradScaler`` is wired in ``steppers.py``), and
:class:`spectramr.models.blocks.mamba_block.MambaBlock` force-runs the SSM/GRU
recurrence in fp32 (``autocast(enabled=False)`` + ``.float()``) regardless of
``amp_dtype``, so the precision-fragile long recurrence (L = H·W) and the
cuDNN-GRU-under-fp16 crash are both avoided. bf16 is still preferred (no
scale-collapse failure mode; identical Tensor-Core throughput on Ada), and
because the recurrence is fp32-pinned either way, fp16's extra mantissa bits
buy nothing on the part that matters.

**fp8 is not supported.** It is not an autocast dtype: PyTorch fp8 *training*
requires ``torchao.float8`` or NVIDIA Transformer-Engine, explicit per-layer
conversion of ``nn.Linear`` (and per-tensor scaling), and Hopper/Ada hardware.
The Mamba selective-scan and ``Conv2d`` stems here are not standard fp8 targets,
so ``amp_dtype`` deliberately rejects ``float8`` rather than advertise an
unwired knob. Adding fp8 would be a separate feature (a ``Float8Linear`` swap
pass behind a new ``parallel``/``optimization`` flag), not a precision toggle.


Distributed training (single-node multi-GPU & multi-node)
==========================================================

DDP is config-driven via ``config.parallel`` (``ParallelismConfigSchema``) and
launched with ``torchrun``; the ``train-distributed`` CLI verb forces
``parallel.strategy='ddp'`` and ``num_devices=WORLD_SIZE`` when a torchrun launch
is detected (:func:`spectramr.pipelines.distributed.run_distributed_training`).
Data is sharded with a ``DistributedSampler``; very large models can shard
parameters with FSDP (``parallel.fsdp.enabled: true``, ``mixed_precision: bf16``).

**Single node, N GPUs:**

.. code-block:: bash

   torchrun --nproc_per_node=4 -m spectramr.cli train-distributed \
       --config experiments/inprogress/hilbert_mamba/exp_hm_05_mm_mamba.yaml

**Multiple nodes** (rendezvous on the first node):

.. code-block:: bash

   torchrun --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=4 \
       --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
       -m spectramr.cli train-distributed --config <yaml>

On SLURM use the committed launcher
``scripts/training/train_distributed.sbatch`` (matches the cluster's
``--account=<your-slurm-account>`` / ``--gres=gpu:ada:N`` conventions; one ``torchrun`` per
node via ``srun``, c10d rendezvous keyed on the job id):

.. code-block:: bash

   # single node × 4 GPUs
   sbatch --nodes=1 --gres=gpu:ada:4 \
          --export=ALL,CONFIG=<yaml> scripts/training/train_distributed.sbatch
   # 2 nodes × 4 GPUs (world_size 8)
   sbatch --nodes=2 --gres=gpu:ada:4 \
          --export=ALL,CONFIG=<yaml> scripts/training/train_distributed.sbatch

Use ``DistributedDataParallel`` (the path above), never ``nn.DataParallel``.
Combine with ``amp_dtype: bfloat16`` for the largest effective batch per GPU.


Validation Errors
=================

``Validation produced zero successful batches``
------------------------------------------------

Every validation batch raised the same exception (a shape / channel mismatch in
the strategy's validation forward, or an OOM), so the run is failed loud rather
than shipped image-less and green (F36, CLAUDE.md #10). The **root-cause
traceback of the first failing batch is embedded in the raised error itself**
(``--- first validation-batch failure (root cause) ---`` block) — earlier it
only said "logged above", and that ``logger.warning`` line was routed to a
handler the per-arm log didn't persist, so cluster triage saw the symptom with
no cause (2026-06-21: the ``exp_vf_ib_infonce_v2`` / ``b17_dice_risk_calibration``
validation crashes). Read that embedded block: it names the exact tensor op and
shapes. Common causes: the model emits 2-channel real-stacked complex while the
validation metric compares against a 1-channel magnitude target; a strategy
``_validation_forward`` that returns ``None``; or a ``val_batch`` whose keys
``_unpack_batch`` doesn't recognise. Pinned by
``tests/unit/pipelines/test_validation_fail_loud.py``.

``RuntimeError: quantile() input tensor is too large``
-------------------------------------------------------

**Root cause**: ``torch.quantile`` sorts the reduced dimension and refuses any
reduced length above ``2**24`` (~16.7M) elements. The digital-twin marker
embedder (``infrastructure/physics/digital_twin_simulator.py``) derives the
tissue-intensity scale from a 0.75 quantile of the anatomy magnitude; on the
**single-coil** path it flattens the *whole* tensor, so a larger **validation**
batch tips over the cap and every validation batch raises → *"Validation
produced zero successful batches"* (the ``exp_vf_ib_infonce_v2`` crash,
2026-06-22 forensics).

**Fix**: the quantile sites now go through ``_robust_quantile`` (same module),
which decimates the reduced dimension with an even stride down to ~16.7M samples
before the quantile when it would overflow — deterministic (no ``randperm``, so
seeding/determinism is preserved) and statistically unbiased for the smooth
0.75/0.99 quantiles the embedder uses. No config change needed. Pinned by
``tests/unit/infrastructure/physics/test_digital_twin_simulator.py::TestRobustQuantile``
(including a real ``>2**24`` tensor where bare ``torch.quantile`` raises).

**Also in that helper (not an error, but it is what runs)**: below the cap
``robust_quantile`` takes an exact selection fast path — two ``torch.kthvalue``
calls interpolated, instead of the full sort ``torch.quantile`` performs to read
at most two order statistics. It is **bit-identical**, not approximate, and
declines to ``torch.quantile`` for any input it cannot reproduce exactly (NaN,
±inf, a negative zero, non-CPU tensors, dtypes outside float32/float64, fewer
than 4096 elements, or a ``q`` outside ``[0, 1]``) -- **and above the cap this
page is about**: ``kthvalue`` carries no size limit of its own, so past
``2**24`` it would answer where ``torch.quantile`` raises the error above.
Declining there is what keeps this error reachable through ``max_elems``, rather
than silently replaced by a number no reference value was ever compared against.
So a number never changes because of it, and profiles taken before 2026-08 will
show a ``sort`` where current ones show ``kthvalue``.

The exactness rests on evaluating the rank ``q * (n - 1)`` in the *tensor's*
dtype rather than in Python ``float``. Issue #1537 measured a divergence, read
it as ``kthvalue`` being inexact on large tensors, and concluded the route was
unusable above ~2**18 elements; that attribution is wrong — a full ``sort``
under the same float64 rank arithmetic diverges identically, and with the rank
in the tensor dtype the route is exact at 262,144 and 1,048,576 both.


Checkpoint Errors
==================

``KeyError: 'state_dict'`` when loading checkpoint
----------------------------------------------------

The checkpoint uses the old ``pth`` format from before v5.0. Convert:

.. code-block:: python

   import torch
   old = torch.load("old_checkpoint.pth")
   # Old format may have different key
   weights = old.get("model_state_dict", old.get("generator", old))
   torch.save({"state_dict": weights}, "converted.pt")

``Checkpoint epoch/step counter mismatch``
------------------------------------------

If resuming after changing ``max_iterations``, the step counter may
be ahead of the new value. Fix:

.. code-block:: yaml

   checkpoint:
     resume_training: false    # Don't restore step counter
     pretrained_path: checkpoints/last.safetensors
     load_optimizer_state: true


Hyper-Mamba Cohort (``hilbert_mamba``) — 2026-06-24 crash triage
================================================================

The ten ``experiments/inprogress/hilbert_mamba/exp_hm_*`` arms crashed in four
distinct families. Two are fixed; two are cluster-data / cluster-runtime
dependent and are documented here with their reproduction state.

``RuntimeError: weight of size [1, 64, 1, 1], expected input … to have 64 channels, but got 1`` (FIXED)
------------------------------------------------------------------------------------------------------

**Symptom**: a Hilbert-Mamba arm with ``optimization.use_gradient_checkpointing:
true`` (``exp_hm_01_ct``/``03_crm``/``04_hw``/``05_mm``) crashes at iteration 1
inside ``self.stem`` — but the reported weight is the model's *last* conv (the
``Conv2d(d_model, out_channels, 1)`` head), not the stem's.

**Root cause**: ``GradientCheckpointing.apply_checkpointing``
(``models/profiling/advanced_profiling.py``) patched every ``Conv2d``/``Linear``
with ``module.forward = lambda x: checkpointed_forward(module, x)`` *inside the
loop*. The lambda captured the loop variables ``module`` and
``checkpointed_forward`` by reference (Python closure late-binding), so after the
loop **every** patched layer's forward resolved to the **last** module's — the
stem ended up running the head's conv on its 1-channel input.

**Fix**: bind each layer's forward once, by value, via a module-scope factory
``_make_checkpointed_forward(orig_forward)``, and use ``use_reentrant=False`` so
gradients still flow when a layer's input does not itself require grad (the old
reentrant default silently produced ``None`` grads for the stem). Regression:
``tests/unit/models/profiling/test_advanced_profiling.py``. This bug affected
*any* model routed through the generic checkpointing path, not just Hyper-Mamba.

``CUDA out of memory. Tried to allocate 19.50 GiB`` in validation (FIXED)
-------------------------------------------------------------------------

**Symptom**: ``exp_hm_02_fe`` trains at ``batch_size: 1`` but OOMs the moment
validation starts.

**Root cause**: a full-resolution 256×256 patch linearizes to a 65,536-token
sequence; the SSM selective-scan tensor for a single forward is ~9.75 GiB, so
``validation.loader.batch_size: 2`` tries to allocate ~19.5 GiB (= 2×) on top of the
model. Validation, not training, is the trigger because training already fit at
batch 1.

**Fix**: ``validation.loader.batch_size: 1`` across all ten arms is the guaranteed lever (a
single forward fits, as the training step proved); the val OOM was a latent
failure for every full-resolution arm once the checkpointing crash above is
cleared. ``optimization.use_amp: true`` + ``amp_dtype: bfloat16`` is also
enabled, but note its OOM relief here is **partial**: ``MambaBlock`` force-runs
the SSM/GRU recurrence in fp32 (``autocast(enabled=False)`` + ``.float()``), so
AMP only shrinks the conv / projection / stem-head / sequence-embedding
activations, not the recurrence — it is not a clean 2× on these models. For
further scaling, reduce ``patch_size`` or run multi-GPU DDP (see the
distributed-training section below).

``num_samples should be a positive integer value, but got num_samples=0`` (cluster-data dependent)
---------------------------------------------------------------------------------------------------

**Symptom**: ``exp_hm_07_25d`` and ``exp_hm_08_sdtw`` — the only two arms with a
3-D ``patch_size`` (depth 16) and 3-D/2.5-D models — die at dataloader build with
an empty training split.

**Status**: NOT reproducible locally. ``data.data_root``
(``databases/ulf_paired/preprocessed``) is cluster-only, and the local proxy
manifest (32 paired records, depths 27/150/156) yields ≥1 depth-16 slab per
volume — so the empty split is specific to the cluster manifest. **Diagnose on
the cluster**: dump the train/val record counts the 3-D slab path
(:class:`~spectramr.data.datasets.slice_dataset.SliceVolumeDataset`) produces for
these two arms; the suspect is the interaction of ``allow_unpaired: true`` with
the slab windowing leaving the train split empty.

Model emits all-NaN → ``Metric 'lpips' … values in range [nan, nan]`` (cluster-runtime dependent)
--------------------------------------------------------------------------------------------------

**Symptom**: ``exp_hm_06_mdi`` (NaN by iter ~10, preceded by
``GRADIENT EXPLOSION DETECTED total_norm=2676``) and ``exp_hm_10_inr`` (NaN by
iter ~40). LPIPS correctly hard-raises on the non-finite prediction (CLAUDE.md
pitfall #9) — the metric is the messenger, not the cause.

**Status**: training instability, not a forward bug — at initialisation both
models produce finite output. Must be reproduced on a box with the real
``mamba_ssm`` kernel (see the next section — the run now *fails loud* without
it). Gradient clipping is already enabled (``gradient_clip_value: 1.0``) yet NaN
still appears, which points at an exploding *forward* activation rather than the
optimiser step. **Recommended cluster-side iteration**: lower ``learning_rate``
(e.g. 5e-5), add LR warmup, and/or bound the output.


``MambaBlock requires the official mamba_ssm selective-scan kernel`` (by design)
=================================================================================

Mamba/SSM models (``hilbert_mamba``, ``geomamba``, ``d2_mamba``, ``bloch_mamba``,
…) **require** the official ``mamba_ssm`` CUDA kernel.
:class:`spectramr.models.blocks.mamba_block.MambaBlock` now **raises** at
construction when it is missing or its kernel failed to build, rather than
silently substituting a Gated-Conv+GRU block — that fallback is **not an SSM**,
so a silent substitution would train a GRU under the "Mamba" label and make
every result scientifically mislabelled (pitfall #9 / #16).

This is also caught **at audit time** (before any GPU work) by the
``mamba_models_require_mamba_ssm`` health check: a ``model_type`` containing
``mamba`` with no importable kernel fails ``spectramr audit`` (``error``), or warns
if ``SPECTRAMR_ALLOW_MAMBA_FALLBACK`` is set (the run would be a non-SSM GRU).

**Fix** — install the kernel (needs CUDA + ``nvcc``):

.. code-block:: bash

   pip install -e '.[mamba]' --no-build-isolation   # mamba-ssm + causal-conv1d
   python -c "import mamba_ssm"                       # verify the kernel imports

The error message distinguishes *not installed* from *installed-but-kernel-broken*
(CUDA/PyTorch version mismatch — rebuild with ``--no-build-isolation``).

**Opt-in GRU fallback (CPU/CI wiring only).** ``SPECTRAMR_ALLOW_MAMBA_FALLBACK=1``
re-enables the GRU approximation with a loud warning, for shape/wiring tests on
boxes without the kernel. The pytest ``conftest`` sets it for the test session
(set ``=0`` to exercise the raise path). **Never** set it for a real Mamba
experiment — the GRU is not an SSM and the numbers are not "Mamba".

**Which blocks run the official kernel.** Everything that builds its sequence
mixer from :class:`MambaBlock` runs the real ``mamba_ssm`` selective scan:
``hilbert_mamba`` (``_MambaEncoder``), ``mamba_unet`` (``MambaLayer2D``),
``geo_mamba_unet`` / ``FiLMMambaBlock``, ``d2_mamba``, ``hdsf``, ``mamba_4d``,
``se3_lie_algebra_mamba``, and (since 2026-06-24) ``swin_mamba_kan``
(``MambaLayer`` was a slow pure-Python ``for t in range(L)`` reimplementation;
it now delegates to ``MambaBlock``). A handful of models keep a **bespoke**
recurrence on purpose — that custom SSM *is* their contribution and must NOT be
swapped for vanilla ``mamba_ssm``: ``bloch_mamba`` / ``bloch_mamba_v2`` (Bloch
T1/T2 physics in the A-matrix), ``diff_mamba`` (Neural-ODE), ``neuro_mamba``
(spiking LIF), ``continuous_sfc_mamba`` (physical arc-length Δt + its own
triton/python scan backend), ``ttt_mamba`` (test-time training), and
``hyper_mamba_bridge`` (a hypernetwork that *generates* SSM parameters).

``continuous_sfc_mamba`` additionally has an **opt-in** ``mamba_ssm`` scan
backend (``kernel_backend='mamba_ssm'``) that routes its diagonal recurrence
through ``mamba_ssm.selective_scan_fn``. It is **not bit-identical** to the
python/triton backends: the kernel forms the decay as ``exp(Δ·mean_c A)``
(mean-before-exp) while the reference backends use ``mean_c exp(Δ·A)``
(mean-after-exp), which the kernel cannot express. It is therefore opt-in only
(``auto`` never selects it, for reproducibility), CUDA-only, and intended for
training from scratch on the fast kernel — not for swapping a checkpoint trained
under a reference backend. See ``models/blocks/triton_scan.py``.

Environment / Dependency Errors
===============================

Verify the declared dependency set with one command
----------------------------------------------------

``scripts/verify/verify_dependencies.py`` checks that every dependency the
project declares in ``pyproject.toml`` (the SSOT — ``[project].dependencies``
plus each ``[project.optional-dependencies]`` group) is installed and
version-correct, and — with ``--import-check`` — actually importable. It imports
only the standard library, so it runs even in a partially-broken environment.

.. code-block:: bash

   python scripts/verify/verify_dependencies.py                 # core deps only
   python scripts/verify/verify_dependencies.py --extras mri,viz # + named groups
   python scripts/verify/verify_dependencies.py --all            # every group
   python scripts/verify/verify_dependencies.py --import-check    # also import each
   python scripts/verify/verify_dependencies.py --json            # machine-readable

Exit code ``0`` means all selected dependencies are satisfied; ``1`` flags a
missing, version-mismatched, or (with ``--import-check``) unimportable
dependency; ``2`` is a usage/environment error. This makes it usable as a
pre-flight gate in a shell script or CI step.

Installed-but-unimportable (the ``torchmetrics`` case)
-------------------------------------------------------

A metadata check alone is **not** sufficient: a distribution can be installed at
a version that satisfies its specifier yet still fail to ``import`` because of a
transitive-dependency conflict. The live example is ``torchmetrics``:
``torchmetrics>=1.0,<2.0`` is satisfied by e.g. ``1.9.0``, but the import raises
when the environment has ``huggingface-hub>=1.0`` (torchmetrics needs ``<1.0``).
Every torchmetrics-backed metric (``ms_ssim``, ``lpips``, ``fid``, ``kid``,
``uqi``) then raises at runtime rather than fabricating a ``0.0`` (the
2026-07-01 M1 fix), so any arm listing one
of them in ``validation.metrics`` crashes at the first validation step.

``--import-check`` is what surfaces this — plain metadata reports ``OK`` while
the import probe reports ``IMP`` with the exact root cause:

.. code-block:: text

   IMP   torchmetrics   1.9.0   <2.0,>=1.0   import torchmetrics: ImportError: huggingface-hub>=0.34.0,<1.0 is required …

**Fix** (env-level; ``pyproject`` deliberately does not pin ``huggingface-hub`` to
avoid cascading a ``transformers`` downgrade):

.. code-block:: bash

   pip install 'huggingface-hub<1.0'
   python scripts/verify/verify_dependencies.py --import-check   # re-verify

Data Loading Errors
====================

``BART dim/payload mismatch for <v>.cfl: header dims … imply N … but the .cfl holds M``
---------------------------------------------------------------------------------------

The BART ``.cfl`` payload is **truncated** — its byte size is smaller than its
``.hdr`` dimension vector implies (a complete payload is ``prod(dims)`` ×
``complex64`` = ``× 8`` bytes). Root cause: a dropped connection during download
ended the stream early and the partial file was renamed as "complete" — the
``size > 0`` completeness test never noticed (``multiecho_radial_b0_r2star``
``v05.cfl``: 790 MB of a ~9 GB payload, which took out the 7 B0-field VF arms
vf_21/25/26/29_real/30 at the first batch).

``download_external_datasets.py`` now (a) **never promotes a short stream** —
it verifies bytes-downloaded == ``Content-Length`` before the atomic rename and
otherwise leaves the ``.part`` for the next Range-resume; (b) treats a
``.cfl`` whose size ≠ ``prod(hdr dims) × 8`` as **not present** (``_present`` /
``--verify``) so the dataset re-fetches; and (c) **unlinks** a previously-truncated
``.cfl`` before re-streaming it. Pinned by
``tests/unit/scripts/test_download_external_datasets.py``.

**Repair**: re-fetch the truncated payload from wherever you obtained it. A
partially-written ``.cfl`` is indistinguishable from a complete one by anything
except its length, so any integrity check you build has to compare the byte count
against the ``.hdr`` dimensions rather than trust a ``status: downloaded`` marker.

.. note::

   The mirroring and manifest tooling referenced throughout this section fetches
   from the maintainers' cluster mirror into their manifest layout, and is not
   part of this distribution. The diagnosis below is portable; the fetch commands
   were not, so they are described rather than quoted.

No manifest regeneration is needed for this fix: the manifest already lists
``v05`` (its ``shape`` comes from the intact ``.hdr``; only the ``.cfl`` payload
was short), and the reader reads the now-intact ``.cfl`` at load time. Regenerate
only if the file *list* changed.

The ``.cfl`` files are streamed **directly** into ``<id>/raw`` (no archive), so
``extract_external_datasets.py`` is not in this path — it only expands archived
bundles and already fails loud on a truncated ``.zip`` (``BadZipFile``).

**"A byte-level check says OK but the loader still raises"** — these disagree only
when they read **different files**. ``inspect_bart.py`` historically globbed a
hard-coded ``--root`` directory, but the loader resolves each ``.cfl`` from the
**index manifest** (``data.index_path`` → ``data_root / relative_path``). A
re-fetched ``raw/v05.cfl`` can pass the directory glob while the manifest still
points the loader at a stale copy (or a missing/renamed file). Use the
paths the ``BartDataset`` index actually reads, rather than globbing the
directory. Per record the useful verdicts are ``OK`` / ``MISMATCH`` (the loader
will raise ``ValueError``) / ``MISSING-CFL`` (``FileNotFoundError``) /
``TRAILING-BYTES``.

The verdict mirrors ``io_strategies.BartCflStrategy`` exactly: ``np.fromfile``
*floors* trailing bytes, so the loader raises iff ``(st_size // 8) !=
prod(dims)`` (the ``account()`` ``loader_will_raise`` key), distinct from the
stricter byte-perfect ``ok``. Path resolution also mirrors the loader's
**basename normalisation** (``BartCflStrategy`` strips a trailing ``.cfl`` then
reads ``<base>.cfl``): a manifest record may carry either the BART bare basename
``relative_path: "v05"`` or the explicit ``"v05.cfl"`` — both resolve to the same
``<data_root>/v05.cfl`` the loader reads, so a bare-basename manifest is **not**
falsely flagged ``MISSING-CFL`` (the 2026-06-22 cluster false-positive: the real
``v05.cfl`` was present and the loader loaded fine, but the inspector had
literal-matched ``<data_root>/v05``). If manifest mode reports ``MISMATCH`` /
``MISSING-CFL`` while a bare glob is clean, the manifest genuinely points at a
**stale/truncated/missing** payload — re-fetch it or regenerate the manifest
(above). Pinned by ``tests/unit/data/test_inspect_bart.py``.

On a non-clean verdict the dataset id is recoverable from the manifest's own
``data_root`` (``.../external/<id>/raw``) or an explicit ``dataset_name`` key,
which is what you need to scope a re-fetch to the one broken dataset.

This is safe to re-run because the downloader is **integrity-aware**: its
``_bart_truncated`` / ``_present`` treat a short ``.cfl`` as *not present*, so a
truncated ``v05.cfl`` falls through to a real re-pull instead of being skipped as
already-downloaded — the same payload check the train-time loader enforces.
Re-generating the manifest (``gen_external_dataset_manifests.py``) is the
complementary fix: its ``_bart_payload_intact`` **quarantines** a truncated
``.cfl`` from ``records[]`` so consuming arms drop the bad file instead of
crashing. Re-fetch restores the file; regenerate drops it — pick by whether the
record is recoverable.

``pairing_policy='ulf_source' produced 0 pairs … fields present = [5.0, 7.0]``
-----------------------------------------------------------------------------

**Symptom**: an mrixfields ``ulf_source`` arm crashes at data-loader build with
*"produced 0 pairs: no group matches the pinned field 0.1 T; fields present =
[5.0, 7.0]"* — even though ``mrixfields2026_train.json`` contains all five field
strengths (9 complete groups, each 0.1/1.5/3/5/7 T).

**Cause**: the manifest records are **field-sorted** (all 0.1 T first … 7 T
last). The upstream train/val split is a flat contiguous record slice, so a
90/10 cut put *every* 0.1 T source in train and left validation with only the
top fields. ``ulf_source`` pins the 0.1 T source, so the val dataset matched
nothing and fail-fasted. It is **not** a missing-data problem — the misleading
"build/point at the full corpus" hint in older builds pointed the wrong way.

**Fix** (``DatasetInstantiator._create_mrixfields``): re-split **group-aware**
(on whole ``pairing_group`` groups) for *every* field-pinned policy —
``multi_source``, ``ulf_source``, ``prior``, ``fixed_target`` — so each split
keeps complete field groups and the pinned field is present in both. See
``tests/unit/data/builders/test_dataset_instantiator.py``.

``No losses were built by LossBuilder. Training cannot proceed`` (ablation arms)
-------------------------------------------------------------------------------

**Symptom**: an ablation arm (e.g. ``mrixfields_b*_ablate_*``) crashes at build
with *"No losses were built by LossBuilder"*.

**Cause**: these strategies compute their objective **directly** (e.g.
``ScatteringBesovStrategy``); the declarative ``losses.image_losses`` list is only
a **LossBuilder gate placeholder** (``- {name: l1, weight: 1.0}``) so the builder
sees ≥1 enabled loss. The ablation YAML dropped that placeholder while keeping
only ``output_domain: image`` + a comment, so the builder counted zero enabled
losses and refused to proceed (a correct fail-loud, but the arm was simply
under-specified).

**Fix**: restore the parent arm's placeholder ``image_losses`` list in the
ablation (the strategy still computes the real, ablated objective — the
placeholder only satisfies the build gate). Mirror the ``metadata.baseline``
parent's ``losses`` block exactly.

``Paired-NIfTI VAE trains HF→ULF (degradation) instead of autoencoding HF``
---------------------------------------------------------------------------

**Symptom**: a ``dataset_type: nifti_paired`` stage-1 VAE (the two-stage LDM
``stage1_vae_*`` arms) shows a **sharp HF** ``input`` and a **noisy ULF**
``target`` with the SAME shape but different statistics in its first-steps
snapshot — the model is minimizing ``||Dec(Enc(HF)) − ULF||``, a degradation
network, and its frozen decoder later emits low-field appearance that corrupts
stage 2.

**Root cause** (2026-07, pitfall #9): the mode name is ``<input>_to_<target>``,
so ``hf_to_ulf`` is a genuine HF→ULF *translation* (input HF, target ULF), NOT an
autoencoder. Earlier it *looked* like an autoencoder only because a depth-0 patch
bug made the ULF target shape-mismatch, tripping a silent ``vae.py`` fallback that
substituted ``target = input``. When ``slice_2d`` + ``sampler.type: full`` fixed
the patch bug, input and target became the same shape, the fallback stopped
firing, and the arm silently trained the wrong direction.

**Fix**: two new single-field autoencode modes, ``hf_to_hf`` and ``ulf_to_ulf``,
which DROP the opposite arm (``target_path`` → ``None`` so the self-supervised
branch aliases ``target = input``) — ``input ≡ target`` by construction. The
silent ``vae.py`` shape-mismatch fallback is REMOVED; a missing/mismatched target
now **raises**. A ``ConfigHealthChecker`` rule
(``check_vae_pretrain_autoencodes_single_field``) rejects a ``vae_pretrain`` arm
on paired data that declares a translation direction. Set the stage-1 arms to
``data.bidirectional_mode: hf_to_hf``. ``hf_to_ulf`` remains valid as a real
bidirectional-translation direction. Pinned by
``tests/unit/data/builders/test_dataset_instantiator.py`` (``_autoencode_field`` +
the ``_create_nifti_universal`` hf_to_hf / ulf_to_ulf / missing-target cases),
``tests/unit/data/test_hf_to_hf_autoencode.py`` (end-to-end input≡target), and
``tests/unit/infrastructure/training/strategies/test_vae.py`` (the raise).


``Manifest not found: data/manifests/train.pkl``
-------------------------------------------------

Manifests are machine-local. Either:

1. Regenerate: ``python scripts/data/regenerate_cluster_manifests.py --data-base databases``
2. Update path: use absolute paths or cluster-relative paths

``Empty DataLoader: 0 samples after split``
--------------------------------------------

The train/val split produced 0 validation samples. Fix by:

.. code-block:: yaml

   validation:
     split: 0.1     # Use 10% for validation

Or provide a separate validation manifest:

.. code-block:: yaml

   data:
     val_manifest: data/manifests/val.pkl

``FileNotFoundError: .h5 not found``
--------------------------------------

Cluster path layout differs from local. Use path aliases:

.. code-block:: yaml

   data:
     data_root: /project/<allocation>/<user>/spectramr/databases/


Training Not Improving
========================

``val_psnr stuck at ~25 dB after 10k steps``
---------------------------------------------

Common causes and fixes:

1. **Learning rate too low** — try ``1e-3`` with cosine warmup
2. **Data consistency disabled** — enable ``physics.data_consistency.enabled``
3. **Loss domain mismatch** — check ``output_domain`` matches loss lists
4. **EMA decay too high early** — use ``warmup_steps: 2000``

``val_lpips not decreasing``
-----------------------------

LPIPS requires perceptual loss during training:

.. code-block:: yaml

   losses:
     image_losses:
       - name: perceptual
         weight: 10.0
         enabled: true
       - name: lpips
         weight: 1.0
         enabled: true

``GAN mode: discriminator loss = 0 immediately``
-------------------------------------------------

Discriminator collapses to always-real prediction. Fix:

.. code-block:: yaml

   optimization:
     discriminator_lr: 4e-4      # Keep D LR > G LR
   training:
     gan:
       n_critic: 1               # Update G and D equally
       label_smoothing: 0.1      # Add label smoothing


Physics / K-Space Errors
=========================

``AssertionError: Expected complex tensor``
-------------------------------------------

The physics operators require complex tensors. Use the framework's
FFT wrapper (not raw ``torch.fft``):

.. code-block:: python

   # ❌
   kspace = torch.fft.fft2(image)

   # ✅
   from spectramr.infrastructure.physics.fft_ops import fft2c
   kspace = fft2c(image)    # handles centering and normalization

``Hermitian symmetry violation``
---------------------------------

Real images must have conjugate-symmetric k-space. Violations typically
come from applying non-symmetric augmentations in k-space. Use:

.. code-block:: python

   kspace = kspace + torch.conj(torch.flip(kspace, dims=[-2, -1])) * 0.5

Validation REAL image is a centre-bright blob; FAKE is black; ``val_psnr`` NaN
------------------------------------------------------------------------------

Symptom (smoke audit 2026-06-13, VF cohort exp_p3 / hyper_mamba_meta /
method_c): the saved ``metrics/real_images`` panel renders as a centre-bright
**k-space** blob instead of a brain, the ``fake_images`` panel is black, and
``val_psnr`` is ``NaN``.

Root cause — a **domain-contract drift**. These arms declare
``data.dataset_type: kspace`` (so the motion / kinematic operator can corrupt
the data in k-space, which is physically correct), but the strategy's
reconstruction loss, ``val_psnr`` and cached ``_last_visual_*`` are all defined
on *images*. If the strategy treats the k-space target as an image and never
IFFTs it, the REAL reference becomes ``|k-space|`` and the image-domain
corruption is applied to k-space → garbage / black FAKE.

Every image-domain strategy must route the dataloader target through the SSOT
seam ``BaseTrainingStrategy._ensure_image_domain_target`` before using it:

.. code-block:: python

   target_complex = self._to_complex(target_batch)
   target_complex = self._ensure_image_domain_target(target_complex)  # k-space -> image (once)

The seam's domain decision is delegated to
:func:`spectramr.infrastructure.training.utils.domain_inference.needs_ifft_for_visualization`,
**not** a raw ``dataset_type == "kspace"`` check — because
``coil_processing_mode: rss_image`` / ``magnitude`` already IFFT inside the
dataset's TorchIO pipeline, so those arms read ``dataset_type: kspace`` yet
deliver an *image*. A naive guard would re-FFT that image into k-space (the
mirror-image regression). The seam is therefore a no-op for ``rss_image`` arms
(exp_p2, eval_c2/c3/c7, exp_c4) and only IFFTs the genuinely k-space-delivering
``svd`` arms.

Config half of the fix: an ``svd`` arm must keep the complex pair so the seam
has an imaginary half to invert — set ``data.target_channels: 2`` (not ``1``,
which strips phase and leaves a single real channel the IFFT cannot use).

``ConcreteVirtualFiducialStrategy`` and ``IBVFStrategy`` carry an equivalent
inline guard; the SE3-navigator, motion-meta, distillation and Bloch-manifold
strategies now route through the shared base seam.


Strategy-Specific Issues
=========================

TTO strategy: ``motion_traj not updating``
-------------------------------------------

``ConcreteTTOStrategy`` is an inference-time strategy, not a training strategy.
Invoke with ``--resume`` on a pre-trained checkpoint:

.. code-block:: bash

   python -m spectramr.cli predict \
       --model checkpoints/hypermamba_best.safetensors \
       --config experiments/training/exp_tto.yaml

Diffusion: ``prediction quality degrades after 500 steps``
------------------------------------------------------------

Cold diffusion with ``prediction_type: sample`` can overfit at high T.
Add importance sampling:

.. code-block:: yaml

   training:
     diffusion:
       importance_sampling: true
       importance_sampling_gamma: 1.0

Diffusion denoiser ignores the measurement (measurement-independent output)
---------------------------------------------------------------------------

Standard image-domain diffusion noises the **target** and trains the denoiser
to invert that noise — the low-res / ULF **input is never fed to the model**, so
the network must hallucinate a specific subject from pure noise. The result is a
measurement-independent solution (pitfall #20): the reconstruction does not
depend on the actual acquisition, and PSNR/SSIM against a fixed validation set
can look plausible while the model has learned a prior, not a reconstruction.

This bit ``exp_hm_09_hld_mamba`` (``in_channels: 1``, no conditioning path). To
condition the denoiser on the measurement, set ``condition_on_input`` and give
the model an extra input channel — the strategy concatenates the (resized) input
onto the noised target along the channel axis:

.. code-block:: yaml

   model:
     in_channels: 2          # [noisy target || conditioning input]
   training:
     diffusion:
       condition_on_input: true

The flag defaults to ``false`` (historical unconditional behaviour) and is a
no-op for cold/latent diffusion and when smaps were already concatenated. It is
declared on ``DiffusionTrainingConfigSchema`` (``training.diffusion``) and read
by ``DiffusionTrainingStrategy._maybe_condition_on_input``.

``N2N mode: train loss not decreasing``
-----------------------------------------

N2N requires at least 2 repetitions in the dataset. Verify:

.. code-block:: yaml

   data:
     dataset_type: m4raw_multi_rep
     use_repetitions: true
     num_repetitions: 3


Diagnostics round (2026-06-25): four genuine crash fixes
========================================================

Four arms in the ``tests_experiments`` diagnostics sweep crashed on genuine,
locally-reproducible code paths (the rest were stale, data-absent, or
GPU-repro-bound). Each fix lands with a regression test.

``field_strength`` missing in validation (calibration / field-renderer arms)
----------------------------------------------------------------------------

Symptom (``mrixfields_b17_dice_risk_calibration``): trains fine, then every
validation batch raises ``AnatomyFieldRenderer.forward() missing 1 required
keyword-only argument: 'field_strength'`` → *zero successful validation batches*
(CLAUDE.md #10).

Root cause — the field-strength injection had been added to
``ReconstructionMixin._prepare_generator_inputs_reconstruction``, but that method
**has no callers**. The live forward seam used by both training and validation is
``ReconstructionTrainingStrategy._prepare_generator_inputs``; ``_validation_forward``
calls it, and it did not inject ``field_strength``. Training survived because the
*loss* path injects it; validation did not. Fix: inject ``field_strength`` /
``contrast_id`` (signature-gated via ``_callable_accepts_kwarg``, target field
preferred, never defaulted) into the **live** method. ``xfield_fm_strategy`` is
unaffected — its override re-sets the field after ``super()``.

``[DomainMismatch] Model outputs N channels, but target provided 1``
--------------------------------------------------------------------

Symptom (``mrixfields_b29_heteroscedastic_ulf``): crash at iter 1 — the model
emits 2 channels (``[mean, logvar]``) for a 1-channel target and the base
``train_step`` width guard rejects it.

Root cause — distributional / parametric heads emit more channels than the target
**by design** and self-compute the likelihood, but the guard only excepted the
hard-coded ``model_type == "evidential_unet"``. Fix: an Open-Closed
``predicts_distribution_params`` class flag on ``BaseTrainingStrategy`` (set
``True`` on ``HeteroscedasticULFStrategy``); the guard defers when it is set. Also
covers ``b29_ablate_var_prior`` (same strategy + ``out_channels: 2``).

``quantile() input tensor is too large`` (digital twin, multi-coil)
-------------------------------------------------------------------

Symptom (``exp_vf_ib_infonce_v2``): ``RuntimeError: quantile() input tensor is too
large`` in ``CornerFiducialEmbedder.forward`` — ``torch.quantile`` refuses a
reduced dim above ``2**24``.

Root cause — the single-coil and per-channel branches already used the
``_robust_quantile`` guard (strided decimation below the cap), but the
**multi-coil RSS branch** still called raw ``torch.quantile(rss.float(), 0.75)``.
A 3-D/5-D coil volume tips it over the cap. Fix: route that branch through
``_robust_quantile`` too.

UNet output 4 px short of target (non-divisible input)
------------------------------------------------------

Symptom (``mrixfields_b25_cartoon_texture``): ``The size of tensor a (432) must
match b (436)`` in the L1 loss — model output is 432 wide, target 436.

Root cause — a ``W=436`` input floors through the strided encoder
(``436 → 27 → ... → 432``). The UNet **deep-supervision** branch interpolates the
output back to ``(input_height, input_width)``, but the standard (non-deep)
branch did not — violating the documented "output preserves input H, W" contract.
Fix: the ``else`` branch now interpolates back to the input size (guarded, so the
common divisible case is an exact no-op). Benefits every ``configurable_unet`` arm
on a non-divisible input, not just the cartoon-texture one.


Useful Debug Commands
======================

.. code-block:: bash

   # Config dry-run (no GPU)
   python -m spectramr.cli train --config my.yaml --dry_run

   # Smoke tests (quick sanity check)
   pytest tests/smoke/ -v --tb=short

   # Complexity audit
   radon cc src/ -s -a --min B

   # Dead code detection
   vulture src/ --min-confidence 80

   # Check manifest integrity
   python -c "
   import pickle
   m = pickle.load(open('data/manifests/train.pkl','rb'))
   print(f'{len(m)} samples, keys: {list(m[0].keys()) if m else []}')
   "

   # Profile GPU memory
   python -c "
   import torch
   print(torch.cuda.memory_summary())
   "
