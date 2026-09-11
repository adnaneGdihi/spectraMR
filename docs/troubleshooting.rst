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

The configuration is nested. Read the full path — a shorter guess raises the
same ``AttributeError`` you are trying to fix.

.. code-block:: python

   # ❌ not on the schema
   config.lr
   config.lambda_l1

   # ✅
   config.optimization.optimizer.learning_rate
   config.losses.image_losses[0].weight

When you are unsure where a field lives, look it up in
:doc:`config_schema_reference` rather than guessing — it lists every key the
schema declares, with its full path.

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
   * - ``Model outputs N channels but the target provides 1``
     - A head that emits distribution parameters (e.g. ``[mean, logvar]``) must
       set ``predicts_distribution_params = True`` on its strategy class, so the
       channel check and the metric reducer read only the mean channel


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
     precision:
       dtype: bfloat16

   # Fix 2: disable AMP entirely
   optimization:
     precision:
       enabled: false

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
     precision:
       enabled: true
       dtype: bfloat16

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
:class:`spectramr.models.blocks.dual_domain_attention_kan.ComplexMHA` would
otherwise form a dense ``[B, h, N, N]`` score matrix over the full feature-map
sequence (``N = H*W``). On a 256² map that is tens of GiB.

``forward`` **chunks over the query dimension** above
``_COMPLEX_MHA_QUERY_CHUNK`` (default 2048): softmax / top-k act per query on
the key dimension, so the chunked result is *numerically identical* to the
dense one, while peak attention memory drops from ``O(N²)`` to ``O(chunk·N)``.
This mirrors the ``max_band_tokens`` cap on ``RadialBandAttention`` and
``MultiScaleFreqBandAttention``. There is no YAML knob — the chunk size is an
internal memory optimisation.

``max_dense_attn_tokens`` is the supported config lever for the score-tensor
peak. When the feature-map sequence ``N = H*W`` exceeds it, the dense
image/cross branches adaptive-pool to a ``round(sqrt(budget))²`` grid before
attending, attend there, then interpolate back; the **output k-space stays full
256²** regardless, so this coarsens only the *global attention mixing*, not the
data or the reconstruction target. Because the budget sets the pooled attention
resolution, keep it uniform across any set of arms you intend to compare, or a
memory-vs-quality difference confounds the comparison.

If that is not enough headroom, gradient checkpointing
(``optimization.gradient.enable_checkpointing: true``) and a larger GPU are the
remaining levers.

This is the right knob precisely because ``data.patch_size`` is **not** safe to
shrink on these k-space arms. For ``dataset_type: m4raw`` the Subject's
``input`` / ``target`` keys hold *k-space*, and ``patch_size`` equals M4Raw's
native 256² acquisition matrix, so at ``256`` it
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
:func:`spectramr.pipelines.train._all_reduce_val_metrics` all-reduce-sums both
the per-metric running-sums and the sample count before dividing, giving the
correct sample-weighted global mean rather than one padded shard's value. It is
a **no-op** when ``torch.distributed`` is not initialised (the default
single-process ``spectramr train`` path), so single-GPU runs are unaffected.


Mixed precision: ``precision.dtype`` selects the autocast dtype
================================================================

``optimization.precision.dtype`` chooses the autocast precision when
``optimization.precision.enabled: true``. It maps to the AMP policy in
:func:`spectramr.infrastructure.training.mixed_precision.resolve_amp_precision`:

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - ``precision.dtype``
     - autocast
     - Notes
   * - *(unset)* / ``float16``
     - fp16
     - Needs a ``GradScaler``; narrow dynamic range
       (max ≈ 65504) can overflow to ``inf`` on unstable models. **Disabled
       automatically for complex/k-space flows** (complex64 cannot mix with
       Half weights).
   * - ``bfloat16``
     - bf16
     - **Preferred.** Same exponent range as fp32 → no loss scaling, no
       overflow, no GradScaler scale-collapse. Halves the *autocast'd*
       activations (convs/projections). Works under complex autocast. Full
       Tensor-Core throughput on Ampere-class and newer GPUs.
   * - ``float32``
     - *(off)*
     - Full precision — AMP is disabled even if ``precision.enabled: true`` (the knob
       cannot silently no-op into fp16).

The resolved value is logged at startup
(``[Mixed Precision] enabled=… precision=…``).

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

A third candidate, ``training.diffusion is not None``, is deliberately not used:
it captures PINN, FNO, Vision-Mamba and VAE reconstruction arms that merely
carry the sub-block, and no true ones the two signals above miss. In a check
that hard-errors, over-capture blocks AMP on arms legitimately entitled to it,
which is worse than under-capture.

Matching on ``model_type`` is *not* how this works, deliberately: that would be
a second resolver for one decision, and AMP has a single owner.

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
the whole answer if it were populated; it is unpopulated on most registered
models, which is why four signals are needed instead of one lookup.

.. rubric:: Getting throughput on a complex arm instead

``bfloat16`` AMP (which *does* work under complex autocast),
``optimization.optimizer.fused``, and ZeRO sharding. See
:doc:`training_throughput`.

For OOM relief prefer ``bfloat16``: it does **not** need loss scaling and does
**not** aggravate the gradient-explosion → NaN instability, whereas fp16 can
collapse the ``GradScaler`` scale and silently skip steps.

**Mamba arms and fp16.** ``precision.dtype: float16`` *is* viable for
image-domain Mamba arms — real fp16 autocast, with the ``GradScaler`` wired in
``steppers.py`` — and
:class:`spectramr.models.blocks.mamba_block.MambaBlock` force-runs the SSM/GRU
recurrence in fp32 (``autocast(enabled=False)`` + ``.float()``) regardless of
``precision.dtype``, so the precision-fragile long recurrence (L = H·W) and the
cuDNN-GRU-under-fp16 crash are both avoided. bf16 is still preferred (no
scale-collapse failure mode; identical Tensor-Core throughput), and
because the recurrence is fp32-pinned either way, fp16's extra mantissa bits
buy nothing on the part that matters.

**fp8 is not supported.** It is not an autocast dtype: PyTorch fp8 *training*
requires ``torchao.float8`` or NVIDIA Transformer-Engine, explicit per-layer
conversion of ``nn.Linear`` (and per-tensor scaling), and Hopper/Ada hardware.
The Mamba selective-scan and ``Conv2d`` stems here are not standard fp8 targets,
so ``precision.dtype`` deliberately rejects ``float8`` rather than advertise an
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

:doc:`distributed_training` carries the launch commands for single-node and
multi-node runs, the rendezvous settings, and the failure modes specific to
each.

Use ``DistributedDataParallel`` (the config-driven path above), never
``nn.DataParallel``.
Combine with ``precision.dtype: bfloat16`` for the largest effective batch per
GPU.


Validation Errors
=================

``Validation produced zero successful batches``
------------------------------------------------

Every validation batch raised the same exception (a shape / channel mismatch in
the strategy's validation forward, or an OOM), so the run fails loud rather than
shipping image-less and green. The **root-cause traceback of the first failing
batch is embedded in the raised error itself**
(``--- first validation-batch failure (root cause) ---`` block). Read that
block: it names the exact tensor op and shapes. Common causes: the model emits 2-channel real-stacked complex while the
validation metric compares against a 1-channel magnitude target; a strategy
``_validation_forward`` that returns ``None``; or a ``val_batch`` whose keys
``_unpack_batch`` doesn't recognise.

``RuntimeError: quantile() input tensor is too large``
-------------------------------------------------------

**Root cause**: ``torch.quantile`` sorts the reduced dimension and refuses any
reduced length above ``2**24`` (~16.7M) elements. The digital-twin marker
embedder (``infrastructure/physics/digital_twin_simulator.py``) derives the
tissue-intensity scale from a 0.75 quantile of the anatomy magnitude; on the
**single-coil** path it flattens the *whole* tensor, so a larger **validation**
batch tips over the cap and every validation batch raises → *"Validation
produced zero successful batches"*.

**Fix**: the quantile sites go through ``_robust_quantile`` (same module),
which decimates the reduced dimension with an even stride down to ~16.7M samples
before the quantile when it would overflow — deterministic (no ``randperm``, so
seeding/determinism is preserved) and statistically unbiased for the smooth
0.75/0.99 quantiles the embedder uses. No config change needed.

Checkpoint Errors
==================

``KeyError: 'state_dict'`` when loading checkpoint
----------------------------------------------------

The checkpoint is in an older format that stores the weights under a different
key. Convert:

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


``MambaBlock requires the official mamba_ssm selective-scan kernel`` (by design)
=================================================================================

Mamba/SSM models (``hilbert_mamba``, ``geomamba``, ``d2_mamba``, ``bloch_mamba``,
…) **require** the official ``mamba_ssm`` CUDA kernel.
:class:`spectramr.models.blocks.mamba_block.MambaBlock` now **raises** at
construction when it is missing or its kernel failed to build, rather than
silently substituting a Gated-Conv+GRU block — that fallback is **not an SSM**,
so a silent substitution would train a GRU under the "Mamba" label and make
every result scientifically mislabelled.

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
``se3_lie_algebra_mamba`` and ``swin_mamba_kan``. A handful of models keep a
**bespoke**
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
``uqi``) then raises at runtime rather than fabricating a ``0.0``, so any arm
listing one of them in ``validation.metrics`` crashes at the first validation
step.

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
ends the stream early and the partial file is renamed as "complete" — a
``size > 0`` completeness test does not notice.

.. note::

   The mirroring and manifest tooling referenced throughout this section fetches
   from the maintainers' cluster mirror into their manifest layout, and is not
   part of this distribution. The diagnosis here is portable; the fetch commands
   are not, so they are described rather than quoted.

**Repair**: re-fetch the truncated payload from wherever you obtained it. A
partially-written ``.cfl`` is indistinguishable from a complete one by anything
except its length, so any integrity check you build has to compare the byte count
against the ``.hdr`` dimensions rather than trust a ``status: downloaded`` marker.

``download_external_datasets.py`` (a) **never promotes a short stream** —
it verifies bytes-downloaded == ``Content-Length`` before the atomic rename and
otherwise leaves the ``.part`` for the next Range-resume; (b) treats a
``.cfl`` whose size ≠ ``prod(hdr dims) × 8`` as **not present** (``_present`` /
``--verify``) so the dataset re-fetches; and (c) **unlinks** a previously-truncated
``.cfl`` before re-streaming it.

No manifest regeneration is needed for this fix: the manifest already lists
``v05`` (its ``shape`` comes from the intact ``.hdr``; only the ``.cfl`` payload
was short), and the reader reads the now-intact ``.cfl`` at load time. Regenerate
only if the file *list* changed.

The ``.cfl`` files are streamed **directly** into ``<id>/raw`` (no archive), so
``extract_external_datasets.py`` is not in this path — it only expands archived
bundles and already fails loud on a truncated ``.zip`` (``BadZipFile``).

**"A byte-level check says OK but the loader still raises"** — these disagree only
when they read **different files**. A check that globs a directory sees whatever
is on disk, but the loader resolves each ``.cfl`` from the **index manifest**
(``data.index_path`` → ``data_root / relative_path``). A re-fetched
``raw/v05.cfl`` can pass the directory glob while the manifest still points the
loader at a stale copy (or a missing/renamed file). Use the paths the
``BartDataset`` index actually reads, rather than globbing the directory. Per
record the useful verdicts are ``OK`` / ``MISMATCH`` (the loader will raise
``ValueError``) / ``MISSING-CFL`` (``FileNotFoundError``) / ``TRAILING-BYTES``.

The verdict mirrors ``io_strategies.BartCflStrategy`` exactly: ``np.fromfile``
*floors* trailing bytes, so the loader raises iff ``(st_size // 8) !=
prod(dims)`` (the ``account()`` ``loader_will_raise`` key), distinct from the
stricter byte-perfect ``ok``. Path resolution also mirrors the loader's
**basename normalisation** (``BartCflStrategy`` strips a trailing ``.cfl`` then
reads ``<base>.cfl``): a manifest record may carry either the BART bare basename
``relative_path: "v05"`` or the explicit ``"v05.cfl"`` — both resolve to the same
``<data_root>/v05.cfl`` the loader reads, so a bare-basename manifest is **not**
falsely flagged ``MISSING-CFL``. If manifest mode reports ``MISMATCH`` /
``MISSING-CFL`` while a bare glob is clean, the manifest genuinely points at a
**stale/truncated/missing** payload — re-fetch it or regenerate the manifest
(above).

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
------------------------------------------------------------------------------

**Symptom**: an mrixfields ``ulf_source`` arm crashes at data-loader build with
*"produced 0 pairs: no group matches the pinned field 0.1 T; fields present =
[5.0, 7.0]"* — even though ``mrixfields2026_train.json`` contains all five field
strengths (9 complete groups, each 0.1/1.5/3/5/7 T).

**Cause**: the manifest records are **field-sorted** (all 0.1 T first … 7 T
last). The upstream train/val split is a flat contiguous record slice, so a
90/10 cut put *every* 0.1 T source in train and left validation with only the
top fields. ``ulf_source`` pins the 0.1 T source, so the val dataset matched
nothing and fail-fasted. It is **not** a missing-data problem, so pointing the
arm at a larger corpus will not help.

**Resolution**: ``DatasetInstantiator._create_mrixfields`` splits
**group-aware** (on whole ``pairing_group`` groups) for *every* field-pinned
policy — ``multi_source``, ``ulf_source``, ``prior``, ``fixed_target`` — so each
split keeps complete field groups and the pinned field is present in both.

``No losses were built by LossBuilder`` (ablation arms)
-------------------------------------------------------

**Symptom**: an ablation arm (e.g. ``mrixfields_b*_ablate_*``) crashes at build
with *"No losses were built by LossBuilder"*. The raise names the strategy it
judged and both remedies; match on that leading phrase, not on the whole
sentence.

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

**The placeholder is not the only route, but it is the one this family needs.**
``LossBuilder.validate()`` accepts an empty stack from a strategy that declares it
owns its *whole* objective — ``inline_losses`` set **and** ``folds_image_losses =
False``. ``ScatteringBesovStrategy`` declares ``folds_image_losses = True``: it does
consume the builder's image list, so it is not exempt and the placeholder stands.
A strategy that declares neither is never exempt — silence is not a claim of
ownership (#1918).

``Paired-NIfTI VAE trains HF→ULF (degradation) instead of autoencoding HF``
---------------------------------------------------------------------------

**Symptom**: a ``dataset_type: nifti_paired`` stage-1 VAE in a two-stage LDM
shows a **sharp HF** ``input`` and a **noisy ULF**
``target`` with the SAME shape but different statistics in its first-steps
snapshot — the model is minimizing ``||Dec(Enc(HF)) − ULF||``, a degradation
network, and its frozen decoder later emits low-field appearance that corrupts
stage 2.

**Root cause**: the mode name is ``<input>_to_<target>``, so ``hf_to_ulf`` is a
genuine HF→ULF *translation* (input HF, target ULF), NOT an autoencoder.

**Fix**: use one of the single-field autoencode modes, ``hf_to_hf`` or
``ulf_to_ulf``, which DROP the opposite arm (``target_path`` → ``None`` so the
self-supervised branch aliases ``target = input``) — ``input ≡ target`` by
construction. A missing or mismatched target **raises** rather than silently
substituting the input. A ``ConfigHealthChecker`` rule
(``check_vae_pretrain_autoencodes_single_field``) rejects a ``vae_pretrain`` arm
on paired data that declares a translation direction. Set a stage-1 VAE to
``data.bidirectional_mode: hf_to_hf``; ``hf_to_ulf`` remains valid as a real
bidirectional-translation direction.


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

Symptom: the saved ``metrics/real_images`` panel renders as a centre-bright
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
deliver an *image*. A naive guard would re-FFT that image into k-space. The seam
is therefore a no-op for ``rss_image`` arms and only IFFTs the genuinely
k-space-delivering ``svd`` arms.

Config half of the fix: an ``svd`` arm must keep the complex pair so the seam
has an imaginary half to invert — set ``data.target_channels: 2`` (not ``1``,
which strips phase and leaves a single real channel the IFFT cannot use).

``ConcreteVirtualFiducialStrategy`` and ``IBVFStrategy`` carry an equivalent
inline guard; the SE3-navigator, motion-meta, distillation and Bloch-manifold
strategies route through the shared base seam.


Strategy-Specific Issues
=========================

TTO strategy: ``motion_traj not updating``
-------------------------------------------

``ConcreteTTOStrategy`` is an inference-time strategy, not a training strategy.
Invoke with ``--resume`` on a pre-trained checkpoint:

.. code-block:: bash

   python -m spectramr.cli predict \
       --model checkpoints/hypermamba_best.safetensors \
       --config experiments/<paradigm>/<your-arm>.yaml

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
measurement-independent solution: the reconstruction does not
depend on the actual acquisition, and PSNR/SSIM against a fixed validation set
can look plausible while the model has learned a prior, not a reconstruction.

This affects any diffusion arm with ``in_channels: 1`` and no conditioning
path. To condition the denoiser on the measurement, set ``condition_on_input`` and give
the model an extra input channel — the strategy concatenates the (resized) input
onto the noised target along the channel axis:

.. code-block:: yaml

   model:
     in_channels: 2          # [noisy target || conditioning input]
   training:
     diffusion:
       condition_on_input: true

The flag defaults to ``false`` (unconditional) and is a
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


Useful Debug Commands
======================

.. code-block:: bash

   # Config dry-run (no GPU)
   python -m spectramr.cli train --config my.yaml --dry_run

   # Smoke tests (quick sanity check)
   pytest tests/smoke/ -v --tb=short

   # Pre-flight a config (schema + health checks, no GPU work)
   spectramr audit my.yaml

   # Check manifest integrity
   python -c "
   import pickle
   m = pickle.load(open('<your-train-manifest>.pkl','rb'))
   print(f'{len(m)} samples, keys: {list(m[0].keys()) if m else []}')
   "

   # Profile GPU memory
   python -c "
   import torch
   print(torch.cuda.memory_summary())
   "
