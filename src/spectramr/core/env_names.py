"""Registry of every environment-variable NAME spectraMR reads.

Names only -- this module registers, it does not read. The parsed accessors live
in :mod:`spectramr.core.env`, which re-exports every constant below, so
``from spectramr.core import env`` remains the one import every consumer needs.
Split out in the Wave 0 exit-criterion work (#1400): ``env.py`` was 372 LOC
against the 300 ceiling (NN20).

The list mirrors ``.env.example`` at the repo root. Adding a new env var requires
three lines: a constant here, a ``.env.example`` entry, and a getter in
``env.py`` (if it has a parsed type).

**The constant must also be listed in ``__all__`` below.** ``env.names()`` is
derived from it, and ``env.py`` imports from it, so a constant that is declared
but not exported is invisible to every consumer *and* to the tests that check
``.env.example`` coverage. That is exactly how ``SPECTRAMR_GPU_MEMORY_FRACTION``
stayed unadvertised while its own resolver claimed it was "registered in
core/env.py". Two tests fail loudly on the same slip now, and they are not
redundant: ``test_env.py::test_all_constants_are_exported`` scans ``vars(env)``
and so only sees constants that were successfully re-exported, while
``test_env_names.py::test_all_constants_are_exported`` scans this module and
catches the one that never left it.

One family is deliberately absent: ``SPECTRAMR_LAUNCH_*`` is composed at runtime
from a prefix in ``infrastructure.execution.backends`` and written by
``spectramr launch`` for the child process to read back as provenance. It is an
internal handoff, not an operator knob -- setting one by hand falsifies the
provenance record. Its membership is derived from ``_LAUNCH_RESOURCE_FIELDS``,
so hand-copying the composed names here would recreate the very drift this
module exists to prevent.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# 1. Path resolution
# ─────────────────────────────────────────────────────────────────────────────

SPECTRAMR_DATA_ROOT = "SPECTRAMR_DATA_ROOT"
PROJECT_ROOT = "PROJECT_ROOT"
#: Repo root used by ``shared.utils.path_normalizer`` to resolve relative paths;
#: falls back to the *current working directory* when unset, so a run launched
#: from elsewhere silently resolves against the wrong tree. Distinct from
#: ``PROJECT_ROOT`` (data/output root) — the two are not interchangeable.
SPECTRAMR_ROOT = "SPECTRAMR_ROOT"
SPECTRAMR_CLUSTER_ROOT = "SPECTRAMR_CLUSTER_ROOT"
SPECTRAMR_CACHE_ROOT = "SPECTRAMR_CACHE_ROOT"
SPECTRAMR_LEGACY_ABS_PREFIXES = "SPECTRAMR_LEGACY_ABS_PREFIXES"
SPECTRAMR_LEGACY_CLUSTER_PREFIX = "SPECTRAMR_LEGACY_CLUSTER_PREFIX"
FASTMRI_DATASETS_ROOT = "FASTMRI_DATASETS_ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Device / determinism / behaviour
# ─────────────────────────────────────────────────────────────────────────────

SPECTRAMR_SUPPRESS_CLINICAL_WARNING = "SPECTRAMR_SUPPRESS_CLINICAL_WARNING"
FORCE_CPU = "FORCE_CPU"
SPECTRAMR_DEVICE = "SPECTRAMR_DEVICE"
SPECTRAMR_NO_GPU_PROBE = "SPECTRAMR_NO_GPU_PROBE"
#: Compute capability of the machine the run will land on, as "<major>.<minor>"
#: (e.g. "7.0" for a V100). Read only when a live probe cannot see a device --
#: `spectramr audit` on a login node. A malformed value RAISES rather than
#: defaulting, and a live probe always wins over it.
SPECTRAMR_TARGET_COMPUTE_CAPABILITY = "SPECTRAMR_TARGET_COMPUTE_CAPABILITY"
# Per-process GPU memory cap applied by ``initialize_device`` — float in
# (0, 1], default 0.85. Invalid values RAISE at device init (pitfall #9/#15).
SPECTRAMR_GPU_MEMORY_FRACTION = "SPECTRAMR_GPU_MEMORY_FRACTION"
PYTHONHASHSEED = "PYTHONHASHSEED"
CUBLAS_WORKSPACE_CONFIG = "CUBLAS_WORKSPACE_CONFIG"
#: Runtime dimension-guard mode — ``off`` / ``observe`` / ``enforce``, default
#: ``observe`` (log-only). An unknown value RAISES in
#: ``infrastructure.validation.dimension_contract.resolve_contract_mode``.
SPECTRAMR_DIMENSION_CONTRACT = "SPECTRAMR_DIMENSION_CONTRACT"
#: Opt-in ONLY, for CPU/CI wiring tests: permits the Gated-Conv+GRU stand-in
#: when the ``mamba_ssm`` selective-scan kernel is unavailable. Without it a
#: missing kernel fails loud, so a GRU is never trained under the "Mamba" label
#: (pitfall #9/#16). Never set it for a real Mamba experiment.
SPECTRAMR_ALLOW_MAMBA_FALLBACK = "SPECTRAMR_ALLOW_MAMBA_FALLBACK"
#: Extra plugin module paths to import at startup, split on ``os.pathsep`` AND
#: whitespace (``plugins._env_paths``). A token that fails to import raises
#: ``PluginImportError`` — and keeps raising on retry, by design.
SPECTRAMR_PLUGINS = "SPECTRAMR_PLUGINS"
#: Escalates execution-ledger substitution records from advisory to fatal.
#: Read for membership in ``{"1", "true", "yes"}`` (``execution_ledger``), so
#: unlike the bare-truthiness flags in section 6, ``=0`` here does mean off.
SPECTRAMR_LEDGER_STRICT = "SPECTRAMR_LEDGER_STRICT"
#: Absolute unix epoch second at which THIS job's allocation ends, exported by
#: the sbatch from Slurm's own remaining-time report. Training yields (saves a
#: checkpoint and stops cleanly) before it, so a requeued task resumes instead
#: of losing everything since the last periodic save.
SPECTRAMR_WALL_CLOCK_DEADLINE = "SPECTRAMR_WALL_CLOCK_DEADLINE"
#: Seconds reserved ahead of that deadline for the final checkpoint write and
#: teardown. Must exceed one logging interval's worth of step time, since the
#: deadline is only inspected at that cadence. Default 900.
SPECTRAMR_WALL_CLOCK_MARGIN_S = "SPECTRAMR_WALL_CLOCK_MARGIN_S"
#: Absolute path the run writes when it yields at the wall clock. The LAUNCHER
#: chooses it and the run obeys, so the two cannot disagree about where to look
#: (non-negotiable 17). Unset falls back to ``<run_dir>/WALL_CLOCK_YIELD``.
SPECTRAMR_WALL_CLOCK_MARKER = "SPECTRAMR_WALL_CLOCK_MARKER"

# ─────────────────────────────────────────────────────────────────────────────
# 3. CUDA / PyTorch tuning
# ─────────────────────────────────────────────────────────────────────────────

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
PYTORCH_CUDA_ALLOC_CONF = "PYTORCH_CUDA_ALLOC_CONF"
CUDA_CACHE_MAXSIZE = "CUDA_CACHE_MAXSIZE"
CUDA_CACHE_CONFIG = "CUDA_CACHE_CONFIG"
#: PTX JIT cache directory -- the NVIDIA-documented spelling, unlike
#: ``CUDA_CACHE_CONFIG`` above it (#2178). Unset, the cache stays in
#: ``~/.nv/ComputeCache``, outside ``SPECTRAMR_CACHE_ROOT``.
CUDA_CACHE_PATH = "CUDA_CACHE_PATH"
#: Inductor's generated modules and compiled kernels. Pinned explicitly
#: because Inductor derives its default from ``tempfile.gettempdir()``,
#: which memoizes -- so inheriting it through ``TMPDIR`` stops holding the
#: moment anything touches tempdir first.
TORCHINDUCTOR_CACHE_DIR = "TORCHINDUCTOR_CACHE_DIR"
TORCH_HOME = "TORCH_HOME"
#: Set to "1" by ``main.py`` to keep the eager-mode CUDA cache manager active.
TORCH_CUDA_EAGER_CACHE_MANAGER = "TORCH_CUDA_EAGER_CACHE_MANAGER"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Threading
# ─────────────────────────────────────────────────────────────────────────────

OMP_NUM_THREADS = "OMP_NUM_THREADS"
MKL_NUM_THREADS = "MKL_NUM_THREADS"
OPENBLAS_NUM_THREADS = "OPENBLAS_NUM_THREADS"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Distributed training (set by torchrun / launchers)
# ─────────────────────────────────────────────────────────────────────────────

RANK = "RANK"
LOCAL_RANK = "LOCAL_RANK"
WORLD_SIZE = "WORLD_SIZE"
#: Ranks on THIS node, as opposed to ``WORLD_SIZE`` across all of them.
#: The two are equal only on a single-node run, and conflating them is
#: what makes a per-node resource (CPU cores) look like a per-job one.
#: Read by ``core.topology`` and ``pipelines.distributed``.
LOCAL_WORLD_SIZE = "LOCAL_WORLD_SIZE"
MASTER_ADDR = "MASTER_ADDR"
MASTER_PORT = "MASTER_PORT"

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI / terminal
# ─────────────────────────────────────────────────────────────────────────────

FORCE_COLOR = "FORCE_COLOR"
NO_COLOR = "NO_COLOR"
#: Print full tracebacks on error (the env twin of ``-v`` / ``--verbose``).
#: Read for bare truthiness in ``cli/app.py``, NOT through :func:`as_bool` — so
#: ``SPECTRAMR_DEBUG=0`` ENABLES it. Unset the variable to turn it off.
SPECTRAMR_DEBUG = "SPECTRAMR_DEBUG"
#: Suppress the CLI startup notice. Same bare-truthiness read as
#: ``SPECTRAMR_DEBUG`` above: ``SPECTRAMR_QUIET=0`` is quiet. Unset to turn off.
SPECTRAMR_QUIET = "SPECTRAMR_QUIET"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Filesystem
# ─────────────────────────────────────────────────────────────────────────────

#: The five variables below plus ``TORCH_HOME``/``CUDA_CACHE_CONFIG`` are set as
#: one block by ``infrastructure.config.env_resolver.configure_cache_environment``
#: -- the SSOT for the cache layout. Do not name a single entry point as "the"
#: setter: this block lived inline in ``main.py``, which meant it applied to
#: ``spectramr train`` and to nothing else, and the multi-GPU path
#: (``torchrun -m spectramr.cli train-distributed``, which never imports
#: ``main.py``) ran with all of them unset until 2026-08-16.
TMPDIR = "TMPDIR"
#: Also the root torch resolves its JIT extension build directory from, via
#: ``torch._appdirs.user_cache_dir('torch_extensions')`` -- so leaving it unset
#: puts DeepSpeed's compiled ops under ``~/.cache``. There is no separate
#: ``TORCH_EXTENSIONS_DIR`` in this framework for exactly that reason.
XDG_CACHE_HOME = "XDG_CACHE_HOME"
#: torchmetrics' cache, pointed at ``cache_root`` with the rest of the block.
TORCH_METRICS_CACHE = "TORCH_METRICS_CACHE"
#: Triton's cache root. Registered because it is the one cache directory that
#: does NOT follow ``XDG_CACHE_HOME`` -- Triton defaults to ``~/.triton``, and
#: DeepSpeed populates it during ``import deepspeed``, so on a cluster home with
#: a quota it fails before any of our code runs. It is the one member of the
#: block applied with ``setdefault`` rather than assignment: an operator's
#: explicit export wins, because DeepSpeed's own startup warning asks them to set
#: it for NFS reasons.
TRITON_CACHE_DIR = "TRITON_CACHE_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 9. Scheduler / container identity (site-specific — change these to move host)
# ─────────────────────────────────────────────────────────────────────────────
#
# (Section 8 in `.env.example` is the sim2rank tooling, which lives entirely
# under ``scripts/`` and so has no constant here — the numbering is kept in
# step with that file rather than made contiguous.)
# Read by ``infrastructure.execution.backends``, which carries the container
# image names as module-level defaults (``spectramr:v6.1``, ``spectramr.sif``).
# The Slurm account and mail address carry NO default -- both are site-specific,
# so there is no value this tree could ship that is correct anywhere but one
# cluster; an unresolved account is raised on at render time rather than
# guessed. ``infrastructure.orchestration.slurm_backend``, the second submitter
# that used to hardcode both and read no environment variable at all, now
# delegates to the same ``ResourceSpec`` resolver (#1146).
#
# Still open: ``#SBATCH`` directives written literally in ``*.sbatch`` files are
# parsed by Slurm rather than the shell, so no variable ever reaches them
# (#1145). Those files are internal and are not part of the public tree.

#: Slurm ``--account``. No default: an allocation name is site-specific, so
#: ``SlurmBackend.render_directives`` raises when this is unset rather than
#: emitting an ``--account`` line that sbatch rejects for an unrelated-looking
#: reason.
SPECTRAMR_SLURM_ACCOUNT = "SPECTRAMR_SLURM_ACCOUNT"
#: Slurm ``--partition``. No default — when unset, ``backends.py`` omits the
#: flag entirely and the cluster's own default partition applies.
SPECTRAMR_SLURM_PARTITION = "SPECTRAMR_SLURM_PARTITION"
#: Docker image tag for the container execution backend.
SPECTRAMR_DOCKER_IMAGE = "SPECTRAMR_DOCKER_IMAGE"
#: Slurm ``--mail-user`` for campaign-generated job scripts. No default -- when
#: unset the ``#SBATCH --mail-user`` line is omitted entirely and Slurm applies
#: its own policy, which is the correct behaviour for anyone who is not this
#: site's original author.
SPECTRAMR_SLURM_MAIL_USER = "SPECTRAMR_SLURM_MAIL_USER"
#: Apptainer/Singularity ``.sif`` image for the container execution backend.
SPECTRAMR_APPTAINER_SIF = "SPECTRAMR_APPTAINER_SIF"


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_CACHE_CONFIG",
    "CUDA_CACHE_MAXSIZE",
    "CUDA_CACHE_PATH",
    "CUDA_VISIBLE_DEVICES",
    "FASTMRI_DATASETS_ROOT",
    "FORCE_COLOR",
    "FORCE_CPU",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "MKL_NUM_THREADS",
    "SPECTRAMR_ALLOW_MAMBA_FALLBACK",
    "SPECTRAMR_APPTAINER_SIF",
    "SPECTRAMR_CACHE_ROOT",
    "SPECTRAMR_CLUSTER_ROOT",
    "SPECTRAMR_DATA_ROOT",
    "SPECTRAMR_DEBUG",
    "SPECTRAMR_DEVICE",
    "SPECTRAMR_DIMENSION_CONTRACT",
    "SPECTRAMR_DOCKER_IMAGE",
    "SPECTRAMR_GPU_MEMORY_FRACTION",
    "SPECTRAMR_LEDGER_STRICT",
    "SPECTRAMR_WALL_CLOCK_DEADLINE",
    "SPECTRAMR_WALL_CLOCK_MARGIN_S",
    "SPECTRAMR_WALL_CLOCK_MARKER",
    "SPECTRAMR_LEGACY_ABS_PREFIXES",
    "SPECTRAMR_LEGACY_CLUSTER_PREFIX",
    "SPECTRAMR_NO_GPU_PROBE",
    "SPECTRAMR_TARGET_COMPUTE_CAPABILITY",
    "SPECTRAMR_PLUGINS",
    "SPECTRAMR_QUIET",
    "SPECTRAMR_ROOT",
    "SPECTRAMR_SLURM_ACCOUNT",
    "SPECTRAMR_SLURM_MAIL_USER",
    "SPECTRAMR_SLURM_PARTITION",
    "SPECTRAMR_SUPPRESS_CLINICAL_WARNING",
    "NO_COLOR",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PROJECT_ROOT",
    "PYTHONHASHSEED",
    "PYTORCH_CUDA_ALLOC_CONF",
    "RANK",
    "TMPDIR",
    "TORCH_CUDA_EAGER_CACHE_MANAGER",
    "TORCH_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_METRICS_CACHE",
    "TRITON_CACHE_DIR",
    "WORLD_SIZE",
    "XDG_CACHE_HOME",
]
