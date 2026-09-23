"""Run provenance & traceability capture — canonical home.

Single source of truth for *what code, what machine, when, how big* — the
metadata that makes a training run reproducible and a bundle self-describing.
Before this module the run log answered "which services started" but nothing
about the commit, host, wall-clock, or model/data size, so a result on the
cluster could not be tied back to the code that produced it.

Every helper is **pure and fail-open**: a missing ``git`` binary, a broken
torch, or an un-dumpable config must never abort a training run, so each
returns a sentinel (``{"available": False}`` / ``{}`` / ``None``) on error.

Public API:

- :func:`torch_runtime` — torch/CUDA/device probe (also consumed by ``doctor``)
- :func:`cpu_resources` — node cores vs. the cores this process may actually use
- :func:`memory_resources` — node RAM vs. the cgroup/scheduler allocation
- :func:`gpu_resources` — GPU count + type census, driver, node-vs-visible split
- :func:`node_resources` — the composite ``{cpu, memory, gpu}`` hardware record
- :func:`runtime_environment` — python/platform/host/pid/user + torch + node
- :func:`git_provenance` — commit sha, branch, dirty flag, subject, time
- :func:`count_parameters` — total/trainable/frozen params + fp32 size (MB)
- :func:`config_fingerprint` — stable sha256 prefix of the resolved config
- :func:`effective_batch_size` — batch x grad-accum x world-size
- :func:`parallel_provenance` — declared ``parallel.strategy`` vs. the live group
- :func:`make_run_id` — correlation id (``name-stamp-gitsha``)
- :func:`collect_run_provenance` — the composite stamped into ``provenance.json``
- :func:`format_provenance_lines` / :func:`log_provenance` — banner rendering
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime
from typing import Any

# CPU/cgroup probes moved to ``core.resources`` so the numbers stamped here and
# the numbers the dataloader-worker clamp decides on come from ONE implementation
# (they used to be reporting-only, and nothing consumed them). Aliased to their
# former private names to keep this module's call sites unchanged.
from spectramr.core.curriculum import resolve_curriculum_state
from spectramr.core.resources import ALLOC_CPU_ENV as _ALLOC_CPU_ENV
from spectramr.core.resources import ALLOC_GPU_ENV as _ALLOC_GPU_ENV
from spectramr.core.resources import cgroup_memory_limit_gb as _cgroup_memory_limit_gb
from spectramr.core.resources import cpu_resources
from spectramr.core.resources import env_int as _env_int

_logger = logging.getLogger(__name__)

# Distributed/cluster env vars that carry the world size, in priority order.
_WORLD_SIZE_ENV = ("WORLD_SIZE", "SLURM_NTASKS", "SLURM_NPROCS")
# Cluster scheduler fields worth stamping for traceability (job correlation) and
# for the *allocation* — what the scheduler actually granted, which is what the
# run really had, not what the node happens to own.
_SLURM_FIELDS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_NODELIST",
    "SLURM_NTASKS",
    "SLURM_GPUS",
    "SLURM_JOB_PARTITION",
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
    "SLURM_MEM_PER_GPU",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_NODE",
    "SLURM_JOB_GPUS",
    # Node topology. Everything above answers "how much did the scheduler grant
    # *on this node*"; without these the record could not answer "how many
    # nodes", so a 2-node run and a 1-node run stamped indistinguishable
    # allocations. Stamped as raw strings by the comprehension in
    # `collect_run_provenance` -- `SLURM_TASKS_PER_NODE` is formatted "4(x2)"
    # for a heterogeneous allocation, so it must never be fed to `_env_int`.
    "SLURM_NNODES",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_TASKS_PER_NODE",
    "SLURM_NODEID",
)
# Launcher-injected env: torchrun / torch.distributed.launch / a spawn wrapper.
# Deliberately NOT folded into `_SLURM_FIELDS`, whose contract above is "what
# the scheduler granted" -- these say how the process group was *wired*, which
# is the other half of "did this run on 4 GPUs?". A 4-GPU allocation launched
# without torchrun trains four identical single-process runs, and only these
# fields distinguish that from a real 4-way shard.
_LAUNCHER_FIELDS = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
)
# Node-count env, in priority order. Slurm spells it two ways depending on
# version; torchrun derives it as world/local-world and exports neither.
_NODE_COUNT_ENV = ("SLURM_NNODES", "SLURM_JOB_NUM_NODES")
# Scheduler env vars carrying the GPU count, in priority order. Owned by
# ``core.resources`` now: `allocated_count` used to be reporting-only, and the
# decision that finally consumes it (`pipelines.distributed.idle_device_refusal`)
# must read the same tuple this record was stamped from.
_BYTES_PER_GB = 1024**3


def torch_runtime() -> dict[str, Any]:
    """Probe torch / CUDA / devices without crashing if torch is absent/broken.

    Shared SSOT for ``spectramr doctor`` and run provenance. Returns
    ``{"available": False, "import_error": ...}`` when torch cannot import.
    """
    info: dict[str, Any] = {"available": False}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch-less env
        info["import_error"] = str(exc)
        return info

    info["available"] = True
    info["version"] = torch.__version__
    info["cuda_compiled"] = getattr(torch.version, "cuda", None)
    cuda_ok = bool(torch.cuda.is_available())
    info["cuda_available"] = cuda_ok
    info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    info["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    devices: list[dict[str, Any]] = []
    if cuda_ok:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            # SM count separates same-named SKUs (e.g. an MIG slice from a whole
            # A100); uuid pins the exact physical card for per-device forensics.
            uuid = getattr(props, "uuid", None)
            devices.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_mem_gb": round(props.total_memory / _BYTES_PER_GB, 2),
                    "capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                    "uuid": str(uuid) if uuid is not None else None,
                }
            )
    info["devices"] = devices
    return info


def memory_resources() -> dict[str, Any]:
    """Node RAM inventory **and** the ceiling this run was actually granted.

    ``usable_gb`` folds in the cgroup limit and the scheduler allocation, so an
    OOM in a 64 GB SLURM allocation on a 1 TB node is diagnosable from the
    record alone.
    """
    info: dict[str, Any] = {
        "total_gb": None,
        "available_gb": None,
        "swap_total_gb": None,
        "cgroup_limit_gb": _cgroup_memory_limit_gb(),
        "allocated_gb": None,
        "usable_gb": None,
    }
    # SLURM reports its memory allocation in MB.
    mem_per_node = _env_int(("SLURM_MEM_PER_NODE",))
    if mem_per_node:
        info["allocated_gb"] = round(mem_per_node / 1024, 2)
    else:
        per_cpu = _env_int(("SLURM_MEM_PER_CPU",))
        cpus = _env_int(_ALLOC_CPU_ENV)
        if per_cpu and cpus:
            info["allocated_gb"] = round(per_cpu * cpus / 1024, 2)
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["total_gb"] = round(vm.total / _BYTES_PER_GB, 2)
        info["available_gb"] = round(vm.available / _BYTES_PER_GB, 2)
        info["swap_total_gb"] = round(psutil.swap_memory().total / _BYTES_PER_GB, 2)
    except Exception:  # psutil absent/unsupported → node totals stay None
        _logger.debug("psutil memory probe failed", exc_info=True)
    limits = [v for v in (info["cgroup_limit_gb"], info["allocated_gb"], info["total_gb"]) if v]
    if limits:
        info["usable_gb"] = min(limits)
    return info


def _nvidia_smi_devices() -> tuple[list[dict[str, Any]], str | None]:
    """Every GPU physically on the node + the driver version (fail-open).

    Complements the torch probe, which only ever sees the ``CUDA_VISIBLE_DEVICES``
    subset — and sees *nothing* when the driver is broken, which is exactly the
    case worth recording under the accelerated-run contract.
    """
    query = "index,name,memory.total,driver_version,uuid"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:  # nvidia-smi absent (CPU node, mac, container without it)
        return [], None
    if out.returncode != 0:
        return [], None
    devices: list[dict[str, Any]] = []
    driver: str | None = None
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx, name, mem_mib, driver_version, uuid = parts
        driver = driver or driver_version
        try:
            total_mem_gb = round(float(mem_mib) / 1024, 2)
        except ValueError:
            total_mem_gb = None
        devices.append(
            {
                "index": int(idx) if idx.isdigit() else idx,
                "name": name,
                "total_mem_gb": total_mem_gb,
                "uuid": uuid,
            }
        )
    return devices, driver


def gpu_resources(torch_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """GPU census: how many, of what type, visible vs. present on the node.

    ``count``/``types`` describe what this process can *use*; ``node_count``/
    ``node_types`` describe what the node *has*. They differ whenever
    ``CUDA_VISIBLE_DEVICES`` masks devices — and the gap is the difference
    between "this node has no GPU" and "this job was given none of its 4".

    Pass an already-probed :func:`torch_runtime` dict to avoid a second probe;
    device enumeration itself stays single-sourced there.
    """
    info = torch_info if torch_info is not None else torch_runtime()
    devices = list(info.get("devices") or [])
    types: dict[str, int] = {}
    for dev in devices:
        name = dev.get("name")
        if name:
            types[name] = types.get(name, 0) + 1
    node_devices, driver = _nvidia_smi_devices()
    node_types: dict[str, int] = {}
    for dev in node_devices:
        name = dev.get("name")
        if name:
            node_types[name] = node_types.get(name, 0) + 1
    return {
        "count": len(devices),
        "types": types,
        "total_mem_gb": round(sum(d.get("total_mem_gb") or 0.0 for d in devices), 2) or None,
        "cuda_available": bool(info.get("cuda_available")),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "allocated_count": _env_int(_ALLOC_GPU_ENV),
        "driver_version": driver,
        "devices": devices,
        "node_count": len(node_devices),
        "node_types": node_types,
        "node_devices": node_devices,
    }


def node_resources(torch_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """The compute node's hardware record: ``{cpu, memory, gpu}`` (fail-open).

    Each sub-probe degrades independently — a node without ``nvidia-smi`` still
    reports cores and RAM.
    """
    record: dict[str, Any] = {}
    for key, probe in (
        ("cpu", cpu_resources),
        ("memory", memory_resources),
        ("gpu", lambda: gpu_resources(torch_info)),
    ):
        try:
            record[key] = probe()
        except Exception:  # one dead probe must not sink the record
            _logger.debug("node %s probe failed", key, exc_info=True)
            record[key] = {}
    return record


def runtime_environment() -> dict[str, Any]:
    """Host + interpreter + torch + node-hardware snapshot (fail-open)."""
    try:
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
    except Exception:  # pragma: no cover - defensive
        user = "unknown"
    # Probe torch once and hand the result to the GPU census, so device
    # enumeration has exactly one site and the two views cannot drift.
    torch_info = torch_runtime()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "user": user,
        "torch": torch_info,
        "node": node_resources(torch_info),
    }


def _git(args: list[str], repo_dir: str | None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # pragma: no cover - git absent / not a repo
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_provenance(repo_dir: str | None = None) -> dict[str, Any]:
    """Capture the code's git identity: sha, branch, dirty flag, subject, time.

    ``dirty=True`` means the working tree had uncommitted changes at run start
    — the single most important traceability flag, because a "clean" sha alone
    does **not** reproduce a run whose tree was dirty.
    """
    sha = _git(["rev-parse", "HEAD"], repo_dir)
    if sha is None:
        return {"available": False}
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    status = _git(["status", "--porcelain"], repo_dir)
    return {
        "available": True,
        "sha": sha,
        "sha_short": sha[:12],
        "branch": branch,
        "dirty": bool(status),
        "subject": _git(["log", "-1", "--pretty=%s"], repo_dir),
        "committed_at": _git(["log", "-1", "--pretty=%cI"], repo_dir),
    }


def count_parameters(model: Any) -> dict[str, Any]:
    """Total / trainable / frozen parameter counts + footprint in MB (fail-open).

    ``size_mb`` uses each parameter's real ``element_size()`` so mixed-dtype
    models report honestly. Empty dict on any error (e.g. ``model is None``).
    """
    try:
        total = 0
        trainable = 0
        n_bytes = 0
        for p in model.parameters():
            n = p.numel()
            total += n
            n_bytes += n * p.element_size()
            if p.requires_grad:
                trainable += n
    except Exception:  # never block training on a parameter count
        return {}
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "size_mb": round(n_bytes / 1024**2, 2),
    }


# How far ``describe_dataloader`` will unwrap a dataset looking for the
# ``provenance_counts`` hook. A ``tio.Queue`` is one hop; this repo's dataset
# wrappers add one or two more. The cap is what makes a self-referential or
# cyclic wrapper terminate instead of hanging the banner.
_UNWRAP_DEPTH = 4


def describe_dataloader(loader: Any) -> dict[str, Any]:
    """Count what a loader will hand out, with every number named by its unit.

    Provenance used to record ``{split: len(loader)}`` -- a bare int whose unit
    lived only in the startup banner's ``" batches"`` suffix. Read from the JSON
    alone, ``train: 768`` against a folder of 1024 files reads as a 25 % data
    loss. It is in fact ``1024 files -> 384 (patient, contrast) groups ->
    x4 samples_per_volume = 1536 patches -> /2 batch_size (drop_last) = 768
    batches``, and every step of that chain is now on the record. So each key
    here carries its unit in its own name.

    Two counts are universal and cheap:

    ``batches``
        ``len(loader)`` -- what the training loop will iterate.
    ``samples``
        ``len(loader.dataset)`` -- what the loader draws from before batching.
        For a ``tio.Queue`` that is the patch count, not the subject count.

    Anything richer is dataset vocabulary, so it is *asked for* rather than
    guessed: a dataset, or anything it wraps, may expose
    ``provenance_counts() -> dict`` and its keys are merged in. Nothing is
    derived from torchio's ``Queue.num_subjects``, which counts index entries
    (here: (patient, contrast) groups) and would land a second, different number
    under a name a reader would take for a patient count -- the exact ambiguity
    this function exists to remove.

    Cheapness is not incidental. ``len(tio.Queue)`` routes to the uncached
    ``iterations_per_epoch``, which walks ``dry_iter()``; that is affordable
    only because ``dry_iter`` returns metadata shells and reads no voxels.
    Do not add a count that touches image data (non-negotiable 9).

    Fail-open per this module's posture, but never silently: a source that
    raises is named in ``incomplete`` rather than dropped (pitfall #16).
    """
    counts: dict[str, Any] = {}
    incomplete: list[str] = []

    try:
        counts["batches"] = len(loader)
    except (TypeError, AttributeError) as exc:
        incomplete.append(f"batches: {type(exc).__name__}")

    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        try:
            counts["samples"] = len(dataset)
        except (TypeError, AttributeError) as exc:
            incomplete.append(f"samples: {type(exc).__name__}")

    # The dataset that knows its own vocabulary may sit behind a wrapper: a
    # ``tio.Queue`` holds it as ``subjects_dataset``, this repo's wrappers as
    # ``dataset``. Walk both, and resolve each hop with an explicit ``is None``
    # rather than ``a or b`` -- an empty dataset is falsy, so the ``or`` idiom
    # would skip past a real (len 0) wrapper and mis-resolve the owner.
    hook_owner = None
    node = dataset
    for _ in range(_UNWRAP_DEPTH):
        if node is None:
            break
        if callable(getattr(node, "provenance_counts", None)):
            hook_owner = node
            break
        nxt = getattr(node, "subjects_dataset", None)
        if nxt is None:
            nxt = getattr(node, "dataset", None)
        node = nxt

    if hook_owner is not None:
        try:
            extra = hook_owner.provenance_counts() or {}
        except Exception as exc:  # a diagnostic count never blocks training
            incomplete.append(f"provenance_counts: {type(exc).__name__}")
            extra = {}
        for key, value in extra.items():
            # ``batches``/``samples`` are the loader's own facts; a dataset that
            # redefines them is a defect worth seeing, not worth honouring.
            if key in counts:
                incomplete.append(f"provenance_counts shadowed {key!r}")
                continue
            counts[key] = value

    if incomplete:
        counts["incomplete"] = incomplete
    return counts


def config_fingerprint(config: Any) -> str | None:
    """Stable 12-char sha256 prefix of the resolved config (fail-open).

    Lets two runs be compared at a glance — identical fingerprint ⇒ identical
    resolved knobs (after defaults + migrations), regardless of YAML formatting
    or which file it came from.
    """
    try:
        if hasattr(config, "model_dump"):
            data = config.model_dump(mode="json")
        elif isinstance(config, dict):
            data = config
        else:
            return None
        blob = json.dumps(data, sort_keys=True, default=str)
    except Exception:  # any dump/serialise failure → no fingerprint, never raise
        return None
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def effective_batch_size(config: Any) -> dict[str, Any]:
    """Resolve the *effective* batch = per-device x grad-accum x world-size.

    The number that actually drives optimization, which the per-device
    ``data.batch_size`` alone hides — a frequent source of "why won't this
    reproduce" confusion across single-GPU vs DDP runs.

    ``world_size`` prefers the **live process group** over the environment, the
    same precedence :func:`parallel_provenance` uses and for the same reason:
    only ``torchrun`` exports ``WORLD_SIZE``. A ``torch.multiprocessing.spawn``
    worker receives its world size as an ``init_process_group`` *argument* and
    the environment is never set, so an env-only read reported
    ``world_size: 1`` — and therefore an ``effective`` batch N times too small
    — for every spawned run. Under torchrun inside a 1-task Slurm allocation
    ``SLURM_NTASKS`` is likewise 1 while the true world size is the GPU count.

    ``world_size_source`` names which of the two won, so a reader can tell a
    genuine single-process run from a distributed one whose group had not been
    initialised when the record was built (the banner renders before Stage B).
    """
    _data = getattr(config, "data", None)
    _opt = getattr(config, "optimization", None)
    per_device = getattr(getattr(_data, "loader", None), "batch_size", None)
    accum = getattr(getattr(_opt, "gradient", None), "accumulation_steps", 1) or 1
    world = 1
    world_source = None
    for key in _WORLD_SIZE_ENV:
        val = os.environ.get(key)
        if val and val.isdigit():
            world = int(val)
            world_source = key
            break
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            world = int(dist.get_world_size())
            world_source = "process_group"
    except Exception:  # pragma: no cover - provenance never blocks training
        _logger.debug("world size probe failed", exc_info=True)
    effective = None
    if isinstance(per_device, int):
        effective = per_device * int(accum) * world
    return {
        "per_device": per_device,
        "grad_accum": int(accum),
        "world_size": world,
        "world_size_source": world_source,
        "effective": effective,
    }


#: Fallback for a compile block that is not a pydantic model. Deliberately not
#: the source of truth -- see :func:`compile_provenance`.
_COMPILE_KEYS_WITHOUT_A_SCHEMA = ("enabled", "mode", "backend", "fullgraph", "dynamic")


def compile_provenance(config: Any, applied: Any = None) -> dict[str, Any]:
    """What compilation was *declared* and what the build *actually did*.

    Two independent facts, and the config can only speak to the first. Whether
    the model ended up wrapped, and at which point in the build, depends on the
    parallel strategy -- ``DDP(compile(m))`` and ``compile(DDP(m))`` are
    different runs with identical YAML, and the second is the one that engages
    DDPOptimizer. A record assembled from the config alone reads the same
    either way.

    ``applied`` is the per-model record the director produced
    (``TrainingEnvironment.compile``); ``{}`` means the arm asked for eager,
    and ``None`` means the build never got far enough to say, which goes in
    ``incomplete`` rather than being flattened into "not compiled".

    Args:
        config: the frozen settings.
        applied: name -> {stage, mode, backend, wrapped, ...}, or None.
    """
    compile_cfg = getattr(getattr(config, "optimization", None), "compile", None)
    # Read off the schema rather than a list kept here. A hand-written list is a
    # second declaration of the block's contents, and the one that is never
    # updated: `regional` and `allow_complex` were added to the schema, read and
    # validated, and left out of the record -- two thirds of non-negotiable 8,
    # where the missing third is what makes a finished run explicable.
    # The tuple survives only for a stub or partial config with no schema.
    fields = getattr(type(compile_cfg), "model_fields", None)
    keys = tuple(fields) if fields else _COMPILE_KEYS_WITHOUT_A_SCHEMA
    declared = {key: getattr(compile_cfg, key, None) for key in keys}
    record: dict[str, Any] = {"declared": declared}
    if applied is None:
        record["incomplete"] = ["applied"]
    else:
        record["applied"] = {name: dict(entry) for name, entry in dict(applied).items()}
    return record


def parallel_provenance(config: Any) -> dict[str, Any]:
    """What parallelism was *declared* and what the process group *actually is*.

    These are two independent facts and the run log conflated them, which is
    the whole reason this exists. ``parallel.strategy`` is an author decision
    in the YAML; the process group is a property of how the job was launched.
    Every combination occurs and they fail in different places:

    * declared ``deepspeed``, launched without torchrun — ``DeepSpeedStrategy.
      adopt`` raises out of ``_require_process_group``, at Stage B, long after
      the model and data are built.
    * declared ``none``, launched under torchrun — ``train-distributed``
      initialises a group that ``_resolve_parallel`` then declines to use, so
      ``world_size`` GPUs each train the identical un-sharded run.

    Reporting only one of the two cannot distinguish those, so both are
    stamped. Fail-open per module contract: a torch that cannot be imported
    degrades ``backend``/``initialized``, never the record.

    The *declared* device and node counts are stamped beside the applied ones
    because "provenance says 1 GPU but I asked for 4" was previously
    unanswerable from the artifact: ``parallel.num_devices`` reached
    ``resolved_config.json`` and stopped there, so a reader comparing the two
    had to open a second file — and ``pipelines/distributed.py`` *overwrites*
    ``num_devices`` from ``LOCAL_WORLD_SIZE``, meaning the resolved config does
    not always hold the authored value either. Side by side, a mismatch is the
    finding rather than a puzzle (non-negotiable 14's declared-vs-applied rule,
    applied to topology).
    """
    parallel = getattr(config, "parallel", None)
    record: dict[str, Any] = {
        # `None` (no block) and "none" (block present, opted out) are the same
        # runtime behaviour but not the same authoring intent — keep them apart.
        "strategy": getattr(parallel, "strategy", None),
        "declared": parallel is not None,
        # What the YAML ASKED for. `None` when no block was authored, so an
        # absent declaration never reads as the schema default of 1.
        "declared_num_devices": getattr(parallel, "num_devices", None),
        "declared_num_nodes": getattr(parallel, "num_nodes", None),
        "declared_backend": getattr(parallel, "backend", None),
        # The idle-GPU refusal is opt-out-able, so whether it was ARMED is part
        # of the run's shape: a 1-rank record on a 4-GPU allocation reads very
        # differently depending on whether someone acknowledged it.
        "declared_allow_idle_devices": getattr(parallel, "allow_idle_devices", None),
        "world_size": effective_batch_size(config).get("world_size", 1),
        "rank": _env_int(("RANK", "SLURM_PROCID")),
        "local_rank": _env_int(("LOCAL_RANK", "SLURM_LOCALID")),
        # torchrun sets these together; their presence is what distinguishes a
        # distributed launch from a plain `spectramr train` before init runs.
        "launcher": "torchrun" if os.environ.get("TORCHELASTIC_RUN_ID") else None,
        # Ranks per node and node count, so a `world_size: 8` record says
        # whether that was 8x1 or 2x4 -- which decides whether a NCCL stall is
        # an interconnect problem or a local one.
        "local_world_size": _env_int(("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE")),
        "node_count": _env_int(_NODE_COUNT_ENV),
        "initialized": None,
        "backend": None,
    }
    try:
        import torch.distributed as dist

        record["initialized"] = bool(dist.is_available() and dist.is_initialized())
        if record["initialized"]:
            record["backend"] = str(dist.get_backend())
            # Prefer the live group over the environment: a launcher can export
            # a WORLD_SIZE that init_process_group was never given.
            record["world_size"] = int(dist.get_world_size())
            record["rank"] = int(dist.get_rank())
    except Exception:  # pragma: no cover - provenance never blocks training
        _logger.debug("distributed state probe failed", exc_info=True)
    # torchrun exports LOCAL_WORLD_SIZE but never a node count, so outside Slurm
    # the only way to get it is to divide. Derived, not measured -- hence the
    # separate key: a reader must be able to tell the two apart.
    if record["node_count"] is None:
        local = record["local_world_size"]
        world = record["world_size"]
        if isinstance(local, int) and local > 0 and isinstance(world, int):
            record["node_count"] = max(1, world // local)
            record["node_count_derived"] = True
    return record


def rank_device_inventory() -> dict[str, Any]:
    """Which host and which physical GPU every rank actually got.

    :func:`gpu_resources` shells out to ``nvidia-smi`` on the **local** host, so
    on a multi-node run rank 0's record described one node and silently implied
    it was the whole job. This gathers one small record per rank instead, which
    is the only way the artifact can show that (say) two ranks landed on the
    same physical device — the failure that looks like a mysterious 2x slowdown
    rather than an error.

    **This is a collective.** ``all_gather_object`` must be reached by every
    rank or the job hangs, which inverts this module's usual fail-open posture:
    here a rank that bails out early is worse than a missing field. Three
    consequences, all deliberate:

    * The guard is ``is_available() and is_initialized()`` and nothing else —
      uniform across ranks by construction, since all of them cleared
      ``init_process_group``. It must **not** be nested inside a rank-0 branch
      or a fallible probe; the caller places it before the pipeline build for
      exactly that reason (both of the build's guards are rank-divergent).
    * The *local* record is built with per-field ``try``, so a broken
      ``nvidia-smi`` or an un-set CUDA device degrades one string and still
      reaches the collective.
    * If the gather itself raises, the failure is recorded in ``incomplete``
      rather than omitted (pitfall #16), because "no inventory" and "inventory
      says one node" must not look the same.

    NCCL note: ``all_gather_object`` moves its pickled buffer to
    ``torch.cuda.current_device()``, so every rank must have called
    ``set_device`` — ``pipelines/distributed.py`` does so *before*
    ``init_process_group`` and ``distributed/launcher.py`` right after. This
    mirrors ``RankUtility.broadcast_object``, which the training loop already
    relies on under the same invariant.

    Returns:
        ``{}`` on a single-process run (nothing to gather, and the existing
        ``gpu_resources`` record is already complete for one host), otherwise
        ``{"ranks": [...], "hosts": [...], "node_count": N}`` plus an
        ``incomplete`` list when anything could not be resolved.
    """
    try:
        import torch
        import torch.distributed as dist
    except Exception:  # pragma: no cover - no torch, nothing to inventory
        return {}
    if not (dist.is_available() and dist.is_initialized()):
        return {}

    local: dict[str, Any] = {}
    for field, probe in (
        ("hostname", socket.gethostname),
        ("rank", dist.get_rank),
        ("local_rank", lambda: _env_int(("LOCAL_RANK", "SLURM_LOCALID"))),
        ("device_index", torch.cuda.current_device),
        ("device_name", lambda: torch.cuda.get_device_name()),
        ("device_uuid", lambda: str(torch.cuda.get_device_properties(0).uuid)),
    ):
        try:
            local[field] = probe()
        except Exception:  # pragma: no cover - one bad probe must not deadlock
            local[field] = None

    world = int(dist.get_world_size())
    holder: list[Any] = [None] * world
    try:
        dist.all_gather_object(holder, local)
    except Exception:
        # Recorded, not swallowed: a reader must not mistake a failed gather for
        # a single-node job.
        _logger.warning(
            "per-rank device inventory could not be gathered across %d ranks; "
            "provenance will describe rank %s's host only",
            world,
            local.get("rank"),
            exc_info=True,
        )
        return {
            "ranks": [local],
            "incomplete": [
                f"all_gather_object failed across {world} ranks; only the local "
                "rank's device is recorded"
            ],
        }

    ranks = [r for r in holder if isinstance(r, dict)]
    hosts = sorted({r.get("hostname") for r in ranks if r.get("hostname")})
    record: dict[str, Any] = {
        "ranks": ranks,
        "hosts": hosts,
        "node_count": len(hosts) or None,
    }
    if len(ranks) != world:
        record["incomplete"] = [f"{len(ranks)} of {world} ranks returned a device record"]
    # Two ranks on one (host, device) is a real misconfiguration, not a note.
    # Ranks whose device could not be resolved are EXCLUDED rather than grouped:
    # `device_index` is `None` on every rank of a CPU run (and on any rank whose
    # probe failed), so keying on it would report the whole world as sharing one
    # device -- a false alarm precisely where the record is least informative.
    # Their unresolved state is already visible in the per-rank entries.
    seen: dict[tuple[Any, Any], int] = {}
    for r in ranks:
        if r.get("device_index") is None:
            continue
        key = (r.get("hostname"), r.get("device_index"))
        seen[key] = seen.get(key, 0) + 1
    collisions = {f"{h}:cuda:{d}": n for (h, d), n in seen.items() if n > 1}
    if collisions:
        record["device_collisions"] = collisions
        _logger.warning(
            "%d rank(s) share a physical device: %s. Each shared device runs "
            "both ranks' work serially, so the run is slower than a "
            "single-process one while reporting the full world size. Check "
            "that the launcher sets LOCAL_RANK and that torch.cuda.set_device "
            "is called per rank.",
            sum(collisions.values()),
            collisions,
        )
    return record


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "run"


def make_run_id(run_name: str, started_at: datetime, git_short: str | None) -> str:
    """Correlation id: ``<slug(name)>-<YYYYmmdd_HHMMSS>-<gitsha|nogit>``.

    Uniquely tags a run so its log lines, bundle dir, and scheduler job can be
    cross-referenced — essential when many arms run concurrently on the cluster.
    """
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    return f"{_slug(run_name)}-{stamp}-{git_short or 'nogit'}"


def collect_run_provenance(
    config: Any,
    *,
    seed: int | None,
    device: Any,
    run_name: str,
    started_at: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the run's provenance record (everything except model/data size,
    which the caller augments once those are built).

    Soft-fails component-by-component: a failure in one probe (git, env, …)
    degrades only that field, never the whole record.
    """
    started_at = started_at or datetime.now().astimezone()
    git = git_provenance()
    env = runtime_environment()
    record: dict[str, Any] = {
        "run_id": make_run_id(run_name, started_at, git.get("sha_short")),
        "run_name": run_name,
        "started_at": started_at.isoformat(timespec="seconds"),
        "seed": seed,
        "device": str(device) if device is not None else None,
        "config_sha256": config_fingerprint(config),
        "model_type": getattr(getattr(config, "model", None), "model_type", None),
        "training_mode": getattr(getattr(config, "training", None), "training_mode", None),
        "batch": effective_batch_size(config),
        "parallel": parallel_provenance(config),
        "git": git,
        "env": env,
        "slurm": {k: os.environ.get(k) for k in _SLURM_FIELDS if os.environ.get(k)},
        # Kept as its own block, not merged into `slurm`: these come from the
        # launcher, and a run can have one without the other (torchrun outside
        # Slurm, or a Slurm job launched with plain `spectramr train`). Raw
        # strings -- `SLURM_TASKS_PER_NODE` is "4(x2)" on a heterogeneous
        # allocation and several of these are host:port pairs.
        "launcher_env": {k: os.environ.get(k) for k in _LAUNCHER_FIELDS if os.environ.get(k)},
    }
    # How a metric reduces a batch and how an epoch reduces its batches. Not a
    # knob -- a convention, stamped because it RESTATES numbers the corpus has
    # already recorded (issue #1347): a run made before the per-sample reduction
    # and a run made after it are not comparable, and nothing else in the
    # artifact would say so. Constant by construction; imported rather than
    # spelled out here so there is one owner (non-negotiable 17).
    try:
        from spectramr.core.metrics.sample_aggregation import aggregation_provenance

        record["metric_aggregation"] = aggregation_provenance()
    except Exception:  # provenance never BLOCKS training; surfaced, not swallowed.
        _logger.debug("metric aggregation provenance capture failed", exc_info=True)
    # Resolved device CAPABILITIES, distinct from ``torch_runtime``'s hardware
    # inventory above: that lists what is visible, this records what the run
    # concluded about it (native bf16, Triton, the compile backend) and where
    # that conclusion came from. Without it "bf16 was declared" and "bf16 was
    # runnable here" are indistinguishable after the fact -- and on a V100 they
    # differ, because torch reports emulated bf16 as supported.
    try:
        from spectramr.core.device_capabilities import probe_device_capabilities

        record["device_capabilities"] = probe_device_capabilities(device).to_dict()
    except Exception:
        _logger.debug("device capability provenance capture failed", exc_info=True)
    # Out-of-tree plugin sources (pitfall #15c — an advertised knob's resolved
    # value is stamped into provenance so a run is traceable to the exact
    # SPECTRAMR_PLUGINS / entry-point / config plugin code it pulled in).
    try:
        from spectramr.plugins import resolve_plugin_provenance

        plugins = resolve_plugin_provenance()
        config_plugins = getattr(config, "plugins", None)
        if config_plugins is not None:
            plugins["config_paths"] = list(getattr(config_plugins, "paths", []) or [])
        if any(plugins.get(k) for k in ("env_var", "entry_points", "config_paths")):
            record["plugins"] = plugins
    except Exception:  # provenance never BLOCKS training, but the failure is
        # surfaced (debug) rather than silently swallowed (pitfall #15c).
        _logger.debug("plugin provenance capture failed", exc_info=True)
    # Resolved LR schedule (pitfall #15c). The declared ``scheduler:`` block used
    # to be discarded wholesale, so every run silently annealed on library
    # defaults and no artifact recorded it (issue #533). Stamp what was actually
    # resolved so a run's LR trajectory is reconstructible from provenance alone.
    try:
        from spectramr.infrastructure.training.scheduler_resolution import (
            resolve_scheduler_spec,
        )

        opt_cfg = getattr(config, "optimization", None)
        if opt_cfg is not None:
            spec = resolve_scheduler_spec(
                opt_cfg,
                max_iterations=getattr(getattr(config, "training", None), "max_iterations", None),
            )
            record["lr_schedule"] = (
                spec.as_provenance() if spec is not None else {"scheduler": None}
            )
    except Exception:  # provenance never BLOCKS training; surfaced, not swallowed.
        _logger.debug("scheduler provenance capture failed", exc_info=True)
    # Cold-diffusion supervision regime (pitfall #15c). Both knobs change what the
    # run actually optimises — which tensor the forward process degrades, and what
    # the magnitude ceiling is referenced to — so a run's numbers are only
    # comparable to another run with the same pair (issue #536).
    try:
        diffusion = getattr(getattr(config, "training", None), "diffusion", None)
        model_kwargs = getattr(getattr(config, "model", None), "model_kwargs", None) or {}
        if diffusion is not None:
            record["cold_diffusion"] = {
                "degradation_source": getattr(diffusion, "degradation_source", None),
                "clip_reference": model_kwargs.get("output_kspace_clip_reference", "global_max"),
            }
    except Exception:
        _logger.debug("cold-diffusion provenance capture failed", exc_info=True)
    # The resolved loss objective (non-negotiable 8's third obligation). A weight
    # is declarable on two surfaces and canonicalised across aliases, so the YAML
    # does NOT say what the run optimised: `lambda_mse` files under `l2`, an
    # equal-weight duplicate reconciles into one entry naming both paths, and a
    # warm-up-gated term resolves to 0.0 for its first N iterations. This stamps
    # the table the run actually trained against, per term, with the declaration
    # each weight came from.
    #
    # `LossWeightTable.provenance()` has existed since the weight-table work and
    # its docstring called itself "the stamp written into the run record" while
    # having no caller outside a unit test -- the advertised-but-unwired shape
    # pitfall #16 is about, in the mechanism meant to enforce pitfall #15.
    try:
        from spectramr.models.losses.weights import build_loss_weight_table

        losses_cfg = getattr(config, "losses", None)
        if losses_cfg is not None:
            record["losses"] = build_loss_weight_table(losses_cfg).provenance()
    except Exception:  # provenance never BLOCKS training; surfaced, not swallowed.
        _logger.debug("loss-weight provenance capture failed", exc_info=True)
    # Launcher resources (pitfall #15c — when started via ``spectramr launch`` the
    # resolved ResourceSpec is handed to this child via SPECTRAMR_LAUNCH_* env, so
    # the run is traceable to the backend + resources it actually ran under). A
    # no-op for a plain ``spectramr train`` (env absent → empty dict).
    try:
        from spectramr.infrastructure.execution.backends import resolve_launch_provenance

        launch = resolve_launch_provenance()
        if launch:
            record["launch"] = launch
    except Exception:  # never BLOCKS training; failure surfaced, not swallowed.
        _logger.debug("launch provenance capture failed", exc_info=True)
    # Run topology (issue: multi-GPU slower than single-GPU). Records the shape
    # the run ACTUALLY had -- ranks, nodes, per-rank CPU share -- plus a
    # ``sources`` map naming the env var each field came from, so a surprising
    # number is traceable rather than merely surprising.
    #
    # Deliberately a top-level key and NOT an extension of ``parallel_provenance``,
    # for two reasons that both hold today:
    #   * It is run SHAPE, not parallelism. It feeds ``workers`` immediately
    #     below -- a dataloader concern -- and is just as meaningful on a
    #     single-process run, where ``parallel`` is thin or absent.
    #   * ``pipelines/train.py`` merges the plugin's runtime record over
    #     ``provenance["parallel"]`` and lets the PLUGIN win collisions, so a key
    #     nested there can be silently overwritten by a strategy plugin that
    #     happens to spell one of its own the same way. Top level is outside
    #     that merge entirely.
    # Do not restore the older justification that ``train.py`` REPLACES the
    # block wholesale -- it merges (see its own "MERGE, do not replace"), and
    # that claim was already false when it was written.
    try:
        from spectramr.core.topology import resolve_run_topology

        topology = resolve_run_topology()
        record["topology"] = topology.to_dict()
        record["workers"] = worker_provenance(config, topology)
    except Exception:
        _logger.debug("topology provenance capture failed", exc_info=True)
    return record


def worker_provenance(config: Any, topology: Any) -> dict[str, Any]:
    """Declared vs actually-running dataloader worker counts, per role.

    The clamp added with ``RunTopology`` lowers ``num_workers`` to this rank's
    share of the node's cores, so from this release the number a YAML declares
    and the number that runs can differ -- and nothing on disk said so. The
    loaders themselves cannot answer it either: on the ``tio.Queue`` path the
    outer ``SubjectsLoader`` is hardcoded to ``num_workers=0`` and the real
    count lives on the Queue, so reading the DataLoader back would report 0 for
    the very path this arm uses.

    Rather than thread a record up through four transient builders, this
    re-derives the decision by calling the SAME pure ``clamp_worker_count`` on
    the SAME two inputs the director reads -- the declared count and the
    topology. Both are pure config reads, so the answer is identical to what
    ran by construction, not by bookkeeping that can drift.

    Describes the TRAINING path (``data_builder`` -> ``build_dataloaders``).
    ``pipelines/make.py`` and ``shared/utils/data_utils.py`` build their own
    directors and may pass a different declared count; those are not covered.
    """
    from spectramr.core.worker_policy import clamp_worker_count

    out: dict[str, Any] = {}
    data_cfg = getattr(config, "data", None)
    if data_cfg is not None:
        declared = getattr(getattr(data_cfg, "loader", None), "num_workers", None)
        if declared is not None:
            out["train"] = clamp_worker_count(
                int(declared), topology, role="train", log=False
            ).to_dict()
    val_cfg = getattr(config, "validation", None)
    if val_cfg is not None:
        declared = getattr(getattr(val_cfg, "loader", None), "num_workers", None)
        if declared is not None:
            out["val"] = clamp_worker_count(
                int(declared), topology, role="val", log=False
            ).to_dict()
    return out


def hardware_summary(prov: dict[str, Any]) -> dict[str, Any]:
    """Flatten the node record into the compact census for ``run_summary.json``.

    The footer reports ``iterations_per_sec``, which is meaningless without the
    hardware behind it — 3 it/s on one RTX 3060 and 3 it/s on eight A100s are
    opposite results. Returns ``{}`` when nothing is known, so the caller can
    omit the key rather than stamp a dict of nulls.
    """
    node = (prov.get("env") or {}).get("node") or {}
    gpu = node.get("gpu") or {}
    cpu = node.get("cpu") or {}
    mem = node.get("memory") or {}
    summary = {
        "gpu_count": gpu.get("count"),
        "gpu_types": gpu.get("types") or None,
        "gpu_node_count": gpu.get("node_count"),
        "gpu_driver_version": gpu.get("driver_version"),
        "gpu_total_mem_gb": gpu.get("total_mem_gb"),
        "cpu_model": cpu.get("model"),
        "cpu_cores_usable": cpu.get("usable_cores"),
        "cpu_cores_total": cpu.get("logical_cores"),
        "memory_usable_gb": mem.get("usable_gb"),
        "memory_total_gb": mem.get("total_gb"),
    }
    return summary if any(v is not None for v in summary.values()) else {}


def _device_census(devices: list[dict[str, Any]]) -> str:
    """``2x A100-SXM4-40GB(40.0GB)`` — grouped, so an 8-GPU node stays readable."""
    census: dict[str, int] = {}
    sizes: dict[str, Any] = {}
    for dev in devices:
        name = dev.get("name") or "unknown"
        census[name] = census.get(name, 0) + 1
        sizes.setdefault(name, dev.get("total_mem_gb"))
    return ", ".join(
        f"{n}x {name}({sizes[name]}GB)" if n > 1 else f"{name}({sizes[name]}GB)"
        for name, n in census.items()
    )


def _format_gpu_line(gpu: dict[str, Any]) -> str | None:
    """Visible-vs-present GPU summary; ``None`` when nothing is known."""
    count = gpu.get("count")
    node_count = gpu.get("node_count")
    if not count and not node_count:
        return None
    parts = [f"{count or 0} visible"]
    if node_count:
        parts[0] += f" / {node_count} on node"
    if gpu.get("allocated_count"):
        parts.append(f"{gpu['allocated_count']} allocated")
    if gpu.get("driver_version"):
        parts.append(f"driver {gpu['driver_version']}")
    visible = gpu.get("visible_devices")
    if visible is not None:
        parts.append(f"CUDA_VISIBLE_DEVICES={visible or '<empty>'}")
    if not count and node_count:
        # The forensically important case under the accelerated-run contract:
        # the hardware is there, torch just cannot reach it.
        parts.append("NOT USABLE BY TORCH")
    return " · ".join(parts)


def _format_node_line(cpu: dict[str, Any], mem: dict[str, Any]) -> str | None:
    """``8/128 cores · 64.0/1007.5 GB RAM · <cpu model>`` (usable/total)."""
    parts: list[str] = []
    if cpu.get("logical_cores"):
        usable = cpu.get("usable_cores") or cpu["logical_cores"]
        parts.append(f"{usable}/{cpu['logical_cores']} cores")
    if mem.get("total_gb"):
        usable_gb = mem.get("usable_gb") or mem["total_gb"]
        parts.append(f"{usable_gb}/{mem['total_gb']} GB RAM")
    if cpu.get("model"):
        parts.append(str(cpu["model"]))
    return " · ".join(parts) or None


def _ranks_on_this_node(par: dict[str, Any]) -> int | None:
    """Ranks launched on THIS node, or ``None`` when the shape cannot say.

    ``declared_num_devices`` is a per-node count on both of its origins:
    ``parallel.num_devices`` is authored beside ``num_nodes``, and
    ``pipelines/distributed.py`` overwrites it from ``LOCAL_WORLD_SIZE``. The
    world size is global. Comparing the two told a correct 2x4 run that half its
    hardware was idle, which is the defect this resolves (#1276).

    Ambiguity passes, the same discipline as the ``idle_device_refusal``
    predicate: a banner warning that is wrong once is distrusted forever, so a
    shape this cannot resolve returns ``None`` and the detector stays quiet.
    Dividing ``world`` by a node count would be a guess about whether the ranks
    were spread evenly, so it is not attempted.
    """
    local = par.get("local_world_size")
    if isinstance(local, int) and local > 0:
        return local
    nodes = par.get("node_count")
    if isinstance(nodes, int) and nodes > 1:
        return None
    # Single node (or no topology at all): every rank in the world is on it.
    # This is the case that keeps the detector's one remaining real job alive --
    # a plain `spectramr train` that declares 4 devices and runs one process,
    # where `num_devices` is still the AUTHORED value because no launcher
    # overwrote it.
    world = par.get("world_size", 1) or 1
    return world if isinstance(world, int) else None


def format_parallel_line(par: dict[str, Any]) -> str:
    """Render :func:`parallel_provenance` as one scannable line.

    Also names the two declared-vs-launched mismatches inline, because both are
    silent at this point in the run: the reader is looking at a log that has so
    far said nothing at all about parallelism, so a bare pair of facts leaves
    them to spot the contradiction themselves.
    """
    strategy = par.get("strategy")
    # Spell out an absent block. It and `strategy: none` reach the same no-op,
    # but a reader who declared one and typo'd the key would otherwise see a
    # plausible-looking line reporting the opt-out they did not write.
    label = "none (no parallel: block)" if strategy is None else str(strategy)

    world = par.get("world_size", 1) or 1
    parts = [label, f"world={world} rank={par.get('rank') or 0}"]

    # Topology, when there is any: `world=8` alone does not say 8x1 or 2x4.
    nodes = par.get("node_count")
    local_world = par.get("local_world_size")
    if isinstance(nodes, int) and nodes > 1:
        suffix = " (derived)" if par.get("node_count_derived") else ""
        parts.append(f"nodes={nodes}x{local_world or '?'}{suffix}")

    if par.get("initialized"):
        parts.append(f"group={par.get('backend')}")
    elif par.get("launcher") or world > 1:
        # Launched distributed but no group yet. Normal when the banner renders
        # before init; not normal afterwards, hence the explicit words.
        parts.append("group=not-initialized")
    else:
        parts.append("single-process")

    if strategy not in (None, "none") and world == 1:
        # NOT gated on the group being absent. That conjunct restricted this to
        # the one state that already raises -- `_require_process_group` refuses
        # a declared strategy with no group inside `adopt` -- and so excluded
        # the state that actually wastes hardware: an INITIALISED one-rank
        # group. The incident log proves the exclusion; it printed `group=nccl`.
        #
        # Kept at banner level rather than left to that raise, because the raise
        # lands at Stage B, after the model and the data are built, while this
        # line renders before anything is.
        #
        # A deliberate single-rank DeepSpeed run -- ZeRO-3 or CPU offload on one
        # GPU is a real memory strategy -- now gets a line it does not need.
        # That is banner noise, not a false alarm: the fact stated is true. It
        # is the cheaper error than a detector that cannot fire at all.
        state = "an initialised 1-rank group" if par.get("initialized") else "a single process"
        parts.append(f"[!] {strategy} declared on {state}")
    elif strategy in (None, "none") and world > 1:
        parts.append(f"[!] {world} ranks each training the SAME un-sharded run")

    # "Provenance says 1 GPU but I asked for 4" -- the whole point of stamping
    # the declared counts. Said inline, because a reader scanning the banner is
    # not going to diff two numbers in a JSON file they have not opened yet.
    declared_devices = par.get("declared_num_devices")
    ranks_on_node = _ranks_on_this_node(par)
    if (
        isinstance(declared_devices, int)
        and declared_devices > 1
        and isinstance(ranks_on_node, int)
        and declared_devices != ranks_on_node
    ):
        # Direction matters: only a declaration that EXCEEDS the ranks leaves
        # devices idle. The reverse is a different finding and saying "the extra
        # devices are NOT being used" about it would be simply false.
        idle = ": the extra devices are NOT being used" if declared_devices > ranks_on_node else ""
        parts.append(
            f"[!] declared num_devices={declared_devices} but "
            f"{ranks_on_node} rank(s) on this node{idle}"
        )
    declared_nodes = par.get("declared_num_nodes")
    if isinstance(declared_nodes, int) and isinstance(nodes, int) and declared_nodes != nodes:
        parts.append(f"[!] declared num_nodes={declared_nodes} but ran on {nodes}")
    return " · ".join(parts)


def format_runtime_knobs(config: Any) -> str:
    """The handful of knobs that change what a run *costs*, on one line.

    Deliberately not "everything": the resolved config is already written to
    ``resolved_config.json`` in full, so this is the subset a reader checks
    when a run is slower, larger or shorter than they expected — and each of
    these is routinely set by ``-O`` at launch, where a typo is otherwise
    invisible.
    """
    opt = getattr(config, "optimization", None)
    grad = getattr(opt, "gradient", None)
    prec = getattr(opt, "precision", None)
    loader = getattr(getattr(config, "data", None), "loader", None)
    training = getattr(config, "training", None)
    schedule = getattr(getattr(config, "validation", None), "schedule", None)
    # `undersampling:` is the only spelling that reaches a resolved settings
    # object -- `acceleration:` is folded onto it by RENAMES at load time, so a
    # second lookup here would be a dead branch, not a fallback.
    accel = getattr(config, "undersampling", None)
    # `validation.sampling.ensemble_samples` multiplies the cost of every
    # validation by N; absent on the `SimpleNamespace` configs tests hand in.
    sampling = getattr(getattr(config, "validation", None), "sampling", None)
    ensemble = getattr(sampling, "ensemble_samples", None)

    # `precision.enabled` gates the dtype: a block reading dtype=bfloat16 with
    # enabled=False runs fp32, so reporting the dtype alone would be a lie.
    if getattr(prec, "enabled", False):
        amp = f"amp={getattr(prec, 'dtype', None)}"
    else:
        amp = "amp=off(fp32)"

    return " · ".join(
        [
            f"grad_ckpt={getattr(grad, 'enable_checkpointing', None)}",
            amp,
            f"accum={getattr(grad, 'accumulation_steps', None)}",
            f"workers={getattr(loader, 'num_workers', None)}",
            f"max_iter={getattr(training, 'max_iterations', None)}",
            f"val_every={getattr(schedule, 'interval_steps', None)}",
            # A DECLARED curriculum that the short-run bypass suppresses is
            # invisible in every other artifact a run produces (#1296), and
            # this line is emitted before `LoggingService.setup` clamps the
            # level -- so it survives on the `level: warning` arms that declare
            # one. `describe()` distinguishes "off" from "declared-but-off".
            resolve_curriculum_state(config).describe(),
        ]
        # DECLARED intent, pre-clamp. The knob fully determines the resolved
        # training/sampling timestep floor (0 when on, otherwise a pure function
        # of the ladder's first rung), so this one token is enough to tell a
        # reader whether the fully-sampled rung is in the run. Omitted entirely
        # when off, which is every non-diffusion arm: an always-present
        # `identity_rung=off` would be noise on 600+ arms. Deliberately NOT
        # logged at WARNING -- `audit` is --strict and warnings exit 2
        # (non-negotiable 4), so an informational line there would fail every
        # smoke run of the arm that opts in.
        + (["identity_rung=on"] if getattr(accel, "train_identity_rung", False) else [])
        # Same policy: only the arms that opt in carry the token. N reverse
        # samples per validation input is a cost multiplier a reader of a slow
        # run needs to see, and it is invisible in every per-step artifact.
        + (
            [f"val_ensemble={int(ensemble)}"]
            if isinstance(ensemble, int) and not isinstance(ensemble, bool) and ensemble > 1
            else []
        )
    )


def log_startup_summary(config: Any, logger: logging.Logger | None = None) -> None:
    """Emit parallelism + cost knobs BEFORE the logging level is configured.

    The full :func:`log_provenance` banner is richer but cannot move here — it
    reports model and dataset sizes, which do not exist until the environment
    is built, and by then ``LoggingService.setup`` has pushed
    ``logging.sinks.level`` onto the root logger, every existing logger and
    every handler. An arm that sets ``level: warning`` (an entirely reasonable
    choice for a long run, and what the attention_shootout arms set) therefore
    discards that banner along with the per-step narration it was aimed at.

    Those are different categories. Per-step narration is noise a long run can
    do without; *what this run is* is not, and a reader who cannot tell whether
    DeepSpeed engaged has no way to interpret anything downstream of it. So the
    identity facts are emitted here instead, in the pre-setup window, rather
    than by overriding the author's level choice.

    Fail-open per module contract: logging never blocks training.
    """
    log = logger or _logger
    try:
        log.info("parallel   : %s", format_parallel_line(parallel_provenance(config)))
        log.info("knobs      : %s", format_runtime_knobs(config))
    except Exception:  # pragma: no cover - logging never blocks training
        _logger.debug("startup summary failed", exc_info=True)


def _format_split_counts(split: str, counts: Any) -> str:
    """Render one split's counts for the banner, each number unit-labelled.

    ``counts`` is a unit -> number mapping since the data block gained explicit
    units; a bare int is the pre-units shape, which always meant batches, so old
    records stay readable. Nested values (``per_contrast``) are left to the JSON
    to keep the line scannable, but an ``incomplete`` marker is always shown --
    a count that could not be taken must not read as a count of zero.
    """
    if not isinstance(counts, dict):
        return f"{split}[batches={counts}]"
    body = ", ".join(
        f"{unit}={value}"
        for unit, value in counts.items()
        if isinstance(value, int) and not isinstance(value, bool)
    )
    if counts.get("incomplete"):
        body = f"{body}, incomplete" if body else "incomplete"
    return f"{split}[{body}]" if body else f"{split}[?]"


def format_provenance_lines(prov: dict[str, Any]) -> list[str]:
    """Render the provenance record as aligned, human-scannable log lines."""
    git = prov.get("git", {}) or {}
    env = prov.get("env", {}) or {}
    torch_i = env.get("torch", {}) or {}
    node = env.get("node", {}) or {}
    batch = prov.get("batch", {}) or {}

    if git.get("available"):
        dirty = " (DIRTY)" if git.get("dirty") else ""
        git_line = f"{git.get('sha_short')} @ {git.get('branch')}{dirty}"
    else:
        git_line = "unavailable"

    if torch_i.get("available"):
        devs = torch_i.get("devices") or []
        dev_line = _device_census(devs) if devs else "CPU-only"
        torch_line = f"{torch_i.get('version')} · {dev_line}"
    else:
        torch_line = "unavailable"

    model = prov.get("model", {}) or {}
    if model.get("total") is not None:
        params = f"{model['total'] / 1e6:.2f}M ({model.get('size_mb')}MB)"
        if model.get("frozen"):
            params += f", {model['frozen'] / 1e6:.2f}M frozen"
    else:
        params = "n/a"

    data = prov.get("data", {}) or {}
    data_line = (
        " / ".join(
            _format_split_counts(split, counts)
            for split, counts in data.items()
            if counts is not None
        )
        or "n/a"
    )

    eff = batch.get("effective")
    batch_line = (
        f"{batch.get('per_device')} x {batch.get('grad_accum')}accum"
        f" x {batch.get('world_size')}gpu = {eff}"
        if eff is not None
        else "n/a"
    )

    lines = [
        f"run_id     : {prov.get('run_id')}",
        f"parallel   : {format_parallel_line(prov.get('parallel', {}) or {})}",
        f"git        : {git_line}",
        f"host       : {env.get('hostname')} (pid {env.get('pid')}, {env.get('user')})",
        f"torch      : {torch_line}",
        f"python     : {env.get('python')}",
        f"model      : {prov.get('model_type')} · {params} params",
        f"data       : {data_line}",
        f"batch(eff) : {batch_line}",
        f"config_sha : {prov.get('config_sha256')}",
        f"seed       : {prov.get('seed')}  device: {prov.get('device')}",
    ]
    gpu_line = _format_gpu_line(node.get("gpu", {}) or {})
    if gpu_line:
        lines.insert(4, f"gpu        : {gpu_line}")
    node_line = _format_node_line(node.get("cpu", {}) or {}, node.get("memory", {}) or {})
    if node_line:
        lines.insert(5 if gpu_line else 4, f"node       : {node_line}")
    if prov.get("slurm"):
        job = prov["slurm"].get("SLURM_JOB_ID")
        nodes = prov["slurm"].get("SLURM_NODELIST")
        count = prov["slurm"].get("SLURM_NNODES") or prov["slurm"].get("SLURM_JOB_NUM_NODES")
        tasks = prov["slurm"].get("SLURM_TASKS_PER_NODE")
        extra = f" n={count}" if count else ""
        extra += f" tasks/node={tasks}" if tasks else ""
        lines.append(f"slurm      : job={job} nodes={nodes}{extra}")
    inv = prov.get("rank_devices") or {}
    if inv.get("ranks"):
        lines.append(f"ranks      : {_format_rank_inventory(inv)}")
    log = prov.get("logging") or {}
    if log.get("resolved_path"):
        note = log.get("resolved_path")
        if log.get("relocated_from"):
            note += f"  [!] RELOCATED from {log['relocated_from']} (temp dir; wiped at teardown)"
        lines.append(f"log        : {note}")
    return lines


def _format_rank_inventory(inv: dict[str, Any]) -> str:
    """One line per host, listing which device each of its ranks holds.

    Collapsed per host rather than per rank: a 32-rank job would otherwise push
    32 lines into a banner whose whole value is being scannable, and the fact a
    reader needs is "did every rank get its own GPU on the host I expected".
    """
    by_host: dict[str, list[str]] = {}
    for r in inv.get("ranks") or []:
        host = str(r.get("hostname") or "?")
        idx = r.get("device_index")
        # "cuda:None" is worse than saying nothing: it looks like a device.
        where = f"cuda:{idx}" if idx is not None else "device=?"
        by_host.setdefault(host, []).append(f"r{r.get('rank')}→{where}")
    rendered = "; ".join(f"{h}[{', '.join(v)}]" for h, v in sorted(by_host.items()))
    if inv.get("device_collisions"):
        rendered += f"  [!] shared devices: {inv['device_collisions']}"
    for note in inv.get("incomplete") or []:
        rendered += f"  [!] {note}"
    return rendered


def log_provenance(prov: dict[str, Any], logger: logging.Logger | None = None) -> None:
    """Emit the provenance banner at INFO (fail-open)."""
    log = logger or _logger
    try:
        log.info("─── run provenance ───")
        for line in format_provenance_lines(prov):
            log.info(line)
    except Exception:  # pragma: no cover - logging never blocks training
        pass


__all__ = [
    "collect_run_provenance",
    "compile_provenance",
    "config_fingerprint",
    "count_parameters",
    "cpu_resources",
    "describe_dataloader",
    "effective_batch_size",
    "format_parallel_line",
    "format_provenance_lines",
    "format_runtime_knobs",
    "git_provenance",
    "gpu_resources",
    "hardware_summary",
    "log_provenance",
    "log_startup_summary",
    "make_run_id",
    "memory_resources",
    "node_resources",
    "parallel_provenance",
    "rank_device_inventory",
    "runtime_environment",
    "torch_runtime",
    "worker_provenance",
]
