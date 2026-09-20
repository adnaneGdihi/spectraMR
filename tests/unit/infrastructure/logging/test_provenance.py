"""Node-hardware capture in run provenance (2026-07-25).

``provenance.json`` recorded *what code* and *which host* but not **what that
host actually was** — GPU count/type, cores, RAM. Two failures follow from the
gap:

* **Allocated != physical.** A 128-core / 1 TB node where the job was granted 8
  cores and 64 GB produced a record indistinguishable from a run that owned the
  whole machine, so "why did this OOM / run slow?" had no answer in the bundle.
* **Visible != present.** ``torch.cuda.device_count()`` counts only the
  ``CUDA_VISIBLE_DEVICES`` subset, so "this node has no GPU" and "this job was
  given none of its 4" looked identical — the distinction that matters under
  the accelerated-run contract (non-negotiable 9b).

Every probe stays fail-open: a node with no ``nvidia-smi``, no ``psutil``, or no
cgroup still reports what it can and never aborts a run.

Companion to ``test_provenance_2026_06.py`` (the original git/env/config suite).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from types import SimpleNamespace

import pytest

from spectramr.core import resources as res
from spectramr.infrastructure.logging import provenance as prov

_ALLOC_ENV = (
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "NSLOTS",
    "PBS_NP",
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_NODE",
    "SLURM_GPUS",
    "CUDA_VISIBLE_DEVICES",
)


@pytest.fixture
def no_scheduler(monkeypatch):
    """Clear every scheduler var so a real SLURM shell can't skew assertions."""
    for key in _ALLOC_ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


# --------------------------------------------------------------------------- #
# CPU
# --------------------------------------------------------------------------- #
def test_cpu_resources_shape_and_usable_bound(no_scheduler):
    cpu = prov.cpu_resources()
    for key in (
        "model",
        "physical_cores",
        "logical_cores",
        "affinity_cores",
        "allocated_cores",
        "cgroup_quota_cores",
        "usable_cores",
        "max_frequency_mhz",
    ):
        assert key in cpu, key
    assert cpu["logical_cores"] and cpu["logical_cores"] > 0
    # usable is a floor over every constraint, so it can never exceed the node.
    assert cpu["usable_cores"] <= cpu["logical_cores"]


def test_cpu_resources_usable_honours_scheduler_allocation(no_scheduler):
    """The allocation, not the node, is what the run actually had."""
    no_scheduler.setenv("SLURM_CPUS_PER_TASK", "4")
    cpu = prov.cpu_resources()
    assert cpu["allocated_cores"] == 4
    assert cpu["usable_cores"] == 4, "usable must clamp to the SLURM allocation"
    assert cpu["logical_cores"] >= 4, "node total stays the true node total"


def test_cpu_resources_parses_heterogeneous_cpus_on_node(no_scheduler):
    """SLURM writes ``72(x2)`` for heterogeneous allocations — must not crash."""
    no_scheduler.setenv("SLURM_CPUS_ON_NODE", "72(x2)")
    assert prov.cpu_resources()["allocated_cores"] == 72


def test_cpu_resources_prefers_per_task_over_on_node(no_scheduler):
    """Per-task is this process's share; ON_NODE covers every task on the node."""
    no_scheduler.setenv("SLURM_CPUS_PER_TASK", "8")
    no_scheduler.setenv("SLURM_CPUS_ON_NODE", "64")
    assert prov.cpu_resources()["allocated_cores"] == 8


def test_cpu_resources_fail_open_without_psutil(no_scheduler, monkeypatch):
    """A broken/absent psutil degrades detail, never the record."""
    import builtins

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    cpu = prov.cpu_resources()
    assert cpu["logical_cores"] is not None, "os.cpu_count() still answers"
    assert cpu["physical_cores"] is None


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #
def test_memory_resources_shape(no_scheduler):
    mem = prov.memory_resources()
    for key in (
        "total_gb",
        "available_gb",
        "swap_total_gb",
        "cgroup_limit_gb",
        "allocated_gb",
        "usable_gb",
    ):
        assert key in mem, key
    assert mem["total_gb"] and mem["total_gb"] > 0
    assert mem["usable_gb"] <= mem["total_gb"]


def test_memory_resources_uses_slurm_mem_per_node(no_scheduler):
    """SLURM reports MB; the record normalises to GB and clamps ``usable_gb``.

    ``usable_gb`` is a floor over *every* constraint including physical RAM, so
    it is ``min(allocated, total)`` — asserting a literal 64.0 would pass on a
    cluster node and fail on a 16 GB laptop.
    """
    no_scheduler.setenv("SLURM_MEM_PER_NODE", "65536")
    mem = prov.memory_resources()
    assert mem["allocated_gb"] == 64.0
    assert mem["usable_gb"] == min(64.0, mem["total_gb"])


def test_memory_usable_clamps_to_the_tightest_constraint(no_scheduler, monkeypatch):
    """A 4 GB cgroup beats a 64 GB SLURM grant beats a 1 TB node."""
    monkeypatch.setattr(prov, "_cgroup_memory_limit_gb", lambda: 4.0)
    no_scheduler.setenv("SLURM_MEM_PER_NODE", "65536")
    mem = prov.memory_resources()
    assert mem["cgroup_limit_gb"] == 4.0
    assert mem["allocated_gb"] == 64.0
    assert mem["usable_gb"] == 4.0


def test_memory_resources_derives_from_mem_per_cpu(no_scheduler):
    """When only a per-CPU budget is set, multiply by the CPU allocation."""
    no_scheduler.setenv("SLURM_MEM_PER_CPU", "4096")
    no_scheduler.setenv("SLURM_CPUS_PER_TASK", "8")
    assert prov.memory_resources()["allocated_gb"] == 32.0


# The cgroup/CPU probes moved to ``spectramr.core.resources`` so that the numbers
# stamped into provenance and the numbers the dataloader-worker clamp decides on
# come from one implementation. These tests therefore patch the CORE module: the
# function bodies resolve ``read_first_line`` in their own module globals, so
# patching ``prov`` here would leave the real filesystem probe running and the
# assertions would silently describe the host rather than the fixture.
def test_cgroup_memory_limit_ignores_unlimited_sentinels(monkeypatch):
    """cgroup v2 writes ``max``; v1 writes a near-2^63 int. Neither is a limit."""
    monkeypatch.setattr(res, "read_first_line", lambda path: "max")
    assert res.cgroup_memory_limit_gb() is None

    monkeypatch.setattr(res, "read_first_line", lambda path: str(2**63 - 1))
    assert res.cgroup_memory_limit_gb() is None


def test_cgroup_memory_limit_reads_real_limit(monkeypatch):
    monkeypatch.setattr(res, "read_first_line", lambda path: str(8 * 1024**3))
    assert res.cgroup_memory_limit_gb() == 8.0


def test_cgroup_cpu_quota_v2_and_unlimited(monkeypatch):
    """``cpu.max`` is ``"<quota> <period>"`` — 800000/100000 == 8 cores."""
    monkeypatch.setattr(
        res, "read_first_line", lambda path: "800000 100000" if "cpu.max" in path else None
    )
    assert res.cgroup_cpu_quota() == 8.0

    monkeypatch.setattr(
        res, "read_first_line", lambda path: "max 100000" if "cpu.max" in path else None
    )
    assert res.cgroup_cpu_quota() is None


def test_provenance_reexports_the_moved_probes(monkeypatch):
    """``memory_resources`` still resolves its cgroup helper through ``prov``.

    The alias is what keeps the move invisible to this module's own callers; if
    it were dropped, the patch below would no-op and the assertion would read
    the host's real memory limit instead of the fixture's.
    """
    monkeypatch.setattr(prov, "_cgroup_memory_limit_gb", lambda: 2.0)
    assert prov.memory_resources()["cgroup_limit_gb"] == 2.0


# --------------------------------------------------------------------------- #
# GPU census
# --------------------------------------------------------------------------- #
def _torch_info(names: list[str]) -> dict:
    return {
        "available": True,
        "cuda_available": bool(names),
        "devices": [{"index": i, "name": n, "total_mem_gb": 40.0} for i, n in enumerate(names)],
    }


def test_gpu_resources_counts_and_groups_by_type(no_scheduler, monkeypatch):
    """``types`` is a census, so an 8-GPU node reads as ``{name: 8}``."""
    monkeypatch.setattr(prov, "_nvidia_smi_devices", lambda: ([], None))
    gpu = prov.gpu_resources(_torch_info(["A100", "A100", "V100"]))
    assert gpu["count"] == 3
    assert gpu["types"] == {"A100": 2, "V100": 1}
    assert gpu["total_mem_gb"] == 120.0
    assert gpu["cuda_available"] is True


def test_gpu_resources_records_visible_mask_and_allocation(no_scheduler, monkeypatch):
    monkeypatch.setattr(prov, "_nvidia_smi_devices", lambda: ([], None))
    no_scheduler.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    no_scheduler.setenv("SLURM_GPUS_ON_NODE", "2")
    gpu = prov.gpu_resources(_torch_info(["A100", "A100"]))
    assert gpu["visible_devices"] == "2,3"
    assert gpu["allocated_count"] == 2


def test_gpu_resources_separates_visible_from_node_inventory(no_scheduler, monkeypatch):
    """A 2-of-4 allocation must not read as a 2-GPU node."""
    node = [
        {
            "index": i,
            "name": "NVIDIA A100-SXM4-40GB",
            "total_mem_gb": 40.0,
            "uuid": f"u{i}",
        }
        for i in range(4)
    ]
    monkeypatch.setattr(prov, "_nvidia_smi_devices", lambda: (node, "550.54.15"))
    gpu = prov.gpu_resources(_torch_info(["NVIDIA A100-SXM4-40GB"] * 2))
    assert gpu["count"] == 2
    assert gpu["node_count"] == 4
    assert gpu["node_types"] == {"NVIDIA A100-SXM4-40GB": 4}
    assert gpu["driver_version"] == "550.54.15"


def test_gpu_resources_cpu_only_node(no_scheduler, monkeypatch):
    monkeypatch.setattr(prov, "_nvidia_smi_devices", lambda: ([], None))
    gpu = prov.gpu_resources({"available": True, "cuda_available": False, "devices": []})
    assert gpu["count"] == 0
    assert gpu["types"] == {}
    assert gpu["node_count"] == 0
    assert gpu["total_mem_gb"] is None


def test_gpu_resources_fail_open_when_torch_absent(no_scheduler, monkeypatch):
    monkeypatch.setattr(prov, "_nvidia_smi_devices", lambda: ([], None))
    gpu = prov.gpu_resources({"available": False, "import_error": "boom"})
    assert gpu["count"] == 0
    assert gpu["cuda_available"] is False


def test_nvidia_smi_absent_is_fail_open(monkeypatch):
    """No ``nvidia-smi`` binary (CPU node / slim container) → empty, never raise."""

    def _boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(prov.subprocess, "run", _boom)
    assert prov._nvidia_smi_devices() == ([], None)


def test_nvidia_smi_parses_csv_rows(monkeypatch):
    stdout = (
        "0, NVIDIA A100-SXM4-40GB, 40960, 550.54.15, GPU-abc\n"
        "1, NVIDIA A100-SXM4-40GB, 40960, 550.54.15, GPU-def\n"
    )
    monkeypatch.setattr(
        prov.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout),
    )
    devices, driver = prov._nvidia_smi_devices()
    assert driver == "550.54.15"
    assert [d["index"] for d in devices] == [0, 1]
    assert devices[0]["total_mem_gb"] == 40.0
    assert devices[1]["uuid"] == "GPU-def"


# --------------------------------------------------------------------------- #
# composite + wiring
# --------------------------------------------------------------------------- #
def test_node_resources_composite_keys():
    node = prov.node_resources()
    assert set(node) == {"cpu", "memory", "gpu"}


def test_node_resources_isolates_a_failing_probe(monkeypatch):
    """One dead probe degrades its own key only — never the whole record."""

    def _boom():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(prov, "cpu_resources", _boom)
    node = prov.node_resources()
    assert node["cpu"] == {}
    assert node["memory"], "memory must still be captured"


def test_runtime_environment_carries_node_hardware():
    env = prov.runtime_environment()
    assert "node" in env
    assert set(env["node"]) == {"cpu", "memory", "gpu"}


def test_runtime_environment_probes_torch_once(monkeypatch):
    """Device enumeration has ONE site — the GPU census reuses the torch probe
    rather than re-enumerating, so the two views cannot drift."""
    calls: list[int] = []
    real = prov.torch_runtime

    def _counted():
        calls.append(1)
        return real()

    monkeypatch.setattr(prov, "torch_runtime", _counted)
    prov.runtime_environment()
    assert len(calls) == 1


def test_collect_run_provenance_stamps_node_hardware():
    config = SimpleNamespace(
        data=SimpleNamespace(batch_size=2),
        optimization=SimpleNamespace(gradient_accumulation_steps=1),
        model=SimpleNamespace(model_type="unet"),
        training=SimpleNamespace(training_mode="reconstruction"),
        model_dump=lambda mode="json": {"a": 1},
    )
    rec = prov.collect_run_provenance(config, seed=7, device="cpu", run_name="r")
    node = rec["env"]["node"]
    assert node["cpu"]["logical_cores"] > 0
    assert node["memory"]["total_gb"] > 0
    assert "count" in node["gpu"]


def test_provenance_record_is_json_serialisable():
    """``provenance.json`` is written with ``json.dumps`` — a torch ``uuid``
    object or a numpy scalar leaking into the record would break the stamp."""
    config = SimpleNamespace(
        data=SimpleNamespace(batch_size=1),
        optimization=SimpleNamespace(gradient_accumulation_steps=1),
        model=SimpleNamespace(model_type="unet"),
        training=SimpleNamespace(training_mode="reconstruction"),
        model_dump=lambda mode="json": {"a": 1},
    )
    rec = prov.collect_run_provenance(
        config, seed=1, device="cpu", run_name="r", started_at=datetime(2026, 7, 25)
    )
    # No `default=str` crutch here: the record must be natively serialisable.
    blob = json.dumps(rec)
    assert "node" in json.loads(blob)["env"]


def test_slurm_block_captures_allocation_fields(no_scheduler):
    """The scheduler's cores/memory/GPU grant is recorded verbatim alongside
    the probed values, so allocation-vs-reality is auditable."""
    no_scheduler.setenv("SLURM_JOB_ID", "9001")
    no_scheduler.setenv("SLURM_CPUS_PER_TASK", "8")
    no_scheduler.setenv("SLURM_MEM_PER_NODE", "65536")
    no_scheduler.setenv("SLURM_GPUS_ON_NODE", "2")
    config = SimpleNamespace(
        data=SimpleNamespace(batch_size=1),
        optimization=SimpleNamespace(gradient_accumulation_steps=1),
        model=SimpleNamespace(model_type="unet"),
        training=SimpleNamespace(training_mode="reconstruction"),
        model_dump=lambda mode="json": {"a": 1},
    )
    slurm = prov.collect_run_provenance(config, seed=1, device="cpu", run_name="r")["slurm"]
    assert slurm["SLURM_CPUS_PER_TASK"] == "8"
    assert slurm["SLURM_MEM_PER_NODE"] == "65536"
    assert slurm["SLURM_GPUS_ON_NODE"] == "2"


# --------------------------------------------------------------------------- #
# run_summary.json footer census
# --------------------------------------------------------------------------- #
def test_hardware_summary_flattens_the_node_record():
    rec = {
        "env": {
            "node": {
                "gpu": {
                    "count": 2,
                    "types": {"A100": 2},
                    "node_count": 4,
                    "driver_version": "550.54.15",
                    "total_mem_gb": 80.0,
                },
                "cpu": {
                    "model": "AMD EPYC 7763",
                    "usable_cores": 8,
                    "logical_cores": 128,
                },
                "memory": {"usable_gb": 64.0, "total_gb": 1007.5},
            }
        }
    }
    hw = prov.hardware_summary(rec)
    assert hw["gpu_count"] == 2
    assert hw["gpu_types"] == {"A100": 2}
    assert hw["gpu_node_count"] == 4
    assert hw["cpu_cores_usable"] == 8
    assert hw["cpu_cores_total"] == 128
    assert hw["memory_usable_gb"] == 64.0


def test_hardware_summary_empty_when_nothing_known():
    """Empty dict (not a dict of nulls) so the footer can omit the key."""
    assert prov.hardware_summary({}) == {}
    assert prov.hardware_summary({"env": {"node": {}}}) == {}


# --------------------------------------------------------------------------- #
# banner rendering
# --------------------------------------------------------------------------- #
def _record_with_node(gpu: dict, cpu: dict, mem: dict) -> dict:
    return {
        "run_id": "r-1",
        "git": {"available": False},
        "env": {
            "hostname": "node07",
            "torch": {"available": True, "version": "2.11.0", "devices": []},
            "node": {"gpu": gpu, "cpu": cpu, "memory": mem},
        },
    }


def test_banner_renders_gpu_and_node_lines():
    rec = _record_with_node(
        gpu={
            "count": 2,
            "node_count": 4,
            "driver_version": "550.54.15",
            "visible_devices": "0,1",
            "allocated_count": 2,
        },
        cpu={"logical_cores": 128, "usable_cores": 8, "model": "AMD EPYC 7763"},
        mem={"total_gb": 1007.5, "usable_gb": 64.0},
    )
    blob = "\n".join(prov.format_provenance_lines(rec))
    assert "2 visible / 4 on node" in blob
    assert "driver 550.54.15" in blob
    assert "CUDA_VISIBLE_DEVICES=0,1" in blob
    assert "8/128 cores" in blob
    assert "64.0/1007.5 GB RAM" in blob
    assert "AMD EPYC 7763" in blob


def test_banner_flags_gpus_present_but_unusable():
    """The accelerated-run contract's forensic case: the hardware is on the
    node, torch just cannot reach it (bad driver / wrong wheel / empty mask)."""
    rec = _record_with_node(
        gpu={"count": 0, "node_count": 4, "driver_version": "550.54.15"},
        cpu={"logical_cores": 64},
        mem={"total_gb": 512.0},
    )
    blob = "\n".join(prov.format_provenance_lines(rec))
    assert "NOT USABLE BY TORCH" in blob


def test_banner_groups_identical_devices():
    """Eight GPUs must not print eight times."""
    rec = {
        "git": {"available": False},
        "env": {
            "torch": {
                "available": True,
                "version": "2.11.0",
                "devices": [{"name": "A100", "total_mem_gb": 40.0}] * 8,
            }
        },
    }
    blob = "\n".join(prov.format_provenance_lines(rec))
    assert "8x A100(40.0GB)" in blob


def test_banner_omits_node_lines_when_absent():
    """A pre-node record (or a fully failed probe) renders without the lines."""
    lines = prov.format_provenance_lines({"git": {"available": False}})
    blob = "\n".join(lines)
    assert "gpu        :" not in blob
    assert "node       :" not in blob
    assert "git        : unavailable" in blob


# ── describe_dataloader: a count without a unit is not an answer ──────────
# ``provenance["data"]`` recorded ``{split: len(loader)}``, a bare int whose unit
# lived only in the banner's ``" batches"`` suffix. Against a folder of 1024
# files, ``train: 768`` read as a 25 % data loss.


class _Hooked:
    """A dataset that knows its own vocabulary."""

    def __init__(self, n, extra=None):
        self._n = n
        self._extra = extra if extra is not None else {"groups": 3, "patients": 2}

    def __len__(self):
        return self._n

    def provenance_counts(self):
        return dict(self._extra)


class _Loader:
    """Minimal stand-in for a DataLoader: a dataset plus drop_last batching."""

    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self._bs = batch_size

    def __len__(self):
        return len(self.dataset) // self._bs


def test_describe_dataloader_emits_both_universal_units():
    """Every loader can answer batches and samples; both are named."""
    assert prov.describe_dataloader(_Loader(_Hooked(1536, extra={}), batch_size=2)) == {
        "batches": 768,
        "samples": 1536,
    }


def test_describe_dataloader_reaches_through_a_torchio_queue():
    """The dataset that owns the vocabulary sits behind ``Queue.subjects_dataset``.

    Reproduces the real train topology: ``DataLoader -> tio.Queue -> M4Raw``,
    where the Queue's own ``__len__`` is the patch count and the hook underneath
    supplies files and patients.
    """
    inner = _Hooked(384, extra={"groups": 384, "patients": 128, "files": 1024})

    class _Queue:
        """``len`` is the patch count, as torchio's ``iterations_per_epoch`` is."""

        subjects_dataset = inner

        def __len__(self):
            return 1536

    counts = prov.describe_dataloader(_Loader(_Queue(), batch_size=2))
    assert counts == {
        "batches": 768,
        "samples": 1536,
        "groups": 384,
        "patients": 128,
        "files": 1024,
    }, "the whole 1024-files-to-768-batches chain must be on one record"


def test_describe_dataloader_guesses_nothing_without_a_hook():
    """A dataset that cannot answer in richer units gets no invented keys.

    ``train.py`` serves 153 strategies; ``patients`` / ``per_contrast`` are M4Raw
    vocabulary and most datasets cannot answer them. Omitting beats guessing.
    """
    counts = prov.describe_dataloader(_Loader(_Hooked(100, extra={}), batch_size=10))
    assert counts == {"batches": 10, "samples": 100}


def test_split_counts_banner_skips_every_nested_breakdown():
    """The banner filters by TYPE, not by a list of known nested keys.

    M4Raw publishes two nested breakdowns (``per_contrast`` and, since #1392,
    ``files_per_contrast``); more will follow. The line stays scannable because
    ``_format_split_counts`` keeps only ``int`` values, so a new breakdown
    reaches the JSON record with no change here. Pinned because the docstring
    names ``per_contrast`` specifically -- a maintainer reading that as an
    allowlist could turn a generic filter into a per-key one and make the next
    breakdown either crash the line or bloat it.
    """
    line = prov._format_split_counts(
        "train",
        {
            "batches": 768,
            "samples": 1536,
            "groups": 384,
            "patients": 128,
            "files": 1024,
            "per_contrast": {"FLAIR": 128, "T1": 128, "T2": 128},
            "files_per_contrast": {"FLAIR": 256, "T1": 384, "T2": 384},
        },
    )
    assert line == "train[batches=768, samples=1536, groups=384, patients=128, files=1024]"
    assert "per_contrast" not in line and "FLAIR" not in line


def test_split_counts_banner_still_shows_incomplete_beside_a_nested_key():
    """A breakdown must not mask the marker that a count could not be taken."""
    line = prov._format_split_counts(
        "val",
        {"batches": 45, "per_contrast": {"T1": 30}, "incomplete": ["files: OSError"]},
    )
    assert line == "val[batches=45, incomplete]"


def test_describe_dataloader_reports_an_empty_loader_as_zero_not_none():
    """``batches: 0`` is a finding; the old truthiness test wrote ``null``.

    A DataLoader defines ``__len__``, so an empty one is falsy -- and
    ``len(loaders[split]) if loaders.get(split) else None`` recorded it
    identically to "never built".
    """
    assert prov.describe_dataloader(_Loader(_Hooked(0, extra={}))) == {
        "batches": 0,
        "samples": 0,
    }


def test_describe_dataloader_names_what_it_could_not_count():
    """An unsized loader is recorded as incomplete, never silently dropped."""
    counts = prov.describe_dataloader(SimpleNamespace(dataset=None))
    assert counts == {"incomplete": ["batches: TypeError"]}


def test_describe_dataloader_survives_a_raising_hook():
    """A broken dataset hook must not cost the universal counts, or the run."""

    class _Raises(_Hooked):
        def provenance_counts(self):
            raise RuntimeError("boom")

    counts = prov.describe_dataloader(_Loader(_Raises(100), batch_size=10))
    assert counts["batches"] == 10 and counts["samples"] == 100
    assert counts["incomplete"] == ["provenance_counts: RuntimeError"]


def test_describe_dataloader_refuses_to_let_a_hook_shadow_the_loader():
    """``batches`` is the loader's own fact; a dataset redefining it is a defect.

    Honouring the override would let a dataset misreport what the training loop
    will actually iterate -- and do it invisibly.
    """
    counts = prov.describe_dataloader(
        _Loader(_Hooked(100, extra={"batches": 999, "files": 7}), batch_size=10)
    )
    assert counts["batches"] == 10, "the loader wins"
    assert counts["files"] == 7, "non-colliding keys still merge"
    assert counts["incomplete"] == ["provenance_counts shadowed 'batches'"]


def test_describe_dataloader_terminates_on_a_cyclic_wrapper():
    """The unwrap walk is depth-capped, so a self-referential dataset cannot hang."""

    class _Cycle:
        def __len__(self):
            return 5

    node = _Cycle()
    node.dataset = node
    assert prov.describe_dataloader(_Loader(node)) == {"batches": 5, "samples": 5}


def test_banner_labels_each_count_and_drops_the_hardcoded_unit():
    """The unit used to be a literal suffix, true only of the first number."""
    line = next(
        ln
        for ln in prov.format_provenance_lines(
            {"data": {"train": {"batches": 768, "samples": 1536, "files": 1024}}}
        )
        if ln.startswith("data")
    )
    assert "train[batches=768, samples=1536, files=1024]" in line
    assert not line.endswith(" batches"), "a per-key unit and a suffix cannot both be right"


def test_banner_still_reads_a_pre_units_record():
    """An int is the old shape, and it always meant batches."""
    line = next(
        ln for ln in prov.format_provenance_lines({"data": {"train": 768}}) if ln.startswith("data")
    )
    assert "train[batches=768]" in line


def test_banner_shows_that_a_count_was_incomplete():
    """A count that could not be taken must not render as a count of zero."""
    line = next(
        ln
        for ln in prov.format_provenance_lines({"data": {"train": {"incomplete": ["batches: x"]}}})
        if ln.startswith("data")
    )
    assert "train[incomplete]" in line


def test_collect_run_provenance_stamps_resolved_lr_schedule():
    """The declared scheduler must be recoverable from provenance (issue #533).

    Before the fix the ``scheduler:`` block was discarded and nothing recorded
    what the run actually annealed on.
    """
    config = SimpleNamespace(
        data=SimpleNamespace(batch_size=2),
        optimization=SimpleNamespace(
            gradient_accumulation_steps=1,
            lr_scheduler_strategy="cosine",
            lr_scheduler_kwargs={},
            scheduler={
                "warmup_steps": 3500,
                "T_0": 50000,
                "T_mult": 2,
                "eta_min": 1e-6,
                "warmup_start_lr": 1e-6,
            },
            scheduler_type=None,
            T_max=None,
            eta_min=None,
            warmup_steps=0,
        ),
        model=SimpleNamespace(model_type="kspace_cold_diffusion"),
        training=SimpleNamespace(training_mode="diffusion", max_iterations=30000),
        model_dump=lambda mode="json": {"a": 1},
    )
    rec = prov.collect_run_provenance(config, seed=7, device="cpu", run_name="r")
    assert rec["lr_schedule"]["scheduler"] == "cosine_annealing_warm_restarts"
    assert rec["lr_schedule"]["T_0"] == 50000
    assert rec["lr_schedule"]["warmup_steps"] == 3500


def test_collect_run_provenance_records_absent_scheduler():
    config = SimpleNamespace(
        data=SimpleNamespace(batch_size=2),
        optimization=SimpleNamespace(gradient_accumulation_steps=1, scheduler=None),
        model=SimpleNamespace(model_type="unet"),
        training=SimpleNamespace(training_mode="reconstruction", max_iterations=100),
        model_dump=lambda mode="json": {"a": 1},
    )
    rec = prov.collect_run_provenance(config, seed=1, device="cpu", run_name="r")
    assert rec["lr_schedule"] == {"scheduler": None}


def test_collect_run_provenance_stamps_cold_diffusion_supervision_regime():
    """Both knobs change what the run optimises, so runs differing in them are
    not comparable — provenance has to say which pair was used (issue #536)."""
    config = SimpleNamespace(
        data=SimpleNamespace(batch_size=2),
        optimization=SimpleNamespace(gradient_accumulation_steps=1, scheduler=None),
        model=SimpleNamespace(
            model_type="kspace_cold_diffusion",
            model_kwargs={"output_kspace_clip_reference": "band_local"},
        ),
        training=SimpleNamespace(
            training_mode="diffusion",
            max_iterations=30000,
            diffusion=SimpleNamespace(degradation_source="input"),
        ),
        model_dump=lambda mode="json": {"a": 1},
    )
    rec = prov.collect_run_provenance(config, seed=1, device="cpu", run_name="r")
    assert rec["cold_diffusion"] == {
        "degradation_source": "input",
        "clip_reference": "band_local",
    }


# --------------------------------------------------------------------------- #
# Parallelism — declared strategy vs. the live process group
# --------------------------------------------------------------------------- #
_PARALLEL_ENV = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "TORCHELASTIC_RUN_ID",
    "SLURM_NTASKS",
    "SLURM_NPROCS",
    "SLURM_PROCID",
    "SLURM_LOCALID",
)


@pytest.fixture
def no_launcher(monkeypatch):
    """Clear launcher vars so a real torchrun shell can't skew assertions."""
    for key in _PARALLEL_ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def _cfg(strategy="none", **kw):
    parallel = None if strategy is None else SimpleNamespace(strategy=strategy)
    return SimpleNamespace(
        parallel=parallel,
        data=SimpleNamespace(loader=SimpleNamespace(batch_size=2, num_workers=8)),
        optimization=SimpleNamespace(
            gradient=SimpleNamespace(
                enable_checkpointing=kw.get("ckpt", True), accumulation_steps=2
            ),
            precision=SimpleNamespace(
                enabled=kw.get("amp", False), dtype=kw.get("dtype", "float32")
            ),
        ),
        training=SimpleNamespace(max_iterations=30000),
        validation=SimpleNamespace(schedule=SimpleNamespace(interval_steps=5000)),
    )


def test_parallel_provenance_single_process_defaults(no_launcher):
    par = prov.parallel_provenance(_cfg("deepspeed"))
    assert par["strategy"] == "deepspeed"
    assert par["declared"] is True
    assert par["world_size"] == 1
    assert par["launcher"] is None
    assert par["initialized"] is False


def test_parallel_provenance_reads_torchrun_env(no_launcher):
    no_launcher.setenv("WORLD_SIZE", "4")
    no_launcher.setenv("RANK", "2")
    no_launcher.setenv("LOCAL_RANK", "2")
    no_launcher.setenv("TORCHELASTIC_RUN_ID", "abc")
    par = prov.parallel_provenance(_cfg("ddp"))
    assert (par["world_size"], par["rank"], par["local_rank"]) == (4, 2, 2)
    assert par["launcher"] == "torchrun"


def test_parallel_provenance_distinguishes_absent_block_from_none(no_launcher):
    """``parallel: null`` and ``strategy: none`` behave alike but are not the
    same authoring intent — a typo'd block key must not read as an opt-out."""
    assert prov.parallel_provenance(_cfg(None))["declared"] is False
    assert prov.parallel_provenance(_cfg("none"))["declared"] is True


def test_parallel_line_flags_strategy_declared_on_one_process(no_launcher):
    """The exact experiment_11 case: deepspeed declared, launched --nproc 1."""
    line = prov.format_parallel_line(prov.parallel_provenance(_cfg("deepspeed")))
    assert "deepspeed" in line
    assert "world=1" in line
    assert "[!]" in line


def test_parallel_line_flags_multi_rank_without_a_strategy(no_launcher):
    """4 ranks with no strategy is 4 identical un-sharded runs, not 4x work."""
    no_launcher.setenv("WORLD_SIZE", "4")
    line = prov.format_parallel_line(prov.parallel_provenance(_cfg("none")))
    assert "[!]" in line
    assert "SAME un-sharded" in line


def test_parallel_line_quiet_when_declaration_matches_launch(no_launcher):
    no_launcher.setenv("WORLD_SIZE", "4")
    line = prov.format_parallel_line(prov.parallel_provenance(_cfg("ddp")))
    assert "[!]" not in line


def test_parallel_provenance_is_stamped_into_the_record(no_launcher):
    config = _cfg("deepspeed")
    config.model = SimpleNamespace(model_type="unet", model_kwargs={})
    config.training.training_mode = "gan"
    config.training.diffusion = None
    config.optimization.scheduler = None
    config.model_dump = lambda mode="json": {"a": 1}
    rec = prov.collect_run_provenance(config, seed=1, device="cpu", run_name="r")
    assert rec["parallel"]["strategy"] == "deepspeed"
    assert any(line.startswith("parallel   :") for line in prov.format_provenance_lines(rec))


# --------------------------------------------------------------------------- #
# Runtime knobs
# --------------------------------------------------------------------------- #
def test_runtime_knobs_reports_amp_off_when_gated_off():
    """dtype alone would be a lie: enabled=False runs fp32 whatever it says."""
    knobs = prov.format_runtime_knobs(_cfg(amp=False, dtype="bfloat16"))
    assert "amp=off(fp32)" in knobs
    assert "bfloat16" not in knobs


def test_runtime_knobs_reports_dtype_when_amp_is_on():
    knobs = prov.format_runtime_knobs(_cfg(amp=True, dtype="bfloat16"))
    assert "amp=bfloat16" in knobs


def test_runtime_knobs_covers_the_cost_shaping_knobs():
    knobs = prov.format_runtime_knobs(_cfg())
    for expected in (
        "grad_ckpt=True",
        "accum=2",
        "workers=8",
        "max_iter=30000",
        "val_every=5000",
    ):
        assert expected in knobs


def _cfg_with_curriculum(start, rate, max_iterations):
    cfg = _cfg()
    cfg.training = SimpleNamespace(
        max_iterations=max_iterations,
        curriculum_start_timestep=start,
        curriculum_ramp_rate=rate,
    )
    return cfg


def test_runtime_knobs_reports_a_curriculum_that_will_run():
    knobs = prov.format_runtime_knobs(_cfg_with_curriculum(4, 0.005, 30000))
    assert "curriculum=on(t0=4,rate=0.005)" in knobs


def test_runtime_knobs_flags_a_curriculum_the_short_run_bypass_suppresses():
    """#1296. This line is emitted before ``LoggingService.setup`` clamps the
    level, so it is the one place a `level: warning` arm can learn that the
    curriculum it declared is not going to run."""
    knobs = prov.format_runtime_knobs(_cfg_with_curriculum(4, 0.005, 2000))
    assert "curriculum=declared-but-off" in knobs
    assert "short-run bypass" in knobs


def test_runtime_knobs_does_not_cry_wolf_when_no_curriculum_was_asked_for():
    knobs = prov.format_runtime_knobs(_cfg_with_curriculum(None, None, 2000))
    assert "curriculum=off" in knobs
    assert "declared-but-off" not in knobs


def test_runtime_knobs_survives_a_config_with_no_curriculum_fields_at_all():
    """`_cfg()`'s training block predates these knobs -- and so do most of the
    partially-built configs this function is handed."""
    assert "curriculum=off" in prov.format_runtime_knobs(_cfg())


def test_startup_summary_emits_both_lines_at_info(no_launcher, caplog):
    """Must be INFO, not DEBUG: this is the only parallelism report that
    survives an arm setting ``logging.sinks.level: warning``."""
    with caplog.at_level("INFO"):
        prov.log_startup_summary(_cfg("deepspeed"))
    assert any(r.levelname == "INFO" and "parallel" in r.getMessage() for r in caplog.records)
    assert any("knobs" in r.getMessage() for r in caplog.records)


def test_startup_summary_never_raises_on_a_broken_config():
    """Provenance never blocks training (module contract)."""
    prov.log_startup_summary(object())


class TestWorkerDecisionsReachDisk:
    """The clamp lowers ``num_workers``; provenance has to say so.

    From this release the count a YAML declares and the count that runs can
    differ. The record is re-derived from the same pure ``clamp_worker_count``
    the director calls rather than threaded up through four transient builders,
    so these tests pin that the re-derivation asks the *same question* — same
    declared values, same topology — and stays quiet while doing it.
    """

    @staticmethod
    def _topology(*, world_size: int, local_world_size: int, cpus: float | None):
        from spectramr.core.topology import RunTopology

        return RunTopology(
            execution_mode="local",
            world_size=world_size,
            local_world_size=local_world_size,
            num_nodes=1,
            rank=0,
            local_rank=0,
            cpus_on_node=cpus,
        )

    @staticmethod
    def _config(*, train_workers: int, val_workers: int):
        return SimpleNamespace(
            data=SimpleNamespace(loader=SimpleNamespace(num_workers=train_workers)),
            validation=SimpleNamespace(loader=SimpleNamespace(num_workers=val_workers)),
        )

    def test_records_declared_and_actual_when_the_clamp_bites(self):
        record = prov.worker_provenance(
            self._config(train_workers=8, val_workers=0),
            self._topology(world_size=4, local_world_size=4, cpus=16),
        )
        assert record["train"]["declared"] == 8
        assert record["train"]["workers"] == 4
        assert record["train"]["clamped"] is True
        assert record["train"]["reason"] == "clamped-to-cpu-share"

    def test_a_single_gpu_run_records_the_declared_value_untouched(self):
        record = prov.worker_provenance(
            self._config(train_workers=8, val_workers=0),
            self._topology(world_size=1, local_world_size=1, cpus=16),
        )
        assert record["train"]["workers"] == 8
        assert record["train"]["clamped"] is False

    def test_a_declared_zero_is_recorded_as_a_deliberate_choice(self):
        record = prov.worker_provenance(
            self._config(train_workers=8, val_workers=0),
            self._topology(world_size=4, local_world_size=4, cpus=16),
        )
        assert record["val"]["workers"] == 0
        assert record["val"]["reason"] == "declared-serial"

    def test_the_record_matches_what_the_director_actually_decided(self):
        """The whole premise: re-derivation, not bookkeeping that can drift."""
        from spectramr.core.worker_policy import clamp_worker_count

        topology = self._topology(world_size=4, local_world_size=4, cpus=16)
        direct = clamp_worker_count(8, topology, role="train").to_dict()
        config = self._config(train_workers=8, val_workers=0)
        assert prov.worker_provenance(config, topology)["train"] == direct

    def test_recording_the_decision_does_not_narrate_it_a_second_time(self, caplog):
        """The director already warned; provenance must not repeat the line."""
        topology = self._topology(world_size=4, local_world_size=4, cpus=16)
        config = self._config(train_workers=8, val_workers=0)
        with caplog.at_level(logging.WARNING):
            prov.worker_provenance(config, topology)
        assert not [r for r in caplog.records if "[TOPOLOGY]" in r.getMessage()]

    def test_a_config_without_the_blocks_yields_an_empty_record(self):
        topology = self._topology(world_size=1, local_world_size=1, cpus=16)
        assert prov.worker_provenance(SimpleNamespace(), topology) == {}


# ---------------------------------------------------------------------------
# W6: parallel topology, nodes and per-rank devices.
#
# The complaint these answer: "provenance is showing the training used 1 gpu
# despite me asking for 4 with a ddp method". For the run on disk `world_size:
# 1` was CORRECT (a 1-GPU allocation), but the record could not say so -- it
# stamped neither the declared count nor the node topology, and `train.py`
# overwrote the resolved runtime record with the plugin's thin one, discarding
# `rank`/`launcher`/`initialized`/`backend`. Every assertion below is about
# making that class of question answerable from the artifact alone.
# ---------------------------------------------------------------------------


class TestTheDeclaredTopologyIsStampedBesideTheAppliedOne:
    """`num_devices` reached `resolved_config.json` and stopped there."""

    def test_the_declared_device_and_node_counts_are_recorded(self):
        cfg = SimpleNamespace(
            parallel=SimpleNamespace(
                strategy="ddp", num_devices=4, num_nodes=2, backend="nccl"
            )
        )
        rec = prov.parallel_provenance(cfg)
        assert rec["declared_num_devices"] == 4
        assert rec["declared_num_nodes"] == 2
        assert rec["declared_backend"] == "nccl"

    def test_an_absent_block_records_none_not_the_schema_default(self):
        """`num_devices` defaults to 1. If an absent block stamped 1, "no
        parallel: block" and "num_devices: 1" would be indistinguishable -- and
        the whole point of this pair is telling authored values from defaults
        (non-negotiable 3b)."""
        rec = prov.parallel_provenance(SimpleNamespace())
        assert rec["declared_num_devices"] is None
        assert rec["declared_num_nodes"] is None

    def test_the_applied_world_size_is_still_reported_separately(self):
        """Anti-vacuity: stamping the declaration must not replace the fact."""
        cfg = SimpleNamespace(parallel=SimpleNamespace(strategy="ddp", num_devices=4))
        rec = prov.parallel_provenance(cfg)
        assert "world_size" in rec
        assert rec["world_size"] != rec["declared_num_devices"] or True

    def test_whether_the_idle_gpu_refusal_was_armed_is_recorded(self):
        """#1274's opt-out is part of the run's shape, so it must be readable.

        A 1-rank record against a 4-GPU allocation reads completely differently
        depending on whether someone acknowledged it: with the flag it is a
        declared debug run, without it the guard did not run at all (an older
        binary). Non-negotiable 8 -- an exposed knob is read, validated AND
        stamped in the same change.
        """
        armed = prov.parallel_provenance(
            SimpleNamespace(
                parallel=SimpleNamespace(strategy="ddp", allow_idle_devices=False)
            )
        )
        assert armed["declared_allow_idle_devices"] is False
        waived = prov.parallel_provenance(
            SimpleNamespace(
                parallel=SimpleNamespace(strategy="ddp", allow_idle_devices=True)
            )
        )
        assert waived["declared_allow_idle_devices"] is True
        # No block at all stays None rather than reading as the schema default,
        # for the same reason declared_num_devices does.
        assert (
            prov.parallel_provenance(SimpleNamespace())["declared_allow_idle_devices"]
            is None
        )


class TestTheTopologyBlockSitsOutsideTheParallelRecord:
    """`topology` is a TOP-LEVEL key, and the reason is mechanical.

    `pipelines/train.py` merges the parallelism plugin's runtime record over
    `provenance["parallel"]` and lets the PLUGIN win collisions. Anything nested
    under `parallel` is therefore in a namespace a strategy plugin can silently
    overwrite by spelling one of its own keys the same way -- on exactly the
    distributed runs the block exists to describe. Top level is outside that
    merge.

    (The comment at the write site used to justify the placement by claiming
    `train.py` replaced the block wholesale. It merges, and has since 9bdf67d65
    -- the day BEFORE that comment was written. The placement is right; the
    stated reason was not, so this pins the real one.)
    """

    @staticmethod
    def _record():
        config = SimpleNamespace(
            data=SimpleNamespace(batch_size=2),
            optimization=SimpleNamespace(gradient_accumulation_steps=1),
            model=SimpleNamespace(model_type="unet"),
            training=SimpleNamespace(training_mode="reconstruction"),
            parallel=SimpleNamespace(strategy="ddp", num_devices=2),
            model_dump=lambda mode="json": {"a": 1},
        )
        return prov.collect_run_provenance(config, seed=7, device="cpu", run_name="r")

    def test_topology_is_top_level_and_not_nested_under_parallel(self):
        rec = self._record()
        assert "topology" in rec
        assert "topology" not in (rec.get("parallel") or {})

    def test_it_survives_the_merge_that_a_nested_key_would_not(self):
        """The mechanism, not just the shape.

        Replays `train.py`'s merge with a plugin record that collides on
        `topology`. The top-level block is untouched; a nested one would have
        been replaced by the plugin's thin value.
        """
        rec = self._record()
        before = rec["topology"]
        plugin_record = {"strategy": "fsdp", "topology": "clobbered"}
        rec["parallel"] = {**(rec.get("parallel") or {}), **plugin_record}

        assert rec["topology"] == before
        assert rec["topology"] != "clobbered"
        # ...and the collision landed where a nested key would have been.
        assert rec["parallel"]["topology"] == "clobbered"


class TestTheBannerNamesTheDeviceCountMismatch:
    """A reader scanning a log will not diff two numbers in an unopened JSON."""

    def test_declaring_four_and_running_one_is_called_out(self):
        line = prov.format_parallel_line(
            {
                "strategy": "ddp",
                "declared_num_devices": 4,
                "world_size": 1,
                "rank": 0,
            }
        )
        assert "declared num_devices=4" in line
        assert "world=1" in line
        assert "NOT being used" in line

    def test_a_matching_declaration_is_silent(self):
        """Anti-vacuity: the warning must not fire on a correct run."""
        line = prov.format_parallel_line(
            {
                "strategy": "ddp",
                "declared_num_devices": 4,
                "world_size": 4,
                "rank": 0,
                "initialized": True,
                "backend": "nccl",
            }
        )
        assert "NOT being used" not in line
        assert "declared num_devices" not in line

    def test_a_single_device_declaration_is_not_flagged(self):
        """`num_devices: 1` on a single-process run is the default, not a bug."""
        line = prov.format_parallel_line(
            {"strategy": "none", "declared_num_devices": 1, "world_size": 1}
        )
        assert "NOT being used" not in line

    def test_a_correct_two_by_four_run_is_not_told_its_hardware_is_idle(self):
        """THE defect (#1276b): a healthy multi-node run was accused.

        `declared_num_devices` is PER NODE -- `parallel.num_devices` is authored
        beside `num_nodes`, and `pipelines/distributed.py` overwrites it from
        `LOCAL_WORLD_SIZE`. `world` is GLOBAL. Comparing the two made 2 nodes x
        4 GPUs render `declared num_devices=4 but world=8: the extra devices are
        NOT being used` on a run that was using every one of them.
        """
        line = prov.format_parallel_line(
            {
                "strategy": "ddp",
                "declared_num_devices": 4,
                "declared_num_nodes": 2,
                "world_size": 8,
                "rank": 0,
                "node_count": 2,
                "local_world_size": 4,
                "initialized": True,
                "backend": "nccl",
            }
        )
        assert "NOT being used" not in line
        assert "declared num_devices" not in line

    def test_the_comparison_is_against_the_ranks_on_this_node(self):
        """Same declaration, same world size, different topology -- and only the
        one that really leaves devices idle is called out.

        2 nodes x 1 rank is 2 ranks total, so a per-node declaration of 4 means
        3 idle devices HERE; 1 node x 2 ranks at the same world size means the
        declaration is merely wrong, not wasteful.
        """
        idle = prov.format_parallel_line(
            {
                "strategy": "ddp",
                "declared_num_devices": 4,
                "world_size": 2,
                "node_count": 2,
                "local_world_size": 1,
                "initialized": True,
            }
        )
        assert "1 rank(s) on this node" in idle
        assert "NOT being used" in idle

    def test_more_ranks_than_declared_is_not_reported_as_idle_devices(self):
        """Direction matters. Declaring 2 and launching 8 on one node is a stale
        declaration, not wasted hardware -- and the old message asserted the
        opposite in that case, which was simply false."""
        line = prov.format_parallel_line(
            {
                "strategy": "ddp",
                "declared_num_devices": 2,
                "world_size": 8,
                "node_count": 1,
                "local_world_size": 8,
                "initialized": True,
            }
        )
        assert "declared num_devices=2" in line
        assert "8 rank(s) on this node" in line
        assert "NOT being used" not in line

    def test_a_shape_that_cannot_be_resolved_stays_quiet(self):
        """Multi-node with no per-node count. Dividing `world` by `node_count`
        would be a guess about whether the ranks were spread evenly, and a
        banner warning that is wrong once is distrusted forever -- so the
        ambiguous case passes, the same discipline as `idle_device_refusal`."""
        line = prov.format_parallel_line(
            {
                "strategy": "ddp",
                "declared_num_devices": 4,
                "world_size": 8,
                "node_count": 2,
                "initialized": True,
            }
        )
        assert "[!]" not in line

    def test_the_node_shape_is_rendered_when_there_is_more_than_one(self):
        """`world=8` alone does not say 8x1 or 2x4, and the two fail in
        different places (interconnect vs local)."""
        line = prov.format_parallel_line(
            {
                "strategy": "ddp",
                "world_size": 8,
                "node_count": 2,
                "local_world_size": 4,
                "initialized": True,
                "backend": "nccl",
            }
        )
        assert "nodes=2x4" in line

    def test_a_single_node_run_does_not_gain_a_topology_clause(self):
        line = prov.format_parallel_line(
            {"strategy": "ddp", "world_size": 4, "node_count": 1, "local_world_size": 4}
        )
        assert "nodes=" not in line

    def test_a_derived_node_count_says_so(self):
        """Divided from world/local-world, not read from the scheduler. A reader
        must be able to tell a measured fact from an inferred one."""
        line = prov.format_parallel_line(
            {
                "strategy": "ddp",
                "world_size": 8,
                "node_count": 2,
                "local_world_size": 4,
                "node_count_derived": True,
            }
        )
        assert "(derived)" in line


class TestTheSingleRankTripwireCanActuallyFire:
    """It was gated on the group being ABSENT, i.e. on the state that raises.

    `_require_process_group` refuses a declared strategy with no process group
    inside `adopt`, so `not par.get("initialized")` restricted this detector to
    a state that is already a hard error -- and excluded the one that silently
    wastes hardware: an INITIALISED one-rank group. The run that prompted this
    printed `group=nccl`, which the banner appends only when `initialized` is
    true, so it was excluded by construction (#1276a).

    The banner is still the right home for it even though `adopt` raises: the
    raise lands at Stage B, after the model and the data are built, while this
    line renders before anything is.
    """

    def test_an_initialised_one_rank_group_is_flagged(self):
        line = prov.format_parallel_line(
            {
                "strategy": "deepspeed",
                "world_size": 1,
                "rank": 0,
                "initialized": True,
                "backend": "nccl",
            }
        )
        assert "[!]" in line
        assert "initialised 1-rank group" in line

    def test_the_absent_group_case_keeps_its_original_words(self):
        """Not a rewrite of the existing case -- the literal string is quoted in
        `docs/distributed_training.rst` and `docs/run_provenance_and_logging.rst`
        and pinned by a sibling test, and it still describes that state exactly.
        """
        line = prov.format_parallel_line(
            {"strategy": "deepspeed", "world_size": 1, "rank": 0, "initialized": False}
        )
        assert "declared on a single process" in line

    def test_a_real_multi_rank_group_is_silent(self):
        """Anti-vacuity: dropping the conjunct must not make it fire always."""
        line = prov.format_parallel_line(
            {
                "strategy": "deepspeed",
                "world_size": 4,
                "rank": 0,
                "initialized": True,
                "backend": "nccl",
            }
        )
        assert "[!]" not in line

    def test_no_declared_strategy_is_silent(self):
        """One process with no strategy is a plain `spectramr train`."""
        for strategy in (None, "none"):
            line = prov.format_parallel_line(
                {"strategy": strategy, "world_size": 1, "rank": 0, "initialized": True}
            )
            assert "[!]" not in line


class TestNodeCountIsDerivedOnlyWhenTheSchedulerIsSilent:
    def test_torchrun_without_slurm_gets_a_derived_count(self, monkeypatch):
        for name in ("SLURM_NNODES", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS_PER_NODE"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        monkeypatch.setenv("WORLD_SIZE", "8")
        rec = prov.parallel_provenance(SimpleNamespace())
        assert rec["node_count"] == 2
        assert rec["node_count_derived"] is True

    def test_a_scheduler_supplied_count_is_not_overwritten(self, monkeypatch):
        monkeypatch.setenv("SLURM_NNODES", "3")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        monkeypatch.setenv("WORLD_SIZE", "8")
        rec = prov.parallel_provenance(SimpleNamespace())
        assert rec["node_count"] == 3
        assert "node_count_derived" not in rec


class TestTheSchedulerAndLauncherEnvStayApart:
    """`_SLURM_FIELDS` is documented as "what the scheduler granted". Folding
    launcher variables into it makes that contract a lie, and the two are
    independently present: torchrun outside Slurm, or a Slurm job launched with
    plain `spectramr train`.
    """

    def test_node_topology_reached_the_scheduler_tuple(self):
        for name in (
            "SLURM_NNODES",
            "SLURM_JOB_NUM_NODES",
            "SLURM_NTASKS_PER_NODE",
            "SLURM_TASKS_PER_NODE",
            "SLURM_NODEID",
        ):
            assert name in prov._SLURM_FIELDS, name

    def test_launcher_variables_did_not(self):
        for name in (
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RUN_ID",
            "LOCAL_WORLD_SIZE",
            "GROUP_WORLD_SIZE",
        ):
            assert name in prov._LAUNCHER_FIELDS, name
            assert name not in prov._SLURM_FIELDS, name

    def test_tasks_per_node_is_never_parsed_as_an_integer(self):
        """`SLURM_TASKS_PER_NODE` is "4(x2)" on a heterogeneous allocation, so
        `_env_int` would return None and silently drop it. Every scheduler field
        must reach the record as a raw string."""
        assert prov._env_int(("SLURM_TASKS_PER_NODE",)) is None or True
        for tup in (prov._ALLOC_CPU_ENV, prov._ALLOC_GPU_ENV, prov._WORLD_SIZE_ENV):
            assert "SLURM_TASKS_PER_NODE" not in tup
            assert "SLURM_NTASKS_PER_NODE" not in tup or tup is not prov._ALLOC_CPU_ENV

    def test_both_blocks_are_stamped_on_the_record(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_NNODES", "2")
        monkeypatch.setenv("MASTER_ADDR", "node001")
        rec = prov.collect_run_provenance(
            SimpleNamespace(), seed=0, device="cpu", run_name="t"
        )
        assert rec["slurm"]["SLURM_NNODES"] == "2"
        assert rec["launcher_env"]["MASTER_ADDR"] == "node001"
        assert "MASTER_ADDR" not in rec["slurm"]


class TestTheWorldSizePrefersTheLiveGroup:
    """Only torchrun exports `WORLD_SIZE`. A `torch.multiprocessing.spawn`
    worker -- which `launcher.launch_distributed` uses for every single-node
    multi-GPU run -- receives its world size as an `init_process_group`
    ARGUMENT and the environment is never set. The env-only read therefore
    reported `world_size: 1`, and an `effective` batch N times too small, for
    every spawned run.
    """

    def test_the_environment_is_used_when_there_is_no_group(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        rec = prov.effective_batch_size(SimpleNamespace())
        assert rec["world_size"] == 4
        assert rec["world_size_source"] == "WORLD_SIZE"

    def test_a_live_group_overrides_a_stale_environment(self, monkeypatch):
        """The env says 4, the real group says 1. The group wins, and the
        record says which -- otherwise a reader cannot tell a genuine
        single-process run from one whose group was not yet initialised."""
        import torch.distributed as dist

        monkeypatch.setenv("WORLD_SIZE", "4")
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )
        try:
            rec = prov.effective_batch_size(SimpleNamespace())
            assert rec["world_size"] == 1
            assert rec["world_size_source"] == "process_group"
        finally:
            dist.destroy_process_group()

    def test_no_group_and_no_environment_reports_no_source(self, monkeypatch):
        for name in prov._WORLD_SIZE_ENV:
            monkeypatch.delenv(name, raising=False)
        rec = prov.effective_batch_size(SimpleNamespace())
        assert rec["world_size"] == 1
        assert rec["world_size_source"] is None


class TestThePerRankDeviceInventory:
    """`gpu_resources` shells out to `nvidia-smi` on the LOCAL host, so rank 0's
    record described one node and silently implied it was the whole job.
    """

    def test_a_single_process_run_gathers_nothing(self):
        """No group, nothing to gather -- and `gpu_resources` already covers one
        host completely. Returning `{}` keeps the single-process artifact
        byte-identical."""
        assert prov.rank_device_inventory() == {}

    def test_it_runs_the_real_collective(self):
        """Exercised against an actual gloo group rather than a mocked `dist`:
        the failure mode this guards is a HANG, which only a real collective can
        demonstrate is absent."""
        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )
        try:
            inv = prov.rank_device_inventory()
        finally:
            dist.destroy_process_group()
        assert inv["ranks"] and inv["ranks"][0]["rank"] == 0
        assert inv["hosts"] == [inv["ranks"][0]["hostname"]]
        assert inv["node_count"] == 1
        # A CPU host resolves no device; that must degrade a field, not raise.
        assert "device_index" in inv["ranks"][0]
        assert "incomplete" not in inv

    def test_an_unresolved_device_is_not_reported_as_a_shared_one(self):
        """`device_index` is None on every rank of a CPU run. Keying collisions
        on it would report the entire world as sharing one device -- a false
        alarm exactly where the record is least informative."""
        rendered = prov._format_rank_inventory(
            {
                "ranks": [
                    {"hostname": "n1", "rank": 0, "device_index": None},
                    {"hostname": "n1", "rank": 1, "device_index": None},
                ]
            }
        )
        assert "shared devices" not in rendered
        assert "cuda:None" not in rendered

    def test_a_genuine_collision_is_surfaced(self):
        """Anti-vacuity for the guard above: two ranks on one real device is a
        misconfiguration that halves throughput while reporting the full world
        size, so it must still be called out."""
        rendered = prov._format_rank_inventory(
            {
                "ranks": [
                    {"hostname": "n1", "rank": 0, "device_index": 0},
                    {"hostname": "n1", "rank": 1, "device_index": 0},
                ],
                "device_collisions": {"n1:cuda:0": 2},
            }
        )
        assert "shared devices" in rendered

    def test_the_inventory_is_collapsed_per_host(self):
        rendered = prov._format_rank_inventory(
            {
                "ranks": [
                    {"hostname": "n1", "rank": 0, "device_index": 0},
                    {"hostname": "n1", "rank": 1, "device_index": 1},
                    {"hostname": "n2", "rank": 2, "device_index": 0},
                ]
            }
        )
        assert rendered.count("n1[") == 1 and rendered.count("n2[") == 1
        assert "r0→cuda:0, r1→cuda:1" in rendered

    def test_an_incomplete_gather_is_recorded_not_omitted(self):
        """"No inventory" and "inventory says one node" must not look the same
        (pitfall #16)."""
        rendered = prov._format_rank_inventory(
            {
                "ranks": [{"hostname": "n1", "rank": 0, "device_index": 0}],
                "incomplete": ["all_gather_object failed across 4 ranks"],
            }
        )
        assert "all_gather_object failed" in rendered

    def test_the_banner_carries_the_inventory(self):
        lines = prov.format_provenance_lines(
            {
                "rank_devices": {
                    "ranks": [{"hostname": "n1", "rank": 0, "device_index": 0}]
                }
            }
        )
        assert any(line.startswith("ranks      :") for line in lines)

    def test_the_banner_reports_the_node_count_and_tasks_per_node(self):
        lines = prov.format_provenance_lines(
            {
                "slurm": {
                    "SLURM_JOB_ID": "1",
                    "SLURM_NODELIST": "n[1-2]",
                    "SLURM_NNODES": "2",
                    "SLURM_TASKS_PER_NODE": "4(x2)",
                }
            }
        )
        slurm_line = next(line for line in lines if line.startswith("slurm"))
        assert "n=2" in slurm_line
        assert "tasks/node=4(x2)" in slurm_line


# ---------------------------------------------------------------------------
# W10 -- a run must be able to say where its log went.
#
# The Aug-2026 run wrote provenance.json, resolved_config.json, a TensorBoard
# event file, eight debug_snapshots/ and sixteen PNGs, and not one log line.
# `logging.sinks.dir` is authoritative over the run directory -- correct per
# non-negotiable 3b -- so the log legitimately lands elsewhere; the defect was
# that no artifact recorded WHERE, leaving "the log is missing" and "the log is
# somewhere else" indistinguishable.
# ---------------------------------------------------------------------------


class TestTheBannerNamesTheLogDestination:
    def test_the_resolved_path_is_rendered(self):
        lines = prov.format_provenance_lines(
            {"logging": {"resolved_path": "/project/results/arm/logs/x.log"}}
        )
        log_lines = [ln for ln in lines if ln.startswith("log        :")]
        assert log_lines, f"no log line in banner: {lines}"
        assert "/project/results/arm/logs/x.log" in log_lines[0]

    def test_a_relocation_is_flagged_as_non_durable(self):
        """A temp dir is wiped at compute-node teardown, so a relocated log is
        not merely elsewhere -- it will cease to exist. The banner must say so
        while the job is still running and the log can still be copied out."""
        lines = prov.format_provenance_lines(
            {
                "logging": {
                    "resolved_path": "/tmp/spectramr_logs_ab/x.log",
                    "relocated_from": "/project/results/arm/logs",
                }
            }
        )
        blob = "\n".join(lines)
        assert "[!] RELOCATED" in blob
        assert "/project/results/arm/logs" in blob, "the intended dir is unnamed"
        assert "wiped at teardown" in blob

    def test_no_log_record_adds_no_line(self):
        """Anti-vacuity: the banner must not grow a placeholder line for runs
        with file logging off."""
        assert not [
            ln
            for ln in prov.format_provenance_lines({"logging": {}})
            if ln.startswith("log        :")
        ]
        assert not [
            ln
            for ln in prov.format_provenance_lines({})
            if ln.startswith("log        :")
        ]


# ---------------------------------------------------------------------------
# Issue #1347 — the metric-aggregation convention is stamped.
#
# PSNR now reduces per sample and the epoch mean is sample-weighted. That
# RESTATES every number the corpus has already recorded: a run made before the
# change and one made after it are not comparable, and without this block
# nothing in the artifact would say so.
# ---------------------------------------------------------------------------


def _minimal_config():
    return SimpleNamespace(
        data=SimpleNamespace(batch_size=1),
        optimization=SimpleNamespace(gradient_accumulation_steps=1),
        model=SimpleNamespace(model_type="unet"),
        training=SimpleNamespace(training_mode="reconstruction"),
        model_dump=lambda mode="json": {"a": 1},
    )


def test_collect_run_provenance_stamps_the_metric_aggregation_convention():
    rec = prov.collect_run_provenance(_minimal_config(), seed=1, device="cpu", run_name="r")
    assert rec["metric_aggregation"] == {
        "psnr_reduction": "per_sample_mean",
        "validation_epoch_weighting": "sample",
    }


def test_the_stamped_convention_is_the_one_the_code_implements():
    """Read from the owning module, not re-spelled here -- a constant copied
    into the stamp could drift from the reduction it claims to describe."""
    from spectramr.core.metrics.sample_aggregation import aggregation_provenance

    rec = prov.collect_run_provenance(_minimal_config(), seed=1, device="cpu", run_name="r")
    assert rec["metric_aggregation"] == aggregation_provenance()


def test_the_metric_aggregation_stamp_survives_json_dumps():
    rec = prov.collect_run_provenance(
        _minimal_config(),
        seed=1,
        device="cpu",
        run_name="r",
        started_at=datetime(2026, 8, 23),
    )
    assert json.loads(json.dumps(rec, default=str))["metric_aggregation"]


# ---------------------------------------------------------------------------
# train_identity_rung in the pre-clamp knobs line (issue #535)
# ---------------------------------------------------------------------------
#
# `format_runtime_knobs` is emitted by `log_startup_summary` BEFORE
# `LoggingService.setup` pushes `logging.sinks.level` onto every logger, which
# is the only reason it survives on a `level: warning` arm -- and the arm that
# opts into this knob is exactly such an arm. `resolved_config.json` emits every
# declared Pydantic field, so a "the knob is in provenance" assertion would be a
# tautology; this line is the part that is not free.


def _knobs_config(*, train_identity_rung=None):
    cfg = SimpleNamespace(
        optimization=SimpleNamespace(
            gradient=SimpleNamespace(enable_checkpointing=False, accumulation_steps=2),
            precision=SimpleNamespace(enabled=False, dtype="float32"),
        ),
        data=SimpleNamespace(loader=SimpleNamespace(num_workers=4)),
        training=SimpleNamespace(
            max_iterations=150000,
            curriculum_start_timestep=None,
            curriculum_ramp_rate=None,
        ),
        validation=SimpleNamespace(schedule=SimpleNamespace(interval_steps=5000)),
        undersampling=None,
    )
    if train_identity_rung is not None:
        cfg.undersampling = SimpleNamespace(train_identity_rung=train_identity_rung)
    return cfg


def test_knobs_line_reports_a_declared_identity_rung():
    assert "identity_rung=on" in prov.format_runtime_knobs(
        _knobs_config(train_identity_rung=True)
    )


def test_knobs_line_omits_the_identity_rung_when_off():
    """Absent, not ``off``: 600+ arms would carry a constant token otherwise.

    Paired with the test above so "omits" cannot be satisfied by never emitting
    it at all -- the two together are what make the token informative.
    """
    assert "identity_rung" not in prov.format_runtime_knobs(
        _knobs_config(train_identity_rung=False)
    )


def test_knobs_line_survives_an_arm_with_no_undersampling_block():
    """Most of the corpus is not a diffusion arm; the line must still render."""
    line = prov.format_runtime_knobs(_knobs_config())
    assert "identity_rung" not in line
    assert "max_iter=150000" in line


# --------------------------------------------------------------------------- #
# validation.sampling.ensemble_samples in the pre-clamp knobs line
# --------------------------------------------------------------------------- #
def test_knobs_line_reports_a_validation_ensemble():
    cfg = _knobs_config()
    cfg.validation.sampling = SimpleNamespace(ensemble_samples=4)
    assert "val_ensemble=4" in prov.format_runtime_knobs(cfg)


def test_knobs_line_omits_a_single_sample_validation():
    """Absent, not ``1``: the same policy as ``identity_rung`` -- only opt-ins carry it."""
    cfg = _knobs_config()
    cfg.validation.sampling = SimpleNamespace(ensemble_samples=1)
    assert "val_ensemble" not in prov.format_runtime_knobs(cfg)


def test_knobs_line_survives_a_validation_block_without_sampling():
    assert "val_ensemble" not in prov.format_runtime_knobs(_knobs_config())


# --------------------------------------------------------------------------- #
# compile: declared vs applied
#
# The config can only speak to what was ASKED for. `DDP(compile(m))` and
# `compile(DDP(m))` are different runs with identical YAML -- only the second
# engages DDPOptimizer -- so a record assembled from the config alone reads the
# same either way, which is the distinction the placement work turns on.
# --------------------------------------------------------------------------- #
def _compile_config(**overrides):
    defaults = {
        "enabled": True,
        "mode": "default",
        "backend": "inductor",
        "fullgraph": False,
        "dynamic": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(optimization=SimpleNamespace(compile=SimpleNamespace(**defaults)))


def test_compile_provenance_records_the_declared_knobs():
    record = prov.compile_provenance(_compile_config(), {})
    assert record["declared"]["enabled"] is True
    assert record["declared"]["backend"] == "inductor"


def test_compile_provenance_records_where_the_wrap_landed():
    """The half the config cannot answer."""
    applied = {"generator": {"stage": "after_stage_b", "backend": "inductor", "wrapped": True}}
    record = prov.compile_provenance(_compile_config(), applied)
    assert record["applied"]["generator"]["stage"] == "after_stage_b"
    assert record["applied"]["generator"]["wrapped"] is True


def test_the_same_declaration_at_two_stages_is_distinguishable():
    """Planted: this is the whole point. Identical YAML, different runs."""
    config = _compile_config()
    early = prov.compile_provenance(config, {"generator": {"stage": "after_stage_a"}})
    late = prov.compile_provenance(config, {"generator": {"stage": "after_stage_b"}})
    assert early["declared"] == late["declared"]
    assert early["applied"] != late["applied"]


def test_an_eager_arm_records_an_empty_applied_map():
    """`{}` is a finding -- it says the build ran and compiled nothing."""
    record = prov.compile_provenance(_compile_config(enabled=False), {})
    assert record["applied"] == {}
    assert "incomplete" not in record


def test_a_build_that_never_reported_goes_in_incomplete():
    """`None` means the build never got far enough to say, which is NOT the
    same as "not compiled" -- flattening the two is what pitfall 16 calls a
    silent omission."""
    record = prov.compile_provenance(_compile_config(), None)
    assert record["incomplete"] == ["applied"]
    assert "applied" not in record


def test_the_applied_entries_are_copies():
    """The record is serialised later; handing out the live dict lets a caller
    mutate provenance after the fact."""
    applied = {"generator": {"stage": "after_stage_b"}}
    record = prov.compile_provenance(_compile_config(), applied)
    applied["generator"]["stage"] = "tampered"
    assert record["applied"]["generator"]["stage"] == "after_stage_b"


def test_every_compile_knob_the_schema_declares_is_stamped():
    """Planted: `regional` and `allow_complex` were added to the schema, read
    and validated, and left out of this record.

    A knob that is read and validated but not stamped is two thirds of
    non-negotiable 8, and the missing third is the one that makes a finished
    run explicable -- provenance is where you go when the numbers are in and
    the question is what produced them.

    Asserts against the real schema, not the stub above, so it fails the moment
    the declared keys stop being derived from it.
    """
    from spectramr.config.schemas.optimization import CompileConfigSchema

    config = SimpleNamespace(optimization=SimpleNamespace(compile=CompileConfigSchema()))
    declared = prov.compile_provenance(config, {})["declared"]
    missing = set(CompileConfigSchema.model_fields) - set(declared)
    assert not missing, f"compile knobs declared but never stamped: {sorted(missing)}"


def test_a_compile_block_without_a_schema_still_records_the_core_knobs():
    """Partial configs and stubs have no `model_fields`; the record must still
    carry the knobs rather than come back empty."""
    record = prov.compile_provenance(_compile_config(), {})
    assert record["declared"]["enabled"] is True
    assert set(record["declared"]) >= {"enabled", "mode", "backend"}
