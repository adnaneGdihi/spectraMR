"""Run-topology resolution: how many ranks, on how many nodes, how many CPUs.

Context: ``experiment_11_attention_none`` ran SLOWER on four GPUs than on one.
The cause was that ``training.max_iterations`` is a per-rank loop bound with no
``world_size`` term, so every rank ran the whole 30 000-iteration experiment;
the aggravating factor was ``num_workers`` having no topology term either, so a
4-rank node spawned four times the declared decoder processes. This module is
the SSOT those decisions were missing. The clamp itself is a separate policy
(``core/worker_policy.py``) and is tested in ``test_worker_policy.py``.

Style note: these tests pass ``environ=`` / ``cpu_probe=`` / ``path_exists=``
explicitly and use **no** monkeypatch, mirroring
``tests/unit/core/test_compute_device.py`` rather than ``test_env.py``. That is
deliberate -- ``resolve_run_topology`` takes an injected mapping precisely so
every branch is testable without a scheduler, a GPU, or a container. It also
matters here because there is no global autouse fixture that snapshots and
restores ``os.environ`` between tests, so a module that leaned on ambient
environment state would leak into its neighbours.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from spectramr.core.topology import (
    EXECUTION_MODES,
    RunTopology,
    TopologyResolutionError,
    resolve_run_topology,
)


def _probe(cores):
    """A stand-in for ``cpu_resources`` returning a fixed usable-core count."""

    def probe(environ=None):
        return {"usable_cores": cores}

    return probe


def _resolve(env=None, *, cores=16, exists=lambda _p: False):
    return resolve_run_topology(env or {}, cpu_probe=_probe(cores), path_exists=exists)


# --------------------------------------------------------------------------- #
# The three compute topologies
# --------------------------------------------------------------------------- #
def test_single_node_single_gpu_is_the_empty_environment():
    """No launcher vars at all must read as a 1-rank, 1-node local run."""
    t = _resolve()
    assert (t.world_size, t.num_nodes, t.local_world_size) == (1, 1, 1)
    assert (t.rank, t.local_rank) == (0, 0)
    assert t.is_rank_zero and t.is_local_rank_zero
    assert not t.is_distributed
    assert t.execution_mode == "local"


def test_single_node_multi_gpu_splits_cores_across_local_ranks():
    """4 ranks on one node share that node's cores; nodes stay at 1."""
    t = _resolve(
        {"WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "2", "LOCAL_WORLD_SIZE": "4"},
        cores=16,
    )
    assert t.num_nodes == 1
    assert t.is_distributed
    assert not t.is_rank_zero and not t.is_local_rank_zero
    assert t.cpus_per_rank == 4  # 16 cores / 4 local ranks


def test_multi_node_multi_gpu_keeps_the_two_rank_zeros_distinct():
    """The rank-5-of-8 process is local rank 1 on node 2: neither zero.

    ``is_rank_zero`` (one owner per JOB) and ``is_local_rank_zero`` (one owner
    per NODE) must stay separate predicates. Collapsing them is what makes a
    job-global artifact get written once per node, with the survivor decided by
    a race.
    """
    env = {
        "WORLD_SIZE": "8",
        "RANK": "5",
        "LOCAL_RANK": "1",
        "LOCAL_WORLD_SIZE": "4",
        "SLURM_JOB_ID": "12345",
    }
    t = _resolve(env, cores=32)
    assert (t.num_nodes, t.local_world_size) == (2, 4)
    assert not t.is_rank_zero and not t.is_local_rank_zero
    # ...while rank 4 is node 2's owner but NOT the job's owner.
    t4 = _resolve({**env, "RANK": "4", "LOCAL_RANK": "0"}, cores=32)
    assert t4.is_local_rank_zero and not t4.is_rank_zero
    # ...and rank 0 is both.
    t0 = _resolve({**env, "RANK": "0", "LOCAL_RANK": "0"}, cores=32)
    assert t0.is_local_rank_zero and t0.is_rank_zero


# --------------------------------------------------------------------------- #
# Resolution precedence, field by field
# --------------------------------------------------------------------------- #
def test_local_world_size_prefers_torchrun_then_slurm_then_world_size():
    assert _resolve({"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "2"}).local_world_size == 2
    t = _resolve({"WORLD_SIZE": "8", "SLURM_NTASKS_PER_NODE": "4"})
    assert (t.local_world_size, t.sources["local_world_size"]) == (
        4,
        "env:SLURM_NTASKS_PER_NODE",
    )
    t = _resolve({"WORLD_SIZE": "8"})
    assert t.local_world_size == 8 and t.sources["local_world_size"] == "derived:world_size"


def test_local_rank_falls_back_to_slurm_localid_then_arithmetic():
    assert _resolve({"WORLD_SIZE": "4", "RANK": "3", "SLURM_LOCALID": "1",
                     "LOCAL_WORLD_SIZE": "2"}).local_rank == 1
    t = _resolve({"WORLD_SIZE": "4", "RANK": "3", "LOCAL_WORLD_SIZE": "2"})
    assert t.local_rank == 1 and t.sources["local_rank"] == "derived:rank%local"


def test_slurm_nnodes_outranks_the_derived_division():
    """A heterogeneous allocation makes world/local wrong; the declaration wins.

    ``SLURM_CPUS_ON_NODE``'s ``"72(x2)"`` form exists because allocations need
    not be uniform, and when they are not, ``world_size // local_world_size`` is
    simply the wrong node count.
    """
    t = _resolve({"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "4", "SLURM_NNODES": "3"})
    assert t.num_nodes == 3  # not 8 // 4 == 2
    assert t.sources["num_nodes"] == "env:SLURM_NNODES"


@pytest.mark.parametrize(
    ("env", "exists", "expected", "source"),
    [
        ({"SPECTRAMR_LAUNCH_BACKEND": "slurm"}, lambda p: False, "slurm",
         "env:SPECTRAMR_LAUNCH_BACKEND"),
        ({"SLURM_JOB_ID": "1"}, lambda p: False, "slurm", "env:SLURM_JOB_ID"),
        ({"APPTAINER_CONTAINER": "/x.sif"}, lambda p: False, "apptainer",
         "env:APPTAINER_CONTAINER"),
        ({"SINGULARITY_CONTAINER": "/x.sif"}, lambda p: False, "apptainer",
         "env:SINGULARITY_CONTAINER"),
        ({}, lambda p: p == "/.dockerenv", "docker", "probe:/.dockerenv"),
        ({}, lambda p: False, "local", "default"),
    ],
)
def test_execution_mode_covers_every_launcher(env, exists, expected, source):
    t = _resolve(env, exists=exists)
    assert t.execution_mode == expected
    assert t.sources["execution_mode"] == source
    assert t.execution_mode in EXECUTION_MODES


def test_launcher_declaration_outranks_a_container_probe():
    """``spectramr launch`` said slurm; running inside a container must not relabel it."""
    t = _resolve(
        {"SPECTRAMR_LAUNCH_BACKEND": "slurm", "APPTAINER_CONTAINER": "/x.sif"},
        exists=lambda p: p == "/.dockerenv",
    )
    assert t.execution_mode == "slurm"


# --------------------------------------------------------------------------- #
# An impossible environment raises rather than resolving to something plausible
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "env",
    [
        {"WORLD_SIZE": "abc"},  # unparseable — must not read as single-process
        {"WORLD_SIZE": "-1"},
        {"WORLD_SIZE": "2", "RANK": "5"},  # rank outside its world
        {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "4"},  # node bigger than the job
        {"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "2", "LOCAL_RANK": "3"},
        {"SPECTRAMR_LAUNCH_BACKEND": "kubernetes"},  # not in the closed set
    ],
)
def test_inconsistent_environment_raises(env):
    with pytest.raises(TopologyResolutionError):
        _resolve(env)


def test_malformed_world_size_never_degrades_to_one():
    """The specific silent failure this guards: ``WORLD_SIZE=abc`` -> a 1-rank run.

    That would look like a successful single-process job while N-1 siblings were
    also running, each believing the same thing.
    """
    with pytest.raises(TopologyResolutionError, match="not an integer"):
        _resolve({"WORLD_SIZE": "abc"})


# --------------------------------------------------------------------------- #
# CPU share
# --------------------------------------------------------------------------- #
def test_cpus_per_rank_floors_at_one():
    """More ranks than cores must still give each rank a worker, never zero."""
    t = _resolve({"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "8"}, cores=4)
    assert t.cpus_per_rank == 1


def test_cpus_per_rank_handles_a_fractional_cgroup_quota():
    """``cpu.max`` can be fractional (1.5 cores), so ``usable_cores`` can be a float."""
    t = _resolve({"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2"}, cores=2.5)
    assert t.cpus_per_rank == 1
    assert isinstance(t.cpus_per_rank, int)


def test_unprobeable_cpus_raise_on_the_decision_path_but_not_on_the_report():
    """The asymmetry: reporting is fail-open, deciding is not.

    ``cpu_resources`` is written fail-open because it is a reporting function.
    A caller that *acts* on the number must treat ``None`` explicitly rather
    than substituting a default -- silently defaulting is exactly how
    ``num_workers`` came to be multiplied by the rank count.
    """
    t = _resolve(cores=None)
    assert t.cpus_on_node is None
    with pytest.raises(TopologyResolutionError, match="could not be probed"):
        _ = t.cpus_per_rank
    assert t.to_dict()["cpus_per_rank"] is None  # provenance still succeeds


def test_to_dict_carries_the_derived_predicates_and_sources():
    rec = _resolve({"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4", "RANK": "1",
                    "LOCAL_RANK": "1"}).to_dict()
    for key in ("is_distributed", "is_rank_zero", "is_local_rank_zero",
                "cpus_per_rank", "sources"):
        assert key in rec
    assert rec["sources"]["cpus_on_node"] == "probe:cpu_resources.usable_cores"


def test_topology_is_frozen():
    """Observed facts must not be mutated after resolution (non-negotiable 1)."""
    t = _resolve()
    with pytest.raises(FrozenInstanceError):
        t.world_size = 4  # type: ignore[misc]
    assert isinstance(t, RunTopology)


# --- GPU facts (#1274) -----------------------------------------------------
# The SSOT had a CPU inventory and no GPU counterpart, so no consumer of the
# run shape could see the hardware half of "4 GPUs allocated, 1 rank launched".
# These are FACTS only; the refusal that reads them is tested in
# tests/unit/pipelines/test_distributed_pipeline.py, at its consumer.


def test_gpu_fields_default_to_unknown_rather_than_zero():
    """Off-scheduler with an unprobeable GPU, both halves must read as unknown.

    ``0`` allocated would mean "granted nothing", which is a different claim from
    "no scheduler here" — and the launcher branches on exactly that difference.
    """
    t = resolve_run_topology(
        {"WORLD_SIZE": "1", "RANK": "0"},
        cpu_probe=_probe(16),
        gpu_probe=lambda: None,
    )
    assert t.visible_gpus is None
    assert t.allocated_gpus_on_node is None
    assert t.sources["allocated_gpus_on_node"] == "unset:no-scheduler-gpu-grant"


def test_the_incident_shape_is_recorded_in_full():
    """job 17762324: ``--gpus=4`` (job total), one rank, four cards visible."""
    t = resolve_run_topology(
        {"WORLD_SIZE": "1", "RANK": "0", "SLURM_JOB_ID": "17762324", "SLURM_GPUS": "4"},
        cpu_probe=_probe(16),
        gpu_probe=lambda: 4,
    )
    assert (t.world_size, t.local_world_size) == (1, 1)
    assert t.visible_gpus == 4
    assert t.allocated_gpus_on_node == 4
    # The record must say the 4 came from a JOB total that was divided, not from
    # a per-node variable — those are different claims on a multi-node run.
    assert t.sources["allocated_gpus_on_node"] == "derived:SLURM_GPUS/1nodes"


def test_a_job_total_is_normalised_against_the_node_count():
    """2 nodes x 4 GPUs: the field is per-node, so it must read 4 and not 8."""
    t = resolve_run_topology(
        {
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "4",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "SLURM_GPUS": "8",
        },
        cpu_probe=_probe(16),
        gpu_probe=lambda: 4,
    )
    assert t.num_nodes == 2
    assert t.allocated_gpus_on_node == 4


def test_a_per_node_variable_is_not_divided_again():
    """``SLURM_GPUS_ON_NODE`` is already in this field's unit."""
    t = resolve_run_topology(
        {
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "4",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "SLURM_GPUS_ON_NODE": "4",
        },
        cpu_probe=_probe(16),
        gpu_probe=lambda: 4,
    )
    assert t.allocated_gpus_on_node == 4
    assert t.sources["allocated_gpus_on_node"] == "env:SLURM_GPUS_ON_NODE"


def test_the_default_gpu_probe_is_the_core_one():
    """Not a local re-derivation.

    If this module probed for itself, the number stamped into provenance and the
    number ``pipelines/distributed.py`` refuses on would be two implementations
    free to drift -- which is exactly why the CPU probes were moved into
    ``core.resources``. Asserted behaviourally (patch the name this module binds)
    rather than by reading the source, so it survives a refactor that keeps the
    contract.
    """
    from unittest.mock import patch

    with patch("spectramr.core.topology.visible_gpu_count", return_value=7) as probe:
        t = resolve_run_topology({"WORLD_SIZE": "1", "RANK": "0"}, cpu_probe=_probe(16))
    probe.assert_called_once_with()
    assert t.visible_gpus == 7
    assert t.sources["visible_gpus"] == "probe:visible_gpu_count"


def test_gpu_facts_reach_provenance_and_the_startup_line():
    t = resolve_run_topology(
        {"WORLD_SIZE": "1", "RANK": "0", "SLURM_GPUS_ON_NODE": "4"},
        cpu_probe=_probe(16),
        gpu_probe=lambda: 4,
    )
    rec = t.to_dict()
    assert rec["visible_gpus"] == 4
    assert rec["allocated_gpus_on_node"] == 4
    assert "gpus_allocated_on_node=4" in t.describe()


# --------------------------------------------------------------------------- #
# RANK from SLURM_PROCID: a hand-rolled srun, no torchrun
# --------------------------------------------------------------------------- #
def test_rank_falls_back_to_slurm_procid_when_world_size_is_declared():
    t = _resolve({"WORLD_SIZE": "4", "SLURM_PROCID": "3", "LOCAL_WORLD_SIZE": "4"})
    assert (t.rank, t.sources["rank"]) == (3, "env:SLURM_PROCID")
    assert not t.is_rank_zero


def test_rank_env_outranks_slurm_procid():
    t = _resolve({"WORLD_SIZE": "4", "RANK": "1", "SLURM_PROCID": "3"})
    assert (t.rank, t.sources["rank"]) == (1, "env:RANK")


def test_slurm_procid_is_ignored_without_a_declared_world_size():
    """``srun -n 4 python train.py`` is four independent jobs, not four ranks."""
    t = _resolve({"SLURM_PROCID": "3"})
    assert (t.rank, t.world_size, t.sources["rank"]) == (0, 1, "default")


def test_slurm_procid_outside_the_declared_world_is_refused():
    with pytest.raises(TopologyResolutionError, match="outside WORLD_SIZE"):
        _resolve({"WORLD_SIZE": "2", "SLURM_PROCID": "5"})


# --------------------------------------------------------------------------- #
# resolve_rank: the one rule the topology, the console gate and setup_distributed share
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"RANK": "2", "WORLD_SIZE": "4"}, (2, "env:RANK")),
        ({"RANK": "2"}, (2, "env:RANK")),
        ({"WORLD_SIZE": "4", "SLURM_PROCID": "3"}, (3, "env:SLURM_PROCID")),
        ({"WORLD_SIZE": "4", "RANK": "1", "SLURM_PROCID": "3"}, (1, "env:RANK")),
        ({"SLURM_PROCID": "3"}, (None, "unresolved")),
        ({}, (None, "unresolved")),
        # Planted: LOCAL_RANK is a rank within one node, never a global rank.
        ({"WORLD_SIZE": "4", "LOCAL_RANK": "1"}, (None, "unresolved")),
    ],
)
def test_resolve_rank(environ, expected):
    from spectramr.core.topology import resolve_rank

    assert resolve_rank(environ) == expected


def test_resolve_rank_refuses_a_malformed_rank():
    from spectramr.core.topology import resolve_rank

    with pytest.raises(TopologyResolutionError, match="not an integer"):
        resolve_rank({"WORLD_SIZE": "2", "RANK": "abc"})


def test_the_topology_and_the_console_gate_agree_on_every_launch(monkeypatch):
    """Planted divergence guard: both consumers read ``resolve_rank``, so they cannot differ."""
    from spectramr.core import env
    from spectramr.core.topology import resolve_rank

    for environ in (
        {"WORLD_SIZE": "4", "RANK": "2"},
        {"WORLD_SIZE": "4", "SLURM_PROCID": "3"},
        {"WORLD_SIZE": "4", "LOCAL_RANK": "1"},
    ):
        for name in ("RANK", "SLURM_PROCID", "LOCAL_RANK", "WORLD_SIZE"):
            monkeypatch.delenv(name, raising=False)
        for name, value in environ.items():
            monkeypatch.setenv(name, value)
        assert env.distributed_rank() == resolve_rank(environ)[0]
