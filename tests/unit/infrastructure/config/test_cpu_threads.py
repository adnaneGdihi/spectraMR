"""SPECTRAMR_CPU_THREADS: override, else the SLURM grant per rank, else 1."""

from __future__ import annotations

import pytest

from spectramr.infrastructure.config.cpu_threads import ENV_VAR, resolve_cpu_threads

pytestmark = pytest.mark.unit


def test_defaults_to_one_outside_a_scheduler_allocation() -> None:
    assert resolve_cpu_threads({}) == 1


def test_blank_override_falls_through_to_the_grant() -> None:
    assert resolve_cpu_threads({ENV_VAR: "  ", "SLURM_CPUS_PER_TASK": "6"}) == 6


def test_inherits_the_slurm_grant() -> None:
    assert resolve_cpu_threads({"SLURM_CPUS_PER_TASK": "8"}) == 8


def test_grant_is_split_across_the_ranks_on_the_node() -> None:
    env = {"SLURM_CPUS_PER_TASK": "16", "LOCAL_WORLD_SIZE": "4"}
    assert resolve_cpu_threads(env) == 4


def test_more_ranks_than_cores_still_gets_one_thread() -> None:
    env = {"SLURM_CPUS_PER_TASK": "2", "LOCAL_WORLD_SIZE": "4"}
    assert resolve_cpu_threads(env) == 1


def test_explicit_override_beats_the_grant() -> None:
    assert resolve_cpu_threads({ENV_VAR: "3", "SLURM_CPUS_PER_TASK": "8"}) == 3


@pytest.mark.parametrize("bad", ["0", "-2", "many", "1.5"])
def test_rejects_a_malformed_value(bad: str) -> None:
    with pytest.raises(ValueError, match=ENV_VAR):
        resolve_cpu_threads({ENV_VAR: bad, "SLURM_CPUS_PER_TASK": "8"})
