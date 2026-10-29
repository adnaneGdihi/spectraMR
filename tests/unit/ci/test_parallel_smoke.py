"""The multi-GPU smoke harness's verdict: what makes a scenario fail.

The harness itself needs real GPUs (scripts/ci/run_parallel_gpu_smoke.sbatch); its
judgement is pure, so each failure shape it claims to catch is planted here.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tests.utils.repo_scripts import require_repo_file

pytest.importorskip("torch")
# The script imports its sibling parallel_smoke_scenarios, so its directory goes on the path.
sys.path.insert(0, str(require_repo_file("scripts/diagnostics/parallel_smoke.py").parent))

from parallel_smoke import _judge

ARGS = SimpleNamespace(min_gpu_util=60.0, max_straggler=1.25)


def _row(ms: float = 10.0, util: float | None = 90.0) -> dict:
    return {"step_ms_median": ms, "gpu_util_mean": util}


def test_healthy_ranks_pass() -> None:
    assert _judge("ddp", [_row(), _row(10.5)], ARGS, cuda=True) == []


def test_a_rank_that_raised_fails_the_scenario() -> None:
    problems = _judge("ddp", [_row(), {"error": "RuntimeError: boom"}], ARGS, cuda=True)
    assert problems == ["ddp: rank 1 raised: RuntimeError: boom"]


def test_an_underused_gpu_fails() -> None:
    problems = _judge("ddp", [_row(), _row(util=12.0)], ARGS, cuda=True)
    assert len(problems) == 1 and "rank 1 GPU utilisation 12.0" in problems[0]


def test_a_gpu_never_sampled_fails_rather_than_passing() -> None:
    """No NVML reading is not evidence the GPU was busy."""
    assert _judge("ddp", [_row(), _row(util=None)], ARGS, cuda=True)


def test_a_straggler_fails() -> None:
    problems = _judge("ddp", [_row(10.0), _row(10.0), _row(30.0)], ARGS, cuda=True)
    assert len(problems) == 1 and "rank 2 straggles" in problems[0]


def test_timing_and_utilisation_are_not_judged_on_cpu() -> None:
    """CPU ranks share cores; only correctness (errors) decides there."""
    rows = [_row(10.0, util=None), _row(40.0, util=None)]
    assert _judge("ddp", rows, ARGS, cuda=False) == []
    assert _judge("ddp", [_row(), {"error": "E"}], ARGS, cuda=False)
