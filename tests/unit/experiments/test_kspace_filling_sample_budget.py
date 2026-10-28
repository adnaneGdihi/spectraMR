"""Every kspace_filling arm trains on the same number of samples and selects on fixed batches.

Arms that run batch 1 x accumulation 4 (for memory) used to share the 2 x 2 arms'
70000-iteration budget, so they saw half the samples and half the optimizer steps,
and every delta against them carried that difference. The budget and the validation
cadence are counted in loader iterations, so each must be equal across arms once
multiplied by the batch size. Validation must not shuffle:
the selector averages per-batch metrics, and a reshuffled pairing makes it move
between selection points without the model changing. The three baselines follow
their own papers' protocols and are exempt. These tests read YAML; none builds a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_INPROGRESS = Path(__file__).resolve().parents[3] / "experiments" / "inprogress"
_TREES = ("kspace_filling", "kspace_filling_fastmri_brain", "kspace_filling_fastmri_knee")

#: Loader-iteration counters that must scale with the batch size, as dotted paths.
_ITERATION_COUNTERS = ("training.max_iterations", "validation.schedule.interval_steps")


def _get(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        node = node.get(key) if isinstance(node, dict) else None
    return node


def _arms(tree: str) -> dict[Path, dict]:
    out = {}
    for path in sorted((_INPROGRESS / tree).rglob("*.yaml")):
        if "baseline_" in path.name:
            continue
        cfg = yaml.safe_load(path.read_text())
        if isinstance(cfg, dict) and _get(cfg, "training.max_iterations"):
            out[path] = cfg
    return out


def _sample_counts(cfg: dict) -> dict[str, int]:
    batch = int(_get(cfg, "training.batch_size"))
    return {path: int(_get(cfg, path)) * batch for path in _ITERATION_COUNTERS}


def _budget_violations(arms: dict[Path, dict]) -> list[str]:
    counts = {path: _sample_counts(cfg) for path, cfg in arms.items()}
    reference = next(iter(counts.values()))
    return [
        f"{path.name}: {key}={value // int(_get(arms[path], 'training.batch_size'))} "
        f"iterations x batch = {value} samples, not {reference[key]}"
        for path, per_arm in counts.items()
        for key, value in per_arm.items()
        if value != reference[key]
    ]


@pytest.mark.parametrize("tree", _TREES)
def test_every_arm_sees_the_same_samples(tree: str) -> None:
    arms = _arms(tree)
    if not arms:
        pytest.skip(f"{tree} not present")
    assert not _budget_violations(arms)


@pytest.mark.parametrize("tree", _TREES)
def test_validation_batches_are_fixed(tree: str) -> None:
    shuffled = [
        path.name
        for path, cfg in _arms(tree).items()
        if _get(cfg, "validation.loader.shuffle") is not False
    ]
    assert not shuffled, f"validation.loader.shuffle is not false on {shuffled}"


def _arm(batch: int, iterations: int, interval: int) -> dict:
    return {
        "training": {"batch_size": batch, "max_iterations": iterations},
        "validation": {"schedule": {"interval_steps": interval}},
    }


def test_a_batch_one_arm_on_the_batch_two_budget_is_caught() -> None:
    """PLANTED: the shape this file exists for, on each counter."""
    good = {
        Path("a.yaml"): _arm(2, 70000, 10000),
        Path("b.yaml"): _arm(1, 140000, 20000),
    }
    assert _budget_violations(good) == []
    for planted in (_arm(1, 70000, 20000), _arm(1, 140000, 10000)):
        arms = {Path("a.yaml"): _arm(2, 70000, 10000), Path("b.yaml"): planted}
        assert len(_budget_violations(arms)) == 1, planted
