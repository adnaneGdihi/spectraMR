"""Every n2n and self_supervised arm writes inside its cohort's ``_v2`` tree.

Protocol v2 keeps the first run readable by writing to
``experiments/results/<cohort>_v2/<arm>`` (#2542). An arm that writes elsewhere
either overwrites a v1 artifact or lands where the runbook and the reporting
scripts do not look; three control arms added after the split did the latter.
The output and log directories must also agree with each other, and no arm
declares a checkpoint path: checkpoints follow ``training.output_dir``.

The predicate is exercised against planted violations (non-negotiable 15).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_INPROGRESS = Path(__file__).resolve().parents[3] / "experiments" / "inprogress"
_COHORTS = ("n2n", "self_supervised")
_ARMS = sorted(arm for cohort in _COHORTS for arm in (_INPROGRESS / cohort).glob("*.yaml"))


def _result_tree_error(cfg: dict[str, Any], cohort: str, arm: str) -> str | None:
    root = f"experiments/results/{cohort}_v2/{arm}"
    # Checkpoints follow output_dir; a checkpoint path key is retired and refused (#2842).
    for retired in ("checkpoint_dir", "save_dir"):
        if retired in (cfg.get("checkpoint") or {}):
            return f"checkpoint.{retired} is declared; checkpoints follow training.output_dir"
    paths = {
        "training.output_dir": ((cfg.get("training") or {}).get("output_dir"), root),
    }
    log_dir = (cfg.get("logging") or {}).get("dir")
    if log_dir is not None:
        paths["logging.dir"] = (log_dir, f"{root}/logs")
    for key, (declared, expected) in paths.items():
        if declared != expected:
            return f"{key} is {declared!r}; expected {expected!r}"
    return None


@pytest.mark.parametrize("arm", _ARMS, ids=[a.stem for a in _ARMS])
def test_arm_writes_inside_its_v2_tree(arm: Path) -> None:
    cfg = yaml.safe_load(arm.read_text()) or {}
    error = _result_tree_error(cfg, arm.parent.name, arm.stem)
    assert error is None, error


def test_both_cohorts_are_scanned() -> None:
    assert {a.parent.name for a in _ARMS} == set(_COHORTS)


@pytest.mark.parametrize(
    ("block", "key", "value", "reason"),
    [
        ("training", "output_dir", "experiments/results/n2n/{arm}", "training.output_dir"),
        ("training", "output_dir", "experiments/results/{arm}", "training.output_dir"),
        ("checkpoint", "checkpoint_dir", "./checkpoints", "checkpoint.checkpoint_dir"),
        ("logging", "dir", "experiments/results/{arm}/logs", "logging.dir"),
    ],
)
def test_the_predicate_rejects_a_planted_path(
    block: str, key: str, value: str, reason: str
) -> None:
    arm = _ARMS[0]
    cfg = yaml.safe_load(arm.read_text()) or {}
    assert _result_tree_error(cfg, arm.parent.name, arm.stem) is None
    cfg.setdefault(block, {})[key] = value.format(arm=arm.stem)
    error = _result_tree_error(cfg, arm.parent.name, arm.stem)
    assert error is not None and error.startswith(reason), error
