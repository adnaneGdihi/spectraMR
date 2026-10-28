"""No two arms of the three kspace_filling trees write to the same place or share a label.

The M4Raw tree and its fastMRI brain and knee ports are launched together, so an
output directory, log directory, run identity or figure label shared between an arm
and its port would make one run overwrite or be read as the other. Data sources are
read-only and are shared on purpose. These tests read YAML only.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import yaml

_INPROGRESS = Path(__file__).resolve().parents[3] / "experiments" / "inprogress"
_TREES = ("kspace_filling", "kspace_filling_fastmri_brain", "kspace_filling_fastmri_knee")

#: Dotted keys that name where a run writes, or how a figure labels it.
_UNIQUE_KEYS = (
    "training.output_dir",
    "metrics.output_dir",
    "loss_logging.output_dir",
    "logging.sinks.dir",
    "logging.identity.experiment",
    "logging.identity.run",
    "reporting.method_name",
    "metadata.name",
)


def _get(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        node = node.get(key) if isinstance(node, dict) else None
    return node


def _arms() -> dict[Path, dict]:
    out = {}
    for tree in _TREES:
        for path in sorted((_INPROGRESS / tree).rglob("*.yaml")):
            cfg = yaml.safe_load(path.read_text())
            if isinstance(cfg, dict) and cfg.get("model"):
                out[path] = cfg
    return out


def _collisions(arms: dict[Path, dict]) -> list[str]:
    seen: dict[tuple[str, Any], list[str]] = defaultdict(list)
    for path, cfg in arms.items():
        for key in _UNIQUE_KEYS:
            value = _get(cfg, key)
            if value is not None:
                seen[(key, value)].append(path.name)
    return [f"{k}={v!r}: {names}" for (k, v), names in seen.items() if len(names) > 1]


@pytest.mark.skipif(not (_INPROGRESS / _TREES[0]).is_dir(), reason="corpus not present")
def test_no_two_arms_share_an_output_or_a_label() -> None:
    arms = _arms()
    assert len(arms) > 200
    assert not _collisions(arms)


def test_an_arm_and_its_port_with_one_label_are_caught() -> None:
    """PLANTED: a port that kept its source's method_name, and one that kept its output_dir."""
    source = {
        "training": {"output_dir": "experiments/results/a"},
        "reporting": {"method_name": "a"},
    }
    label = {"training": {"output_dir": "experiments/results/b"}, "reporting": {"method_name": "a"}}
    path = {"training": {"output_dir": "experiments/results/a"}, "reporting": {"method_name": "c"}}
    assert len(_collisions({Path("s.yaml"): source, Path("p.yaml"): label})) == 1
    assert len(_collisions({Path("s.yaml"): source, Path("p.yaml"): path})) == 1
