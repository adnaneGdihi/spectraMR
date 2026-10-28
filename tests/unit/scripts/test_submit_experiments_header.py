"""``submit_experiments.sh`` requests an untyped GPU and tolerates an unset PYTHONPATH.

The clusters mix Ada and V100 nodes, so a typed ``--gres`` pins one family (#2650), and the
rendered job script may run under ``set -u``, where a bare ``${PYTHONPATH}`` aborts it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts/training/submit_experiments.sh"

_TYPED_GPU = re.compile(r"\bGPUS=\w+:\d+|--gres=gpu:[A-Za-z]\w*:\d+")
_BARE_PYTHONPATH = re.compile(r"\$\{PYTHONPATH\}")


def test_the_gpu_request_names_no_type() -> None:
    assert _TYPED_GPU.search(SCRIPT.read_text()) is None


def test_pythonpath_expands_with_a_default() -> None:
    assert _BARE_PYTHONPATH.search(SCRIPT.read_text()) is None


def test_the_detectors_turn_red_on_a_planted_violation() -> None:
    assert _TYPED_GPU.search("    GPUS=ada:1")
    assert _TYPED_GPU.search("#SBATCH --gres=gpu:v100:2")
    assert _BARE_PYTHONPATH.search("export PYTHONPATH=src:${PYTHONPATH}")
    assert not _TYPED_GPU.search("#SBATCH --gres=gpu:${GPUS}")
