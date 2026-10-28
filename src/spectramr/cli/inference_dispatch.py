"""The per-task launch pieces of ``campaign infer``: weights to grade and where the pass writes.

An inference task is the arm's own ``train --validate-only`` on one resolved checkpoint, so the run
scores the held-out test split exactly as the training run's own test pass did. It writes into
``<base>/<stem>/inference`` and never into the arm's run directory.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from spectramr.core.run_artifacts import INFERENCE_SUBDIR

SOURCE_JSON = "inference_source.json"


def checkpoint_for_task(checkpoints_file: str | Path, index: int, what: str = "checkpoint") -> str:
    """Line ``index`` of a frozen per-task list aligned with the manifest; a short or blank line raises."""
    lines = Path(checkpoints_file).read_text().splitlines()
    if not 0 <= index < len(lines) or not lines[index].strip():
        raise IndexError(
            f"{checkpoints_file} has no {what} for task {index} ({len(lines)} lines); "
            "the list is frozen next to the manifest and must stay aligned with it."
        )
    return lines[index].strip()


def inference_run_dir(output_base: str | Path, config: str | Path) -> Path:
    return Path(output_base) / Path(config).stem / INFERENCE_SUBDIR


def inference_verb_args(checkpoint: str) -> list[str]:
    """Flags that make ``train`` grade ``checkpoint`` once; an explicit path is never ``if-present``."""
    return ["--validate-only", "--resume", checkpoint]


def write_source(run_dir: Path, *, checkpoint: str, config: str) -> Path:
    """Stamp what was graded before the pass starts, so a crashed task still says so."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / SOURCE_JSON
    path.write_text(
        json.dumps({"checkpoint": checkpoint, "config": config, "git_head": head or None}, indent=2)
    )
    return path


__all__ = [
    "SOURCE_JSON",
    "checkpoint_for_task",
    "inference_run_dir",
    "inference_verb_args",
    "write_source",
]
