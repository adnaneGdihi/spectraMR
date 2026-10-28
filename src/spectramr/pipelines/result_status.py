"""What a training pipeline's result says about whether the run succeeded.

``run_training_pipeline`` returns a dict and reports most failures by returning
``{"success": False, ...}`` rather than raising. Every entry point therefore has to turn that
dict into an exit code, and the rule must be the same everywhere: only an explicit ``True`` is
success, and an absent flag is *unknown*, not a pass (#181). ``spectramr train`` applied it;
``train-distributed`` returned 0 unconditionally, so a failed run under ``torchrun`` (a
resume that found no checkpoint, a pipeline that refused to build) reported "Training
completed" and let a Slurm script go on to inference on a checkpoint that was never written.
"""

from __future__ import annotations

from typing import Any


def failure_reason(result: Any) -> str | None:
    """Why ``result`` is not a success, or ``None`` when it is one."""
    if isinstance(result, dict) and result.get("success") is True:
        return None
    if isinstance(result, dict):
        return str(result.get("error", "Unknown pipeline error"))
    return f"pipeline returned {type(result).__name__}, not a result dict"
