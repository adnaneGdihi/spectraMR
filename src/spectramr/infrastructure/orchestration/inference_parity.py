"""Does an arm's re-run test pass reproduce its training-time one?

``campaign infer`` is first-run code on these arms, so its result is only trusted for an arm whose
``inference/final_eval.json`` matches the ``final_eval.json`` the training run wrote, case by case.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from spectramr.core.run_artifacts import INFERENCE_SUBDIR

FINAL_EVAL = "final_eval.json"


@dataclass
class ParityResult:
    arm: str
    ok: bool
    reason: str | None = None
    #: metric -> (cases compared, largest absolute difference, largest relative difference)
    metrics: dict[str, tuple[int, float, float]] = field(default_factory=dict)


def _cases(payload: dict) -> dict[str, dict[str, float]]:
    """``{metric: {case: value}}``; a scalar (a run without per-case rows) becomes one case."""
    return {m: v if isinstance(v, dict) else {"mean": float(v)} for m, v in payload.items()}


def check_arm(run_dir: str | Path, *, rtol: float = 1e-3, atol: float = 1e-6) -> ParityResult:
    run = Path(run_dir)
    arm = run.name
    ref_path, new_path = run / FINAL_EVAL, run / INFERENCE_SUBDIR / FINAL_EVAL
    for path in (ref_path, new_path):
        if not path.is_file():
            return ParityResult(arm, False, f"missing {path}")
    ref, new = _cases(json.loads(ref_path.read_text())), _cases(json.loads(new_path.read_text()))
    if set(ref) != set(new):
        return ParityResult(arm, False, f"metric sets differ: {sorted(set(ref) ^ set(new))}")
    result = ParityResult(arm, True)
    for metric in sorted(ref):
        a, b = ref[metric], new[metric]
        if set(a) != set(b):
            result.ok = False
            result.reason = f"{metric}: case ids differ ({len(set(a) ^ set(b))} not shared)"
            return result
        diffs = [(abs(a[c] - b[c]), abs(a[c] - b[c]) / max(abs(a[c]), atol)) for c in a]
        worst_abs = max((d for d, _ in diffs), default=0.0)
        worst_rel = max((r for _, r in diffs), default=0.0)
        result.metrics[metric] = (len(a), worst_abs, worst_rel)
        if any(not math.isclose(a[c], b[c], rel_tol=rtol, abs_tol=atol) for c in a):
            result.ok = False
            result.reason = result.reason or f"{metric}: differs beyond rtol={rtol}, atol={atol}"
    return result


__all__ = ["ParityResult", "check_arm"]
