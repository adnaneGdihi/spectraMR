"""Turn a Scalene profile into ``hotspots.json``: the functions and classes to look at.

Each function Scalene reports becomes one row carrying its measured cost
(CPU split into Python / native / system, GPU, memory peak, copy volume), the
share of CPU it spent under training vs validation, and the static rule findings
from :mod:`profile_static` for the same function. A row is listed when a
measured threshold trips or a static finding sits in it; classes are the
roll-up of their methods.

Scalene's percentages are shares of the profiled wall time, never seconds. When
a mode did not collect a column (``--gpu`` turns memory off, ``--cpu-only`` turns
both off) that column is ``null`` and its flag cannot fire: absent is reported,
not read as zero.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from spectramr.cli.profile_phases import PHASES, classify_stack, parse_frame
from spectramr.cli.profile_static import RULES, StaticFinding, scan_file

SCHEMA = "spectramr.profile_hotspots/1"

#: Measured-flag thresholds, stamped into the report so a reader can re-derive it.
THRESHOLDS: dict[str, float] = {
    "cpu_hot_percent": 5.0,
    "python_bound_min_cpu_percent": 2.0,
    "python_bound_share": 0.7,
    "system_heavy_percent": 2.0,
    "memory_peak_mb": 256.0,
    "copy_heavy_mb_s": 100.0,
    "gpu_hot_percent": 5.0,
}

MAX_FUNCTIONS = 100
MAX_STATIC_ONLY = 50
MAX_CLASSES = 30


def _relative(path: str) -> str:
    """Path from the package root (``pipelines/train.py``), or unchanged if absent."""
    _, sep, tail = path.rpartition("/spectramr/")
    return tail if sep else path


def _class_of(qualname: str) -> str | None:
    parts = qualname.split(".")
    if len(parts) < 2 or "<locals>" in parts:
        return None
    return ".".join(parts[:-1])


def _phase_shares(profile: dict[str, Any]) -> dict[tuple[str, str], dict[str, float]]:
    """Leaf-frame CPU share per ``(relative file, function)``, split by phase."""
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: dict.fromkeys(PHASES, 0.0))
    for entry in profile.get("stacks") or []:
        if not (isinstance(entry, list) and len(entry) == 2):
            continue
        frames, payload = entry
        leaf = next((f for f in (parse_frame(r) for r in reversed(frames)) if f is not None), None)
        if leaf is None:
            continue
        out[(_relative(leaf.filename), leaf.function)][classify_stack(frames)] += float(
            payload.get("cpu_samples", 0.0) or 0.0
        )
    return out


def _measured(fn: dict[str, Any], *, gpu: bool, memory: bool) -> dict[str, Any]:
    python = float(fn.get("n_cpu_percent_python", 0.0) or 0.0)
    native = float(fn.get("n_cpu_percent_c", 0.0) or 0.0)
    system = float(fn.get("n_sys_percent", 0.0) or 0.0)
    return {
        "cpu_percent": round(python + native + system, 3),
        "python_percent": round(python, 3),
        "native_percent": round(native, 3),
        "system_percent": round(system, 3),
        "gpu_percent": round(float(fn.get("n_gpu_percent", 0.0) or 0.0), 3) if gpu else None,
        "peak_mb": round(float(fn.get("n_peak_mb", 0.0) or 0.0), 3) if memory else None,
        "copy_mb_s": round(float(fn.get("n_copy_mb_s", 0.0) or 0.0), 3) if memory else None,
    }


def _flags(m: dict[str, Any]) -> list[str]:
    t, cpu = THRESHOLDS, m["cpu_percent"]
    flags = []
    if cpu >= t["cpu_hot_percent"]:
        flags.append("cpu_hot")
    if (
        cpu >= t["python_bound_min_cpu_percent"]
        and m["python_percent"] / cpu >= t["python_bound_share"]
    ):
        flags.append("python_bound")
    if m["system_percent"] >= t["system_heavy_percent"]:
        flags.append("system_heavy")
    if m["gpu_percent"] is not None and m["gpu_percent"] >= t["gpu_hot_percent"]:
        flags.append("gpu_hot")
    if m["peak_mb"] is not None and m["peak_mb"] >= t["memory_peak_mb"]:
        flags.append("memory_peak")
    if m["copy_mb_s"] is not None and m["copy_mb_s"] >= t["copy_heavy_mb_s"]:
        flags.append("copy_heavy")
    return flags


def _finding_row(f: StaticFinding) -> dict[str, Any]:
    return {"rule": f.rule, "lineno": f.lineno, "detail": f.detail}


def _static_by_function(
    files: list[str],
) -> tuple[dict[tuple[str, str], list[StaticFinding]], list[str]]:
    found: dict[tuple[str, str], list[StaticFinding]] = defaultdict(list)
    skipped: list[str] = []
    for filename in files:
        try:
            findings = scan_file(Path(filename))
        except (OSError, SyntaxError):
            skipped.append(_relative(filename))
            continue
        for f in findings:
            found[(_relative(filename), f.function)].append(f)
    return found, skipped


def _class_rows(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for fn in functions:
        if fn["class"] is not None:
            grouped[(fn["file"], fn["class"])].append(fn)
    rows = []
    for (file, cls), methods in grouped.items():
        gpu = [m["gpu_percent"] for m in methods if m["gpu_percent"] is not None]
        peaks = [m["peak_mb"] for m in methods if m["peak_mb"] is not None]
        rows.append(
            {
                "file": file,
                "class": cls,
                "cpu_percent": round(sum(m["cpu_percent"] for m in methods), 3),
                "gpu_percent": round(sum(gpu), 3) if gpu else None,
                "peak_mb": max(peaks) if peaks else None,
                "methods_listed": len(methods),
                "methods": [m["function"] for m in methods],
            }
        )
    rows.sort(key=lambda r: r["cpu_percent"] + (r["gpu_percent"] or 0.0), reverse=True)
    return rows[:MAX_CLASSES]


def build_hotspot_report(profile: dict[str, Any], *, profile_path: str = "") -> dict[str, Any]:
    """Build the report dict from one loaded Scalene profile."""
    gpu, memory = bool(profile.get("gpu")), bool(profile.get("memory"))
    files = profile.get("files") or {}
    static, skipped = _static_by_function(list(files))
    phases = _phase_shares(profile)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for filename, body in files.items():
        rel = _relative(filename)
        for fn in body.get("functions") or []:
            name = fn.get("line")
            if not isinstance(name, str):
                continue
            seen.add((rel, name))
            m = _measured(fn, gpu=gpu, memory=memory)
            flags, findings = _flags(m), static.get((rel, name), [])
            if not flags and not findings:
                continue
            rows.append(
                {
                    "file": rel,
                    "function": name,
                    "class": _class_of(name),
                    "lineno": fn.get("lineno"),
                    **m,
                    "phase_cpu_share": {
                        k: round(v, 6) for k, v in phases.get((rel, name), {}).items()
                    }
                    or None,
                    "measured_flags": flags,
                    "static_findings": [_finding_row(f) for f in findings],
                    "score": round(m["cpu_percent"] + (m["gpu_percent"] or 0.0), 3),
                }
            )
    rows.sort(key=lambda r: (r["score"], len(r["static_findings"])), reverse=True)

    cold = [
        {"file": rel, "function": name, "static_findings": [_finding_row(f) for f in fs]}
        for (rel, name), fs in static.items()
        if (rel, name) not in seen
    ]
    return {
        "schema": SCHEMA,
        "source": {
            "profile": profile_path,
            "elapsed_time_sec": profile.get("elapsed_time_sec"),
            "gpu_profiled": gpu,
            "memory_profiled": memory,
        },
        "metric": "percent of profiled wall time (scalene n_*_percent); NOT seconds",
        "thresholds": THRESHOLDS,
        "rules": RULES,
        "static_scan_skipped": skipped,
        "summary": {
            "functions_flagged": len(rows),
            "functions_truncated": max(0, len(rows) - MAX_FUNCTIONS),
            "static_only": len(cold),
        },
        "functions": rows[:MAX_FUNCTIONS],
        "classes": _class_rows(rows[:MAX_FUNCTIONS]),
        "static_only": cold[:MAX_STATIC_ONLY],
    }


def write_hotspots(profile_path: Path, out_path: Path) -> dict[str, Any]:
    """Read ``profile_path``, write ``out_path``; return the manifest record."""
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    report = build_hotspot_report(profile, profile_path=str(profile_path))
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"status": "written", "path": str(out_path), **report["summary"]}


__all__ = ["THRESHOLDS", "build_hotspot_report", "write_hotspots"]
