"""Which profiler ``spectramr profile`` runs: Scalene, or ``torch.profiler`` when Scalene's GPU mode is unsuitable.

Scalene's ``--gpu`` mode was measured at 12-25x the unprofiled time, starts a stack-recording torch
profiler that crashes intermittently, and stalls on GPUs in a restricted compute mode. The torch backend
(:mod:`spectramr.cli.profile_torch_runner`) attaches one ``torch.profiler`` window to the training steps
instead. It reports GPU kernel time but has no line-level CPU/memory columns, no phase split and no
``hotspots.json``; the manifest says so rather than leaving those files to be missed.

The choice is a request resolved once, to a :class:`BackendChoice` that carries its reason, so the manifest
records both what was asked and what ran. Nothing here degrades silently: an explicit backend whose
requirement is absent raises, and ``auto`` states why it picked what it picked.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

BACKEND_CHOICES = ("auto", "scalene", "torch")

#: Steps skipped and discarded before the window opens. The first 8 training steps write debug
#: snapshots with the GPU idle, so a window opened earlier averages in a one-off cost.
TORCH_WAIT_STEPS = 10
TORCH_WARMUP_STEPS = 5
TORCH_MAX_ACTIVE_STEPS = 20
#: The smallest ``--iterations`` that leaves at least one recorded step after wait + warmup.
TORCH_MIN_ITERATIONS = TORCH_WAIT_STEPS + TORCH_WARMUP_STEPS + 1


@dataclass(frozen=True)
class BackendChoice:
    """The resolved backend, with what was requested and why this one."""

    requested: str
    applied: str
    reason: str


def scalene_gpu_flag(mode_flags: tuple[str, ...]) -> bool:
    """Whether the mode's Scalene flags leave its GPU sampler on.

    No flag (``--mode full``) samples CPU, GPU and memory; ``--gpu`` keeps the GPU sampler; the other
    flags turn it off (see the table beside ``PROFILE_MODES``).
    """
    return not mode_flags or "--gpu" in mode_flags


def select_backend(
    requested: str,
    *,
    mode_flags: tuple[str, ...],
    cuda_available: bool,
    scalene_available: bool,
) -> BackendChoice:
    """Resolve ``requested`` to a backend, or raise when its requirement is absent.

    ``auto`` picks torch when Scalene would sample the GPU on a CUDA machine, or when Scalene is not
    installed but CUDA is visible; otherwise Scalene.
    """
    if requested not in BACKEND_CHOICES:
        raise ValueError(f"--backend must be one of {BACKEND_CHOICES}, got {requested!r}.")
    if requested == "torch":
        if not cuda_available:
            raise RuntimeError(
                "--backend torch profiles GPU time and no CUDA device is visible; it does not "
                "fall back to the CPU. Use --backend scalene --mode cpu-only for a CPU profile."
            )
        return BackendChoice(requested, "torch", "requested explicitly")
    if requested == "scalene":
        return BackendChoice(requested, "scalene", "requested explicitly")
    if not scalene_available:
        if not cuda_available:
            # Neither can run; Scalene is returned so its own missing-install error (with the
            # install command) is the one raised, and a dry run can still print its command.
            return BackendChoice(
                requested, "scalene", "scalene is not installed and no CUDA device is visible"
            )
        return BackendChoice(requested, "torch", "scalene is not installed")
    if scalene_gpu_flag(mode_flags) and cuda_available:
        return BackendChoice(
            requested,
            "torch",
            "the mode samples the GPU through Scalene, which runs 12-25x slower and crashes "
            "intermittently; use --backend scalene to force it",
        )
    return BackendChoice(requested, "scalene", "the mode needs no GPU sampling")


def torch_window(iterations: int | None) -> tuple[int, int, int]:
    """``(wait, warmup, active)`` for a run capped at ``iterations`` training steps."""
    if iterations is None or iterations < TORCH_MIN_ITERATIONS:
        raise ValueError(
            f"--backend torch needs --iterations >= {TORCH_MIN_ITERATIONS} (a {TORCH_WAIT_STEPS}-step "
            f"wait and {TORCH_WARMUP_STEPS}-step warm-up come before the recorded steps), "
            f"got {iterations}."
        )
    active = min(TORCH_MAX_ACTIVE_STEPS, iterations - TORCH_WAIT_STEPS - TORCH_WARMUP_STEPS)
    return TORCH_WAIT_STEPS, TORCH_WARMUP_STEPS, active


def build_torch_command(child: list[str], *, out_dir: Path, iterations: int | None) -> list[str]:
    """``python -m spectramr.cli.profile_torch_runner --out ... -- <verb> ...``; pure."""
    wait, warmup, active = torch_window(iterations)
    return [
        sys.executable,
        "-m",
        "spectramr.cli.profile_torch_runner",
        "--out",
        str(out_dir),
        "--wait",
        str(wait),
        "--warmup",
        str(warmup),
        "--active",
        str(active),
        "--",
        *child,
    ]


#: The artifact whose presence says the torch backend produced a profile.
TORCH_SUMMARY_FILE = "summary.json"

_TORCH_NO_DERIVED_REPORTS = (
    "the torch backend records GPU kernel time per operator (kernels.txt, trace.json, summary.json); "
    "it has no Scalene line profile to split by phase or rank into hotspots"
)


def cuda_visible() -> bool:
    """Whether a CUDA device is visible, importing torch only when asked."""
    import torch

    return bool(torch.cuda.is_available())


def stamp_backend(
    manifest: dict,
    choice: BackendChoice,
    *,
    window: tuple[int, int, int] | None,
    fell_back_from: str | None = None,
) -> None:
    """Record the request and the outcome, so the manifest says which profiler's numbers these are."""
    manifest["modes"]["backend_requested"] = choice.requested
    manifest["modes"]["backend_applied"] = choice.applied
    manifest["modes"]["backend_reason"] = choice.reason
    manifest["modes"]["backend_fell_back_from"] = fell_back_from
    if choice.applied == "torch":
        import torch

        manifest["scalene"] = None
        manifest["torch_profiler"] = {
            "version": torch.__version__,
            "wait_warmup_active": list(window) if window else None,
        }
        manifest["phase_split"] = {"status": "unavailable", "reason": _TORCH_NO_DERIVED_REPORTS}
        manifest["hotspots"] = {"status": "unavailable", "reason": _TORCH_NO_DERIVED_REPORTS}


__all__ = [
    "BACKEND_CHOICES",
    "TORCH_MIN_ITERATIONS",
    "TORCH_SUMMARY_FILE",
    "BackendChoice",
    "build_torch_command",
    "cuda_visible",
    "scalene_gpu_flag",
    "select_backend",
    "stamp_backend",
    "torch_window",
]
