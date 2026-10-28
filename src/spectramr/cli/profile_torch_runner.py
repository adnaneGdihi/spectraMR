"""The in-process ``torch.profiler`` launcher behind ``spectramr profile --backend torch``.

Scalene's ``--gpu`` samples through NVML and starts a stack-recording torch profiler every
100 steps, which crashes intermittently and stalls on GPUs set to a restricted compute mode.
This launcher runs the verb in this process and attaches one ``torch.profiler`` window to the
training steps: ``wait`` steps skipped, ``warmup`` discarded, ``active`` recorded.

It writes, under ``--out``: ``kernels.txt`` (top operators by device time, then by CPU time),
``trace.json`` (open in chrome://tracing or Perfetto), and ``summary.json`` (wall time of the
active window, summed kernel time, and their ratio). The ratio sums kernels across streams, so
it can pass 1.0 only if kernels overlap on several streams.

The first 8 training steps write debug snapshots (about 0.8 s each, GPU idle), so a window that
starts earlier than step 9 averages in a one-off cost. Keep ``--wait`` + ``--warmup`` >= 8 and
end the window before the first validation step.

Usage (from the repo root, ``PYTHONPATH=src``; everything after ``--`` goes to the verb)::

    python -m spectramr.cli.profile_torch_runner --out /tmp/tp \
        -- train --config experiments/inprogress/self_supervised/ssdu_m4raw_r4.yaml \
        --override training.max_iterations=80 \
        --override validation.schedule.interval_steps=40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _device_us(event: Any) -> float:
    """Self device time in microseconds; the attribute was renamed across torch releases."""
    for name in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(event, name):
            return float(getattr(event, name))
    return 0.0


def summarise(prof: Any, wall_s: float, steps: int) -> dict[str, float]:
    """Wall seconds of the window, device seconds, and their ratio.

    The device total is torch's own ``total_average`` (the table footer): summing the
    ``key_averages`` rows counts a kernel under its op and again under ``ProfilerStep*``.
    """
    kernel_s = _device_us(prof.key_averages().total_average()) / 1e6
    return {
        "active_steps": float(steps),
        "wall_s": wall_s,
        "device_kernel_s": kernel_s,
        "device_over_wall": kernel_s / wall_s if wall_s > 0 else 0.0,
    }


def write_reports(prof: Any, out: Path, wall_s: float, steps: int) -> dict[str, float]:
    out.mkdir(parents=True, exist_ok=True)
    avg = prof.key_averages()
    sort_device = (
        "self_device_time_total"
        if hasattr(avg[0], "self_device_time_total")
        else "self_cuda_time_total"
    )
    (out / "kernels.txt").write_text(
        avg.table(sort_by=sort_device, row_limit=30, max_name_column_width=70)
        + "\n\n"
        + avg.table(sort_by="self_cpu_time_total", row_limit=30, max_name_column_width=70)
    )
    prof.export_chrome_trace(str(out / "trace.json"))
    summary = summarise(prof, wall_s, steps)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--wait", type=int, default=10, help="steps skipped before warmup")
    ap.add_argument("--warmup", type=int, default=5, help="steps profiled then discarded")
    ap.add_argument("--active", type=int, default=20, help="steps recorded")
    ap.add_argument("verb", nargs=argparse.REMAINDER, help="-- <verb> <args> for spectramr")
    args = ap.parse_args(argv)
    verb = [a for a in args.verb if a != "--"]
    if not verb:
        ap.error("give the verb after --, e.g. -- train --config <yaml>")

    import torch
    from torch.profiler import ProfilerActivity, profile, schedule

    from spectramr.cli.app import main as spectramr_main
    from spectramr.infrastructure.training.step_executor import StepExecutor

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: this profiles GPU time and will not fall back to CPU")
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=1),
    )
    first_active = args.wait + args.warmup
    last_active = first_active + args.active
    stamps: dict[str, float] = {}
    count = {"n": 0}
    real = StepExecutor.execute_step

    def stepped(self, *a, **k):
        out = real(self, *a, **k)
        count["n"] += 1
        if count["n"] == first_active:
            torch.cuda.synchronize()
            stamps["start"] = time.perf_counter()
        if count["n"] == last_active:
            torch.cuda.synchronize()
            stamps["end"] = time.perf_counter()
        prof.step()
        return out

    StepExecutor.execute_step = stepped
    prof.start()
    try:
        code = spectramr_main(verb)
    finally:
        prof.stop()
    if "start" not in stamps or "end" not in stamps:
        print(
            f"the run ended after {count['n']} steps, before the {last_active}-step window closed",
            file=sys.stderr,
        )
        return code or 2
    summary = write_reports(prof, args.out, stamps["end"] - stamps["start"], args.active)
    print(json.dumps(summary, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
