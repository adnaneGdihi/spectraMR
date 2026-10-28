"""Print the checkpoint a finished run's post-training inference should evaluate.

``python -m spectramr.cli.resolve_checkpoint <run>/checkpoints`` prints the run's best
checkpoint, else its latest step, of any format, and exits 0; or names the directory
on stderr and exits 1. Generated SLURM scripts call it so the choice has one owner
(``checkpointing.select``). Torch-free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Print the chosen checkpoint and return 0, or return 1 when there is none."""
    parser = argparse.ArgumentParser(
        prog="python -m spectramr.cli.resolve_checkpoint",
        description="Print the best-then-latest checkpoint in a run's checkpoints directory.",
    )
    parser.add_argument("checkpoints_dir", type=Path, help="The run's checkpoints/ directory.")
    args = parser.parse_args(argv)

    from spectramr.infrastructure.checkpointing.layout import CHECKPOINT_SUBDIR
    from spectramr.infrastructure.checkpointing.payload import CONVERTER
    from spectramr.infrastructure.checkpointing.select import (
        ConversionPendingError,
        best_then_latest,
        pre_v1_names,
    )

    # The run is the argument's parent; any other directory name would silently read a sibling.
    if args.checkpoints_dir.name != CHECKPOINT_SUBDIR:
        parser.error(f"expected a run's {CHECKPOINT_SUBDIR}/ directory, got {args.checkpoints_dir}")
    run_dir = args.checkpoints_dir.parent
    try:
        checkpoint, _rule = best_then_latest(run_dir)
    except ConversionPendingError as exc:
        print(f"resolve_checkpoint: {exc}", file=sys.stderr)
        return 1
    if checkpoint is None:
        legacy = pre_v1_names(run_dir)
        hint = ""
        if legacy:
            shown = ", ".join(legacy[:5]) + (
                f" and {len(legacy) - 5} more" if len(legacy) > 5 else ""
            )
            hint = (
                f"; it holds only pre-1.0 checkpoint files ({shown}). "
                f"Convert the run first: python {CONVERTER} {run_dir}"
            )
        print(
            f"resolve_checkpoint: no checkpoint in {args.checkpoints_dir} (looked for "
            f"best.<format>, then the highest step_NNNNNNNN.<format>){hint}",
            file=sys.stderr,
        )
        return 1
    print(checkpoint)
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI entry
    sys.exit(main())
