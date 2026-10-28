"""Fail if any file under ``scratch/`` is tracked: it is the host-local, gitignored bench.

Reusable probes belong in ``tools/``, producers in ``scripts/``; anything else is deleted.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


def tracked_under_scratch(paths: Iterable[str]) -> list[str]:
    """Return the paths whose first segment is exactly ``scratch``."""
    return sorted(p for p in paths if p.split("/", 1)[0] == "scratch" and "/" in p)


def main(repo_root: Path | None = None) -> int:
    root = repo_root or Path(__file__).resolve().parents[2]
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "scratch"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    offenders = tracked_under_scratch(p for p in out.split("\0") if p)
    for path in offenders:
        print(f"tracked under scratch/: {path}  (move to tools/ or scripts/, or delete)")
    return 1 if offenders else 0


if __name__ == "__main__":
    sys.exit(main())
