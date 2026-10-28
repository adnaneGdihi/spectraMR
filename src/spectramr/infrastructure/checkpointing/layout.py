"""Where a run's checkpoints live and the two names they may have.

A run writes ``<run_dir>/checkpoints/step_{N:08d}.{ext}`` (rolling) and ``best.{ext}``
(single). Why a file was written is recorded inside it, not in its name. Torch-free so
discovery can be imported from ``cli/`` without the startup cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spectramr.core.checkpoint_formats import CHECKPOINT_FORMATS as FORMATS

CHECKPOINT_SUBDIR = "checkpoints"

Role = Literal["step", "best"]

_NAME_RE = re.compile(rf"(?:step_(?P<step>\d{{8,}})|(?P<best>best))\.(?P<fmt>{'|'.join(FORMATS)})")


def validate_format(fmt: str) -> str:
    """Return ``fmt`` if it is a supported checkpoint format, else raise."""
    if fmt not in FORMATS:
        raise ValueError(f"checkpoint format {fmt!r} is not one of {list(FORMATS)}")
    return fmt


def native_tag(step: int) -> str:
    """The stem of a step checkpoint, shared with a strategy-native tag directory."""
    return f"step_{step:08d}"


@dataclass(frozen=True)
class CheckpointName:
    """A checkpoint file recognised by :func:`parse`."""

    path: Path
    role: Role
    step: int | None
    fmt: str


def parse(path: Path) -> CheckpointName | None:
    """Classify ``path`` as a checkpoint, or ``None`` for any other entry.

    Symlinks are refused so a stray alias can never be selected or pruned twice.
    """
    match = _NAME_RE.fullmatch(path.name)
    if match is None or path.is_symlink() or not path.is_file():
        return None
    step = match.group("step")
    role: Role = "best" if match.group("best") else "step"
    return CheckpointName(path, role, int(step) if step else None, match.group("fmt"))


def run_dir_of(path: str | Path) -> Path | None:
    """The run directory a checkpoint path names, read from the path alone.

    ``None`` when the path is not ``<run_dir>/checkpoints/<checkpoint name>``. The
    file need not exist yet, so a downstream arm can name an upstream one's output.
    """
    p = Path(path)
    if p.parent.name != CHECKPOINT_SUBDIR or _NAME_RE.fullmatch(p.name) is None:
        return None
    return p.parent.parent


@dataclass(frozen=True)
class CheckpointLayout:
    """The checkpoint directory of one run; construction touches no file system."""

    run_dir: Path

    @classmethod
    def for_run(cls, run_dir: str | Path | None) -> CheckpointLayout:
        # No fallback location: a missing run dir once put checkpoints under the CWD.
        if run_dir is None or not str(run_dir).strip():
            raise ValueError(
                "a checkpoint layout needs the run's output directory; none was given "
                "and there is no default location"
            )
        return cls(Path(run_dir))

    @property
    def directory(self) -> Path:
        return self.run_dir / CHECKPOINT_SUBDIR

    def ensure(self) -> Path:
        """Create the checkpoint directory; the run calls this once, at start."""
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory

    def step_path(self, step: int, fmt: str) -> Path:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"checkpoint step must be a non-negative int, got {step!r}")
        return self.directory / f"{native_tag(step)}.{validate_format(fmt)}"

    def best_path(self, fmt: str) -> Path:
        return self.directory / f"best.{validate_format(fmt)}"

    def entries(self) -> list[CheckpointName]:
        """Every checkpoint file in the directory, of every format; empty if absent."""
        if not self.directory.is_dir():
            return []
        return [name for p in sorted(self.directory.iterdir()) if (name := parse(p))]


__all__ = [
    "CHECKPOINT_SUBDIR",
    "run_dir_of",
    "FORMATS",
    "CheckpointLayout",
    "CheckpointName",
    "native_tag",
    "parse",
    "validate_format",
]
