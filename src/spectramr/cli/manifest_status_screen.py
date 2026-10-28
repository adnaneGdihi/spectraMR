"""Submit-time screen that keeps launch-refused arms out of an array manifest.

``run_training_pipeline`` refuses an arm whose ``metadata.status`` is in
``LAUNCH_REFUSED_STATUSES`` unless ``--allow-status`` names it. The status is a
static field of the YAML, so an array task that reaches that refusal has spent a
node, a GPU and the audit import on an answer the submitter could have read
(#2049). This screen applies the same rule to the frozen manifest before the
array range is fixed, and reports every arm it drops.

The status set and the ``--allow-status`` semantics are the pipeline's, imported
rather than restated, so the screen cannot refuse an arm that ``train`` would
run or pass one it would refuse.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import yaml

from spectramr.config.schemas.base import LAUNCH_REFUSED_STATUSES

#: Verbs whose pipeline applies the refusal. The others never read the status,
#: so screening them would drop arms their verb runs.
SCREENED_VERBS: frozenset[str] = frozenset({"train"})


def read_declared_status(config_path: str | Path) -> str | None:
    """Return the arm's declared ``metadata.status``, or ``None`` if it has none.

    A raw read, like ``manifest_dispatch.read_parallel_strategy``: the value is
    a closed vocabulary validated at load, so the file's spelling is what the
    pipeline sees.
    """
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    metadata = cfg.get("metadata") if isinstance(cfg, dict) else None
    status = metadata.get("status") if isinstance(metadata, dict) else None
    return None if status is None else str(status)


def allow_status_from_verb_args(verb_args: Sequence[str]) -> str | None:
    """Return the status ``--allow-status`` names in the frozen verb args, if any.

    Last occurrence wins, which is how argparse resolves a repeated option.
    """
    allowed: str | None = None
    for i, arg in enumerate(verb_args):
        if arg == "--allow-status" and i + 1 < len(verb_args):
            allowed = verb_args[i + 1]
        elif arg.startswith("--allow-status="):
            allowed = arg.split("=", 1)[1]
    return allowed


def screen_manifest(
    lines: Iterable[str], *, allow_status: str | None, repo_root: Path
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split manifest lines into (kept, refused) under the pipeline's rule.

    A line whose file cannot be found is kept: resolving it is the dispatcher's
    job, and its error names the missing path.

    Returns:
        The kept lines in order, and ``(status, line)`` for each refused arm.
    """
    kept: list[str] = []
    refused: list[tuple[str, str]] = []
    for line in lines:
        path = Path(line)
        if not path.is_file():
            path = repo_root / line
        status = read_declared_status(path) if path.is_file() else None
        if status in LAUNCH_REFUSED_STATUSES and status != allow_status:
            refused.append((status, line))
        else:
            kept.append(line)
    return kept, refused


def main(argv: list[str] | None = None) -> int:
    """Rewrite ``--manifest`` in place without refused arms; list them in ``--refused``.

    Exit 0 when the screen ran (even if it refused every arm; the caller reports
    an empty manifest), 2 on bad arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--refused", required=True, type=Path)
    parser.add_argument("--verb", default="train")
    parser.add_argument("--verb-args-file", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    args.refused.write_text("")
    if args.verb not in SCREENED_VERBS:
        return 0
    verb_args = (
        [ln for ln in args.verb_args_file.read_text().splitlines() if ln]
        if args.verb_args_file is not None
        else []
    )
    lines = [ln for ln in args.manifest.read_text().splitlines() if ln.strip()]
    kept, refused = screen_manifest(
        lines,
        allow_status=allow_status_from_verb_args(verb_args),
        repo_root=args.repo_root,
    )
    args.manifest.write_text("".join(f"{ln}\n" for ln in kept))
    args.refused.write_text("".join(f"{status}\t{line}\n" for status, line in refused))
    return 0


if __name__ == "__main__":
    sys.exit(main())
