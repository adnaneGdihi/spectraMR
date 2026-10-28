"""Submit-time snapshot of an array's code and configs, checked at every task start.

An array manifest lists config *paths*, and each task reads its YAML and imports
``src/`` from the live checkout when it starts. A ``git pull`` between two task
starts therefore makes one array run two protocols: array 17863804 index 8
started 37 minutes before indices 9 and 10, before the protocol-v2 pull, and
trained the v1 config into the v1 results tree without a word (#2662).

Runs follow ``dev``. The submitter refuses a checkout that is not on the ``dev``
branch and records HEAD, the branch and a sha256 per listed YAML beside the
manifest (:func:`write_snapshot`). Each task start, including a requeued link,
checks the live checkout (:func:`verify_snapshot`): it must still be on ``dev``
and still contain the submitted commit, so dev advancing is the code the task
runs, while a feature branch, a detached HEAD, a rewound dev or an uncommitted
YAML edit is refused by name. ``SPECTRAMR_DISPATCH_ALLOW_DRIFT=1`` proceeds
anyway, and the verdict, with the commit each task actually ran, reaches the
run's provenance through ``SPECTRAMR_DISPATCH_SNAPSHOT_RECORD`` either way.

Stdlib only: the submit wrapper runs this with whatever ``python3`` it finds,
before any venv is guaranteed to exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spectramr.core.dispatch_git import (
    DispatchSnapshotError,
    committed,
    contains,
    git_dirty,
    git_head,
    git_src_tree,
    off_run_branch,
    run_branch,
)
from spectramr.core.env_names import (
    SPECTRAMR_DISPATCH_ALLOW_DRIFT,
    SPECTRAMR_DISPATCH_SNAPSHOT_RECORD,
)

SCHEMA_VERSION = 1


def snapshot_path(manifest: str | Path) -> Path:
    """Where the record for ``manifest`` lives: beside it, named after it."""
    manifest = Path(manifest)
    return manifest.with_name(f"{manifest.stem}.snapshot.json")


def manifest_entries(manifest: str | Path) -> list[str]:
    """The manifest's config lines, blank lines dropped, as the dispatcher reads them."""
    return [ln.strip() for ln in Path(manifest).read_text().splitlines() if ln.strip()]


def _src_tree_or_none(repo_root: Path) -> str | None:
    """``git_src_tree``, or ``None`` for a checkout with no ``src/`` at HEAD (provenance only)."""
    try:
        return git_src_tree(repo_root)
    except DispatchSnapshotError:
        return None


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve(entry: str, repo_root: Path) -> Path:
    """A manifest entry as the task will open it: relative entries are taken
    from the repo root, which is the cwd both submitters give their tasks."""
    path = Path(entry)
    return path if path.is_absolute() else repo_root / path


def write_snapshot(manifest: str | Path, repo_root: str | Path) -> Path:
    """Record HEAD, the dirty flag and a sha256 per listed YAML beside ``manifest``.

    Raises :class:`DispatchSnapshotError` when the checkout is not on
    the run branch (``dev``), HEAD cannot be resolved or a listed YAML cannot be read: an
    array submitted without a record would be checked by nothing.
    """
    root = Path(repo_root).resolve()
    head = git_head(root)  # not a checkout at all is reported as such, before the branch
    problem = off_run_branch(root)
    if problem:
        raise DispatchSnapshotError(f"{problem}; check out {run_branch()} and submit again.")
    configs: dict[str, str] = {}
    for entry in manifest_entries(manifest):
        target = _resolve(entry, root)
        try:
            configs[entry] = sha256_file(target)
        except OSError as exc:
            raise DispatchSnapshotError(f"cannot hash manifest entry {entry!r}: {exc}") from exc
    record = {
        "schema": SCHEMA_VERSION,
        "repo_root": str(root),
        "head": head,
        "branch": run_branch(),
        "src_tree": _src_tree_or_none(root),
        "dirty": git_dirty(root),
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "configs": configs,
    }
    out = snapshot_path(manifest)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return out


def allow_drift_requested(env: Mapping[str, str] | None = None) -> bool:
    """Read ``SPECTRAMR_DISPATCH_ALLOW_DRIFT``: unset, ``0`` or ``1``; anything else raises."""
    raw = (os.environ if env is None else env).get(SPECTRAMR_DISPATCH_ALLOW_DRIFT, "")
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{SPECTRAMR_DISPATCH_ALLOW_DRIFT}={raw!r} must be unset, '0' or '1'.")


def _drifts(record: dict[str, Any], entry: str) -> tuple[list[str], dict[str, Any]]:
    """Every way this task differs from the record, plus the values compared."""
    root = Path(record["repo_root"])
    drifts: list[str] = []
    try:
        head_now: str | None = git_head(root)
    except DispatchSnapshotError as exc:
        head_now = None
        drifts.append(f"HEAD of {root} cannot be resolved ({exc}); recorded {record['head']}")
    # Runs follow dev: dev advancing is the code a task should run, so only leaving
    # dev, or a dev that no longer contains the submitted commit, is drift. The live
    # checkout decides, so arrays recorded before the rule are checked the same way.
    problem = off_run_branch(root)
    if problem:
        drifts.append(problem)
    if head_now is not None and head_now != record["head"]:
        if not contains(root, record["head"], head_now):
            drifts.append(
                f"submitted commit {record['head']} is not in the history of {head_now}: "
                f"{run_branch()} was rewound, or the checkout is another line"
            )

    recorded_sha = record["configs"].get(entry)
    target = _resolve(entry, root)
    try:
        current_sha: str | None = sha256_file(target)
    except OSError:
        current_sha = None
    if recorded_sha is None:
        drifts.append(f"config {entry} is not in the submit-time snapshot")
    elif current_sha is None:
        drifts.append(f"config {entry} is no longer readable at {target}")
    elif current_sha != recorded_sha and not committed(target):
        drifts.append(
            f"config {entry} changed since submit: sha256 {recorded_sha} at submit, "
            f"{current_sha} now"
        )
    values = {
        "recorded_head": record["head"],
        "current_head": head_now,
        "dirty_at_submit": record["dirty"],
        "recorded_sha256": recorded_sha,
        "current_sha256": current_sha,
    }
    return drifts, values


def verify_snapshot(
    manifest: str | Path,
    entry: str,
    *,
    allow_drift: bool,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare this task's HEAD and config against the submit-time record.

    Returns the verdict to stamp into provenance, whose ``status`` is
    ``verified``, ``drift_allowed`` or ``absent`` (a manifest written before the
    record existed). Raises :class:`DispatchSnapshotError` naming every
    difference, with both values, unless ``allow_drift``.
    """
    source = os.environ if env is None else env
    verdict: dict[str, Any] = {
        "manifest": str(manifest),
        "snapshot": str(snapshot_path(manifest)),
        "config": entry,
        "restart_count": int(source.get("SLURM_RESTART_COUNT") or 0),
        "allow_drift": allow_drift,
        "drift": [],
    }
    path = snapshot_path(manifest)
    if not path.is_file():
        verdict["status"] = "absent"
        return verdict
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DispatchSnapshotError(f"snapshot {path} is unreadable: {exc}") from exc
    if record.get("schema") != SCHEMA_VERSION:
        raise DispatchSnapshotError(
            f"snapshot {path} has schema {record.get('schema')!r}; this dispatcher "
            f"reads schema {SCHEMA_VERSION}"
        )

    drifts, values = _drifts(record, entry)
    verdict.update(values, drift=drifts)
    if drifts and not allow_drift:
        raise DispatchSnapshotError(
            "this task no longer matches the array it was submitted in:\n  - "
            + "\n  - ".join(drifts)
            + f"\nResubmit the array from the current checkout, or resubmit with "
            f"{SPECTRAMR_DISPATCH_ALLOW_DRIFT}=1 to run it anyway (recorded in provenance). "
            "Any checkpoint an earlier link of this task saved is intact."
        )
    verdict["status"] = "drift_allowed" if drifts else "verified"
    return verdict


def report_snapshot(verdict: Mapping[str, Any]) -> None:
    """One line for a verified task; a loud stderr block for anything else."""
    link = f" (requeued link {verdict['restart_count']})" if verdict["restart_count"] else ""
    if verdict["status"] == "verified":
        print(f"[SNAPSHOT] verified{link}: HEAD {verdict['current_head']}, config sha256 unchanged")
        return
    if verdict["status"] == "absent":
        print(
            f"[SNAPSHOT] WARNING{link}: no submit-time snapshot at {verdict['snapshot']} "
            "(manifest predates #2662); HEAD and config drift are NOT checked for this task.",
            file=sys.stderr,
        )
        return
    lines = "\n".join(f"[SNAPSHOT]   - {d}" for d in verdict["drift"])
    print(
        f"[SNAPSHOT] DRIFT ALLOWED by {SPECTRAMR_DISPATCH_ALLOW_DRIFT}=1{link}; this task "
        f"does NOT run what the array was submitted with:\n{lines}",
        file=sys.stderr,
    )


def resolve_dispatch_snapshot_provenance(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The verdict the dispatcher handed this run, or ``{}`` outside an array."""
    raw = (os.environ if env is None else env).get(SPECTRAMR_DISPATCH_SNAPSHOT_RECORD)
    if not raw:
        return {}
    return dict(json.loads(Path(raw).read_text()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dispatch_snapshot",
        description="Record HEAD and per-config sha256 beside an array manifest.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args(argv)
    try:
        out = write_snapshot(args.manifest, args.repo_root)
    except DispatchSnapshotError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    record = json.loads(out.read_text())
    dirty = " (tracked files modified: uncommitted code is NOT covered)" if record["dirty"] else ""
    print(
        f"Code snapshot: HEAD {record['head']}{dirty}, {len(record['configs'])} config(s) -> {out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI entry
    raise SystemExit(main())
