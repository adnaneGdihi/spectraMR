#!/usr/bin/env python3
"""Ruff, scoped to the lines a change actually adds.

Why not just ``ruff check <changed files>``
-------------------------------------------
Because ``dev`` carries **13,225** pre-existing ruff findings across **2,526**
unformatted files. A whole-file gate therefore fails on debt the change did not
write: touch one line of a physics module and you inherit all of its ``N806``
findings -- and ``N806`` (non-lowercase-variable-in-function) fires on ``B0``,
``T1``, ``M0``, ``S``, ``F``, which are the *correct* symbols in MRI. The gate was
unsatisfiable for any change touching a legacy file, which is the same as no gate
at all: a check that can never be green teaches everyone to merge red, and then a
real failure walks straight through.

The rules
---------
* **Added file** -> must be clean end to end (``ruff check`` AND ``ruff format``).
  New code has no inherited debt and no excuse.
* **Modified file** -> only findings ON LINES THE CHANGE ADDED are errors. Format is
  not checked: reformatting a legacy file to satisfy a two-line diff is pure churn,
  and it buries the actual change in a 400-line whitespace commit.

So a change is free to leave old debt alone, and is not free to add new debt.
Cleaning a legacy file remains welcome -- it just is not conscripted.

Two modes, one policy
---------------------
* ``--base``/``--head`` -- a merge-base range. What ``pr-required.yml`` runs.
* ``--staged`` -- the index against ``HEAD``. What the pre-commit hook runs.

``--staged`` takes line numbers from the index while ruff reads the WORKING TREE.
pre-commit stashes unstaged changes before running hooks, so under the hook the two
are the same bytes. Run by hand over a partially-staged file they are not and the
line numbers drift, so that case is reported rather than silently answered.

Scope: every Python file the change touches
-------------------------------------------
Not ``src/`` and ``tests/``. This script is the single owner of the added-line policy
(non-negotiable 24) and it is what the pre-commit hook runs, so a root allowlist here
is not a narrowing of *this* gate -- it is a hole in the only one. Held to
``src/``+``tests/`` it leaves **448** tracked ``.py`` files unlinted in CI, **362** of
them under ``scripts/``, which is where the CI gates themselves live. Those files were
linted at commit time before this change -- by the whole-file hook it replaces, which
ran on every Python file -- so pointing the hook here without widening the scope is
what would have dropped them.

WHICH paths are exempt is ruff's question, not this script's. ``--force-exclude`` makes
``[tool.ruff] extend-exclude`` apply even to paths passed explicitly -- ruff otherwise
lints a file you name, because the exclusion governs directory *walking* -- and
``per-file-ignores`` covers the rest. One owner for which files, one for which lines.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

#: ``@@ -old,+new @@`` -- we only care about the post-image (the ``+`` side), because
#: that is what the change is proposing to land.
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

#: git's canonical empty tree. ``--staged`` on an unborn branch has no ``HEAD`` to
#: diff against, and every staged file is then an addition.
_EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def _in_scope(path: str) -> bool:
    """Every Python file, with no root allowlist -- see the module docstring."""
    return path.endswith(".py")


def staged_base() -> str:
    """``HEAD``, or the empty tree when the branch is unborn."""
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "HEAD"], capture_output=True, text=True
    )
    return proc.stdout.strip() or _EMPTY_TREE


def changed_files(diff_args: list[str]) -> tuple[list[str], list[str]]:
    """``(added, modified)`` Python paths for a git diff selector.

    ``diff_args`` is ``["<base>...<head>"]`` for a range or ``["--cached", "<base>"]``
    for the index. Three-dot for the range: two-dot would also blame this change for
    commits that landed on the base branch after the branch forked.

    Scoping happens in Python, NOT via a ``src/**/*.py`` pathspec: git's wildmatch wants
    an intermediate directory for ``**``, so that pathspec matches
    ``src/spectramr/models/x.py`` but silently MISSES ``src/x.py``. The whole-file gate
    this replaces had exactly that hole -- it only ever worked because everything in
    this repo happens to sit one level down, under ``src/spectramr/``.
    """
    out = _git("diff", "--name-status", "--diff-filter=ACM", *diff_args)
    added: list[str] = []
    modified: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        status, _, path = line.partition("\t")
        if not _in_scope(path):
            continue
        if status.startswith("A"):
            added.append(path)
        else:  # C (copied) counts as new content too, but git reports it rarely.
            modified.append(path)
    return added, modified


def added_lines(diff_args: list[str], path: str) -> set[int]:
    """1-indexed line numbers this change ADDS to ``path``, in the post-image.

    ``-U0`` so the hunk headers carry no context lines -- a context line is code the
    change did not touch, and blaming it for a finding there is the whole bug.
    """
    out = _git("diff", "-U0", *diff_args, "--", path)
    lines: set[int] = set()
    for raw in out.splitlines():
        m = _HUNK.match(raw)
        if m:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            lines.update(range(start, start + count))
    return lines


def drifted(paths: list[str]) -> list[str]:
    """Staged paths whose working-tree bytes differ from the index.

    ruff reads the working tree and the line numbers come from the index, so for these
    the two disagree. pre-commit stashes unstaged changes, so this is empty under the
    hook and non-empty only for a hand run over a partially-staged file.
    """
    if not paths:
        return []
    out = _git("diff", "--name-only", "--", *paths)
    return [p for p in out.splitlines() if p.strip()]


def ruff_findings(paths: list[str]) -> list[dict]:
    """``ruff check --output-format=json``. Exit code is ignored: findings are the signal."""
    if not paths:
        return []
    proc = subprocess.run(
        ["ruff", "check", "--force-exclude", "--output-format=json", "--", *paths],
        capture_output=True,
        text=True,
    )
    if not proc.stdout.strip():
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"ruff failed to run: {proc.stderr.strip()}")
        return []
    return json.loads(proc.stdout)


def unformatted(paths: list[str]) -> list[str]:
    """Added files ``ruff format`` would rewrite, read off the EXIT CODE.

    Not off stdout. This parsed a ``Would reformat: `` prefix that ruff 0.16.2 does not
    emit -- it prints a diagnostic block headed ``unformatted: File would be
    reformatted`` -- so the comprehension yielded ``[]`` for every input and the
    added-file format half of the policy stopped firing at the version bump (#1418,
    #1810). An exit code is the tool's API; its prose is not.

    One invocation per file, because a batch run collapses to a single exit code while
    the report has to name the file. Added files are few.
    """
    out: list[str] = []
    for path in paths:
        proc = subprocess.run(
            ["ruff", "format", "--check", "--force-exclude", "--", path],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 1:
            out.append(path)
        elif proc.returncode != 0:
            raise RuntimeError(f"ruff format failed on {path}: {proc.stderr.strip()}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", help="merge-base side SHA (range mode)")
    ap.add_argument("--head", help="head SHA (range mode)")
    ap.add_argument(
        "--staged",
        action="store_true",
        help="lint the index against HEAD instead of a range (pre-commit mode)",
    )
    args = ap.parse_args()

    if args.staged:
        if args.base or args.head:
            ap.error("--staged takes the index against HEAD; it cannot also take a range")
        diff_args = ["--cached", staged_base()]
    elif args.base and args.head:
        diff_args = [f"{args.base}...{args.head}"]
    else:
        ap.error("--base and --head are both required unless --staged is given")

    added, modified = changed_files(diff_args)
    if not added and not modified:
        print("No changed Python files.")
        return 0

    print(f"{len(added)} added file(s), {len(modified)} modified file(s)")

    if args.staged:
        for path in drifted(added + modified):
            print(
                f"  note: {path} has unstaged changes, so ruff reads bytes the "
                f"line numbers below do not describe"
            )

    violations: list[str] = []

    root = Path.cwd().resolve()

    def _rel(p: str) -> str:
        try:
            return str(Path(p).resolve().relative_to(root))
        except ValueError:
            return p

    # Added files: every finding counts. New code has no inherited debt.
    for f in ruff_findings(added):
        loc = f.get("location") or {}
        violations.append(
            f"{_rel(f['filename'])}:{loc.get('row', '?')}:{loc.get('column', '?')}: "
            f"{f.get('code')} {f.get('message')}  [new file]"
        )
    for path in unformatted(added):
        violations.append(f"{_rel(path)}: not ruff-formatted  [new file]")

    # Modified files: only findings on lines this change added.
    scoped = {p: added_lines(diff_args, p) for p in modified}
    for f in ruff_findings(modified):
        loc = f.get("location") or {}
        path = _rel(f["filename"])
        if loc.get("row") in scoped.get(path, ()):
            violations.append(
                f"{path}:{loc.get('row')}:{loc.get('column', '?')}: "
                f"{f.get('code')} {f.get('message')}  [line added by this change]"
            )

    if violations:
        print(f"\n{len(violations)} ruff violation(s) introduced by this change:\n")
        for v in violations:
            print(f"  {v}")
        print(
            "\nOnly NEW code is gated: added files must be clean end to end, and "
            "modified files only on the lines you added. Pre-existing findings "
            "elsewhere in a file you touched are NOT your problem."
        )
        return 1

    print("No ruff violations on lines this change added.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
