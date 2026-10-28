#!/usr/bin/env python3
"""Every repo path a doc page tells you to run must exist in the tree being shipped.

The failure this catches is the most visible one a new user can hit: they copy a
command out of the documentation, paste it, and get ``No such file or directory``.
It is invisible to Sphinx -- a code block is opaque text, so a build with zero
warnings says nothing at all about whether the commands in it can run -- and it is
invisible to the test suite, which never reads prose.

It bites hardest in an *exported* tree, because the export drops whole roots
(``experiments/``, ``TODO/``, ``scratch/``, most of ``scripts/``) that the private
tree has. A page that was accurate on ``dev`` becomes a lie in the distribution
without one character of it changing. So this gate is written to run against
whichever tree it is pointed at, and is meant to be run against the export.

Placeholders are exempt, and deliberately so: ``experiments/<paradigm>/<arm>.yaml``
teaches a shape and no reader will paste it verbatim. A token is a placeholder when
it carries ``<``/``>``, ``...``, ``{``/``}``, ``$`` or a ``*`` glob. Everything else
is a concrete claim that a file is there.

Exit codes: 0 clean, 1 findings.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Path-shaped tokens under a root the export is entitled to drop. Anchored on the
# root name so a prose mention of the word "scripts" is not a finding.
# The optional ``./`` is not decoration. Without it the negative lookbehind -- which
# is there so ``src/scripts/x.py`` does not match on its tail -- also rejects
# ``./scripts/run.sh``, one of the shapes readers most reliably paste. That blindness
# was found by planting it, not by reading the pattern.
_PATH_RE = re.compile(
    r"(?<![\w/.-])(?:\./)?"
    r"((?:scripts|tools|experiments|TODO|scratch|runners|examples|tests)"
    r"/[A-Za-z0-9_<>{}$*./-]*"
    r"[A-Za-z0-9_<>{}$*-]"
    r"\.(?:py|sh|sbatch|ya?ml|json|txt|cfg|toml|md|rst))"
)

# A token nobody will paste verbatim.
_PLACEHOLDER = ("<", ">", "...", "{", "}", "$", "*")

# Two findings with the same shape are not the same defect, and the fix differs, so
# they are reported apart rather than summed.
#
#   COMMAND   -- the line is something a reader pastes into a shell. A dead path here
#                is a `No such file or directory` in the reader's terminal. This is
#                the class that must reach zero.
#   REFERENCE -- the line cites a file in prose ("this knob is read at ..."). A dead
#                path here is a dangling pointer: worth fixing, not user-breaking.
#
# Classification is by command shape rather than by parsing RST/MyST block structure,
# because it is the leading verb that decides whether a line gets pasted -- an
# indented literal block and a fenced code block are the same hazard, and a prose
# sentence quoting `python foo.py` is the same hazard again.
# ``\./`` is split out from the verb list rather than sitting inside it: every other
# alternative is a command *name* and so is followed by whitespace, but ``./run.sh``
# has none -- the path is glued straight onto the prefix. Folding it in with the verbs
# made the gate classify ``./scripts/run.sh`` as prose.
_COMMAND_RE = re.compile(
    r"(?:^|[\s`(|])"
    r"(?:(?:python3?|bash|sh|sbatch|srun|pytest|make|spectramr|uv run|pip install -e)\s"
    r"|\./)"
)


def is_command(line: str) -> bool:
    return bool(_COMMAND_RE.search(line))


# A repo-root script carries no directory to anchor on -- ``bash download_datasets.sh``
# is as pasteable as ``bash scripts/x.sh`` and just as dead once the export drops it,
# but the root-anchored pattern above cannot see it. This one only fires on a line
# already established as a command, so a bare ``config.yaml`` in prose stays quiet.
_BARE_SCRIPT_RE = re.compile(
    r"(?:^|[\s`(|])(?:bash|sh|sbatch|srun|source|\./)\s?"
    r"([A-Za-z0-9_<>{}$*.-]+\.(?:sh|sbatch))(?![\w/])"
)


def is_placeholder(token: str) -> bool:
    return any(marker in token for marker in _PLACEHOLDER)


# A tutorial that prints a listing and says "save this as X" has *given* the reader
# the file; the command that runs X afterwards is correct, and the file's absence from
# the repository is the point rather than a defect. Without this the gate reports the
# best-written tutorials as the most broken ones -- and a gate that punishes good prose
# gets switched off. The exemption is per page and per path, so a page that authors
# scripts/a.py earns nothing for a stray scripts/b.py.
_AUTHORS_RE = re.compile(
    r"(?:save|store|write|put|create|add)\b[^.\n]{0,40}?\b(?:as|to|into)\b[^\n]{0,20}?"
    r"[`:\"']((?:scripts|tools|experiments|TODO|scratch|runners|examples|tests)/[^`\"'\s]+)"
    r"|(?:^|[\s])(?:create|write|save|add)\s+(?:a\s+|the\s+|new\s+)*"
    r"[`:\"']*((?:scripts|tools|experiments|TODO|scratch|runners|examples|tests)/[^`\"'\s,]+)",
    re.IGNORECASE,
)


def paths_the_page_tells_you_to_create(text: str) -> set[str]:
    found: set[str] = set()
    for a, b in _AUTHORS_RE.findall(text):
        for token in (a, b):
            if token:
                found.add(token.rstrip(":`\"'.,"))
    return found


def shipped_pages(root: Path, doc_dir: Path) -> list[Path]:
    """Every prose page a reader of the distribution can paste a command out of.

    ``docs/`` is the obvious half. The other half is the root-level set --
    ``README.md``, ``CONTRIBUTING.md`` and their siblings -- which are the most
    read pages in the repository and were outside this gate's scan root until
    2026-08-29. A scan root is an unaudited constant: no violation planted
    *inside* ``docs/`` can show that ``docs/`` is not the whole surface.
    """
    pages = [p for p in doc_dir.rglob("*") if p.suffix in (".rst", ".md") and p.is_file()]
    pages += [p for p in root.iterdir() if p.suffix in (".rst", ".md") and p.is_file()]
    return sorted(set(pages))


def scan(root: Path, doc_dir: Path) -> list[tuple[Path, int, str, str]]:
    findings: list[tuple[Path, int, str, str]] = []
    for page in shipped_pages(root, doc_dir):
        text = page.read_text(errors="ignore")
        authored = paths_the_page_tells_you_to_create(text)
        for lineno, line in enumerate(text.splitlines(), 1):
            command = is_command(line)
            tokens = list(_PATH_RE.findall(line))
            if command:
                tokens += _BARE_SCRIPT_RE.findall(line)
            for token in tokens:
                if is_placeholder(token) or token in authored:
                    continue
                if not (root / token).exists():
                    kind = "COMMAND" if command else "REFERENCE"
                    findings.append((page.relative_to(root), lineno, token, kind))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--commands-only",
        action="store_true",
        help="Fail only on COMMAND findings -- the pasteable ones. REFERENCE findings "
        "are still listed, but do not set the exit code.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Tree to check. Defaults to this script's repository -- pass the export "
        "explicitly when checking a distribution.",
    )
    args = ap.parse_args()
    root = args.root.resolve()
    doc_dir = root / "docs"

    # A repo-rooted script run from outside its repo resolves its root to somewhere
    # with no pages and reports all-clean, which reads exactly like a pass. Refuse.
    #
    # The refusal keys on the PAGE SET, not on ``docs/``, and neither half of that
    # implies the other. A ``docs/`` that exists but holds no page scans nothing
    # and prints OK -- indistinguishable from a clean tree. A distribution whose
    # prose is entirely root-level is a legitimate tree with no ``docs/`` at all,
    # and refusing that one would leave this gate unable to check ``README.md``
    # and ``CONTRIBUTING.md``, the two pages its scan root was widened to cover.
    # This gate sets the exit code of the public export now, so "no page was
    # checked" must never read as "every page is clean" -- and "checked only the
    # root pages" must not be refused as though it were that.
    if not shipped_pages(root, doc_dir):
        missing = " (no docs/, and no root-level page)" if not doc_dir.is_dir() else ""
        print(f"FAIL: no .rst/.md page under {root}{missing} -- refusing to report a vacuous pass.")
        return 1

    findings = scan(root, doc_dir)
    commands = [f for f in findings if f[3] == "COMMAND"]
    references = [f for f in findings if f[3] == "REFERENCE"]

    if not findings:
        print(
            f"OK: every concrete repo path cited under {doc_dir} "
            f"and in the root-level pages exists in {root}."
        )
        return 0

    for label, group in (("COMMAND", commands), ("REFERENCE", references)):
        if not group:
            continue
        print(f"\n=== {label}: {len(group)} citation(s) absent from {root} ===")
        for page, lineno, token, _ in group:
            print(f"  {page}:{lineno}: {token}")

    print(
        f"\nTotals: {len(commands)} COMMAND, {len(references)} REFERENCE.\n"
        "A COMMAND finding is a line a reader pastes and watches fail -- fix by "
        "pointing at a path that ships, by rewriting so it names no file, or, when "
        "the shape is the point, by making it an explicit <placeholder>. A REFERENCE "
        "finding is a dangling pointer in prose: say where the file lives instead of "
        "citing a path the reader cannot open."
    )
    if args.commands_only:
        return 1 if commands else 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
