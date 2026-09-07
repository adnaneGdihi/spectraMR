#!/usr/bin/env python3
"""Move the version in every file that states it, in one reviewed command.

Four files state the version independently -- ``src/spectramr/__init__.py``,
``CHANGELOG.md``, ``CITATION.cff`` and the git tag -- because each is read by
something that cannot import the package. ``build_dist.py`` is the sole
**comparator** of that set (non-negotiable 17) and fails the build on any
disagreement. This is the sole **writer** for the same set, and it imports
``build_dist``'s readers *and its comparator* rather than restating them: a
writer with its own idea of where the version lives is exactly the second owner
that rule forbids. ``show`` did restate it once, as ``len(set(values)) != 1``,
which cannot express a dev build -- so it reported ``DISAGREEMENT`` on every
tree this file's own ``nightly`` mode had just written correctly.

Modes, and why they are separate:

``major`` / ``minor`` / ``build``
    Stable bumps. Each requires the current version to be stable -- from a
    ``.devN`` the intent is genuinely ambiguous (finish this series, or start a
    different one?), so it raises and names the two modes that say which.

``nightly``
    Advance the dev counter of the version being worked *towards*:
    ``0.1.0 -> 0.1.1.dev1 -> 0.1.1.dev2``. Leaves ``CHANGELOG.md`` alone. A dev
    build is not a release, and giving each one a changelog heading makes the
    file's release history unreadable.

``release``
    Finish a dev series: ``0.1.1.dev4 -> 0.1.1``. The only mode that touches the
    changelog -- it cuts the accumulated ``[Unreleased]`` section into a dated
    heading and opens a fresh empty one.

``set``
    An explicit version, for the cases the rules above deliberately refuse.

Dry-run by default: it prints the plan and writes nothing until ``--apply``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dist import (
    changelog_version,
    citation_version,
    declared_version,
    version_disagreements,
)

ROOT = Path(__file__).resolve().parents[2]
INIT = "src/spectramr/__init__.py"
CITATION = "CITATION.cff"
CHANGELOG = "CHANGELOG.md"

_VERSION = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<build>\d+)(?:\.dev(?P<dev>\d+))?$")
_UNRELEASED_LINK = re.compile(r"^\[Unreleased\]:\s*(?P<base>\S+)/compare/v\S+\.\.\.HEAD\s*$", re.M)


@dataclass(frozen=True)
class Version:
    """``MAJOR.MINOR.BUILD`` with an optional ``.devN`` pre-release counter."""

    major: int
    minor: int
    build: int
    dev: int | None = None

    def __str__(self) -> str:
        core = f"{self.major}.{self.minor}.{self.build}"
        return core if self.dev is None else f"{core}.dev{self.dev}"

    @property
    def release(self) -> Version:
        """The stable version this one is working towards (itself, if stable)."""
        return Version(self.major, self.minor, self.build)


def parse(text: str) -> Version:
    """``'0.1.1.dev4'`` -> ``Version``. Anything else raises, never defaults."""
    m = _VERSION.match(text.strip())
    if not m:
        raise ValueError(
            f"{text!r} is not MAJOR.MINOR.BUILD[.devN]. Local versions such as "
            "'0.1.1+build4' are rejected by PyPI outright and cannot be used here."
        )
    dev = m.group("dev")
    return Version(int(m["major"]), int(m["minor"]), int(m["build"]), dev and int(dev))


def advance(current: Version, mode: str) -> Version:
    """The version ``mode`` moves to, or a raise naming the mode that applies."""
    if mode in {"major", "minor", "build"} and current.dev is not None:
        raise ValueError(
            f"{current} is a dev version: a '{mode}' bump from it is ambiguous. "
            f"Use 'release' to finish {current.release}, 'nightly' to advance the "
            "counter, or 'set' to state a version outright."
        )
    if mode == "major":
        return Version(current.major + 1, 0, 0)
    if mode == "minor":
        return Version(current.major, current.minor + 1, 0)
    if mode == "build":
        return Version(current.major, current.minor, current.build + 1)
    if mode == "nightly":
        if current.dev is not None:
            return Version(current.major, current.minor, current.build, current.dev + 1)
        return Version(current.major, current.minor, current.build + 1, 1)
    if mode == "release":
        if current.dev is None:
            raise ValueError(f"{current} is already a release; 'release' has nothing to finish.")
        return current.release
    raise ValueError(f"unknown mode {mode!r}")


def rewrite_init(text: str, new: Version) -> str:
    """Replace ``__version__``'s literal, leaving its quoting style alone."""
    m = re.search(r'^__version__\s*=\s*(["\'])(?P<v>[^"\']+)\1', text, re.M)
    if m is None:
        raise ValueError(f"no __version__ assignment found in {INIT}")
    return text[: m.start("v")] + str(new) + text[m.end("v") :]


def rewrite_citation(text: str, new: Version) -> str:
    """Replace CITATION.cff's ``version:`` value, read without a YAML dependency."""
    m = re.search(r'^version:\s*["\']?(?P<v>[^"\'\s]+)', text, re.M)
    if m is None:
        raise ValueError(f"no 'version:' key found in {CITATION}")
    return text[: m.start("v")] + str(new) + text[m.end("v") :]


def rewrite_changelog(text: str, new: Version, today: str) -> str:
    """Cut ``[Unreleased]`` into a dated heading for ``new`` and reopen it empty.

    Raises rather than guessing on every shape it cannot handle: a missing
    ``[Unreleased]`` heading, a missing link reference, or an ``[Unreleased]``
    section with no entries in it. Releasing nothing is a mistake worth stopping
    on, and ``set`` is the documented way past it.
    """
    heading = re.search(r"^##[ 	]*\[Unreleased\][ 	]*$", text, re.M)
    if heading is None:
        raise ValueError(f"no '## [Unreleased]' heading in {CHANGELOG}")
    nxt = re.search(r"^##\s+", text[heading.end() :], re.M)
    body = text[heading.end() :][: nxt.start()] if nxt else text[heading.end() :]
    if not body.strip():
        raise ValueError(
            f"the [Unreleased] section of {CHANGELOG} is empty -- there is nothing "
            f"to release as {new}."
        )
    link = _UNRELEASED_LINK.search(text)
    if link is None:
        raise ValueError(f"no '[Unreleased]: .../compare/v...HEAD' link reference in {CHANGELOG}")
    base = link.group("base")

    out = text[: heading.end()] + f"\n\n## [{new}] - {today}" + text[heading.end() :]
    out = _UNRELEASED_LINK.sub(f"[Unreleased]: {base}/compare/v{new}...HEAD", out, count=1)
    anchor = f"[Unreleased]: {base}/compare/v{new}...HEAD"
    return out.replace(anchor, f"{anchor}\n[{new}]: {base}/releases/tag/v{new}", 1)


def sources(root: Path) -> dict[str, str | None]:
    """Each file that states the version *as a version*, via ``build_dist``'s readers.

    ``CHANGELOG.md`` is deliberately not here. It states the version as a dated
    heading, which a dev build deliberately has none of, so it cannot join a
    string comparison -- ``build_dist.changelog_disagreement`` judges it instead.
    """
    return {
        INIT: declared_version((root / INIT).read_text()),
        CITATION: citation_version((root / CITATION).read_text()),
    }


def plan(root: Path, new: Version, touch_changelog: bool, today: str) -> list[tuple[str, str]]:
    """``(path, new_text)`` for every file this bump rewrites."""
    edits = [
        (INIT, rewrite_init((root / INIT).read_text(), new)),
        (CITATION, rewrite_citation((root / CITATION).read_text(), new)),
    ]
    if touch_changelog:
        edits.append((CHANGELOG, rewrite_changelog((root / CHANGELOG).read_text(), new, today)))
    return edits


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "mode",
        choices=["show", "major", "minor", "build", "nightly", "release", "set"],
        help="'show' reports every source and any disagreement; the rest move the number.",
    )
    p.add_argument("version", nargs="?", help="the target version, for 'set' only")
    p.add_argument("--root", type=Path, default=ROOT, help="repository root (default: this one)")
    p.add_argument("--apply", action="store_true", help="write the files (default: dry run)")
    args = p.parse_args(argv)

    have = sources(args.root)
    unreadable = [k for k, v in have.items() if v is None]
    if unreadable:
        print(f"UNREADABLE: {', '.join(unreadable)}", file=sys.stderr)
        return 2

    changelog_text = (args.root / CHANGELOG).read_text()

    if args.mode == "show":
        for path, value in have.items():
            print(f"  {value:<16} {path}")
        newest = changelog_version(changelog_text)
        print(f"  {newest or '-':<16} {CHANGELOG}   (newest released heading)")
        problems = version_disagreements(
            have[INIT], {k: v for k, v in have.items() if k != INIT}, changelog_text
        )
        if problems:
            print("DISAGREEMENT:", file=sys.stderr)
            for why in problems:
                print(f"  {why}", file=sys.stderr)
            return 1
        print(f"\nversion: {have[INIT]}")
        return 0

    current = parse(have[INIT])
    if args.mode == "set":
        if not args.version:
            print("'set' needs a version argument", file=sys.stderr)
            return 2
        new = parse(args.version)
    else:
        if args.version:
            print(
                f"'{args.mode}' takes no version argument; use 'set' to state one", file=sys.stderr
            )
            return 2
        new = advance(current, args.mode)

    touch_changelog = new.dev is None
    edits = plan(args.root, new, touch_changelog, date.today().isoformat())

    print(
        f"{current} -> {new}" + ("" if touch_changelog else "   (dev build: CHANGELOG untouched)")
    )
    for path, text in edits:
        changed = text != (args.root / path).read_text()
        print(
            f"  {'write' if args.apply else 'would write'}  {path}"
            + ("" if changed else "  (no change)")
        )
        if args.apply:
            (args.root / path).write_text(text)
    if not args.apply:
        print("\ndry run -- nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
