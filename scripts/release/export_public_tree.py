#!/usr/bin/env python3
"""Export the public spectraMR tree from a pinned commit.

Both halves of the export are pinned to that SHA: the candidate file list comes
from ``git ls-tree``, and the allowlist that selects from it is read with
``git show`` rather than from the working tree. Pinning only the first half meant
an unchanged SHA could yield two different trees.

The file list comes from ``git ls-tree`` at an explicit SHA -- never a filesystem
walk. A walk picks up untracked files, which breaks the provenance claim that the
manifest's ``git_sha`` pins its file set; this checkout in particular carries
several GB of untracked cluster output.

The allowlist is **fail-closed**: a path ships only if some pattern names it.
Forgetting a pattern loses a file (visible, recoverable); forgetting a denial
would publish one (invisible, not recoverable).

A line beginning with ``!`` is a **denial**, and denials only ever *subtract* --
a path ships iff some allowance names it AND no denial does. That direction is
what keeps the gate fail-closed: a wrong denial loses a file, which is the same
visible, recoverable failure the allowlist already prefers, whereas a denial
that could *add* a path would reintroduce the invisible one. They exist because
``tests/`` is not separable by prefix: the ~100 test modules whose subject is a
``scripts/`` tool that does not ship sit in the same directories as the tests
that must ship, so "everything under tests/ except these" has no allow-only
spelling.

A third mechanism decides CONTENT rather than membership: ``--overlay`` names a
directory whose files **replace** their exported counterparts. It is replace-only
and never adds a path, so the allowlist remains the single answer to "what ships"
and the overlay answers only "what does this shipped file contain". See
``read_overlay``.

Three ratchets close rather than open, all reported as errors under --strict:

* a top-level root matched by nothing is listed, so dropping a whole directory
  is always a stated decision rather than an oversight;
* a pattern that changes nothing is **dead** -- delete it. An allowance is dead
  when it ships no file; a denial is dead when it removes none. The rule is
  symmetric on purpose: a denial kept after its target stopped existing reads
  as a stated exclusion while excluding nothing. The same rule the rename guard
  applies to its ALLOWED_ROOTS;
* an overlay file that replaces no shipped path is **dead** by the same symmetry --
  it reads as a stated substitution while substituting nothing, and the file it was
  meant to correct ships uncorrected.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def git(*args: str, cwd: Path, binary: bool = False):
    out = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True
    ).stdout
    return out if binary else out.decode("utf-8", "surrogateescape")


def parse_allowlist(text: str, origin: str) -> tuple[list[str], list[str]]:
    """Split the text into (allowances, denials); ``!`` prefixes a denial.

    Takes text rather than a path so the caller decides *where the text came
    from*. That choice is the provenance claim -- see ``read_allowlist``.
    """
    allows: list[str] = []
    denies: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        (denies if line.startswith("!") else allows).append(line.removeprefix("!").strip())
    # Keyed on the ALLOWANCES alone. A file holding nothing but denials names
    # no path to ship, so it is the empty case wearing a different shape.
    if not allows:
        raise SystemExit(f"{origin}: no allowances -- refusing to export an empty tree")
    return allows, denies


def read_allowlist(
    repo: Path, resolved: str, rel: Path, from_worktree: bool
) -> tuple[str, str]:
    """Return (text, origin) for the allowlist, pinned to ``resolved`` by default.

    This closes a provenance hole rather than tidying one. Every blob written by
    this script is read at the pinned SHA, and the manifest's ``git_sha`` is sold
    as pinning the exported file set -- but the allowlist *selects* that set, and
    reading it from the working tree meant an unchanged SHA could produce two
    different exports. It did: 4672 files one run and 4673 the next, same HEAD,
    the only difference an uncommitted allowlist edit. A frozen SHA plus a
    floating selector is not a freeze.

    ``--allowlist-from-worktree`` stays available because iterating on the
    allowlist without committing each attempt is the normal way to build one. It
    is refused under ``--strict``, and stamped into the manifest when used, so a
    tree exported that way can never be mistaken for a reproducible one.
    """
    if from_worktree:
        path = rel if rel.is_absolute() else repo / rel
        return path.read_text(encoding="utf-8"), f"{path} (WORKTREE, unpinned)"
    if rel.is_absolute():
        raise SystemExit(
            f"--allowlist {rel} is an absolute path, so it cannot be read from "
            f"{resolved[:12]}. Pass a repo-relative path, or accept an unpinned "
            "export with --allowlist-from-worktree."
        )
    try:
        text = git("show", f"{resolved}:{rel.as_posix()}", cwd=repo)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"{rel} does not exist at {resolved[:12]}. Commit it before exporting, "
            "or pass --allowlist-from-worktree to accept an unpinned export.\n"
            f"  git: {exc.stderr.decode('utf-8', 'replace').strip()}"
        ) from exc
    return text, f"{rel} @ {resolved[:12]}"


def matches(rel: str, pattern: str) -> bool:
    """A trailing '/' makes a pattern a directory prefix; otherwise fnmatch."""
    if pattern.endswith("/"):
        return rel.startswith(pattern)
    return fnmatch.fnmatch(rel, pattern)


def read_overlay(repo: Path, resolved: str, overlay_root: str) -> dict[str, bytes]:
    """Read the public overlay, pinned to ``resolved`` like everything else.

    An overlay file at ``<overlay_root>/<path>`` REPLACES the content of the exported
    ``<path>``. It exists for the handful of files that are correct on this branch and
    wrong in the distribution -- the docs sidebar being the clearest case, because the
    published site's ``toctree`` must name only pages that ship, while this tree's must
    also reach ``docs/api/`` (175 pages), ``docs/contributing/`` and
    ``cluster_verification``. Those are two different documents for two different sites,
    not one invariant with two owners.

    **Replace-only, never add.** An overlay whose target is not already shipping is
    reported as dead rather than written. This is the same fail-closed direction the
    allowlist enforces: if the overlay could ADD a path, it would become a second way to
    publish a file, and the allowlist would stop being the single answer to "what ships".
    The overlay decides only what a shipped file CONTAINS.

    Before this existed the substitution was done by hand in the published repository,
    which is how a 105-line ``docs/index.rst`` came to live there and nowhere else --
    invisible to this branch, and destroyed by the next export.
    """
    listing = git("ls-tree", "-r", "-z", "--name-only", f"{resolved}:{overlay_root}",
                  cwd=repo).split("\0")
    return {
        rel: git("show", f"{resolved}:{overlay_root}/{rel}", cwd=repo, binary=True)
        for rel in listing
        if rel
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sha", required=True, help="commit to export (pinned, not a branch)")
    ap.add_argument("--out", required=True, type=Path, help="output directory (must not exist)")
    ap.add_argument("--repo", default=".", type=Path)
    ap.add_argument("--allowlist", type=Path, default=Path("scripts/release/public_allowlist.txt"))
    ap.add_argument(
        "--allowlist-from-worktree",
        action="store_true",
        help="read the allowlist from the working tree instead of --sha "
        "(unpinned: the export is then not reproducible from the SHA alone)",
    )
    ap.add_argument("--overlay", default="scripts/release/public_overlay",
                    help="repo-relative directory whose files REPLACE their exported "
                         "counterparts (replace-only; never adds a path)")
    ap.add_argument("--strict", action="store_true", help="exit 2 on any dead allowance")
    args = ap.parse_args()

    repo = args.repo.resolve()
    resolved = git("rev-parse", args.sha, cwd=repo).strip()

    # ``-z`` is load-bearing, not a tidiness flag. Without it git C-QUOTES any
    # path holding a double-quote, backslash, control character or non-ASCII
    # byte -- wrapping it in '"' and escaping the interior -- so the parser
    # stores git's *rendering* of the path instead of the path. This tree has
    # four such entries today; none is allowlisted, so the bug is currently
    # silent, and a fail-closed gate must not carry a silent path hole.
    # NUL-separated records cannot be ambiguous and are therefore never quoted.
    #
    # Read MODES, not just names. A tree holds three kinds of entry and only one
    # of them is a file: 160000 is a submodule gitlink (``git show`` exits 128 on
    # it) and 120000 is a symlink (whose blob is the target path -- writing that
    # as a regular file silently converts the link into a file containing text).
    entries: dict[str, str] = {}
    for record in git("ls-tree", "-r", "-z", resolved, cwd=repo).split("\0"):
        if not record:
            continue
        meta, rel = record.split("\t", 1)
        mode = meta.split(" ", 1)[0]
        entries[rel] = mode
    tracked = list(entries)
    if args.allowlist_from_worktree and args.strict:
        raise SystemExit(
            "--allowlist-from-worktree is refused under --strict: a strict export "
            "asserts the tree is reproducible from --sha, and an uncommitted "
            "allowlist is exactly what makes it not."
        )
    allowlist_text, allowlist_origin = read_allowlist(
        repo, resolved, args.allowlist, args.allowlist_from_worktree
    )
    allowlist_sha256 = hashlib.sha256(allowlist_text.encode("utf-8")).hexdigest()
    allows, denies = parse_allowlist(allowlist_text, allowlist_origin)

    shipped: list[str] = []
    denied: list[str] = []
    used_allow: set[str] = set()
    used_deny: set[str] = set()
    overlay_sources: list[str] = []
    for rel in tracked:
        if rel == args.overlay or rel.startswith(f"{args.overlay}/"):
            # Overlay storage is outside the allowlist's jurisdiction -- the allowlist
            # never inspects the overlay -- so it is skipped BEFORE the allowlist is
            # consulted. Without this a bare `scripts/release/` allowance publishes the
            # overlay tree at its STORAGE path as well as at its mapped path: the same
            # bytes twice under two names, and the storage copy is the one nothing reads.
            # It is neither shipped nor `excluded`; it is content that ships elsewhere.
            overlay_sources.append(rel)
            continue
        allowed_by = next((p for p in allows if matches(rel, p)), None)
        if allowed_by is None:
            continue
        denied_by = next((p for p in denies if matches(rel, p)), None)
        if denied_by is not None:
            used_deny.add(denied_by)
            denied.append(rel)
            continue
        # Marked live only once the path actually SHIPS, so an allowance whose
        # every match is cancelled by a denial reads as dead rather than as
        # working -- the pair is then deleted together instead of drifting.
        used_allow.add(allowed_by)
        shipped.append(rel)

    dead = [p for p in allows if p not in used_allow] + [
        f"!{p}" for p in denies if p not in used_deny
    ]
    excluded = set(tracked) - set(shipped) - set(overlay_sources)
    dropped_roots = sorted({r.split("/")[0] for r in excluded} - {s.split("/")[0] for s in shipped})

    if args.out.exists():
        raise SystemExit(f"{args.out} exists -- refusing to write into a non-empty tree")
    args.out.mkdir(parents=True)

    try:
        overlay = read_overlay(repo, resolved, args.overlay)
    except subprocess.CalledProcessError:
        overlay = {}          # no overlay directory at this SHA
    dead_overlay = sorted(set(overlay) - set(shipped))
    overlaid: list[str] = []
    # An overlay whose content EQUALS the tracked blob it replaces. The dead-overlay
    # ratchet cannot see this shape -- the target ships, so the substitution is
    # structurally valid -- and today the two copies agree, so nothing looks wrong. The
    # first edit to the tracked file is then silently discarded in the export and the
    # published page freezes, with no gate red. One invariant, one owner: the overlay
    # copy is only entitled to exist while it says something different.
    redundant_overlay: list[str] = []

    manifest = []
    submodules = []
    for rel in shipped:
        mode = entries[rel]
        dest = args.out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if mode == "160000":
            # A gitlink carries no content. Record the pinned commit and let
            # .gitmodules describe it; extracting it would vendor third-party
            # code the licence review treats as a pointer.
            submodules.append({"path": rel, "commit": git("rev-parse", f"{resolved}:{rel}", cwd=repo).strip()})
            continue
        blob = git("show", f"{resolved}:{rel}", cwd=repo, binary=True)
        if rel in overlay:
            if overlay[rel] == blob:
                redundant_overlay.append(rel)
            # Substituted BEFORE the manifest hash is taken, so `sha256` describes
            # what is on disk rather than what git holds. A manifest that hashed the
            # pre-overlay blob would be a provenance claim about a file the export
            # does not contain.
            blob = overlay[rel]
            overlaid.append(rel)
        if mode == "120000":
            dest.symlink_to(blob.decode("utf-8", "surrogateescape"))
        else:
            dest.write_bytes(blob)
            if mode == "100755":
                dest.chmod(0o755)
        manifest.append(
            {"path": rel, "mode": mode, "size": len(blob),
             "sha256": hashlib.sha256(blob).hexdigest()}
        )

    (args.out / "EXPORT_MANIFEST.json").write_text(
        json.dumps(
            {
                "git_sha": resolved,
                "allowlist_origin": allowlist_origin,
                "allowlist_sha256": allowlist_sha256,
                "allowlist_pinned": not args.allowlist_from_worktree,
                "file_count": len(manifest),
                "total_bytes": sum(m["size"] for m in manifest),
                "excluded_count": len(excluded),
                "submodules": submodules,
                "dropped_top_level_roots": dropped_roots,
                "dead_allowlist_patterns": dead,
                "overlay_root": args.overlay,
                "overlaid_paths": sorted(overlaid),
                "dead_overlay_files": dead_overlay,
                "redundant_overlay_files": sorted(redundant_overlay),
                "overlay_source_paths": sorted(overlay_sources),
                "denied_patterns": denies,
                "denied_paths": sorted(denied),
                "files": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"sha        : {resolved}")
    print(f"allowlist  : {allowlist_origin}")
    print(f"           : sha256 {allowlist_sha256[:16]}")
    if args.allowlist_from_worktree:
        print("WARNING: allowlist read from the working tree -- this export is NOT")
        print("         reproducible from --sha alone. Commit it and re-export.")
    print(f"tracked    : {len(tracked)}")
    print(f"shipped    : {len(shipped)}")
    print(f"excluded   : {len(excluded)}")
    print(f"denied     : {len(denied)} path(s) removed by {len(denies)} denial(s)")
    print(f"bytes      : {sum(m['size'] for m in manifest):,}")
    print(f"overlaid   : {len(overlaid)} file(s) replaced from {args.overlay}/")
    # Every path, not just the count. An overlay substitution is VALID whenever
    # its target ships, so the dead-overlay ratchet cannot flag one you did not
    # intend -- a stray `README.md` at the overlay root silently replaced the
    # project README, and only the count being 2 instead of 1 gave it away.
    for p_ in sorted(overlaid):
        print(f"           : {p_}")
    print(f"submodules : {len(submodules)} (recorded as pinned commits, not vendored)")
    print(f"dropped roots ({len(dropped_roots)}): {', '.join(dropped_roots) or '-'}")
    if dead_overlay:
        # Symmetric with a dead allowance: an overlay that replaces nothing reads as a
        # stated substitution while substituting nothing, and the file it was meant to
        # correct ships uncorrected.
        print(f"DEAD OVERLAY ({len(dead_overlay)}) -- these replace no shipped path:")
        for p_ in dead_overlay:
            print(f"  {p_}")
        if args.strict:
            return 2

    if redundant_overlay:
        print(f"REDUNDANT OVERLAY ({len(redundant_overlay)}) -- identical to the tracked file:")
        for p_ in redundant_overlay:
            print(f"  {p_}")
        if args.strict:
            return 2

    if dead:
        print(f"DEAD PATTERNS ({len(dead)}) -- delete these from the allowlist:")
        for p in dead:
            print(f"  {p}")
        if args.strict:
            return 2

    # Non-negotiable 16: this gate was written for exactly this tree and had
    # never once run against it. Its only invocations are in the *published*
    # `pr-required.yml`, and spectramr receives snapshots as direct pushes to
    # `main`, never PRs -- so the single lane that runs it cannot fire on a
    # snapshot. The research repository's CI does not run it either. Pointing it
    # at the export here is what turns it from a script into a gate.
    #
    # It runs from THIS checkout, never from the export's own shipped copy: the
    # export is the subject under test, and a tree that stopped shipping the
    # gate would otherwise quietly stop being checked by it.
    doc_pages = [r for r in shipped if r.endswith((".rst", ".md"))]
    if not doc_pages:
        # Not a vacuous pass. An export shipping no prose has no pasteable line
        # to dangle, and the shape where docs were MEANT to ship and did not is
        # already owned by the dead-allowance check above -- every `docs/`
        # allowance would read as dead. One owner per invariant (17).
        print("docs gate  : skipped -- this export ships no .rst/.md page")
    else:
        gate = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_docs_paths_exist.py"
        if not gate.is_file():
            print(f"FAIL: {gate} is missing -- refusing to publish an unchecked doc set.")
            return 2
        # The count is the TRIGGER, not the scan: the gate reads docs/ and the
        # root-level pages, while this counts every shipped .rst/.md. Saying
        # "over N pages" would overstate what was checked.
        print(f"docs gate  : {len(doc_pages)} .rst/.md shipped -- running {gate.name}")
        rc = subprocess.run(
            [sys.executable, str(gate), "--root", str(args.out), "--commands-only"],
            check=False,
        ).returncode
        if rc != 0:
            print(
                "FAIL: a published page tells a reader to run a path this export does\n"
                "      not carry. This is unconditional, unlike a dead allowance: it is\n"
                "      a defect in the product rather than in the export's configuration,\n"
                "      so there is no non-strict reading under which it is acceptable."
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
