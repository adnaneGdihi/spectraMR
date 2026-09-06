"""Contract tests for ``scripts/release/export_public_tree.py``.

The export script is a **fail-closed gate**: a tracked path ships only if the
allowlist names it. Non-negotiable 15 says a gate is only a gate for the
violation shape it has been watched to fail on, so every clause below is
exercised against a synthetic repository carrying each entry shape a real tree
holds -- regular file, executable, symlink, submodule gitlink, and a path whose
name forces ``git`` to C-quote it.

That last shape is not hypothetical. Four tracked paths under
``experiments/inference/`` contain a double quote, and reading ``git ls-tree``
without ``-z`` stores git's *rendering* of such a path (``"a/\"b\".yaml"``)
rather than the path. It shipped nothing wrong, because none of the four is
allowlisted -- it corrupted the dropped-roots ratchet instead, which reported a
top-level root that does not exist.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "release" / "export_public_tree.py"
REAL_ALLOWLIST = SCRIPT.parent / "public_allowlist.txt"

# A gitlink may point at a commit that exists nowhere; git records the id
# without resolving it, which is what makes this plantable without a submodule.
FAKE_SUBMODULE_SHA = "a" * 40

# Two shapes that force git to C-quote a path in non-``-z`` output.
QUOTED_BY_DQUOTE = 'odd/"quoted"_name.yaml'
QUOTED_BY_NONASCII = "odd/café.yaml"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway repo holding one of every tree-entry shape."""
    root = tmp_path / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "secrets").mkdir()
    (root / "odd").mkdir()

    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True, capture_output=True)
    _git("config", "user.email", "test@example.invalid", cwd=root)
    _git("config", "user.name", "Test", cwd=root)

    (root / "src" / "pkg" / "mod.py").write_text("VALUE = 1\n")
    (root / "src" / "pkg" / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (root / "src" / "pkg" / "run.sh").chmod(0o755)
    (root / "src" / "pkg" / "link.py").symlink_to("mod.py")
    (root / "secrets" / "token.txt").write_text("DO-NOT-SHIP\n")
    (root / "README.md").write_text("# readme\n")
    (root / QUOTED_BY_DQUOTE).write_text("quoted: true\n")
    (root / QUOTED_BY_NONASCII).write_text("nonascii: true\n")

    _git("add", "-A", cwd=root)
    # A submodule gitlink, planted directly into the index. No submodule needed:
    # git stores the id it is given and never resolves it.
    _git(
        "update-index", "--add", "--cacheinfo", f"160000,{FAKE_SUBMODULE_SHA},vendor/dep", cwd=root
    )
    _git("commit", "-qm", "seed", cwd=root)
    return root


def _allowlist(tmp_path: Path, *patterns: str, name: str = "allow.txt") -> Path:
    path = tmp_path / name
    path.write_text("".join(f"{p}\n" for p in patterns))
    return path


REL_ALLOWLIST = "release_allowlist.txt"


def _export(repo: Path, allowlist: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    """Run the export, committing the allowlist into the fixture repo first.

    The script reads the allowlist at ``--sha``, not from the working tree, so a
    caller passing an out-of-repo path exercises the ``--allowlist-from-worktree``
    escape hatch instead of the default. Committing it here means all 30-odd
    tests below cover the path the release actually uses; the escape hatch gets
    its own tests in the provenance section.
    """
    (repo / REL_ALLOWLIST).write_text(allowlist.read_text())
    _git("add", REL_ALLOWLIST, cwd=repo)
    _git("commit", "-qm", "allowlist", "--allow-empty", cwd=repo)
    sha = _git("rev-parse", "HEAD", cwd=repo).strip()
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sha",
            sha,
            "--out",
            str(out),
            "--repo",
            str(repo),
            "--allowlist",
            REL_ALLOWLIST,
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def _export_unpinned(
    repo: Path, allowlist: Path, out: Path, *extra: str
) -> subprocess.CompletedProcess:
    """Export reading the allowlist from the working tree, without committing."""
    sha = _git("rev-parse", "HEAD", cwd=repo).strip()
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sha",
            sha,
            "--out",
            str(out),
            "--repo",
            str(repo),
            "--allowlist",
            str(allowlist),
            "--allowlist-from-worktree",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def _manifest(out: Path) -> dict:
    return json.loads((out / "EXPORT_MANIFEST.json").read_text())


# --------------------------------------------------------------------------- #
# Fail-closed: the whole point of an allowlist
# --------------------------------------------------------------------------- #
def test_an_unlisted_path_does_not_ship(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert _export(repo, _allowlist(tmp_path, "src/"), out).returncode == 0
    assert (out / "src" / "pkg" / "mod.py").exists()
    assert not (out / "secrets").exists(), "an unlisted root was exported"
    assert not any(m["path"].startswith("secrets/") for m in _manifest(out)["files"])


def test_an_unlisted_path_is_counted_as_excluded(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _export(repo, _allowlist(tmp_path, "src/"), out)
    manifest = _manifest(out)
    assert manifest["excluded_count"] > 0
    assert "secrets" in manifest["dropped_top_level_roots"]


# --------------------------------------------------------------------------- #
# Entry shapes: a tree holds three kinds of thing and only one is a file
# --------------------------------------------------------------------------- #
def test_executable_bit_survives_the_export(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _export(repo, _allowlist(tmp_path, "src/"), out)
    exported = out / "src" / "pkg" / "run.sh"
    assert exported.stat().st_mode & 0o111, "100755 was exported without its executable bit"


def test_a_symlink_stays_a_symlink(repo: Path, tmp_path: Path) -> None:
    """A 120000 blob IS the target path; writing it as a file yields a
    plain-text file whose contents happen to read ``mod.py``."""
    out = tmp_path / "out"
    _export(repo, _allowlist(tmp_path, "src/"), out)
    exported = out / "src" / "pkg" / "link.py"
    assert exported.is_symlink(), "symlink was materialised as a regular file"
    assert Path.readlink(exported) == Path("mod.py")


def test_a_gitlink_is_recorded_and_never_extracted(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    proc = _export(repo, _allowlist(tmp_path, "src/", "vendor/"), out)
    assert proc.returncode == 0, proc.stderr
    manifest = _manifest(out)
    assert manifest["submodules"] == [{"path": "vendor/dep", "commit": FAKE_SUBMODULE_SHA}]
    assert not (out / "vendor" / "dep").exists(), "a gitlink was written to disk"
    assert not any(m["path"] == "vendor/dep" for m in manifest["files"])


# --------------------------------------------------------------------------- #
# C-quoted paths -- the regression this file was written for
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel", [QUOTED_BY_DQUOTE, QUOTED_BY_NONASCII])
def test_a_quoted_path_exports_under_its_real_name(repo: Path, tmp_path: Path, rel: str) -> None:
    out = tmp_path / "out"
    proc = _export(repo, _allowlist(tmp_path, "odd/"), out)
    assert proc.returncode == 0, proc.stderr
    assert (out / rel).read_text() == (repo / rel).read_text()
    assert any(m["path"] == rel for m in _manifest(out)["files"]), (
        "the manifest records git's quoted rendering, not the path"
    )


@pytest.mark.parametrize("rel", [QUOTED_BY_DQUOTE, QUOTED_BY_NONASCII])
def test_a_quoted_path_stays_excluded_when_unlisted(repo: Path, tmp_path: Path, rel: str) -> None:
    out = tmp_path / "out"
    _export(repo, _allowlist(tmp_path, "src/"), out)
    assert not (out / "odd").exists()
    assert not any(m["path"] == rel for m in _manifest(out)["files"])


def test_dropped_roots_never_names_a_root_that_does_not_exist(repo: Path, tmp_path: Path) -> None:
    """The quoting bug invented a phantom root by quoting the whole path, so the
    first path segment carried a leading double quote."""
    out = tmp_path / "out"
    _export(repo, _allowlist(tmp_path, "src/"), out)
    real_roots = {
        p.split("/")[0]
        for p in _git("ls-tree", "-r", "-z", "--name-only", "HEAD", cwd=repo).split("\0")
        if p
    }
    assert set(_manifest(out)["dropped_top_level_roots"]) <= real_roots


# --------------------------------------------------------------------------- #
# Ratchets close rather than open
# --------------------------------------------------------------------------- #
def test_a_dead_allowance_is_reported(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    proc = _export(repo, _allowlist(tmp_path, "src/", "nothing/matches/this/"), out)
    assert _manifest(out)["dead_allowlist_patterns"] == ["nothing/matches/this/"]
    assert "DEAD PATTERNS" in proc.stdout


def test_a_dead_allowance_fails_only_under_strict(repo: Path, tmp_path: Path) -> None:
    allow = _allowlist(tmp_path, "src/", "nothing/matches/this/")
    assert _export(repo, allow, tmp_path / "lax").returncode == 0
    assert _export(repo, allow, tmp_path / "strict", "--strict").returncode == 2


def test_a_live_allowlist_is_clean_under_strict(repo: Path, tmp_path: Path) -> None:
    """The strict gate must be able to pass, or it says nothing when it fails."""
    proc = _export(repo, _allowlist(tmp_path, "src/"), tmp_path / "out", "--strict")
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------- #
# Denials -- one test per shape the ``!`` mechanism can take
#
# Denials exist because ``tests/`` is not separable by prefix: ~100 test modules
# whose subject is a ``scripts/`` tool that does not ship sit in the same
# directories as tests that must. They are safe only because they SUBTRACT --
# every clause below pins that direction, since a denial that could add a path
# would undo the fail-closed property the rest of this file exists to defend.
# --------------------------------------------------------------------------- #
def test_a_denial_removes_an_allowed_file(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert _export(repo, _allowlist(tmp_path, "src/", "!src/pkg/mod.py"), out).returncode == 0
    assert not (out / "src" / "pkg" / "mod.py").exists()
    # The siblings prove the denial was surgical rather than the allowance failing.
    assert (out / "src" / "pkg" / "run.sh").exists()


def test_a_denial_never_adds_a_path(repo: Path, tmp_path: Path) -> None:
    """The direction that keeps the gate fail-closed: denials only subtract."""
    plain = tmp_path / "plain"
    _export(repo, _allowlist(tmp_path, "src/", name="a.txt"), plain)
    with_deny = tmp_path / "denied"
    _export(repo, _allowlist(tmp_path, "src/", "!src/pkg/mod.py", name="b.txt"), with_deny)
    before = {f["path"] for f in _manifest(plain)["files"]}
    after = {f["path"] for f in _manifest(with_deny)["files"]}
    assert after < before, "a denial must produce a strict subset, never a new path"


def test_a_denial_may_be_a_directory_prefix(repo: Path, tmp_path: Path) -> None:
    """The trailing-slash arm of ``matches`` -- the shape the real list uses most."""
    out = tmp_path / "out"
    allow = _allowlist(tmp_path, "src/", "README.md", "!src/pkg/")
    assert _export(repo, allow, out).returncode == 0
    assert not (out / "src").exists()
    assert (out / "README.md").exists()


def test_a_denial_may_be_a_glob(repo: Path, tmp_path: Path) -> None:
    """The fnmatch arm -- a shape a prefix denial cannot express."""
    out = tmp_path / "out"
    assert _export(repo, _allowlist(tmp_path, "src/", "!*.sh"), out).returncode == 0
    assert not (out / "src" / "pkg" / "run.sh").exists()
    assert (out / "src" / "pkg" / "mod.py").exists()


@pytest.mark.parametrize("order", [("!src/pkg/mod.py", "src/"), ("src/", "!src/pkg/mod.py")])
def test_a_denial_wins_regardless_of_line_order(
    repo: Path, tmp_path: Path, order: tuple[str, ...]
) -> None:
    """Allowances short-circuit on first match; denials must not be order-sensitive."""
    out = tmp_path / "out"
    assert _export(repo, _allowlist(tmp_path, *order), out).returncode == 0
    assert not (out / "src" / "pkg" / "mod.py").exists()


def test_a_dead_denial_is_reported_and_fails_under_strict(repo: Path, tmp_path: Path) -> None:
    """Symmetric with a dead allowance: a denial that removes nothing is a lie.

    It reads as a stated exclusion while excluding nothing, so a reader auditing
    the list believes a path is held out that was never there to begin with.

    The live-denial leg is what makes this test discriminate. Without it the
    dead leg passes under PURE-ALLOW semantics too: a parser that never splits
    on ``!`` reads the raw line as an allowance, finds no path starting with
    ``!``, and reports the identical string as dead. Only a build that
    understands denials can leave a WORKING one unreported.
    """
    dead = _allowlist(tmp_path, "src/", "!nothing/matches/this/", name="dead.txt")
    out = tmp_path / "lax"
    proc = _export(repo, dead, out)
    assert proc.returncode == 0
    assert _manifest(out)["dead_allowlist_patterns"] == ["!nothing/matches/this/"]
    assert _export(repo, dead, tmp_path / "strict", "--strict").returncode == 2

    live = _allowlist(tmp_path, "src/", "!src/pkg/mod.py", name="live.txt")
    out_live = tmp_path / "live"
    proc_live = _export(repo, live, out_live, "--strict")
    assert proc_live.returncode == 0, proc_live.stdout + proc_live.stderr
    assert _manifest(out_live)["dead_allowlist_patterns"] == []


def test_an_allowance_cancelled_entirely_by_a_denial_reads_as_dead(
    repo: Path, tmp_path: Path
) -> None:
    """Both halves of a fully-cancelled pair are reported, so both get deleted."""
    out = tmp_path / "out"
    proc = _export(repo, _allowlist(tmp_path, "README.md", "src/", "!src/"), out)
    assert set(_manifest(out)["dead_allowlist_patterns"]) == {"src/"}
    assert proc.returncode == 0


def test_the_manifest_records_the_denials_as_a_stated_decision(repo: Path, tmp_path: Path) -> None:
    """An exclusion this script makes is recorded, never merely performed."""
    out = tmp_path / "out"
    _export(repo, _allowlist(tmp_path, "src/", "!src/pkg/mod.py"), out)
    manifest = _manifest(out)
    assert manifest["denied_patterns"] == ["src/pkg/mod.py"]
    assert manifest["denied_paths"] == ["src/pkg/mod.py"]


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_an_allowlist_of_only_denials_is_refused(repo: Path, tmp_path: Path) -> None:
    """The empty case wearing a different shape: denials name nothing to ship."""
    proc = _export(repo, _allowlist(tmp_path, "!src/", "!README.md"), tmp_path / "out")
    assert proc.returncode != 0
    assert "refusing to export an empty tree" in proc.stderr


def test_an_empty_allowlist_is_refused(repo: Path, tmp_path: Path) -> None:
    allow = _allowlist(tmp_path, "# only a comment", "")
    proc = _export(repo, allow, tmp_path / "out")
    assert proc.returncode != 0
    assert "refusing to export an empty tree" in proc.stderr


def test_an_existing_out_dir_is_refused(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    proc = _export(repo, _allowlist(tmp_path, "src/"), out)
    assert proc.returncode != 0
    assert "refusing to write" in proc.stderr


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Provenance: the SHA must pin BOTH halves
#
# ``git_sha`` is sold as pinning the exported file set. The blobs were always
# read at that SHA, but the allowlist -- which *selects* which blobs -- was read
# from the working tree, so an unchanged SHA could produce two different trees.
# It did: 4672 files one run and 4673 the next, same HEAD, the only difference an
# uncommitted allowlist edit. Each clause below is a planted violation of the
# fixed behaviour, and the escape-hatch test is what stops the first one from
# passing vacuously -- a flag that read the same bytes either way would satisfy
# "the edit was ignored" while proving nothing.
# --------------------------------------------------------------------------- #
def test_an_uncommitted_allowlist_edit_does_not_change_a_pinned_export(
    repo: Path, tmp_path: Path
) -> None:
    """The plant: widen the allowlist in the worktree only, keep the SHA."""
    out_before = tmp_path / "before"
    assert _export(repo, _allowlist(tmp_path, "src/"), out_before).returncode == 0
    assert not (out_before / "secrets").exists()

    sha = _git("rev-parse", "HEAD", cwd=repo).strip()
    (repo / REL_ALLOWLIST).write_text("src/\nsecrets/\n")  # dirty, uncommitted

    out_after = tmp_path / "after"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sha",
            sha,
            "--out",
            str(out_after),
            "--repo",
            str(repo),
            "--allowlist",
            REL_ALLOWLIST,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (out_after / "secrets").exists(), (
        "an uncommitted allowlist edit changed a pinned export -- the SHA does not pin the tree"
    )
    assert _manifest(out_after)["file_count"] == _manifest(out_before)["file_count"]


def test_the_worktree_escape_hatch_actually_reads_the_worktree(repo: Path, tmp_path: Path) -> None:
    """Anti-vacuity partner to the test above.

    If ``--allowlist-from-worktree`` read the pinned bytes too, the pinning test
    would pass for the wrong reason. This proves the dirty edit is visible when
    the flag asks for it, so ignoring it above is a real decision.
    """
    assert _export(repo, _allowlist(tmp_path, "src/"), tmp_path / "seed").returncode == 0
    (repo / REL_ALLOWLIST).write_text("src/\nsecrets/\n")  # same dirty edit

    out = tmp_path / "out"
    proc = _export_unpinned(repo, repo / REL_ALLOWLIST, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "secrets" / "token.txt").exists(), (
        "the escape hatch did not read the working tree, so the pinning test above proves nothing"
    )


def test_the_manifest_records_whether_the_allowlist_was_pinned(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _export(repo, _allowlist(tmp_path, "src/"), out)
    manifest = _manifest(out)
    assert manifest["allowlist_pinned"] is True
    assert len(manifest["allowlist_sha256"]) == 64
    assert _git("rev-parse", "HEAD", cwd=repo).strip()[:12] in manifest["allowlist_origin"]


def test_an_unpinned_export_says_so_in_its_manifest(repo: Path, tmp_path: Path) -> None:
    """An unpinned tree must never be mistakable for a reproducible one."""
    allow = _allowlist(tmp_path, "src/")
    out = tmp_path / "out"
    proc = _export_unpinned(repo, allow, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = _manifest(out)
    assert manifest["allowlist_pinned"] is False
    assert "WORKTREE" in manifest["allowlist_origin"]
    assert "not" in proc.stdout.lower() and "reproducible" in proc.stdout.lower()


def test_an_unpinned_export_is_refused_under_strict(repo: Path, tmp_path: Path) -> None:
    """--strict asserts the tree is reproducible from --sha; unpinned is not."""
    proc = _export_unpinned(repo, _allowlist(tmp_path, "src/"), tmp_path / "out", "--strict")
    assert proc.returncode != 0
    assert "refused under --strict" in proc.stderr


def test_an_allowlist_absent_at_the_sha_is_a_hard_error(repo: Path, tmp_path: Path) -> None:
    """Not a silent fall back to the working tree, which is the whole defect."""
    (repo / "only_in_worktree.txt").write_text("src/\n")
    sha = _git("rev-parse", "HEAD", cwd=repo).strip()
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sha",
            sha,
            "--out",
            str(tmp_path / "out"),
            "--repo",
            str(repo),
            "--allowlist",
            "only_in_worktree.txt",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "does not exist at" in proc.stderr
    assert not (tmp_path / "out").exists()


def test_an_absolute_allowlist_path_is_refused_without_the_escape_hatch(
    repo: Path, tmp_path: Path
) -> None:
    """An out-of-repo file cannot be pinned, so it must be asked for explicitly."""
    allow = _allowlist(tmp_path, "src/")
    sha = _git("rev-parse", "HEAD", cwd=repo).strip()
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sha",
            sha,
            "--out",
            str(tmp_path / "out"),
            "--repo",
            str(repo),
            "--allowlist",
            str(allow),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "absolute path" in proc.stderr


def test_the_manifest_pins_the_resolved_sha(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _export(repo, _allowlist(tmp_path, "src/"), out)
    assert _manifest(out)["git_sha"] == _git("rev-parse", "HEAD", cwd=repo).strip()


def test_the_manifest_hashes_the_bytes_that_were_written(repo: Path, tmp_path: Path) -> None:
    import hashlib

    out = tmp_path / "out"
    _export(repo, _allowlist(tmp_path, "src/"), out)
    for rec in _manifest(out)["files"]:
        if rec["mode"] == "120000":
            continue
        blob = (out / rec["path"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == rec["sha256"]
        assert len(blob) == rec["size"]


# --------------------------------------------------------------------------- #
# The one fail-OPEN direction, pinned so the next pattern is written knowingly
# --------------------------------------------------------------------------- #
def test_a_star_pattern_crosses_a_slash(repo: Path, tmp_path: Path) -> None:
    """``fnmatch``'s ``*`` is not a shell glob: it matches ``/`` too, so
    ``*.md`` ships every markdown file at every depth. This is the only way the
    allowlist can over-ship, and the real allowlist therefore carries no
    ``*`` pattern (see the test below)."""
    out = tmp_path / "out"
    _export(repo, _allowlist(tmp_path, "*.txt"), out)
    shipped = {m["path"] for m in _manifest(out)["files"]}
    assert "secrets/token.txt" in shipped, "'*' no longer crosses '/' -- re-audit the allowlist"


def test_the_shipped_allowlist_carries_no_unbounded_glob() -> None:
    globbed = [
        line.split("#", 1)[0].strip()
        for line in REAL_ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip() and "*" in line.split("#", 1)[0]
    ]
    assert globbed == [], (
        f"'*' crosses '/' in fnmatch, so these patterns ship more than they read as: {globbed}"
    )


def _shipped_experiment_yamls() -> tuple[list[Path], Path]:
    """The experiment YAMLs the real allowlist ships, and the repo root.

    Extracted so the two guards below cannot drift about what "shipped" means
    (non-negotiable 17). Derived from ``parse_allowlist``/``matches`` and
    ``git ls-tree``, never from a directory listing: the exporter drives off a
    pinned SHA, so an untracked YAML in ``experiments/`` is something a listing
    would judge and the exporter would never see.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_export_public_tree", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    allows, denies = mod.parse_allowlist(
        REAL_ALLOWLIST.read_text(encoding="utf-8"), str(REAL_ALLOWLIST)
    )
    repo_root = SCRIPT.resolve().parents[2]

    from tests.utils.corpus import tracked_yamls

    shipped = [
        p
        for p in tracked_yamls(repo_root / "experiments")
        for rel in [p.relative_to(repo_root).as_posix()]
        if any(mod.matches(rel, a) for a in allows) and not any(mod.matches(rel, d) for d in denies)
    ]
    # Anti-vacuity: an empty shipped set passes every assertion downstream, and is
    # exactly what a mistyped allowance would produce.
    assert shipped, (
        "no experiment YAML ships at all -- either the allowlist stopped naming "
        "experiments/ or a denial cancelled every match; the guards would "
        "otherwise pass by having nothing to check"
    )
    return shipped, repo_root


def test_no_shipped_experiment_yaml_declares_a_key_the_schema_discards() -> None:
    """A shipped arm must not teach a knob that does nothing.

    ``extra="ignore"`` blocks drop an undeclared key at parse time with no error
    and no warning: the YAML still shows it, the run never sees it, the arm trains
    on the default (``docs/known_limitations.rst``). In an arm shipped as an
    EXEMPLAR that is worse than in a private one -- it is the first configuration a
    reader copies, and the key they copy is inert.

    Two of the three exemplar arms were in exactly that state: 8 declarations
    across 7 keys, including ``undersampling.acceleration_factor`` and
    ``training.diffusion.num_timesteps`` -- the first two knobs anyone edits.
    ``acceleration_factor`` resolved to the schema default of 4.0, which happened
    to equal the 4 the arm intended, so it was right by luck rather than by
    declaration.

    Uses the repo's own detector rather than re-deriving the predicate; the
    ledger's ``extra_ignore_dropped`` classification is the single owner of what
    "discarded" means.
    """
    import importlib.util

    shipped, repo_root = _shipped_experiment_yamls()
    detector_path = repo_root / "scripts" / "ci" / "report_discarded_config_keys.py"
    spec = importlib.util.spec_from_file_location("_discarded_keys_detector", detector_path)
    assert spec is not None and spec.loader is not None
    detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector)

    offenders: dict[str, list[str]] = {}
    for path in shipped:
        found = detector.discarded_keys(path)
        if found:
            offenders[path.relative_to(repo_root).as_posix()] = sorted(k for k, _ in found)

    assert not offenders, (
        "shipped experiment YAML(s) declare keys the schema silently discards. "
        "Delete the key, or declare it on the schema -- do not leave a shipped "
        "exemplar teaching an inert knob:\n"
        + "\n".join(f"  {f}:\n    " + ", ".join(v) for f, v in offenders.items())
    )


def _shipped_package_name(repo_root: Path) -> str:
    """The installed package name, derived rather than spelled."""
    owners = sorted((repo_root / "src").glob("*/config/settings.py"))
    assert len(owners) == 1, (
        f"expected exactly one src/*/config/settings.py, found {owners} -- "
        "these tests derive the package name from it rather than spelling it"
    )
    return owners[0].parent.parent.name


def test_every_experiment_yaml_that_ships_audits_clean() -> None:
    """Parsing is not passing. A shipped config must survive `audit` too.

    ``test_every_experiment_yaml_that_ships_actually_loads`` below asks only
    whether pydantic accepts the file. It does not run one health check, so a
    template can load perfectly while declaring two knobs that contradict each
    other -- and a reader who copies it meets the contradiction at training
    time, not at copy time.

    Measured, not hypothesised. On 2026-08-29 the one shipped template exited 2
    with two ``health:domain_alignment`` errors: ``model.in_channels=2`` /
    ``out_channels=2`` against ``data.coil_processing_mode='rss_image'``, which
    the channel table in ``config_health_checker.py`` maps to 1 channel. It got
    there because two commits the same afternoon each fixed the template on its
    own branch -- ``7a96822e32`` set the channels for ``rss``, ``af3ca50873``
    set the mode to ``rss_image`` -- and each was genuinely exit 0 where it was
    written. The merge combined them and nothing looked again: no gate ran
    ``audit`` on this file. That is what this test is.

    Run through the CLI in a subprocess rather than by calling the checker in
    process: ``audit`` is the command a reader actually types, and its exit
    code is the contract (0 clean, 2 warnings-under-strict). An in-process call
    would also inherit whatever this session has already imported.
    """
    import subprocess
    import sys

    shipped, repo_root = _shipped_experiment_yamls()
    pkg = _shipped_package_name(repo_root)

    failures = []
    for path in shipped:
        proc = subprocess.run(
            [sys.executable, "-m", f"{pkg}.cli", "audit", str(path)],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=900,
        )
        if proc.returncode != 0:
            tail = "\n      ".join(
                line
                for line in (proc.stdout + proc.stderr).splitlines()
                if "\u274c" in line or "MISMATCH" in line
            )
            failures.append(
                f"{path.relative_to(repo_root)}: audit exited {proc.returncode}\n      {tail}"
            )

    assert not failures, (
        "these YAMLs ship but do not pass `audit`, so a reader copying one "
        "inherits a config the framework's own gate rejects:\n  " + "\n  ".join(failures)
    )


def test_every_experiment_yaml_that_ships_actually_loads() -> None:
    """A shipped template must parse at the current schema.

    A template is *copied*. One that does not load is therefore worse than an
    absent one: the reader's first act is to inherit the defect, and the
    ``config_version`` error they get names a version they never chose. Two of
    the three files under ``experiments/templates/`` were in exactly that state
    (``categorical_values_reference.yaml`` has no ``config_version`` at all;
    ``comprehensive_config_template_v5.0.yaml`` declares the retired ``5.0``,
    which the loader now *rejects* rather than folds), and both are denied.

    The denial is data, so without this test nothing notices when the next
    template is added -- or when one of these two is "fixed" by deleting its
    denial rather than its defect. Driving the shipped set off ``parse_allowlist``
    and ``matches`` rather than off a directory listing is what makes this a
    guard on the *export* instead of a guard on the filesystem: the two differ
    precisely when a denial is wrong.
    """
    import importlib

    # Shipped set derived once, in _shipped_experiment_yamls, so this guard and the
    # discarded-key guard cannot drift about what "shipped" means (NN17). The package
    # directory is renamed by Workstream A, and this module's own header says a gate is
    # only a gate for the shape it was watched to fail on -- a hardcoded package name
    # fails as a COLLECTION error, which reads as infrastructure breakage rather than as
    # this guard firing. So _shipped_package_name derives it, and refuses an ambiguous
    # answer rather than picking one.
    shipped, repo_root = _shipped_experiment_yamls()
    settings_mod = importlib.import_module(f"{_shipped_package_name(repo_root)}.config.settings")

    failures = []
    for path in shipped:
        try:
            settings_mod.TrainingSettings.from_yaml(str(path))
        except Exception as exc:  # the failure text IS the report; narrowing hides a shape
            failures.append(f"{path.relative_to(repo_root)}: {type(exc).__name__}: {exc}")

    assert not failures, (
        "these YAMLs ship but do not load, so a reader copying one inherits a "
        "config the framework rejects:\n  " + "\n  ".join(failures)
    )


# --------------------------------------------------------------------------- #
# The export ships a .gitignore, and the new repo's first act is `git add`
# --------------------------------------------------------------------------- #
NEGATION_FOR_THE_GOLDEN = "!tests/unit/data/builders/_coil_processing_parity_golden.pt"


def _ignored_among(repo_root: Path, rels: list[str]) -> list[str]:
    """Of ``rels``, the ones ``.gitignore`` alone would have git skip.

    ``--no-index`` is not tidiness, it is the whole measurement. Without it git
    consults the index first and reports every *tracked* path as not-ignored,
    whatever the rules say -- and every path this guard checks is tracked, so
    the guard would return an empty list unconditionally and pass forever.
    Measured on the defect this test was written for: the same query returns
    exit 1 ("not ignored") with the index and exit 0 ("ignored") without it.

    That gap is the bug's whole hiding place. A file force-added once is tracked
    for good, so the source checkout behaves correctly no matter what the rules
    say; only a tree that has never seen the file -- the new repository -- obeys
    them. The export ships ``.gitignore`` verbatim, so that tree is the one the
    release creates.
    """
    if not rels:
        return []
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=repo_root,
        input="\n".join(rels),
        capture_output=True,
        text=True,
    )
    # 0 = some path is ignored, 1 = none is. Anything else is git failing, and
    # git failing writes nothing to stdout -- which parses as "nothing ignored"
    # and turns this guard into an unconditional pass. The planted test below
    # found exactly that: it ran check-ignore outside a repository, git exited
    # 128, and the guard reported clean.
    if proc.returncode not in (0, 1):
        raise AssertionError(
            f"git check-ignore failed in {repo_root} (exit {proc.returncode}): "
            f"{proc.stderr.strip()}\nNo answer is not a clean answer."
        )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _shipped_paths() -> tuple[Path, list[str]]:
    """(repo root, every tracked path the allowlist ships), via the exporter."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_export_public_tree", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    allows, denies = mod.parse_allowlist(
        REAL_ALLOWLIST.read_text(encoding="utf-8"), str(REAL_ALLOWLIST)
    )
    repo_root = SCRIPT.resolve().parents[2]
    tracked = [p for p in _git("ls-files", "-z", cwd=repo_root).split("\0") if p]
    shipped = [
        rel
        for rel in tracked
        if any(mod.matches(rel, a) for a in allows) and not any(mod.matches(rel, d) for d in denies)
    ]
    return repo_root, shipped


def test_no_shipped_path_is_ignored_by_the_shipped_gitignore() -> None:
    """Every file the allowlist ships must survive the new repo's first commit.

    Workstream F pushes the export as a single initial commit. That commit is a
    ``git add`` in a tree with no index, so the shipped ``.gitignore`` -- not the
    source repo's history -- decides what lands. A shipped path matching an
    ignore rule is dropped silently: the export manifest still lists it, the
    commit does not contain it, and the two disagree with nothing reporting it.
    """
    repo_root, shipped = _shipped_paths()

    # Anti-vacuity: an empty shipped list satisfies the assertion below, and is
    # what a mistyped allowance or a broken ls-files would produce.
    assert len(shipped) > 1000, (
        f"only {len(shipped)} paths ship -- the selection is broken, and this "
        "guard would otherwise pass by having nothing to check"
    )

    ignored = _ignored_among(repo_root, shipped)
    assert not ignored, (
        f"{len(ignored)} path(s) ship but the shipped .gitignore tells git to "
        "skip them, so the new repo's initial commit omits them while the "
        "export manifest still claims they shipped. Add a negation next to the "
        f"others near {NEGATION_FOR_THE_GOLDEN!r}:\n  " + "\n  ".join(ignored)
    )


def test_the_gitignore_guard_reports_the_defect_it_was_written_for(tmp_path: Path) -> None:
    """Plant: remove the negation and the guard must go red on that exact file.

    Not a synthetic ``*.pt`` in an empty fixture -- the real ``.gitignore`` with
    one line taken out, which is the state this repo was actually in. A guard
    that only fires on a hand-made example is not known to fire on the shape it
    exists to catch.
    """
    repo_root, shipped = _shipped_paths()
    golden = NEGATION_FOR_THE_GOLDEN.lstrip("!")
    assert golden in shipped, (
        f"{golden} no longer ships; this plant pins a path that must exist for "
        "the guard below to mean anything"
    )

    original = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert NEGATION_FOR_THE_GOLDEN in original, (
        "the negation this plant removes is gone -- either it was deleted (the "
        "defect is back) or it was reworded (update this test)"
    )

    # Plant into a copy, never the checkout: a concurrent run in the same
    # worktree would read the defective file and report a failure nothing did.
    planted = tmp_path / "repo"
    planted.mkdir()
    # check-ignore only runs inside a repository; without this it exits 128.
    subprocess.run(["git", "init", "-q", str(planted)], check=True, capture_output=True)
    (planted / ".gitignore").write_text(
        original.replace(NEGATION_FOR_THE_GOLDEN + "\n", ""), encoding="utf-8"
    )

    assert golden in _ignored_among(planted, shipped), (
        "the guard did not report the golden fixture with its negation removed, "
        "so it would not have caught the defect it was written for"
    )


# ---------------------------------------------------------------------------
# Import reachability: a shipped test may not import an unshipped module
# ---------------------------------------------------------------------------
#
# The path census that found 91 broken test files scanned *path construction*.
# It is structurally blind to a test that names its subject through the import
# system instead of the filesystem, and blind again to a helper module that
# constructs the path one level of indirection away from any test. Both shapes
# were found by hand, after the fact, in the export's failure list.
#
# This is the standing detector for the import shape. It runs under pytest --
# note that the blocking CI lane only *collects* tests, so this is a guard on
# every real test run, including the export verification suite, and not a guard
# the required lane executes.

# A dotted name whose first segment is one of these is *in-tree*: whether it
# resolves is decided by the allowlist, not by the environment. Everything else
# (``torch``, ``numpy``, and the installed package itself, which lives under
# ``src/``) is a dependency question rather than an export question.
_IN_TREE_ROOTS = frozenset(
    {"scripts", "tests", "tools", "experiments", "runners", "scratch", "TODO"}
)

# The one unguarded import that is accepted. ``dropped_key_baseline`` imports
# this only inside ``_regenerate()``, which is reachable via ``python -m ...
# --update`` and never under pytest, so the import cannot fail a test run.
# The pair is asserted to still be *observed* below: when the import goes, the
# waiver goes red rather than rotting into a permanent exemption.
_WAIVED_IMPORTS = frozenset(
    {("tests/utils/dropped_key_baseline.py", "tests.smoke.test_deep_config_integrity")}
)


def _module_ships(dotted: str, shipped: set[str]) -> bool:
    """Does ``dotted`` resolve against the **shipped set** (not the filesystem)?

    Resolving against the filesystem is the trap: on the private tree every
    denied module is still on disk, so the check would report clean here and
    red in the export -- the one place it cannot be run before publishing.
    """
    base = dotted.replace(".", "/")
    if f"{base}.py" in shipped or f"{base}/__init__.py" in shipped:
        return True
    # A namespace package (``tests/`` and ``scripts/`` are both) resolves when
    # anything beneath it ships.
    return any(p.startswith(f"{base}/") for p in shipped)


def _importorskip_arguments(tree: ast.AST) -> set[str]:
    """Dotted names passed to ``pytest.importorskip`` anywhere in the file.

    Matched as an ``ast.Call`` on the *name* ``importorskip``, never as a
    substring of the source: a presence pin on text is satisfied by a docstring
    or a comment mentioning the call, which is how a prose-satisfiable check
    scores a file green without a guard.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "importorskip" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.add(first.value)
    return found


def _unshipped_imports(source: str, shipped: set[str]) -> list[str]:
    """In-tree modules ``source`` imports that neither ship nor are guarded.

    ``ast.walk``, not a pass over the module body: the shape this detector
    exists for is the **function-local** import. Every line-anchored check in
    this repo has been blind to exactly that -- CLAUDE.md records it for the
    five ``^``-anchored greps in ``check_layering.sh`` -- and nearly all of the
    unshipped imports found by hand were inside methods.
    """
    tree = ast.parse(source)
    guards = _importorskip_arguments(tree)

    def is_guarded(dotted: str) -> bool:
        # Guarding an ancestor guards its descendants: if ``scripts.sim2rank``
        # is absent then ``scripts.sim2rank.scoring`` is too, and one skip
        # covers both.
        parts = dotted.split(".")
        return any(".".join(parts[: i + 1]) in guards for i in range(len(parts)))

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative: resolves inside the shipped tree by construction.
                continue
            names = [node.module] if node.module else []
        else:
            continue
        for dotted in names:
            if dotted.split(".")[0] not in _IN_TREE_ROOTS:
                continue
            if _module_ships(dotted, shipped) or is_guarded(dotted):
                continue
            found.add(dotted)
    return sorted(found)


def test_no_shipped_test_imports_a_module_that_does_not_ship() -> None:
    """A shipped test may not import an in-tree module the allowlist drops.

    Such a test does not fail informatively -- it raises ``ModuleNotFoundError``
    at collection, which reads as a broken install rather than as a deliberate
    scope decision. The fix is one of three, in order of preference: guard the
    import with ``pytest.importorskip`` (keeps the file's other tests), deny
    the test file (when every test in it depends on the absent subject), or
    ship the module.
    """
    repo_root, shipped_list = _shipped_paths()
    shipped = set(shipped_list)

    scanned = 0
    offenders: dict[str, list[str]] = {}
    observed_waivers: set[tuple[str, str]] = set()

    for rel in shipped_list:
        if not (rel.startswith("tests/") and rel.endswith(".py")):
            continue
        scanned += 1
        try:
            source = (repo_root / rel).read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - a shipped path that cannot be read
            continue
        for dotted in _unshipped_imports(source, shipped):
            if (rel, dotted) in _WAIVED_IMPORTS:
                observed_waivers.add((rel, dotted))
            else:
                offenders.setdefault(rel, []).append(dotted)

    # Anti-vacuity: an empty scan satisfies every assertion below, and is what a
    # mistyped allowance or a shipped-set computation that stopped returning
    # ``tests/`` would produce.
    assert scanned > 500, (
        f"only {scanned} shipped test modules were scanned -- the export ships "
        "thousands, so the shipped set is wrong and this test is not checking "
        "anything"
    )

    assert observed_waivers == set(_WAIVED_IMPORTS), (
        "stale waiver in _WAIVED_IMPORTS: "
        f"{sorted(set(_WAIVED_IMPORTS) - observed_waivers)} was not observed. "
        "The import it excuses is gone, so the waiver must go with it -- a "
        "waiver kept past its subject silently excuses the next import too."
    )

    assert not offenders, "shipped tests import modules that do not ship:\n" + "\n".join(
        f"  {path}: {', '.join(mods)}" for path, mods in sorted(offenders.items())
    )


# -- planted violations (non-negotiable 15) ---------------------------------
#
# One plant per *shape*, not one per rule: ``import x`` and ``from x import y``
# are different AST nodes, and each arrives both at the top of a file and inside
# a function. A detector watched to fail on only the top-of-file form is exactly
# the blindness this file's own header describes.

_PLANT_SHIPPED = {
    "tests/unit/test_thing.py",
    "tests/utils/helper.py",
    "tests/utils/__init__.py",
    "scripts/ci/present.py",
}

_MUST_FLAG = {
    "top-level Import": "import scripts.sim2rank\n",
    "top-level ImportFrom": "from scripts.sim2rank.scoring import compute_adr\n",
    "function-local Import": (
        "def test_x():\n    import scripts.sim2rank\n    assert scripts.sim2rank\n"
    ),
    "function-local ImportFrom": (
        "def test_x():\n    from scripts.sim2rank.scoring import compute_adr\n"
        "    assert compute_adr\n"
    ),
    "class-body ImportFrom": (
        "class TestX:\n    def test_y(self):\n"
        "        from scripts.sim2rank.degradation import apply\n        assert apply\n"
    ),
}

_MUST_NOT_FLAG = {
    "guarded at module level": (
        "import pytest\n\npytest.importorskip('scripts.sim2rank.scoring')\n"
        "from scripts.sim2rank.scoring import compute_adr\n"
    ),
    "guarded by an ancestor": (
        "import pytest\n\npytest.importorskip('scripts.sim2rank')\n"
        "def test_x():\n    from scripts.sim2rank.scoring import compute_adr\n"
        "    assert compute_adr\n"
    ),
    "relative import": "from ..utils.helper import thing\n",
    "shipped in-tree module": "from scripts.ci.present import thing\n",
    "third-party module": "import torch\nfrom numpy import ndarray\n",
    "guard named in a comment only": (
        # Prose must NOT satisfy the guard: this file has no real call.
        "# pytest.importorskip('scripts.sim2rank.scoring') would guard this\n"
        "import scripts.ci.present\n"
    ),
}


@pytest.mark.parametrize("shape", sorted(_MUST_FLAG))
def test_detector_flags_each_planted_import_shape(shape: str) -> None:
    """Every violation shape turns the detector red -- watched, not assumed."""
    found = _unshipped_imports(_MUST_FLAG[shape], _PLANT_SHIPPED)
    assert found, f"planted {shape} was not detected"
    assert all(f.startswith("scripts.sim2rank") for f in found), found


@pytest.mark.parametrize("shape", sorted(_MUST_NOT_FLAG))
def test_detector_stays_silent_on_each_legitimate_shape(shape: str) -> None:
    """A detector that flags these is a detector nobody can leave switched on."""
    assert _unshipped_imports(_MUST_NOT_FLAG[shape], _PLANT_SHIPPED) == [], shape


def test_the_prose_only_guard_plant_would_flag_without_its_comment() -> None:
    """Pins that the prose-only case is silent for the right reason.

    ``scripts.ci.present`` ships, so the comment plant above is silent whether
    or not comments count as guards -- which would make it a plant that proves
    nothing. Swapping in an unshipped module shows the comment does not guard.
    """
    prose = _MUST_NOT_FLAG["guard named in a comment only"].replace(
        "import scripts.ci.present", "import scripts.sim2rank.scoring"
    )
    assert _unshipped_imports(prose, _PLANT_SHIPPED) == ["scripts.sim2rank.scoring"]


# -- a shipped test may not construct a path to a DENIED file ----------------
#
# The import census above scans imports. It is blind to a test that names its
# subject as a **path**, and blind again when that subject is another *test*
# file rather than a ``src/`` module or a corpus root -- which is how
# ``tests/unit/architecture/test_required_lane_composition_plants.py`` shipped
# while ``tests/architecture/test_required_lane_composition.py``, the detector
# it plants violations against, was explicitly denied. Result in the export: 15
# collected tests, 15 ``FileNotFoundError``.
#
# A detector and its plant harness are one unit, and there was no gate that
# said so. This is that gate, scoped deliberately narrowly: it asks only about
# paths the allowlist **explicitly denies** (``!path``), never about paths that
# merely do not ship. A denial is a decision someone wrote down, so a shipped
# file reaching for one is always a mistake; the much larger "names a path that
# is simply absent" population is issue #1589's backlog and would swamp this.


def _denied_file_paths() -> list[str]:
    """Concrete file paths the real allowlist denies with ``!``.

    Directory prefixes and globs are excluded: this gate compares against a
    file's own name, and ``!experiments/`` names no file.
    """
    denied: list[str] = []
    for line in REAL_ALLOWLIST.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#")[0].strip()
        if not stripped.startswith("!"):
            continue
        pattern = stripped[1:].strip()
        if pattern and not pattern.endswith("/") and "*" not in pattern:
            denied.append(pattern)
    return denied


def _denied_paths_constructed(source: str, denied: list[str]) -> list[str]:
    """Denied paths this source builds a filesystem path to.

    Two decisions, both load-bearing:

    **Only string literals inside a path expression count.** A ``/`` ``BinOp``
    or a ``Path``/``joinpath``/``open`` call -- not any occurrence of the name.
    ``tests/unit/test_no_cluster_paths.py`` carries a waiver table whose *keys*
    are three denied paths; those entries are correct (the files exist in the
    private tree) and inert in the export, so a mention-based check would report
    a defect that is not one. Measured over 2,593 tracked test modules and 208
    denials: mention-based finds 2, path-construction finds 1 -- the real one.

    **Equality, never containment.** ``test_required_lane_composition_plants``
    opens with a docstring naming its subject's full path in prose; a substring
    test flags that docstring rather than line 37's actual ``Path`` expression,
    and would keep flagging a file whose defect had been fixed. Equality against
    the basename or the full path is prose-immune, per the same rule the
    ``importorskip`` scan states above.
    """
    tree = ast.parse(source)

    def string_constants(node: ast.AST) -> set[str]:
        return {
            n.value
            for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }

    # A `/` chain rooted at a pytest temp fixture builds a file the test CREATES,
    # in a directory that does not exist yet. It is definitionally not a read of a
    # repository file, so it cannot be an export-boundary defect -- and because
    # the match below also accepts a BASENAME, it is otherwise indistinguishable
    # from one: `tmp_path / "workflow_backlog.md"` scored identically to
    # `REPO_ROOT / "docs" / "reference" / "workflow_backlog.md"`. That false
    # positive was live and named test_check_docs_navigation.py, whose line is a
    # synthetic fixture page. A gate that reports a defect which is not one gets
    # switched off, so precision here is worth as much as recall.
    synthetic_roots = {"tmp_path", "tmp_path_factory", "tmpdir", "tmpdir_factory"}

    def chain_root(node: ast.AST) -> ast.AST:
        while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            node = node.left
        return node

    literals: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            root = chain_root(node)
            if isinstance(root, ast.Name) and root.id in synthetic_roots:
                continue
            literals |= string_constants(node)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in {"Path", "PurePath", "joinpath", "open", "read_text"}:
                for arg in node.args:
                    literals |= string_constants(arg)

    found = {path for path in denied if path in literals or PurePosixPath(path).name in literals}
    return sorted(found)


def test_no_shipped_test_constructs_a_path_to_a_denied_file() -> None:
    """A shipped test may not read a file the allowlist explicitly denies.

    The failure is not informative when it happens: ``FileNotFoundError`` at
    call time reads as a broken checkout, not as a scope decision. Fix by
    denying the reader in the same allowlist entry that denies its subject --
    a detector and the harness that plants violations against it are one unit
    and cannot be split across the export boundary.
    """
    repo_root, shipped_list = _shipped_paths()
    denied = _denied_file_paths()

    scanned = 0
    offenders: dict[str, list[str]] = {}
    for rel in shipped_list:
        if not (rel.startswith("tests/") and rel.endswith(".py")):
            continue
        scanned += 1
        try:
            source = (repo_root / rel).read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - a shipped path that cannot be read
            continue
        try:
            hits = _denied_paths_constructed(source, denied)
        except SyntaxError:  # pragma: no cover - not this gate's question
            continue
        if hits:
            offenders[rel] = hits

    # Anti-vacuity, both sides. An empty scan satisfies the assertion, and so
    # does an empty denial list -- and the denial list is read from a file, so
    # a rename of the allowlist would silently produce one.
    assert scanned > 500, (
        f"only {scanned} shipped test modules were scanned -- the shipped set "
        "is wrong and this gate is not checking anything"
    )
    assert len(denied) > 50, (
        f"only {len(denied)} concrete file denials were parsed from "
        f"{REAL_ALLOWLIST} -- the allowlist moved or its syntax changed, and "
        "this gate would pass by having nothing to compare against"
    )

    assert not offenders, (
        "shipped test(s) build a path to a file the allowlist denies:\n"
        + "\n".join(f"  {path}: {', '.join(hits)}" for path, hits in sorted(offenders.items()))
        + "\nDeny the reader beside its subject, or ship the subject."
    )


_PLANT_DENIED = ["tests/architecture/test_required_lane_composition.py"]

_MUST_FLAG_PATH = {
    "segment chain off a root": (
        "ROOT = Path(__file__).resolve().parents[3]\n"
        'D = ROOT / "tests" / "architecture" / "test_required_lane_composition.py"\n'
    ),
    "full path in one literal": (
        'D = Path("tests/architecture/test_required_lane_composition.py")\n'
    ),
    "joinpath call": (
        'ROOT = Path(".")\n'
        'D = ROOT.joinpath("tests", "architecture", '
        '"test_required_lane_composition.py")\n'
    ),
    "function-local construction": (
        "def test_x():\n"
        '    d = Path(".") / "tests" / "architecture" / '
        '"test_required_lane_composition.py"\n'
        "    assert d.exists()\n"
    ),
    "read_text on the literal": (
        'def test_x():\n    open("tests/architecture/test_required_lane_composition.py").read()\n'
    ),
}

_MUST_NOT_FLAG_PATH = {
    # The real false positive this detector was tuned against.
    "a data-table key, not a path": (
        '_WAIVERS = {\n    "tests/architecture/test_required_lane_composition.py": (3, "why"),\n}\n'
    ),
    "named in a docstring": (
        '"""``tests/architecture/test_required_lane_composition.py`` asserts '
        'that the lane runs."""\n'
    ),
    "named in a comment": (
        "# see tests/architecture/test_required_lane_composition.py\n"
        'D = Path(".") / "tests" / "conftest.py"\n'
    ),
    "a path to a file that is not denied": (
        'D = Path(".") / "tests" / "architecture" / "test_structure_guard.py"\n'
    ),
    "the basename as a bare comparison": (
        'if name == "test_required_lane_composition":\n    pass\n'
    ),
}


@pytest.mark.parametrize("shape", sorted(_MUST_FLAG_PATH))
def test_denied_path_detector_flags_each_planted_shape(shape: str) -> None:
    """Every construction shape turns it red -- watched, not assumed."""
    assert _denied_paths_constructed(_MUST_FLAG_PATH[shape], _PLANT_DENIED) == _PLANT_DENIED, (
        f"planted {shape} was not detected"
    )


@pytest.mark.parametrize("shape", sorted(_MUST_NOT_FLAG_PATH))
def test_denied_path_detector_stays_silent_on_each_legitimate_shape(shape: str) -> None:
    """Flagging these would make the gate one nobody can leave switched on."""
    assert _denied_paths_constructed(_MUST_NOT_FLAG_PATH[shape], _PLANT_DENIED) == [], shape


def test_the_not_denied_plant_would_flag_if_that_path_were_denied() -> None:
    """Pins that "a path to a file that is not denied" is silent for the right reason.

    Without this, that plant passes whether the detector consults the denial
    list or is simply broken -- a negative plant that proves nothing. Denying
    the path it names must make the same source flag.
    """
    source = _MUST_NOT_FLAG_PATH["a path to a file that is not denied"]
    other = ["tests/architecture/test_structure_guard.py"]
    assert _denied_paths_constructed(source, other) == other


# ---------------------------------------------------------------------------
# A numbered issue link is a promise the destination repo cannot keep
# ---------------------------------------------------------------------------
#
# Workstream F pushes this tree as a SINGLE INITIAL COMMIT into a repository
# with no history, no issues and no pull requests. Every `#N` in the shipped
# text was minted by the private repo's counter, and none of those numbers mean
# anything on the other side. There are two failure shapes and the second is the
# dangerous one:
#
#   * a link into a repository of this owner that is not published at all 404s
#     for every reader -- visibly, unhelpfully wrong;
#   * a link to the repo that IS published (`.../spectramr/issues/1497`) 404s
#     only until the new repo's counter reaches 1497, at which point it starts
#     resolving to an unrelated issue. That one never announces itself.
#
# Third-party numbered links are legitimate and must keep working: the tree
# cites `github.com/quiqi/relu_kan` as upstream provenance for a model. So the
# rule is scoped to THIS project's owner, and the owner is read from
# `[project.urls] Repository` rather than typed here -- a hardcoded handle is a
# second owner of the project's identity, and it would go quietly stale on the
# next rename (`tests/architecture/test_no_stale_package_name.py` is the first
# owner, and it deliberately ALLOWS `github.com/<owner>/<old-name>` as a
# historical repository URL -- a correct waiver for the question IT asks, and
# the reason nothing in the tree was asking this one).


def _project_owner() -> str:
    """The GitHub owner this project publishes under, from ``[project.urls]``."""
    import tomllib

    repo_root = SCRIPT.resolve().parents[2]
    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    url = data["project"]["urls"]["Repository"]
    owner = PurePosixPath(url.split("github.com/", 1)[1]).parts[0]
    assert owner, f"no owner derivable from Repository={url!r}"
    return owner


def _numbered_link_pattern(owner: str) -> re.Pattern[str]:
    """Links to a numbered issue/PR under ``owner``, in ANY repo of theirs.

    Any repo, not just the published one, because the old name is exactly the
    case that produced the two live hits this gate was written for.
    """
    return re.compile(
        rf"github\.com/{re.escape(owner)}/[A-Za-z0-9._-]+/(?:issues|pull)/\d+",
        re.IGNORECASE,
    )


#: Strings the detector MUST flag, and the shape each one stands for.
_MUST_FLAG_LINK = {
    "an issue in the published repo": "see https://github.com/{owner}/spectramr/issues/1497 for why",
    # A repo OTHER than the published one -- deliberately not spelled with the
    # pre-rename package name, which `tests/architecture/test_no_stale_package_name.py`
    # owns and would (correctly) report as a stale-name hit in this file.
    "an issue in another repo of the same owner": (
        '"issue": "https://github.com/{owner}/some-other-repo/issues/1585"'
    ),
    "a pull request": "found_by: https://github.com/{owner}/some-other-repo/pull/1584",
    "an rst hyperlink target": "(`#1497 <https://github.com/{owner}/spectramr/issues/1497>`_)",
    "a differently-cased owner": "https://github.com/{lower}/spectramr/issues/12",
}

#: Strings the detector MUST NOT flag. Each is a real shape in the shipped tree.
_MUST_NOT_FLAG_LINK = {
    "a third-party issue, cited as upstream provenance": (
        "adapted from https://github.com/quiqi/relu_kan/issues/3"
    ),
    "the unnumbered issue tracker from [project.urls]": (
        'Issues = "https://github.com/{owner}/spectramr/issues"'
    ),
    "a repository url with no path": 'Repository = "https://github.com/{owner}/spectramr"',
    "a blob link, which survives a fresh history": (
        "https://github.com/{owner}/spectramr/blob/main/CHANGELOG.md"
    ),
    "a bare issue reference with no url": "the num_timesteps fold (#980) explains this",
}


@pytest.mark.parametrize("shape", sorted(_MUST_FLAG_LINK))
def test_the_numbered_link_detector_fires(shape: str) -> None:
    owner = _project_owner()
    text = _MUST_FLAG_LINK[shape].format(owner=owner, lower=owner.lower())
    assert _numbered_link_pattern(owner).search(text), f"missed: {shape}"


@pytest.mark.parametrize("shape", sorted(_MUST_NOT_FLAG_LINK))
def test_the_numbered_link_detector_stays_silent(shape: str) -> None:
    owner = _project_owner()
    text = _MUST_NOT_FLAG_LINK[shape].format(owner=owner)
    assert not _numbered_link_pattern(owner).search(text), f"false positive: {shape}"


#: Shipped files allowed to carry such a link, and the reason for each. Keyed by
#: PATH, so an entry names a decision about a file rather than a pattern that
#: would quietly waive the next file to grow one.
#:
#: EMPTY, and that is a finished state rather than a missing entry. The one waiver
#: this held was `tests/unit/_known_sim2rank_docstring_counts.json`, whose `issue`
#: and `found_by` keys linked the private tracker. Its reason argued that
#: rewriting the repository NAME turns a working link into a 404 or, worse, into
#: a plausible link to an unrelated issue -- which is true, and is an argument
#: against rewriting the owner, not against the fix this module's own failure
#: message prescribes. Stating the reference as text (`internal issue #1585`)
#: keeps the number, cannot resolve to anything at all, and is what the shipped
#: `docs/index.rst` already tells a reader to expect: naming an internal source is
#: fine, pointing at one is not. Neither key had a consumer --
#: `tests/unit/test_sim2rank_docstring_counts.py` reads only `waived`.
_LINK_WAIVERS: dict[str, str] = {}


def test_every_link_waiver_still_waives_something() -> None:
    """A waiver whose file stopped carrying a link is dead and must be deleted.

    Same ratchet direction as the exporter's own dead-denial check: an entry
    that waives nothing reads as a standing decision while holding nothing
    back, and the next link to land in that file ships unremarked.
    """
    repo_root, shipped = _shipped_paths()
    pattern = _numbered_link_pattern(_project_owner())
    for rel, reason in sorted(_LINK_WAIVERS.items()):
        assert rel in shipped, f"waived path does not ship: {rel}"
        assert reason.strip(), f"waiver for {rel} states no reason"
        text = (repo_root / rel).read_text(encoding="utf-8")
        assert pattern.search(text), (
            f"{rel} no longer carries a numbered link -- delete its waiver "
            "rather than leaving an allowance that allows nothing."
        )


def test_a_waiver_is_keyed_on_the_path_not_the_link() -> None:
    """The same link in a DIFFERENT file is still a finding.

    Planted because a path-keyed waiver is one careless edit away from a
    substring test, and that edit is invisible: everything stays green.

    Driven from a SYNTHETIC mapping rather than from `_LINK_WAIVERS`, which is
    empty above. A plant that reads the live mapping stops holding at exactly the
    moment the last waiver is retired -- and this one did not even degrade
    quietly: `next(iter(...))` raises `StopIteration` on an empty dict, so
    retiring the last waiver turned this plant into an ERROR rather than into a
    vacuous pass. The sibling owner already settled the shape --
    `_build_allowed_path` in `tests/architecture/test_no_stale_package_name.py` is
    "pure and total, so the tests below drive it with synthetic sets rather than
    with whatever the surrounding tree happens to contain" -- so this follows it.
    """
    waived = "some/waived/file.json"
    waivers = {waived: "a reason"}

    assert waived not in "some/other/file.py", "plant is malformed"
    for other in ("docs/contributing/ci.rst", "README.md", waived + ".bak"):
        assert other not in waivers, f"{other} must not be waived"

    # And the live mapping obeys the same convention. Vacuous while it is empty;
    # this is what catches the first entry keyed on a link fragment instead.
    for rel in _LINK_WAIVERS:
        assert "/" in rel and not rel.startswith("http"), (
            f"waiver key {rel!r} is not a repository-relative path"
        )


def test_this_modules_own_plants_do_not_hardcode_the_owner() -> None:
    """This file must not become its own finding, and not by being exempted.

    The plants above are the only place in the tree that deliberately spells the
    offending shape, so the discipline that keeps them harmless -- templating the
    owner rather than typing it -- has to be pinned somewhere. Typing a literal
    owner into any plant turns this red, which is the correct signal: the fix is
    to template it, not to add an exemption.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    hits = _numbered_link_pattern(_project_owner()).findall(source)
    assert not hits, (
        f"{len(hits)} plant string(s) in this module spell the real owner. "
        "Template it as {owner} so the file is not its own finding."
    )


def test_no_shipped_file_links_to_a_numbered_issue_in_this_project() -> None:
    """No shipped text may link to an issue/PR number the new repo will not have."""
    repo_root, shipped = _shipped_paths()
    owner = _project_owner()
    pattern = _numbered_link_pattern(owner)

    # This module is NOT exempt from its own rule, and does not need to be: the
    # plant strings above template the owner as `{owner}`, so the text on disk
    # reads `github.com/{owner}/...` and matches nothing. A self-exemption was
    # written first and removed -- planted by deleting it, the run stayed green,
    # which is the proof that it was dead code. `test_this_modules_own_plants_
    # do_not_hardcode_the_owner` is what actually holds that property.
    offenders: list[str] = []
    scanned = 0
    for rel in shipped:
        if rel in _LINK_WAIVERS:
            continue
        try:
            text = (repo_root / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; no links to find
        scanned += 1
        for match in pattern.finditer(text):
            offenders.append(f"{rel}: {match.group(0)}")

    assert scanned > 1000, (
        f"only {scanned} shipped files were read -- the corpus is broken, and "
        "this guard would otherwise pass by having nothing to scan"
    )
    assert not offenders, (
        f"{len(offenders)} shipped file(s) link to a numbered issue/PR under "
        f"{owner!r}:\n"
        + "\n".join(f"  {o}" for o in sorted(offenders))
        + "\n\nThe public repository starts from a fresh single-commit history "
        "and mints its own numbers. Such a link either 404s or, worse, silently "
        "resolves to an unrelated issue once the counter passes it. State the "
        "reference as text (e.g. 'internal issue #1497') instead of linking it."
    )


# ---------------------------------------------------------------------------
# The overlay (non-negotiable 15)
#
# The overlay decides CONTENT; the allowlist decides MEMBERSHIP. Keeping those two
# separate is the whole safety property, so each way of blurring them is planted.
#
# One plant is not hypothetical. The first version of the overlay shipped its own
# README at ``public_overlay/README.md``, which replaced the PROJECT README -- the
# published front page read "# Public overlay". The dead-overlay ratchet cannot
# catch that, because ``README.md`` IS a shipped path: the substitution was valid,
# just unintended. Only the run summary reporting two overlaid files when one was
# placed gave it away, which is why the summary now prints every path.
# ---------------------------------------------------------------------------

OVERLAY_DIR = "public_overlay"


def _stage_overlay(repo: Path, files: dict[str, str], root: str = OVERLAY_DIR) -> None:
    """Write overlay files into the fixture repo and stage them.

    Staged rather than committed: ``_export`` commits everything staged when it
    commits the allowlist, so the overlay lands in the same pinned SHA the export
    reads. Committing here as well would work but would hide that coupling.
    """
    for rel, text in files.items():
        dest = repo / root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    _git("add", root, cwd=repo)


def test_an_overlay_replaces_a_shipped_files_content(repo: Path, tmp_path: Path) -> None:
    _stage_overlay(repo, {"src/pkg/mod.py": "VALUE = 99\n"})
    out = tmp_path / "out"
    proc = _export(
        repo, _allowlist(tmp_path, "src/", OVERLAY_DIR + "/"), out, "--overlay", OVERLAY_DIR
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "src/pkg/mod.py").read_text() == "VALUE = 99\n"


def test_an_overlay_never_adds_a_path(repo: Path, tmp_path: Path) -> None:
    """The direction that keeps the allowlist the single answer to 'what ships'.

    If the overlay could add, it would be a second publication route and the
    fail-closed allowlist would stop being the only one.
    """
    _stage_overlay(repo, {"secrets/token.txt": "LEAKED\n"})
    out = tmp_path / "out"
    proc = _export(
        repo, _allowlist(tmp_path, "src/", OVERLAY_DIR + "/"), out, "--overlay", OVERLAY_DIR
    )
    assert not (out / "secrets" / "token.txt").exists()
    assert "DEAD OVERLAY" in proc.stdout


def test_an_overlay_whose_target_does_not_ship_is_dead_and_fails_under_strict(
    repo: Path, tmp_path: Path
) -> None:
    _stage_overlay(repo, {"secrets/token.txt": "x\n"})
    allow = _allowlist(tmp_path, "src/", OVERLAY_DIR + "/")

    lenient = _export(repo, allow, tmp_path / "a", "--overlay", OVERLAY_DIR)
    assert lenient.returncode == 0
    assert "DEAD OVERLAY" in lenient.stdout

    strict = _export(repo, allow, tmp_path / "b", "--overlay", OVERLAY_DIR, "--strict")
    assert strict.returncode == 2, strict.stdout


def test_the_manifest_hashes_what_is_on_disk_not_the_git_blob(repo: Path, tmp_path: Path) -> None:
    """The manifest is a provenance claim about the EXPORT, not about the commit.

    Hashing the pre-overlay blob would describe a file the export does not contain.
    """
    replaced = "VALUE = 99\n"
    _stage_overlay(repo, {"src/pkg/mod.py": replaced})
    out = tmp_path / "out"
    _export(repo, _allowlist(tmp_path, "src/", OVERLAY_DIR + "/"), out, "--overlay", OVERLAY_DIR)

    entry = next(f for f in _manifest(out)["files"] if f["path"] == "src/pkg/mod.py")
    assert entry["sha256"] == hashlib.sha256(replaced.encode()).hexdigest()
    assert entry["size"] == len(replaced)


def test_the_manifest_records_the_overlay_as_a_stated_decision(repo: Path, tmp_path: Path) -> None:
    _stage_overlay(repo, {"src/pkg/mod.py": "VALUE = 99\n"})
    out = tmp_path / "out"
    _export(repo, _allowlist(tmp_path, "src/", OVERLAY_DIR + "/"), out, "--overlay", OVERLAY_DIR)

    manifest = _manifest(out)
    assert manifest["overlaid_paths"] == ["src/pkg/mod.py"]
    assert manifest["dead_overlay_files"] == []
    assert manifest["overlay_root"] == OVERLAY_DIR


def test_a_root_level_overlay_file_replaces_a_root_file_and_is_named_in_the_summary(
    repo: Path, tmp_path: Path
) -> None:
    """The README collision, planted.

    A file at the overlay ROOT maps to a file at the export root. That is the
    mechanism working as designed, and it is also how the overlay's own README
    silently became the project's. The ratchet cannot flag it -- the target ships,
    so the substitution is valid -- so the guard is that the summary NAMES it.
    """
    _stage_overlay(repo, {"README.md": "# Public overlay\n"})
    out = tmp_path / "out"
    proc = _export(
        repo, _allowlist(tmp_path, "README.md", OVERLAY_DIR + "/"), out, "--overlay", OVERLAY_DIR
    )

    assert (out / "README.md").read_text() == "# Public overlay\n"
    assert "DEAD OVERLAY" not in proc.stdout, "a valid substitution, which is the trap"
    assert "README.md" in proc.stdout, "an unintended substitution must at least be visible"


def test_an_overlay_source_never_ships_at_its_storage_path(repo: Path, tmp_path: Path) -> None:
    """The overlay tree is content, not membership -- so it must not also ship as itself.

    The real allowlist carries a bare ``scripts/release/`` allowance, which matched the
    overlay's own storage paths and published them verbatim ALONGSIDE their mapped
    copies: the same bytes at two paths, and the visible one is the copy nothing reads.

    The differential is what pins it, rather than a hardcoded count that would only
    describe this fixture: with the overlay ACTIVE the storage path leaves ``shipped``
    and does NOT arrive in ``excluded``, because it is neither -- it is content that
    ships elsewhere.
    """
    _stage_overlay(repo, {"src/pkg/mod.py": "VALUE = 99\n"})
    allow = _allowlist(tmp_path, "src/", OVERLAY_DIR + "/")

    # Overlay inactive: the storage tree is ordinary allowlisted content and ships.
    off = _export(repo, allow, tmp_path / "off", "--overlay", "no_such_dir")
    assert off.returncode == 0, off.stderr
    m_off = _manifest(tmp_path / "off")
    assert (tmp_path / "off" / OVERLAY_DIR / "src/pkg/mod.py").exists()

    # Overlay active: the same path is skipped before the allowlist is consulted.
    on = _export(repo, allow, tmp_path / "on", "--overlay", OVERLAY_DIR)
    assert on.returncode == 0, on.stderr
    m_on = _manifest(tmp_path / "on")

    assert not (tmp_path / "on" / OVERLAY_DIR).exists(), "the overlay shipped at its storage path"
    assert (tmp_path / "on" / "src/pkg/mod.py").read_text() == "VALUE = 99\n"
    assert f"{OVERLAY_DIR}/src/pkg/mod.py" in m_on["overlay_source_paths"]
    assert not any(r["path"].startswith(OVERLAY_DIR + "/") for r in m_on["files"])

    assert m_on["file_count"] == m_off["file_count"] - 1, "one fewer file shipped"
    assert m_on["excluded_count"] == m_off["excluded_count"], (
        "the storage path was counted as excluded -- it is neither shipped nor "
        "excluded, and counting it as excluded would misreport what the export dropped"
    )


def test_an_allowance_naming_only_the_overlay_root_is_dead(repo: Path, tmp_path: Path) -> None:
    """The new semantics, made checkable rather than left as prose.

    "The allowlist never inspects the overlay" used to be a comment. Now that overlay
    storage is skipped BEFORE the allowlist is consulted, an allowance that names only
    the overlay root cancels nothing and is reported dead -- which is the correct
    guidance, because such a line is a claim the exporter no longer honours. The real
    allowlist is unaffected: its ``scripts/release/`` allowance stays live on the four
    other files it ships.
    """
    _stage_overlay(repo, {"src/pkg/mod.py": "VALUE = 99\n"})
    allow = _allowlist(tmp_path, "src/", OVERLAY_DIR + "/")

    lenient = _export(repo, allow, tmp_path / "a", "--overlay", OVERLAY_DIR)
    assert lenient.returncode == 0
    assert "DEAD PATTERNS" in lenient.stdout
    assert OVERLAY_DIR + "/" in lenient.stdout

    strict = _export(repo, allow, tmp_path / "b", "--overlay", OVERLAY_DIR, "--strict")
    assert strict.returncode == 2, strict.stdout


def test_an_overlay_identical_to_its_target_is_redundant_and_fails_under_strict(
    repo: Path, tmp_path: Path
) -> None:
    """The blindness the dead-overlay ratchet has: a VALID substitution that changes nothing.

    Dead-overlay fires when the target does not ship. Here the target ships and the
    substitution succeeds -- it just writes the bytes that were already there. Nothing
    looks wrong while the two copies agree, and the first edit to the tracked file is
    then silently discarded in the export while the published copy freezes. Found in the
    live configuration, not hypothesised: ``public_overlay/docs/index.rst`` was
    byte-identical to ``docs/index.rst`` and no gate said so.
    """
    tracked = (repo / "src" / "pkg" / "mod.py").read_text()
    _stage_overlay(repo, {"src/pkg/mod.py": tracked})
    allow = _allowlist(tmp_path, "src/")

    lenient = _export(repo, allow, tmp_path / "a", "--overlay", OVERLAY_DIR)
    assert lenient.returncode == 0, lenient.stderr
    assert "REDUNDANT OVERLAY" in lenient.stdout
    assert _manifest(tmp_path / "a")["redundant_overlay_files"] == ["src/pkg/mod.py"]

    strict = _export(repo, allow, tmp_path / "b", "--overlay", OVERLAY_DIR, "--strict")
    assert strict.returncode == 2, strict.stdout


def test_an_overlay_that_differs_is_not_reported_as_redundant(repo: Path, tmp_path: Path) -> None:
    """Negative control: the ratchet must not fire on the case the overlay exists for."""
    _stage_overlay(repo, {"src/pkg/mod.py": "VALUE = 99\n"})
    out = tmp_path / "out"
    proc = _export(repo, _allowlist(tmp_path, "src/"), out, "--overlay", OVERLAY_DIR, "--strict")
    assert proc.returncode == 0, proc.stdout
    assert "REDUNDANT OVERLAY" not in proc.stdout
    assert _manifest(out)["redundant_overlay_files"] == []


def test_an_absent_overlay_directory_is_not_an_error(repo: Path, tmp_path: Path) -> None:
    """An export with no overlay is the normal case, not a degraded one."""
    out = tmp_path / "out"
    proc = _export(repo, _allowlist(tmp_path, "src/"), out, "--overlay", "no_such_dir", "--strict")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _manifest(out)["overlaid_paths"] == []


def test_the_shipped_overlay_only_replaces_paths_the_shipped_allowlist_ships() -> None:
    """The real overlay, against the real allowlist -- a negative control.

    Bounds the live configuration rather than a fixture: every file under
    ``scripts/release/public_overlay/`` must name a path the real allowlist ships,
    or the next release export exits 2.
    """
    root = Path(__file__).resolve().parents[3]
    overlay_root = root / "scripts" / "release" / "public_overlay"
    if not overlay_root.is_dir():
        pytest.skip("no overlay in this tree")

    targets = sorted(
        p.relative_to(overlay_root).as_posix() for p in overlay_root.rglob("*") if p.is_file()
    )
    assert targets, "an empty overlay directory should be deleted, not kept"

    # SHIPS, not exists. The overlay is replace-only: read_overlay writes a file
    # only where the allowlist already selects one, and reports every other entry
    # as a DEAD OVERLAY (exit 2). Existence and membership come apart in exactly
    # the case that matters -- a path that is present in this tree and DENIED --
    # so a `.exists()` check is green on the one configuration this guard exists
    # to reject. It was written as `.exists()` while its own docstring said
    # "ships"; the two agreed only because every overlay file happened to be
    # allowed. Ask the allowlist, through the exporter's own parser, so this
    # guard and the release path cannot disagree (non-negotiable 17).
    import importlib.util

    spec = importlib.util.spec_from_file_location("_export_public_tree", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    allows, denies = mod.parse_allowlist(
        REAL_ALLOWLIST.read_text(encoding="utf-8"), str(REAL_ALLOWLIST)
    )

    for rel in targets:
        assert (root / rel).exists(), (
            f"overlay file {rel} names a path that does not exist in this tree; "
            "it would be reported as a DEAD OVERLAY at release time"
        )
        allowed = any(mod.matches(rel, a) for a in allows)
        denied = any(mod.matches(rel, d) for d in denies)
        assert allowed and not denied, (
            f"overlay file {rel} names a path the allowlist does not ship "
            f"(allowed={allowed}, denied={denied}). The overlay never widens "
            f"membership -- it only replaces CONTENT for a path already shipping -- "
            f"so this entry is written nowhere and the export exits 2 with "
            f"DEAD OVERLAY. Add the allowance (or drop the denial) in the same "
            f"commit as the overlay file."
        )


def test_the_overlay_f821_baseline_is_derived_from_the_real_one() -> None:
    """The overlay baseline must be its source minus exactly the unshipped rows.

    A full-file overlay is a copy, and a copy is a second owner that nothing diffs
    (non-negotiable 17). For a *derived* file the drift is checkable, so it is
    checked here rather than trusted: the overlay must equal the real baseline with
    every row whose path does not ship removed, and nothing else changed.

    The row set cannot simply be dropped from the real baseline instead. Those names
    still exist on this branch, so removing their rows would un-exempt them and
    redden this tree's own gate -- and the ratchet may only ever move DOWN
    (non-negotiable 20). Two file sets, two baselines, one derivation.
    """
    import importlib.util

    repo_root = SCRIPT.resolve().parents[2]
    overlay = repo_root / "scripts/release/public_overlay/scripts/ci/baselines/f821.txt"
    source = repo_root / "scripts/ci/baselines/f821.txt"
    if not overlay.exists():
        pytest.skip("no f821 overlay in this tree")

    spec = importlib.util.spec_from_file_location("_export_public_tree", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    allows, denies = mod.parse_allowlist(
        REAL_ALLOWLIST.read_text(encoding="utf-8"), str(REAL_ALLOWLIST)
    )

    def ships(rel: str) -> bool:
        return any(mod.matches(rel, a) for a in allows) and not any(
            mod.matches(rel, d) for d in denies
        )

    expected = []
    dropped = []
    for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            expected.append(line)
            continue
        parts = stripped.split("\t")
        (expected if len(parts) < 2 or ships(parts[1]) else dropped).append(line)

    # Anti-vacuity: if nothing is dropped the overlay is a pure copy and should be
    # deleted, not maintained.
    assert dropped, (
        "every row of the real f821 baseline names a shipping path, so the overlay "
        "copy has no reason to exist -- delete it and let the real baseline ship"
    )
    assert overlay.read_text(encoding="utf-8") == "".join(expected), (
        "the f821 overlay has drifted from its source. Regenerate it as the real "
        "baseline minus the rows whose path does not ship."
    )


# ---------------------------------------------------------------------------
# A shipped pre-commit hook must have a shipped subject.
#
# `.pre-commit-config.yaml` ships, and the published pr-advisory lane runs
# `pre-commit run --all-files`, so every local hook it declares executes in the
# public repo. A hook whose `entry` names a path the allowlist drops fails there
# with "No such file or directory" -- a red X that says nothing about the PR,
# which is precisely what the overlay header of pr-advisory.yml exists to avoid.
#
# The two neighbouring detectors both miss this shape, which is why it is a
# separate one rather than a widened predicate. `_unshipped_imports` reads
# `import` statements, and a pre-commit entry is not an import; the denied-path
# scan asks whether a path is *denied*, and an entry pointing at a merely
# unlisted file is excluded without ever being denied. Detector blindness is
# multidimensional: covering one axis says nothing about the next.
# ---------------------------------------------------------------------------

_PRE_COMMIT_CONFIG = ".pre-commit-config.yaml"


def _entry_path_tokens(entry: str) -> list[str]:
    """Tokens of a hook ``entry`` that could name an in-tree path.

    ``entry`` is an argv prefix rather than a shell string, so a plain split is
    what pre-commit itself does. A token is a candidate when it carries a
    separator, which drops the interpreter (``python``) and bare flags without
    needing to enumerate either.
    """
    return [t for t in entry.split() if "/" in t and not t.startswith("-")]


def _unshipped_entry_paths(config_text: str, shipped: set[str], repo_root: Path) -> list[str]:
    """Entry paths that exist in the source tree but are absent from *shipped*.

    A token that names nothing in this tree is not this detector's business: it
    is a tool name that happens to contain a separator, or a defect for the hook
    to report. The question asked here is the narrower one the export can
    actually answer -- does the allowlist drop something a shipped hook needs?
    """
    offenders: list[str] = []
    for repo in yaml.safe_load(config_text).get("repos", []):
        if repo.get("repo") != "local":
            continue
        for hook in repo.get("hooks", []):
            for token in _entry_path_tokens(str(hook.get("entry", ""))):
                if not (repo_root / token).exists():
                    continue
                if token in shipped:
                    continue
                # A directory counts as shipped when any file under it ships.
                if any(s.startswith(token.rstrip("/") + "/") for s in shipped):
                    continue
                offenders.append(f"{hook.get('id', '?')}: {token}")
    return sorted(offenders)


def _synthetic_config(entry: str) -> str:
    """A minimal one-hook config, built here rather than read from disk.

    Planting against the real file would make the plant depend on the very
    content under test.
    """
    return (
        "repos:\n"
        "  - repo: local\n"
        "    hooks:\n"
        "      - id: planted\n"
        f"        entry: {entry}\n"
        "        language: system\n"
    )


def test_every_shipped_pre_commit_hook_entry_ships_its_subject() -> None:
    """The real assertion: no shipped hook points at a path the export drops."""
    repo_root, shipped_list = _shipped_paths()
    shipped = set(shipped_list)

    assert _PRE_COMMIT_CONFIG in shipped, (
        f"{_PRE_COMMIT_CONFIG} no longer ships, so no hook of it runs in the "
        "published repo and every assertion below is vacuous. Either restore the "
        "allowance or delete this detector -- do not leave it passing."
    )

    text = (repo_root / _PRE_COMMIT_CONFIG).read_text(encoding="utf-8")
    scanned = sum(
        len(_entry_path_tokens(str(hook.get("entry", ""))))
        for repo in yaml.safe_load(text).get("repos", [])
        if repo.get("repo") == "local"
        for hook in repo.get("hooks", [])
    )
    assert scanned >= 3, (
        f"only {scanned} entry path token(s) parsed out of {_PRE_COMMIT_CONFIG}; "
        "each local hook names a script, so the parse is wrong and a clean "
        "result here means nothing"
    )

    offenders = _unshipped_entry_paths(text, shipped, repo_root)
    assert not offenders, (
        "the exported .pre-commit-config.yaml declares local hooks whose entry "
        "paths the allowlist drops. In the published repo `pre-commit run "
        "--all-files` fails on each with 'No such file or directory'.\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


@pytest.mark.parametrize(
    "entry,expected",
    [
        (
            "python scripts/ci/check_no_stale_package_name.py",
            "scripts/ci/check_no_stale_package_name.py",
        ),
        (
            "python scripts/ci/check_witness_corpus.py experiments/inprogress",
            "experiments/inprogress",
        ),
    ],
)
def test_entry_detector_flags_a_dropped_path(entry: str, expected: str) -> None:
    """Planted red: with nothing shipped, every real entry path is an offence."""
    repo_root, _ = _shipped_paths()
    assert (repo_root / expected).exists(), f"plant is stale: {expected} is gone"
    found = _unshipped_entry_paths(_synthetic_config(entry), set(), repo_root)
    assert any(expected in f for f in found), f"detector missed {expected}: {found}"


def test_entry_detector_stays_silent_when_the_subject_ships() -> None:
    repo_root, _ = _shipped_paths()
    token = "scripts/ci/check_test_paired_with_source.py"
    assert _unshipped_entry_paths(_synthetic_config(f"python {token}"), {token}, repo_root) == []


def test_entry_detector_accepts_a_directory_whose_files_ship() -> None:
    """A directory token ships when anything under it does -- not as a literal."""
    repo_root, _ = _shipped_paths()
    under = "experiments/inprogress/reconstruction"
    assert (
        _unshipped_entry_paths(
            _synthetic_config("python scripts/ci/check_witness_corpus.py experiments/inprogress"),
            {"scripts/ci/check_witness_corpus.py", under + "/anything.yaml"},
            repo_root,
        )
        == []
    )


def test_entry_detector_ignores_a_token_that_names_nothing() -> None:
    """A tool name carrying a separator is not a missing file."""
    repo_root, _ = _shipped_paths()
    assert _unshipped_entry_paths(_synthetic_config("some/tool --fix"), set(), repo_root) == []


def test_entry_detector_ignores_hooks_from_remote_repos() -> None:
    """Only `repo: local` entries resolve against this tree."""
    repo_root, _ = _shipped_paths()
    remote = (
        "repos:\n"
        "  - repo: https://github.com/example/thing\n"
        "    rev: v1\n"
        "    hooks:\n"
        "      - id: whatever\n"
        "        entry: python scripts/ci/check_no_stale_package_name.py\n"
    )
    assert _unshipped_entry_paths(remote, set(), repo_root) == []


# ---------------------------------------------------------------------------
# A shipped test whose bare `conftest` import the allowlist drops.
#
# The sibling detector above catches a pre-commit hook naming a dropped script.
# This is the same export-only class one layer down, and it is the more damaging
# half: the allowlist deliberately tolerates ~46 shipped tests whose *subject* is
# denied, and each costs exactly one failure. A dropped conftest.py costs the
# whole directory -- pytest aborts collection for every file beside it. It also
# reports itself badly, because the bare name still resolves, to the ROOT
# conftest.py, so the error is "cannot import name X from conftest" and reads
# like a broken helper rather than a missing file.
#
# Observed 2026-09-04: tests/unit/release/conftest.py was denied when its only
# two importers were denied with it. test_bump_version.py then shipped, imported
# it, and took tests/unit/release/ from 102 passing to "Interrupted: 1 error
# during collection" -- invisible in the private checkout, where the file exists.
# ---------------------------------------------------------------------------


def _imports_bare_conftest(text: str) -> bool:
    """Does this source import the top-level ``conftest`` module by bare name?

    ``level == 0`` is load-bearing: ``ast.ImportFrom.module`` drops the leading
    dots, so ``from .conftest import x`` also presents as ``module == "conftest"``
    and would false-positive without it. ``ast.walk`` is deliberate rather than a
    top-level scan -- a function-local import fails just as hard, only later.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(
        (isinstance(n, ast.ImportFrom) and n.level == 0 and n.module == "conftest")
        or (isinstance(n, ast.Import) and any(a.name == "conftest" for a in n.names))
        for n in ast.walk(tree)
    )


def _conftest_importers(paths: list[str], repo_root: Path) -> list[str]:
    """Which of ``paths`` import a bare ``conftest``."""
    found = []
    for rel in paths:
        if not rel.endswith(".py"):
            continue
        text = (repo_root / rel).read_text(encoding="utf-8", errors="replace")
        if "conftest" in text and _imports_bare_conftest(text):
            found.append(rel)
    return found


def _unshipped_conftest_imports(shipped: set[str], repo_root: Path) -> list[str]:
    """Shipped files importing a bare ``conftest`` whose sibling does not ship."""
    offenders = []
    for rel in _conftest_importers(sorted(shipped), repo_root):
        sibling = str(PurePosixPath(rel).parent / "conftest.py")
        if sibling not in shipped:
            offenders.append(f"{rel} imports `conftest`, but {sibling} does not ship")
    return offenders


def test_conftest_import_detector_reads_the_real_tree() -> None:
    """Non-vacuity: the parser finds real importers before any verdict is read."""
    repo_root, _ = _shipped_paths()
    tracked = [p for p in _git("ls-files", "-z", cwd=repo_root).split("\0") if p]
    importers = _conftest_importers(tracked, repo_root)
    assert len(importers) >= 3, (
        f"only {len(importers)} file(s) parsed as importing a bare `conftest`; "
        "the repo has more, so the AST scan is wrong and a clean verdict below "
        f"means nothing: {importers}"
    )


def test_no_shipped_test_imports_a_dropped_conftest() -> None:
    repo_root, shipped_list = _shipped_paths()
    shipped = set(shipped_list)

    assert _conftest_importers(sorted(shipped), repo_root), (
        "no shipped file imports a bare `conftest`, so this detector is vacuous. "
        "Either that is a real change (delete this test) or the shipped set is "
        "wrong -- do not leave it passing."
    )

    offenders = _unshipped_conftest_imports(shipped, repo_root)
    assert not offenders, (
        "the export ships a test importing a `conftest` the allowlist drops. In "
        "the published repo pytest aborts collection for that whole directory, "
        "and the error names the root conftest.py rather than the missing one.\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


@pytest.mark.parametrize(
    "source",
    [
        "from conftest import load_release_module\n",
        "import conftest\n",
        "def t():\n    from conftest import load_release_module\n    return load_release_module\n",
    ],
    ids=["top-level-from", "top-level-import", "function-local"],
)
def test_conftest_detector_flags_every_import_shape(source: str) -> None:
    """Planted red: each shape a bare `conftest` import can take."""
    assert _imports_bare_conftest(source), f"detector missed: {source!r}"


@pytest.mark.parametrize(
    "source",
    [
        "from .conftest import load_release_module\n",
        "from conftest_helpers import x\n",
        "conftest = 1\n",
        "# from conftest import load_release_module\n",
    ],
    ids=["relative", "different-module", "bare-name", "comment"],
)
def test_conftest_detector_stays_silent_on_a_non_import(source: str) -> None:
    assert not _imports_bare_conftest(source), f"detector false-positived: {source!r}"


def test_conftest_detector_flags_the_real_file_when_its_loader_is_dropped() -> None:
    """Planted red against the real tree: drop the loader, expect the offence."""
    repo_root, shipped_list = _shipped_paths()
    shipped = set(shipped_list)
    loader = "tests/unit/release/conftest.py"
    assert loader in shipped, f"plant is stale: {loader} no longer ships"
    offenders = _unshipped_conftest_imports(shipped - {loader}, repo_root)
    assert any(loader in o for o in offenders), f"detector missed the drop: {offenders}"
