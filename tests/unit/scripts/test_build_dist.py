"""Contract tests for ``scripts/release/build_dist.py``.

Non-negotiable 15: a gate is only a gate for the violation shape it has been
watched to fail on. This one guards a **one-shot, irreversible** act -- PyPI
refuses to re-upload a filename it has already seen, so a defect that reaches
the index costs a version number rather than an amend. Every clause below is
therefore planted red, and each is planted at the **call site** (``main``) as
well as in its helper: a helper-only pin scores green on a ``main`` that
computes the answer and forgets to record it, which is the exact shape of a
detector that never fires.
"""

from __future__ import annotations

import importlib.util
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "release" / "build_dist.py"


def _load():
    """Load the script by path -- ``tests/unit/scripts`` shadows the root ``scripts``."""
    spec = importlib.util.spec_from_file_location("_build_dist_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


bd = _load()


# --------------------------------------------------------------------------
# completeness
# --------------------------------------------------------------------------
def test_completeness_is_clean_when_the_two_sets_agree() -> None:
    assert bd.completeness({"a.py", "py.typed"}, {"a.py", "py.typed"}) == ([], [])


def test_a_file_dropped_from_the_wheel_is_reported_missing() -> None:
    missing, extra = bd.completeness({"a.py"}, {"a.py", "config/presets/x.yaml"})
    assert missing == ["config/presets/x.yaml"] and extra == []


def test_a_file_only_in_the_wheel_is_reported_extra() -> None:
    missing, extra = bd.completeness({"a.py", "stowaway.pth"}, {"a.py"})
    assert missing == [] and extra == ["stowaway.pth"]


# --------------------------------------------------------------------------
# contamination -- the shape produced by measuring a tree before building it
# --------------------------------------------------------------------------
def test_a_clean_payload_reports_no_contamination() -> None:
    assert bd.contamination({"a.py", "py.typed", "config/presets/x.yaml"}) == []


@pytest.mark.parametrize(
    "planted",
    ["a.pyc", "sub/b.pyo", "__pycache__/c.cpython-312.pyc", "deep/__pycache__/d.pyc"],
)
def test_each_build_artefact_shape_is_caught(planted: str) -> None:
    assert bd.contamination({"a.py", planted}) == [planted]


def test_a_file_merely_named_like_a_cache_is_not_flagged() -> None:
    """Precision: ``__pycache__`` must be a path *component*, not a substring."""
    assert bd.contamination({"tools/__pycache__helper.py"}) == []


# --------------------------------------------------------------------------
# version agreement across the four independent statements of it
# --------------------------------------------------------------------------
def test_declared_version_reads_the_package_dunder() -> None:
    assert bd.declared_version('x = 1\n__version__ = "0.1.0"\n') == "0.1.0"


def test_declared_version_ignores_a_dunder_that_is_not_at_line_start() -> None:
    assert bd.declared_version('# __version__ = "9.9.9" in a comment\n') is None


def test_changelog_version_skips_the_unreleased_heading() -> None:
    text = "# Changelog\n\n## [Unreleased]\n\n## [0.1.0] - 2026-08-31\n"
    assert bd.changelog_version(text) == "0.1.0"


def test_changelog_version_is_none_when_only_unreleased_exists() -> None:
    assert bd.changelog_version("## [Unreleased]\n") is None


@pytest.mark.parametrize("line", ["version: 0.1.0", 'version: "0.1.0"', "version: '0.1.0'"])
def test_citation_version_reads_quoted_and_bare_forms(line: str) -> None:
    assert bd.citation_version(f"cff-version: 1.2.0\n{line}\n") == "0.1.0"


RELEASED_CHANGELOG = "# Changelog\n\n## [Unreleased]\n\n## [0.1.0] - 2026-08-31\n"


def test_versions_that_all_agree_produce_no_complaint() -> None:
    others = dict.fromkeys(["__init__.py", "CITATION.cff"], "0.1.0")
    assert bd.version_disagreements("0.1.0", others, RELEASED_CHANGELOG) == []


@pytest.mark.parametrize("stale", ["__init__.py", "CITATION.cff"])
def test_any_single_stale_version_source_is_named(stale: str) -> None:
    others = dict.fromkeys(["__init__.py", "CITATION.cff"], "0.1.0")
    others[stale] = "0.0.9"
    problems = bd.version_disagreements("0.1.0", others, RELEASED_CHANGELOG)
    assert len(problems) == 1 and problems[0].startswith(f"{stale}: ")


def test_an_unreadable_version_is_a_failure_not_a_pass() -> None:
    """A regex that stops matching must not read as agreement."""
    problems = bd.version_disagreements("0.1.0", {"CITATION.cff": None}, RELEASED_CHANGELOG)
    assert problems == ["CITATION.cff: unreadable"]


def test_an_unreadable_reference_is_a_failure_not_a_pass() -> None:
    """A wheel filename the regex stops matching must not compare as agreement."""
    problems = bd.version_disagreements(None, {"CITATION.cff": "0.1.0"}, RELEASED_CHANGELOG)
    assert problems == ["reference version: unreadable"]


# --------------------------------------------------------------------------
# the CHANGELOG is judged by SHAPE, because a release and a dev build state
# themselves there differently. One planted violation per shape the rule takes.
# --------------------------------------------------------------------------
DEV_CHANGELOG = (
    "# Changelog\n\n## [Unreleased]\n\n### Added\n- a thing\n\n## [0.1.2] - 2026-09-01\n"
)


def test_a_dev_build_agrees_with_an_open_unreleased_section() -> None:
    """The shape that made `nightly` unbuildable: 0.1.3.dev1 beside a 0.1.2 heading.

    `bump_version.py nightly` writes no heading for a dev build, so equality
    against the newest heading rejected every tree it had just written correctly.
    """
    assert bd.changelog_disagreement(DEV_CHANGELOG, "0.1.3.dev1") is None


def test_a_dev_build_without_an_unreleased_section_is_named() -> None:
    assert "no '## [Unreleased]' section" in str(
        bd.changelog_disagreement("## [0.1.2] - 2026-09-01\n", "0.1.3.dev1")
    )


def test_a_dev_build_of_an_already_released_version_is_named() -> None:
    """0.1.2.dev4 beside a dated [0.1.2] is a build of a version that shipped."""
    assert "already has a dated heading" in str(
        bd.changelog_disagreement(DEV_CHANGELOG, "0.1.2.dev4")
    )


def test_a_release_whose_heading_is_stale_is_named() -> None:
    assert bd.changelog_disagreement(DEV_CHANGELOG, "0.1.3") == "newest heading '0.1.2' != '0.1.3'"


def test_a_release_with_no_dated_heading_at_all_is_named() -> None:
    """`[Unreleased]` alone satisfies a dev build and must not satisfy a release."""
    assert "no released heading" in str(bd.changelog_disagreement("## [Unreleased]\n", "0.1.3"))


def test_a_release_that_matches_the_newest_heading_agrees() -> None:
    assert bd.changelog_disagreement(DEV_CHANGELOG, "0.1.2") is None


# --------------------------------------------------------------------------
# console script
# --------------------------------------------------------------------------
def test_entry_point_module_path_drops_the_package_segment() -> None:
    assert bd.entry_point_module_path("spectramr.cli.app:main") == "cli/app.py"


def test_entry_point_target_is_none_when_undeclared() -> None:
    assert bd.entry_point_target({"project": {}}) is None


# --------------------------------------------------------------------------
# call-site plants: main() must RECORD what the helpers compute
# --------------------------------------------------------------------------
#: The real exporter, copied into every synthetic repo. A stub parser would make
#: the reuse untested -- the point of load_allowances is that build_dist.py does
#: NOT own the allowlist grammar, and only the real module can show that holds.
EXPORTER = SCRIPT.parent / "export_public_tree.py"


def _fake_repo(
    tmp: Path,
    *,
    version: str = "0.1.0",
    typed: bool = True,
    payload: dict[str, bytes] | None = None,
    sdist_extra: tuple[str, ...] = (),
    allowlist: str | None = "src/\nscripts/\npyproject.toml\n",
    changelog: str | None = None,
) -> Path:
    pkg = tmp / "src" / "spectramr"
    (pkg / "cli").mkdir(parents=True)
    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (pkg / "cli" / "app.py").write_text("def main() -> int:\n    return 0\n")
    if typed:
        (pkg / "py.typed").write_text("")
    (tmp / "CHANGELOG.md").write_text(
        changelog if changelog is not None else f"## [Unreleased]\n\n## [{version}] - 2026-08-31\n"
    )
    (tmp / "CITATION.cff").write_text(f'cff-version: 1.2.0\nversion: "{version}"\n')
    (tmp / "pyproject.toml").write_text(
        '[project]\nname = "spectramr"\n\n[project.scripts]\nspectramr = "spectramr.cli.app:main"\n'
    )

    if allowlist is not None:
        rel = tmp / "scripts" / "release"
        rel.mkdir(parents=True)
        (rel / "public_allowlist.txt").write_text(allowlist)
        (rel / "export_public_tree.py").write_text(EXPORTER.read_text(encoding="utf-8"))

    dist = tmp / "dist"
    dist.mkdir()
    default = {"spectramr/__init__.py": b"x", "spectramr/cli/app.py": b"x"}
    if typed:
        default["spectramr/py.typed"] = b""
    members = payload if payload is not None else default
    with zipfile.ZipFile(dist / f"spectramr-{version}-py3-none-any.whl", "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
        z.writestr(f"spectramr-{version}.dist-info/METADATA", "Name: spectramr\n")
    with tarfile.open(dist / f"spectramr-{version}.tar.gz", "w:gz") as t:
        t.add(tmp / "pyproject.toml", arcname=f"spectramr-{version}/pyproject.toml")
        for root in sdist_extra:
            t.add(tmp / "pyproject.toml", arcname=f"spectramr-{version}/{root}")
    return dist


@pytest.fixture()
def no_twine(monkeypatch: pytest.MonkeyPatch):
    """Stub the subprocess leg so these tests measure main()'s own bookkeeping."""
    monkeypatch.setattr(bd, "_run", lambda cmd, cwd: 0)


def test_main_accepts_a_consistent_distribution(tmp_path: Path, no_twine, capsys) -> None:
    dist = _fake_repo(tmp_path)
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 0
    assert "OK --" in capsys.readouterr().out


def test_main_fails_when_the_wheel_drops_package_data(tmp_path: Path, no_twine, capsys) -> None:
    dist = _fake_repo(
        tmp_path, payload={"spectramr/__init__.py": b"x", "spectramr/cli/app.py": b"x"}
    )  # py.typed dropped
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 1
    assert "missing from wheel: py.typed" in capsys.readouterr().out


def test_main_fails_when_the_wheel_carries_a_file_the_source_tree_lacks(
    tmp_path: Path, no_twine, capsys
) -> None:
    """The 'extra' half of completeness, planted at the call site.

    Found by mutation: deleting the ``only in wheel`` row from main()'s report
    loop left every other test in this file green, because ``extra`` was pinned
    only in its helper.
    """
    dist = _fake_repo(
        tmp_path,
        payload={
            "spectramr/__init__.py": b"x",
            "spectramr/cli/app.py": b"x",
            "spectramr/py.typed": b"",
            "spectramr/stowaway.pth": b"x",
        },
    )
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 1
    assert "only in wheel: stowaway.pth" in capsys.readouterr().out


def test_main_fails_on_a_contaminated_payload(tmp_path: Path, no_twine, capsys) -> None:
    dist = _fake_repo(
        tmp_path,
        payload={
            "spectramr/__init__.py": b"x",
            "spectramr/cli/app.py": b"x",
            "spectramr/py.typed": b"",
            "spectramr/__pycache__/__init__.cpython-312.pyc": b"x",
        },
    )
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 1
    assert "build artefact in payload" in capsys.readouterr().out


def test_main_fails_when_the_package_never_declared_py_typed(
    tmp_path: Path, no_twine, capsys
) -> None:
    """Distinct from a *dropped* py.typed, which completeness already catches.

    Absent from wheel AND source, the two sets still agree, so only the explicit
    clause can see it -- and the classifier keeps advertising 'Typing :: Typed'.
    Found by mutation: deleting that clause left every other test green.
    """
    dist = _fake_repo(tmp_path, typed=False)
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 1
    assert "py.typed absent" in capsys.readouterr().out


def test_main_fails_when_the_changelog_version_is_stale(tmp_path: Path, no_twine, capsys) -> None:
    dist = _fake_repo(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text("## [Unreleased]\n\n## [0.0.9] - 2026-01-01\n")
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 1
    assert "version disagreement" in capsys.readouterr().out


def test_main_accepts_a_dev_series_tree(tmp_path: Path, no_twine, capsys) -> None:
    """The lane `.github/workflows/dev-publish.yml` builds from. Was exit 1.

    A call site that still compared the newest heading by string would fail here
    while every helper test above stayed green -- which is how this shipped.
    """
    dist = _fake_repo(
        tmp_path,
        version="0.1.3.dev7",
        changelog="## [Unreleased]\n\n### Added\n- a thing\n\n## [0.1.2] - 2026-09-01\n",
    )
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 0
    assert "OK --" in capsys.readouterr().out


def test_main_fails_a_dev_tree_whose_unreleased_section_was_deleted(
    tmp_path: Path, no_twine, capsys
) -> None:
    dist = _fake_repo(tmp_path, version="0.1.3.dev7", changelog="## [0.1.2] - 2026-09-01\n")
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 1
    assert "no '## [Unreleased]' section" in capsys.readouterr().out


def test_main_fails_when_the_console_script_names_a_missing_module(
    tmp_path: Path, no_twine, capsys
) -> None:
    dist = _fake_repo(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "spectramr"\n\n[project.scripts]\n'
        'spectramr = "spectramr.cli.renamed:main"\n'
    )
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 1
    assert "absent from the wheel" in capsys.readouterr().out


def test_main_fails_when_twine_rejects_the_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The twine leg must be wired into the verdict, not merely executed."""
    dist = _fake_repo(tmp_path)
    monkeypatch.setattr(bd, "_run", lambda cmd, cwd: 1)
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 1
    assert "twine check failed" in capsys.readouterr().out


def test_a_stale_artefact_from_another_version_is_refused(tmp_path: Path, no_twine) -> None:
    """Two wheels in dist/ means an old one could be checked -- or published."""
    dist = _fake_repo(tmp_path)
    (dist / "spectramr-0.0.9-py3-none-any.whl").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    with pytest.raises(SystemExit, match="expected exactly one"):
        bd.main(["--repo", str(tmp_path), "--check-only", str(dist)])


# --------------------------------------------------------------------------
# --expect-version: the tag must agree with the tree it is tagging.
#
# This is the only disagreement in the file that cannot be repaired after the
# fact. Every other check catches something a rebuild fixes; a tag pushed at a
# tree declaring a different version uploads that other version's filename, and
# PyPI never lets it go.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("refs/tags/v0.1.0", "0.1.0"),  # $GITHUB_REF
        ("v0.1.0", "0.1.0"),  # ${{ github.ref_name }}
        ("0.1.0", "0.1.0"),  # a hand-typed version
        ("v1.2.3rc1", "1.2.3rc1"),  # PEP 440 pre-release
        ("  v0.1.0\n", "0.1.0"),  # $(git describe) keeps the newline
    ],
)
def test_tag_version_reads_every_shape_ci_actually_hands_over(ref, expected) -> None:
    assert bd.tag_version(ref) == expected


@pytest.mark.parametrize("ref", ["main", "refs/heads/dev", "v", "release-0.1.0", ""])
def test_tag_version_refuses_a_ref_that_states_no_version(ref) -> None:
    """None is reported as unreadable downstream -- never silently as agreement."""
    assert bd.tag_version(ref) is None


def test_main_accepts_a_tag_that_matches_the_tree(tmp_path: Path, no_twine, capsys) -> None:
    """The flag must be able to pass, or the plants below prove only that it always fails."""
    dist = _fake_repo(tmp_path, version="0.1.0")
    assert (
        bd.main(
            [
                "--repo",
                str(tmp_path),
                "--check-only",
                str(dist),
                "--expect-version",
                "refs/tags/v0.1.0",
            ]
        )
        == 0
    )
    assert "git tag=0.1.0" in capsys.readouterr().out


def test_main_fails_when_the_tag_names_a_version_the_tree_does_not_declare(
    tmp_path: Path, no_twine, capsys
) -> None:
    """v0.1.1 at a 0.1.0 tree publishes spectramr-0.1.0 and burns that filename."""
    dist = _fake_repo(tmp_path, version="0.1.0")
    assert (
        bd.main(["--repo", str(tmp_path), "--check-only", str(dist), "--expect-version", "v0.1.1"])
        == 1
    )
    assert "version disagreement" in capsys.readouterr().out


def test_main_fails_when_the_expected_ref_is_not_a_version_at_all(
    tmp_path: Path, no_twine, capsys
) -> None:
    """A workflow passing a branch name must not be read as 'nothing to compare'."""
    dist = _fake_repo(tmp_path, version="0.1.0")
    assert (
        bd.main(
            [
                "--repo",
                str(tmp_path),
                "--check-only",
                str(dist),
                "--expect-version",
                "refs/heads/dev",
            ]
        )
        == 1
    )
    assert "version disagreement" in capsys.readouterr().out


def test_main_without_the_flag_does_not_invent_a_fifth_source(
    tmp_path: Path, no_twine, capsys
) -> None:
    """Omitting --expect-version must leave the tag out, not add it as unreadable."""
    dist = _fake_repo(tmp_path, version="0.1.0")
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 0
    assert "git tag" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# The sdist must not carry a path the public allowlist does not admit.
#
# Hatchling's sdist is "everything not gitignored" filtered by a *denylist*, so
# it ships whatever nobody remembered to exclude. Run in the research checkout
# this is not hypothetical: the sdist carries .agent, CLAUDE.md, TODO/, paper/,
# data/ and the cluster scripts. Uploading it publishes the internal tree, and
# PyPI does not take it back.
# --------------------------------------------------------------------------


def test_unadmitted_roots_flags_a_root_no_allowance_can_reach() -> None:
    assert bd.unadmitted_roots({"src", "TODO"}, ["src/", "docs/index.md"]) == ["TODO"]


def test_unadmitted_roots_admits_a_root_some_allowance_names() -> None:
    """One-directional by design: a partial allowance is the export's call, not ours."""
    assert bd.unadmitted_roots({"docs"}, ["docs/index.md"]) == []


def test_unadmitted_roots_honours_a_glob_allowance() -> None:
    assert bd.unadmitted_roots({"README.md", "SECRETS.md"}, ["READ*.md"]) == ["SECRETS.md"]


def test_pkg_info_is_the_only_exempt_root() -> None:
    """It is written by the backend, so it can never appear in an allowlist."""
    assert bd.unadmitted_roots({"PKG-INFO"}, ["src/"]) == []
    assert bd.unadmitted_roots({"PKG-INFO.bak"}, ["src/"]) == ["PKG-INFO.bak"]


def test_main_fails_when_the_sdist_carries_an_internal_path(
    tmp_path: Path, no_twine, capsys
) -> None:
    dist = _fake_repo(tmp_path, sdist_extra=("CLAUDE.md", "TODO"))
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 1
    out = capsys.readouterr().out
    assert "internal path in sdist: CLAUDE.md" in out
    assert "internal path in sdist: TODO" in out


def test_main_does_not_flag_the_backend_written_pkg_info(tmp_path: Path, no_twine, capsys) -> None:
    """Real sdists always carry it; flagging it would make the check unusable."""
    dist = _fake_repo(tmp_path, sdist_extra=("PKG-INFO",))
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 0
    assert "sdist leaks  : 0" in capsys.readouterr().out


def test_main_fails_when_the_allowlist_is_absent_rather_than_passing(
    tmp_path: Path, no_twine, capsys
) -> None:
    """Unverifiable is a failure. A soft-skip here passes exactly when it matters."""
    dist = _fake_repo(tmp_path, sdist_extra=("CLAUDE.md",), allowlist=None)
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 1
    assert "not verifiable against the public allowlist" in capsys.readouterr().out


def test_main_fails_when_the_allowlist_holds_only_denials(tmp_path: Path, no_twine, capsys) -> None:
    """parse_allowlist raises SystemExit on an allowance-free file; catching it is
    what stops the run from aborting mid-report with the other findings unprinted."""
    dist = _fake_repo(tmp_path, sdist_extra=("CLAUDE.md",), allowlist="!TODO/\n# nothing\n")
    assert bd.main(["--repo", str(tmp_path), "--check-only", str(dist)]) == 1
    assert "not verifiable against the public allowlist" in capsys.readouterr().out
