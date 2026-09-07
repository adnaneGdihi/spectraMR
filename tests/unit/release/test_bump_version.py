"""Pairs with ``scripts/release/bump_version.py``.

The load-bearing test here is not any single rewrite -- it is
``test_every_source_build_dist_reads_agrees_after_a_bump``. ``build_dist.py`` is
the sole *comparator* of the version set and this module is the sole *writer*;
if the two ever disagree about where the version lives, the writer moves two of
three files and the disagreement surfaces at release time, on a tag, which is
the most expensive moment to find it. So the writer is checked by asking the
comparator, never by re-listing the paths.
"""

from __future__ import annotations

import pytest

from conftest import load_release_module

mod = load_release_module("bump_version")
build_dist = load_release_module("build_dist", alias="build_dist")

V = mod.Version


# ----------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("text", "want"),
    [("0.1.0", V(0, 1, 0)), ("1.2.3", V(1, 2, 3)), ("0.1.1.dev4", V(0, 1, 1, 4))],
)
def test_a_valid_version_round_trips_through_its_own_text(text, want):
    assert mod.parse(text) == want
    assert str(want) == text


@pytest.mark.parametrize("text", ["0.1", "0.1.0.1", "v0.1.0", "0.1.0rc1", "", "nightly"])
def test_a_version_that_is_not_major_minor_build_raises(text):
    """No defaulting: an unreadable version is reported, never inferred (NN18)."""
    with pytest.raises(ValueError):
        mod.parse(text)


def test_a_local_version_is_rejected_because_pypi_rejects_it():
    """`0.1.1+build4` is the natural spelling of "build 4" and is unpublishable.

    PyPI refuses local versions outright, so accepting one here would produce an
    artefact that fails at UPLOAD time -- after the tag exists, which is the one
    moment a version number cannot be taken back.
    """
    with pytest.raises(ValueError, match="PyPI"):
        mod.parse("0.1.1+build4")


def test_a_dev_version_sorts_before_the_release_it_precedes():
    """The claim the whole nightly branch rests on, asserted rather than stated."""
    packaging_version = pytest.importorskip("packaging.version")
    assert packaging_version.Version("0.1.1.dev4") < packaging_version.Version("0.1.1")
    assert packaging_version.Version("0.1.0") < packaging_version.Version("0.1.1.dev1")


# ----------------------------------------------------------------- advancing


@pytest.mark.parametrize(
    ("current", "mode", "want"),
    [
        ("0.1.0", "major", "1.0.0"),
        ("0.1.0", "minor", "0.2.0"),
        ("0.1.0", "build", "0.1.1"),
        ("1.4.7", "major", "2.0.0"),
        ("1.4.7", "minor", "1.5.0"),
        ("0.1.0", "nightly", "0.1.1.dev1"),
        ("0.1.1.dev1", "nightly", "0.1.1.dev2"),
        ("0.1.1.dev9", "release", "0.1.1"),
    ],
)
def test_each_mode_moves_the_number_it_names(current, mode, want):
    assert str(mod.advance(mod.parse(current), mode)) == want


@pytest.mark.parametrize("mode", ["major", "minor", "build"])
def test_a_stable_bump_from_a_dev_version_raises_rather_than_choosing(mode):
    """Genuinely ambiguous, so it names the two modes that resolve it.

    From `0.1.1.dev4`, "bump the build" could mean finish 0.1.1 or start 0.1.2.
    Picking either silently is the defaulting NN3 forbids.
    """
    with pytest.raises(ValueError, match="ambiguous"):
        mod.advance(mod.parse("0.1.1.dev4"), mode)


def test_release_from_a_stable_version_raises():
    with pytest.raises(ValueError, match="nothing to finish"):
        mod.advance(mod.parse("0.1.1"), "release")


# ----------------------------------------------------------------- rewriting


def test_the_init_rewrite_touches_only_the_version_literal():
    text = '"""doc."""\n\n__version__ = "0.1.0"\n__all__ = ["0.1.0-lookalike"]\n'
    out = mod.rewrite_init(text, V(0, 2, 0))
    assert '__version__ = "0.2.0"' in out
    assert '__all__ = ["0.1.0-lookalike"]' in out, "a lookalike elsewhere was rewritten"


def test_the_citation_rewrite_touches_only_the_version_key():
    text = "cff-version: 1.2.0\nversion: 0.1.0\ntitle: x\n"
    out = mod.rewrite_citation(text, V(0, 2, 0))
    assert "version: 0.2.0" in out
    assert "cff-version: 1.2.0" in out, "the cff-version key was rewritten"


def _changelog(body: str = "\n\n### Added\n- a thing\n") -> str:
    return (
        f"# Changelog\n\n## [Unreleased]{body}\n## [0.1.0] - 2026-09-04\n\n- first\n\n"
        "[Unreleased]: https://example.invalid/r/compare/v0.1.0...HEAD\n"
        "[0.1.0]: https://example.invalid/r/releases/tag/v0.1.0\n"
    )


def test_the_changelog_cut_dates_the_release_and_reopens_unreleased():
    out = mod.rewrite_changelog(_changelog(), V(0, 1, 1), "2026-09-04")
    assert "## [Unreleased]\n\n## [0.1.1] - 2026-09-04" in out
    assert "### Added\n- a thing" in out, "the accumulated entries were lost"
    assert "[Unreleased]: https://example.invalid/r/compare/v0.1.1...HEAD" in out
    assert "[0.1.1]: https://example.invalid/r/releases/tag/v0.1.1" in out
    assert "[0.1.0]: https://example.invalid/r/releases/tag/v0.1.0" in out, "history was dropped"


def test_the_link_base_is_read_from_the_file_not_hardcoded():
    """One owner for the repository URL: the file already states it."""
    out = mod.rewrite_changelog(
        _changelog().replace("https://example.invalid/r", "https://other.invalid/q"),
        V(0, 1, 1),
        "2026-09-04",
    )
    assert "https://other.invalid/q/compare/v0.1.1...HEAD" in out
    assert "example.invalid" not in out


def test_an_empty_unreleased_section_raises_rather_than_releasing_nothing():
    with pytest.raises(ValueError, match="empty"):
        mod.rewrite_changelog(_changelog(body="\n\n"), V(0, 1, 1), "2026-09-04")


@pytest.mark.parametrize(
    ("mangle", "match"),
    [
        (lambda t: t.replace("## [Unreleased]", "## Unreleased"), "Unreleased"),
        (
            lambda t: t.replace(
                "[Unreleased]: https://example.invalid/r/compare/v0.1.0...HEAD\n", ""
            ),
            "link reference",
        ),
    ],
)
def test_a_changelog_shape_it_cannot_handle_raises(mangle, match):
    with pytest.raises(ValueError, match=match):
        mod.rewrite_changelog(mangle(_changelog()), V(0, 1, 1), "2026-09-04")


# ------------------------------------------------------- writer meets comparator


@pytest.fixture
def repo(tmp_path):
    """A miniature tree carrying all three version statements."""
    (tmp_path / "src" / "spectramr").mkdir(parents=True)
    (tmp_path / "src" / "spectramr" / "__init__.py").write_text('__version__ = "0.1.0"\n')
    (tmp_path / "CITATION.cff").write_text("cff-version: 1.2.0\nversion: 0.1.0\n")
    (tmp_path / "CHANGELOG.md").write_text(_changelog())
    return tmp_path


def test_every_source_build_dist_reads_agrees_after_a_bump(repo):
    """The writer is checked by asking the comparator, not by re-listing paths."""
    assert mod.main(["minor", "--root", str(repo), "--apply"]) == 0
    others = {
        "init": build_dist.declared_version((repo / "src/spectramr/__init__.py").read_text()),
        "citation": build_dist.citation_version((repo / "CITATION.cff").read_text()),
    }
    changelog = (repo / "CHANGELOG.md").read_text()
    assert build_dist.version_disagreements("0.2.0", others, changelog) == []


def test_a_nightly_bump_leaves_the_changelog_alone(repo):
    before = (repo / "CHANGELOG.md").read_text()
    assert mod.main(["nightly", "--root", str(repo), "--apply"]) == 0
    assert (repo / "CHANGELOG.md").read_text() == before, "a dev build got a release heading"
    assert '__version__ = "0.1.1.dev1"' in (repo / "src/spectramr/__init__.py").read_text()


def test_show_accepts_the_tree_a_nightly_bump_just_wrote(repo, capsys):
    """`show` and `build_dist` must agree, because they are one comparator now.

    They were two: `show` compared `set(values)` and could not express a dev
    build, so it printed DISAGREEMENT on the exact tree `nightly` produces --
    the state the whole 0.1.3 series has been in.
    """
    assert mod.main(["nightly", "--root", str(repo), "--apply"]) == 0
    assert mod.main(["show", "--root", str(repo)]) == 0
    assert "0.1.1.dev1" in capsys.readouterr().out


def test_show_still_reports_a_genuinely_stale_citation(repo, capsys):
    """Accepting the dev shape must not accept a source that really did drift."""
    assert mod.main(["nightly", "--root", str(repo), "--apply"]) == 0
    (repo / "CITATION.cff").write_text("cff-version: 1.2.0\nversion: 0.1.0\n")
    assert mod.main(["show", "--root", str(repo)]) == 1
    assert "DISAGREEMENT" in capsys.readouterr().err


def test_show_reports_a_dev_tree_whose_unreleased_section_was_deleted(repo, capsys):
    assert mod.main(["nightly", "--root", str(repo), "--apply"]) == 0
    (repo / "CHANGELOG.md").write_text("# Changelog\n\n## [0.1.0] - 2026-08-31\n")
    assert mod.main(["show", "--root", str(repo)]) == 1
    assert "no '## [Unreleased]' section" in capsys.readouterr().err


def test_a_dry_run_writes_nothing(repo):
    before = {p.name: p.read_text() for p in repo.rglob("*") if p.is_file()}
    assert mod.main(["minor", "--root", str(repo)]) == 0
    after = {p.name: p.read_text() for p in repo.rglob("*") if p.is_file()}
    assert before == after, "a dry run wrote to the tree"


def test_show_reports_a_disagreement_as_a_nonzero_exit(repo, capsys):
    (repo / "CITATION.cff").write_text("cff-version: 1.2.0\nversion: 9.9.9\n")
    assert mod.main(["show", "--root", str(repo)]) == 1
    assert "DISAGREEMENT" in capsys.readouterr().err


def test_a_full_nightly_series_lands_back_on_an_agreeing_release(repo):
    """0.1.0 -> dev1 -> dev2 -> 0.1.1, and the comparator is happy at the end."""
    for _ in range(2):
        assert mod.main(["nightly", "--root", str(repo), "--apply"]) == 0
    assert '__version__ = "0.1.1.dev2"' in (repo / "src/spectramr/__init__.py").read_text()
    assert mod.main(["release", "--root", str(repo), "--apply"]) == 0
    assert mod.main(["show", "--root", str(repo)]) == 0
