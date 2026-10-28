"""``docs/contributing/ci.rst`` explains the accessor escape hatch; pin its claims.

The page tells a reader what ``external_reads`` does, how many paths the first
declaration carries, and which example path is covered. Those are code facts
restated in prose, and prose has no compiler -- ``ruff`` never opens this page and
Sphinx builds it without ``-W``, so a number that goes stale here goes stale
silently and is then trusted *instead of* being measured.

**This is a fidelity check, not a truth check**, and the distinction matters
(``test_known_limitations_schema_claims.py`` states it well: a test that shares
its subject's blind spot verifies agreement, not correctness). Whether 112 is the
*right* answer is settled elsewhere, by an executed oracle:
``test_weights.py::TestAccessorReadPathsAreTheExecutedReadSet`` moves one schema
field at a time and observes whether the built table changes. What is checked
*here* is narrower and still worth checking -- that the page agrees with the code
it describes.

Every test below asserts its own premise before scoring. A regex that matches
nothing must fail, never sweep an empty set and report success.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from spectramr.config.key_reachability_model import ReadEvidence
from spectramr.models.losses.weights import accessor_read_paths

DOC = Path(__file__).resolve().parents[3] / "docs" / "contributing" / "ci.rst"

#: The heading the module docstring of ``config/key_reachability.py`` points at.
#: Pinned in both places so renaming the section cannot silently rot the pointer.
SECTION_TITLE = "Declaring a read the token index cannot see"

SOURCE_WITH_POINTER = (
    Path(__file__).resolve().parents[3] / "src" / "spectramr" / "config" / "key_reachability.py"
)


@pytest.fixture(scope="module")
def section() -> str:
    """The text of the section, from its heading to the next same-level heading."""
    text = DOC.read_text(encoding="utf-8")
    start = text.find(SECTION_TITLE + "\n" + "=" * len(SECTION_TITLE))
    assert start != -1, (
        f"{DOC} has no section titled {SECTION_TITLE!r}. Either the section was "
        "renamed -- in which case update SECTION_TITLE here and the pointer in "
        "config/key_reachability.py's module docstring together -- or the "
        "documentation for `external_reads` was dropped."
    )
    # Step past the heading AND its own underline -- searching for the next
    # ``=``-run from immediately after the title matches that underline at
    # offset 0 and yields an empty body, which the length assertion below
    # catches rather than letting three tests scan an empty string.
    after_underline = start + len(SECTION_TITLE) * 2 + 1
    rest = text[after_underline:]
    end = rest.find("\n" + "=" * 10)
    body = rest if end == -1 else rest[:end]
    assert len(body) > 500, f"section body is only {len(body)} chars -- did it get truncated?"
    return body


def test_the_path_count_matches_the_accessor(section: str) -> None:
    """The figure the page publishes is the number the accessor actually returns."""
    claimed = re.findall(r"returns the (\d+) paths", section)
    assert len(claimed) == 1, (
        "expected exactly one 'returns the N paths' claim in the section, found "
        f"{claimed!r} -- the regex and the prose have diverged"
    )
    assert int(claimed[0]) == len(accessor_read_paths()), (
        f"the page says {claimed[0]} paths; accessor_read_paths() returns "
        f"{len(accessor_read_paths())}. Update the page."
    )


def test_the_worked_example_path_is_really_covered(section: str) -> None:
    """The page names one path as the motivating case. It must be in the mapping."""
    example = "losses.physics.lambda_bloch_residual"
    assert example in section, f"the section no longer names {example!r}"
    assert example in accessor_read_paths(), (
        f"the page presents {example!r} as the path the declaration rescues, but "
        "accessor_read_paths() does not claim it -- the example is now fiction"
    )


def test_the_contrasting_path_is_really_not_covered(section: str) -> None:
    """The page's full-path-keying argument rests on a twin that stays uncovered."""
    assert "training.multi.stages" in section, (
        "the section no longer contrasts the multi-stage twin, which is what makes "
        "the full-path-vs-leaf argument concrete"
    )
    twins = [path for path in accessor_read_paths() if path.startswith("training.multi.stages")]
    assert twins == [], (
        "the page argues the identical leaf under 'training.multi.stages' stays "
        f"unread, but accessor_read_paths() now claims {twins!r}"
    )


def test_the_evidence_kind_named_in_prose_exists() -> None:
    """The page names a ``ReadEvidence`` member; a renamed member must fail here."""
    text = DOC.read_text(encoding="utf-8")
    assert "ACCESSOR_READ" in text, "the page no longer names the evidence kind"
    assert hasattr(ReadEvidence, "ACCESSOR_READ")
    assert ReadEvidence.ACCESSOR_READ is not ReadEvidence.LIVE_READ, (
        "the page's central claim is that an accessor read is deliberately NOT a "
        "live read; collapsing the two would make the documentation wrong"
    )


def test_the_module_docstring_pointer_resolves() -> None:
    """``key_reachability.py`` points at this section by title. Pin the pointer.

    A source-text pointer to a doc heading is exactly the reference that breaks on
    a rename without any gate noticing: the page still builds, the module still
    imports, and the reader follows a heading that is not there.
    """
    source = SOURCE_WITH_POINTER.read_text(encoding="utf-8")
    assert "docs/contributing/ci.rst" in source, (
        "config/key_reachability.py no longer points at the CI page; either restore "
        "the pointer or delete this test with the section it guards"
    )
    assert SECTION_TITLE in source, (
        f"the module docstring cites a section title that is not {SECTION_TITLE!r} -- "
        "the pointer and the heading have drifted apart"
    )
