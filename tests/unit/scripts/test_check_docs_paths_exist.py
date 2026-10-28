"""The docs-path gate must go red on a real broken command, not just on a unit case.

Non-negotiable 15: a gate is only a gate for the violation shape it has been watched
failing on. So every assertion here drives ``scan`` over a real temporary tree -- the
call site -- rather than probing ``is_placeholder`` alone. A helper-only pin scores a
call-site plant green, and this gate's whole value is that it runs over a *tree*: the
same page is clean in the private checkout and broken in the export, and only a
tree-driven check can express that.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_GATE = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "check_docs_paths_exist.py"
_spec = importlib.util.spec_from_file_location("check_docs_paths_exist", _GATE)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _tree(tmp_path: Path, page: str, body: str, present: tuple[str, ...] = ()) -> Path:
    """Build a miniature repo: one doc page, plus whichever files really exist."""
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / page).write_text(body)
    for rel in present:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")
    return tmp_path


# --- the plants: each of these MUST turn the gate red -------------------------------
#
# One per shape the rule can take. The shapes are not interchangeable -- an early
# revision of this gate classified by RST block structure and was blind to the
# MyST-fenced and inline-prose forms, which is why the leading verb decides instead.
_PLANTS = [
    ("rst_codeblock", "p.rst", "Run it:\n\n.. code-block:: bash\n\n   python scripts/gone.py\n"),
    ("md_fence", "p.md", "Run it:\n\n```bash\npython scripts/gone.py\n```\n"),
    ("inline_prose", "p.rst", "Invoke ``python scripts/gone.py`` to rebuild.\n"),
    ("bash_verb", "p.rst", ".. code-block:: bash\n\n   bash scripts/gone.sh\n"),
    ("sbatch_verb", "p.rst", ".. code-block:: bash\n\n   sbatch scripts/gone.sbatch\n"),
    ("dot_slash", "p.rst", ".. code-block:: bash\n\n   ./scripts/gone.sh --flag\n"),
    ("pytest_verb", "p.rst", ".. code-block:: bash\n\n   pytest tests/unit/gone.py\n"),
    (
        "config_yaml",
        "p.rst",
        ".. code-block:: bash\n\n   spectramr train -c experiments/gone.yaml\n",
    ),
    # A repo-root script has no directory to anchor on, so the root-anchored pattern
    # is structurally blind to it -- and the export drops several such files.
    (
        "bare_root_script",
        "p.rst",
        ".. code-block:: bash\n\n   bash download_datasets.sh m4raw\n",
    ),
    ("bare_root_sbatch", "p.rst", ".. code-block:: bash\n\n   sbatch run_all.sbatch\n"),
    ("bare_root_dot_slash", "p.rst", ".. code-block:: bash\n\n   ./cluster_update.sh\n"),
]


@pytest.mark.parametrize("shape,page,body", _PLANTS, ids=[p[0] for p in _PLANTS])
def test_a_dead_path_in_a_pasteable_line_is_a_command_finding(
    tmp_path: Path, shape: str, page: str, body: str
) -> None:
    root = _tree(tmp_path, page, body)
    findings = gate.scan(root, root / "docs")
    assert findings, f"{shape}: the gate saw nothing -- it is blind to this shape"
    assert [f[3] for f in findings] == ["COMMAND"], (
        f"{shape}: classified {[f[3] for f in findings]}, but a reader pastes this line"
    )


def test_the_same_path_is_clean_once_the_file_exists(tmp_path: Path) -> None:
    """The gate must key on the tree, not on the text -- that is its entire premise."""
    body = ".. code-block:: bash\n\n   python scripts/here.py\n"
    absent = _tree(tmp_path / "absent", "p.rst", body)
    assert gate.scan(absent, absent / "docs"), "same text, no file: must be a finding"
    present = _tree(tmp_path / "present", "p.rst", body, present=("scripts/here.py",))
    assert gate.scan(present, present / "docs") == [], "same text, file there: must be clean"


@pytest.mark.parametrize(
    "phrasing",
    [
        "Save this as `scripts/made.py`:",
        "Save it as `scripts/made.py`:",
        "Create ``scripts/made.py``:",
        "Write this to `scripts/made.py`:",
    ],
)
def test_a_page_that_hands_you_the_file_may_then_run_it(tmp_path: Path, phrasing: str) -> None:
    """The listing IS the file. Its absence from the repo is the point."""
    body = f"{phrasing}\n\n.. code-block:: python\n\n   pass\n\nRun it:\n\n"
    body += ".. code-block:: bash\n\n   python scripts/made.py\n"
    root = _tree(tmp_path, "p.rst", body)
    assert gate.scan(root, root / "docs") == [], f"{phrasing!r} was not recognised"


def test_the_authoring_exemption_does_not_leak_to_a_different_path(
    tmp_path: Path,
) -> None:
    """A page that authors scripts/made.py earns nothing for scripts/other.py.
    Without this the exemption would whitewash every page containing one listing."""
    body = "Save this as `scripts/made.py`:\n\n.. code-block:: bash\n\n"
    body += "   python scripts/other.py\n"
    root = _tree(tmp_path, "p.rst", body)
    findings = gate.scan(root, root / "docs")
    assert [f[2] for f in findings] == ["scripts/other.py"]


def test_a_bare_script_name_in_prose_is_not_a_finding(tmp_path: Path) -> None:
    """The bare-name pattern is command-gated on purpose. Without that gate a prose
    sentence mentioning ``setup.sh`` -- any ``.sh`` the reader is not being told to
    run -- becomes a finding, and a gate that cries wolf gets switched off."""
    root = _tree(tmp_path, "p.rst", "The historical entrypoint was named setup.sh.\n")
    assert gate.scan(root, root / "docs") == []


def test_a_dead_path_in_prose_is_a_reference_not_a_command(tmp_path: Path) -> None:
    """Both are worth fixing; only one breaks a reader's terminal, so they differ."""
    root = _tree(tmp_path, "p.rst", "The knob is read in :file:`scripts/gone.py`.\n")
    findings = gate.scan(root, root / "docs")
    assert [f[3] for f in findings] == ["REFERENCE"]


@pytest.mark.parametrize(
    "token",
    [
        "experiments/<paradigm>/<arm>.yaml",
        "experiments/inprogress/{cohort}/arm.yaml",
        "scripts/data/gen_*.py",
        "experiments/$COHORT/arm.yaml",
        "experiments/.../arm.yaml",
    ],
)
def test_an_explicit_placeholder_is_exempt(tmp_path: Path, token: str) -> None:
    """Teaching a shape is not claiming a file. Nobody pastes an angle bracket."""
    root = _tree(tmp_path, "p.rst", f".. code-block:: bash\n\n   spectramr audit {token}\n")
    assert gate.scan(root, root / "docs") == []


def test_a_root_without_docs_refuses_rather_than_reporting_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo-rooted script run from outside its repo reports all-clean, and that
    reads exactly like a pass. It must refuse instead."""
    monkeypatch.setattr("sys.argv", ["check_docs_paths_exist.py", "--root", str(tmp_path)])
    assert gate.main() == 1
    assert "refusing to report a vacuous pass" in capsys.readouterr().out


def test_commands_only_still_lists_references_but_does_not_fail_on_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tree(tmp_path, "p.rst", "See :file:`scripts/gone.py` for the reader.\n")
    monkeypatch.setattr(
        "sys.argv", ["check_docs_paths_exist.py", "--root", str(root), "--commands-only"]
    )
    assert gate.main() == 0
    out = capsys.readouterr().out
    assert "scripts/gone.py" in out, "a reference must still be reported, just not fatal"


# --- scan ROOT: the blindness no plant inside docs/ could ever reveal ---------------
#
# Until 2026-08-29 this gate scanned ``docs/`` and nothing else, so README.md and
# CONTRIBUTING.md -- the two most-read pages in the distribution -- were outside its
# scan root. All eleven plants above lived under ``docs/`` and all eleven passed while
# a genuine broken command sat in CONTRIBUTING.md. Measured on the export that day:
# the pre-change gate printed "OK" and exited 0 on a planted root-level violation.
#
# A scan root is an unaudited constant. Widening the *pattern* would not have found
# this; only a plant placed OUTSIDE the scope can.


def _root_page_tree(tmp_path: Path, name: str, body: str) -> Path:
    """A repo whose docs/ is clean and whose ROOT-level page carries the plant."""
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "index.rst").write_text("Nothing broken here.\n")
    (tmp_path / name).write_text(body)
    return tmp_path


@pytest.mark.parametrize(
    "name,body",
    [
        ("README.md", "Install:\n\n```bash\npython scripts/gone.py\n```\n"),
        ("CONTRIBUTING.md", "Then run ``bash scripts/gone.sh`` before the PR.\n"),
        ("SECURITY.md", ".. code-block:: bash\n\n   pytest tests/unit/gone.py\n"),
        ("CHANGELOG.rst", "Run ``python tools/gone.py`` to regenerate.\n"),
    ],
)
def test_a_broken_command_in_a_root_level_page_is_found(
    tmp_path: Path, name: str, body: str
) -> None:
    root = _root_page_tree(tmp_path, name, body)
    findings = gate.scan(root, root / "docs")
    assert [(str(f[0]), f[3]) for f in findings] == [(name, "COMMAND")]


def test_a_root_level_page_is_clean_once_the_file_exists(tmp_path: Path) -> None:
    """Tree-keyed, not text-keyed: the identical page must go green on a fuller tree."""
    root = _root_page_tree(tmp_path, "README.md", "```bash\npython scripts/here.py\n```\n")
    assert gate.scan(root, root / "docs") != []
    (root / "scripts").mkdir()
    (root / "scripts" / "here.py").write_text("")
    assert gate.scan(root, root / "docs") == []


def test_root_pages_are_scanned_in_addition_to_docs_not_instead_of_them(
    tmp_path: Path,
) -> None:
    """Widening the scope must not drop the original one."""
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    (docs / "guide.rst").write_text("Run ``python scripts/in_docs.py`` now.\n")
    (tmp_path / "README.md").write_text("Run ``python scripts/in_root.py`` now.\n")
    pages = {str(f[0]) for f in gate.scan(tmp_path, docs)}
    assert pages == {"docs/guide.rst", "README.md"}


def test_shipped_pages_does_not_descend_below_the_repository_root(tmp_path: Path) -> None:
    """Only top-level prose joins docs/ -- not every .md buried in src/ or tests/."""
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    (docs / "index.rst").write_text("ok\n")
    (tmp_path / "README.md").write_text("ok\n")
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    (nested / "NOTES.md").write_text("Run ``python scripts/gone.py``.\n")
    names = {p.name for p in gate.shipped_pages(tmp_path, docs)}
    assert names == {"index.rst", "README.md"}


def test_a_docs_dir_holding_no_page_refuses_rather_than_reporting_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal one level down from the missing-``docs/`` case.

    An empty ``docs/`` scans nothing and prints OK, which is indistinguishable
    from a clean tree. That was survivable while this gate only ran in CI; it
    sets the exit code of the public export now, and there "no page was checked"
    must never be readable as "every page is clean".
    """
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr("sys.argv", ["check_docs_paths_exist.py", "--root", str(tmp_path)])
    assert gate.main() == 1
    assert "refusing to report a vacuous pass" in capsys.readouterr().out


def test_a_tree_whose_prose_is_all_root_level_is_scanned_not_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``docs/`` at all, but a root page carrying a real plant.

    Every other root-level case above builds a ``docs/`` first, so this shape --
    a distribution whose prose is entirely root-level -- was refused as a
    misconfigured scan root rather than scanned. The refusal keys on the page
    set now, and the plant must be found.
    """
    (tmp_path / "README.md").write_text("Install:\n\n```bash\npython scripts/gone.py\n```\n")
    assert not (tmp_path / "docs").exists()
    monkeypatch.setattr("sys.argv", ["check_docs_paths_exist.py", "--root", str(tmp_path)])
    assert gate.main() == 1
    out = capsys.readouterr().out
    assert "scripts/gone.py" in out
    assert "COMMAND" in out
    assert "vacuous" not in out, "this must fail on the finding, not on the missing docs/"


def test_the_same_root_only_tree_passes_once_the_file_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The widened shape must be able to pass, or its failure says nothing."""
    (tmp_path / "README.md").write_text("Install:\n\n```bash\npython scripts/gone.py\n```\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "gone.py").write_text("")
    monkeypatch.setattr("sys.argv", ["check_docs_paths_exist.py", "--root", str(tmp_path)])
    assert gate.main() == 0
    assert "OK:" in capsys.readouterr().out
