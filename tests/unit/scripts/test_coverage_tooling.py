"""Unit tests for the coverage triage scripts.

The three scripts live under ``scripts/coverage/`` and are reporting
tools — they never exit non-zero on missing coverage, they just emit
tables. The tests verify:

* ``print_per_layer.main`` groups files into top-level layers and
  computes per-layer line coverage from ``coverage.xml``
* ``missing_test_files`` enumerates ``src/*.py`` and checks for paired
  ``tests/unit/...``
* ``shape_coverage_report`` enumerates the live model / loss / metric
  registries and matches each name against ``test_*shapes*.py`` files
  (it used to scan src for ``@register_*`` decorators; that design, and
  the tests written against it, are gone — see issue #280)

Each script is run as a sub-process via ``runpy`` so we exercise the
``main()`` entry point exactly the same way the Makefile target does.

These tests do *not* require pytorch — they only touch the filesystem
and parse small XML / regex inputs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.utils.repo_scripts import require_repo_file

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(module_name: str, file_name: str):
    """Import a script by file path (no need to put scripts/ on sys.path).

    Guarded: scripts/coverage/ ships only in part, so in the export this load
    raised FileNotFoundError. Per-call rather than module-level -- the classes
    covering the scripts that DO ship keep running there.
    """
    path = require_repo_file(f"scripts/coverage/{file_name}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ``print_per_layer`` is covered by tests/unit/coverage/test_print_per_layer.py,
# which owns it alone. It used to be tested here too, from a fixture whose
# filenames carried a ``spectramr/`` prefix that coverage does not emit for this
# repo; the assertions passed because that fixture and the parser's leading-
# component strip were wrong in mirror-image ways. One owner, one fixture.

# ── missing_test_files ─────────────────────────────────────────────


class TestMissingTestFiles:
    """``scripts/coverage/missing_test_files.py``.

    Rewritten 2026-07-14 (issue #280). The previous class tested an API that no
    longer exists — ``_find_paired_test()``, ``_enumerate_src_modules()``, a
    ``TEST_ROOT``/``UNIT_ROOT`` constant pair and a ``--csv`` flag. The script was
    redesigned: instead of looking for a *file* named ``test_<stem>.py`` next to
    each module, it now asks whether the module's stem is **mentioned anywhere**
    in the test corpus, always emits CSV, and gates on ``--min-loc``.

    The stale fixtures patched constants that were gone, and because
    ``monkeypatch.setattr`` defaults to ``raising=True`` every test errored at
    SETUP — so the assertions never ran and nothing here was actually verified.
    That matters more than usual: this script is the coverage side of the
    source<->test pairing gate (CLAUDE.md non-negotiable #10).
    """

    @pytest.fixture
    def fake_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A tiny ``src/`` + ``tests/`` tree with the module constants rerouted."""
        src = tmp_path / "src" / "core"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("")
        (src / "alpha.py").write_text("def f(): pass\n")
        (src / "beta.py").write_text("def g(): pass\n")
        (src / "gamma.py").write_text("def h(): pass\n")

        tests = tmp_path / "tests" / "unit" / "core"
        tests.mkdir(parents=True)
        # alpha + beta are mentioned by name; gamma is mentioned nowhere.
        (tests / "test_alpha.py").write_text("from core.alpha import f\ndef test_a(): pass\n")
        (tests / "test_beta_shapes.py").write_text("from core.beta import g\ndef test_b(): pass\n")

        mod = _load("missing_test_files_test", "missing_test_files.py")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(mod, "SRC_ROOT", tmp_path / "src")
        monkeypatch.setattr(mod, "TESTS_ROOT", tmp_path / "tests")

        return mod, tmp_path

    def test_corpus_collects_every_test_file(self, fake_repo) -> None:
        mod, _ = fake_repo
        corpus = mod._build_test_corpus()
        assert sorted(Path(p).name for p in corpus) == [
            "test_alpha.py",
            "test_beta_shapes.py",
        ]

    def test_count_substantive_lines_ignores_blanks_and_comments(
        self, fake_repo, tmp_path: Path
    ) -> None:
        mod, _ = fake_repo
        f = tmp_path / "counted.py"
        f.write_text("import os\n\n# a comment\n\ndef g():\n    return 1\n")
        # import + def + return == 3; the blank lines and the comment do not count.
        assert mod._count_substantive_lines(f) == 3

    def test_layer_label_derives_from_path(self, fake_repo, tmp_path: Path) -> None:
        mod, _ = fake_repo
        assert mod._layer_of(tmp_path / "src" / "core" / "alpha.py") == "core"
        assert mod._layer_of(tmp_path / "src" / "top.py") == "(root)"

    def test_main_reports_unreferenced_module(self, fake_repo, capsys) -> None:
        mod, _ = fake_repo
        # min-loc 0: the fixture's modules are one-liners and would otherwise be
        # filtered out by the default 40-LOC floor.
        rc = mod.main(["--min-loc", "0"])
        assert rc == 0

        out = capsys.readouterr().out
        assert out.splitlines()[0] == "module_path,referenced,layer"
        rows = {
            line.split(",")[0].rsplit("/", 1)[-1]: line.split(",")[1]
            for line in out.splitlines()[1:]
            if line.strip()
        }
        # __init__.py is always skipped.
        assert "__init__.py" not in rows
        assert rows["alpha.py"] == "True"
        assert rows["beta.py"] == "True"
        # gamma is named in no test file → unreferenced.
        assert rows["gamma.py"] == "False"

    def test_unreferenced_only_filters_referenced_rows(self, fake_repo, capsys) -> None:
        mod, _ = fake_repo
        rc = mod.main(["--min-loc", "0", "--unreferenced-only"])
        assert rc == 0

        out = capsys.readouterr().out
        assert "gamma.py" in out
        assert "alpha.py" not in out
        assert "beta.py" not in out

    def test_min_loc_floor_excludes_trivial_modules(self, fake_repo, capsys) -> None:
        """The default floor is what keeps the report signal-bearing."""
        mod, _ = fake_repo
        rc = mod.main([])  # default --min-loc 40; every fixture module is 1 LOC
        assert rc == 0

        body = capsys.readouterr().out.splitlines()[1:]
        assert [line for line in body if line.strip()] == []

    def test_main_returns_2_when_src_root_is_missing(
        self, fake_repo, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A vanished src root must fail loudly, not report 0 modules as 'all clear'."""
        mod, _ = fake_repo
        monkeypatch.setattr(mod, "SRC_ROOT", tmp_path / "does_not_exist")
        assert mod.main(["--min-loc", "0"]) == 2


# ── shape_coverage_report ──────────────────────────────────────────


class TestShapeCoverageReport:
    """``scripts/coverage/shape_coverage_report.py``.

    Rewritten 2026-07-14 (issue #280). The previous class tested an API that no
    longer exists: ``_scan_decorators()``, ``_shape_test_for()``, a ``SRC_ROOT``
    constant, a ``TESTS_UNIT_ROOT`` constant and a ``--kind`` flag. The script was
    rewritten to enumerate names from the LIVE registries (``MODEL_REGISTRY`` /
    ``LossRegistry`` / ``MetricsRegistry``) instead of grepping ``src/`` for
    ``@register_*`` decorators, so all of those seams disappeared with it.

    The tests kept monkeypatching the vanished constants, and
    ``monkeypatch.setattr`` defaults to ``raising=True`` — so each one blew up at
    SETUP with ``AttributeError``. Net effect: the tooling that backs the
    source<->test pairing gate (CLAUDE.md non-negotiable #10) was itself entirely
    untested, and nobody saw it because Actions is disabled on this repo.

    These tests target the current API: ``_build_shapes_corpus`` (glob over
    ``TESTS_ROOT``), ``_is_covered`` (name-mention lookup) and ``main``.
    """

    @pytest.fixture
    def fake_tests_tree(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        tests = tmp_path / "tests" / "unit"
        tests.mkdir(parents=True)
        (tests / "test_alpha_model_shapes.py").write_text(
            "def test_alpha_shape():\n    name = 'alpha_model'\n"
        )
        # A registry-wide parametrised shapes test that merely MENTIONS the name.
        (tests / "test_loss_registry_shapes.py").write_text(
            "def test_loss_registry():\n    name = 'beta_loss'\n"
        )
        # Names gamma_metric, but is NOT a shapes file — must not count as coverage.
        (tests / "test_unrelated.py").write_text("def test_x():\n    name = 'gamma_metric'\n")

        mod = _load("shape_coverage_report_test", "shape_coverage_report.py")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(mod, "TESTS_ROOT", tmp_path / "tests")

        return mod, tmp_path

    def test_corpus_collects_only_shapes_files(self, fake_tests_tree) -> None:
        mod, _ = fake_tests_tree
        corpus = mod._build_shapes_corpus()
        assert sorted(p.name for p, _ in corpus) == [
            "test_alpha_model_shapes.py",
            "test_loss_registry_shapes.py",
        ]

    def test_is_covered_direct_match(self, fake_tests_tree) -> None:
        mod, _ = fake_tests_tree
        covered, hit = mod._is_covered("alpha_model", mod._build_shapes_corpus())
        assert covered
        assert hit is not None and hit.name == "test_alpha_model_shapes.py"

    def test_is_covered_via_parametrised_registry_test(self, fake_tests_tree) -> None:
        """A registry-wide shapes test that mentions the name counts as coverage."""
        mod, _ = fake_tests_tree
        covered, hit = mod._is_covered("beta_loss", mod._build_shapes_corpus())
        assert covered
        assert hit is not None and "loss_registry_shapes" in hit.name

    def test_mention_in_a_non_shapes_file_is_not_coverage(self, fake_tests_tree) -> None:
        """``gamma_metric`` appears only in ``test_unrelated.py``.

        The corpus glob is what makes the number mean "has a SHAPE-contract test"
        rather than "is mentioned anywhere in tests/". If this ever passes, the
        report is silently over-counting coverage.
        """
        mod, _ = fake_tests_tree
        covered, hit = mod._is_covered("gamma_metric", mod._build_shapes_corpus())
        assert not covered
        assert hit is None

    def test_main_returns_zero_it_is_a_report_not_a_gate(self, fake_tests_tree) -> None:
        mod, _ = fake_tests_tree
        assert mod.main([]) == 0

    def test_csv_header_matches_the_writer(self, fake_tests_tree, capsys) -> None:
        mod, _ = fake_tests_tree
        rc = mod.main(["--csv"])
        assert rc == 0
        header = capsys.readouterr().out.splitlines()[0]
        assert header == "registry,name,has_shapes_test,test_file"
