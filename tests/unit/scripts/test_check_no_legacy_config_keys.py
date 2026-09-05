"""The corpus gate must find a retired key only where the rename says it lives.

The gate originally matched the *leaf* key at any indentation, with a comment
claiming it confirmed the owning block afterwards. It did not. That was invisible
while the only rename was ``optimization.accumulate_grad_steps`` — a leaf name
that appears nowhere else — and became an 836-hit false-positive storm across 636
configs the moment a rename named ``name``, which every loss-list entry and every
metadata block also uses.

The fix is structural rather than a better regex: the gate now runs the *fixer*
with ``apply=False``. One code path means the check cannot claim a key the fixer
would not touch, nor miss one it would.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from spectramr.config.schemas.renames import RenameRecord
from tests.utils.repo_scripts import require_repo_file

_CI = Path(__file__).resolve().parents[3] / "scripts" / "ci"

#: The gate ships; this does not. `_version_status` reaches it per scanned file
#: through `_version_migrator()`, which execs it by path, so the failure surfaces
#: as FileNotFoundError from inside a shipped script rather than at import.
_VERSION_MIGRATOR = "scripts/migrations/migrate_config_version_to_v1.py"


@pytest.fixture
def version_migrator() -> None:
    """Require the migrator the gate classifies versions with.

    Marked on the tests that actually reach it, not on `_with_table`: a scan that
    never classifies a version (TestScope's v5 config is skipped before the
    classifier runs) has no migrator dependency and must keep running in the
    export.
    """
    require_repo_file(_VERSION_MIGRATOR)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", _CI / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gate():
    return _load("check_no_legacy_config_keys")


def _with_table(gate, records: list[RenameRecord]):
    """Point BOTH the gate and the fixer it loads at ``records``."""
    table = {r.legacy: r for r in records}
    gate.RENAMES = table
    original = gate._load_fixer

    def patched():
        mod = original()
        mod.RENAMES = table
        return mod

    gate._load_fixer = patched


_ARM = """\
config_version: '1.0'

workflow:
  name: mri_structural
  task: reconstruction

metadata:
  name: an arm whose metadata block also has a name

losses:
  image_losses:
    - name: l1
      weight: 1.0
"""


@pytest.mark.usefixtures("version_migrator")
class TestBlockAwareness:
    def test_flags_the_key_only_under_its_declared_block(self, gate, tmp_path, capsys):
        (tmp_path / "arm.yaml").write_text(_ARM)
        _with_table(
            gate,
            [
                RenameRecord(
                    legacy="workflow.name",
                    canonical="workflow.regime",
                    since="2026-07-31",
                    reason="`name` reads like a label; this is the physical regime.",
                )
            ],
        )
        assert gate.main([str(tmp_path)]) == 1
        out = capsys.readouterr().out
        assert "1 retired key(s)" in out, (
            "the gate matched `name:` outside `workflow:` — it is scanning leaf "
            f"names again, not paths.\n{out}"
        )
        assert "workflow.name -> workflow.regime" in out

    def test_a_file_without_the_owning_block_is_clean(self, gate, tmp_path, capsys):
        """`name:` under `metadata:` and inside a loss list is not the retired key."""
        (tmp_path / "arm.yaml").write_text(
            "config_version: '1.0'\n"
            "metadata:\n"
            "  name: no workflow block here\n"
            "losses:\n"
            "  image_losses:\n"
            "    - name: l1\n"
        )
        _with_table(
            gate,
            [
                RenameRecord(
                    legacy="workflow.name",
                    canonical="workflow.regime",
                    since="2026-07-31",
                    reason="x",
                )
            ],
        )
        assert gate.main([str(tmp_path)]) == 0
        assert "0 retired key(s)" in capsys.readouterr().out


class TestScope:
    def test_a_v5_config_is_not_scanned(self, gate, tmp_path, capsys):
        """v5.0 is rejected by `validate_config_version` before the schema is
        built, so a retired key there cannot break a run."""
        (tmp_path / "old.yaml").write_text(
            "config_version: '5.0'\nworkflow:\n  name: mri_structural\n"
        )
        _with_table(
            gate,
            [
                RenameRecord(
                    legacy="workflow.name",
                    canonical="workflow.regime",
                    since="2026-07-31",
                    reason="x",
                )
            ],
        )
        assert gate.main([str(tmp_path)]) == 0
        assert "scanned 0 loadable config(s)" in capsys.readouterr().out


@pytest.mark.usefixtures("version_migrator")
class TestRefusalsAreReportedSeparately:
    def test_a_disagreement_is_not_offered_as_auto_fixable(self, gate, tmp_path, capsys):
        """Both spellings with different values needs a human, so printing the
        fixer command against it would send the author in a circle."""
        (tmp_path / "arm.yaml").write_text(
            "config_version: '1.0'\nworkflow:\n  name: mri_structural\n  regime: mri_quantitative\n"
        )
        _with_table(
            gate,
            [
                RenameRecord(
                    legacy="workflow.name",
                    canonical="workflow.regime",
                    since="2026-07-31",
                    reason="x",
                )
            ],
        )
        assert gate.main([str(tmp_path)]) == 1
        out = capsys.readouterr().out
        assert "REFUSES" in out and "decide which the arm meant" in out
        assert "the fixer can rewrite" not in out


@pytest.mark.usefixtures("version_migrator")
class TestTheCheckIsTheFixer:
    def test_gate_reports_exactly_what_the_fixer_would_change(self, gate, tmp_path):
        """The whole point of delegating: run both and compare."""
        (tmp_path / "arm.yaml").write_text(_ARM)
        records = [
            RenameRecord(
                legacy="workflow.name",
                canonical="workflow.regime",
                since="2026-07-31",
                reason="x",
            )
        ]
        _with_table(gate, records)
        fixer = gate._load_fixer()
        direct = fixer.migrate_file(tmp_path / "arm.yaml", apply=False)
        assert len(direct) == 1 and "MIGRATED" in direct[0]
        assert gate.main([str(tmp_path)]) == 1


@pytest.mark.usefixtures("version_migrator")
class TestLegacyVersionCountdown:
    """The gate also counts LEGACY SCHEMA VERSIONS, next to the staged keys.

    The schema version is staged exactly like a key: 6.0/6.1 still load because
    the loader folds them to 1.0, and the fold is deleted when the count reaches
    zero. It is reported by this gate rather than by the version migrator alone
    so one command shows the whole ratchet -- a countdown kept somewhere else is
    a countdown nobody runs, and "staged" quietly becomes "permanent".

    The classifier is IMPORTED from the migrator, never re-implemented: a second
    copy of "is this version legacy?" is how a countdown and its fixer come to
    disagree, which is the defect class this gate exists to catch.
    """

    def test_the_gate_reuses_the_migrators_classifier(self, gate) -> None:
        """The classifier is imported, not re-implemented -- so when the legacy
        tier emptied on 2026-08-08 the gate followed the migrator automatically.

        A retired version now files as `unloadable` rather than `legacy`, which
        is the honest answer: the loader refuses it. Had the gate carried its
        own copy of "is this version legacy?", it would still be counting these
        toward a countdown that can never reach zero.
        """
        from spectramr.config.schemas.base import LEGACY_CONFIG_VERSIONS

        assert frozenset() == LEGACY_CONFIG_VERSIONS
        assert gate._version_status("config_version: '6.0'\n") == (
            "unloadable",
            "6.0",
        )

    def test_canonical_is_not_counted_as_legacy(self, gate) -> None:
        from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION

        status, _ = gate._version_status(f"config_version: '{CANONICAL_CONFIG_VERSION}'\n")
        assert status == "canonical"

    def test_the_migrator_module_is_loaded_once(self, gate) -> None:
        """Cached on purpose: this feeds a per-file call over ~830 configs, and
        re-execing a module per file is the shape of the 139k-parse waste that
        made an earlier corpus sweep unusable."""
        assert gate._version_migrator() is gate._version_migrator()


class TestPerRecordCountdown:
    """The promotion rule is PER RECORD; the gate only printed one aggregate.

    ``schemas/renames.py`` says: "When a fold record's count reaches zero, flip
    ``posture`` to ``raise``." But the gate printed a single total across all
    fold records (29,250 at the time of writing), so nobody could see WHICH of
    the 161 records were promotable without re-deriving the whole scan by hand.
    A ratchet whose next step is invisible does not get taken.

    The unreachable-key blocker was likewise global -- "NO record may be
    promoted until this is 0" -- while the 108 unreachable declarations belong
    to 8 records that each already carry 700+ staged hits. Globally that is a
    false blocker for every record that is genuinely drained; per record it is
    exactly right, because a record with unreachable declarations still has a
    countdown the line-scanner cannot see.
    """

    @pytest.mark.usefixtures("version_migrator")
    def test_per_record_counts_are_reported(self, gate, tmp_path, capsys) -> None:
        rec = RenameRecord(
            legacy="losses.output_domain",
            canonical="losses.policy.output_domain",
            since="2026-08-08",
            reason="test record.",
            posture="fold",
        )
        _with_table(gate, [rec])
        (tmp_path / "arm.yaml").write_text(
            "config_version: '1.0'\nlosses:\n  output_domain: kspace\n"
        )
        gate.main([str(tmp_path)])
        out = capsys.readouterr().out
        assert "losses.output_domain" in out, (
            "the countdown must name the record, not just report a total"
        )

    @pytest.mark.usefixtures("version_migrator")
    def test_a_drained_record_is_listed_as_promotable(self, gate, tmp_path, capsys) -> None:
        """Zero staged AND zero unreachable is the promotion condition."""
        rec = RenameRecord(
            legacy="losses.output_domain",
            canonical="losses.policy.output_domain",
            since="2026-08-08",
            reason="test record.",
            posture="fold",
        )
        _with_table(gate, [rec])
        (tmp_path / "arm.yaml").write_text(
            "config_version: '1.0'\nlosses:\n  policy:\n    output_domain: kspace\n"
        )
        gate.main([str(tmp_path)])
        out = capsys.readouterr().out
        assert "promotable" in out.lower()
        assert "losses.output_domain" in out

    def test_the_corpus_is_enumerated_from_git_not_an_on_disk_glob(self, gate) -> None:
        """An on-disk glob has a different subject on every machine.

        Cluster job 8004252 reported failures against two arms that exist in no
        git history on any branch -- generated output, never added, still
        resident on the cluster's working tree. A countdown that GATES A
        PROMOTION must not vary between the cluster and a dev box: the whole
        point is deciding whether a record is safe to flip to `raise`.

        ``CLAUDE.md`` already names the authority in the census it publishes:
        "committed files via ``git ls-files``, not an on-disk glob".
        """
        import inspect

        src = inspect.getsource(gate.main)
        assert "rglob" not in src, "main() must not enumerate the corpus with an on-disk glob"
        assert hasattr(gate, "_iter_corpus_yamls")

    def test_git_enumeration_excludes_an_untracked_file(self, gate, tmp_path) -> None:
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        tracked = tmp_path / "tracked.yaml"
        tracked.write_text("config_version: '1.0'\n")
        subprocess.run(["git", "add", "tracked.yaml"], cwd=tmp_path, check=True)
        (tmp_path / "untracked.yaml").write_text("config_version: '1.0'\n")

        found = {p.name for p in gate._iter_corpus_yamls(str(tmp_path))}
        assert "tracked.yaml" in found
        assert "untracked.yaml" not in found, (
            "a generated-but-never-added arm must not enter the countdown"
        )

    def test_a_non_git_root_announces_that_it_globbed_instead(self, gate, tmp_path, capsys) -> None:
        """Returning [] outside a checkout would report "scanned 0" as though
        the corpus were clean. Globbing is right there -- but it must SAY so,
        because the numbers then describe a different subject."""
        (tmp_path / "arm.yaml").write_text("config_version: '1.0'\n")
        found = gate._iter_corpus_yamls(str(tmp_path))
        assert [p.name for p in found] == ["arm.yaml"]
        assert "not inside a git checkout" in capsys.readouterr().out


class TestTheFixersDependenciesAreTheGates:
    """#976: ``ruamel.yaml`` went undeclared for the whole life of this gate.

    ``_load_fixer`` execs ``migrate_config_keys.py`` on purpose, so the two can
    never disagree about what a rename IS -- which also makes the fixer's
    imports the gate's runtime requirements. On a fresh clone the gate therefore
    died on ``ModuleNotFoundError: No module named 'ruamel'`` with a traceback
    ending in a file the caller never invoked, and hand-editing ``_DRAINED`` --
    which the gate's own docstring forbids -- was the only route left.
    """

    def test_a_missing_fixer_dependency_names_the_gate_and_the_fix(self, gate, monkeypatch) -> None:
        """The traceback pointed at the fixer; the reader invoked the gate."""
        import importlib.util

        real_exec = importlib.util.module_from_spec

        def explode(spec):
            mod = real_exec(spec)
            original_loader_exec = spec.loader.exec_module

            def failing(_mod):
                raise ModuleNotFoundError("No module named 'ruamel'", name="ruamel")

            spec.loader.exec_module = failing
            del original_loader_exec
            return mod

        monkeypatch.setattr(importlib.util, "module_from_spec", explode)

        with pytest.raises(ModuleNotFoundError) as excinfo:
            gate._load_fixer()

        message = str(excinfo.value)
        assert "check_no_legacy_config_keys.py cannot run" in message
        assert "ruamel" in message
        assert "pip install -e '.[dev]'" in message
        # The cause is preserved, so the original traceback is still reachable.
        assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)

    def test_ruamel_is_declared_in_the_dev_extra(self) -> None:
        """Anti-regression: the diagnostic is a consolation prize, not the fix.

        Derived from the scripts rather than hardcoded, so deleting the
        declaration while the imports remain fails here.
        """
        import tomllib

        repo = Path(__file__).resolve().parents[3]
        importers = [
            path
            for directory in ("scripts", "tools")
            for path in (repo / directory).rglob("*.py")
            if "ruamel" in path.read_text(encoding="utf-8", errors="ignore")
        ]
        assert importers, "precondition: something still imports ruamel.yaml"

        pyproject = tomllib.loads((repo / "pyproject.toml").read_text())
        dev = pyproject["project"]["optional-dependencies"]["dev"]
        assert any(spec.startswith("ruamel.yaml") for spec in dev), (
            f"{len(importers)} script(s) import ruamel.yaml but the dev extra "
            "does not declare it (#976)"
        )

    def test_src_does_not_import_ruamel(self) -> None:
        """Why it belongs in `dev` and not a runtime extra.

        If this ever fails, the dependency has become a runtime one and the
        declaration has to move -- a shipped wheel would be missing it.
        """
        repo = Path(__file__).resolve().parents[3]
        offenders = [
            path.relative_to(repo)
            for path in (repo / "src").rglob("*.py")
            if "ruamel" in path.read_text(encoding="utf-8", errors="ignore")
        ]
        assert not offenders, f"src/ imports ruamel.yaml: {offenders}"
