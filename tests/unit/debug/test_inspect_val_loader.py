"""Staleness ratchet for ``tests/debug/inspect_val_loader.py`` (#1280).

That script is not collected by pytest (``python_files`` is ``test_*.py``), so
nothing ever executed it and it rotted on two independent axes at once: it named
an experiment YAML from before the ``inprogress/<paradigm>/`` reorg, and it read
``config.data.validation_split``, a spelling retired to
``data.split.validation_fraction``. Both failures printed and returned ``0``.

These tests are the gate that was missing. They are deliberately *structural*
rather than source-text greps: the module declares the config paths it reads in
``INSPECTED_CONFIG_PATHS`` and its default arm in ``DEFAULT_CONFIG``, and the
tests resolve both against the live schema and the live ``RENAMES`` table. So
they go red on the NEXT rename or reorg too, not only on the ones already fixed.

No dataset, no model, no forward pass -- import + schema walk + an argparse call
on a path that does not exist.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import typing
from pathlib import Path

import pytest

from spectramr.config.schemas.renames import RENAMES
from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.directors.data_pipeline_director import (
    DataPipelineDirector,
)
from tests.utils.repo_scripts import require_repo_file

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tests" / "debug" / "inspect_val_loader.py"


def _import_script() -> types.ModuleType:
    """Import the debug script by path; it is not on any package path."""
    spec = importlib.util.spec_from_file_location("inspect_val_loader_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script() -> types.ModuleType:
    return _import_script()


def test_script_exists() -> None:
    assert SCRIPT.is_file(), f"debug script missing: {SCRIPT}"


def test_import_does_not_run_the_inspection(script: types.ModuleType) -> None:
    """Importing must be side-effect free -- the work is behind ``__main__``."""
    assert callable(script.main)
    assert callable(script.inspect_validation_data)


def test_default_config_exists(script: types.ModuleType) -> None:
    """The #1280 path defect: a default arm that no longer exists on disk."""
    default = script.DEFAULT_CONFIG
    # require_repo_file, not `.is_file()`: this script ships in the public export
    # while `experiments/` does not, so there the arm's absence is the publication
    # boundary and not #1280. In every other tree a default that does not resolve
    # is still exactly the defect this test was written for, and still fails loud.
    assert require_repo_file(str(default.relative_to(REPO))).is_file(), (
        f"DEFAULT_CONFIG points at a file that does not exist: {default}. "
        "The experiments tree was reorganised; repoint it at a live arm."
    )


def test_default_config_is_under_inprogress(script: types.ModuleType) -> None:
    """Keep the default on the tree this repo actually develops (NN11)."""
    rel = script.DEFAULT_CONFIG.relative_to(REPO)
    assert rel.parts[:2] == ("experiments", "inprogress"), rel


def test_inspected_paths_are_not_retired_spellings(script: types.ModuleType) -> None:
    """The #1280 rename defect, pinned against the live RENAMES table.

    ``data.validation_split`` is in ``RENAMES``; reading it raises. Asserting
    against the table rather than against the one known-bad string means the
    next retirement fails here too.
    """
    retired = [p for p in script.INSPECTED_CONFIG_PATHS if p in RENAMES]
    assert not retired, (
        f"inspect_val_loader reads retired config spellings: {retired}. "
        + "; ".join(f"{p} -> {RENAMES[p].canonical}" for p in retired)
    )


def _resolve_schema_path(model: type, dotted: str) -> bool:
    """True if ``dotted`` resolves through nested pydantic models from ``model``."""
    node: typing.Any = model
    for part in dotted.split("."):
        fields = getattr(node, "model_fields", None)
        if not fields or part not in fields:
            return False
        ann = fields[part].annotation
        # Unwrap Optional[X] / X | None to the first non-None member.
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        node = args[0] if args else ann
    return True


@pytest.mark.parametrize("dotted", ["data.split.validation_fraction"])
def test_inspected_paths_resolve_on_the_schema(dotted: str) -> None:
    """Not-retired is necessary but not sufficient -- the path must also exist."""
    assert _resolve_schema_path(TrainingSettings, dotted), (
        f"{dotted} does not resolve on TrainingSettings"
    )


def test_declared_paths_match_the_parametrised_set(script: types.ModuleType) -> None:
    """Guard the test above from going vacuous if the script adds a path."""
    assert set(script.INSPECTED_CONFIG_PATHS) == {"data.split.validation_fraction"}, (
        "INSPECTED_CONFIG_PATHS changed; extend "
        "test_inspected_paths_resolve_on_the_schema's parametrisation to match."
    )


def test_routes_through_the_maintained_director(script: types.ModuleType) -> None:
    """The script must not re-implement the director's build order (NN17).

    An identity check, not a source grep: it stays true if the import is moved
    or aliased, and goes red only if the binding stops being the real director.
    """
    assert script.DataPipelineDirector is DataPipelineDirector


def test_missing_config_exits_nonzero(script: types.ModuleType, tmp_path: Path) -> None:
    """The silent-failure defect: every failure used to print and return 0."""
    missing = tmp_path / "does_not_exist.yaml"
    assert script.main(["--config", str(missing)]) == 1


def test_src_is_on_sys_path_after_import(script: types.ModuleType) -> None:
    """The old ``sys.path.append(os.getcwd())`` never made ``spectramr`` importable."""
    assert str(REPO / "src") in sys.path
