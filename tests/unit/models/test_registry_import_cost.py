"""Populating the model registry must not import heavy optional packages.

Those packages are needed only when a specific model is *built*, so pulling
them in during the registry walk costs seconds of startup for every command.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests.unit.models.test_registry_census import GLOBAL_FLOOR

pytestmark = pytest.mark.unit

HEAVY = ("timm", "diffusers", "transformers", "matplotlib.pyplot", "monai")

_PROBE = """
import json, sys, warnings
warnings.simplefilter("ignore")
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY
populate_model_registry()
print(json.dumps({"n": len(MODEL_REGISTRY), "loaded": [m for m in %r if m in sys.modules]}))
"""


@pytest.fixture(scope="module")
def probe() -> dict:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE % (list(HEAVY),)],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", HEAVY)
def test_heavy_module_not_imported_by_registry_walk(probe: dict, module: str) -> None:
    assert module not in probe["loaded"]


def test_registered_count_matches_in_process_walk(probe: dict) -> None:
    """The subprocess walk registers exactly what the census fixture sees."""
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert probe["n"] == len(MODEL_REGISTRY)
    assert probe["n"] >= GLOBAL_FLOOR
