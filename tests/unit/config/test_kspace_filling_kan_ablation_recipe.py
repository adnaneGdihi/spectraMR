"""Every KAN dual-domain ablation keeps its parent's KAN parameter-group recipe.

``experiment_11_kan_dual_domain`` trains the KAN attention block at
``kan_lr_ratio: 0.1`` and ``kan_weight_decay_ratio: 0.1``. All eleven arms of
``ablations_kan_dual_domain/`` omitted both keys, so they defaulted to 1.0 and
trained that block at ten times the parent's learning rate. Each ablation then
differed from its parent on two axes, and the four ``no_*`` loss ablations
diverged near step 4000 on 2026-09-20.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3] / "experiments" / "inprogress" / "kspace_filling"
_PARENT = _ROOT / "experiment_11_kan_dual_domain.yaml"
_ABLATIONS = sorted((_ROOT / "ablations_kan_dual_domain").glob("*.yaml"))

#: Keys the ablations inherit from the parent unchanged; no ablation's axis is one of them.
INHERITED_KEYS: tuple[str, ...] = ("kan_lr_ratio", "kan_weight_decay_ratio")


def _model_kwargs(path: Path) -> dict:
    return yaml.safe_load(path.read_text())["model"]["model_kwargs"]


def recipe_drift(parent: dict, arm: dict, keys: tuple[str, ...] = INHERITED_KEYS) -> dict:
    """``{key: (parent value, arm value)}`` for every inherited key the arm does not match."""
    missing = object()
    return {
        k: (parent.get(k, missing), arm.get(k, missing))
        for k in keys
        if arm.get(k, missing) != parent.get(k, missing)
    }


def test_the_parent_declares_the_recipe() -> None:
    parent = _model_kwargs(_PARENT)
    assert all(k in parent for k in INHERITED_KEYS)


def test_the_directory_holds_the_ablations() -> None:
    assert len(_ABLATIONS) >= 11


@pytest.mark.parametrize("arm", _ABLATIONS, ids=lambda p: p.stem)
def test_each_ablation_inherits_the_kan_recipe(arm: Path) -> None:
    drift = recipe_drift(_model_kwargs(_PARENT), _model_kwargs(arm))
    assert not drift, f"{arm.name} differs from its parent on {drift}"


def test_an_arm_that_omits_the_recipe_is_caught() -> None:
    """PLANTED: the shape all eleven arms had, keys absent and so defaulted."""
    parent = _model_kwargs(_PARENT)
    drift = recipe_drift(parent, {"attention_type": "kan_dual_domain"})
    assert set(drift) == set(INHERITED_KEYS)


def test_an_arm_that_overrides_the_recipe_is_caught() -> None:
    parent = _model_kwargs(_PARENT)
    drift = recipe_drift(parent, {**parent, "kan_lr_ratio": 1.0})
    assert set(drift) == {"kan_lr_ratio"}
