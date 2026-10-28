"""The kspace_filling standardizer never conforms an arm's model-selection keys.

``scripts/migrations/standardize_kspace_filling_cohort.py`` conforms every arm to
a reference arm except for the paths ``protected_for`` returns. #1977: the keys
that decide which checkpoint an arm keeps were not among them, so a run would
rewrite an arm's selection metric to the reference's under an unchanged name.

Each case plants a reference and an arm in ``tmp_path`` and calls ``plan``, the
function that queues the edits; the real cohort is never read or written. One
case per way a key can drift -- the arm's value differs (SET), the arm omits it
(INSERT), the reference omits it (DELETE) -- and an unprotected key that must
still be conformed, so the protection cannot pass by conforming nothing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]


def _load():
    path = REPO / "scripts/migrations/standardize_kspace_filling_cohort.py"
    spec = importlib.util.spec_from_file_location("standardize_kspace_filling_cohort", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Loaded per module rather than at import: the script needs ruamel.yaml, and a
# missing dev dependency should fail these tests, not abort the whole collection.
@pytest.fixture(scope="module")
def std():
    return _load()


REFERENCE = """\
metadata:
  name: reference
  primary_metric: hfen
early_stopping:
  metric: val_hfen_mean
  mode: min
metrics:
  best_metric_name: val_hfen_mean
  best_metric_mode: min
optimization:
  learning_rate: 0.0001
"""

# The arm selects on robust PSNR, declares its own scoring headline (absent from
# the reference) and omits metadata.primary_metric (present in the reference).
ARM = """\
metadata:
  name: arm
early_stopping:
  metric: val_robust_mri_psnr_mean
  mode: max
metrics:
  best_metric_name: val_robust_mri_psnr_mean
  best_metric_mode: max
validation:
  scoring:
    primary: psnr
optimization:
  learning_rate: 0.0003
"""

SELECTION_SET = [
    "early_stopping.metric",
    "early_stopping.mode",
    "metrics.best_metric_name",
    "metrics.best_metric_mode",
]


def _plan(std, tmp_path: Path) -> tuple[dict, list[str], dict]:
    ref_path = tmp_path / "reference.yaml"
    arm_path = tmp_path / "arm.yaml"
    ref_path.write_text(REFERENCE)
    arm_path.write_text(ARM)
    tpl = std.Arm(ref_path)
    arm = std.Arm(arm_path)
    intended, counts = std.plan(arm, tpl, "kspace_filling/arm", std.flat(tpl.doc))
    written = yaml.safe_load(arm.render())
    return intended, list(counts), std.flat(written)


@pytest.mark.parametrize("key", SELECTION_SET)
def test_a_differing_selection_key_is_not_set(std, tmp_path: Path, key: str) -> None:
    _intended, ops, written = _plan(std, tmp_path)
    assert f"set {key}" not in ops
    assert written[key] == std.flat(yaml.safe_load(ARM))[key]


def test_a_missing_primary_metric_is_not_inserted(std, tmp_path: Path) -> None:
    _intended, ops, written = _plan(std, tmp_path)
    assert not any(op.endswith("metadata.primary_metric") for op in ops)
    assert "metadata.primary_metric" not in written


def test_a_differing_primary_metric_is_not_set(std, tmp_path: Path) -> None:
    ref_path = tmp_path / "reference.yaml"
    arm_path = tmp_path / "arm.yaml"
    ref_path.write_text(REFERENCE)
    arm_path.write_text(ARM.replace("  name: arm\n", "  name: arm\n  primary_metric: psnr\n"))
    tpl = std.Arm(ref_path)
    arm = std.Arm(arm_path)
    _intended, counts = std.plan(arm, tpl, "kspace_filling/arm", std.flat(tpl.doc))
    assert "set metadata.primary_metric" not in counts
    assert yaml.safe_load(arm.render())["metadata"]["primary_metric"] == "psnr"


def test_a_scoring_headline_the_reference_lacks_is_not_deleted(std, tmp_path: Path) -> None:
    _intended, ops, written = _plan(std, tmp_path)
    assert not any("validation" in op for op in ops)
    assert written["validation.scoring.primary"] == "psnr"


def test_an_unprotected_key_is_still_conformed(std, tmp_path: Path) -> None:
    """Control: the protection is specific, not a standardizer that does nothing."""
    _intended, ops, written = _plan(std, tmp_path)
    assert "set optimization.learning_rate" in ops
    assert written["optimization.learning_rate"] == 0.0001


def test_every_selection_key_is_protected_on_every_planned_arm(std) -> None:
    """Protection holds for an unlisted arm and for every arm with an AXIS row."""
    for rel in ["kspace_filling/arm", *std.AXIS]:
        exact, prefixes = std.protected_for(rel)
        for key in std.MODEL_SELECTION:
            assert std.is_protected(key, exact, prefixes), (rel, key)


def test_the_excluded_arm_paths_cannot_touch_a_selection_key(std) -> None:
    """EXCLUDED arms bypass ``plan`` and ``protected_for``: their only edits are the
    HYGIENE_ONLY deletes and EXCLUDED_ALSO_CONFORM sets, so neither may name one."""
    for listed in (*std.HYGIENE_ONLY, *std.EXCLUDED_ALSO_CONFORM):
        for key in std.MODEL_SELECTION:
            assert not (listed == key or key.startswith(f"{listed}.")), (listed, key)
