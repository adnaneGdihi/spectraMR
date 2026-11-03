"""Tests for the debug-snapshot audit (scripts/diagnostics/debug_snapshot_audit.py).

Every finding code ships with a **planted** manifest that turns it red, and -- for
S2, whose whole difficulty is not over-firing -- planted clean manifests that must
stay green (non-negotiable 15). S2 compares the model input the forward hooks
recorded with ``input_raw``; a strategy that degrades inside the step feeds a
different tensor, so it no longer needs an exemption.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from tests.utils.repo_scripts import require_repo_file

_SCRIPT_REL = "scripts/diagnostics/debug_snapshot_audit.py"


def _load():
    script = require_repo_file(_SCRIPT_REL)
    spec = importlib.util.spec_from_file_location("debug_snapshot_audit", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tensor(key: str, *, std: float = 1.0, nan: int = 0, inf: int = 0, mean: float = 0.0):
    return {
        "key": key,
        "shape": [1, 8, 16, 16],
        "dtype": "torch.float32",
        "nan_count": nan,
        "inf_count": inf,
        "min": -1.0,
        "max": 1.0,
        "mean": mean,
        "std": std,
        "png": [],
    }


def _write_manifest(
    arm: pathlib.Path, step: int = 1, *, run: str = "run-a", stages=None, provenance=None
):
    d = arm / "snapshots" / run / "train" / f"step_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": "T",
        "phase": "train",
        "step": step,
        "provenance": provenance or {"declared": {}, "applied": {}, "incomplete": []},
        "stages": stages if stages is not None else _stages(),
    }
    (d / "manifest.json").write_text(json.dumps(payload))
    return d


def _stages(*, model_std: float = 0.5, target_std: float = 1.0, nan: int = 0, model: bool = True):
    """Raw batch, prepared input and (by default) a model input DIFFERENT from raw."""
    stages = [
        {"group": "batch", "name": "input_raw", "tensors": [_tensor("input"), _tensor("target")]},
        {
            "group": "prepare",
            "name": "input_prepared",
            "tensors": [_tensor("input_prepared"), _tensor("target", std=target_std, nan=nan)],
        },
    ]
    if model:
        stages.append(
            {
                "group": "model",
                "name": "input[generator]",
                "tensors": [_tensor("arg.0", std=model_std)],
            }
        )
    return stages


def test_s1_fires_when_no_snapshot_exists(tmp_path):
    mod = _load()
    arm = tmp_path / "arm_without_snapshots"
    arm.mkdir()
    rec = mod.audit_arm(arm)
    assert "S1" in rec["codes"]
    assert rec["snapshot"] is None


def test_s1_fires_on_unreadable_manifest(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    d = _write_manifest(arm)
    (d / "manifest.json").write_text("{not json")
    assert "S1" in mod.audit_arm(arm)["codes"]


def test_s2_fires_when_the_model_was_fed_the_raw_input(tmp_path):
    """Planted violation: the model input is the loader's input, so a declared
    degradation never reached the model."""
    mod = _load()
    arm = tmp_path / "arm"
    _write_manifest(arm, stages=_stages(model_std=1.0))
    assert "S2" in mod.audit_arm(arm)["codes"]


def test_s2_is_silent_when_the_model_input_differs(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    _write_manifest(arm, stages=_stages(model_std=0.25))
    assert "S2" not in mod.audit_arm(arm)["codes"]


def _identical_with(arm, *, declared=None, transforms=None):
    provenance = {
        "declared": {} if declared is None else declared,
        "applied": {"transforms": transforms or []},
        "incomplete": [],
    }
    _write_manifest(arm, stages=_stages(model_std=1.0), provenance=provenance)


def test_s2_fires_when_a_degradation_is_declared_and_nothing_evidences_it(tmp_path):
    """Planted: undersampling declared, no masking transform applied, model input == raw."""
    mod = _load()
    arm = tmp_path / "arm"
    _identical_with(arm, declared={"acquisition": None, "undersampling": {"acceleration": 4}})
    assert "S2" in mod.audit_arm(arm)["codes"]


def test_s2_is_silent_when_no_degradation_is_declared(tmp_path):
    """Noise2Noise, repetition-pair and reconstruction-only arms: the 26 false positives."""
    mod = _load()
    arm = tmp_path / "arm"
    _identical_with(arm, declared={"acquisition": None, "undersampling": None})
    rec = mod.audit_arm(arm)
    assert "S2" not in rec["codes"]
    assert any("no degradation was declared" in n for n in rec["notes"])


def test_s2_is_silent_for_a_populated_undersampling_block_that_masks_nothing(tmp_path):
    """The three VAEs: the block is present with base == max == 1.0, i.e. full sampling."""
    mod = _load()
    arm = tmp_path / "arm"
    _identical_with(
        arm,
        declared={
            "acquisition": None,
            "undersampling": {
                "base_acceleration": 1.0,
                "max_acceleration": 1.0,
                "center_fraction": 0.08,
            },
        },
    )
    assert "S2" not in mod.audit_arm(arm)["codes"]


def test_s2_fires_for_an_acceleration_ladder_that_starts_at_one(tmp_path):
    """Planted: base 1.0 but max 8.0 still asks for degraded inputs later in the run."""
    mod = _load()
    arm = tmp_path / "arm"
    _identical_with(
        arm,
        declared={
            "acquisition": None,
            "undersampling": {"base_acceleration": 1.0, "max_acceleration": 8.0},
        },
    )
    assert "S2" in mod.audit_arm(arm)["codes"]


def test_s2_is_silent_when_a_declared_degradation_is_applied_by_a_masking_transform(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    _identical_with(
        arm,
        declared={"acquisition": None, "undersampling": {"acceleration": 2}},
        transforms=["EnsureSpatialConsistency", "ComplementaryRepMasking"],
    )
    rec = mod.audit_arm(arm)
    assert "S2" not in rec["codes"]
    assert any("ComplementaryRepMasking" in n for n in rec["notes"])


def test_s2_still_fires_when_the_record_does_not_say_what_was_declared(tmp_path):
    """Absent is not 'none declared': an older snapshot without the keys keeps its S2."""
    mod = _load()
    arm = tmp_path / "arm"
    _identical_with(arm, declared={"target_mode": "r2r"})
    assert "S2" in mod.audit_arm(arm)["codes"]


@pytest.mark.parametrize("field", ["nan_count", "inf_count"])
def test_s3_fires_on_non_finite_values(tmp_path, field):
    mod = _load()
    arm = tmp_path / "arm"
    stages = _stages()
    stages[1]["tensors"][0][field] = 7
    _write_manifest(arm, stages=stages)
    rec = mod.audit_arm(arm)
    assert "S3" in rec["codes"]
    assert any("input_prepared" in n for n in rec["notes"])


def test_s4_fires_on_unresolved_provenance(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    _write_manifest(
        arm, provenance={"declared": {}, "applied": {}, "incomplete": ["dataset_chain unresolved"]}
    )
    rec = mod.audit_arm(arm)
    assert "S4" in rec["codes"]
    assert any("dataset_chain" in n for n in rec["notes"])


def test_s5_fires_when_no_model_call_was_recorded(tmp_path):
    """A strategy running a model outside ``env.models`` leaves no model stage."""
    mod = _load()
    arm = tmp_path / "arm"
    _write_manifest(arm, stages=_stages(model=False))
    rec = mod.audit_arm(arm)
    assert "S5" in rec["codes"]
    assert "S2" not in rec["codes"]


def test_s5_is_silent_when_the_model_input_is_recorded(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    _write_manifest(arm)
    assert "S5" not in mod.audit_arm(arm)["codes"]


def test_s6_fires_on_a_constant_target(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    _write_manifest(arm, stages=_stages(target_std=0))
    assert "S6" in mod.audit_arm(arm)["codes"]


def test_s7_fires_when_the_snapshot_predates_the_run_window(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    d = _write_manifest(arm)
    mtime = (d / "manifest.json").stat().st_mtime
    assert "S7" in mod.audit_arm(arm, since=mtime + 60)["codes"]
    assert "S7" not in mod.audit_arm(arm, since=mtime - 60)["codes"]


def test_newest_step_of_the_newest_run_wins(tmp_path):
    """Steps are zero-padded, so the newest is the highest number; an older run's
    higher step must not win over the newest run."""
    import os

    mod = _load()
    arm = tmp_path / "arm"
    old = _write_manifest(arm, step=40, run="run-old")
    os.utime(old / "manifest.json", (1, 1))
    _write_manifest(arm, step=1, run="run-new")
    _write_manifest(arm, step=12, run="run-new")
    got = mod.newest_manifest(arm)
    assert got.parent.name == "step_000012" and got.parents[2].name == "run-new"


def test_panel_paths_resolve_the_manifest_pngs(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    stages = _stages()
    d = _write_manifest(arm, stages=stages)
    root = d.parents[1]
    for stage in stages:
        for t in stage["tensors"]:
            rel = f"train/{d.name}/{stage['name']}__{t['key']}.png"
            (root / rel).write_bytes(b"")
            t["png"] = [rel]
    _write_manifest(arm, stages=stages)
    names = [n for n, _ in mod.panel_paths(mod.audit_arm(arm))]
    assert names == list(mod.CANONICAL_KEYS)


def test_parse_since_accepts_the_compilers_window_format():
    mod = _load()
    assert mod.parse_since("2026-09-20 08:41:42") == pytest.approx(
        mod.parse_since("2026-09-20 08:41") + 42
    )
    assert mod.parse_since(None) is None
    assert mod.parse_since("none") is None
    with pytest.raises(SystemExit):
        mod.parse_since("last tuesday")
