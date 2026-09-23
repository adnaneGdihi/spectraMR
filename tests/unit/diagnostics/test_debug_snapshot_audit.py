"""Tests for the debug-snapshot audit (scripts/diagnostics/debug_snapshot_audit.py).

Every finding code ships with a **planted** snapshot that turns it red, and --
for the two codes whose whole difficulty is not over-firing -- a planted clean
snapshot that must stay green (non-negotiable 15). The S2 pair is the reason
this file exists: an unconditional "input_prepared == input_raw" check fired on
67 of 67 arms in the downloaded corpus, because every one is a diffusion arm
that declares the in-step carve-out, where prepared == raw is the documented
correct result (.claude/rules/debug-snapshots.md). A detector that fires on the
whole corpus is not a detector.
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


def _tensor(name: str, *, std: float = 1.0, nan: int = 0, inf: int = 0, mean: float = 0.0):
    return {
        "name": name,
        "shape": [2, 8, 16, 16],
        "dtype": "torch.float32",
        "nan_count": nan,
        "inf_count": inf,
        "min": -1.0,
        "max": 1.0,
        "mean": mean,
        "std": std,
    }


def _write_snapshot(arm: pathlib.Path, tag: str, step: int = 1, **body) -> pathlib.Path:
    d = arm / "debug_snapshots" / f"{tag}_step_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "tag": tag,
        "paradigm": "T",
        "tensors": [],
        "extra": {},
        "provenance": {"declared": {}, "applied": {}, "incomplete": []},
    }
    payload.update(body)
    (d / "snapshot.json").write_text(json.dumps(payload))
    return d


def _canonical(*, prepared_std: float = 0.5, target_std: float = 1.0, nan: int = 0):
    """The three canonical keys, with prepared DIFFERENT from raw by default."""
    return [
        _tensor("input_raw", std=1.0),
        _tensor("input_prepared", std=prepared_std),
        _tensor("target", std=target_std, nan=nan),
    ]


def test_s1_fires_when_no_snapshot_exists(tmp_path):
    mod = _load()
    arm = tmp_path / "arm_without_snapshots"
    arm.mkdir()
    rec = mod.audit_arm(arm)
    assert "S1" in rec["codes"]
    assert rec["snapshot"] is None


def test_s1_fires_on_unreadable_snapshot_json(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    d = _write_snapshot(arm, "first_steps")
    (d / "snapshot.json").write_text("{not json")
    assert "S1" in mod.audit_arm(arm)["codes"]


def test_s2_fires_when_prepared_is_raw_and_no_carve_out(tmp_path):
    """Planted violation: identical stats, strategy claims prepared IS the model
    input -- so the declared degradation never reached the model."""
    mod = _load()
    arm = tmp_path / "arm"
    _write_snapshot(
        arm,
        "first_steps",
        tensors=_canonical(prepared_std=1.0),
        extra={"prepared_equals_model_input": True},
    )
    assert "S2" in mod.audit_arm(arm)["codes"]


def test_s2_is_silent_under_the_declared_carve_out(tmp_path):
    """The over-firing shape: identical stats are CORRECT when the strategy
    degrades the input inside the step and says so."""
    mod = _load()
    arm = tmp_path / "arm"
    _write_snapshot(
        arm,
        "first_steps",
        tensors=_canonical(prepared_std=1.0),
        extra={"prepared_equals_model_input": False, "model_input_snapshot_tag": "diffusion_step"},
    )
    _write_snapshot(arm, "diffusion_step", tensors=_canonical())
    rec = mod.audit_arm(arm)
    assert "S2" not in rec["codes"], rec["notes"]
    assert any("carve-out" in n for n in rec["notes"])


def test_s2_is_silent_when_prepared_genuinely_differs(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    _write_snapshot(
        arm,
        "first_steps",
        tensors=_canonical(prepared_std=0.25),
        extra={"prepared_equals_model_input": True},
    )
    assert "S2" not in mod.audit_arm(arm)["codes"]


@pytest.mark.parametrize("field", ["nan_count", "inf_count"])
def test_s3_fires_on_non_finite_values(tmp_path, field):
    mod = _load()
    arm = tmp_path / "arm"
    bad = _tensor("input_prepared", std=0.5)
    bad[field] = 7
    _write_snapshot(
        arm,
        "first_steps",
        tensors=[_tensor("input_raw"), bad, _tensor("target")],
        extra={"prepared_equals_model_input": True},
    )
    rec = mod.audit_arm(arm)
    assert "S3" in rec["codes"]
    assert any("input_prepared" in n for n in rec["notes"])


def test_s4_fires_on_unresolved_provenance(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    _write_snapshot(
        arm,
        "first_steps",
        tensors=_canonical(),
        extra={"prepared_equals_model_input": True},
        provenance={"declared": {}, "applied": {}, "incomplete": ["dataset_chain unresolved"]},
    )
    rec = mod.audit_arm(arm)
    assert "S4" in rec["codes"]
    assert any("dataset_chain" in n for n in rec["notes"])


def test_s5_fires_when_the_carve_out_keeps_no_record(tmp_path):
    """The carve-out's own obligation: name a tag AND emit that snapshot."""
    mod = _load()
    arm = tmp_path / "arm"
    _write_snapshot(
        arm,
        "first_steps",
        tensors=_canonical(prepared_std=1.0),
        extra={"prepared_equals_model_input": False, "model_input_snapshot_tag": "diffusion_step"},
    )
    rec = mod.audit_arm(arm)
    assert "S5" in rec["codes"]
    assert rec["model_input_snapshot"] is None


def test_s5_is_silent_when_the_named_snapshot_exists(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    _write_snapshot(
        arm,
        "first_steps",
        tensors=_canonical(prepared_std=1.0),
        extra={"prepared_equals_model_input": False, "model_input_snapshot_tag": "diffusion_step"},
    )
    _write_snapshot(arm, "diffusion_step", tensors=_canonical())
    rec = mod.audit_arm(arm)
    assert "S5" not in rec["codes"]
    assert rec["model_input_snapshot"] is not None


def test_s6_fires_on_a_constant_target(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    _write_snapshot(
        arm,
        "first_steps",
        tensors=_canonical(target_std=0),
        extra={"prepared_equals_model_input": True},
    )
    assert "S6" in mod.audit_arm(arm)["codes"]


def test_s7_fires_when_the_snapshot_predates_the_run_window(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    d = _write_snapshot(
        arm, "first_steps", tensors=_canonical(), extra={"prepared_equals_model_input": True}
    )
    mtime = (d / "snapshot.json").stat().st_mtime
    assert "S7" in mod.audit_arm(arm, since=mtime + 60)["codes"]
    assert "S7" not in mod.audit_arm(arm, since=mtime - 60)["codes"]


def test_newest_step_wins_not_lexicographic_first(tmp_path):
    """Steps are zero-padded, so the newest is the highest number -- pin it,
    because the validation-image picker got exactly this wrong."""
    mod = _load()
    arm = tmp_path / "arm"
    _write_snapshot(arm, "first_steps", step=1, tensors=_canonical())
    _write_snapshot(arm, "first_steps", step=12, tensors=_canonical())
    assert mod.newest_snapshot(arm, "first_steps").name.endswith("000012")


def test_panel_paths_include_the_carve_out_model_input(tmp_path):
    mod = _load()
    arm = tmp_path / "arm"
    first = _write_snapshot(
        arm,
        "first_steps",
        tensors=_canonical(prepared_std=1.0),
        extra={"prepared_equals_model_input": False, "model_input_snapshot_tag": "diffusion_step"},
    )
    alt = _write_snapshot(arm, "diffusion_step", tensors=_canonical())
    for key in mod.CANONICAL_KEYS:
        (first / f"{key}.png").write_bytes(b"")
    (alt / "model_input__kspace.png").write_bytes(b"")
    names = [n for n, _ in mod.panel_paths(mod.audit_arm(arm))]
    assert names == [*mod.CANONICAL_KEYS, "model_input"]


def test_parse_since_accepts_the_compilers_window_format():
    mod = _load()
    assert mod.parse_since("2026-09-20 08:41:42") == pytest.approx(
        mod.parse_since("2026-09-20 08:41") + 42
    )
    assert mod.parse_since(None) is None
    assert mod.parse_since("none") is None
    with pytest.raises(SystemExit):
        mod.parse_since("last tuesday")
