"""The gate that decides whether a re-run test pass is trusted."""

from __future__ import annotations

import json
from pathlib import Path

from spectramr.infrastructure.orchestration.inference_parity import check_arm


def _write(run: Path, payload: dict, *, inference: bool) -> None:
    target = run / "inference" if inference else run
    target.mkdir(parents=True, exist_ok=True)
    (target / "final_eval.json").write_text(json.dumps(payload))


BASE = {"psnr": {"s1/T1/0": 25.0, "s1/T1/1": 26.0}, "mse": {"s1/T1/0": 0.002, "s1/T1/1": 0.003}}


def test_identical_rows_pass(tmp_path):
    _write(tmp_path, BASE, inference=False)
    _write(tmp_path, BASE, inference=True)
    assert check_arm(tmp_path).ok


def test_planted_value_drift_fails_and_names_the_metric(tmp_path):
    _write(tmp_path, BASE, inference=False)
    _write(tmp_path, {**BASE, "psnr": {"s1/T1/0": 25.0, "s1/T1/1": 26.5}}, inference=True)
    res = check_arm(tmp_path)
    assert not res.ok and "psnr" in res.reason and res.metrics["psnr"][1] == 0.5


def test_planted_missing_case_fails(tmp_path):
    _write(tmp_path, BASE, inference=False)
    _write(tmp_path, {**BASE, "mse": {"s1/T1/0": 0.002}}, inference=True)
    assert "case ids differ" in check_arm(tmp_path).reason


def test_planted_missing_metric_fails(tmp_path):
    _write(tmp_path, BASE, inference=False)
    _write(tmp_path, {"psnr": BASE["psnr"]}, inference=True)
    assert "metric sets differ" in check_arm(tmp_path).reason


def test_a_missing_rerun_fails(tmp_path):
    _write(tmp_path, BASE, inference=False)
    assert "missing" in check_arm(tmp_path).reason


def test_scalar_payloads_compare_as_one_case(tmp_path):
    _write(tmp_path, {"psnr": 25.0}, inference=False)
    _write(tmp_path, {"psnr": 25.0}, inference=True)
    assert check_arm(tmp_path).ok


def test_the_verdict_is_written_and_read_back(tmp_path):
    from spectramr.core.run_artifacts import read_parity, write_parity

    (tmp_path / "inference").mkdir()
    assert read_parity(tmp_path) == "unchecked"
    write_parity(tmp_path, ok=False, reason="psnr", rtol=1e-3, atol=1e-6)
    assert read_parity(tmp_path) == "mismatch"
    write_parity(tmp_path, ok=True, reason=None, rtol=1e-3, atol=1e-6)
    assert read_parity(tmp_path) == "ok"
