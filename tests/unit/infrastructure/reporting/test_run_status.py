"""The one reader of ``run_summary.json`` ``success`` (#2525)."""

import json
import logging

import pytest

from spectramr.infrastructure.reporting.run_status import (
    RunStatus,
    log_status,
    read_run_status,
    status_notes,
)

CRASH = {
    "success": False,
    "error_type": "RuntimeError",
    "error": 'Traceback (most recent call last):\n  File "x"\nRuntimeError: CUDA error',
    "best_metrics": None,
    "duration_sec": None,
    "model_params": None,
}


def _write(run, payload, where="run_summary.json"):
    path = run / where
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return run


def test_success_true_is_ok(tmp_path):
    assert read_run_status(_write(tmp_path, {"success": True})) == RunStatus("ok")


def test_a_crash_summary_is_failed_with_its_error_type(tmp_path):
    status = read_run_status(_write(tmp_path, CRASH))
    assert status.state == "failed"
    assert status.error_type == "RuntimeError"
    assert status.label == "crashed: RuntimeError"
    assert status.error_line == "RuntimeError: CUDA error"


def test_a_crash_without_an_error_type_is_still_failed(tmp_path):
    status = read_run_status(_write(tmp_path, {"success": False}))
    assert status.state == "failed"
    assert "not recorded" in status.label


def test_the_summary_under_logs_is_read(tmp_path):
    assert read_run_status(_write(tmp_path, CRASH, "logs/run_summary.json")).state == "failed"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (None, "no run_summary.json"),
        ({"seed": 1}, "no success flag"),
        ({"success": None}, "null"),
        ("{not json", "unreadable"),
        ([1, 2], "not a JSON object"),
    ],
)
def test_missing_evidence_is_unknown_never_ok(tmp_path, payload, reason):
    run = tmp_path if payload is None else _write(tmp_path, payload)
    status = read_run_status(run)
    assert status.state == "unknown"
    assert reason in status.reason
    assert status.label.startswith("status unknown")


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_a_non_boolean_success_raises(tmp_path, value):
    with pytest.raises(ValueError, match="must be true, false or null"):
        read_run_status(_write(tmp_path, {"success": value}))


def test_status_notes_name_only_runs_that_are_not_ok():
    notes = status_notes({"a": RunStatus("ok"), "b": RunStatus("unknown", "no run_summary.json")})
    assert notes == {"b": "status unknown: no run_summary.json"}


def test_the_log_line_carries_the_error_type_but_never_the_traceback(caplog):
    status = RunStatus("failed", error_type="RuntimeError", error=CRASH["error"])
    with caplog.at_level(logging.INFO):
        log_status("crashed", status, action="excluded")
        log_status("fine", RunStatus("ok"), action="excluded")
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "crashed: RuntimeError" in caplog.text
    assert "Traceback" not in caplog.text


def test_to_dict_carries_no_traceback():
    d = RunStatus("failed", "x", error_type="RuntimeError", error=CRASH["error"]).to_dict()
    assert d == {
        "status": "failed",
        "reason": "x",
        "error_type": "RuntimeError",
        "error": "RuntimeError: CUDA error",
    }
