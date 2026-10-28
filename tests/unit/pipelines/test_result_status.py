"""One rule for "did the pipeline succeed", shared by every entry point.

Only an explicit ``True`` is success, and an absent flag is unknown, not a pass (#181).
"""

from __future__ import annotations

import sys

import pytest

from spectramr.pipelines.result_status import failure_reason

pytestmark = pytest.mark.unit


class TestFailureReason:
    def test_an_explicit_true_is_a_success(self) -> None:
        assert failure_reason({"success": True}) is None
        assert failure_reason({"success": True, "error": "ignored"}) is None

    def test_a_failed_result_names_its_error(self) -> None:
        assert failure_reason({"success": False, "error": "Resume failed: x"}) == "Resume failed: x"

    def test_a_failed_result_without_an_error_still_says_something(self) -> None:
        assert failure_reason({"success": False}) == "Unknown pipeline error"

    @pytest.mark.parametrize("flag", [None, 1, "true", "yes", [], 0])
    def test_anything_but_an_explicit_true_is_not_a_pass(self, flag) -> None:
        assert failure_reason({"success": flag}) is not None

    def test_an_absent_flag_is_unknown_not_a_pass(self) -> None:
        assert failure_reason({}) == "Unknown pipeline error"

    @pytest.mark.parametrize("result", [None, "done", 0, ["success"]])
    def test_a_non_dict_is_refused_with_its_type(self, result) -> None:
        reason = failure_reason(result)
        assert reason is not None and type(result).__name__ in reason


class TestPythonMDistributedEntryPoint:
    """``python -m spectramr.pipelines.distributed`` ignored the result entirely."""

    @staticmethod
    def _main(monkeypatch, result):
        import spectramr.pipelines.distributed as distributed

        monkeypatch.setattr(distributed, "run_distributed_training", lambda **kw: result)
        monkeypatch.setattr(sys, "argv", ["distributed", "--config", "a.yaml"])
        distributed.main()

    def test_a_successful_run_returns_normally(self, monkeypatch) -> None:
        self._main(monkeypatch, {"success": True})

    def test_planted_a_failed_run_exits_nonzero(self, monkeypatch) -> None:
        with pytest.raises(SystemExit) as raised:
            self._main(monkeypatch, {"success": False, "error": "boom"})
        assert raised.value.code == 1

    def test_a_missing_result_exits_nonzero(self, monkeypatch) -> None:
        with pytest.raises(SystemExit) as raised:
            self._main(monkeypatch, None)
        assert raised.value.code == 1
