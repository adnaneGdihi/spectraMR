"""The held-out test pass turns per-slice rows into the report's ``final_eval.json`` shape."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from spectramr.pipelines.test_pass import case_id, final_eval_from_rows


def _write(path: Path, rows: list[dict]) -> Path:
    header = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(i, psnr, **extra):
    return {
        "iteration": 0,
        "epoch": 0,
        "batch_index": i,
        "n_samples": 1,
        "subject_id": "s1",
        "contrast": "T1",
        "slice_index": i,
        "repetition_group": "g",
        "file_id": "f",
        "val_psnr": psnr,
        **extra,
    }


def test_rows_become_metric_by_case_with_the_split_prefix_stripped(tmp_path):
    rows = _write(tmp_path / "r.csv", [_row(0, 20.0), _row(1, 22.0)])
    out = final_eval_from_rows(rows)
    assert out == {"psnr": {"s1/T1/f/0": 20.0, "s1/T1/f/1": 22.0}}


def test_a_non_finite_value_is_left_out_not_zeroed(tmp_path):
    rows = _write(tmp_path / "r.csv", [_row(0, 20.0), _row(1, "nan")])
    assert final_eval_from_rows(rows) == {"psnr": {"s1/T1/f/0": 20.0}}


def test_identifier_columns_are_never_metrics(tmp_path):
    out = final_eval_from_rows(_write(tmp_path / "r.csv", [_row(0, 20.0)]))
    assert set(out) == {"psnr"}


def test_the_same_slice_gets_the_same_id_in_every_arm():
    assert case_id(_row(3, 1.0)) == case_id(_row(3, 99.0))


def test_a_row_without_identifiers_falls_back_to_its_batch_index():
    assert case_id({"batch_index": "7"}) == "batch7"


class _Loop:
    """A stand-in loop whose ``evaluate`` writes the rows a real test pass would."""

    def __init__(self, out_dir_holder, seen):
        self.holder, self.seen = out_dir_holder, seen

    def evaluate(self, *, iteration, epoch, csv_file, split):
        self.seen.update(split=split, sinks=dict(self.holder.sinks_during))
        _write(Path(csv_file).parent / "validation_batches.csv", [_row(0, 21.0), _row(1, 23.0)])
        return {"val_psnr": 22.0}


class _Strategy:
    def __init__(self):
        self._per_case_metric_sink = object()
        self._report_case_recorder = object()
        self.sinks_during = {}


def _config(tmp_path):
    return SimpleNamespace(
        data=SimpleNamespace(source=SimpleNamespace(test_index_path="test.json"))
    )


def test_a_test_pass_writes_final_eval_and_leaves_the_report_sinks_as_it_found_them(tmp_path):
    import json

    from spectramr.pipelines.test_pass import run_test_pass

    strategy, seen = _Strategy(), {}
    sink, recorder = strategy._per_case_metric_sink, strategy._report_case_recorder

    class Loop(_Loop):
        def evaluate(self, **kw):
            strategy.sinks_during = {
                "sink": strategy._per_case_metric_sink,
                "recorder": strategy._report_case_recorder,
            }
            return super().evaluate(**kw)

    stale = tmp_path / "test"
    stale.mkdir()
    _write(
        stale / "validation_batches.csv", [_row(9, 1.0)]
    )  # a rerun's leftover must not be scored
    pipeline = type("P", (), {"data_loaders": {"test": object()}})()
    summary = run_test_pass(
        Loop(strategy, seen),
        pipeline,
        strategy,
        _config(tmp_path),
        tmp_path,
        iteration=5,
        weights="best",
        weights_iteration=3,
    )
    assert seen["split"] == "test"
    assert seen["sinks"] == {"sink": None, "recorder": None}
    assert (strategy._per_case_metric_sink, strategy._report_case_recorder) == (sink, recorder)
    assert json.loads((tmp_path / "final_eval.json").read_text()) == {
        "psnr": {"s1/T1/f/0": 21.0, "s1/T1/f/1": 23.0}
    }
    assert (summary["n_cases"], summary["weights"]) == (2, "best")
    written = json.loads((tmp_path / "test_metrics.json").read_text())
    assert (written["iteration"], written["weights_iteration"]) == (5, 3)


def test_a_declared_test_set_with_no_loader_raises(tmp_path):
    import pytest

    from spectramr.pipelines.test_pass import run_test_pass

    pipeline = type("P", (), {"data_loaders": {}})()
    with pytest.raises(ValueError, match="no test loader"):
        run_test_pass(
            None,
            pipeline,
            _Strategy(),
            _config(tmp_path),
            tmp_path,
            iteration=0,
            weights="best",
        )


def test_an_arm_without_a_test_set_is_not_scored(tmp_path):
    from spectramr.pipelines.test_pass import run_test_pass

    bare = SimpleNamespace(data=SimpleNamespace(source=SimpleNamespace(test_index_path=None)))
    assert run_test_pass(None, None, None, bare, tmp_path, iteration=0, weights="final") is None
    assert not (tmp_path / "final_eval.json").exists()


def _hook(monkeypatch, result, *, sanity=False, declared=True, trigger="train_end"):
    from spectramr.pipelines import test_pass, train

    calls = []
    monkeypatch.setattr(
        test_pass,
        "run_test_pass",
        lambda *a, **kw: calls.append(kw) or {"metrics": {"psnr": 1.0}},
    )
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            source=SimpleNamespace(
                test_index_path="t.json" if declared else None, test_pass=trigger
            )
        )
    )
    train._run_held_out_test_pass(
        None, None, None, cfg, {"run_output_dir": "/tmp/x"}, result, is_sanity_check=sanity
    )
    return calls


def test_test_pass_infer_skips_the_end_of_training_pass(monkeypatch):
    result = {"success": True, "best_weights_restored": True, "iterations_completed": 9}
    assert _hook(monkeypatch, result, trigger="infer") == []


def test_test_pass_infer_is_scored_by_a_validate_only_run(monkeypatch):
    """`infer --test-split` is a validate-only run; it must reach the test pass."""
    result = {"success": True, "validate_only": True, "resumed_step": 7, "weights": "ckpt"}
    assert len(_hook(monkeypatch, result, trigger="infer")) == 1


def test_a_finished_run_is_tested_and_its_result_carries_the_numbers(monkeypatch):
    result = {"success": True, "best_weights_restored": True, "iterations_completed": 9}
    calls = _hook(monkeypatch, result)
    assert calls[0]["weights"] == "best" and result["test_metrics"] == {"psnr": 1.0}


def test_restored_best_weights_name_their_own_step_not_the_last_one(monkeypatch):
    """Planted: test_metrics said iteration 84001 for weights restored from step 54000."""
    result = {"success": True, "best_weights_restored": True, "iterations_completed": 84001,
              "best_iteration": 54000}
    assert _hook(monkeypatch, result)[0]["weights_iteration"] == 54000
    final = {"success": True, "best_weights_restored": False, "best_iteration": 54000}
    assert _hook(monkeypatch, final)[0]["weights_iteration"] is None


def test_a_run_without_restored_best_weights_says_it_scored_the_final_ones(monkeypatch):
    calls = _hook(monkeypatch, {"success": True, "best_weights_restored": False})
    assert calls[0]["weights"] == "final"


def test_a_validate_only_run_names_the_checkpoint_it_loaded(monkeypatch):
    result = {"success": True, "validate_only": True, "weights": "ckpt.pt", "resumed_step": 46080,
              "iterations_completed": 0}
    calls = _hook(monkeypatch, result)
    assert calls[0]["weights"] == "ckpt.pt"
    # Planted: a validate-only pass was stamped iteration 0 with no weights step.
    assert (calls[0]["iteration"], calls[0]["weights_iteration"]) == (46080, 46080)


def test_nothing_is_tested_for_a_yield_a_failure_a_sanity_run_or_an_undeclared_set(monkeypatch):
    assert _hook(monkeypatch, {"wall_clock_yield": True}) == []
    assert _hook(monkeypatch, {"success": False}) == []
    assert _hook(monkeypatch, {"success": True}, sanity=True) == []
    assert _hook(monkeypatch, {"success": True}, declared=False) == []


def test_the_test_pass_attaches_a_slice_writer_and_writes_the_report(tmp_path):
    from spectramr.pipelines.test_pass import run_test_pass

    strategy, seen = _Strategy(), {}

    class Loop(_Loop):
        def evaluate(self, **kw):
            seen["writer"] = strategy._test_slice_writer
            return super().evaluate(**kw)

    pipeline = type("P", (), {"data_loaders": {"test": object()}})()
    run_test_pass(Loop(strategy, seen), pipeline, strategy, _config(tmp_path), tmp_path, iteration=1, weights="w")
    assert seen["writer"] is not None and strategy._test_slice_writer is None
    assert (tmp_path / "test" / "report.html").is_file()
    assert (tmp_path / "test" / "summary_by_group.csv").is_file()


def test_a_poisoned_test_input_refuses_to_report(tmp_path):
    import pytest

    from spectramr.pipelines.test_pass import run_test_pass

    strategy = _Strategy()

    class Loop(_Loop):
        def evaluate(self, **kw):
            strategy._test_slice_writer.poisoned = "the test input equals the target"
            return super().evaluate(**kw)

    pipeline = type("P", (), {"data_loaders": {"test": object()}})()
    with pytest.raises(ValueError, match="refusing to report"):
        run_test_pass(Loop(strategy, {}), pipeline, strategy, _config(tmp_path), tmp_path, iteration=1, weights="w")
    assert not (tmp_path / "final_eval.json").exists()
