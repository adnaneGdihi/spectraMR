"""Per-case rows at one checkpoint pass."""

import pandas as pd

from spectramr.infrastructure.reporting.per_case import nearest_step, per_case_rows, read_per_call


def _calls():
    rows = []
    for step in (100, 200):
        for r in (2.0, 8.0):
            for b in range(3):
                rows.append(
                    {
                        "case_id": f"val_step{step}_R{int(r)}x",
                        "split": "val",
                        "step": step,
                        "acceleration_level": r,
                        "batch_index": b,
                        "heldout": False,
                        "val_psnr": 30.0 - r + b + step / 100,
                    }
                )
    return pd.DataFrame(rows)


def test_one_pass_only():
    rows = per_case_rows(_calls(), step=100, method="m")
    assert set(rows["step"]) == {100}
    assert len(rows) == 6
    assert set(rows["method"]) == {"m"}


def test_none_takes_the_last_pass():
    assert set(per_case_rows(_calls())["step"]) == {200}


def test_batch_index_makes_the_case_unique():
    rows = per_case_rows(_calls(), step=200)
    assert rows["case_id"].nunique() == 6


def test_bookkeeping_and_booleans_are_not_metrics():
    rows = per_case_rows(_calls())
    assert set(rows["metric"]) == {"val_psnr"}
    assert set(rows["base_metric"]) == {"psnr"}
    assert set(rows["acceleration"]) == {2.0, 8.0}


def test_nearest_step_takes_the_pass_at_or_before():
    assert nearest_step(_calls(), 150) == 100
    assert nearest_step(_calls(), 50) is None


def test_read_per_call_absent_is_none(tmp_path):
    assert read_per_call(tmp_path) is None
    _calls().to_csv(tmp_path / "per_call_metrics.csv", index=False)
    assert len(read_per_call(tmp_path)) == 12


def test_empty_input_gives_an_empty_frame():
    assert per_case_rows(None).empty


def test_validation_batch_size_reads_the_validation_loader(tmp_path):
    import json

    from spectramr.infrastructure.reporting.per_case import validation_batch_size

    assert validation_batch_size(tmp_path) is None
    (tmp_path / "resolved_config.json").write_text(
        json.dumps({"training": {"batch_size": 4}, "validation": {"loader": {"batch_size": 2}}})
    )
    assert validation_batch_size(tmp_path) == 2


def test_call_unit_never_calls_a_batch_mean_a_case():
    from spectramr.infrastructure.reporting.per_case import call_unit

    assert call_unit([1, 1]) == "slices"
    assert call_unit([2, 2]) == "validation calls (means of 2 slices)"
    assert call_unit([2, None]) == "validation calls (batch means)"
    assert "case" not in call_unit([None])

def test_a_checkpoint_before_every_pass_yields_nothing_not_every_pass():
    rows = per_case_rows(_calls(), step=50)
    assert rows.empty


def test_a_replayed_pass_is_deduplicated():
    doubled = pd.concat([_calls(), _calls()], ignore_index=True)
    rows = per_case_rows(doubled, step=200)
    assert rows["pair_key"].is_unique

def test_rows_without_a_batch_index_are_all_kept():
    """GAN-style strategies feed the sink with no batch index; de-duplicating on
    pair_key used to cut every such pass to one row."""
    frame = pd.DataFrame(
        [{"case_id": "val_step1000", "split": "val", "step": 1000, "val_psnr": 30.0 + i}
         for i in range(5)]
    )
    rows = per_case_rows(frame, step=1000)
    assert len(rows) == 5 and rows["pair_key"].is_unique


def test_a_cascade_logged_in_one_call_keeps_its_rungs_apart():
    """`val_psnr_2x` / `_8x` in one row are two rungs; `_mean` is not a case."""
    frame = pd.DataFrame(
        [{"case_id": "c", "split": "val", "step": 1, "batch_index": b,
          "val_psnr_2x": 30.0, "val_psnr_8x": 22.0, "val_psnr_mean": 26.0} for b in range(4)]
    )
    rows = per_case_rows(frame, step=1)
    assert set(rows["acceleration"]) == {2.0, 8.0}
    assert len(rows) == 8
    assert not rows["metric"].str.endswith("_mean").any()
