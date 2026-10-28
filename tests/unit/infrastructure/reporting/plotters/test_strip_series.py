"""What the QC strip distributes: one pass, no bookkeeping columns."""

import pandas as pd

from spectramr.infrastructure.reporting.plotters.qc._strip_series import _collect, _one_pass


def _calls():
    return pd.DataFrame(
        [
            {
                "case_id": "c",
                "split": "val",
                "step": s,
                "batch_index": b,
                "heldout": False,
                "acceleration_level": 2.0,
                "psnr": float(s + b),
            }
            for s in (100, 200)
            for b in range(4)
        ]
    )


def test_one_pass_reads_the_requested_checkpoint():
    rows = _one_pass(pd.DataFrame(), _calls(), checkpoint_step=150)
    assert set(rows["step"]) == {100}


def test_a_checkpoint_before_every_pass_reads_nothing():
    assert _one_pass(pd.DataFrame(), _calls(), checkpoint_step=50).empty


def test_bookkeeping_columns_are_not_metrics():
    series = _collect(pd.DataFrame(), _calls(), None)
    assert set(series) == {"psnr"}
