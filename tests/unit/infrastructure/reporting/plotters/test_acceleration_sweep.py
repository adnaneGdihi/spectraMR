"""acceleration_sweep: per-R columns and per-call rows, never a trajectory band."""

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metric_names import annotate_frame
from spectramr.infrastructure.reporting.plotters import get
from spectramr.infrastructure.reporting.plotters.mri_specific import (  # noqa: F401
    acceleration_sweep,
)


def _logged(steps=(10, 20)):
    """The aggregator shape: ``val_psnr_<R>x`` columns over validation passes."""
    rows = []
    for step in steps:
        for r in (2, 8, 32):
            rows.append(
                {
                    "run_id": "a",
                    "method": "a",
                    "split": "val",
                    "step": step,
                    "metric": f"val_psnr_{r}x",
                    "value": 40.0 - r / 2 + step / 100,
                }
            )
    return annotate_frame(pd.DataFrame(rows))


def _per_call(step=20, n=6):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        [
            {
                "case_id": f"val_step{step}_R{r}x",
                "split": "val",
                "step": step,
                "acceleration_level": float(r),
                "batch_index": b,
                "val_psnr": 40.0 - r / 2 + rng.normal(0, 0.5),
            }
            for r in (2, 8, 32)
            for b in range(n)
        ]
    )


def test_renders_from_the_logged_per_r_columns(tmp_path):
    res = get("acceleration_sweep")(_logged(), tmp_path / "s.pdf", formats=("pdf",), dpi=80)
    assert res is not None and res.exists()


def test_renders_with_a_band_from_per_call_rows(tmp_path):
    res = get("acceleration_sweep")(
        _logged(), tmp_path / "s.pdf", per_call_df=_per_call(), formats=("pdf",), dpi=80
    )
    assert res is not None and res.exists()


def test_reads_one_pass_from_the_logged_columns():
    rows = acceleration_sweep._from_frame(_logged(), "final")
    assert set(rows["step"]) == {20}
    assert len(rows) == 3


def test_per_call_rows_are_cut_to_the_checkpoint_pass():
    two = pd.concat([_per_call(step=10), _per_call(step=20)], ignore_index=True)
    rows = acceleration_sweep._from_per_call(_logged(), two, "final")
    assert set(rows["step"]) == {20}
    assert len(rows) == 18


def test_a_single_acceleration_is_no_sweep(tmp_path):
    df = _logged()
    df = df[df["acceleration"] == 2.0]
    assert get("acceleration_sweep")(df, tmp_path / "s.pdf", formats=("pdf",)) is None


def test_the_main_runs_checkpoint_is_used_not_the_first_sorted_run():
    rows = acceleration_sweep._from_per_call(
        _logged(), pd.concat([_per_call(step=10), _per_call(step=20)]), "final", checkpoint_step=10
    )
    assert set(rows["step"]) == {10}


def test_n_counts_every_arm(tmp_path, monkeypatch):
    from spectramr.infrastructure.reporting.per_case import per_case_rows

    rows = pd.concat([
        per_case_rows(_per_call(step=20, n=10), method="a"),
        per_case_rows(_per_call(step=20, n=3), method="b"),
    ])
    titles = []
    monkeypatch.setattr("matplotlib.axes.Axes.set_title", lambda self, t, **k: titles.append(t))
    get("acceleration_sweep")(_logged(), tmp_path / "s.png", per_case_df=rows, formats=("png",), dpi=50)
    assert any("n = 3-10" in t for t in titles), titles
