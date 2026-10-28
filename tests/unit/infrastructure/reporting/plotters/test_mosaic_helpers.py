"""Shared mosaic machinery: one window, one error scale, honest annotation."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.plotters import _mosaic, get
from spectramr.infrastructure.reporting.plotters.mri_specific import recon_panel


def _pair(gain=1.0, seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.uniform(0, 1, (32, 32))
    return ref * gain, ref


def test_prediction_shares_the_target_window_so_gain_errors_show():
    pred, ref = _pair(gain=0.5)
    win = _mosaic.window(ref)
    assert _mosaic.to_display(pred, win).mean() < 0.6 * _mosaic.to_display(ref, win).mean()


def test_shared_vmax_is_one_readable_ceiling():
    errs = [np.full((4, 4), 0.03), np.full((4, 4), 0.12)]
    assert _mosaic.shared_vmax(errs) == 0.15


def test_zoom_lands_on_the_error():
    err = np.zeros((64, 64))
    err[40:50, 10:20] = 1.0
    r0, r1, c0, c1 = _mosaic.zoom_box([err], err.shape)
    assert r0 <= 45 < r1 and c0 <= 15 < c1


def test_zoom_skips_a_noisy_background_floor():
    err = np.zeros((64, 64))
    err[:20, :20] = 1.0  # strongest error sits in empty background
    err[40:50, 40:50] = 0.5
    ref = np.zeros((64, 64))
    ref[28:60, 28:60] = 1.0
    r0, _, c0, _ = _mosaic.zoom_box([err], err.shape, ref)
    assert r0 >= 28 and c0 >= 28


def _cases(with_input=True):
    out = []
    for i in range(2):
        pred, ref = _pair(gain=0.9, seed=i)
        case = {
            "prediction": pred,
            "target": ref,
            "case_id": f"c{i}",
            "metrics": {"val_psnr": 30.0 + i},
        }
        if with_input:
            case["input"] = ref * 0.7
        out.append(case)
    return out


def test_recon_panel_renders_with_and_without_an_input(tmp_path):
    for with_input in (True, False):
        res = recon_panel.make(
            pd.DataFrame(),
            tmp_path / f"p{with_input}.png",
            cases=_cases(with_input),
            formats=("png",),
            dpi=60,
        )
        assert res is not None and res.exists()


def test_kspace_recon_skips_without_a_recorded_input(tmp_path):
    cases = [{"pred": c["prediction"], "fs_ref": c["target"]} for c in _cases(False)]
    assert (
        get("mri_a7_kspace_recon")(
            pd.DataFrame(), tmp_path / "k.png", cases=cases, formats=("png",)
        )
        is None
    )


def test_kspace_recon_draws_a_baseline_column_only_when_one_was_recorded(tmp_path):
    cases = [
        {"zero_filled": c["input"], "pred": c["prediction"], "fs_ref": c["target"]}
        for c in _cases()
    ]
    res = get("mri_a7_kspace_recon")(
        pd.DataFrame(), tmp_path / "k.png", cases=cases, formats=("png",), dpi=60
    )
    assert res is not None and res.exists()


def test_failure_gallery_reads_the_recorders_prediction_key(tmp_path):
    """The recorder writes ``prediction``; the gallery read only ``predicted`` and
    drew a bare target with no residual for every recorded failure."""
    from spectramr.infrastructure.reporting.plotters import failure_gallery

    case = _cases(False)[0]
    assert failure_gallery._prediction(case) is case["prediction"]
    res = get("fig_1_12_failure_gallery")(pd.DataFrame(), tmp_path / "f.png", cases=[case])
    assert res is not None and res.exists()


def test_annotated_psnr_uses_the_reference_peak_not_one():
    """A fixed data_range of 1.0 printed negative dB on magnitude images."""
    from spectramr.infrastructure.reporting.plotters._comparison_helpers import (
        _format_metric_str,
    )

    ref = np.full((8, 8), 100.0)
    ref[0, 0] = 50.0
    text = _format_metric_str(ref + 1.0, ref, show=("psnr",))
    assert float(text.split()[1]) > 0
