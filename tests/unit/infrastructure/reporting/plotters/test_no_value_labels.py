"""Publication figures carry no numeric value labels and no on-canvas provenance."""

import numpy as np
import pandas as pd
import pytest

from spectramr.infrastructure.reporting import metadata as meta_mod
from spectramr.infrastructure.reporting.plotters import get
from spectramr.infrastructure.reporting.plotters.mri_specific import (
    kspace_error_spectrum,
    recon_panel,
)


def _captured_figures(monkeypatch, module):
    figs = []
    real = module.save_figure

    def spy(fig, *a, **k):
        figs.append(fig)
        return real(fig, *a, **k)

    monkeypatch.setattr(module, "save_figure", spy)
    return figs


def _cases():
    rng = np.random.default_rng(0)
    ref = np.zeros((64, 64), np.float32)
    ref[16:48, 16:48] = 1.0
    return [
        {
            "prediction": ref * 0.9 + 0.01 * rng.random((64, 64), dtype=np.float32),
            "target": ref + 0.02 * rng.random((64, 64), dtype=np.float32),
            "rank": "median",
            "metrics": {"val_psnr": 31.2, "val_ssim": 0.91},
            "case_id": "s0",
        }
    ]


def test_recon_panel_has_no_metric_overlay(tmp_path, monkeypatch):
    mod = recon_panel
    figs = _captured_figures(monkeypatch, mod)
    get("mri_recon_panel")(
        pd.DataFrame(), tmp_path / "p.pdf", cases=_cases(), formats=("pdf",), dpi=72
    )
    assert not [t.get_text() for ax in figs[0].axes for t in ax.texts]


def test_spectrum_has_no_clipped_corner_bins_and_no_single_entry_legend(tmp_path, monkeypatch):
    figs = _captured_figures(monkeypatch, kspace_error_spectrum)
    get("kspace_error_spectrum")(
        pd.DataFrame(), tmp_path / "s.pdf", cases=_cases(), formats=("pdf",), dpi=72
    )
    ax = figs[0].axes[0]
    assert ax.get_legend() is None
    assert ax.lines[0].get_xdata()[-1] == pytest.approx(1.0)


def test_stamp_figure_leaves_the_canvas_alone():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    meta_mod.stamp_figure(fig, meta_mod.RunMetadata())
    assert not fig.texts
    plt.close(fig)
