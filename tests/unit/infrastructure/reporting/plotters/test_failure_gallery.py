"""Tests for the failure-case gallery plotter.

Targets ``spectramr.infrastructure.reporting.plotters.failure_gallery``. The
regression of note: with only 1-3 cases the grid must stay compact rather than
producing a mostly-empty ultra-wide canvas (the provenance stamp pins
``bbox_inches`` to full width, so the layout itself must adapt).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.plotters import failure_gallery


def _case(seed, name):
    rng = np.random.default_rng(seed)
    tgt = rng.random((16, 16)).astype(np.float32)
    return {"name": name, "predicted": tgt + 0.1 * rng.random((16, 16)),
            "target": tgt, "residual": float(seed)}


def _aspect(path):
    from PIL import Image
    w, h = Image.open(path).size
    return w / h


def test_soft_skip_when_no_cases(tmp_path):
    assert failure_gallery.make(pd.DataFrame(), tmp_path / "f.png", cases=[]) is None
    assert failure_gallery.make(pd.DataFrame(), tmp_path / "f.png", cases=None) is None


def test_single_case_is_not_ultra_wide(tmp_path):
    out = failure_gallery.make(pd.DataFrame(), tmp_path / "f.png",
                               cases=[_case(0, "case_0")])
    assert out is not None and out.exists()
    # regression: a 1-case gallery used to render at ~6.5:1; the adaptive grid
    # keeps it compact (image + histogram = 2 columns, one row).
    assert _aspect(out) < 4.0


def test_three_cases_compact(tmp_path):
    out = failure_gallery.make(pd.DataFrame(), tmp_path / "f.png",
                               cases=[_case(i, f"c{i}") for i in range(3)])
    assert out is not None
    assert _aspect(out) < 6.0


def test_many_cases_wrap_to_grid(tmp_path):
    # 6 cases → 4 per row → 2 rows; should render fine and not be a single strip
    out = failure_gallery.make(pd.DataFrame(), tmp_path / "f.png",
                               cases=[_case(i, f"c{i}") for i in range(6)])
    assert out is not None and out.exists()


def test_a_path_is_refused_rather_than_opened_here(tmp_path):
    """File loading belongs to the data layer (#2501); a path used to be opened
    with nib/np/PIL and a failure silently dropped the panel."""
    import pytest

    case = _case(0, "c0")
    case["target"] = str(tmp_path / "missing.nii.gz")
    with pytest.raises(TypeError, match="takes arrays"):
        failure_gallery.make(pd.DataFrame(), tmp_path / "f.png", cases=[case])
