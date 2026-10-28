"""Every observed test slice becomes one panel PNG and one row of ``slice_index.csv``."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from spectramr.infrastructure.reporting.test_split.degradation_guard import DegradationGuard
from spectramr.infrastructure.reporting.test_split.slice_writer import (
    SLICE_INDEX_CSV,
    HeldOutSliceWriter,
    magnitude,
    render_panel,
)

pytestmark = pytest.mark.unit


def test_magnitude_of_real_stacked_channels_is_the_complex_rss() -> None:
    c = torch.randn(1, 2, 4, 4, dtype=torch.complex64)
    stacked = torch.cat([c.real, c.imag], dim=1)
    np.testing.assert_allclose(magnitude(stacked), magnitude(c), rtol=1e-5)


def test_panel_is_four_target_widths_and_blanks_an_input_off_the_grid() -> None:
    t = np.random.rand(8, 6).astype(np.float32)
    panel = render_panel(np.random.rand(4, 3), t, t)
    assert panel.shape == (8, 24) and panel.dtype == np.uint8
    assert not panel[:, :6].any()


def _batch(mask):
    return {
        "mask": mask,
        "subject_id": ["s1", "s2"],
        "contrast": ["T1", "T2"],
        "file_id": ["s1_T101", "s2_T201"],
        "slice_index": torch.tensor([3, 4]),
    }


def test_each_slice_is_rendered_and_indexed_with_its_realised_acceleration(tmp_path) -> None:
    mask = torch.zeros(2, 1, 8, 8)
    mask[..., ::4] = 1
    writer = HeldOutSliceWriter(tmp_path / "slices", DegradationGuard(4.0))
    writer.observe(
        input_view=torch.rand(2, 1, 8, 8),
        pred_view=torch.rand(2, 1, 8, 8),
        target_view=torch.rand(2, 1, 8, 8),
        batch_data=_batch(mask),
        batch_index=5,
    )
    rows = list(csv.DictReader(writer.write_index().open()))
    assert [r["subject_id"] for r in rows] == ["s1", "s2"]
    assert [r["slice_index"] for r in rows] == ["3", "4"]
    assert float(rows[0]["realised_acceleration"]) == pytest.approx(4.0)
    assert rows[0]["rate_source"] == "mask"
    image = Image.open(tmp_path / rows[1]["image"])
    assert image.size == (32, 8)
    assert (tmp_path / "slices" / SLICE_INDEX_CSV).is_file()


def test_a_poisoned_first_batch_renders_nothing(tmp_path) -> None:
    writer = HeldOutSliceWriter(tmp_path / "slices", DegradationGuard(4.0))
    x = torch.rand(2, 1, 8, 8)
    writer.observe(
        input_view=x,
        pred_view=x,
        target_view=x.clone(),
        batch_data=_batch(torch.ones(2, 1, 8, 8)),
        batch_index=0,
    )
    assert writer.poisoned and not writer.rows
