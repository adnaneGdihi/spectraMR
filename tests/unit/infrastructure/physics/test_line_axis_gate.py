"""The first-batch line-axis gate (#2160), on non-square planes only.

On a square plane a swapped H/W is indistinguishable from a correct one, so
every case here uses 48x80 or 40x64.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.line_axis_gate import (
    LineAxisMismatchError,
    check_line_axis,
    structural_zero_lines,
)


def _kspace(h: int, w: int, *, zero_cols: int = 0, zero_rows: int = 0) -> torch.Tensor:
    x = torch.randn(2, 4, h, w)
    if zero_cols:
        x[..., :zero_cols] = 0
    if zero_rows:
        x[..., :zero_rows, :] = 0
    return x


def _line_mask(h: int, w: int, dim: int) -> torch.Tensor:
    mask = torch.zeros(h, w, dtype=torch.bool)
    if dim == -1:
        mask[:, ::4] = True
    else:
        mask[::4, :] = True
    return mask


@pytest.mark.parametrize(("h", "w"), [(48, 80), (40, 64)])
def test_counts_zero_lines_per_dimension(h: int, w: int) -> None:
    assert structural_zero_lines(_kspace(h, w, zero_cols=7)) == {-1: 7, -2: 0}
    assert structural_zero_lines(_kspace(h, w, zero_rows=5)) == {-1: 0, -2: 5}


def test_a_line_that_is_zero_in_one_coil_only_is_not_structural() -> None:
    x = _kspace(48, 80)
    x[0, 0, :, 3] = 0
    assert structural_zero_lines(x)[-1] == 0


@pytest.mark.parametrize(("h", "w"), [(48, 80), (40, 64)])
@pytest.mark.parametrize("dim", [-1, -2])
def test_agreement_is_recorded(h: int, w: int, dim: int) -> None:
    data = _kspace(h, w, zero_cols=6) if dim == -1 else _kspace(h, w, zero_rows=6)
    record = check_line_axis(data, _line_mask(h, w, dim), split="train")
    assert record["verdict"] == "agree"
    assert record["data_axis"] == dim and record["mask_axis"] == dim
    assert record["shape"] == [h, w]


@pytest.mark.parametrize(("h", "w"), [(48, 80), (40, 64)])
@pytest.mark.parametrize("mask_dim", [-1, -2])
def test_a_planted_mismatch_raises(h: int, w: int, mask_dim: int) -> None:
    """PLANTED: the pre-#2160 shape -- the data's unacquired lines on one axis,
    the mask removing lines along the other."""
    data = _kspace(h, w, zero_rows=6) if mask_dim == -1 else _kspace(h, w, zero_cols=6)
    with pytest.raises(LineAxisMismatchError, match="readout"):
        check_line_axis(data, _line_mask(h, w, mask_dim), split="validation")


@pytest.mark.parametrize(("h", "w"), [(48, 80), (40, 64)])
def test_zero_free_data_is_inconclusive_never_agreement(h: int, w: int) -> None:
    record = check_line_axis(_kspace(h, w), _line_mask(h, w, -2), split="train")
    assert record["verdict"] == "inconclusive" and record["data_axis"] is None


def test_zero_lines_on_both_axes_are_inconclusive() -> None:
    data = _kspace(48, 80, zero_cols=4, zero_rows=3)
    assert check_line_axis(data, _line_mask(48, 80, -1), split="train")["verdict"] == (
        "inconclusive"
    )


@pytest.mark.parametrize(
    "mask",
    [torch.ones(48, 80, dtype=torch.bool), torch.rand(48, 80) > 0.5],
    ids=["fully_sampled", "point_pattern"],
)
def test_a_mask_without_a_line_axis_is_not_compared(mask: torch.Tensor) -> None:
    record = check_line_axis(_kspace(48, 80, zero_cols=6), mask, split="train")
    assert record["verdict"] == "not_line_structured" and record["mask_axis"] is None
