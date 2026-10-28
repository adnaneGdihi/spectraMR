"""The bins Algorithm F fits over: phase-encode rows, or shells of radial |k|."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.diffusion.reverse_decay_axis import (
    DECAY_AXES,
    LINE_DECAY_AXES,
    build_decay_axis,
    column_axis,
    radial_axis,
    row_axis,
    row_index,
)

pytestmark = pytest.mark.unit

CPU = torch.device("cpu")


def _plane(seed: int = 0) -> torch.Tensor:
    return torch.rand(2, 3, 8, 6, generator=torch.Generator().manual_seed(seed))


def test_the_row_axis_bins_are_the_rows() -> None:
    axis = row_axis(8, 6, CPU)
    t = _plane()
    assert axis.num_bins == 8
    assert torch.equal(axis.coord, row_index(8, CPU))
    assert torch.equal(axis.reduce_sum(t), t.sum(dim=(-3, -1)))
    assert torch.equal(axis.reduce_max(t), t.amax(dim=(-3, -1)))
    assert torch.equal(
        axis.spread(axis.reduce_sum(t)).expand(2, 1, 8, 6)[:, 0, :, 0], axis.reduce_sum(t)
    )


def test_a_radial_bin_is_the_rounded_distance_to_the_centre() -> None:
    axis = radial_axis(8, 6, CPU)
    ky = row_index(8, CPU).view(-1, 1)
    kx = row_index(6, CPU).view(1, -1)
    assert torch.equal(axis.bins, torch.sqrt(ky * ky + kx * kx).round().long())
    assert axis.bins[4, 3] == 0
    assert axis.num_bins == int(axis.bins.max()) + 1
    assert torch.equal(axis.coord_map()[0, 0], axis.bins.float())


def test_radial_reductions_match_a_per_bin_loop() -> None:
    axis = radial_axis(8, 6, CPU)
    t = _plane()
    total = axis.reduce_sum(t)
    peak = axis.reduce_max(t)
    for b in range(axis.num_bins):
        members = axis.bins == b
        assert torch.allclose(total[:, b], t[..., members].sum(dim=(-2, -1)))
        assert torch.equal(peak[:, b], t[..., members].amax(dim=(-2, -1)))
    spread = axis.spread(total)
    assert spread.shape == (2, 1, 8, 6)
    assert torch.equal(spread[:, 0], total[:, axis.bins])


def test_every_named_axis_builds_and_an_unknown_one_raises() -> None:
    for name in DECAY_AXES:
        assert build_decay_axis(name, 8, 6, CPU).name == name
    with pytest.raises(ValueError, match="decay axis"):
        build_decay_axis("angular", 8, 6, CPU)


@pytest.mark.parametrize(("h", "w"), [(8, 6), (40, 64)])
def test_the_column_axis_is_the_row_axis_of_the_transposed_plane(h: int, w: int) -> None:
    """A mask that removes columns (#2160) bins its lines along dim -1."""
    column, row = column_axis(h, w, CPU), row_axis(w, h, CPU)
    t = torch.rand(2, 3, h, w, generator=torch.Generator().manual_seed(1))
    tt = t.transpose(-1, -2)
    assert column.num_bins == w and column.line_dim == -1
    assert torch.equal(column.coord, row_index(w, CPU))
    assert torch.equal(column.reduce_sum(t), row.reduce_sum(tt))
    assert torch.equal(column.reduce_max(t), row.reduce_max(tt))
    per_bin = column.reduce_sum(t)
    assert torch.equal(
        column.spread(per_bin).expand(2, 1, h, w), row.spread(per_bin).expand(2, 1, w, h).mT
    )
    assert torch.equal(column.coord_map().expand(1, 1, h, w)[0, 0, 0], row_index(w, CPU))


def test_the_mask_line_axis_selects_the_binning() -> None:
    assert LINE_DECAY_AXES == {"y": "row", "x": "column", None: "row"}
