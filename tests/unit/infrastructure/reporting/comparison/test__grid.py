"""``comparison/_grid.py``: chunks, hidden axes and numbered continuation files."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from spectramr.infrastructure.reporting.comparison._grid import chunks, new_grid, write_chunks


def test_chunks_keep_order_and_the_remainder():
    assert chunks(list("abcde"), 2) == [["a", "b"], ["c", "d"], ["e"]]
    assert chunks([], 3) == []
    with pytest.raises(ValueError, match="at least 1"):
        chunks(["a"], 0)


def test_unused_axes_are_hidden_and_panels_are_lettered():
    fig, axes = new_grid(5)
    assert len(axes) == 5
    all_axes = fig.axes
    assert sum(not a.get_visible() for a in all_axes) == 3  # 2 rows x 4 columns, 5 used
    plt.close(fig)


def test_a_single_panel_carries_no_letter():
    fig, axes = new_grid(1)
    assert len(axes) == 1 and not axes[0].texts
    plt.close(fig)


def test_continuation_files_are_numbered_from_two(tmp_path):
    def draw(chunk):
        fig, _ = new_grid(len(chunk))
        return fig

    first = write_chunks(tmp_path / "f.png", list(range(7)), 3, draw, formats=("png",), dpi=40)
    assert first == tmp_path / "f.png"
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["f.png", "f_2.png", "f_3.png"]
    assert write_chunks(tmp_path / "g.png", [], 3, draw, formats=("png",)) is None
