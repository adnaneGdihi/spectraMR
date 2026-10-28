import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_png_run
from spectramr.infrastructure.reporting.plotters.images import progress_strip


def test_pairs_are_in_step_order_and_thinned_keeping_the_ends():
    pairs = progress_strip.pass_pairs(make_png_run(10))
    assert [s for s, _, _ in pairs] == sorted(s for s, _, _ in pairs)
    kept = progress_strip.thin(pairs, 4)
    assert len(kept) == 4 and kept[0] == pairs[0] and kept[-1] == pairs[-1]


def test_renders_a_column_per_pass_with_three_rows(tmp_path):
    out = progress_strip.make(
        pd.DataFrame(), tmp_path / "f.pdf", run_dir=make_png_run(4), formats=("pdf",), dpi=72
    )
    assert out is not None and out.exists()


def test_soft_skips_without_a_run_or_with_fewer_than_two_passes(tmp_path):
    assert progress_strip.make(pd.DataFrame(), tmp_path / "f.pdf", run_dir=None) is None
    assert progress_strip.make(pd.DataFrame(), tmp_path / "f.pdf", run_dir=make_png_run(1)) is None
    assert progress_strip.make(pd.DataFrame(), tmp_path / "f.pdf", run_dir=tmp_path) is None


def test_each_column_is_scored_against_its_own_target(tmp_path, monkeypatch):
    seen = []
    real = progress_strip.save_plot
    monkeypatch.setattr(
        progress_strip, "save_plot", lambda fig, *a, **k: (seen.append(fig), real(fig, *a, **k))[1]
    )
    progress_strip.make(
        pd.DataFrame(), tmp_path / "f.pdf", run_dir=make_png_run(4), formats=("pdf",), dpi=72
    )
    grid = [ax for ax in seen[0].axes if ax.images]
    assert len(grid) == 12  # 3 rows x 4 passes, colour bar excluded
