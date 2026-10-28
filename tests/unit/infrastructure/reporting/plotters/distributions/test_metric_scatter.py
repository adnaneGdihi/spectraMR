import pandas as pd

from spectramr.infrastructure.reporting.plotters.distributions import metric_scatter


def test_renders_and_carries_no_value_text(tmp_path, slice_df, figures, drawn_texts):
    out = metric_scatter.make(
        pd.DataFrame(), tmp_path / "f.pdf", slice_df=slice_df, formats=("pdf",), dpi=72
    )
    assert out is not None and out.exists()
    assert not drawn_texts(figures[0])


def test_legend_stays_within_twelve_entries(tmp_path, slice_df, figures):
    metric_scatter.make(
        pd.DataFrame(), tmp_path / "f.pdf", slice_df=slice_df, formats=("pdf",), dpi=72
    )
    legends = [lg for lg in figures[0].legends if lg is not None]
    assert all(len(lg.get_texts()) <= 12 for lg in legends)


def test_soft_skips_without_slice_rows(tmp_path):
    assert metric_scatter.make(pd.DataFrame(), tmp_path / "f.pdf", slice_df=None) is None
    assert metric_scatter.make(pd.DataFrame(), tmp_path / "f.pdf", slice_df=pd.DataFrame()) is None
