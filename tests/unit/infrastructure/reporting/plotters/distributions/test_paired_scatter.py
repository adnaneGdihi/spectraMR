import pandas as pd

from spectramr.infrastructure.reporting.plotters.distributions import paired_scatter


def test_pairs_each_arm_with_the_baseline_and_draws_no_text(
    tmp_path, slice_df, figures, drawn_texts
):
    out = paired_scatter.make(
        pd.DataFrame(),
        tmp_path / "f.pdf",
        slice_df=slice_df,
        baseline="arm0",
        formats=("pdf",),
        dpi=72,
    )
    assert out is not None and out.exists()
    # arm0 is the baseline, so two arms remain; every other panel is switched off
    visible = [ax for ax in figures[0].axes if ax.axison]
    assert len(visible) == 2
    assert not drawn_texts(figures[0])


def test_needs_two_arms(tmp_path, slice_df):
    one = slice_df[slice_df["method"] == "arm0"]
    assert paired_scatter.make(pd.DataFrame(), tmp_path / "f.pdf", slice_df=one) is None
