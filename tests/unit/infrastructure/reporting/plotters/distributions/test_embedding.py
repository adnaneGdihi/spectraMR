import pandas as pd

from spectramr.infrastructure.reporting.plotters.distributions import embedding


def test_renders_and_carries_no_value_text(tmp_path, slice_df, figures, drawn_texts):
    out = embedding.make(
        pd.DataFrame(), tmp_path / "f.pdf", slice_df=slice_df, formats=("pdf",), dpi=72
    )
    assert out is not None and out.exists()
    assert not drawn_texts(figures[0])


def test_legend_stays_within_twelve_entries(tmp_path, slice_df, figures):
    embedding.make(pd.DataFrame(), tmp_path / "f.pdf", slice_df=slice_df, formats=("pdf",), dpi=72)
    legends = [lg for lg in figures[0].legends if lg is not None]
    assert all(len(lg.get_texts()) <= 12 for lg in legends)


def test_soft_skips_without_slice_rows(tmp_path):
    assert embedding.make(pd.DataFrame(), tmp_path / "f.pdf", slice_df=None) is None
    assert embedding.make(pd.DataFrame(), tmp_path / "f.pdf", slice_df=pd.DataFrame()) is None


def test_tsne_returns_two_deterministic_coordinates_per_row():
    import numpy as np

    z = np.random.default_rng(0).normal(size=(40, 5))
    first, again = embedding.tsne_2d(z), embedding.tsne_2d(z)
    assert first.shape == (40, 2)
    np.testing.assert_allclose(first, again)


def test_tsne_separates_two_well_separated_clusters():
    import numpy as np

    rng = np.random.default_rng(1)
    z = np.vstack([rng.normal(0, 0.1, (30, 4)), rng.normal(8, 0.1, (30, 4))])
    xy = embedding.tsne_2d(z)
    gap = np.linalg.norm(xy[:30].mean(axis=0) - xy[30:].mean(axis=0))
    spread = max(xy[:30].std(), xy[30:].std())
    assert gap > 3 * spread


def test_tsne_needs_four_points_and_scikit_learn(monkeypatch):
    import numpy as np
    import pytest

    with pytest.raises(ValueError, match="at least four"):
        embedding.tsne_2d(np.zeros((3, 2)))
    monkeypatch.setattr(embedding, "HAVE_SKLEARN", False)
    with pytest.raises(RuntimeError, match="scikit-learn"):
        embedding.tsne_2d(np.zeros((10, 2)))


def test_the_figure_soft_skips_without_scikit_learn(tmp_path, slice_df, monkeypatch):
    monkeypatch.setattr(embedding, "HAVE_SKLEARN", False)
    assert embedding.make(pd.DataFrame(), tmp_path / "f.pdf", slice_df=slice_df) is None


def test_pca_axes_are_ordered_by_variance():
    import numpy as np

    z = np.random.default_rng(1).normal(size=(200, 3)) * np.array([5.0, 1.0, 0.1])
    xy = embedding.pca_2d(z)
    assert xy[:, 0].var() > xy[:, 1].var()
