import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.plotters import get
from spectramr.infrastructure.reporting.plotters import metric_distribution  # noqa: F401


def _frame():
    rng = np.random.default_rng(0)
    rows = []
    for method, mu in (("baseline", 30.0), ("ours", 33.0)):
        for s in range(40):
            rows.append({"method": method, "metric": "psnr",
                         "value": float(rng.normal(mu, 1.0)),
                         "subject_id": f"s{s}", "split": "test"})
    return pd.DataFrame(rows)


def test_metric_distribution_renders(tmp_path):
    out = tmp_path / "metric_distribution.pdf"
    res = get("metric_distribution")(_frame(), out, metrics=["psnr"],
                                     formats=("pdf",), dpi=150)
    assert res is not None and res.exists()


def test_metric_distribution_does_not_touch_global_rng(tmp_path):
    import numpy as np

    np.random.seed(123)
    before = np.random.get_state()[1][0]
    get("metric_distribution")(_frame(), tmp_path / "d.pdf", metrics=["psnr"],
                               formats=("pdf",), dpi=80)
    after = np.random.get_state()[1][0]
    # the plotter must use a local seeded RNG, leaving the global RNG untouched
    assert before == after


def test_metric_distribution_multi_metric_panels_render(tmp_path):
    """2026-07-01 clarity pass: pretty y-labels + panel letters on multi-panel
    layouts must not break rendering."""
    import numpy as np
    import pandas as pd

    from spectramr.infrastructure.reporting.plotters import metric_distribution

    rng = np.random.default_rng(0)
    rows = []
    for metric in ("psnr", "ssim"):
        for method in ("ours", "baseline"):
            for i in range(12):
                rows.append({"run_id": method, "method": method,
                             "metric": metric, "subject_id": f"s{i}",
                             "value": rng.uniform(0.5, 1.0), "split": "test",
                             "step": -1})
    out = metric_distribution.make(pd.DataFrame(rows), tmp_path / "dist.pdf",
                                   metrics=["psnr", "ssim"],
                                   formats=("pdf",), dpi=150)
    assert out is not None and out.exists()


def _texts(fig_path, monkeypatch, slice_df, metrics):
    """Render and return every text the figure drew."""
    import matplotlib.figure

    from spectramr.infrastructure.reporting.plotters import metric_distribution as md

    seen = []
    real = matplotlib.figure.Figure.savefig

    def spy(self, *a, **k):
        seen.extend(t.get_text() for ax in self.axes for t in ax.texts)
        seen.append(self._suptitle.get_text() if self._suptitle else "")
        return real(self, *a, **k)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    md.make(None, fig_path, slice_df=slice_df, metrics=metrics, formats=("png",), dpi=40)
    return seen


def test_slice_rows_get_a_wilcoxon_bracket_over_subjects(tmp_path, monkeypatch):
    from spectramr.infrastructure.reporting.catalog.examples_payloads import make_slice_df

    texts = _texts(tmp_path / "d.png", monkeypatch, make_slice_df(n_arms=3, n_subjects=6), ["psnr"])
    assert any("n=6 subjects" in t for t in texts)
    assert any("validation, selected pass" in t for t in texts)


def test_rows_without_slice_identity_get_no_bracket(tmp_path, monkeypatch):
    from spectramr.infrastructure.reporting.catalog.examples_payloads import make_slice_df

    rows = make_slice_df(n_arms=2).assign(case_id=lambda d: "call#" + d.index.astype(str))
    texts = _texts(tmp_path / "d.png", monkeypatch, rows, ["psnr"])
    assert not any("subjects" in t for t in texts)


def test_the_best_two_follow_the_metric_direction():
    from spectramr.infrastructure.reporting.catalog.examples_payloads import make_slice_df
    from spectramr.infrastructure.reporting.plotters import metric_distribution as md
    from spectramr.infrastructure.reporting.plotters.distributions import _frame

    tab = _frame.wide(make_slice_df(n_arms=3))
    med = {a: tab["nmse"].xs(a, level="method").median() for a in ("arm0", "arm1", "arm2")}
    best = sorted(med, key=med.get)[:2]  # nmse: lower is better
    assert md._subject_test(tab, "nmse", *best) is not None
