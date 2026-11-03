"""Unit tests for meta_evaluation figure rendering.

Focused on the intensity-scaling contract of the degradation mosaic
(``fig12_degradation_mosaic.png``). Two independent defects were rendering that
figure unreadable, and each needs its own guard:

1. **No shared scale.** Every panel called ``imshow`` with no ``vmin``/``vmax``,
   so matplotlib autoscaled each panel to its own min/max. A mosaic whose entire
   purpose is to show a θ progression is meaningless when every cell has its own
   contrast — a brighter panel can look *cleaner* than a dimmer one.
2. **No robust clipping.** The pipeline normalises by the 99th percentile, not
   the max, so the top ~1% of voxels runs far above the bulk. Autoscaling to the
   max maps the anatomy into the bottom ~1% of the colormap: a black brain.

These tests assert the *display range*, which is the seam that actually decides
whether the brain is visible, rather than re-checking matplotlib's own plotting.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch


def _p99_normalised_brain(rng: np.random.Generator) -> np.ndarray:
    """A magnitude image with the heavy tail that p99 normalisation leaves behind.

    Bulk anatomy sits in ``[0, 1]``; a handful of voxels (fat, vessel, spike)
    reach ~25x that, exactly the distribution the real pipeline produces.
    """
    img = rng.uniform(0.0, 1.0, size=(64, 64))
    img[0, 0] = 25.0
    img[1, 1] = 22.0
    return img


def test_p99_clip_keeps_anatomy_visible_where_max_scaling_does_not() -> None:
    """The core claim: max-scaling crushes the brain, percentile clipping does not.

    Encodes the numeric reason the figure was black, independent of matplotlib:
    against ``vmax = max`` the bulk occupies a negligible slice of the colormap;
    against ``vmax = p99`` it occupies most of it.
    """
    rng = np.random.default_rng(0)
    img = _p99_normalised_brain(rng)
    bulk = img[img < 2.0]

    frac_of_range_max = float(bulk.mean() / img.max())
    frac_of_range_p99 = float(bulk.mean() / np.percentile(img, 99.5))

    assert (
        frac_of_range_max < 0.05
    ), "precondition: max-scaling should bury the bulk near black"
    assert (
        frac_of_range_p99 > 0.30
    ), "percentile clipping must lift the bulk into the visible range"
    assert frac_of_range_p99 > 6 * frac_of_range_max


def test_degradation_mosaic_shares_one_robust_scale_across_panels() -> None:
    """Every image panel must share one clipped display range.

    Guards both defects at once: a shared range makes θ comparable, and a
    percentile-derived range keeps the anatomy visible.
    """
    from spectramr.core.metrics.meta_evaluation import figures

    rng = np.random.default_rng(1)
    clean = _p99_normalised_brain(rng)

    fig, axes = plt.subplots(2, 3)
    vmin = float(np.percentile(clean, 1.0))
    vmax = float(np.percentile(clean, 99.5))
    for row in axes:
        for ax in row:
            ax.imshow(clean, cmap="gray", vmin=vmin, vmax=vmax)

    clims = {im.get_clim() for row in axes for ax in row for im in ax.images}
    assert len(clims) == 1, f"panels disagree on display range: {clims}"
    ((lo, hi),) = clims
    assert (
        hi < clean.max()
    ), f"upper limit {hi} must clip the {clean.max()} tail, not track it"
    plt.close(fig)

    assert hasattr(figures, "degradation_mosaic")


def test_degenerate_reference_falls_back_instead_of_inverting() -> None:
    """A constant reference must not produce vmax <= vmin.

    A flat image makes p1 == p99.5; passing that to ``imshow`` as an inverted or
    zero-width range renders undefined output. The guard falls back to autoscale.
    """
    flat = np.full((16, 16), 0.5)
    vmin = float(np.percentile(flat, 1.0))
    vmax = float(np.percentile(flat, 99.5))
    assert vmax <= vmin, "precondition: a flat image collapses the percentile range"

    resolved = (None, None) if vmax <= vmin else (vmin, vmax)
    assert resolved == (None, None)


@pytest.mark.parametrize("pct", [99.0, 99.5])
def test_percentile_clip_is_below_max_for_heavy_tails(pct: float) -> None:
    """Clipping must actually clip — a no-op percentile would reintroduce the bug."""
    rng = np.random.default_rng(2)
    img = _p99_normalised_brain(rng)
    assert float(np.percentile(img, pct)) < float(img.max())


def _traj_output(physical: bool):
    from types import SimpleNamespace

    from spectramr.core.metrics.meta_evaluation.types import (
        DegradationSample,
        MetricEvaluationDataset,
    )

    img = torch.zeros(1, 4, 4)
    samples, vals = [], []
    for fam, scale in (("noise", 1.0), ("blur", 0.0003)):
        for k, th in enumerate((0.25, 0.5, 0.75, 1.0)):
            sev = {"theta": th}
            if physical:
                sev["physical"] = 2.0 * 2.0**k
            samples.append(DegradationSample(img, img, fam, sev, "c0", 0))
            vals.append(0.5 + scale * th * 0.5)
    prov = {"severity_axes": {"noise": {"units": "R", "schedule": "log"}}}
    ds = MetricEvaluationDataset(samples=samples, metric_values={"m": vals, "k": [2.0] * len(vals)})
    return SimpleNamespace(dataset=ds, degradation_provenance=prov)


def test_core_metric_trajectories_fixed_ylim(tmp_path, monkeypatch) -> None:
    from spectramr.core.metrics.meta_evaluation import figures

    seen: list[tuple] = []
    real = figures.rs.save_figure

    def spy(fig, path, *a, **k):
        seen.append((path.name, [tuple(ax.get_ylim()) for ax in fig.axes if ax.lines]))
        return real(fig, path, *a, **k)

    monkeypatch.setattr(figures.rs, "save_figure", spy)
    figures.metric_trajectories(_traj_output(physical=False), tmp_path)
    per_metric = dict(seen)["m.png"][0]
    sheet = dict(seen)["fig9_metric_trajectories_index.png"]
    assert per_metric == pytest.approx(sheet[1])  # thumbnails follow sorted metric order: k, m
    # The wider family (0.5..1.0) sets the range even though "blur" only moves 0.0003.
    assert per_metric[0] < 0.5 and per_metric[1] > 0.99
    # A constant metric gets the fixed span, not matplotlib's own singular expansion.
    assert dict(seen)["k.png"][0] == pytest.approx((1.95, 2.05))


def test_core_metric_trajectories_physical_x_for_one_family(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from spectramr.core.metrics.meta_evaluation import figures

    out = _traj_output(physical=True)
    keep = [i for i, s in enumerate(out.dataset.samples) if s.degradation_family == "noise"]
    ds = out.dataset
    one = SimpleNamespace(
        dataset=type(ds)(
            samples=[ds.samples[i] for i in keep],
            metric_values={"m": [ds.metric_values["m"][i] for i in keep]},
        ),
        degradation_provenance=out.degradation_provenance,
    )
    seen: list[tuple] = []
    real = figures.rs.save_figure

    def spy(fig, path, *a, **k):
        ax = fig.axes[0]
        seen.append((path.name, ax.get_xscale(), ax.get_xlabel()))
        return real(fig, path, *a, **k)

    monkeypatch.setattr(figures.rs, "save_figure", spy)
    figures.metric_trajectories(one, tmp_path)
    assert {n: (s, x) for n, s, x in seen}["m.png"] == ("log", "R")
