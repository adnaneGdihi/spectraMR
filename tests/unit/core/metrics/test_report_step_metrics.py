"""Tests for ``spectramr.core.metrics.report_step_metrics``.

Regression coverage for :class:`RadialKSpaceError` (registered as
``radial_k_error``). The metric bins ``|F·x̂ - F·x|`` by integer radial
k-space frequency and returns the mean over the radial bins. The hardening
fix divides the per-radius sums by their counts only where ``counts > 0``
(``np.divide(..., where=)``) and reduces with ``np.nanmean`` so that any
zero-count bin can never inject a ``0/0 = NaN`` into the final mean.

Contract these tests pin (the observable behaviour the fix guarantees):

* A NON-SQUARE ``[1, 1, 16, 32]`` pair returns a FINITE non-negative
  scalar (regression).
* A square ``[1, 1, 32, 32]`` sanity case returns a finite scalar.
* A perfect reconstruction (``pred == target``) returns finite ~0.
* The returned scalar equals an independent count-weighted mean of the
  per-radius error, i.e. exactly ``nanmean`` over the POPULATED bins — so a
  future regression that re-introduced naive ``sums / counts`` (which would
  divide-by-zero on any empty bin) is caught by the value, not just by
  finiteness.

Reachability note (verified empirically while authoring): for a centred
Cartesian grid the integer-radius binning never actually leaves an interior
or trailing zero-count bin — the corner reaches the maximum integer radius
and the long axis sweeps every integer below it — so the original *symptom*
(NaN from a non-square grid) does not reproduce from grid geometry alone.
The fix is therefore a correct, harmless hardening rather than a
behaviour-changing bug-fix; these tests guard the count-weighted contract so
the hardening cannot silently rot.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from spectramr.core.metrics.report_step_metrics import (
    FocalFrequencyMetric,
    MMDMetric,
    RadialKSpaceError,
    SlicedWassersteinMetric,
    Wasserstein1D,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "metric_cls", [Wasserstein1D, SlicedWassersteinMetric, MMDMetric]
)
def test_distributional_metrics_accept_unequal_batch(metric_cls) -> None:
    """A distribution-vs-distribution metric compares two point sets that may
    differ in cardinality (N degraded vs M clean images). The matching-shape
    guard must NOT reject that — it previously turned every summary metric into
    NaN in the novel pipeline (#311).
    """
    torch.manual_seed(0)
    pred = torch.rand(4, 3, 16, 16)  # 4 "degraded"
    target = torch.rand(2, 3, 16, 16)  # 2 "clean" — deliberately unequal
    metric = metric_cls()

    # The opt-out frees the SAMPLE axis only. It used to clear
    # REQUIRES_MATCHING_SHAPES, which switched off shape checking entirely and let
    # a channel/spatial mismatch through as a finite, meaningless number — see
    # TestDistributionMetricShapeGuard below.
    assert metric.ALLOWS_UNEQUAL_SAMPLE_COUNT is True
    assert metric.REQUIRES_MATCHING_SHAPES is True
    val = float(metric(pred, target))
    assert math.isfinite(val), f"{metric_cls.__name__} returned non-finite {val}"


def _reference_radial_k_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Independent count-weighted mean of the per-radius |F·x̂ − F·x|.

    Mirrors the metric's intent without copying its NaN-handling, so the
    metric's value can be cross-checked. Skips index 0 (the DC bin) to match
    the metric's ``binned[1:]`` reduction.
    """
    p = pred.detach().cpu().numpy().astype(np.float64).squeeze()
    t = target.detach().cpu().numpy().astype(np.float64).squeeze()
    err = np.abs(np.fft.fftshift(np.fft.fft2(p)) - np.fft.fftshift(np.fft.fft2(t)))
    h, w = err.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(err.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    counts = np.bincount(r.ravel())
    sums = np.bincount(r.ravel(), weights=err.ravel())
    populated = counts[1:] > 0
    return float((sums[1:][populated] / counts[1:][populated]).mean())


def test_radial_k_error_non_square_returns_finite() -> None:
    """REGRESSION: a non-square [1,1,16,32] pair yields a FINITE scalar.

    The hardened ``np.divide(where=counts>0)`` + ``np.nanmean`` path must
    never surface a NaN from an (empty) radial bin.
    """
    torch.manual_seed(0)
    metric = RadialKSpaceError()
    pred = torch.rand(1, 1, 16, 32)
    target = torch.rand(1, 1, 16, 32)

    out = metric(pred, target)

    assert isinstance(out, torch.Tensor)
    val = float(out)
    assert math.isfinite(val), f"non-square radial k-error was non-finite: {val}"
    assert val >= 0.0


def test_radial_k_error_non_square_matches_count_weighted_mean() -> None:
    """The returned scalar equals an independent count-weighted (populated-bin)
    mean — pins the value so a regression to naive ``sums / counts`` is caught.
    """
    torch.manual_seed(7)
    metric = RadialKSpaceError()
    pred = torch.rand(1, 1, 16, 32)
    target = torch.rand(1, 1, 16, 32)

    val = float(metric(pred, target))
    ref = _reference_radial_k_error(pred, target)

    assert math.isfinite(val)
    assert val == pytest.approx(ref, rel=1e-6, abs=1e-9)


def test_radial_k_error_non_square_perfect_recon_is_finite_zero() -> None:
    """A perfect (pred == target) reconstruction on a non-square grid yields a
    finite, ~0 error rather than NaN (the original symptom of the 0/0 bug)."""
    metric = RadialKSpaceError()
    target = torch.rand(1, 1, 16, 32)
    pred = target.clone()

    val = float(metric(pred, target))

    assert math.isfinite(val), f"perfect recon should be finite, got {val}"
    assert val == pytest.approx(0.0, abs=1e-6)


def test_radial_k_error_square_sanity_is_finite() -> None:
    """Square [1,1,32,32] sanity case still returns a finite non-negative scalar."""
    torch.manual_seed(1)
    metric = RadialKSpaceError()
    pred = torch.rand(1, 1, 32, 32)
    target = torch.rand(1, 1, 32, 32)

    val = float(metric(pred, target))

    assert math.isfinite(val)
    assert val >= 0.0


# ── FocalFrequencyMetric ─────────────────────────────────────────────────────
# The non-saturating spectral metric added for the MRIxFields2026 saturation
# work: it must (a) register with the right lower-is-better direction, (b) be
# ~0 on a perfect reconstruction, and (c) keep DISCRIMINATING as spectral
# corruption grows — the property SSIM lacks near its ceiling.


def test_focal_frequency_registered_and_lower_is_better() -> None:
    """Registered under name + aliases, and centrally declared lower-is-better."""
    from spectramr.core.metrics.metric_directions import METRIC_HIGHER_IS_BETTER
    from spectramr.core.metrics.registry import MetricsRegistry
    from spectramr.core.metrics.types import DEFAULT_METRIC_DIRECTIONS, MetricMode

    # Resolvable by canonical name and by alias.
    inst = MetricsRegistry.get("focal_frequency")
    assert isinstance(inst, FocalFrequencyMetric)
    assert isinstance(MetricsRegistry.get("ffl"), FocalFrequencyMetric)

    # Direction is a single source of truth in two consumer maps; both must
    # agree that lower is better (a spectral distance).
    assert METRIC_HIGHER_IS_BETTER["focal_frequency"] is False
    assert DEFAULT_METRIC_DIRECTIONS["focal_frequency"] is MetricMode.MIN
    # The decorator injects the boolean onto the class from the SSOT map.
    assert inst.higher_is_better is False


def test_focal_frequency_perfect_recon_is_finite_zero() -> None:
    """pred == target yields a finite ~0 spectral distance."""
    torch.manual_seed(3)
    metric = FocalFrequencyMetric()
    target = torch.rand(2, 1, 32, 32)
    pred = target.clone()

    val = float(metric(pred, target))

    assert math.isfinite(val), f"perfect recon should be finite, got {val}"
    assert val == pytest.approx(0.0, abs=1e-6)


def test_focal_frequency_monotone_in_spectral_corruption() -> None:
    """More spectral corruption -> strictly larger focal-frequency distance.

    This is the load-bearing property: unlike SSIM (which plateaus near the
    reconstruction ceiling), the focal-frequency distance keeps growing with
    added high-frequency error, so it can separate arms that SSIM cannot.
    """
    torch.manual_seed(11)
    metric = FocalFrequencyMetric()
    target = torch.rand(1, 1, 32, 32)
    noise = torch.rand_like(target)
    pred_close = target + 0.01 * noise
    pred_far = target + 0.20 * noise

    d_close = float(metric(pred_close, target))
    d_far = float(metric(pred_far, target))

    assert math.isfinite(d_close) and math.isfinite(d_far)
    assert d_close >= 0.0
    assert (
        d_far > d_close
    ), f"expected monotonic growth, got close={d_close}, far={d_far}"


# ── Shape-guard scope: distribution metrics free the SAMPLE axis only ──────


class TestDistributionMetricShapeGuard:
    """Regression for the over-broad shape opt-out.

    ``Wasserstein1D`` / ``SlicedWassersteinMetric`` / ``MMDMetric`` genuinely
    compare point sets of differing cardinality, but they were given
    ``REQUIRES_MATCHING_SHAPES = False``, which switches off shape checking
    ENTIRELY. Since ``_flatten_pair`` ravels both tensors, a ``[B, 2, H, W]``
    distribution head then scored a finite, plausible, meaningless number against a
    ``[B, 1, H, W]`` target — the silent-broadcast class the guard exists to stop,
    and the exact heteroscedastic ``[mean, logvar]`` case named in its docstring.
    The blast radius reached the training validation loop, because
    ``MetricsComputer._align_prediction`` also early-returns on that flag.
    """

    @pytest.mark.parametrize(
        "cls_name", ["Wasserstein1D", "SlicedWassersteinMetric", "MMDMetric"]
    )
    def test_uses_narrow_optout_not_the_blanket_one(self, cls_name):
        import spectramr.core.metrics.report_step_metrics as mod

        cls = getattr(mod, cls_name)
        assert cls.ALLOWS_UNEQUAL_SAMPLE_COUNT is True
        assert cls.REQUIRES_MATCHING_SHAPES is True, (
            f"{cls_name} disables shape checking wholesale; use "
            "ALLOWS_UNEQUAL_SAMPLE_COUNT to free only the sample axis."
        )

    def test_unequal_sample_count_is_allowed(self):
        from spectramr.core.metrics.report_step_metrics import Wasserstein1D

        value = Wasserstein1D()(torch.rand(8, 1, 16, 16), torch.rand(5, 1, 16, 16))
        assert torch.isfinite(torch.as_tensor(value))

    def test_channel_mismatch_still_raises(self):
        from spectramr.core.metrics.report_step_metrics import Wasserstein1D

        with pytest.raises(ValueError, match="trailing"):
            Wasserstein1D()(torch.rand(2, 2, 16, 16), torch.rand(2, 1, 16, 16))

    def test_spatial_mismatch_still_raises(self):
        from spectramr.core.metrics.report_step_metrics import Wasserstein1D

        with pytest.raises(ValueError, match="trailing"):
            Wasserstein1D()(torch.rand(1, 1, 128, 128), torch.rand(1, 1, 64, 64))


class TestUnsignedAgreementMetrics:
    """Darkening and brightening are both disagreement; the score must not see the sign."""

    @staticmethod
    def _score(key: str, pred: torch.Tensor, target: torch.Tensor) -> float:
        from spectramr.core.metrics.registry import MetricsRegistry

        return float(MetricsRegistry.get(key).compute_metric(pred, target))

    @pytest.fixture
    def target(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.rand(1, 1, 32, 32) + 0.5

    @pytest.mark.parametrize("key", ["abs_bland_altman_bias", "limits_of_agreement_halfwidth"])
    def test_darkening_and_brightening_move_the_score_the_same_way(self, key, target) -> None:
        clean = self._score(key, target, target)
        torch.manual_seed(1)
        noise = 0.05 * torch.randn_like(target)
        dark = self._score(key, 0.7 * target + noise, target) - clean
        bright = self._score(key, 1.3 * target + noise, target) - clean
        assert dark > 0 and bright > 0

    def test_the_signed_bias_is_why(self, target) -> None:
        dark = self._score("bland_altman_bias", 0.7 * target, target)
        bright = self._score("bland_altman_bias", 1.3 * target, target)
        assert dark < 0 and bright > 0

    @pytest.mark.parametrize("key", ["abs_bland_altman_bias", "limits_of_agreement_halfwidth"])
    def test_registered_lower_is_better(self, key) -> None:
        from spectramr.core.metrics.metric_directions import METRIC_HIGHER_IS_BETTER
        from spectramr.core.metrics.registry import MetricsRegistry

        assert METRIC_HIGHER_IS_BETTER[key] is False
        assert MetricsRegistry.get(key).higher_is_better is False
