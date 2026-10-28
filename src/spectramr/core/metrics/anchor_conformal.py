"""Per-voxel prediction intervals calibrated on a known fiducial.

A super-resolution network emits a point estimate. Deciding whether a structure
in that estimate is trustworthy needs an interval, and an interval is only worth
having if it comes with a coverage guarantee. Split conformal prediction
supplies one — distribution-free, finite-sample — but it needs a calibration set
whose residuals are **exchangeable** with the test residuals, and on real data
that means held-out ground truth, which is exactly what ULF-to-HF translation
does not have at inference time.

The fiducial supplies one. Its true value is known everywhere, so residuals on
marker support are computable **without any ground truth on the anatomy**.

That is not free. Marker residuals are exchangeable with anatomy residuals only
if the two populations are comparable, and a Gaussian probe is not a brain.
Conditioning on a difficulty score makes the assumption narrower and, crucially,
**testable**: stratify by difficulty, calibrate within each stratum, then measure
per-stratum coverage on paired validation where ground truth does exist. If a
stratum's empirical coverage misses its nominal level, exchangeability has
failed there and the interval carries no guarantee — a reportable outcome, not
something to patch around by loosening the stratification.

This is a different construction from
:mod:`spectramr.core.metrics.meta_evaluation.conformal`, which is conformal RISK
control for selecting metrics. Nothing is shared beyond the name.

References
----------
* V. Vovk, A. Gammerman, G. Shafer, *Algorithmic Learning in a Random World*,
  Springer, 2005.
* J. Lei et al., "Distribution-free predictive inference for regression,"
  *JASA* 113(523), 2018 — the split-conformal quantile correction used here.
* Y. Romano, M. Sesia, E. Candès, "Classification with valid and adaptive
  coverage," *NeurIPS*, 2020 — conditioning on a score to sharpen coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "AnchorConformalCalibrator",
    "CoverageReport",
    "conformal_quantile",
    "local_detail_score",
]


def conformal_quantile(scores: Tensor, alpha: float) -> Tensor:
    """The split-conformal quantile, with the finite-sample correction.

    Takes the ``ceil((n+1)(1-alpha)) / n`` empirical quantile rather than the
    plain ``1-alpha`` one. The correction is what makes coverage hold at finite
    ``n`` instead of asymptotically, and dropping it is the standard way a
    conformal implementation quietly under-covers on a small calibration set —
    which is the only kind this cohort has.

    Args:
        scores: ``[n]`` non-conformity scores (here, absolute residuals).
        alpha: Miscoverage level; the interval targets ``1 - alpha``.

    Returns:
        Scalar tensor. ``inf`` when ``n`` is too small for the level to be
        attainable, which is the honest answer: with ``n < 1/alpha - 1`` no
        finite interval can carry a ``1-alpha`` guarantee.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = scores.numel()
    if n == 0:
        raise ValueError("cannot calibrate on an empty score set")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return torch.tensor(float("inf"), device=scores.device, dtype=scores.dtype)
    return torch.sort(scores.reshape(-1)).values[k - 1]


def local_detail_score(x: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Difficulty proxy: local gradient magnitude, normalised per sample.

    Edges and texture are where a super-resolution network's residuals live, so
    an unconditional interval is far too wide in flat regions and too narrow at
    boundaries. This score is computable on the marker and on the anatomy
    alike, which is what lets one calibrate the other.

    Args:
        x: ``[B, C, H, W]``.

    Returns:
        ``[B, 1, H, W]`` in ``[0, 1]``, per-sample max-normalised.
    """
    if x.ndim != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(x.shape)}")
    m = x.abs() if torch.is_complex(x) else x
    if m.shape[1] > 1:
        m = m.mean(dim=1, keepdim=True)
    dy = torch.zeros_like(m)
    dx = torch.zeros_like(m)
    dy[..., 1:, :] = m[..., 1:, :] - m[..., :-1, :]
    dx[..., :, 1:] = m[..., :, 1:] - m[..., :, :-1]
    g = (dy.pow(2) + dx.pow(2)).sqrt()
    peak = g.amax(dim=(-2, -1), keepdim=True)
    return g / peak.clamp(min=eps)


@dataclass(frozen=True)
class CoverageReport:
    """Per-stratum coverage, and whether the guarantee survived it.

    Attributes:
        nominal: Target coverage, ``1 - alpha``.
        tolerance: Absolute slack allowed either side of nominal before a stratum fails.
        counts: ``[n_strata]`` calibration points per stratum.
        coverage: ``[n_strata]`` empirical coverage measured on held-out data.
        passed: ``[n_strata]`` booleans.
    """

    nominal: float
    tolerance: float
    counts: Tensor
    coverage: Tensor
    passed: Tensor

    @property
    def guaranteed(self) -> bool:
        """True only if EVERY stratum held.

        Deliberately not "most strata" or "on average". Marginal coverage can
        look fine while a whole difficulty regime is badly under-covered, and
        that regime is usually the interesting one.

        An UNMEASURED stratum (NaN) also fails, because a comparison against NaN
        is False. That is the intended reading: no evidence is not evidence of
        coverage. ``conformal_n_unmeasured`` distinguishes the two cases in the
        report so "not certified" is not mistaken for "certified to fail".
        """
        return bool(self.passed.all())

    def as_dict(self, prefix: str = "conformal") -> dict[str, float]:
        """Flat logging view. The per-stratum numbers are reported, not just the
        verdict, so a near-miss is distinguishable from a collapse."""
        measured = self.coverage[~torch.isnan(self.coverage)]
        out: dict[str, float] = {
            f"{prefix}_nominal": self.nominal,
            f"{prefix}_guaranteed": float(self.guaranteed),
            # Over MEASURED strata only. A plain `min` would return NaN as soon
            # as one stratum went unpopulated, hiding the coverage that WAS
            # measured behind a missing-data marker.
            f"{prefix}_worst_coverage": (
                float(measured.min()) if measured.numel() else float("nan")
            ),
            # Reported separately, because "not certified" and "certified to
            # fail" are different states and the gate collapses them to 0.
            f"{prefix}_n_unmeasured": float(self.coverage.numel() - measured.numel()),
        }
        for i in range(self.coverage.numel()):
            out[f"{prefix}_coverage_s{i}"] = float(self.coverage[i])
            out[f"{prefix}_n_calib_s{i}"] = float(self.counts[i])
        return out


class AnchorConformalCalibrator:
    """Split conformal calibrated on fiducial residuals, stratified by difficulty.

    Args:
        alpha: Miscoverage level. ``0.1`` targets 90% coverage.
        n_strata: Difficulty bins. More strata make the conditional guarantee
            sharper and each calibration set smaller; with a small marker
            support that trade-off is real, and an empty stratum raises rather
            than silently borrowing another's quantile.
        tolerance: Absolute slack below nominal before a stratum is failed.
    """

    def __init__(self, alpha: float = 0.1, n_strata: int = 4, tolerance: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_strata < 1:
            raise ValueError(f"n_strata must be >= 1, got {n_strata}")
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        self.alpha = alpha
        self.n_strata = n_strata
        self.tolerance = tolerance
        self._edges: Tensor | None = None
        self._quantiles: Tensor | None = None
        self._counts: Tensor | None = None

    # ── fitting ──────────────────────────────────────────────────────────

    @property
    def quantiles(self) -> Tensor:
        """``[n_strata]`` interval half-widths. Raises before ``fit``."""
        if self._quantiles is None:
            raise RuntimeError("calibrator has not been fitted")
        return self._quantiles

    @property
    def counts(self) -> Tensor:
        """``[n_strata]`` calibration points per stratum."""
        if self._counts is None:
            raise RuntimeError("calibrator has not been fitted")
        return self._counts

    def _assign(self, difficulty: Tensor) -> Tensor:
        if self._edges is None:
            raise RuntimeError("calibrator has not been fitted")
        return torch.bucketize(difficulty.reshape(-1), self._edges).clamp(0, self.n_strata - 1)

    def fit(
        self, residual: Tensor, difficulty: Tensor, support: Tensor | None = None
    ) -> AnchorConformalCalibrator:
        """Calibrate from residuals whose truth is KNOWN.

        Args:
            residual: Signed or absolute residuals; the absolute value is used.
                On the fiducial these are ``prediction - known_marker``.
            difficulty: Same shape, the conditioning score.
            support: Optional boolean/weight mask selecting where the residuals
                are meaningful — on a fiducial, the marker's own footprint.
                Everywhere else the "truth" is a zero background and the
                residuals would be trivially small, which would shrink every
                interval to nothing.
        """
        if residual.shape != difficulty.shape:
            raise ValueError(
                f"residual {tuple(residual.shape)} and difficulty "
                f"{tuple(difficulty.shape)} must match"
            )
        r = residual.abs().reshape(-1)
        d = difficulty.reshape(-1)
        if support is not None:
            keep = support.reshape(-1) > 0
            r, d = r[keep], d[keep]
        if r.numel() == 0:
            raise ValueError(
                "no calibration points inside the support. A conformal interval "
                "fitted on nothing would report a guarantee it cannot hold."
            )
        # Equal-mass strata: quantile edges of the observed difficulty, so no
        # stratum is empty by construction on the calibration set itself.
        qs = torch.linspace(0.0, 1.0, self.n_strata + 1, device=d.device)[1:-1]
        self._edges = (
            torch.quantile(d.float(), qs) if qs.numel() else torch.empty(0, device=d.device)
        )
        idx = self._assign(d)
        quantiles, counts = [], []
        for s in range(self.n_strata):
            scores = r[idx == s]
            if scores.numel() == 0:
                raise ValueError(
                    f"stratum {s} of {self.n_strata} received no calibration "
                    "points. Reduce n_strata rather than letting it borrow "
                    "another stratum's quantile, which would report a "
                    "conditional guarantee that was never conditioned."
                )
            quantiles.append(conformal_quantile(scores, self.alpha))
            counts.append(scores.numel())
        self._quantiles = torch.stack(quantiles)
        self._counts = torch.tensor(counts, device=r.device)
        return self

    # ── prediction ───────────────────────────────────────────────────────

    def half_width(self, difficulty: Tensor) -> Tensor:
        """Per-element interval half-width for a difficulty map."""
        q = self.quantiles.to(difficulty.device)
        return q[self._assign(difficulty)].reshape(difficulty.shape)

    def interval(self, prediction: Tensor, difficulty: Tensor) -> tuple[Tensor, Tensor]:
        """``(lower, upper)`` around ``prediction``, same shape."""
        if prediction.shape != difficulty.shape:
            raise ValueError(
                f"prediction {tuple(prediction.shape)} and difficulty "
                f"{tuple(difficulty.shape)} must match"
            )
        h = self.half_width(difficulty)
        return prediction - h, prediction + h

    # ── the test the guarantee is gated on ───────────────────────────────

    def coverage(
        self, residual: Tensor, difficulty: Tensor, support: Tensor | None = None
    ) -> CoverageReport:
        """Per-stratum empirical coverage on data the calibrator did NOT see.

        This is the exchangeability test, and it is allowed to fail. A stratum
        outside ``nominal +/- tolerance`` means fiducial residuals do not stand
        in for anatomy residuals at that difficulty. Over-coverage fails too: an
        interval that passes by being too wide carries no information. Reporting
        that is the correct outcome; widening the tolerance until it passes is not.
        """
        r = residual.abs().reshape(-1)
        d = difficulty.reshape(-1)
        if support is not None:
            keep = support.reshape(-1) > 0
            r, d = r[keep], d[keep]
        idx = self._assign(d)
        q = self.quantiles.to(r.device)
        nominal = 1.0 - self.alpha
        cov, counts = [], []
        for s in range(self.n_strata):
            sel = idx == s
            n = int(sel.sum())
            counts.append(n)
            # An unmeasured stratum is recorded as NaN, never as 1.0: "no data"
            # and "perfect coverage" must not look the same in a report.
            cov.append(float((r[sel] <= q[s]).float().mean()) if n else float("nan"))
        coverage = torch.tensor(cov)
        return CoverageReport(
            nominal=nominal,
            tolerance=self.tolerance,
            counts=torch.tensor(counts),
            coverage=coverage,
            passed=(coverage - nominal).abs() <= self.tolerance,
        )
