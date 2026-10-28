"""Derivative-free fitting of a compounded degradation chain to a quality target.

scipy, not CMA-ES: ``scipy>=1.17`` is already a hard dependency and ``cma`` is not,
and ``differential_evolution`` takes the ``[0, 1]^K`` box natively with a seed. Adding
a dependency for this would be gratuitous.

Derivative-free rather than gradient-based because the interesting axes are not
differentiable in theta: line masks, spoke counts and partial-Fourier fractions are
discrete, and those are precisely the axes that dominate a low-field quality gap.

This module fits ATTRIBUTES only; it never sees a header. Voxel geometry is a header
fact to be imposed on the target grid, not a quantity to optimise -- fitting it
against a sharpness proxy would let a blur term absorb a geometry error and still
report a good residual.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize

from spectramr.infrastructure.physics.degradation_chain import ChainLink, DegradationChain
from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY
from spectramr.infrastructure.physics.quality_descriptors import measure_attributes

__all__ = [
    "DegenerateFitError",
    "FitResult",
    "acquisition_warm_start",
    "fit_chain",
    "warm_start_theta",
]

_METHODS = ("differential_evolution", "nelder_mead")


class DegenerateFitError(RuntimeError):
    """The fit ran but did not do its job -- the mechanism-fires guard (pitfall #16).

    Distinct from a crash: the optimiser converged, produced numbers, and would have
    emitted a plausible-looking calibration. This is the error that stops a chain
    which moved the image but not toward the target from being written to disk.
    """


@dataclass(frozen=True, slots=True)
class FitResult:
    """A fitted chain plus everything needed to audit the fit."""

    chain: DegradationChain
    achieved: Mapping[str, float]
    target: Mapping[str, float]
    weights: Mapping[str, float]
    residual: float
    initial_residual: float
    gap_closed: float
    n_evals: int
    method: str
    seed: int


def warm_start_theta(axis: str, target_value: float, *, param: str | None = None) -> float:
    """Invert the axis's declared affine severity; clamped to ``[0, 1]``.

    ``PhysicalParam.value_at`` is affine, so this inverse is exact where the target
    lies inside the declared range and saturates outside it. Handles decreasing
    parameters (e.g. ``complex_gaussian``'s SNR, 40 dB -> 2 dB) because the span is
    signed.

    Only meaningful when ``target_value`` is expressed in the axis's OWN declared
    units. It is not a general warm start for arbitrary no-reference attributes --
    inverting a blur severity in pixels against a Tenengrad variance is a category
    error.
    """
    spec = DEGRADATION_REGISTRY[axis].severity
    if param is None:
        p = spec.primary
    else:
        p = next((q for q in spec.params if q.name == param), None)
        if p is None:
            declared = ", ".join(q.name for q in spec.params)
            raise KeyError(
                f"{param!r} is not a declared parameter of {axis!r} (declared: {declared})"
            )
    span = p.at_theta_max - p.at_theta_min
    if span == 0.0:
        return 0.0
    theta = (float(target_value) - p.at_theta_min) / span
    return float(min(1.0, max(0.0, theta)))


def acquisition_warm_start(
    axes: Sequence[str],
    snr_delta_db: float,
    *,
    default: float = 0.5,
) -> list[float]:
    """``theta0`` with the noise axis set from an acquisition-derived SNR prediction.

    The noise axis is DISCOVERED from the registry -- any axis whose declared primary
    parameter is an SNR in dB -- rather than matched against a hardcoded name. A
    hardcoded ``"complex_gaussian"`` would silently stop applying the prior the moment
    a chain used a different noise operator, and nothing would report it.

    ``snr_delta_db`` is the predicted change relative to the axis's own CLEAN endpoint
    (``at_theta_min``), so a -20 dB prediction on a 40 dB clean endpoint targets 20 dB
    and inverts to the theta that realises it. Non-noise axes keep ``default``: the
    acquisition says nothing about how much motion or blur to expect.
    """
    out: list[float] = []
    for axis in axes:
        primary = DEGRADATION_REGISTRY[axis].severity.primary
        if primary.name == "snr" and primary.units == "dB":
            out.append(warm_start_theta(axis, primary.at_theta_min + float(snr_delta_db)))
        else:
            out.append(float(default))
    return out


def _weighted_residual(
    achieved: Mapping[str, float],
    target: Mapping[str, float],
    weights: Mapping[str, float],
    keys: Sequence[str],
) -> float:
    return float(
        np.sqrt(sum(weights[k] * (float(achieved[k]) - float(target[k])) ** 2 for k in keys))
    )


def _default_weights(target: Mapping[str, float], keys: Sequence[str]) -> dict[str, float]:
    """Reciprocal-square-magnitude weighting, so no attribute dominates by unit scale.

    Tenengrad variance and a noise-colour statistic differ by orders of magnitude; an
    unweighted L2 would optimise whichever happens to be numerically largest and
    ignore the rest.
    """
    return {k: 1.0 / max(abs(float(target[k])), 1e-12) ** 2 for k in keys}


def fit_chain(
    x_hq: torch.Tensor,
    *,
    axes: Sequence[str],
    target: Mapping[str, float],
    attributes: Sequence[str],
    weights: Mapping[str, float] | None = None,
    theta0: Sequence[float] | None = None,
    seed: int = 0,
    max_evals: int = 400,
    method: str = "differential_evolution",
    min_gap_closed: float = 0.5,
    max_residual: float | None = None,
) -> FitResult:
    """Fit per-axis severities so ``x_hq``'s attributes match ``target``.

    Args:
        x_hq: clean volume, ``[S, H, W]`` or ``[H, W]``.
        axes: DEGRADATION_REGISTRY axes composing the chain, in application order.
        target: attribute name -> desired value.
        attributes: the matched attributes (keys of ``target``).
        weights: per-attribute residual weights; defaults to reciprocal-square target.
        theta0: optional warm start in ``[0, 1]^K``; defaults to mid-box.
        min_gap_closed: minimum fraction of the initial gap the fit must close.
        max_residual: absolute ceiling on the final weighted residual, or ``None``.

    Raises:
        DegenerateFitError: the chain ran but closed less than ``min_gap_closed`` of
            the initial distance, or every theta saturated on a box bound.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown fit method {method!r}. Valid: {list(_METHODS)}.")

    keys = list(attributes)
    missing = [k for k in keys if k not in target]
    if missing:
        raise ValueError(
            f"target is missing a value for {missing}; every matched attribute needs "
            "a target, or the residual silently ignores it."
        )

    if theta0 is not None:
        if len(theta0) != len(axes):
            raise ValueError(f"theta0 length {len(theta0)} does not match the {len(axes)} axes")
        if not all(0.0 <= float(t) <= 1.0 for t in theta0):
            raise ValueError(f"theta0 entries must lie in [0, 1]; got {list(theta0)}")

    w = dict(weights) if weights is not None else _default_weights(target, keys)

    base = DegradationChain(links=tuple(ChainLink(axis=a, theta=0.5) for a in axes))
    vol = x_hq if x_hq.dim() == 3 else x_hq.unsqueeze(0)
    batched = vol.unsqueeze(1)

    calls = {"n": 0}

    def _evaluate(thetas: Sequence[float]) -> float:
        calls["n"] += 1
        chain = base.with_thetas([float(t) for t in thetas])
        out = chain.apply(batched, seed=seed).squeeze(1).abs()
        return _weighted_residual(measure_attributes(out, attributes=keys), target, w, keys)

    initial_residual = _weighted_residual(
        measure_attributes(vol.abs(), attributes=keys), target, w, keys
    )

    x0 = (
        np.asarray(theta0, dtype=float)
        if theta0 is not None
        else np.full(len(axes), 0.5, dtype=float)
    )
    bounds = [(0.0, 1.0)] * len(axes)

    if method == "differential_evolution":
        popsize = 5
        maxiter = max(1, max_evals // max(1, popsize * len(axes)))
        opt = differential_evolution(
            _evaluate,
            bounds=bounds,
            # `rng`, not the legacy `seed` alias: scipy 1.15 introduced `rng` as the
            # canonical spelling and `seed` is retained only for back-compat.
            rng=seed,
            maxiter=maxiter,
            popsize=popsize,
            polish=False,
            x0=x0,
        )
        best = np.asarray(opt.x, dtype=float)
    else:
        opt = minimize(
            _evaluate,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxfev": max_evals},
        )
        best = np.asarray(opt.x, dtype=float)

    # Local polish from the global optimum. The objective is deterministic for a
    # fixed seed, so re-evaluating `best` is a fair comparison.
    best_value = _evaluate(best.tolist())
    polished = minimize(
        _evaluate,
        best,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxfev": max(20, max_evals // 4)},
    )
    if float(polished.fun) < best_value:
        best = np.asarray(polished.x, dtype=float)

    fitted = base.with_thetas([float(t) for t in best])
    out = fitted.apply(batched, seed=seed).squeeze(1).abs()
    achieved = measure_attributes(out, attributes=keys)
    residual = _weighted_residual(achieved, target, w, keys)

    gap_closed = (
        0.0 if initial_residual <= 0.0 else float(max(0.0, 1.0 - residual / initial_residual))
    )

    if gap_closed < min_gap_closed:
        raise DegenerateFitError(
            f"the fitted chain closed only {gap_closed:.1%} of the initial gap "
            f"(residual {initial_residual:.4g} -> {residual:.4g}), below the required "
            f"{min_gap_closed:.0%}. Either the target is unreachable by degradation, "
            f"or the declared axis set {list(axes)} cannot express it."
        )
    if max_residual is not None and residual > max_residual:
        # Relative: the weights are 1/target^2, so the residual is a relative error norm.
        raise DegenerateFitError(
            f"the fitted chain's residual {residual:.4g} exceeds the absolute ceiling "
            f"{max_residual:.4g}. The gap closed ({gap_closed:.1%}) is relative only, so "
            "a fit can halve a large gap and still miss the target."
        )
    if all(t <= 1e-6 or t >= 1.0 - 1e-6 for t in best):
        raise DegenerateFitError(
            f"every fitted theta saturated on a box bound ({list(best)}), so the "
            "search found no interior optimum and the chain is not calibrated."
        )

    return FitResult(
        chain=fitted,
        achieved=achieved,
        target=dict(target),
        weights=w,
        residual=residual,
        initial_residual=initial_residual,
        gap_closed=gap_closed,
        n_evals=calls["n"],
        method=method,
        seed=seed,
    )
