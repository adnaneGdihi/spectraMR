"""Metric values of every catalogue experiment on every clean content.

A violation is a statement about ``E q(Y_a) - E q(Y_b)``, which depends on each experiment's
marginal law alone. Each experiment is therefore simulated once per content and draw.
Draw ``r`` reuses one noise array across all experiments (common random numbers), which
recovers most of the variance reduction that coupling the pair would give.

The display-canonicalised value ``q#`` (Theorem C1.5, Gaussian-blur class) filters the
complex data in k-space before the magnitude display. That class is closed under
composition (variances add). The sup is taken over a mean across draws, never per draw,
because a per-draw sup is biased upward and the bias grows with the noise level. The
filter is chosen on the even draws and scored on the odd ones: choosing and scoring on
the same draws would carry the same bias, largest for the noisiest experiments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

from .catalogue import DiagonalExperiment, gaussian_log_transfer, radial_frequency

MetricFn = Callable[..., float]
#: ``(xk, experiment, rms) -> (B, H, W)`` post-filters for one content and experiment;
#: entry 0 must be the identity.
EnvelopeFn = Callable[[torch.Tensor, Any, float], torch.Tensor]
#: ``(xk, experiment, noise, rms, envelope) -> (B, R, H, W)`` magnitude displays.
Realiser = Callable[[torch.Tensor, Any, torch.Tensor, float, torch.Tensor], torch.Tensor]


def displays(
    xk: torch.Tensor,
    experiment: DiagonalExperiment,
    noise: torch.Tensor,
    rms: float,
    envelope: torch.Tensor,
) -> torch.Tensor:
    """Magnitude displays ``(B, R, H, W)`` for ``B`` envelope filters and ``R`` draws.

    Args:
        xk: centred k-space of the clean content, ``(H, W)`` complex.
        experiment: the experiment to realise.
        noise: unit ``CN(0, 1)`` draws, ``(R, H, W)``.
        rms: the content's RMS; noise is ``experiment.noise_std * rms``.
        envelope: k-space transfer of each post-processing filter, ``(B, H, W)`` real.
    """
    h, g = (t.to(device=xk.device, dtype=torch.float32) for t in experiment.transfers())
    sigma = experiment.noise_std * rms
    y = (h * xk)[None] + (sigma * g) * noise  # unacquired coefficients stay zero
    filtered = envelope[:, None].to(y.dtype) * y[None]
    return ifft2c(filtered).abs()


def blur_envelope(
    shape: tuple[int, int], widths: Sequence[float], device: torch.device | None
) -> torch.Tensor:
    """k-space transfers of the Gaussian post-processing blurs; width 0 is the identity."""
    rho = radial_frequency(shape, device)
    return torch.stack([torch.exp(gaussian_log_transfer(rho, w)) for w in widths]).float()


def wiener_envelope(lambdas: Sequence[float]) -> EnvelopeFn:
    """Oracle Wiener post-filters ``conj(h) p / (|h|^2 p + lambda sigma^2 |g|^2)``, identity first.

    ``p = |x_k|^2`` is the clean content's own power spectrum. That makes this an oracle,
    per-image envelope (``pi = delta_x``). At ``lambda = 1`` it is the exact oracle-LSI
    envelope of squared error (Theorem C1.5), which no Gaussian blur can match on a
    post-noise low-pass. The family is not closed under composition, so for other
    metrics the canonicalised rate is a diagnostic, not the least monotone majorant.
    Only diagonal experiments have the ``h`` and ``g`` it needs.
    """
    lams = [float(x) for x in lambdas]
    if not lams or any(x <= 0 for x in lams):
        raise ValueError(f"Wiener lambdas must be positive, got {lambdas}")

    def envelope(xk: torch.Tensor, exp: Any, rms: float) -> torch.Tensor:
        h, g = (t.to(device=xk.device, dtype=torch.float64) for t in exp.transfers())
        power = xk.abs().double().pow(2)
        noise = (exp.noise_std * rms) ** 2 * g.pow(2)
        filters = [torch.ones_like(power)]
        for lam in lams:
            den = h.pow(2) * power + lam * noise
            filters.append(torch.where(den > 0, h * power / den.clamp_min(1e-300), 0.0))
        return torch.stack(filters).float()

    return envelope


def _call(fn: MetricFn, image: torch.Tensor, clean: torch.Tensor, context: object) -> float:
    if context is not None:
        return float(fn(image[None], clean[None], context=context))
    return float(fn(image[None], clean[None]))


def metric_table(
    metrics: Mapping[str, MetricFn],
    orientation: Mapping[str, float],
    contents: Sequence[torch.Tensor],
    experiments: Sequence[Any],
    *,
    n_draws: int,
    envelope_widths: Sequence[float],
    seed: int,
    contexts: Mapping[str, Sequence[object]] | None = None,
    realise: Realiser = displays,
    noise_dims: tuple[int, ...] = (),
    envelope_fn: EnvelopeFn | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Oriented (higher is better) metric values per content, experiment and draw.

    Returns ``(raw, sharp)`` of float64 tensors. ``raw[m]`` is ``(C, E, R)`` under the
    identity display. ``sharp[m]`` is ``(C, E, R // 2)``: the odd draws under the filter
    whose mean over the even draws is largest, so ``sharp.mean(-1)`` estimates ``q#``.
    With a single envelope width it is ``raw``. A NaN from a metric call propagates into
    that content and experiment only.

    Args:
        contents: clean references, each ``(H, W)`` real or complex, all one shape.
        contexts: per metric, one ``MetricContext`` per content for metrics that need one.
        realise: builds the displays of one experiment; :func:`displays` for the diagonal
            class. Every realiser shares this function's draws, envelope and split.
        noise_dims: extra leading noise axes per draw (the coils of a multi-coil realiser).
        envelope_fn: per-content and per-experiment post-filters (identity first), in place
            of the shared Gaussian blurs of ``envelope_widths``, which are then ignored
            apart from their identity check.
    """
    if not contents:
        raise ValueError("metric_table needs at least one content")
    shape = (int(contents[0].shape[-2]), int(contents[0].shape[-1]))
    if any(tuple(c.shape) != shape for c in contents):
        raise ValueError("every content must share one (H, W) shape")
    widths = list(envelope_widths)
    if not widths or widths[0] != 0.0:
        raise ValueError("envelope_widths must start with 0.0 (the identity display)")
    device = contents[0].device
    shared = blur_envelope(shape, widths, device)
    n_filters = (
        len(widths)
        if envelope_fn is None
        else int(envelope_fn(fft2c(contents[0]), experiments[0], 1.0).shape[0])
    )
    if n_filters > 1 and n_draws < 2:
        raise ValueError(
            "choosing an envelope filter needs n_draws >= 2 (one to choose, one to score)"
        )
    names = list(metrics)
    n_c, n_e = len(contents), len(experiments)
    full = (n_c, n_e, n_draws)
    raw = {m: torch.full(full, float("nan"), dtype=torch.float64) for m in names}
    held = full if n_filters == 1 else (n_c, n_e, n_draws // 2)
    sharp = {m: torch.full(held, float("nan"), dtype=torch.float64) for m in names}
    gen = torch.Generator(device=device)
    for c, clean in enumerate(contents):
        gen.manual_seed(seed + c)
        noise = torch.randn(
            (n_draws, *noise_dims, *shape), dtype=torch.complex64, generator=gen, device=device
        )
        xk = fft2c(clean)
        ref = clean.abs().float()
        rms = float(ref.pow(2).mean().sqrt())
        for e, exp in enumerate(experiments):
            envelope = shared if envelope_fn is None else envelope_fn(xk, exp, rms)
            disp = realise(xk, exp, noise, rms, envelope)
            for m in names:
                ctx = contexts[m][c] if contexts and m in contexts else None
                vals = (
                    torch.tensor(
                        [
                            [_call(metrics[m], disp[b, r], ref, ctx) for r in range(n_draws)]
                            for b in range(n_filters)
                        ],
                        dtype=torch.float64,
                    )
                    * orientation[m]
                )
                raw[m][c, e] = vals[0]
                if n_filters == 1:
                    sharp[m][c, e] = vals[0]
                    continue
                choose = vals[:, 0::2].mean(dim=1)
                if not torch.isnan(choose).any():
                    sharp[m][c, e] = vals[int(choose.argmax()), 1::2]
    return raw, sharp


def oracle_lsi_control(
    contents: Sequence[torch.Tensor], experiments: Sequence[DiagonalExperiment]
) -> torch.Tensor:
    """Closed-form oracle-LSI envelope of squared error, ``(C, E)`` float64 (Theorem C1.5).

    ``-sum_k |x_k|^2 / (1 + |x_k|^2 n_k / rms^2)``: concave and Loewner-isotone in the
    profile, so it is coherent by construction and serves as the Bayes-type control.
    """
    out = torch.empty((len(contents), len(experiments)), dtype=torch.float64)
    for c, clean in enumerate(contents):
        p = fft2c(clean).abs().double().pow(2).cpu()
        rms2 = float(clean.abs().double().pow(2).mean())
        for e, exp in enumerate(experiments):
            n = torch.exp(exp.log_info.cpu()) / rms2
            out[c, e] = -(p / (1.0 + p * n)).sum()
    return out


__all__ = [
    "EnvelopeFn",
    "blur_envelope",
    "displays",
    "metric_table",
    "oracle_lsi_control",
    "wiener_envelope",
]
