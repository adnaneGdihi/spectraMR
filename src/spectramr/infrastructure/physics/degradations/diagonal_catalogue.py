"""Registry degradations that are known diagonal masks or filters in k-space, as experiments.

Each listed operator was probed to be deterministic, linear and diagonal in centred
k-space with a real, nonnegative transfer ``h`` (the probe is rerun on every build). Where
it acts relative to the thermal noise decides its information:

* ``before_noise``: acquisition masks and signal weighting during the echo train. The data
  are ``h x + sigma n`` on the acquired coefficients, so the information is
  ``h^2 / sigma^2``. ``h`` is divided by its peak (noise referenced to the strongest
  echo), because some kernels raise the signal above 1 (``tse_t2_blur`` reaches e at
  theta = 1). Taken literally, that would certify a blur as more informative than noise
  alone. The data are then shown at unit DC gain. Without that, the peak normalisation
  dimmed ``tse_t2_blur`` to 0.37 of its brightness, and squared error read the dimming
  as lost information.
* ``after_noise``: the reconstruction filter and the receiver gain stage, which filter the
  noise with the signal. The data are ``h (x + sigma n)``, so the information is
  ``1 / sigma^2`` wherever ``h > 0``. Such an operator is Blackwell-equivalent to noise
  alone, which is what makes it a test of display sensitivity.

Operators that multiply by a pure phase (motion shifts, drifts, ghosts) are left out:
with the phase known they would be equivalent to noise alone, but the reader does not
know the trajectory, so they belong with the nuisance-mixture experiments.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from spectramr.core.metrics.meta_evaluation.blackwell import DiagonalExperiment
from spectramr.infrastructure.physics.degradations.registry import apply_degradation
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

#: Audited noise placement of every admitted code (checked against each declaration).
PLACEMENT: dict[str, str] = {
    "cartesian_undersamp": "before_noise",
    "vd_cartesian": "before_noise",
    "vd_undersamp": "before_noise",
    "radial_undersamp": "before_noise",
    "partial_fourier": "before_noise",
    "zero_fill_interp": "before_noise",  # reduced matrix: the outer lines are never acquired
    "gibbs": "before_noise",  # truncated acquisition radius
    "t2star_blur": "before_noise",
    "t1_shot_blur": "before_noise",
    "tse_t2_blur": "before_noise",
    "sar_limited_flip": "before_noise",
    "slice_pve": "before_noise",  # a signal-averaging proxy, applied as a k-space window
    "recon_apodization": "after_noise",
    "receiver_gain_drift": "after_noise",
}
DEFAULT_CODES = tuple(PLACEMENT)
#: Probe residual tolerance; float32 round-off sits near 1e-6.
PROBE_TOL = 1e-4
#: ``log h`` is snapped to this grid, so probe round-off cannot split equal values.
LOG_STEP = 1e-4


def probe_transfer(
    code: str, theta: float, seed: int, shape: tuple[int, int]
) -> tuple[torch.Tensor, dict[str, float]]:
    """The snapped real transfer ``h`` of ``code`` at ``theta``, and the probe residuals.

    Raises:
        ValueError: the operator is not deterministic, linear, diagonal or real
            nonnegative within :data:`PROBE_TOL` at this setting.
    """
    gen = torch.Generator().manual_seed(1234)
    x1, x2 = (torch.randn((1, 1, *shape), dtype=torch.complex64, generator=gen) for _ in "ab")

    def op(x: torch.Tensor) -> torch.Tensor:
        return apply_degradation(code, x, theta, seed=seed)

    y1, y2, y12 = op(x1), op(x2), op(x1 + x2)
    if not y1.is_complex():
        raise ValueError(f"{code}: returns a real image, so it is not a linear k-space filter")
    h1 = (fft2c(y1) / fft2c(x1))[0, 0].to(torch.complex128)
    h2 = (fft2c(y2) / fft2c(x2))[0, 0].to(torch.complex128)
    peak = float(h1.abs().max())
    res = {
        "determinism": float((op(x1) - y1).abs().max() / y1.abs().max()),
        "linearity": float((y12 - y1 - y2).norm() / y12.norm()),
        "diagonality": float((h1 - h2).abs().max() / peak),
        "imaginary": float(h1.imag.abs().max() / peak),
        "negative": float((-h1.real).clamp_min(0).max() / peak),
    }
    bad = {k: v for k, v in res.items() if not v <= PROBE_TOL}
    if bad:
        raise ValueError(f"{code} at theta={theta}: not a real diagonal filter, residuals {bad}")
    h = ((h1 + h2) / 2).real.clamp_min(0.0)
    h[h < PROBE_TOL * peak] = 0.0
    pos = h > 0
    h[pos] = torch.exp(torch.round(torch.log(h[pos]) / LOG_STEP) * LOG_STEP)
    return h, res


def check_on_contents(
    code: str, theta: float, seed: int, transfer: torch.Tensor, contents: Sequence[torch.Tensor]
) -> float:
    """Largest relative gap between ``code`` and its probed transfer on the real contents.

    Random complex probes can miss a branch that depends on the image, so the transfer is
    re-checked on what the ranker will actually evaluate.

    Raises:
        ValueError: the gap exceeds :data:`PROBE_TOL` on some content.
    """
    worst = 0.0
    for x in contents:
        xc = x.to(torch.complex64)[None, None]
        y = apply_degradation(code, xc, theta, seed=seed)[0, 0]
        pred = ifft2c(transfer.to(device=x.device, dtype=torch.float32) * fft2c(xc[0, 0]))
        worst = max(worst, float((y - pred).norm() / y.norm().clamp_min(1e-12)))
    if worst > PROBE_TOL:
        raise ValueError(
            f"{code} at theta={theta}: probed transfer misses a content by {worst:.1e}"
        )
    return worst


def _registry_experiment(
    code: str, theta: float, h: torch.Tensor, sigma: float
) -> tuple[DiagonalExperiment, float]:
    peak = float(h.max())
    acquired = h > 0
    neg_inf = torch.full_like(h, -math.inf)
    if PLACEMENT[code] == "before_noise":
        rel = h / peak
        log_info = torch.where(acquired, 2 * torch.log(rel) - 2 * math.log(sigma), neg_inf)
        # Displayed at unit DC gain: a known global rescale is invertible, so it leaves the
        # information unchanged and only stops the peak normalisation darkening the image.
        dc = float(h[h.shape[0] // 2, h.shape[1] // 2])
        display = peak / dc if dc > 0 else 1.0
        signal = rel * display
        noise = acquired.double() * display
    else:
        signal, noise = h, h
        log_info = torch.where(acquired, torch.full_like(h, -2 * math.log(sigma)), neg_inf)
    exp = DiagonalExperiment(f"{code}_{theta:g}", code, theta, log_info, signal, noise, sigma)
    return exp, peak


def registry_catalogue(
    contents: Sequence[torch.Tensor],
    *,
    codes: Sequence[str] = DEFAULT_CODES,
    thetas: Sequence[float] = (0.33, 0.67, 1.0),
    seed: int = 3,
    base_noise: float = 0.1,
    noise_levels: Sequence[float] = (0.05, 0.1, 0.15, 0.2),
) -> tuple[list[DiagonalExperiment], dict[str, object]]:
    """Noise ladder plus every admitted code at every theta, at ``base_noise``.

    Returns the experiments and their provenance: per entry the placement, the peak gain
    before normalisation and the probe residuals.
    """
    unknown = [c for c in codes if c not in PLACEMENT]
    if unknown:
        raise ValueError(
            f"codes {unknown} are not audited diagonal filters; known: {DEFAULT_CODES}"
        )
    if not contents or any(c.dim() != 2 for c in contents):
        raise ValueError("registry_catalogue needs 2-D (H, W) contents")
    shape = (int(contents[0].shape[-2]), int(contents[0].shape[-1]))
    exps = [
        DiagonalExperiment(
            f"noise_{s:g}", "noise", s, torch.full(shape, -2 * math.log(s), dtype=torch.float64)
        )
        for s in noise_levels
    ]
    entries: list[dict[str, object]] = []
    for code in codes:
        for theta in thetas:
            h, res = probe_transfer(code, theta, seed, shape)
            gap = check_on_contents(code, theta, seed, h, contents)
            exp, peak = _registry_experiment(code, theta, h, base_noise)
            exps.append(exp)
            entries.append(
                {
                    "name": exp.name,
                    "placement": PLACEMENT[code],
                    "max_gain": peak,
                    "content_gap": gap,
                    **res,
                }
            )
    provenance: dict[str, object] = {
        "catalogue": "registry_diagonal",
        "seed": seed,
        "thetas": list(thetas),
        "base_noise": base_noise,
        "noise_levels": list(noise_levels),
        "gain_convention": "before_noise information divided by the peak gain; shown at unit DC gain",
        "entries": entries,
    }
    return exps, provenance


__all__ = [
    "DEFAULT_CODES",
    "PLACEMENT",
    "check_on_contents",
    "probe_transfer",
    "registry_catalogue",
]
