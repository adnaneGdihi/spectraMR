"""Repetition-split risk: a reference-free error estimate for tissue-map models (R1).

With the input built from one repetition and the target from an independent one
(``M4RawRepetitionTripletDataset``), the target's noise is independent of the prediction.
In the power domain the RSS magnitude's noncentral-chi bias is exact: ``E[M^2] = A^2 + P``,
with ``P = f C`` in noise units for ``C`` coils and a fraction ``f`` of k-space acquired. So
``M_b^2 - P`` is an unbiased target for the noise-free power ``A^2``, and

    E || (g r)^2 - (M_b^2 - P) ||^2  =  || (g r)^2 - A^2 ||^2  +  Var(M_b^2)

ranks models and settings (unroll depth, prior weight) by their true power error, without
a reference map. The gain ``g`` is fitted on the INPUT repetition only, so the target is
never read to make the prediction. ``val_rep_split_bias`` (the mean signed residual)
should sit near zero for a calibrated model.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

_NAMES = ("t1", "t2", "flair")


def power_residual(
    render: torch.Tensor,
    xk: torch.Tensor,
    yk: torch.Tensor,
    power: torch.Tensor,
    *,
    gain2: torch.Tensor | None = None,
    signal_floor: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked ``(g r)^2 - (M_b^2 - P)`` for one contrast, and its input-built mask.

    Args:
        render: ``(B, 1, H, W)`` render of the contrast.
        xk: ``(B, 1, H, W)`` input repetition: builds the mask and, with ``gain2`` None,
            the closed-form power-domain gain.
        yk: ``(B, 1, H, W)`` independent target repetition.
        power: ``(B,)`` expected noise power ``P = f C`` of the target series.
        gain2: a known squared gain (calibrated mode); None fits it on the input.
    """
    power = power.to(yk).view(-1, 1, 1, 1)  # device and dtype of the images
    w = (xk > signal_floor).to(render.dtype)
    if gain2 is None:
        gain2 = power_gain2(render, xk, power, w)
    return (gain2 * render * render - (yk**2 - power)) * w, w


def power_gain2(
    render: torch.Tensor, xk: torch.Tensor, power: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """``(B, 1, 1, 1)`` squared gain fitted to the INPUT repetition in the power domain.

    A magnitude fit to the input would inherit its noise bias and push the predicted
    power up by about ``P``.
    """
    r2 = render * render
    power = power.to(xk).view(-1, 1, 1, 1)
    num = (r2 * (xk**2 - power) * w).flatten(1).sum(1)
    return (num / (r2 * r2 * w).flatten(1).sum(1).clamp_min(1e-12)).view(-1, 1, 1, 1)


def power_projected_gains(
    renders: Sequence[torch.Tensor],
    x: torch.Tensor,
    noise_power: torch.Tensor,
    *,
    signal_floor: float = 3.0,
) -> torch.Tensor:
    """``(B, 3)`` per-image magnitude gains ``sqrt(gain2)``, fitted on the input repetition only.

    The gauge prior reads these on a repetition split, so it never sees the target.
    """
    gains = []
    for k, r in enumerate(renders):
        xk = x[:, k : k + 1]
        w = (xk > signal_floor).to(r.dtype)
        gains.append(power_gain2(r, xk, noise_power[:, k], w).clamp_min(0.0).sqrt().flatten())
    return torch.stack(gains, dim=1)


def repetition_loss(
    render: torch.Tensor,
    xk: torch.Tensor,
    yk: torch.Tensor,
    power: torch.Tensor,
    *,
    gain2: torch.Tensor | None = None,
    signal_floor: float = 3.0,
) -> torch.Tensor:
    """Training objective on a repetition split: the masked mean of the weighted power residual.

    Its minimiser is the noise-free power ``A^2``, where a magnitude loss against ``M_b``
    converges to ``E[M_b]``, above ``A`` at low SNR (#2976). The residual is divided by
    ``2 sqrt(max(x^2 - P, floor^2))`` (delta method), which brings it back to magnitude
    units, so a prior weight keeps the meaning it had against the magnitude loss. The
    weight is built from the input repetition, independent of the target, so the
    per-voxel minimiser stays ``A^2``.
    """
    resid, w = power_residual(render, xk, yk, power, gain2=gain2, signal_floor=signal_floor)
    p = power.to(xk).view(-1, 1, 1, 1)
    scale = 2.0 * (xk * xk - p).clamp_min(signal_floor**2).sqrt()
    return (resid / scale).pow(2).sum() / w.sum().clamp_min(1.0)


def repetition_risk(
    renders: Sequence[torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    noise_power: torch.Tensor,
    *,
    signal_floor: float = 3.0,
) -> dict[str, float]:
    """Power-domain risk and bias per contrast, against an independent repetition.

    Args:
        renders: one ``(B, 1, H, W)`` render per contrast.
        x: ``(B, 3, H, W)`` input repetition (noise units): fits the gain and the mask.
        y: ``(B, 3, H, W)`` independent target repetition (noise units).
        noise_power: ``(B, 3)`` expected noise power ``P = f C`` of each target series.
    """
    out: dict[str, float] = {}
    risks, biases = [], []
    for k, (r, name) in enumerate(zip(renders, _NAMES, strict=True)):
        resid, w = power_residual(
            r, x[:, k : k + 1], y[:, k : k + 1], noise_power[:, k], signal_floor=signal_floor
        )
        n = w.sum().clamp_min(1.0)
        risk, bias = float(resid.pow(2).sum() / n), float(resid.sum() / n)
        out[f"val_rep_split_risk_{name}"], out[f"val_rep_split_bias_{name}"] = risk, bias
        risks.append(risk)
        biases.append(bias)
    out["val_rep_split_risk"] = sum(risks) / len(risks)
    out["val_rep_split_bias"] = sum(biases) / len(biases)
    return out
