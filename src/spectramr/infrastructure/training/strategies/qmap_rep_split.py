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
        xk, yk = x[:, k : k + 1], y[:, k : k + 1]
        power = noise_power[:, k].to(yk).view(-1, 1, 1, 1)  # device and dtype of the images
        w = (xk > signal_floor).to(r.dtype)
        # The gain is fitted in the power domain too: a magnitude fit to the input would
        # inherit its noise bias and push the predicted power up by about P.
        r2 = r * r
        gain2 = (r2 * (xk**2 - power) * w).flatten(1).sum(1) / (r2 * r2 * w).flatten(1).sum(
            1
        ).clamp_min(1e-12)
        predicted = gain2.view(-1, 1, 1, 1) * r2
        target = yk**2 - power
        resid = (predicted - target) * w
        n = w.sum().clamp_min(1.0)
        risk, bias = float(resid.pow(2).sum() / n), float(resid.sum() / n)
        out[f"val_rep_split_risk_{name}"], out[f"val_rep_split_bias_{name}"] = risk, bias
        risks.append(risk)
        biases.append(bias)
    out["val_rep_split_risk"] = sum(risks) / len(risks)
    out["val_rep_split_bias"] = sum(biases) / len(biases)
    return out
