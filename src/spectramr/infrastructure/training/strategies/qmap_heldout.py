"""Readings for one held-out contrast in the leave-one-contrast-out pass.

``val_psnr_loo`` scales the held-out render by a gain fitted to its own target: the
pre-registered, gain-invariant synthesis score. It is blind to the gain, so it cannot show
whether the gauge prior works. With a prior, the held-out gain is also *predicted* from the
available contrasts (``val_psnr_loo_prior_gain``), and that prediction, which never sees
the label, feeds the conformal residuals (``qmap_conformal``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from spectramr.infrastructure.training.strategies.qmap_availability import leave_one_out
from spectramr.infrastructure.training.strategies.qmap_conformal import SubjectResiduals
from spectramr.infrastructure.training.strategies.qmap_gauge_prior import (
    GainRatioPrior,
    prior_predicted_gains,
    projected_gains,
)


def heldout_readings(
    k: int,
    renders: list[torch.Tensor],
    available: torch.Tensor,
    *,
    targets: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    fitted_prediction: Callable[[int, torch.Tensor], torch.Tensor],
    psnr: Callable[[torch.Tensor, torch.Tensor], float],
    prior: GainRatioPrior | None,
    residuals: SubjectResiduals | None,
    subjects: Sequence[str],
) -> dict[str, float]:
    """Score held-out contrast ``k``; only available contrasts predict its gain."""
    y = targets[k]
    out = {"val_psnr_loo": psnr(fitted_prediction(k, renders[k]), y)}
    if prior is None:
        return out
    with torch.no_grad():
        fitted = projected_gains(renders, targets, masks)
        gain = prior_predicted_gains(fitted, available, prior)[:, k].view(-1, 1, 1, 1)
    prediction = gain * renders[k]
    out["val_psnr_loo_prior_gain"] = psnr(prediction, y)
    if residuals is not None:
        residuals.add(subjects, k, (prediction - y).abs(), masks[k])
    return out


def heldout_pass(
    render: Callable[[torch.Tensor], list[torch.Tensor]],
    y: torch.Tensor,
    masks: Sequence[torch.Tensor],
    *,
    fitted_prediction: Callable[[int, torch.Tensor], torch.Tensor],
    psnr: Callable[[torch.Tensor, torch.Tensor], float],
    prior: GainRatioPrior | None,
    residuals: SubjectResiduals | None,
    subjects: Sequence[str],
) -> dict[str, float]:
    """The whole leave-one-contrast-out pass over a ``(B, 3, H, W)`` target triplet."""
    targets = [y[:, j : j + 1] for j in range(y.shape[1])]
    return leave_one_out(
        render,
        batch_size=y.shape[0],
        device=y.device,
        score=lambda k, renders, available: heldout_readings(
            k,
            renders,
            available,
            targets=targets,
            masks=masks,
            fitted_prediction=fitted_prediction,
            psnr=psnr,
            prior=prior,
            residuals=residuals,
            subjects=[str(s) for s in subjects],
        ),
    )
