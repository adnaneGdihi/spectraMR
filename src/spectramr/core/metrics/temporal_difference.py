r"""Inter-frame difference error of a reconstructed series.

For a real series ``[B, T, ...]`` with time on axis 1, the metric is the
normalised RMSE of the forward temporal differences :math:`\Delta_t x = x_{t+1}-x_t`,

.. math::

    \mathrm{TD\text{-}NRMSE}(\hat x, x) =
        \frac{\lVert \Delta_t \hat x - \Delta_t x \rVert_2}{\lVert \Delta_t x \rVert_2},

computed per sample and averaged over the samples whose reference moves. Lower is
better; 0 means every frame-to-frame change is reproduced exactly, and 1 is what
a reconstruction that emits one static frame (zero motion) scores.

It is not a second ``temporal_fidelity``. That metric is a per-voxel Pearson
correlation and therefore invariant to per-voxel scale: a reconstruction that
halves the motion amplitude still scores 1.0 there, and 0.5 here. Read together,
fidelity says whether the shape of the dynamics survived and this says whether
their amplitude did.

Conventions follow :mod:`spectramr.core.metrics.temporal_metrics`: real input
only, and degenerate input (complex, a single frame, a shape mismatch, or a
reference with no motion anywhere) returns ``NaN`` (skip), never a value in the
metric's range.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.config.schemas.enums import Regime, Task
from spectramr.core.metrics.registry import register_metric
from spectramr.core.metrics.temporal_metrics import _MIN_TEMPORAL_STD, _reject_non_series

logger = logging.getLogger(__name__)


@register_metric(
    "temporal_difference_nrmse",
    aliases=["inter_frame_nrmse", "TemporalDifferenceNRMSE"],
    direction="lower",
    workflows=frozenset({Regime.FUNCTIONAL, Regime.DYNAMIC}),
    tasks=frozenset({Task.RECONSTRUCTION, Task.DENOISING, Task.SUPER_RESOLUTION}),
)
class TemporalDifferenceNRMSE:
    r"""NRMSE of the inter-frame differences, per sample, averaged over moving samples."""

    @property
    def name(self) -> str:
        return "temporal_difference_nrmse"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        del kwargs  # domain / data_range: the ratio is scale-free.
        reason = _reject_non_series(prediction, "prediction") or _reject_non_series(
            target, "target"
        )
        if reason is None and prediction.shape != target.shape:
            reason = f"shape mismatch {tuple(prediction.shape)} vs {tuple(target.shape)}"
        if reason is not None:
            logger.debug("temporal_difference_nrmse skipped: %s.", reason)
            return float("nan")

        pred = prediction.float().flatten(2)
        ref = target.float().flatten(2)
        d_ref = ref[:, 1:] - ref[:, :-1]
        d_err = (pred[:, 1:] - pred[:, :-1]) - d_ref

        # Per-sample RMS so the score does not depend on how many elements a sample has.
        ref_rms = d_ref.square().mean(dim=(1, 2)).sqrt()
        err_rms = d_err.square().mean(dim=(1, 2)).sqrt()
        moving = ref_rms > _MIN_TEMPORAL_STD
        if not bool(moving.any()):
            logger.debug(
                "temporal_difference_nrmse skipped: the reference has no inter-frame "
                "change in any sample, so there is no motion to grade."
            )
            return float("nan")
        return float((err_rms[moving] / ref_rms[moving]).mean())


__all__ = ["TemporalDifferenceNRMSE"]
