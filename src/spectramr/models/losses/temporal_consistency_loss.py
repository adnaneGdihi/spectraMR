r"""Temporal consistency loss: match the inter-frame differences of a series.

For a cine / dynamic series laid out ``[B, T, ...]`` with time on axis 1 (the
convention of :mod:`spectramr.core.metrics.temporal_metrics`), the forward
temporal difference is :math:`\Delta_t x = x_{t+1} - x_t`. The loss penalises the
mismatch between the predicted and the reference differences,

.. math::

    \mathcal{L}_{\mathrm{tc}} = \bigl\lVert \Delta_t \hat x - \Delta_t x \bigr\rVert_p ,
    \qquad p \in \{1, 2\},

so the reconstruction is asked to reproduce the *motion* between frames, not
only each frame. Per-frame pixel losses reduce over the time axis and cannot see
that difference.

The term is target-referenced on purpose. A reference-free smoothness penalty on
:math:`\Delta_t \hat x` alone (temporal TV, the ``storm`` Laplacian) is minimised
by emitting the temporal mean at every frame, which is the collapse
``TemporalFidelity`` exists to detect; here that collapse scores
:math:`\lVert \Delta_t x \rVert_p`, its worst value.

Reference: the temporal-gradient matching term of dynamic-MRI and video
restoration objectives, e.g. Qin et al., "Convolutional Recurrent Neural
Networks for Dynamic MR Image Reconstruction", IEEE TMI 38(1), 2019.
"""

from __future__ import annotations

import torch
from torch import nn

from spectramr.config.schemas.enums import Regime
from spectramr.models.losses.registry import register_loss

#: The advertised option sets; anything else raises at construction.
SUPPORTED_NORMS: tuple[str, ...] = ("l1", "l2")
SUPPORTED_REDUCTIONS: tuple[str, ...] = ("mean", "sum")


def temporal_difference(series: torch.Tensor) -> torch.Tensor:
    """Forward difference along axis 1: ``series[:, 1:] - series[:, :-1]``."""
    return series[:, 1:] - series[:, :-1]


def _check_series(series: torch.Tensor, label: str) -> None:
    if series.is_complex():
        raise ValueError(
            f"temporal_consistency: {label} is complex; pass a real series "
            "(magnitude, or real/imag stacked on a non-time axis)."
        )
    if series.ndim < 3:
        raise ValueError(
            f"temporal_consistency: {label} must be [B, T, ...] with time on axis 1, "
            f"got shape {tuple(series.shape)}."
        )
    if series.shape[1] < 2:
        raise ValueError(
            f"temporal_consistency: {label} has {series.shape[1]} frame(s) on axis 1; "
            "a single frame has no inter-frame difference, so the term would be "
            "identically zero. Feed a series with T >= 2 frames."
        )


@register_loss(
    name="temporal_consistency",
    aliases=["temporal_difference_loss", "inter_frame_consistency"],
    domain="image",
    workflows=frozenset({Regime.DYNAMIC, Regime.FUNCTIONAL}),
)
class TemporalConsistencyLoss(nn.Module):
    r"""Inter-frame difference matching :math:`\lVert\Delta_t\hat x-\Delta_t x\rVert_p`.

    **DOMAIN**: IMAGE — a real ``[B, T, ...]`` series, time on axis 1.

    Args:
        norm: ``"l1"`` (default, robust to a few mis-registered frames) or ``"l2"``.
        reduction: ``"mean"`` (default) or ``"sum"`` over every difference element.

    Raises:
        ValueError: on an unsupported ``norm``/``reduction``, and at call time on a
            complex input, a shape mismatch, or fewer than two frames.
    """

    def __init__(self, norm: str = "l1", reduction: str = "mean") -> None:
        super().__init__()
        if norm not in SUPPORTED_NORMS:
            raise ValueError(f"norm must be one of {SUPPORTED_NORMS}, got {norm!r}")
        if reduction not in SUPPORTED_REDUCTIONS:
            raise ValueError(f"reduction must be one of {SUPPORTED_REDUCTIONS}, got {reduction!r}")
        self.norm = norm
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        _check_series(prediction, "prediction")
        _check_series(target, "target")
        if prediction.shape != target.shape:
            raise ValueError(
                "temporal_consistency: shape mismatch "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
            )
        residual = temporal_difference(prediction) - temporal_difference(target)
        per_elem = residual.abs() if self.norm == "l1" else residual.square()
        return per_elem.sum() if self.reduction == "sum" else per_elem.mean()


__all__ = [
    "SUPPORTED_NORMS",
    "SUPPORTED_REDUCTIONS",
    "TemporalConsistencyLoss",
    "temporal_difference",
]
