"""Phase-contrast / 4D-flow metrics (regime: ``mri_flow``).

Velocity is ``[B, n_dir, *extra, *spatial]`` in m/s: component ``i`` is the
velocity along spatial axis ``i`` (the axes in the order they appear), and any
axes between the component axis and the spatial ones (cardiac frames) are
pooled. The spatial rank is never guessed from the shape; it is the length of
``context.voxel_size_mm`` or the rank of a plane mask.

Everything beyond ``(prediction, target)`` arrives through
:class:`~spectramr.core.metrics.context.MetricContext` and is declared in
``needs``; a metric handed a context missing a declared field raises
:class:`MetricNotApplicableError` rather than inventing a value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from spectramr.config.schemas.enums import Regime
from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.outcome import MetricNotApplicableError, NotApplicableReason
from spectramr.core.metrics.registry import register_metric

_FLOW = frozenset({Regime.FLOW})


def _missing(name: str, ctx: MetricContext, *fields: str) -> MetricNotApplicableError:
    absent = [f for f in fields if getattr(ctx, f, None) is None]
    return MetricNotApplicableError(
        name, NotApplicableReason.MISSING_MEASUREMENT_CONTEXT, f"context lacks {absent}"
    )


def _require(name: str, ctx: MetricContext, *fields: str) -> None:
    if not ctx.has(*fields):
        raise _missing(name, ctx, *fields)


def _masked_mean(values: Tensor, weight: Tensor | None) -> Tensor:
    if weight is None:
        return values.mean()
    w = torch.broadcast_to(weight.to(values.dtype), values.shape)
    return (w * values).sum() / w.sum().clamp_min(torch.finfo(values.dtype).eps)


def _velocity_jacobian(velocity: Tensor, spacing_mm: Tensor) -> list[list[Tensor]]:
    """``d v_i / d x_j`` in 1/s by central differences, ``[i][j] -> [B, *extra, *spatial]``."""
    rank = spacing_mm.numel()
    spatial_dims = tuple(range(velocity.ndim - 1 - rank, velocity.ndim - 1))
    spacing_m = spacing_mm.to(velocity) * 1e-3
    rows = []
    for i in range(velocity.shape[1]):
        partials = torch.gradient(velocity[:, i], dim=spatial_dims)
        rows.append([partials[j] / spacing_m[j] for j in range(rank)])
    return rows


def _plane_flow(velocity: Tensor, plane: Tensor) -> Tensor:
    """Flow rate through a plane, ``[B, *extra]`` in mL/s (m/s x mm^2 = mL/s)."""
    rank = plane.ndim - 1
    v = velocity.flatten(start_dim=velocity.ndim - rank)
    return torch.einsum("bn...s,ns->b...", v, plane.flatten(start_dim=1).to(v))


def _divergence(
    name: str, velocity: Tensor, ctx: MetricContext
) -> tuple[list[list[Tensor]], Tensor]:
    """The velocity Jacobian and the ROI it is graded on."""
    # The ROI is required: outside the vessel the phase is noise or air, and a
    # whole-grid mean would grade that instead of the flow.
    spacing, roi = ctx.voxel_size_mm, ctx.foreground_mask
    if spacing is None or roi is None:
        raise _missing(name, ctx, "voxel_size_mm", "foreground_mask")
    rank = spacing.numel()
    if velocity.shape[1] < rank:
        raise MetricNotApplicableError(
            name,
            NotApplicableReason.VELOCITY_DIRECTIONS_BELOW_SPATIAL_RANK,
            f"{velocity.shape[1]} encoded direction(s) over a rank-{rank} grid; "
            "divergence needs every in-grid component (through-plane data has one)",
        )
    return _velocity_jacobian(velocity[:, :rank], spacing), roi


@register_metric(
    "divergence",
    aliases=["Divergence"],
    requires_reference=False,
    needs=("voxel_size_mm", "foreground_mask"),
    workflows=_FLOW,
)
class Divergence(nn.Module):  # type: ignore[misc]
    """Mean ``|div v|`` (1/s) over the vessel ROI; zero for incompressible flow."""

    def forward(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> Tensor:
        ctx = resolve_context(context, kwargs)
        jac, roi = _divergence("divergence", prediction, ctx)
        div = sum(jac[i][i] for i in range(len(jac)))
        return _masked_mean(div.abs(), roi)


@register_metric(
    "divergence_normalised",
    requires_reference=False,
    needs=("voxel_size_mm", "foreground_mask"),
    workflows=_FLOW,
)
class NormalisedDivergence(nn.Module):  # type: ignore[misc]
    """``sum |div v| / sum_ij |d v_i / d x_j|`` over the ROI, in ``[0, 1]``.

    Raw divergence rewards a field that is flat or zero; dividing by the total
    velocity gradient scores how much of the field's variation is compressive.
    """

    def forward(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> Tensor:
        ctx = resolve_context(context, kwargs)
        jac, roi = _divergence("divergence_normalised", prediction, ctx)
        div = sum(jac[i][i] for i in range(len(jac))).abs()
        total = sum(d.abs() for row in jac for d in row)
        w = torch.broadcast_to(roi.to(div), div.shape)
        denom = (w * total).sum()
        if denom <= 0:
            raise MetricNotApplicableError(
                "divergence_normalised",
                NotApplicableReason.DEGENERATE_VELOCITY_FIELD,
                "the velocity field has no gradient in the ROI",
            )
        return (w * div).sum() / denom


@register_metric(
    "mass_conservation",
    aliases=["MassConservation"],
    requires_reference=False,
    needs=("inlet_mask", "outlet_mask"),
    workflows=_FLOW,
)
class ConsistencyOfMass(nn.Module):  # type: ignore[misc]
    """Inlet-minus-outlet flow imbalance, in percent of the inlet flow.

    Pooled over batch and frames, ``100 * sum |Q_in - Q_out| / sum |Q_in|``, so a
    diastolic frame with near-zero inflow cannot dominate the ratio.
    """

    def forward(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> Tensor:
        ctx = resolve_context(context, kwargs)
        _require("mass_conservation", ctx, "inlet_mask", "outlet_mask")
        q_in = _plane_flow(prediction, ctx.inlet_mask)
        q_out = _plane_flow(prediction, ctx.outlet_mask)
        inflow = q_in.abs().sum()
        if inflow <= 0:
            raise MetricNotApplicableError(
                "mass_conservation",
                NotApplicableReason.DEGENERATE_VELOCITY_FIELD,
                "no flow through the inlet plane",
            )
        return 100.0 * (q_in - q_out).abs().sum() / inflow


@register_metric(
    "net_flow_error",
    aliases=["NetFlowError"],
    requires_reference=True,
    needs=("inlet_mask",),
    workflows=_FLOW,
)
class NetFlowError(nn.Module):  # type: ignore[misc]
    """``|mean_t Q_pred - mean_t Q_target|`` through the inlet plane, mL/s, batch-averaged."""

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> Tensor:
        ctx = resolve_context(context, kwargs)
        _require("net_flow_error", ctx, "inlet_mask")
        q_pred = _plane_flow(prediction, ctx.inlet_mask).flatten(start_dim=1).mean(dim=1)
        q_true = _plane_flow(target, ctx.inlet_mask).flatten(start_dim=1).mean(dim=1)
        return (q_pred - q_true).abs().mean()


@register_metric(
    "vnr",
    aliases=["VelocityNoiseRatio"],
    requires_reference=False,
    needs=("foreground_mask", "static_mask"),
    workflows=_FLOW,
)
class VelocityNoiseRatio(nn.Module):  # type: ignore[misc]
    """Mean speed in the vessel over the per-component velocity RMS in static tissue."""

    def forward(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> Tensor:
        ctx = resolve_context(context, kwargs)
        _require("vnr", ctx, "foreground_mask", "static_mask")
        speed = prediction.norm(dim=1)
        noise = _masked_mean(prediction.pow(2).mean(dim=1), ctx.static_mask).sqrt()
        return _masked_mean(speed, ctx.foreground_mask) / noise.clamp_min(1e-12)


@register_metric(
    "velocity_rmse",
    aliases=["VelocityRMSE"],
    requires_reference=True,
    workflows=_FLOW,
)
class VelocityRMSE(nn.Module):  # type: ignore[misc]
    """Root-mean-square error between predicted and reference velocity fields."""

    def forward(self, v_pred: Tensor, v_target: Tensor) -> Tensor:
        """Compute velocity RMSE over all components/voxels."""
        return torch.sqrt(torch.mean((v_pred - v_target) ** 2))


@register_metric(
    "velocity_rmse_roi",
    requires_reference=True,
    needs=("foreground_mask",),
    workflows=_FLOW,
)
class VelocityRMSEROI(nn.Module):  # type: ignore[misc]
    """Per-component velocity RMSE (m/s) restricted to the vessel ROI."""

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> Tensor:
        ctx = resolve_context(context, kwargs)
        _require("velocity_rmse_roi", ctx, "foreground_mask")
        return _masked_mean((prediction - target).pow(2).mean(dim=1), ctx.foreground_mask).sqrt()


@register_metric(
    "peak_velocity_error",
    aliases=["PeakVelocityError"],
    requires_reference=True,
    workflows=_FLOW,
)
class PeakVelocityError(nn.Module):  # type: ignore[misc]
    """Absolute error in peak velocity magnitude (a key clinical PC-MRI readout)."""

    def forward(self, v_pred: Tensor, v_target: Tensor) -> Tensor:
        """Compute |max|v_pred| - max|v_target||."""
        peak_pred = v_pred.norm(dim=1).amax()
        peak_target = v_target.norm(dim=1).amax()
        return (peak_pred - peak_target).abs()
