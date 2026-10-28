r"""Conditional Flow Matching (CFM) and Rectified Flow training losses.

Both are direct regression objectives on a *velocity field* that
transports a base distribution :math:`\mathbf{x}_0\sim p_0` to the data
distribution :math:`\mathbf{x}_1\sim p_{\text{data}}`. They differ
only in how the (training) endpoints :math:`(\mathbf{x}_0,\mathbf{x}_1)`
are paired and what the conditioning variable :math:`t` parameterises.

* **Conditional flow matching** (Lipman *et al.* 2023): for each
  example sample :math:`t\sim U(0,1)` and a base point
  :math:`\mathbf{x}_0\sim\mathcal{N}(0,\mathbf{I})`. The conditional
  probability path is the straight line
  :math:`\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1`, whose
  velocity is :math:`\mathbf{x}_1-\mathbf{x}_0`. The loss is

  .. math::
      \mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\mathbf{x}_0,\mathbf{x}_1}\,
        \|v_{\boldsymbol{\theta}}(\mathbf{x}_t,t)-(\mathbf{x}_1-\mathbf{x}_0)\|^2.

* **Rectified flow** (Liu, Gong, Liu 2023): same loss form but with
  :math:`(\mathbf{x}_0,\mathbf{x}_1)` taken from a previously-trained
  flow's deterministic ODE pairing rather than independent draws —
  *straightens* the trajectories, enabling 1- or 2-step generation.
  The rectified-flow loss therefore behaves identically *given* the
  paired sample, so we re-export the same implementation under a
  separate registry entry to make the choice explicit in YAML.

References:
    [35] Y. Lipman *et al.*, ICLR 2023.
    [36] A. Tong *et al.*, "Improving and generalizing flow-based
        generative models with minibatch optimal transport," TMLR, 2024.
    [37] X. Liu, C. Gong, Q. Liu, ICLR 2023.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


def _require_x0(ctx: dict[str, Any], loss_name: str) -> torch.Tensor:
    """Return the base sample the strategy built ``x_t`` from, or raise.

    The regression target ``x1 - x0`` is only meaningful for the ``x0`` that
    produced the network input; a freshly drawn one is independent of it, so
    the optimum would be ``E[x1 | x_t]`` rather than the flow velocity.
    """
    x0 = ctx.get("x0")
    if x0 is None:
        raise ValueError(
            f"{loss_name} requires context['x0']: the base sample used to build "
            "x_t = (1 - t) x0 + t x1 for the velocity prediction."
        )
    if not isinstance(x0, torch.Tensor):
        raise TypeError(f"context['x0'] must be a tensor, got {type(x0).__name__}")
    return x0


def _flow_matching_loss(
    velocity_pred: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    norm: str,
) -> torch.Tensor:
    """Regress the prediction onto the straight-line velocity ``x1 - x0``."""
    target_v = x1 - x0  # straight-line velocity field
    if velocity_pred.shape != target_v.shape:
        raise ValueError(
            f"velocity prediction shape {tuple(velocity_pred.shape)} != "
            f"target velocity shape {tuple(target_v.shape)}"
        )
    if norm == "l1":
        return F.l1_loss(velocity_pred, target_v)
    return F.mse_loss(velocity_pred, target_v)


@register_loss(
    name="conditional_flow_matching",
    aliases=["cfm", "CFMLoss", "flow_matching"],
    domain="image",
)
class ConditionalFlowMatchingLoss(nn.Module):
    r"""Velocity-regression loss for conditional flow matching.

    Forward expects:

    * ``prediction``: velocity prediction :math:`v_{\boldsymbol{\theta}}
      (\mathbf{x}_t, t)`. Same shape as the data tensor.
    * ``target``: the data sample :math:`\mathbf{x}_1`. Same shape.
    * ``context["x0"]``: required base sample :math:`\mathbf{x}_0` — the
      one the strategy used to build the network input :math:`\mathbf{x}_t`.

    The time :math:`t` enters only through :math:`\mathbf{x}_t`; the
    straight-line target does not depend on it.

    Args:
        norm: ``"l2"`` (default, paper) or ``"l1"``.
    """

    def __init__(self, norm: str = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
        self.norm = norm

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        x0 = _require_x0(context or {}, type(self).__name__)
        return _flow_matching_loss(prediction, x0, target, self.norm)


@register_loss(
    name="rectified_flow",
    aliases=["RectifiedFlowLoss", "reflow"],
    domain="image",
)
class RectifiedFlowLoss(nn.Module):
    r"""Rectified-flow velocity loss (paper-identical to CFM regression
    but with caller-supplied straight-line pairs).

    The strategy is responsible for sampling :math:`(\mathbf{x}_0,
    \mathbf{x}_1)` from the previous flow's deterministic ODE and
    passing ``x0`` via ``context["x0"]`` (required).

    Args:
        norm: ``"l2"`` (default) or ``"l1"``.
    """

    def __init__(self, norm: str = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
        self.norm = norm

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        x0 = _require_x0(context or {}, type(self).__name__)
        return _flow_matching_loss(prediction, x0, target, self.norm)


def _hungarian_pair(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Find an optimal-transport pairing of (x0, x1) within the batch.

    Returns the permutation ``perm`` of ``x0`` such that ``x0[perm][j]`` is the
    partner of ``x1[j]`` under the minimum squared-Euclidean cost. Uses the
    Hungarian algorithm via ``scipy.optimize.linear_sum_assignment``
    on the detached cost matrix (the loss itself remains
    differentiable through the resulting straight-line velocity).
    """
    from scipy.optimize import (  # type: ignore[import-untyped]  # local: keep startup cheap
        linear_sum_assignment,
    )

    flat_x0 = x0.reshape(x0.shape[0], -1)
    flat_x1 = x1.reshape(x1.shape[0], -1)
    cost = torch.cdist(flat_x0, flat_x1).pow(2).detach().cpu().numpy()
    # linear_sum_assignment pairs x0[i] with x1[col[i]]; argsort inverts that
    # to the x0 row matched to each x1[j].
    _, col = linear_sum_assignment(cost)
    col_t = torch.as_tensor(col, device=x0.device, dtype=torch.long)
    return torch.argsort(col_t)


@register_loss(
    name="ot_conditional_flow_matching",
    aliases=["ot_cfm", "OTCFMLoss", "ot_flow_matching"],
    domain="image",
)
class OTConditionalFlowMatchingLoss(nn.Module):
    r"""CFM with minibatch optimal-transport pairing of (x_0, x_1).

    Identical loss form to plain CFM, but the noise samples
    :math:`\mathbf{x}_0` are re-paired with the data samples
    :math:`\mathbf{x}_1` using the Hungarian algorithm on the
    minibatch cost matrix :math:`C_{ij} = \|x_{0,i}-x_{1,j}\|^2`.
    The OT pairing produces *straighter* probability paths, which in
    practice lets ODE samplers integrate in many fewer steps.

    Forward expects:

    * ``prediction``: velocity prediction at the (re-paired)
      :math:`\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1`. The
      loss does **not** itself rebuild :math:`\mathbf{x}_t` — the
      strategy must build it after the OT pairing is returned via the
      auxiliary ``permutation`` (see Notes). This is the canonical
      stateless form: the strategy gets ``permutation``, indexes
      :math:`\mathbf{x}_0` accordingly, builds :math:`\mathbf{x}_t`,
      runs the network, and hands the result back as ``prediction``.
    * ``target``: data sample :math:`\mathbf{x}_1`.
    * ``context["x0"]``: required :math:`\mathbf{x}_0` (unpaired).

    Notes:
        ``context["permutation"]`` indexes ``x0``: ``x0[permutation][j]``
        is the partner of ``x1[j]``. When ``context["return_permutation"]
        = True`` the loss attaches the permutation it solved so the strategy
        can reuse it. The loss assumes the strategy has already built
        :math:`\mathbf{x}_t` using the *same* permutation — when
        ``context`` provides a pre-computed ``permutation`` we honour it;
        otherwise we Hungarian-pair on the fly and the strategy is
        expected to have done the same.

    Args:
        norm: ``"l2"`` (default) or ``"l1"``.
    """

    def __init__(self, norm: str = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
        self.norm = norm

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        ctx = context if context is not None else {}
        x0 = _require_x0(ctx, type(self).__name__)
        perm = ctx.get("permutation")
        if perm is None:
            perm = _hungarian_pair(x0, target)
            if context is not None and ctx.get("return_permutation"):
                ctx["permutation"] = perm
        x0_paired = x0[perm]
        return _flow_matching_loss(prediction, x0_paired, target, self.norm)
