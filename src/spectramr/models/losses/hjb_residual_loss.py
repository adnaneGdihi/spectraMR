r"""Hamilton-Jacobi-Bellman residual loss for trajectory / waveform design.

Backs the ``hjb_trajectory`` and ``hjb_waveform`` strategy keys with the
real optimal-control mathematics they advertise, instead of routing a
vanilla L1/L2 reconstruction objective. The value function
:math:`V(\mathbf{k},t)` of the optimal-control problem satisfies the HJB
PDE [Fleming-Soner 2006, §III.5]

.. math::

    \partial_t V + \min_{\|\mathbf u\|\le G_{\max}}
        \bigl\{\nabla_{\mathbf k} V\cdot\mathbf u + L(\mathbf k,t)\bigr\} = 0,
    \qquad
    \min_{\|\mathbf u\|\le G_{\max}}\nabla_{\mathbf k}V\cdot\mathbf u
        = -G_{\max}\,\|\nabla_{\mathbf k}V\|.

This module treats the network output as a sampled value-function /
trajectory / waveform field and forms the HJB **residual operator**

.. math::

    \mathcal R[V] \;=\; \partial_t V \;-\; G_{\max}\,\|\nabla_{\mathbf k} V\|

via :func:`spectramr.infrastructure.physics.hjb_optimal_control.hjb_residual_loss`
(the deep-Galerkin residual of [Sirignano-Spiliopoulos 2018, JCP]). Because
the absolute running cost :math:`L` of a recon task is unknown, the loss is a
**discrepancy** formulation: it penalises the difference between the
residual operator evaluated on the prediction and on the target value
field (equivalently, it uses the target's residual operator as the running
cost). A field that satisfies the same HJB relation as the target scores
~0; any departure — in particular one that changes the Hamiltonian
:math:`G_{\max}\|\nabla V\|` term — is penalised.

See :mod:`spectramr.infrastructure.physics.hjb_optimal_control` for the
underlying primitives.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.nn.functional import grid_sample

from spectramr.infrastructure.physics.hjb_optimal_control import hjb_residual_loss
from spectramr.models.losses.registry import register_loss


@register_loss(
    name="hjb_residual",
    aliases=["hjb_trajectory", "hjb_waveform"],
    domain="image",
)
class HJBResidualLoss(nn.Module):
    r"""Hamilton-Jacobi-Bellman PDE-residual discrepancy loss.

    The input tensor is read as a sampled value function
    :math:`V(\mathbf k, t)`. The **channel axis is the control time axis**
    and the trailing spatial axes are the (2-D) state coordinates
    :math:`\mathbf k`; i.e. ``prediction`` / ``target`` are
    ``[B, T, H, W]``. A differentiable interpolant of the field is built
    with bilinear ``grid_sample`` so that the HJB primitive can take
    autograd derivatives :math:`\partial_t V` and :math:`\nabla_{\mathbf k}V`
    at continuous collocation points.

    The loss is the squared discrepancy between the HJB residual operator
    :math:`\partial_t V - G_{\max}\|\nabla_{\mathbf k}V\|` evaluated on the
    prediction and on the target — the target's residual plays the role of
    the (otherwise unknown) running cost :math:`L`.

    Args:
        g_max: Hardware gradient-amplitude bound :math:`G_{\max}` entering
            the min-control Hamiltonian term. Must be > 0. Default 1.0.
        num_collocation: Number of random collocation points per batch
            element at which the HJB residual is evaluated. Must be > 0.
            Default 256.
        eps: Floor on :math:`\|\nabla V\|` for numerical stability.
        reduction: ``"mean"`` or ``"sum"`` over the batch.
    """

    def __init__(
        self,
        g_max: float = 1.0,
        num_collocation: int = 256,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if g_max <= 0:
            raise ValueError(f"g_max must be > 0, got {g_max!r}")
        if num_collocation <= 0:
            raise ValueError(f"num_collocation must be > 0, got {num_collocation!r}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps!r}")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean/sum, got {reduction!r}")
        self.g_max = g_max
        self.num_collocation = num_collocation
        self.eps = eps
        self.reduction = reduction

    @staticmethod
    def _as_btwh(x: torch.Tensor) -> torch.Tensor:
        """Coerce to ``[B, T, H, W]`` (time on the channel axis)."""
        if x.ndim == 4:
            return x
        if x.ndim == 3:
            return x.unsqueeze(0)
        raise ValueError(
            f"HJBResidualLoss expects a 3-D [T,H,W] or 4-D [B,T,H,W] value "
            f"field; got {x.ndim} dims with shape {tuple(x.shape)}."
        )

    def _make_value_fn(
        self, field: torch.Tensor
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        r"""Build a differentiable ``value_fn(states, times) -> (N,)``.

        ``field`` is a single ``[T, H, W]`` value-function volume. The
        returned callable bilinearly samples it at continuous state/time
        collocation coordinates so the HJB primitive can autograd-difference
        :math:`\partial_t V` and :math:`\nabla_{\mathbf k}V`.
        """
        t_dim = field.shape[0]
        # Treat the [T, H, W] volume as a [1, 1, T, H*?] ... we sample with
        # grid_sample over a synthetic image whose "channels" carry time.
        # Layout for grid_sample: input [N=1, C=T, H, W]; we sample a
        # spatial (state) grid and select the time slice by linear blend.
        vol = field.unsqueeze(0)  # [1, T, H, W]

        def value_fn(states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
            # states: (N, 2) normalised state coords in [-1, 1];
            # times:  (N,)  normalised time coord in [-1, 1].
            n = states.shape[0]
            grid = states.view(1, n, 1, 2)  # [1, N, 1, 2] -> (x, y)
            # Sample every time slice at the state points: [1, T, N, 1].
            sampled = grid_sample(
                vol, grid, mode="bilinear", align_corners=True, padding_mode="border"
            )
            sampled = sampled.view(t_dim, n)  # [T, N]
            # Linear interpolation along the time axis using `times`.
            t_cont = (times + 1.0) * 0.5 * (t_dim - 1)  # -> [0, T-1]
            t0 = t_cont.floor().clamp(0, t_dim - 1).long()
            t1 = (t0 + 1).clamp(0, t_dim - 1)
            frac = (t_cont - t0.to(t_cont.dtype)).clamp(0.0, 1.0)
            idx = torch.arange(n, device=field.device)
            v0 = sampled[t0, idx]
            v1 = sampled[t1, idx]
            return v0 * (1.0 - frac) + v1 * frac  # (N,)

        return value_fn

    def _sample_collocation(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.num_collocation
        states = (torch.rand(n, 2, device=device, dtype=dtype) * 2.0) - 1.0
        times = (torch.rand(n, device=device, dtype=dtype) * 2.0) - 1.0
        return states, times

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        pred = self._as_btwh(prediction)
        tgt = self._as_btwh(target)
        if pred.shape != tgt.shape:
            raise ValueError(
                f"prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(tgt.shape)}."
            )

        batch = pred.shape[0]
        per_sample: list[torch.Tensor] = []
        for b in range(batch):
            states, times = self._sample_collocation(pred.device, pred.dtype)

            pred_value_fn = self._make_value_fn(pred[b])
            tgt_value_fn = self._make_value_fn(tgt[b])

            # The running cost for the prediction's residual is taken to be
            # the target's HJB residual operator -G_max||grad V_tgt|| + d_t
            # V_tgt, so that the primitive returns the squared discrepancy
            # ||R[V_pred] - R[V_tgt]||^2. We realise this by passing a
            # running_cost callable that re-evaluates the target operator on
            # the SAME collocation points the primitive uses internally.
            def _target_residual_cost(
                s: torch.Tensor,
                t: torch.Tensor,
                _fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tgt_value_fn,
            ) -> torch.Tensor:
                s = s.detach().requires_grad_(True)
                t = t.detach().requires_grad_(True)
                v = _fn(s, t)
                go = torch.ones_like(v)
                nabla, dvdt = torch.autograd.grad(v, [s, t], grad_outputs=go, create_graph=True)
                norm = nabla.norm(dim=-1).clamp_min(self.eps)
                # cost L such that R[V_tgt] = d_t V_tgt - g_max||grad V_tgt||
                # The primitive computes residual = d_t V + (-g_max||grad V||)
                # + L. Choosing L = -(d_t V_tgt - g_max||grad V_tgt||) makes
                # residual = R[V_pred] - R[V_tgt].
                cost: torch.Tensor = -(dvdt - self.g_max * norm)
                return cost

            residual = hjb_residual_loss(
                value_fn=pred_value_fn,
                states=states,
                times=times,
                running_cost=_target_residual_cost,
                g_max=self.g_max,
                eps=self.eps,
            )
            per_sample.append(residual)

        stacked = torch.stack(per_sample)
        return stacked.mean() if self.reduction == "mean" else stacked.sum()
