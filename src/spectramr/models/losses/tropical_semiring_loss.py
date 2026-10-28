r"""Tropical-semiring consistency loss for MRF / quantitative-map paradigms.

The ``tropical_mrf`` and ``tropical_quantitative_maps`` strategies model the
log-magnitude signal (resp. the :math:`T_1, T_2, M_0` maps) as a tropical
polynomial over the max-plus semiring
:math:`(\mathbb{R}\cup\{-\infty\}, \max, +)` — i.e. as a continuous
piecewise-affine function whose pieces are the cones of a polyhedral fan
[Maclagan & Sturmfels 2015; Zhang-Naitzat-Lim 2018]. A vanilla
:math:`\ell_1` / :math:`\ell_2` reconstruction loss is *blind* to that
structure: it penalises Euclidean residuals but never the tropical
(``min``/``max``-plus) reduction that defines the paradigm.

This loss closes that gap by consuming the genuine primitive at
:mod:`spectramr.infrastructure.physics.tropical_geometry`. Each pixel's channel
vector :math:`\mathbf{r}\in\mathbb{R}^{d}` (``d = C``, e.g.
:math:`(\log T_1, \log T_2, \log M_0)`) is treated as a point in tropical
coordinate space. Against a fixed reference fan
:math:`p(\mathbf{r})=\max_i\langle a_i,\mathbf{r}\rangle+b_i` the loss combines:

#. a **tropicalisation term** — the deviation between the smooth max-plus
   reduction :func:`log_sum_exp_tropical` of the prediction and of the target,
   so the network must reproduce the target's piecewise-affine envelope, not
   merely its Euclidean values;
#. a **cone / variety term** — a penalty that the prediction lies on the same
   top cone of the polyhedral fan as the target (the cone classifier
   :func:`active_cone_indices`), measured as the slack of the prediction
   against the target's active monomial via :func:`tropical_polynomial`. A
   prediction sitting on the tropical variety of the *wrong* cone is penalised
   even when its raw values are close.

See :mod:`spectramr.infrastructure.physics.tropical_geometry` for the algebraic
primitives.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.tropical_geometry import (
    active_cone_indices,
    log_sum_exp_tropical,
    tropical_polynomial,
)
from spectramr.models.losses.registry import register_loss


@register_loss(
    name="tropical_semiring_consistency",
    aliases=["tropical_mrf_consistency", "tropical_quantitative_consistency"],
    domain="image",
)
class TropicalSemiringConsistencyLoss(nn.Module):
    r"""Max-plus (tropical) consistency penalty between prediction and target.

    Treats the ``C`` channels at each spatial location as a point
    :math:`\mathbf{r}\in\mathbb{R}^{C}` in tropical coordinate space and
    penalises the deviation of the prediction's tropical reduction from the
    target's, plus a cone-membership term. Inputs are ``[B, C, H, W]`` image
    tensors with matching shapes.

    Args:
        n_monomials: Number :math:`K` of monomials (linear pieces / top cones)
            in the reference tropical fan. Default 8.
        beta: Log-sum-exp sharpness :math:`\beta>0` for the smooth
            tropicalisation; larger is closer to the hard max-plus. Default
            16.0 (matching the primitive default).
        cone_weight: Coefficient on the active-cone / tropical-variety term.
            Set 0.0 to use the tropicalisation term alone. Default 1.0.
        reduction: ``"mean"`` or ``"sum"`` over the per-pixel penalties.
        seed: Seed for the deterministic reference fan so the loss is
            reproducible across runs. Default 0.
    """

    _slopes: torch.Tensor
    _intercepts: torch.Tensor

    def __init__(
        self,
        n_monomials: int = 8,
        beta: float = 16.0,
        cone_weight: float = 1.0,
        reduction: str = "mean",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if n_monomials <= 0:
            raise ValueError(f"n_monomials must be positive; got {n_monomials}.")
        if beta <= 0:
            raise ValueError(f"beta must be positive; got {beta}.")
        if cone_weight < 0:
            raise ValueError(f"cone_weight must be >= 0; got {cone_weight}.")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean/sum, got {reduction!r}.")
        self.n_monomials = n_monomials
        self.beta = beta
        self.cone_weight = cone_weight
        self.reduction = reduction
        self.seed = seed
        # The reference fan is built lazily once C is known (forward), so the
        # loss adapts to whatever channel count the paradigm uses. We cache it
        # as a buffer keyed by the coordinate dim.
        self._fan_dim: int | None = None
        self.register_buffer("_slopes", torch.empty(0), persistent=False)
        self.register_buffer("_intercepts", torch.empty(0), persistent=False)

    def _ensure_fan(self, dim: int, device: torch.device, dtype: torch.dtype) -> None:
        """Build a deterministic reference tropical fan in :math:`\\mathbb{R}^d`."""
        if self._fan_dim == dim and self._slopes.numel() > 0:
            if self._slopes.device != device or self._slopes.dtype != dtype:
                self._slopes = self._slopes.to(device=device, dtype=dtype)
                self._intercepts = self._intercepts.to(device=device, dtype=dtype)
            return
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        scale = 1.0 / max(1.0, dim**0.5)
        slopes = torch.randn(self.n_monomials, dim, generator=gen) * scale
        intercepts = torch.randn(self.n_monomials, generator=gen) * scale
        self._slopes = slopes.to(device=device, dtype=dtype)
        self._intercepts = intercepts.to(device=device, dtype=dtype)
        self._fan_dim = dim

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have the same shape; got "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
            )
        if prediction.ndim != 4:
            raise ValueError(f"expected [B, C, H, W] inputs; got ndim={prediction.ndim}.")

        c = prediction.shape[1]
        # Points: one C-dim tropical coordinate per spatial location.
        # (B, C, H, W) -> (B*H*W, C)
        pred_pts = prediction.permute(0, 2, 3, 1).reshape(-1, c)
        targ_pts = target.permute(0, 2, 3, 1).reshape(-1, c)

        self._ensure_fan(c, prediction.device, prediction.dtype)
        slopes = self._slopes
        intercepts = self._intercepts

        reduce = torch.mean if self.reduction == "mean" else torch.sum

        # (1) Tropicalisation term: the smooth max-plus reduction must agree.
        pred_trop = log_sum_exp_tropical(pred_pts, slopes, intercepts, beta=self.beta)
        targ_trop = log_sum_exp_tropical(targ_pts, slopes, intercepts, beta=self.beta)
        trop_term = reduce((pred_trop - targ_trop).pow(2))

        if self.cone_weight == 0.0:
            return trop_term

        # (2) Cone / tropical-variety term. The target's active cone fixes the
        # monomial i* that *should* dominate. We penalise how far the
        # prediction is from realising that same monomial as its tropical
        # value, i.e. the slack p(pred) - <a_{i*}, pred> - b_{i*} >= 0. This is
        # zero iff the prediction sits inside the target's top cone (on the
        # tropical variety boundary the slack vanishes), and grows as the
        # prediction drifts to a different cone.
        with torch.no_grad():
            cone_idx = active_cone_indices(targ_pts, slopes, intercepts)  # (N,)
        a_star = slopes[cone_idx]  # (N, C)
        b_star = intercepts[cone_idx]  # (N,)
        target_monomial = (pred_pts * a_star).sum(dim=-1) + b_star  # (N,)
        pred_envelope = tropical_polynomial(pred_pts, slopes, intercepts)  # (N,)
        cone_slack = pred_envelope - target_monomial  # >= 0 by construction
        cone_term = reduce(cone_slack)

        return trop_term + self.cone_weight * cone_term

    def extra_repr(self) -> str:
        return (
            f"n_monomials={self.n_monomials}, beta={self.beta}, "
            f"cone_weight={self.cone_weight}, reduction={self.reduction}"
        )
