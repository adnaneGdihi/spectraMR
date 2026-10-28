r"""Sliced Score Matching (SSM) — score estimation without explicit noise.

Standard denoising score matching adds noise to obtain a tractable
target :math:`\boldsymbol{\epsilon}/\sigma`. Sliced score matching
(Song *et al.* 2019) avoids the noise altogether by reducing the
intractable :math:`\mathrm{tr}(\nabla s)` term to its
random-projection scalar form

.. math::
    \mathcal{L}_{\text{SSM}} = \mathbb{E}_{\mathbf{x},\mathbf{v}}\!\left[
        \tfrac12 (\mathbf{v}^\top s_{\boldsymbol{\theta}}(\mathbf{x}))^2
        + \mathbf{v}^\top \nabla_{\mathbf{x}}\!\left(\mathbf{v}^\top s_{\boldsymbol{\theta}}(\mathbf{x})\right)
    \right],

with :math:`\mathbf{v}` a Rademacher (or normal) projection vector.
This is unbiased for the Fisher score and avoids the
:math:`\boldsymbol{\epsilon}/\sigma` blow-up when :math:`\sigma\to 0`.

The Hessian-vector product is computed by autograd (``torch.autograd.grad``
with ``create_graph=True``), so this loss requires the score model to
be supplied via ``context`` and the input :math:`\mathbf{x}` to be a
leaf with ``requires_grad=True``. The loss handles that internally
when ``prediction`` is itself the score :math:`s_\theta(\mathbf{x})`.

Reference:
    Y. Song *et al.*, "Sliced score matching: A scalable approach to
    density and score estimation," *Proc. UAI*, 2019.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


def _sliced_score_loss(
    score: torch.Tensor,
    x: torch.Tensor,
    n_slices: int,
    rademacher: bool,
) -> torch.Tensor:
    """Per-batch sliced-score-matching estimator.

    score: ``s_theta(x)`` of same shape as ``x``.
    x: input tensor with ``requires_grad=True``.
    """
    loss = score.new_zeros(())
    for _ in range(n_slices):
        if rademacher:
            v = torch.randint_like(x, 0, 2).mul_(2).sub_(1).to(x.dtype)
        else:
            v = torch.randn_like(x)
        v_score = (v * score).flatten(1).sum(dim=1)  # (B,)
        # ∇_x (v · s_theta(x)), summed over batch ⇒ vector field grad_x
        grad = torch.autograd.grad(v_score.sum(), x, create_graph=True, retain_graph=True)[0]
        # Term 1: ½ (v · s)²; Term 2: v · ∇(v · s)
        term1 = 0.5 * v_score.pow(2).mean()
        term2 = (v * grad).flatten(1).sum(dim=1).mean()
        loss = loss + term1 + term2
    return loss / n_slices


@register_loss(
    name="sliced_score_matching",
    aliases=["SlicedScoreMatching", "ssm"],
    domain="image",
)
class SlicedScoreMatchingLoss(nn.Module):
    r"""Sliced score matching with random projections.

    Two usage modes:

    1. **Pre-computed score** — caller passes ``prediction = s_theta(x)``
       and supplies the *input* tensor :math:`\mathbf{x}` (with
       ``requires_grad=True``) via ``context["x_input"]``. The loss
       takes the autograd Hessian-vector product on the fly.

    2. **Model-driven** — caller passes ``context["model"]`` and
       ``context["x_input"]``; the loss invokes the model itself,
       enabling stand-alone score-matching training where the strategy
       only needs to hand off the network.

    Args:
        n_slices: Number of random projection vectors per call.
            Default 1 (paper).
        rademacher: Use ``±1`` Rademacher slices (default) or Gaussian.
    """

    def __init__(self, n_slices: int = 1, rademacher: bool = True) -> None:
        super().__init__()
        if n_slices < 1:
            raise ValueError(f"n_slices must be >= 1, got {n_slices}")
        self.n_slices = n_slices
        self.rademacher = rademacher

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if context is None or "x_input" not in context:
            raise ValueError(
                "SlicedScoreMatchingLoss requires context['x_input'] "
                "(the score's *input* tensor with requires_grad=True)."
            )
        x = context["x_input"]
        if not x.requires_grad:
            raise ValueError(
                "context['x_input'] must have requires_grad=True so that "
                "the autograd Hessian-vector product is well-defined."
            )
        model = context.get("model")
        score: torch.Tensor = model(x) if model is not None else prediction
        if score.shape != x.shape:
            raise ValueError(f"score shape {tuple(score.shape)} != x shape {tuple(x.shape)}")
        return _sliced_score_loss(score, x, self.n_slices, self.rademacher)
