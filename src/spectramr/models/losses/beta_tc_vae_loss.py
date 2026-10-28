r"""β-TCVAE Total Correlation Loss for Disentangled Representations.

Decomposes the standard VAE Evidence Lower Bound (ELBO) into three
independent terms (Chen et al., *NeurIPS* 2018):

.. math::

    \mathcal{L}_{\text{ELBO}}
        = \mathbb{E}_{q}\bigl[\log p(x \mid z)\bigr]
          - \underbrace{\alpha \cdot I_q(z; x)}_{\text{Mutual Info}}
          - \underbrace{\beta \cdot \text{TC}(z)}_{\text{Total Correlation}}
          - \underbrace{\gamma \cdot \text{DW-KL}(z)}_{\text{Dim-wise KL}}

Only the **Total Correlation** term is aggressively penalized (β ≫ 1),
forcing statistical independence between latent dimensions without
collapsing the posterior (which naive KL penalties cause).

The TC is estimated via the **minibatch-weighted sampling** trick:

.. math::

    \text{TC}(z) \approx \frac{1}{N} \sum_{i=1}^{N}
        \bigl[\log q(z_i) - \sum_{j} \log q(z_{i,j})\bigr]

where :math:`q(z) = \frac{1}{N} \sum_i q(z \mid x_i)` is the aggregated
posterior, approximated by the minibatch.

Reference:
    Chen, Li, Grosse, Duvenaud, "Isolating Sources of Disentanglement
    in Variational Autoencoders", *NeurIPS* (2018).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from spectramr.models.losses.registry import register_loss


def _log_density_gaussian(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    r"""Evaluate log N(x; μ, σ²) element-wise.

    .. math::

        \log \mathcal{N}(x; \mu, \sigma^2)
            = -\tfrac{1}{2}\bigl[\log 2\pi + \log\sigma^2
              + (x - \mu)^2 / \sigma^2\bigr]

    Args:
        x: Sample points ``(B, D)``.
        mu: Means ``(B, D)``.
        logvar: Log-variances ``(B, D)``.

    Returns:
        Log-density ``(B, D)``.
    """
    norm = -0.5 * (math.log(2 * math.pi) + logvar)
    log_density = norm - 0.5 * ((x - mu).pow(2) * torch.exp(-logvar))
    return log_density


def _log_importance_weights(
    batch_size: int, dataset_size: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    r"""Log-weights :math:`\log w_{ij}` of :math:`q(z_i)\approx\sum_j w_{ij}\,q(z_i\mid x_j)`.

    Of the :math:`N` terms of :math:`q(z_i)=\frac1N\sum_n q(z_i\mid x_n)`, the one
    for :math:`x_i` is present exactly (weight :math:`1/N`); the other :math:`N-1`
    are represented by the :math:`B-1` remaining batch members, each weighted
    :math:`(N-1)/(N(B-1))`. Every row sums to one; :math:`N=B` gives the uniform
    minibatch mixture.
    """
    if batch_size < 2:
        return torch.full(
            (batch_size, batch_size), -math.log(dataset_size), device=device, dtype=dtype
        )
    off_diag = (dataset_size - 1) / (dataset_size * (batch_size - 1))
    weights = torch.full((batch_size, batch_size), off_diag, device=device, dtype=dtype)
    weights.fill_diagonal_(1.0 / dataset_size)
    return weights.log()


@register_loss(
    name="beta_tc_vae",
    aliases=["total_correlation", "tc_kl", "BetaTCVAELoss"],
    domain="latent",
)
class BetaTCVAELoss(nn.Module):
    r"""β-TCVAE loss with minibatch-weighted TC estimation.

    **DOMAIN**: LATENT SPACE
    **Input**: ``mu`` ``[B, D]``, ``log_var`` ``[B, D]``, ``z`` ``[B, D]``
    **3D Support**: No (flat latent embeddings)

    The three ELBO components are returned as a weighted sum::

        loss = α · MI + β · TC + γ · DW_KL

    Default weights (β=6, α=γ=1) follow the best-performing configuration
    reported in Table 2 of the original paper.

    Example::

        loss_fn = BetaTCVAELoss(beta=6.0)
        z = reparameterize(mu, log_var)
        loss = loss_fn(z, mu, log_var)

    Args:
        alpha: Weight on mutual information :math:`I_q(z; x)`.
        beta: Weight on total correlation :math:`\text{TC}(z)`.
            Higher values enforce stronger independence between
            physics and anatomy latent dimensions.
        gamma: Weight on dimension-wise KL divergence.
        dataset_size: Total dataset size for IS correction.
            When ``None``, uses the minibatch size (unbiased but
            higher variance).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 6.0,
        gamma: float = 1.0,
        dataset_size: int | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dataset_size = dataset_size

    def forward(
        self,
        z: Tensor,
        mu: Tensor,
        log_var: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute β-TCVAE loss.

        Args:
            z: Reparameterized samples ``(B, D)``.
            mu: Posterior means ``(B, D)``.
            log_var: Posterior log-variances ``(B, D)``.

        Returns:
            Scalar loss value.
        """
        B = z.shape[0]
        N = self.dataset_size if self.dataset_size is not None else B

        # ── 1. log q(z|x) — log-density of z under the individual posteriors
        log_qz_given_x = _log_density_gaussian(z, mu, log_var).sum(dim=1)
        # shape: (B,)

        # ── 2. log q(z) — log-density under the aggregated posterior
        #    q(z) ≈ (1/M) Σ_i q(z | x_i)  where M is the minibatch size
        #
        #    For each sample z_i, evaluate q(z_i | x_j) for all j in the batch.
        #    log_qz_given_xj has shape (B, B, D):
        #      [i, j, d] = log q(z_i_d | x_j)
        _z = z.unsqueeze(1)  # (B, 1, D)
        _mu = mu.unsqueeze(0)  # (1, B, D)
        _logvar = log_var.unsqueeze(0)  # (1, B, D)

        log_qz_given_xj = _log_density_gaussian(_z, _mu, _logvar)
        # (B, B, D)

        log_importance = _log_importance_weights(
            B, N, z.device, torch.promote_types(z.dtype, torch.float32)
        )

        # log q(z) — sum over D, logsumexp over batch dimension
        log_qz = torch.logsumexp(log_qz_given_xj.sum(dim=2) + log_importance, dim=1)
        # (B,)

        # log Π_d q(z_d) — product of marginals
        log_prod_qz_d = torch.logsumexp(log_qz_given_xj + log_importance.unsqueeze(-1), dim=1).sum(
            dim=1
        )
        # (B,)

        # ── 3. log p(z) — standard normal prior
        log_pz = -0.5 * (math.log(2 * math.pi) + z.pow(2)).sum(dim=1)
        # (B,)

        # ── 4. ELBO decomposition
        # MI:    E_q[log q(z|x) - log q(z)]
        mi_loss = (log_qz_given_x - log_qz).mean()

        # TC:    E_q[log q(z) - log Π_d q(z_d)]
        tc_loss = (log_qz - log_prod_qz_d).mean()

        # DW-KL: E_q[log Π_d q(z_d) - log p(z)]
        dw_kl_loss = (log_prod_qz_d - log_pz).mean()

        # Weighted combination
        loss = self.alpha * mi_loss + self.beta * tc_loss + self.gamma * dw_kl_loss

        return loss

    def extra_repr(self) -> str:
        return f"α={self.alpha}, β={self.beta}, γ={self.gamma}"


__all__ = ["BetaTCVAELoss"]
