"""Aggregate-posterior estimator of ``BetaTCVAELoss`` (Chen et al. 2018).

``q(z_i) = (1/N) sum_n q(z_i | x_n)`` over the whole dataset. In a minibatch the
self term ``q(z_i | x_i)`` is present exactly and carries weight ``1/N``; the other
``N - 1`` terms are represented by the ``B - 1`` batch neighbours.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.losses.beta_tc_vae_loss import BetaTCVAELoss, _log_importance_weights


@pytest.mark.parametrize(("batch", "dataset"), [(8, 8), (8, 1000), (2, 50)])
def test_importance_weights_are_a_mixture_with_self_weight_one_over_n(
    batch: int, dataset: int
) -> None:
    w = _log_importance_weights(batch, dataset, torch.device("cpu"), torch.float64).exp()
    torch.testing.assert_close(w.sum(dim=1), torch.ones(batch, dtype=torch.float64))
    torch.testing.assert_close(
        torch.diagonal(w), torch.full((batch,), 1.0 / dataset, dtype=torch.float64)
    )


def test_mutual_information_is_log_n_for_disjoint_posteriors() -> None:
    """Non-overlapping posteriors make ``x`` recoverable from ``z``: ``I(x; z) = log N``.

    Every off-diagonal ``q(z_i | x_j)`` underflows to zero, so the estimate reduces
    to the self term, and only a self weight of ``1/N`` yields ``log N``.
    """
    torch.manual_seed(0)
    batch, dim, dataset = 8, 3, 1000
    mu = torch.arange(batch, dtype=torch.float64).unsqueeze(1).repeat(1, dim) * 100.0
    log_var = torch.full((batch, dim), math.log(1e-2), dtype=torch.float64)
    z = mu + 0.1 * torch.randn(batch, dim, dtype=torch.float64)

    mi_only = BetaTCVAELoss(alpha=1.0, beta=0.0, gamma=0.0, dataset_size=dataset)
    assert float(mi_only(z, mu, log_var)) == pytest.approx(math.log(dataset), rel=1e-6)


def test_default_dataset_size_uses_the_uniform_minibatch_mixture() -> None:
    torch.manual_seed(1)
    batch, dim = 6, 4
    mu, log_var = torch.randn(batch, dim), 0.1 * torch.randn(batch, dim)
    z = mu + torch.randn(batch, dim) * (0.5 * log_var).exp()

    log_q = -0.5 * (
        math.log(2 * math.pi) + log_var + (z[:, None] - mu[None]).pow(2) / log_var.exp()
    )
    log_qz = torch.logsumexp(log_q.sum(2), dim=1) - math.log(batch)
    log_qz_given_x = torch.diagonal(log_q.sum(2))
    expected_mi = (log_qz_given_x - log_qz).mean()

    mi_only = BetaTCVAELoss(alpha=1.0, beta=0.0, gamma=0.0)
    torch.testing.assert_close(mi_only(z, mu, log_var), expected_mi)
