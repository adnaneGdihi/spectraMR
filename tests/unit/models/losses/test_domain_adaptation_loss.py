"""``DomainAdversarialLoss`` (``models/losses/domain_adaptation_loss.py``)."""

from __future__ import annotations

import torch

from spectramr.models.losses.domain_adaptation_loss import (
    DomainAdversarialConfig,
    DomainAdversarialLoss,
)


def _loss() -> DomainAdversarialLoss:
    torch.manual_seed(0)
    cfg = DomainAdversarialConfig(
        domain_classifier_hidden=8, domain_classifier_dropout=0.0, gradient_penalty_weight=0.0
    )
    return DomainAdversarialLoss(cfg, feature_dim=4)


def test_a_batch_of_one_matches_its_label_shape() -> None:
    """A bare ``squeeze()`` turned the (1, 1) logit into a 0-d tensor that BCE
    rejected against its (1,) label."""
    loss = _loss().compute_domain_loss(torch.randn(1, 4), torch.tensor([1.0]))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_the_encoder_gradient_is_reversed() -> None:
    loss_fn = _loss()
    feats = torch.randn(6, 4, requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    loss_fn.compute_domain_loss(feats, labels).backward()
    reversed_grad = feats.grad
    assert reversed_grad is not None

    plain = feats.detach().clone().requires_grad_(True)
    logits = loss_fn.domain_classifier(plain).squeeze(-1)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).backward()
    assert plain.grad is not None
    coeff = loss_fn.grl._get_coefficient()
    torch.testing.assert_close(reversed_grad, -coeff * plain.grad)
