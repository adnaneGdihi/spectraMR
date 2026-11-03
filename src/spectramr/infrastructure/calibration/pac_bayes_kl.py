r"""Inputs of the R5 PAC-Bayes certificate: ``KL(Q||P)`` and the Gibbs empirical risk.

A point-mass posterior has infinite KL to any continuous prior, so the certificate uses
the Dziugaite-Roy (2017) Gaussian form: :math:`Q = \mathcal N(w_Q, \sigma_Q^2 I)` around the
certified checkpoint and :math:`P = \mathcal N(w_P, \sigma_P^2 I)` around the prior
checkpoint. Both sigmas are fixed by config before the bound sample is seen; tuning them on
that sample would need a union-bound term the certificate does not carry.

The McAllester bound holds for the risk of the stochastic (Gibbs) predictor, so the
empirical risk is a Monte-Carlo estimate of :math:`E_{\theta\sim Q}[\hat L(\theta)]` over
weight perturbations, not the loss at the mean.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from spectramr.core.metrics.quantitative.pac_bayes_certificate import kl_isotropic_gaussian

KLEstimator = Callable[..., tuple[float, int, float]]
KL_ESTIMATOR_REGISTRY: dict[str, KLEstimator] = {}


def register_kl_estimator(name: str) -> Callable[[KLEstimator], KLEstimator]:
    """Register a ``KL(Q||P)`` estimator under ``name`` (the config's ``kl_estimator``)."""

    def deco(fn: KLEstimator) -> KLEstimator:
        if name in KL_ESTIMATOR_REGISTRY:
            raise ValueError(f"KL estimator {name!r} is already registered")
        KL_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return deco


def resolve_kl_estimator(name: str) -> KLEstimator:
    """Return the estimator registered as ``name``; raise on an unknown name."""
    if name not in KL_ESTIMATOR_REGISTRY:
        raise ValueError(
            f"unknown certification.pac_bayes.kl_estimator {name!r}; "
            f"registered: {sorted(KL_ESTIMATOR_REGISTRY)}"
        )
    return KL_ESTIMATOR_REGISTRY[name]


def _matched_prior(
    posterior: Mapping[str, torch.Tensor], prior: Mapping[str, torch.Tensor]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pair every posterior parameter with its prior tensor; raise on any mismatch."""
    missing = sorted(set(posterior) - set(prior))
    if missing:
        raise ValueError(
            f"prior checkpoint lacks {len(missing)} posterior parameters (e.g. {missing[:3]}); "
            "a partial KL(Q||P) would understate the complexity term."
        )
    pairs = []
    for name, w_q in posterior.items():
        w_p = prior[name]
        if tuple(w_p.shape) != tuple(w_q.shape):
            raise ValueError(
                f"prior parameter {name!r} has shape {tuple(w_p.shape)}, posterior "
                f"{tuple(w_q.shape)}; the prior must share the certified architecture."
            )
        pairs.append((w_q.detach(), w_p.detach().to(device=w_q.device, dtype=w_q.dtype)))
    return pairs


@register_kl_estimator("isotropic_gaussian")
def isotropic_gaussian_kl(
    posterior: Mapping[str, torch.Tensor],
    prior: Mapping[str, torch.Tensor],
    *,
    posterior_sigma: float,
    prior_sigma: float,
) -> tuple[float, int, float]:
    r"""``(KL, n_parameters, ||w_Q - w_P||_2)`` for isotropic Gaussians around two checkpoints.

    KL is shift-invariant, so each parameter's term is the zero-mean-prior closed form of
    :func:`kl_isotropic_gaussian` evaluated at :math:`\mu = w_Q - w_P`.
    """
    if posterior_sigma <= 0 or prior_sigma <= 0:
        raise ValueError("posterior_sigma and prior_sigma must be positive")
    log_var = 2.0 * math.log(posterior_sigma)
    prior_var = prior_sigma**2
    kl = torch.zeros((), dtype=torch.float64)
    sq = torch.zeros((), dtype=torch.float64)
    count = 0
    for w_q, w_p in _matched_prior(posterior, prior):
        delta = (w_q - w_p).double().flatten()
        kl = kl + kl_isotropic_gaussian(delta, torch.full_like(delta, log_var), prior_var).cpu()
        sq = sq + delta.pow(2).sum().cpu()
        count += delta.numel()
    return float(kl), count, float(sq.sqrt())


def nmse_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image normalised squared error ``||pred - target||^2 / ||target||^2``, shape ``(B,)``."""
    if pred.shape != target.shape:
        raise ValueError(
            f"pac_bayes risk: prediction {tuple(pred.shape)} and target {tuple(target.shape)} "
            "differ in shape; the per-image loss is undefined."
        )
    err = (pred - target).abs().pow(2).flatten(start_dim=1).sum(dim=1)
    ref = target.abs().pow(2).flatten(start_dim=1).sum(dim=1)
    return err / ref.clamp_min(torch.finfo(ref.dtype).tiny)


RISK_LOSSES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "nmse": nmse_per_image,
}


@dataclass(frozen=True)
class GibbsRisk:
    """Monte-Carlo Gibbs loss per scored image, before bounding and grouping."""

    per_image: list[float]
    mc_standard_error: float
    n_truncated: int


@torch.no_grad()
def gibbs_per_image_loss(
    model: nn.Module,
    batches: Iterable[Any],
    predict: Callable[[Any, nn.Module], tuple[torch.Tensor, torch.Tensor]],
    *,
    risk_loss: str,
    posterior_sigma: float,
    n_mc_samples: int,
    seed: int,
    loss_bound: float,
    out_of_range: str,
) -> GibbsRisk:
    """Mean loss of each image over ``n_mc_samples`` draws ``w ~ N(w_Q, sigma_Q^2 I)``.

    The weights are perturbed in place and restored exactly afterwards, even on error.
    """
    if risk_loss not in RISK_LOSSES:
        raise ValueError(f"unknown pac_bayes risk_loss {risk_loss!r}; known: {sorted(RISK_LOSSES)}")
    if out_of_range not in ("raise", "truncate"):
        raise ValueError(f"unknown pac_bayes loss_out_of_range {out_of_range!r}")
    if n_mc_samples < 1:
        raise ValueError("n_mc_samples must be >= 1")
    loss_fn = RISK_LOSSES[risk_loss]
    batches = list(batches)
    params = list(model.parameters())
    if not params:
        raise ValueError("pac_bayes: the certified model has no parameters to perturb")
    saved = [p.detach().clone() for p in params]
    gen = torch.Generator(device=params[0].device).manual_seed(seed)
    was_training = model.training
    model.eval()
    draws: list[torch.Tensor] = []
    try:
        for _ in range(n_mc_samples):
            noise = [
                torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype) for p in params
            ]
            torch._foreach_add_(params, noise, alpha=posterior_sigma)
            draws.append(torch.cat([loss_fn(*predict(b, model)).float() for b in batches]))
            torch._foreach_copy_(params, saved)
    finally:
        torch._foreach_copy_(params, saved)
        model.train(was_training)
    losses = torch.stack(draws)  # (n_mc, n_images)
    over = losses > loss_bound
    n_over = int(over.sum())
    if n_over and out_of_range == "raise":
        raise ValueError(
            f"pac_bayes: {n_over} per-image {risk_loss} losses exceed loss_bound_M={loss_bound} "
            f"(max {float(losses.max()):.4g}); the McAllester bound holds only for losses in "
            "[0, M]. Raise loss_bound_M or declare loss_out_of_range: truncate."
        )
    losses = losses.clamp(max=loss_bound)
    per_image = losses.mean(dim=0)
    run_means = losses.mean(dim=1)
    se = float(run_means.std(unbiased=True) / math.sqrt(n_mc_samples)) if n_mc_samples > 1 else 0.0
    return GibbsRisk(per_image=per_image.tolist(), mc_standard_error=se, n_truncated=n_over)


def group_draws(per_image: list[float], subject_ids: list[str] | None) -> list[float]:
    """One draw per independent unit: subject means when ids are given, else the images."""
    if subject_ids is None:
        return list(per_image)
    if len(subject_ids) != len(per_image):
        raise ValueError(f"{len(subject_ids)} subject ids for {len(per_image)} scored images")
    groups: dict[str, list[float]] = {}
    for sid, loss in zip(subject_ids, per_image, strict=True):
        groups.setdefault(sid, []).append(loss)
    return [sum(v) / len(v) for v in groups.values()]


__all__ = [
    "KL_ESTIMATOR_REGISTRY",
    "RISK_LOSSES",
    "GibbsRisk",
    "gibbs_per_image_loss",
    "group_draws",
    "isotropic_gaussian_kl",
    "nmse_per_image",
    "register_kl_estimator",
    "resolve_kl_estimator",
]
