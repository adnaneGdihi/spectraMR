"""Heteroscedastic ULF restoration loss (MICCAI MRIxFields2026, B-2.9).

A Gaussian negative-log-likelihood over a predicted per-voxel mean and log-variance,
regularised by a prior that pulls the predicted variance toward the physically
calibrated ULF noise level :math:`\\sigma(B_0)^2` (Hoult). The NLL is a strictly
proper scoring rule (its minimiser is the true predictive variance); the physics
prior grounds the uncertainty map in the acquisition noise where data are sparse.
"""

from __future__ import annotations

import torch
from torch import nn

from spectramr.models.losses.registry import register_loss


def heteroscedastic_nll(
    mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    r"""Gaussian NLL ``0.5 * (exp(-logvar)·(target-mean)^2 + logvar)`` (mean-reduced)."""
    inv_var = torch.exp(-logvar)
    return 0.5 * torch.mean(inv_var * (target - mean) ** 2 + logvar)


def heteroscedastic_ulf_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    sigma_target: float | torch.Tensor,
    lambda_var_prior: float = 0.1,
    lambda_mean: float = 0.0,
    logvar_min: float = -10.0,
    logvar_max: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Gaussian NLL + Hoult variance prior, with anti-degeneracy guards (B-2.9).

    Args:
        mean, logvar: predicted per-voxel mean and log-variance ``[B, 1, H, W]``.
        target: ground-truth higher-field image.
        sigma_target: physical ULF noise std (scalar or per-sample) the variance is
            anchored toward.
        lambda_var_prior: weight of ``(sigma^2 - sigma_target^2)^2`` (0 disables).
        lambda_mean: weight of a direct L1 anchor ``|mean - target|`` on the mean
            channel. Guards the DEGENERATE hedge (#20) where the model inflates the
            variance to shrink ``exp(-logvar)*(y-mean)^2`` and lets the mean go lazy —
            the NLL alone under-constrains the mean once the variance is large. 0
            disables (bare NLL behaviour).
        logvar_min, logvar_max: hard clamp on ``logvar`` before the NLL, bounding the
            predicted variance to a physical band so it can neither blow up (a
            measurement-independent "I don't know" solution) nor collapse to a spuriously
            over-confident zero. For [0,1] data ``sigma in [~0.007, ~2.7]``.

    Returns:
        dict with ``loss_total``, ``loss_nll``, ``loss_var_prior`` (+ ``loss_mean_anchor``
        when ``lambda_mean > 0``).
    """
    logvar = logvar.clamp(logvar_min, logvar_max)  # bound variance (anti-degeneracy #20)
    nll = heteroscedastic_nll(mean, logvar, target)
    if lambda_var_prior > 0.0:
        var = torch.exp(logvar)
        st = (
            sigma_target
            if isinstance(sigma_target, torch.Tensor)
            else torch.as_tensor(sigma_target, device=var.device, dtype=var.dtype)
        )
        target_var = (st.reshape(-1, 1, 1, 1) ** 2).expand_as(var)
        var_prior = lambda_var_prior * torch.mean((var - target_var) ** 2)
    else:
        var_prior = nll.new_zeros(())
    out = {
        "loss_total": nll + var_prior,
        "loss_nll": nll,
        "loss_var_prior": var_prior,
    }
    if lambda_mean > 0.0:
        mean_anchor = lambda_mean * torch.mean(torch.abs(mean - target))
        out["loss_mean_anchor"] = mean_anchor
        out["loss_total"] = out["loss_total"] + mean_anchor
    return out


@register_loss(
    name="heteroscedastic_ulf",
    aliases=["het_ulf_nll"],
    domain="image",
    compatible_with=["heteroscedastic_ulf", "reconstruction"],
)
class HeteroscedasticULFLoss(nn.Module):
    r"""Gaussian NLL + Hoult variance prior over a 2-channel ``[mean, logvar]`` pred.

    ``prediction`` is ``[B, 2, H, W]`` (channel 0 = mean, channel 1 = log-variance);
    ``target`` is ``[B, 1, H, W]``. ``sigma_target`` (the Hoult ULF noise std) and
    ``lambda_var_prior`` are read from kwargs (the strategy supplies the per-sample
    value); defaults make the registered loss usable standalone.
    """

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        # The docstring says [B, 2, H, W] and the maths depends on it: with
        # C=1, `prediction[:, 1:2]` is an EMPTY tensor, so `logvar` is empty,
        # every reduction over it is nan, and the loss reports nan with a zero
        # gradient rather than failing. Non-negotiable #3: raise.
        if prediction.dim() < 2 or prediction.shape[1] != 2:
            raise ValueError(
                f"HeteroscedasticULFLoss needs a 2-channel prediction "
                f"[B, 2, H, W] (channel 0 = mean, channel 1 = log-variance); "
                f"got shape {tuple(prediction.shape)}. A single-channel "
                f"prediction carries no variance head to score."
            )
        mean = prediction[:, 0:1]
        logvar = prediction[:, 1:2]
        sigma_target = kwargs.get("sigma_target", 0.05)
        lam = kwargs.get("lambda_var_prior", 0.1)
        if not isinstance(sigma_target, (int, float, torch.Tensor)):
            raise TypeError(
                f"sigma_target must be a float or tensor, got {type(sigma_target).__name__}"
            )
        if not isinstance(lam, (int, float)):
            raise TypeError(f"lambda_var_prior must be a float, got {type(lam).__name__}")
        return heteroscedastic_ulf_loss(
            mean,
            logvar,
            target,
            sigma_target,
            lambda_var_prior=float(lam),
        )["loss_total"]


__all__ = [
    "HeteroscedasticULFLoss",
    "heteroscedastic_nll",
    "heteroscedastic_ulf_loss",
]
