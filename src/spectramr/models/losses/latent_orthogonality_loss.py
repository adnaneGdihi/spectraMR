"""Orthogonality between two blocks of a VAE latent: ``|| z_p^T z_a ||_F^2``.

The posterior mean ``mu`` (``[B, D]``) is split into a physics block ``z_p = mu[:, :split]``
and an anatomy block ``z_a = mu[:, split:]``. Each block is centred over the batch, and
the penalty is the squared Frobenius norm of their cross-covariance
``C = z_p_c^T z_a_c / B``, which is 0 iff the two blocks are linearly decorrelated
across samples. One sample has no covariance, so a batch of 1 raises rather than
returning a penalty that is identically 0.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from spectramr.models.losses.registry import register_loss


def latent_cross_covariance(mu: Tensor, split: int) -> Tensor:
    """``|| z_p_c^T z_a_c / B ||_F^2`` for ``z_p = mu[:, :split]``, ``z_a = mu[:, split:]``."""
    if mu.dim() != 2:
        mu = mu.flatten(1)
    batch, dim = mu.shape
    if not 0 < split < dim:
        raise ValueError(f"latent_orthogonality split={split} must lie in (0, {dim}).")
    if batch < 2:
        raise ValueError(
            "latent_orthogonality needs a batch of at least 2: a single sample has no "
            "cross-covariance, so the penalty would be 0 by construction."
        )
    centred = mu - mu.mean(dim=0, keepdim=True)
    cross = centred[:, :split].transpose(0, 1) @ centred[:, split:] / batch
    return cross.pow(2).sum()


@register_loss("latent_orthogonality", aliases=["latent_ortho"])
class LatentOrthogonalityLoss(nn.Module):
    """Decorrelate the physics and anatomy blocks of the VAE posterior mean.

    Args:
        split: width of the physics block ``z_p``; the anatomy block ``z_a`` is the rest.
    """

    def __init__(self, split: int) -> None:
        super().__init__()
        if isinstance(split, bool) or not isinstance(split, int):
            raise TypeError(f"latent_orthogonality split must be an int, got {split!r}.")
        self.split = split

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        posterior: tuple[Tensor, Tensor] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Read ``mu`` from ``posterior``; ``prediction``/``target`` are the images, unused."""
        del prediction, target, kwargs
        if posterior is None:
            raise ValueError(
                "latent_orthogonality needs the posterior (mu, logvar) from the strategy."
            )
        mu = posterior[0]
        return latent_cross_covariance(mu.float(), self.split)


__all__ = ["LatentOrthogonalityLoss", "latent_cross_covariance"]
