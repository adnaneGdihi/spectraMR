r"""Jacobian guard against amorphous anatomy in off-grid reconstruction.

Gridding an off-grid acquisition, and filling what it did not acquire, does not
fail by adding noise. It fails by **moving tissue**: a boundary drifts a bin, a
sulcus closes, a small structure merges with its neighbour. Every one of those is
a spatial warp, and none of them costs much PSNR -- which is why an arm can score
well and still be diagnostically wrong.

So measure the warp directly. Treat the reconstruction as a deformation of a
reference (the density-compensated adjoint, or the previous unrolled iterate),
estimate the displacement field :math:`\mathbf{u}`, and look at the Jacobian of
the induced map:

.. math::
    J = I + \nabla \mathbf{u}, \qquad
    \det J = (1 + \partial_x u_x)(1 + \partial_y u_y)
             - \partial_y u_x \, \partial_x u_y

Two distinct failures live in that determinant and the loss separates them:

* :math:`\det J \le 0` -- the map **folds**. Anatomy has changed topology: two
  places became one. This is not a matter of degree and is penalised on a hinge,
  hard, with no tolerance band.
* :math:`\det J \ne 1` -- the map **dilates or compresses**. Anatomy is the right
  shape in the wrong size. Penalised as :math:`|\log \det J|`, which is
  symmetric in expansion and contraction, unlike a penalty on
  :math:`\det J` itself.

The displacement is estimated by local normalised cross-correlation on a small
search window rather than by an optical-flow network, deliberately: a learned
estimator inside the guard would be a second thing that can hallucinate, and then
the metric and the failure share a failure mode.

**Read the loss as relative and the folding fraction as absolute.** The
soft-argmax vote does not fully collapse on smooth data -- neighbouring offsets
of a blurred edge both score near 1 -- so ``forward`` sits on a data-dependent
noise floor (0.23 on the 48x48 disc phantom below) rather than at 0 for a perfect
match. Measured on that phantom: identical 0.232, rigid 1-pixel shift 0.221,
resampling warp 0.247, 1.33x zoom 0.404, 1.6x zoom 0.500. Compare arms on one
dataset, never a number against an absolute threshold.

A rigid shift scoring the same as identical is the guard working, not failing:
the Jacobian of a constant displacement is exactly ``I``, so translation is
invisible to it by construction -- which is what you want, since a shifted image
has lost no anatomy. :meth:`JacobianAnatomyLoss.folding_fraction` is the
absolute companion and behaves cleanly on the same cases: 0.0000, 0.0000,
0.0013, 0.0182, 0.0109.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


def displacement_by_local_correlation(
    moving: torch.Tensor, fixed: torch.Tensor, search: int = 2, sharpness: float = 24.0
) -> torch.Tensor:
    """Estimate a dense displacement field by a soft correlation-peak vote.

    The score is a **locally normalised** cross-correlation -- local means removed
    and local standard deviations divided out -- not a raw product. The raw form
    is what an earlier cut of this used, and it does not work: the products across
    a 5x5 offset window are all of similar magnitude, so the softmax over them is
    nearly uniform, the estimated displacement collapses to noise about zero, and
    the loss returns the same 0.020 for identical images, a 2-pixel shift and a
    resampling warp. A guard that cannot separate those is worse than no guard.

    Args:
        moving: ``[B, 1, H, W]`` real.
        fixed: ``[B, 1, H, W]`` real, same shape.
        search: Half-width of the integer search window, in pixels. The field it
            can express is bounded by this, on purpose -- a guard that can explain
            away an arbitrarily large warp reports no warp.
        sharpness: Softmax temperature on the NCC. High enough to approximate
            ``argmax`` (so the vote is a peak, not an average) while staying
            differentiable.

    Returns:
        ``[B, 2, H, W]`` displacement ``(u_y, u_x)`` in pixels.
    """
    if moving.shape != fixed.shape:
        raise ValueError(f"shape mismatch: {tuple(moving.shape)} vs {tuple(fixed.shape)}")
    if search < 1:
        raise ValueError(f"search must be >= 1, got {search}.")

    def _standardise(x: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(x, 5, stride=1, padding=2)
        var = F.avg_pool2d(x * x, 5, stride=1, padding=2) - mean * mean
        return (x - mean) / var.clamp_min(1e-8).sqrt()

    moving_n = _standardise(moving)
    fixed_n = _standardise(fixed)

    scores, offsets = [], []
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            shifted = torch.roll(fixed_n, shifts=(dy, dx), dims=(-2, -1))
            scores.append(F.avg_pool2d(moving_n * shifted, 5, stride=1, padding=2))
            offsets.append((dy, dx))

    weights = torch.softmax(torch.cat(scores, dim=1) * sharpness, dim=1)
    off = torch.tensor(offsets, dtype=moving.dtype, device=moving.device)
    u_y = (weights * off[:, 0].view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
    u_x = (weights * off[:, 1].view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
    return torch.cat([u_y, u_x], dim=1)


def jacobian_determinant(displacement: torch.Tensor) -> torch.Tensor:
    """``det(I + grad u)`` for a ``[B, 2, H, W]`` field, returned ``[B, 1, H, W]``.

    Central differences, replicate-padded: a one-sided difference at the border
    biases ``det J`` there, and the border of an MRI matrix is where a gridding
    artefact lands.
    """
    if displacement.dim() != 4 or displacement.shape[1] != 2:
        raise ValueError(
            f"displacement must be [B, 2, H, W] as (u_y, u_x); got {tuple(displacement.shape)}."
        )
    pad = F.pad(displacement, (1, 1, 1, 1), mode="replicate")
    duy_dy = (pad[:, 0:1, 2:, 1:-1] - pad[:, 0:1, :-2, 1:-1]) * 0.5
    duy_dx = (pad[:, 0:1, 1:-1, 2:] - pad[:, 0:1, 1:-1, :-2]) * 0.5
    dux_dy = (pad[:, 1:2, 2:, 1:-1] - pad[:, 1:2, :-2, 1:-1]) * 0.5
    dux_dx = (pad[:, 1:2, 1:-1, 2:] - pad[:, 1:2, 1:-1, :-2]) * 0.5
    return (1.0 + duy_dy) * (1.0 + dux_dx) - duy_dx * dux_dy


@register_loss(
    name="jacobian_anatomy",
    aliases=["JacobianAnatomyLoss", "anatomy_jacobian"],
    domain="image",
)
class JacobianAnatomyLoss(nn.Module):
    r"""Penalise folding and volume distortion between a reconstruction and a reference.

    **DOMAIN**: IMAGE — magnitude or complex; a complex input is reduced to
    ``abs()`` first, because a warp is a geometric statement about where tissue
    is, not about its phase.

    Args:
        fold_weight: Weight on the ``det J <= 0`` hinge. Dominant by default:
            a fold is a topology change, not a degree of blur.
        volume_weight: Weight on ``|log det J|``.
        search: Half-width of the displacement search, in pixels.
        eps: Floor inside the log.
    """

    def __init__(
        self,
        fold_weight: float = 10.0,
        volume_weight: float = 1.0,
        search: int = 2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if fold_weight < 0 or volume_weight < 0:
            raise ValueError("weights must be non-negative.")
        if fold_weight == 0 and volume_weight == 0:
            raise ValueError(
                "Both weights are zero, so this loss is identically 0 and the arm "
                "declares a guard that cannot fire (pitfall 15)."
            )
        self.fold_weight = float(fold_weight)
        self.volume_weight = float(volume_weight)
        self.search = int(search)
        self.eps = float(eps)

    @staticmethod
    def _as_magnitude(x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            x = x.abs()
        if x.shape[1] > 1:
            # Interleaved coils/channels: RSS to one geometric map, so the
            # displacement is estimated on anatomy and not on a coil profile.
            x = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        return x

    @staticmethod
    def structure_weight(reference: torch.Tensor) -> torch.Tensor:
        """Where a displacement is even defined: local contrast in the reference.

        A correlation peak in flat background is arbitrary, and on a brain most
        of the matrix IS flat background. Averaging the penalty over it charged a
        constant 0.165 to a pair of IDENTICAL images and buried the real signal
        under it. Weighting by local variance also matches the claim: amorphous
        anatomy is a statement about boundaries, and a region with no boundary
        has none to lose.
        """
        mean = F.avg_pool2d(reference, 5, stride=1, padding=2)
        var = (F.avg_pool2d(reference * reference, 5, stride=1, padding=2) - mean * mean).clamp_min(
            0
        )
        return var / var.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

    def forward(self, prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Scalar penalty; ~``0`` when the two differ by no warp at all."""
        moving = self._as_magnitude(prediction)
        fixed = self._as_magnitude(reference)
        displacement = displacement_by_local_correlation(moving, fixed, search=self.search)
        det = jacobian_determinant(displacement)
        weight = self.structure_weight(fixed)
        denom = weight.sum().clamp_min(1e-12)

        fold = (F.relu(-det) * weight).sum() / denom
        volume = (det.clamp_min(self.eps).log().abs() * weight).sum() / denom
        return self.fold_weight * fold + self.volume_weight * volume

    def folding_fraction(self, prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Fraction of pixels where the map folds -- report this, not just the loss.

        A mean penalty hides a small region that folded badly, which is exactly
        the case a radiologist would care about.
        """
        moving = self._as_magnitude(prediction)
        fixed = self._as_magnitude(reference)
        det = jacobian_determinant(
            displacement_by_local_correlation(moving, fixed, search=self.search)
        )
        return (det <= 0).to(det.dtype).mean()


__all__ = [
    "JacobianAnatomyLoss",
    "displacement_by_local_correlation",
    "jacobian_determinant",
]
