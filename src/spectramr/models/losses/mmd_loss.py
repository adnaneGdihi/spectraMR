r"""Maximum Mean Discrepancy (MMD) — kernel-based two-sample distance.

For two empirical distributions :math:`P, Q` with samples
:math:`\{x_i\}, \{y_j\}` and a positive-definite kernel :math:`k`,

.. math::
    \mathrm{MMD}^2(P, Q) =
        \tfrac{1}{n(n-1)}\sum_{i\ne j} k(x_i, x_j)
      + \tfrac{1}{m(m-1)}\sum_{i\ne j} k(y_i, y_j)
      - \tfrac{2}{nm}\sum_{i,j} k(x_i, y_j).

In MRI MMD is the canonical *unbalanced* harmonisation /
domain-alignment loss: it forces feature distributions across two
sites or two acquisition protocols to match without requiring paired
data, and unlike adversarial alignment it has a closed-form gradient
and no min-max instabilities.

This implementation uses a sum-of-Gaussian-kernels (multi-bandwidth)
:math:`k(x,y) = \sum_s \exp(-\|x-y\|^2/(2 s^2 \sigma_{med}^2))`, where
:math:`\sigma_{med}^2` is the median squared distance over all pairs of the
pooled sample (Gretton et al. 2012). Squared distances between flattened
images grow with the number of pixels, so a fixed absolute bandwidth sends
every off-diagonal kernel value to 0 and MMD to a constant with no gradient
(#2277); scaling by the median keeps the kernel in its informative range at
any image size.

Reference:
    A. Gretton *et al.*, "A kernel two-sample test," *JMLR*, 13, 2012.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss

#: Accepted ``bandwidth_mode`` values; anything else raises.
BANDWIDTH_MODES = ("median", "absolute")


def check_bandwidth_mode(mode: str) -> str:
    """Return ``mode`` if it is a known bandwidth mode, else raise."""
    if mode not in BANDWIDTH_MODES:
        raise ValueError(f"bandwidth_mode must be one of {BANDWIDTH_MODES}; got {mode!r}")
    return mode


def pooled_gaussian_kernel_sum(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: tuple[float, ...],
    bandwidth_mode: str = "median",
) -> torch.Tensor:
    """Sum of Gaussian kernels over the pooled sample ``[x; y]``, shape ``(n+m, n+m)``.

    With ``bandwidth_mode="median"`` each ``sigma`` multiplies the median pairwise
    distance of the pooled sample; with ``"absolute"`` it is used as given. The
    median is detached so the bandwidth cannot become a path that lowers the loss.
    """
    z = torch.cat([x, y], dim=0)
    d2 = torch.cdist(z, z).pow(2)
    if check_bandwidth_mode(bandwidth_mode) == "median":
        n = z.shape[0]
        rows, cols = torch.triu_indices(n, n, offset=1, device=z.device)
        pairs = d2.detach()[rows, cols]
        # Median over distinct pairs: duplicates would drive it to 0 and turn the
        # kernel into an exact-match test. With no distinct pair every kernel is 1
        # at any bandwidth, so the fill value is immaterial.
        distinct = torch.where(pairs > 0, pairs, torch.nan)
        scale = torch.nan_to_num(distinct.nanmedian(), nan=1.0)
    else:
        scale = torch.ones((), device=d2.device, dtype=d2.dtype)
    kernel = torch.zeros_like(d2)
    for sigma in sigmas:
        kernel = kernel + torch.exp(-d2 / (2.0 * sigma * sigma * scale))
    return kernel


def _multi_kernel_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: tuple[float, ...],
    bandwidth_mode: str = "median",
) -> torch.Tensor:
    """Squared MMD with sum-of-Gaussian-kernels (unbiased estimator)."""
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        raise ValueError(f"MMD needs at least 2 samples per side, got n={n}, m={m}.")
    kernel = pooled_gaussian_kernel_sum(x, y, sigmas, bandwidth_mode)
    k_xx, k_yy, k_xy = kernel[:n, :n], kernel[n:, n:], kernel[:n, n:]
    sum_xx = (k_xx.sum() - k_xx.diagonal().sum()) / (n * (n - 1))
    sum_yy = (k_yy.sum() - k_yy.diagonal().sum()) / (m * (m - 1))
    return sum_xx + sum_yy - 2.0 * k_xy.mean()


@register_loss(name="mmd", aliases=["MMDLoss", "maximum_mean_discrepancy"], domain="agnostic")
class MMDLoss(nn.Module):
    r"""Multi-bandwidth Gaussian-kernel MMD.

    Forward signature: ``prediction``/``target`` are flattened per-batch
    to ``(N, D)`` feature vectors before computing MMD. They do *not*
    need to be paired and may have different batch sizes.

    Args:
        sigmas: Bandwidths for the sum-of-Gaussians kernel: multiples of the
            median pairwise distance under ``bandwidth_mode="median"``,
            distances in feature units under ``"absolute"``. The default
            :math:`(1/4, \ldots, 4)` spans two octaves either side of the median.
        flatten_spatial: If True (default) flatten ``(B, C, H, W) →
            (B, C·H·W)`` before computing MMD. If False the input must
            already be ``(B, D)``.
        bandwidth_mode: ``"median"`` (default) or ``"absolute"``; see above.
    """

    def __init__(
        self,
        sigmas: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
        flatten_spatial: bool = True,
        bandwidth_mode: str = "median",
    ) -> None:
        super().__init__()
        if not sigmas:
            raise ValueError("sigmas must contain at least one bandwidth.")
        if any(s <= 0 for s in sigmas):
            raise ValueError(f"all sigmas must be > 0, got {sigmas}")
        self.sigmas = tuple(sigmas)
        self.flatten_spatial = flatten_spatial
        self.bandwidth_mode = check_bandwidth_mode(bandwidth_mode)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if self.flatten_spatial:
            x = prediction.reshape(prediction.shape[0], -1)
            y = target.reshape(target.shape[0], -1)
        else:
            x, y = prediction, target
        return _multi_kernel_mmd(x, y, self.sigmas, self.bandwidth_mode)
