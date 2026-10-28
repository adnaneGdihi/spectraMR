r"""High-Frequency Error Norm (HFEN) Metric.

Applies a Laplacian of Gaussian (LoG) filter to both prediction and target
before computing the L1 norm.  This highlights edges and fine anatomical
detail, proving that high-frequency content (clinical edges, tumour margins,
microtrabecular bone) is geometrically preserved by the reconstruction.

.. math::

    \text{HFEN} = \frac{\| \text{LoG}(\hat{x}) - \text{LoG}(x) \|_1}
                       {\| \text{LoG}(x) \|_1 + \epsilon}

Higher HFEN indicates worse edge preservation (lower is better).

Reference:
    Ravishankar & Bresler, "MR Image Reconstruction From Highly
    Undersampled k-Space Data by Dictionary Learning", *IEEE TMI* (2011).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from spectramr.core.metrics.registry import register_metric


def _log_kernel(size: int = 15, sigma: float = 1.5) -> Tensor:
    """Build a 2D Laplacian-of-Gaussian (LoG) kernel.

    Args:
        size: Kernel spatial extent (odd integer).
        sigma: Gaussian standard deviation.

    Returns:
        LoG kernel ``(1, 1, size, size)`` normalised to unit sum-of-squares.
    """
    ax = torch.arange(size, dtype=torch.float32) - size // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    r2 = xx**2 + yy**2
    s2 = sigma**2

    # LoG formula:  (r² - 2σ²) / σ⁴ · exp(-r² / 2σ²)
    g = torch.exp(-r2 / (2.0 * s2))
    log = ((r2 - 2.0 * s2) / (s2**2)) * g

    # Zero-mean normalisation
    log = log - log.mean()

    # Unit energy normalisation
    log = log / (log.norm() + 1e-8)

    return log.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


@register_metric("hfen", aliases=["HFEN", "HighFrequencyErrorNorm"])
class HFENMetric:
    """High-Frequency Error Norm.

    Measures fidelity of high-frequency (edge) content between prediction
    and ground truth, complementing PSNR/SSIM which are dominated by
    low-frequency structure.

    Example::

        from spectramr.core.metrics import compute_metric
        score = compute_metric("hfen", pred, target)  # lower is better
    """

    def __init__(
        self,
        kernel_size: int = 15,
        sigma: float = 1.5,
        **kwargs,  # absorb unknown kwargs (e.g. device= from InfrastructureBuilder)
    ) -> None:
        """Initialise HFEN.

        Args:
            kernel_size: LoG kernel spatial extent.
            sigma: LoG Gaussian σ.
        """
        self._kernel = _log_kernel(kernel_size, sigma)
        self._pad = kernel_size // 2
        # A pageable host-to-device copy is a sync, so the kernel is uploaded once
        # per (device, dtype) rather than on every call.
        self._kernel_cache: dict[tuple[torch.device, torch.dtype], Tensor] = {}

    @property
    def name(self) -> str:
        """Metric canonical name."""
        return "hfen"

    @property
    def higher_is_better(self) -> bool:
        """Lower HFEN → better edge preservation."""
        return False

    def _apply_log(self, x: Tensor) -> Tensor:
        """Apply LoG filter to a ``(B, C, H, W)`` or ``(B, C, D, H, W)`` tensor.

        Accepts 2D-spatial (4D) and 3D-spatial (5D) inputs. 5D volumes
        are processed slice-by-slice along ``D``; the kernel is a
        2D LoG. Rank-3 ``(C, H, W)`` and rank-2 ``(H, W)`` inputs are
        promoted with leading singleton batch dims.

        Previous implementation used ``reshape(B*C, 1, H, W)`` against
        ``x.shape[-2:]``, which silently mismatched the element count
        for 5D inputs (``B*C*D*H*W ≠ B*C*H*W``) and triggered the
        ``"shape ... invalid for input of size N"`` warning observed in
        the 2026-05-14 smoke run.
        """
        key = (x.device, x.dtype)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self._kernel.to(device=x.device, dtype=x.dtype)
            self._kernel_cache[key] = kernel

        # Promote rank to at least 4D.
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        elif x.dim() == 3:
            x = x.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.reshape(B * C, 1, H, W)
            filtered = F.conv2d(x_flat, kernel, padding=self._pad)
            return filtered.reshape(B, C, H, W)

        if x.dim() == 5:
            # (B, C, D, H, W) — fold depth into the per-slice batch
            B, C, D, H, W = x.shape
            x_flat = x.reshape(B * C * D, 1, H, W)
            filtered = F.conv2d(x_flat, kernel, padding=self._pad)
            return filtered.reshape(B, C, D, H, W)

        raise ValueError(
            f"[HFEN] Unsupported input rank {x.dim()} (shape={tuple(x.shape)}). "
            "Expected 2D/3D/4D/5D tensor; a 2D LoG kernel cannot be applied "
            "to higher-rank tensors without an explicit reduction policy."
        )

    @torch.no_grad()
    def __call__(
        self,
        prediction: Tensor,
        target: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute HFEN.

        Args:
            prediction: Model output ``(B, C, H, W)``.
            target: Ground truth ``(B, C, H, W)``.

        Returns:
            0-d tensor on the input's device (lower = better). It stays on the
            device so the metrics computer's fused host transfer is the only sync.
        """
        # Handle complex inputs → magnitude
        if torch.is_complex(prediction):
            prediction = prediction.abs()
        if torch.is_complex(target):
            target = target.abs()

        pred_log = self._apply_log(prediction)
        tgt_log = self._apply_log(target)

        numerator = (pred_log - tgt_log).abs().sum()
        denom = tgt_log.abs().sum() + 1e-8

        return numerator / denom


__all__ = ["HFENMetric"]
