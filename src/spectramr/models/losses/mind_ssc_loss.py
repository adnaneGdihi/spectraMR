"""MIND-SSC Loss for intensity-invariant anatomical structure matching.

MIND-SSC (Modality Independent Neighbourhood Descriptor with Self-Similarity Context)
computes self-similarity features that are robust to intensity transformations between
different MRI contrasts (e.g., ULF vs HF).

References:
    Heinrich et al. "MIND: Modality independent neighbourhood descriptor for multi-modal
    deformable registration." Medical Image Analysis, 2012.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


@register_loss("mind_ssc", aliases=["MINDSSCLoss", "anat_lock", "mind_loss"], domain="image")
class MINDSSCLoss(nn.Module):
    """MIND-SSC loss for anatomical structure preservation.

    **DOMAIN**: IMAGE-SPACE ONLY
    **Input**: [B, 1, H, W] or [B, C, H, W] image-space intensity
    **3D Support**: No (2-D neighbourhood; a 5-D input raises)

    Intensity-invariant self-similarity descriptors robust to non-linear intensity
    transformations (e.g., ULF vs HF contrasts). Do NOT use with k-space [B, 2, H, W].

    Computes self-similarity features by comparing local patches using
    Gaussian-weighted differences.

    Args:
        patch_size: Size of local neighborhood for descriptor (default: 7)
        search_radius: Radius for self-similarity context (default: 3)
        gaussian_sigma: Sigma for Gaussian weighting (default: 1.0)
        use_self_similarity: Enable self-similarity context (default: True)

    Forward Args:
        pred: Predicted image [B, C, H, W]
        target: Target image [B, C, H, W]

    Returns:
        MIND-SSC loss (L2 distance between descriptors)

    Example:
        >>> loss_fn = MINDSSCLoss(patch_size=7, search_radius=3)
        >>> loss = loss_fn(pred_image, target_image)
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{MIND} = \\| \\text{MIND}(\\hat{y}) - \\text{MIND}(y) \\|_2^2"""

    gaussian_kernel: torch.Tensor
    mean_kernel: torch.Tensor

    def __init__(
        self,
        patch_size: int = 7,
        search_radius: int = 3,
        gaussian_sigma: float = 1.0,
        use_self_similarity: bool = True,
    ) -> None:
        """__init__.

        Args:
            patch_size (int): Description.
            search_radius (int): Description.
            gaussian_sigma (float): Description.
            use_self_similarity (bool): Description.
        """
        super().__init__()
        self.patch_size = patch_size
        self.search_radius = search_radius
        self.gaussian_sigma = gaussian_sigma
        self.use_self_similarity = use_self_similarity
        self.padding = patch_size // 2

        # Pre-compute the 8 directional Gaussian kernels for the MIND descriptor.
        # Directions: the full 3x3 neighborhood (excluding the centre). Using the
        # COMPLETE, symmetric neighbourhood is what makes MIND insensitive to
        # orientation. The previous ``directions[:6]`` slice kept an asymmetric
        # subset -- it dropped (1, 0) and (1, 1) while keeping (-1, 0)/(-1, 1) --
        # which biased the descriptor toward the upper-left and broke that
        # rotation-insensitivity.
        self.directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),  # Top row
            (0, -1),
            (0, 1),  # Middle row (skip center)
            (1, -1),
            (1, 0),
            (1, 1),  # Bottom row
        ]
        self.register_buffer(
            "gaussian_kernel",
            self._compute_gaussian_kernel(patch_size, gaussian_sigma).view(
                1, 1, patch_size, patch_size
            ),
            persistent=False,
        )
        self.register_buffer(
            "mean_kernel",
            torch.full((1, 1, patch_size, patch_size), 1.0 / patch_size**2),
            persistent=False,
        )

    def _compute_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2

        g = coords**2
        g = (-g / (2 * sigma**2)).exp()

        kernel = g.unsqueeze(-1) @ g.unsqueeze(0)
        kernel /= kernel.sum()
        return kernel

    def _compute_mind_descriptor(self, img: torch.Tensor) -> torch.Tensor:
        """Compute MIND descriptor for image.

        Args:
            img: Input image [B, C, H, W]

        Returns:
            MIND descriptor [B, C*8, H, W] (8 directional features per channel)
        """
        if img.dim() != 4:
            raise ValueError(f"MINDSSCLoss expects [B, C, H, W], got {tuple(img.shape)}")

        # Squared differences to every shifted copy, stacked direction-major so a
        # single grouped conv applies the per-channel Gaussian to all of them.
        diffs = [
            (img - torch.roll(img, shifts=(dy, dx), dims=(-2, -1))) ** 2
            for dx, dy in self.directions
        ]
        stacked = torch.cat(diffs, dim=1)  # [B, C*8, H, W]
        groups = stacked.shape[1]
        gaussian = self.gaussian_kernel.to(device=img.device, dtype=img.dtype)
        gaussian = gaussian.expand(groups, 1, -1, -1)
        mind = F.conv2d(stacked, gaussian, padding=self.padding, groups=groups)

        # Variance normalization (self-similarity)
        if self.use_self_similarity:
            box = self.mean_kernel.to(device=mind.device, dtype=mind.dtype)
            box = box.expand(groups, 1, -1, -1)
            local_mean = F.conv2d(mind, box, padding=self.padding, groups=groups)
            local_var = F.conv2d(mind**2, box, padding=self.padding, groups=groups) - local_mean**2

            # Normalize by local variance (add epsilon for stability)
            # [FIX] Clamp variance to be non-negative AND add epsilon before sqrt to prevent NaN gradient at 0
            # Variance computation (E[x^2] - E[x]^2) can be negative due to float precision
            local_var = torch.clamp(local_var, min=1e-6)
            mind = mind / local_var.sqrt()

        return mind

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute MIND-SSC loss between predicted and target images.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]

        Returns:
            Scalar loss value

        forward method for MINDSSCLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Compute MIND descriptors for both images
        mind_pred = self._compute_mind_descriptor(pred)
        mind_target = self._compute_mind_descriptor(target)

        # L2 distance between descriptors
        loss = F.mse_loss(mind_pred, mind_target)

        return loss
