import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


@register_loss(name="histogram_consistency", aliases=["HistogramConsistencyLoss"], domain="image")
class HistogramConsistencyLoss(nn.Module):
    """Differentiable Histogram Consistency Loss.

    **DOMAIN**: IMAGE-SPACE ONLY
    **Input**: [B, 1, H, W] or [B, C, H, W] image-space intensity
    **3D Support**: Yes (one histogram per sample, over all channels and voxels)

    Computes distance between intensity histograms using kernel density estimation
    (soft-binning) to ensure differentiability. Earth Mover's Distance approximation.
    Do NOT use with k-space [B, 2, H, W].

    References:
    - "Differentiable Histogram Loss"
    - "Earth Mover's Distance" approximations

    Args:
        bins (int): Number of histogram bins.
        min_val (float): Minimum value of the range.
        max_val (float): Maximum value of the range.
        sigma (float): Sigma for Gaussian kernel (soft binning width).
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{Histogram} = \\| \\text{CDF}(\\hat{y}) - \\text{CDF}(y) \\|_1"""

    bin_centers: torch.Tensor

    def __init__(
        self,
        bins: int = 100,
        min_val: float = 0.0,
        max_val: float = 1.0,
        sigma: float | None = None,
    ) -> None:
        """__init__.

        Args:
            bins (int): Description.
            min_val (float): Description.
            max_val (float): Description.
            sigma (float): Description.
        """
        super().__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        # Define bin centers
        self.delta = (max_val - min_val) / bins
        self.register_buffer(
            "bin_centers",
            torch.linspace(min_val + self.delta / 2, max_val - self.delta / 2, bins),
        )

        self.sigma = sigma if sigma is not None else self.delta

    def _compute_soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute soft histogram of x."""
        # Flatten x: [B, N]
        x_flat = x.reshape(x.size(0), -1)

        # [B, N, 1] - [1, 1, Bins] -> [B, N, Bins]
        x_expanded = x_flat.unsqueeze(2)
        centers_expanded = self.bin_centers.view(1, 1, -1)

        # Gaussian Kernel
        # exp( - (x - center)^2 / (2 * sigma^2) )
        x_minus_center = x_expanded - centers_expanded
        soft_assignment = torch.exp(-torch.pow(x_minus_center, 2) / (2 * self.sigma**2))

        # Sum over N pixels -> [B, Bins]
        hist = soft_assignment.sum(dim=1)

        # Normalize to probability sum=1
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)

        return hist

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
           input: Input tensor [B, C, H, W]
           target: Target tensor [B, C, H, W]

        forward method for HistogramConsistencyLoss.

        Executes PyTorch tensor operations.

        Args:
            input (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Handle multi-channel: treat each channel separately or flatten?
        # Usually we want consistent intensity distribution regardless of position.

        hist_input = self._compute_soft_histogram(input)
        hist_target = self._compute_soft_histogram(target)

        # Compute L1 distance between histograms (approx Earth Mover's if CDF used, else straightforward divergence)
        # Using CDF is more robust for EMD-like behavior (Wasserstein-1).

        cdf_input = torch.cumsum(hist_input, dim=1)
        cdf_target = torch.cumsum(hist_target, dim=1)

        # EMD approximation: L1 distance between CDFs
        loss = torch.mean(torch.abs(cdf_input - cdf_target))

        return loss
