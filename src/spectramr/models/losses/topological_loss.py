import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


@register_loss(name="topological", aliases=["PersistentHomologyLoss"])
class PersistentHomologyLoss(nn.Module):
    """Topological Loss based on Persistent Homology Approximation.

    Instead of full Cubical Complex calculation (which is computationally expensive),
    we approximate the 0-th order persistence (connected components) via
    differentiable Euler Characteristic (EC) on super-level sets.

    This penalizes the network if it hallucinates handles (b_1) or breaks components (b_0)
    relative to the topological structure of the target.

    Reference:
        Hu, X., et al. (2019). Topology-Aware Segmentation using Discrete Morse Theory. ICLR.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{PH} = \\mathbb{E} \\left[ (\\chi(\\hat{y}_{>t}) - \\chi(y_{>t}))^2 \\right]"""

    def __init__(self, levels: int = 10):
        """__init__.

        Args:
            levels (int): Description.
        """
        super().__init__()
        self.levels = levels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probability or intensity map [B, 1, H, W] (normalized 0-1)
            target: Ground truth [B, 1, H, W] (normalized 0-1)

        Returns:
            Scalar loss

        forward method for PersistentHomologyLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Squash an out-of-[0, 1] prediction; the branch is taken on device
        # (torch.where) because a Python ``if`` on a tensor syncs every step.
        out_of_range = (pred.amax() > 1.0) | (pred.amin() < 0.0)
        pred = torch.where(out_of_range, torch.sigmoid(pred), pred)

        loss = torch.zeros((), device=pred.device, dtype=pred.dtype)

        # Iterate through filtration levels (t)
        # We compare the Euler Characteristic (EC) of the super-level sets L(t) = {x | f(x) > t}
        level_values = torch.linspace(0.1, 0.9, self.levels, device=pred.device)

        for t in level_values:
            # Differentiable approximation of super-level set
            # Sigmoid with large scale acts as soft step function
            scale = 20.0
            pred_set = torch.sigmoid(scale * (pred - t))
            # Use same soft approximation for target to ensure consistency at identity
            # (or use hard target if strict segmentation is required, but for loss stability soft is better)
            target_set = torch.sigmoid(scale * (target - t))

            # Compute EC for this level set
            ec_pred = self.compute_euler_characteristic_2d(pred_set)
            # Target EC is constant w.r.t network weights, but we compute it to match
            ec_target = self.compute_euler_characteristic_2d(target_set)

            # Loss is MSE between EC curves.
            # CRITICAL: do NOT call .item() here — that detaches from autograd.
            # The previous version of this loss was effectively a no-op for
            # backprop because of an .item() in this exact spot.
            loss = loss + (ec_pred - ec_target).pow(2).mean()

        return loss / self.levels

    def compute_euler_characteristic_2d(self, img: torch.Tensor) -> torch.Tensor:
        """
        Computes Euler Characteristic (EC) = V - E + F for a binary/soft image.
        Uses 2x2 local configurations.

        EC = (1/4) * (N1 - N3 + 2*N4Diag) ?
        Or standard summing local contributions.

        Standard formula for 4-connectivity:
        EC = Sum(Vertices) - Sum(Edges) + Sum(Faces)

        V: Active pixels (d)
        E: Active Horizontal + Active Vertical edges
        F: Active 2x2 squares

        Soft approximation:
        V = sum(p)
        e_h = sum(p[i,j] * p[i, j+1])
        e_v = sum(p[i,j] * p[i+1, j])
        F = sum(p[i,j] * p[i,j+1] * p[i+1, j] * pi+1, j+1])
        """
        # Pad to avoid boundary issues
        # img: [B, C, H, W]
        p = F.pad(img, (0, 1, 0, 1), mode="constant", value=0)

        # Vertices (Pixels)
        V = p.sum(dim=(-1, -2))

        # Horizontal neighbors (Edges)
        # p * p_right
        h_neighbors = p[..., :, :-1] * p[..., :, 1:]
        e_h = h_neighbors.sum(dim=(-1, -2))

        # Vertical neighbors (Edges)
        v_neighbors = p[..., :-1, :] * p[..., 1:, :]
        e_v = v_neighbors.sum(dim=(-1, -2))

        # Faces (2x2 squares)
        # p * p_right * p_down * p_down_right
        faces = p[..., :-1, :-1] * p[..., :-1, 1:] * p[..., 1:, :-1] * p[..., 1:, 1:]
        f_quad = faces.sum(dim=(-1, -2))

        # EC = V - E + F = V - (e_h + e_v) + f_quad
        # Note: This formula depends on connectivity (4 vs 8).
        # For 4-connectivity (objects) and 8-connectivity (holes):
        # EC = V - E + F

        ec = V - (e_h + e_v) + f_quad

        return ec
