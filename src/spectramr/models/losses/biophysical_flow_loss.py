import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


@register_loss(name="biophysical_flow", aliases=["BiophysicalFlowLoss", "velocity_prediction"])
class BiophysicalFlowLoss(nn.Module):
    """Joint loss function for LaNS (Flow) and HoBS (Biophysics).
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{BiophysicalFlow} = \\lambda_{dc} \\|k_{pred} - k_{target}\\|_1 + \\lambda_{bloch} \\mathcal{L}_{bloch} + \\lambda_{sparsity} \\|\\alpha\\|_1 + \\lambda_{smooth} \\|\\psi\\|_2^2"""

    def __init__(
        self,
        lambda_dc: float = 1.0,
        lambda_bloch: float = 0.1,
        lambda_sparsity: float = 0.01,
        lambda_smoothness: float = 0.1,
        t1_range: tuple[float, float] = (10.0, 3000.0),  # ms
        t2_range: tuple[float, float] = (5.0, 500.0),  # ms
    ) -> None:
        """__init__.

        Args:
            lambda_dc (float): Description.
            lambda_bloch (float): Description.
            lambda_sparsity (float): Description.
            lambda_smoothness (float): Description.
            t1_range (tuple): Description.
            t2_range (tuple): Description.
        """
        super().__init__()
        self.lambda_dc = lambda_dc
        self.lambda_bloch = lambda_bloch
        self.lambda_sparsity = lambda_sparsity
        self.lambda_smoothness = lambda_smoothness
        self.t1_range = t1_range
        self.t2_range = t2_range

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        k_pred: torch.Tensor,
        k_target: torch.Tensor,
        biophysics_map: torch.Tensor | None = None,  # (B, N, 3) [M0, T1, T2]
        opacity: torch.Tensor | None = None,
        psi: torch.Tensor | None = None,  # Vector Potential for smoothness
    ) -> dict[str, torch.Tensor]:
        """forward.

        Args:
            k_pred (torch.Tensor): Description.
            k_target (torch.Tensor): Description.
            biophysics_map (Optional[torch.Tensor]): Description.
            opacity (Optional[torch.Tensor]): Description.
            psi (Optional[torch.Tensor]): Description.
        Returns:
            dict[str, torch.Tensor]: Description.

        forward method for BiophysicalFlowLoss.

        Executes PyTorch tensor operations.

        Args:
            k_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            k_target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            biophysics_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            opacity (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            psi (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        losses: dict[str, torch.Tensor] = {}

        # 1. Data Consistency (DC)
        # Handle complex
        if torch.is_complex(k_pred) or torch.is_complex(k_target):
            # Mean modulus, the same per-element reduction as the real branch, so
            # lambda_dc does not scale with the number of k-space samples.
            dc_loss = (k_pred - k_target).abs().mean()
        else:
            dc_loss = self.l1(k_pred, k_target)

        losses["dc_loss"] = dc_loss
        total_loss = self.lambda_dc * dc_loss

        # 2. Bloch Constraints (Physical Plausibility)
        if biophysics_map is not None:
            # Assuming biophysics_map already processed to physical units OR raw logits?
            # If raw, we should map them here or assume map is standardized.
            # Let's assume passed map is in physical units (ms).
            t1 = biophysics_map[..., 1]
            t2 = biophysics_map[..., 2]

            # Penalize values outside range
            t1_loss = torch.mean(
                torch.relu(self.t1_range[0] - t1) + torch.relu(t1 - self.t1_range[1])
            )
            t2_loss = torch.mean(
                torch.relu(self.t2_range[0] - t2) + torch.relu(t2 - self.t2_range[1])
            )

            # Penalize T2 > T1 (physically impossible)
            physics_violation = torch.mean(torch.relu(t2 - t1))

            bloch_loss = t1_loss + t2_loss + physics_violation
            losses["bloch_loss"] = bloch_loss
            # FIXED 2026-05-04: removed .item() — was breaking autograd graph and
            # forcing GPU sync inside forward(). total_loss stays a tensor so
            # downstream .backward() can flow through every term.
            total_loss = total_loss + self.lambda_bloch * bloch_loss

        # 3. Sparsity
        if opacity is not None:
            sparsity_loss = torch.mean(torch.abs(opacity))
            losses["sparsity_loss"] = sparsity_loss
            total_loss = total_loss + self.lambda_sparsity * sparsity_loss

        # 4. Smoothness (Flow)
        if psi is not None:
            # Psi is (B, N, 3) or (B, N, 4) if spatiotemporal.
            # We want local smoothness. If points are unordered, this is hard on a cloud.
            # If Psi comes from SIREN on a grid (implicit), we can compute grad squared.
            # But here Psi is likely sampled at points?
            # If SIREN, we should regularize the network weights or sample gradients.
            # For prototype: L2 penalty on Psi magnitude to prevent explosion.
            smoothness = torch.mean(psi**2)
            losses["smoothness_loss"] = smoothness
            total_loss = total_loss + self.lambda_smoothness * smoothness

        losses["total_loss"] = total_loss
        return losses
