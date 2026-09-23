"""
Physics-Informed Loss Functions.

This module provides a collection of loss functions that incorporate MRI physics
constraints, including k-space consistency, Bloch equation residuals, and
spectral domain losses.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.core.coil_map_names import kwargs_accepted_by
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c, sense_forward
from spectramr.models.losses.registry import register_loss


@register_loss(name="non_cartesian_graph", aliases=["NonCartesianGraphLoss"])
class NonCartesianGraphLoss(nn.Module):
    r"""Cross-Domain Spectral Loss for Non-Cartesian MRI.

    Bridges the gap between the Point Cloud domain (Graph output) and the
    Image domain (Ground Truth) using a differentiable Non-Uniform FFT (NUFFT).
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{NonCartesian} = \| \mathcal{F}^{-1}(\hat{k}) - y \|_1"""

    def __init__(self, nufft_operator):
        """
        Args:
            nufft_operator: Instance of NUFFTOperator (physics forward model).
        """
        super().__init__()
        self.nufft = nufft_operator
        self.l1 = nn.L1Loss()

    def forward(
        self,
        pred_graph_nodes: torch.Tensor,
        target_image: torch.Tensor,
        trajectory: torch.Tensor,
        dcf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_graph_nodes: [B, N, 2] Predicted k-space samples (Real, Imag)
            target_image: [B, 1, H, W] Ground Truth Image (Real/Magnitude)
            trajectory: [B, N, 2] Trajectory coordinates (Kx, Ky)
            dcf: [B, N] Optional density compensation weights for forward operator optimization

        forward method for NonCartesianGraphLoss.

        Executes PyTorch tensor operations.

        Args:
            pred_graph_nodes (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            trajectory (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            dcf (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Ensure graph nodes are complex for NUFFT
        if pred_graph_nodes.shape[-1] == 2 and not pred_graph_nodes.is_complex():
            pred_kspace = torch.complex(pred_graph_nodes[..., 0], pred_graph_nodes[..., 1])
        else:
            pred_kspace = pred_graph_nodes

        # Handle batch dimension in NUFFT if needed.
        if pred_kspace.ndim == 2:  # [B, N]
            pred_kspace = pred_kspace.unsqueeze(1)  # [B, 1, N]

        # 1. Physics Forward Pass (Graph -> Image)
        pred_image = self.nufft.adjoint(pred_kspace)

        # 2. Image Domain Loss (Perceptual/Structural)
        image_loss = self.l1(pred_image, target_image)

        return image_loss


@register_loss(name="graph_consistency", aliases=["GraphConsistencyLoss"])
class GraphConsistencyLoss(nn.Module):
    r"""Physics-consistent Graph-to-Graph Loss.

    Computes weighted error directly on the k-space manifold nodes.
    Avoids differentiable NUFFT during training, resolving instability and computational bottlenecks.

    Formula:
        Loss = mean( |Predicted - Target|^2 * DCF )

    Args:
        density_weighting (bool): Whether to weight errors by Density Compensation Function (DCF).
                                  Essential for non-uniform sampling to balance frequency importance.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{GraphConsistency} = \mathbb{E}[ dcf \odot \| \hat{k}_{graph} - k_{graph} \|_2^2 ]"""

    def __init__(self, density_weighting: bool = True):
        """__init__.

        Args:
            density_weighting (bool): Description.
        """
        super().__init__()
        self.density_weighting = density_weighting
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        pred_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        dcf: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_nodes: [B, N, 2] (Real, Imag) or [B, C, H, W] grid.
            target_nodes: [B, N, 2] (Real, Imag) or [B, C, H, W] grid.
            dcf: [B, N] Density weights

        forward method for GraphConsistencyLoss.

        Executes PyTorch tensor operations.

        Args:
            pred_nodes (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_nodes (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            dcf (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Convert grid inputs [B, C, H, W] to flat graph [B, H*W, C] -> [B, N, 2]
        if pred_nodes.dim() == 4:
            B, C, H, W = pred_nodes.shape
            pred_nodes = pred_nodes.view(B, C, -1).permute(0, 2, 1)
        if target_nodes.dim() == 4:
            B, C, H, W = target_nodes.shape
            target_nodes = target_nodes.view(B, C, -1).permute(0, 2, 1)

        # Compute Squared Error per node: [B, N, 2]
        error = (pred_nodes - target_nodes) ** 2
        # Sum Real+Imag error: [B, N]
        error_mag = error.sum(dim=-1)

        if self.density_weighting:
            if dcf is None:
                # Warn once or raise? raising ensures correctness.
                pass  # Assume dcf might be missing if implicit uniform, but for spiral it's critical.
            else:
                # Apply DCF Weighting
                # If dcf is 2D grid, flatten
                if dcf.dim() == 4:
                    dcf = dcf.view(dcf.shape[0], dcf.shape[1], -1).permute(0, 2, 1).squeeze(-1)
                elif dcf.dim() == 3:
                    dcf = dcf.view(dcf.shape[0], -1)

                # Check shapes for broadcasting
                if dcf.ndim == 2 and error_mag.ndim == 2:
                    weighted_error = error_mag * dcf
                else:
                    weighted_error = error_mag  # Fallback if shapes mismatch

                return weighted_error.mean()

        return error_mag.mean()


@register_loss(name="bloch_residual", aliases=["BlochResidualLoss"])
class BlochResidualLoss(nn.Module):
    """
    Physics-Informed Bloch Residual Loss.
    Enforces that the generated magnetization time-series M(t) obeys the Bloch Differential Equation.

    Equation: dM/dt = M x (gamma * B) - R * (M - M0)
    """

    def __init__(self, dt: float = 1e-3, gamma: float = 267.52e6):
        """
        Args:
           dt: Time step between frames in seconds (e.g. 1ms)
           gamma: Gyromagnetic ratio (rad/s/T)
        """
        super().__init__()
        self.dt = dt
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(
        self,
        M_pred: torch.Tensor,
        T1_map: torch.Tensor,
        T2_map: torch.Tensor,
        PD_map: torch.Tensor = None,
        B0_map: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            M_pred: [Batch, Time, 3, H, W] Predicted Magnetization vectors (Mx, My, Mz)
            T1_map: [Batch, 1, H, W] T1 relaxation time (seconds)
            T2_map: [Batch, 1, H, W] T2 relaxation time (seconds)
            PD_map: [Batch, 1, H, W] Proton Density (M0 magnitude)
            B0_map: [Batch, 1, H, W] Off-resonance map (Tesla) - Optional

        Returns:
            Scalar Loss (Residual of Bloch Equation)

        forward method for BlochResidualLoss.

        Executes PyTorch tensor operations.

        Args:
            M_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            T1_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            T2_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            PD_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            B0_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 0. Shapes and Time
        B, T, C, H, W = M_pred.shape
        assert C == 3, "Output must have 3 channels (Mx, My, Mz)"

        # The maps are documented `[B, 1, H, W]` and the code below relies on it:
        # their dim-1 of size 1 broadcasts against the time axis directly. A
        # `[B, H, W]` map would need an extra axis, and mixing the two is how
        # this loss broke -- see the note on the removed `unsqueeze(1)` calls.
        #
        # Guarded rather than assumed because the failure mode is SILENT on the
        # relaxation path: an extra axis makes `-Mx * R1` broadcast
        # `[B, T-1, H, W]` against `[B, 1, 1, H, W]` into `[B, B, T-1, H, W]`,
        # inventing a batch axis instead of raising.
        for _name, _m in (
            ("T1_map", T1_map),
            ("T2_map", T2_map),
            ("PD_map", PD_map),
            ("B0_map", B0_map),
        ):
            if _m is None:
                continue
            if _m.dim() != 4 or _m.shape[0] != B or _m.shape[1] != 1:
                raise ValueError(
                    f"{_name} must be [B, 1, H, W] with B={B}; got "
                    f"{tuple(_m.shape)}. A parameter map with a different rank "
                    f"broadcasts silently against the time axis and yields a "
                    f"finite loss computed over the wrong tensor."
                )

        # 1. Compute Numerical Time Derivative (dM/dt)
        # Using forward difference: (M[t+1] - M[t]) / dt
        dM_dt_pred = (M_pred[:, 1:] - M_pred[:, :-1]) / self.dt

        # We evaluate the Bloch equation at time t (from 0 to T-1)
        M_current = M_pred[:, :-1]

        # 2. Compute Physical Derivative (RHS of Bloch)
        Mx, My, Mz = M_current[:, :, 0], M_current[:, :, 1], M_current[:, :, 2]

        if B0_map is not None:
            # Expand B0 to the time dim. NO `unsqueeze(1)`: B0_map is already
            # `[B, 1, H, W]`, so unsqueezing made it 5-D and `expand` then got 4
            # sizes for 5 dimensions -- a hard error.
            Bz = B0_map.expand(-1, T - 1, -1, -1)
            omega_z = self.gamma * Bz

            # Cross product M x B => (My*Bz - Mz*By, Mz*Bx - Mx*Bz, Mx*By - My*Bx)
            # Assuming Bx = By = 0
            prec_x = My * omega_z
            prec_y = -Mx * omega_z
            prec_z = torch.zeros_like(Mz)
        else:
            prec_x = torch.zeros_like(Mx)
            prec_y = torch.zeros_like(My)
            prec_z = torch.zeros_like(Mz)

        # 2b. Relaxation term: -R * (M - M0)
        M0_z = PD_map.expand(-1, T - 1, -1, -1) if PD_map is not None else torch.ones_like(Mz)

        # Avoid div by zero. NO `unsqueeze(1)` here either: the maps are
        # `[B, 1, H, W]` and that size-1 axis already broadcasts against
        # `[B, T-1, H, W]`. With the unsqueeze they were `[B, 1, 1, H, W]`, and
        # `-Mx * R1` broadcast to `[B, B, T-1, H, W]` WITHOUT error -- the loss
        # only failed later, at the MSE against `dM_dt_pred`.
        R1 = 1.0 / (T1_map + 1e-6)
        R2 = 1.0 / (T2_map + 1e-6)

        relax_x = -Mx * R2
        relax_y = -My * R2
        relax_z = -(Mz - M0_z) * R1

        target_dMx = prec_x + relax_x
        target_dMy = prec_y + relax_y
        target_dMz = prec_z + relax_z

        target_dM_dt = torch.stack([target_dMx, target_dMy, target_dMz], dim=2)

        # 3. Residual Loss
        return self.mse(dM_dt_pred, target_dM_dt)


@register_loss(name="physics_constraint", aliases=["PhysicsConstraintLoss"])
class PhysicsConstraintLoss(nn.Module):
    r"""Physics-informed constraint loss for MRI reconstruction.

    Enforces physical consistency constraints such as energy conservation,
    k-space consistency, and smoothness.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{PhysicsConstraint} = \lambda_{smooth} \mathcal{L}_{smooth} + \lambda_{energy} \mathcal{L}_{energy} + \lambda_{kspace} \mathcal{L}_{kspace}"""

    def __init__(
        self,
        smoothness_weight: float = 0.1,
        energy_weight: float = 0.1,
        kspace_weight: float = 0.1,
    ):
        """__init__.

        Args:
            smoothness_weight (float): Description.
            energy_weight (float): Description.
            kspace_weight (float): Description.
        """
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.energy_weight = energy_weight
        self.kspace_weight = kspace_weight

    def compute_smoothness_loss(self, u: torch.Tensor) -> torch.Tensor:
        """Compute Total Variation (smoothness) loss.

        Args:
            u: Input tensor [B, C, H, W]

        Returns:
            Smoothness loss
        """
        # Calculate gradients
        dy = torch.abs(u[..., 1:, :] - u[..., :-1, :])
        dx = torch.abs(u[..., :, 1:] - u[..., :, :-1])

        loss = torch.mean(dx) + torch.mean(dy)
        return loss

    def compute_energy_conservation_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute energy conservation loss (Parseval's theorem).

        Args:
            pred: Predicted image (real or complex)
            target: Target image (real or complex)

        Returns:
            Energy conservation loss
        """
        # Use |x|^2 for complex tensors so we never call mse on ComplexFloat
        pred_energy = torch.mean(pred.abs() ** 2 if torch.is_complex(pred) else pred**2)
        target_energy = torch.mean(target.abs() ** 2 if torch.is_complex(target) else target**2)
        # Both are real scalars now — safe to subtract and square
        return (pred_energy - target_energy) ** 2

    def compute_kspace_consistency_loss(
        self,
        pred: torch.Tensor,
        kspace_target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute k-space consistency loss.

        Args:
            pred: Predicted image [B, C, H, W]
            kspace_target: Acquired k-space data
            mask: Sampling mask

        Returns:
            Consistency loss
        """
        # Improved FFT using canonical centered transform
        pred_kspace = fft2c(pred)

        # Use magnitude-based MSE to avoid 'mse_cpu not implemented for ComplexFloat'
        if mask is not None:
            diff = (pred_kspace * mask - kspace_target * mask).abs()
        else:
            diff = (pred_kspace - kspace_target).abs()

        return torch.mean(diff**2)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        kspace_target: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined physics loss.

        Returns:
            Tuple of (total_loss, metrics_dict)

        forward method for PhysicsConstraintLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            kspace_target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, dict]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        metrics = {}
        # Accumulate as a graph-connected tensor — never via ``.item()`` (which
        # would return a gradient-free Python float and silently zero the loss).
        total_loss = torch.zeros((), device=pred.device)

        # Smoothness
        if self.smoothness_weight > 0:
            smooth_loss = self.compute_smoothness_loss(pred)
            total_loss = total_loss + self.smoothness_weight * smooth_loss
            metrics["loss/physics_smoothness"] = smooth_loss.item()

        # Energy
        if self.energy_weight > 0:
            energy_loss = self.compute_energy_conservation_loss(pred, target)
            total_loss = total_loss + self.energy_weight * energy_loss
            metrics["loss/physics_energy"] = energy_loss.item()

        # K-space
        if self.kspace_weight > 0 and kspace_target is not None:
            kspace_loss = self.compute_kspace_consistency_loss(pred, kspace_target, mask)
            total_loss = total_loss + self.kspace_weight * kspace_loss
            metrics["loss/physics_kspace"] = kspace_loss.item()

        return total_loss, metrics


@register_loss(name="snr_preserving", aliases=["SNRPreservingLoss"])
class SNRPreservingLoss(nn.Module):
    """Loss function that penalizes SNR Degradation.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{SNR} = \\lambda \\max(0, \text{SNR}_{target} - \text{SNR}_{current})"""

    def __init__(self, target_snr: float = 20.0, weight: float = 1.0):
        """__init__.

        Args:
            target_snr (float): Description.
            weight (float): Description.
        """
        super().__init__()
        self.target_snr = target_snr
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Signal power
        """forward.

        Args:
            pred (torch.Tensor): Description.
            target (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SNRPreservingLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        signal = torch.mean(target**2)

        # Noise power (mse)
        noise = torch.mean((pred - target) ** 2)

        # Estimated SNR
        current_snr = 10 * torch.log10(signal / (noise + 1e-8))

        # Penalty if SNR is below target
        penalty = F.relu(self.target_snr - current_snr)

        return self.weight * penalty


@register_loss(name="physics_informed", aliases=["PhysicsInformedLoss"])
class PhysicsInformedLoss(nn.Module):
    r"""Physics-informed loss functions for MRI reconstruction.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{PhysicsInformed} = \lambda_{L2} \| \hat{y} - y \|_2^2 + \lambda_{dc} \| (\hat{k} - k) \odot M \|_2^2 + \lambda_{reg} \mathcal{L}_{TV}"""

    def __init__(
        self,
        lambda_l2: float = 1.0,
        lambda_dc: float = 1.0,
        lambda_reg: float = 0.01,
    ):
        """__init__.

        Args:
            lambda_l2 (float): Description.
            lambda_dc (float): Description.
            lambda_reg (float): Description.
        """
        super().__init__()
        self.lambda_l2 = lambda_l2
        self.lambda_dc = lambda_dc
        self.lambda_reg = lambda_reg

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_kspace: torch.Tensor | None = None,
        target_kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute physics-informed losses.

        Args:
            pred: Predicted image
            target: Target image
            pred_kspace: Predicted k-space (optional)
            target_kspace: Target k-space (optional)
            mask: Sampling mask (optional)

        Returns:
            Dictionary of loss components

        forward method for PhysicsInformedLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            pred_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        losses = {}

        # L2 loss in image space
        losses["l2_image"] = self.lambda_l2 * F.mse_loss(pred, target)

        # Data consistency loss in k-space
        if pred_kspace is not None and target_kspace is not None and mask is not None:
            masked_pred = pred_kspace * mask.unsqueeze(1)
            masked_target = target_kspace * mask.unsqueeze(1)
            losses["dc_kspace"] = self.lambda_dc * F.mse_loss(
                masked_pred,
                masked_target,
            )

        # Regularization (Enhanced TV + Wavelet)
        if len(pred.shape) >= 4:  # Has spatial dimensions
            # Anisotropic TV (better edge preservation than isotropic).
            # Differences must be taken along the two SPATIAL axes (H=-2, W=-1)
            # of a [B, C, H, W] tensor — not along the channel axis.
            diff_x = pred[..., :, :, 1:] - pred[..., :, :, :-1]  # along W
            diff_y = pred[..., :, 1:, :] - pred[..., :, :-1, :]  # along H

            # L1 norm (anisotropic TV) - preserves edges better than L2
            tv_loss = torch.mean(torch.abs(diff_x)) + torch.mean(torch.abs(diff_y))

            # Optionally add epsilon for smoothness (Huber-like)
            epsilon = 1e-3
            tv_loss_smooth = torch.mean(torch.sqrt(diff_x**2 + epsilon)) + torch.mean(
                torch.sqrt(diff_y**2 + epsilon)
            )

            # Wavelet-based regularization (approximate with Laplacian pyramid)
            # High-frequency component suppression for smoother reconstruction
            # Laplacian: approximated as image - blur(image)
            kernel_size = 5
            padding = kernel_size // 2

            # Simple box blur as approximation
            # Depthwise box blur: a grouped conv with ``groups=C`` needs a
            # weight of shape ``(C, 1, k, k)`` (out_channels divisible by groups),
            # not ``(1, C, k, k)`` — the latter raises for C > 1.
            blur_kernel = torch.ones(
                pred.shape[1], 1, kernel_size, kernel_size, device=pred.device
            ) / (kernel_size**2)
            blurred = F.conv2d(pred, blur_kernel, padding=padding, groups=pred.shape[1])

            high_freq = pred - blurred
            wavelet_loss = torch.mean(torch.abs(high_freq))

            # Combine regularizations
            losses["regularization"] = self.lambda_reg * (
                0.6 * tv_loss_smooth  # Smooth anisotropic TV
                + 0.3 * tv_loss  # Hard anisotropic TV
                + 0.1 * wavelet_loss  # High-frequency suppression
            )

        # Total loss
        total_loss = torch.tensor(0.0, device=pred.device)
        for loss_value in losses.values():
            if isinstance(loss_value, torch.Tensor):
                total_loss = total_loss + loss_value
            else:
                total_loss = total_loss + torch.tensor(loss_value, device=pred.device)
        losses["total"] = total_loss

        return losses


@register_loss(name="parallel_imaging_kspace", aliases=["ParallelImagingKSpaceLoss"])
class ParallelImagingKSpaceLoss(nn.Module):
    """
    K-Space Consistency Loss with Parallel Imaging Support.

    Computes loss between predicted and measured K-space data, handling
    multi-coil sensitivities and forward transforms.
    """

    def __init__(self, fft_transformer: Any | None = None):
        """
        Args:
            fft_transformer: Object dealing with FFT operations (must implement fft2c).
                           If None, standard torch.fft is used.
        """
        super().__init__()
        self.fft_transformer = fft_transformer
        self.l1 = nn.L1Loss()

    def forward(
        self,
        hr_fakes: torch.Tensor,
        measured_kspace: torch.Tensor,
        coil_sensitivities: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hr_fakes: [B, C, H, W] Predicted image (Magnitude or Complex)
            measured_kspace: [B, Coils, H, W] or [B, H, W] Measured K-space
            coil_sensitivities: [B, Coils, H, W] Optional sensitivity maps
            mask: Optional sampling mask

        forward method for ParallelImagingKSpaceLoss.

        Executes PyTorch tensor operations.

        Args:
            hr_fakes (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measured_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            coil_sensitivities (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Forward Operator: Image -> K-Space
        # 1. Forward Operator: Image -> K-Space
        if coil_sensitivities is not None:
            # Parallel Imaging Forward Model using Physics-consistent SENSE Operator
            # Ensure complex input handled by sense_forward internally
            # but we assume hr_fakes matches input convention
            pred_kspace = sense_forward(hr_fakes, smaps=coil_sensitivities)

        elif self.fft_transformer is not None:
            # Use provided transformer
            pred_kspace = self.fft_transformer.fft2c(hr_fakes)
        else:
            # Standard FFT (single coil assumption if no sens provided)
            # Ensure complex input for FFT
            hr_complex = self._ensure_complex(hr_fakes)
            # Use centered FFT to match standard MRI convention
            # (Matches fft2c behavior)
            from spectramr.infrastructure.physics.fft_ops import fft2c

            pred_kspace = fft2c(hr_complex)

        # 2. Alignment & Loss
        target_kspace = measured_kspace
        if pred_kspace.shape != target_kspace.shape:
            # Handle complex view mismatch (last dim 2 vs complex64/128)
            pred_kspace, target_kspace = self._align_complex_shapes(pred_kspace, target_kspace)

        if mask is not None:
            return self.l1(pred_kspace * mask, target_kspace * mask)
        return self.l1(pred_kspace, target_kspace)

    def _ensure_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is in complex format (not B,C,H,W with last dim 2)."""
        if torch.is_complex(x):
            return x

        # If (B, C, H, W) with even C, view as complex
        if x.dim() == 4 and x.shape[1] % 2 == 0:
            B, C, H, W = x.shape
            x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            return torch.view_as_complex(x_reshaped).permute(0, 3, 1, 2)

        # If (B, H, W, 2), view as complex
        if x.shape[-1] == 2:
            return torch.view_as_complex(x)

        # If real (B, 1, H, W) or (B, H, W), treat as real part
        return x

    def _align_complex_shapes(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align shapes if one is complex and other is 2-channel real."""
        # Convert 2-channel real to complex for comparison
        if not pred.is_complex() and target.is_complex():
            if pred.shape[-1] == 2:
                pred = torch.view_as_complex(pred)
            elif pred.shape[1] % 2 == 0 and pred.dim() == 4:  # (B, 2C, H, W)
                B, C, H, W = pred.shape
                pred_reshaped = pred.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
                pred = torch.view_as_complex(pred_reshaped).permute(0, 3, 1, 2)

        if pred.is_complex() and not target.is_complex():
            if target.shape[-1] == 2:
                target = torch.view_as_complex(target)
            elif target.shape[1] % 2 == 0 and target.dim() == 4:
                B, C, H, W = target.shape
                target_reshaped = target.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
                target = torch.view_as_complex(target_reshaped).permute(0, 3, 1, 2)

        return pred, target


@register_loss(name="spectral_kspace", aliases=["SpectralKSpaceLoss", "kspace"])
class SpectralKSpaceLoss(nn.Module):
    r"""Log-L1 K-Space Loss ("Spectral Loss") for frequency-aware optimization.

    Penalizes errors in the frequency domain with logarithmic scaling,
    preventing the DC component (contrast) from dominating high-frequency details.

    Equation: L = Mean( | log(|F(x)| + eps) - log(|F(y)| + eps) | )
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{SpectralKSpace} = \| \log(|\mathcal{F}(\hat{y})| + \epsilon) - \log(|\mathcal{F}(y)| + \epsilon) \|_1"""

    def __init__(self, epsilon: float = 1e-8):
        """__init__.

        Args:
            epsilon (float): Description.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
           pred_img: [B, C, H, W]
           target_img: [B, C, H, W]

        forward method for SpectralKSpaceLoss.

        Executes PyTorch tensor operations.

        Args:
            pred_img (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_img (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Ensure complex input for standard physics convention if needed,
        # or just take FFT of image representation
        if pred_img.shape[-1] == 2 and not pred_img.is_complex():
            pred_complex = torch.view_as_complex(pred_img)
        elif pred_img.shape[1] % 2 == 0 and not pred_img.is_complex():
            B, C, H, W = pred_img.shape
            pred_reshaped = pred_img.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            pred_complex = torch.view_as_complex(pred_reshaped).permute(0, 3, 1, 2)
        else:
            pred_complex = pred_img  # Assume magnitude or already complex

        if target_img.shape[-1] == 2 and not target_img.is_complex():
            target_complex = torch.view_as_complex(target_img)
        elif target_img.shape[1] % 2 == 0 and not target_img.is_complex():
            B, C, H, W = target_img.shape
            target_reshaped = target_img.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            target_complex = torch.view_as_complex(target_reshaped).permute(0, 3, 1, 2)
        else:
            target_complex = target_img

        # Transform to frequency domain using canonical centered FFT
        pred_k = fft2c(pred_complex)
        target_k = fft2c(target_complex)

        # Log-Frequency Loss
        # Log of Magnitude
        log_pred = torch.log(torch.abs(pred_k) + self.epsilon)
        log_target = torch.log(torch.abs(target_k) + self.epsilon)

        loss = F.l1_loss(log_pred, log_target)
        return loss


# NOTE: Not @register_loss-decorated. The canonical "hfen" entry lives at
# ``src/models/losses/hfen_loss.py`` (kernel_size=15, sigma=1.5,
# normalize=True), which the registry resolves for any YAML using the
# ``hfen`` key. This legacy class is kept for direct-import callers and
# its complex/2-channel handling, but the public surface goes through the
# newer implementation. See TODO/audit/12_losses.md F1.
class HFENLoss(nn.Module):
    """
    High-Frequency Error Norm (HFEN) Loss - IMAGE-SPACE ONLY (legacy).

    **DOMAIN**: IMAGE-SPACE (spatial domain gradients required) ⚠️
    **Input**: [B, 1, H, W] or [B, C, H, W] image-space intensity
    **3D Support**: No (2D Laplacian of Gaussian only)

    Uses isotropic Laplacian of Gaussian (LoG) kernel to extract edges
    and penalizes differences in edge maps. Critical for trabecular structure.

    ⚠️ CRITICAL WARNING: Do NOT use with k-space inputs [B, 2, H, W]!
    K-space HFEN computes meaningless k-space gradients (not image edges).

    For k-space reconstruction, choose ONE:
    1. Disable: lambda_hfen = 0.0
    2. Convert: img = ifft2c(kspace); loss_hfen(img, target_img)
    3. Use k-space native: complex_l1, weighted_kspace_l1, data_consistency
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        """__init__.

        Args:
            kernel_size (int): Description.
            sigma (float): Description.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Construct LoG kernel
        grid = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        xx, yy = torch.meshgrid(grid, grid, indexing="ij")

        # Laplacian of Gaussian Formula:
        # LoG(x,y) = -1/(pi * sigma^4) * (1 - (x^2+y^2)/(2*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2))

        r2 = xx**2 + yy**2
        sigma2 = sigma**2

        kernel = (
            -(1 / (torch.pi * sigma2**2)) * (1 - r2 / (2 * sigma2)) * torch.exp(-r2 / (2 * sigma2))
        )

        # Normalize to zero sum to ignore DC
        kernel = kernel - kernel.mean()

        # Reshape for conv2d: [Out, In, H, W] -> [1, 1, K, K]
        # We will apply it channel-wise using groups=C
        self.register_buffer("kernel", kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W] Input image
            target: [B, C, H, W] Target image

        forward method for HFENLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Handle Complex or 2-channel
        # Typically computed on Magnitude
        if pred.is_complex():
            pred_mag = pred.abs()
        elif pred.shape[1] % 2 == 0 and not pred.is_complex():
            B, C, H, W = pred.shape
            pred_reshaped = pred.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            pred_complex = torch.view_as_complex(pred_reshaped).permute(0, 3, 1, 2)
            pred_mag = pred_complex.abs()
        elif pred.shape[-1] == 2:
            pred_mag = torch.sqrt(pred[..., 0] ** 2 + pred[..., 1] ** 2 + 1e-8).unsqueeze(1)
        else:
            pred_mag = pred

        if target.is_complex():
            target_mag = target.abs()
        elif target.shape[1] % 2 == 0 and not target.is_complex():
            B, C, H, W = target.shape
            target_reshaped = target.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            target_complex = torch.view_as_complex(target_reshaped).permute(0, 3, 1, 2)
            target_mag = target_complex.abs()
        elif target.shape[-1] == 2:
            target_mag = torch.sqrt(target[..., 0] ** 2 + target[..., 1] ** 2 + 1e-8).unsqueeze(1)
        else:
            target_mag = target

        # Ensure channel dimension matches groups
        # If input has C channels, we replicate kernel
        channels = pred_mag.shape[1]
        kernel_expanded = self.kernel.repeat(channels, 1, 1, 1)

        # Convolve
        pred_edges = F.conv2d(
            pred_mag, kernel_expanded, padding=self.kernel_size // 2, groups=channels
        )
        target_edges = F.conv2d(
            target_mag, kernel_expanded, padding=self.kernel_size // 2, groups=channels
        )

        return F.l1_loss(pred_edges, target_edges)


# =============================================================================
# ARCHITECTURAL REMEDIATION LOSSES
# =============================================================================
# Physics-informed losses for production-ready MRI reconstruction
# =============================================================================


@register_loss(name="focal_frequency", aliases=["FocalFrequencyLoss", "InvertedFocalFrequencyLoss"])
class InvertedFocalFrequencyLoss(nn.Module):
    """Inverted Focal Frequency Loss for Image Reconstruction.

    Prevents hallucination of fine structures (trabecular bone, lesions) by
    penalizing frequency-domain differences with adaptive weighting.
    The highest energy in natural images resides at the DC component.
    By shifting the FFT and applying a quadratic radial distance multiplier,
    the optimizer faces an exponentially steeper gradient at high frequencies.

    L_FFL = Σ w(u,v) * |F(pred) - F(target)|²
    where w(u,v) = 1.0 + α * r² (focal weighting)
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{InvertedFocal} = \\mathbb{E} \\left[ (1 + \alpha |k|^2) | \\hat{k} - k |^2 \right]"""

    def __init__(self, alpha: float = 2.0, **kwargs):
        """
        Args:
            alpha: Controls the steepness of the high-frequency penalty.
        """
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # If real, convert to complex automatically by fft2c (it handles it)
        # 1. Compute 2D FFT with orthonormal normalization & shift (using physics module)
        # fft2c returns centered k-space with orthonormal normalization
        """forward.

        Args:
            pred (torch.Tensor): Description.
            target (torch.Tensor): Description.
            weight_matrix (Optional[torch.Tensor]): Description.
        Returns:
            torch.Tensor: Description.

        forward method for InvertedFocalFrequencyLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            weight_matrix (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        pred_shifted = fft2c(pred)
        target_shifted = fft2c(target)

        B, C, H, W = pred.shape

        # 3. Construct normalized radial distance mask [-1, 1]
        # We cache this to avoid recomputing every forward pass in a real implementation
        Y, X = torch.meshgrid(
            torch.linspace(-1, 1, H, device=pred.device),
            torch.linspace(-1, 1, W, device=pred.device),
            indexing="ij",
        )
        radial_dist = torch.sqrt(X**2 + Y**2)

        # 4. Apply Inverted Weighting: 1 + alpha * r^2
        # Center (DC) has weight 1. Periphery (High Freq) has weight heavily scaled
        weight = 1.0 + self.alpha * (radial_dist**2)
        weight = weight.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

        if weight_matrix is not None:
            weight = weight * weight_matrix.unsqueeze(0).unsqueeze(0)

        # 5. Calculate weighted L2 norm in the frequency domain (complex difference magnitude)
        diff = torch.abs(pred_shifted - target_shifted)
        loss = torch.mean((diff**2) * weight)

        return loss


class ManifoldConstrainedGradient(nn.Module):
    """Manifold Constrained Gradient (MCG) for Physics-Consistent Inference.

    Projects diffusion model predictions back onto the valid measurement subspace.
    Ensures that generated images are strictly consistent with acquired k-space.

    x_corrected = x̂ + A^H(y - A(x̂))

    where:
    - x̂: model prediction
    - y: measured k-space
    - A: forward operator (FFT + mask)
    - A^H: adjoint operator

    Reference: Chung et al., "Score-based diffusion models for accelerated MRI"
               (Medical Image Analysis, 2022)
    """

    def __init__(self, step_size: float = 1.0):
        """
        Args:
            step_size: Data consistency step size (lambda)
        """
        super().__init__()
        self.step_size = step_size

    def forward(
        self,
        x_pred: torch.Tensor,
        y_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply MCG projection.

        Args:
            x_pred: Model prediction [B, C, H, W] (image domain)
            y_measured: Measured k-space [B, C, H, W] (complex)
            mask: Undersampling mask [B, 1, H, W]

        Returns:
            Physics-corrected prediction [B, C, H, W]

        forward method for ManifoldConstrainedGradient.

        Executes PyTorch tensor operations.

        Args:
            x_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            y_measured (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Ensure complex for FFT
        if not x_pred.is_complex():
            if x_pred.shape[1] % 2 == 0:
                B, C, H, W = x_pred.shape
                x_reshaped = x_pred.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
                x_pred = torch.view_as_complex(x_reshaped).permute(0, 3, 1, 2)
            else:
                x_pred = x_pred.to(torch.complex64)

        # Forward: image -> k-space using canonical centered FFT
        x_pred_kspace = fft2c(x_pred)

        # Data consistency residual (only at measured locations)
        residual = mask * (y_measured - x_pred_kspace)

        # Adjoint: k-space -> image using canonical centered IFFT
        correction = ifft2c(residual)

        # Apply correction
        x_corrected = x_pred + self.step_size * correction

        return x_corrected


@register_loss(name="divergence_free", aliases=["DivergenceFreeLoss"])
class DivergenceFreeLoss(nn.Module):
    """Divergence-Free Penalty for Motion/Deformation Fields.

    Enforces continuity equation (∇·v = 0) for incompressible tissue.

    Reference: VoxelMorph (Balakrishnan et al., IEEE TMI 2019)
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{DivergenceFree} = \\mathbb{E}[ (\nabla \\cdot \vec{v})^2 ]"""

    def __init__(self):
        """__init__."""
        super().__init__()

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """Compute divergence penalty for 2D flow field [B, 2, H, W].

        forward method for DivergenceFreeLoss.

        Executes PyTorch tensor operations.

        Args:
            flow (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        vx, vy = flow[:, 0:1], flow[:, 1:2]

        # Central differences
        dvx_dx = (vx[:, :, :, 2:] - vx[:, :, :, :-2]) / 2.0
        dvy_dy = (vy[:, :, 2:, :] - vy[:, :, :-2, :]) / 2.0

        # Align shapes
        min_h = min(dvx_dx.shape[2], dvy_dy.shape[2])
        min_w = min(dvx_dx.shape[3], dvy_dy.shape[3])

        divergence = dvx_dx[:, :, :min_h, :min_w] + dvy_dy[:, :, :min_h, :min_w]

        return torch.mean(divergence**2)


@register_loss(name="jacobian_determinant", aliases=["JacobianDeterminantLoss"])
class JacobianDeterminantLoss(nn.Module):
    r"""Jacobian Determinant Loss for Deformation Fields.

    Penalizes negative Jacobian determinants (folding/topology violations).

    Reference: VoxelMorph (Balakrishnan et al., IEEE TMI 2019)
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{JacobianDet} = \mathbb{E}[ \max(0, -|J_v|) ]"""

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian penalty for 2D flow [B, 2, H, W].

        forward method for JacobianDeterminantLoss.

        Executes PyTorch tensor operations.

        Args:
            flow (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        dx, dy = flow[:, 0:1], flow[:, 1:2]

        # Forward differences
        ddx_dx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
        ddx_dy = dx[:, :, 1:, :] - dx[:, :, :-1, :]
        ddy_dx = dy[:, :, :, 1:] - dy[:, :, :, :-1]
        ddy_dy = dy[:, :, 1:, :] - dy[:, :, :-1, :]

        min_h = min(ddx_dx.shape[2], ddx_dy.shape[2])
        min_w = min(ddx_dx.shape[3], ddx_dy.shape[3])

        det_J = (1 + ddx_dx[:, :, :min_h, :min_w]) * (1 + ddy_dy[:, :, :min_h, :min_w]) - ddx_dy[
            :, :, :min_h, :min_w
        ] * ddy_dx[:, :, :min_h, :min_w]

        # Penalize negative determinants (folding)
        return torch.mean(F.relu(-det_J))


@register_loss(name="data_consistency", aliases=["DataConsistencyLoss"])
class DataConsistencyLoss(nn.Module):
    """Data Consistency Loss for Unrolled Optimization.

    **DOMAIN**: K-SPACE NATIVE
    **Input**: Predicted image [B, C, H, W], measured k-space [B, 2, H, W], mask [B, 1, H, W]
    **3D Support**: Yes

    Enforces k-space consistency to prevent hallucinations by projecting predictions
    back to measured k-space frequency components. Critical for physics-informed reconstruction.

    Reference: Hammernik et al., "Learning a Variational Network" (MRM 2018)
    """

    def __init__(self, lambda_dc: float = 1.0):
        """__init__.

        Args:
            lambda_dc (float): Description.
        """
        super().__init__()
        self.lambda_dc = lambda_dc

    def forward(
        self,
        pred_image: torch.Tensor,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DC loss.

        forward method for DataConsistencyLoss.

        Executes PyTorch tensor operations.

        Args:
            pred_image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measured_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if not pred_image.is_complex():
            pred_image = pred_image.to(torch.complex64)

        pred_kspace = fft2c(pred_image)
        residual = (pred_kspace - measured_kspace) * mask

        return self.lambda_dc * torch.mean(torch.abs(residual) ** 2)


@register_loss(name="qmap_physics", aliases=["SelfSupervisedPhysicsLoss", "PhysicsForwardLoss"])
class SelfSupervisedPhysicsLoss(nn.Module):
    """Self-Supervised Physics-Driven Loss for Quantitative Mapping.

    Bridges the gap between predicted tissue parameter maps (T1, T2, PD)
    and acquired k-space data via the SPGR forward model.

    Physics Pipeline:
        1. Bloch Simulation: (T1, T2, PD) -> Image Magnitude
        2. Sensitivity Encoding: Image * Sensitivity Maps -> Coil Images
        3. FFT: Coil Images -> K-space
        4. Data Consistency: Compare simulated vs. measured k-space

    This loss enforces that predicted tissue parameters, when forward-modeled
    through MRI physics, reproduce the acquired k-space data.

    Args:
        bloch_layer: Instance of DifferentiableBlochLayer.
        use_sensitivity_maps (bool): Whether to apply sensitivity encoding.
        loss_weight (float): Overall loss scaling factor.
        kspace_weight (float): Weight for k-space data consistency term.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{SelfSup} = \\lambda \\| \\mathcal{F}(\text{Bloch}(T_1, T_2, PD)) \\odot M - k_{target} \\odot M \\|_1"""

    def __init__(
        self,
        bloch_layer: nn.Module,
        use_sensitivity_maps: bool = True,
        loss_weight: float = 1.0,
        kspace_weight: float = 1.0,
        device: torch.device = None,
    ):
        """
        Args:
            bloch_layer: DifferentiableBlochLayer instance for SPGR simulation.
            use_sensitivity_maps: If True, apply SENSE model for multicoil simulation.
            loss_weight: Overall multiplicative weight for this loss.
            kspace_weight: Weight applied to k-space error term.
            device: Device for tensor operations.
        """
        super().__init__()
        self.bloch_layer = bloch_layer
        self.use_sensitivity_maps = use_sensitivity_maps
        self.loss_weight = loss_weight
        self.kspace_weight = kspace_weight
        self.device = device or torch.device("cpu")

        self.l1_loss = nn.L1Loss(reduction="mean")

    def forward_operator(
        self,
        sim_image: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply SENSE forward model: Image -> Coil Images -> K-space.

        Args:
            sim_image (Tensor): Simulated image [B, 1, H, W] (real-valued magnitude).
            sensitivity_maps (Tensor): Coil sensitivity maps [B, C, H, W] (complex).
                                      If None, assume single-coil (coil multiplicity = 1).

        Returns:
            kspace_pred (Tensor): Predicted k-space [B, C, H, W] (complex).
        """
        # Convert magnitude image to complex (phase = 0 for now)
        if not sim_image.is_complex():
            sim_image_complex = torch.complex(sim_image, torch.zeros_like(sim_image))
        else:
            sim_image_complex = sim_image

        if self.use_sensitivity_maps and sensitivity_maps is not None:
            # Apply SENSE encoding: multiply by coil sensitivity maps
            # [B, 1, H, W] * [B, C, H, W] -> [B, C, H, W]
            coil_images = sim_image_complex * sensitivity_maps
        else:
            # Single-coil or raw image: treat as coil 0
            coil_images = sim_image_complex

        # Transform to k-space via FFT.
        # ``fft2c`` is centered + ortho-normalized by construction; it takes no
        # ``centered`` kwarg (passing one raises TypeError at runtime).
        kspace_pred = fft2c(coil_images)

        return kspace_pred

    def forward(
        self,
        pred_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
        acq_params: dict | None = None,
    ) -> torch.Tensor:
        """
        Compute self-supervised physics loss.

        Args:
            pred_params (Tuple): Predicted tissue maps (t1, t2, pd).
                                Each [B, 1, H, W].
            target_kspace (Tensor): Measured k-space [B, C, H, W] (complex).
            mask (Tensor): Undersampling mask [B, 1, H, W] or [B, C, H, W].
                          Binary: 1 = sampled, 0 = unsampled.
            sensitivity_maps (Tensor): Coil sensitivity maps [B, C, H, W] (complex).
                                      If None, single-coil simulation.
            acq_params (dict): Sequence parameters:
                - 'TR': Repetition time (ms)
                - 'TE': Echo time (ms)
                - 'flip_angle': Flip angle (radians)

        Returns:
            loss (Tensor): Scalar loss value.

        Physics Forward Pass:
            1. SPGR Bloch simulation: (T1, T2, PD) -> Image magnitude
            2. SENSE forward model: Image * Sensitivity -> Coil images -> K-space
            3. Data consistency: Compare masked k-space regions

        forward method for SelfSupervisedPhysicsLoss.

        Executes PyTorch tensor operations.

        Args:
            pred_params (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            sensitivity_maps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            acq_params (dict | None): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if acq_params is None:
            raise ValueError("acq_params dict required: must contain TR, TE, flip_angle")

        t1_map, t2_map, pd_map = pred_params

        # 1. Bloch Simulation: Predict steady-state signal magnitude
        sim_image_mag = self.bloch_layer(
            t1_map,
            t2_map,
            pd_map,
            tr=acq_params["TR"],
            te=acq_params["TE"],
            flip_angle_rad=acq_params["flip_angle"],
        )

        # 2. Apply forward operator (SENSE + FFT)
        sim_kspace = self.forward_operator(sim_image_mag, sensitivity_maps)

        # 3. Apply sampling mask
        sim_kspace_masked = sim_kspace * mask
        target_kspace_masked = target_kspace * mask

        # 4. Compute k-space data consistency loss
        # Convert complex tensors to real representation for L1Loss
        # torch.view_as_real: [B, C, H, W] complex -> [B, C, H, W, 2] real
        sim_kspace_real = torch.view_as_real(sim_kspace_masked)
        target_kspace_real = torch.view_as_real(target_kspace_masked)

        loss_kspace = self.l1_loss(sim_kspace_real, target_kspace_real)

        # Apply loss weight
        total_loss = self.loss_weight * self.kspace_weight * loss_kspace

        return total_loss


@register_loss(name="qmap_physics_advanced", aliases=["AdvancedPhysicsLoss"])
class AdvancedSelfSupervisedPhysicsLoss(SelfSupervisedPhysicsLoss):
    r"""Extended Physics Loss with B0/B1 Field Corrections.

    Builds on SelfSupervisedPhysicsLoss to include:
    - Off-resonance (B0) phase accumulation
    - Transmit field (B1) correction with smoothness regularization
    - Support for 5-parameter mapping (T1, T2, PD, B0, B1)
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{AdvSelfSup} = \mathcal{L}_{SelfSup} + \lambda_{B1} \mathcal{L}_{smooth}(B_1)"""

    def _compute_b1_smoothness_loss(
        self, b1_map: torch.Tensor, weight: float = 10.0
    ) -> torch.Tensor:
        """
        Compute smoothness regularization for B1 field.

        B1 maps should be spatially smooth (low frequency). This loss penalizes
        sharp gradients to enforce physical plausibility.

        Args:
            b1_map (Tensor): B1 field map [B, 1, H, W].
            weight (float): Regularization strength (10.0 recommended).

        Returns:
            loss (Tensor): Scalar smoothness penalty.
        """
        # Compute spatial gradients
        dy = torch.abs(b1_map[:, :, 1:, :] - b1_map[:, :, :-1, :])
        dx = torch.abs(b1_map[:, :, :, 1:] - b1_map[:, :, :, :-1])

        # Mean absolute gradient
        smoothness_penalty = torch.mean(dx) + torch.mean(dy)

        return weight * smoothness_penalty

    def forward(
        self,
        pred_params,
        target_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
        acq_params: dict | None = None,
        b1_smoothness_weight: float = 10.0,
    ) -> torch.Tensor:
        """
        Compute self-supervised physics loss with field corrections.

        Args:
            pred_params: Predicted maps. Can be:
                - 3-tuple (t1, t2, pd): Uses parent class behavior
                - 5-tuple (t1, t2, pd, b0, b1): Full field correction
            target_kspace (Tensor): Measured k-space [B, C, H, W] (complex).
            mask (Tensor): Undersampling mask [B, 1, H, W] or [B, C, H, W].
            sensitivity_maps (Tensor): Coil sensitivity maps (optional).
            acq_params (dict): Sequence parameters (TR, TE, flip_angle).
            b1_smoothness_weight (float): Weight for B1 smoothness regularization.

        Returns:
            loss (Tensor): Scalar loss value.
        """
        if acq_params is None:
            raise ValueError("acq_params dict required: must contain TR, TE, flip_angle")

        # Handle both 3-param and 5-param cases
        if len(pred_params) == 3:
            # Original 3-param case: use parent implementation
            t1_map, t2_map, pd_map = pred_params
            b0_map = None
            b1_map = None
        elif len(pred_params) == 5:
            # New 5-param case with B0, B1
            t1_map, t2_map, pd_map, b0_map, b1_map = pred_params
        else:
            raise ValueError(f"Expected 3 or 5 parameter maps, got {len(pred_params)}")

        # 1. Bloch Simulation: Pass optional B0, B1
        sim_image = self.bloch_layer(
            t1_map,
            t2_map,
            pd_map,
            tr=acq_params["TR"],
            te=acq_params["TE"],
            flip_angle_rad=acq_params["flip_angle"],
            b0_map=b0_map,
            b1_map=b1_map,
        )

        # 2. Apply forward operator (SENSE + FFT)
        sim_kspace = self.forward_operator(sim_image, sensitivity_maps)

        # 3. Apply sampling mask
        sim_kspace_masked = sim_kspace * mask
        target_kspace_masked = target_kspace * mask

        # 4. Compute k-space data consistency loss
        sim_kspace_real = torch.view_as_real(sim_kspace_masked)
        target_kspace_real = torch.view_as_real(target_kspace_masked)
        loss_kspace = self.l1_loss(sim_kspace_real, target_kspace_real)

        # 5. Compute B1 smoothness regularization (if provided)
        loss_b1_smooth = 0.0
        if b1_map is not None:
            loss_b1_smooth = self._compute_b1_smoothness_loss(b1_map, weight=b1_smoothness_weight)

        # 6. Total loss: data consistency + B1 smoothness
        total_loss = self.loss_weight * (self.kspace_weight * loss_kspace + loss_b1_smooth)

        return total_loss


@register_loss(
    name="energy_conservation",
    aliases=["parseval_energy", "kspace_energy_conservation"],
)
class EnergyConservationLoss(nn.Module):
    """Parseval's Theorem Energy Conservation Loss.

    Enforces that the total energy in k-space matches the energy in image space,
    as dictated by Parseval's theorem:

    $$\\sum_k |K(k)|^2 = \\sum_x |x|^2 / (N_x N_y)$$

    This loss prevents the model from "stealing" energy from the DC component
    and redistributing it, which would be invisible to pixel-space metrics but
    would create unrealistic k-space patterns.

    **DOMAIN**: K-space
    **Input**: [B, C, H, W] (complex-valued k-space or real [B, C, 2, H, W])
    **3D Support**: Yes

    Args:
        energy_method (str): How to compute energy. Options:
            - "magnitude_squared" (default): E = sum(|K|^2)
            - "complex_norm": E = sum(real^2 + imag^2) for real/imag channels
        reduction (str): Reduction method ('mean' or 'mse')
        margin (float): Allow up to this fraction of energy deviation (e.g., 0.05 = ±5%)
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{EnergyConservation} = \\left( \frac{\\sum |\\hat{k}|^2}{\\sum |k|^2} - 1 \right)^2"""

    def __init__(
        self,
        energy_method: str = "magnitude_squared",
        reduction: str = "mse",
        margin: float = 0.05,
    ):
        """__init__.

        Args:
            energy_method (str): Description.
            reduction (str): Description.
            margin (float): Description.
        """
        super().__init__()
        if energy_method not in ("magnitude_squared", "complex_norm"):
            msg = f"Unknown energy_method: {energy_method}"
            raise ValueError(msg)
        if reduction not in ("mean", "mse"):
            msg = f"Unknown reduction: {reduction}"
            raise ValueError(msg)
        self.energy_method = energy_method
        self.reduction = reduction
        self.margin = margin

    def forward(
        self,
        pred_kspace: torch.Tensor,
        target_kspace: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy conservation loss.

        Args:
            pred_kspace: Predicted k-space [B, C, H, W] or [B, C, 2, H, W] (real/imag)
            target_kspace: Target k-space [B, C, H, W] or [B, C, 2, H, W]

        Returns:
            Scalar loss value penalizing energy deviation

        forward method for EnergyConservationLoss.

        Executes PyTorch tensor operations.

        Args:
            pred_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""

        # Helper to compute energy from k-space representation
        def compute_energy(kspace):
            """compute_energy.

            Args:
                kspace (Any): Description.
            Returns:
                Any: Description.
            """
            if kspace.is_complex():
                # Complex tensor: E = sum(|K|^2)
                return torch.sum(torch.abs(kspace) ** 2, dim=(-2, -1))

            # Real/imaginary channels: [B, C, 2, H, W]
            if kspace.shape[-3] == 2:
                real = kspace[..., 0, :, :]  # [B, C, H, W]
                imag = kspace[..., 1, :, :]  # [B, C, H, W]
                return torch.sum(real**2 + imag**2, dim=(-2, -1))

            # Fallback: treat as magnitude squared
            return torch.sum(kspace**2, dim=(-2, -1))

        # Compute energies
        pred_energy = compute_energy(pred_kspace)  # [B, C]
        target_energy = compute_energy(target_kspace)  # [B, C]

        # Normalize by spatial dimensions to account for FFT scaling
        h, w = pred_kspace.shape[-2:]
        norm_factor = h * w  # Parseval accounts for this

        pred_energy_normalized = pred_energy / norm_factor
        target_energy_normalized = target_energy / norm_factor

        # Compute energy ratio to detect "energy theft"
        # Ratio should be close to 1.0
        energy_ratio = pred_energy_normalized / (target_energy_normalized + 1e-8)

        # Penalize deviations from unity
        if self.energy_method == "magnitude_squared":
            if self.reduction == "mse":
                loss = torch.mean((energy_ratio - 1.0) ** 2)
            else:  # mean
                loss = torch.mean(torch.abs(energy_ratio - 1.0))

        else:  # complex_norm
            # More lenient: allow ±margin deviation
            normalized_ratio = (energy_ratio - 1.0) / (self.margin + 1e-8)
            if self.reduction == "mse":
                loss = torch.mean(torch.clamp(torch.abs(normalized_ratio) - 1.0, min=0.0) ** 2)
            else:  # mean
                loss = torch.mean(torch.clamp(torch.abs(normalized_ratio) - 1.0, min=0.0))

        return loss


class DifferentiableFourierBridge(nn.Module):
    """
    Bridges pure k-space predictions into the image domain for spatial loss
    calculations, maintaining full autograd tracking for complex tensors.
    """

    def __init__(self, spatial_loss_fn=None, return_complex: bool = False):
        """__init__.

        Args:
            spatial_loss_fn (Any): Description.
            return_complex (bool): Description.
        """
        super().__init__()
        self.spatial_loss_fn = spatial_loss_fn
        self.return_complex = return_complex

    def _kspace_to_image(self, kspace: torch.Tensor) -> torch.Tensor:
        # 1. Ensure input is complex [B, C, H, W] where C=1 (if coils are combined)
        """_kspace_to_image.

        Args:
            kspace (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if not torch.is_complex(kspace):
            if kspace.shape[-1] == 2:
                # [..., 2] format: convert to complex
                kspace = torch.view_as_complex(kspace)
            else:
                # [B, 2*C, H, W] stacked: reshape to complex
                B, _, H, W = kspace.shape
                C = kspace.shape[1] // 2
                kspace = torch.view_as_complex(
                    kspace.view(B, C, 2, H, W).permute(0, 1, 3, 4, 2).contiguous()
                )

        # 2. Inverse FFT with orthonormal scaling (crucial for gradient magnitude)
        # Use the verified ifft2c from physics ops which handles shift order correctly
        img_complex = ifft2c(kspace)

        # 3. Return magnitude for spatial loss (or keep complex for phase losses)
        if self.return_complex:
            return img_complex
        return torch.abs(img_complex)

    def forward(
        self,
        k_pred: torch.Tensor,
        k_target: torch.Tensor,
        spatial_mask: torch.Tensor = None,
        **kwargs,
    ):
        """forward.

        Args:
            k_pred (torch.Tensor): Description.
            k_target (torch.Tensor): Description.
            spatial_mask (torch.Tensor): Description.
        Returns:
            Any: Description.

        forward method for DifferentiableFourierBridge.

        Executes PyTorch tensor operations.

        Args:
            k_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            k_target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            spatial_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.spatial_loss_fn is None:
            raise ValueError("spatial_loss_fn must be provided to compute loss natively")

        # Project both prediction and target to image space
        img_pred = self._kspace_to_image(k_pred)
        img_target = self._kspace_to_image(k_target)

        # Apply the spatial operation (e.g., Background Suppression)
        if spatial_mask is not None:
            img_pred = img_pred * spatial_mask
            img_target = img_target * spatial_mask

        # The builder's wrapper forwards everything, so this is the hop that narrows
        # kwargs to the inner loss, and the one that must re-file the coil maps.
        accepted = kwargs_accepted_by(self.spatial_loss_fn, kwargs)
        return self.spatial_loss_fn(img_pred, img_target, **accepted)


@register_loss(name="background_suppression", aliases=["BackgroundSuppressionLoss"])
class BackgroundSuppressionLoss(nn.Module):
    r"""Background (Noise) Suppression Loss.

    Suppresses artifacts in background/noise regions by forcing predictions to zero
    where the ground truth is near the noise floor. This addresses the "gray fog"
    problem in MRI reconstruction where background noise appears as reduced contrast.

    Physics:
        MRI images have anatomically defined backgrounds (brain void, air) where
        signal should be zero. Yet standard L1 reconstruction predicts non-zero
        values in these regions due to noise fitting. This loss identifies the
        noise floor in the target and penalizes predictions there.

    Algorithm (use_fourier_bridge=True, k-space inputs):
        1. Transform k-space to image domain via DifferentiableFourierBridge (iFFT, norm="ortho")
        2. Compute noise floor as 10th percentile of GT image magnitude
        3. Identify background as pixels < threshold_ratio * noise_floor
        4. Penalize |prediction| in background regions (gradients flow back through iFFT)

    Algorithm (use_fourier_bridge=False, image-domain inputs):
        1. Compute noise floor on raw GT magnitude directly
        2. Identify background mask
        3. Penalize |prediction| in background regions

    Args:
        threshold_ratio (float): Background threshold as multiple of noise floor.
            Default: 1.5 (targets ~15th percentile of GT distribution).
        use_fourier_bridge (bool): Whether to project k-space inputs to image
            domain via DifferentiableFourierBridge before computing the loss.
            - True (default): use when pred/gt are k-space tensors (Experiment 11).
            - False: use when pred/gt are already in image domain.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{Background} = \mathbb{E}[ | \hat{y} | \odot M_{bg} ]"""

    def __init__(self, threshold_ratio: float = 1.5, use_fourier_bridge: bool = True):
        """__init__.

        Args:
            threshold_ratio (float): Description.
            use_fourier_bridge (bool): Description.
        """
        super().__init__()
        self.ratio = threshold_ratio
        self.use_fourier_bridge = use_fourier_bridge

        if use_fourier_bridge:
            # Define the spatial loss logic (mean absolute error of masked image)
            def l1_mae(pred, target):
                """l1_mae.

                Args:
                    pred (Any): Description.
                    target (Any): Description.
                Returns:
                    Any: Description.
                """
                return pred.abs().mean()

            # Wrap the spatial logic in the differentiable bridge to preserve Parseval's theorem
            self.bridge = DifferentiableFourierBridge(spatial_loss_fn=l1_mae)
        else:
            self.bridge = None

    def forward(self, k_pred: torch.Tensor, k_gt: torch.Tensor) -> torch.Tensor:
        """
        Compute background suppression loss.

        Args:
            k_pred: Predicted tensor — k-space [B, C, H, W...] if use_fourier_bridge=True,
                    image-domain magnitude [B, C, H, W] if use_fourier_bridge=False.
            k_gt: Ground truth tensor (same domain as k_pred).

        Returns:
            Scalar loss (penalizes nonzero pred in background regions)

        forward method for BackgroundSuppressionLoss.

        Executes PyTorch tensor operations.

        Args:
            k_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            k_gt (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.use_fourier_bridge:
            # ── K-SPACE PATH ──────────────────────────────────────────────────
            # Derive background mask from GT without gradients (mask is non-differentiable)
            with torch.no_grad():
                img_gt = self.bridge._kspace_to_image(k_gt)
                B = img_gt.shape[0]
                flat_gt = img_gt.reshape(B, -1)
                noise_level = torch.quantile(flat_gt, 0.1, dim=1, keepdim=True)
                noise_level = noise_level.view(B, 1, 1, 1)
                bg_mask = (img_gt < (self.ratio * noise_level)).float()

            # Bridge projects k_pred into image space (differentiable) then masks + MAE
            return self.bridge(k_pred=k_pred, k_target=k_gt, spatial_mask=bg_mask)

        else:
            # ── IMAGE-DOMAIN PATH ─────────────────────────────────────────────
            # pred/gt are already image-domain magnitudes
            img_pred = k_pred.abs() if torch.is_complex(k_pred) else k_pred
            img_gt = k_gt.abs() if torch.is_complex(k_gt) else k_gt

            B = img_gt.shape[0]
            flat_gt = img_gt.reshape(B, -1)
            noise_level = torch.quantile(flat_gt, 0.1, dim=1, keepdim=True)
            noise_level = noise_level.view(B, 1, 1, 1)
            bg_mask = (img_gt < (self.ratio * noise_level)).float()

            return (img_pred.abs() * bg_mask).mean()


@register_loss(
    name="complex_spatial_gradient",
    aliases=["ComplexGradientLoss", "complex_gradient"],
    domain="kspace",
    compatible_with=["kspace", "complex"],
)
class ComplexGradientLoss(nn.Module):
    """Penalizes first-order spatial derivatives of the complex image.

    Enforces both magnitude sharpness (edges) and local phase coherence by
    directly penalizing complex spatial differences dx and dy. This prevents
    hallucinated sharpness with scrambled phase — a critical failure mode in
    k-space MRI reconstruction.

    Given complex images X_pred = A + iB and X_gt = C + iD (after iFFT):
        loss = mean(|dx_pred - dx_gt|) + mean(|dy_pred - dy_gt|)

    where |·| is the complex magnitude: sqrt(Re² + Im²), so both amplitude
    and phase gradients are penalized simultaneously.

    Args:
        use_fourier_bridge (bool): If True (default), inputs are k-space tensors
            and will be projected into complex image space via DifferentiableFourierBridge
            (iFFT, norm="ortho", return_complex=True) before taking finite differences.
            If False, inputs must already be complex image-domain tensors.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{ComplexGradient} = \\| \nabla \\hat{k} - \nabla k \\|_p"""

    def __init__(self, use_fourier_bridge: bool = True):
        """__init__.

        Args:
            use_fourier_bridge (bool): Description.
        """
        super().__init__()
        self.use_fourier_bridge = use_fourier_bridge

        def complex_gradient_penalty(
            img_pred: torch.Tensor, img_target: torch.Tensor
        ) -> torch.Tensor:
            """Compute L1 norm of complex spatial gradient differences.

            Args:
                img_pred: Complex predicted image [B, C, H, W] (complex dtype)
                img_target: Complex target image [B, C, H, W] (complex dtype)

            Returns:
                Scalar loss combining magnitude and phase gradient penalties.
            """
            # Finite differences along width (x) axis
            dx_pred = torch.diff(img_pred, dim=-1)
            dx_target = torch.diff(img_target, dim=-1)

            # Finite differences along height (y) axis
            dy_pred = torch.diff(img_pred, dim=-2)
            dy_target = torch.diff(img_target, dim=-2)

            # torch.abs() on a complex tensor computes: sqrt(Re² + Im²)
            # This simultaneously penalizes magnitude AND phase gradient errors
            loss_dx = torch.abs(dx_pred - dx_target).mean()
            loss_dy = torch.abs(dy_pred - dy_target).mean()

            return loss_dx + loss_dy

        if use_fourier_bridge:
            # Bridge maps k-space → complex image domain (return_complex=True)
            # Gradients flow back through the iFFT to k-space predictions
            self.bridge = DifferentiableFourierBridge(
                spatial_loss_fn=complex_gradient_penalty,
                return_complex=True,
            )
        else:
            self.bridge = None
            self._spatial_loss_fn = complex_gradient_penalty

    def forward(self, k_pred: torch.Tensor, k_target: torch.Tensor) -> torch.Tensor:
        """Compute complex spatial gradient loss.

        Args:
            k_pred: Predicted k-space tensor [B, C, H, W] if use_fourier_bridge=True,
                    or complex image tensor if use_fourier_bridge=False.
            k_target: Ground truth tensor (same domain as k_pred).

        Returns:
            Scalar loss value.

        forward method for ComplexGradientLoss.

        Executes PyTorch tensor operations.

        Args:
            k_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            k_target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.use_fourier_bridge:
            return self.bridge(k_pred, k_target)
        # Inputs are already complex image-domain tensors
        return self._spatial_loss_fn(k_pred, k_target)


@register_loss(name="sense_adjoint_phase", aliases=["SENSEAdjointPhaseLoss"])
class SENSEAdjointPhaseLoss(nn.Module):
    r"""SENSE-Adjoint Phase Loss.

    Validates the predicted K-Space using standard iFFT combined securely
    with the $S_c^*$ sensitivity estimated maps to enforce phase consistency.

    L_SENSE = || conj(S_c) * iFFT(K_pred) - conj(S_c) * iFFT(K_true) ||_1
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{SENSEAdjointPhase} = \| \sum_c S_c^* \mathcal{F}^{-1}(\hat{k}_c) - \sum_c S_c^* \mathcal{F}^{-1}(k_c) \|_1"""

    def __init__(self):
        """__init__."""
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        pred_kspace: torch.Tensor,
        target_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
    ) -> torch.Tensor:
        # Convert to complex if needed
        """forward.

        Args:
            pred_kspace (torch.Tensor): Description.
            target_kspace (torch.Tensor): Description.
            sensitivity_maps (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SENSEAdjointPhaseLoss.

        Executes PyTorch tensor operations.

        Args:
            pred_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            sensitivity_maps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if pred_kspace.shape[-1] == 2 and not pred_kspace.is_complex():
            pred_kspace = torch.view_as_complex(pred_kspace)
        if target_kspace.shape[-1] == 2 and not target_kspace.is_complex():
            target_kspace = torch.view_as_complex(target_kspace)
        if sensitivity_maps.shape[-1] == 2 and not sensitivity_maps.is_complex():
            sensitivity_maps = torch.view_as_complex(sensitivity_maps)

        # iFFT
        pred_img = ifft2c(pred_kspace)
        target_img = ifft2c(target_kspace)

        # Combine with conjugate of sensitivity maps
        # sensitivity_maps: [B, C, H, W]
        S_conj = torch.conj(sensitivity_maps)

        # Multiply and sum over coils (SENSE Adjoint operation)
        # Resulting image: [B, 1, H, W]
        pred_combined = torch.sum(pred_img * S_conj, dim=1, keepdim=True)
        target_combined = torch.sum(target_img * S_conj, dim=1, keepdim=True)

        # Phase extraction or direct complex L1
        # L1 on the combined complex images intrinsically penalizes phase
        return self.l1(pred_combined, target_combined)


@register_loss(name="sobolev_frequency", aliases=["SobolevFrequencyLoss"])
class SobolevFrequencyLoss(nn.Module):
    """Sobolev Frequency Loss.

    L_Sob = || (1 + sqrt(k_x^2 + k_y^2)) \\odot (K_pred - K_true) ||_1
    Directly weights the edges in frequency domain.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{SobolevFrequency} = \\mathbb{E}_{k} \\left[ (1 + |k|) | \\hat{k} - k | \right]"""

    def __init__(self):
        """__init__."""
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred_kspace: torch.Tensor, target_kspace: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            pred_kspace (torch.Tensor): Description.
            target_kspace (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SobolevFrequencyLoss.

        Executes PyTorch tensor operations.

        Args:
            pred_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if pred_kspace.shape[-1] == 2 and not pred_kspace.is_complex():
            pred_kspace = torch.view_as_complex(pred_kspace)
        if target_kspace.shape[-1] == 2 and not target_kspace.is_complex():
            target_kspace = torch.view_as_complex(target_kspace)

        # Create Sobolev weight mask
        H, W = pred_kspace.shape[-2:]
        Y, X = torch.meshgrid(
            torch.linspace(-1, 1, H, device=pred_kspace.device),
            torch.linspace(-1, 1, W, device=pred_kspace.device),
            indexing="ij",
        )

        # Radial distance from center
        radial_dist = torch.sqrt(X**2 + Y**2)
        weight = 1.0 + radial_dist

        # Reshape for broadcasting [1, 1, H, W]
        while weight.dim() < pred_kspace.dim():
            weight = weight.unsqueeze(0)

        diff = pred_kspace - target_kspace
        weighted_diff = diff * weight

        # L1 norm of complex difference
        return self.l1(weighted_diff, torch.zeros_like(weighted_diff))


@register_loss(name="unit_norm_coil", aliases=["UnitNormCoilLoss"])
class UnitNormCoilLoss(nn.Module):
    """Enforces the Uecker NLINV unit-norm coil constraint:
    \\sum_{c} |S_c|^2 = 1.0 everywhere in the spatial domain.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{UnitNormCoil} = \\mathbb{E} \\left[ (1 - \\sum_c |S_c|^2)^2 \right]"""

    def __init__(self):
        super().__init__()

    def forward(self, S_real: torch.Tensor, S_imag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S_real: (N, C) or (B, C, H, W) Real sensitivities
            S_imag: (N, C) or (B, C, H, W) Imag sensitivities
        """
        S_mag_sq = S_real**2 + S_imag**2

        # Determine coil dimension
        if S_mag_sq.dim() == 2:  # (N, C)
            sum_mag_sq = S_mag_sq.sum(dim=-1)
        elif S_mag_sq.dim() == 4:  # (B, C, H, W)
            sum_mag_sq = S_mag_sq.sum(dim=1)
        else:
            raise ValueError(f"Unexpected shape for sensitivities: {S_mag_sq.shape}")

        return torch.mean((1.0 - sum_mag_sq) ** 2)


@register_loss(name="magnitude_tv", aliases=["MagnitudeTVLoss"])
class MagnitudeTVLoss(nn.Module):
    r"""
    Total Variation on the Magnitude of the Complex Fields.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, S_real: torch.Tensor, S_imag: torch.Tensor, shape: tuple[int, int] = None
    ) -> torch.Tensor:
        """
        Args:
            S_real: (N, C) or (B, C, H, W) Real sensitivities
            S_imag: (N, C) or (B, C, H, W) Imag sensitivities
            shape: (H, W) required if input is (N, C) points
        """
        S_mag = torch.sqrt(S_real**2 + S_imag**2 + 1e-8)

        if S_mag.dim() == 2:
            import math

            N, C = S_mag.shape
            if shape is not None:
                H, W = shape
            else:
                H = W = int(math.sqrt(N))  # Assume square
            # S_mag: (N, C) -> (C, N) -> (C, H, W)
            S_mag_2d = S_mag.T.view(C, H, W).unsqueeze(0)  # (1, C, H, W)
        elif S_mag.dim() == 4:
            S_mag_2d = S_mag
        else:
            raise ValueError(f"Unexpected shape for sensitivities: {S_mag.shape}")

        diff_h = torch.abs(S_mag_2d[..., 1:, :] - S_mag_2d[..., :-1, :]).mean()
        diff_w = torch.abs(S_mag_2d[..., :, 1:] - S_mag_2d[..., :, :-1]).mean()

        return diff_h + diff_w
