"""Disentangled MRI Representation Learning
========================================

Implementation of Experiment 1.2 from the Research Roadmap.
Uses Dual-Stream Encoder and AdaIN Decoder for disentangled synthesis.

Key Concepts:
- ContentEncoder: Uses InstanceNorm to strip style, preserving spatial structure (anatomy)
- StyleEncoder: Uses GlobalAvgPool to destroy spatial info, keeping global statistics (contrast)
- AdaIN: Re-injects style into content, forcing "Structure A + Texture B" synthesis

This enables training on UNPAIRED data (T1 from Patient A, T2 from Patient B).
"""

import logging
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.constants import (
    BIAS_FIELD_MAX_VALUE,
    BIAS_FIELD_MIN_VALUE,
    BIAS_FIELD_POLYNOMIAL_COEFFICIENTS,
    BIAS_FIELD_SMOOTHING_KERNEL_SIZE,
    BIAS_FIELD_SMOOTHING_SIGMA,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

#: Width of the [TR, TE, TI, B0] vector the disentangled_vae strategy conditions on.
PHYSICS_VECTOR_DIM = 4


class BiasFieldGenerator(nn.Module):
    """Generates smooth, low-frequency bias field for MRI receive coil sensitivity.

    Outputs multiplicative field B(r) ∈ [0.3, 3.0] modeling:
    - Receive coil sensitivity variations
    - Transmit field (B1+) inhomogeneity

    Physics: MRI signal ∝ B⁻(r) · S_tissue · B⁺(r)

    This module generates B(r) to be multiplied with tissue synthesis.
    """

    def __init__(
        self,
        in_dim: int = 256,
        order: Literal["polynomial_3", "lowrank_unet"] = "polynomial_3",
        output_resolution: tuple = (256, 256),
    ):
        """Initialize bias field generator.

        Args:
            in_dim: Input feature dimension (from content encoder)
            order: 'polynomial_3' (10 coefficients) or 'lowrank_unet' (learned UNet)
            output_resolution: (H, W) for upsampling if needed
        """
        super().__init__()
        self.order = order
        self.output_resolution = output_resolution

        if order == "polynomial_3":
            # Polynomial basis: fit spatial field with 3rd-order polynomial
            # BIAS_FIELD_POLYNOMIAL_COEFFICIENTS: 1, x, y, x², xy, y², x³, x²y, xy², y³
            self.poly_head = nn.Sequential(
                nn.Conv2d(in_dim, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),  # Global context pooling
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, BIAS_FIELD_POLYNOMIAL_COEFFICIENTS),
            )
            self.register_buffer(
                "smoothing_kernel",
                self._create_gaussian_kernel(
                    sigma=BIAS_FIELD_SMOOTHING_SIGMA,
                    size=BIAS_FIELD_SMOOTHING_KERNEL_SIZE,
                ),
            )

        elif order == "lowrank_unet":
            # Lightweight 2-layer UNet for bias field
            self.unet = nn.Sequential(
                # Feature extraction
                nn.Conv2d(in_dim, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                # Down
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                # Up
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                # Output
                nn.Conv2d(32, 1, 3, padding=1),
                nn.Sigmoid(),  # Constrain to [0, 1]
            )
        else:
            raise ValueError(
                f"Unknown bias_field order: {order!r}; expected 'polynomial_3' or 'lowrank_unet'"
            )

    def _create_gaussian_kernel(
        self,
        sigma: float = BIAS_FIELD_SMOOTHING_SIGMA,
        size: int = BIAS_FIELD_SMOOTHING_KERNEL_SIZE,
    ) -> torch.Tensor:
        """Create 2D Gaussian kernel for field smoothing."""
        x = torch.arange(size).float() - size // 2
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate bias field from content features.

        Args:
            features: [B, C, H_feat, W_feat] content features from encoder

        Returns:
            bias_field: [B, 1, H_out, W_out] multiplicative field ∈ [0.3, 3.0]

        forward method for BiasFieldGenerator.

        Executes PyTorch tensor operations.

        Args:
            features (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, H_feat, W_feat = features.shape[0], features.shape[2], features.shape[3]
        H_out, W_out = self.output_resolution
        device = features.device

        if self.order == "polynomial_3":
            # Get polynomial coefficients from content features
            coeffs = self.poly_head(features)  # [B, 10]

            # Create coordinate grids (normalized to [-1, 1])
            y_coords = torch.linspace(-1, 1, H_out, device=device)
            x_coords = torch.linspace(-1, 1, W_out, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")  # [H_out, W_out]

            # Evaluate 3rd-order polynomial basis
            basis = torch.stack(
                [
                    torch.ones_like(xx),  # 1
                    xx,
                    yy,  # x, y
                    xx**2,
                    xx * yy,
                    yy**2,  # x², xy, y²
                    xx**3,
                    (xx**2) * yy,
                    xx * (yy**2),
                    yy**3,  # x³, x²y, xy², y³
                ],
                dim=0,
            )  # [10, H_out, W_out]

            # Weighted sum: bias_field = Σ coeff_i * basis_i
            bias_field = torch.einsum("bi,ihw->bhw", coeffs, basis)  # [B, H_out, W_out]
            bias_field = bias_field.unsqueeze(1)  # [B, 1, H_out, W_out]

            # Apply Gaussian smoothing to ensure low-frequency
            padding = self.smoothing_kernel.shape[-1] // 2
            bias_field = F.conv2d(bias_field, self.smoothing_kernel.to(device), padding=padding)

            # Scale to realistic range [BIAS_FIELD_MIN_VALUE, BIAS_FIELD_MAX_VALUE]
            # (receive field variations from ~30% to ~300%)
            bias_field = torch.tanh(bias_field)  # Squash to [-1, 1]
            bias_field = 1.0 + 0.5 * bias_field  # Intermediate scale
            # Map to [BIAS_FIELD_MIN_VALUE, BIAS_FIELD_MAX_VALUE]
            bias_field = (BIAS_FIELD_MIN_VALUE + BIAS_FIELD_MAX_VALUE) / 2.0 + (
                (BIAS_FIELD_MAX_VALUE - BIAS_FIELD_MIN_VALUE) / 2.0
            ) * (2.0 * bias_field - 2.0)

        elif self.order == "lowrank_unet":
            # Upsample features to output resolution if needed
            if (H_feat, W_feat) != (H_out, W_out):
                features_up = F.interpolate(
                    features, size=(H_out, W_out), mode="bilinear", align_corners=False
                )
            else:
                features_up = features

            # Generate field via UNet
            bias_field_01 = self.unet(features_up)  # [B, 1, H_out, W_out] ∈ [0, 1]

            # Scale to [BIAS_FIELD_MIN_VALUE, BIAS_FIELD_MAX_VALUE] (wider range for learned flexibility)
            bias_field = (
                BIAS_FIELD_MIN_VALUE + (BIAS_FIELD_MAX_VALUE - BIAS_FIELD_MIN_VALUE) * bias_field_01
            )

        return bias_field


class AdaIN(nn.Module):
    """Adaptive Instance Normalization.

    Mathematically: output = (1 + gamma) * InstanceNorm(x) + beta
    Where gamma, beta are derived from style vector.
    """

    def __init__(self, style_dim: int, num_features: int):
        """__init__.

        Args:
            style_dim (int): Description.
            num_features (int): Description.
        """
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=1e-3)
        self.fc = nn.Linear(style_dim, num_features * 2)
        # [FIX] Initialize AdaIN to identity transformation (gamma=1, beta=0)
        # This prevents signal saturation at Tanh output
        # gamma = 1, beta = 0 means: output = 1*x + 0 = x (identity)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)  # Small weights
        nn.init.zeros_(self.fc.bias)  # Zero all bias first
        self.fc.bias.data[:num_features] = 1.0  # gamma = 1 (identity scale)
        self.fc.bias.data[num_features:] = 0.0  # beta = 0 (no shift)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] (Content)
        # s: [B, Style_Dim] (Style)
        """forward.

        Args:
            x (torch.Tensor): Description.
            s (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for AdaIN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            s (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        # [FIX] Clamp gamma/beta via tanh to prevent exponential magnification.
        # Without this, 4 stacked ResBlocks with gamma~10 multiply gradients by
        # 11^4 ≈ 14 641, causing the observed total_norm > 6e14.
        # tanh keeps gamma in (-3, 3) → scale factor (1+gamma) stays in (-2, 4).
        gamma = torch.tanh(gamma) * 3.0
        beta = torch.tanh(beta) * 3.0
        return (1 + gamma) * self.norm(x) + beta


class ResBlock(nn.Module):
    """Residual Block with optional AdaIN injection."""

    def __init__(self, dim: int, style_dim: int = None):
        """__init__.

        Args:
            dim (int): Description.
            style_dim (int): Description.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, padding_mode="reflect")
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, padding_mode="reflect")
        # [FIX] Increase eps for InstanceNorm stability
        self.norm = nn.InstanceNorm2d(dim, eps=1e-3) if style_dim is None else None
        self.adain = AdaIN(style_dim, dim) if style_dim is not None else None
        self.act = nn.LeakyReLU(
            0.2, inplace=False
        )  # [FIX] inplace=False to prevent gradient corruption

    def forward(self, x: torch.Tensor, s: torch.Tensor = None) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            s (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            s (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.conv1(x)
        h = self.norm(h) if self.norm else self.adain(h, s)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm(h) if self.norm else self.adain(h, s)
        return x + h


class ContentEncoder(nn.Module):
    """Content (Anatomy) Encoder.

    Uses InstanceNorm to remove mean/variance (style statistics),
    leaving only structural information.
    """

    def __init__(self, in_channels: int = 1, dim: int = 64, n_downsample: int = 2):
        """__init__.

        Args:
            in_channels (int): Description.
            dim (int): Description.
            n_downsample (int): Description.
        """
        super().__init__()
        # [FIX] Use larger epsilon in InstanceNorm to prevent gradient explosion
        # Default eps=1e-5 causes instability when input variance is near-zero
        layers = [
            nn.Conv2d(in_channels, dim, 7, 1, 3, padding_mode="reflect"),
            nn.InstanceNorm2d(dim, eps=0.1),  # [FIX] Large eps for near-constant inputs
            nn.LeakyReLU(0.2, inplace=False),  # [FIX] inplace=False for gradient stability
        ]

        # Downsampling preserves Spatial Structure (Anatomy)
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, 2, 1),
                nn.InstanceNorm2d(dim * 2, eps=0.1),  # [FIX] Large eps
                nn.LeakyReLU(0.2, inplace=False),  # [FIX] inplace=False
            ]
            dim *= 2

        self.model = nn.Sequential(*layers)

        # [FIX] Add LayerNorm to bottleneck to constrain latent distribution
        # We permute to [B, H, W, C] before determining norm, so we normalize over C (dim)
        # This allows spatial resolution to vary (good for multi-scale) rather than hardcoding 64x64
        self.output_norm = nn.LayerNorm(dim)
        self.output_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FIX] Handle 5D input for 2.5D slab mode
        # [B, C, D, H, W] -> [B, C*D, H, W]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ContentEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        original_shape = None
        if x.ndim == 5:
            original_shape = x.shape
            b, c, d, h, w = x.shape
            if d == 1:
                x = x.squeeze(2)
            else:
                x = x.reshape(b, c * d, h, w)

        # [FIX] Handle malformed shapes where spatial dims got shuffled into channels
        # If 4D but channels dimension looks wrong (too large), try to correct it
        if x.ndim == 4 and x.shape[1] > 256:
            # Likely [B, H_or_D, H, W] instead of [B, C, H, W]
            # Try to infer correct shape
            raise RuntimeError(
                f"ContentEncoder received unexpected input shape {x.shape}. "
                f"Expected [B, C, H, W] where C<=256, got C={x.shape[1]}. "
                f"This usually indicates a data loading or batching issue."
            )

        out = self.model(x)

        # [FIX] Apply LayerNorm to standardize latent distribution (prevent collapse)
        # Handle shape [B, C, H, W] -> [B, H, W, C] for LayerNorm -> [B, C, H, W]
        out = out.permute(0, 2, 3, 1)  # Put C last
        out = self.output_norm(out)
        out = out.permute(0, 3, 1, 2)  # Put C back

        # [PERF] Clamp activations to prevent gradient explosion
        # Range [-6, 6] allows ~6σ spread while preventing extreme values
        # Observed issue: c_a range up to 5.7 without clamping
        return torch.clamp(out, min=-6.0, max=6.0)


class StyleEncoder(nn.Module):
    """Style (Contrast/Modality) Encoder.

    Uses GlobalAveragePooling to destroy spatial information,
    leaving only global statistics (contrast/noise level).

    [VAE MODE] When use_vae=True, outputs (mu, logvar) tuple for reparameterization.
    [DETERMINISTIC MODE] When use_vae=False, outputs single style tensor (backward compatible).
    """

    def __init__(
        self,
        in_channels: int = 1,
        dim: int = 64,
        style_dim: int = 8,
        n_downsample: int = 4,
        use_vae: bool = False,  # [NEW] VAE mode flag
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            dim (int): Description.
            style_dim (int): Description.
            n_downsample (int): Description.
            use_vae (bool): Description.
        """
        super().__init__()
        self.use_vae = use_vae

        layers = [
            nn.Conv2d(in_channels, dim, 7, 1, 3, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        ]

        # Global Downsampling to destroy spatial info and keep only Texture/Contrast
        for _ in range(n_downsample):
            layers += [nn.Conv2d(dim, dim * 2, 4, 2, 1), nn.LeakyReLU(0.2)]
            dim *= 2

        self.model = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # [VAE] Dual heads for mu (mean) and logvar (log variance)
        self.fc_mu = nn.Linear(dim, style_dim)
        self.fc_var = nn.Linear(dim, style_dim)

        # [BACKWARD COMPAT] Keep legacy fc for deterministic mode
        self.fc = self.fc_mu  # Alias for backward compatibility

        self.output_dim = style_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Description.

        forward method for StyleEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.ndim == 5:
            b, c, d, h, w = x.shape
            if d == 1:
                x = x.squeeze(2)
            else:
                x = x.reshape(b, c * d, h, w)

        h_feat = self.model(x)
        h_feat = self.pool(h_feat).view(h_feat.size(0), -1)

        mu = self.fc_mu(h_feat)
        # [FIX] Clamp mu in both VAE and deterministic modes.
        # In VAE mode mu was previously returned raw (only logvar was clamped),
        # letting unbounded mu values reach AdaIN and drive gamma explosion.
        mu = torch.clamp(mu, min=-10.0, max=10.0)

        if self.use_vae:
            # VAE mode: return (mu, logvar) tuple
            logvar = self.fc_var(h_feat)
            # Clamp logvar to prevent numerical explosion
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            return mu, logvar
        else:
            # Deterministic mode: return clamped style (backward compatible)
            style = torch.clamp(mu, min=-10.0, max=10.0)
            return style


class BiophysicalBoundsActivation(nn.Module):
    """Bounds physical parameters to strictly valid physiological ranges."""

    def __init__(self):
        """__init__."""
        super().__init__()
        # Define strict physiological bounds: [min, max]
        self.bounds = {
            "rho": (0.0, 1.0),
            "T1": (100.0, 5000.0),  # ms (Fat to CSF)
            "T2": (10.0, 2500.0),  # ms
            "T2star": (5.0, 2000.0),  # ms
            "ADC": (0.1e-3, 3.5e-3),  # mm^2/s
        }

    def forward(
        self,
        rho_raw: torch.Tensor,
        t1_raw: torch.Tensor,
        t2_raw: torch.Tensor,
        t2star_raw: torch.Tensor,
        adc_raw: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """forward.

        Args:
            rho_raw (torch.Tensor): Description.
            t1_raw (torch.Tensor): Description.
            t2_raw (torch.Tensor): Description.
            t2star_raw (torch.Tensor): Description.
            adc_raw (torch.Tensor | None): Description.
        Returns:
            dict[str, torch.Tensor]: Description.

        forward method for BiophysicalBoundsActivation.

        Executes PyTorch tensor operations.

        Args:
            rho_raw (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t1_raw (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t2_raw (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t2star_raw (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            adc_raw (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        rho = (
            torch.sigmoid(rho_raw) * (self.bounds["rho"][1] - self.bounds["rho"][0])
            + self.bounds["rho"][0]
        )
        t1 = (
            torch.sigmoid(t1_raw) * (self.bounds["T1"][1] - self.bounds["T1"][0])
            + self.bounds["T1"][0]
        )
        t2 = (
            torch.sigmoid(t2_raw) * (self.bounds["T2"][1] - self.bounds["T2"][0])
            + self.bounds["T2"][0]
        )
        t2star = (
            torch.sigmoid(t2star_raw) * (self.bounds["T2star"][1] - self.bounds["T2star"][0])
            + self.bounds["T2star"][0]
        )

        res = {"rho": rho, "t1": t1, "t2": t2, "t2star": t2star}

        if adc_raw is not None:
            adc = (
                torch.sigmoid(adc_raw) * (self.bounds["ADC"][1] - self.bounds["ADC"][0])
                + self.bounds["ADC"][0]
            )
            res["adc"] = adc

        return res


class SequenceParameterMLP(nn.Module):
    """Predicts sequence parameters (TR, TE, TI, alpha) from style code."""

    def __init__(self, style_dim: int):
        """__init__.

        Args:
            style_dim (int): Description.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(style_dim, 128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(128, 4),  # TR, TE, TI, alpha
        )
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, style: torch.Tensor) -> dict[str, torch.Tensor]:
        """forward.

        Args:
            style (torch.Tensor): Description.
        Returns:
            dict[str, torch.Tensor]: Description.

        forward method for SequenceParameterMLP.

        Executes PyTorch tensor operations.

        Args:
            style (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        out = self.mlp(style)
        # TR: ~10 to >5000 ms
        tr = 10.0 + self.softplus(out[:, 0:1]) * 1000.0
        # TE: ~1 to >200 ms
        te = 1.0 + self.softplus(out[:, 1:2]) * 50.0
        # TI: 0 to ~3000 ms
        ti = self.softplus(out[:, 2:3]) * 1000.0
        # alpha: 1 to >90 degrees (flip angle)
        alpha = 1.0 + self.softplus(out[:, 3:4]) * 20.0

        # Reshape to [B, 1, 1, 1] for broadcasting
        return {
            "predicted_tr": tr.view(-1, 1, 1, 1),
            "predicted_te": te.view(-1, 1, 1, 1),
            "predicted_ti": ti.view(-1, 1, 1, 1),
            "predicted_alpha": alpha.view(-1, 1, 1, 1),
        }


class Generator(nn.Module):
    """AdaIN-based Generator/Decoder.

    Uses AdaIN to inject style into content features.
    Can output either direct image or tissue parameters (ρ, T1, T2).
    """

    def __init__(
        self,
        content_dim: int = 256,
        style_dim: int = 8,
        n_res: int = 4,
        out_channels: int = 1,
        output_tissue_params: bool = False,
        include_adc: bool = False,  # For DWI support
        n_upsample: int = 2,  # [NEW] Number of upsampling layers
    ):
        """__init__.

        Args:
            content_dim (int): Description.
            style_dim (int): Description.
            n_res (int): Description.
            out_channels (int): Description.
            output_tissue_params (bool): Description.
            include_adc (bool): Description.
            n_upsample (int): Description.
        """
        super().__init__()
        self.output_tissue_params = output_tissue_params
        self.include_adc = include_adc

        # Decoder uses AdaIN to inject Style into Content
        # [FIX] Architectural Refinement: NO style injection for tissue param mode.
        # Tissue properties (ρ, T1, T2, T2*) are biological invariants of the
        # patient and must NOT be conditioned on scanner sequence parameters.
        # Style → sequence params mapping is handled by DisentangledMRI, not here.
        res_block_style_dim = None if self.output_tissue_params else style_dim
        self.res_blocks = nn.ModuleList(
            [ResBlock(content_dim, res_block_style_dim) for _ in range(n_res)]
        )

        # Upsampling path (Dynamic based on n_upsample)
        layers = []
        curr_dim = content_dim
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(curr_dim, curr_dim // 2, 5, 1, 2, padding_mode="reflect"),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.LeakyReLU(0.2, inplace=False),
            ]
            curr_dim //= 2

        self.upsample_path = nn.Sequential(*layers)
        final_dim = curr_dim

        if self.output_tissue_params:
            # Shared feature extraction
            self.shared_head = nn.Conv2d(final_dim, final_dim, 7, 1, 3, padding_mode="reflect")

            # Raw logits for biophysical bounds
            self.rho_head = nn.Conv2d(final_dim, 1, 3, 1, 1, padding_mode="reflect")
            self.t1_head = nn.Conv2d(final_dim, 1, 3, 1, 1, padding_mode="reflect")
            self.t2_head = nn.Conv2d(final_dim, 1, 3, 1, 1, padding_mode="reflect")
            self.t2star_head = nn.Conv2d(final_dim, 1, 3, 1, 1, padding_mode="reflect")

            # Bijector for physiological bounds
            self.biophysical_bounds = BiophysicalBoundsActivation()

            # Susceptibility/field inhomogeneity head [NEW]
            # Maps to phase offset (-π, π) representing ΔB₀ field effects
            self.susceptibility_head = nn.Conv2d(final_dim, 1, 3, 1, 1, padding_mode="reflect")

            # [NEW] B1+ Transmit Field Map
            self.b1_plus_head = nn.Sequential(
                nn.Conv2d(final_dim, 1, 3, 1, 1, padding_mode="reflect"), nn.Sigmoid()
            )

            if self.include_adc:
                self.adc_head = nn.Conv2d(final_dim, 1, 3, 1, 1, padding_mode="reflect")
        else:
            # Standard image output
            self.final = nn.Sequential(
                nn.Conv2d(final_dim, out_channels, 7, 1, 3, padding_mode="reflect"),
                nn.Tanh(),
            )
            # [FIX] Zero-init the final Conv2d layer to prevent Tanh saturation
            # This forces initial output to be 0.0 (gray) instead of +1.0 (white)
            # Gradient at Tanh(0) = 1.0 (maximum flow), enabling immediate learning
            final_conv = self.final[0]  # The Conv2d before Tanh
            nn.init.normal_(final_conv.weight, mean=0.0, std=1e-4)
            if final_conv.bias is not None:
                nn.init.zeros_(final_conv.bias)

    def forward(
        self, content: torch.Tensor, style: torch.Tensor
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """forward.

        Args:
            content (torch.Tensor): Description.
            style (torch.Tensor): Description.
        Returns:
            torch.Tensor | dict[str, torch.Tensor]: Description.

        forward method for Generator.

        Executes PyTorch tensor operations.

        Args:
            content (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            style (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor | dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = content
        for block in self.res_blocks:
            # [FIX] Issue #1: Do not inject style into anatomy (tissue parameters)
            x = block(x, style if not self.output_tissue_params else None)

        x = self.upsample_path(x)

        if self.output_tissue_params:
            # Shared processing
            features = self.shared_head(x)

            # Generate raw logits
            rho_raw = self.rho_head(features)
            t1_raw = self.t1_head(features)
            t2_raw = self.t2_head(features)
            t2star_raw = self.t2star_head(features)
            adc_raw = self.adc_head(features) if self.include_adc else None

            # Apply strict physiological bounds
            tissue_params = self.biophysical_bounds(
                rho_raw=rho_raw,
                t1_raw=t1_raw,
                t2_raw=t2_raw,
                t2star_raw=t2star_raw,
                adc_raw=adc_raw,
            )

            # [NEW] Susceptibility/field inhomogeneity head
            chi_logits = self.susceptibility_head(features)
            # Retain unmodified chi for 3D Dipole Kernel mapping, no clamp to [-π, π] needed directly
            tissue_params["chi"] = chi_logits

            # [NEW] B1+ Transmit field map
            b1_plus_raw = self.b1_plus_head(features)
            tissue_params["b1_plus"] = b1_plus_raw + 0.5  # Scale [0, 1] to [0.5, 1.5]

            # [FIX] Architectural Refinement: Sequence parameters are NOT predicted
            # inside the tissue decoder. They are predicted at DisentangledMRI level
            # from the style code, ensuring strict tissue/sequence decoupling.

            return tissue_params
        else:
            return self.final(x)


@register_model(
    name="disentangled_mri",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class DisentangledMRI(nn.Module, IGenerator):
    """Disentangled MRI Synthesis Model.

    Separates Anatomy (Content) from Physics/Contrast (Style) using:
    - ContentEncoder: InstanceNorm strips style, keeps structure
    - StyleEncoder: GlobalAvgPool strips structure, keeps contrast
    - Generator: AdaIN injects style into content

    Args:
        in_channels: Number of input channels (1 for grayscale MRI)
        out_channels: Number of output channels
        dim: Base feature dimension
        style_dim: Dimension of style vector
        n_downsample: Number of downsample layers in content encoder
        n_res: Number of residual blocks in generator
        output_tissue_params: If True, decoder outputs tissue params (ρ, T1, T2)
        enable_bloch_synthesis: If True, synthesize images via Bloch equations
        enable_bias_field: If True, generate and apply multiplicative bias field [NEW]
        bias_field_order: Polynomial order for bias field generation [NEW]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 64,
        style_dim: int = 8,
        n_downsample: int = 2,
        n_res: int = 4,
        output_tissue_params: bool = False,
        enable_bloch_synthesis: bool = False,
        include_adc: bool = False,  # For DWI support
        enable_bias_field: bool = False,  # [NEW] Enable bias field generation
        bias_field_order: str = "polynomial_3",  # [NEW] 'polynomial_3' or 'lowrank_unet'
        use_vae: bool = False,  # [VAE] Enable VAE mode with reparameterization
        spatial_dims: int = 2,  # [NEW] Store spatial dims for reshaping config
        weight_init: str = "kaiming",  # [FIX] Configurable initialization: 'kaiming', 'xavier', 'normal'
        # Aliases for config compatibility
        anatomy_dim: int | None = None,
        **kwargs,  # Ignore extra kwargs for forward compatibility
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            dim (int): Description.
            style_dim (int): Description.
            n_downsample (int): Description.
            n_res (int): Description.
            output_tissue_params (bool): Description.
            enable_bloch_synthesis (bool): Description.
            include_adc (bool): Description.
            enable_bias_field (bool): Description.
            bias_field_order (str): Description.
            use_vae (bool): Description.
            spatial_dims (int): Description.
            weight_init (str): Description.
            anatomy_dim (int | None): Description.
        """
        super().__init__()
        # Use anatomy_dim if provided (config alias for dim)
        if anatomy_dim is not None:
            dim = anatomy_dim

        # Support z_dim as alias for style_dim (common in VAE configs)
        if "z_dim" in kwargs:
            style_dim = kwargs["z_dim"]

        # [FIX] Handle output_mode passed as direct kwarg (factory passes them directly)
        if "output_mode" in kwargs:
            output_mode = kwargs["output_mode"]
            if output_mode == "tissue_params":
                output_tissue_params = True
        if "enable_bloch_synthesis" in kwargs:
            enable_bloch_synthesis = kwargs["enable_bloch_synthesis"]
        if "include_adc" in kwargs:
            include_adc = kwargs["include_adc"]
        if "enable_bias_field" in kwargs:
            enable_bias_field = kwargs["enable_bias_field"]
        if "bias_field_order" in kwargs:
            bias_field_order = kwargs["bias_field_order"]
        if "weight_init" in kwargs:
            weight_init = kwargs["weight_init"]

        # Also check nested model_kwargs for backward compatibility
        if "model_kwargs" in kwargs:
            if "output_mode" in kwargs["model_kwargs"]:
                output_mode = kwargs["model_kwargs"]["output_mode"]
                if output_mode == "tissue_params":
                    output_tissue_params = True
            if "enable_bloch_synthesis" in kwargs["model_kwargs"]:
                enable_bloch_synthesis = kwargs["model_kwargs"]["enable_bloch_synthesis"]
            if "include_adc" in kwargs["model_kwargs"]:
                include_adc = kwargs["model_kwargs"]["include_adc"]
            if "enable_bias_field" in kwargs["model_kwargs"]:
                enable_bias_field = kwargs["model_kwargs"]["enable_bias_field"]
            if "bias_field_order" in kwargs["model_kwargs"]:
                bias_field_order = kwargs["model_kwargs"]["bias_field_order"]
            if "weight_init" in kwargs["model_kwargs"]:
                weight_init = kwargs["model_kwargs"]["weight_init"]

        self.output_tissue_params = output_tissue_params
        self.enable_bloch_synthesis = enable_bloch_synthesis
        self.include_adc = include_adc
        self.enable_bias_field = enable_bias_field  # [NEW]
        self.bias_field_order = bias_field_order  # [NEW]
        self.weight_init = weight_init  # [NEW]

        self.use_vae = use_vae  # [VAE] Store for reparameterization

        if "num_classes" in kwargs:
            self.num_classes = kwargs["num_classes"]
        elif "model_kwargs" in kwargs and "num_classes" in kwargs["model_kwargs"]:
            self.num_classes = kwargs["model_kwargs"]["num_classes"]
        else:
            self.num_classes = None

        if self.num_classes is not None and self.num_classes > 0:
            self.class_embedding = nn.Embedding(self.num_classes, style_dim)
        else:
            self.class_embedding = None

        # For 2.5D slab processing, we expect D=16 when spatial_dims=3, modifying in_channels
        self._d_depth = 16 if spatial_dims == 3 else 1
        eff_in_channels = in_channels * self._d_depth

        self.enc_c = ContentEncoder(eff_in_channels, dim, n_downsample)
        self.enc_s = StyleEncoder(eff_in_channels, dim, style_dim, use_vae=use_vae)
        # The disentangled_vae strategy conditions on [TR, TE, TI, B0]; built here so the
        # optimizer and a fresh state_dict both see the projection (#2388).
        self.phys_to_style_proj = nn.Linear(PHYSICS_VECTOR_DIM, style_dim)
        self.gen = Generator(
            self.enc_c.output_dim,
            style_dim,
            n_res,
            out_channels,
            output_tissue_params=output_tissue_params,
            include_adc=include_adc,
            n_upsample=n_downsample,  # [FIX] Match encoder downsampling
        )

        # [NEW] Bias field generator for receive coil sensitivity modeling
        if self.enable_bias_field:
            self.bias_field_generator = BiasFieldGenerator(
                in_dim=self.enc_c.output_dim,
                order=bias_field_order,
                output_resolution=(
                    256,
                    256,
                ),  # Default resolution, will adapt at runtime
            )

        # [PHYSICS] Bloch Regressor Head
        self.bloch_regressor = BlochRegressor(style_dim, hidden_dim=32, out_dim=4)

        # [FIX] Sequence Parameter MLP — lives at model level, NOT in tissue decoder.
        # Tissue decoder outputs scanner-agnostic biology (ρ, T1, T2, T2*, χ, B1+).
        # Sequence params (TR, TE, TI, α) are derived from style and merged into
        # the output dict here, keeping the two pathways decoupled until Bloch synthesis.
        if self.output_tissue_params:
            self.sequence_mlp = SequenceParameterMLP(style_dim)

        # Store config for interface compliance
        self._in_channels = in_channels
        self._out_channels = out_channels

        # Initialize weights for stability
        self.apply(self._init_weights)

        # [FIX] Re-initialize AdaIN layers after apply(_init_weights).
        # apply() overwrites the carefully set small weights (std=0.01, identity bias)
        # in AdaIN.__init__ with Kaiming Normal weights on every nn.Linear, which
        # is a much larger initialization and is the root cause of early gamma explosion.
        for m in self.modules():
            if isinstance(m, AdaIN):
                nn.init.normal_(m.fc.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.fc.bias)
                n = m.fc.out_features // 2
                m.fc.bias.data[:n] = 1.0  # gamma starts at 1 (identity scale)
                m.fc.bias.data[n:] = 0.0  # beta starts at 0 (no shift)

        # [FIX] Initialize physics heads with physiologically plausible values
        # This MUST be called after apply(_init_weights) to avoid being overwritten
        self._init_physics_heads()

        # Validate no NaN in initial weights
        self._validate_weights()

    def _init_physics_heads(self):
        """Initialize physics heads to start in valid physiological ranges."""
        if not self.output_tissue_params:
            return

        logger = logging.getLogger(__name__)

        # 1. Proton Density (Rho) -> Target Sigmoid(bias) ≈ 0.8 => bias ≈ 1.4
        if hasattr(self.gen, "rho_head") and self.gen.rho_head.bias is not None:
            nn.init.constant_(self.gen.rho_head.bias, 1.4)
            logger.info("  Initialized rho_head bias to 1.4 (Rho ~ 0.8)")

        # 2. T1 Relaxation -> Bounds are [100, 5000]. Target ≈ 1850ms.
        # 1850 = 100 + 4900 * Sigmoid(bias) => Sigmoid(bias) = 1750/4900 ≈ 0.357
        # bias = log(0.357 / (1 - 0.357)) ≈ -0.588
        if hasattr(self.gen, "t1_head") and self.gen.t1_head.bias is not None:
            nn.init.constant_(self.gen.t1_head.bias, -0.588)
            logger.info("  Initialized t1_head bias to -0.588 (T1 ~ 1850ms)")

    def _validate_weights(self):
        """Check for NaN/Inf in model weights."""
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                raise RuntimeError(f"NaN detected in {name} weights after initialization!")
            if torch.isinf(param).any():
                raise RuntimeError(f"Inf detected in {name} weights after initialization!")

    def _check_nan(self, tensor: torch.Tensor | dict, name: str):
        """Check tensor or dict of tensors for NaN/Inf."""
        if isinstance(tensor, dict):
            for k, v in tensor.items():
                self._check_nan(v, f"{name}[{k}]")
            return

        if tensor is None:
            return

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            stats = (
                f"min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}"
                if tensor.numel() > 0
                else "empty"
            )
            raise RuntimeError(
                f"NaN/Inf detected in {name}!\n"
                f"  NaNs: {nan_count}/{tensor.numel()}\n"
                f"  Infs: {inf_count}\n"
                f"  Stats: {stats}"
            )

    def _init_weights(self, m):
        """Initialize weights to prevent initial explosion."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # [FIX] Configurable initialization
            if self.weight_init == "kaiming":
                # Kaiming Normal for LeakyReLU (a=0.2)
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            elif self.weight_init == "xavier":
                nn.init.xavier_normal_(m.weight)
            else:
                # Default Normal
                nn.init.normal_(m.weight, 0.0, 0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "DisentangledMRI"

    @property
    def in_channels(self) -> int:
        """in_channels.

        Returns:
            int: Description.
        """
        return self._in_channels

    @property
    def content_encoder(self) -> nn.Module:
        """Expose content encoder for latent consistency loss."""
        return self.enc_c

    @property
    def style_encoder(self) -> nn.Module:
        """Expose style encoder for latent consistency loss."""
        return self.enc_s

    @property
    def generator(self) -> nn.Module:
        """Expose generator for latent consistency loss."""
        return self.gen

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def forward(
        self,
        x_content_source: torch.Tensor,
        x_style_source: torch.Tensor = None,
        class_idx: torch.Tensor = None,
    ) -> tuple[torch.Tensor | dict, torch.Tensor, torch.Tensor]:
        """Forward pass for disentangled synthesis.

        Args:
            x_content_source: Image to extract anatomy from (e.g., T1)
            x_style_source: Image to extract contrast from (e.g., T2)
                           If None, uses same image (self-reconstruction)
            class_idx: Optional condition label tensor of shape [B]

        Returns:
            Tuple of (output, mu, logvar) for VAE strategy compatibility.
            output: Either image tensor or dict of tissue params {rho, t1, t2, t2star, chi, adc}
            mu: Style encoding
            logvar: Zeros (no KL regularization)

        forward method for DisentangledMRI.

        Executes PyTorch tensor operations.

        Args:
            x_content_source (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_style_source (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            class_idx (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor | dict, torch.Tensor, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Input validation
        if torch.isnan(x_content_source).any():
            logger = logging.getLogger(__name__)
            nan_count = torch.isnan(x_content_source).sum().item()
            logger.error(
                f"[MODEL NaN] NaN in input x_content_source!\n"
                f"  NaNs: {nan_count}/{x_content_source.numel()}\n"
                f"  Range (excluding NaN): [{x_content_source[~torch.isnan(x_content_source)].min():.4f}, {x_content_source[~torch.isnan(x_content_source)].max():.4f}]\n"
                f"  Replacing with zeros!"
            )
            x_content_source = torch.nan_to_num(x_content_source, nan=0.0)

        # 1. Extract Anatomy from Source A (e.g., T1)
        original_shape_a = x_content_source.shape
        content = self.enc_c(x_content_source)
        self._check_nan(content, "ContentEncoder Output")

        # 2. Extract Contrast from Source B (e.g., T2), or self if not provided
        if x_style_source is None:
            x_style_source = x_content_source
        else:
            if torch.isnan(x_style_source).any():
                logger = logging.getLogger(__name__)
                nan_count = torch.isnan(x_style_source).sum().item()
                logger.error(
                    f"[MODEL NaN] NaN in input x_style_source!\n"
                    f"  NaNs: {nan_count}/{x_style_source.numel()}\n"
                    f"  Replacing with zeros!"
                )
                x_style_source = torch.nan_to_num(x_style_source, nan=0.0)

        # [VAE] Encode style - returns (mu, logvar) tuple in VAE mode, single tensor otherwise
        # GUARD: If x_style_source is a 2D physics vector [B, params] (from strategy),
        # skip the StyleEncoder (which uses Conv2d and requires 4D image input).
        # Use the vector directly as the style code.
        if x_style_source.ndim == 2:
            style_output = self._style_from_vector(x_style_source)
            # In VAE mode, create dummy logvar for compatibility
            if self.use_vae:
                style_output = (style_output, torch.zeros_like(style_output))
        else:
            style_output = self.enc_s(x_style_source)

        if self.use_vae:
            # VAE mode: unpack mu and logvar, apply reparameterization
            mu, logvar = style_output
            self._check_nan(mu, "StyleEncoder mu")
            self._check_nan(logvar, "StyleEncoder logvar")
            style = self.reparameterize(mu, logvar)
        else:
            # Deterministic mode: style is the output directly
            style = style_output
            mu = style
            logvar = torch.zeros_like(style)
            self._check_nan(style, "StyleEncoder Output")

        style = self.condition_style(style, class_idx)
        output = self.decode_tissue(content, style)

        # Reshape output backward to 5D if we received 5D with depth > 1
        if isinstance(output, dict) and len(original_shape_a) == 5 and original_shape_a[2] > 1:
            for k, v in output.items():
                # We assume tissue params have 1 intrinsic channel, so when stacked it's 1 * D
                target_channels = 1 * self._d_depth
                if v.ndim == 4 and v.shape[1] == target_channels:
                    output[k] = v.view(v.shape[0], 1, self._d_depth, v.shape[2], v.shape[3])
        elif torch.is_tensor(output) and len(original_shape_a) == 5 and original_shape_a[2] > 1:
            if output.shape[1] == self._out_channels * self._d_depth:
                output = output.view(
                    output.shape[0],
                    self._out_channels,
                    self._d_depth,
                    output.shape[2],
                    output.shape[3],
                )

        # 5. Return VAE-compatible tuple (output, mu, logvar)
        return output, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE Reparameterization Trick: z = mu + sigma * epsilon.

        Enables gradient flow through the sampling operation.
        During training: samples from N(mu, sigma^2)
        During inference: returns mu (deterministic)
        """
        if self.training:
            # [FIX] Clamp logvar to prevent exp() overflow and NaNs
            logvar = torch.clamp(logvar, min=-20.0, max=20.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Deterministic at inference time
            return mu

    def encode_content(self, x: torch.Tensor) -> torch.Tensor:
        """Extract content (anatomy) code."""
        return self.enc_c(x)

    def encode_style(self, x: torch.Tensor) -> torch.Tensor:
        """Extract style (contrast) code."""
        return self.enc_s(x)

    def _style_from_vector(self, vec: torch.Tensor) -> torch.Tensor:
        """A 2-D style input: already a style code, or the physics vector to project."""
        style_dim = self.enc_s.output_dim
        if vec.shape[-1] == style_dim:
            return vec
        if vec.shape[-1] == PHYSICS_VECTOR_DIM:
            return self.phys_to_style_proj(vec)
        raise ValueError(
            f"style vector has width {vec.shape[-1]}; expected style_dim={style_dim} "
            f"or the {PHYSICS_VECTOR_DIM}-element physics vector"
        )

    def decode(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Decode content + style to image.

        Automatically projects physics vectors to style_dim if needed.
        """
        if style.ndim == 2:
            style = self._style_from_vector(style)
        return self.decode_tissue(content, style)

    def condition_style(self, style: torch.Tensor, class_idx: torch.Tensor | None) -> torch.Tensor:
        """Add the contrast-class embedding; out-of-range labels clamp to the last bin."""
        if class_idx is None or self.class_embedding is None:
            return style
        idx = class_idx.to(style.device).clamp(0, self.class_embedding.num_embeddings - 1)
        return style + self.class_embedding(idx)

    def decode_tissue(self, content: torch.Tensor, style: torch.Tensor) -> Any:
        """The one decode: gen, then sequence params (tissue mode), then the bias field.

        forward, decode and the disentangled training strategy all go through here, so
        training, validation and inference decode the same way.
        """
        output = self.gen(content, style)
        self._check_nan(output, "Generator Output")
        if isinstance(output, dict) and self.output_tissue_params:
            # Sequence params come from the style code, never from the tissue decoder.
            output.update(self.sequence_mlp(style))
        if self.enable_bias_field:
            bias_field = self.bias_field_generator(content)
            self._check_nan(bias_field, "BiasFieldGenerator Output")
            if isinstance(output, dict):
                output["bias_field"] = bias_field  # applied once, at render time
            else:
                output = output * bias_field
        return output

    def generate(
        self, x: torch.Tensor, target_style: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """Generate with style transfer (IGenerator interface).

        Args:
            x: Content source image
            target_style: Style vector or image to extract style from
        """
        content = self.enc_c(x)

        if target_style is None:
            style = self.enc_s(x)
        elif target_style.dim() == 1 or (
            target_style.dim() == 2 and target_style.shape[-1] == self.enc_s.output_dim
        ):
            # Already a style vector
            style = target_style
        else:
            # It's an image, encode it
            style = self.enc_s(target_style)

        return self.gen(content, style)

    def predict_physics(self, style: torch.Tensor) -> torch.Tensor:
        """Predict physics parameters from style code (Bloch Regressor).

        Args:
            style: Latent style vector [B, style_dim]

        Returns:
            Predicted physics vector [B, 4] (TR, TE, TI, B0)
        """
        return self.bloch_regressor(style)


class BlochRegressor(nn.Module):
    """Physics Parameter Regressor (The 'Bloch' Head).

    Maps latent style code z_p to physics vector P = [TR, TE, TI, B0].
    """

    def __init__(self, style_dim: int, hidden_dim: int = 32, out_dim: int = 4):
        """__init__.

        Args:
            style_dim (int): Description.
            hidden_dim (int): Description.
            out_dim (int): Description.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),  # Normalized physics parameters are in [0, 1]
        )

    def forward(self, style: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            style (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for BlochRegressor.

        Executes PyTorch tensor operations.

        Args:
            style (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(style)
