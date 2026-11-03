"""Task 1: SNR & Resolution Reconstruction Generators.

Physics-informed generator architectures that compose existing physics
operators with learnable regularisers for marker-anchored MRI reconstruction.

Models
------
- :class:`MoDLSRGenerator`           — Sub-voxel kinematic super-resolution
- :class:`WienerUNetGenerator`       — T2* Lorentzian de-apodization
- :class:`DiffNUFFTGenerator`        — Trajectory auto-calibration
- :class:`KalmanPLLGenerator`        — Phase-locked coherent averaging
- :class:`DirichletESPIRiTGenerator` — CSM boundary extrapolation
- :class:`VCCVarNetGenerator`        — Phase-constrained VCC
- :class:`LSISTAGenerator`           — L+S Unrolled ISTA
- :class:`HardDCForcingGenerator`    — Deep DC forcing
- :class:`PDHGReconGenerator`        — PDHG spatially variant reg
- :class:`NeuralComplexSumGenerator` — bSSFP banding mitigation

All models accept ``marker_prior`` via ``**kwargs`` from the training
strategy.  When available, physics operators are deterministically
extracted from the marker; when not, each model falls back to its learned
CNN pathway so that existing configs remain functional.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared mixin for IGenerator abstract methods
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _Task1Mixin:
    """Default implementations for IGenerator abstract methods."""

    def generate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate output (alias for forward)."""
        return self.forward(x, **kwargs)  # type: ignore[attr-defined]

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the output shape for a given input shape."""
        return input_shape

    def get_parameter_count(self) -> int:
        """Returns total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # type: ignore[attr-defined]


def _zero_init_conv(layer: nn.Module) -> None:
    """Zero-initialize a Conv2d/Linear so residual identity dominates at init."""
    if hasattr(layer, "weight"):
        nn.init.zeros_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared building blocks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _complex_to_ri(x: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor [B, C, H, W] to 2-channel real/imag [B, 2C, H, W]."""
    return torch.cat([x.real, x.imag], dim=1)


def _ensure_real_stacked(x: torch.Tensor) -> torch.Tensor:
    """Guarantee [B, 2, H, W] real tensor from any input format.

    Handles:
      - Complex [B, 1, H, W] → stack real/imag → [B, 2, H, W]
      - Complex [B, C, H, W] → stack real/imag → [B, 2C, H, W]
      - Real [B, 2, H, W] → passthrough
    """
    if torch.is_complex(x):
        return torch.cat([x.real, x.imag], dim=1)
    return x


def _ensure_complex(x: torch.Tensor) -> torch.Tensor:
    """Guarantee complex [B, 1, H, W] from any input format."""
    if torch.is_complex(x):
        return x if x.shape[1] == 1 else x[:, 0:1]
    return torch.complex(x[:, 0:1], x[:, 1:2] if x.shape[1] > 1 else torch.zeros_like(x[:, 0:1]))


class _ResBlock2D(nn.Module):
    """Residual block: Conv-BN-ReLU-Conv-BN + identity."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _ResBlock2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.block(x) + x


class _ConvBlock(nn.Module):
    """Double convolution block: Conv-BN-ReLU × 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _ConvBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.block(x)


class _SmallUNet(nn.Module):
    """Compact U-Net regulariser for use inside unrolled cascades."""

    def __init__(self, in_ch: int, out_ch: int, chans: int = 32) -> None:
        super().__init__()
        self.enc1 = _ConvBlock(in_ch, chans)
        self.enc2 = _ConvBlock(chans, chans * 2)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(chans * 2, chans, 2, stride=2)
        self.dec1 = _ConvBlock(chans * 2, chans)
        self.head = nn.Conv2d(chans, out_ch, 1)
        _zero_init_conv(self.head)  # identity at init for residual skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _SmallUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)
        # Handle size mismatch from odd spatial dims
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def _extract_marker_psf(
    corrupted_input: torch.Tensor,
    marker_prior: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Extract empirical PSF from marker: H(k) = F(m_blurred) / F(m_prior).

    Handles both complex and real-stacked inputs.  Returns complex PSF in
    frequency domain [B, 1, H, W].
    """
    m_blur = _ensure_complex(corrupted_input)
    m_prior = _ensure_complex(marker_prior)

    M_blur = fft2c(m_blur)
    M_prior = fft2c(m_prior)

    # Wiener deconvolution: H = M_blur * conj(M_prior) / (|M_prior|^2 + eps)
    H = M_blur * M_prior.conj() / (M_prior.abs().pow(2) + eps)

    # Normalize spatial kernel to absolute sum = 1.0 (preserves phase/shift)
    h = ifft2c(H)
    h_sum = h.abs().sum(dim=(-2, -1), keepdim=True) + eps
    h_norm = h / h_sum
    return fft2c(h_norm)


def _extract_marker_phase(
    marker_prior: torch.Tensor,
) -> torch.Tensor:
    """Extract low-frequency absolute phase from the marker prior.

    Args:
        marker_prior: Complex marker tensor or real-stacked [B, 2, H, W].

    Returns:
        Phase map [B, 1, H, W] (real-valued).
    """
    if torch.is_complex(marker_prior):
        m = marker_prior
    else:
        m = torch.complex(marker_prior[:, 0:1], marker_prior[:, 1:2])
    return torch.angle(m)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. MoDL Super-Resolution (exp_vf_01)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _MarkerAffineTracker(nn.Module):
    """Extract sub-voxel affine shift from marker_prior vs corrupted marker.

    Uses least-squares phase-gradient estimation in k-space to derive
    the translational shift Δr per repetition without image-space
    interpolation.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        corrupted: torch.Tensor,
        marker_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate sub-voxel shift [B, 2] (Δx, Δy).

        Uses the Fourier Shift Theorem: spatial shift → linear phase ramp
        in k-space.  We extract the phase difference and fit a linear
        plane via least-squares.

        forward method for _MarkerAffineTracker.

        Executes PyTorch tensor operations.

        Args:
            corrupted (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_prior (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Convert to complex if needed
        corrupted = _ensure_complex(corrupted)
        marker_prior = _ensure_complex(marker_prior)

        K_corr = fft2c(corrupted)
        K_prior = fft2c(marker_prior)

        # Phase difference (mask low-magnitude regions)
        mag = K_prior.abs().clamp(min=1e-6)
        phase_diff = torch.angle(K_corr * K_prior.conj())

        B, C, H, W = phase_diff.shape
        # Build k-space coordinate grids
        ky = torch.linspace(-0.5, 0.5, H, device=phase_diff.device)
        kx = torch.linspace(-0.5, 0.5, W, device=phase_diff.device)
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")

        # Weight by magnitude (SNR-aware)
        w = mag[:, 0, :, :].expand(B, -1, -1)  # [B, H, W]
        phi = phase_diff[:, 0, :, :]  # [B, H, W]

        # Weighted least squares for Δr: φ(k) ≈ 2π(kx·Δx + ky·Δy)
        w_flat = w.reshape(B, -1)
        phi_flat = phi.reshape(B, -1)
        kx_flat = kx_grid.reshape(1, -1).expand(B, -1)
        ky_flat = ky_grid.reshape(1, -1).expand(B, -1)

        # A = [kx, ky], solve A^T W A δ = A^T W φ / (2π)
        wkx = w_flat * kx_flat
        wky = w_flat * ky_flat
        wphi = w_flat * phi_flat / (2.0 * math.pi)

        ATA = torch.stack(
            [
                torch.stack(
                    [torch.sum(wkx * kx_flat, dim=1), torch.sum(wkx * ky_flat, dim=1)],
                    dim=1,
                ),
                torch.stack(
                    [torch.sum(wky * kx_flat, dim=1), torch.sum(wky * ky_flat, dim=1)],
                    dim=1,
                ),
            ],
            dim=1,
        )  # [B, 2, 2]

        ATb = torch.stack(
            [
                torch.sum(wkx * phi_flat / (2.0 * math.pi), dim=1),
                torch.sum(wky * phi_flat / (2.0 * math.pi), dim=1),
            ],
            dim=1,
        )  # [B, 2]

        # Solve with regularization for numerical stability
        ATA = ATA + 1e-4 * torch.eye(2, device=ATA.device).unsqueeze(0)
        delta_r = torch.linalg.solve(ATA, ATb.unsqueeze(-1)).squeeze(-1)  # [B, 2]

        return delta_r


class _CGDataConsistency(nn.Module):
    """Conjugate Gradient data consistency solver.

    Solves (A^H A + λI) x = A^H y + λ R(x) via CG iterations.
    Now supports marker-derived affine shift correction by applying
    the inverse phase ramp before DC enforcement.
    """

    def __init__(self, num_cg_iters: int = 6, lambda_reg: float = 0.1) -> None:
        super().__init__()
        self.num_cg_iters = num_cg_iters
        self.lambda_reg = lambda_reg

    def forward(
        self,
        x_reg: torch.Tensor,
        x_input: torch.Tensor,
        mask: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Soft DC in k-space with optional shift correction.

        Args:
            x_reg: Regularised image from CNN ``[B, C, H, W]``.
            x_input: Input (zero-filled or previous estimate).
            mask: k-space sampling mask ``[1, 1, H, W]``.
            shift: Sub-voxel shift ``[B, 2]`` from MarkerAffineTracker.

        Returns:
            Data-consistent image ``[B, C, H, W]``.

        forward method for _CGDataConsistency.

        Executes PyTorch tensor operations.

        Args:
            x_reg (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_input (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            shift (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if mask is None:
            return x_reg

        k_reg = fft2c(x_reg)
        k_input = fft2c(x_input)

        # Apply inverse phase ramp if shift is provided
        if shift is not None:
            B, C, H, W = x_reg.shape
            ky = torch.linspace(-0.5, 0.5, H, device=x_reg.device)
            kx = torch.linspace(-0.5, 0.5, W, device=x_reg.device)
            ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
            # Phase correction: exp(-i 2π k·Δr)
            phase = (
                2.0
                * math.pi
                * (
                    kx_grid.unsqueeze(0) * shift[:, 0:1].unsqueeze(-1)
                    + ky_grid.unsqueeze(0) * shift[:, 1:2].unsqueeze(-1)
                )
            )
            correction = torch.exp(-1j * phase).unsqueeze(1)
            k_input = k_input * correction

        # Soft DC: weighted combination at sampled locations
        lam = self.lambda_reg
        k_dc = mask * (k_input + lam * k_reg) / (1.0 + lam) + (1 - mask) * k_reg

        return _complex_to_ri(ifft2c(k_dc))


@register_model(name="modl_sr", training_mode="reconstruction")
class MoDLSRGenerator(_Task1Mixin, IGenerator, nn.Module):
    """3D MoDL Super-Resolution with Marker Affine Tracking.

    Unrolled Model-Based Deep Learning network with marker-derived
    sub-voxel shift tracking and CG data consistency.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_cascades: Number of unrolled CG+CNN cascades.
        chans: CNN internal channels.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_cascades: int = 5,
        chans: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.tracker = _MarkerAffineTracker()
        self.regularisers = nn.ModuleList(
            [_SmallUNet(in_channels, in_channels, chans) for _ in range(num_cascades)]
        )
        self.dc_layers = nn.ModuleList([_CGDataConsistency() for _ in range(num_cascades)])
        self.output_proj = nn.Conv2d(in_channels, out_channels, 1)
        _zero_init_conv(self.output_proj)

    @property
    def name(self) -> str:
        return "modl_sr"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for MoDLSRGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        mask = kwargs.get("mask")
        marker_prior = kwargs.get("marker_prior")

        # Extract sub-voxel shift from marker if available
        shift = None
        if marker_prior is not None:
            shift = self.tracker(x, marker_prior)

        x_cur = x
        for reg, dc in zip(self.regularisers, self.dc_layers, strict=True):
            x_reg = reg(x_cur) + x_cur  # residual regularisation
            x_cur = dc(x_reg, x, mask, shift=shift)
        return self.output_proj(x_cur) + x  # global residual


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Wiener-UNet (exp_vf_02)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _WienerFilterLayer(nn.Module):
    r"""Differentiable Wiener deconvolution in frequency domain.

    When ``marker_prior`` is available, dynamically computes the PSF:
    H(k) = F(m_blurred) / F(m_prior).

    Otherwise falls back to a static learnable SNR parameter:
    X(k) = Y(k) · H*(k) / (|H(k)|² + 1/SNR)
    """

    def __init__(self, snr_init: float = 20.0) -> None:
        super().__init__()
        self.log_snr = nn.Parameter(torch.tensor(math.log(snr_init)))

    def forward(
        self,
        y: torch.Tensor,
        psf_fft: torch.Tensor | None = None,
        marker_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply Wiener filter with marker-derived or explicit PSF.

        Args:
            y: Input image ``[B, C, H, W]``.
            psf_fft: Pre-computed PSF in frequency domain (optional).
            marker_prior: Marker prior ``[B, 2, H, W]`` (optional).

        Returns:
            Deconvolved image ``[B, C, H, W]``.

        forward method for _WienerFilterLayer.

        Executes PyTorch tensor operations.

        Args:
            y (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            psf_fft (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_prior (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Dynamically extract PSF from marker if available
        if psf_fft is None and marker_prior is not None:
            psf_fft = _extract_marker_psf(y, marker_prior)

        if psf_fft is None:
            return y

        snr = torch.exp(self.log_snr)
        Y = fft2c(y)
        H = psf_fft
        # Wiener filter: X = Y · H* / (|H|² + 1/SNR)
        X = Y * H.conj() / (H.abs().pow(2) + 1.0 / snr)
        return _complex_to_ri(ifft2c(X))


_SWINIR_REFINER_KEYS = frozenset(
    {"embed_dim", "depths", "num_heads", "window_size", "mlp_ratio", "img_size"}
)


def _build_resnet_refiner(
    m: nn.Module, cin: int, cout: int, features: int, num_resblocks: int, **_: Any
) -> None:
    m.stem = nn.Conv2d(cin, features, 3, padding=1)
    m.resblocks = nn.Sequential(*[_ResBlock2D(features) for _ in range(num_resblocks)])
    m.head = nn.Conv2d(features, cout, 1)
    _zero_init_conv(m.head)


def _apply_resnet_refiner(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return m.head(m.resblocks(m.stem(x)))


def _build_swinir_refiner(
    m: nn.Module, cin: int, cout: int, refiner_kwargs: dict[str, Any], **_: Any
) -> None:
    from spectramr.models.generators.swinir_generator import SwinIRGenerator

    unknown = set(refiner_kwargs) - _SWINIR_REFINER_KEYS
    if unknown:
        raise ValueError(
            f"wiener_unet: unknown refiner_kwargs {sorted(unknown)} for refiner='swinir'; "
            f"allowed: {sorted(_SWINIR_REFINER_KEYS)}"
        )
    kw = dict(refiner_kwargs)
    for key in ("depths", "num_heads"):
        if key in kw:
            kw[key] = tuple(kw[key])
    m.swinir = SwinIRGenerator(in_channels=cin, out_channels=cout, in_chans=cin, upscale=1, **kw)
    _zero_init_conv(m.swinir.upsample[-1])


def _apply_swinir_refiner(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return m.swinir(x)


#: wiener_unet's post-Wiener refiner by ``model_kwargs.refiner``: (build, apply). The ResNet
#: keeps its original attribute names so existing checkpoints still load.
_WIENER_REFINERS = {
    "resnet": (_build_resnet_refiner, _apply_resnet_refiner),
    "swinir": (_build_swinir_refiner, _apply_swinir_refiner),
}


@register_model(name="wiener_unet", training_mode="reconstruction")
class WienerUNetGenerator(_Task1Mixin, IGenerator, nn.Module):
    """Dual-domain Wiener-UNet with marker-derived PSF extraction.

    Dynamically computes the empirical PSF from the marker_prior,
    applies a Wiener filter in k-space, then cleans residual
    high-frequency noise with a spatial ResNet.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_resblocks: Number of ResNet blocks (``refiner='resnet'``).
        features: Hidden feature width (``refiner='resnet'``).
        refiner: Post-Wiener refiner, ``'resnet'`` or ``'swinir'``.
        refiner_kwargs: SwinIR hyper-parameters (``refiner='swinir'`` only).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_resblocks: int = 8,
        features: int = 64,
        refiner: str = "resnet",
        refiner_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if refiner not in _WIENER_REFINERS:
            raise ValueError(
                f"wiener_unet: unknown refiner {refiner!r}; expected one of {sorted(_WIENER_REFINERS)}"
            )
        if refiner_kwargs and refiner != "swinir":
            raise ValueError("wiener_unet: refiner_kwargs is read only with refiner='swinir'")
        self.refiner_name = refiner
        # use_marker=False forces the no-marker path (blind Wiener — the model
        # already branches on ``marker_prior is None``), making the paired
        # ablation baseline a one-knob config change (pitfall #17).
        self.use_marker = bool(kwargs.get("use_marker", True))
        # zero-init residual ⇒ identity at init by design (learns the residual);
        # opt out of the Tier-2 identity-collapse FP (shape/OOM/input-invariance
        # probe checks still apply). Same as SiameseEPITracker.
        self.synthetic_forward_probe_skip = {"identity_collapse"}
        self.wiener = _WienerFilterLayer()
        build, self._apply_refiner = _WIENER_REFINERS[refiner]
        build(
            self,
            in_channels,
            out_channels,
            features=features,
            num_resblocks=num_resblocks,
            refiner_kwargs=refiner_kwargs or {},
        )

    @property
    def name(self) -> str:
        return "wiener_unet"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for WienerUNetGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        psf_fft = kwargs.get("psf_fft")
        marker_prior = kwargs.get("marker_prior")
        if not self.use_marker:
            marker_prior = None  # blind-deconvolution baseline

        # Wiener deconvolution (dynamically extracts PSF from marker)
        x_wiener = self.wiener(x, psf_fft, marker_prior=marker_prior)

        # Spatial refiner (ResNet or SwinIR) cleans the amplified high-frequency noise
        return self._apply_refiner(self, x_wiener) + x  # global residual


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Differentiable NUFFT Trajectory Calibrator (exp_vf_03)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="diff_nufft", training_mode="reconstruction")
class DiffNUFFTGenerator(_Task1Mixin, IGenerator, nn.Module):
    """Differentiable NUFFT trajectory calibrator.

    Optimises learnable trajectory delay parameters Δk via SSIM against
    the marker_prior.  The correction is applied as a k-space phase ramp.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Learnable trajectory delay parameters (init=0 → identity)
        self.delay_ro = nn.Parameter(torch.zeros(1))
        self.delay_pe = nn.Parameter(torch.zeros(1))
        # Post-correction refinement CNN (zero-init last layer for identity)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )
        _zero_init_conv(self.refine[-1])

    @property
    def name(self) -> str:
        return "diff_nufft"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for DiffNUFFTGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        B, C, H, W = x.shape
        marker_prior = kwargs.get("marker_prior")

        # Build phase ramp from delay parameters
        kx = torch.linspace(-0.5, 0.5, W, device=x.device)
        ky = torch.linspace(-0.5, 0.5, H, device=x.device)
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
        phase = 2 * math.pi * (kx_grid * self.delay_ro + ky_grid * self.delay_pe)
        correction = torch.exp(1j * phase).unsqueeze(0).unsqueeze(0)

        # Apply correction in k-space via centered FFT
        X_k = fft2c(x)
        X_k = X_k * correction
        x_corrected = _complex_to_ri(ifft2c(X_k))

        # If marker_prior available, use SSIM loss against it for
        # self-supervised trajectory calibration (marker = known reference)
        if marker_prior is not None and self.training:
            mp = _ensure_real_stacked(marker_prior)
            if mp.shape[0] != B:
                mp = mp.expand(B, -1, -1, -1)
            # Store marker reference for external loss computation
            self._last_marker_target = mp

        return self.refine(x_corrected) + x_corrected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Kalman-PLL (exp_vf_04)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="kalman_pll", training_mode="reconstruction")
class KalmanPLLGenerator(_Task1Mixin, IGenerator, nn.Module):
    """Extended Kalman Filter Phase-Locked Loop for B0 drift demodulation.

    When ``marker_prior`` is available, extracts the 0th-order global
    phase offset by computing the complex inner product (matched filter)
    between the corrupted marker and the pristine prior:
        φ₀ = ∠( Σ_k Y_m(k) · M_prior*(k) )

    This deterministic phase is then conjugate-multiplied out of the
    input signal before a lightweight CNN refinement.

    Without a marker, falls back to a learned CNN phase estimator with
    a gated identity initialisation.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        hidden_dim: CNN hidden dimension for fallback estimator.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Fallback CNN phase estimator (gated → identity at init)
        self.phase_estimator = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Tanh(),
        )
        # Learnable phase gate (init=-5 → sigmoid≈0 → no CNN phase at start)
        self.phase_gate = nn.Parameter(torch.tensor(-5.0))
        # Demodulation refinement
        self.refine = _SmallUNet(in_channels, out_channels, 32)

    @property
    def name(self) -> str:
        return "kalman_pll"

    def _extract_marker_phase_scalar(
        self,
        corrupted_input: torch.Tensor,
        marker_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Extract 0th-order phase via matched filter inner product.

        φ₀ = ∠( Σ_k F(corrupted) · F(prior)* )

        Returns:
            Phase scalar [B, 1, 1, 1] for broadcasting.
        """
        corr = _ensure_complex(corrupted_input)
        prior = _ensure_complex(marker_prior)

        K_corr = fft2c(corr)
        K_prior = fft2c(prior)

        # Complex inner product (matched filter)
        inner = torch.sum(K_corr * K_prior.conj(), dim=(-2, -1), keepdim=True)
        phi_0 = torch.angle(inner)  # [B, 1, 1, 1]
        return phi_0

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for KalmanPLLGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")

        if marker_prior is not None:
            # Deterministic phase extraction from marker
            phi_0 = self._extract_marker_phase_scalar(x, marker_prior)
        else:
            # Fallback: CNN-estimated phase (gated for identity init)
            phi_0 = self.phase_estimator(x) * math.pi * self.phase_gate.sigmoid()

        # Demodulation: rotate by negative estimated phase
        cos_p = torch.cos(phi_0)
        sin_p = torch.sin(phi_0)

        if x.shape[1] >= 2:
            x_real = x[:, 0:1] * cos_p + x[:, 1:2] * sin_p
            x_imag = -x[:, 0:1] * sin_p + x[:, 1:2] * cos_p
            x_demod = torch.cat([x_real, x_imag], dim=1)
        else:
            x_demod = x * cos_p

        return self.refine(x_demod) + x_demod


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Dirichlet-ESPIRiT (exp_vf_05)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _PowerIterationBlock(nn.Module):
    """One step of learned power iteration for CSM estimation."""

    def __init__(self, chans: int = 18) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(chans, chans, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, chans, 3, padding=1),
        )

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """forward method for _PowerIterationBlock.

        Executes PyTorch tensor operations.

        Args:
            S (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return F.normalize(self.refine(S) + S, dim=1)


@register_model(name="dirichlet_espirit", training_mode="reconstruction")
class DirichletESPIRiTGenerator(_Task1Mixin, IGenerator, nn.Module):
    """Unrolled Power Iteration ESPIRiT with Dirichlet boundary forcing.

    Refines coil sensitivity maps via learned power iterations
    constrained by the known marker boundary condition.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_power_iters: Number of power iteration steps.
        chans: Internal channels.
        use_marker: ``False`` drops the Dirichlet marker anchoring, so the same
            power-iteration network calibrates natively: the marker-off control.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_power_iters: int = 6,
        chans: int = 18,
        use_marker: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not isinstance(use_marker, bool):
            raise TypeError(f"use_marker must be a bool, got {use_marker!r}.")
        self.use_marker = use_marker
        self.stem = nn.Conv2d(in_channels, chans, 3, padding=1)
        self.power_iters = nn.ModuleList(
            [_PowerIterationBlock(chans) for _ in range(num_power_iters)]
        )
        self.recon = _SmallUNet(chans + in_channels, out_channels, 32)
        # Learned boundary value from the marker amplitude; built only when used so
        # the marker-off control carries no parameter outside the autograd graph.
        self.marker_embed = nn.Conv2d(1, chans, 3, padding=1) if use_marker else None

    @property
    def name(self) -> str:
        return "dirichlet_espirit"

    def _marker_boundary(
        self, marker_prior: torch.Tensor, batch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the binary marker footprint ``w`` and the unit-norm boundary value ``g``.

        The footprint uses the embedder's own rule (``|m| > 0.01 max|m|``), which is
        scale-invariant, so it reproduces ``CornerFiducialEmbedder.marker_mask``.
        """
        mp = _ensure_real_stacked(marker_prior)
        if mp.shape[0] != batch:
            mp = mp.expand(batch, -1, -1, -1)
        mag = mp.abs().amax(dim=1, keepdim=True)
        mag = mag / mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        w = (mag > 0.01).to(mag.dtype)
        if self.marker_embed is None:
            raise RuntimeError("_marker_boundary called on a use_marker=False model.")
        g = F.normalize(self.marker_embed(mag), dim=1)
        return w, g

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for DirichletESPIRiTGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")

        boundary = None
        if marker_prior is not None and self.use_marker:
            boundary = self._marker_boundary(marker_prior, x.shape[0])

        # Dirichlet condition: S is pinned to g on the marker footprint and re-imposed
        # after every power iteration, so the conv steps carry it into the interior.
        S = self.stem(x)
        if boundary is not None:
            S = S * (1.0 - boundary[0]) + boundary[1] * boundary[0]
        for pi in self.power_iters:
            S = pi(S)
            if boundary is not None:
                S = S * (1.0 - boundary[0]) + boundary[1] * boundary[0]

        # Concatenate refined CSM features with input + global residual
        return self.recon(torch.cat([S, x], dim=1)) + x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. VCC-VarNet (exp_vf_06)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _conj_reflect_kspace(k: torch.Tensor) -> torch.Tensor:
    """Return ``y*(-k)`` for a CENTERED (fft2c-layout) spectrum.

    Frequency negation about the centre bin on an even grid is flip followed
    by a one-bin roll: a bare flip reflects about the inter-pixel centre,
    shifting every frequency by one bin — a global linear-phase error on the
    virtual-conjugate channel.
    """
    return k.conj().flip(-1, -2).roll(shifts=(1, 1), dims=(-1, -2))


class _VCCBlock(nn.Module):
    """Virtual Conjugate Coil synthesis block.

    When ``marker_prior`` is available, extracts the deterministic
    low-frequency absolute phase φ_syn from the marker and emits the
    image-domain VCC channel
        x_vcc(r) = F⁻¹{ y*(-k) } · e^{i 2φ_syn(r)}
    concatenated with the image-domain input (the k-space VCC signal is
    ``F{x_vcc}``; the cascades here operate on image-domain features, so
    the outer transform is not applied).

    Otherwise falls back to a learned 1×1 phase estimator.
    """

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.phase_net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        marker_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """forward method for _VCCBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_prior (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if marker_prior is not None:
            # Deterministic phase from marker
            phi = _extract_marker_phase(marker_prior)  # [B, 1, H, W]
            # Image-domain VCC channel: x_vcc = F⁻¹{y*(-k)} · e^{i 2φ}
            x_complex = _ensure_complex(x)
            x_conj_reflected = ifft2c(_conj_reflect_kspace(fft2c(x_complex)))
            # Phase rotation
            x_vcc = x_conj_reflected * torch.exp(1j * 2 * phi)
            x_vcc_ri = _complex_to_ri(x_vcc)
            # Handle channel count mismatch
            if x_vcc_ri.shape[1] != x.shape[1]:
                x_vcc_ri = x_vcc_ri[:, : x.shape[1]]
            return torch.cat([x, x_vcc_ri], dim=1)
        else:
            # Fallback: learned phase approximation
            phase = self.phase_net(x) * math.pi
            x_conj = x * torch.cos(2 * phase) + x.flip(-1) * torch.sin(2 * phase)
            return torch.cat([x, x_conj], dim=1)


class _VarNetCascade(nn.Module):
    """One VarNet cascade: regulariser + soft DC."""

    def __init__(self, channels: int, chans: int = 18) -> None:
        super().__init__()
        self.reg = nn.Sequential(
            nn.Conv2d(channels, chans, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, chans, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, channels, 3, padding=1),
        )
        self.dc_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        """forward method for _VarNetCascade.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_ref (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        reg = self.reg(x)
        dc = self.dc_weight * (x - x_ref)
        return x - reg - dc


@register_model(name="vcc_varnet", training_mode="reconstruction")
class VCCVarNetGenerator(_Task1Mixin, IGenerator, nn.Module):
    """Phase-Constrained VCC-VarNet with marker-derived phase.

    When marker_prior is available, extracts the deterministic absolute
    phase for Hermitian symmetry enforcement via Virtual Conjugate Coils,
    doubling effective k-space sampling and breaking the thermal noise floor.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_cascades: Number of VarNet cascades.
        chans: Internal cascade channels.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_cascades: int = 8,
        chans: int = 18,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.vcc = _VCCBlock(in_channels)
        doubled = in_channels * 2
        self.cascades = nn.ModuleList([_VarNetCascade(doubled, chans) for _ in range(num_cascades)])
        self.output_proj = nn.Conv2d(doubled, out_channels, 1)
        _zero_init_conv(self.output_proj)

    @property
    def name(self) -> str:
        return "vcc_varnet"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for VCCVarNetGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")
        x_vcc = self.vcc(x, marker_prior=marker_prior)
        x_ref = x_vcc
        x_cur = x_vcc
        for cascade in self.cascades:
            x_cur = cascade(x_cur, x_ref)
        return self.output_proj(x_cur) + x  # global residual


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. L+S Unrolled ISTA (exp_vf_07)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _CNNThresholder(nn.Module):
    """Learned soft-thresholder replacing hand-tuned threshold."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _CNNThresholder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(x)


@register_model(name="ls_ista", training_mode="reconstruction")
class LSISTAGenerator(_Task1Mixin, IGenerator, nn.Module):
    """L+S Unrolled ISTA with CNN-parameterised thresholders.

    Decomposes the input into Low-rank background (SVD) and
    Sparse dynamics (CNN soft-thresholding).

    When marker_prior is available, the rank estimation is guided
    by the marker's temporal subspace singular values.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_iters: Number of ISTA unrolling iterations.
        rank: Target rank for low-rank component.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_iters: int = 8,
        rank: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_iters = num_iters
        self.rank = rank
        self.thresholders = nn.ModuleList([_CNNThresholder(in_channels) for _ in range(num_iters)])
        self.step_sizes = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(num_iters)]
        )
        self.output_proj = nn.Conv2d(in_channels, out_channels, 1)
        _zero_init_conv(self.output_proj)

    @property
    def name(self) -> str:
        return "ls_ista"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for LSISTAGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        B, C, H, W = x.shape
        marker_prior = kwargs.get("marker_prior")

        # 1. Low-rank background (L) projection via marker Temporal Subspace
        if marker_prior is not None and marker_prior.shape[1] == C:
            # Ensure marker batch matches input batch (last val batch may differ)
            mp = marker_prior
            if mp.shape[0] != B:
                mp = mp[:B] if mp.shape[0] > B else mp
            # If batch still doesn't match, skip subspace projection
            if mp.shape[0] != B:
                L = torch.zeros_like(x)
                x_sparse_cur = x
            else:
                # Ensure marker spatial dims match input
                if mp.shape[-2:] != x.shape[-2:]:
                    if mp.is_complex():
                        mp_r = torch.nn.functional.interpolate(
                            mp.real,
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        )
                        mp_i = torch.nn.functional.interpolate(
                            mp.imag,
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        )
                        mp = torch.complex(mp_r, mp_i)
                    else:
                        mp = torch.nn.functional.interpolate(
                            mp,
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        )
                # Form Casorati matrix [B, C, H*W]
                M_cas = mp.reshape(B, C, -1)
                # SVD of marker Casorati matrix
                U, S_vals, Vh = torch.linalg.svd(M_cas, full_matrices=False)

                # Keep top 'rank' singular vectors
                r = min(self.rank, C)
                U_r = U[:, :, :r]  # [B, C, rank]

                # Project input onto the marker's low-rank temporal subspace
                X_cas = x.reshape(B, C, -1)
                # L = U_r U_r^H X
                L_cas = torch.bmm(U_r, torch.bmm(U_r.transpose(-2, -1).conj(), X_cas))
                L = L_cas.reshape(B, C, H, W)

                # Sparse component is the residual
                x_sparse_cur = x - L
        else:
            L = torch.zeros_like(x)
            x_sparse_cur = x

        # 2. Sparse (S) dynamics reconstruction via CNN-ISTA
        for i in range(self.num_iters):
            # Gradient step w.r.t data consistency
            grad = (L + x_sparse_cur) - x
            x_sparse_cur = x_sparse_cur - self.step_sizes[i] * grad

            # Sparse thresholding via CNN
            x_sparse_cur = self.thresholders[i](x_sparse_cur) + x_sparse_cur

        return self.output_proj(L + x_sparse_cur) + x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Hard DC Forcing (exp_vf_08)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _HardDCLayer(nn.Module):
    """Hard data-consistency forcing in k-space.

    Replaces measured k-space locations with ground-truth values:
    X_out(k) = M · Y + (1-M) · F(CNN(x))

    When marker_prior is available, additionally enforces the marker's
    k-space lines: X_out[M_syn_k] = Y_syn_prior[M_syn_k]
    """

    def forward(
        self,
        x_cnn: torch.Tensor,
        x_input: torch.Tensor,
        mask: torch.Tensor | None = None,
        marker_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """forward method for _HardDCLayer.

        Executes PyTorch tensor operations.

        Args:
            x_cnn (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_input (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_prior (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if mask is None and marker_prior is None:
            return x_cnn

        k_cnn = fft2c(x_cnn)

        if mask is not None:
            k_input = fft2c(x_input)
            k_out = mask * k_input + (1 - mask) * k_cnn
        else:
            k_out = k_cnn

        # Additional marker k-space forcing
        if marker_prior is not None:
            k_marker = fft2c(marker_prior)
            # Derive mask from marker magnitude (non-zero regions)
            marker_k_mask = (k_marker.abs() > 1e-6).float()
            k_out = marker_k_mask * k_marker + (1 - marker_k_mask) * k_out

        return _complex_to_ri(ifft2c(k_out))


@register_model(name="hard_dc_forcing", training_mode="reconstruction")
class HardDCForcingGenerator(_Task1Mixin, IGenerator, nn.Module):
    """CNN with hard k-space data consistency forcing.

    Appends a deterministic DC projection after every CNN block,
    anchored by the marker prior mask.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_cascades: Number of CNN + DC cascades.
        chans: CNN internal channels.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_cascades: int = 8,
        chans: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.cascades = nn.ModuleList(
            [_SmallUNet(in_channels, in_channels, chans) for _ in range(num_cascades)]
        )
        self.dc_layers = nn.ModuleList([_HardDCLayer() for _ in range(num_cascades)])
        self.output_proj = nn.Conv2d(in_channels, out_channels, 1)
        _zero_init_conv(self.output_proj)

    @property
    def name(self) -> str:
        return "hard_dc_forcing"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for HardDCForcingGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        mask = kwargs.get("mask")
        marker_prior = kwargs.get("marker_prior")
        x_cur = x
        for cnn, dc in zip(self.cascades, self.dc_layers, strict=True):
            x_cnn = cnn(x_cur) + x_cur
            x_cur = dc(x_cnn, x, mask, marker_prior=marker_prior)
        return self.output_proj(x_cur) + x  # global residual


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. PDHG Reconstruction (exp_vf_09)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _VarianceEstimatorCNN(nn.Module):
    """Estimates spatially-variant noise variance.

    When marker_prior is available, computes the empirical noise
    variance from the marker ROI and broadcasts it spatially.
    Otherwise uses a learned CNN estimator.
    """

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Softplus(),  # ensure positive
        )

    def forward(
        self,
        x: torch.Tensor,
        marker_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """forward method for _VarianceEstimatorCNN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_prior (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if marker_prior is not None:
            # Empirical noise variance from marker residual
            residual = x - marker_prior
            var = residual.var(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
            return var  # [B, 1, 1, 1]
        return self.net(x).clamp(min=1e-6)


@register_model(name="pdhg_recon", training_mode="reconstruction")
class PDHGReconGenerator(_Task1Mixin, IGenerator, nn.Module):
    """Primal-Dual Hybrid Gradient reconstruction.

    Unrolled PDHG with marker-derived variance-weighted Total Variation.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_iters: Number of PDHG iterations.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_iters: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_iters = num_iters
        self.var_estimator = _VarianceEstimatorCNN(in_channels)

        # Learned step sizes
        self.tau = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(num_iters)])
        self.sigma = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(num_iters)])
        # Primal proximal CNNs
        self.primal_cnns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, in_channels, 3, padding=1),
                )
                for _ in range(num_iters)
            ]
        )
        self.output_proj = nn.Conv2d(in_channels, out_channels, 1)
        _zero_init_conv(self.output_proj)

    @property
    def name(self) -> str:
        return "pdhg_recon"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for PDHGReconGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")
        var_map = self.var_estimator(x, marker_prior=marker_prior)

        B, C, H, W = x.shape
        x_cur = x
        p = torch.zeros(B, 2 * C, H, W, device=x.device, dtype=x.dtype)

        for i in range(self.num_iters):
            # Dual update: p = proj(p + σ∇x̄)
            dx = torch.diff(x_cur, dim=-1, prepend=x_cur[..., :1])
            dy = torch.diff(x_cur, dim=-2, prepend=x_cur[..., :1, :])
            grad_x = torch.cat([dy, dx], dim=1)
            p = p + self.sigma[i] * grad_x
            # Project onto variance-weighted ball
            p_norm = p.norm(dim=1, keepdim=True).clamp(min=1e-8)
            p_max = var_map.expand_as(p_norm)
            p = p / (p_norm / p_max).clamp(min=1.0)

            # Primal update: x = prox(x - τ·div(p))
            div_p = -(
                torch.diff(p[:, :C], dim=-2, append=p[:, :C, -1:, :])
                + torch.diff(p[:, C:], dim=-1, append=p[:, C:, ..., -1:])
            )
            x_cur = x_cur - self.tau[i] * div_p
            x_cur = self.primal_cnns[i](x_cur) + x_cur
            # Data fidelity
            x_cur = (x_cur + self.tau[i] * x) / (1.0 + self.tau[i])

        return self.output_proj(x_cur) + x  # global residual


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Neural Complex-Sum (exp_vf_10)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@register_model(name="neural_complex_sum", training_mode="reconstruction")
class NeuralComplexSumGenerator(_Task1Mixin, IGenerator, nn.Module):
    """Neural Complex-Sum combiner for bSSFP banding mitigation.

    Voxel-wise MLP (1×1 convolutions) that learns optimal
    combination weights across phase cycles, conditioned on ΔB0.

    When marker_prior is available, extracts the ΔB0 map from the
    marker phase to condition the combination weights.

    Args:
        in_channels: Input channels (stacked phase cycles).
        out_channels: Output channels.
        hidden_dim: MLP hidden dimension.
        num_layers: Number of MLP layers.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # use_marker=False forces the no-ΔB0 path (uniform/input-only combine —
        # the model already branches on ``marker_prior is None``) → paired
        # baseline is one config knob (pitfall #17).
        self.use_marker = bool(kwargs.get("use_marker", True))
        # zero-init residual ⇒ identity at init by design (learns the residual);
        # opt out of the Tier-2 identity-collapse FP (shape/OOM/input-invariance
        # probe checks still apply). Same as SiameseEPITracker.
        self.synthetic_forward_probe_skip = {"identity_collapse"}
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_dim, hidden_dim, 1),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1))
        self.mlp = nn.Sequential(*layers)
        _zero_init_conv(self.mlp[-1])  # identity at init

        # Spatial softmax weights conditioned on input (or ΔB0)
        self.weight_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1),
        )
        _zero_init_conv(self.weight_net[-1])  # sigmoid(0)=0.5 is acceptable when mlp=0

    @property
    def name(self) -> str:
        return "neural_complex_sum"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for NeuralComplexSumGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = _ensure_real_stacked(x)  # guard: complex64 → [B, 2, H, W] real
        marker_prior = kwargs.get("marker_prior")
        if not self.use_marker:
            marker_prior = None  # uniform-combine baseline (no marker ΔB0)

        # If marker is available, extract ΔB0 and concatenate as conditioning
        if marker_prior is not None:
            delta_b0 = _extract_marker_phase(marker_prior)  # [B, 1, H, W]
            # Expand ΔB0 to match input channel count for conditioning
            delta_b0_expanded = delta_b0.expand_as(x[:, :1])
            # Use ΔB0 to modulate the weight network
            weight_input = x + delta_b0_expanded.expand_as(x) * 0.1
        else:
            weight_input = x

        # MLP prediction
        x_mlp = self.mlp(x)
        # Per-element attention gating
        weights = torch.sigmoid(self.weight_net(weight_input))
        return x_mlp * weights + x  # global residual


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# __all__
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    "DiffNUFFTGenerator",
    "DirichletESPIRiTGenerator",
    "HardDCForcingGenerator",
    "KalmanPLLGenerator",
    "LSISTAGenerator",
    "MoDLSRGenerator",
    "NeuralComplexSumGenerator",
    "PDHGReconGenerator",
    "VCCVarNetGenerator",
    "WienerUNetGenerator",
]
