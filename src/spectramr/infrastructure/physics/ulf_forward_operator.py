"""Differentiable ULF Forward Operator (𝒜_ULF).

Maps 3T quantitative parameter maps (ρ, T₁, T₂) to physically accurate 64mT
k-space through 5 sequential electrodynamic stages:

    Stage 1: power-law T₁ dispersion + Ernst-angle Mxy rendering
    Stage 2: Gradient Non-Linearity (GNL) warping via spherical harmonics
    Stage 3: Maxwell concomitant phase + B₀ off-resonance injection
    Stage 4: Time-Segmented NUFFT (temporal integration)
    Stage 5: Thermal noise (B₀^{-7/4} scaling) + structured EMI

All stages are fully differentiable. Composes existing codebase primitives
without duplication.

References:
    Bottomley, P. A. et al. (1984). Medical Physics.
    Bernstein, M. A. et al. (1998). MRM 39(2):300-308.
    Noll, D. C. et al. (1991). IEEE TMI.
    Winter, L. et al. (2020). Nature Scientific Reports.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.dispersion import (
    power_law_t1_transport,
    power_law_t2_transport,
)
from spectramr.infrastructure.physics.emi_degradation import readout_tone_field
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.field_simulation import B0MapSimulator
from spectramr.infrastructure.physics.spherical_harmonics import (
    _build_2d_polynomial_basis,
)

logger = logging.getLogger(__name__)

# Lazy NUFFT import
_NUFFT_AVAILABLE: bool | None = None


def _check_nufft() -> bool:
    global _NUFFT_AVAILABLE
    if _NUFFT_AVAILABLE is None:
        try:
            import torchkbnufft  # noqa: F401

            _NUFFT_AVAILABLE = True
        except ImportError:
            _NUFFT_AVAILABLE = False
    return _NUFFT_AVAILABLE


# ──────────────────────────────────────────────────────────────────
# Stage 1: power-law dispersion + Ernst Mxy
# ──────────────────────────────────────────────────────────────────


def power_law_dispersion(
    t1_source: torch.Tensor,
    t2_source: torch.Tensor,
    b0_source: float = 3.0,
    b0_target: float = 0.064,
    beta: float = 0.35,
    gamma_t2: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport (T₁, T₂) from a source field to a target field by power law.

    T₁(B₀) ∝ B₀^β  (Bottomley 1984, β ≈ 0.3–0.4 for brain tissue)
    T₂(B₀) ∝ B₀^γ  (weak, exponent ≈ 0.05)

    Renamed from ``bpp_dispersion``: both halves are empirical power laws, and the
    only citation the original docstring carried was Bottomley 1984, which is the
    power-law reference. Bloembergen-Purcell-Pound is a different model with pool
    amplitudes and correlation times, and it lives in
    :mod:`spectramr.infrastructure.physics.dispersion` as :func:`dispersion_t1` /
    :func:`dispersion_t2`. Calling this one BPP made two distinct laws read as one.

    Both halves now delegate to that module, which owns the power law and has the
    production importers (non-negotiable 17). This function stays as the composite
    the operator's stage 1 calls, not as a second implementation.

    Args:
        t1_source: T₁ at source field [ms].
        t2_source: T₂ at source field [ms].
        b0_source: Source field strength [T].
        b0_target: Target field strength [T].
        beta: T₁ dispersion exponent.
        gamma_t2: T₂ dispersion exponent.

    Returns:
        (t1_target, t2_target) in ms.

    Raises:
        ValueError: when a field strength is non-positive. Delegation supplies this
            guard; the previous inline arithmetic raised a bare ``ZeroDivisionError``
            at ``b0_source=0`` and emitted ``inf`` for a tensor field.
    """
    return (
        power_law_t1_transport(t1_source, b0_source, b0_target, beta),
        power_law_t2_transport(t2_source, b0_source, b0_target, gamma_t2),
    )


def ernst_magnetization(
    rho: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    tr: float,
    te: float,
    alpha: float,
    b1_plus: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute steady-state transverse magnetization (Ernst equation).

    Mxy = ρ · sin(α·B₁⁺) · (1 - E₁) / (1 - cos(α·B₁⁺)·E₁) · E₂

    Args:
        rho: Proton density map.
        t1: T₁ map [ms].
        t2: T₂ map [ms].
        tr: Repetition time [ms].
        te: Echo time [ms].
        alpha: Nominal flip angle [radians].
        b1_plus: B₁⁺ transmit efficiency map (1.0 = nominal). If None, uniform.

    Returns:
        Complex Mxy (real-valued magnitude, zero imaginary).
    """
    t1 = t1.clamp(min=1e-3)
    t2 = t2.clamp(min=1e-3)

    if b1_plus is not None:
        effective_alpha = alpha * b1_plus
    else:
        effective_alpha = torch.tensor(alpha, device=rho.device, dtype=rho.dtype)
    E1 = torch.exp(-tr / t1)
    E2 = torch.exp(-te / t2)

    sin_a = torch.sin(effective_alpha)
    cos_a = torch.cos(effective_alpha)

    numerator = rho * sin_a * (1.0 - E1)
    denominator = (1.0 - cos_a * E1).clamp(min=1e-8)

    mxy = (numerator / denominator) * E2
    return mxy


# ──────────────────────────────────────────────────────────────────
# Stage 2: GNL Warping
# ──────────────────────────────────────────────────────────────────


def gnl_warp(
    image: torch.Tensor,
    sh_coefficients: torch.Tensor | None = None,
    max_order: int = 3,
) -> torch.Tensor:
    """Warp image using spherical harmonic GNL displacement field.

    Uses polynomial basis from existing spherical_harmonics module.

    Args:
        image: Input [B, C, H, W].
        sh_coefficients: SH coefficients for displacement. If None, identity.
        max_order: Maximum SH order.

    Returns:
        Warped image [B, C, H, W].
    """
    if sh_coefficients is None:
        return image

    B, C, H, W = image.shape
    device = image.device

    # Build polynomial basis and evaluate displacement
    basis = _build_2d_polynomial_basis((H, W), max_order, device=device)  # [M, K]
    K = basis.shape[1]

    # sh_coefficients: [2, K] for (dx, dy) displacement
    if sh_coefficients.shape[-1] != K:
        raise ValueError(
            f"GNL SH coefficients last-dim {sh_coefficients.shape[-1]} != "
            f"polynomial basis size {K} for max_order={max_order}. "
            "Coefficients were provided deliberately, so a size mismatch is a "
            "config/wiring error; pass coefficients sized to the basis or set "
            "gnl_coefficients=None to disable GNL."
        )

    dx = (basis @ sh_coefficients[0]).reshape(H, W)  # displacement in x
    dy = (basis @ sh_coefficients[1]).reshape(H, W)  # displacement in y

    # Build sampling grid in [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )

    # Apply displacement (normalized to [-1, 1] range)
    grid_x_warped = (grid_x + dx).clamp(-1, 1)
    grid_y_warped = (grid_y + dy).clamp(-1, 1)

    grid = torch.stack([grid_x_warped, grid_y_warped], dim=-1)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


# ──────────────────────────────────────────────────────────────────
# Stage 5: ULF Noise
# ──────────────────────────────────────────────────────────────────


def inject_ulf_noise(
    kspace: torch.Tensor,
    time_vector: torch.Tensor,
    b0: float = 0.064,
    thermal_scale: float = 1e-5,
    emi_frequencies: tuple[float, ...] = (60.0, 120.0, 180.0),
    emi_amplitudes: tuple[float, ...] = (5e-3, 2e-3, 1e-3),
    enable_emi: bool = True,
) -> torch.Tensor:
    """Inject ULF-specific noise: thermal (B₀^{-7/4}) + structured EMI.

    Args:
        kspace: Complex k-space [B, C, K].
        time_vector: Readout timestamps [K].
        b0: Main field [T].
        thermal_scale: Thermal noise scaling factor.
        emi_frequencies: EMI frequencies in Hz (60Hz powerline + harmonics).
        emi_amplitudes: Amplitude per EMI component.
        enable_emi: Whether to add EMI.

    Returns:
        Noisy k-space.
    """
    # Thermal noise: σ ∝ 1/B₀^{7/4}
    sigma = thermal_scale / (b0**1.75)
    thermal = sigma * torch.complex(
        torch.randn_like(kspace.real),
        torch.randn_like(kspace.real),
    )

    result = kspace + thermal

    if enable_emi and len(emi_frequencies) > 0:
        t = time_vector.to(kspace.device)
        for freq, amp in zip(emi_frequencies, emi_amplitudes, strict=True):
            phase = torch.rand(1, device=kspace.device) * 2.0 * math.pi
            # Sinusoidal interference in time domain → localized k-space spike
            emi_signal = amp * torch.exp(1j * (2.0 * math.pi * freq * t + phase))
            # Broadcast to match kspace shape
            while emi_signal.ndim < kspace.ndim:
                emi_signal = emi_signal.unsqueeze(0)
            result = result + emi_signal

    return result


# ──────────────────────────────────────────────────────────────────
# Main Forward Operator
# ──────────────────────────────────────────────────────────────────


class DifferentiableULFForwardOperator(nn.Module):
    r"""Differentiable forward operator mapping 3T parameters to 64mT k-space.

    Composes existing codebase primitives:
    - BlochSimulator → Ernst-angle Mxy rendering
    - SphericalHarmonics → GNL displacement basis
    - B0MapSimulator → concomitant field generation
    - TimeSegmentedNUFFT → temporal integration
    - NoiseSimulator → thermal baseline

    Args:
        b0_target: Target field strength in Tesla. Default: 0.064.
        b0_source: Source field strength in Tesla. Default: 3.0.
        beta: Power-law T₁ dispersion exponent. Default: 0.35.
        gamma_t2: T₂ dispersion exponent. Default: 0.05.
        tr: Repetition time in ms. Default: 500.0.
        te: Echo time in ms. Default: 14.0.
        alpha: Nominal flip angle in radians. Default: π/2.
        num_time_segments: TS-NUFFT temporal bins. Default: 6.
        image_shape: Spatial dimensions (H, W). Default: (256, 256).
        gx: Readout gradient amplitude in mT/m. Default: 15.0.
        gy: Phase-encode gradient amplitude in mT/m. Default: 15.0.
        gz: Slice-select gradient amplitude in mT/m. Default: 0.0.
        slice_offset_m: Slice position from isocentre in metres. Default: 0.0. Every
            concomitant term needs ``gz`` or a slice offset, so with both at zero the
            Maxwell stage is exactly zero (an axial slice at isocentre).
        emi_frequencies: EMI offsets from the Larmor frequency in Hz; on the Cartesian
            path a tone at ``f`` lands ``f * T_read`` pixels from the readout centre.
        thermal_scale: Thermal noise scaling. Default: 1e-5.
        enable_emi: Whether to inject EMI. Default: True.
        enable_gnl: Whether to apply GNL warping. Default: True.
        enable_maxwell: Whether to apply concomitant phase. Default: True.
        enable_noise: Whether to inject noise. Default: True.
        thermal_drift_rate: B₀ drift rate in ppm/s. Default: 1.0.
    Mathematical Formulation:
    .. math::

        \mathcal{O}_{ULF}(x) = \mathcal{F}(x_{ULF} \odot S) + \mathcal{N}_{EMI}"""

    def __init__(
        self,
        b0_target: float = 0.064,
        b0_source: float = 3.0,
        beta: float = 0.35,
        gamma_t2: float = 0.05,
        tr: float = 500.0,
        te: float = 14.0,
        alpha: float = math.pi / 2,
        num_time_segments: int = 6,
        image_shape: tuple[int, int] = (256, 256),
        gx: float = 15.0,
        gy: float = 15.0,
        gz: float = 0.0,
        slice_offset_m: float = 0.0,
        thermal_scale: float = 1e-5,
        emi_frequencies: tuple[float, ...] = (60.0, 120.0, 180.0),
        emi_amplitudes: tuple[float, ...] = (5e-3, 2e-3, 1e-3),
        enable_emi: bool = True,
        enable_gnl: bool = True,
        enable_maxwell: bool = True,
        enable_noise: bool = True,
        thermal_drift_rate: float = 1.0,
    ) -> None:
        super().__init__()
        self.b0_target = b0_target
        self.b0_source = b0_source
        self.beta = beta
        self.gamma_t2 = gamma_t2
        self.tr = tr
        self.te = te
        self.alpha = alpha
        self.L = num_time_segments
        self.image_shape = image_shape
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.slice_offset_m = slice_offset_m
        self.thermal_scale = thermal_scale
        self.emi_frequencies = emi_frequencies
        self.emi_amplitudes = emi_amplitudes
        self.enable_emi = enable_emi
        self.enable_gnl = enable_gnl
        self.enable_maxwell = enable_maxwell
        self.enable_noise = enable_noise
        self.thermal_drift_rate = thermal_drift_rate
        self.gamma_hz = 42.576e6  # Hz/T

        # Lazy-init NUFFT operators (only when forward() is called with trajectory)
        self._nufft_ob = None

    def _ensure_nufft(self, device: torch.device) -> None:
        """Lazy-initialize NUFFT operator on first use."""
        if self._nufft_ob is not None:
            return
        if not _check_nufft():
            raise ImportError(
                "torchkbnufft is required for DifferentiableULFForwardOperator. "
                "Install with: pip install torchkbnufft"
            )
        import torchkbnufft as tkbn

        self._nufft_ob = tkbn.KbNufft(im_size=self.image_shape).to(device)

    def forward(
        self,
        rho: torch.Tensor,
        t1_3t: torch.Tensor,
        t2_3t: torch.Tensor,
        b1_plus: torch.Tensor | None = None,
        b0_map: torch.Tensor | None = None,
        trajectory: torch.Tensor | None = None,
        time_vector: torch.Tensor | None = None,
        gnl_coefficients: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the full 5-stage ULF forward model.

        Args:
            rho: Proton density [B, 1, H, W].
            t1_3t: T₁ at 3T [ms] [B, 1, H, W].
            t2_3t: T₂ at 3T [ms] [B, 1, H, W].
            b1_plus: B₁⁺ transmit map [B, 1, H, W]. None → uniform.
            b0_map: ΔB₀ off-resonance [Hz] [B, 1, H, W]. None → zero.
            trajectory: k-space trajectory [2, K] in [-π, π]. None → Cartesian FFT.
            time_vector: Readout timestamps [K] in seconds. None → uniform.
            gnl_coefficients: SH coefficients [2, K_sh]. None → no GNL.

        Returns:
            Synthetic 64mT k-space [B, 1, H, W] (Cartesian) or [B, 1, K] (NUFFT).

        forward method for DifferentiableULFForwardOperator.

        Executes PyTorch tensor operations.

        Args:
            rho (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t1_3t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t2_3t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            b1_plus (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            b0_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            trajectory (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            time_vector (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            gnl_coefficients (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        device = rho.device

        # ── Stage 1: power-law dispersion + Ernst Mxy ──
        t1_ulf, t2_ulf = power_law_dispersion(
            t1_3t,
            t2_3t,
            b0_source=self.b0_source,
            b0_target=self.b0_target,
            beta=self.beta,
            gamma_t2=self.gamma_t2,
        )

        mxy = ernst_magnetization(
            rho,
            t1_ulf,
            t2_ulf,
            tr=self.tr,
            te=self.te,
            alpha=self.alpha,
            b1_plus=b1_plus,
        )

        # ── Stage 2: GNL Warping ──
        if self.enable_gnl and gnl_coefficients is not None:
            mxy = gnl_warp(mxy, gnl_coefficients)

        # ── Stage 3: Maxwell + B₀ Phase ──
        if self.enable_maxwell:
            H, W = mxy.shape[-2], mxy.shape[-1]
            phi_c = B0MapSimulator.generate_concomitant_field(
                shape=(H, W),
                field_strength=self.b0_target,
                gx=self.gx,
                gy=self.gy,
                gz=self.gz,
                z_offset=self.slice_offset_m,
                device=device,
            )
            # φ = 2π · Δf · TE_seconds
            te_seconds = self.te * 1e-3
            phase_maxwell = 2.0 * math.pi * phi_c * te_seconds
            mxy = mxy * torch.exp(-1j * phase_maxwell.to(mxy.dtype))

        # Cartesian: the off-resonance phase at the echo. The TS-NUFFT path integrates
        # b0 over the readout time itself, so applying TE here too would count it twice.
        if b0_map is not None and trajectory is None:
            te_seconds = self.te * 1e-3
            phase_b0 = 2.0 * math.pi * b0_map * te_seconds
            if not torch.is_complex(mxy):
                mxy = mxy.to(torch.complex64)
            mxy = mxy * torch.exp(-1j * phase_b0)

        # Ensure complex for k-space operations
        if not torch.is_complex(mxy):
            mxy = mxy.to(torch.complex64)

        # ── Stage 4: Fourier Encoding ──
        if trajectory is not None:
            # Non-Cartesian: use TS-NUFFT
            kspace = self._forward_ts_nufft(mxy, b0_map, trajectory, time_vector)
        else:
            # Cartesian: centered ortho FFT via the SSOT. The previous inline
            # ``fftshift(fft2(ifftshift(x)))`` used the opposite shift convention
            # to ``fft2c`` (``ifftshift(fft2(fftshift(x)))``); they agree for even
            # dims but mis-center synthetic 64mT k-space on odd dims.
            kspace = fft2c(mxy)

        # ── Stage 5: ULF Noise ──
        if self.enable_noise:
            if kspace.ndim == 4:
                # Cartesian: each PE line is its own readout. Thermal noise is white; EMI
                # is a tone per readout row whose phase is not locked between lines.
                B, C, H, W = kspace.shape
                if time_vector is not None and time_vector.numel() != W:
                    raise ValueError(
                        f"Cartesian time_vector must hold the {W} readout timestamps; "
                        f"got {time_vector.numel()}."
                    )
                t_read = (
                    0.03
                    if time_vector is None
                    else float(time_vector[-1] - time_vector[0]) * W / max(W - 1, 1)
                )
                kspace = inject_ulf_noise(
                    kspace.reshape(B, C, -1),
                    torch.zeros(H * W, device=device),
                    b0=self.b0_target,
                    thermal_scale=self.thermal_scale,
                    enable_emi=False,
                ).reshape(B, C, H, W)
                if self.enable_emi and len(self.emi_frequencies) > 0:
                    freqs = torch.tensor(self.emi_frequencies, device=device)
                    amps = torch.tensor(self.emi_amplitudes, device=device)
                    kspace = kspace + readout_tone_field(
                        (B, C, H, W), freqs * t_read, amps, device=device
                    )
            else:
                if time_vector is None:
                    time_vector = torch.linspace(0, 0.03, kspace.shape[-1], device=device)
                kspace = inject_ulf_noise(
                    kspace,
                    time_vector,
                    b0=self.b0_target,
                    thermal_scale=self.thermal_scale,
                    emi_frequencies=self.emi_frequencies,
                    emi_amplitudes=self.emi_amplitudes,
                    enable_emi=self.enable_emi,
                )

        return kspace

    def _forward_ts_nufft(
        self,
        mxy: torch.Tensor,
        b0_map: torch.Tensor | None,
        trajectory: torch.Tensor,
        time_vector: torch.Tensor | None,
    ) -> torch.Tensor:
        """Time-segmented NUFFT with B₀ drift and interpolation.

        Delegates to torchkbnufft with L temporal bins.
        """
        self._ensure_nufft(mxy.device)
        device = mxy.device
        B = mxy.shape[0]
        K = trajectory.shape[-1]

        if time_vector is None:
            time_vector = torch.linspace(1e-3, 30e-3, K, device=device)

        tau_bins = torch.linspace(
            time_vector.min(),
            time_vector.max(),
            self.L,
            device=device,
        )

        kspace_total = torch.zeros(B, 1, K, dtype=torch.complex64, device=device)
        dt = (tau_bins[1] - tau_bins[0]).clamp(min=1e-8)

        for l in range(self.L):
            tau = tau_bins[l]

            # B₀ drift + off-resonance phase
            if b0_map is not None:
                # B0(t) = B0 (1 + r 1e-6 t), so the drift is gamma B0 r 1e-6 t in Hz.
                drift_hz = self.gamma_hz * self.b0_target * self.thermal_drift_rate * 1e-6 * tau
                drifted_b0 = b0_map + drift_hz
                phase = 2.0 * math.pi * drifted_b0 * tau
                m_shifted = mxy * torch.exp(-1j * phase)
            else:
                m_shifted = mxy

            # NUFFT forward
            traj_batch = (
                trajectory.unsqueeze(0).expand(B, -1, -1) if trajectory.ndim == 2 else trajectory
            )
            k_bin = self._nufft_ob(m_shifted, traj_batch)

            # Triangular interpolation weights
            w = torch.clamp(1.0 - torch.abs(time_vector - tau) / dt, min=0.0)
            w = w.view(1, 1, -1)
            kspace_total = kspace_total + w * k_bin

        return kspace_total


__all__ = [
    "DifferentiableULFForwardOperator",
    "ernst_magnetization",
    "gnl_warp",
    "inject_ulf_noise",
    "power_law_dispersion",
]
