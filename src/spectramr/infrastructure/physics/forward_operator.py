"""Differentiable Multi-Coil Forward Operator for MRI Reconstruction.

This module implements the forward MRI acquisition model with:
- Multi-coil sensitivity encoding (SENSE)
- Null-space and range-space projection operators
- Differentiable FFT operations

Mathematical Model:
    y = A·x = M·F·S·x

    where:
        x: Image to reconstruct (N×H×W complex)
        S: Coil sensitivity maps (C×H×W complex)
        F: 2D Fourier Transform
        M: Undersampling mask (H×W binary)
        y: Acquired k-space data (C×H×W complex)

    Null-space projection: P_N(A) = I - A†A
    Range-space projection: P_R = A†A

References:
    - [1] PA-PCFM: Projected Conditional Flow Matching
    - [30] ESPIRiT coil sensitivity estimation

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import (
    fft1_uncentered,
    fft1c,
    fft2_uncentered_last2,
    fft2c,
    ifft1_uncentered,
    ifft1c,
    ifft2_uncentered_last2,
    ifft2c,
)


class MultiCoilForwardOperator(nn.Module):
    r"""Differentiable multi-coil MRI forward operator.

    Implements y = M·F·P·S·x where:
        - S applies coil sensitivities
        - P applies B₀ / T₂* / gradient NL physics
        - F applies 2D FFT
        - M applies undersampling mask

    **B₀ Phase Model**

    Two modes are supported:

    1. **Static phase** (default): ``b0_map`` is treated as a phase error
       in radians applied at echo-top.  Equivalent to a per-voxel complex
       multiplication ``exp(i·φ)``.
    2. **Time-dependent phase** (Koolstra et al. 2021, Eq. 1) —
       **not implemented; rejected at construction.** The intended model
       accumulates per-readout-sample phase
       :math:`P(x, t_i) = \exp(-i\,2\pi\,\omega_{\text{eff}}(x)\,(t_i + t_{\text{shift}}))`
       with :math:`\omega_{\text{eff}} = \Delta B_0 + \Delta G_{\text{NL}}`,
       but the current hybrid-space forward and its adjoint are inconsistent
       (not an adjoint pair) and the phase tensor mis-broadcasts against the
       coil axis, so ``b0_units='hz'`` raises ``NotImplementedError``. Use
       ``b0_units='rad'`` (static phase at echo-top).

    **Gradient Nonlinearity Fusion**

    If ``gradient_nonlinearity_map`` is provided (in Hz), it is added to
    ``b0_map`` to form ``ω_eff``, the effective off-resonance map.

    Args:
        coil_sensitivities: Complex tensor (num_coils, H, W) of coil maps.
        mask: Binary mask (H, W) for undersampling pattern.
        centered: Whether FFT should be centered (shift DC to center).
        b0_map: Field inhomogeneity map (H, W).  Units depend on mode:
            *radians* for static, *Hz* for time-dependent.
        t2_star_map: T₂* relaxation map (H, W) — weighting factor.
        dt: Dwell time (time per sample) in seconds.
        t_shift: Readout gradient time-shift in seconds (Koolstra §2.2).
            Setting a non-zero value activates time-dependent B₀ phase.
        bw_hz: Readout bandwidth in Hz (reciprocal of dwell time).
        b0_units: ``'rad'`` (static) or ``'hz'`` (time-dependent).
        gradient_nonlinearity_map: Gradient NL contribution (H, W) in Hz.
            Added to ``b0_map`` to form :math:`\omega_{\text{eff}}`.

    Shape:
        - Input: (batch, 1, H, W) complex or (batch, 2, H, W) real
        - Output: (batch, num_coils, H, W) complex k-space
    """

    def __init__(
        self,
        coil_sensitivities: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        centered: bool = True,
        b0_map: torch.Tensor | None = None,
        t2_star_map: torch.Tensor | None = None,
        dt: float = 1e-5,
        t_shift: float = 0.0,
        bw_hz: float = 0.0,
        b0_units: str = "rad",
        gradient_nonlinearity_map: torch.Tensor | None = None,
    ) -> None:
        """Initialise the forward operator.

        Args:
            coil_sensitivities: Complex tensor (num_coils, H, W).
            mask: Binary mask (H, W) or (1, 1, H, W).
            centered: Whether to use centered FFT.
            b0_map: Field map (H, W) in rad or Hz.
            t2_star_map: T₂* weighting map (H, W).
            dt: Dwell time in seconds.
            t_shift: Readout time-shift in seconds.
            bw_hz: Readout bandwidth in Hz.
            b0_units: ``'rad'`` or ``'hz'``.
            gradient_nonlinearity_map: Gradient NL map (H, W) in Hz.
        """
        super().__init__()
        if b0_units not in ("rad", "hz"):
            raise ValueError(f"b0_units must be 'rad' or 'hz', got {b0_units!r}.")
        if b0_units == "hz":
            # The time-segmented (Koolstra) hz path is not implemented: its
            # hybrid-space forward and image-space adjoint are not an
            # adjoint pair (breaks CG/`normal()`), and the per-sample phase
            # tensor [1, H, W, N_ro] mis-broadcasts against coil_images
            # [B, C, H, W]. Fail loud (CLAUDE.md #9/#16) rather than silently
            # mis-compute; use b0_units='rad' (static phase at echo-top).
            raise NotImplementedError(
                "b0_units='hz' (time-segmented B0 phase) is not implemented; "
                "the hz forward/adjoint are inconsistent. Use b0_units='rad'."
            )
        self.centered = centered
        self.dt = dt
        self.t_shift = t_shift
        self.bw_hz = bw_hz
        self.b0_units = b0_units

        # Register as buffers (not parameters - not learned)
        self.register_buffer("coil_sensitivities", coil_sensitivities)
        self.register_buffer("mask", mask)
        self.register_buffer("b0_map", b0_map)
        self.register_buffer("t2_star_map", t2_star_map)
        self.register_buffer("gradient_nonlinearity_map", gradient_nonlinearity_map)

    def set_coil_sensitivities(self, coil_maps: torch.Tensor) -> None:
        """Update coil sensitivity maps.

        Args:
            coil_maps: Complex tensor (num_coils, H, W)
        """
        self.coil_sensitivities = coil_maps

    def set_mask(self, mask: torch.Tensor) -> None:
        """Update undersampling mask.

        Args:
            mask: Binary tensor (H, W) or (1, 1, H, W)
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        self.mask = mask

    def set_physics_maps(
        self,
        b0_map: torch.Tensor | None = None,
        t2_star_map: torch.Tensor | None = None,
        gradient_nonlinearity_map: torch.Tensor | None = None,
    ) -> None:
        """Update physics maps.

        Args:
            b0_map: (H, W) phase error map (rad) or field map (Hz)
            t2_star_map: (H, W) relaxation map
            gradient_nonlinearity_map: (H, W) gradient NL in Hz
        """
        self.b0_map = b0_map
        self.t2_star_map = t2_star_map
        if gradient_nonlinearity_map is not None:
            self.gradient_nonlinearity_map = gradient_nonlinearity_map

    def _effective_b0(self) -> torch.Tensor | None:
        """Compute effective off-resonance: ΔB₀ + ΔG_NL.

        Returns:
            Effective off-resonance map or None.
        """
        if self.b0_map is None:
            return self.gradient_nonlinearity_map
        if self.gradient_nonlinearity_map is None:
            return self.b0_map
        return self.b0_map + self.gradient_nonlinearity_map

    def _build_readout_time_vector(self, n_ro: int, device: torch.device) -> torch.Tensor:
        """Build the readout time vector centred on the echo.

        Args:
            n_ro: Number of readout samples.
            device: Target device.

        Returns:
            Time vector [1, 1, 1, N_ro] in seconds.
        """
        dwell = 1.0 / self.bw_hz if self.bw_hz > 0 else self.dt
        total = n_ro * dwell
        t_ro = torch.linspace(-total / 2, total / 2, n_ro, device=device)  # [N_ro]
        return t_ro.reshape(1, 1, 1, -1)  # broadcastable [1, 1, 1, N_ro]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward operator: y = M·F·P·S·x.

        Args:
            x: Image tensor (batch, 1, H, W) complex

        Returns:
            k-space data (batch, num_coils, H, W) complex

        forward method for MultiCoilForwardOperator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Ensure complex
        if not torch.is_complex(x):
            # Assume (batch, 2, H, W) real format
            x = torch.complex(x[:, 0], x[:, 1]).unsqueeze(1)

        # Apply coil sensitivities: S·x -> (batch, num_coils, H, W)
        if self.coil_sensitivities is not None:
            # Broadcast: (batch, 1, H, W) * (num_coils, H, W) -> (batch, num_coils, H, W)
            coil_images = x * self.coil_sensitivities.unsqueeze(0)
        else:
            coil_images = x

        # ── B₀ / T₂* Physics ──────────────────────────────────────
        omega_eff = self._effective_b0()  # ΔB₀ + optional grad NL

        # Note: b0_units == "hz" is rejected in __init__, so the hz branches
        # in this method (and in adjoint) are unreachable and kept only as a
        # record of the intended-but-broken time-segmented model.
        if omega_eff is not None and self.b0_units == "hz" and self.bw_hz > 0:
            # Time-dependent phase model (Koolstra Eq. 1):
            #   P(x, t_i) = exp(-i·2π·ω_eff(x)·(t_i + t_shift))
            # Applied per readout sample AFTER FFT (in hybrid space).
            pass  # Phase applied after FFT — see below
        elif omega_eff is not None:
            # Static phase model (legacy): b0_map in radians
            ones = torch.ones_like(omega_eff)
            phase_error = torch.polar(ones, omega_eff).unsqueeze(0).unsqueeze(1)
            coil_images = coil_images * phase_error

        if self.t2_star_map is not None:
            decay = self.t2_star_map.unsqueeze(0).unsqueeze(1)
            coil_images = coil_images * decay

        # ── FFT ───────────────────────────────────────────────────
        # Centered path routes through the fft_ops SSOT (centered, norm="ortho",
        # FP32-autocast-guarded against NaN gradients under AMP) — CLAUDE.md
        # non-negotiable #2. Identical to the previous manual transform on even
        # grids; on odd grids it now matches the SSOT shift convention used
        # everywhere else.
        kspace = fft2c(coil_images) if self.centered else fft2_uncentered_last2(coil_images)

        # ── Time-dependent B₀ phase (post-FFT, per readout sample) ─
        if omega_eff is not None and self.b0_units == "hz" and self.bw_hz > 0:
            N_ro = kspace.shape[-1]
            t_ro = self._build_readout_time_vector(N_ro, kspace.device)
            # ω_eff: [H, W] → [1, 1, H, 1]  (broadcast over readout)
            omega_2d = omega_eff.squeeze()
            if omega_2d.dim() == 1:
                omega_2d = omega_2d.unsqueeze(0)
            omega_4d = omega_2d.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
            if omega_4d.dim() == 3:
                omega_4d = omega_4d.unsqueeze(0)

            # Apply in hybrid space: FFT along PE (already done) but
            # modulate each readout column with time-dependent phase.
            # Transform to hybrid (y, t): IFFT along readout
            # ``ifft1c`` is ifftshift -> ifft -> fftshift along the readout
            # axis (fft_ops); the inline spelling this replaced used the two
            # shifts in the opposite order, which is a no-op on even lengths
            # and mis-centres an odd readout axis by one index.
            hybrid = ifft1c(kspace, dim=-1) if self.centered else ifft1_uncentered(kspace, dim=-1)

            # Modulate: hybrid * exp(-i·2π·ω_eff·(t + t_shift))
            phase = -2.0 * math.pi * omega_4d * (t_ro + self.t_shift)
            hybrid = hybrid * torch.exp(1j * phase)

            # Transform back to k-space: FFT along readout
            kspace = fft1c(hybrid, dim=-1) if self.centered else fft1_uncentered(hybrid, dim=-1)

        # Apply undersampling mask
        if self.mask is not None:
            kspace = kspace * self.mask

        return kspace

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint operator: x = A†y = S†·F†·M†·y.

        Args:
            y: k-space data (batch, num_coils, H, W) complex

        Returns:
            Image (batch, 1, H, W) complex
        """
        # Apply mask (self-adjoint)
        if self.mask is not None:
            y = y * self.mask

        # Apply inverse FFT. Centered path uses the ifft2c SSOT (the true
        # adjoint of fft2c used in forward); identical on even grids, SSOT-
        # consistent on odd grids, and FP32-autocast-guarded.
        coil_images = ifft2c(y) if self.centered else ifft2_uncentered_last2(y)

        # ── Adjoint B₀ Phase ──────────────────────────────────────
        omega_eff = self._effective_b0()

        if omega_eff is not None and self.b0_units == "hz" and self.bw_hz > 0:
            # Time-dependent adjoint: conjugate phase per readout sample
            N_ro = coil_images.shape[-1]
            t_ro = self._build_readout_time_vector(N_ro, coil_images.device)
            omega_2d = omega_eff.squeeze()
            if omega_2d.dim() == 1:
                omega_2d = omega_2d.unsqueeze(0)
            omega_4d = omega_2d.unsqueeze(0).unsqueeze(-1)
            if omega_4d.dim() == 3:
                omega_4d = omega_4d.unsqueeze(0)

            # Conjugate phase: +i·2π·ω_eff·(t + t_shift)
            phase = 2.0 * math.pi * omega_4d * (t_ro + self.t_shift)
            coil_images = coil_images * torch.exp(1j * phase)
        elif omega_eff is not None:
            # Static adjoint: conjugate phase exp(-i·φ)
            ones = torch.ones_like(omega_eff)
            phase_conj = torch.polar(ones, -1.0 * omega_eff).unsqueeze(0).unsqueeze(1)
            coil_images = coil_images * phase_conj

        if self.t2_star_map is not None:
            # Decay is real and self-adjoint (diagonal scaling)
            decay = self.t2_star_map.unsqueeze(0).unsqueeze(1)
            coil_images = coil_images * decay

        # Combine coils using conjugate sensitivities
        if self.coil_sensitivities is not None:
            # S†: multiply by conjugate and sum over coils
            sens_conj = self.coil_sensitivities.conj().unsqueeze(0)
            combined = (coil_images * sens_conj).sum(dim=1, keepdim=True)
        else:
            combined = coil_images.sum(dim=1, keepdim=True)

        return combined

    def normal(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normal operator: A†Ax.

        This is the range-space projection when mask is applied.

        Args:
            x: Image tensor (batch, 1, H, W) complex

        Returns:
            Projected image (batch, 1, H, W) complex
        """
        return self.adjoint(self.forward(x))

    def simulate_measurement(
        self,
        x: torch.Tensor,
        snr: float = 30.0,
    ) -> torch.Tensor:
        """Simulate full acquisition with noise.

        Args:
            x: Ground truth image (batch, 1, H, W)
            snr: Signal-to-Noise Ratio in dB

        Returns:
            Noisy k-space (batch, num_coils, H, W)
        """
        # 1. Forward model (Physics: S -> B0 -> FFT -> Mask)
        y_clean = self.forward(x)

        # 2. Add Thermal Noise (Complex Gaussian in k-space)
        # P_signal = mean(|y|^2) or peak?
        # Typically SNR is defined relative to signal peak or mean power.
        # We use P95 of magnitude.

        # Calculate signal amplitude reference
        y_mag = torch.abs(y_clean)
        batch_size = x.shape[0]

        # Per-sample scaling
        sig_level = torch.quantile(y_mag.flatten(1), 0.95, dim=1).view(batch_size, 1, 1, 1)

        # SNR formula: SNR_dB = 20 * log10(A_signal / sigma_noise)
        # sigma_noise = A_signal / 10^(SNR/20)
        snr_linear = 10.0 ** (snr / 20.0)
        sigma = sig_level / (snr_linear + 1e-8)

        # Complex Gaussian Noise n ~ CN(0, sigma^2)
        # Real/Imag parts each have std = sigma/sqrt(2)
        noise = torch.randn_like(y_clean) + 1j * torch.randn_like(y_clean)
        noise = noise * (sigma / math.sqrt(2))

        return y_clean + noise


class NullSpaceProjection(nn.Module):
    """Null-space projection operator: P_N = I - A†A.

    Projects input onto the null space of the forward operator.
    This is the space of information NOT constrained by measurements.

    For MRI: The null space contains unmeasured k-space frequencies.

    Mathematical Property:
        x = x_measured + x_null
        x_measured = A†A·x  (range space)
        x_null = (I - A†A)·x  (null space)

    Used in PA-PCFM to ensure network only generates null-space content.
    """

    def __init__(self, forward_operator: MultiCoilForwardOperator) -> None:
        """__init__.

        Args:
            forward_operator (MultiCoilForwardOperator): Description.
        """
        super().__init__()
        self.A = forward_operator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto null space: (I - A†A)x.

        Args:
            x: Image tensor (batch, 1, H, W) complex

        Returns:
            Null-space component (batch, 1, H, W) complex

        forward method for NullSpaceProjection.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Note: Ideally this would be x - A_pinv(A(x)), but A†A is used as approximation
        # for computational efficiency in PA-PCFM.
        return x - self.A.normal(x)

    def range_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto range space: A†Ax.

        Args:
            x: Image tensor (batch, 1, H, W) complex

        Returns:
            Range-space component (batch, 1, H, W) complex
        """
        return self.A.normal(x)


class MultiCoilDataConsistency(nn.Module):
    """Data consistency layer for iterative reconstruction.

    Enforces that reconstruction is consistent with acquired k-space:
        x_dc = x - λ·A†(Ax - y)

    Or hard replacement in k-space:
        x_dc = F†[M·y + (1-M)·Fx]

    Args:
        forward_operator: Multi-coil forward operator
        mode: 'soft' for gradient step, 'hard' for k-space replacement
        lambda_dc: Weight for soft data consistency (default 1.0)
    """

    def __init__(
        self,
        forward_operator: MultiCoilForwardOperator,
        mode: Literal["soft", "hard"] = "soft",
        lambda_dc: float = 1.0,
    ) -> None:
        """__init__.

        Args:
            forward_operator (MultiCoilForwardOperator): Description.
            mode (Literal['soft', 'hard']): Description.
            lambda_dc (float): Description.
        """
        super().__init__()
        self.A = forward_operator
        self.mode = mode
        self.lambda_dc = lambda_dc

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_residual: bool = False,
    ) -> torch.Tensor:
        """Apply data consistency.

        Args:
            x: Current reconstruction (batch, 1, H, W) complex
            y: Acquired k-space (batch, num_coils, H, W) complex
            return_residual: If True, also return ||Ax - y||

        Returns:
            Data-consistent reconstruction (batch, 1, H, W) complex

        forward method for MultiCoilDataConsistency.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            y (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            return_residual (bool): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.mode == "soft":
            # Gradient step: x - λ·A†(Ax - y)
            residual = self.A.forward(x) - y
            correction = self.A.adjoint(residual)
            x_dc = x - self.lambda_dc * correction
        else:
            # Hard replacement in k-space
            # Get current k-space
            kspace_pred = self.A.forward(x)

            # Replace measured locations with acquired data
            if self.A.mask is not None:
                kspace_dc = y * self.A.mask + kspace_pred * (1 - self.A.mask)
            else:
                kspace_dc = y

            # Transform back
            x_dc = self.A.adjoint(kspace_dc)

        if return_residual:
            residual_norm = torch.norm(self.A.forward(x_dc) - y)
            return x_dc, residual_norm

        return x_dc


def create_radial_mask(
    height: int,
    width: int,
    num_spokes: int,
    golden_angle: bool = True,
) -> torch.Tensor:
    """Create a radial undersampling mask.

    Args:
        height: Image height
        width: Image width
        num_spokes: Number of radial spokes
        golden_angle: If True, use golden angle spacing

    Returns:
        Binary mask (height, width)
    """
    mask = torch.zeros(height, width)
    center_h, center_w = height // 2, width // 2

    # Golden angle in radians. Use the radial golden angle π(√5−1)/2 ≈ 111.25°
    # (matching radial_golden_angle / trajectories.py); the previous
    # π(3−√5) ≈ 137.5° is the phyllotaxis angle and contradicted the "~111.25°"
    # comment and the rest of the repo's radial generators.
    if golden_angle:
        angle_increment = torch.pi * (torch.sqrt(torch.tensor(5.0)) - 1.0) / 2.0  # ~111.25°
    else:
        angle_increment = torch.pi / num_spokes

    for i in range(num_spokes):
        angle = i * angle_increment

        # Compute spoke coordinates using parametric line
        t = torch.linspace(-1, 1, max(height, width))
        x = (center_w + t * width * 0.5 * torch.cos(angle)).long()
        y = (center_h + t * height * 0.5 * torch.sin(angle)).long()

        # Clip to valid indices
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        mask[y[valid], x[valid]] = 1.0

    return mask


def create_cartesian_mask(
    height: int,
    width: int,
    acceleration: int = 4,
    center_fraction: float = 0.08,
    random_seed: int | None = None,
) -> torch.Tensor:
    """Create a Cartesian undersampling mask.

    Args:
        height: Image height (phase encoding)
        width: Image width (frequency encoding - fully sampled)
        acceleration: Undersampling factor
        center_fraction: Fraction of center lines to fully sample
        random_seed: Seed for reproducibility

    Returns:
        Binary mask (height, width)
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Always include center lines
    center_lines = int(height * center_fraction)
    center_start = height // 2 - center_lines // 2
    center_end = center_start + center_lines

    # Random selection for outer lines
    num_outer = (height - center_lines) // acceleration
    outer_indices = torch.randperm(height - center_lines)[:num_outer]

    # Adjust indices to skip center
    outer_indices = torch.where(
        outer_indices >= center_start,
        outer_indices + center_lines,
        outer_indices,
    )

    # Create mask
    mask = torch.zeros(height, width)
    mask[center_start:center_end, :] = 1.0  # Center lines

    for idx in outer_indices:
        if idx < height:
            mask[idx, :] = 1.0

    return mask


__all__ = [
    "MultiCoilDataConsistency",
    "MultiCoilForwardOperator",
    "NullSpaceProjection",
    "create_cartesian_mask",
    "create_radial_mask",
]
