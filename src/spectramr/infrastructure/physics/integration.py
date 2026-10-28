"""Advanced Physics and Data Consistency Integration for MRI Reconstruction

This module provides physics-based components for MRI reconstruction including:
- Forward operator composition and factory functions
- Data consistency layers
- Coil sensitivity maps
- Bloch equation solvers
"""

import logging

import torch
from torch import nn

from spectramr.core.types import CoilSens, Image, KSpace, Mask, Tensor
from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.infrastructure.physics.implementations import (  # noqa: F401  (import for register_operator side effects: fft2d/nufft must be in the registry)
    FFT2DOperator,
    NUFFTOperator,
)
from spectramr.infrastructure.physics.registry import IForwardOperator, create_operator
from spectramr.shared.utils.safe_io import safe_torch_load

try:
    import sigpy as sp

    SIGPY_AVAILABLE = True
except ImportError:
    SIGPY_AVAILABLE = False
    sp = None

try:
    import torchkbnufft as tkbn

    TORCHKBNUFFT_AVAILABLE = True
except ImportError:
    TORCHKBNUFFT_AVAILABLE = False
    tkbn = None

logger = logging.getLogger(__name__)


# Import from registry and implementations


class OperatorProjectionDataConsistency(nn.Module):
    """Data consistency layer for reconstruction networks."""

    def __init__(self, operator: IForwardOperator, lambda_dc: float = 1.0):
        """__init__.

        Args:
            operator (IForwardOperator): Description.
            lambda_dc (float): Description.
        """
        super().__init__()
        self.operator = operator
        self.lambda_dc = lambda_dc

    def forward(
        self,
        pred_image: Image,
        ref_kspace: KSpace,
        mask: Mask,
        coil_sens: CoilSens | None = None,
    ) -> Image:
        """Apply data consistency.

        Args:
            pred_image: Predicted image [B, H, W, 2] (complex)
            ref_kspace: Reference k-space [B, C, H, W, 2] (complex)
            mask: Sampling mask [B, H, W]
            coil_sens: Coil sensitivity maps [B, C, H, W, 2] (optional)

        Returns:
            Data-consistent image [B, H, W, 2]

        forward method for OperatorProjectionDataConsistency.

        Executes PyTorch tensor operations.

        Args:
            pred_image (Image): Expected input tensor.
            ref_kspace (KSpace): Expected input tensor.
            mask (Mask): Expected input tensor.
            coil_sens (CoilSens | None): Expected input tensor.

        Returns:
            Image: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Forward model: pred -> k-space
        pred_kspace = self.operator.forward(pred_image, coil_sens)

        # Replace sampled k-space points with reference
        # Robust mask expansion for [B, H, W, 2] (4D complex-last) or [B, C, H, W] (4D channel-first)
        if pred_kspace.ndim == 4:
            # Heuristic: Check last dim. If 2, likely complex-last [B, H, W, 2]
            # (Unless W=2, which is rare for MRI)
            if pred_kspace.shape[-1] == 2 and pred_kspace.shape[1] != 2:
                mask_expanded = mask.unsqueeze(-1)  # [B, H, W, 1]
            else:
                mask_expanded = mask.unsqueeze(1)  # [B, 1, H, W]
        else:
            mask_expanded = mask.unsqueeze(1)  # [B, 1, H, W] or 5D compatibility

        dc_kspace = pred_kspace * (1 - mask_expanded) + ref_kspace * mask_expanded

        # Adjoint: k-space -> image
        dc_image = self.operator.adjoint(dc_kspace, coil_sens)

        # Weighted combination
        output = (1 - self.lambda_dc) * pred_image + self.lambda_dc * dc_image

        return output


class CoilSensitivityEstimator(nn.Module):
    """Coil sensitivity map estimation using ESPIRiT."""

    def __init__(
        self,
        calib_size: int = 24,
        thresh: float = 0.02,
        kernel_size: int = 6,
    ):
        """__init__.

        Args:
            calib_size (int): Description.
            thresh (float): Description.
            kernel_size (int): Description.
        """
        super().__init__()
        self.calib_size = calib_size
        self.thresh = thresh
        self.kernel_size = kernel_size

    def estimate_espirit(
        self, kspace: KSpace, kernel_size: int = 6, threshold: float = 0.02
    ) -> CoilSens:
        """ESPIRiT-based sensitivity estimation."""
        try:
            import sigpy.mri as mr

            # Convert to numpy
            kspace_np = kspace.detach().cpu().numpy()

            # ESPIRiT
            sens_np = mr.app.EspiritCalib(
                kspace_np, calib_width=kernel_size, thresh=threshold
            ).run()

            # Convert back to torch
            return torch.from_numpy(sens_np).to(kspace.device)
        except ImportError as exc:
            # No silent fallbacks (pitfall #9): an explicit "espirit" request
            # must surface the missing optional dependency rather than silently
            # substituting a different algorithm. The forward(method="auto")
            # path is the only place that may downgrade to JSENSE — it catches
            # this ImportError itself.
            raise ImportError(
                "sigpy is required for ESPIRiT coil-sensitivity estimation; "
                'install sigpy or use method="jsense" (or method="auto" for '
                "an automatic JSENSE fallback)."
            ) from exc

    def estimate_jsense(
        self, kspace: KSpace, num_iterations: int = 10, reg_param: float = 0.01
    ) -> CoilSens:
        """JSENSE-inspired joint sensitivity and image estimation.

        Robust alternative to ESPIRiT that doesn't require external dependencies.

        Args:
            kspace: Multi-coil k-space [B, C, H, W, 2]
            num_iterations: Number of alternating optimization iterations
            reg_param: Regularization parameter for Tikhonov

        Returns:
            Estimated sensitivit maps [B, C, H, W, 2]
        """
        B, C, H, W, _ = kspace.shape
        device = kspace.device
        dtype = kspace.dtype

        # Initial image estimate (centered IFFT — SSOT, CLAUDE.md pitfall #2).
        # ``ifft2c`` accepts the framework-canonical [..., H, W, 2] real layout
        # via ``_to_complex`` and returns a complex tensor [..., H, W].
        images = ifft2c(kspace)
        rss_image = torch.sqrt(torch.sum(torch.abs(images) ** 2, dim=1, keepdim=True) + 1e-8)

        # Initialize sensitivities (normalized by RSS)
        sens_complex = images / (rss_image + 1e-8)

        # Alternating optimization
        for _ in range(num_iterations):
            # Update image given sensitivities
            weighted_images = sens_complex.conj() * images
            image_estimate = weighted_images.sum(dim=1, keepdim=True)
            denom = torch.sum(torch.abs(sens_complex) ** 2, dim=1, keepdim=True) + reg_param
            image_estimate = image_estimate / denom

            # Update sensitivities given image
            numerator = images * image_estimate.conj()
            denominator = torch.abs(image_estimate) ** 2 + reg_param
            sens_complex = numerator / denominator

        # Convert back to real/imag
        sens_real_imag = torch.view_as_real(sens_complex)
        return sens_real_imag

    def estimate_rss(self, kspace: KSpace) -> CoilSens:
        """Root sum of squares sensitivity estimation."""
        image = ifft2c(kspace)
        rss = torch.sqrt(torch.sum(torch.abs(image) ** 2, dim=1, keepdim=True) + 1e-8)
        sens = image / (rss + 1e-8)
        return torch.view_as_real(sens)

    def forward(self, kspace: KSpace, method: str = "auto") -> CoilSens:
        """Estimate coil sensitivities.

        Args:
            kspace: Multi-coil k-space [B, C, H, W, 2]
            method: "espirit", "jsense", "rss", or "auto"

        Returns:
            Estimated sensitivity maps [B, C, H, W, 2]

        forward method for CoilSensitivityEstimator.

        Executes PyTorch tensor operations.

        Args:
            kspace (KSpace): Expected input tensor.
            method (str): Expected input tensor.

        Returns:
            CoilSens: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if method == "auto":
            # "auto" may downgrade to JSENSE ONLY when the optional sigpy
            # dependency is missing (ImportError). Any OTHER failure (NaN,
            # shape, CUDA OOM, sigpy bug) is a genuine error and must propagate
            # — never silently degrade to a lower-quality estimate on a real
            # failure (CLAUDE.md pitfall #9). The previous catch-all
            # `except (ImportError, Exception)` swallowed every error.
            try:
                return self.estimate_espirit(kspace)
            except ImportError:
                return self.estimate_jsense(kspace)
        elif method == "espirit":
            return self.estimate_espirit(kspace)
        elif method == "jsense":
            return self.estimate_jsense(kspace)
        elif method == "rss":
            return self.estimate_rss(kspace)
        else:
            raise ValueError(f"Unknown method: {method}")


class BlochEquationSolver(nn.Module):
    """High-fidelity Bloch equation solver with rigorous physics modeling.

    Includes:
    - B0 inhomogeneity with vectorized multi-isochromat T2* dephasing
    - Exact Stejskal-Tanner diffusion from gradient waveforms
    - Full 3D magnetization vector tracking (T1 recovery, T2 decay)
    """

    def __init__(
        self,
        t1: float = 1000.0,
        t2: float = 100.0,
        tr: float = 500.0,
        te: float = 10.0,
        b0: float = 3.0,  # Field strength in Tesla
        diffusion_coeff: float = 0.0,  # Diffusion coefficient (mm^2/s)
        voxel_size: float = 1.0,  # Nominal voxel size in mm
    ):
        """__init__.

        Args:
            t1 (float): Description.
            t2 (float): Description.
            tr (float): Description.
            te (float): Description.
            b0 (float): Description.
            diffusion_coeff (float): Description.
            voxel_size (float): Description.
        """
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.tr = tr
        self.te = te
        self.b0 = b0
        self.diffusion_coeff = diffusion_coeff
        self.gamma = 42.58  # MHz/T
        self.voxel_size = voxel_size

    def compute_diffusion_attenuation(
        self,
        gradient: torch.Tensor,
        te: float,
    ) -> torch.Tensor:
        """Compute diffusion attenuation using the Stejskal-Tanner equation.

        Assumes a standard Pulsed Gradient Spin Echo (PGSE) sequence where:
        - delta (gradient duration) = te / 3
        - Delta (gradient separation) = te / 2

        Args:
            gradient: Gradient vector [3] (Gx, Gy, Gz) in T/m
            te: Echo time in seconds

        Returns:
            Attenuation factor exp(-b * D)
        """
        # Analytical solution for PGSE
        # b = (gamma * G * delta)^2 * (Delta - delta/3)
        delta = te / 3.0
        Delta = te / 2.0

        gamma_rad = 2 * torch.pi * self.gamma * 1e6  # rad/s/T
        G_mag = torch.norm(gradient)  # T/m

        b_val_si = (gamma_rad * G_mag * delta) ** 2 * (Delta - delta / 3.0)  # s/m^2
        b_val_mm = b_val_si * 1e-6  # s/mm^2

        return torch.exp(-b_val_mm * self.diffusion_coeff)

    def evolve_phase_multi_isochromat(
        self,
        b0_map: Image | None,
        te: float,
        n_isochromats: int = 50,
    ) -> Tensor:
        """Simulate intra-voxel T2* dephasing using vectorized Monte Carlo integration.

        Args:
            b0_map: B0 off-resonance map in Tesla (or Hz if scaled) [B, H, W]
                    If None, assumes perfect homogeneity (no T2* decay beyond T2).
            te: Echo time in seconds.
            n_isochromats: Number of intra-voxel spins.

        Returns:
            Complex decay factor [B, H, W]
        """
        if b0_map is None:
            # Perfect homogeneity: unity phasor. Return a 0-dim (scalar) tensor with
            # no device pin so it broadcasts onto a CUDA `m_complex` in forward()
            # without a cross-device RuntimeError. (A 1-elem CPU tensor does NOT
            # scalar-broadcast and raises on GPU; the removed self.b0_device attr was
            # never assigned anywhere in the class, so the hasattr guard was always
            # False and the device was always "cpu".)
            return torch.ones(())

        device = b0_map.device
        B, H, W = b0_map.shape
        gamma_hz = self.gamma * 1e6

        # Calculate field gradients (spatial derivatives of B0 map)
        # Gradient in Hz/mm (since voxel_size is mm)
        # Using central finite differences
        grad_y = torch.zeros_like(b0_map)
        grad_x = torch.zeros_like(b0_map)

        # Scale B0 map to Hz if it's in Tesla?
        # Assuming b0_map is already in Hz for legacy consistency, or we strictly define it.
        # integration.py docstring says "B0 map in Tesla relative to 0 or including main field".
        # But previous code used self.gamma * 1e6 * full_b0.
        # CAUTION: Metric definitions usually provide B0 map in Hz directly.
        # Let's assume input is in Tesla for physics consistency with gamma.

        # Central difference: (f(x+1) - f(x-1)) / 2
        # Units: Tesla/pixel. Convert to Tesla/meter?
        # voxel_size is in mm.
        pixel_spacing_m = self.voxel_size * 1e-3

        grad_y[:, 1:-1, :] = (b0_map[:, 2:, :] - b0_map[:, :-2, :]) / (2 * pixel_spacing_m)
        grad_x[:, :, 1:-1] = (b0_map[:, :, 2:] - b0_map[:, :, :-2]) / (2 * pixel_spacing_m)

        # Generate random offsets within voxel [-0.5*v, 0.5*v]
        # Shape: [N_iso, 2] -> [1, 1, 1, N_iso, 2]
        offsets = (torch.rand(n_isochromats, 2, device=device) - 0.5) * pixel_spacing_m
        dy = offsets[:, 0].view(1, 1, 1, n_isochromats)
        dx = offsets[:, 1].view(1, 1, 1, n_isochromats)

        # Expand gradients for isochromats: [B, H, W, N_iso]
        gy_expanded = grad_y.unsqueeze(-1)
        gx_expanded = grad_x.unsqueeze(-1)
        b0_expanded = b0_map.unsqueeze(-1)

        # Local field at each isochromat: B(r) = B0(center) + G \cdot r
        # We model linear gradients within voxel.
        local_b = b0_expanded + (gy_expanded * dy + gx_expanded * dx)

        # Phase evolution: phi = gamma * B * TE
        # gamma is MHz/T * 1e6 = Hz/T. * 2pi for radians.
        omega = 2 * torch.pi * gamma_hz * local_b
        phase = omega * te

        # Sum isochromats: (1/N) * sum(exp(i*phi))
        phasors = torch.exp(1j * phase)
        signal_decay = phasors.mean(dim=-1)  # Average over isochromats

        return signal_decay

    def compute_exact_diffusion(
        self,
        gradient_waveform: Tensor,
        dt: float,
    ) -> Tensor:
        """Compute exact diffusion attenuation for arbitrary gradient waveforms.

        Uses the exact integral: b = gamma^2 * integral(|k(t)|^2) dt
        where k(t) = integral(G(tau)) dtau (from t to TE).

        Args:
            gradient_waveform: [T, 3] or [B, T, 3] in T/m
            dt: Dwell time per point in seconds

        Returns:
            Attenuation factor exp(-b * D) in range [0, 1]
        """
        # Ensure waveform is [B, T, 3]
        if gradient_waveform.ndim == 2:
            gradient_waveform = gradient_waveform.unsqueeze(0)

        gamma_rad = 2 * torch.pi * self.gamma * 1e6  # rad/s/T

        # Calculate k(t) trajectory: integral of gradient from 0 to t
        # Note: Standard formulation usually defines k(t) = gamma * integral_0^t G(tau).
        # But 'b' value definition varies.
        # Standard: b = integral_0^TE |k(t)|^2 dt ?
        # Actually: k(t) is spatial frequency weighting. The attenuation is due to phase distribution variance.
        # b = \int_0^{TE} || k(t) ||^2 dt  where k(t) = \gamma \int_0^t G(\tau) d\tau

        # Cumulative sum for integral G
        k_traj = gamma_rad * torch.cumsum(gradient_waveform, dim=1) * dt

        # Magnitude squared of k-vector
        k_sq = torch.sum(k_traj**2, dim=-1)  # [B, T]

        # Integrate |k(t)|^2 dt -> b-value
        b_val_sim = torch.sum(k_sq, dim=-1) * dt  # s/m^2

        # Convert s/m^2 to s/mm^2 for D (which is mm^2/s)
        # 1 s/m^2 = 1e-6 s/mm^2
        b_val_mm = b_val_sim * 1e-6

        return torch.exp(-b_val_mm * self.diffusion_coeff)

    def update_bloch_vector(
        self,
        m_current: Tensor,
        duration: float,
    ) -> Tensor:
        """Rigorous 3D Bloch relaxation step.

        Evolves magnetization vector M over time `duration` accounting for
        T1 recovery (Mz -> 1) and T2 decay (Mxy -> 0).

        Args:
            m_current: [..., 3] (Mx, My, Mz)
            duration: Time step in seconds
        """
        mx, my, mz = m_current[..., 0], m_current[..., 1], m_current[..., 2]

        e1 = torch.exp(torch.tensor(-duration / self.t1, device=m_current.device))
        e2 = torch.exp(torch.tensor(-duration / self.t2, device=m_current.device))

        # Mz recovery: Mz(t) = Mz(0)E1 + M0(1-E1) (Normalized M0=1)
        mz_new = mz * e1 + (1.0 - e1)

        # Transverse decay
        mx_new = mx * e2
        my_new = my * e2

        return torch.stack([mx_new, my_new, mz_new], dim=-1)

    def forward(
        self,
        m0: Tensor,
        flip_angle: Tensor | None = None,
        gradient_waveform: Tensor | None = None,
        dt_waveform: float = 1e-3,
        b0_map: Image | None = None,
    ) -> Image:
        """Solve Bloch-Torrey equations for signal prediction at TE.

        Args:
            m0: Initial magnetization [B, H, W, 3]
            flip_angle: Excitation angle [B, H, W]
            gradient_waveform: Diffusion gradients [B, T, 3] (optional)
            dt_waveform: Time step for gradient integration (s)

        Returns:
            Signal [B, H, W, 2] (real, imag)

        forward method for BlochEquationSolver.

        Executes PyTorch tensor operations.

        Args:
            m0 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            flip_angle (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            gradient_waveform (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            dt_waveform (float): Expected input tensor.
            b0_map (Image | None): Expected input tensor.

        Returns:
            Image: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if flip_angle is None:
            flip_angle = torch.full_like(m0[..., 0], torch.pi / 2)

        # 1. RF Excitation (Instantaneous)
        cos_fa = torch.cos(flip_angle)
        sin_fa = torch.sin(flip_angle)
        mx0, my0, mz0 = m0[..., 0], m0[..., 1], m0[..., 2]

        # Mx' = Mx
        # My' = My*cos - Mz*sin
        # Mz' = My*sin + Mz*cos
        mx_post = mx0
        my_post = my0 * cos_fa - mz0 * sin_fa
        mz_post = my0 * sin_fa + mz0 * cos_fa

        # 2. Transverse Evolution to TE
        # T2 decay
        e2_te = torch.exp(torch.tensor(-self.te / self.t2, device=m0.device))
        mx_te = mx_post * e2_te
        my_te = my_post * e2_te

        # T2* Multi-Isochromat Dephasing
        phasor = self.evolve_phase_multi_isochromat(b0_map, self.te)

        m_complex = torch.complex(mx_te, my_te) * phasor

        # 3. Exact Diffusion Attenuation
        if gradient_waveform is not None and self.diffusion_coeff > 0:
            diff_factor = self.compute_exact_diffusion(gradient_waveform, dt_waveform)
            # Expand diff_factor dimensions to match image [B, 1, 1] if strictly global gradients
            if diff_factor.dim() == 1:
                diff_factor = diff_factor.view(-1, 1, 1)
            m_complex = m_complex * diff_factor

        return torch.stack([m_complex.real, m_complex.imag], dim=-1)

    def simulate_sequence(
        self,
        m0: torch.Tensor,
        sequence_params: dict[str, torch.Tensor],
        b0_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Simulate a complete pulse sequence (TR loop)."""
        flip_angles = sequence_params.get("flip_angles")
        gradients = sequence_params.get("gradient_waveforms")  # [N_pulses, T, 3]
        dt = sequence_params.get("dt", 1e-3)

        num_pulses = flip_angles.shape[0] if flip_angles is not None else 1
        signals = []
        m_curr = m0.clone()

        for i in range(num_pulses):
            fa = flip_angles[i] if flip_angles is not None else None
            grad = gradients[i] if gradients is not None else None

            # Predict signal at TE
            signal = self.forward(m_curr, fa, grad, dt, b0_map)
            signals.append(signal)

            # Evolve system state:
            # 1. Apply RF to M_curr
            cos = torch.cos(fa) if fa is not None else 1.0
            sin = torch.sin(fa) if fa is not None else 0.0
            mx, my, mz = m_curr[..., 0], m_curr[..., 1], m_curr[..., 2]

            my_rot = my * cos - mz * sin
            mz_rot = my * sin + mz * cos
            mx_rot = mx

            m_post_rf = torch.stack([mx_rot, my_rot, mz_rot], dim=-1)

            # 2. Relax for full TR period
            # (Approximation: We ignore gradient-induced dephasing persistence across TRs
            #  aka usually gradients serve as crushers or we are in spoiled gradient echo)
            m_curr = self.update_bloch_vector(m_post_rf, self.tr)

            # Perfect spoiling assumption: Transverse mag destroyed before next pulse
            # m_curr[..., 0] = 0
            # m_curr[..., 1] = 0

        return torch.stack(signals, dim=0)


class MRIReconstructionPipeline(nn.Module):
    """Complete MRI reconstruction pipeline with physics integration."""

    def __init__(
        self,
        operator: IForwardOperator,
        dc_layer: OperatorProjectionDataConsistency,
        coil_estimator: CoilSensitivityEstimator | None = None,
    ):
        """__init__.

        Args:
            operator (IForwardOperator): Description.
            dc_layer (OperatorProjectionDataConsistency): Description.
            coil_estimator (Optional[CoilSensitivityEstimator]): Description.
        """
        super().__init__()
        self.operator = operator
        self.dc_layer = dc_layer
        self.coil_estimator = coil_estimator

    def forward(
        self,
        kspace: KSpace,
        mask: Mask,
        initial_image: Image | None = None,
    ) -> Image:
        """Run reconstruction pipeline.

        Args:
            kspace: Input k-space [B, C, H, W, 2]
            mask: Sampling mask [B, H, W]
            initial_image: Initial image estimate [B, H, W, 2] (optional)

        Returns:
            Reconstructed image [B, H, W, 2]

        forward method for MRIReconstructionPipeline.

        Executes PyTorch tensor operations.

        Args:
            kspace (KSpace): Expected input tensor.
            mask (Mask): Expected input tensor.
            initial_image (Image | None): Expected input tensor.

        Returns:
            Image: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Estimate coil sensitivities if needed
        coil_sens = None
        if self.coil_estimator is not None and kspace.shape[1] > 1:
            coil_sens = self.coil_estimator(kspace)

        # Initial reconstruction (zero-filled)
        if initial_image is None:
            initial_image = self.operator.adjoint(kspace, coil_sens)

        # Apply data consistency
        reconstructed = self.dc_layer(initial_image, kspace, mask, coil_sens)

        return reconstructed


# ``PhysicsInformedLoss`` moved to the canonical loss home at
# ``spectramr.models.losses.physics_informed_integration_loss`` 2026-05-14
# per ``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 5. The
# re-export from this module was dropped because it created a circular
# import (physics/integration → models/losses/__init__ → physics/...).
# Callers must import from the canonical location directly.


def create_physics_operator(operator_type: str = "fft2d", **kwargs) -> IForwardOperator:
    """Factory function for creating k-space operators via the operator registry.

    ``fft2d``, ``nufft`` and ``multi_coil_fft`` are registered at import time by
    ``infrastructure.physics.implementations`` (no hardcoded if/elif fallback).
    An unknown ``operator_type`` raises ``ValueError`` (no silent fallback, pitfall #9).
    Requesting ``nufft`` when ``torchkbnufft`` is unavailable raises (never silently
    downgrades to FFT2D) because ``NUFFTOperator`` raises ``ImportError`` on construct.
    """
    try:
        return create_operator(operator_type, **kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown operator type: {operator_type}") from exc


def create_dc_layer(
    operator: IForwardOperator,
    lambda_dc: float = 1.0,
) -> OperatorProjectionDataConsistency:
    """Factory function for creating data consistency layers."""
    return OperatorProjectionDataConsistency(operator, lambda_dc)


def create_coil_estimator(
    method: str = "espirit",
    **kwargs,
) -> CoilSensitivityEstimator:
    """Factory function for creating coil sensitivity estimators."""
    if method.lower() == "espirit":
        return CoilSensitivityEstimator(**kwargs)
    raise ValueError(f"Unknown coil estimation method: {method}")


# Utility functions


def rss_coil_combine(kspace: torch.Tensor) -> torch.Tensor:
    """Root sum of squares coil combination."""
    # Convert to complex
    kspace_complex = torch.complex(kspace[..., 0], kspace[..., 1])

    # IFFT
    image = ifft2c(kspace_complex)

    # RSS combination
    rss = torch.sqrt(torch.sum(torch.abs(image) ** 2, dim=1))

    # Convert back to real/imag representation
    return torch.stack([rss.real, rss.imag], dim=-1)


def load_coil_sensitivity_maps(path: str) -> torch.Tensor:
    """Load pre-computed coil sensitivity maps from various formats."""
    from pathlib import Path

    import numpy as np

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Coil sensitivity file not found: {path}")

    # Determine file format and load accordingly
    if path.suffix.lower() in [".nii", ".nii.gz"]:
        # NIfTI format
        try:
            import nibabel as nib

            nii_img = nib.load(str(path))
            coil_maps = torch.from_numpy(nii_img.get_fdata()).float()
        except ImportError:
            raise ImportError("nibabel required for NIfTI loading")

    elif path.suffix.lower() in [".h5", ".hdf5"]:
        # HDF5 format
        try:
            import h5py

            with h5py.File(str(path), "r") as f:
                # Try different dataset names
                for name in [
                    "coil_sensitivity",
                    "coil_maps",
                    "sensitivity",
                    "sens",
                    "data",
                ]:
                    if name in f:
                        coil_maps = torch.from_numpy(f[name][()]).float()
                        break
                else:
                    raise ValueError(f"No data found in {path}")
        except ImportError:
            raise ImportError("h5py required for HDF5 loading")

    elif path.suffix.lower() == ".npy":
        # NumPy format
        coil_maps = torch.from_numpy(np.load(str(path))).float()

    elif path.suffix.lower() == ".pt":
        # PyTorch format
        coil_maps = safe_torch_load(str(path), map_location="cpu").float()

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Validate coil sensitivity maps
    if coil_maps.ndim < 3:
        raise ValueError(f"Maps need >= 3 dims, got {coil_maps.ndim}")

    # Ensure complex format if real-valued (assume magnitude only)
    if coil_maps.dtype == torch.float32 and coil_maps.shape[-1] != 2:
        # Convert magnitude to complex (zero phase)
        coil_maps = torch.complex(coil_maps, torch.zeros_like(coil_maps))

    return coil_maps


def validate_physics_consistency(
    image: Image,
    kspace: KSpace,
    operator: IForwardOperator,
    tolerance: float = 1e-6,
) -> bool:
    """Validate forward/adjoint operator consistency."""
    # Forward: image -> k-space
    kspace_pred = operator.forward(image)

    # Adjoint: k-space -> image
    image_pred = operator.adjoint(kspace_pred)

    # Check consistency
    diff = torch.abs(image - image_pred).mean()
    return bool((diff < tolerance).item())
