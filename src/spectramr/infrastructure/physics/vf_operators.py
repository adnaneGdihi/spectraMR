"""Virtual Fiducial Physics Operators for Marker-Anchored Reconstruction.

This module implements the operator chain for the Marker-Anchored Digital Twin
Pipeline.  Each operator is a differentiable ``nn.Module`` that can be composed
into the master forward chain:

.. math::

    \\mathbf{y} = \\mathcal{U} \\cdot \\mathcal{G} \\cdot \\mathcal{C}
                 \\cdot \\mathcal{B} \\cdot \\mathcal{W} \\cdot \\mathcal{H}
                 \\cdot \\mathcal{M} \\cdot \\mathbf{x}

Operators
---------
- :class:`MarkerPriorProjection`   — ``P``: anchors marker voxels to ideal prior
- :class:`RigidKinematicOperator`  — ``M``: inverse phase ramp from 6-DOF motion
- :class:`B0PhaseOperator`         — ``B``: unwind :math:`\\Delta\\omega(r)` dephasing
- :class:`GeometricWarpOperator`   — ``W``: differentiable spatial transformer
- :class:`ToeplitzPSFOperator`     — ``H``: PSF convolution / deconvolution
- :class:`NoisePreWhitening`       — ``N^{-1/2}``: Cholesky decorrelation
- :class:`MasterForwardChain`      — composition of all operators
- :class:`ADMMSolver`              — ADMM splitting with prior projection

Reference
---------
    Systemic Physical Invariance framework for marker-anchored MRI.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.fft_ops import ifft2c

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Prior Projection Operator  P
# ======================================================================


class MarkerPriorProjection(nn.Module):
    r"""Diagonal projection that anchors marker voxels to their ideal prior.

    In every iterative reconstruction step the operator replaces the voxels
    at the absolute coordinates of the fiducial markers with the known ideal
    image, preventing the optimiser from drifting in those regions.

    .. math::

        \mathcal{P}(\mathbf{x}) =
            (1 - \mathbf{m}) \odot \mathbf{x}
            + \mathbf{m} \odot \mathbf{x}_{\text{prior}}

    Args:
        marker_mask:  Binary spatial mask ``[1, 1, H, W]``  (1 = marker region).
        marker_prior: Ideal marker image ``[1, 1, H, W]``.
    """

    #: Recorded as a ``post`` stage of the debug snapshot (``snapshot.hooks``).
    snapshot_stage: ClassVar[str] = "post"

    def __init__(
        self,
        marker_mask: torch.Tensor,
        marker_prior: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", marker_mask.float())
        self.register_buffer("prior", marker_prior)

    @staticmethod
    def _match_channels(t: torch.Tensor, c: int) -> torch.Tensor:
        """Broadcast a marker mask/prior to ``c`` channels, block-layout aware.

        The VF strategies real-stack complex tensors as ``cat([real, imag])`` —
        a BLOCK layout ``[r0..r_{n-1}, i0..i_{n-1}]``. A single-marker prior is
        ``[m_r, m_i]`` (2ch); anchoring a multi-coil prediction ``[B, 2*ncoil]``
        means tiling each marker channel over its coil block, i.e.
        ``repeat_interleave`` → ``[m_r * ncoil, m_i * ncoil]`` — NOT ``repeat``
        (which would interleave r/i and mis-anchor every coil). Raises when the
        channel counts are incommensurate, rather than silently mis-projecting
        (CLAUDE.md pitfall #9).
        """
        cm = t.shape[1]
        if cm == c:
            return t
        if c % cm == 0:
            return t.repeat_interleave(c // cm, dim=1)
        raise ValueError(
            f"MarkerPriorProjection: cannot broadcast {cm}-channel mask/prior to "
            f"a {c}-channel input ({c} is not a multiple of {cm}). Marker and "
            "prediction channel layouts are incompatible."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply prior projection.

        Args:
            x: Image estimate ``[B, C, H, W]``. When ``C`` exceeds the marker's
                channel count (multi-coil prediction vs single-marker prior) the
                mask/prior are block-broadcast across coils — see
                :meth:`_match_channels`.

        Returns:
            Anchored image ``[B, C, H, W]``.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        c = x.shape[1]
        mask = self._match_channels(self.mask, c)
        prior = self._match_channels(self.prior, c)
        return (1.0 - mask) * x + mask * prior


# ======================================================================
# 1b. K-Space Data Consistency Projection  P_k
# ======================================================================


class MarkerKSpaceDCProjection(nn.Module):
    r"""K-space data consistency projection for marker-anchored reconstruction.

    Unlike the image-space :class:`MarkerPriorProjection`, this operator
    enforces consistency directly in k-space:

    .. math::

        \mathcal{P}_k(\hat{x}) = \mathcal{F}^{-1}\bigl[
            \mathbf{M}_k \odot \mathbf{Y}_{\text{prior}}
            + (1 - \mathbf{M}_k) \odot \mathcal{F}(\hat{x})
        \bigr]

    where :math:`\mathbf{M}_k` is the k-space support of the known marker
    and :math:`\mathbf{Y}_{\text{prior}}` is the ideal marker k-space.

    Args:
        marker_mask: Image-space marker mask ``[1, 1, H, W]``.
        marker_prior: Ideal marker image ``[1, 1, H, W]``.
    """

    #: Recorded as a ``post`` stage of the debug snapshot (``snapshot.hooks``).
    snapshot_stage: ClassVar[str] = "post"

    def __init__(
        self,
        marker_mask: torch.Tensor,
        marker_prior: torch.Tensor,
    ) -> None:
        super().__init__()
        from spectramr.infrastructure.physics.fft_ops import fft2c as _fft2c

        self.register_buffer("mask", marker_mask.float())
        self.register_buffer("prior", marker_prior)
        # Pre-compute k-space of the masked prior
        masked_prior = marker_mask.float() * marker_prior
        if not masked_prior.is_complex():
            masked_prior = torch.complex(masked_prior, torch.zeros_like(masked_prior))
        self.register_buffer("prior_kspace", _fft2c(masked_prior))
        # K-space support mask (non-zero entries of prior k-space)
        self.register_buffer("kspace_mask", (self.prior_kspace.abs() > 1e-8).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply k-space DC projection.

        Args:
            x: Image estimate ``[B, C, H, W]``.

        Returns:
            DC-projected image ``[B, C, H, W]``.

        forward method for MarkerKSpaceDCProjection.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        from spectramr.infrastructure.physics.fft_ops import fft2c as _fft2c

        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))

        x_k = _fft2c(x)
        # Replace marker k-space support with the known prior
        x_k = (1.0 - self.kspace_mask) * x_k + self.kspace_mask * self.prior_kspace
        return ifft2c(x_k)


# ======================================================================
# 2. Rigid Kinematic Operator  M
# ======================================================================


class RigidKinematicOperator(nn.Module):
    r"""Inverse rigid-body **translation** correction via k-space phase ramps.

    Given per-line motion parameters ``θ(t) = [dx, dy, dθ]``, applies the
    *inverse* phase ramp in k-space to undo translational motion:

    .. math::

        k_{\text{corrected}}(k_y) =
            k(k_y) \cdot e^{+i 2\pi (k_x \cdot dx(k_y) + k_y \cdot dy(k_y))}

    .. note::

        Per-line **rotation** (``dθ``) correction requires k-space regridding
        (the rotated object samples each line along a rotated k-space line),
        which is not implemented. Rather than silently ignoring the ``dθ``
        channel (an inert-mechanism facade), the operator **raises** when a
        non-zero rotation is supplied; pass ``dθ = 0`` for translation-only
        correction.

    Args:
        im_size: Spatial dimensions ``(H, W)``.
    """

    def __init__(self, im_size: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        H, W = im_size
        # Centered normalized k-grid for the Fourier shift theorem. Use
        # fftshift(fftfreq) (step 1/N, exact DC zero at index N//2), NOT
        # linspace(-0.5, 0.5, N) (step 1/(N-1), no exact DC): the latter applies
        # a commanded dx-pixel shift as dx*N/(N-1) and off-center, so it is not
        # an exact integer-pixel circular shift.
        ky = torch.fft.fftshift(torch.fft.fftfreq(H)).unsqueeze(1)
        kx = torch.fft.fftshift(torch.fft.fftfreq(W)).unsqueeze(0)
        self.register_buffer("kx", kx)  # [1, W]
        self.register_buffer("ky", ky)  # [H, 1]

    def forward(
        self,
        kspace: torch.Tensor,
        motion_params: torch.Tensor,
    ) -> torch.Tensor:
        """Apply inverse motion correction to k-space.

        Args:
            kspace: Complex k-space ``[B, C, H, W]``.
            motion_params: ``[B, 3, H]``  per-line ``(dx, dy, dθ)`` in pixels/radians.

        Returns:
            Motion-corrected k-space ``[B, C, H, W]``.

        forward method for RigidKinematicOperator.

        Executes PyTorch tensor operations.

        Args:
            kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            motion_params (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = kspace.shape
        dx = motion_params[:, 0, :]  # [B, H]
        dy = motion_params[:, 1, :]  # [B, H]
        dtheta = motion_params[:, 2, :]  # [B, H] per-line rotation (radians)
        if dtheta.abs().max() > 1e-6:
            raise NotImplementedError(
                "RigidKinematicOperator: non-zero rotation (dθ) requires k-space "
                "regridding, which is not implemented. Pass dθ=0 for "
                "translation-only correction."
            )

        # Phase ramp per readout line: φ(ky, kx) = 2π(kx·dx + ky·dy)
        # dx[b, ky_idx] needs broadcasting with kx [1, W]
        phase = (
            2.0
            * math.pi
            * (
                self.kx.unsqueeze(0) * dx.unsqueeze(-1)  # [B, H, W]
                + self.ky.squeeze(-1).unsqueeze(0) * dy.unsqueeze(-1)
            )
        )  # [B, H, W]

        # Inverse correction: multiply by conjugate phase
        correction = torch.polar(torch.ones_like(phase), phase)  # [B, H, W]
        correction = correction.unsqueeze(1)  # [B, 1, H, W]

        return kspace * correction


# ======================================================================
# 3. B0 Phase Aberration Operator  B
# ======================================================================


class B0PhaseOperator(nn.Module):
    r"""Unwind B0-induced phase errors in image space.

    Applies pixel-wise phase correction:

    .. math::

        \mathcal{B}(\mathbf{x}) = \mathbf{x} \cdot e^{+i \Delta\omega(\mathbf{r}) \cdot TE}

    Args:
        b0_map: Field map in Hz ``[1, 1, H, W]``.
        te: Echo time in seconds.
    """

    def __init__(
        self,
        b0_map: torch.Tensor | None = None,
        te: float = 0.01,
    ) -> None:
        super().__init__()
        self.te = te
        if b0_map is not None:
            self.register_buffer("b0_map", b0_map)
        else:
            self.b0_map = None

    def set_b0_map(self, b0_map: torch.Tensor) -> None:
        """Update the B0 field map.

        Args:
            b0_map: Field map in Hz ``[1, 1, H, W]``.
        """
        if hasattr(self, "b0_map") and isinstance(self.b0_map, torch.Tensor):
            self.b0_map = b0_map.to(self.b0_map.device)
        else:
            self.register_buffer("b0_map", b0_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply B0 phase unwinding.

        Args:
            x: Complex image ``[B, C, H, W]``.

        Returns:
            Phase-corrected image ``[B, C, H, W]``.

        forward method for B0PhaseOperator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.b0_map is None:
            return x
        # Phase to correct: +Δω·TE  (inverse of the corruption -Δω·TE)
        phase = 2.0 * math.pi * self.b0_map * self.te  # [1, 1, H, W]
        correction = torch.polar(torch.ones_like(phase), phase)
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        return x * correction


# ======================================================================
# 4. Geometric Warp Operator  W
# ======================================================================


class GeometricWarpOperator(nn.Module):
    r"""Differentiable spatial transformer for geometric distortion correction.

    Applies a displacement field (flow) to push voxels back to their correct
    positions, reversing gradient non-linearity and B0-induced geometric
    distortion.

    Args:
        im_size: Spatial dimensions ``(H, W)``.
    Mathematical Formulation:
    .. math::

        \mathcal{O}_{GeometricWarp}(x, v) = x(p + v(p))"""

    def __init__(self, im_size: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        self.im_size = im_size
        H, W = im_size
        # Base identity grid in normalised coordinates [-1, 1]
        grid_y = torch.linspace(-1, 1, H)
        grid_x = torch.linspace(-1, 1, W)
        grid = torch.stack(
            torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=-1
        )  # [H, W, 2]  (y, x)
        # grid_sample expects (x, y) ordering
        grid = grid[..., [1, 0]]  # [H, W, 2]  → (x, y)
        self.register_buffer("base_grid", grid.unsqueeze(0))  # [1, H, W, 2]

    def forward(
        self,
        x: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """Apply geometric correction.

        Args:
            x: Image ``[B, C, H, W]`` (real).
            flow: Displacement field ``[B, 2, H, W]`` in pixels (dx, dy).

        Returns:
            Warped image ``[B, C, H, W]``.

        forward method for GeometricWarpOperator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            flow (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = x.shape[0]
        H, W = self.im_size

        # Convert pixel displacement to normalised coordinates
        flow_norm = flow.permute(0, 2, 3, 1).clone()  # [B, H, W, 2]
        flow_norm[..., 0] = flow_norm[..., 0] * 2.0 / W
        flow_norm[..., 1] = flow_norm[..., 1] * 2.0 / H

        grid = self.base_grid.expand(B, -1, -1, -1) + flow_norm

        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


# ======================================================================
# 5. Toeplitz PSF Operator  H
# ======================================================================


class ToeplitzPSFOperator(nn.Module):
    """Point Spread Function convolution / deconvolution operator.

    Models spatially-invariant blur as convolution with a PSF kernel
    :math:`h`.  The adjoint is correlation with the conjugate-reflected
    kernel (Wiener-like deconvolution in the Fourier domain).

    Args:
        kernel_size: Spatial extent of the PSF kernel.
    Mathematical Formulation:
    .. math::

        \\mathcal{O}_{ToeplitzPSF}(x) = x * \text{PSF}"""

    def __init__(self, kernel_size: int = 15) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        # Initialise with identity (delta function)
        kernel = torch.zeros(1, 1, kernel_size, kernel_size)
        center = kernel_size // 2
        kernel[0, 0, center, center] = 1.0
        self.psf = nn.Parameter(kernel)

    def set_psf(self, psf: torch.Tensor) -> None:
        """Set the PSF kernel from an external estimate.

        Args:
            psf: PSF kernel ``[1, 1, kH, kW]``.
        """
        with torch.no_grad():
            self.psf.copy_(psf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PSF convolution (forward blur).

        Args:
            x: Image ``[B, C, H, W]``.

        Returns:
            Blurred image ``[B, C, H, W]``.

        forward method for ToeplitzPSFOperator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        pad = self.kernel_size // 2
        # Normalise PSF to unit sum
        psf_norm = self.psf / (self.psf.sum() + 1e-8)
        psf_expanded = psf_norm.expand(C, -1, -1, -1)
        return F.conv2d(x, psf_expanded, padding=pad, groups=C)

    def adjoint(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adjoint (correlation / approximate deconvolution).

        Args:
            x: Image ``[B, C, H, W]``.

        Returns:
            Deblurred image ``[B, C, H, W]``.
        """
        B, C, H, W = x.shape
        pad = self.kernel_size // 2
        psf_norm = self.psf / (self.psf.sum() + 1e-8)
        # Adjoint = correlation = convolution with flipped kernel
        psf_flipped = torch.flip(psf_norm, dims=[-2, -1])
        psf_expanded = psf_flipped.expand(C, -1, -1, -1)
        return F.conv2d(x, psf_expanded, padding=pad, groups=C)


# ======================================================================
# 6. Noise Pre-Whitening  N^{-1/2}
# ======================================================================


class NoisePreWhitening(nn.Module):
    r"""Noise pre-whitening via Cholesky decomposition.

    Given an estimated noise covariance matrix :math:`\Psi` from the marker
    background, computes :math:`L = \mathrm{chol}(\Psi)` and applies
    :math:`L^{-1}` to each k-space readout to decorrelate thermal noise.

    Args:
        num_coils: Number of receive coils.
    """

    def __init__(self, num_coils: int = 1) -> None:
        super().__init__()
        self.num_coils = num_coils
        # Identity initialisation
        self.register_buffer(
            "L_inv",
            torch.eye(num_coils, dtype=torch.complex64),
        )

    def estimate_from_noise_region(
        self,
        kspace: torch.Tensor,
        noise_mask: torch.Tensor,
    ) -> None:
        """Estimate noise covariance from marker background region.

        Args:
            kspace: Multi-coil k-space ``[B, num_coils, H, W]`` complex.
            noise_mask: Binary mask ``[1, 1, H, W]`` (1 = noise-only region).
        """
        with torch.no_grad():
            # Extract noise samples: [num_coils, N_samples]
            mask_flat = noise_mask.flatten().bool()
            samples = kspace[0][:, mask_flat]  # [num_coils, N]

            # Covariance: Ψ = (1/N) · samples · samples^H
            N = samples.shape[1]
            if N < 2:
                logger.warning("Too few noise samples for covariance estimation.")
                return

            samples_centered = samples - samples.mean(dim=1, keepdim=True)
            psi = (samples_centered @ samples_centered.conj().T) / (N - 1)

            # Cholesky: Ψ = L · L^H → L^{-1}
            try:
                L = torch.linalg.cholesky(psi)
                L_inv = torch.linalg.inv(L)
                self.L_inv = L_inv
            except torch.linalg.LinAlgError:
                logger.warning("Cholesky decomposition failed. Keeping identity.")

    def forward(self, kspace: torch.Tensor) -> torch.Tensor:
        """Apply pre-whitening to multi-coil k-space.

        Args:
            kspace: ``[B, num_coils, H, W]`` complex.

        Returns:
            Pre-whitened k-space ``[B, num_coils, H, W]``.

        forward method for NoisePreWhitening.

        Executes PyTorch tensor operations.

        Args:
            kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.num_coils <= 1:
            return kspace

        B, C, H, W = kspace.shape
        # Reshape: [B, C, H*W] → matmul with L_inv [C, C]
        kspace_flat = kspace.reshape(B, C, -1)
        whitened = torch.matmul(self.L_inv, kspace_flat)
        return whitened.reshape(B, C, H, W)


# ======================================================================
# 7. Master Forward Chain  A
# ======================================================================


class MasterForwardChain(nn.Module):
    r"""Composable forward encoding chain for MRI acquisition.

    Chains operators sequentially (right-to-left application):

    .. math::

        \mathbf{y} = \mathcal{U} \cdot \mathcal{F} \cdot \mathcal{C}
                     \cdot \mathcal{B} \cdot \mathcal{W} \cdot \mathcal{H}
                     \cdot \mathcal{M} \cdot \mathbf{x}

    Each operator is optional.  If not provided, that step is an identity.

    Args:
        operators: Ordered sequence of ``nn.Module`` operators to chain.
    """

    def __init__(self, operators: Sequence[nn.Module] | None = None) -> None:
        super().__init__()
        self.chain = nn.ModuleList(operators or [])

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the full forward chain.

        Only single-argument image->image (or k-space->k-space) operators are
        supported in the chain; ``**kwargs`` is reserved and currently ignored.
        Multi-argument operators (e.g. ``RigidKinematicOperator`` which needs
        ``motion_params``, ``RFModulationOperator`` which needs
        ``b1_plus``/``b1_minus``, ``GNLUnwarpingOperator`` which needs ``flow``)
        cannot be driven through the chain and will raise ``TypeError`` if added.

        Args:
            x: Input image/k-space.
            **kwargs: Reserved; ignored (not forwarded to chained operators).

        Returns:
            Encoded data.

        forward method for MasterForwardChain.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for op in self.chain:
            x = op(x)
        return x

    def adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the adjoint chain (reverse order, each op's adjoint).

        Like :meth:`forward`, only single-argument operators are supported;
        ``**kwargs`` is reserved and currently ignored (not forwarded to the
        chained operators).

        Args:
            y: Encoded data.
            **kwargs: Reserved; ignored (not forwarded to chained operators).

        Returns:
            Reconstructed image.
        """
        x = y
        for op in reversed(self.chain):
            if hasattr(op, "adjoint"):
                x = op.adjoint(x)
            else:
                x = op(x)
        return x


# ======================================================================
# 8. ADMM Solver
# ======================================================================


class ADMMSolver(nn.Module):
    r"""ADMM-based iterative reconstruction with prior projection.

    Alternating Direction Method of Multipliers splits the inverse problem
    into three sub-problems:

    1. **Data Consistency**: :math:`\mathbf{x} \leftarrow \mathbf{x} - \mu \mathcal{A}^H(\mathcal{A}\mathbf{x} - \mathbf{y})`
    2. **Regularisation**: :math:`\mathbf{z} \leftarrow \mathrm{prox}_R(\mathbf{x} + \mathbf{u})`
    3. **Prior Projection**: :math:`\mathcal{P}(\mathbf{z})`

    Args:
        forward_op: Forward operator (or :class:`MasterForwardChain`).
        regulariser: Neural network denoiser / regulariser.
        prior_projection: :class:`MarkerPriorProjection` instance.
        num_iters: Number of ADMM iterations.
        mu: Step size for data consistency.
        rho: ADMM penalty parameter.
    """

    def __init__(
        self,
        forward_op: nn.Module,
        regulariser: nn.Module,
        prior_projection: MarkerPriorProjection | None = None,
        num_iters: int = 8,
        mu: float = 0.1,
        rho: float = 1.0,
    ) -> None:
        super().__init__()
        self.forward_op = forward_op
        self.regulariser = regulariser
        self.prior_projection = prior_projection
        self.num_iters = num_iters
        self.mu = mu
        self.rho = rho

    def forward(
        self,
        y: torch.Tensor,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run ADMM reconstruction.

        Args:
            y: Measured k-space ``[B, C, H, W]``.
            x_init: Initial image estimate (default: adjoint of ``y``).

        Returns:
            Reconstructed image ``[B, C, H, W]``.

        forward method for ADMMSolver.

        Executes PyTorch tensor operations.

        Args:
            y (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_init (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Initialise
        if x_init is not None:
            x = x_init
        elif hasattr(self.forward_op, "adjoint"):
            x = self.forward_op.adjoint(y)
        else:
            x = ifft2c(y)

        z = x.clone()
        u = torch.zeros_like(x)  # Dual variable

        for _i in range(self.num_iters):
            # Step 1: Data Consistency (gradient descent)
            residual = self.forward_op(x) - y
            if hasattr(self.forward_op, "adjoint"):
                grad = self.forward_op.adjoint(residual)
            else:
                grad = ifft2c(residual)

            x = x - self.mu * grad + self.rho * (z - x - u)

            # Step 2: Regularisation (proximal / denoiser)
            z = self.regulariser(x + u)

            # Step 3: Prior Projection (anchor marker)
            if self.prior_projection is not None:
                z = self.prior_projection(z)

            # Step 4: Dual variable update
            u = u + x - z

        return z


__all__ = [
    "ADMMSolver",
    "B0PhaseOperator",
    "GeometricWarpOperator",
    "MarkerKSpaceDCProjection",
    "MarkerPriorProjection",
    "MasterForwardChain",
    "NoisePreWhitening",
    "RigidKinematicOperator",
    "ToeplitzPSFOperator",
]
