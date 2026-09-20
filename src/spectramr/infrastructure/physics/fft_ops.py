r"""Unified FFT Operations Module
============================

Centralized FFT operations for MRI reconstruction with multi-coil support.
Consolidates all FFT-related functionality from physics modules.

.. math::

    X[k] = \sum_{n=0}^{N-1} x[n] e^{-i 2\pi k n / N}
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from spectramr.core.compile_fences import dynamo_disable

from .interfaces import IPhysicsOperator

logger = logging.getLogger(__name__)


def _to_complex(x: torch.Tensor, *, strict: bool = False) -> torch.Tensor:
    """Ensure a tensor is complex dtype.

    Accepts:
    - complex tensors (returned unchanged)
    - real tensors with last dim==2 encoding (real, imag) -> view_as_complex
    - real tensors [B, 2, H, W] where 2 is real/imag channels
    - real tensors -> cast to complex with zero imaginary part (unless strict=True)

    Args:
        x: Input tensor to convert to complex.
        strict: If True, raises ValueError for pure real tensors that cannot be
            unambiguously converted to complex. Use strict=True in physics-critical
            code paths to catch bugs where complex inputs are expected but real
            provided. Default False for backward compatibility.

    Returns:
        Complex tensor.

    Raises:
        ValueError: If strict=True and input is a pure real tensor without
            explicit real/imag encoding.

    Example:
        >>> # Normal usage (backward compatible)
        >>> img = _to_complex(real_tensor)  # Casts to complex64 with zero imag
        >>>
        >>> # Strict mode for physics-critical code
        >>> kspace = _to_complex(data, strict=True)  # Raises if not properly encoded
    """
    if torch.is_complex(x):
        return x

    # [FIX] Prioritize last-dim real/imag encoding (view-as-real format)
    # This prevents misinterpreting 5D tensors [B, C, H, W, 2] as interleaved
    # when the spatial dimension (H or W) is even.
    if x.shape[-1] == 2:
        return torch.view_as_complex(x.contiguous())

    # CRITICAL: Handle [..., C, H, W] where C is even (real/imag interleaved)
    # This is the PyTorch CNN standard format for complex data in this project
    if x.ndim >= 3 and x.shape[-3] % 2 == 0:
        # Extract real and imaginary parts
        # Assuming interleaved format: [R1, I1, R2, I2, ...]
        shape = list(x.shape)
        C = shape[-3]
        spatial_dims = shape[-2:]
        batch_dims = shape[:-3]

        x_reshaped = x.view(*batch_dims, C // 2, 2, *spatial_dims)
        real_part = x_reshaped[..., 0, :, :]  # [..., C//2, H, W]
        imag_part = x_reshaped[..., 1, :, :]  # [..., C//2, H, W]

        # Create complex tensor
        return torch.complex(real_part, imag_part)  # [..., C//2, H, W] complex

    # [AUDIT FIX] Strict mode to catch physics bugs
    if strict:
        raise ValueError(
            f"_to_complex(strict=True): Cannot convert pure real tensor {x.shape} "
            f"to complex. Expected complex input or real/imag encoded tensor "
            f"(shape [B, 2, H, W] or (..., 2)). This may indicate a physics bug "
            f"where complex k-space/image data was expected but real provided."
        )

    # [RELAXED CONSTRAINT] Allow real inputs by casting to complex (zero phase).
    # This prevents crashes in standard tests and non-physics-strict pipelines.
    logger.debug(
        f"_to_complex: Casting real tensor {x.shape} to complex64 with zero imaginary. "
        f"Use strict=True to catch potential physics bugs."
    )
    return x.to(dtype=torch.complex64)


def _to_ri(x: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to real-imag stacked at last dim."""
    if not torch.is_complex(x):
        return torch.stack((x, torch.zeros_like(x)), dim=-1)
    return torch.view_as_real(x)


@dynamo_disable
def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT with orthonormal normalization.

    [STABILIZATION FIX 3.3] Wrapped in FP32 context to prevent NaN gradients
    during mixed-precision training.

    ``ifftshift`` moves the image centre (index ``N // 2``) to index 0, where
    ``fft2`` expects the origin; ``fftshift`` then moves DC back to ``N // 2``.
    Using the two in the opposite order is a no-op on EVEN lengths (there
    ``fftshift == ifftshift``) but mis-centres every ODD axis by one index.

    Args:
        x: complex tensor (..., H, W), image centre at index ``N // 2``

    Returns:
        complex tensor (..., H, W) k-space with DC at index ``N // 2``
    """
    with torch.amp.autocast("cuda", enabled=False):
        x = _to_complex(x).to(torch.complex64)
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        k = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        k = torch.fft.fftshift(k, dim=(-2, -1))
        return k


@dynamo_disable
def ifft2c(k: torch.Tensor) -> torch.Tensor:
    """Centered 2D inverse FFT with orthonormal normalization.

    [STABILIZATION FIX 3.3] Wrapped in FP32 context to prevent NaN gradients
    during mixed-precision training.

    Args:
        k: complex tensor (..., H, W) k-space with DC at index ``N // 2``

    Returns:
        complex tensor (..., H, W), image centre at index ``N // 2``
    """
    with torch.amp.autocast("cuda", enabled=False):
        k = _to_complex(k).to(torch.complex64)
        k = torch.fft.ifftshift(k, dim=(-2, -1))
        x = torch.fft.ifft2(k, dim=(-2, -1), norm="ortho")
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x


def fft1c(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Centered 1-D FFT along ``dim`` with orthonormal normalization.

    The single-axis counterpart of :func:`fft2c`, for operators that work in
    hybrid space (one axis in k-space, the other in the image domain): the
    readout-axis transform of ``MultiCoilForwardOperator`` used to spell the
    three shift/transform calls inline (non-negotiable 2). ``ifftshift`` moves
    the centre of ``dim`` to index 0, the transform runs along ``dim`` alone,
    and ``fftshift`` returns DC to ``N // 2``. FP32 like the 2-D helpers.

    Args:
        x: complex (or real / real-imag) tensor; centre of ``dim`` at ``N // 2``
        dim: the axis to transform

    Returns:
        complex64 tensor with DC of ``dim`` at index ``N // 2``
    """
    with torch.amp.autocast("cuda", enabled=False):
        x = _to_complex(x).to(torch.complex64)
        x = torch.fft.ifftshift(x, dim=dim)
        k = torch.fft.fft(x, dim=dim, norm="ortho")
        return torch.fft.fftshift(k, dim=dim)


def ifft1c(k: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Centered 1-D inverse FFT along ``dim`` with orthonormal normalization (see :func:`fft1c`)."""
    with torch.amp.autocast("cuda", enabled=False):
        k = _to_complex(k).to(torch.complex64)
        k = torch.fft.ifftshift(k, dim=dim)
        x = torch.fft.ifft(k, dim=dim, norm="ortho")
        return torch.fft.fftshift(x, dim=dim)


def fft1_uncentered(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Uncentered 1-D FFT along ``dim`` (DC at index 0), orthonormal, FP32.

    For data whose convention keeps DC at index 0 (``centered=False`` on the
    forward operator); the centered helpers above are the default.
    """
    with torch.amp.autocast("cuda", enabled=False):
        return torch.fft.fft(_to_complex(x).to(torch.complex64), dim=dim, norm="ortho")


def ifft1_uncentered(k: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Uncentered 1-D inverse FFT along ``dim`` (see :func:`fft1_uncentered`)."""
    with torch.amp.autocast("cuda", enabled=False):
        return torch.fft.ifft(_to_complex(k).to(torch.complex64), dim=dim, norm="ortho")


def fft2_uncentered_last2(x: torch.Tensor) -> torch.Tensor:
    """Uncentered 2-D FFT over the trailing two axes of any rank (DC at index 0), orthonormal, FP32.

    Distinct from :func:`fft_volume_spatial_uncentered`, which takes a 5-D
    ``(B, C, D, H, W)`` volume and whose inverse returns the real part: this pair
    keeps the rank and the complex values, which is what the forward operator's
    ``centered=False`` branch needs on ``(B, coils, H, W)`` k-space.
    """
    with torch.amp.autocast("cuda", enabled=False):
        return torch.fft.fft2(_to_complex(x).to(torch.complex64), dim=(-2, -1), norm="ortho")


def ifft2_uncentered_last2(k: torch.Tensor) -> torch.Tensor:
    """Uncentered 2-D inverse FFT over the trailing two axes, complex out (see :func:`fft2_uncentered_last2`)."""
    with torch.amp.autocast("cuda", enabled=False):
        return torch.fft.ifft2(_to_complex(k).to(torch.complex64), dim=(-2, -1), norm="ortho")


def fft2c_masked(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Centered 2D FFT with optional sampling mask.

    Args:
        x: complex tensor (..., H, W)
        mask: real sampling mask broadcastable to k-space (..., H, W)

    Returns:
        complex tensor (..., H, W)

    """
    kspace = fft2c(x)

    if mask is not None:
        # Ensure mask is real and broadcastable
        if torch.is_complex(mask):
            mask = mask.real
        kspace = kspace * mask

    return kspace


def ifft2c_masked(k: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Centered 2D inverse FFT with optional sampling mask.

    Args:
        k: complex tensor (..., H, W)
        mask: real sampling mask broadcastable to k-space (..., H, W)

    Returns:
        complex tensor (..., H, W)

    """
    # Apply mask before inverse FFT if provided
    if mask is not None:
        if torch.is_complex(mask):
            mask = mask.real
        k = k * mask

    return ifft2c(k)


def spatial_dims(x: torch.Tensor) -> tuple[int, ...]:
    """Spatial axes of a ``[B, C, *spatial]`` tensor (2-D or 3-D).

    Args:
        x: Tensor shaped ``[B, C, H, W]`` or ``[B, C, H, W, D]``.

    Returns:
        The trailing axis indices, i.e. ``(2, 3)`` or ``(2, 3, 4)``.

    Raises:
        ValueError: If the tensor is neither 2-D nor 3-D spatially. No silent
            fallback to a guessed rank (non-negotiable 3).
    """
    ndim = x.ndim - 2
    if ndim not in (2, 3):
        raise ValueError(f"expected [B, C, H, W] or [B, C, H, W, D], got shape {tuple(x.shape)}")
    return tuple(range(2, x.ndim))


def _resolve_dims(x: torch.Tensor, dims: tuple[int, ...] | None) -> tuple[int, ...]:
    """Normalise an explicit ``dims`` override, or fall back to every spatial axis."""
    if dims is None:
        return spatial_dims(x)
    return tuple(a % x.ndim for a in dims)


def fftnc(x: torch.Tensor, dims: tuple[int, ...] | None = None) -> torch.Tensor:
    """CENTRED N-D FFT over the spatial axes, each channel independently.

    The N-D companion to :func:`fft2c`: ``ifftshift -> fftn(norm="ortho") ->
    fftshift``, so DC lands at the geometric centre and the transform is
    unitary. Works on 2-D and 3-D spatial ranks alike.

    Unlike :func:`fft2c` this does **not** call :func:`_to_complex`, and that is
    the whole reason both exist. ``_to_complex`` reinterprets a *real* tensor
    with an even channel count as interleaved real/imag pairs, so it changes the
    channel count: a real 8-frame magnitude stack silently collapses to 4
    complex images, and ``[B, 2, H, W]`` comes back as ``[B, 1, H, W]``.

    Choose :func:`fft2c` when the tensor really is real-interleaved complex MRI
    k-space and the caller wants that reinterpretation. Choose this when the
    channel axis must survive the transform -- including when *correcting an
    existing uncentered call*, where swapping in ``fft2c`` would smuggle a
    channel-layout change into what should be a centering-only fix.

    Args:
        x: Tensor shaped ``[B, C, H, W]`` or ``[B, C, H, W, D]``.
        dims: Axes to transform. Defaults to every spatial axis. Pass an
            explicit tuple (e.g. ``(-2, -1)``) for an in-plane transform of a
            volume, where the default would give a 3-D transform instead.

    Returns:
        Complex centred spectrum of the same shape.
    """
    d = _resolve_dims(x, dims)
    z = torch.fft.ifftshift(x.to(torch.complex64), dim=d)
    k = torch.fft.fftn(z, dim=d, norm="ortho")
    return torch.fft.fftshift(k, dim=d)


def ifftnc(k: torch.Tensor, dims: tuple[int, ...] | None = None) -> torch.Tensor:
    """Inverse of :func:`fftnc` -- CENTRED N-D inverse FFT.

    Args:
        k: Centred spectrum shaped ``[B, C, H, W]`` or ``[B, C, H, W, D]``.
        dims: Axes to transform; must match the forward call. Defaults to every
            spatial axis.

    Returns:
        Complex image of the same shape.
    """
    d = _resolve_dims(k, dims)
    z = torch.fft.ifftshift(k.to(torch.complex64), dim=d)
    x = torch.fft.ifftn(z, dim=d, norm="ortho")
    return torch.fft.fftshift(x, dim=d)


def coil_combine_rss(img_coils: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Root-sum-of-squares coil combination in image domain.

    Args:
        img_coils: complex image tensor (B, C_coils, H, W)

    Returns:
        real magnitude image (B, 1, H, W)

    """
    img_coils = _to_complex(img_coils)
    mag2 = (img_coils.real**2 + img_coils.imag**2).sum(dim=1, keepdim=True)
    return torch.sqrt(mag2 + eps)


def apply_mask(kspace: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Apply a sampling mask to k-space if provided.

    mask should be broadcastable to kspace (B, 1 or C, H, W) and real-valued in
    {0,1}.
    """
    if mask is None:
        return kspace
    # Ensure mask is real and broadcastable
    if torch.is_complex(mask):
        mask = mask.real
    return kspace * mask


def sense_forward(
    img: torch.Tensor,
    smaps: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward operator A: image -> k-space.

    Args:
        img: real image (B, 1 or C, H, W)
        smaps: complex sensitivity maps (B, C_coils, H, W) or None for
            single-coil
        mask: real sampling mask broadcastable to k-space
    Returns:
        kspace: complex k-space (B, C_coils or C, H, W)

    """
    # Cast image to complex
    img_c = _to_complex(img)
    if smaps is not None:
        smaps_c = _to_complex(smaps)

        # Handle channel broadcasting
        # img: (B, C_img, H, W)
        # smaps: (B, C_coils, H, W)

        # If img has 1 channel, broadcast to all coils
        if img_c.shape[1] == 1:
            coil_imgs = img_c * smaps_c
        else:
            # If img has multiple channels, we assume they map to coils 1-to-1
            # OR we need to broadcast.
            if img_c.dim() == smaps_c.dim() and img_c.shape[1] == smaps_c.shape[1]:
                # Element-wise multiplication
                coil_imgs = img_c * smaps_c
            else:
                coil_imgs = img_c * smaps_c

        kspace = fft2c(coil_imgs)
    else:
        kspace = fft2c(img_c)
    return apply_mask(kspace, mask)


def sense_adjoint(
    kspace: torch.Tensor,
    smaps: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    combine_rss: bool = False,
) -> torch.Tensor:
    """Adjoint operator A^H: k-space -> image.

    Args:
        kspace: complex k-space (B, C_coils or C, H, W)
        smaps: complex sensitivity maps (B, C_coils, H, W) or None
        mask: optional sampling mask
        combine_rss: if True and smaps provided, return RSS magnitude;
            else SENSE combine
    Returns:
        complex image (B, 1 or C, H, W) if smaps provided; else complex image
        matching input channels

    """
    kspace = apply_mask(_to_complex(kspace), mask)
    img = ifft2c(kspace)
    if smaps is not None:
        smaps_c = _to_complex(smaps)
        if combine_rss:
            return coil_combine_rss(img)
        # SENSE combine: sum over coils of conj(smaps) * img_coil
        img = (img * torch.conj(smaps_c)).sum(dim=1, keepdim=True)
    return img


def coil_combine(
    coil_imgs: torch.Tensor, method: str = "rss", smaps: torch.Tensor | None = None
) -> torch.Tensor:
    """Combine coil images ``(B, C, H, W)`` -> ``(B, 1, H, W)`` real magnitude.

    Raises on unknown method or sense-without-smaps (no silent fallback)."""
    if method == "rss":
        return coil_combine_rss(coil_imgs)
    if method == "sense":
        if smaps is None:
            raise ValueError("coil_combine method='sense' requires smaps")
        # Roemer/SENSE-optimal combine = (sum_c conj(S_c) I_c) / (sum_c |S_c|^2).
        # The denominator is what distinguishes the *combine* (an unbiased
        # magnetization estimate) from the matched-filter *adjoint*
        # (``sense_adjoint``, which correctly omits it). Without it the result is
        # shaded by sum_c|S_c|^2 unless the maps are unit-RSS-normalized.
        smaps_c = _to_complex(smaps)
        num = (_to_complex(coil_imgs) * torch.conj(smaps_c)).sum(dim=1, keepdim=True)
        denom = (smaps_c.abs() ** 2).sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (num / denom).abs()
    raise ValueError(f"Unknown coil combine method {method!r}. Valid: rss, sense.")


class NUFFTOperators(IPhysicsOperator):
    """Optional NUFFT wrapper (requires `torchkbnufft`).

    Exposes simple forward/adjoint call signatures similar to Cartesian ops.
    Mathematical Formulation:
    .. math::

        \\mathcal{O}_{NUFFT}(x) = \\sum_{j} x_j e^{-i \vec{k} \\cdot \vec{r}_j}"""

    def __init__(
        self,
        om: torch.Tensor,
        im_size: tuple[int, int],
        grid_size: tuple[int, int] | None = None,
    ):
        """__init__.

        Args:
            om (torch.Tensor): Description.
            im_size (tuple[int, int]): Description.
            grid_size (tuple[int, int] | None): Description.
        Returns:
            Any: Description.
        """
        if not self._has_torchkbnufft():
            raise ImportError(
                "torchkbnufft not installed. Install it to use NUFFT ops.",
            )
        import torchkbnufft as tkbn

        self.nufft = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)
        self.adj_nufft = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)
        self.om = om  # k-space trajectory (M, 2)
        self._im_size = im_size

    @staticmethod
    def _has_torchkbnufft() -> bool:
        """_has_torchkbnufft.

        Returns:
            bool: Description.
        """
        try:
            import torchkbnufft  # noqa: F401

            return True
        except ImportError:
            return False

    def forward(
        self,
        img: torch.Tensor,
        smaps: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """forward.

        Args:
            img (torch.Tensor): Description.
            smaps (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for NUFFTOperators.

        Executes PyTorch tensor operations.

        Args:
            img (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            smaps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        img_c = _to_complex(img)
        if smaps is not None:
            smaps_c = _to_complex(smaps)
            while img_c.dim() < smaps_c.dim():
                img_c = img_c.unsqueeze(1)
            coil_imgs = img_c * smaps_c
            kspace = self.nufft(coil_imgs, self.om)
        else:
            kspace = self.nufft(img_c, self.om)
        return kspace

    def adjoint(
        self,
        kspace: torch.Tensor,
        smaps: torch.Tensor | None = None,
        combine_rss: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """adjoint.

        Args:
            kspace (torch.Tensor): Description.
            smaps (torch.Tensor | None): Description.
            combine_rss (bool): Description.
        Returns:
            torch.Tensor: Description.
        """
        kspace_c = _to_complex(kspace)
        img = self.adj_nufft(kspace_c, self.om)
        if smaps is not None:
            smaps_c = _to_complex(smaps)
            if combine_rss:
                return coil_combine_rss(img)
            img = (img * torch.conj(smaps_c)).sum(dim=1, keepdim=True)
        return img

    @property
    def img_size(self) -> tuple[int, int]:
        """Get image size (H, W)."""
        return self._im_size

    def get_operator_type(self) -> str:
        """Get string identifier for operator type."""
        return "nufft"


def fft_volume_slice_dimension_uncentered(volume: torch.Tensor) -> torch.Tensor:
    """Apply FFT along slice dimension (z-axis).

    Args:
        volume: (B, C, D, H, W) or (D, H, W)

    Returns:
        (B, C, D, H, W) or (D, H, W) in Fourier domain
    """
    # Ensure 3D
    if volume.ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    elif volume.ndim == 4:
        volume = volume.unsqueeze(1)  # (B, 1, D, H, W)

    # Apply batched FFT along depth dimension (dim=2)
    kspace = torch.fft.fft(volume, dim=2, norm="ortho")
    return kspace


def ifft_volume_slice_dimension_uncentered(
    kspace: torch.Tensor, return_complex: bool = False
) -> torch.Tensor:
    """Inverse FFT along slice dimension.

    Args:
        kspace: (B, C, D, H, W) or (D, H, W)
        return_complex: If True, returns the full complex tensor.
            Default False preserves legacy behavior (returns ``.real``)
            but discards phase. Phase-aware reconstruction MUST pass
            ``return_complex=True``. See findings booklet 2026-05-05 P-1.

    Returns:
        Real part of inverse FFT (default) or complex tensor.
    """
    image = torch.fft.ifft(kspace, dim=2, norm="ortho")
    return image if return_complex else image.real


def fft_volume_spatial_uncentered(volume: torch.Tensor) -> torch.Tensor:
    """Apply FFT along spatial dimensions (H, W).

    Args:
        volume: (B, C, D, H, W)

    Returns:
        (B, C, D, H, W) in Fourier domain
    """
    # Reshape for batched FFT: combine batch + channel + depth
    original_shape = volume.shape
    B, C, D, H, W = original_shape

    # Flatten spatial batch dimension
    volume_flat = volume.contiguous().view(B * C * D, H, W)

    # Apply 2D FFT (batched across first dimension)
    kspace_flat = torch.fft.fft2(volume_flat, norm="ortho")

    # Reshape back
    kspace = kspace_flat.contiguous().view(original_shape)
    return kspace


def ifft_volume_spatial_uncentered(
    kspace: torch.Tensor, return_complex: bool = False
) -> torch.Tensor:
    """Inverse FFT along spatial dimensions (H, W).

    Args:
        kspace: (B, C, D, H, W)
        return_complex: If True, returns the full complex tensor.
            Default False preserves legacy behavior (returns ``.real``)
            but discards phase. See findings booklet 2026-05-05 P-1.

    Returns:
        Real part of inverse FFT (default) or complex tensor.
    """
    original_shape = kspace.shape
    B, C, D, H, W = original_shape

    kspace_flat = kspace.contiguous().view(B * C * D, H, W)
    image_flat = torch.fft.ifft2(kspace_flat, norm="ortho")
    image = image_flat.contiguous().view(original_shape)
    return image if return_complex else image.real


def fft_volume_full3d_uncentered(volume: torch.Tensor) -> torch.Tensor:
    """Apply 3D FFT (most efficient for isotropic volumes).

    Args:
        volume: (B, C, D, H, W)

    Returns:
        (B, C, D, H, W) in Fourier domain
    """
    kspace = torch.fft.fftn(volume, dim=(2, 3, 4), norm="ortho")
    return kspace


def ifft_volume_full3d_uncentered(
    kspace: torch.Tensor, return_complex: bool = False
) -> torch.Tensor:
    """Inverse 3D FFT.

    Args:
        kspace: (B, C, D, H, W)
        return_complex: If True, returns the full complex tensor.
            Default False preserves legacy behavior (returns ``.real``)
            but discards phase. Phase-aware reconstruction MUST pass
            ``return_complex=True``. See findings booklet 2026-05-05 P-1.

    Returns:
        Real part of inverse FFT (default) or complex tensor.
    """
    image = torch.fft.ifftn(kspace, dim=(2, 3, 4), norm="ortho")
    return image if return_complex else image.real


def get_fft_ops() -> dict[str, Callable]:
    """Get dictionary of FFT operation functions.

    Returns:
        dict mapping operation names to callable functions

    """
    return {
        "fft2c": fft2c,
        "ifft2c": ifft2c,
        "fft2c_masked": fft2c_masked,
        "ifft2c_masked": ifft2c_masked,
        "fftnc": fftnc,
        "ifftnc": ifftnc,
        "fft1c": fft1c,
        "ifft1c": ifft1c,
        "fft1_uncentered": fft1_uncentered,
        "ifft1_uncentered": ifft1_uncentered,
        "fft2_uncentered_last2": fft2_uncentered_last2,
        "ifft2_uncentered_last2": ifft2_uncentered_last2,
        "sense_forward": sense_forward,
        "sense_adjoint": sense_adjoint,
        "coil_combine_rss": coil_combine_rss,
        "apply_mask": apply_mask,
        "fft_volume_slice_dimension_uncentered": fft_volume_slice_dimension_uncentered,
        "ifft_volume_slice_dimension_uncentered": ifft_volume_slice_dimension_uncentered,
        "fft_volume_spatial_uncentered": fft_volume_spatial_uncentered,
        "ifft_volume_spatial_uncentered": ifft_volume_spatial_uncentered,
        "fft_volume_full3d_uncentered": fft_volume_full3d_uncentered,
        "ifft_volume_full3d_uncentered": ifft_volume_full3d_uncentered,
    }


__all__ = [
    "FFTTransformer",
    "NUFFTOperators",
    "_to_complex",
    "_to_ri",
    "apply_mask",
    "coil_combine",
    "coil_combine_rss",
    "fft1_uncentered",
    "fft1c",
    "fft2c",
    "fft2c_masked",
    "fft_volume_full3d_uncentered",
    "fft_volume_slice_dimension_uncentered",
    "fft_volume_spatial_uncentered",
    "fftnc",
    "get_fft_ops",
    "ifft1_uncentered",
    "ifft1c",
    "fft2_uncentered_last2",
    "ifft2_uncentered_last2",
    "ifft2c",
    "ifft2c_masked",
    "ifft_volume_full3d_uncentered",
    "ifft_volume_slice_dimension_uncentered",
    "ifft_volume_spatial_uncentered",
    "ifftnc",
    "sense_adjoint",
    "sense_forward",
    "spatial_dims",
]


class FFTTransformer:
    """Transformer class for FFT operations.

    Unified transformer that handles device management and provides
    consistent interface for FFT operations across strategies.
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        """Initialize the FFT transformer.

        Args:
            device: Device for tensor operations. If None, uses input tensor device.
        """
        self.device = torch.device(device) if device else None

    def _ensure_device(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the configured device."""
        if self.device and x.device != self.device:
            return x.to(self.device)
        return x

    def fft2_uncentered(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D FFT along spatial dimensions, WITHOUT centering.

        DC lands at the array corner. Do not pair the result with a centred
        mask or a centred measured k-space -- see :meth:`fft2c`.
        """
        x = self._ensure_device(x)
        return fft_volume_spatial_uncentered(x)

    def ifft2_uncentered(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D IFFT along spatial dimensions, WITHOUT centering."""
        x = self._ensure_device(x)
        return ifft_volume_spatial_uncentered(x)

    def fftn_uncentered(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3D FFT, WITHOUT centering. DC lands at the array corner."""
        x = self._ensure_device(x)
        return fft_volume_full3d_uncentered(x)

    def ifftn_uncentered(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3D IFFT, WITHOUT centering."""
        x = self._ensure_device(x)
        return ifft_volume_full3d_uncentered(x)

    def fftnc(self, x: torch.Tensor, dims: tuple[int, ...] | None = None) -> torch.Tensor:
        """Centred N-D FFT over the spatial axes (2-D or 3-D)."""
        x = self._ensure_device(x)
        return fftnc(x, dims)

    def ifftnc(self, x: torch.Tensor, dims: tuple[int, ...] | None = None) -> torch.Tensor:
        """Centred N-D inverse FFT over the spatial axes (2-D or 3-D)."""
        x = self._ensure_device(x)
        return ifftnc(x, dims)

    def fft2c(self, x: torch.Tensor) -> torch.Tensor:
        """Centered 2D FFT."""
        x = self._ensure_device(x)
        return fft2c(x)

    def ifft2c(self, x: torch.Tensor) -> torch.Tensor:
        """Centered 2D inverse FFT."""
        x = self._ensure_device(x)
        return ifft2c(x)

    def fft2c_masked(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """2D forward FFT with undersampling mask.

        Args:
            x: Input tensor of shape [B, C, H, W]
            mask: Binary mask of shape [B, 1, H, W] or [1, H, W]

        Returns:
            Masked FFT result of shape [B, C, H, W]
        """
        x = self._ensure_device(x)
        mask = self._ensure_device(mask)
        return fft2c_masked(x, mask)

    def ifft2c_masked(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """2D inverse FFT with mask compensation.

        Args:
            x: Input tensor of shape [B, C, H, W]
            mask: Binary mask of shape [B, 1, H, W] or [1, H, W]

        Returns:
            Reconstructed result of shape [B, C, H, W]
        """
        x = self._ensure_device(x)
        mask = self._ensure_device(mask)
        return ifft2c_masked(x, mask)

    def image_to_kspace(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image domain data to k-space.

        Args:
            image: Image domain data [B, C, H, W]

        Returns:
            K-space data [B, C, H, W] (complex)
        """
        image = self._ensure_device(image)
        return fft2c(image)

    def kspace_to_image(self, kspace: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Convert k-space to image domain.

        Args:
            kspace: K-space data [B, C, H, W]
            normalize: Whether to normalize output

        Returns:
            Image domain data [B, C, H, W]
        """
        kspace = self._ensure_device(kspace)
        img = ifft2c(kspace)
        if normalize:
            img_abs = torch.abs(img) if torch.is_complex(img) else img
            min_val = img_abs.amin(dim=(-2, -1), keepdim=True)
            max_val = img_abs.amax(dim=(-2, -1), keepdim=True)
            img = (img_abs - min_val) / (max_val - min_val + 1e-8)
        return img
