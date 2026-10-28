"""
Physics-informed loss functions for k-space cold diffusion.

Implements frequency-weighted losses that account for k-space spectral properties
(1/f decay) and enforce high-frequency reconstruction accuracy.
"""

import math
from collections.abc import Callable, Sequence
from itertools import pairwise

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss

VALID_REDUCTIONS: frozenset[str] = frozenset({"mean", "sum", "none"})
#: Ring edges as fractions of the Nyquist radius ``min(H, W) / 2`` (the PR1 ring rule).
DEFAULT_RING_EDGES: tuple[float, ...] = (1 / 8, 1 / 4, 1 / 2, 3 / 4)


def check_reduction(reduction: str, owner: str) -> None:
    """Raise at construction on an unknown reduction instead of at the first step."""
    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            f"Unknown reduction {reduction!r} for {owner}. Valid: {sorted(VALID_REDUCTIONS)}."
        )


def _fold_depth_into_batch(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, D, H, W]`` -> ``[B*D, C, H, W]``, keeping each slice's channels together.

    A bare ``reshape`` would interleave C with D, so the real/imag channel pairs
    of a stacked tensor would pair two depth slices instead.
    """
    b, c, d, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)


@register_loss("frequency_weighted_l1_kspace")
class FrequencyWeightedL1Loss(nn.Module):
    """
    L1 loss with frequency-dependent weighting for k-space reconstruction.

    Addresses the natural 1/f spectral decay in MRI by upweighting high-frequency
    components. Prevents blur artifacts where standard losses allow high-frequency
    errors to dominate during optimization.

    The weight map W(r) increases radially from DC (weight=1.0) at center to
    edge (weight=1.0+alpha*r_normalized) at domain boundary.

    Args:
        height (int): Height of k-space data.
        width (int): Width of k-space data.
        alpha (float): Slope of the linear radial weight ``1 + alpha * r``; the
                      map is then rescaled to unit mean. Default=2.0.
        reduction (str): 'mean', 'sum', or 'none'. Default='mean'.

    Shape:
        - Input: (B, 2*C, H, W) where 2*C represents stacked real/imaginary
                 OR (B, C, H, W) for complex64/complex128 tensors
        - Target: Same shape as input
        - Output: Scalar loss (if reduction='mean' or 'sum') or (B, 2*C, H, W)

    Physics:
        Natural k-space decay follows approximately 1/r due to distance-dependent
        signal attenuation. This loss compensates by upweighting periphery via:

            W(r) = 1.0 + alpha * (r_norm / r_max)

        where r_norm is radial distance from center and r_max is maximum radius.
        Applied element-wise: loss = mean(W(r) * |pred - target|)

    Example:
        >>> loss_fn = FrequencyWeightedL1Loss(height=256, width=256, alpha=2.0)
        >>> pred = torch.randn(2, 2, 256, 256)  # [B, 2*C, H, W] for complex
        >>> target = torch.randn(2, 2, 256, 256)
        >>> loss = loss_fn(pred, target)
        >>> loss.backward()
    """

    def __init__(
        self,
        height: int = 256,
        width: int = 256,
        alpha: float = 2.0,
        reduction: str = "mean",
        profile: str = "linear",
        ring_edges: Sequence[float] | None = None,
        ring_gain: Sequence[float] | None = None,
        ring_gain_outside: float | None = None,
    ) -> None:
        """Initialize FrequencyWeightedL1Loss.

        Note: height and width are optional defaults. The loss will dynamically
        adapt to input tensor shapes during forward pass.

        ``profile='ring_band'`` replaces the linear ramp with equal total weight per
        ring: ring ``r`` carries total weight ``ring_gain[r]`` spread over its bins, and
        every bin outside the rings forms one group of total ``ring_gain_outside``.
        Groups are counted on the full grid, as the linear map is, and the map is then
        renormalised to unit mean. A per-bin weight for the outside group would make
        the low band ~N_ring times heavier than a ring bin's weight (or starve it),
        so it is a group total instead.
        """
        super().__init__()
        check_reduction(reduction, "frequency_weighted_l1_kspace")
        if profile not in _PROFILE_BUILDERS:
            raise ValueError(
                f"Unknown profile {profile!r} for frequency_weighted_l1_kspace. "
                f"Valid: {sorted(_PROFILE_BUILDERS)}."
            )
        self.default_height = height
        self.default_width = width
        self.alpha = alpha
        self.reduction = reduction
        self.profile = profile
        self.ring_edges: tuple[float, ...] | None = None
        self.ring_gain: tuple[float, ...] | None = None
        self.ring_gain_outside: float | None = None
        if profile == "ring_band":
            self._configure_rings(ring_edges, ring_gain, ring_gain_outside)
        elif any(v is not None for v in (ring_edges, ring_gain, ring_gain_outside)):
            raise ValueError(
                "ring_edges, ring_gain and ring_gain_outside apply only to profile: ring_band."
            )

        # Cache for dynamically computed weight maps (keyed by (H, W))
        self._weight_map_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _configure_rings(
        self,
        ring_edges: Sequence[float] | None,
        ring_gain: Sequence[float] | None,
        ring_gain_outside: float | None,
    ) -> None:
        """Validate and store the ring-band geometry and gains (raises on any bad value)."""
        edges = tuple(float(e) for e in (DEFAULT_RING_EDGES if ring_edges is None else ring_edges))
        if len(edges) < 2 or edges[0] < 0 or any(b <= a for a, b in pairwise(edges)):
            raise ValueError(f"ring_edges must be >= 2 strictly increasing values >= 0: {edges}")
        n_rings = len(edges) - 1
        gains = tuple(1.0 for _ in range(n_rings)) if ring_gain is None else tuple(ring_gain)
        gains = tuple(float(g) for g in gains)
        outside = 1.0 if ring_gain_outside is None else float(ring_gain_outside)
        if len(gains) != n_rings:
            raise ValueError(f"ring_gain needs one value per ring ({n_rings}), got {len(gains)}")
        if any(not (math.isfinite(g) and g >= 0) for g in (*gains, outside)):
            raise ValueError(f"ring gains must be finite and >= 0: {gains}, outside={outside}")
        if sum(gains) + outside <= 0:
            raise ValueError("ring_band needs at least one positive gain")
        self.ring_edges = edges
        self.ring_gain = gains
        self.ring_gain_outside = outside

    def _ring_band_map(self, height: int, width: int) -> torch.Tensor:
        """Unit-mean ``(1, 1, H, W)`` map giving each ring equal total weight per ``ring_gain``."""
        assert self.ring_edges is not None and self.ring_gain is not None
        assert self.ring_gain_outside is not None
        ky = torch.arange(height, dtype=torch.float64) - height // 2
        kx = torch.arange(width, dtype=torch.float64) - width // 2
        nyquist = min(height, width) / 2.0
        rho = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2) / nyquist
        edges = torch.tensor(self.ring_edges, dtype=torch.float64)
        n_rings = len(self.ring_edges) - 1
        # bucketize gives 0 below the first edge and n_rings at or beyond the last one.
        ring = torch.bucketize(rho, edges, right=True) - 1
        slot = torch.where((ring >= 0) & (ring < n_rings), ring, torch.full_like(ring, n_rings))
        gains = torch.tensor((*self.ring_gain, self.ring_gain_outside), dtype=torch.float64)
        counts = torch.bincount(slot.flatten(), minlength=n_rings + 1).to(torch.float64)
        raw = gains[slot] / counts.clamp_min(1.0)[slot]
        return (raw / raw.mean()).to(torch.float32)[None, None]

    def _get_weight_map(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Get or create frequency weight map for given dimensions.

        Caches computed weight maps to avoid recomputation on same shapes.

        Args:
            height (int): Height of required weight map.
            width (int): Width of required weight map.
            device (torch.device): Device where weight map should be placed.
            dtype (torch.dtype): Data type for weight map.

        Returns:
            torch.Tensor: Weight map shape (1, 1, height, width).
        """
        cache_key = (height, width)
        # Ensure weights are always real, matching precision of input
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

        # Return cached weight map if available on the right device AND precision:
        # a float64 map cached by a complex128 call would silently promote later losses.
        if cache_key in self._weight_map_cache:
            cached = self._weight_map_cache[cache_key]
            if cached.device == device and cached.dtype == real_dtype:
                return cached

        # Compute new weight map
        weight_map = _PROFILE_BUILDERS[self.profile](self, height, width)
        weight_map = weight_map.to(device=device, dtype=real_dtype)

        # Cache it
        self._weight_map_cache[cache_key] = weight_map

        return weight_map

    def _create_frequency_weight_map(
        self,
        height: int,
        width: int,
        alpha: float,
    ) -> torch.Tensor:
        """
        Create radial frequency weight map.

        Computes 2D Gaussian-like radial distance from center, then applies
        linear scaling. Center (DC) has weight ~1.0, edges have weight ~1.0+alpha.

        Args:
            height (int): Height of map.
            width (int): Width of map.
            alpha (float): Scaling exponent.

        Returns:
            torch.Tensor: Weight map shape (1, 1, height, width) normalized to [1, 1+alpha].
        """
        # Create meshgrid centered at (0, 0)
        y = torch.arange(height, dtype=torch.float32) - height / 2.0
        x = torch.arange(width, dtype=torch.float32) - width / 2.0
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Radial distance from center
        r = torch.sqrt(xx**2 + yy**2)

        # Normalize to [0, 1]
        r_max = torch.sqrt(
            torch.tensor((height / 2.0) ** 2 + (width / 2.0) ** 2, dtype=torch.float32)
        )
        r_norm = r / (r_max + 1e-8)

        # Linear weight: 1.0 (center) to 1.0+alpha (edges)
        weight = 1.0 + alpha * r_norm

        # Reshape for broadcasting: [1, 1, H, W]
        weight_map = weight.unsqueeze(0).unsqueeze(0)

        # Normalize mean weight to 1.0 for stable training
        weight_map = weight_map / (weight_map.mean() + 1e-8)

        return weight_map

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        acquired: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute frequency-weighted L1 loss.

        Args:
            pred (torch.Tensor): Predicted k-space, shape (B, 2*C, H, W) or (B, C, H, W) for complex.
            target (torch.Tensor): Target k-space, same shape as pred.
            acquired (torch.Tensor, optional): ``[B, 1, H, W]`` plane of the lines the
                scanner wrote. When given, the never-acquired lines (exact zeros in the
                target, ~24 % of an M4Raw plane and ~32 % of this map's weight) carry no
                weight and the mean runs over the acquired bins only.

        Returns:
            torch.Tensor: Loss value (scalar or per-sample depending on reduction).

        Raises:
            RuntimeError: If shapes mismatch.
        """
        if pred.shape != target.shape:
            raise RuntimeError(f"Pred and target shape mismatch: {pred.shape} vs {target.shape}")

        # Handle both 4D (B,C,H,W) and 5D (B,C,D,H,W) tensors.
        # For 5D inputs (volumetric/multi-slice), collapse depth into batch.
        if pred.dim() == 5:
            depth = pred.shape[2]
            pred = _fold_depth_into_batch(pred)
            target = _fold_depth_into_batch(target)
            if acquired is not None:
                acquired = acquired.repeat_interleave(depth, dim=0)
        elif pred.dim() == 4:
            pass  # Already (B, C, H, W)
        else:
            raise RuntimeError(f"Expected 4D or 5D tensor, got {pred.dim()}D: {pred.shape}")

        # Get weight map dynamically based on spatial dims
        height, width = pred.shape[-2], pred.shape[-1]
        weight_map = self._get_weight_map(height, width, pred.device, pred.dtype)

        # Handle both stacked real/imag [B, 2*C, H, W] and complex tensors
        if pred.is_complex():
            # Modulus of the complex DIFFERENCE, as in the formula: ||pred| - |target||
            # would score a k-space phase ramp (an image shift) as a perfect match.
            unweighted_loss = torch.abs(pred - target)
        else:
            # Stacked real/imag: [B, 2*C, H, W]
            unweighted_loss = torch.nn.functional.l1_loss(pred, target, reduction="none")

        # Apply frequency weights: broadcast weight_map [1, 1, H, W] to loss [B, C, H, W]
        if acquired is not None:
            weight_map = weight_map * acquired.to(device=pred.device, dtype=weight_map.dtype)
        weighted_loss = unweighted_loss * weight_map

        if self.reduction == "mean":
            if acquired is not None:
                count = acquired.to(weighted_loss.dtype).expand_as(weighted_loss).sum()
                return weighted_loss.sum() / count.clamp_min(1.0)
            return weighted_loss.mean()
        if self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"default_height={self.default_height}, default_width={self.default_width}, "
            f"alpha={self.alpha}, reduction='{self.reduction}', profile='{self.profile}', "
            f"(dynamically adapts to input shape)"
        )


def _linear_profile(loss: FrequencyWeightedL1Loss, height: int, width: int) -> torch.Tensor:
    """The radial ramp ``1 + alpha * r`` renormalised to unit mean (the original map)."""
    return loss._create_frequency_weight_map(height, width, loss.alpha)


#: Weight-map builders keyed by ``profile``; an unknown name is rejected at construction.
_PROFILE_BUILDERS: dict[str, Callable[[FrequencyWeightedL1Loss, int, int], torch.Tensor]] = {
    "linear": _linear_profile,
    "ring_band": FrequencyWeightedL1Loss._ring_band_map,
}


@register_loss("sobolev_kspace")
class SobolevKSpaceLoss(nn.Module):
    """Fourier-Domain Sobolev Loss.

    By the derivative theorem of Fourier transforms, spatial derivatives map to
    multiplying k-space by frequency coordinates. This loss enforces:

    L_Sobolev = mean_{kx, ky} (1 + kx^2 + ky^2) * |K_pred(kx, ky) - K_true(kx, ky)|_1

    This natively up-weights high-frequency edge sharpness directly in k-space.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{SobolevKSpace} = \\mathbb{E}_{k_x, k_y} \\left[ (1 + k_x^2 + k_y^2)^{s/2} | \\hat{k} - k | \\right]
    """

    def __init__(
        self, order: float = 2.0, reduction: str = "mean", restrict_to_acquired: bool = False
    ):
        """Fourier-domain Sobolev loss of order ``order``.

        Args:
            order: The ``s`` in ``(1 + k_x^2 + k_y^2)^{s/2}``. The default 2.0
                reproduces the exponent this class hardcoded until 2026-08, so
                existing arms are unaffected. ``order=1.0`` is the
                ``sqrt(1 + |k|^2)`` weighting that 56 exp_11 arms asked for via
                the dead ``sobolev_order`` key (#560, #615); note
                ``sobolev_frequency`` is a separate registered loss weighting by
                ``(1 + |k|)`` exactly, which may be the closer intent.
                Immutable after construction — the weight-map cache is keyed on
                shape alone.
            reduction: Reduction passed to the underlying magnitude penalty.
            restrict_to_acquired: Score only the lines the scanner wrote, so the never-acquired
                zero lines do not dilute the mean. Off by default: it moves the loss scale of
                every arm that already declares this loss.
        """
        super().__init__()
        check_reduction(reduction, "sobolev_kspace")
        self.restrict_to_acquired = bool(restrict_to_acquired)
        self.order = float(order)
        self.reduction = reduction
        self._weight_map_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _get_weight_map(
        self, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """_get_weight_map.

        Args:
            height (int): Description.
            width (int): Description.
            device (torch.device): Description.
            dtype (torch.dtype): Description.
        Returns:
            torch.Tensor: Description.
        """
        cache_key = (height, width)
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        if cache_key in self._weight_map_cache:
            cached = self._weight_map_cache[cache_key]
            if cached.device == device and cached.dtype == real_dtype:
                return cached

        # Create meshgrid with frequencies normalized to [-1, 1]
        y = torch.arange(height, dtype=torch.float32) - height / 2.0
        x = torch.arange(width, dtype=torch.float32) - width / 2.0

        # Normalize to maximum frequency 1.0 at the edge
        y = y / (height / 2.0)
        x = x / (width / 2.0)

        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # W(kx, ky) = (1 + kx^2 + ky^2)^(s/2); s=2 is the historical hardcoded
        # case, so the default `order` leaves every existing arm unchanged.
        weight = (1.0 + xx**2 + yy**2) ** (self.order / 2.0)

        weight_map = weight.unsqueeze(0).unsqueeze(0)
        weight_map = weight_map.to(device=device, dtype=real_dtype)

        self._weight_map_cache[cache_key] = weight_map
        return weight_map

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        acquired: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """forward.

        Args:
            pred (torch.Tensor): Description.
            target (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SobolevKSpaceLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        if pred.shape != target.shape:
            raise RuntimeError(f"Shape mismatch: {pred.shape} vs {target.shape}")

        # Handle both 4D (B,C,H,W) and 5D (B,C,D,H,W) tensors.
        if pred.dim() == 5:
            depth = pred.shape[2]
            pred = _fold_depth_into_batch(pred)
            target = _fold_depth_into_batch(target)
            if acquired is not None:
                acquired = acquired.repeat_interleave(depth, dim=0)
        elif pred.dim() != 4:
            raise RuntimeError(f"Expected 4D or 5D tensor, got {pred.dim()}D: {pred.shape}")

        height, width = pred.shape[-2], pred.shape[-1]
        weight_map = self._get_weight_map(height, width, pred.device, pred.dtype)

        if pred.is_complex():
            diff_mag = torch.abs(pred - target)
        else:
            if pred.ndim == 4 and pred.shape[1] % 2 == 0:
                # Real-stacked: compute magnitude of diff
                B, C, H, W = pred.shape
                diff = pred - target
                diff_reshaped = diff.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
                diff_complex = torch.view_as_complex(diff_reshaped).permute(0, 3, 1, 2)
                diff_mag = torch.abs(diff_complex)
            else:
                diff_mag = torch.abs(pred - target)

        weighted_loss = diff_mag * weight_map

        if self.restrict_to_acquired:
            if acquired is None:
                raise ValueError(
                    "sobolev_kspace has restrict_to_acquired=True but no `acquired` plane reached it."
                )
            plane = acquired.to(device=weighted_loss.device, dtype=weighted_loss.dtype)
            weighted_loss = weighted_loss * plane
            if self.reduction == "mean":
                return weighted_loss.sum() / plane.expand_as(weighted_loss).sum().clamp_min(1.0)

        if self.reduction == "mean":
            return weighted_loss.mean()
        if self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss
