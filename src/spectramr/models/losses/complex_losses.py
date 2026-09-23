"""Complex Loss Functions
======================

This module contains loss functions specifically designed for complex-valued data,
addressing phase blindness issues in MRI reconstruction.
"""

import logging

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.registry import register_loss

logger = logging.getLogger(__name__)


@register_loss(name="complex_l1", aliases=["ComplexL1Loss"], domain="agnostic")
class ComplexL1Loss(nn.Module):
    r"""L1 Loss for complex-valued data.

    **DOMAIN**: K-SPACE NATIVE / AGNOSTIC ✓
    **Input**: [B, 2, H, W] or torch.complex64 (k-space or image)
    **3D Support**: Yes

    Computes mean absolute error between complex tensors via magnitude:
    loss = mean(sqrt((real_diff)² + (imag_diff)²))

    Works on both k-space and image domains. Phase errors correctly penalized.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{ComplexL1} = \| \hat{y} - y \|_1 = \sqrt{ (\Re(\hat{y}) - \Re(y))^2 + (\Im(\hat{y}) - \Im(y))^2 + \epsilon^2 }
    """

    def __init__(self, reduction: str = "mean"):
        """__init__.

        Args:
            reduction (str): Description.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute complex L1 loss.

        Args:
            input: Input tensor (complex or real)
            target: Target tensor (complex or real)

        Returns:
            Loss value

        forward method for ComplexL1Loss.

        Executes PyTorch tensor operations.

        Args:
            input (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        # DEBUG: Check input types before conversion
        logger.debug(
            f"[ComplexL1Loss ENTRY] input: shape={input.shape}, complex={torch.is_complex(input)}, "
            f"target: shape={target.shape}, complex={torch.is_complex(target)}"
        )

        # Ensure inputs are complex
        # 1. Handle Channel-First REAL: [B, C, H, W] -> Complex (C must be even)
        if not input.is_complex() and input.ndim == 4 and input.shape[1] % 2 == 0:
            B, C, H, W = input.shape
            input_reshaped = input.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            input = torch.view_as_complex(input_reshaped).permute(0, 3, 1, 2)
        # 2. Handle Channel-Last REAL: [..., 2] -> Complex (Legacy support)
        elif not input.is_complex() and input.shape[-1] == 2:
            input = torch.view_as_complex(input)

        # Handle target: might be real [B, C, H, W] or already complex [B, C_complex, H, W]
        if not target.is_complex() and target.ndim == 4 and target.shape[1] % 2 == 0:
            # Real tensor with even channels - convert
            B, C, H, W = target.shape
            target_reshaped = target.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            target = torch.view_as_complex(target_reshaped).permute(0, 3, 1, 2)
        elif not target.is_complex() and target.shape[-1] == 2:
            # Real tensor with last dim=2 encoding
            target = torch.view_as_complex(target)
        elif target.is_complex() and target.ndim == 4 and target.shape[1] % 2 == 0:
            # CRITICAL FIX: Target is already complex but with wrong channel structure
            # This happens when target comes from physics ops that output complex64
            # with odd number of channels. We need to ensure it matches input shape.
            # For now, keep it as-is (it's already the right type)
            pass

        # DEBUG: Check after conversion
        logger.debug(
            f"[ComplexL1Loss AFTER_CONVERT] input: shape={input.shape}, complex={torch.is_complex(input)}, "
            f"target: shape={target.shape}, complex={torch.is_complex(target)}"
        )

        # Handle channel mismatch
        if input.shape != target.shape:
            logger.warning(
                f"[ComplexL1Loss] Shape mismatch: input={input.shape}, target={target.shape}. "
                f"Attempting to fix..."
            )
            # If input has fewer channels than target, expand
            if input.shape[1] < target.shape[1]:
                # Repeat input channels to match target
                ratio = target.shape[1] / input.shape[1]
                if ratio == int(ratio):
                    input = input.repeat(1, int(ratio), 1, 1)
            # If target has fewer, repeat target
            elif target.shape[1] < input.shape[1]:
                ratio = input.shape[1] / target.shape[1]
                if ratio == int(ratio):
                    target = target.repeat(1, int(ratio), 1, 1)

            # If still mismatched, slice to same size
            if input.shape[1] != target.shape[1]:
                min_c = min(input.shape[1], target.shape[1])
                input = input[:, :min_c, :, :]
                target = target[:, :min_c, :, :]
                logger.warning(
                    f"[ComplexL1Loss] Sliced to minimum channels: {min_c}. New shapes: "
                    f"input={input.shape}, target={target.shape}"
                )

        # Compute difference
        diff = input - target

        # [AUDIT FIX] Numerical Stability
        eps = 1e-8
        if torch.is_complex(diff):
            loss = torch.sqrt(diff.real.pow(2) + diff.imag.pow(2) + eps**2)
        else:
            # Handle real inputs (imag is 0)
            loss = torch.sqrt(diff.pow(2) + eps**2)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


@register_loss(name="weighted_kspace_l1", aliases=["WeightedKSpaceL1Loss"], domain="kspace")
class WeightedKSpaceL1Loss(nn.Module):
    r"""Frequency-Weighted L1 Loss for K-Space data.

    **DOMAIN**: K-SPACE NATIVE ✓
    **Input**: [B, 2, H, W] or torch.complex64 (k-space only)
    **3D Support**: Yes

    Upweights high-frequency components to counteract spectral bias (1/f decay).
    Weight: W(k) = (1 + normalized_radius)^exponent

    Addresses natural signal spectral bias for balanced reconstruction.

    Attributes:
        exponent: Power (alpha). Higher = more high-freq emphasis.
        reduction: 'mean' or 'sum'.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{WeightedKSpaceL1} = W(k_x, k_y) \odot | \hat{k} - k |"""

    def __init__(self, exponent: float = 1.0, reduction: str = "mean"):
        """__init__.

        Args:
            exponent (float): Description.
            reduction (str): Description.
        """
        super().__init__()
        self.exponent = exponent
        self.reduction = reduction
        self._weight_cache = {}

    def _get_weight_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generate or retrieve pre-calculated frequency weight mask."""
        key = (height, width, str(device))
        if key not in self._weight_cache:
            # Create grid of distances from center
            y = torch.arange(height, device=device).float()
            x = torch.arange(width, device=device).float()

            # Center coordinates
            cy, cx = height / 2.0, width / 2.0

            # Create meshgrid
            # Normalize coordinates to [-1, 1] for resolution independence
            y_norm = (y - cy) / cy
            x_norm = (x - cx) / cx

            Y, X = torch.meshgrid(y_norm, x_norm, indexing="ij")

            # Distance from center (radius)
            R = torch.sqrt(Y**2 + X**2)

            # Weight formula: (1 + R)^exponent
            # Base weight is 1.0 (at DC), grows with radius
            W = (1.0 + R) ** self.exponent

            self._weight_cache[key] = W

        return self._weight_cache[key]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted complex L1 loss.

        Args:
            input: Input k-space (complex)
            target: Target k-space (complex)

        forward method for WeightedKSpaceL1Loss.

        Executes PyTorch tensor operations.

        Args:
            input (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        # Ensure inputs are complex
        if not input.is_complex() and input.ndim == 4 and input.shape[1] % 2 == 0:
            B, C, H, W = input.shape
            input_reshaped = input.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            input = torch.view_as_complex(input_reshaped).permute(0, 3, 1, 2)
        elif not input.is_complex() and input.shape[-1] == 2:
            input = torch.view_as_complex(input)

        if not target.is_complex() and target.ndim == 4 and target.shape[1] % 2 == 0:
            B, C, H, W = target.shape
            target_reshaped = target.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            target = torch.view_as_complex(target_reshaped).permute(0, 3, 1, 2)
        elif not target.is_complex() and target.shape[-1] == 2:
            target = torch.view_as_complex(target)

        # Basic difference magnitude
        diff_mag = torch.abs(input - target)

        # Get weight mask
        H, W = diff_mag.shape[-2:]
        weight_mask = self._get_weight_mask(H, W, diff_mag.device)

        # Apply weights (broadcast over batch/channels)
        weighted_diff = diff_mag * weight_mask

        if self.reduction == "mean":
            return weighted_diff.mean()
        elif self.reduction == "sum":
            return weighted_diff.sum()
        else:
            return weighted_diff


@register_loss(name="log_spectral_phase", aliases=["LogSpectralPhaseLoss"])
class LogSpectralPhaseLoss(nn.Module):
    """Log-Spectral Phase Loss (Frequency-Weighted).

    Combines log-compressed magnitude loss with magnitude-weighted phase loss.
    Designed for k-space reconstruction where dynamic range is large and phase is critical.

    Loss = L1(log(1+|k_pred|), log(1+|k_target|)) + lambda * weighted_phase_diff
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{LogSpectralPhase} = | \\log(1 + |\\hat{y}|) - \\log(1 + |y|) | + \\lambda_{phase} W_{phase} \\odot | e^{j \angle \\hat{y}} - e^{j \angle y} |
    """

    def __init__(self, phase_weight: float = 0.1, reduction: str = "mean"):
        """__init__.

        Args:
            phase_weight (float): Description.
            reduction (str): Description.
        """
        super().__init__()
        self.phase_weight = phase_weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Complex tensor [B, C, H, W] (or [B, 2, H, W] as real/imag)
            target: Complex tensor [B, C, H, W]

        forward method for LogSpectralPhaseLoss.

        Executes PyTorch tensor operations.

        Args:
            input (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        # Ensure inputs are complex
        if not input.is_complex() and input.shape[1] % 2 == 0:
            B, C, H, W = input.shape
            input_reshaped = input.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            input_complex = torch.view_as_complex(input_reshaped).permute(0, 3, 1, 2)
        else:
            input_complex = input

        if not target.is_complex() and target.shape[1] % 2 == 0:
            B, C, H, W = target.shape
            target_reshaped = target.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            target_complex = torch.view_as_complex(target_reshaped).permute(0, 3, 1, 2)
        else:
            target_complex = target

        # [AUDIT FIX] Numerical Stability: Stabilized Magnitude
        # Mag = sqrt(real^2 + imag^2 + eps^2)
        eps = 1e-8

        def stable_mag(c: torch.Tensor) -> torch.Tensor:
            """stable_mag.

            Args:
                c (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            return torch.sqrt(c.real.pow(2) + c.imag.pow(2) + eps**2)

        input_mag = stable_mag(input_complex)
        target_mag = stable_mag(target_complex)

        # 1. Log-Magnitude Loss
        # Compress dynamic range: log(1 + |k|)
        input_log = torch.log1p(input_mag)
        target_log = torch.log1p(target_mag)

        mag_loss = torch.abs(input_log - target_log)

        # 2. Phase Loss (Weighted by Target Magnitude)
        # Phase coherence matters most where signal is strong
        input_phase = torch.angle(input_complex)
        target_phase = torch.angle(target_complex)

        # Phase difference on unit circle: |e^(i*phi1) - e^(i*phi2)|
        # This handles 2pi wrap-around correctly
        # [AUDIT FIX] Stabilize phase diff magnitude
        phase_vec_diff = torch.exp(1j * input_phase) - torch.exp(1j * target_phase)
        phase_diff = stable_mag(phase_vec_diff)

        # Weight phase error by target magnitude (don't care about phase of noise)
        # weight_map = |k| / (max(|k|) + eps)
        # Using stabilized magnitude for weight map as well
        weight_map = target_mag / (torch.max(target_mag) + 1e-6)

        weighted_phase_loss = phase_diff * weight_map

        # Combine
        total_loss = mag_loss + self.phase_weight * weighted_phase_loss

        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        else:
            return total_loss


@register_loss(name="complex_mse", aliases=["complex_l2", "ComplexMSELoss"])
class ComplexMSELoss(nn.Module):
    r"""MSE Loss for complex-valued data.

    Computes |z|^2 = Re^2 + Im^2 for the difference.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{ComplexMSE} = (\Re(\hat{y}) - \Re(y))^2 + (\Im(\hat{y}) - \Im(y))^2
    """

    def __init__(self, reduction: str = "mean"):
        """__init__.

        Args:
            reduction (str): Description.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute complex MSE loss.

        forward method for ComplexMSELoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        # Handle 2-channel format
        if not pred.is_complex() and pred.ndim == 4 and pred.shape[1] % 2 == 0:
            B, C, H, W = pred.shape
            pred_reshaped = pred.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            pred = torch.view_as_complex(pred_reshaped).permute(0, 3, 1, 2)
        if not target.is_complex() and target.ndim == 4 and target.shape[1] % 2 == 0:
            B, C, H, W = target.shape
            target_reshaped = target.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            target = torch.view_as_complex(target_reshaped).permute(0, 3, 1, 2)

        if torch.is_complex(pred):
            diff = pred - target
            # |z|^2 = Re^2 + Im^2
            loss = diff.real**2 + diff.imag**2
        else:
            loss = (pred - target) ** 2

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@register_loss(
    name="frequency_domain_consistency",
    aliases=["FrequencyDomainLoss", "frequency_domain"],
)
class FrequencyDomainLoss(nn.Module):
    r"""Frequency Domain (K-Space) Consistency Loss.

    Penalizes errors in k-space for physics consistency.
    Projects images to frequency domain and compares.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{FrequencyDomain} = \| \mathcal{F}(\hat{y}) - \mathcal{F}(y) \|_2"""

    def __init__(self, use_log: bool = True, reduction: str = "mean"):
        """__init__.

        Args:
            use_log (bool): Description.
            reduction (str): Description.
        """
        super().__init__()
        self.use_log = use_log
        self.reduction = reduction

    def forward(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """Compute frequency domain loss.

        Args:
            pred_img: Predicted image [B, C, H, W]
            target_img: Target image [B, C, H, W]

        forward method for FrequencyDomainLoss.

        Executes PyTorch tensor operations.

        Args:
            pred_img (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_img (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        # Handle complex images
        if torch.is_complex(pred_img):
            pred_img = torch.abs(pred_img)
        if torch.is_complex(target_img):
            target_img = torch.abs(target_img)

        # Project to k-space using canonical centered FFT
        pred_k = fft2c(pred_img)
        target_k = fft2c(target_img)

        if self.use_log:
            # Log-magnitude to weight high-frequencies more evenly
            pred_k = torch.log(torch.abs(pred_k) + 1e-8)
            target_k = torch.log(torch.abs(target_k) + 1e-8)
            loss = torch.abs(pred_k - target_k)
        else:
            # Complex difference magnitude
            diff = pred_k - target_k
            loss = torch.sqrt(diff.real**2 + diff.imag**2 + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@register_loss(
    name="sense_adjoint_l1",
    aliases=["SENSESpatialLoss"],
    domain="kspace",
    compatible_with=["kspace", "image"],
)
class SENSEAdjointL1Loss(nn.Module):
    """SENSE Image-Domain Complex Loss as a Module.

    Acts as a bridge to compute_sense_loss with unified signature support.
    Expected to receive 'smaps' injected automatically via _call_safe_loss.

    The loss bridges from k-space itself (``use_fourier_bridge=True``, the
    default): it iFFTs the prediction before the coil-adjoint combination. It
    must therefore be declared under ``losses.kspace_losses`` so the LossBuilder
    adds no second bridge — see issue #467 and ``docs/losses_reference.rst``
    (``docs/loss_domain_bridging.rst``, cited here since #467, was never written).
    """

    def __init__(
        self,
        reduction: str = "mean",
        use_fourier_bridge: bool = True,
        normalize: bool = False,
        min_support_frac: float = 1e-2,
    ):
        """__init__.

        Args:
            reduction (str): Description.
            use_fourier_bridge (bool): When True (default) ``pred``/``target``
                are k-space and this loss applies the iFFT itself. Set False when
                an outer bridge (LossBuilder's ``image_losses`` /
                ``complex_losses`` wrapper) has already moved them to the image
                domain, so the transform is applied exactly once. The attribute
                is what LossBuilder inspects to reject a double bridge.
            normalize (bool): Divide the coil combination by ``sum_c |S_c|^2``,
                i.e. score the Roemer estimate of the magnetization rather than
                the matched-filter adjoint. The adjoint's output is
                ``(sum_c|S_c|^2) * m``, so without this the term weights the
                image by the coil intensity profile -- a spatial weight on a
                *fidelity* term that nobody chose and that varies per subject.
                Defaults to ``False`` because flipping it moves the loss on every
                declaring arm; it is opted into per-arm from the YAML ``kwargs:``
                so the recipe stays visible in the config.
            min_support_frac (float): Forwarded to ``coil_combine_sense``; inert
                unless ``normalize``. Outside the object ``sum_c|S_c|^2 -> 0``,
                and an unprotected divide there makes amplified air noise the
                dominant term.
        """
        super().__init__()
        self.reduction = reduction
        self.use_fourier_bridge = use_fourier_bridge
        self.normalize = normalize
        self.min_support_frac = float(min_support_frac)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smaps: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """forward.

        Args:
            pred (torch.Tensor): Description.
            target (torch.Tensor): Description.
            smaps (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SENSEAdjointL1Loss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            smaps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        if smaps is None:
            # 2026-05-30 DC-blob fix: a fidelity term that SILENTLY returns
            # ``pred.sum() * 0.0`` when smaps is absent is the exact
            # root-cause pattern of the DC blob — a data-fidelity term
            # multiplied by zero so the network is never pressured toward a
            # measurement-consistent output. We keep the zero ONLY for the
            # explicitly-flagged startup/no-coil context (eval mode, or an
            # explicit ``allow_missing_smaps`` opt-out) and RAISE during real
            # training so a 0.3-weighted SENSE-adjoint fidelity term can never
            # silently vanish (CLAUDE.md pitfalls #9 / #15).
            if getattr(self, "allow_missing_smaps", False) or not self.training:
                import logging as _logging

                _logging.getLogger(__name__).debug(
                    "SENSEAdjointL1Loss: smaps absent in startup/no-coil "
                    "context (eval or allow_missing_smaps); returning zero."
                )
                return pred.sum() * 0.0

            raise ValueError(
                "SENSEAdjointL1Loss received smaps=None during training. This "
                "SENSE-adjoint data-fidelity term would silently contribute "
                "zero (the DC-blob root-cause pattern: a fidelity term "
                "multiplied by zero). Coil sensitivities must be injected into "
                "the loss kwargs at train time (via _call_safe_loss from the "
                "batch). Set the instance attribute allow_missing_smaps=True "
                "only for an explicit no-coil run."
            )

        # Ensure complex tensors
        if not pred.is_complex() and pred.shape[1] % 2 == 0:
            B, C, H, W = pred.shape
            pred_complex = torch.view_as_complex(
                pred.permute(0, 2, 3, 1)
                .contiguous()
                .view(B, H, W, C // 2, 2)
                .permute(0, 3, 1, 2, 4)
                .contiguous()
            )
        else:
            pred_complex = pred

        if not target.is_complex() and target.shape[1] % 2 == 0:
            B, C, H, W = target.shape
            target_complex = torch.view_as_complex(
                target.permute(0, 2, 3, 1)
                .contiguous()
                .view(B, H, W, C // 2, 2)
                .permute(0, 3, 1, 2, 4)
                .contiguous()
            )
        else:
            target_complex = target

        if not smaps.is_complex() and smaps.shape[1] % 2 == 0:
            B, C, H, W = smaps.shape
            smaps_complex = torch.view_as_complex(
                smaps.permute(0, 2, 3, 1)
                .contiguous()
                .view(B, H, W, C // 2, 2)
                .permute(0, 3, 1, 2, 4)
                .contiguous()
            )
        else:
            smaps_complex = smaps

        # Finite-guard the coil maps before they scale the prediction: a
        # non-finite smaps (singular/rank-deficient ESPIRiT) would poison this
        # weight-bearing fidelity gradient and diverge training. Mirror the
        # None-guard above: raise at train time, tolerate at eval (#9/#16).
        if self.training and not torch.isfinite(torch.view_as_real(smaps_complex)).all():
            raise ValueError(
                "SENSEAdjointL1Loss received non-finite smaps during training "
                "(NaN/Inf coil maps, e.g. rank-deficient ESPIRiT calibration). "
                "Refusing to poison the fidelity gradient; fix coil estimation."
            )

        from spectramr.infrastructure.physics.fft_ops import ifft2c

        # 1. Transform pure complex k-space to spatial domain FIRST.
        # Skipped when an outer bridge already did it (use_fourier_bridge=False),
        # so the iFFT is applied exactly once on the loss path (issue #467).
        if self.use_fourier_bridge:
            img_pred = ifft2c(pred_complex)
            img_target = ifft2c(target_complex)
        else:
            img_pred = pred_complex
            img_target = target_complex

        # 2. Coil combination -> a single complex image. The matched-filter form
        # is A^H and carries a sum|S|^2 shading; ``normalize`` divides it out
        # through the owner of that divide. Slicing first handles the
        # cross-contrast mismatch, where the model emits fewer coils than smaps.
        pred_coils = img_pred.shape[1]
        smap_coils = smaps_complex.shape[1]
        if smap_coils != pred_coils:
            smaps_complex = smaps_complex[:, :pred_coils]
        if self.normalize:
            from spectramr.infrastructure.physics.coil_sensitivity import coil_combine_sense

            sense_pred = coil_combine_sense(
                img_pred, smaps_complex, min_support_frac=self.min_support_frac
            )
            sense_target = coil_combine_sense(
                img_target, smaps_complex, min_support_frac=self.min_support_frac
            )
        else:
            sense_pred = torch.sum(img_pred * smaps_complex.conj(), dim=1, keepdim=True)
            sense_target = torch.sum(img_target * smaps_complex.conj(), dim=1, keepdim=True)

        # 3. Direct Complex L1 Loss on the true anatomical manifold (NO RSS!)
        loss = torch.nn.functional.l1_loss(
            torch.view_as_real(sense_pred),
            torch.view_as_real(sense_target),
            reduction=self.reduction,
        )

        return loss
