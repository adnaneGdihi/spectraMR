"""Data Consistency (DC) layers for MRI reconstruction
===================================================

Provides simple DC modules compatible with Cartesian sampling and optional
multi-coil sensitivity maps.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from spectramr.infrastructure.physics.dc_mask import align_dc_mask
from spectramr.infrastructure.physics.dc_settings import SUPPORTED_NOISE_TYPES

from .fft_ops import _to_complex, fft2c, ifft2c

logger = logging.getLogger(__name__)

#: Single advertised set of data-consistency methods (SSOT). Both the
#: model-internal DC builder (``KSpaceColdDiffusionGenerator``) and the
#: reverse-diffusion sampler (``PhysicsInformedColdDiffusion``) validate against
#: this one frozenset so a method one accepts, the other can honour. ``hard`` /
#: ``soft`` (and its alias ``noise_adjusted``) are closed-form k-space ops; the
#: rest are learned ``nn.Module`` layers reused from the model at sampling time.
#: Divergence here is what caused the 2026-07-05 ``dc_method='adaptive'`` sampler
#: crash (generator built ``AdaptiveDataConsistency``; sampler knew only hard/soft).
#: ``noise_adaptive`` is a closed-form Wiener/SNR trust layer (``NoiseAdaptiveDataConsistency``);
#: unlike the ``noise_adjusted`` alias (which just maps to soft) it genuinely adapts the
#: measurement trust per k-space bin from the measured SNR — denoising the low-field periphery.
VALID_DC_METHODS: frozenset[str] = frozenset(
    {
        "hard",
        "soft",
        "noise_adjusted",
        "adaptive",
        "kan_adaptive",
        "target_aware_fsdc",
        "noise_adaptive",
    }
)


def _validate_noise_type(noise_type: str, owner: str) -> str:
    """Normalise and validate a ``noise_type``, raising on anything unsupported.

    Three DC layers take this parameter and, before #1525, applied three
    different policies to it: one validated, one stored it unchecked, and one
    accepted it and never stored it at all. One owner, one policy (NN17), and an
    unsupported value RAISES rather than degrading to Gaussian (NN3).

    Args:
        noise_type: The declared noise model.
        owner: Class name, so the message names the layer that rejected it.

    Returns:
        The lower-cased, validated noise type.

    Raises:
        ValueError: If ``noise_type`` is not implemented by any DC layer.
    """
    resolved = str(noise_type).lower()
    if resolved not in SUPPORTED_NOISE_TYPES:
        raise ValueError(
            f"{owner}: unsupported noise_type={noise_type!r}; "
            f"valid values: {sorted(SUPPORTED_NOISE_TYPES)}"
        )
    return resolved


class NoiseSimulatingDataConsistency(nn.Module):
    r"""
    Physics-Informed Data Consistency Layer with Realistic Noise Simulation.
    """

    def __init__(
        self,
        train_noise_level: float = 0.01,
        eval_noise_level: float = 0.005,
        noise_type: str = "gaussian",
        noise_lvl: float | None = None,
    ):
        """__init__.

        Args:
            train_noise_level (float): Description.
            eval_noise_level (float): Description.
            noise_type (str): Description.
            noise_lvl (float | None): Description.
        """
        super().__init__()
        if noise_lvl is not None:
            train_noise_level = noise_lvl
            eval_noise_level = noise_lvl

        self.train_noise_level = train_noise_level
        self.eval_noise_level = eval_noise_level
        self.noise_type = _validate_noise_type(noise_type, type(self).__name__)
        self.noise_lvl = noise_lvl

    def forward(self, predicted_img, measured_kspace, mask):
        # 1. Transform estimate to k-space
        """forward.

        Args:
            predicted_img (Any): Description.
            measured_kspace (Any): Description.
            mask (Any): Description.
        Returns:
            Any: Description.

        forward method for NoiseSimulatingDataConsistency.

        Executes PyTorch tensor operations.

        Args:
            predicted_img (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            measured_kspace (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        if not torch.is_complex(predicted_img) and predicted_img.shape[1] % 2 == 0:
            # Interleaved parsing
            real = predicted_img[:, 0::2, ...]
            imag = predicted_img[:, 1::2, ...]
            x_complex = torch.complex(real, imag)
        elif not torch.is_complex(predicted_img):
            x_complex = _to_complex(predicted_img)
        else:
            x_complex = predicted_img

        kspace_pred = fft2c(x_complex)

        # 2. Enforce Consistency
        if mask.dim() < kspace_pred.dim():
            while mask.dim() < kspace_pred.dim():
                mask = mask.unsqueeze(1)

        if not torch.is_complex(measured_kspace):
            if measured_kspace.shape[1] % 2 == 0:
                real = measured_kspace[:, 0::2, ...]
                imag = measured_kspace[:, 1::2, ...]
                measured_kspace = torch.complex(real, imag)
            else:
                measured_kspace = _to_complex(measured_kspace)

        if torch.is_complex(mask):
            mask = mask.real

        # Ensure mask is float for arithmetic (bool tensors don't support subtraction)
        if mask.dtype == torch.bool:
            mask = mask.float()

        measured_kspace_noisy = self._add_realistic_noise(measured_kspace)
        kspace_consistent = (1 - mask) * kspace_pred + mask * measured_kspace_noisy

        # 3. Return to image space
        image_out = ifft2c(kspace_consistent)

        # Convert back to original format (Interleaved)
        if not torch.is_complex(predicted_img):
            if predicted_img.shape[1] % 2 == 0:
                image_out = torch.stack([image_out.real, image_out.imag], dim=2).flatten(1, 2)
            else:
                image_out = image_out.abs()

        return image_out

    def _add_realistic_noise(self, kspace: torch.Tensor) -> torch.Tensor:
        """_add_realistic_noise.

        Args:
            kspace (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        noise_std = self.train_noise_level if self.training else self.eval_noise_level
        if noise_std == 0.0:
            return kspace
        # noise_type is validated to be 'gaussian' in __init__; no silent fallback.
        return kspace + torch.randn_like(kspace) * noise_std


from .implementations.fft_operator import MultiCoilFFTOperator


class AdaptiveDataConsistency(nn.Module):
    """Adaptive Data Consistency with learnable noise weighting."""

    def __init__(self):
        """__init__."""
        super().__init__()
        self.noise_estimator = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.mc_operator = MultiCoilFFTOperator(normalized=True, center=True)

    def forward(
        self,
        img_pred: torch.Tensor,
        measured_kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        sensitivity_maps: torch.Tensor | None = None,
        is_kspace_domain: bool = False,
        acs_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply adaptive data consistency.

        Supports both image-domain and k-space-domain inputs. When the input
        is already in k-space (``is_kspace_domain=True``), the FFT step is
        skipped to avoid double-FFT corruption.

        Args:
            img_pred: Predicted tensor — image ``[B, C, H, W]`` or k-space
                (interleaved real/imag channels).
            measured_kspace: Acquired k-space measurements.
            mask: Sampling mask ``[B, 1, H, W]``.
            sensitivity_maps: Optional coil sensitivity maps.
            is_kspace_domain: If ``True``, treat *img_pred* as k-space and
                skip the forward FFT.
            acs_mask: Optional ACS-region mask (accepted for API compat,
                currently unused).

        Returns:
            Data-consistent tensor in the same domain as *img_pred*.

        forward method for AdaptiveDataConsistency.

        Executes PyTorch tensor operations.

        Args:
            img_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measured_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            sensitivity_maps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            is_kspace_domain (bool): Expected input tensor.
            acs_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        if measured_kspace is None or mask is None:
            return img_pred

        # Auto-detect k-space domain when not explicitly set
        if not is_kspace_domain:
            is_kspace_domain = (
                not torch.is_complex(img_pred)
                and img_pred.ndim == 4
                and img_pred.shape[1] % 2 == 0
                and measured_kspace is not None
                and not torch.is_complex(measured_kspace)
                and measured_kspace.ndim == 4
                and measured_kspace.shape[1] == img_pred.shape[1]
            )

        is_complex_input = torch.is_complex(img_pred)

        # Convert prediction to complex
        if not is_complex_input and img_pred.shape[1] % 2 == 0:
            real = img_pred[:, 0::2, ...]
            imag = img_pred[:, 1::2, ...]
            x = torch.complex(real, imag)
        else:
            x = _to_complex(img_pred)

        # Get k-space prediction: skip FFT if already in k-space domain
        if is_kspace_domain:
            k_pred = x
        elif sensitivity_maps is not None:
            k_pred = self.mc_operator.forward(x, smaps=sensitivity_maps, mask=None)
        else:
            k_pred = fft2c(x)

        # Convert measured k-space to complex
        if not torch.is_complex(measured_kspace):
            real = measured_kspace[:, 0::2, ...]
            imag = measured_kspace[:, 1::2, ...]
            measured_kspace = torch.complex(real, imag)

        # Estimate spatially-varying noise weight from measured data. The mask is
        # aligned against the MEASURED channel count here and against the PREDICTED
        # one below, because a SENSE forward can leave the two different.
        mag = torch.abs(measured_kspace) + 1e-8
        k_mag = torch.log(mag)
        if mask is not None:
            k_mag = k_mag * align_dc_mask(mask, measured_kspace.shape[-3])
        if k_mag.shape[1] > 1:
            k_mag = k_mag.mean(dim=1, keepdim=True)

        lambda_map = self.noise_estimator(k_mag)
        k_new = (1.0 - lambda_map) * k_pred + lambda_map * measured_kspace

        k_out = torch.where(align_dc_mask(mask, k_pred.shape[-3]).bool(), k_new, k_pred)

        # Convert back to output domain
        if is_kspace_domain:
            # Stay in k-space — no IFFT needed
            if is_complex_input:
                return k_out
            elif img_pred.shape[1] % 2 == 0:
                return torch.stack([k_out.real, k_out.imag], dim=2).flatten(1, 2)
            return k_out.real

        # Image-domain output path
        if sensitivity_maps is not None:
            x_out = self.mc_operator.adjoint(k_out, smaps=sensitivity_maps, combine_method="sense")
        else:
            x_out = ifft2c(k_out)

        if is_complex_input:
            return x_out
        elif img_pred.shape[1] % 2 == 0:
            return torch.stack([x_out.real, x_out.imag], dim=2).flatten(1, 2)
        return x_out.real


class SimpleDataConsistency(AdaptiveDataConsistency):
    """Backward compatibility wrapper."""

    def __init__(
        self,
        weight: float = 1.0,
        method: str = "hard",
        train_noise_level: float = 0.01,
        eval_noise_level: float = 0.005,
        noise_type: str = "gaussian",
        **kwargs,
    ):
        """__init__.

        Args:
            weight (float): Description.
            method (str): Description.
            train_noise_level (float): Description.
            eval_noise_level (float): Description.
            noise_type (str): Description.
        """
        super(AdaptiveDataConsistency, self).__init__()
        self.weight = nn.Parameter(torch.tensor(float(weight)))
        self.method = method
        self.train_noise_level = train_noise_level
        self.eval_noise_level = eval_noise_level
        # Validated rather than merely lower-cased: an unsupported value used to
        # be stored and then ignored by ``_add_realistic_noise``, which is the
        # silent-fallback shape non-negotiable 3 forbids (#1525).
        self.noise_type = _validate_noise_type(noise_type, type(self).__name__)

    def forward(
        self,
        img_pred,
        measured_kspace=None,
        mask=None,
        kspace_obs=None,
        is_kspace_domain=False,
        acs_mask=None,
    ):
        """forward.

        Args:
            img_pred (Any): Description.
            measured_kspace (Any): Description.
            mask (Any): Description.
            kspace_obs (Any): Description.
            is_kspace_domain (Any): Description.
            acs_mask (Any): Description.
        Returns:
            Any: Description.
        """
        if kspace_obs is not None and measured_kspace is None:
            measured_kspace = kspace_obs
        if measured_kspace is None or mask is None:
            return img_pred

        if not is_kspace_domain:
            is_kspace_domain = (
                not torch.is_complex(img_pred)
                and img_pred.ndim == 4
                and img_pred.shape[1] % 2 == 0
                and measured_kspace is not None
                and not torch.is_complex(measured_kspace)
                and measured_kspace.ndim == 4
                and measured_kspace.shape[1] == img_pred.shape[1]
            )

        lambda_val = torch.abs(self.weight)

        if is_kspace_domain:
            # K-SPACE Path
            if mask.dim() < img_pred.dim():
                mask = mask.unsqueeze(1)
            mask_num = mask.float()

            k_unsampled = img_pred * (1.0 - mask_num)
            if self.method == "hard":
                # Hard DC: the measurement replaces the prediction exactly at
                # sampled lines (the λ → ∞ limit). ``self.method`` was previously
                # stored but never read, so 'hard' silently produced the soft
                # blend below — an inert knob (CLAUDE.md pitfall #15).
                k_sampled = mask_num * measured_kspace
            else:
                # Soft proximal blend.
                k_sampled = mask_num * (
                    (img_pred + lambda_val * measured_kspace) / (1.0 + lambda_val)
                )
            return k_unsampled + k_sampled

        # IMAGE Path
        if not torch.is_complex(img_pred) and img_pred.shape[1] % 2 == 0:
            real = img_pred[:, 0::2, ...]
            imag = img_pred[:, 1::2, ...]
            x = torch.complex(real, imag)
        else:
            x = _to_complex(img_pred)

        k = fft2c(x)
        if not torch.is_complex(measured_kspace):
            if measured_kspace.shape[1] % 2 == 0:
                real = measured_kspace[:, 0::2, ...]
                imag = measured_kspace[:, 1::2, ...]
                measured_k = torch.complex(real, imag)
            else:
                measured_k = _to_complex(measured_kspace)
        else:
            measured_k = measured_kspace
            if measured_k.ndim == 3:
                measured_k = measured_k.unsqueeze(1)

        measured_k = self._add_realistic_noise(measured_k)

        while mask.dim() < k.dim():
            mask = mask.unsqueeze(1)
        mask_num = mask.float()
        if mask_num.ndim == 4 and mask_num.shape[1] > k.shape[1]:
            mask_num = mask_num[:, : k.shape[1], ...]

        k_unsampled = k * (1.0 - mask_num)
        if self.method == "hard":
            # Hard DC: exact measurement replacement at sampled lines (see the
            # k-space branch above — same inert-knob fix, pitfall #15).
            k_sampled = mask_num * measured_k
        else:
            k_sampled = mask_num * ((k + lambda_val * measured_k) / (1.0 + lambda_val))
        k_out = k_unsampled + k_sampled

        x_out = ifft2c(k_out)
        if torch.is_complex(img_pred):
            return x_out
        elif img_pred.shape[1] % 2 == 0:
            return torch.stack([x_out.real, x_out.imag], dim=2).flatten(1, 2)
        return x_out.real

    def _add_realistic_noise(self, kspace: torch.Tensor) -> torch.Tensor:
        """_add_realistic_noise.

        Args:
            kspace (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        noise_std = self.train_noise_level if self.training else self.eval_noise_level
        if noise_std == 0.0:
            return kspace
        # noise_type is validated to be 'gaussian' in __init__; no silent fallback.
        return kspace + torch.randn_like(kspace) * noise_std


class HardDataConsistency(nn.Module):
    """Hard Data Consistency Layer."""

    def __init__(
        self,
        mask=None,
        kspace_obs=None,
        train_noise_level=0.0,
        eval_noise_level=0.0,
        noise_type="gaussian",
    ):
        """__init__.

        Args:
            mask (Any): Description.
            kspace_obs (Any): Description.
            train_noise_level: Acquisition-noise std added to the measurement
                before it is pinned. **0.0** -- hard DC holds the measurement and
                nothing else, which is the contract its name states and the one
                the reverse loop's ``_apply_observed_dc`` already implemented.
                The previous 0.01/0.005 defaults were reached by four
                no-argument construction sites here, two of them on inference
                paths, so a deployed reconstruction perturbed its own input.
            eval_noise_level: As above. Non-zero at eval also contradicts the
                ``sampler_sigma: 0.0`` determinism every cohort arm declares
                (#1689).
            noise_type (Any): Description.
        """
        super().__init__()
        self.register_buffer("mask", mask)
        self.register_buffer("kspace_obs", kspace_obs)
        self.train_noise_level = train_noise_level
        self.eval_noise_level = eval_noise_level
        # ``noise_type`` used to be accepted here and then dropped on the floor:
        # the parameter arrived, was documented, and was never stored, so an
        # unsupported value degraded silently to Gaussian (#1445, #1525,
        # pitfall #9). Validated the way ``NoiseSimulatingDataConsistency`` already does,
        # so the three implementations of this parameter share one policy (NN17).
        self.noise_type = _validate_noise_type(noise_type, type(self).__name__)

    def forward(
        self,
        reconstruction,
        kspace_obs=None,
        mask=None,
        measured_kspace=None,
        is_kspace_domain=False,
        acs_mask=None,
    ):
        """forward.

        Args:
            reconstruction (Any): Description.
            kspace_obs (Any): Description.
            mask (Any): Description.
            measured_kspace (Any): Description.
            is_kspace_domain (Any): Description.
            acs_mask (Any): Description.
        Returns:
            Any: Description.

        forward method for HardDataConsistency.

        Executes PyTorch tensor operations.

        Args:
            reconstruction (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            kspace_obs (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            measured_kspace (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            is_kspace_domain (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            acs_mask (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        if measured_kspace is not None and kspace_obs is None:
            kspace_obs = measured_kspace
        if kspace_obs is None:
            kspace_obs = self.kspace_obs
        if mask is None:
            mask = self.mask

        kspace_obs = self._add_realistic_noise(kspace_obs)

        if not is_kspace_domain:
            is_kspace_domain = (
                not torch.is_complex(reconstruction)
                and reconstruction.ndim == 4
                and reconstruction.shape[1] % 2 == 0
            )

        if is_kspace_domain:
            while mask.dim() < reconstruction.dim():
                mask = mask.unsqueeze(1)
            mask_num = mask.float()
            if mask_num.shape[1] > reconstruction.shape[1]:
                mask_num = mask_num[:, : reconstruction.shape[1], ...]
            if kspace_obs.shape[1] > reconstruction.shape[1]:
                kspace_obs = kspace_obs[:, : reconstruction.shape[1], ...]

            result = (1.0 - mask_num) * reconstruction + mask_num * kspace_obs
            return result

        # IMAGE Path
        if not torch.is_complex(reconstruction) and reconstruction.shape[1] % 2 == 0:
            real = reconstruction[:, 0::2, ...]
            imag = reconstruction[:, 1::2, ...]
            recon_complex = torch.complex(real, imag)
        else:
            recon_complex = _to_complex(reconstruction)

        k_pred = fft2c(recon_complex)
        if not torch.is_complex(kspace_obs):
            if kspace_obs.shape[1] % 2 == 0:
                real = kspace_obs[:, 0::2, ...]
                imag = kspace_obs[:, 1::2, ...]
                kspace_obs = torch.complex(real, imag)
            else:
                kspace_obs = _to_complex(kspace_obs)

        while mask.dim() < k_pred.dim():
            mask = mask.unsqueeze(1)
        mask_num = mask.float()
        if mask_num.shape[1] > k_pred.shape[1]:
            mask_num = mask_num[:, : k_pred.shape[1], ...]
        if kspace_obs.shape[1] > k_pred.shape[1]:
            kspace_obs = kspace_obs[:, : k_pred.shape[1], ...]

        k_fused = (1.0 - mask_num) * k_pred + mask_num * kspace_obs

        img_fused = ifft2c(k_fused)
        if not torch.is_complex(reconstruction):
            if reconstruction.shape[1] % 2 == 0:
                img_fused = torch.stack([img_fused.real, img_fused.imag], dim=2).flatten(1, 2)
            else:
                img_fused = img_fused.abs()
        return img_fused

    def _add_realistic_noise(self, kspace):
        """_add_realistic_noise.

        Args:
            kspace (Any): Description.
        Returns:
            Any: Description.
        """
        noise_std = self.train_noise_level if self.training else self.eval_noise_level
        if noise_std == 0.0:
            return kspace
        # noise_type is validated to be 'gaussian' in __init__; no silent fallback.
        return kspace + torch.randn_like(kspace) * noise_std


class SoftDataConsistency(nn.Module):
    """Soft Data Consistency Layer."""

    def __init__(self, lambda_init: float = 10.0, learnable: bool = True):
        """__init__.

        Args:
            lambda_init (float): Description.
            learnable (bool): Description.
        """
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor(float(lambda_init)), requires_grad=learnable)

    def forward(
        self,
        predicted_img,
        measured_kspace,
        mask,
        acs_mask=None,
        is_kspace_domain=False,
    ):
        """forward.

        Args:
            predicted_img (Any): Description.
            measured_kspace (Any): Description.
            mask (Any): Description.
            acs_mask (Any): Description.
            is_kspace_domain (Any): Description.
        Returns:
            Any: Description.

        forward method for SoftDataConsistency.

        Executes PyTorch tensor operations.

        Args:
            predicted_img (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            measured_kspace (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            acs_mask (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            is_kspace_domain (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        if not is_kspace_domain:
            is_kspace_domain = (
                not torch.is_complex(predicted_img)
                and predicted_img.ndim == 4
                and predicted_img.shape[1] % 2 == 0
            )

        is_complex_input = torch.is_complex(predicted_img)
        if is_kspace_domain:
            if not is_complex_input:
                real = predicted_img[:, 0::2, ...]
                imag = predicted_img[:, 1::2, ...]
                k_pred = torch.complex(real, imag)
            else:
                k_pred = predicted_img
        else:
            if not is_complex_input and predicted_img.shape[1] % 2 == 0:
                real = predicted_img[:, 0::2, ...]
                imag = predicted_img[:, 1::2, ...]
                x_complex = torch.complex(real, imag)
            else:
                x_complex = _to_complex(predicted_img)
            k_pred = fft2c(x_complex)

        if not torch.is_complex(measured_kspace):
            if measured_kspace.shape[1] % 2 == 0:
                real = measured_kspace[:, 0::2, ...]
                imag = measured_kspace[:, 1::2, ...]
                measured_kspace = torch.complex(real, imag)
            else:
                measured_kspace = _to_complex(measured_kspace)

        while mask.dim() < k_pred.dim():
            mask = mask.unsqueeze(1)
        mask_real = mask.real if torch.is_complex(mask) else mask
        if mask_real.ndim == 4 and mask_real.shape[1] > k_pred.shape[1]:
            mask_real = mask_real[:, : k_pred.shape[1], ...]

        lam = torch.clamp(self.lambda_param, min=1e-6)
        num = k_pred + lam * measured_kspace * mask_real
        den = 1.0 + lam * mask_real
        k_out = num / den

        if is_kspace_domain:
            if not is_complex_input:
                return torch.stack([k_out.real, k_out.imag], dim=2).flatten(1, 2)
            return k_out
        else:
            x_out = ifft2c(k_out)
            if not is_complex_input:
                if predicted_img.shape[1] % 2 == 0:
                    return torch.stack([x_out.real, x_out.imag], dim=2).flatten(1, 2)
                return x_out.abs()
            return x_out


class NoiseAdaptiveDataConsistency(nn.Module):
    r"""Physics-based noise-adaptive data consistency (Wiener / LMMSE trust).

    Blends the network's k-space prediction with the measurement on the sampled
    support using a per-bin Wiener shrinkage weight derived from the measured
    signal-to-noise ratio::

        P(k)   = |y(k)|^2                                    # measured power
        lam(k) = P(k) / (P(k) + softplus(beta) * sigma^2)    # in [0, 1)
        k_new  = (1 - lam) * k_pred + lam * y                # convex blend
        k_out  = where(mask, k_new, k_pred)                  # unsampled = prediction

    High-SNR bins (``P >> sigma^2``, e.g. the k-space centre) get ``lam -> 1`` and
    approach hard DC; noise-dominated bins (``P <~ sigma^2``, the low-field
    periphery) get ``lam -> 0`` and keep the network's denoised prediction. That
    is exactly the low-field denoising behaviour. Unlike the CNN
    ``AdaptiveDataConsistency`` it uses NO learned per-pixel map (only a single
    scalar temperature ``beta``), so it cannot manufacture a measurement-
    independent central blob: ``lam`` is a deterministic function of the *measured*
    magnitude, and unsampled bins are always pure prediction.

    ``sigma^2`` defaults to a self-contained robust estimate from the outermost
    *sampled* (noise-dominated) k-space bins; an explicit ``noise_sigma`` (a noise
    standard deviation, e.g. from M4Raw rep-variance) overrides it.

    Boundedness: ``lam in [0, 1)`` makes ``k_new`` a convex combination of ``y`` and
    ``k_pred``, so ``|k_new| <= max(|y|, |k_pred|)`` per bin (no amplification).

    ``sensitivity_maps`` is accepted for interface parity with the other adaptive
    DC layers but is unused: this closed-form layer is coil-wise (each coil's
    k-space is made consistent independently), which needs no SENSE operator.
    """

    def __init__(
        self,
        beta: float = 1.0,
        learnable_beta: bool = False,
        outer_radius_frac: float = 0.7,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)), requires_grad=bool(learnable_beta))
        self.outer_radius_frac = float(outer_radius_frac)
        self.eps = float(eps)
        self._coord_cache_shape: tuple[int, int] | None = None
        self.register_buffer("_coord_radius", torch.empty(0), persistent=False)
        # Telemetry: [center_mean, periphery_mean, overall_mean, overall_std],
        # populated every forward (parity with KANAdaptiveDataConsistency).
        self.register_buffer("_last_trust_stats", torch.zeros(4), persistent=False)

    @torch.no_grad()
    def _build_coords(self, H: int, W: int, device) -> None:
        ky = torch.linspace(-(H // 2), H - 1 - H // 2, H, device=device)
        kx = torch.linspace(-(W // 2), W - 1 - W // 2, W, device=device)
        ky_g, kx_g = torch.meshgrid(ky, kx, indexing="ij")
        r = torch.sqrt(ky_g**2 + kx_g**2)
        r_max = float(r.max().item())
        if r_max <= 0:
            r_max = 1.0
        self._coord_radius = r / r_max  # [0, 1]; 0 at the k-space centre
        self._coord_cache_shape = (H, W)

    @torch.no_grad()
    def _estimate_sigma2(self, power: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Robust per-sample noise variance from the outermost sampled bins.

        ``power`` is ``[B, 1, H, W]`` (mean-over-coils ``|y|^2``); ``obs`` is the
        observed support ``[B, 1, H, W]``. Uses the median of ``|y|^2`` over the
        sampled bins in the outer radial annulus (noise-dominated); falls back to
        the median over all sampled bins when the annulus is too sparse (e.g. very
        high acceleration). Returns ``[B, 1, 1, 1]``.
        """
        B, _, H, W = power.shape
        if self._coord_cache_shape != (H, W):
            self._build_coords(H, W, power.device)
        outer = (self._coord_radius > self.outer_radius_frac).view(1, 1, H, W)
        region = (obs > 0) & outer
        sigma2 = power.new_empty(B, 1, 1, 1)
        for b in range(B):
            vals = power[b][region[b]]
            if vals.numel() < 8:
                vals = power[b][obs[b] > 0]
            sigma2[b] = torch.median(vals) if vals.numel() > 0 else power.new_tensor(self.eps)
        return sigma2.clamp_min(self.eps)

    def _compute_lambda(
        self,
        measured_kspace: torch.Tensor,
        mask: torch.Tensor,
        noise_sigma: torch.Tensor | float | None,
    ) -> torch.Tensor:
        """Return the Wiener trust map ``[B, 1, H, W]`` in ``[0, 1)``."""
        power = measured_kspace.abs() ** 2
        if power.shape[1] > 1:
            power = power.mean(dim=1, keepdim=True)  # share trust across coils
        _, _, H, W = power.shape
        if self._coord_cache_shape != (H, W):
            self._build_coords(H, W, power.device)

        if noise_sigma is not None:
            if torch.is_tensor(noise_sigma):
                sigma2 = noise_sigma.to(power) ** 2
                while sigma2.dim() < 4:
                    sigma2 = sigma2.unsqueeze(-1)
            else:
                sigma2 = power.new_tensor(float(noise_sigma) ** 2).view(1, 1, 1, 1)
            sigma2 = sigma2.clamp_min(self.eps)
        else:
            obs = align_dc_mask(mask, 1)
            sigma2 = self._estimate_sigma2(power, obs)

        beta = torch.nn.functional.softplus(self.beta)
        lam = power / (power + beta * sigma2 + self.eps)

        with torch.no_grad():
            r = self._coord_radius
            if r.numel() > 0:
                center = (r < 0.08).float().view(1, 1, H, W)
                periph = 1.0 - center
                lam_c = (lam * center).sum(dim=(-2, -1)) / center.sum().clamp_min(1.0)
                lam_p = (lam * periph).sum(dim=(-2, -1)) / periph.sum().clamp_min(1.0)
                self._last_trust_stats[0] = lam_c.mean().detach()
                self._last_trust_stats[1] = lam_p.mean().detach()
                self._last_trust_stats[2] = lam.mean().detach()
                self._last_trust_stats[3] = lam.std().detach()
        return lam

    def forward(
        self,
        img_pred: torch.Tensor,
        measured_kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        sensitivity_maps: torch.Tensor | None = None,
        is_kspace_domain: bool = False,
        acs_mask: torch.Tensor | None = None,
        noise_sigma: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Mirror the ``KANAdaptiveDataConsistency`` interface (see class docstring)."""
        del acs_mask, sensitivity_maps  # coil-wise: no SENSE operator needed
        if measured_kspace is None or mask is None:
            return img_pred

        if not is_kspace_domain:
            is_kspace_domain = (
                not torch.is_complex(img_pred)
                and img_pred.ndim == 4
                and img_pred.shape[1] % 2 == 0
                and not torch.is_complex(measured_kspace)
                and measured_kspace.ndim == 4
                and measured_kspace.shape[1] == img_pred.shape[1]
            )

        is_complex_input = torch.is_complex(img_pred)
        if not is_complex_input and img_pred.shape[1] % 2 == 0:
            x = torch.complex(img_pred[:, 0::2, ...], img_pred[:, 1::2, ...])
        else:
            x = _to_complex(img_pred)

        k_pred = x if is_kspace_domain else fft2c(x)

        if not torch.is_complex(measured_kspace):
            measured_kspace = torch.complex(
                measured_kspace[:, 0::2, ...], measured_kspace[:, 1::2, ...]
            )

        lambda_map = self._compute_lambda(measured_kspace, mask, noise_sigma)
        k_new = (1.0 - lambda_map) * k_pred + lambda_map * measured_kspace

        k_out = torch.where(align_dc_mask(mask, k_pred.shape[-3]).bool(), k_new, k_pred)

        if is_kspace_domain:
            if is_complex_input:
                return k_out
            if img_pred.shape[1] % 2 == 0:
                return torch.stack([k_out.real, k_out.imag], dim=2).flatten(1, 2)
            return k_out.real

        x_out = ifft2c(k_out)
        if is_complex_input:
            return x_out
        if img_pred.shape[1] % 2 == 0:
            return torch.stack([x_out.real, x_out.imag], dim=2).flatten(1, 2)
        return x_out.real


def data_consistency(x, k_measured, mask, weight=1.0):
    """Soft (proximal) data-consistency blend of a prediction with measurements.

    Blends at sampled k-space lines as ``(k_pred + weight·k_meas)/(1 + weight)``:
    ``weight=0`` is a pure passthrough (returns the prediction), ``weight→∞``
    approaches hard replacement by the measurement. ``method="soft"`` is passed
    explicitly so ``weight`` is a live knob — the default ``SimpleDataConsistency``
    method is ``"hard"``, which ignores ``weight`` (that would make this helper's
    ``weight`` argument inert, CLAUDE.md pitfall #15).

    Args:
        x: Predicted image (complex) or interleaved real/imag ``[B, 2C, H, W]``.
        k_measured: Measured k-space (complex or interleaved).
        mask: Sampling mask (broadcastable to the k-space shape).
        weight: Soft-DC weight ``λ`` (0 → passthrough, large → hard-like).
    Returns:
        The blended image in the same domain/dtype layout as ``x``.
    """
    dc_layer = SimpleDataConsistency(weight=weight, method="soft")
    return dc_layer(x, k_measured, mask)


class TargetAwareFSDC(nn.Module):
    r"""Target-Aware Frequency-Split Data Consistency (TA-FSDC).

    Physics-informed DC that applies **different consistency strategies** to
    reference vs. target contrast channels, and **frequency-splits** the
    target channel update into low-frequency (ACS) and high-frequency bands.

    In a multi-contrast cold-diffusion pipeline the predicted tensor contains
    both fully-sampled reference contrasts and undersampled target contrasts
    concatenated along the channel axis.  A monolithic DC operator would
    degrade the reference priors and introduce spectral discontinuities on the
    target.  This module avoids both failure modes:

    * **Reference channels**: strict Hard DC — perfectly anchors the
      fully-sampled structural prior manifold that the cross-attention
      module queries.
    * **Target ACS (low-frequency)**: Hard DC — rigorously locks the
      target's macroscopic tissue contrast encoded in the centre of
      k-space (dictated by T₁, T₂, PD relaxometry).
    * **Target HF sampled**: Soft proximal DC — blends the sparse, aliased
      high-frequency target measurements with the coherent structural
      phase synthesised by the network via a learnable weight, stabilising
      the reverse diffusion trajectory.
    * **Target unsampled**: Pure prediction — no measured data exists, so
      the network's cross-contrast synthesis is trusted entirely.

    References
    ----------
    * Ehrhardt et al., "Multicontrast MRI Reconstruction with
      Structure-Guided Total Variation", SIAM J. Imaging Sci. (2015).
    * Chung et al., "Score-based diffusion models for accelerated MRI",
      Med. Image Anal. (2022).
    """

    def __init__(
        self,
        target_channels: int = 8,
        center_fraction: float = 0.03,
        hf_lambda: float = 0.5,
        acs_taper: str | None = None,
    ):
        """Initialise TA-FSDC.

        Args:
            target_channels: Number of interleaved-real/imag channels that
                belong to the **target** contrast (placed at the *end* of the
                channel dimension).  The remaining leading channels are treated
                as fully-sampled reference contrasts.
            center_fraction: Fraction of k-space height/width that defines the
                Auto-Calibration Signal (ACS) low-frequency band.
            hf_lambda: Initial value for the learnable high-frequency blend
                weight (passed through sigmoid before use).
            acs_taper: Optional apodization for the ACS mask edge. ``None``
                (default) keeps the legacy hard-rectangle window: a binary
                ``2·pad × 2·pad`` box at k-space centre.  The hard rectangle
                has an ``IFFT`` that's a 2-D sinc with strong sidelobes — for
                an under-trained model whose HF prediction is near-zero, the
                ACS-only reconstruction produces a concentrated bright lobe at
                image centre (the "white spot" surfaced by
                ``experiment_130_ti_ccd`` in audit-2026-05-14 §1 round-5).
                Set ``acs_taper="hann"`` to smoothly attenuate the ACS mask
                at its boundary (separable Hann taper over the outer 25% of
                the ACS box's width/height). The resulting ACS window has a
                wider but smoother main lobe in image space — early-training
                renders no longer show a hard saturated central cluster, and
                trained-model behaviour is essentially unchanged (the HF
                prediction dominates outside ACS).
        """
        super().__init__()
        self.target_channels = target_channels
        self.center_fraction = center_fraction
        if acs_taper not in (None, "hann"):
            raise ValueError(
                f"TargetAwareFSDC: acs_taper={acs_taper!r} unsupported. "
                "Valid options: None (hard rectangle, legacy) or 'hann' "
                "(separable Hann taper on outer 25% of ACS box). See "
                "audit-2026-05-14 §1 round-5 for the under-training "
                "central-spot rationale."
            )
        self.acs_taper = acs_taper

        # Learnable λ for the high-frequency soft proximal blend.
        # σ(λ_hf) = 0  → trust prediction fully
        # σ(λ_hf) = 1  → trust measurement fully
        self.lambda_hf = nn.Parameter(torch.tensor(float(hf_lambda)))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_acs_mask(self, H: int, W: int, reference: torch.Tensor) -> torch.Tensor:
        """Return a mask selecting the ACS (centre-of-k-space) region.

        Shape: ``[1, 1, H, W]``, same device/dtype as *reference*.

        With ``acs_taper=None`` the mask is binary 0/1 (legacy hard
        rectangle). With ``acs_taper='hann'`` the outer 25% of the box's
        width / height is tapered by a separable Hann window so the IFFT
        of the ACS-only k-space has reduced sidelobes (audit-2026-05-14
        §1 round-5).
        """
        pad_h = max(1, int(H * self.center_fraction))
        pad_w = max(1, int(W * self.center_fraction))
        ch, cw = H // 2, W // 2

        acs = torch.zeros(1, 1, H, W, device=reference.device, dtype=reference.dtype)

        if self.acs_taper is None:
            acs[:, :, ch - pad_h : ch + pad_h, cw - pad_w : cw + pad_w] = 1.0
            return acs

        # Hann-apodized ACS window: a full-period Hann window spans the
        # ACS box on each axis. The window peaks at 1.0 at the box centre
        # (preserving hard-DC at the macroscopic-contrast core) and rolls
        # smoothly to 0 at the box edges. IFFT-domain effect: vs. the
        # hard rectangle, the IFFT has roughly half the central-peak
        # amplitude and dramatically suppressed sidelobes (Gibbs ringing
        # gone). For an under-trained cold-diffusion model whose HF
        # prediction is near-zero, the central cluster amplitude drops
        # by ~78% (measured on the ``experiment_130_ti_ccd`` fixture)
        # so the renderer's foreground-percentile normalisation no
        # longer saturates a single bright clipped region.
        box_h, box_w = 2 * pad_h, 2 * pad_w
        if box_h < 4 or box_w < 4:
            # ACS too small to taper meaningfully — ``hann_window(2)``
            # returns ``[0, 0]`` (zeros at both endpoints), producing an
            # empty mask. Fall back to the binary rectangle so a tiny
            # ``center_fraction`` still yields a sensible DC anchor.
            acs[:, :, ch - pad_h : ch + pad_h, cw - pad_w : cw + pad_w] = 1.0
            return acs
        prof_h = torch.hann_window(
            box_h,
            periodic=False,
            device=reference.device,
            dtype=reference.dtype,
        )
        prof_w = torch.hann_window(
            box_w,
            periodic=False,
            device=reference.device,
            dtype=reference.dtype,
        )
        win_2d = prof_h.unsqueeze(1) * prof_w.unsqueeze(0)
        acs[:, :, ch - pad_h : ch + pad_h, cw - pad_w : cw + pad_w] = win_2d
        return acs

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        kspace_pred: torch.Tensor,
        kspace_measured: torch.Tensor,
        mask: torch.Tensor,
        is_kspace_domain: bool = True,
        acs_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply Target-Aware Frequency-Split Data Consistency.

        Args:
            kspace_pred: Model-predicted k-space ``[B, C_total, H, W]``
                (interleaved real/imag channels).
            kspace_measured: Acquired k-space measurements.  May be
                ``[B, C_total, H, W]`` (full multi-contrast) or
                ``[B, C_target, H, W]`` (target-only).
            mask: Binary undersampling mask ``[B, 1, H, W]`` (1 = sampled).
            is_kspace_domain: Accepted for API compatibility (TA-FSDC always
                operates in k-space; the flag is ignored).
            acs_mask: Optional external ACS mask.  When ``None`` the ACS region
                is derived from ``center_fraction``.

        Returns:
            Data-consistent k-space ``[B, C_total, H, W]``.

        forward method for TargetAwareFSDC.

        Executes PyTorch tensor operations.

        Args:
            kspace_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            kspace_measured (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            is_kspace_domain (bool): Expected input tensor.
            acs_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        B, C_total, H, W = kspace_pred.shape

        # Ensure mask is broadcastable  [B, 1, H, W]
        if mask.dim() < kspace_pred.dim():
            mask = mask.unsqueeze(1)
        # The source/target split slices the channel axis, so a mask holding one
        # channel per interleaved Re/Im half no longer lines up with either side.
        mask_f = align_dc_mask(mask, 1).float()

        # ── 0. Determine reference / target split ────────────────────
        C_target = min(self.target_channels, C_total)
        C_source = C_total - C_target

        source_pred = kspace_pred[:, :C_source, ...]
        target_pred = kspace_pred[:, C_source:, ...]

        # Align kspace_measured to the target region
        C_meas = kspace_measured.shape[1]
        if C_meas >= C_total:
            # Full multi-contrast measurement: split source + target
            source_meas = kspace_measured[:, :C_source, ...]
            target_meas = kspace_measured[:, C_source:, ...]
        elif C_meas >= C_target:
            # Target-only measurement
            source_meas = None
            target_meas = kspace_measured[:, :C_target, ...]
        else:
            # Fewer measured channels than the target region — a wiring/contract
            # violation (predicted k-space needs >= C_target measured channels to
            # apply DC). Silently skipping DC here masks a dropped physics step
            # (CLAUDE.md pitfalls #4/#9/#10), so fail loud instead of degrading.
            raise ValueError(
                f"TA-FSDC: kspace_measured has {C_meas} channels but needs "
                f">= {C_target} target channels; cannot apply data consistency. "
                f"This is a channel-count wiring bug, not a recoverable runtime "
                f"condition."
            )

        # ── 1. Reference channels: strict Hard DC ────────────────────
        # Replace sampled k-space lines with measurements exactly;
        # unsampled regions keep the prediction.
        if C_source > 0 and source_meas is not None:
            source_out = torch.where(mask_f.bool(), source_meas, source_pred)
        else:
            source_out = source_pred

        # ── 2. Target channels: frequency-split DC ───────────────────
        # Build ACS (low-frequency) mask
        if acs_mask is not None:
            lf_mask = acs_mask.float()
        else:
            lf_mask = self._build_acs_mask(H, W, kspace_pred)

        hf_mask = mask_f * (1.0 - lf_mask)  # sampled HF lines
        unsampled_mask = 1.0 - mask_f  # never-acquired lines

        # 2a. ACS region — Hard DC (lock macroscopic tissue contrast)
        target_lf = torch.where(
            lf_mask.bool(),
            target_meas,  # measurement replaces prediction exactly
            target_pred,  # outside ACS → keep prediction
        )

        # 2b. HF sampled — Soft proximal DC (prevent manifold departure)
        blend = torch.sigmoid(self.lambda_hf)
        hf_blend = (1.0 - blend) * target_pred + blend * target_meas

        # 2c. Compose: LF uses hard-replaced values, HF-sampled uses blend,
        #     unsampled keeps pure prediction.
        target_out = lf_mask * target_lf + hf_mask * hf_blend + unsampled_mask * target_pred

        # ── 3. Re-concatenate source + target ────────────────────────
        if C_source > 0:
            return torch.cat([source_out, target_out], dim=1)
        return target_out


def dc_passthrough_center_patch(
    x_out: torch.Tensor,
    measured_kspace: torch.Tensor | None,
    mask: torch.Tensor | None,
    patch_size: tuple[int, int] | int,
) -> torch.Tensor:
    """Hardcode the central k-space patch of the model output to the measurement.

    The DC bin (and a small region around it) is determined by acquisition,
    not by reconstruction — for any sampled bin in the central low-frequency
    region, the measured value is exact. Routing it around the CNN prevents
    the centre-blob ("white spot") failure mode that dominates iter-0
    visualisations of k-space cold-diffusion variants (see
    docs/validation_image_audit.rst).

    Mathematically identical to the corner case of
    :class:`AdaptiveDataConsistency` with ``lambda_map ≡ 1`` on the centre
    patch; the difference is that this passthrough is deterministic from
    iter 0, whereas AdaptiveDC has to learn the same fact.

    Args:
        x_out: Model output, ``[B, C, H, W]``. May be complex or real-stacked.
        measured_kspace: Measurements, same shape semantics as ``x_out``.
        mask: Sampling mask, broadcastable on the channel and spatial axes.
        patch_size: ``(h, w)`` of the central patch to passthrough. Pass
            ``1`` or ``(1, 1)`` for DC-only.

    Returns:
        ``x_out`` with the centre patch overwritten at sampled bins (mask=1).
        Unsampled bins inside the patch are left as the CNN predicted them.
        No-op if any required argument is ``None``.
    """
    if measured_kspace is None or mask is None:
        return x_out

    H, W = x_out.shape[-2], x_out.shape[-1]
    if isinstance(patch_size, int):
        h, w = patch_size, patch_size
    else:
        h, w = patch_size
    h = max(1, min(int(h), H))
    w = max(1, min(int(w), W))

    # ``fft2c`` uses fftshift, so DC lives at index ``H // 2`` (not
    # ``(H - 1) // 2``). Centre the patch on the DC bin: for h=1 the
    # slice is exactly the DC index; for h=3 it extends one bin to
    # either side.
    cy, cx = H // 2, W // 2
    h_start = max(0, cy - h // 2)
    w_start = max(0, cx - w // 2)
    h_end = min(H, h_start + h)
    w_end = min(W, w_start + w)

    # Channel alignment — for cross-contrast layouts the measurement
    # tensor carries ``[source || target]`` while ``x_out`` is only the
    # target half. Take the LAST C channels to align (same convention as
    # ``_slice_to_target_contrast_single`` and the surrounding DC code).
    if measured_kspace.shape[1] > x_out.shape[1]:
        measured_kspace = measured_kspace[:, -x_out.shape[1] :]

    mask_real = mask.real if torch.is_complex(mask) else mask
    if mask_real.dim() >= 2 and mask_real.shape[1] > 1 and mask_real.shape[1] != x_out.shape[1]:
        mask_real = mask_real[:, -x_out.shape[1] :]

    measured_patch = measured_kspace[..., h_start:h_end, w_start:w_end]
    mask_patch = mask_real[..., h_start:h_end, w_start:w_end]
    out_patch = x_out[..., h_start:h_end, w_start:w_end]

    # Batch-dim broadcast for the 5D → 4D slice-flatten case (validation
    # path in ``diffusion.py::validation_step``). When the dataloader emits
    # ``(B, C, H, W, D)`` and the validator flattens to ``(B*D, C, H, W)``,
    # ``x_out`` arrives with batch ``B*D`` while ``mask`` and
    # ``measured_kspace`` were captured pre-flatten at batch ``B``.
    # ``.expand_as(out_patch)`` then fails because expand only handles
    # size-1 dimensions — 2 → 18 is not a singleton expansion. Repeat
    # each batch element ``D`` times to match, which is the same
    # broadcast semantics the IFFT pipeline applies elsewhere.
    # Anchor: experiment_11 cluster run 2026-05-28 19:30 traceback.
    if out_patch.shape[0] != mask_patch.shape[0]:
        if mask_patch.shape[0] > 0 and out_patch.shape[0] % mask_patch.shape[0] == 0:
            n_repeat = out_patch.shape[0] // mask_patch.shape[0]
            mask_patch = mask_patch.repeat_interleave(n_repeat, dim=0)
            if measured_patch.shape[0] == out_patch.shape[0] // n_repeat:
                measured_patch = measured_patch.repeat_interleave(n_repeat, dim=0)
        else:
            # Cannot align batch dims by integer repetition — bail
            # rather than silently corrupt the centre patch.
            return x_out

    # Cast measured to x_out dtype for safe blend (cross-contrast paths
    # can mix complex64 and float32 across the boundary).
    if measured_patch.dtype != out_patch.dtype:
        if torch.is_complex(out_patch) and not torch.is_complex(measured_patch):
            # real-stacked measured → take pairs as complex
            if measured_patch.shape[-3] % 2 == 0:
                measured_patch = torch.complex(
                    measured_patch[..., 0::2, :, :],
                    measured_patch[..., 1::2, :, :],
                )
            else:
                return x_out  # cannot align; bail rather than corrupt
        elif not torch.is_complex(out_patch) and torch.is_complex(measured_patch):
            measured_patch = torch.stack([measured_patch.real, measured_patch.imag], dim=2).flatten(
                1, 2
            )[..., h_start:h_end, w_start:w_end]
        measured_patch = measured_patch.to(out_patch.dtype)

    if mask_patch.shape != out_patch.shape:
        # Broadcast mask to match — repeating along channel if needed.
        mask_patch = mask_patch.expand_as(out_patch)

    new_patch = torch.where(mask_patch.bool(), measured_patch, out_patch)
    x_out_new = x_out.clone()
    x_out_new[..., h_start:h_end, w_start:w_end] = new_patch
    return x_out_new


class ConformalDataConsistency(nn.Module):
    """DC step in conformally-mapped k-space coordinates (§2 of the
    SFC / conformal plan).

    Given a Riemann map ``Φ : Ω → 𝔻`` that takes the non-Cartesian
    sampling region to the unit disk, the forward MRI model becomes

    .. math::
        \\tilde y(z) = M(\\Phi^{-1}(z))\\,(F S x)(\\Phi^{-1}(z)),
        \\quad z\\in\\mathbb{D},

    with the conformal Jacobian ``|Φ'(z)|^2`` absorbed into the
    sampling-density correction. The DC update is

    .. math::
        (\\mathrm{DC}_\\Phi x)(z) = x(z)
        + \\alpha\\,|\\Phi'(z)|^{-1}\\,
        \\bigl(\\tilde y(z) - M(\\Phi^{-1}(z))(F x)(\\Phi^{-1}(z))\\bigr).

    Implementation: caller supplies a pre-computed ``conformal_jacobian``
    tensor (``|Φ'(z)|`` on the canonical polar grid in :math:`\\mathbb{D}`),
    the mapped observation ``y_tilde``, and a sampling mask. The
    operator is a single multiplicative correction — no learnable
    parameters — and is therefore safe to register as a fixed layer
    in any reconstruction strategy.

    References:
        [3] L. V. Ahlfors, *Complex Analysis*, 3rd ed., McGraw-Hill, 1979.
        [4] Driscoll & Trefethen, *Schwarz-Christoffel Mapping*, CUP, 2002.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        reconstruction: torch.Tensor,
        *,
        y_tilde: torch.Tensor,
        mask: torch.Tensor,
        conformal_jacobian: torch.Tensor,
    ) -> torch.Tensor:
        """Apply conformal DC.

        Args:
            reconstruction: ``[B, C, H, W]`` current estimate, in the
                conformally-mapped k-space coordinates ``z ∈ 𝔻``.
            y_tilde: Observation in the same coordinates, same shape.
            mask: Binary mask of acquired samples, broadcastable.
            conformal_jacobian: ``|Φ'(z)|`` on the same grid, broadcastable.

        Returns:
            Updated reconstruction.
        """
        inv_jac = conformal_jacobian.clamp_min(1e-6).reciprocal()
        residual = (y_tilde - reconstruction) * mask
        return reconstruction + self.alpha * inv_jac * residual


__all__ = [
    "AdaptiveDataConsistency",
    "ConformalDataConsistency",
    "HardDataConsistency",
    "SimpleDataConsistency",
    "SoftDataConsistency",
    "TargetAwareFSDC",
    "data_consistency",
    "dc_passthrough_center_patch",
]
