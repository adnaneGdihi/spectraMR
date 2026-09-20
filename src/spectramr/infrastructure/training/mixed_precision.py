import logging
from dataclasses import dataclass
from typing import ContextManager

import torch
from torch.amp import GradScaler as NativeScaler

from spectramr.core.compute_device import resolve_torch_device
from spectramr.core.device_capabilities import (
    MIN_NATIVE_BF16_CAPABILITY,
    TARGET_CAPABILITY_ENV,
    DeviceCapabilities,
    probe_device_capabilities,
)

logger = logging.getLogger(__name__)


@dataclass
class MixedPrecisionConfig:
    """Configuration for Mixed Precision Training."""

    enabled: bool = (
        False  # Default to False (FP32), enable only via config.optimization.precision.enabled
    )
    precision: str = "fp16"  # 'fp16' or 'bf16'
    optimize_memory: bool = True
    loss_scaling: float | str = "dynamic"  # fixed value or "dynamic"
    dynamic_loss_scaling: bool = True
    enable_complex_support: bool = True  # Enable for MRI/Complex flows, disable for standard float


# Maps the schema-facing ``optimization.precision.dtype`` to MixedPrecisionConfig.precision.
_AMP_DTYPE_TO_PRECISION = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp16"}


def resolve_amp_precision(use_amp: bool, amp_dtype: str | None) -> tuple[bool, str]:
    """Resolve ``(enabled, precision)`` for :class:`MixedPrecisionConfig`.

    Threads the schema knob ``optimization.precision.dtype`` into the AMP policy so
    autocast actually runs in the requested dtype (previously the policy
    hardcoded fp16 and the knob was inert — pitfall #15).

    - ``amp_dtype=None`` keeps the historical ``float16`` default (back-compat:
      existing ``use_amp: true`` configs are unchanged).
    - ``"bfloat16"`` → ``"bf16"`` (no loss scaling; fp32 exponent range — the
      safer choice for unstable models, and it works under complex autocast).
    - ``"float32"`` means full precision → AMP is **disabled** regardless of
      ``use_amp`` (so the knob can't silently no-op into fp16).

    Raises:
        ValueError: on a dtype outside the schema-validated set.
    """
    dtype = (amp_dtype or "float16").lower()
    if dtype not in _AMP_DTYPE_TO_PRECISION:
        raise ValueError(
            f"amp_dtype must be one of {sorted(_AMP_DTYPE_TO_PRECISION)}, got {amp_dtype!r}"
        )
    enabled = bool(use_amp) and dtype != "float32"
    return enabled, _AMP_DTYPE_TO_PRECISION[dtype]


class AMPPrecisionUnsupportedError(RuntimeError):
    """The resolved AMP dtype has no native support on the resolved device."""


def assert_amp_dtype_supported(
    precision: str,
    *,
    capabilities: DeviceCapabilities,
) -> None:
    """Raise when *precision* would run emulated on this device.

    Only ``bf16`` is gated, and only on CUDA. Pre-Ampere cards (sm_70 V100,
    sm_75 Turing) have no bf16 tensor cores; ``torch`` will still *run* bf16
    there by emulating it, which is slow and numerically unlike what the arm
    asked for.

    **Raises rather than downgrading to fp16.** A downgrade would be a silent
    fallback (pitfall #9), and it would also be wrong here: ``get_autocast_context``
    turns fp16 into a ``nullcontext`` for complex arms, so an automatic
    downgrade would quietly produce fp32 on exactly the arms this framework
    cares most about.

    An *unknown* capability does not raise. The audit legitimately runs on a
    login node with a different GPU or none at all, so "cannot tell" is reported
    and the run proceeds — the health check carries the same polarity.
    """
    if precision != "bf16" or capabilities.device_type != "cuda":
        return
    if "capability" in capabilities.incomplete:
        logger.warning(
            "AMP dtype bfloat16 requested but the compute capability could not be read "
            "(source=%s). Native bf16 needs sm_80+; set %s to declare the target.",
            capabilities.source,
            TARGET_CAPABILITY_ENV,
        )
        return
    if capabilities.native_bf16:
        return
    raise AMPPrecisionUnsupportedError(
        f"optimization.precision.dtype='bfloat16' on a device with compute capability "
        f"{capabilities.capability_str} (source={capabilities.source}). Native bf16 needs "
        f"sm_{MIN_NATIVE_BF16_CAPABILITY[0]}{MIN_NATIVE_BF16_CAPABILITY[1]}+; below that "
        f"torch emulates it, which is slower than fp16 and numerically different from what "
        f"this arm declared. Supported here: {list(capabilities.amp_dtypes)}. Note that "
        f"torch.cuda.is_bf16_supported() answers True on this device -- it admits emulation "
        f"by default, which is why this check reads the capability instead."
    )


class MixedPrecisionIntegrationHelper:
    """Helper to integrate AMP into training strategies."""

    def __init__(
        self,
        config: MixedPrecisionConfig,
        device: torch.device | str | None = None,
        capabilities: DeviceCapabilities | None = None,
    ):
        """Initialize the AMP integration helper.

        Args:
            config: Mixed-precision settings for this run.
            device: The device the run already resolved. AMP must autocast on
                the device the tensors are actually on; deriving it a second
                time from ``torch.cuda.is_available()`` made this a competing
                device resolver (non-negotiable 9b/17) that disagreed with
                ``env.device`` whenever the user opted into CPU on a CUDA box,
                or the run was pinned to a device other than ``cuda:0``.
                ``None`` falls back to the 9b resolver rather than to a raw
                availability probe.
            capabilities: Injected device capabilities, for tests and for callers
                that already probed. ``None`` probes *device*.

        Raises:
            AMPPrecisionUnsupportedError: AMP is on and the resolved dtype has
                no native support here -- today only bf16 below sm_80.
        """
        self.config = config
        self.scaler = None
        if device is None:
            device = resolve_torch_device(
                None,
                pipeline="mixed_precision",
                source="MixedPrecisionIntegrationHelper(device=None)",
            ).device
        self.device_type = torch.device(device).type
        self.enabled = config.enabled
        if self.enabled:
            # Probed here rather than threaded in: this runs once per helper, not
            # per step, and the probe is a `get_device_capability` call plus a
            # `find_spec`. Injectable so the gate is testable without a GPU.
            assert_amp_dtype_supported(
                config.precision,
                capabilities=capabilities or probe_device_capabilities(device),
            )
        if config.enabled and config.precision == "fp16":
            if config.enable_complex_support:
                # Use our new ComplexGradScaler instead of native one
                self.scaler = ComplexGradScaler(
                    self.device_type,
                    enabled=True,
                    init_scale=(
                        config.loss_scaling if isinstance(config.loss_scaling, float) else 2.0**16
                    ),
                )
            else:
                # Use NativeScaler for standard float training (faster, TF32 allowed)
                self.scaler = NativeScaler(
                    self.device_type,
                    enabled=True,
                    init_scale=(
                        config.loss_scaling if isinstance(config.loss_scaling, float) else 2.0**16
                    ),
                )

    def configure_model_for_amp(self, model: torch.nn.Module) -> None:
        """Calculates modification on the model if needed (usually none for PyTorch AMP)."""
        pass

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss for mixed precision.

        Args:
            loss: The loss tensor to scale.

        Returns:
            The scaled loss tensor.
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscales gradients before optimizer step.

        Args:
            optimizer: The optimizer containing scaled gradients.
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def step_optimizer(
        self, optimizer: torch.optim.Optimizer, loss: torch.Tensor | None = None
    ) -> bool:
        """Steps the optimizer and updates the scaler state.

        Args:
            optimizer: The optimizer to step.
            loss: Optional loss tensor for reference.

        Returns:
            Always returns True as the step is performed.
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
            return True
        optimizer.step()
        return True

    def is_enabled(self) -> bool:
        """Check if AMP is enabled."""
        return self.config.enabled

    def get_loss_scale(self) -> float:
        """Get current loss scale value."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0

    def get_autocast_context(self) -> ContextManager:
        """Returns the appropriate autocast context manager.

        For complex-valued MRI pipelines, fp16 autocast is fundamentally
        incompatible because complex64 tensors cannot interact with Half-precision
        model weights (PyTorch has no complex16 type). When complex support is
        enabled with fp16, we disable autocast entirely to prevent
        'Input type (complex<float>) and bias type (Half)' crashes.
        """
        from contextlib import nullcontext

        if not self.config.enabled:
            return nullcontext()

        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16

        # If complex support is disabled, use standard autocast (allows TF32 if globally enabled)
        if not self.config.enable_complex_support:
            return torch.amp.autocast(device_type=self.device_type, dtype=dtype, enabled=True)

        # [FIX] Complex + fp16 is fundamentally broken: complex64 cannot mix
        # with Half weights. Disable autocast entirely for fp16 complex flows.
        # bfloat16 CAN work with complex on some hardware but is still risky.
        if dtype == torch.float16:
            return nullcontext()

        # Use our SafeComplexAutocast with proper device_type and TF32 disabled
        return SafeComplexAutocast(
            device_type=self.device_type, dtype=dtype, enabled=True, disable_tf32=True
        )


class ComplexGradScaler(NativeScaler):
    """
    Extends GradScaler to safely handle Complex64/128 gradients.

    Standard NativeScaler typically expects real-valued losses to derive scale factors.
    When working with complex-valued pipelines, gradients are complex.
    PyTorch's AMP scaler logic involves checking for Inf/NaNs in gradients.

    This implementation ensures that:
    1. Validation of inputs prevents complex-valued LOSS being scaled directly (must be real).
    2. Unscaling handles complex gradients correctly (checking magnitudes/real/imag).
    """

    def __init__(
        self,
        device="cuda",
        enabled=True,
        init_scale=2.0**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    ):
        """__init__.

        Args:
            device (Any): Description.
            enabled (Any): Description.
            init_scale (Any): Description.
            growth_factor (Any): Description.
            backoff_factor (Any): Description.
            growth_interval (Any): Description.
        """
        super().__init__(
            device=device,
            enabled=enabled,
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        )

    def scale(self, outputs):
        """
        Scales the loss. The loss must be real-valued.
        """
        # AMP scaling logic typically requires real-valued loss
        if torch.is_complex(outputs):
            raise ValueError(
                "Loss passed to Scaler.scale() must be Real-valued (e.g. abs() or norm()). "
                "Optimization on complex numbers requires a real-valued objective function."
            )

        return super().scale(outputs)

    def _unscale_grads_(self, optimizer, *args, **kwargs):
        """
        Unscales the gradients of optimizer's assigned parameters.
        Native implementation usually works on underlying .grad attributes.
        We hook here just to provide observability or custom handling if PyTorch versions change.

        Note: PyTorch's native _unscale_grads_ iterates over parameters and multiplies .grad by inv_scale.
        For complex tensors, multiplying by a real scalar (inv_scale) works correctly
        (scales both real and imag parts).

        The critical part is INF/NaN checking which is done in `step`.
        """
        return super()._unscale_grads_(optimizer, *args, **kwargs)

    def step(self, optimizer, *args, **kwargs):
        """
        Step the optimizer.
        Native Scaler checks for Inf/NaN in gradients before stepping.
        Complex gradients with Inf/NaN in either part should be detected.
        Process:
        1. _unscale_grads_ is called (if not already).
        2. Check intersection of "found_inf" across all separate grad blocks.

        Standard PyTorch check uses `torch.isfinite`.
        For complex numbers, `isfinite` checks both parts.
        So native `step` should be robust, provided `unscale` worked.
        """
        return super().step(optimizer, *args, **kwargs)


class SafeComplexAutocast(torch.amp.autocast):
    """
    Context Manager for "Safe" Complex Autocast.

    Enforces policies to prevent instability in MRI reconstruction:
    1. Disables TensorFloat32 (TF32) for matmuls if requested (Precision > Speed).
    2. (Optionally) Manages FFT precision fallback (though PyTorch FFTs are mostly FP32).
    """

    def __init__(
        self,
        device_type: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enabled: bool = True,
        cache_enabled: bool = True,
        disable_tf32: bool = True,
    ):
        """__init__.

        Args:
            device_type (str): Description.
            dtype (torch.dtype): Description.
            enabled (bool): Description.
            cache_enabled (bool): Description.
            disable_tf32 (bool): Description.
        """
        super().__init__(
            device_type=device_type,
            dtype=dtype,
            enabled=enabled,
            cache_enabled=cache_enabled,
        )
        self.disable_tf32 = disable_tf32
        self._prev_tf32_matmul = None
        self._prev_tf32_cudnn = None

    def __enter__(self):
        """__enter__.

        Returns:
            Any: Description.
        """
        super().__enter__()
        if self.disable_tf32 and torch.cuda.is_available():
            self._prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
            self._prev_tf32_cudnn = torch.backends.cudnn.allow_tf32

            # Disable TF32 to ensure full FP32 precision where FP16 isn't used
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore flags
        """__exit__.

        Args:
            exc_type (Any): Description.
            exc_val (Any): Description.
            exc_tb (Any): Description.
        Returns:
            Any: Description.
        """
        if self.disable_tf32 and torch.cuda.is_available():
            if self._prev_tf32_matmul is not None:
                torch.backends.cuda.matmul.allow_tf32 = self._prev_tf32_matmul
            if self._prev_tf32_cudnn is not None:
                torch.backends.cudnn.allow_tf32 = self._prev_tf32_cudnn

        super().__exit__(exc_type, exc_val, exc_tb)
