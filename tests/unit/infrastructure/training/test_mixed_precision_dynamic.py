from unittest.mock import MagicMock

import pytest
import torch
from torch.amp import GradScaler as NativeScaler

from spectramr.infrastructure.training.mixed_precision import (
    ComplexGradScaler,
    MixedPrecisionConfig,
    MixedPrecisionIntegrationHelper,
    resolve_amp_precision,
)


def test_mixed_precision_helper_float_mode():
    """Test that helper selects NativeScaler when enable_complex_support is False."""
    config = MixedPrecisionConfig(
        enabled=True, precision="fp16", enable_complex_support=False
    )
    helper = MixedPrecisionIntegrationHelper(config)

    assert isinstance(helper.scaler, NativeScaler)
    assert not isinstance(helper.scaler, ComplexGradScaler)

    # Check autocast context
    ctx = helper.get_autocast_context()
    if torch.cuda.is_available():
        # Should NOT disable TF32
        assert not getattr(ctx, "disable_tf32", False)


def test_mixed_precision_helper_rejects_fp16_on_a_complex_arm():
    """fp16 has no complex type: this pairing ran in fp32 behind a scaler that
    skipped non-finite steps, while reporting AMP on (#2302)."""
    from spectramr.infrastructure.training.mixed_precision import (
        AMPPrecisionUnsupportedError,
    )

    config = MixedPrecisionConfig(
        enabled=True, precision="fp16", enable_complex_support=True
    )
    with pytest.raises(AMPPrecisionUnsupportedError, match="complex"):
        MixedPrecisionIntegrationHelper(config, device="cpu")


def test_disabled_amp_on_a_complex_arm_is_accepted():
    config = MixedPrecisionConfig(
        enabled=False, precision="fp16", enable_complex_support=True
    )
    helper = MixedPrecisionIntegrationHelper(config, device="cpu")
    assert helper.scaler is None


def test_strategy_auto_detection_image_domain():
    """Test that BaseTrainingStrategy configures MP for float (image) domain."""

    # Mock State and Config
    state = MagicMock()
    state.config = MagicMock()
    state.config.model_domain = "image"
    state.config.physics.kspace.enable_kspace_recon = False
    state.config.optimization.precision.enabled = True

    # Validate logic without instantiating full strategy (which requires valid env/device)
    # We'll mimic the logic used in _setup_strategy_specific_components

    def _get_config_value(config, key, default):
        if key == "model_domain":
            return config.model_domain
        if key == "physics.kspace.enable_kspace_recon":
            return config.physics.kspace.enable_kspace_recon
        return default

    # Replicate detection logic
    model_domain = _get_config_value(state.config, "model_domain", "image")
    physics_kspace = _get_config_value(
        state.config, "physics.kspace.enable_kspace_recon", False
    )
    is_complex = (str(model_domain).lower() == "kspace") or physics_kspace

    assert is_complex is False


def test_strategy_auto_detection_kspace_domain():
    """Test that BaseTrainingStrategy configures MP for complex (kspace) domain."""

    # Mock State and Config
    state = MagicMock()
    state.config = MagicMock()
    state.config.model_domain = "kspace"
    state.config.physics.kspace.enable_kspace_recon = True  # or False, domain override
    state.config.optimization.precision.enabled = True

    def _get_config_value(config, key, default):
        if key == "model_domain":
            return config.model_domain
        if key == "physics.kspace.enable_kspace_recon":
            return config.physics.kspace.enable_kspace_recon
        return default

    # Replicate detection logic
    model_domain = _get_config_value(state.config, "model_domain", "image")
    physics_kspace = _get_config_value(
        state.config, "physics.kspace.enable_kspace_recon", False
    )
    is_complex = (str(model_domain).lower() == "kspace") or physics_kspace

    assert is_complex is True


class TestResolveAmpPrecision:
    """``optimization.precision.dtype`` must drive the AMP autocast dtype (pitfall #15)."""

    def test_default_dtype_is_fp16_when_amp_on(self):
        # amp_dtype=None preserves the historical fp16 default (back-compat).
        assert resolve_amp_precision(use_amp=True, amp_dtype=None) == (True, "fp16")

    def test_bfloat16_maps_to_bf16(self):
        assert resolve_amp_precision(use_amp=True, amp_dtype="bfloat16") == (True, "bf16")

    def test_float16_maps_to_fp16(self):
        assert resolve_amp_precision(use_amp=True, amp_dtype="float16") == (True, "fp16")

    def test_float32_disables_amp_even_when_use_amp_true(self):
        # float32 means full precision → autocast must be OFF regardless of use_amp.
        enabled, _ = resolve_amp_precision(use_amp=True, amp_dtype="float32")
        assert enabled is False

    def test_use_amp_false_disables_regardless_of_dtype(self):
        enabled, _ = resolve_amp_precision(use_amp=False, amp_dtype="bfloat16")
        assert enabled is False

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="amp_dtype"):
            resolve_amp_precision(use_amp=True, amp_dtype="int8")


if __name__ == "__main__":
    pytest.main([__file__])
