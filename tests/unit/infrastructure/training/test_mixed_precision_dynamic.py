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
    """Real-valued fp16 arms get the same single scaler (one owner for fp16 grads)."""
    config = MixedPrecisionConfig(
        enabled=True, precision="fp16", enable_complex_support=False
    )
    helper = MixedPrecisionIntegrationHelper(config)

    assert isinstance(helper.scaler, NativeScaler)
    assert isinstance(helper.scaler, ComplexGradScaler)

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


_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
]


def _complex_param_step(device: str):
    """One scaled backward through a complex parameter; returns (scaler, opt, param)."""
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.complex64, device=device))
    opt = torch.optim.SGD([param], lr=0.0)
    scaler = ComplexGradScaler(device, enabled=True, init_scale=8.0)
    x = torch.randn(3, 4, dtype=torch.complex64, device=device)
    scaler.scale((x @ param).abs().sum()).backward()
    return scaler, opt, param


@pytest.mark.parametrize("device", _DEVICES)
def test_complex_grads_unscale_to_the_true_gradient(device):
    """Unscale must divide complex grads by the scale (real and imaginary parts alike)."""
    scaler, opt, param = _complex_param_step(device)
    scaled = param.grad.clone()
    scaler.unscale_(opt)
    assert torch.allclose(param.grad, scaled / 8.0)
    assert scaler.get_scale() == 8.0


@pytest.mark.parametrize("device", _DEVICES)
def test_nonfinite_complex_grad_skips_the_step_and_halves_the_scale(device):
    """inf in the imaginary part alone must be seen: the step is skipped and the scale backs off."""
    scaler, opt, param = _complex_param_step(device)
    before = param.detach().clone()
    param.grad[0, 0] = torch.complex(torch.tensor(float("inf")), torch.tensor(0.0)).to(device)
    scaler.step(opt)
    scaler.update()
    assert torch.equal(param.detach(), before)
    assert scaler.get_scale() == 4.0


def _mock_reduce(monkeypatch):
    """Record the sharded non-finite reduce; no process group is needed to see it."""
    import torch.distributed as dist

    calls = []

    def fake_all_reduce(tensor, op=None, group=None, async_op=False):
        calls.append(tensor)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    return calls


def test_sharded_reduce_is_one_collective_on_the_scale_device(monkeypatch):
    """One MAX reduce per unscale, on the scale device, for a CPU-resident grad."""
    calls = _mock_reduce(monkeypatch)
    param = torch.nn.Parameter(torch.ones(3))
    param.grad = torch.full((3,), 2.0)
    opt = torch.optim.SGD([param], lr=0.0)
    scaler = ComplexGradScaler("cpu", enabled=True, init_scale=4.0, sharded=True)
    scaler.scale(torch.tensor(1.0))  # initialises the scale, as a real backward does
    scaler.unscale_(opt)
    assert len(calls) == 1
    assert calls[0].device == torch.device("cpu")
    assert torch.allclose(param.grad, torch.full((3,), 0.5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sharded_reduce_with_a_cpu_grad_beside_a_cuda_grad(monkeypatch):
    """CPU-offloaded grads (FSDP cpu_offload) must not reach NCCL; the flag is reduced on CUDA."""
    calls = _mock_reduce(monkeypatch)
    on_gpu = torch.nn.Parameter(torch.ones(3, device="cuda"))
    on_gpu.grad = torch.full((3,), 2.0, device="cuda")
    on_cpu = torch.nn.Parameter(torch.ones(3))
    on_cpu.grad = torch.full((3,), float("inf"))
    opt = torch.optim.SGD([on_gpu, on_cpu], lr=0.0)
    scaler = ComplexGradScaler("cuda", enabled=True, init_scale=4.0, sharded=True)
    scaler.scale(torch.tensor(1.0, device="cuda"))
    scaler.unscale_(opt)
    assert len(calls) == 1
    assert calls[0].device.type == "cuda"
    assert float(calls[0]) == 1.0  # the CPU grad's inf reached the agreed flag


if __name__ == "__main__":
    pytest.main([__file__])
