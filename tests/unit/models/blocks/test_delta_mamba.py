"""Unit tests for DeltaMamba: an external Δ scale and shift on the selective scan."""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from spectramr.models.blocks.delta_mamba import DeltaMamba, DeltaMambaStack

B, L, D = 2, 16, 32  # DeltaMamba d_inner is expand * D = 2 * D

# The official kernel needs both mamba_ssm and a GPU; the cpu_tests lane skips these.
requires_mamba_ssm_gpu = pytest.mark.skipif(
    importlib.util.find_spec("mamba_ssm") is None or not torch.cuda.is_available(),
    reason="needs mamba_ssm and a CUDA device",
)


def _block(**kw) -> DeltaMamba:
    torch.manual_seed(0)
    return DeltaMamba(D, d_state=4, backend="reference", **kw).eval()


def _input() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(B, L, D)


def test_delta_scale_changes_the_output():
    block, x = _block(), _input()
    base = block(x)
    scaled = block(x, delta_scale=torch.full((2 * D,), 0.5))
    assert not torch.allclose(base, scaled)


def test_delta_shift_changes_the_output():
    block, x = _block(), _input()
    base = block(x)
    shifted = block(x, delta_shift=torch.full((2 * D,), 2.0))
    assert not torch.allclose(base, shifted)


def test_unit_scale_and_zero_shift_are_the_identity_of_the_unmodulated_scan():
    block, x = _block(), _input()
    base = block(x)
    unit = block(x, delta_scale=torch.ones(2 * D), delta_shift=torch.zeros(2 * D))
    torch.testing.assert_close(base, unit)


def test_zero_delta_scale_leaves_only_the_skip_path():
    """Δ -> 0 freezes the state, so y = D * u * silu(z) and nothing else."""
    block, x = _block(), _input()
    out = block(x, delta_scale=torch.zeros(2 * D))

    xs, z = block.in_proj(x).transpose(1, 2).chunk(2, dim=1)
    xs = F.silu(block.conv1d(xs)[..., :L])
    skip = (block.D[None, :, None] * xs * F.silu(z)).transpose(1, 2)
    torch.testing.assert_close(out, block.out_proj(skip))


def test_stack_applies_the_same_modulation_to_every_layer_and_keeps_shape():
    torch.manual_seed(0)
    stack = DeltaMambaStack(D, num_layers=2, d_state=4, backend="reference").eval()
    x = _input()
    out = stack(x, delta_scale=torch.full((2 * D,), 0.5), delta_shift=torch.zeros(2 * D))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_gradients_reach_the_delta_projection_and_the_state_matrix():
    torch.manual_seed(0)
    stack = DeltaMambaStack(D, num_layers=1, d_state=4, backend="reference")
    stack(_input(), delta_scale=torch.ones(2 * D)).sum().backward()
    layer = stack.layers[0]
    assert layer.dt_proj.weight.grad is not None and layer.dt_proj.weight.grad.abs().sum() > 0
    assert layer.A_log.grad is not None and layer.A_log.grad.abs().sum() > 0


def test_cuda_backend_raises_without_mamba_ssm_even_with_the_fallback_env(monkeypatch):
    """The GRU fallback has no Δ, so DeltaMamba must not degrade to it."""
    real_find_spec = importlib.util.find_spec

    def no_mamba(name, *a, **k):
        return None if name == "mamba_ssm" else real_find_spec(name, *a, **k)

    monkeypatch.setenv("SPECTRAMR_ALLOW_MAMBA_FALLBACK", "1")
    monkeypatch.setattr(importlib.util, "find_spec", no_mamba)
    with pytest.raises(ImportError, match="mamba_ssm"):
        DeltaMamba(D, backend="cuda")


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        DeltaMamba(D, backend="gru")


@requires_mamba_ssm_gpu
def test_cuda_scan_matches_reference_scan_on_gpu():
    torch.manual_seed(0)
    cuda = DeltaMamba(D, d_state=4, backend="cuda").cuda().eval()
    ref = DeltaMamba(D, d_state=4, backend="reference").cuda().eval()
    ref.load_state_dict(cuda.state_dict())
    x = _input().cuda()
    scale = torch.rand(2 * D, device="cuda") + 0.5
    torch.testing.assert_close(
        cuda(x, delta_scale=scale), ref(x, delta_scale=scale), atol=1e-4, rtol=1e-4
    )


@requires_mamba_ssm_gpu
def test_unit_scale_matches_official_mamba_on_gpu():
    from mamba_ssm import Mamba

    torch.manual_seed(0)
    official = Mamba(D, d_state=4, use_fast_path=False).cuda().eval()
    block = DeltaMamba(D, d_state=4, backend="cuda").cuda().eval()
    block.load_state_dict(official.state_dict())
    x = _input().cuda()
    torch.testing.assert_close(block(x), official(x), atol=1e-4, rtol=1e-4)


def test_freeze_dt_bias_stops_the_bias_gradient_only():
    stack = DeltaMambaStack(D, num_layers=1, d_state=4, backend="reference")
    stack.freeze_dt_bias()
    stack(_input(), delta_scale=torch.ones(2 * D)).sum().backward()
    layer = stack.layers[0]
    assert layer.dt_proj.bias.grad is None
    assert layer.dt_proj.weight.grad is not None
