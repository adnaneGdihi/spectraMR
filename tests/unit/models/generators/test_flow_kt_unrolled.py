"""flow_kt_unrolled on tiny synthetic tensors (CPU; no arm model is built)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import sense_adjoint, sense_forward
from spectramr.models.generators.flow_kt_unrolled import FlowKTUnrolled

B, C, E, T, H, W = 2, 3, 4, 3, 8, 6


def _maps(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(B, C, H, W, generator=g, dtype=torch.complex64)
    return raw / raw.abs().pow(2).sum(dim=1, keepdim=True).sqrt()


def _batch(full: bool = False, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(B, E, T, H, W, generator=g, dtype=torch.complex64)
    maps = _maps(seed)
    mask = (
        torch.ones(B, E, T, H, W)
        if full
        else (torch.rand(B, E, T, H, W, generator=g) > 0.5).float()
    )
    kspace = sense_forward(images[:, None], maps[:, :, None, None], mask[:, None])
    return images, kspace, mask, maps


def _zero_regularisers(model: FlowKTUnrolled) -> None:
    with torch.no_grad():
        for p in model.regularisers.parameters():
            p.zero_()


def test_registered_for_the_flow_regime() -> None:
    from spectramr.config.schemas.enums import Regime
    from spectramr.models.registry import get_model_capabilities, get_model_class

    assert get_model_class("flow_kt_unrolled") is FlowKTUnrolled
    caps = get_model_capabilities("flow_kt_unrolled")
    assert caps.accepts_complex and caps.input_domain == "kspace" and caps.output_domain == "image"
    assert Regime.FLOW in caps.workflows
    # [B, C, E, T, H, W] has no [B, C, *spatial] reading, so no rank is declared and
    # the generic image probe is skipped rather than fed a batch it would reject.
    assert caps.spatial_dims is None
    assert "forward" in FlowKTUnrolled.synthetic_forward_probe_skip


def test_the_six_dimensional_sense_pair_is_adjoint() -> None:
    """The model relies on sense_forward/adjoint broadcasting over encodings and frames."""
    x, _, mask, maps = _batch()
    y = torch.randn(B, C, E, T, H, W, dtype=torch.complex64)
    ax = sense_forward(x[:, None], maps[:, :, None, None], mask[:, None])
    ahy = sense_adjoint(y, maps[:, :, None, None], mask[:, None])[:, 0]
    lhs, rhs = (ax.conj() * y).sum(), (x.conj() * ahy).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-4)


def test_output_shape_and_finite_gradients() -> None:
    _, kspace, mask, maps = _batch()
    model = FlowKTUnrolled(encodings=E, cascades=2, features=8, layers=2)
    out = model(kspace, mask, maps)
    assert out.shape == (B, E, T, H, W) and out.dtype == torch.complex64
    out.abs().pow(2).mean().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_fully_sampled_data_is_reconstructed_exactly_without_a_regulariser() -> None:
    images, kspace, mask, maps = _batch(full=True)
    model = FlowKTUnrolled(encodings=E, cascades=3, features=8, layers=2)
    _zero_regularisers(model)
    assert torch.allclose(model(kspace, mask, maps), images, atol=1e-4)


def test_output_scales_with_the_measurement() -> None:
    _, kspace, mask, maps = _batch()
    model = FlowKTUnrolled(encodings=E, cascades=2, features=8, layers=2).eval()
    with torch.no_grad():
        base, scaled = model(kspace, mask, maps), model(7.5 * kspace, mask, maps)
    assert torch.allclose(scaled, 7.5 * base, rtol=1e-4, atol=1e-5)


def test_checkpointing_changes_neither_output_nor_gradient() -> None:
    _, kspace, mask, maps = _batch()
    plain = FlowKTUnrolled(encodings=E, cascades=2, features=8, layers=2)
    ckpt = FlowKTUnrolled(encodings=E, cascades=2, features=8, layers=2, checkpoint_cascades=True)
    ckpt.load_state_dict(plain.state_dict())
    outs = []
    for model in (plain, ckpt):
        out = model(kspace, mask, maps)
        out.abs().sum().backward()
        outs.append((out.detach(), model.step.grad.clone()))
    assert torch.allclose(outs[0][0], outs[1][0]) and torch.allclose(outs[0][1], outs[1][1])


@pytest.mark.parametrize(
    ("kwargs", "call", "match"),
    [
        ({"features": 12}, None, "multiple of 8"),
        ({"encodings": 1}, None, "encodings >= 2"),
        ({}, "no_mask", "sampling mask"),
        ({"encodings": 2}, "wrong_encodings", "E=2"),
    ],
)
def test_invalid_construction_or_input_raises(kwargs, call, match) -> None:
    _, kspace, mask, maps = _batch()
    with pytest.raises(ValueError, match=match):
        model = FlowKTUnrolled(**{"features": 8, "layers": 2, **kwargs})
        if call == "no_mask":
            model(kspace)
        elif call == "wrong_encodings":
            model(kspace, mask, maps)
