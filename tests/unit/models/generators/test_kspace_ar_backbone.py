"""kspace_ar: prefix-causal line transformer (acquired prefix, causal missing lines)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.kspace_ar_backbone import KSpaceARBackbone

SIZE, C = 16, 8


def _net(axis: int = -2) -> KSpaceARBackbone:
    torch.manual_seed(0)
    net = KSpaceARBackbone(
        in_channels=C, out_channels=C, image_size=SIZE, dim=32, depth=2, heads=2, line_axis=axis
    )
    with torch.no_grad():  # adaLN-Zero makes a fresh trunk the identity
        for p in net.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=0.05)
    return net


def _rows(keep: list[int]) -> torch.Tensor:
    m = torch.zeros(1, C, SIZE, SIZE)
    m[:, :, keep, :] = 1
    return m


@pytest.mark.parametrize("axis", [-2, -1])
def test_forward_shape(axis: int) -> None:
    m = _rows([0, 4, 8, 12])
    if axis == -1:
        m = m.transpose(-1, -2).contiguous()
    out = _net(axis)(torch.randn(2, C, SIZE, SIZE), timesteps=torch.zeros(2), mask=m)
    assert out.shape == (2, C, SIZE, SIZE) and torch.isfinite(out).all()


def test_missing_line_input_is_ignored_and_acquired_line_is_seen() -> None:
    net, m = _net(), _rows([0, 4, 8, 12])
    x = torch.randn(1, C, SIZE, SIZE)
    miss, seen = x.clone(), x.clone()
    miss[:, :, 5, :] += 3.0
    seen[:, :, 8, :] += 3.0
    assert torch.allclose(net(x, mask=m), net(miss, mask=m), atol=1e-5)
    assert not torch.allclose(net(x, mask=m), net(seen, mask=m))


def test_acquired_line_prediction_ignores_missing_slots() -> None:
    """Acquired lines attend only to acquired ones, so the missing-line token cannot reach them."""
    net, m = _net(), _rows([0, 4, 8, 12])
    x = torch.randn(1, C, SIZE, SIZE)
    base = net(x, mask=m)
    with torch.no_grad():
        net.mask_token.add_(1.0)
    moved = net(x, mask=m)
    for acquired in (0, 4, 8, 12):
        assert torch.allclose(base[:, :, acquired], moved[:, :, acquired], atol=1e-5)
    assert not torch.allclose(base[:, :, 5], moved[:, :, 5])


def test_requires_mask() -> None:
    with pytest.raises(ValueError, match="needs mask"):
        _net()(torch.randn(1, C, SIZE, SIZE))


def test_rejects_mask_varying_within_line() -> None:
    m = torch.zeros(1, C, SIZE, SIZE)
    m[..., 3] = 1
    with pytest.raises(ValueError, match="line_axis"):
        _net(-2)(torch.randn(1, C, SIZE, SIZE), mask=m)


def test_rejects_image_feature_domain() -> None:
    with pytest.raises(ValueError, match="k-space"):
        KSpaceARBackbone(image_size=SIZE, dim=32, depth=1, heads=2, feature_domain="image")


def test_all_ones_mask_is_finite() -> None:
    out = _net()(torch.randn(1, C, SIZE, SIZE), mask=torch.ones(1, C, SIZE, SIZE))
    assert torch.isfinite(out).all()


def test_grads_flow() -> None:
    net = _net()
    net(
        torch.randn(2, C, SIZE, SIZE), timesteps=torch.zeros(2), mask=_rows([0, 8])
    ).sum().backward()
    assert all(p.grad is not None for p in net.parameters())


def test_non_square_field_raises_instead_of_dropping_lines() -> None:
    with pytest.raises(ValueError, match="square"):
        _net()(torch.randn(1, C, SIZE // 2, SIZE), mask=torch.ones(1, C, SIZE // 2, SIZE))


def test_wrong_size_mask_raises_every_call() -> None:
    net, x = _net(), torch.randn(1, C, SIZE, SIZE)
    net(x, mask=_rows([0, 8]))
    with pytest.raises(ValueError, match="square"):
        net(x, mask=torch.ones(1, C, SIZE // 2, SIZE))
