"""kspace_masked: bidirectional line transformer with a learned mask token."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.kspace_masked_backbone import KSpaceMaskedBackbone

SIZE, C = 16, 8


def _net(axis: int = -2) -> KSpaceMaskedBackbone:
    torch.manual_seed(0)
    net = KSpaceMaskedBackbone(
        in_channels=C, out_channels=C, image_size=SIZE, dim=32, depth=2, heads=2, line_axis=axis
    )
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=0.05)
    return net


def _rows(keep: list[int], batch: int = 1) -> torch.Tensor:
    m = torch.zeros(batch, C, SIZE, SIZE)
    m[:, :, keep, :] = 1
    return m


@pytest.mark.parametrize("axis", [-2, -1])
def test_forward_shape(axis: int) -> None:
    m = _rows(list(range(0, SIZE, 2)))
    if axis == -1:
        m = m.transpose(-1, -2).contiguous()
    out = _net(axis)(torch.randn(2, C, SIZE, SIZE), timesteps=torch.zeros(2), mask=m)
    assert out.shape == (2, C, SIZE, SIZE) and torch.isfinite(out).all()


def test_masked_line_input_is_ignored() -> None:
    net, m = _net(), _rows([0, 4, 8, 12])
    x = torch.randn(1, C, SIZE, SIZE)
    y = x.clone()
    y[:, :, 5, :] += 3.0  # line 5 is masked
    assert torch.allclose(net(x, mask=m), net(y, mask=m), atol=1e-5)


def test_acquired_line_change_reaches_every_line() -> None:
    net, m = _net(), _rows([0, 4, 8, 12])
    x = torch.randn(1, C, SIZE, SIZE)
    y = x.clone()
    y[:, :, 8, :] += 3.0
    a, b = net(x, mask=m), net(y, mask=m)
    assert not torch.allclose(a[:, :, 1], b[:, :, 1]) and not torch.allclose(
        a[:, :, 14], b[:, :, 14]
    )


def test_all_ones_mask_is_finite() -> None:
    out = _net()(torch.randn(1, C, SIZE, SIZE), mask=torch.ones(1, C, SIZE, SIZE))
    assert torch.isfinite(out).all()


def test_mask_batch_one_broadcasts() -> None:
    out = _net()(torch.randn(3, C, SIZE, SIZE), mask=_rows([0, 8]))
    assert out.shape[0] == 3


def test_requires_mask_and_rejects_cross_axis_mask() -> None:
    net = _net()
    with pytest.raises(ValueError, match="needs mask"):
        net(torch.randn(1, C, SIZE, SIZE))
    cols = torch.zeros(1, C, SIZE, SIZE)
    cols[..., 3] = 1
    with pytest.raises(ValueError, match="line_axis"):
        net(torch.randn(1, C, SIZE, SIZE), mask=cols)


def test_wrong_size_mask_raises() -> None:
    with pytest.raises(ValueError, match="lines"):
        _net()(torch.randn(1, C, SIZE, SIZE), mask=torch.ones(1, C, 1, SIZE))


def test_mask_orientation_is_rechecked_when_switching_to_eval() -> None:
    net = _net(-2)
    good = _rows([0, 4, 8, 12])
    bad = torch.zeros(1, C, SIZE, SIZE)
    bad[..., 3] = 1
    x = torch.randn(1, C, SIZE, SIZE)
    net.train()(x, mask=good)
    with pytest.raises(ValueError, match="line_axis"):
        net.eval()(x, mask=bad)
