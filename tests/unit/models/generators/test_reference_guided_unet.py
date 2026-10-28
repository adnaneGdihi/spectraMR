"""The two-encoder reference-guided U-Net (improvements 4 and 5 on #2452's arms)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.reference_guided_unet import ReferenceGuidedUNet


def _net(**kw) -> ReferenceGuidedUNet:
    args = {
        "in_channels": 24,
        "out_channels": 8,
        "reference_channels": 8,
        "width": 8,
        "levels": 3,
        "heads": 2,
        "max_timesteps": 24.0,
    }
    args.update(kw)
    torch.manual_seed(0)
    return ReferenceGuidedUNet(**args)


def test_output_is_the_target_width() -> None:
    out = _net()(torch.randn(2, 24, 16, 16), timesteps=torch.tensor([3, 20]))
    assert out.shape == (2, 8, 16, 16)


def test_the_reference_changes_the_output() -> None:
    """Planted against the failure the backbone exists to fix: a reference the
    output does not depend on (#2452's concatenated T1 was at least mixed in)."""
    net = _net().eval()
    x = torch.randn(1, 24, 16, 16)
    swapped = x.clone()
    swapped[:, :8] = torch.randn(1, 8, 16, 16)
    t = torch.tensor([5])
    assert not torch.allclose(net(x, timesteps=t), net(swapped, timesteps=t))


def test_the_source_encoder_receives_gradient() -> None:
    net = _net()
    net(torch.randn(2, 24, 16, 16), timesteps=torch.tensor([1, 2])).square().mean().backward()
    for name in ("source_stem", "source_enc", "fusion", "cross"):
        grads = [p.grad for n, p in net.named_parameters() if n.startswith(name)]
        assert grads and all(g is not None and g.abs().sum() > 0 for g in grads), name


def test_the_residual_base_is_the_noisy_state_not_the_smaps() -> None:
    """With the head silenced the output is x_t, the channels right after the reference."""
    net = _net().eval()
    for p in net.head.parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(1, 24, 16, 16)
    assert torch.equal(net(x, timesteps=torch.tensor([0])), x[:, 8:16])


def test_the_timestep_conditions_the_output() -> None:
    net = _net().eval()
    x = torch.randn(1, 24, 16, 16)
    assert not torch.allclose(
        net(x, timesteps=torch.tensor([1])), net(x, timesteps=torch.tensor([20]))
    )


def test_a_contrast_embedding_is_projected_or_refused() -> None:
    net = _net(contrast_emb_dim=16).eval()
    x = torch.randn(1, 24, 16, 16)
    t = torch.tensor([4])
    a = net(x, timesteps=t, contrast_emb=torch.randn(1, 16))
    b = net(x, timesteps=t, contrast_emb=torch.randn(1, 16))
    assert not torch.allclose(a, b)
    with pytest.raises(ValueError):
        net(x, timesteps=t, contrast_emb=torch.randn(1, 5))


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"reference_channels": 0}, "needs a reference"),
        ({"reference_channels": 20}, "target channels"),
        ({"levels": 0}, "levels"),
        ({"heads": 3}, "divide"),
    ],
)
def test_illegal_construction_raises(kw, match) -> None:
    with pytest.raises(ValueError, match=match):
        _net(**kw)


def test_a_size_the_levels_cannot_halve_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _net()(torch.randn(1, 24, 18, 18), timesteps=torch.tensor([0]))
