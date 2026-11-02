"""MAEGenerator masking contract (#2393)."""

from __future__ import annotations

import torch

from spectramr.models.mae.mae_generator import MAEGenerator


def _tiny(mask_ratio: float = 0.75) -> MAEGenerator:
    torch.manual_seed(0)
    return MAEGenerator(
        img_size=(32, 32),
        patch_size=(16, 16),
        in_channels=1,
        embed_dim=32,
        encoder_depth=1,
        decoder_depth=1,
        num_heads=2,
        mask_ratio=mask_ratio,
    ).eval()


def test_disable_spatial_masking_is_read_and_deterministic() -> None:
    g = _tiny()
    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        a = g(x, disable_spatial_masking=True)
        b = g(x, disable_spatial_masking=True)
    assert torch.equal(a, b)


def test_default_forward_still_drops_patches() -> None:
    g = _tiny()
    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        torch.manual_seed(1)
        a = g(x)
        torch.manual_seed(2)
        b = g(x)
    assert not torch.allclose(a, b)


def test_zero_ratio_keeps_patch_order() -> None:
    g = _tiny(mask_ratio=0.0)
    tokens = torch.randn(2, 4, 8)
    kept, mask, ids_restore = g.random_masking(tokens, 0.0)
    assert torch.equal(kept, tokens)
    assert mask.sum() == 0
    assert torch.equal(ids_restore, torch.arange(4).expand(2, 4))


def test_output_rank_matches_a_2d_input() -> None:
    g = _tiny()
    with torch.no_grad():
        out = g(torch.randn(2, 1, 32, 32), disable_spatial_masking=True)
    assert out.shape == (2, 1, 32, 32)


def test_declares_it_owns_spatial_masking() -> None:
    assert MAEGenerator.owns_spatial_masking is True


def test_positions_follow_spatial_location_not_keep_order() -> None:
    g = MAEGenerator(
        img_size=(16, 16),
        patch_size=(4, 4),
        in_channels=1,
        embed_dim=16,
        encoder_depth=1,
        decoder_depth=1,
        num_heads=2,
        mask_ratio=0.75,
    )
    pos = g.encoder.encoder.pos_embed
    torch.manual_seed(3)
    _, mask, _ = g.forward_encoder(torch.randn(2, 1, 16, 16), 0.75)
    torch.manual_seed(3)
    g(torch.randn(2, 1, 16, 16)).square().mean().backward()
    touched = pos.grad[0].abs().sum(-1) > 0
    kept = (mask == 0).any(0)
    assert touched[0]
    # A patch row gets gradient exactly when that spatial patch was visible in some sample.
    assert torch.equal(touched[1:], kept)
    assert kept.sum() > mask.shape[1] * 0.25
