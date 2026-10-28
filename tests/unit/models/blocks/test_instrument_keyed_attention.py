"""Tests for instrument-keyed frame fusion.

The claim this block exists to support is that marker-keyed and content-keyed
attention are DIFFERENT mechanisms, not two names for one. A forward pass that
does not crash proves nothing about that, so the decisive tests here compare the
attention maps directly and check the parameter counts that make the ablation
capacity-matched.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.blocks.instrument_keyed_attention import (  # noqa: E402
    InstrumentKeyedCrossAttention,
)

B, N, C, H, W = 2, 8, 16, 10, 10


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(B, N, C, H, W), torch.randn(B, N, C, H, W)


def _block(keys: str, **kw) -> InstrumentKeyedCrossAttention:
    torch.manual_seed(0)
    return InstrumentKeyedCrossAttention(channels=C, n_frames=N, keys=keys, heads=4, **kw)


# ── contract ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("keys", ["content", "marker", "none"])
def test_every_routing_returns_the_same_shape(keys: str) -> None:
    frames, inst = _inputs()
    assert _block(keys)(frames, inst).shape == (B, C, H, W)


def test_unknown_routing_raises() -> None:
    """A silent fallback to content keying would let an arm report an ablation
    it never ran (CLAUDE.md #9)."""
    with pytest.raises(ValueError, match="not one of"):
        _block("psychic")


def test_marker_routing_without_an_instrument_raises() -> None:
    """Otherwise this becomes content-keyed attention under a marker label."""
    frames, _ = _inputs()
    with pytest.raises(ValueError, match="needs the instrument features"):
        _block("marker")(frames)


def test_instrument_of_the_wrong_shape_raises() -> None:
    frames, _ = _inputs()
    with pytest.raises(ValueError, match="must match frames"):
        _block("marker")(frames, torch.randn(B, N, C, H + 2, W))


def test_frame_count_is_a_contract() -> None:
    """The frame embedding is per-frame, so a mismatch is not broadcastable."""
    with pytest.raises(ValueError, match="built for n_frames"):
        _block("content")(torch.randn(B, N + 1, C, H, W))


def test_rank_is_checked() -> None:
    with pytest.raises(ValueError, match=r"\[B, N, C, H, W\]"):
        _block("content")(torch.randn(B, N, H, W))


def test_heads_must_divide_channels() -> None:
    with pytest.raises(ValueError, match="divisible"):
        InstrumentKeyedCrossAttention(channels=15, n_frames=N, heads=4)


# ── the mechanism ────────────────────────────────────────────────────────────


def test_marker_and_content_keying_produce_different_attention() -> None:
    """The decisive check. Identical seeds, identical frames, identical
    parameters — the ONLY difference is where the queries and keys are read
    from. If these maps matched, the ablation would be measuring nothing."""
    frames, inst = _inputs()
    content, marker = _block("content"), _block("marker")
    w_content = content.attention_weights(frames)
    w_marker = marker.attention_weights(frames, inst)
    assert w_content.shape == w_marker.shape
    assert not torch.allclose(w_content, w_marker)
    assert float((w_content - w_marker).detach().abs().max()) > 0.05


def test_marker_keying_ignores_the_anatomy_it_fuses() -> None:
    """The identifiability claim: with the instrument held fixed, changing the
    anatomy must not move the attention map. That independence is what makes
    the fusion weights a function of geometry alone."""
    frames, inst = _inputs()
    blk = _block("marker")
    a = blk.attention_weights(frames, inst)
    b = blk.attention_weights(torch.randn_like(frames), inst)
    assert torch.allclose(a, b, atol=1e-6)


def test_content_keying_does_not_ignore_the_anatomy() -> None:
    """The contrast to the test above — otherwise that one proves nothing."""
    frames, _ = _inputs()
    blk = _block("content")
    a = blk.attention_weights(frames)
    b = blk.attention_weights(torch.randn_like(frames))
    assert not torch.allclose(a, b, atol=1e-6)


def test_content_and_marker_are_capacity_matched() -> None:
    """The comparison between them is the only one not confounded by size."""
    n_content = sum(p.numel() for p in _block("content").parameters())
    n_marker = sum(p.numel() for p in _block("marker").parameters())
    assert n_content == n_marker


def test_none_drops_exactly_the_frame_attention_parameters() -> None:
    """The stated capacity gap is 4C^2 + NC, the projections plus frame embedding."""
    gap = sum(p.numel() for p in _block("content").parameters())
    assert gap == 4 * C * C + N * C


def test_none_carries_no_attention_parameters() -> None:
    """Absent, not disabled: a switched-off attention that still ran its
    projections would be a different model, not an ablation of one. The arms
    that compare against it must state the capacity gap."""
    blk = _block("none")
    assert sum(p.numel() for p in blk.parameters()) == 0
    assert blk.attn is None
    with pytest.raises(ValueError, match="computes no attention"):
        blk.attention_weights(*_inputs())


@pytest.mark.parametrize("keys", ["content", "marker", "none"])
def test_identity_at_init(keys: str) -> None:
    """out_proj starts at zero, so every routing begins at the frame mean and
    the attention has to EARN its contribution. Across eight exp-11 arms, a
    block whose output norm started 7-190x its input never recovered."""
    frames, inst = _inputs()
    assert torch.allclose(_block(keys)(frames, inst), frames.mean(dim=1), atol=1e-6)


def test_attention_trains_away_from_the_frame_mean() -> None:
    """Identity-at-init must not mean identity-forever: a block that cannot
    leave its initialisation is a facade with a gradient (#16)."""
    frames, inst = _inputs()
    target = frames[:, 0]
    blk = _block("marker")
    opt = torch.optim.Adam(blk.parameters(), lr=1e-2)
    for _ in range(30):
        loss = (blk(frames, inst) - target).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert not torch.allclose(blk(frames, inst), frames.mean(dim=1), atol=1e-4)


def test_query_key_path_is_dormant_at_init_and_live_once_trained() -> None:
    """A consequence of the zero-initialised residual worth stating explicitly.

    The instrument enters ONLY through the queries and keys, and at step 0
    ``out_proj`` is zero, so the whole attention branch contributes nothing and
    the gradient reaching the instrument is exactly zero. Only ``out_proj`` and
    ``v_proj`` learn on the first step; ``q_proj``/``k_proj`` — and any future
    learnable fiducial — start moving once ``out_proj`` leaves zero.

    This is the standard zero-init residual behaviour (ReZero, diffusion UNet
    output blocks), not a dead path, and the second half proves it: after a few
    steps the instrument gradient is non-zero.
    """
    frames, inst = _inputs()
    blk = _block("marker")

    probe = inst.clone().requires_grad_(True)
    blk(frames, probe).pow(2).mean().backward()
    assert probe.grad is not None
    assert float(probe.grad.norm()) == 0.0, "zero-init residual should be dormant"

    opt = torch.optim.Adam(blk.parameters(), lr=1e-2)
    for _ in range(10):
        loss = (blk(frames, inst) - frames[:, 0]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    probe = inst.clone().requires_grad_(True)
    blk.zero_grad(set_to_none=True)
    blk(frames, probe).pow(2).mean().backward()
    assert float(probe.grad.norm()) > 0.0, "the marker path never woke up"


# ── registry integration ─────────────────────────────────────────────────────


def test_block_is_registered_and_reachable_through_the_registry() -> None:
    """The repo resolves blocks through BLOCK_REGISTRY (CLAUDE.md #6), and 47
    others already do. A block that is merely importable is reachable from
    Python and NOT from the dispatch path every other block uses."""
    from spectramr.models.blocks import create_block, list_registered_blocks

    assert "instrument_keyed_attention" in list_registered_blocks()
    blk = create_block(
        "instrument_keyed_attention", channels=C, n_frames=N, keys="marker", heads=4
    )
    assert isinstance(blk, InstrumentKeyedCrossAttention)
    frames, inst = _inputs()
    assert blk(frames, inst).shape == (B, C, H, W)


def test_block_is_exported_from_the_package() -> None:
    """`from spectramr.models.blocks import InstrumentKeyedCrossAttention` must
    work, as it does for every other canonical block."""
    import spectramr.models.blocks as blocks

    assert hasattr(blocks, "InstrumentKeyedCrossAttention")
    assert "InstrumentKeyedCrossAttention" in blocks.__all__
    assert "AttentionKeys" in blocks.__all__
