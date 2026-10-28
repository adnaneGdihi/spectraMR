"""SwinBlock: the shifted-window mask must follow the resolution (#1345).

The mask used to be a persistent buffer pinned to ``input_resolution``, and the
branch in ``forward`` that noticed a changed resolution was an empty ``pass``.
``WindowAttention`` reads its window count off ``mask.shape[0]``, so the stale
mask did not merely mis-weight the attention -- it reshaped the attention
tensor by the wrong factor. With 16 real windows and a 4-window mask the view
became ``(4 batches, 4 windows)`` and succeeded; at 9 real windows it raised a
``RuntimeError`` from deep inside the attention.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.swin import SwinBlock, WindowAttention
from spectramr.models.blocks.swin_windows import build_shifted_window_mask

WINDOW = 8
DIM = 32


def _block(resolution: tuple[int, int] | None = (16, 16)) -> SwinBlock:
    torch.manual_seed(0)
    return SwinBlock(
        dim=DIM,
        num_heads=4,
        window_size=WINDOW,
        shift_size=WINDOW // 2,
        input_resolution=resolution,
    ).eval()


@pytest.mark.parametrize(
    ("height", "width", "windows"),
    [(16, 16, 4), (32, 32, 16), (24, 24, 9), (40, 16, 10)],
)
def test_the_mask_follows_the_resolution_in_hand(height, width, windows):
    """Declared (16, 16); the mask must still be right at every other size."""
    block = _block((16, 16))
    mask = block._resolve_attn_mask(height, width, device=torch.device("cpu"), dtype=torch.float32)
    assert mask.shape[0] == windows


@pytest.mark.parametrize(("height", "width"), [(16, 16), (32, 32), (24, 24), (40, 16)])
def test_forward_runs_at_a_resolution_other_than_the_declared_one(height, width):
    """(24, 24) used to raise: 9 real windows do not divide by a 4-window mask."""
    block = _block((16, 16))
    out = block(torch.randn(1, height * width, DIM), input_resolution=(height, width))
    assert out.shape == (1, height * width, DIM)
    assert torch.isfinite(out).all()


def test_the_stale_mask_was_not_inert():
    """Plant the old behaviour: applying the (16, 16) mask at (32, 32) changes
    the output. Without this, the tests above would pass just as well if the
    mask made no difference at all, and they would be proving nothing.
    """
    attention = WindowAttention(dim=DIM, window_size=WINDOW, num_heads=4)
    tokens = torch.randn(16, WINDOW * WINDOW, DIM)
    # Applied at WindowAttention rather than through SwinBlock: the block now
    # states its window count, so the guard rejects the stale mask before it
    # can be measured. The point here is what the mask *did*, not whether the
    # guard fires -- that is pinned separately.
    with torch.no_grad():
        correct = attention(tokens, mask=build_shifted_window_mask(32, 32, WINDOW, WINDOW // 2))
        degraded = attention(tokens, mask=build_shifted_window_mask(16, 16, WINDOW, WINDOW // 2))

    assert not torch.allclose(correct, degraded), (
        "the stale mask changes nothing, so these tests cannot see the defect "
        "they were written for -- re-derive them"
    )


def test_a_block_with_no_declared_resolution_is_still_masked():
    """``input_resolution=None`` used to mean no mask at all for a *shifted*
    block, which is not Swin -- the shifted windows attended across the cyclic
    wrap. Three construction sites in the repo are built that way.
    """
    block = _block(None)
    x = torch.randn(1, 32 * 32, DIM)
    mask = block._resolve_attn_mask(32, 32, device=torch.device("cpu"), dtype=torch.float32)
    assert mask.shape[0] == 16
    assert (mask == -100.0).any()
    assert torch.isfinite(block(x, input_resolution=(32, 32))).all()


def test_an_unshifted_block_passes_no_mask():
    """W-MSA needs no mask; only SW-MSA does."""
    block = SwinBlock(
        dim=DIM, num_heads=4, window_size=WINDOW, shift_size=0, input_resolution=(16, 16)
    ).eval()
    assert block.attn_mask is None


def test_the_declared_resolution_mask_is_unchanged():
    """Behaviour preservation: the mask that *was* correct still is, exactly."""
    block = _block((16, 16))
    reference = build_shifted_window_mask(16, 16, WINDOW, WINDOW // 2)
    assert torch.equal(block.attn_mask, reference)


def test_the_mask_is_not_in_the_state_dict():
    """It is derived state. As a persistent buffer it locked a checkpoint to
    the resolution its run was built at.
    """
    assert "attn_mask" not in _block((16, 16)).state_dict()


def test_a_pre_fix_checkpoint_loads_at_any_resolution():
    """Every checkpoint written before this change carries an ``attn_mask``
    sized for that run's resolution. Loading one into a block built at another
    resolution used to fail with a size mismatch.
    """
    legacy = dict(_block((16, 16)).state_dict())
    legacy["attn_mask"] = torch.zeros(4, WINDOW * WINDOW, WINDOW * WINDOW)
    target = _block((32, 32))
    target.load_state_dict(legacy, strict=True)  # raises if the key is not dropped


def test_the_mask_is_built_once_per_resolution_not_once_per_forward():
    """Non-negotiable 9: no rebuilt tensors on the warm path."""
    block = _block((16, 16))
    x = torch.randn(1, 32 * 32, DIM)
    for _ in range(5):
        block(x, input_resolution=(32, 32))
    assert len(block._mask_cache) == 2, "one for the declared size, one for (32, 32)"


def test_window_attention_raises_on_a_mask_from_another_resolution():
    """The guard on the contract itself. ``nW`` is read off the mask, so the
    reshape below it succeeds whenever the counts happen to divide -- 16 rows
    against a 4-window mask became ``(4, 4)`` with no complaint.
    """
    attention = WindowAttention(dim=DIM, window_size=WINDOW, num_heads=4)
    tokens = torch.randn(16, WINDOW * WINDOW, DIM)
    stale = torch.zeros(4, WINDOW * WINDOW, WINDOW * WINDOW)

    # The shape checks alone cannot see this one: 16 % 4 == 0, so the reshape
    # succeeds. This is the exact shape #1345 is about, and it is caught only
    # because SwinBlock now states how many windows it partitioned into.
    with pytest.raises(ValueError, match="describes 4 windows but 16 were"):
        attention(tokens, mask=stale, num_windows=16)

    # Without that, the same call is silently accepted -- stated as a test so
    # the guard is not mistaken for complete coverage of an unco-operative
    # caller (the three other WindowAttention users do not pass the count).
    with torch.no_grad():
        assert attention(tokens, mask=stale).shape == tokens.shape

    # A count that does not divide is caught either way.
    with pytest.raises(ValueError, match="describes 5 windows but 16"):
        attention(tokens, mask=torch.zeros(5, WINDOW * WINDOW, WINDOW * WINDOW))

    with pytest.raises(ValueError, match="different window_size"):
        attention(tokens, mask=torch.zeros(16, 4, 4))


def test_window_attention_still_applies_a_correct_mask():
    """Anti-vacuity for the guard: the valid case is not now rejected, and the
    mask still reaches the softmax.
    """
    attention = WindowAttention(dim=DIM, window_size=WINDOW, num_heads=4)
    tokens = torch.randn(16, WINDOW * WINDOW, DIM)
    mask = build_shifted_window_mask(32, 32, WINDOW, WINDOW // 2)

    with torch.no_grad():
        masked = attention(tokens, mask=mask)
        unmasked = attention(tokens, mask=None)
    assert masked.shape == tokens.shape
    assert not torch.allclose(masked, unmasked)


def _explicit_window_attention(m, x, mask):
    """The pre-SDPA reference: scores + relative bias (+ shift mask), softmax, weighted sum."""
    import torch

    b_, N, C = x.shape
    qkv = m.qkv(x).reshape(b_, N, 3, m.num_heads, C // m.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0] * m.scale, qkv[1], qkv[2]
    attn = q @ k.transpose(-2, -1)
    bias = m.relative_position_bias_table[m.relative_position_index.view(-1)].view(N, N, -1)
    attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
    if mask is not None:
        n_win = mask.shape[0]
        attn = attn.view(b_ // n_win, n_win, m.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, m.num_heads, N, N)
    out = (torch.softmax(attn, dim=-1) @ v).transpose(1, 2).reshape(b_, N, C)
    return m.proj(out)


def test_window_attention_sdpa_matches_explicit_softmax() -> None:
    import torch

    from spectramr.models.blocks.swin import WindowAttention

    torch.manual_seed(0)
    m = WindowAttention(dim=16, window_size=4, num_heads=2).eval()
    torch.nn.init.normal_(m.relative_position_bias_table, std=0.5)
    x = torch.randn(6, 16, 16)  # 3 images x n_win=2 windows, N = 4*4
    shift_mask = torch.where(torch.rand(2, 16, 16) > 0.7, -100.0, 0.0)
    for mask in (None, shift_mask):
        torch.testing.assert_close(
            m(x, mask), _explicit_window_attention(m, x, mask), rtol=1e-4, atol=1e-5
        )
