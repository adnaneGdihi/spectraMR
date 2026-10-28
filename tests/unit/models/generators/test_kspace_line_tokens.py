"""Line-token helpers shared by the kspace_ar / kspace_masked backbones."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.diffusion_transformer_stem import TokenAttention
from spectramr.models.generators.kspace_line_tokens import (
    LineHead,
    LineStem,
    centre_out_order,
    line_support,
    prefix_causal_mask,
    validate_line_axis,
)


def test_centre_out_order_n8() -> None:
    assert centre_out_order(8).tolist() == [4, 5, 3, 6, 2, 7, 1, 0]


@pytest.mark.parametrize("n", [7, 8, 32])
def test_centre_out_order_is_permutation(n: int) -> None:
    assert sorted(centre_out_order(n).tolist()) == list(range(n))


@pytest.mark.parametrize("axis", [-2, -1])
def test_stem_head_round_trip_shape(axis: int) -> None:
    x = torch.randn(2, 8, 16, 16)
    tokens = LineStem(8, 16, 32, axis)(x)
    assert tokens.shape == (2, 16, 32)
    assert LineHead(32, 8, 16, axis)(tokens, torch.randn(2, 32)).shape == x.shape


def test_stem_rejects_wrong_line_length() -> None:
    with pytest.raises(ValueError, match="line length"):
        LineStem(8, 16, 32, -2)(torch.randn(1, 8, 16, 8))


def test_line_support_rows_and_columns() -> None:
    mask = torch.zeros(1, 2, 6, 6)
    mask[:, :, [1, 4], :] = 1
    assert line_support(mask, -2, "t")[0].nonzero().flatten().tolist() == [1, 4]
    cols = torch.zeros(1, 2, 6, 6)
    cols[:, :, :, [0, 5]] = 1
    assert line_support(cols, -1, "t")[0].nonzero().flatten().tolist() == [0, 5]


def test_line_support_raises_when_mask_varies_within_line() -> None:
    cols = torch.zeros(1, 1, 6, 6)
    cols[:, :, :, 2] = 1
    with pytest.raises(ValueError, match="line_axis=-2"):
        line_support(cols, -2, "t")


def test_line_support_raises_on_none() -> None:
    with pytest.raises(ValueError, match="needs mask"):
        line_support(None, -2, "t")


def test_validate_line_axis_rejects_0() -> None:
    with pytest.raises(ValueError, match="line_axis"):
        validate_line_axis(0)


def test_token_attention_attn_mask_blocks_masked_keys() -> None:
    torch.manual_seed(0)
    attn = TokenAttention(16, 2)
    x = torch.randn(1, 6, 16)
    y = x.clone()
    y[:, 3] += 1.0
    causal = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    a, b = attn(x, causal), attn(y, causal)
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6)
    assert not torch.allclose(a[:, 3:], b[:, 3:])


def test_token_attention_default_is_bidirectional() -> None:
    torch.manual_seed(0)
    attn = TokenAttention(16, 2)
    x = torch.randn(1, 6, 16)
    y = x.clone()
    y[:, 3] += 1.0
    assert not torch.allclose(attn(x)[:, :3], attn(y)[:, :3])


def test_prefix_causal_mask_semantics() -> None:
    known = torch.tensor([[True, False, True, False, False]])
    m = prefix_causal_mask(known)[0, 0]
    assert m.shape == (5, 5)
    assert m[0].tolist() == [True, False, True, False, False]  # known: acquired keys only
    assert m[1].tolist() == [True, True, True, False, False]  # missing: prefix + itself
    assert m[3].tolist() == [True, True, True, True, False]  # + earlier missing, not later
    assert m[4].tolist() == [True, True, True, True, True]
    assert m.diagonal().all()


def test_later_calls_still_reject_a_misaligned_mask_without_a_host_sync() -> None:
    cols = torch.zeros(1, 1, 6, 6)
    cols[:, :, :, 2] = 1
    with pytest.raises(RuntimeError, match="varies along the line"):
        line_support(cols, -2, "t", check=False)
    assert line_support(cols, -1, "t", check=False).shape == (1, 6)
