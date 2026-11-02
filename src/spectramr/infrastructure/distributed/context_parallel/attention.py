"""``TokenAttention`` over a sequence split across the context-parallel group.

Each rank holds a contiguous chunk of the tokens. Queries stay local; keys and values
are all-gathered so every local query attends to the whole sequence, then the local
SDPA runs on a ``(n / cp) x n`` score matrix. Parameters are untouched, so state-dict
keys, shapes and optimizer state equal the unsplit model's.

All-gather rather than ring attention: the clusters run V100 and P100, which have no
flash kernel, and torch's ring attention needs those SDPA backends and cannot run on
CPU, so it could not be tested here.

Under tensor x context parallelism a ``TensorParallelAttention`` (its own heads' q, k, v
rows and ``proj`` columns) is adopted again as :class:`TensorContextParallelAttention`:
the two splits are orthogonal, so it gathers the keys and values of its own heads over
the context group and sums its partial output over the tensor group.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from spectramr.infrastructure.distributed.context_parallel.comms import gather_keys_values
from spectramr.infrastructure.distributed.tensor_parallel.comms import (
    copy_to_tensor_parallel,
    reduce_from_tensor_parallel,
)
from spectramr.infrastructure.distributed.tensor_parallel.layers import TensorParallelAttention
from spectramr.models.generators.diffusion_transformer_stem import TokenAttention


def local_mask_rows(mask: torch.Tensor | None, local: int, group: Any) -> torch.Tensor | None:
    """The rows of a full ``[..., N, N]`` attention mask that this rank's queries use.

    A causal or any other explicit mask is indexed by global token position, so the
    rows are offset by ``rank * local``. A mask broadcast over queries passes unchanged.
    """
    if mask is None or mask.dim() < 2 or mask.shape[-2] == 1:
        return mask
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if mask.shape[-2] != local * world:
        raise ValueError(
            f"attention mask has {mask.shape[-2]} query rows; context parallelism expects "
            f"the full sequence ({local * world}) and slices this rank's {local}"
        )
    return mask.narrow(-2, rank * local, local)


class ContextParallelAttention(TokenAttention):
    """``TokenAttention`` whose input is this rank's chunk of the sequence."""

    cp_group: Any

    @classmethod
    def adopt(cls, attn: TokenAttention, group: Any) -> ContextParallelAttention:
        if type(attn) is not TokenAttention:
            raise TypeError(
                f"context parallelism adopts plain TokenAttention, got {type(attn).__name__}"
            )
        attn.__class__ = cls
        attn.cp_group = group
        return attn

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = x.shape
        heads = self.heads
        qkv = self.qkv(x)
        q = qkv[..., :d].reshape(b, n, heads, d // heads).transpose(1, 2)
        kv = gather_keys_values(qkv[..., d:], self.cp_group)  # one collective for k and v
        k, v = kv.reshape(b, kv.shape[1], 2, heads, d // heads).permute(2, 0, 3, 1, 4)
        mask = local_mask_rows(attn_mask, n, self.cp_group)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class TensorContextParallelAttention(TokenAttention):
    """A tensor-parallel ``TokenAttention`` whose input is this rank's sequence chunk.

    Adopted from a :class:`TensorParallelAttention`, whose split weights, state-dict
    hooks and ``tp_sharding`` it keeps, so checkpoints and optimizer state stay unsplit.
    """

    tp_group: Any
    cp_group: Any
    local_heads: int

    @classmethod
    def adopt(cls, attn: TokenAttention, group: Any) -> TensorContextParallelAttention:
        if type(attn) is not TensorParallelAttention:
            raise TypeError(
                "tensor x context parallelism adopts TensorParallelAttention, got "
                f"{type(attn).__name__}"
            )
        attn.__class__ = cls
        attn.cp_group = group
        return attn

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        heads = self.local_heads
        width = self.qkv.out_features // 3  # this rank's heads x head dim
        qkv = self.qkv(copy_to_tensor_parallel(x, self.tp_group))
        q = qkv[..., :width].reshape(b, n, heads, width // heads).transpose(1, 2)
        kv = gather_keys_values(qkv[..., width:], self.cp_group)  # 1/tp of the heads' k, v
        k, v = kv.reshape(b, kv.shape[1], 2, heads, width // heads).permute(2, 0, 3, 1, 4)
        mask = local_mask_rows(attn_mask, n, self.cp_group)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        partial = F.linear(out.transpose(1, 2).reshape(b, n, width), self.proj.weight)
        return reduce_from_tensor_parallel(partial, self.tp_group) + self.proj.bias


#: The attention class each split form is adopted from, and its context-parallel form.
CP_ADOPTERS = {
    TokenAttention: ContextParallelAttention.adopt,
    TensorParallelAttention: TensorContextParallelAttention.adopt,
}

#: Every attention whose input is a sequence chunk (what the gate checks).
CP_ATTENTIONS = (ContextParallelAttention, TensorContextParallelAttention)


__all__ = [
    "CP_ADOPTERS",
    "CP_ATTENTIONS",
    "ContextParallelAttention",
    "TensorContextParallelAttention",
    "local_mask_rows",
]
