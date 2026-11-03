"""Tensor-parallel attention and MLP, adopted in place from the unsplit modules.

``qkv`` is laid out q-major (all q heads, then k, then v), so a contiguous split of
its output rows would hand one rank all of q and half of k. Each rank instead keeps
the q, k and v rows of its own heads and the matching input columns of ``proj``; the
MLP keeps a slice of the hidden units. Parameters keep their names; a state-dict hook
gathers full-shape tensors on save and a load pre-hook slices them on load, so a
checkpoint is interchangeable with the unsplit model's.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

from spectramr.infrastructure.distributed.tensor_parallel.comms import (
    copy_to_tensor_parallel,
    gather_along,
    reduce_from_tensor_parallel,
)
from spectramr.models.generators.diffusion_transformer_stem import TokenAttention


def _replace(linear: nn.Linear, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    linear.weight = nn.Parameter(weight.detach().clone())
    if bias is not None:
        linear.bias = nn.Parameter(bias.detach().clone())
    linear.out_features, linear.in_features = weight.shape


class _Sharding:
    """How each sharded tensor maps between its full and local shapes.

    Under FSDP2 a parameter is a DTensor sharded over the data-parallel mesh. The save
    hook then gathers it over dp before the tp gather and re-shards the full tensor over
    the same dp placements, so the state dict looks like an FSDP-only unsplit model.
    """

    def __init__(self, group: Any, slices: dict[str, tuple]) -> None:
        self.group, self.slices = group, slices  # key -> (to_local, view_local, cat_dim, to_full)

    def full(self, key: str, local: torch.Tensor) -> torch.Tensor:
        """The unsplit tensor from every tensor rank's slice (collective over the group)."""
        _, view_local, dim, to_full = self.slices[key]
        return to_full(gather_along(view_local(local), dim, self.group))

    def local(self, key: str, full: torch.Tensor) -> torch.Tensor:
        """This tensor rank's slice of the unsplit tensor."""
        return self.slices[key][0](full)

    def save(self, module: nn.Module, state: dict, prefix: str, _meta: Any) -> dict:
        if _LOCAL_STATE.depth:
            return state
        for key in self.slices:
            value = state[prefix + key]
            local = value.full_tensor() if isinstance(value, DTensor) else value
            state[prefix + key] = _like(self.full(key, local), value)
        return state

    def load(self, module: nn.Module, state: dict, prefix: str, *_args: Any) -> None:
        for key in self.slices:
            tensor = state.get(prefix + key)
            param = module.get_parameter(key)
            # Only a full-shape tensor is sliced; an already-local one is this rank's own.
            if tensor is not None and tensor.shape != param.shape:
                full = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
                state[prefix + key] = _like(self.local(key, full), param)


def _like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """``tensor`` with the dp placements of ``reference`` when that is a DTensor."""
    if not isinstance(reference, DTensor):
        return tensor
    # Every dp rank holds the same full tensor, so each slices its own shard locally.
    return distribute_tensor(
        tensor, reference.device_mesh, reference.placements, src_data_rank=None
    )


class _LocalStateDepth:
    depth = 0


_LOCAL_STATE = _LocalStateDepth()


@contextlib.contextmanager
def tensor_parallel_local_state() -> Iterator[None]:
    """``state_dict()`` returns each rank's own split tensors, with no collective.

    For a blend between two modules split alike (the EMA shadow and the live model).
    Never wrap a checkpoint write in it: the result is one tensor rank's slice.
    """
    _LOCAL_STATE.depth += 1
    try:
        yield
    finally:
        _LOCAL_STATE.depth -= 1


class TensorParallelAttention(TokenAttention):
    """``TokenAttention`` with heads split across the tensor group."""

    tp_group: Any

    @classmethod
    def adopt(cls, attn: TokenAttention, group: Any) -> TensorParallelAttention:
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        heads, dim = attn.heads, attn.proj.in_features
        if heads % size:
            raise ValueError(f"{heads} attention heads do not split across {size} tensor ranks")
        hl, hd = heads // size, dim // heads
        sl = slice(rank * hl, (rank + 1) * hl)

        def qkv_w(w):  # [3*dim, dim] -> [3*hl*hd, dim]
            return w.reshape(3, heads, hd, dim)[:, sl].reshape(3 * hl * hd, dim)

        def qkv_b(b):
            return b.reshape(3, heads, hd)[:, sl].reshape(-1)

        def proj_w(w):  # [dim, dim] -> [dim, hl*hd]
            return w[:, rank * hl * hd : (rank + 1) * hl * hd]

        _replace(attn.qkv, qkv_w(attn.qkv.weight), qkv_b(attn.qkv.bias))
        _replace(attn.proj, proj_w(attn.proj.weight), None)
        sharding = _Sharding(
            group,
            {
                "qkv.weight": (
                    qkv_w,
                    lambda t: t.reshape(3, hl, hd, dim),
                    1,
                    lambda t: t.reshape(3 * dim, dim),
                ),
                "qkv.bias": (qkv_b, lambda t: t.reshape(3, hl, hd), 1, lambda t: t.reshape(-1)),
                "proj.weight": (proj_w, lambda t: t, 1, lambda t: t),
            },
        )
        attn._register_state_dict_hook(sharding.save)
        attn._register_load_state_dict_pre_hook(sharding.load, with_module=True)
        attn.__class__ = cls
        attn.tp_group, attn.local_heads = group, hl
        attn.tp_sharding, attn.tp_sharded_keys = sharding, tuple(sharding.slices)
        return attn

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = x.shape
        x = copy_to_tensor_parallel(x, self.tp_group)
        hd = self.qkv.out_features // (3 * self.local_heads)
        qkv = self.qkv(x).reshape(b, n, 3, self.local_heads, hd)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        partial = F.linear(
            out.transpose(1, 2).reshape(b, n, self.local_heads * hd), self.proj.weight
        )
        return reduce_from_tensor_parallel(partial, self.tp_group) + self.proj.bias


class TensorParallelMLP(nn.Sequential):
    """``Sequential(Linear, activation, Linear)`` with the hidden units split across ranks."""

    tp_group: Any

    @classmethod
    def adopt(cls, mlp: nn.Sequential, group: Any) -> TensorParallelMLP:
        if len(mlp) != 3 or not (isinstance(mlp[0], nn.Linear) and isinstance(mlp[2], nn.Linear)):
            raise ValueError("tensor-parallel MLP expects Sequential(Linear, activation, Linear)")
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        hidden = mlp[0].out_features
        if hidden % size:
            raise ValueError(f"{hidden} hidden units do not split across {size} tensor ranks")
        hs = slice(rank * hidden // size, (rank + 1) * hidden // size)
        _replace(mlp[0], mlp[0].weight[hs], mlp[0].bias[hs] if mlp[0].bias is not None else None)
        _replace(mlp[2], mlp[2].weight[:, hs], None)
        sharding = _Sharding(
            group,
            {
                "0.weight": (lambda w: w[hs], lambda t: t, 0, lambda t: t),
                "0.bias": (lambda b: b[hs], lambda t: t, 0, lambda t: t),
                "2.weight": (lambda w: w[:, hs], lambda t: t, 1, lambda t: t),
            },
        )
        if mlp[0].bias is None:
            del sharding.slices["0.bias"]
        mlp._register_state_dict_hook(sharding.save)
        mlp._register_load_state_dict_pre_hook(sharding.load, with_module=True)
        mlp.__class__ = cls
        mlp.tp_group, mlp.tp_sharding = group, sharding
        mlp.tp_sharded_keys = tuple(sharding.slices)
        return mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = copy_to_tensor_parallel(x, self.tp_group)
        partial = F.linear(self[1](self[0](x)), self[2].weight)
        out = reduce_from_tensor_parallel(partial, self.tp_group)
        return out if self[2].bias is None else out + self[2].bias


__all__ = ["TensorParallelAttention", "TensorParallelMLP", "tensor_parallel_local_state"]
