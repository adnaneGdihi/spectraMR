"""Neural Cache Generator for retrieval-augmented MRI reconstruction.

Two memories, selected by ``cache_mode``:

* a learned key-value bank (``nn.Parameter``), always on, read by every slice;
* ``cache_mode: sequential`` adds a continuous cache in the sense of Grave et al.: while
  the slices of a ``[B, C, H, W, S]`` sequence are processed in order, slice ``t`` also
  attends over pooled feature tokens of slices ``0..t-1``. The tokens reuse the query
  projection, so ``off`` and ``sequential`` have identical parameters and differ in the
  cross-slice read alone.

The slice axis reaches the model only when the arm declares ``model.spatial_dims: 3``
(the training loop folds depth into the batch for 2-D models), so ``sequential``
requires it at construction and refuses a 4-D input in ``forward``.

Reference:
    Grave et al., "Improving Neural Language Models with a Continuous Cache",
    ICLR 2017.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.generators.option_parsing import parse_enum_option
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

__all__ = ["CacheMode", "NeuralCacheGenerator"]


class CacheMode(StrEnum):
    """Which memories the final cache layer reads."""

    OFF = "off"
    SEQUENTIAL = "sequential"


class _CacheAttention(nn.Module):
    """Cross-attention between spatial features and a key-value memory bank."""

    def __init__(self, ch: int, cache_size: int, kv_dim: int) -> None:
        super().__init__()
        self.keys = nn.Parameter(torch.randn(cache_size, kv_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(cache_size, kv_dim) * 0.02)
        self.q_proj = nn.Conv2d(ch, kv_dim, 1)
        self.out_proj = nn.Conv2d(kv_dim, ch, 1)

    def tokens(self, x: torch.Tensor, pool_size: int) -> torch.Tensor:
        """Pooled cache tokens ``[B, pool_size**2, kv_dim]`` summarising ``x [B, C, H, W]``."""
        pooled = F.adaptive_avg_pool2d(x, pool_size)
        return self.q_proj(pooled).flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        """Residual read of the learned bank, plus ``memory [B, M, kv_dim]`` when given."""
        B, _, H, W = x.shape
        q = self.q_proj(x).flatten(2).transpose(1, 2)  # [B, HW, D]
        k = self.keys.unsqueeze(0).expand(B, -1, -1)
        v = self.values.unsqueeze(0).expand(B, -1, -1)
        if memory is not None:
            k = torch.cat([k, memory], dim=1)
            v = torch.cat([v, memory], dim=1)
        out = F.scaled_dot_product_attention(q, k, v)  # fused kernel, scale = D**-0.5
        out = out.transpose(1, 2).reshape(B, -1, H, W)
        return x + self.out_proj(out)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"neural_cache {name} must be a positive int, got {value!r}.")
    return value


@register_model(
    name="neural_cache",
    training_mode="reconstruction",
    spatial_dims=(2, 3),
    input_domain="image",
    output_domain="image",
)
class NeuralCacheGenerator(IGenerator, nn.Module):
    """Retrieval-augmented reconstruction network with a learned bank and a slice cache."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        feature_dim = _positive_int("feature_dim", kwargs.get("feature_dim", 64))
        num_layers = _positive_int("num_layers", kwargs.get("num_layers", 8))
        cache_size = _positive_int("cache_size", kwargs.get("cache_size", 1024))
        kv_dim = _positive_int("key_value_dim", kwargs.get("key_value_dim", 64))
        self.cache_mode = parse_enum_option(
            CacheMode, kwargs.get("cache_mode", CacheMode.OFF.value), "cache_mode"
        )
        self.cache_pool_size = _positive_int("cache_pool_size", kwargs.get("cache_pool_size", 16))
        # How many preceding slices a slice reads; None reads all of them. Capping it at the
        # training window keeps whole-volume validation inside the trained regime.
        window = kwargs.get("cache_window")
        self.cache_window = None if window is None else _positive_int("cache_window", window)
        spatial_dims = kwargs.get("spatial_dims")
        if self.cache_mode is CacheMode.SEQUENTIAL and spatial_dims != 3:
            raise ValueError(
                "neural_cache cache_mode='sequential' reads the slice axis, which reaches the "
                "model only when the arm declares model.spatial_dims: 3 (2-D arms have depth "
                f"folded into the batch); got spatial_dims={spatial_dims!r}."
            )

        self.intro = nn.Conv2d(in_channels, feature_dim, 3, padding=1)
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(feature_dim, feature_dim, 3, padding=1))
            layers.append(nn.GroupNorm(min(32, feature_dim), feature_dim))
            layers.append(nn.SiLU())
            if i == num_layers // 2:
                layers.append(_CacheAttention(feature_dim, cache_size, kv_dim))
        self.backbone = nn.Sequential(*layers)
        self.cache_attn = _CacheAttention(feature_dim, cache_size, kv_dim)
        self.out_conv = nn.Conv2d(feature_dim, out_channels, 3, padding=1)

    @property
    def name(self) -> str:
        return "neural_cache"

    @property
    def mechanism_record(self) -> dict[str, Any]:
        """The cache configuration this run trained, stamped into provenance."""
        return {
            "cache_mode": self.cache_mode.value,
            "cache_pool_size": self.cache_pool_size,
            "cache_tokens_per_slice": self.cache_pool_size**2,
            "cache_window": self.cache_window,
            "learned_bank_size": int(self.cache_attn.keys.shape[0]),
        }

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """``[B, C, H, W]`` or a slice sequence ``[B, C, H, W, S]`` to the same layout."""
        if x.ndim == 4:
            if self.cache_mode is CacheMode.SEQUENTIAL:
                raise ValueError(
                    "neural_cache cache_mode='sequential' needs a [B, C, H, W, S] slice "
                    f"sequence; got a 4-D input {tuple(x.shape)}, which has no cache to read."
                )
            return self.out_conv(self.cache_attn(self.backbone(self.intro(x))))
        if x.ndim != 5:
            raise ValueError(f"neural_cache expects a 4-D or 5-D input, got {tuple(x.shape)}.")

        B, C, H, W, S = x.shape
        folded = x.permute(0, 4, 1, 2, 3).reshape(B * S, C, H, W)
        h = self.backbone(self.intro(folded))
        if self.cache_mode is CacheMode.OFF:
            h = self.cache_attn(h)
        else:
            h = self._sequential_read(h.reshape(B, S, *h.shape[1:]))
        out = self.out_conv(h)
        return out.reshape(B, S, *out.shape[1:]).permute(0, 2, 3, 4, 1)

    def _sequential_read(self, h: torch.Tensor) -> torch.Tensor:
        """Slice ``t`` of ``h [B, S, F, H, W]`` reads the bank plus tokens of the slices before it."""
        S = h.shape[1]
        # Tokens of every slice in one batched pass; the loop below is the causal order.
        tokens = self.cache_attn.tokens(h.flatten(0, 1), self.cache_pool_size)
        tokens = tokens.reshape(h.shape[0], S * tokens.shape[1], tokens.shape[2])
        per_slice = self.cache_pool_size**2
        outs = [self.cache_attn(h[:, 0])]
        for t in range(1, S):
            first = 0 if self.cache_window is None else max(0, t - self.cache_window)
            memory = tokens[:, first * per_slice : t * per_slice]
            outs.append(self.cache_attn(h[:, t], memory=memory))
        return torch.stack(outs, dim=1).flatten(0, 1)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
