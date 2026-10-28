"""Line tokens for the LLM-style k-space backbones (``kspace_ar``, ``kspace_masked``).

One token is one k-space line along ``line_axis``, because the cohort masks
remove whole lines: a square patch is often half-sampled, which would leave
"masked token" and "centre-out order" undefined. The trunk is DiT's adaLN-Zero
block stack, so conditioning on the timestep and contrast is the DiT path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from spectramr.models.generators.diffusion_transformer_stem import (
    TimeConditioning,
    apply_contrast_conditioning,
    build_contrast_projection,
    modulate,
)
from spectramr.models.generators.dit_backbone import DiTBlock

LINE_AXES = (-2, -1)
LINE_BACKBONES = frozenset({"kspace_ar", "kspace_masked"})


def validate_line_axis(line_axis: int) -> int:
    if line_axis not in LINE_AXES:
        raise ValueError(f"line_axis must be one of {LINE_AXES}, got {line_axis!r}.")
    return int(line_axis)


def validate_feature_domain_kspace(feature_domain: str, owner: str) -> None:
    if feature_domain != "kspace":
        raise ValueError(
            f"{owner} tokenises k-space lines and has no image-domain view; "
            f"feature_domain={feature_domain!r} is not supported (use force_pure_kspace: true)."
        )


def centre_out_order(n: int) -> torch.Tensor:
    """Line order ``n//2, +1, -1, +2, -2, ...`` as a permutation of ``range(n)``."""
    c = n // 2
    order = [c]
    for step in range(1, n):
        for line in (c + step, c - step):
            if 0 <= line < n:
                order.append(line)
    return torch.tensor(order, dtype=torch.long)


def line_support(
    mask: torch.Tensor | None, line_axis: int, owner: str, check: bool = True
) -> torch.Tensor:
    """``[B, N]`` bool, True where the line is acquired.

    The mask must be all-or-nothing along each line. ``check`` raises a readable
    ``ValueError`` (one host sync, so the backbones pass it on the first forward in
    each mode); otherwise the same condition is asserted asynchronously.
    """
    if mask is None:
        raise ValueError(
            f"{owner} needs mask= (the support known at this step) to know which lines are missing."
        )
    if mask.dim() != 4:
        raise ValueError(f"{owner} expects a 4D mask [B, C, H, W], got {tuple(mask.shape)}.")
    support = mask.bool().any(dim=1)  # [B, H, W]
    within = -1 if line_axis == -2 else -2  # axis running along one line
    aligned = (support.all(dim=within) == support.any(dim=within)).all()
    if check:
        if not bool(aligned):
            raise ValueError(
                f"{owner}: the mask varies along the line (line_axis={line_axis}); the acquisition "
                f"removes lines along the other axis, set model_kwargs.line_axis={within}."
            )
    else:
        # Later calls assert on-device so the training loop takes no host sync.
        torch._assert_async(
            aligned, f"{owner}: mask varies along the line (line_axis={line_axis})."
        )
    return support.any(dim=within)


def prefix_causal_mask(known: torch.Tensor) -> torch.Tensor:
    """``[B, 1, N, N]`` bool attention mask for tokens laid out in centre-out order.

    Acquired (``known``) tokens attend to the acquired ones only, in both
    directions; a missing token attends to every acquired token and to the missing
    tokens at or before its own position, so missing lines are filled coarse to
    fine. Every token may attend itself, so no softmax row is empty.
    """
    n = known.shape[1]
    idx = torch.arange(n, device=known.device)
    earlier = (idx[None, :] <= idx[:, None])[None]
    key_known, query_known = known[:, None, :], known[:, :, None]
    return (key_known | (~query_known & ~key_known & earlier)).unsqueeze(1)


def expand_batch(support: torch.Tensor, batch: int, owner: str) -> torch.Tensor:
    if support.shape[0] == batch:
        return support
    if support.shape[0] == 1:
        return support.expand(batch, -1)
    if batch % support.shape[0]:
        raise ValueError(f"{owner}: mask batch {support.shape[0]} does not divide batch {batch}.")
    return support.repeat_interleave(batch // support.shape[0], dim=0)


class LineStem(nn.Module):
    """``[B, C, H, W]`` -> ``[B, N, dim]``, one token per line, natural order."""

    def __init__(self, channels: int, line_len: int, dim: int, line_axis: int):
        super().__init__()
        self.line_axis, self.line_len = validate_line_axis(line_axis), int(line_len)
        self.proj = nn.Linear(channels * line_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if (w if self.line_axis == -2 else h) != self.line_len:
            raise ValueError(
                f"line length {self.line_len} (model image_size) does not match the field "
                f"{h}x{w} along line_axis={self.line_axis}."
            )
        lines = (
            x.permute(0, 2, 1, 3).reshape(b, h, c * w)
            if self.line_axis == -2
            else (x.permute(0, 3, 1, 2).reshape(b, w, c * h))
        )
        return self.proj(lines)


class LineHead(nn.Module):
    """adaLN-modulated projection of ``[B, N, dim]`` tokens back to ``[B, C, H, W]``."""

    def __init__(self, dim: int, channels: int, line_len: int, line_axis: int):
        super().__init__()
        self.line_axis, self.channels = validate_line_axis(line_axis), int(channels)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, channels * line_len)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        # Non-zero like DomainHead: a zero head is the measurement-independent
        # output the Tier-2 probe rejects.
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        out = self.proj(modulate(self.norm(tokens), shift, scale))
        b, n, _ = out.shape
        out = out.reshape(b, n, self.channels, -1)  # [B, N, C, L]
        return out.permute(0, 2, 1, 3) if self.line_axis == -2 else out.permute(0, 2, 3, 1)


class LinePositions(nn.Module):
    """Learned per-line position table, resampled if a field has another line count."""

    def __init__(self, dim: int, n: int):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(1, n, dim))
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self, n: int) -> torch.Tensor:
        if n == self.table.shape[1]:
            return self.table
        t = F.interpolate(self.table.transpose(1, 2), size=n, mode="linear", align_corners=False)
        return t.transpose(1, 2)


class LineTrunk(nn.Module):
    """Conditioning plus the adaLN-Zero block stack; ``attn_mask`` restricts attention."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        max_timesteps: float | None,
        contrast_emb_dim: int | None,
        owner: str,
    ):
        super().__init__()
        self.owner = owner
        self.time = TimeConditioning(dim, max_timesteps=max_timesteps)
        self.contrast_proj = build_contrast_projection(contrast_emb_dim, dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])

    def condition(
        self,
        timesteps: torch.Tensor | None,
        batch: int,
        device: torch.device,
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        cond = self.time(
            timesteps,
            batch,
            device,
            max_timesteps=kwargs.get("max_timesteps"),
            scale=kwargs.get("timestep_scale", "unit_interval"),
        )
        return apply_contrast_conditioning(
            cond, kwargs.get("contrast_emb"), self.contrast_proj, owner=self.owner
        )

    def forward(
        self,
        tokens: torch.Tensor,
        cond: torch.Tensor,
        ckpt: bool,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            tokens = (
                checkpoint(block, tokens, cond, attn_mask, use_reentrant=False)
                if ckpt
                else block(tokens, cond, attn_mask)
            )
        return tokens
