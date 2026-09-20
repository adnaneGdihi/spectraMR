"""Domain-aware stem, head and time conditioning for the diffusion transformers.

DiT, U-ViT, HAT and DiffiT were published on natural images: one tensor, one
domain, a patch is a picture. MRI arrives k-space-native, and a patch of k-space
is a frequency band whose neighbours share no spatial locality at all -- so a
stem that patch-embeds whichever tensor it was handed silently means two
different things depending on ``force_pure_kspace``.

This module is the one owner of that difference for all four backbones. The stem
embeds **both** views of the field and sums them, deriving the second through
``ifft2c``/``fft2c`` from the declared domain, exactly as ``DualDomainAttention``
already conjugates its two branches. ``feature_domain`` therefore selects which
view needs the transform rather than being recorded and ignored (pitfall 15), and
the head returns the prediction in the domain the caller passed in.

The transformer trunks that consume these tokens are the published architectures;
only the stem and head are the k-space adaptation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.blocks.attention_domains import (
    complex_to_interleaved,
    interleaved_to_complex,
    validate_feature_domain,
)
from spectramr.models.blocks.timestep_embedding import sinusoidal_timestep_embedding


class TokenAttention(nn.Module):
    """Multi-head self-attention over ``[B, N, D]`` tokens via SDPA."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must divide evenly into heads={heads}.")
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation ``x * (1 + scale) + shift`` over ``[B, N, D]`` tokens.

    ``1 + scale`` rather than ``scale`` so a zero-initialised projection is the
    identity, which is what makes adaLN-Zero start as a no-op.
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def both_domain_views(x: torch.Tensor, feature_domain: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(kspace_view, image_view)`` of an interleaved field.

    Args:
        x: ``[B, 2C, H, W]`` interleaved real/imag, living in ``feature_domain``.
        feature_domain: ``"kspace"`` or ``"image"`` -- which one ``x`` already is.

    Returns:
        Both views, each ``[B, 2C, H, W]`` interleaved. The one matching
        ``feature_domain`` is ``x`` itself, untransformed.
    """
    h = interleaved_to_complex(x)
    if feature_domain == "kspace":
        return x, complex_to_interleaved(ifft2c(h))
    return complex_to_interleaved(fft2c(h)), x


class TimeConditioning(nn.Module):
    """Scalar timestep -> ``[B, D]`` conditioning vector.

    ``max_timesteps`` can arrive twice: once at construction (what
    ``backbone_builders.py`` supplies) and once per forward call (what
    ``kspace_cold_diffusion_generator.py`` knows once it has computed its own
    diffusion horizon, which is after every backbone here is already built --
    so the constructor argument is structurally ``None`` on the production
    path). ``forward`` reconciles the two per call rather than mutating
    ``self.max_timesteps``, so the embedding never depends on a prior call's
    kwargs the way a stored override would.
    """

    def __init__(self, dim: int, max_timesteps: float | None = None):
        super().__init__()
        self.dim = dim
        self.max_timesteps = max_timesteps
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(
        self,
        timesteps: torch.Tensor | None,
        batch: int,
        device: torch.device,
        max_timesteps: float | None = None,
    ) -> torch.Tensor:
        """Embed ``timesteps``; a caller that passes none conditions on step 0."""
        if timesteps is None:
            timesteps = torch.zeros(batch, device=device)
        t = timesteps.float().flatten()
        if t.numel() == 1 and batch > 1:
            t = t.expand(batch)
        if t.shape[0] != batch:
            t = t[:batch]
        effective_max = self.max_timesteps
        if max_timesteps is not None:
            if effective_max is not None and float(effective_max) != float(max_timesteps):
                raise ValueError(
                    f"TimeConditioning received max_timesteps={max_timesteps!r} at "
                    f"forward time but was constructed with {effective_max!r}; the "
                    "two owners of the diffusion horizon disagree."
                )
            if effective_max is None:
                t = t / float(max_timesteps)
        basis = sinusoidal_timestep_embedding(t, self.dim, max_timesteps=effective_max)
        return self.mlp(basis)


def build_contrast_projection(contrast_emb_dim: int | None, dim: int) -> nn.Linear | None:
    """Size a projection from the generator's contrast width onto this backbone's ``dim``.

    ``kspace_cold_diffusion_generator.py`` builds ``contrast_emb`` at
    ``time_embedding_dim`` (256 by default), never at a backbone's own token
    width, so an exact-width check silently drops the declared FiLM
    conditioning on every arm that widens the trunk. Returns ``None`` when the
    two already agree, so :func:`apply_contrast_conditioning` adds directly
    instead of routing through an identity-shaped ``Linear``. Left at its
    default (non-zero) init deliberately: every caller already gates its own
    conditioning pathway at init (adaLN-Zero, or an explicitly zero-initialised
    FiLM), so a zero-init projection on top would compose two zeros into the
    same saddle issue #471 found in the attention blocks.
    """
    return (
        nn.Linear(int(contrast_emb_dim), dim)
        if contrast_emb_dim is not None and int(contrast_emb_dim) != dim
        else None
    )


def apply_contrast_conditioning(
    cond: torch.Tensor,
    contrast: torch.Tensor | None,
    contrast_proj: nn.Linear | None,
    *,
    owner: str,
) -> torch.Tensor:
    """Reconcile a contrast embedding onto ``cond``, or raise naming both widths.

    An exact width adds directly; a width ``contrast_proj`` was built for gets
    projected first; anything else is unreconcilable and raises rather than
    silently dropping the arm's declared FiLM conditioning (pitfall 9).
    """
    if contrast is None:
        return cond
    if contrast.shape[-1] == cond.shape[-1]:
        return cond + contrast[: cond.shape[0]]
    if contrast_proj is not None and contrast_proj.in_features == contrast.shape[-1]:
        return cond + contrast_proj(contrast[: cond.shape[0]])
    raise ValueError(
        f"{owner} received contrast_emb width {contrast.shape[-1]} that matches "
        f"neither cond width {cond.shape[-1]} nor the built contrast_proj "
        f"({contrast_proj.in_features if contrast_proj is not None else 'absent'}). "
        "Set model_kwargs.contrast_emb_dim (or time_embedding_dim) to match the "
        "arm's contrast embedding so the declared FiLM conditioning is not "
        "silently dropped."
    )


class DomainStem(nn.Module):
    """Patch-embed both domain views of an interleaved field into tokens.

    Each view gets its OWN projection: the two carry different statistics (a
    k-space patch is dominated by its distance from the centre, an image patch by
    local structure), so sharing weights would force one scale on both.
    """

    def __init__(self, in_channels: int, dim: int, patch_size: int, feature_domain: str):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(
                f"DomainStem expects an interleaved real/imag field with an even "
                f"channel count, got in_channels={in_channels}."
            )
        self.feature_domain = validate_feature_domain(feature_domain)
        self.patch_size = int(patch_size)
        self.dim = int(dim)
        self.proj_kspace = nn.Conv2d(in_channels, dim, self.patch_size, stride=self.patch_size)
        self.proj_image = nn.Conv2d(in_channels, dim, self.patch_size, stride=self.patch_size)

    def grid(self, height: int, width: int) -> tuple[int, int]:
        """Token grid for a field of this size, or raise if it does not tile.

        Cropping or padding to fit would change the field of view of every arm
        that happens to pick a non-dividing size, and in k-space a crop discards
        the highest frequencies outright (pitfall 9).
        """
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"patch_size={self.patch_size} does not tile a {height}x{width} field. "
                f"Set model_kwargs.patch_size to a divisor of both."
            )
        return height // self.patch_size, width // self.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 2C, H, W]`` -> ``[B, N, D]`` tokens."""
        self.grid(x.shape[-2], x.shape[-1])
        k_view, i_view = both_domain_views(x, self.feature_domain)
        tokens = self.proj_kspace(k_view) + self.proj_image(i_view)
        return tokens.flatten(2).transpose(1, 2)


class DomainHead(nn.Module):
    """Project tokens back to an interleaved field in the stem's input domain.

    The adaLN modulation is zero-initialised, so the head starts as a plain
    ``LayerNorm -> Linear``; the projection deliberately is NOT. DiT zero-inits
    its final layer so the whole model emits zero at step 0, and that is exactly
    the measurement-independent output this framework's Tier-2 probe rejects --
    a forward whose result does not move when its input does is the DC-blob
    facade class, and the probe cannot tell "untrained" from "ignores its
    measurement". The property the paper credits for stability lives in the
    blocks' adaLN-Zero gates, which are kept.
    """

    def __init__(self, dim: int, out_channels: int, patch_size: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, tokens: torch.Tensor, cond: torch.Tensor, grid: tuple[int, int]
    ) -> torch.Tensor:
        """``[B, N, D]`` -> ``[B, out_channels, H, W]``."""
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        h = self.proj(modulate(self.norm(tokens), shift, scale))
        gh, gw = grid
        p, c = self.patch_size, self.out_channels
        h = h.reshape(h.shape[0], gh, gw, p, p, c)
        h = h.permute(0, 5, 1, 3, 2, 4)
        return h.reshape(h.shape[0], c, gh * p, gw * p)


class LearnedPositions(nn.Module):
    """Interpolatable learned position table.

    Held at a reference grid and resampled bilinearly, because an arm may
    validate at a different matrix size than it trains on and a fixed table would
    raise on the first such batch.
    """

    def __init__(self, dim: int, reference_grid: int):
        super().__init__()
        self.reference_grid = int(reference_grid)
        self.table = nn.Parameter(torch.zeros(1, dim, reference_grid, reference_grid))
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self, grid: tuple[int, int]) -> torch.Tensor:
        """``[1, N, D]`` positions for a ``grid`` token layout."""
        gh, gw = grid
        table = self.table
        if (gh, gw) != (self.reference_grid, self.reference_grid):
            table = nn.functional.interpolate(
                table, size=(gh, gw), mode="bilinear", align_corners=False
            )
        return table.flatten(2).transpose(1, 2)
