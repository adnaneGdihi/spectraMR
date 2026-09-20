r"""Per-annulus latent tokens, indexed by acceleration rung, read at every up-step.

The ``kspace_filling`` cohort is architecturally low-pass before it is trained:
at initialisation every arm in the attention shootout returns
``outer_band_retention`` between **0.198 and 0.239** against a 0.50 threshold, and
no attention type can close it -- ``worst_rho`` is 1.00-1.10 across all ten,
because a softmax redistributes energy and does not create it (#2117).
:class:`~spectramr.models.blocks.radial_band_gain.RadialBandGain` answers that
with one real scalar per annulus on the model's output; this block answers the
other half, since the attenuation is introduced *inside* the decoder, compounds
across up-steps, and asks for a different correction at R=2 than at R=32. A owns
the output level, this owns the shape along the way, and an arm may carry both.

Four constraints are load-bearing; the docs page carries the evidence.

**The partition is not re-derived.** :func:`~spectramr.infrastructure.physics.radial_bands.radial_bins`
is its one owner and is what the spectral-transfer probe measures. Decoder level
*i* runs on a centre crop of full k-space, so binning it with
``n_bands * (H_i / H_full)`` bins makes the local bin index *exactly* the global
annulus index -- annulus 3 is one physical band at every up-step and in the probe.

**Scale only, never a shift**, because a constant added to a k-space annulus is a
Dirac in image space -- the reason every ``ComplexConv2d`` here is ``bias=False``.
The scale is ``exp`` of a clamped logit, so it is strictly positive and leaves
:math:`\arg z` unchanged elementwise. That is a property of **this block**, not of
the network: downstream ``ComplexConv2d`` and ``ModReLU`` still rotate phase. A
can make the end-to-end claim because it sits after ``final_conv``; this cannot,
and reading the two alike is how a ``val_band_*`` drift gets misattributed.

**The bound compounds** -- ``n`` sites reach ``exp(n * max_log_scale)``, so the
default 0.40 puts four up-steps at the ``exp(1.6)`` A argues from the measured
1.71, where A's per-site 1.6 would reach ``exp(6.4)``.

**Identity at initialisation, with exactly one zero.** The write head is zeroed,
so every scale is exactly 1.0 and an arm is bit-identical to its control at step
0, while everything upstream stays random. That is what keeps this off the
issue-#471 saddle, which needs a zero gate *composed with* an identity-at-init
inner block so both gradients vanish together.

**No mask dependence**: the reverse sampler calls ``model(x, t)`` bare, so a
mechanism needing the mask trains and then does not run where the reported
numbers come from. The rung, however, *is* ``t``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.radial_bands import band_counts, radial_bins
from spectramr.models.blocks.attention_domains import (
    complex_to_interleaved,
    interleaved_to_complex,
)

__all__ = ["DEFAULT_MAX_LOG_SCALE", "RadialBandTokens"]

#: Per-site bound on ``|log m|``; it compounds across up-steps, hence not A's 1.6.
DEFAULT_MAX_LOG_SCALE = 0.40

#: ``DEFAULT_MAX_LOG_SCALE`` times the cohort's up-step count -- the network-level
#: ceiling the default is chosen to hit, pinned by a test so it cannot drift.
COMPOUNDED_LOG_CEILING = 1.6


class RadialBandTokens(nn.Module):
    """Learned (annulus, rung) tokens driving a per-annulus, per-channel scale.

    Args:
        level_channels: Complex width of each decoder up-step, in run order; one
            write head is built per entry.
        n_bands: Annuli over the full grid. Must be >= 2 and must leave every
            decoder level an integer, non-degenerate share.
        n_rungs: Size of the rung axis -- the process's timestep count. On a
            one-rung-per-timestep ladder the rung *is* the timestep.
        token_dim: Token width, and the context head's. Deliberately independent
            of the decoder width: tying them would make the bank a property of
            the backbone.
        max_log_scale: Per-site bound on ``|log m|``.
        token_init_std: Bank init scale. Not zero -- a second zero would make the
            context a weighted sum of zeros and strand the write head.

    Shape:
        ``features``: complex ``[B, C, H, W]`` or interleaved ``[B, 2C, H, W]``;
        returns the same layout and dtype.
    """

    def __init__(
        self,
        level_channels: tuple[int, ...],
        n_bands: int = 8,
        n_rungs: int = 29,
        token_dim: int = 64,
        max_log_scale: float = DEFAULT_MAX_LOG_SCALE,
        token_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        level_channels = tuple(int(c) for c in level_channels)
        if not level_channels:
            raise ValueError("RadialBandTokens: level_channels must name at least one up-step.")
        if any(c < 1 for c in level_channels):
            raise ValueError(
                f"RadialBandTokens: level_channels must be >= 1, got {level_channels}."
            )
        if n_bands < 2:
            raise ValueError(
                f"RadialBandTokens: n_bands must be >= 2, got {n_bands}. One annulus is a "
                "global scalar wearing a band's name."
            )
        if n_rungs < 1:
            raise ValueError(f"RadialBandTokens: n_rungs must be >= 1, got {n_rungs}.")
        if max_log_scale <= 0:
            raise ValueError(f"RadialBandTokens: max_log_scale must be > 0, got {max_log_scale}.")

        self.level_channels = level_channels
        self.n_bands = int(n_bands)
        self.n_rungs = int(n_rungs)
        self.token_dim = int(token_dim)
        self.max_log_scale = float(max_log_scale)

        # Stored rung-major so gathering a sample's rung is a contiguous
        # index_select on dim 0; the logical index is still (annulus, rung).
        self.bank = nn.Parameter(
            torch.randn(self.n_rungs, self.n_bands, self.token_dim) * token_init_std
        )
        self.band_embedding = nn.Embedding(self.n_bands, self.token_dim)
        self.q_proj = nn.Linear(self.token_dim + 1, self.token_dim, bias=True)
        self.context = nn.Sequential(nn.Linear(self.token_dim, self.token_dim), nn.SiLU())
        self.write_heads = nn.ModuleList(nn.Linear(self.token_dim, c) for c in level_channels)
        # The one zero in the chain: every scale is exp(0) == 1.0 at step 0, so
        # the arm reproduces its control bit-for-bit.
        for head in self.write_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self._grid_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}
        self._rung_checked: set[torch.device] = set()

    def _grid(self, height: int, width: int, full: tuple[int, int], device: torch.device):
        """Bin index / disc mask / counts / this level's share of the annuli.

        Cached per (grid, full grid, device): rebuilding would allocate in the loop.
        """
        key = (height, width, full[0], full[1], device)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached

        scale_h, scale_w = height / full[0], width / full[1]
        if abs(scale_h - scale_w) > 1e-9:
            raise ValueError(
                f"RadialBandTokens: {height}x{width} is not a uniform crop of "
                f"{full[0]}x{full[1]}; an annulus index would mean two physical bands."
            )
        exact = self.n_bands * scale_h
        n_level = round(exact)
        if abs(exact - n_level) > 1e-9:
            raise ValueError(
                f"RadialBandTokens: n_bands={self.n_bands} does not divide at {height}x"
                f"{width} of {full[0]}x{full[1]} ({self.n_bands} * {scale_h:g} = {exact:g}); "
                f"use a multiple of {round(1.0 / scale_h)}."
            )
        if n_level < 2:
            raise ValueError(
                f"RadialBandTokens: {height}x{width} of {full[0]}x{full[1]} gets {n_level} "
                f"annulus, a global scalar; raise n_bands to {round(2 / scale_h)}."
            )

        index, inside, _edges = radial_bins(height, width, n_level, device)
        counts = band_counts(index, inside, n_level)
        if int((counts == 0).sum()) > 0:
            empty = [i for i, c in enumerate(counts.tolist()) if c == 0]
            raise ValueError(
                f"RadialBandTokens: annuli {empty} are empty on {height}x{width}, so their "
                "token would be fitted on no data. Lower n_bands or raise the grid size."
            )
        self._grid_cache[key] = (index, inside, counts, n_level)
        return self._grid_cache[key]

    def _validate_rung(self, rung: torch.Tensor, batch: int) -> None:
        """Refuse a rung this bank cannot index, rather than wrapping or clamping.

        The range check costs one host sync, so on an accelerator it runs once per
        device and ``index_select``'s device-side assert is the backstop; reading a
        CPU scalar is not a sync, so there it runs every call.
        """
        if rung.dim() != 1 or rung.shape[0] != batch:
            raise ValueError(
                f"RadialBandTokens: rung must be [B]; got {tuple(rung.shape)} for batch "
                f"{batch}. A broadcast rung conditions samples on the wrong step, silently."
            )
        if rung.is_floating_point() or rung.dtype == torch.bool:
            raise ValueError(
                f"RadialBandTokens: rung must be an integer step index, got {rung.dtype}. A "
                "normalised float collapses every sample onto rung 0, wired but inert."
            )
        if rung.device.type != "cpu" and rung.device in self._rung_checked:
            return
        self._rung_checked.add(rung.device)
        lo, hi = int(rung.min()), int(rung.max())
        if lo < 0 or hi >= self.n_rungs:
            raise ValueError(
                f"RadialBandTokens: rung range [{lo}, {hi}] is outside the bank's [0, "
                f"{self.n_rungs - 1}]; n_rungs must equal the process's timestep count."
            )

    def _scales(
        self, band_ids: torch.Tensor, amplitude: torch.Tensor, rung: torch.Tensor, level: int
    ) -> torch.Tensor:
        """``[B, n_level, C]`` strictly-positive scales from the bank."""
        batch = amplitude.shape[0]
        embedded = self.band_embedding(band_ids).unsqueeze(0).expand(batch, -1, -1)
        query = self.q_proj(torch.cat([embedded, amplitude.unsqueeze(-1)], dim=-1))
        # Keys and values are the whole rung, so a coarse level whose queries
        # cover only inner annuli can still read what the outer bands carry.
        keys = self.bank.index_select(0, rung)
        attended = F.scaled_dot_product_attention(
            query.unsqueeze(1), keys.unsqueeze(1), keys.unsqueeze(1)
        ).squeeze(1)
        logits = self.write_heads[level](self.context(attended))
        return torch.exp(torch.clamp(logits, -self.max_log_scale, self.max_log_scale))

    def forward(
        self,
        features: torch.Tensor,
        rung: torch.Tensor,
        *,
        level: int,
        full_size: tuple[int, int],
    ) -> torch.Tensor:
        complex_features = features if features.is_complex() else interleaved_to_complex(features)
        batch, channels, height, width = complex_features.shape
        if channels != self.level_channels[level]:
            raise ValueError(
                f"RadialBandTokens: up-step {level} was built for "
                f"{self.level_channels[level]} complex channels, got {channels}."
            )
        self._validate_rung(rung, batch)
        index, inside, counts, n_level = self._grid(
            height, width, full_size, complex_features.device
        )

        # Mean amplitude per annulus in one scatter; corners contribute zero, the
        # population band_counts measured. float32 because an fp16 scatter over
        # ~1e5 terms loses accuracy log1p then bakes into the query.
        amplitude = complex_features.abs().mean(dim=1)
        masked = (amplitude * inside).reshape(batch, -1).float()
        pooled = torch.zeros(batch, n_level, device=masked.device, dtype=masked.dtype).index_add_(
            1, index.reshape(-1), masked
        )
        pooled = torch.log1p((pooled / counts.clamp_min(1.0)).clamp_min(0.0))
        # Centred, so the query is blind to the global level ComplexRMSNorm
        # renormalises away and sees only the shape this block can act on.
        pooled = pooled - pooled.mean(dim=1, keepdim=True)

        band_ids = torch.arange(n_level, device=complex_features.device)
        scales = self._scales(band_ids, pooled, rung.long(), level)

        per_bin = scales.permute(0, 2, 1)[:, :, index]
        per_bin = torch.where(inside.unsqueeze(0).unsqueeze(0), per_bin, torch.ones_like(per_bin))
        scaled = complex_features * per_bin.to(complex_features.dtype)
        return scaled if features.is_complex() else complex_to_interleaved(scaled)

    def band_scales(
        self,
        level: int = -1,
        rungs: torch.Tensor | None = None,
        band_amplitude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[len(rungs), n_bands, C]`` scales, read without a forward pass.

        This is what falsifies the block, so it is readable from the bank alone:
        a mechanism whose decision can only be inferred from a ratio of outputs
        is graded by proxy. ``band_amplitude`` defaults to a flat spectrum -- the
        neutral probe of what the bank encodes, not what one batch elicited.
        """
        device = self.bank.device
        if rungs is None:
            rungs = torch.arange(self.n_rungs, device=device)
        if band_amplitude is None:
            band_amplitude = torch.zeros(self.n_bands, device=device)
        band_ids = torch.arange(self.n_bands, device=device)
        amplitude = band_amplitude.to(device).reshape(1, -1).expand(rungs.shape[0], -1)
        return self._scales(band_ids, amplitude, rungs.to(device).long(), level)

    def differentiation(self, level: int = -1) -> dict[str, float]:
        """Spread of ``log`` scale across annuli and across rungs.

        Both are exactly zero at initialisation, so a non-zero reading is
        attributable to training rather than to the draw. Near zero on the
        annulus axis means the bank is a global embedding; near zero on the rung
        axis means that axis is unused and should be dropped.
        """
        with torch.no_grad():
            log_scales = self.band_scales(level=level).log()
            return {
                "across_annuli": float(log_scales.std(dim=1).mean()),
                "across_rungs": float(log_scales.std(dim=0).mean()),
            }
