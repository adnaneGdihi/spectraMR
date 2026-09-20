r"""Phase-exact per-annulus magnitude gain on a k-space prediction.

The ``kspace_filling`` cohort's reconstructions carry roughly 0.30 of the
target's energy at R=32 and roll off past ``r/Nyq ~ 0.5``. Measured at
initialisation, where ``ComplexUNet`` is exactly linear, every arm in the
attention shootout returns ``outer_band_retention`` between **0.198 and 0.239**
against a 0.50 low-pass threshold, and the trained control's outer-band gain
back-solves to within ~1.2x of that untrained floor. The deficit tracks
``2^-(up_steps)``: it is a *multiplicative* attenuation introduced by the
decoder, so the matched correction is a *multiplicative* one.

Attention cannot supply it. Every block in that cohort is gain-bounded --
the energy probe puts ``worst_rho`` at 1.00-1.10 across all ten arms -- because
a softmax redistributes energy and does not create it. What the measurement asks
for is a scalar per band: ``val_pred_target_optimal_gain_32x`` is **1.71**.

**The annulus-aware selector already exists and is not enough**, which is the
evidence this block rests on rather than an argument for it.
:class:`~spectramr.models.blocks.dual_domain_attention_kan.RadialBandAttention`
restricts self-attention to concentric annular bands and is deployed in the
k-space branch of ``attention_type: kan_dual_domain``. Its three arms measure
``outer_band_retention`` 0.213 / 0.222 / 0.223 against ``none``'s 0.198 -- a real
improvement, and roughly a tenth of the way to the 0.78 that removing
``kspace_feature_norm`` reaches. Band-awareness is not the missing ingredient;
unbounded-in-the-right-direction *scale* is. This block adds the second without
touching the first, so an arm may carry both.

Its partition is deliberately **not** shared with that block: ``RadialBandAttention``
spaces bands logarithmically to resolve DC finely, while these annuli are
equal-width in ``r`` because they must match
:func:`~spectramr.infrastructure.physics.radial_bands.radial_bins`, which is what
the spectral-transfer probe measures. Two partitions for two purposes, stated
rather than silently divergent.

The gain is real and strictly positive, so

.. math::

    \hat k_b \;\mapsto\; m_b \cdot \hat k_b, \qquad m_b = e^{\gamma \cdot f_b} > 0

leaves :math:`\arg \hat k_b` **algebraically** unchanged. That is not a soft
penalty: phase error cannot originate here, so a drifting ``val_band_*``
argument localises the fault upstream instead of being confounded with it. A
complex gain would give no such separation, and the reverse loop's magnitude
ceiling is already phase-invariant by construction, so an energy-only check
cannot tell a correct band from a rotated one.

``gamma`` is zero-initialised, so at step 0 every :math:`m_b \equiv 1` and the
arm is bit-identical to its control -- the same identity-at-init convention the
attention wrapper uses, and what makes the A/B honest.

**Conditioning deliberately excludes the measurement.** The band features are
the prediction's own per-annulus amplitude plus a timestep embedding, both
available on every path. The measured band energy would be a better feature and
is *not* available to the reverse sampler, which calls ``model(x, t)`` bare --
the same gap that leaves the generator's data-consistency layer inert at
validation. A mechanism that silently weakens where the reported numbers come
from is the defect this cohort keeps rediscovering (pitfall 16).

**No null-space restriction, on purpose.** Hard data consistency overwrites the
observed support after this runs, so a gain applied there is discarded rather
than harmful, and skipping the ``(1 - M)`` gate keeps this block independent of
mask availability.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.radial_bands import (
    band_counts,
    band_reduce,
    radial_bins,
)

__all__ = ["DEFAULT_MAX_LOG_GAIN", "RadialBandGain"]

#: Symmetric bound on ``|log m_b|``, admitting ``m_b`` in ``[0.20, 4.95]`` --
#: the measured 1.71 deficit with room, and still a refusal of a runaway. One
#: owner: the generator reads this rather than repeating the literal.
DEFAULT_MAX_LOG_GAIN = 1.6


class RadialBandGain(nn.Module):
    """Learned real gain per radial annulus, conditioned on band and timestep.

    Args:
        n_bands: Number of annuli. Must be >= 2 and must leave no bin empty on
            the grids the arm runs at.
        time_embed_dim: Width of the timestep embedding this receives. ``0``
            disables timestep conditioning, leaving a per-band scalar.
        hidden: Width of the per-band MLP.
        max_log_gain: Symmetric bound on ``|log m_b|``. The default 1.6 admits
            ``m_b`` in ``[0.20, 4.95]``, which covers the measured 1.71 deficit
            with room and still refuses a runaway.

    Shape:
        ``kspace``: complex ``[B, C, H, W]`` or interleaved real
        ``[B, 2C, H, W]``; returns the same layout and dtype.
    """

    def __init__(
        self,
        n_bands: int = 8,
        time_embed_dim: int = 0,
        hidden: int = 32,
        max_log_gain: float = DEFAULT_MAX_LOG_GAIN,
    ) -> None:
        super().__init__()
        if n_bands < 2:
            raise ValueError(f"RadialBandGain: n_bands must be >= 2, got {n_bands}.")
        if max_log_gain <= 0:
            raise ValueError(f"RadialBandGain: max_log_gain must be > 0, got {max_log_gain}.")
        self.n_bands = int(n_bands)
        self.time_embed_dim = int(time_embed_dim)
        self.max_log_gain = float(max_log_gain)

        self.band_embedding = nn.Embedding(self.n_bands, hidden)
        # Features per band: the embedding, log1p of the band's own amplitude,
        # and the timestep embedding when the caller supplies one.
        in_features = hidden + 1 + self.time_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # Identity at init: gamma = 0 => log m_b = 0 => m_b = 1 exactly.
        self.gamma = nn.Parameter(torch.zeros(1))
        self._grid_cache: dict[tuple[int, int, torch.device], tuple[torch.Tensor, ...]] = {}

    def _grid(self, height: int, width: int, device: torch.device):
        """Bin index / mask / per-band counts, built once per (grid, device).

        Rebuilding them every step would allocate inside the training loop
        (non-negotiable 9); they depend only on the grid, so they are cached.
        """
        key = (height, width, device)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        index, inside, _edges = radial_bins(height, width, self.n_bands, device)
        counts = band_counts(index, inside, self.n_bands)
        if int((counts == 0).sum()) > 0:
            empty = [i for i, c in enumerate(counts.tolist()) if c == 0]
            raise ValueError(
                f"RadialBandGain: n_bands={self.n_bands} leaves annuli {empty} empty on a "
                f"{height}x{width} grid, so their gain would be fitted on no data. "
                f"Lower n_bands or raise the grid size."
            )
        self._grid_cache[key] = (index, inside, counts)
        return index, inside, counts

    def band_gains(
        self, kspace: torch.Tensor, time_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The per-sample, per-band gains ``[B, n_bands]``, strictly positive.

        Exposed so a probe can read what the mechanism decided without re-running
        the forward and without inferring it from a ratio -- the per-band gain is
        the quantity that falsifies this block, and a mechanism whose own output
        is unreadable is graded by proxy.
        """
        complex_k = _as_complex(kspace)
        b, _c, h, w = complex_k.shape
        index, inside, counts = self._grid(h, w, complex_k.device)

        # Mean amplitude per annulus, averaged over coils, per sample.
        amplitude = complex_k.abs().mean(dim=1)  # [B, H, W]
        per_band = torch.stack(
            [band_reduce(amplitude[i], index, inside, self.n_bands) for i in range(b)]
        )
        per_band = per_band / counts.clamp_min(1.0)

        band_ids = torch.arange(self.n_bands, device=complex_k.device)
        features = [
            self.band_embedding(band_ids).unsqueeze(0).expand(b, -1, -1),
            torch.log1p(per_band.clamp_min(0.0)).unsqueeze(-1),
        ]
        if self.time_embed_dim:
            if time_embedding is None:
                raise ValueError(
                    "RadialBandGain was built with time_embed_dim="
                    f"{self.time_embed_dim} but received no time_embedding. A "
                    "timestep-conditioned gain that silently drops the timestep "
                    "is a different mechanism from the one the arm declared."
                )
            if time_embedding.shape[-1] != self.time_embed_dim:
                raise ValueError(
                    f"RadialBandGain: time_embedding width {time_embedding.shape[-1]} "
                    f"does not match time_embed_dim={self.time_embed_dim}."
                )
            features.append(time_embedding.unsqueeze(1).expand(-1, self.n_bands, -1))

        raw = self.mlp(torch.cat(features, dim=-1)).squeeze(-1)  # [B, n_bands]
        log_gain = torch.clamp(self.gamma * raw, -self.max_log_gain, self.max_log_gain)
        return torch.exp(log_gain)

    def forward(
        self, kspace: torch.Tensor, time_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        gains = self.band_gains(kspace, time_embedding)  # [B, n_bands]
        complex_k = _as_complex(kspace)
        _b, _c, h, w = complex_k.shape
        index, inside, _counts = self._grid(h, w, complex_k.device)

        # Gather each bin's gain, leaving bins outside the Nyquist disc at 1.0 --
        # they are the corners `radial_bins` excludes, and scaling a population
        # the bands were never fitted on would be inventing a band.
        per_bin = gains[:, index]  # [B, H, W]
        per_bin = torch.where(inside.unsqueeze(0), per_bin, torch.ones_like(per_bin))
        scaled = complex_k * per_bin.unsqueeze(1).to(complex_k.dtype)
        return _restore_layout(scaled, kspace)


def _as_complex(x: torch.Tensor) -> torch.Tensor:
    """Interleaved ``[R0, I0, R1, I1, ...]`` -> complex, or pass a complex tensor."""
    if x.is_complex():
        return x
    if x.dim() != 4:
        raise ValueError(f"RadialBandGain expects a 4-D k-space tensor, got {tuple(x.shape)}.")
    if x.shape[1] % 2 != 0:
        raise ValueError(
            "RadialBandGain: interleaved Re/Im layout requires an even channel "
            f"count, got {tuple(x.shape)}."
        )
    return torch.complex(x[:, 0::2], x[:, 1::2])


def _restore_layout(scaled: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return ``scaled`` in whichever layout ``reference`` arrived in."""
    if reference.is_complex():
        return scaled
    out = torch.zeros_like(reference)
    out[:, 0::2] = scaled.real
    out[:, 1::2] = scaled.imag
    return out
