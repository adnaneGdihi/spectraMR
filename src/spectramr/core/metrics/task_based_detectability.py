r"""Task-based detectability metric (NR battery §8 / A-8.1 ``task_based_detectability``).

A **model-observer** image-quality metric. Where PSNR / SSIM report a *global*
fidelity, this reports how detectable a standard clinical-style signal (a small
Gaussian "lesion") remains against the **reconstruction's own noise** — the
detectability index :math:`d'` of a channelized Hotelling observer (CHO), the
principled, reproducible surrogate for a human-observer ROC study (Barrett &
Myers, *Foundations of Image Science*, ch. 13-14).

Why a full-reference, anatomy-removed noise ensemble
----------------------------------------------------
The CHO estimates a channel covariance :math:`K_g` from an ensemble of **noise**
realisations. Pixels taken straight from a batch of *different* slices are
dominated by **anatomy**, not noise, so a naive ensemble confounds :math:`K_g`
and the reported :math:`d'` means nothing. We instead form the residual
``R = pred - target`` (the shared anatomy cancels, leaving the recon error
field), tile it into patches as a *stationary-noise* ensemble, and insert a fixed
signal template ``s`` to build the signal-present class. Then

.. math::

    d'^2 = (\mathbf{U}^\top \mathbf{s})^\top K_g^{-1} (\mathbf{U}^\top \mathbf{s}),

so a recon whose noise carries **little energy in the signal band** yields a
large :math:`d'` (the lesion stays detectable → good), while a recon that injects
spurious band-matched texture *masks* the lesion (small :math:`d'` → bad).

Two distinct, separately-guarded properties make this useful where PSNR is not:

1. **Colour-sensitivity at fixed PSNR.** Two recons with identical MSE (hence
   identical PSNR) but differently *coloured* noise score differently here — PSNR is
   invariant to noise colour, this metric is not.
   (``test_sensitive_to_noise_colour_at_fixed_psnr``.)
2. **Task-specificity.** Because the channel weighting follows the lesion's own
   spectrum, equal-energy noise placed *inside* the lesion's spatial-frequency band
   masks it more (lower :math:`d'`) than noise placed *outside* it — the metric is
   tuned to the detection task, not merely to "any structure".
   (``test_task_specific_to_lesion_band``.)

Assumptions (documented, **not** validated at call time): ``pred`` and ``target``
are spatially **aligned** (so the residual is noise, not mis-registration), and the
residual noise is approximately **stationary** across the tiled patches. The metric
is **undefined for a (near-)perfect reconstruction** — a zero residual has no noise
to measure detectability against, so it returns ``nan`` rather than a
ridge-determined constant (pitfall #20: no measurement-independent output).

Scale-invariance
----------------
The residual and the lesion are normalised by a robust intensity scale of the
target (95th percentile of ``|target|``) before the observer, so signal and noise
share one scale and :math:`d'` is invariant to a global intensity rescale of the
``(pred, target)`` pair (the B-2.7 lesson: never let a fixed absolute threshold
decide a physical quantity).

Contract: full-reference (``requires_reference=True``), ``higher_is_better``;
returns ``float('nan')`` on degenerate input (never raises inside ``__call__`` —
the metrics computer would otherwise abort the whole validation).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.nr_texture import _as_magnitude_2d, _param
from spectramr.core.metrics.registry import register_metric
from spectramr.infrastructure.physics.model_observer import (
    ChannelizedHotellingObserver,
    gabor_channel_bank,
)

# Channel banks are fixed (data-independent) templates -> cache per (patch, n_channels).
_BANK_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _bank(patch: int, n_channels: int) -> torch.Tensor:
    key = (patch, n_channels)
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = gabor_channel_bank((patch, patch), n_channels=n_channels)
    return _BANK_CACHE[key]


def _signal_template(patch: int, radius_frac: float) -> torch.Tensor:
    """Fixed centred Gaussian-blob 'lesion' of unit peak, flattened ``[patch*patch]``."""
    coords = torch.arange(patch, dtype=torch.float32) - (patch - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    sigma = max(1.0, radius_frac * patch)
    blob = torch.exp(-(yy * yy + xx * xx) / (2.0 * sigma * sigma))
    return blob.reshape(-1)


@register_metric(
    "task_based_detectability",
    aliases=["cho_detectability", "task_detectability"],
    requires_reference=True,
    direction="higher",
)
class TaskBasedDetectability:
    r"""CHO detectability :math:`d'` of a standard synthetic lesion against the
    reconstruction's own (anatomy-removed) noise. Higher = a clinical-style
    signal stays detectable = better recon.

    Args:
        patch: side of the square noise patches tiled from the residual.
        n_channels: number of Gabor observer channels.
        lesion_contrast: lesion peak as a fraction of the target's robust scale.
        lesion_radius_frac: Gaussian-blob std as a fraction of ``patch``.
        ridge: ridge added to the (normalised) channel covariance before solve.
    """

    def __init__(
        self,
        *,
        patch: int = 32,
        n_channels: int = 6,
        lesion_contrast: float = 0.1,
        lesion_radius_frac: float = 0.12,
        ridge: float = 1e-4,
    ) -> None:
        self.patch = int(patch)
        self.n_channels = int(n_channels)
        self.lesion_contrast = float(lesion_contrast)
        self.lesion_radius_frac = float(lesion_radius_frac)
        self.ridge = float(ridge)

    @property
    def name(self) -> str:
        return "task_based_detectability"

    @property
    def higher_is_better(self) -> bool:
        return True

    def _noise_patches(self, residual: torch.Tensor) -> torch.Tensor | None:
        """Tile ``[B,1,H,W]`` into ``[patch*patch, N]`` noise columns (one per tile)."""
        p = self.patch
        _, _, h, w = residual.shape
        if h < p or w < p:
            return None
        cols = F.unfold(residual, kernel_size=p, stride=p)  # [B, p*p, L]
        cols = cols.permute(1, 0, 2).reshape(p * p, -1)  # [p*p, B*L]
        if cols.shape[1] < 2:  # need >= 2 realisations to estimate a covariance
            return None
        return cols

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if target is None:
            return float("nan")
        pred = _as_magnitude_2d(prediction)
        tgt = _as_magnitude_2d(target)
        if pred is None or tgt is None or pred.shape != tgt.shape:
            return float("nan")

        contrast = _param(ctx, "taskdet_contrast", self.lesion_contrast)

        # Robust intensity scale -> normalise so signal & noise share one scale
        # (exact rescale-invariance + a well-conditioned absolute ridge).
        scale = float(torch.quantile(tgt.flatten().abs(), 0.95).item())
        if not math.isfinite(scale) or scale <= 0:
            scale = float(tgt.abs().mean().item())
        if scale <= 0:
            return float("nan")

        residual = (pred - tgt) / scale  # anatomy cancels -> O(1) recon-noise field
        cols = self._noise_patches(residual)
        if cols is None:
            return float("nan")
        if float(cols.std()) < 1e-6:
            # (Near-)perfect reconstruction: no noise to estimate K_g from, so d'
            # would collapse to a ridge-determined constant independent of the data.
            # The figure of merit is undefined here (pitfall #20) -> nan. Real recons,
            # whose residual is always >> 1e-6 of the signal scale, never trigger this.
            return float("nan")

        s = (
            _signal_template(self.patch, self.lesion_radius_frac).to(
                device=cols.device, dtype=cols.dtype
            )
            * contrast
        )
        present = cols + s.unsqueeze(1)  # signal + noise
        absent = cols  # noise only

        bank = _bank(self.patch, self.n_channels).to(device=cols.device, dtype=cols.dtype)
        observer = ChannelizedHotellingObserver(bank, ridge=self.ridge)
        d_sq = observer.detectability(present, absent)
        if not math.isfinite(d_sq):
            return float("nan")
        return math.sqrt(max(0.0, d_sq))
