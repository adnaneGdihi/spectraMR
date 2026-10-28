r"""Reconstruction error on the unmeasured k-space, split into frequency bands.

``null_band_energy_deficit`` says how much of the missing energy came back, over
the whole unacquired band at once. That band's energy sits at its low-|k| end,
so the number cannot tell a model that restores structure at every frequency
from one that restores the low frequencies and blurs the rest.

These metrics sort each sample's unmeasured bins by radial :math:`|k|` and split
them into three bands of equal count, ``inner``, ``middle`` and ``outer``:

.. math::

    e_b = \frac{\lVert (\mathbf{F}\hat{\mathbf{x}} - \mathbf{F}\mathbf{x})_{B_b} \rVert_2}
               {\lVert (\mathbf{F}\mathbf{x})_{B_b} \rVert_2}

where :math:`B_b` is the band's set of unmeasured bins. A zero-filled
reconstruction scores 1.0 in every band and an exact one 0, so ``1 - e_b`` is
the fraction of that band's missing content recovered, and ``outer`` is the
highest-frequency third of what the mask left out.

The bands follow the mask rather than fixed radii. A fixed-radius shell is empty
whenever a dense calibration core or a variable-density mask measures all of it
-- on the cohort's ladder the innermost fixed shells hold no unmeasured bin at
2x -- and an empty shell has no score. Every accelerated rung leaves unmeasured
bins, so every band here holds a third of them and always has a value. Like the
deficit, the images are reduced to one channel first.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.nr_consistency import _single_channel_complex
from spectramr.core.metrics.registry import register_metric

NAN = float("nan")

#: ``(name, band index)``; band 0 holds the unmeasured bins nearest the centre.
BANDS: tuple[tuple[str, int], ...] = (
    ("null_band_nrmse_inner", 0),
    ("null_band_nrmse_middle", 1),
    ("null_band_nrmse_outer", 2),
)
NUM_BANDS = len(BANDS)


def _radius(height: int, width: int, device: torch.device) -> Tensor:
    ky = torch.arange(height, device=device, dtype=torch.float32) - height // 2
    kx = torch.arange(width, device=device, dtype=torch.float32) - width // 2
    return torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)


def _unmeasured(mask: Tensor, batch: int, height: int, width: int, device: torch.device) -> Tensor:
    """``[B, H, W]`` bool of the bins no channel measured."""
    m = mask.real if torch.is_complex(mask) else mask
    m = m.to(device=device, dtype=torch.float32)
    while m.dim() < 4:
        m = m.unsqueeze(0)
    return (m.amax(dim=1) <= 0).expand(batch, height, width)


def band_errors(prediction: Tensor, target: Tensor, mask: Tensor) -> list[float]:
    """Per-band NRMSE over the batch: error and target energy summed per band, then divided."""
    from spectramr.infrastructure.physics.fft_ops import fft2c

    k_pred = fft2c(_single_channel_complex(prediction))[:, 0]
    k_tgt = fft2c(_single_channel_complex(target))[:, 0]
    batch, height, width = k_tgt.shape
    radius = _radius(height, width, k_tgt.device).flatten()
    unmeasured = _unmeasured(mask, batch, height, width, k_tgt.device)
    err = (k_pred - k_tgt).abs().pow(2).reshape(batch, -1)
    ref = k_tgt.abs().pow(2).reshape(batch, -1)
    err_sum = torch.zeros(NUM_BANDS, device=k_tgt.device)
    ref_sum = torch.zeros(NUM_BANDS, device=k_tgt.device)
    for b in range(batch):
        idx = unmeasured[b].flatten().nonzero().flatten()
        if idx.numel() == 0:
            continue
        ordered = idx[torch.argsort(radius[idx], stable=True)]
        for band, part in enumerate(torch.tensor_split(ordered, NUM_BANDS)):
            err_sum[band] += err[b, part].sum()
            ref_sum[band] += ref[b, part].sum()
    # A band with no target energy holds no missing content: report no error rather than 0/0.
    ratio = torch.where(ref_sum > 0, err_sum / ref_sum.clamp_min(1e-30), torch.zeros_like(err_sum))
    return [float(v) for v in ratio.sqrt()]


class _NullBandNRMSE:
    """NRMSE of one equal-count band of the unmeasured bins (↓, full-reference)."""

    band: int = 0
    metric_name: str = ""

    @property
    def name(self) -> str:
        return self.metric_name

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        # Without a mask there is no unmeasured band; the validation seam always
        # supplies one, and the context-free pass's nan is overwritten there.
        if target is None or not ctx.has("mask") or ctx.mask is None:
            return NAN
        if _single_channel_complex(prediction).shape != _single_channel_complex(target).shape:
            return NAN
        return band_errors(prediction, target, ctx.mask)[self.band]


def _register(name: str, band: int) -> type:
    cls = type(
        f"NullBandNRMSE_{name.rsplit('_', 1)[-1]}",
        (_NullBandNRMSE,),
        {"band": band, "metric_name": name, "__doc__": _NullBandNRMSE.__doc__},
    )
    register_metric(name, requires_reference=True, needs=("mask",), direction="lower")(cls)
    return cls


BAND_METRICS: dict[str, type] = {name: _register(name, band) for name, band in BANDS}

__all__ = ["BANDS", "BAND_METRICS", "NUM_BANDS", "band_errors"]
