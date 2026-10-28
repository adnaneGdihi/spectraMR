r"""Reconstruction error on the acquired and on the acquirable-but-unacquired k-space bins.

The null-band metrics score only the bins the mask left out. A cold-diffusion arm with
hard data consistency pins every acquired bin of its output to the noisy measurement, so
its error there is not the model's: it is the distance between one noisy excitation and
the NEX-mean target, and it is the same for every network. A denoising consistency step
(soft, adaptive, noise-adaptive) changes exactly that number, so the two families can be
compared only if it is reported beside the unacquired-bin error:

.. math::

    e_S = \frac{\lVert (\hat{\mathbf{k}} - \mathbf{k})_S \rVert_2}
               {\lVert \mathbf{k}_S \rVert_2},

with :math:`S` the acquired bins (``acquired_bin_nrmse``) or the acquirable bins the mask
left out (``unacquired_bin_nrmse``).

**Domain.** The inputs are the linear (log-scaling undone) per-coil complex k-space, not
images. Hard consistency makes the bin-wise identity the point, and a coil-combined image
re-transformed to k-space mixes acquired and unacquired bins. The validation seam hands
them over before any image transform.

**Acquirable.** M4Raw stores 195 of 256 phase-encode lines; the rest are exact zeros in
input and target. A bin is acquirable when neither its row nor its column of the TARGET
is identically zero over all coils, which is ``acquired_lines_of``'s rule without naming
the axis. Never-acquired lines are dropped from both numerator and denominator of both
metrics. ``null_band_*`` keeps them in the numerator (a prediction that fills them counts
as error there) and, being zero in the target, never in the denominator.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.nr_consistency import _as_complex_image
from spectramr.core.metrics.registry import register_metric

NAN = float("nan")

ACQUIRED_NAME = "acquired_bin_nrmse"
UNACQUIRED_NAME = "unacquired_bin_nrmse"


def _measured(mask: Tensor, like: Tensor) -> Tensor:
    """``[B, H, W]`` bool of the bins any channel measured."""
    m = mask.real if torch.is_complex(mask) else mask
    m = m.to(device=like.device, dtype=torch.float32)
    while m.dim() < 4:
        m = m.unsqueeze(0)
    return (m.amax(dim=1) > 0).expand(like.shape[0], like.shape[-2], like.shape[-1])


def acquirable_bins(target: Tensor) -> Tensor:
    """``[B, H, W]`` bool: bins on no all-zero row or column of the target."""
    energy = target.abs().pow(2).sum(dim=1)
    return (energy.sum(dim=-1) > 0)[:, :, None] & (energy.sum(dim=-2) > 0)[:, None, :]


def bin_errors(prediction: Tensor, target: Tensor, mask: Tensor) -> tuple[float, float]:
    """``(acquired, unacquired)`` NRMSE over the batch: sums over bins, then the ratio."""
    k_pred = _as_complex_image(prediction)
    k_tgt = _as_complex_image(target)
    acquirable = acquirable_bins(k_tgt)
    measured = _measured(mask, k_tgt)
    err = (k_pred - k_tgt).abs().pow(2).sum(dim=1)
    ref = k_tgt.abs().pow(2).sum(dim=1)
    out = []
    for sel in (measured & acquirable, ~measured & acquirable):
        e, r = err[sel].sum(), ref[sel].sum()
        # No target energy on the set means no content to be wrong about: 0, not 0/0.
        out.append(float((e / r).sqrt()) if float(r) > 0 else 0.0)
    return out[0], out[1]


class _BinNRMSE:
    """NRMSE of the per-coil complex k-space on one bin set (lower is better, full-reference)."""

    index: int = 0
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
        # No mask, no split; the validation seam always supplies one and overwrites this nan.
        if target is None or not ctx.has("mask") or ctx.mask is None:
            return NAN
        if _as_complex_image(prediction).shape != _as_complex_image(target).shape:
            return NAN
        return bin_errors(prediction, target, ctx.mask)[self.index]


def _register(name: str, index: int) -> type:
    cls = type(
        f"BinNRMSE_{name.split('_', 1)[0]}",
        (_BinNRMSE,),
        {"index": index, "metric_name": name, "__doc__": _BinNRMSE.__doc__},
    )
    register_metric(name, requires_reference=True, needs=("mask",), direction="lower")(cls)
    return cls


BIN_METRICS: dict[str, type] = {
    ACQUIRED_NAME: _register(ACQUIRED_NAME, 0),
    UNACQUIRED_NAME: _register(UNACQUIRED_NAME, 1),
}

__all__ = ["ACQUIRED_NAME", "BIN_METRICS", "UNACQUIRED_NAME", "bin_errors"]
