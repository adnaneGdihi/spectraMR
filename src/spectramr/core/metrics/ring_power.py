r"""Radial power of the acquired per-coil k-space in three rings, against the target.

The scalar errors the cohort reports (``hfen``, ``unacquired_bin_nrmse``) do not say which
radii carry the deficit. A reconstruction can recover the centre and blur every ring beyond
it, and a whole-band average hides that. These metrics split the acquired bins into rings of
radius :math:`\rho = |k| / (\min(H,W)/2)`:

.. math::

    \text{mid1} \in [\tfrac18, \tfrac14), \quad
    \text{mid2} \in [\tfrac14, \tfrac12), \quad
    \text{high} \in [\tfrac12, \tfrac34)

and report, per ring, the power ratio :math:`P_{\hat{x}}/P_x`, the NRMSE and the target SNR
against :func:`noise_floor`. ``ring_power_ratio`` is read "higher is better" because the cohort
lives below 1 (a deficit); a ratio above 1 is the hallucination side and is policed by the
selection guard, not by the direction.

Bins are the per-coil complex k-space the validation seam hands over, not images: a SENSE or
RSS image re-transformed to k-space mixes acquired and unacquired bins. The acquired bins are
``acquirable_bins`` of the target (the same rule ``unacquired_bin_nrmse`` uses), so the
never-acquired phase-encode lines are excluded from every ring. Rings are scored over every
acquired bin in them, measured or not.
"""

from __future__ import annotations

import math
from functools import cache

import torch
from torch import Tensor

from spectramr.core.metrics.acquired_bin_error import acquirable_bins
from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.nr_consistency import _as_complex_image
from spectramr.core.metrics.registry import register_metric

NAN = float("nan")

#: Ring edges as fractions of the Nyquist radius ``min(H, W) / 2``.
EDGES: tuple[float, ...] = (1 / 8, 1 / 4, 1 / 2, 3 / 4)
#: ``(ring name, index)``; the index selects a column of :class:`RingStats`.
RINGS: tuple[tuple[str, int], ...] = (("mid1", 0), ("mid2", 1), ("high", 2))
#: The noise floor is read from bins at or beyond this fraction of the Nyquist radius.
NOISE_FROM = 0.85
#: Columns of the per-ring feature stack: error power, target power, prediction power, bin count.
_FEATURES = 4
#: Scatter slot for bins outside every ring; sliced away before use.
_DUMP = len(RINGS)


@cache
def _edges(device: torch.device) -> Tensor:
    """Ring edges on ``device``, built once: a per-call ``torch.tensor`` is a host-to-device copy."""
    return torch.tensor(EDGES, device=device)


def _radius(height: int, width: int, device: torch.device) -> Tensor:
    """``[H*W]`` radius in units of the Nyquist radius, DC at ``(H//2, W//2)``."""
    ky = torch.arange(height, device=device, dtype=torch.float32) - height // 2
    kx = torch.arange(width, device=device, dtype=torch.float32) - width // 2
    nyquist = min(height, width) / 2
    return torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).flatten() / nyquist


def noise_floor(target: Tensor) -> Tensor:
    """Per-sample per-coil noise power sigma^2 estimated from the target's outer acquired k-space.

    The median of an exponential power spectrum is ``ln 2`` times its mean, so the median of the
    acquired bins at ``rho >= 0.85`` divided by ``ln 2`` estimates sigma^2 for complex Gaussian
    noise. Returns ``[B]``; a sample with no such bin is NaN, not a guess.
    """
    k = _as_complex_image(target)
    batch, _, height, width = k.shape
    outer = (_radius(height, width, k.device) >= NOISE_FROM).view(1, 1, -1)
    keep = acquirable_bins(k).view(batch, 1, -1) & outer
    values = (k.abs().pow(2).reshape(batch, k.shape[1], -1)).masked_fill(~keep, NAN)
    return torch.nanmedian(values.reshape(batch, -1), dim=1).values / math.log(2.0)


def ring_statistics(
    prediction: Tensor, target: Tensor, *, with_noise: bool = True
) -> dict[str, Tensor]:
    """Batch-summed error, target and prediction power, bin count per ring, and sigma^2.

    Each entry is a ``[len(RINGS)]`` tensor except ``sigma2``, a scalar (the batch mean of
    :func:`noise_floor`, NaN unless ``with_noise``: the median sort is only needed for SNR).
    """
    pred = _as_complex_image(prediction)
    tgt = _as_complex_image(target)
    batch, _, height, width = tgt.shape
    acquired = acquirable_bins(tgt).view(batch, -1).to(torch.float32)
    rho = _radius(height, width, tgt.device)
    bucket = torch.bucketize(rho, _edges(tgt.device), right=True)
    ring = bucket - 1
    slot = torch.where((ring >= 0) & (ring < len(RINGS)), ring, torch.full_like(ring, _DUMP))
    p_pow = tgt.abs().pow(2).sum(dim=1).reshape(batch, -1)
    q_pow = pred.abs().pow(2).sum(dim=1).reshape(batch, -1)
    e_pow = (pred - tgt).abs().pow(2).sum(dim=1).reshape(batch, -1)
    features = torch.stack([e_pow, p_pow, q_pow, torch.ones_like(p_pow)], dim=1)
    features = features * acquired[:, None, :]
    # One GEMM against the ring membership matrix: the batch and feature axes fold into rows.
    membership = torch.nn.functional.one_hot(slot, _DUMP + 1).to(features.dtype)
    sums = torch.matmul(features, membership)[:, :, : len(RINGS)]
    summed = sums.sum(dim=0)
    sigma2 = noise_floor(tgt).nanmean() if with_noise else torch.full((), NAN, device=tgt.device)
    return {
        "error": summed[0],
        "target": summed[1],
        "prediction": summed[2],
        "count": summed[3],
        "coils": torch.full((), float(tgt.shape[1]), device=tgt.device),
        "sigma2": sigma2,
    }


#: The last ``(prediction, target, versions, statistics)``. The validation seam scores all
#: nine ring metrics on the same pair, so they share one pass instead of nine.
_SHARED: list[tuple[Tensor, Tensor, tuple[int, int], dict[str, Tensor]]] = []


def _shared_statistics(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    """``ring_statistics`` with the noise floor, reused while the same tensors are unmodified.

    Keyed on object identity plus the in-place version counters; holding the pair keeps its
    memory from being reused by a later batch, which a ``data_ptr`` key could not rule out.
    """
    versions = (prediction._version, target._version)
    if _SHARED:
        held_pred, held_target, held_versions, held_stats = _SHARED[0]
        if held_pred is prediction and held_target is target and held_versions == versions:
            return held_stats
    stats = ring_statistics(prediction, target, with_noise=True)
    _SHARED[:] = [(prediction, target, versions, stats)]
    return stats


def _ring_value(stats: dict[str, Tensor], kind: str, index: int) -> float:
    """One ring's scalar for ``kind``; NaN when the ring holds no acquired energy."""
    picked = [stats["error"][index], stats["target"][index], stats["prediction"][index]]
    picked += [stats["count"][index], stats["coils"], stats["sigma2"]]
    values: list[float] = torch.stack(picked).tolist()
    err, tgt, pred, count, coils, sigma2 = values
    if count <= 0 or tgt <= 0:
        return NAN
    if kind == "power_ratio":
        return pred / tgt
    if kind == "nrmse":
        return math.sqrt(err / tgt)
    return (tgt / (count * coils)) / sigma2 if sigma2 > 0 else NAN


class _RingMetric:
    """One ring's statistic of the per-coil k-space prediction against the target (full-reference)."""

    ring: int = 0
    kind: str = ""
    metric_name: str = ""

    @property
    def name(self) -> str:
        return self.metric_name

    @property
    def higher_is_better(self) -> bool:
        return self.kind != "nrmse"

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        # The prediction is per-coil k-space, which only the strategy that produced it
        # knows is k-space; ``pred_kspace`` declares that domain, so a context-free image
        # pass returns NaN and the seam owns the value.
        if target is None or not resolve_context(context, kwargs).has("pred_kspace"):
            return NAN
        if _as_complex_image(prediction).shape != _as_complex_image(target).shape:
            return NAN
        stats = _shared_statistics(prediction, target)
        return _ring_value(stats, self.kind, self.ring)


def _register(kind: str, ring_name: str, index: int) -> type:
    name = f"ring_{kind}_{ring_name}"
    cls = type(
        f"RingMetric_{kind}_{ring_name}",
        (_RingMetric,),
        {"ring": index, "kind": kind, "metric_name": name, "__doc__": _RingMetric.__doc__},
    )
    direction = "lower" if kind == "nrmse" else "higher"
    register_metric(name, requires_reference=True, needs=("pred_kspace",), direction=direction)(cls)
    return cls


#: ``{"ring_<kind>_<ring>": class}``; kinds are ``power_ratio``, ``nrmse`` and ``target_snr``.
RING_METRICS: dict[str, type] = {
    f"ring_{kind}_{ring_name}": _register(kind, ring_name, index)
    for kind in ("power_ratio", "nrmse", "target_snr")
    for ring_name, index in RINGS
}

__all__ = ["EDGES", "RINGS", "RING_METRICS", "noise_floor", "ring_statistics"]
