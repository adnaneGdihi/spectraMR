"""Field-strength contrast change: re-evaluate the signal equation at a lower B0 from q-maps.

``out = x * S(B0_target) / S(B0_src)`` voxel-wise, with S the stage-1 signal of
:class:`DifferentiableULFForwardOperator` (power-law T1/T2 transport, Bottomley 1984, then the
Ernst equation, Ernst & Anderson 1966) run with PD = 1, so PD cancels and the ratio is pure
contrast. ``x`` is the image acquired at ``b0_T``: it supplies shape, device, dtype and the
absolute intensity scale. The result is rescaled per sample to x's mean intensity over its support, so only
tissue contrast changes. Theta moves the target field log-linearly from ``b0_T`` to 0.05 T.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry
from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.infrastructure.physics.ulf_forward_operator import (
    DifferentiableULFForwardOperator,
)

__all__ = [
    "FLOOR_FIELD_T",
    "SEQUENCE_FLIP",
    "SUPPORT_FRACTION",
    "degrade_field_contrast",
    "field_contrast_condition",
    "field_contrast_volume",
    "target_field",
]

#: Field at theta = 1.
FLOOR_FIELD_T = 0.05
#: The operator's documented defaults (ms, ms), used when a sequence time is not supplied.
DEFAULT_TR_MS = 500.0
DEFAULT_TE_MS = 14.0
#: Object support for the intensity match: |x| above this fraction of its per-sample maximum.
SUPPORT_FRACTION = 0.1
_QMAP_KEYS = frozenset({"t1", "t2", "pd"})
#: Sequences the Ernst equation describes -> the flip angle policy. Spin echo is the 90 deg
#: saturation-recovery limit; spoiled GRE reads fa_deg (operator default 90 deg).
SEQUENCE_FLIP: dict[str, bool] = {"spin_echo": False, "spoiled_gre": True}


def target_field(b0_src: float, theta: float) -> float:
    """B0_target = B0_src * (0.05 / B0_src) ** theta (log-linear)."""
    return b0_src * (FLOOR_FIELD_T / b0_src) ** theta


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"field_contrast: {name} must be a finite number, got {value!r}")
    if value <= 0:
        raise ValueError(f"field_contrast: {name} must be > 0, got {value!r}")
    return float(value)


def _alpha(sequence: str, fa_deg: float | None) -> float:
    if sequence not in SEQUENCE_FLIP:
        raise ValueError(
            f"field_contrast: sequence {sequence!r} is not described by the Ernst equation "
            f"of the ULF forward operator; supported: {sorted(SEQUENCE_FLIP)} (inversion, "
            "echo-train, bSSFP and EPI signals need their own equations)"
        )
    if fa_deg is None:
        return math.pi / 2
    fa = _positive("fa_deg", fa_deg)
    if fa > 180:
        raise ValueError(f"field_contrast: fa_deg must be <= 180, got {fa_deg!r}")
    if not SEQUENCE_FLIP[sequence] and fa != 90.0:
        raise ValueError(f"field_contrast: {sequence} is a 90 degree sequence, got fa_deg={fa}")
    return math.radians(fa)


def _maps(qmaps: Mapping[str, Tensor], x: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(qmaps, Mapping) or not {"t1", "t2"} <= set(qmaps):
        raise ValueError("field_contrast: qmaps must be a mapping with 't1' and 't2' in ms")
    if extra := set(qmaps) - _QMAP_KEYS:
        raise ValueError(f"field_contrast: unknown qmaps keys {sorted(extra)}")
    spatial = x.shape[2:]
    for key, m in qmaps.items():
        ok = (
            isinstance(m, Tensor)
            and m.dim() == x.dim()
            and m.shape[0] in (1, x.shape[0])
            and m.shape[1] in (1, x.shape[1])
            and m.shape[2:] == spatial
        )
        if not ok:
            raise ValueError(
                f"field_contrast: qmaps[{key!r}] must be (B|1, C|1, ...) with spatial shape "
                f"{tuple(spatial)}, got {getattr(m, 'shape', type(m))}"
            )
    return qmaps["t1"], qmaps["t2"]


def _signal(t1: Tensor, t2: Tensor, op: DifferentiableULFForwardOperator) -> Tensor:
    # The operator's own call path (as SimulateULFFromQMaps): k-space, then |ifft2c|.
    return ifft2c(op(torch.ones_like(t1), t1, t2, trajectory=None, gnl_coefficients=None)).abs()


def _fold(m: Tensor, batch: int) -> Tensor:
    """(B|1, C|1, [D,] H, W) -> (B[*D], C|1, H, W): slices become the operator's batch."""
    m = m.expand(batch, *m.shape[1:])
    return m if m.dim() == 4 else m.permute(0, 2, 1, 3, 4).reshape(-1, m.shape[1], *m.shape[-2:])


def _contrast_ratio(
    t1: Tensor,
    t2: Tensor,
    b0_src: float,
    b0_tgt: float,
    tr: float,
    te: float,
    alpha: float,
    beta: float,
    gamma_t2: float,
) -> Tensor:
    """S(b0_tgt) / S(b0_src) for 4-D maps (N, C, H, W); 1 where a map is not positive."""
    shape = (t1.shape[-2], t1.shape[-1])

    def op(b0_target: float) -> DifferentiableULFForwardOperator:
        return DifferentiableULFForwardOperator(
            b0_target=b0_target,
            b0_source=b0_src,
            beta=beta,
            gamma_t2=gamma_t2,
            tr=tr,
            te=te,
            alpha=alpha,
            image_shape=shape,
            enable_emi=False,
            enable_gnl=False,
            enable_maxwell=False,
            enable_noise=False,
        )

    s_src, s_tgt = _signal(t1, t2, op(b0_src)), _signal(t1, t2, op(b0_tgt))
    tissue = (t1 > 0) & (t2 > 0)
    return torch.where(tissue, s_tgt / s_src.clamp_min(torch.finfo(s_src.dtype).tiny), 1.0)


def _match_mean_intensity(x: Tensor, y: Tensor) -> Tensor:
    """Scale ``y`` per sample so its mean |y| over x's support equals that of |x| (device-side).

    The overall brightness is a separate (gain / SNR) axis; this keeps only the tissue contrast.
    """
    ax, ay = x.abs().flatten(1), y.abs().flatten(1)
    support = ax > SUPPORT_FRACTION * ax.amax(dim=1, keepdim=True)
    tiny = torch.finfo(ax.dtype).tiny
    scale = (ax * support).sum(1) / (ay * support).sum(1).clamp_min(tiny)
    return y * scale.view(-1, *([1] * (y.dim() - 1)))


def field_contrast_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry | None = None,
    *,
    qmaps: Mapping[str, Tensor],
    b0_T: float = 3.0,  # noqa: N803 -- tesla
    tr_ms: float | None = None,
    te_ms: float | None = None,
    ti_ms: float | None = None,
    fa_deg: float | None = None,
    sequence: str = "spoiled_gre",
    beta: float = 0.35,
    gamma_t2: float = 0.05,
) -> Tensor:
    """``x`` (B, C, [D,] H, W) re-weighted to the target field; any mode (voxel-wise, folded)."""
    b0_src = _positive("b0_T", b0_T)
    if b0_src <= FLOOR_FIELD_T:
        raise ValueError(f"field_contrast: b0_T must exceed {FLOOR_FIELD_T} T, got {b0_T!r}")
    if ti_ms is not None:
        raise ValueError("field_contrast: ti_ms is not read; the operator has no inversion signal")
    tr = DEFAULT_TR_MS if tr_ms is None else _positive("tr_ms", tr_ms)
    te = DEFAULT_TE_MS if te_ms is None else _positive("te_ms", te_ms)
    if te >= tr:
        raise ValueError(f"field_contrast: te_ms={te} must be < tr_ms={tr}")
    alpha = _alpha(sequence, fa_deg)
    beta, gamma_t2 = _positive("beta", beta), _positive("gamma_t2", gamma_t2)
    t1, t2 = _maps(qmaps, x)
    if theta == 0.0:
        return x  # the registry convention: the input tensor itself, not a copy
    dev, work = x.device, torch.promote_types(x.real.dtype, torch.float32)
    t1, t2 = (_fold(m.to(dev, work), x.shape[0]) for m in (t1, t2))
    ratio = _contrast_ratio(
        t1, t2, b0_src, target_field(b0_src, theta), tr, te, alpha, beta, gamma_t2
    )
    if x.dim() == 5:
        ratio = ratio.reshape(x.shape[0], x.shape[2], -1, *x.shape[-2:]).permute(0, 2, 1, 3, 4)
    return _match_mean_intensity(x, x * ratio.to(x.real.dtype))


def degrade_field_contrast(x: Tensor, theta: float, *, seed: int = 0, **kwargs: Any) -> Tensor:
    """The slice_2d kernel: ``x`` is (B, C, H, W)."""
    return field_contrast_volume(x, theta, seed, None, **kwargs)


def field_contrast_condition(profile: Any, theta: float, geometry: Any, shape: Any) -> dict:
    """The profile's field and sequence; only the sequence times it states (None = unknown)."""
    facts = {
        "b0_T": profile.b0_T,
        "sequence": str(profile.sequence),
        "tr_ms": profile.tr_ms,
        "te_ms": profile.te_ms,
        "ti_ms": profile.ti_ms,
        "fa_deg": profile.fa_deg,
    }
    return {k: v for k, v in facts.items() if v is not None}
