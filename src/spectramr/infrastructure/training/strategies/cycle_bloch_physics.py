"""Physics of the CycleBlochStrategy cycle: the tissue-map estimator and backward renders.

The backward builders take the ``training.cycle_bloch`` block and
``physics.field_strength`` and return the render (estimated tissue maps -> simulated
ULF image) plus the provenance record the strategy stamps into its snapshot.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.biophysics.bloch import BlochSimulator
from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.infrastructure.physics.tse_forward import _recovery as inversion_recovery
from spectramr.infrastructure.physics.tse_forward import echo_index, tse_signal
from spectramr.infrastructure.physics.ulf_forward_operator import (
    DifferentiableULFForwardOperator,
    power_law_dispersion,
)

#: Physiological bounds of the estimated maps, in the ms every backward model reads.
T1_RANGE_MS = (100.0, 4000.0)
T2_RANGE_MS = (10.0, 300.0)
ESTIMATOR_RANGES = {"t1_range_ms": T1_RANGE_MS, "t2_range_ms": T2_RANGE_MS}


class ParameterEstimator(nn.Module):
    """
    Lightweight network to estimate quantitative tissue maps from a generated
    High-Field image: PD in (0, 1), T1 in ``T1_RANGE_MS`` and T2 in ``T2_RANGE_MS``
    (milliseconds), plus a relative B1+ map in (0, 2) when ``with_b1`` is set.
    """

    def __init__(self, in_channels=1, with_b1: bool = False):
        """__init__.

        Args:
            in_channels (Any): Description.
            with_b1: also emit a fourth channel, the relative transmit field.
        """
        super().__init__()
        self.with_b1 = with_b1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 4 if with_b1 else 3, 1),  # PD, T1, T2[, B1+]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, 3 or 4, H, W]: (PD, T1 ms, T2 ms)[, B1+]."""
        out = self.net(x)
        s = torch.sigmoid(out[:, :3])
        (t1_lo, t1_hi), (t2_lo, t2_hi) = T1_RANGE_MS, T2_RANGE_MS
        # An unscaled O(1) head read as ms gave exp(-TE/T2) ~ 1e-9, a vanishing cycle term.
        maps = torch.cat(
            [
                s[:, 0:1],
                t1_lo + (t1_hi - t1_lo) * s[:, 1:2],
                t2_lo + (t2_hi - t2_lo) * s[:, 2:3],
            ],
            dim=1,
        )
        if not self.with_b1:
            return maps
        # 2 * sigmoid is 1 (nominal field) at a zero logit and bounded in (0, 2).
        return torch.cat([maps, 2.0 * torch.sigmoid(out[:, 3:4])], dim=1)


#: The historical backward render: spin echo at a fixed ULF sequence, no field dependence.
_TR_FIXED_MS = 500.0
_TE_FIXED_MS = 14.0

#: (PD, T1, T2, B1+ or None[, contrast_id]) [B, 1, H, W] each -> simulated ULF magnitude.
BackwardRender = Callable[..., torch.Tensor]
#: (training.cycle_bloch block, physics.field_strength) -> (render, provenance).
BackwardBuilder = Callable[[Any, float | None], tuple[BackwardRender, dict[str, Any]]]


def _bloch_fixed(cfg: Any, field_strength: float | None) -> tuple[BackwardRender, dict[str, Any]]:
    def render(
        pd: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, b1: Any, *_: Any
    ) -> torch.Tensor:
        return BlochSimulator.spin_echo(m0=pd, t1=t1, t2=t2, te=_TE_FIXED_MS, tr=_TR_FIXED_MS)

    return render, {"tr_ms": _TR_FIXED_MS, "te_ms": _TE_FIXED_MS}


def _scalar_sequence(cfg: Any) -> dict[str, Any]:
    """The scalar ``ulf_*`` fields as the per-contrast sequence view the renders read."""
    return {
        "tr_ms": cfg.ulf_tr_ms,
        "te_ms": cfg.ulf_te_ms,
        "ti_ms": None,
        "flip_deg": cfg.ulf_flip_deg,
        "echo_spacing_ms": cfg.echo_spacing_ms,
        "echo_train_length": cfg.echo_train_length,
        "refocus_deg": cfg.refocus_deg,
    }


def _ulf_forward_render(
    seq: dict[str, Any], cfg: Any, field_strength: float | None
) -> tuple[BackwardRender, dict[str, Any]]:
    if field_strength is None:
        raise ValueError(
            "training.cycle_bloch.backward_model=ulf_forward renders at "
            "physics.field_strength, which is not declared."
        )
    # Noise, EMI, GNL and Maxwell are off: the cycle target must be deterministic, and the
    # Cartesian Maxwell phase does not change the magnitude the cycle loss compares.
    op = DifferentiableULFForwardOperator(
        b0_target=float(field_strength),
        b0_source=float(cfg.tissue_field_tesla),
        tr=float(seq["tr_ms"]),
        te=float(seq["te_ms"]),
        alpha=math.radians(float(seq["flip_deg"])),
        enable_emi=False,
        enable_gnl=False,
        enable_maxwell=False,
        enable_noise=False,
    )
    provenance = {
        "tr_ms": op.tr,
        "te_ms": op.te,
        "flip_deg": float(seq["flip_deg"]),
        "b0_target_tesla": op.b0_target,
        "tissue_field_tesla": op.b0_source,
    }
    ti = seq["ti_ms"]
    if ti is None:

        def render(
            pd: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, b1: Any, *_: Any
        ) -> torch.Tensor:
            return ifft2c(op(pd, t1, t2, b1_plus=b1)).abs()

        return render, provenance

    def render_ir(
        pd: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, b1: Any, *_: Any
    ) -> torch.Tensor:
        # The operator has no inversion stage; with its other stages off it is |Mxy| after
        # dispersion, so the IR signal takes the same dispersion with an ideal inversion.
        t1_ulf, t2_ulf = power_law_dispersion(
            t1,
            t2,
            b0_source=op.b0_source,
            b0_target=op.b0_target,
            beta=op.beta,
            gamma_t2=op.gamma_t2,
        )
        alpha = op.alpha if b1 is None else op.alpha * b1
        recovery = inversion_recovery(t1_ulf.clamp(min=1e-3), family=1, tr=op.tr, ti=float(ti))
        return (
            pd
            * torch.sin(torch.as_tensor(alpha))
            * recovery
            * torch.exp(-op.te / t2_ulf.clamp(min=1e-3))
        )

    return render_ir, {**provenance, "ti_ms": float(ti)}


def _epg_tse_render(
    seq: dict[str, Any], cfg: Any, field_strength: float | None
) -> tuple[BackwardRender, dict[str, Any]]:
    sequence = {
        "tr": float(seq["tr_ms"]),
        "te": float(seq["te_ms"]),
        "esp": float(seq["echo_spacing_ms"]),
        "etl": int(seq["echo_train_length"]),
        "refocus_deg": float(seq["refocus_deg"]),
    }
    ti = seq["ti_ms"]
    k = echo_index(sequence["te"], sequence["esp"], sequence["etl"])  # TE must lie in the train
    # Echo k depends only on the first k refocusing pulses, so the train stops there.
    simulated = {**sequence, "etl": k, "family": 0 if ti is None else 1, "ti": float(ti or 0.0)}

    def render(
        pd: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, b1: Any, *_: Any
    ) -> torch.Tensor:
        return tse_signal(pd, t1, t2, b1_plus=b1, **simulated)

    provenance = {
        "tr_ms": sequence["tr"],
        "te_ms": sequence["te"],
        "echo_spacing_ms": sequence["esp"],
        "echo_train_length": sequence["etl"],
        "refocus_deg": sequence["refocus_deg"],
    }
    return render, provenance if ti is None else {**provenance, "ti_ms": float(ti)}


_SEQUENCE_RENDERS = {"ulf_forward": _ulf_forward_render, "epg_tse": _epg_tse_render}


def _ulf_forward(cfg: Any, field_strength: float | None) -> tuple[BackwardRender, dict[str, Any]]:
    return _ulf_forward_render(_scalar_sequence(cfg), cfg, field_strength)


def _epg_tse(cfg: Any, field_strength: float | None) -> tuple[BackwardRender, dict[str, Any]]:
    return _epg_tse_render(_scalar_sequence(cfg), cfg, field_strength)


BACKWARD_MODELS: dict[str, BackwardBuilder] = {
    "bloch_fixed": _bloch_fixed,
    "ulf_forward": _ulf_forward,
    "epg_tse": _epg_tse,
}


def build_backward(
    cfg: Any, field_strength: float | None, contrast_map: dict[str, int] | None
) -> tuple[BackwardRender, dict[str, Any]]:
    """The backward render; with ``cfg.sequences``, one render per contrast picked by ``contrast_id``."""
    backward_model = cfg.backward_model if cfg is not None else "bloch_fixed"
    if backward_model not in BACKWARD_MODELS:
        raise ValueError(
            f"training.cycle_bloch.backward_model={backward_model!r}; "
            f"supported: {sorted(BACKWARD_MODELS)}."
        )
    table = getattr(cfg, "sequences", None) if cfg is not None else None
    if table is None:
        return BACKWARD_MODELS[backward_model](cfg, field_strength)
    if not contrast_map or set(contrast_map) != set(table):
        raise ValueError(
            f"training.cycle_bloch.sequences names {sorted(table)} but "
            f"data.multi_contrast.contrast_map names {sorted(contrast_map or {})}; every "
            "contrast the loader stamps needs exactly one sequence (enable data.multi_contrast)."
        )
    entries = []
    provenance: dict[str, Any] = {}
    for name, seq in table.items():
        render, prov = _SEQUENCE_RENDERS[backward_model](seq.model_dump(), cfg, field_strength)
        entries.append((int(contrast_map[name]), render))
        provenance[name] = prov

    def render_per_contrast(
        pd: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, b1: Any, contrast_id: Any = None
    ) -> torch.Tensor:
        if contrast_id is None:
            raise ValueError(
                "training.cycle_bloch.sequences renders per contrast, but the batch carries "
                "no contrast_id (data.multi_contrast must stamp contrast_idx)."
            )
        # Every contrast is rendered for the whole batch: the ids arrive on the device, and
        # choosing which renders to run would need a host sync each step.
        cid = torch.as_tensor(contrast_id, device=pd.device).reshape(-1, 1, 1, 1)
        # NaN for an id the table does not name, so it poisons the cycle loss instead of passing.
        out = torch.full_like(pd, float("nan"))
        for idx, render in entries:
            out = torch.where(cid == idx, render(pd, t1, t2, b1), out)
        return out

    return render_per_contrast, {"sequences": provenance}
