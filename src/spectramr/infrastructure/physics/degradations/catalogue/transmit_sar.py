"""sar_limited_flip: a SAR-reduced TSE refocusing angle reweights the echo train.

Lowering the refocusing flip angle from 180 degrees towards 60 degrees (a modelling choice: the
range of variable-flip-angle 3-D TSE trains and SAR-limited high-field 2-D TSE) to meet the SAR
limit (Hennig, Concepts Magn Reson 3:125, 1991; Busse et al., MRM 55:1030, 2006) changes the
amplitude of echo ``n`` from the CPMG ``exp(-n ESP / T2)`` to the EPG value ``A_n(alpha)``
(stimulated-echo pathways included; ``epg.simulate_full_epg_fse`` is the one EPG owner). Each
k-space line is acquired at one echo of the train, so it is scaled by ``A_n(alpha) / A_n(180)``:
the identity at 180 degrees. The weights are not renormalised to the centre echo, so the axis
carries both the PSF change across the train and the signal-level change at TE_eff.
Lines map to echoes in linear order: the ky lines ranked by acquisition time (``line_times``) are
cut into ETL equal consecutive groups (``shot_index``), cyclically shifted so the k-space centre
line falls at the effective echo (the one nearest ``te_ms``, via ``tse_forward.echo_index``; the
middle of the train, ``ETL // 2``, when no TE is given). Every slice, and every kz partition of a
3-D acquisition (SPACE-like), is its own ky train.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import (
    line_times,
    shot_index,
)
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    run_as_volume,
    slice_geometry,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
)
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import ENCODING_DIMS
from spectramr.infrastructure.physics.epg import simulate_full_epg_fse
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc
from spectramr.infrastructure.physics.tse_forward import echo_index

__all__ = [
    "DEFAULT_ESP_MS",
    "DEFAULT_ETL",
    "DEFAULT_T1_MS",
    "DEFAULT_T2_MS",
    "MAX_REDUCTION_DEG",
    "degrade_sar_limited_flip",
    "echo_amplitudes",
    "echo_weights",
    "sar_limited_flip_volume",
]

#: Refocusing angle is ``180 - 120 theta`` degrees.
MAX_REDUCTION_DEG = 120.0
#: Declared defaults of a call without a profile train (T1, T2 are tissue-like, in ms).
DEFAULT_ETL = 16
DEFAULT_ESP_MS = 10.0
DEFAULT_T1_MS = 1000.0
DEFAULT_T2_MS = 80.0
_EXCITATION_DEG = 90.0


def echo_amplitudes(
    refocus_deg: float, etl: int, esp_ms: float, t1_ms: float, t2_ms: float
) -> Tensor:
    """``[ETL]`` EPG echo magnitudes of a 90-degree-excited CPMG train, unit equilibrium."""
    one = torch.ones(1, 1, 1, 1)
    train, *_ = simulate_full_epg_fse(
        one,
        t1_ms * one,
        t2_ms * one,
        one,
        excitation_fa=_EXCITATION_DEG,
        refocusing_fa=refocus_deg,
        echo_train_length=etl,
        te_spacing=esp_ms,
    )
    return train[0, :, 0, 0].abs()


def echo_weights(refocus_deg: float, etl: int, esp_ms: float, t1_ms: float, t2_ms: float) -> Tensor:
    """``[ETL]`` per-echo scale ``A_n(refocus_deg) / A_n(180)``."""
    ratio = echo_amplitudes(refocus_deg, etl, esp_ms, t1_ms, t2_ms)
    return ratio / echo_amplitudes(180.0, etl, esp_ms, t1_ms, t2_ms)


def _echo_of_line(h: int, etl: int, centre_echo: int) -> Tensor:
    """``[1, H]`` zero-based echo number of each ky line; the centre line is at ``centre_echo``."""
    times = line_times(slice_geometry(), 1, h, tr_s=1.0)
    # shot_index with one line per shot is the time rank; ranks are cut into ETL equal groups.
    group = shot_index(times, h) * etl // h
    return (group - group[0, h // 2] + (centre_echo - 1)) % etl


def sar_limited_flip_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    etl: int = DEFAULT_ETL,
    esp_ms: float = DEFAULT_ESP_MS,
    te_ms: float | None = None,
    t1_ms: float = DEFAULT_T1_MS,
    t2_ms: float = DEFAULT_T2_MS,
) -> Tensor:
    """Scale each ky line by ``A_n(alpha) / A_n(180)`` of its echo ``n``.

    The axis carries the PSF change across the train and the signal-level change at TE_eff (the
    weights are not renormalised to the centre echo: that variant is not monotone in theta on the
    sim2rank grid). The centre line sits at the echo nearest ``te_ms``, or the middle of the train.
    """
    check_input(x, geometry, "sar_limited_flip")
    if not (isinstance(etl, int) and etl >= 1):
        raise ValueError(f"etl must be a positive integer, got {etl!r}")
    if not (esp_ms > 0.0 and t1_ms > 0.0 and t2_ms > 0.0):
        raise ValueError(
            f"esp_ms, t1_ms and t2_ms must be positive, got {esp_ms}, {t1_ms}, {t2_ms}"
        )
    centre_echo = max(1, etl // 2) if te_ms is None else echo_index(te_ms, esp_ms, etl)
    if theta <= 0.0:
        return x
    w = echo_weights(180.0 - MAX_REDUCTION_DEG * theta, etl, esp_ms, t1_ms, t2_ms)
    w = w.to(x.device)
    echo = _echo_of_line(x.shape[-2], etl, centre_echo).to(x.device)
    dims = ENCODING_DIMS[geometry.mode]
    k = fftnc(_ensure_complex(x), dims=dims) * w[echo].view(1, 1, 1, -1, 1)
    return ifftnc(k, dims=dims)


def degrade_sar_limited_flip(
    x: Tensor, theta: float, seed: int = 0, *, voxel_mm=None, **kwargs: Any
) -> Tensor:
    """2-D ``sar_limited_flip`` on ``(B, C, H, W)``: one ky echo train."""
    return run_as_volume(sar_limited_flip_volume, x, theta, seed, voxel_mm, **kwargs)
