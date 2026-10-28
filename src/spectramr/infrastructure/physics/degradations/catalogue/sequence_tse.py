"""tse_t2_blur: T2 decay across the echo train of a turbo spin echo, as a phase-encode filter.

Line ``n`` is acquired at echo ``e(n)`` of the train, ``TE_e = (e + 1) ESP``, and carries the signal
weight ``exp(-TE_e / T2)`` (Hennig, Nauerth & Friedburg, MRM 3:823, 1986; Mulkern et al., Magn Reson
Imaging 8:557, 1990 for the PSF). Segmented ordering assigns the lines, in acquisition order, to echo
``floor(rank * ETL / n_lines)``, so linear order puts the k-space centre mid-train (asymmetric weights:
edge enhancement on one side) and centric order puts it at the first echo (decay towards the edges:
blur). Weights are normalised to the k-space centre line, which keeps the contrast at TE_eff.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import shot_index
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    check_input,
    run_as_volume,
)
from spectramr.infrastructure.physics.degradations.catalogue.sequence_common import (
    apply_line_weights,
    check_positive,
    ordered_line_times,
)
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionGeometry

__all__ = [
    "DEFAULT_ESP_MS",
    "DEFAULT_ETL",
    "MAX_TRAIN_OVER_T2",
    "degrade_tse_t2_blur",
    "tse_t2_blur_volume",
    "tse_weights",
]

#: Declared defaults when neither a kwarg nor a TSE scanner profile gives them.
DEFAULT_ETL = 16
DEFAULT_ESP_MS = 10.0
#: ETL * ESP / T2 at theta = 1 is 1; the severity range (0 to 2) is 2 * theta.
MAX_TRAIN_OVER_T2 = 2.0


def tse_weights(echo: Tensor, etl: int, esp_ms: float, theta: float) -> Tensor:
    """``exp(-(TE_e - TE_centre) / T2_eff)`` for echo indices ``echo`` ``[D, H]``; ``T2_eff = ETL ESP / (2 theta)``."""
    te = (echo.to(torch.float64) + 1.0) * esp_ms
    t2_eff = etl * esp_ms / (MAX_TRAIN_OVER_T2 * theta)
    centre = te[echo.shape[0] // 2, echo.shape[1] // 2]
    return torch.exp(-(te - centre) / t2_eff).to(torch.float32)


def tse_t2_blur_volume(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    etl: int = DEFAULT_ETL,
    esp_ms: float = DEFAULT_ESP_MS,
    order: str | None = None,
) -> Tensor:
    check_input(x, geometry, "tse_t2_blur")
    if isinstance(etl, bool) or not isinstance(etl, int) or etl < 2:
        raise ValueError(
            f"etl must be an integer >= 2 (one echo has no train to decay in), got {etl!r}"
        )
    esp_ms = check_positive("esp_ms", esp_ms)
    d, h = x.shape[-3:-1]
    times, _ = ordered_line_times(geometry, d, h, order)
    if theta == 0:
        return x
    # A train longer than the lines leaves its last echoes unused: each line is then its own echo.
    echo = shot_index(times, min(etl, times.numel())).to(x.device)
    return apply_line_weights(x, tse_weights(echo, etl, esp_ms, theta), geometry.mode)


def degrade_tse_t2_blur(x: Tensor, theta: float, seed: int = 0, **kwargs) -> Tensor:
    """2-D slice ``(B, C, H, W)``: one TSE echo train over the phase-encode lines."""
    return run_as_volume(tse_t2_blur_volume, x, theta, seed, **kwargs)
