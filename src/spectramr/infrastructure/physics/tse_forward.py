"""Known-protocol TSE / IR-TSE signal: ideal T1 recovery times the EPG echo-train decay.

The echo amplitude comes from the one EPG owner (``epg.simulate_full_epg_fse``); only the
T1 recovery between excitations is added here, in its ideal spin-echo form, which
neglects recovery lost during the echo train (ETL * ESP << TR).
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.epg import simulate_full_epg_fse


def echo_index(te_ms: float, esp_ms: float, etl: int) -> int:
    """1-based echo whose time is closest to TE; raises when TE is outside the train."""
    k = round(te_ms / esp_ms)
    if not 1 <= k <= etl:
        raise ValueError(
            f"TE={te_ms} ms with echo spacing {esp_ms} ms puts the echo at {k}, outside 1..{etl}"
        )
    return k


def _recovery(t1: torch.Tensor, *, family: int, tr: float, ti: float) -> torch.Tensor:
    if family == 0:
        return 1 - torch.exp(-tr / t1)
    if family == 1:
        return (1 - 2 * torch.exp(-ti / t1) + torch.exp(-tr / t1)).abs()
    raise ValueError(f"family={family}; 0 (SE/TSE) and 1 (IR) are modelled")


def tse_signal(
    rho: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    *,
    family: int,
    tr: float,
    te: float,
    ti: float,
    esp: float,
    etl: int,
    refocus_deg: float,
    b1_plus: torch.Tensor | None = None,
) -> torch.Tensor:
    """``rho * recovery(T1) * |EPG echo at TE|`` for one series' protocol (times in ms).

    ``b1_plus`` (relative transmit field, 1 = nominal) scales the excitation and the
    refocusing angles of the echo train; ``None`` is a uniform nominal field.
    """
    recovery = _recovery(t1, family=family, tr=tr, ti=ti)
    k = echo_index(te, esp, etl)
    train, *_ = simulate_full_epg_fse(
        torch.ones_like(rho),
        t1,
        t2,
        torch.ones_like(rho) if b1_plus is None else b1_plus,
        excitation_fa=90.0,
        refocusing_fa=refocus_deg,
        echo_train_length=etl,
        te_spacing=esp,
    )
    return rho * recovery * train[:, k - 1 : k].abs()
