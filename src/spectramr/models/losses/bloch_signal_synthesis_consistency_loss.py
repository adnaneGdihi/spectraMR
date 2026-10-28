r"""Bloch signal-synthesis consistency loss (idea 10 §10.2).

For one-shot multi-parameter mapping: given network-predicted
maps :math:`(T_1, T_2, \mathrm{PD})` and the per-contrast acquisition
parameters :math:`\{(\mathrm{TR}_i, \mathrm{TE}_i, \mathrm{FA}_i)\}`,
re-synthesise the per-contrast magnitude signal via the Bloch closed
form and penalise the deviation from the observed contrast images.

Mathematical formulation
------------------------
Spoiled gradient-recalled-echo (SPGR) signal model:

.. math::

    S_i(\mathbf r) = \mathrm{PD}(\mathbf r) \cdot
        \frac{\sin(\alpha_i)\,\bigl(1 - e^{-\mathrm{TR}_i / T_1(\mathbf r)}\bigr)}
             {1 - \cos(\alpha_i)\,e^{-\mathrm{TR}_i / T_1(\mathbf r)}} \,
        e^{-\mathrm{TE}_i / T_2(\mathbf r)}

The loss is

.. math::

    \mathcal{L}_{\mathrm{bloch}} = \lambda \,
        \frac{1}{N_c}\sum_i \lVert S_i^{\mathrm{pred}} - S_i^{\mathrm{obs}}\rVert_2^2.

Provides the *physics anchor* that lets the otherwise-underdetermined
multi-parameter regression generalise from very small training sets.

Reference
---------
Kendall et al. *CVPR* 2018 — multi-task uncertainty weighting (the
host strategy uses this loss as one of the objectives).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from spectramr.config.schemas.enums import Regime, Task
from spectramr.models.losses.registry import register_loss

_T1_ALIASES = ("t1", "t1_ms", "t1map", "t1_map")
_T2_ALIASES = ("t2", "t2_ms", "t2map", "t2_map")
_PD_ALIASES = ("pd", "m0", "rho", "proton_density", "pd_map")


def split_t1_t2_pd(
    maps: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract the ``(T1, T2, PD)`` tensors from a parameter-map dict.

    Callers of :class:`BlochSignalSynthesisConsistencyLoss` produce a dict of
    per-parameter heads whose keys vary by config (``t1``/``T1``/``t1_ms`` …).
    This resolves them by case-insensitive alias and **raises** (no silent
    fallback, pitfall #9) if any of the three required maps is absent, so a
    caller cannot accidentally feed the wrong tensor positionally.

    Args:
        maps: mapping of parameter name -> ``[B, 1, H, W]`` tensor.

    Returns:
        ``(t1_ms, t2_ms, pd)`` tensors.
    """
    lut = {str(k).lower(): v for k, v in maps.items()}

    def _get(aliases: Sequence[str], label: str) -> torch.Tensor:
        for name in aliases:
            if name in lut:
                return lut[name]
        raise KeyError(
            f"parameter-map dict is missing a {label} map; looked for "
            f"{aliases!r} in keys {sorted(lut)!r}."
        )

    return (
        _get(_T1_ALIASES, "T1"),
        _get(_T2_ALIASES, "T2"),
        _get(_PD_ALIASES, "PD"),
    )


@register_loss(
    name="bloch_signal_synthesis_consistency",
    aliases=["BlochSignalSynthesisConsistencyLoss", "bloch_synthesis_consistency"],
    domain="bloch_synthesis",
    workflows=frozenset({Regime.QUANTITATIVE}),
    tasks=frozenset({Task.PARAMETER_MAPPING}),
)
class BlochSignalSynthesisConsistencyLoss(nn.Module):
    r"""Resynthesise multi-contrast magnitudes from ``(T1, T2, PD)`` and penalise mismatch.

    Tagged ``mri_quantitative``: it drives predicted maps through the actual SPGR
    signal equation and grades the resynthesised contrast, so the objective
    contains relaxometry physics rather than a themed name over an L1. It hard-
    raises on missing ``TR_ms``/``TE_ms``/``FA_deg`` rather than assuming
    defaults, which is what makes the tag safe to rely on.

    Args:
        weight: Loss multiplier ``λ ≥ 0``.
        eps: Numerical floor on T1, T2 to keep the exponentials finite.

    Inputs to ``forward``:
        - ``t1_ms``: ``[B, 1, H, W]`` predicted T1 (ms, > 0).
        - ``t2_ms``: ``[B, 1, H, W]`` predicted T2 (ms, > 0).
        - ``pd``:    ``[B, 1, H, W]`` predicted proton density.
        - ``observed_contrasts``: ``[B, N_c, H, W]`` measured magnitudes.
        - ``acquisition_params``: list of length ``N_c`` of dicts with
          keys ``TR_ms``, ``TE_ms``, ``FA_deg``. The order must match
          the channel order of ``observed_contrasts``.
    """

    def __init__(self, weight: float = 1.0, eps: float = 1e-3) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError(f"weight must be ≥ 0; got {weight}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0; got {eps}")
        self.weight = float(weight)
        self.eps = float(eps)

    def forward(
        self,
        t1_ms: torch.Tensor,
        t2_ms: torch.Tensor,
        pd: torch.Tensor,
        observed_contrasts: torch.Tensor,
        acquisition_params: Sequence[Mapping[str, Any]],
        **_: object,
    ) -> torch.Tensor:
        if observed_contrasts.dim() != 4:
            raise ValueError(
                f"observed_contrasts must be 4-D [B, N_c, H, W]; "
                f"got {tuple(observed_contrasts.shape)}"
            )
        n_c = observed_contrasts.shape[1]
        if len(acquisition_params) != n_c:
            raise ValueError(
                f"acquisition_params has {len(acquisition_params)} entries "
                f"but observed_contrasts has {n_c} channels — must match."
            )
        for shape_name, t in (("t1_ms", t1_ms), ("t2_ms", t2_ms), ("pd", pd)):
            if t.dim() != 4 or t.shape[1] != 1:
                raise ValueError(f"{shape_name} must be [B, 1, H, W]; got {tuple(t.shape)}")

        t1_safe = t1_ms.clamp_min(self.eps)
        t2_safe = t2_ms.clamp_min(self.eps)

        synth_channels = []
        for params in acquisition_params:
            try:
                tr = float(params["TR_ms"])
                te = float(params["TE_ms"])
                fa_deg = float(params["FA_deg"])
            except KeyError as exc:
                raise ValueError(
                    f"acquisition_params entry missing key {exc.args[0]!r}; "
                    "expected TR_ms, TE_ms, FA_deg."
                ) from exc
            fa_rad = math.radians(fa_deg)
            sin_a = math.sin(fa_rad)
            cos_a = math.cos(fa_rad)

            e1 = torch.exp(-tr / t1_safe)
            spgr = (sin_a * (1.0 - e1)) / (1.0 - cos_a * e1).clamp_min(self.eps)
            t2_decay = torch.exp(-te / t2_safe)
            synth = pd * spgr * t2_decay  # [B, 1, H, W]
            synth_channels.append(synth)

        synth_stack = torch.cat(synth_channels, dim=1)  # [B, N_c, H, W]
        sq_err = (synth_stack - observed_contrasts) ** 2
        return self.weight * sq_err.mean()


__all__ = ["BlochSignalSynthesisConsistencyLoss", "split_t1_t2_pd"]
