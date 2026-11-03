"""PMPS Phase-3 — paired-synthesis generator (frozen orchestrator).

Composes the frozen tissue prior (Phase 1), the Bloch forward
operator, and the frozen ULF + HF corruption operators (Phase 2) into a
single forward pass that emits a dict of
``{ulf, hf, ulf_clean, hf_clean, theta, residual*, defensibility_flag}``.

Inputs: a sampled tissue map ``theta`` (B, 3, H, W). Acquisition
vectors ``P_ULF`` and ``P_HF`` are passed via kwargs.

Outputs: dict with the synthesised pair plus metadata. The
``defensibility_flag`` is a Boolean tensor (B,) derived from the
noise-standardised residual of both images.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(
    name="paired_synthesis_generator",
    training_mode="paired_synthesis",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=False,
    supports_contrast_conditioning=True,
)
class PairedSynthesisGenerator(nn.Module, IGenerator):
    """Frozen Bloch + corruption-operator composition for synthesis.

    The Bloch step is intentionally simple — a SPGR-shaped magnitude
    map that depends on (T1, T2, M0, TE, TR, FA, B0). For full physics
    correctness at inference time the strategy may replace this with
    :class:`DifferentiableBlochLayer` (the encoder path) — that is a
    runtime-component swap, not a schema change.

    The defensibility flag reads both images. For each, the residual is
    ``mean((y - y_clean)^2) / (2 sigma^2)`` with ``sigma`` the operator's
    ``noise_sigma()``: about 1 in air and 0.5 on high-SNR tissue when the
    corruption is noise alone, larger as the field and motion add deviation.
    A pair passes when the larger of the two is below ``defensibility_threshold``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        ulf_corruption: nn.Module | None = None,
        hf_corruption: nn.Module | None = None,
        defensibility_threshold: float = 1.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.ulf_corruption = ulf_corruption
        self.hf_corruption = hf_corruption
        self.defensibility_threshold = defensibility_threshold
        # Frozen and empty: the harness builds an optimizer over parameters(), and torch refuses [].
        self.frozen_anchor = nn.Parameter(torch.zeros(0), requires_grad=False)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def name(self) -> str:
        return "paired_synthesis_generator"

    def get_parameter_count(self) -> int:
        # All sub-modules are frozen; report 0 trainable params.
        return 0

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self._out_channels, *input_shape[2:])

    def _bloch_spgr(
        self,
        theta: torch.Tensor,
        te: float,
        tr: float,
        flip_angle_rad: float,
    ) -> torch.Tensor:
        """Minimal SPGR Bloch magnitude on (T1, T2, M0) maps."""
        t1 = theta[:, 0:1].clamp_min(1e-6)
        t2 = theta[:, 1:2].clamp_min(1e-6)
        m0 = theta[:, 2:3]
        # Normalised maps → scale to physical units for the formula.
        # Convention: t1 in normalised [0,1] → physical 10-5000 ms.
        t1_ms = 10.0 + 4990.0 * t1
        t2_ms = 1.0 + 2499.0 * t2
        sin_a = torch.sin(torch.tensor(flip_angle_rad, device=theta.device, dtype=theta.dtype))
        cos_a = torch.cos(torch.tensor(flip_angle_rad, device=theta.device, dtype=theta.dtype))
        e1 = torch.exp(-tr / t1_ms)
        e2 = torch.exp(-te / t2_ms)
        denom = (1.0 - cos_a * e1).clamp_min(1e-6)
        return m0 * sin_a * (1.0 - e1) / denom * e2

    @staticmethod
    def _corrupt(
        op: nn.Module | None, clean: torch.Tensor, role: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply ``op`` and return the output with its noise-standardised residual (B,)."""
        if op is None:
            return clean, clean.new_zeros(clean.shape[0])
        noise_sigma = getattr(op, "noise_sigma", None)
        if not callable(noise_sigma):
            raise TypeError(
                f"PairedSynthesisGenerator: {role} must expose noise_sigma() so the "
                f"defensibility gate can standardise its residual; got {type(op).__name__}."
            )
        out = op(clean)
        sigma = noise_sigma().to(out).view(1, -1, 1, 1).expand(1, out.shape[1], 1, 1)
        residual = ((out - clean).pow(2) / (2.0 * sigma.pow(2))).mean(dim=(1, 2, 3))
        return out, residual

    def forward(
        self,
        x: torch.Tensor,
        *,
        p_ulf: dict[str, float] | None = None,
        p_hf: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] < 3:
            raise ValueError(
                f"PairedSynthesisGenerator expects (B, ≥3, H, W) tissue maps; got {tuple(x.shape)}."
            )
        if p_ulf is None or p_hf is None:
            raise ValueError(
                "PairedSynthesisGenerator needs both p_ulf and p_hf protocols "
                "({'te', 'tr', 'fa_rad'}); with one shared default the pair is identical."
            )
        ulf_clean = self._bloch_spgr(x, p_ulf["te"], p_ulf["tr"], p_ulf["fa_rad"])
        hf_clean = self._bloch_spgr(x, p_hf["te"], p_hf["tr"], p_hf["fa_rad"])
        ulf, residual_ulf = self._corrupt(self.ulf_corruption, ulf_clean, "ulf_corruption")
        hf, residual_hf = self._corrupt(self.hf_corruption, hf_clean, "hf_corruption")
        residual = torch.maximum(residual_ulf, residual_hf)
        flag = residual < self.defensibility_threshold
        return {
            "ulf": ulf,
            "hf": hf,
            "ulf_clean": ulf_clean,
            "hf_clean": hf_clean,
            "theta": x,
            "residual": residual,
            "residual_ulf": residual_ulf,
            "residual_hf": residual_hf,
            "defensibility_flag": flag,
        }

    def generate(self, z: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.forward(z, **kwargs)


__all__ = ["PairedSynthesisGenerator"]
