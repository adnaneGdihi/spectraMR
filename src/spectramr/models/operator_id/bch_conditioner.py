r"""BCH operator conditioner (Proposal 1).

The conditioner :math:`\psi` maps the contrast/field coordinate
:math:`u=(c,\beta)` to the parameters of the identified degradation
operator:

* per-mode severities :math:`s_\psi(u)\in\mathbb R^{K}_{\ge 0}` (softplus),
  scaling the fixed Lie-algebra generators
  (:mod:`spectramr.infrastructure.physics.degradation_generators`);
* the noise power spectral density :math:`\rho_\psi(u)` over k-space radial
  bins (softplus), the diagonal part of the structured covariance;
* the low-rank covariance factors via per-sample complex column gains
  :math:`g_\psi(u)\in\mathbb C^{r}` applied to shared learnable basis
  images, giving :math:`U_\psi(u)\in\mathbb C^{N\times r}`.

Contrast conditioning is FiLM-style: a learned contrast embedding plus the
scalar field coordinate feed an MLP trunk that modulates three heads. This
mirrors the contrast-embedding conditioning path used elsewhere in the
framework rather than reimplementing a bespoke mechanism.

The module is consumed by ``OperatorIdBCHTrainingStrategy``; it is a
*conditioner*, not an image-to-image network, so the generic Tier-2 probe
(which assumes image→image) does not apply — the operator-ID forward probe
lives in ``tests/physics/test_operator_id_physics.py``.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.structured_covariance import lowrank_from_basis
from spectramr.models.registry import register_model

__all__ = ["BCHOperatorConditioner"]


@register_model(
    name="bch_operator_conditioner",
    training_mode="operator_id",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=False,
    supports_contrast_conditioning=True,
)
class BCHOperatorConditioner(nn.Module):
    r"""Maps :math:`u=(c,\beta)` to operator severities and covariance parameters.

    Args:
        num_modes: Number of deterministic generators ``K`` (length of the
            deterministic subset of ``mode_dictionary``).
        num_contrasts: Cardinality of the contrast vocabulary ``c``.
        covariance_rank: Rank ``r`` of the low-rank covariance factor.
        rho_bins: Number of k-space radial PSD bins.
        channels, height, width: Image dims (``N = C*H*W``); used to size
            the shared low-rank basis images.
        embed_dim: Contrast-embedding width.
        hidden_dim: MLP trunk width.
    """

    # The conditioner is not an image-to-image net; suppress the generic
    # residual identity-collapse heuristic if the probe is ever run.
    synthetic_forward_probe_skip: ClassVar[set[str]] = {"identity_collapse"}

    def __init__(
        self,
        num_modes: int = 4,
        num_contrasts: int = 3,
        covariance_rank: int = 8,
        rho_bins: int = 16,
        channels: int = 1,
        height: int = 64,
        width: int = 64,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.num_contrasts = int(num_contrasts)
        self.covariance_rank = int(covariance_rank)
        self.rho_bins = int(rho_bins)
        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)

        self.contrast_embedding = nn.Embedding(self.num_contrasts, embed_dim)
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.severity_head = nn.Linear(hidden_dim, self.num_modes)
        self.rho_head = nn.Linear(hidden_dim, self.rho_bins)
        self.gain_real_head = nn.Linear(hidden_dim, self.covariance_rank)
        self.gain_imag_head = nn.Linear(hidden_dim, self.covariance_rank)

        # Shared low-rank basis images (real + imag parts), small init so the
        # low-rank covariance term starts as a gentle perturbation.
        r, c, h, w = self.covariance_rank, self.channels, self.height, self.width
        self.basis_real = nn.Parameter(0.01 * torch.randn(r, c, h, w))
        self.basis_imag = nn.Parameter(0.01 * torch.randn(r, c, h, w))

    # ── helpers ───────────────────────────────────────────────────────
    @property
    def basis(self) -> torch.Tensor:
        """Complex low-rank basis images ``[r, C, H, W]``."""
        return torch.complex(self.basis_real, self.basis_imag)

    def build_lowrank(self, gains: torch.Tensor) -> torch.Tensor:
        """Assemble the low-rank covariance factor ``U`` (``[B, N, r]``) from gains."""
        return lowrank_from_basis(self.basis.to(gains.dtype), gains)

    # ── forward ───────────────────────────────────────────────────────
    def forward(
        self,
        contrast_idx: torch.Tensor | None = None,
        field_coord: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        r"""Predict operator parameters from the conditioning coordinate.

        Args:
            contrast_idx: Long tensor ``[B]`` of contrast ids. A float /
                image tensor (e.g. from the generic probe) is treated as a
                batch-size hint and replaced with zeros.
            field_coord: Float tensor ``[B]`` of the field coordinate
                :math:`\beta` (e.g. ``log10(0.3)`` for 0.3 T). Defaults to
                zeros.

        Returns:
            Dict with keys ``severities`` ``[B, K]`` (≥ 0),
            ``rho_radial`` ``[B, rho_bins]`` (> 0), and ``gains``
            ``[B, r]`` (complex).
        """
        # Resolve batch size / coerce probe-style image inputs.
        if contrast_idx is None:
            batch = int(field_coord.shape[0]) if field_coord is not None else 1
            device = field_coord.device if field_coord is not None else self.basis_real.device
            contrast_idx = torch.zeros(batch, dtype=torch.long, device=device)
        elif contrast_idx.dtype not in (torch.long, torch.int32, torch.int64):
            # A float / multi-dim tensor: use only its batch dim.
            batch = int(contrast_idx.shape[0])
            contrast_idx = torch.zeros(batch, dtype=torch.long, device=contrast_idx.device)
        contrast_idx = contrast_idx.reshape(-1)
        batch = int(contrast_idx.shape[0])
        device = contrast_idx.device

        if field_coord is None:
            field_coord = torch.zeros(batch, device=device, dtype=self.basis_real.dtype)
        field_coord = field_coord.reshape(batch, 1).to(self.basis_real.dtype)

        emb = self.contrast_embedding(contrast_idx)  # [B, embed_dim]
        feats = torch.cat([emb, field_coord], dim=-1)
        h = self.trunk(feats)

        severities = F.softplus(self.severity_head(h))  # [B, K] ≥ 0
        rho_radial = F.softplus(self.rho_head(h)) + 1e-4  # [B, bins] > 0
        gains = torch.complex(self.gain_real_head(h), self.gain_imag_head(h))  # [B, r]

        return {
            "severities": severities,
            "rho_radial": rho_radial,
            "gains": gains,
        }
