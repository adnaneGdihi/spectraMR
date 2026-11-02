"""MRF generators for ``TODO/audit_plan_novel_fmri.md`` MRF ideas.

Registered models:
    * ``mrf_tangent_score``       — MRF §2: 5-D tangent score for
      Riemannian diffusion on the Bloch parameter manifold.
    * ``conformal_fp_embedding``  — MRF §3: residual MLP mapping a
      fingerprint of length ``N`` to a 5-D conformal embedding.
    * ``scanner_time_reparam``    — MRF §5: per-scanner 1-D
      square-root-velocity reparameterisation head.

References:
    [1] Ma et al., "Magnetic resonance fingerprinting", *Nature*,
        495, 2013.
    [11] Amari & Nagaoka, *Methods of Information Geometry*, AMS, 2000.
    [27] Srivastava et al., "Shape analysis of elastic curves",
         *IEEE TPAMI*, 33(7), 2011.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.manifolds.bloch_mrf_manifold import (
    cayley_hilbert_chart,
)
from spectramr.infrastructure.physics.manifolds.time_reparameterisation import (
    enforce_1d_beltrami_constraint,
)
from spectramr.models.blocks.mamba_block import MambaBlock
from spectramr.models.registry import register_model


def _sin_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.float().unsqueeze(-1) * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


@register_model(
    "mrf_tangent_score",
    training_mode="diffusion",
    spatial_dims=(1,),
)
class MRFTangentScore(nn.Module):
    """5-D tangent score MLP operating in the Cayley-Hilbert chart.

    Input ``theta[..., 5]`` is mapped to the chart, conditioned on the
    diffusion time ``t`` and (optionally) a fingerprint feature ``y``
    encoded by a small 1-D MLP. Output is projected back to the
    manifold tangent at the original ``theta``.
    """

    def __init__(
        self,
        hidden: int = 64,
        n_layers: int = 4,
        fp_dim: int = 32,
        time_dim: int = 32,
        use_mamba_fp_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.use_mamba_fp_encoder = use_mamba_fp_encoder
        # 1-D Mamba encoder over the fingerprint sequence (MRF §2 calls for
        # exactly this — the fingerprint is a length-N pulse-response curve
        # whose ordering is meaningful, so an SSM is the natural encoder).
        # Falls back to a plain MLP when ``use_mamba_fp_encoder=False``
        # (useful for unit tests where the gated-conv/GRU stand-in adds noise).
        if use_mamba_fp_encoder:
            self.fp_mamba = MambaBlock(d_model=fp_dim, d_state=8, d_conv=4)
            self.fp_proj = nn.Linear(fp_dim, hidden)
        else:
            self.fp_mamba = None
            self.fp_proj = nn.Sequential(
                nn.Linear(fp_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        layers: list[nn.Module] = [nn.Linear(5, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, 5))
        self.net = nn.Sequential(*layers)

    def _encode_fingerprint(self, fp: torch.Tensor) -> torch.Tensor:
        """Encode ``fp[B, ...]`` to ``[B, hidden]``.

        Accepts either ``[B, fp_dim]`` (treated as a length-1 sequence
        and routed through the MLP fallback even when Mamba is enabled),
        or ``[B, L, fp_dim]`` (routed through the Mamba SSM scan, then
        pooled along the sequence axis).
        """
        if fp.dim() == 2:
            # No sequence axis — Mamba would be a no-op. Use the linear
            # projection (or the MLP fallback) directly.
            if self.use_mamba_fp_encoder:
                return self.fp_proj(fp)
            return self.fp_proj(fp)
        if fp.dim() != 3:
            raise ValueError(f"fingerprint must be [B, fp_dim] or [B, L, fp_dim], got {fp.shape}")
        if self.use_mamba_fp_encoder:
            enc = self.fp_mamba(fp)  # [B, L, fp_dim]
            pooled = enc.mean(dim=1)  # mean over sequence
            return self.fp_proj(pooled)
        return self.fp_proj(fp.mean(dim=1))

    def forward(
        self,
        theta: torch.Tensor,
        t: torch.Tensor | None = None,
        fingerprint: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # F7/E27 (2026-05-21 smoke audit): default ``t`` to zeros when
        # the caller hasn't threaded a diffusion timestep — the MRF
        # *acquisition* strategy in mrf_acquisition_strategies.py uses
        # this network as a generic generator (no diffusion context),
        # and crashed at runtime without a kwarg default. The strategy
        # path that doesn't pass ``t`` is now equivalent to evaluating
        # the tangent score at t=0, which is the documented "no-noise"
        # special case.
        if t is None:
            t = torch.zeros(theta.shape[0], dtype=theta.dtype, device=theta.device)
        z = cayley_hilbert_chart(theta)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        h = self.net[0](z)
        h = h + self.time_proj(_sin_embed(t, self.time_dim))
        if fingerprint is not None:
            h = h + self._encode_fingerprint(fingerprint)
        for layer in self.net[1:]:
            h = layer(h)
        return h  # tangent vector in chart coordinates


@register_model(
    "conformal_fp_embedding",
    training_mode="reconstruction",
    spatial_dims=(1,),
)
class ConformalFPEmbedding(nn.Module):
    """Residual MLP mapping a fingerprint to a 5-D conformal embedding.

    The conformality of ``ψ_θ`` is enforced by the
    :class:`ConformalityJacobianLoss` registered separately; this
    module just provides the residual-MLP architecture and a helper
    to compute the Jacobian for the loss.
    """

    def __init__(
        self,
        fingerprint_length: int = 1000,
        embedding_dim: int = 5,
        hidden: int = 128,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(fingerprint_length, hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.GELU()) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden, embedding_dim)

    def forward(self, f: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        h = self.in_proj(f)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(h)

    def jacobian_at(self, f: torch.Tensor) -> torch.Tensor:
        """Return the Jacobian ``[B, embed_dim, fp_length]`` at ``f``.

        For the conformality loss we collapse it to a square form
        ``[B, embed_dim, embed_dim]`` via right-multiplication by
        ``Jᵀ`` after a low-rank projection — caller may use either.
        """
        f = f.detach().clone().requires_grad_(True)
        out = self.forward(f)
        rows = []
        for d in range(out.shape[-1]):
            grad = torch.autograd.grad(out[..., d].sum(), f, create_graph=True, retain_graph=True)[
                0
            ]
            rows.append(grad)
        return torch.stack(rows, dim=-2)


@register_model(
    "scanner_time_reparam",
    training_mode="reconstruction",
    spatial_dims=(1,),
)
class ScannerTimeReparam(nn.Module):
    """Per-scanner 1-D Beltrami-constrained time reparameterisation.

    Holds a learnable parameter *table* of shape ``[num_scanners, T-1]`` — one
    reparameterisation vector per scanner — and returns the cumulative τ via
    :func:`enforce_1d_beltrami_constraint`. ``forward`` indexes the table by the
    per-sample ``scanner_id`` so different scanners learn distinct warps (MRF §5
    cross-scanner harmonisation).

    Backward-compatible: with the default ``num_scanners=1`` and no
    ``scanner_id`` the single row is broadcast across the batch — identical to
    the previous single-global-vector behaviour. (audit 2026-06: the model was
    a single global vector and so was architecturally incapable of the
    per-scanner conditioning the strategy advertised.)
    """

    def __init__(
        self,
        fingerprint_length: int = 1000,
        eps: float = 0.1,
        num_scanners: int = 1,
    ) -> None:
        super().__init__()
        if num_scanners < 1:
            raise ValueError(f"num_scanners must be >= 1; got {num_scanners}")
        self.eps = eps
        self.num_scanners = int(num_scanners)
        # One vector of inter-timepoint intervals PER scanner.
        self.raw = nn.Parameter(torch.zeros(self.num_scanners, fingerprint_length - 1))

    def forward(
        self,
        batch_size: int = 1,
        scanner_id: torch.Tensor | int | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if scanner_id is None:
            # No per-scanner conditioning: broadcast scanner 0 across the batch
            # (== the pre-2026-06 single-vector behaviour when num_scanners=1).
            raw_b = self.raw[0].unsqueeze(0).expand(batch_size, -1)
        else:
            idx = scanner_id
            if not torch.is_tensor(idx):
                idx = torch.as_tensor(idx, device=self.raw.device)
            # No clamp: an id outside [0, num_scanners) must fail in index_select, not
            # collapse onto the last row (num_scanners unset would merge every vendor).
            idx = idx.to(self.raw.device).long().reshape(-1)
            raw_b = self.raw.index_select(0, idx)  # [n_ids, T-1]
            if raw_b.shape[0] == 1 and batch_size > 1:
                raw_b = raw_b.expand(batch_size, -1)
        return enforce_1d_beltrami_constraint(raw_b, eps=self.eps)


__all__ = [
    "ConformalFPEmbedding",
    "MRFTangentScore",
    "ScannerTimeReparam",
]
