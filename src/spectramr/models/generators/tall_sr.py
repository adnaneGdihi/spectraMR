r"""TALL-SR — stand-alone super-resolution backbone over the TALL traversal.

Implements ``IMPLEMENTATION_SPEC.md`` §9.5 **task 2.3** (super-resolution,
the primary stand-alone evaluation for Phase 3): a *Hilbert-Mamba* SR
network in which the fixed Hilbert space-filling curve is replaced by the
content-adaptive, Gumbel-softmax-differentiable
:class:`spectramr.models.blocks.tall.TALL` traversal.

Pipeline (2-D, image → image)::

    x ──lift(1×1 conv)──▶ feat (B, dim, S, S)
        │
        ├─ TALL ─▶ seq (B, dim, N), perm (B, N)      # content-adaptive order
        │
        ├─ MambaBlock × n_layers (pre-norm residual) over the sequence
        │
        └─ scatter-unflatten(perm) ─▶ grid (B, dim, S, S)
              │
              └─ project(1×1 conv) + x (residual) ─▶ SR output

The TALL traversal feeds the Mamba state-space stack a content-adaptive
1-D ordering; because the (soft) linearised sequence is differentiable
through the policy, the super-resolution task loss back-propagates into
TALL's orientation policy — the traversal is *exercised*, not inert.

``TALL`` requires a power-of-two **square** spatial grid (``S = 2**order``);
this model enforces that contract in :meth:`forward` and raises on a
mismatch rather than silently resampling (CLAUDE.md pitfall #9).

References
----------
* IMPLEMENTATION_SPEC.md §9 (TALL — Texture-Adaptive Learnable L-system),
  §9.5 task 2.3 (super-resolution), §9.6 acceptance criteria.
* E. Jang, S. Gu, B. Poole, "Categorical reparameterization with
  Gumbel-softmax," ICLR 2017.
* A. Gu, T. Dao, "Mamba: Linear-time sequence modeling with selective
  state spaces," 2023.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from spectramr.models.blocks.mamba_block import MambaBlock
from spectramr.models.blocks.tall import TALL
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


@register_model(
    name="tall_sr",
    training_mode="reconstruction",
    # 2-D image-domain super-resolution backbone. The TALL traversal and
    # the per-pixel scatter-unflatten are intrinsically 2-D; the 3-D
    # variant (K=24 motifs) is a separate registration once it lands.
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class TALLSuperResolution(nn.Module):
    r"""Texture-Adaptive Learnable L-system super-resolution backbone.

    Args:
        in_channels:   Input image channels (1 for magnitude SR).
        out_channels:  Output image channels. When equal to
            ``in_channels`` the network learns a residual (``out + x``).
        order:         TALL recursion depth ``p``; the model accepts only
            ``S × S`` inputs with ``S = 2**order`` (e.g. ``order=7`` →
            128×128).
        dim:           Hidden width carried through lift → TALL → Mamba.
        n_layers:      Number of stacked :class:`MambaBlock` mixers.
        d_state:       Mamba state dimension.
        policy_hidden: Hidden width of TALL's orientation-policy MLP.
        tau_init / tau_min / anneal_steps: Gumbel-softmax temperature
            schedule handed straight to :class:`TALL`.
        canonical_init: Initialise the TALL policy to emit canonical
            Hilbert in expectation at step 0 (Spec §9.4).

    Notes:
        ``**_unused`` absorbs config keys the schema may pass that this
        model does not consume; every key it *does* read is named
        explicitly so the audit can verify wiring.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        order: int = 7,
        dim: int = 32,
        n_layers: int = 2,
        d_state: int = 16,
        policy_hidden: int = 64,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        anneal_steps: int = 10000,
        canonical_init: bool = True,
        **_unused: Any,
    ) -> None:
        super().__init__()
        if order < 1:
            raise ValueError(f"TALLSuperResolution: order must be >= 1, got {order}")
        if n_layers < 1:
            raise ValueError(f"TALLSuperResolution: n_layers must be >= 1, got {n_layers}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.order = int(order)
        self.side = 2**self.order
        self.dim = int(dim)
        self._residual = self.in_channels == self.out_channels

        self.lift = nn.Conv2d(self.in_channels, self.dim, kernel_size=1)
        # feature_dim = dim so TALL's policy sees the full lifted channel
        # vector with no pad/truncate inside ``TALL._cell_features``.
        self.tall = TALL(
            order=self.order,
            feature_dim=self.dim,
            policy_hidden=policy_hidden,
            tau_init=tau_init,
            tau_min=tau_min,
            anneal_steps=anneal_steps,
            canonical_init=canonical_init,
            n_dims=2,
        )
        self.norms = nn.ModuleList(nn.LayerNorm(self.dim) for _ in range(n_layers))
        self.mixers = nn.ModuleList(
            MambaBlock(d_model=self.dim, d_state=d_state) for _ in range(n_layers)
        )
        self.project = nn.Conv2d(self.dim, self.out_channels, kernel_size=1)

        # Stash the last traversal so a TALL-aware strategy (or a test) can
        # read the locality regulariser (Spec §9.3) without re-running TALL.
        self._last_permutation: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve ``x``.

        Args:
            x: ``(B, in_channels, S, S)`` with ``S = 2**order``.

        Returns:
            ``(B, out_channels, S, S)`` super-resolved image.
        """
        if x.dim() != 4:
            raise ValueError(
                f"TALLSuperResolution expects a 4-D (B, C, H, W) tensor, got shape {tuple(x.shape)}"
            )
        B, _C, H, W = x.shape
        if (self.side, self.side) != (H, W):
            raise ValueError(
                f"TALLSuperResolution(order={self.order}) requires a "
                f"{self.side}×{self.side} grid; got {H}×{W}. Set "
                f"data.patch_size to a power-of-two square or change `order`."
            )

        feat = self.lift(x)  # (B, dim, S, S)
        seq, perm = self.tall(feat)  # seq (B, dim, N), perm (B, N)
        self._last_permutation = perm

        h = seq.transpose(1, 2)  # (B, N, dim) — Mamba layout
        for norm, mixer in zip(self.norms, self.mixers, strict=True):
            h = h + mixer(norm(h))  # pre-norm residual SSM block

        # Scatter the processed sequence back to its original grid position.
        # perm[b, i] is the source flat-index that sequence-slot i came from,
        # so the inverse placement writes h[b, i] back to perm[b, i].
        n = self.side * self.side
        grid_flat = x.new_zeros(B, self.dim, n)
        grid_flat.scatter_(2, perm.unsqueeze(1).expand(-1, self.dim, -1), h.transpose(1, 2))
        grid = grid_flat.reshape(B, self.dim, self.side, self.side)

        out = self.project(grid)
        if self._residual:
            out = out + x
        return out

    @property
    def track_locality(self) -> bool:
        """Keep the TALL traversal graph for the next forward (see ``TALL``)."""
        return self.tall.track_locality

    @track_locality.setter
    def track_locality(self, value: bool) -> None:
        self.tall.track_locality = bool(value)

    def last_locality_loss(self) -> torch.Tensor:
        r"""Differentiable locality regulariser (Spec §9.3) of the latest forward.

        Mean step length between consecutive entries of the soft traversal,
        computed from the Gumbel-softmax choice so the gradient reaches the
        TALL policy. Consumed by ``ReconstructionTrainingStrategy`` when
        ``training.tall_sr.lambda_locality > 0``.

        Raises:
            RuntimeError: if called before any :meth:`forward`.
        """
        if self._last_permutation is None:
            raise RuntimeError(
                "last_locality_loss() called before forward(); no traversal has been computed yet."
            )
        return self.tall.last_locality_loss()


__all__ = ["TALLSuperResolution"]
