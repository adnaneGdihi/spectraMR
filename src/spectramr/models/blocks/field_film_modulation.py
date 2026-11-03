r"""Field-aware FiLM modulation block (XField-FM / idea 2).

Implements a small MLP that consumes the static-field strength
:math:`B_0` plus a sequence-conditioning vector :math:`\tau` and emits
per-channel :math:`(\gamma, \beta)` modulation parameters. These can
then multiply / shift any backbone activation in the FiLM style:

.. math::

    h' = \gamma(B_0, \tau) \odot h + \beta(B_0, \tau).

The block is the workhorse primitive of the cross-field foundation
model — it lets a single backbone produce cross-field-aware
predictions without re-training per scanner. See
``TODO/integration_plan_ulf_cheap_fast_mri.md`` §2.2.

Reference
---------
Perez et al. *AAAI* 2018 — FiLM: Visual Reasoning with a General
Conditioning Layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.blocks.hypernet_api import Hypernet, register_hypernet


class FieldFiLMBlock(nn.Module):
    r"""Generate per-channel FiLM ``(γ, β)`` from a field-strength + sequence vector.

    Args:
        num_channels: Number of feature channels in the host activation
            (output ``γ, β`` have the same channel count).
        sequence_dim: Length of the sequence-conditioning vector
            :math:`\tau` (e.g. one-hot of (T1, T2, FLAIR) plus
            normalised TR/TE/TI). 0 conditions on the field alone; the
            caller then passes an empty ``[B, 0]`` vector.
        hidden_dim: Width of the conditioning MLP.

    The block initialises the final layer with zero weights so the
    block starts as the identity (``γ = 1``, ``β = 0``). This
    preserves the host backbone's behaviour when XField-FM is added
    to an existing recipe (no warm-up shock).

    Inputs
    ------
    field_strength_T : Tensor of shape ``[B]`` or ``[B, 1]`` — the
        static-field strength in Tesla per batch element.
    sequence_features : Tensor of shape ``[B, sequence_dim]``.

    Outputs
    -------
    gamma, beta : Tensors of shape ``[B, num_channels]``. The host
        block applies them as ``γ_b * h_b + β_b`` where the
        modulation is broadcast across the spatial dims.
    """

    def __init__(
        self,
        num_channels: int,
        sequence_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be ≥ 1; got {num_channels}")
        if sequence_dim < 0:
            raise ValueError(f"sequence_dim must be ≥ 0; got {sequence_dim}")

        self.num_channels = num_channels
        self.sequence_dim = sequence_dim
        # Input: B0 (1D) + sequence vector  →  hidden  →  2·C  (γ, β stacked)
        self.mlp = nn.Sequential(
            nn.Linear(1 + sequence_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * num_channels),
        )

        # Identity initialisation: zero-init the final layer so γ=1, β=0
        # at start.  We add 1.0 to γ in forward to recover identity.
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        field_strength_T: torch.Tensor,
        sequence_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce ``(γ, β)`` of shape ``[B, num_channels]``.

        Raises:
            ValueError: when batch sizes disagree or shapes are wrong.
        """
        if field_strength_T.dim() == 1:
            field_strength_T = field_strength_T.unsqueeze(-1)
        if field_strength_T.dim() != 2 or field_strength_T.shape[-1] != 1:
            raise ValueError(
                f"field_strength_T must have shape [B] or [B, 1]; "
                f"got {tuple(field_strength_T.shape)}"
            )
        if sequence_features.dim() != 2:
            raise ValueError(
                f"sequence_features must be 2-D [B, sequence_dim]; "
                f"got {tuple(sequence_features.shape)}"
            )
        if sequence_features.shape[-1] != self.sequence_dim:
            raise ValueError(
                f"sequence_dim mismatch: expected {self.sequence_dim}, "
                f"got {sequence_features.shape[-1]}"
            )
        if field_strength_T.shape[0] != sequence_features.shape[0]:
            raise ValueError("Batch-size mismatch between field_strength_T and sequence_features.")

        x = torch.cat([field_strength_T, sequence_features], dim=-1)
        gamma_beta = self.mlp(x)  # [B, 2C]
        gamma_offset, beta = gamma_beta.chunk(2, dim=-1)
        # Identity init means gamma_offset=0; we shift to 1 so γ starts at 1.
        gamma = 1.0 + gamma_offset
        return gamma, beta

    def apply_to(
        self,
        activation: torch.Tensor,
        field_strength_T: torch.Tensor,
        sequence_features: torch.Tensor,
    ) -> torch.Tensor:
        r"""Apply the modulation: ``γ ⊙ h + β`` (FiLM).

        Args:
            activation: ``[B, C, *spatial]`` backbone activation.
            field_strength_T, sequence_features: see :py:meth:`forward`.

        Returns:
            Same-shape tensor.
        """
        gamma, beta = self.forward(field_strength_T, sequence_features)
        # Broadcast (B, C) → (B, C, 1, 1, …) over the spatial dims.
        view = (activation.shape[0], activation.shape[1]) + (1,) * (activation.dim() - 2)
        return activation * gamma.view(*view) + beta.view(*view)


@register_hypernet("field_film")
class FieldFiLMHypernet(Hypernet):
    r"""Adapter that exposes :class:`FieldFiLMBlock` through the :class:`Hypernet` API.

    The wrapped block consumes ``(B0, sequence_features)`` and emits
    per-channel ``(γ, β)`` FiLM parameters. This adapter:

    - presents the canonical ``forward(context) -> dict[str, Tensor]``
      contract so the block is usable through ``get_hypernet`` and
      :class:`HypernetWrappedModule`;
    - encodes the context as ``[B0, *sequence_features]`` — the leading
      column is the field strength in Tesla, the remainder is the
      sequence-conditioning vector :math:`\tau`.

    Args:
        context_dim: Total context width. Must be ``1 + sequence_dim``
            (the leading dimension is reserved for :math:`B_0`).
        param_shapes: Must be exactly ``{"gamma": (C,), "beta": (C,)}``
            where ``C`` is the number of FiLM channels. Any other key
            set raises ``ValueError``.
        embedding_dim: Width of the inner FiLM MLP (``hidden_dim`` in
            :class:`FieldFiLMBlock`).
    """

    def __init__(
        self,
        context_dim: int = 2,
        param_shapes: dict[str, tuple[int, ...]] | None = None,
        *,
        embedding_dim: int = 64,
    ) -> None:
        if param_shapes is None:
            param_shapes = {"gamma": (1,), "beta": (1,)}
        if set(param_shapes) != {"gamma", "beta"}:
            raise ValueError(
                "FieldFiLMHypernet requires param_shapes={'gamma': (C,), "
                f"'beta': (C,)}}; got keys {sorted(param_shapes)}"
            )
        gamma_shape = param_shapes["gamma"]
        beta_shape = param_shapes["beta"]
        if gamma_shape != beta_shape or len(gamma_shape) != 1:
            raise ValueError(
                "FieldFiLMHypernet requires gamma and beta to share a "
                f"1-D shape (C,); got gamma={gamma_shape}, beta={beta_shape}"
            )
        if context_dim < 2:
            raise ValueError(
                "FieldFiLMHypernet requires context_dim >= 2 "
                "(1 for B0 + at least 1 sequence dim); "
                f"got context_dim={context_dim}"
            )

        # nn.Module init must precede attribute assignments.
        nn.Module.__init__(self)
        self.context_dim = context_dim
        self.param_shapes = dict(param_shapes)
        self.embedding_dim = embedding_dim

        num_channels = int(gamma_shape[0])
        sequence_dim = context_dim - 1
        self.block = FieldFiLMBlock(
            num_channels=num_channels,
            sequence_dim=sequence_dim,
            hidden_dim=embedding_dim,
        )

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split ``context`` into ``(B0, τ)`` and return ``{"gamma", "beta"}``."""
        if context.dim() != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(
                "FieldFiLMHypernet expects context of shape "
                f"[B, {self.context_dim}]; got {tuple(context.shape)}"
            )
        field_strength_T = context[:, :1]
        sequence_features = context[:, 1:]
        gamma, beta = self.block(field_strength_T, sequence_features)
        return {"gamma": gamma, "beta": beta}


__all__ = ["FieldFiLMBlock", "FieldFiLMHypernet"]
