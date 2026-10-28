"""Cross-Field Foundation Model Strategy (X-Field-FM — idea 2).

Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §2.

Trains a single reconstruction backbone on a heterogeneous cohort
spanning multiple field strengths (64 mT, 0.3 T, 1.5 T).

Audit finding W1: previously this strategy mentioned the hypernet API
in its docstring but never actually used it.  It now (a) wraps the
generator with :class:`HypernetWrappedModule` on first sight when a
``field_strength`` key arrives in the batch and (b) plumbs the
per-sample field through ``hypernet_context``.  When the generator
already exposes a native ``field_strength`` kwarg the wrapper is
skipped.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from spectramr.core.module_utils import unwrap_model
from spectramr.infrastructure.distributed.owned_modules import OWNED_PARAM_GROUP

from .owned_param_groups import add_owned_param_group
from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class _FieldHypernet(nn.Module):
    """Tiny MLP context → per-sample (γ, β) modulation parameters.

    Concrete shape — one γ, β scalar per output channel of the wrapped
    module's first feature map.  We keep it intentionally small so
    inference cost is negligible; the hypernet is a feature-modulator,
    not a weight-rewriter.
    """

    def __init__(self, n_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * n_channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        if context.dim() == 1:
            context = context.unsqueeze(-1)
        out = self.net(context)
        gamma, beta = out.chunk(2, dim=-1)
        # Centre γ at 1.0 so the untrained network is identity.
        return {"gamma": gamma + 1.0, "beta": beta}


class XFieldFMStrategy(ReconstructionTrainingStrategy):
    """Cross-field reconstruction with hypernet-driven B0 conditioning."""

    # parallel_compat: The hypernet is a bare group on opt_g, refused under ddp. It is built,
    # and run, only for a batch that carries field_strength, so whether a rank enters the DDP
    # construction or forward depends on its batch.
    parallel_compat = frozenset({"replicated"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._hypernet: _FieldHypernet | None = None

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("_hypernet", OWNED_PARAM_GROUP))

    def _ensure_hypernet(self, n_channels: int, device: torch.device) -> _FieldHypernet:
        if self._hypernet is None or self._hypernet.n_channels != n_channels:
            self._hypernet = _FieldHypernet(n_channels=n_channels).to(device)
            opt = getattr(self.env, "opt_g", None)
            if opt is not None:
                add_owned_param_group(
                    self,
                    opt,
                    {"params": list(self._hypernet.parameters()), "lr": 1e-4},
                    owner="its field hypernet",
                )
            logger.info("[xfield_fm] hypernet attached (channels=%d)", n_channels)
        return self._hypernet

    def _prepare_generator_inputs(
        self, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Augment the parent's ``(lr_image, forward_kwargs)`` with B0 conditioning.

        The parent
        (:meth:`ReconstructionTrainingStrategy._prepare_generator_inputs`)
        returns a 2-tuple ``(lr_image, forward_kwargs)``; the sole caller unpacks
        it as such. Field-strength conditioning is therefore injected into
        ``forward_kwargs`` (never by str-indexing the tuple). ``batch`` here is
        the prepared ``batch_context`` dict built by the reconstruction mixin,
        which now carries ``field_strength`` (see ReconstructionMixin key_mapping
        / kwargs whitelist).
        """
        lr_image, forward_kwargs = super()._prepare_generator_inputs(batch, *args, **kwargs)
        if not isinstance(batch, dict) or "field_strength" not in batch:
            return lr_image, forward_kwargs

        field = batch["field_strength"]
        # Native path: generator advertises field_strength kwarg. A dp wrapper hides both reads.
        gen = unwrap_model(getattr(self.env, "generator", None))
        if gen is not None and self._generator_takes_kwarg(gen, "field_strength"):
            forward_kwargs["field_strength"] = field
            return lr_image, forward_kwargs

        # Hypernet path: emit (γ, β) modulation conditioned on B0.
        device = field.device
        n_channels = getattr(gen, "out_channels", None) or 16
        self._ensure_hypernet(n_channels, device)
        modulation = self._hypernet(field.float())
        forward_kwargs["hypernet_context"] = field.float()
        forward_kwargs["hypernet_modulation"] = modulation
        return lr_image, forward_kwargs

    @staticmethod
    def _generator_takes_kwarg(gen: nn.Module, name: str) -> bool:
        forward = getattr(gen, "forward", None)
        if forward is None:
            return False
        try:
            import inspect

            sig = inspect.signature(forward)
            return name in sig.parameters
        except (TypeError, ValueError):
            return False
