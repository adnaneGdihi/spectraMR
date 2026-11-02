"""Scout-Conditioned Adaptive Sampling Strategy (SCAS — idea 5).

Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §5.

A low-resolution scout of each sample (``batch["scout"]``, produced by the
``scout_acquisition`` transform) is fed to :class:`ScoutHypernet`, which emits one
mask logit per phase-encode column. The per-sample mask replaces LOUPE's shared
logits: training draws it through :func:`gumbel_sigmoid` (straight-through, so
the hypernet receives gradient from the reconstruction loss), validation keeps
the deterministic ``undersampling.adaptive.budget`` selection, and an
:func:`expected_density` penalty pulls the sampling rate toward
``undersampling.adaptive.target_rate``. The mask is applied in k-space by
LOUPE's :meth:`~LOUPEStrategy._apply_learned_mask` before the reconstruction
forward.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.optimization.relaxed_bernoulli import (
    expected_density,
    gumbel_sigmoid,
)

from .loupe_strategy import LOUPEStrategy
from .mixins.reconstruction import _as_aux_mapping
from .mixins.validation_context import flatten_like
from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)

_MISSING_SCOUT = (
    "scas requires batch['scout'] (the low-resolution scout view) to condition the "
    "acquisition hypernet, but the batch supplies {supplied}. Declare the producer:\n"
    "  data:\n"
    "    processing:\n"
    "      transforms:\n"
    "        - name: scout_acquisition\n"
    "          kwargs: {{scout_resolution: [32, 32], domain: image}}\n"
    "Without it this arm is plain LOUPE, not SCAS."
)


class ScoutHypernet(nn.Module):
    """Scout image ``(B, C, H, W)`` -> one mask logit per column ``(B, W)``.

    The encoder pools over ``H``, so the head sees one feature vector per
    phase-encode column ``W``: the axis LOUPE's mask indexes.
    """

    def __init__(self, in_channels: int = 1, hidden: int = 64, init_logit: float = 0.0) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, None)),  # collapse H
        )
        self.head = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)
        # Start at the target density, as LOUPE initialises its shared logits.
        nn.init.constant_(self.head.bias, init_logit)

    def forward(self, scout: torch.Tensor) -> torch.Tensor:
        f = self.encoder(scout).squeeze(-2)  # (B, hidden, W)
        return self.head(f).squeeze(1)  # (B, W)


class SCASStrategy(LOUPEStrategy):
    """Scout-conditioned, per-sample learnable mask via relaxed-Bernoulli sampling."""

    # parallel_compat: The scout hypernet is a bare group on opt_g, so DDP never reduces it.
    parallel_compat = frozenset()

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        # The hypernet replaces LOUPE's shared ``_mask_logits``, which SCAS never builds.
        return super(LOUPEStrategy, cls).owned_module_declarations(config)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self._adaptive_cfg.enabled:
            raise ValueError(
                "training_mode 'scas' learns its mask from undersampling.adaptive "
                "(target_rate, budget, annealing, density_penalty_lambda); set "
                "undersampling.adaptive.enabled: true."
            )
        self._hypernet: ScoutHypernet | None = None

    def _ensure_hypernet(
        self, in_channels: int, device: torch.device, hidden: int = 64
    ) -> ScoutHypernet:
        if self._hypernet is None:
            init = float(torch.logit(torch.tensor(self._adaptive_cfg.target_rate)))
            self._hypernet = ScoutHypernet(in_channels, hidden, init_logit=init).to(device)
            self._register_on_opt_g(self._hypernet)
            logger.info("[scas] scout-hypernet attached (in_channels=%d)", in_channels)
        return self._hypernet

    def _materialize_owned_state(self, state: dict[str, Any]) -> None:
        """Rebuild the saved hypernet at its saved width so resume can restore it."""
        super()._materialize_owned_state(state)
        saved = state.get("modules", {}).get("_hypernet")
        if saved is not None and self._hypernet is None:
            hidden, in_channels = saved["encoder.0.weight"].shape[:2]
            self._ensure_hypernet(int(in_channels), self.device, hidden=int(hidden))

    @staticmethod
    def _scout_from(raw_batch: Any, like: torch.Tensor) -> torch.Tensor:
        """The batch's scout as a real ``(B, C, H, W)`` magnitude on ``like``'s device."""
        raw = _as_aux_mapping(raw_batch) if raw_batch is not None else {}
        scout = raw.get("scout") if isinstance(raw, dict) else None
        if isinstance(scout, dict):
            scout = scout.get("data")
        if not isinstance(scout, torch.Tensor):
            supplied = sorted(raw) if isinstance(raw, dict) else type(raw_batch).__name__
            raise ValueError(_MISSING_SCOUT.format(supplied=supplied))
        # The loop's depth-into-batch flatten, then a hard shape check against the input.
        scout = flatten_like(scout, like, name="scout")
        return (scout.abs() if torch.is_complex(scout) else scout).float()

    def _per_sample_mask(self, scout: torch.Tensor, training: bool) -> tuple[torch.Tensor, ...]:
        """``(mask (B, 1, 1, W), logits (B, W))`` for this batch's scouts."""
        logits = self._ensure_hypernet(scout.shape[1], scout.device)(scout)
        if training:
            # Built before opt_g existed (e.g. a step-0 validation)? Register now; dedup'd.
            self._register_on_opt_g(self._hypernet)
            mask = gumbel_sigmoid(logits, tau=self._current_tau(), hard=True)
        else:
            mask = self._eval_mask(logits.detach())
        return mask[:, None, None, :], logits

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(input_batch, torch.Tensor):
            raise TypeError(f"scas expects a tensor input_batch, got {type(input_batch).__name__}")
        training = bool(self.generator_model.training)
        scout = self._scout_from(kwargs.get("batch"), input_batch)
        mask, logits = self._per_sample_mask(scout, training)
        masked = self._apply_learned_mask(input_batch, mask)
        # Skip LOUPE's body: it would apply its shared mask on top of this one.
        base = ReconstructionTrainingStrategy._compute_losses_impl(
            self, input_batch=masked, target_batch=target_batch, epoch=epoch, **kwargs
        )
        if not training:
            return base
        self._steps_seen += 1
        cfg = self._adaptive_cfg
        density_loss = cfg.density_penalty_lambda * (
            expected_density(logits) - cfg.target_rate
        ).pow(2)
        base["loss_scas_density"] = density_loss
        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                base[total_key] = base[total_key] + density_loss
                break
        return base

    def validation_batch_context(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch_data: Any,
    ) -> dict[str, Any] | None:
        """The bare validation context plus the batch's scout, for the metric pass and the preview."""
        if batch_data is None:
            return None
        return {
            "use_dc": False,
            "measured_kspace": None,
            "scout": self._scout_from(batch_data, input_batch),
        }

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        if kwargs.get("batch_context") is None:
            kwargs["batch_context"] = self.validation_batch_context(
                input_batch, target_batch, batch_data
            )
        return super().validation_step(input_batch, target_batch, batch_data=batch_data, **kwargs)

    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        batch_context: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Grade the deterministic per-sample mask the hypernet selects for each scout."""
        scout = (batch_context or {}).get("scout")
        if not isinstance(scout, torch.Tensor):
            raise ValueError(_MISSING_SCOUT.format(supplied=sorted(batch_context or {})))
        if scout.shape[0] != input_batch.shape[0]:
            # A chunked forward slices the input; the scout rows follow it.
            raise ValueError(
                f"scas: validation scout has {scout.shape[0]} rows for "
                f"{input_batch.shape[0]} input rows."
            )
        mask, _ = self._per_sample_mask(scout.to(input_batch.device), training=False)
        masked = self._apply_learned_mask(input_batch, mask)
        return ReconstructionTrainingStrategy._validation_forward(
            self, masked, batch_context, **kwargs
        )
