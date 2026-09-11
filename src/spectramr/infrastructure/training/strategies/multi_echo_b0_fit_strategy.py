"""Multi-echo B0 field-fit strategy (audit 2026-07 I1).

Physics-in-the-loop B0 estimation. The generator refines the noisy complex
multi-echo input; the **differentiable** analytic fit
``infrastructure.physics.b0_mapping`` (conjugate-product phase difference +
Tikhonov CG — the SSOT) recovers ΔB0 in Hz from the first two refined echoes;
and the field is graded against a reference ``b0_map`` with ``B0FieldRMSE``.

Contrast with :class:`~spectramr.infrastructure.training.strategies.b0_mapping_strategy.B0MappingStrategy`,
which despite its name is a deformable image registration and never produces a Hz
field. Here the physics fit is a genuine layer in the training loop (gradient
flows through the phase-difference + Tikhonov solve back to the generator) and
the headline metric measures the claimed physical quantity (Hz), closing the
metric↔claim gap (pitfall #18) and the facade (pitfall #16).

Batch contract:
    - model input: complex multi-echo images as ``[B, 2E, H, W]`` (real/imag
      interleaved per echo) or a complex ``[B, E, H, W]`` tensor, E ≥ 2.
    - ``batch['b0_map']``: reference off-resonance field, Hz, ``[B, 1, H, W]``
      (or a real ``target_batch``). Required — the fit has nothing to be graded
      against otherwise, so its absence fails loud.
    - ``batch['target_echoes']`` (optional): clean complex echoes for the
      echo-consistency term (only when ``lambda_echo > 0``).
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.config.schemas.enums import Regime, Task
from spectramr.core.metrics.b0_field_rmse import B0FieldRMSE
from spectramr.data.batch_types import read_batch_field
from spectramr.infrastructure.physics.b0_mapping import (
    estimate_b0_tikhonov,
    phase_difference,
)
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.capabilities import StrategyCapabilities


class MultiEchoB0FitStrategy(BaseTrainingStrategy):
    """Physics-in-the-loop multi-echo B0 field fit (the real B0 estimator)."""

    #: Quantitative MRI: the differentiable b0_mapping fit (phase difference +
    #: Tikhonov CG) is a real layer in the training loop and yields the claimed
    #: physical quantity, dB0 in Hz, graded by B0FieldRMSE. Requires E >= 2 echoes
    #: (Axis.ECHO -- exactly what the quantitative profile requires) and raises
    #: otherwise. Contrast B0MappingStrategy, which is deliberately UNTAGGED: its
    #: own docstring records that it never produces a Hz field.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.QUANTITATIVE}),
        tasks=frozenset({Task.PARAMETER_MAPPING}),
    )

    def _setup_strategy_specific_components(self) -> None:
        cfg = getattr(self.config.training, "multi_echo_b0_fit", None)
        if cfg is None:
            raise ValueError(
                "MultiEchoB0FitStrategy requires training.multi_echo_b0_fit "
                "(at least t_shift, the readout time-shift ΔTE in seconds)."
            )
        self._t_shift = float(cfg.t_shift)
        self._gamma = float(cfg.gamma)
        self._max_cg = int(cfg.max_cg_iterations)
        self._lambda_field = float(cfg.lambda_field)
        self._lambda_echo = float(cfg.lambda_echo)
        self._b0_metric = B0FieldRMSE()

    @staticmethod
    def _as_complex_echoes(refined: torch.Tensor) -> torch.Tensor:
        """``[B, 2E, H, W]`` real/imag-interleaved (or ``[B, E, H, W]`` complex)
        -> ``[B, E, H, W]`` complex, with E ≥ 2 enforced."""
        if torch.is_complex(refined):
            echoes = refined
        else:
            if refined.dim() != 4 or refined.shape[1] % 2 != 0:
                raise ValueError(
                    "MultiEchoB0FitStrategy expects real echoes as [B, 2E, H, W] "
                    "(real/imag interleaved per echo) or a complex [B, E, H, W]; "
                    f"got shape {tuple(refined.shape)}."
                )
            b, c, h, w = refined.shape
            r = refined.reshape(b, c // 2, 2, h, w)
            echoes = torch.complex(r[:, :, 0], r[:, :, 1])
        if echoes.shape[1] < 2:
            raise ValueError(
                "MultiEchoB0FitStrategy needs >= 2 echoes for the phase-difference "
                f"fit; got {echoes.shape[1]}."
            )
        return echoes

    def _fit_b0(self, echoes: torch.Tensor) -> torch.Tensor:
        """Analytic ΔB0 (Hz) from the first two echoes.

        ``map_b0`` (phase_difference + estimate_b0_tikhonov) is written per image
        (global median + squeeze), so we loop over the batch. The CG solve is
        differentiable, so gradients flow back to the echoes.
        """
        e0 = echoes[:, 0:1]
        e1 = echoes[:, 1:2]
        maps = []
        for b in range(echoes.shape[0]):
            phi = phase_difference(e0[b : b + 1], e1[b : b + 1])
            maps.append(
                estimate_b0_tikhonov(
                    phi,
                    self._t_shift,
                    gamma=self._gamma,
                    max_cg_iterations=self._max_cg,
                )
            )
        return torch.cat(maps, dim=0)  # [B, 1, H, W] real Hz

    @staticmethod
    def _resolve_reference(batch: dict, target_batch: Any) -> torch.Tensor | None:
        ref = read_batch_field(batch, "b0_map")
        if isinstance(ref, torch.Tensor):
            return ref
        # A real target_batch may carry the Hz reference directly.
        if isinstance(target_batch, torch.Tensor) and not torch.is_complex(target_batch):
            return target_batch
        return None

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        batch = kwargs.get("batch") if isinstance(kwargs.get("batch"), dict) else {}
        refined = self.generator_model(input_batch)
        echoes = self._as_complex_echoes(refined)
        b0_pred = self._fit_b0(echoes)  # [B, 1, H, W] Hz, grad-carrying

        b0_ref = self._resolve_reference(batch, target_batch)
        if b0_ref is None:
            raise ValueError(
                "MultiEchoB0FitStrategy requires a reference B0 field (Hz) as "
                "batch['b0_map'] (or a real target_batch); none was provided, so "
                "the field fit has nothing to be graded against."
            )
        field_loss = F.mse_loss(b0_pred, b0_ref.float())
        total = self._lambda_field * field_loss
        out: dict[str, torch.Tensor] = {"loss_b0_field_mse": field_loss.detach()}

        if self._lambda_echo > 0.0:
            tgt = read_batch_field(batch, "target_echoes")
            if not isinstance(tgt, torch.Tensor):
                raise ValueError(
                    "MultiEchoB0FitStrategy: lambda_echo>0 requires "
                    "batch['target_echoes'] (clean complex echoes) for the "
                    "echo-consistency term."
                )
            echo_loss = (echoes - self._as_complex_echoes(tgt)).abs().mean()
            total = total + self._lambda_echo * echo_loss
            out["loss_echo"] = (self._lambda_echo * echo_loss).detach()

        out["g_total_loss"] = total
        return out

    @torch.no_grad()
    def validation_step(
        self, input_batch: torch.Tensor, target_batch: Any = None, **kwargs: Any
    ) -> dict[str, float]:
        batch = kwargs.get("batch") if isinstance(kwargs.get("batch"), dict) else {}
        refined = self.generator_model(input_batch)
        echoes = self._as_complex_echoes(refined)
        b0_pred = self._fit_b0(echoes)
        b0_ref = self._resolve_reference(batch, target_batch)
        metrics: dict[str, float] = {}
        if isinstance(b0_ref, torch.Tensor):
            # B0FieldRMSE refuses to grade a complex tensor as a field (returns
            # NaN), so b0_pred (real Hz) is the only thing it will score — the
            # metric measures exactly the claimed physical quantity.
            rmse = float(self._b0_metric(b0_pred, b0_ref.float()))
            metrics["val_b0_field_rmse"] = rmse
            metrics["val_loss"] = rmse
        return metrics
