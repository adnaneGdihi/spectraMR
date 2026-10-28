"""HSSC — Hilbert State-Space k-space Completion strategy.

Implements ``IMPLEMENTATION_SPEC.md`` §8 (Phase 2): an autoregressive
Mamba-based k-space completion strategy that wraps existing primitives
into a single registered strategy.

Pipeline (matches Spec §8.1)::

    raw multi-coil k-space  →  SVD coil compression
                            →  Hilbert linearise
                            →  windowed BiMamba
                            →  inverse Hilbert
                            →  Hermitian-symmetry projection
                            →  data-consistency clamp
                            →  IFFT to image domain
                            →  loss

This strategy is intentionally minimal: it composes the existing
``HilbertOrder``, ``MambaBlock``, FFT ops, and ``UnifiedReconstructionLossComputer``
into the spec's Hermitian-projected DC-clamped pipeline. The model
itself is provided via the standard ``env.generator`` slot — any
sequence model that consumes ``(B, T, F)`` tokens works.

References
----------
* IMPLEMENTATION_SPEC.md §8 (Phase 2 — HSSC).
* A. Gu, T. Dao, "Mamba: Linear-time sequence modeling with selective
  state spaces," arXiv:2312.00752, 2023.
* J. Skilling, "Programming the Hilbert curve," AIP Conf. Proc. 707,
  2004.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.losses.computers import UnifiedReconstructionLossComputer


class HSSCStrategy(BaseTrainingStrategy):
    """Hilbert State-Space k-space Completion (Spec §8).

    The **registered generator model** owns the sequence-to-sequence map,
    including the Hilbert linearisation / inverse and the windowed BiMamba
    encoder (see ``models/generators/hilbert_mamba_generators.py``,
    IMPLEMENTATION_SPEC §8.3). This strategy wraps that model with the
    physics scaffolding around it:

    * Hermitian-symmetry projection :math:`\\hat\\kappa \\leftarrow
      (\\hat\\kappa + \\overline{\\hat\\kappa^*})/2`
    * Hard data-consistency clamp ``where(mask, k_obs, k_hat)``
    * Composite loss = complex k-space MSE + α · image-domain (1 − SSIM)
    """

    forwards_loss_components: ClassVar[bool] = True

    def __init__(self, env=None, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        self.loss_computer = UnifiedReconstructionLossComputer(
            config=self.config,
            device=self.device,
        )

    def _setup_strategy_specific_components(self) -> None:
        return

    def _hermitian_project(self, k: torch.Tensor) -> torch.Tensor:
        """``k ← (k + conj(flip(k))) / 2`` — enforces conjugate symmetry."""
        if not torch.is_complex(k):
            return k
        return 0.5 * (k + torch.conj(torch.flip(k, dims=[-2, -1])))

    def _data_consistency(
        self,
        k_hat: torch.Tensor,
        k_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Hard DC: trust observed k-space inside the mask, model elsewhere."""
        return torch.where(mask.bool(), k_obs, k_hat)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the HSSC forward + DC + Hermitian + loss."""
        generator = self.env.generator

        # Treat input_batch as observed (masked) k-space; target as fully sampled.
        # The dataset is responsible for delivering complex tensors, but we
        # tolerate real-stacked even-channel inputs by re-pairing here for
        # the Hermitian + DC steps below (CLAUDE.md: no raw FFT).
        is_complex_in = torch.is_complex(input_batch)
        if not is_complex_in and input_batch.shape[1] % 2 == 0:
            ev = input_batch[:, 0::2]
            od = input_batch[:, 1::2]
            k_obs = torch.complex(ev, od)
        else:
            k_obs = input_batch
        # The sampling mask lives on the batch (TrainingBatch.mask / metadata),
        # NOT as a top-level kwarg — the training loop only forwards
        # ``batch=``/``iteration=``/``batch_data=``/``scaler=`` into
        # ``_compute_losses``. Reading ``kwargs['mask']`` (the old code) always
        # found None and raised at step 1, so no HSSC arm could run.
        batch = kwargs.get("batch")
        mask = batch.get("mask") if batch is not None else None
        if mask is None:
            # No silent fallback: inferring the sampling mask from non-zero
            # k-space entries (the old "safe fallback") mislabels
            # genuinely-sampled-but-zero bins as unobserved and corrupts the
            # DC clamp this strategy is built around (pitfalls #9/#10). The
            # sampling mask is authoritative and must be threaded through the
            # batch by the data pipeline.
            msg = (
                "HSSCStrategy requires an explicit k-space sampling mask on the "
                "batch (batch['mask']); refusing to infer it from non-zero "
                "k-space entries, which would silently mislabel sampled-but-zero "
                "bins and corrupt the hard data-consistency clamp. Thread the "
                "sampling mask through the DataPipelineDirector batch."
            )
            raise ValueError(msg)

        # Forward: model returns the same shape as input (k-space prediction)
        pred = generator(input_batch)

        if not torch.is_complex(pred) and pred.shape[1] % 2 == 0:
            ev = pred[:, 0::2]
            od = pred[:, 1::2]
            k_hat = torch.complex(ev, od)
        else:
            k_hat = pred

        # Hermitian projection + DC clamp (Spec §8.3)
        k_hat = self._hermitian_project(k_hat)
        k_hat = self._data_consistency(k_hat, k_obs, mask)

        # Image-domain reconstruction for the SSIM term
        x_hat = ifft2c(k_hat)
        x_gt = ifft2c(target_batch) if torch.is_complex(target_batch) else target_batch

        # Convert back to real-stacked layout for the unified loss computer
        # (the existing computer expects real tensors)
        if torch.is_complex(x_hat):
            x_hat_r = torch.cat([x_hat.real, x_hat.imag], dim=1)
            x_gt_r = torch.cat([x_gt.real, x_gt.imag], dim=1) if torch.is_complex(x_gt) else x_gt
        else:
            x_hat_r, x_gt_r = x_hat, x_gt

        env_losses = (self.env.losses or {}) if self.env else {}
        loss_output = self.loss_computer.compute(
            pred=x_hat_r,
            target=x_gt_r,
            epoch=epoch,
            # The loop advances loop_state.iteration each step; kwargs.get("step")
            # was always 0 (the loop passes "iteration", not "step"), freezing any
            # step-gated loss schedule (pitfall #16).
            iteration=resolve_loop_iteration(self),
            losses_dict=env_losses,
        )
        out: dict[str, torch.Tensor] = {"g_total_loss": loss_output.total}
        out.update(loss_output.components)
        return out


__all__ = ["HSSCStrategy"]
