"""BALD — Bayesian Active Learning by Disagreement (PR-25 / Part-I G).

Closed-loop active acquisition. The strategy maintains a running mask
of acquired k-space lines, and every ``retrain_every`` gradient steps
uses a posterior-disagreement score to pick the next ``query_batch``
lines from the unsampled set.

Reference: Houlsby et al. "Bayesian Active Learning for Classification
and Preference Learning" (arXiv:1112.5745); Sanchez et al. "Scalable
learning-based sampling optimization" (2020).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from spectramr.config.schemas.enums import Regime, Task
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.capabilities import StrategyCapabilities

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


def posterior_samples(config: Any) -> int:
    """``model_kwargs.n_posterior_samples`` (default 8); BALD needs at least two."""
    model_kwargs = getattr(config.model, "model_kwargs", None) or {}
    n = int(model_kwargs.get("n_posterior_samples", 8))
    if n < 2:
        raise ValueError(f"model_kwargs.n_posterior_samples={n}: BALD compares at least 2 samples")
    return n


class BALDAcquisitionStrategy(ReconstructionTrainingStrategy):
    """Active-acquisition strategy with MC-dropout posterior."""

    #: Declared rather than inherited — see :class:`PILOTStrategy` for why.
    #: BALD *chooses which k-space lines to acquire next* from a posterior
    #: disagreement score, which is acquisition design in the closed-loop sense,
    #: and reconstructs from what it has acquired.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.STRUCTURAL}),
        tasks=frozenset({Task.RECONSTRUCTION, Task.ACQUISITION_DESIGN}),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active = self.config.acquisition.active
        if not self._active.enabled:
            logger.warning(
                "BALDAcquisitionStrategy instantiated but acquisition.active.enabled is False — "
                "the active loop is dormant; falling back to pure reconstruction."
            )
        self._acquired_mask: torch.Tensor | None = None
        self._n_acquired: int = 0  # host-side count, so the budget check needs no device read
        self._step_count: int = 0
        self._n_samples = posterior_samples(self.config)

    def _initialize_seed_mask(self, w: int) -> torch.Tensor:
        """Acquire low-frequency centre lines as the seed mask."""
        n_lf = max(1, int(self._active.seed_mask_lf_fraction * w))
        mask = torch.zeros(w, device=self.device, dtype=torch.bool)
        centre = w // 2
        half = n_lf // 2
        mask[centre - half : centre + half + 1] = True
        self._n_acquired = len(range(w)[centre - half : centre + half + 1])
        return mask

    @torch.no_grad()
    def _bald_score(self, k_input: torch.Tensor) -> torch.Tensor:
        """Predictive disagreement across MC-dropout samples per k-space line.

        The variance is taken over the k-space of the predicted images (``fft2c``), one
        value per phase-encode line (dim -1), the axis the mask acquires along.
        Cost-aware utility (Sanchez 2020 Eq. 6): score = variance / cost,
        where cost = 1 + alpha * |k|² so high-frequency lines pay more.
        """
        model = self.env.generator
        inner = getattr(model, "module", model)
        if getattr(inner, "mc_dropout_p", 0.0) <= 0.0:
            raise RuntimeError(
                "BALD needs MC dropout (model_kwargs.mc_dropout_p > 0); without it the "
                "posterior samples are identical and every line scores zero"
            )
        # Eval mode keeps normalisation statistics fixed; mc_active keeps dropout on.
        was_training = model.training
        model.eval()
        inner.mc_active = True
        try:
            preds = torch.stack([model(k_input) for _ in range(self._n_samples)], dim=0)
        finally:
            inner.mc_active = False
            model.train(was_training)
        k_pred = fft2c(torch.complex(preds.float(), torch.zeros_like(preds.float())))
        spread = (k_pred - k_pred.mean(dim=0, keepdim=True)).abs().pow(2).mean(dim=0)
        per_line_var = spread.mean(dim=tuple(range(spread.dim() - 1)))

        # Cost-aware utility: |k|^2 penalty + acquisition-time prior.
        W = per_line_var.shape[0]
        k = torch.arange(W, device=per_line_var.device).float() - W / 2
        cost = 1.0 + 1e-3 * k.pow(2)
        return per_line_var / cost

    def _maybe_acquire_lines(self, batch: torch.Tensor) -> None:
        """Run the BALD inner loop if it's time and the strategy is enabled."""
        self._step_count += 1
        if not self._active.enabled:
            return
        if self._step_count % self._active.retrain_every:
            return

        w = batch.shape[-1]
        if self._n_acquired >= self._active.target_rate * w:
            return
        scores = self._bald_score(self._masked(batch))
        scores = scores.masked_fill(self._acquired_mask, float("-inf"))
        n_new = min(self._active.query_batch, w - self._n_acquired)
        self._acquired_mask.scatter_(0, scores.topk(n_new).indices, True)  # index_put would sync
        self._n_acquired += n_new
        logger.debug(
            "BALD acquired %d lines (total %d / %d budget)",
            n_new,
            self._n_acquired,
            int(self._active.target_rate * w),
        )

    def _masked(self, k_input: torch.Tensor) -> torch.Tensor:
        """Zero the k-space lines (dim -1) not yet acquired; the seed mask on first use."""
        if not self._active.enabled:
            return k_input
        if self._acquired_mask is None:
            self._acquired_mask = self._initialize_seed_mask(k_input.shape[-1])
        return k_input * self._acquired_mask.to(k_input.dtype)

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # The BALD loop runs before the forward pass, so the backbone sees the union of
        # the seed and the queried lines, never the fully sampled input.
        self._maybe_acquire_lines(input_batch)
        return super()._compute_losses_impl(
            input_batch=self._masked(input_batch), target_batch=target_batch, epoch=epoch, **kwargs
        )

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Grade the acquired mask, not the fully sampled input."""
        return super().validation_step(
            self._masked(input_batch), target_batch, batch_data=batch_data, **kwargs
        )
