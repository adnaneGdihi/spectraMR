r"""Sparsifying-frame coherence-optimisation training (A-6.5 *SCO-Frame*).

Trains a :class:`~spectramr.models.generators.tight_frame_learner.TightFrameLearner`
(a learnable convolutional sparsifying frame, sparse-coding autoencoder) with a
reconstruction loss PLUS two structural penalties on the learned atoms:

* a **Parseval-tightness** term (always-on, ``lambda_parseval``) so the
  analyse/synthesise pair stays a tight frame (energy-preserving round-trip);
* a **mutual-coherence** term (the A-6.5 **one-knob**, ``lambda_coherence``) whose form is
  ``coherence_penalty``: the default ``smooth_max`` (log-sum-exp over the squared atom Gram
  off-diagonal, sharpness ``coherence_beta``) minimises ``mu`` itself; ``frobenius`` lowers
  the average squared coherence only.

The one-knob ablation (``training.sparse_frame.lambda_coherence`` 0 vs >0) isolates
coherence optimisation from tightness and reconstruction: the oracle is that the
treatment's learned mutual coherence (``val_mutual_coherence``) is lower than the
control's at matched reconstruction. Knobs are read from config (#15); the
resolved values + initial coherence are stamped into provenance.

The coherence term is the registered ``frame_coherence`` loss (a parameter
penalty, wired here via ``frame=`` so it is never inert, #16); tightness is the
model's own ``parseval_penalty()``. The penalty is added to the canonical
grad-carrying ``g_total_loss`` key (the key the loop reads for backward).
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.models.losses.registry import create_loss

from .reconstruction import ReconstructionTrainingStrategy


class SparseFrameStrategy(ReconstructionTrainingStrategy):
    """Train a learnable sparsifying frame with coherence optimisation (A-6.5)."""

    def _setup_strategy_specific_components(self) -> None:
        super()._setup_strategy_specific_components()
        cfg = getattr(self.config.training, "sparse_frame", None)
        self._lambda_coherence = float(getattr(cfg, "lambda_coherence", 0.0)) if cfg else 0.0
        self._lambda_parseval = float(getattr(cfg, "lambda_parseval", 0.0)) if cfg else 0.0
        penalty = str(getattr(cfg, "coherence_penalty", "smooth_max")) if cfg else "smooth_max"
        beta = float(getattr(cfg, "coherence_beta", 1000.0)) if cfg else 1000.0
        # The loss constructor validates both and raises on an unknown penalty or bad beta.
        self._coherence_loss = create_loss("frame_coherence", penalty=penalty, beta=beta)
        self.sparse_frame_provenance: dict[str, Any] = {
            "lambda_coherence": self._lambda_coherence,
            "lambda_parseval": self._lambda_parseval,
            "coherence_penalty": penalty,
            "coherence_beta": beta,
        }
        gen = getattr(getattr(self, "env", None), "generator", None)
        if hasattr(gen, "atoms"):
            # Constant log(N)/beta the logged smooth-max sits above mu^2 at most.
            n_atoms = int(gen.atoms().shape[0])
            offset = self._coherence_loss.penalty_offset(n_atoms)
            self.sparse_frame_provenance["coherence_offset"] = offset

        if getattr(self, "logging_service", None):
            mu = gen.mutual_coherence() if hasattr(gen, "mutual_coherence") else float("nan")
            self.logging_service.log_info(
                f"[A-6.5 SCO-Frame] resolved knobs {self.sparse_frame_provenance}; "
                f"initial mutual_coherence={mu:.4g}"
            )

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        losses = super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )
        gen = getattr(self.env, "generator", None)
        total = losses.get("g_total_loss") if isinstance(losses, dict) else None
        if gen is None or not (isinstance(total, torch.Tensor) and total.requires_grad):
            return losses

        extra: torch.Tensor | None = None
        lam_c = getattr(self, "_lambda_coherence", 0.0)
        lam_p = getattr(self, "_lambda_parseval", 0.0)
        if lam_c > 0.0:
            if not hasattr(gen, "atoms"):
                raise ValueError(
                    f"lambda_coherence={lam_c} > 0 but generator {type(gen).__name__} exposes "
                    "no atoms(); the coherence penalty has nothing to act on."
                )
            coh = self._coherence_loss(frame=gen)
            losses["frame_coherence"] = coh.detach()
            extra = lam_c * coh if extra is None else extra + lam_c * coh
        if lam_p > 0.0 and hasattr(gen, "parseval_penalty"):
            par = gen.parseval_penalty()
            losses["frame_parseval"] = par.detach()
            extra = lam_p * par if extra is None else extra + lam_p * par
        if extra is not None:
            losses["g_total_loss"] = total + extra
        return losses

    def validation_step(self, input_batch: Any, target_batch: Any, **kwargs: Any) -> Any:
        """Emit the mutual-coherence witness alongside the usual metrics."""
        metrics = super().validation_step(input_batch, target_batch, **kwargs)
        if isinstance(metrics, dict):
            gen = getattr(self.env, "generator", None)
            if hasattr(gen, "mutual_coherence"):
                metrics["val_mutual_coherence"] = float(gen.mutual_coherence())
        return metrics
