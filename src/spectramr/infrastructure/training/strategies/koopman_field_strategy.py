"""Koopman linear field-propagator training (MICCAI MRIxFields2026, B-3.10).

Trains a :class:`~spectramr.models.generators.koopman_field_propagator.KoopmanFieldPropagator`
to translate any source field to any target field. Loss = L1 on the field-propagated prediction
+ an autoencoder-consistency term (``dtau=0`` -> identity propagation -> ``psi(phi(x_s)) ~ x_s``)
so the linear Koopman operator acts on a faithful observable space. Any-to-any (Task 3):
conditions on BOTH source and target field. The Koopman semigroup is STRUCTURAL on the model,
so the one-knob ablation (``model_kwargs.use_koopman_semigroup``) lives there.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.infrastructure.physics.koopman_operator import koopman_numerical_abscissa

from .reconstruction import ReconstructionTrainingStrategy

# Below this relative output spread, a monitor is treated as a degenerate collapse.
_COLLAPSE_EPS = 1e-3


def compute_koopman_field_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    lambda_l1: float = 1.0,
    lambda_recon: float = 0.5,
    lambda_stability: float = 0.0,
) -> dict[str, torch.Tensor]:
    """L1 on the field-propagated prediction + autoencoder consistency (dtau=0).

    When ``lambda_stability > 0`` and the model carries a linear generator ``A``
    (the semigroup arm), an operator-stability penalty ``relu(mu2(A))`` is added, where
    ``mu2(A)=lambda_max(0.5(A+A^T))`` is the DIFFERENTIABLE numerical abscissa that upper-bounds
    the spectral abscissa — driving the continuous semigroup ``exp(dtau*A)`` toward
    non-expansive. The nonlinear ablation (``model.generator is None``) has no operator to
    stabilise, so the term is structurally skipped there.
    """
    x = batch["input"]
    y = batch["target"]
    b_s = batch["field_strength"].reshape(-1).float()
    b_t = batch["field_strength_target"].reshape(-1).float()
    pred = model(x, field_strength=b_s, field_strength_target=b_t)
    loss_l1 = F.l1_loss(pred, y)
    # dtau=0 -> exp(0*A)=I -> decode(encode(x_s)). A consistency regulariser that pushes phi,psi
    # toward APPROXIMATE invertibility (zero recon loss structurally penalises a lossy bottleneck:
    # a non-injective phi cannot have a deterministic psi reconstruct every x), not a hard guarantee.
    recon = model(x, field_strength=b_s, field_strength_target=b_s)
    loss_recon = F.l1_loss(recon, x)
    total = lambda_l1 * loss_l1 + lambda_recon * loss_recon
    out = {
        "loss_total": total,
        "loss_l1": loss_l1,
        "loss_recon": loss_recon,
        # Grad-carrying pair for the loss-SSOT seam (popped in _compute_losses_impl):
        # sharpen the field-propagated prediction, not the identity-recon term.
        "prediction": pred,
        "target_image": y,
    }
    gen_a = getattr(model, "generator", None)
    if lambda_stability > 0.0 and gen_a is not None:
        loss_stability = torch.relu(koopman_numerical_abscissa(gen_a))
        out["loss_stability"] = loss_stability
        out["loss_total"] = out["loss_total"] + lambda_stability * loss_stability
    return out


class KoopmanFieldStrategy(ReconstructionTrainingStrategy):
    """Train the continuous Koopman cross-field propagator (B-3.10)."""

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = True

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("koopman_field", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
            gen = getattr(self.env, "generator", None)
            abscissa = gen.spectral_abscissa() if hasattr(gen, "spectral_abscissa") else None
            self.logging_service.log_info(
                f"[B-3.10] koopman semigroup={getattr(gen, 'use_koopman_semigroup', '?')}, "
                f"koopman_dim={getattr(gen, 'koopman_dim', '?')}, "
                f"init spectral_abscissa={abscissa}"
            )
        cfg = getattr(self.config.training, "koopman_field", None)
        self._kf_lambda_l1 = self._declared_inline_l1_weight()
        self._kf_lambda_recon = float(getattr(cfg, "lambda_recon", 0.5)) if cfg is not None else 0.5
        self._kf_lambda_stability = (
            float(getattr(cfg, "lambda_stability", 0.0)) if cfg is not None else 0.0
        )
        if getattr(self, "logging_service", None):
            self.logging_service.log_info(
                f"[B-3.10] loss weights: lambda_l1={self._kf_lambda_l1}, "
                f"lambda_recon={self._kf_lambda_recon}, "
                f"lambda_stability={self._kf_lambda_stability} "
                "(relu(numerical-abscissa(A)); semigroup arm only)"
            )

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):  # dict OR TrainingBatch (canonical pipeline)
            raise ValueError(
                "KoopmanFieldStrategy requires a mapping batch (dict/TrainingBatch) from the "
                f"mrixfields dataset; got {type(batch)!r}."
            )
        out = compute_koopman_field_loss(
            self.env.generator,
            batch,
            lambda_l1=getattr(self, "_kf_lambda_l1", 1.0),
            lambda_recon=getattr(self, "_kf_lambda_recon", 0.5),
            lambda_stability=getattr(self, "_kf_lambda_stability", 0.0),
        )
        # Loss-SSOT seam: fold declarative image losses (hfen/ms_ssim) onto the inline
        # objective; the inline l1 is skipped by the seam (no double-count).
        pred = out.pop("prediction", None)
        target_image = out.pop("target_image", None)
        if pred is not None and target_image is not None:
            aux = self._apply_builder_image_losses(pred, target_image, out)
            if aux is not None:
                out["loss_total"] = out["loss_total"] + aux
            self._last_prediction = pred
            self._last_target = target_image
        return out

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        field_strength: Any = None,
        field_strength_target: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the per-sample source + target fields into batch_context (gated seam)."""
        bc = kwargs.pop("batch_context", None) or {}
        if field_strength is not None:
            bc["field_strength"] = field_strength
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        metrics = super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)
        if isinstance(metrics, dict):
            metrics.update(self._score_koopman(input_batch, bc))
        return metrics

    def _score_koopman(self, input_batch: Any, batch_context: dict[str, Any]) -> dict[str, float]:
        """Provenance + mechanism-fires + degeneracy monitors (self-gating; no new YAML knob).

        - ``val_koopman_spectral_abscissa`` = max Re(lambda(A)) of the learned generator — the
          interpretable operator-stability witness (semigroup arm only).
        - ``val_field_sensitivity`` = relative output change when ONLY the field swaps (two
          IN-DISTRIBUTION pairs over {0.1..7} T, same input). Detects a FIELD-blind collapse
          (output ignores ``b_s,b_t``). NOTE: this measures field-EFFECT, NOT input-dependence —
          a constant-encoder + nonzero ``A`` still scores >0 here, so it is emitted (it is also
          ~0 by design during early warmup for the zero-init linear arm) but not warned on.
        - ``val_input_sensitivity`` = relative output change when ONLY the input swaps (same
          field, horizontally-flipped image). Detects the COMPLEMENTARY degeneracy — an
          INPUT-blind / collapsed encoder producing measurement-independent output (#20); a value
          ~0 is never legitimate (even a random autoencoder is input-dependent), so it warns.
        """
        out: dict[str, float] = {}
        gen = self.env.generator
        abscissa = gen.spectral_abscissa() if hasattr(gen, "spectral_abscissa") else None
        if abscissa is not None:
            out["val_koopman_spectral_abscissa"] = float(abscissa)
        x = batch_context.get("input", input_batch)
        if isinstance(x, torch.Tensor) and x.numel() > 0 and x.ndim >= 2:
            n = x.shape[0]
            dev = x.device

            def _full(v: float) -> torch.Tensor:
                return torch.full((n,), float(v), device=dev)

            with torch.no_grad():
                # field sensitivity: two in-distribution field pairs, fixed input.
                o1 = gen(x, field_strength=_full(0.1), field_strength_target=_full(7.0))
                o2 = gen(x, field_strength=_full(1.5), field_strength_target=_full(3.0))
                # input sensitivity: same field, a different image (horizontal flip).
                oi = gen(
                    torch.flip(x, dims=[-1]),
                    field_strength=_full(0.1),
                    field_strength_target=_full(7.0),
                )
            denom = o1.abs().mean().clamp_min(1e-8)
            f_sens = float(((o1 - o2).abs().mean() / denom).detach())
            i_sens = float(((o1 - oi).abs().mean() / denom).detach())
            out["val_field_sensitivity"] = f_sens
            out["val_input_sensitivity"] = i_sens
            if i_sens < _COLLAPSE_EPS and getattr(self, "logging_service", None):
                self.logging_service.log_warning(
                    f"[B-3.10] val_input_sensitivity={i_sens:.2e} ~0 — the encoder may have "
                    "collapsed to measurement-independent (input-blind) output (pitfall #20)."
                )
        return out

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = propagate source->target through the Koopman operator; clamp."""
        x0 = batch_context.get("input", input_batch)
        b_s = batch_context.get("field_strength", kwargs.get("field_strength"))
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_s is None or b_t is None:
            raise ValueError(
                "KoopmanFieldStrategy validation requires per-sample 'field_strength' AND "
                "'field_strength_target'. Got "
                f"field_strength={b_s is not None}, field_strength_target={b_t is not None}."
            )
        b_s = torch.as_tensor(b_s, device=x0.device).reshape(-1).float()
        b_t = torch.as_tensor(b_t, device=x0.device).reshape(-1).float()
        return self.env.generator(x0, field_strength=b_s, field_strength_target=b_t).clamp(0.0, 1.0)
