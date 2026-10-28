"""Deep-VIB recoverability-ceiling training (MICCAI MRIxFields2026, B-2.3).

Trains a :class:`~spectramr.models.generators.recoverability_vib_net.RecoverabilityVIBNet` on the
information-bottleneck Lagrangian ``recon + beta * rate``, where ``rate = KL(q(z|x) || N(0,I))``
is the ``I(X;Z)`` upper bound (the RECOVERABILITY BUDGET, nats/sample). Sweeping ``beta`` traces
the rate-distortion curve; the low-``beta`` limit is the recoverability ceiling for the degraded
input. The headline witness ``val_recoverability_rate`` is the reported budget.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy

#: Reductions ``rate_reduction`` may name. Registered here so an unknown value RAISES
#: instead of silently degrading to the legacy sum (pitfall #9/#15).
_RATE_REDUCTIONS = ("sum", "mean_per_dim")


def _kl_rate(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """KL(q(z|x) || N(0,I)) per sample, in nats.

    The I(X;Z) rate UPPER bound.

    ``sum`` (legacy) integrates over every latent dimension and divides only by batch.
    That makes the rate — and hence the effective ``beta`` — scale with the latent
    grid, which is the units trap that kept collapsing b23: its latent is
    ``16 x 91 x 109 = 158,704`` dims, so ``beta=1e-3`` on the summed rate is
    equivalent to a per-dimension ``beta`` of **159**, and the tight arm's ``1e-1`` to
    **15,870** — both far past the collapse threshold, which is exactly why the two
    ends of the sweep landed at the SAME SSIM (a dead rate-distortion curve) and why
    ``beta_warmup_iters`` and ``free_bits`` each failed to save it.

    ``mean_per_dim`` divides by the latent dimension count instead, so ``beta`` is
    dimensionless and portable across latent geometry (the usual VIB convention, where
    ``beta ~ 1e-3..1e-1`` is the useful range).
    """
    if reduction not in _RATE_REDUCTIONS:
        raise ValueError(
            f"recoverability_vib.rate_reduction={reduction!r} is not implemented; "
            f"supported: {_RATE_REDUCTIONS}."
        )
    per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
    if reduction == "mean_per_dim":
        return per_dim.mean()
    return per_dim.sum() / mu.shape[0]


def _kl_rate_free_bits(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float,
    reduction: str = "sum",
) -> torch.Tensor:
    """The rate PENALTY with a free-bits floor: ``max(rate, free_bits)``.

    Below the floor the ``clamp_min`` has zero gradient, so ``beta * rate`` can no
    longer push the encoder toward posterior collapse (pitfall #20). Above it the
    penalty equals the true rate, so ``beta`` still shapes the trade-off. The REPORTED
    ``val_recoverability_rate`` stays the true (unclamped) KL, so the recoverability
    budget stays honest above the floor. Distinct from KL-capacity annealing, which
    PINS the rate to a target and would corrupt the measurement.

    Two properties this guard does NOT have, both learned the hard way on b23:

    1. ``free_bits`` is in whatever units ``reduction`` produces. Under ``sum`` the
       0.2 configured for b23 was 1.3e-6 nats/dim, while a fully collapsed encoder
       already sits at ~0.044 aggregate — a floor 4.6x above total collapse, i.e. no
       floor at all.
    2. It only removes DOWNWARD pressure. Nothing here pushes the rate back up, so an
       encoder that has already collapsed stays collapsed. It prevents drift; it does
       not rescue. Set it in units that bind from iteration 0.
    """
    rate = _kl_rate(mu, logvar, reduction)
    return rate.clamp_min(free_bits) if free_bits > 0.0 else rate


def compute_recoverability_vib_loss(
    model: Any,
    batch: dict[str, Any],
    *,
    beta: float = 1.0e-3,
    lambda_recon: float = 1.0,
    free_bits: float = 0.0,
    rate_reduction: str = "sum",
) -> dict[str, torch.Tensor]:
    """IB Lagrangian: reconstruction + beta * rate (the recoverability budget).

    ``free_bits`` floors the rate PENALTY (not the reported rate) so the encoder
    cannot collapse to a measurement-independent DC blob (see ``_kl_rate_free_bits``).
    ``rate_reduction`` sets the units of both the rate and that floor (see
    :func:`_kl_rate`).
    """
    x = batch["input"]
    y = batch["target"]
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    recon = model.decode(z, x.shape[-2:])
    loss_recon = F.l1_loss(recon, y)
    # honest reported budget (unclamped), in the configured units
    rate = _kl_rate(mu, logvar, rate_reduction)
    rate_penalty = _kl_rate_free_bits(mu, logvar, free_bits, rate_reduction)
    total = lambda_recon * loss_recon + beta * rate_penalty
    # rate_nats is a REPORTED metric (the recoverability budget); detach so logging never
    # converts a grad tensor to a scalar. loss_total retains grad via `total` above.
    return {
        "loss_total": total,
        "loss_recon": loss_recon.detach(),
        "rate_nats": rate.detach(),
    }


class RecoverabilityVIBStrategy(ReconstructionTrainingStrategy):
    """Train the deep-VIB recoverability-ceiling restorer (B-2.3)."""

    #: Loss ownership (mrixfields review 2026-09-03): computes its objective inline and folds NOTHING else: any other declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("recoverability_vib", "reconstruction"))
        cfg = getattr(self.config.training, "recoverability_vib", None)
        self._vib_beta = float(getattr(cfg, "beta", 1.0e-3)) if cfg is not None else 1.0e-3
        self._vib_lambda_recon = self._declared_inline_l1_weight()
        # Anti-posterior-collapse (#20): ramp beta 0 -> target over this horizon so recon
        # establishes z-usage before the rate budget tightens (the 'recon floor').
        self._vib_beta_warmup = int(getattr(cfg, "beta_warmup_iters", 0)) if cfg is not None else 0
        # Anti-posterior-collapse (#20): free-bits floor on the rate PENALTY so beta
        # cannot drive KL to ~0 (the b23 rate->0.002 degeneracy). 0 disables.
        self._vib_free_bits = float(getattr(cfg, "free_bits", 0.0)) if cfg is not None else 0.0
        # Units of the rate (and hence of beta and free_bits). Validated here rather
        # than only at the tensor op so a bad value fails at construction, not 5000
        # iterations in (#15).
        self._vib_rate_reduction = (
            str(getattr(cfg, "rate_reduction", "sum")) if cfg is not None else "sum"
        )
        if self._vib_rate_reduction not in _RATE_REDUCTIONS:
            raise ValueError(
                "recoverability_vib.rate_reduction="
                f"{self._vib_rate_reduction!r} is not implemented; supported: "
                f"{_RATE_REDUCTIONS}."
            )
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
            self.logging_service.log_info(
                f"[B-2.3] deep-VIB recoverability ceiling: beta={self._vib_beta} "
                f"beta_warmup_iters={self._vib_beta_warmup} "
                f"free_bits={self._vib_free_bits} "
                f"rate_reduction={self._vib_rate_reduction} "
                "(higher beta = tighter rate budget = lower recoverable quality; "
                "beta is comparable across arms only at a FIXED rate_reduction — "
                "under 'sum' it also scales with the latent grid)"
            )

    def _effective_beta(self, iteration: int) -> float:
        """Linear beta warmup: ramp 0 -> target over ``beta_warmup_iters`` (#20 guard)."""
        warmup = getattr(self, "_vib_beta_warmup", 0)
        beta = getattr(self, "_vib_beta", 1.0e-3)
        if warmup and warmup > 0:
            return beta * min(1.0, max(0, iteration) / warmup)
        return beta

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
                "RecoverabilityVIBStrategy requires a mapping batch (dict/TrainingBatch); "
                f"got {type(batch)!r}."
            )
        iteration = int(kwargs.get("iteration", 0) or 0)
        return compute_recoverability_vib_loss(
            self.env.generator,
            batch,
            beta=self._effective_beta(iteration),
            lambda_recon=getattr(self, "_vib_lambda_recon", 1.0),
            free_bits=getattr(self, "_vib_free_bits", 0.0),
            rate_reduction=getattr(self, "_vib_rate_reduction", "sum"),
        )

    def validation_step(self, input_batch: Any, target_batch: Any, **kwargs: Any) -> Any:
        kwargs.pop("batch_context", None)
        metrics = super().validation_step(input_batch, target_batch, **kwargs)
        if isinstance(metrics, dict):
            metrics.update(self._score_recoverability(input_batch))
        return metrics

    def _score_recoverability(self, input_batch: Any) -> dict[str, float]:
        """Witnesses: the recoverability RATE (budget, nats) + an input-dependence sanity check.

        ``val_recoverability_rate`` = KL(q(z|x)||N(0,I)) nats (spatially integrated) — the reported
        budget; it should respond to ``beta`` (higher beta -> lower converged rate). Two collapse
        witnesses guard pitfall #20: ``val_encoder_l2_norm`` (RMS of the posterior mean mu) and
        ``val_encoder_logvar_mean`` directly probe the BOTTLENECK (catching a posterior collapse
        at its source, in the same step, before it shows as rate~0); ``val_input_sensitivity``
        (relative output change when the input is flipped in BOTH spatial dims) probes the OUTPUT
        for a measurement-independent collapse. Near-zero encoder norm or input sensitivity WARNS.
        Self-gating; no new YAML knob.
        """
        x = input_batch
        if not isinstance(x, torch.Tensor) or x.numel() == 0 or x.ndim < 2:
            return {}
        gen = self.env.generator
        with torch.no_grad():
            mu, logvar = gen.encode(x)
            # Same units as the training objective — otherwise the reported budget and
            # the penalised one differ by the latent dimension count (158,704 on b23).
            rate = float(_kl_rate(mu, logvar, getattr(self, "_vib_rate_reduction", "sum")))
            mu_l2 = float((mu.pow(2).mean()).sqrt())  # RMS of the posterior mean
            logvar_mean = float(logvar.mean())
            o1 = gen(x)
            o2 = gen(
                torch.flip(x, dims=[-2, -1])
            )  # flip BOTH spatial dims (any-direction collapse)
        denom = o1.abs().mean().clamp_min(1e-8)
        i_sens = float(((o1 - o2).abs().mean() / denom).detach())
        out = {
            "val_recoverability_rate": rate,
            "val_input_sensitivity": i_sens,
            "val_encoder_l2_norm": mu_l2,
            "val_encoder_logvar_mean": logvar_mean,
        }
        log = getattr(self, "logging_service", None)
        if log is not None:
            if mu_l2 < 1e-2:
                log.log_warning(
                    f"[B-2.3] val_encoder_l2_norm={mu_l2:.2e} ~0 — encoder posterior mean is "
                    "near-silent; posterior collapse suspected (pitfall #20)."
                )
            if i_sens < 1e-3:
                log.log_warning(
                    f"[B-2.3] val_input_sensitivity={i_sens:.2e} ~0 — the VIB decoder may have "
                    "collapsed to a measurement-independent output (pitfall #20)."
                )
        return out

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = deterministic posterior-mean restoration; clamp [0,1]."""
        x0 = batch_context.get("input", input_batch)
        return self.env.generator(x0).clamp(0.0, 1.0)
