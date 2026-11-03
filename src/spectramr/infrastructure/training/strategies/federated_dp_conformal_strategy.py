"""Federated DP-Conformal ULF Strategy: DP-SGD on the reconstruction loop.

Conformal calibration runs after training in
:class:`~spectramr.infrastructure.training.strategies.conformal_calibration_strategy.ConformalCalibrationStrategy`.
DP knobs come from the typed ``config.privacy`` block; the sample rate
``q = batch_size / dataset_size`` uses the live train dataset length. An
absent block, an out-of-range knob or an unknown dataset size raises. The live
budget (``dp/epsilon``, ``dp/steps``, ``q``, dataset and batch size) is merged into
the step metrics after every noised step, so it reaches the training CSV, and
the accountant's step count is checkpointed so a resume continues the ledger.

DP-SGD follows Abadi et al. 2016: :meth:`_compute_losses` runs the loss one
example at a time, clips each example's gradient to ``max_grad_norm``
(:mod:`spectramr.infrastructure.services.privacy.per_sample_clipping`) and
returns a surrogate loss whose gradient is the mean clipped gradient. The
``StepExecutor`` backpropagates it unchanged, then calls
:meth:`_clip_and_log_gradients` before ``opt_g.step()``; that hook adds
``N(0, (noise_multiplier * max_grad_norm / batch_size)^2)`` noise and advances
the accountant. The hook raises if the step did not go through the per-sample
path, so the reported ``(epsilon, delta)`` always matches the applied mechanism.
With ``training.federated_config`` it runs FedAvg / FedBN rounds with per-site DP
(:mod:`.mixins.federated_rounds_mixin`).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from spectramr.config.schemas.training.privacy import PrivacyConfig
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.services.privacy.dp_sgd import add_gaussian_noise
from spectramr.infrastructure.services.privacy.per_sample_clipping import (
    accumulate_clipped_gradient,
    gradient_surrogate,
    slice_sample,
)

from .mixins.dp_mixin import DPMixin
from .mixins.federated_rounds_mixin import FederatedRoundsMixin
from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class FederatedDPConformalULFStrategy(
    FederatedRoundsMixin, DPMixin, ReconstructionTrainingStrategy
):
    """Federated reconstruction with DP-SGD + post-hoc conformal."""

    # Resolved in _configure_from_settings; plain attrs since a strategy is not an nn.Module.
    _dp_batch_size: int = 4
    _dp_dataset_size: int = 0
    _dp_sample_rate: float = 0.0
    # Set by the per-sample loss path, consumed by the noise hook.
    _dp_per_sample_clipped: bool = False
    _dp_step_examples: int = 1
    # Restored in order: the two sizes fix q, so check_strategy_state refuses a mismatch first.
    checkpointed_counters: ClassVar[tuple[str, ...]] = (
        "_dp_dataset_size",
        "_dp_batch_size",
        "_dp_accountant_steps",
    )
    _DP_METRIC_KEYS: ClassVar[frozenset[str]] = frozenset(
        ("dp/epsilon", "dp/delta", "dp/steps", "dp/noise_multiplier", "dp/max_grad_norm")
    ) | frozenset(("dp/sample_rate", "dp/dataset_size", "dp/batch_size"))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._configure_from_settings()

    def _resolve_dataset_size(self, priv: PrivacyConfig, data: Any) -> int:
        """DP dataset size: live loader length, else privacy/data ``dataset_size``; else raise."""
        env = getattr(self, "env", None)
        loader = getattr(env, "train_loader", None) if env is not None else None
        dataset = getattr(loader, "dataset", None) if loader is not None else None
        if dataset is not None:
            try:
                n = len(dataset)
            except TypeError:
                n = 0
            if n >= 1:
                return int(n)

        for source in (priv, data):
            ds = getattr(source, "dataset_size", None) if source is not None else None
            if ds is not None and ds >= 1:
                return int(ds)

        raise ConfigurationError(
            "FederatedDPConformalULFStrategy cannot derive a DP sample rate: "
            "no train_loader dataset length, no config.privacy.dataset_size, "
            "and no config.data.dataset_size. Set one so the reported "
            "(epsilon, delta) budget is honest rather than tied to a literal."
        )

    def _configure_from_settings(self) -> None:
        """Read and validate the DP knobs from ``config``; raise rather than default."""
        config = getattr(self, "config", None)
        priv = getattr(config, "privacy", None) if config is not None else None
        data = getattr(config, "data", None) if config is not None else None

        if priv is None:
            raise ConfigurationError(
                "FederatedDPConformalULFStrategy requires a `privacy` config "
                "block (noise_multiplier, max_grad_norm, target_delta). It is "
                "absent, so DP-SGD cannot be configured; refusing to train a "
                "DP strategy without a privacy guarantee."
            )
        # Coerce a raw mapping (e.g. a caller still passing an untyped dict)
        # into the typed, range-validated schema. A PrivacyConfig passes
        # through unchanged.
        if not isinstance(priv, PrivacyConfig):
            priv = PrivacyConfig.model_validate(priv)

        nm = priv.noise_multiplier
        mgn = priv.max_grad_norm
        delta = priv.target_delta
        # The sample rate (hence epsilon) depends on the real batch size: never guess it.
        if data is None:
            bs = 4
        else:
            loader = getattr(data, "loader", None)
            if loader is None:
                raise ConfigurationError(
                    "FederatedDPConformalULFStrategy could not read "
                    "`data.loader.batch_size` (got "
                    f"data={type(data).__name__}). The DP sample rate is "
                    "computed from it, so a substituted default would silently "
                    "invalidate the reported privacy guarantee."
                )
            bs = loader.batch_size
        self._dp_batch_size = int(bs) if bs and bs >= 1 else 1

        ds = self._resolve_dataset_size(priv, data)
        sample_rate = max(min(self._dp_batch_size / ds, 1.0), 1e-4)
        self._dp_dataset_size = ds
        self._dp_sample_rate = sample_rate

        # Range checks live in configure_dp; let its ValueError propagate so an
        # invalid budget fails at construction rather than degrading to a
        # non-private run (no try/except swallow).
        self.configure_dp(
            noise_multiplier=nm,
            max_grad_norm=mgn,
            sample_rate=sample_rate,
            target_delta=delta,
        )
        self._require_per_sample_compatible_optimisation(config)

        # Stamp the resolved budget into provenance (one-time, structured).
        logger.info(
            "[FederatedDPConformalULF] DP configured from config.privacy: "
            "noise_multiplier=%.4f, max_grad_norm=%.4f, target_delta=%.2e, "
            "batch_size=%d, dataset_size=%d, sample_rate=%.6f, clipping=per_sample",
            nm,
            mgn,
            delta,
            self._dp_batch_size,
            ds,
            sample_rate,
        )

    def _require_per_sample_compatible_optimisation(self, config: Any) -> None:
        """Raise on settings under which per-sample clipping would be silently wrong."""
        opt = getattr(config, "optimization", None)
        precision = getattr(opt, "precision", None)
        if precision is not None and precision.enabled:
            raise ConfigurationError(
                "federated_dp_conformal clips per-sample gradients without a loss "
                "scaler, so fp16 underflow would zero them; set "
                "optimization.precision.enabled: false."
            )
        gradient = getattr(opt, "gradient", None)
        accum = getattr(gradient, "accumulation_steps", 1)
        if accum != 1:
            raise ConfigurationError(
                "federated_dp_conformal adds noise once per optimiser step, calibrated "
                f"to one batch; optimization.gradient.accumulation_steps={accum} "
                "would break the privacy accounting. Set it to 1."
            )

    def _compute_losses(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Per-example loss + clipped gradient; returns the DP-SGD surrogate loss."""
        if not (self._dp_enabled and torch.is_grad_enabled()):
            return super()._compute_losses(input_batch, target_batch, epoch, **kwargs)
        params = [p for p in self.generator_model.parameters() if p.requires_grad]
        n = int(input_batch.shape[0])
        acc = [torch.zeros_like(p) for p in params]
        sums: dict[str, Any] = {}
        # One backward per example is inherent to per-sample clipping; batch 2 here.
        for i in range(n):
            losses, metrics = super()._compute_losses(
                slice_sample(input_batch, i, n),
                slice_sample(target_batch, i, n),
                epoch,
                **slice_sample(kwargs, i, n),
            )
            accumulate_clipped_gradient(losses["g_total_loss"], params, self._max_grad_norm, acc)
            for k, v in losses.items():
                v = v.detach() if isinstance(v, torch.Tensor) else v
                sums[k] = v if k not in sums else sums[k] + v
        out = {k: v / n for k, v in sums.items()}
        mean_grads = torch._foreach_div(acc, float(n))
        out["g_total_loss"] = gradient_surrogate(params, mean_grads, out["g_total_loss"])
        self._dp_per_sample_clipped = True
        self._dp_step_examples = n
        return out, metrics

    @torch.no_grad()
    def _param_grad_norms(self) -> torch.Tensor:
        """Per-*parameter* gradient L2 norms (a diagnostic, not on the DP path)."""
        gen = getattr(self.env, "generator", None)
        if gen is None:
            return torch.zeros(1)
        norms = torch.cat(
            [p.grad.flatten().norm().unsqueeze(0) for p in gen.parameters() if p.grad is not None]
        )
        return norms

    @torch.no_grad()
    def _apply_dp_to_grads(self, model: torch.nn.Module) -> None:
        """Add ``N(0, (sigma * C / B)^2)`` to the mean clipped gradient, in place."""
        if not self._dp_enabled:
            return
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        for g in grads:
            g.copy_(
                add_gaussian_noise(
                    g,
                    noise_multiplier=self._noise_multiplier,
                    max_grad_norm=self._max_grad_norm,
                    batch_size=self._dp_step_examples,
                )
            )

    def _clip_and_log_gradients(
        self,
        model: torch.nn.Module,
        epoch: int,
        step: int,
        model_name: str = "model",
    ) -> float | None:
        """DP-SGD noise hook, called by ``StepExecutor`` after backward, before ``opt_g.step()``.

        Raises when DP is on and the step's gradient did not come from the
        per-sample path, since the accountant would then report a false budget.
        """
        if self._dp_enabled and not self._dp_per_sample_clipped:
            raise RuntimeError(
                "federated_dp_conformal: the gradient was not produced by the "
                "per-sample clipping path, so the DP accountant would misreport "
                "(epsilon, delta). Refusing to step."
            )
        self._dp_per_sample_clipped = False
        self._apply_dp_to_grads(model)
        self.record_dp_step()
        # Host floats from the accountant, no device sync; the loop reads these via get_last_metrics.
        self._last_step_metrics = {**getattr(self, "_last_step_metrics", {}), **self.post_step()}
        return super()._clip_and_log_gradients(model, epoch, step, model_name)

    def declared_metric_keys(self) -> frozenset[str]:
        """The ``dp/*`` budget columns, stamped on every noised step when DP is on."""
        keys = super().declared_metric_keys()
        return keys | self._DP_METRIC_KEYS if self._dp_enabled else keys

    # Read by strategy_state_dict, written back on resume through restore_dp_steps.
    _dp_accountant_steps = property(
        lambda self: int(self.accountant.steps) if self._dp_enabled else 0,
        lambda self, steps: self.restore_dp_steps(steps),
    )

    def check_strategy_state(self, state: dict[str, Any] | None) -> None:
        """Refuse a resume whose saved q differs, since the replayed epsilon would be wrong."""
        super().check_strategy_state(state)
        counters = (state or {}).get("counters", {})
        live = {"_dp_dataset_size": self._dp_dataset_size, "_dp_batch_size": self._dp_batch_size}
        saved = {k: counters.get(k) for k in (*live, "_dp_accountant_steps")}
        if saved.pop("_dp_accountant_steps") is None or saved != live:
            raise RuntimeError(
                f"federated_dp_conformal: checkpoint DP ledger {counters} lacks the accountant "
                f"steps or differs from the live q inputs {live}; epsilon would be misreported."
            )

    def post_step(self) -> dict[str, float]:
        """Live ``(ε, δ)`` budget plus the q / dataset / batch size it was computed from."""
        snapshot = {k: float(v) for k, v in self.dp_metrics().items()}
        snapshot["dp/sample_rate"] = float(self._dp_sample_rate)
        snapshot["dp/dataset_size"] = float(self._dp_dataset_size)
        snapshot["dp/batch_size"] = float(self._dp_batch_size)
        return snapshot
