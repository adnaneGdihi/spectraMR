r"""Equivariance-Defect Conformal Calibration Strategy (B-1 / Idea 1).

A *degenerate* training strategy: it performs **no gradient updates**. It holds
a frozen, pretrained reconstructor :math:`G_\phi` and a calibration loader,
computes the equivariance-defect non-conformity score
(:mod:`spectramr.core.metrics.equivariance_defect`) per calibration sample over
``K`` random rotations, accumulates the empirical distribution, and emits the
split-conformal quantile :math:`q_\alpha` as a JSON artefact.

This is the reference-free *and* marker-free sibling of
:class:`~spectramr.infrastructure.training.strategies.conformal_calibration_strategy.ConformalCalibrationStrategy`
(which scores against a known marker subspace). They are deliberately distinct
classes.

Dispatch (lead wires this): add to
``spectramr/infrastructure/training/strategy_factory.py::STRATEGY_CLASS_PATHS``::

    "equivariance_conformal": (
        "spectramr.infrastructure.training.strategies.equivariance_conformal_strategy."
        "EquivarianceConformalCalibrationStrategy"
    )
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from spectramr.core.metrics.equivariance_defect import (
    conformal_quantile,
    equivariance_defect_score,
)
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin
from spectramr.infrastructure.training.strategies.mixins.validation import ValidationMixin

logger = logging.getLogger(__name__)


def _extract_measurement(batch: Any) -> torch.Tensor:
    """Pull the measurement tensor ``y`` from a dataloader batch.

    Accepts a bare tensor, an ``(input, target)`` tuple/list, or a dict with an
    ``"input"`` / ``"kspace"`` / ``"image"`` key. Raises if none match — no
    silent fallback (CLAUDE.md #9).
    """
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (tuple, list)) and batch:
        return batch[0]
    if isinstance(batch, dict):
        for key in ("input", "kspace", "image", "y"):
            val = batch.get(key)
            if torch.is_tensor(val):
                return val
    raise ValueError(
        "Could not extract a measurement tensor from the batch. Expected a "
        "tensor, an (input, target) sequence, or a dict with an 'input'/"
        "'kspace'/'image' key."
    )


def run_equivariance_calibration(
    recon_fn: Callable[..., torch.Tensor],
    dataloader: Any,
    *,
    alpha: float = 0.1,
    num_rotations: int = 8,
    group: str = "SO2",
    seed: int = 0,
    max_shift_px: int = 4,
    device: torch.device | str = "cpu",
    exchangeability_test: str = "ks_test",
    exchangeability_ks_pvalue_threshold: float = 0.01,
) -> dict[str, Any]:
    r"""Run the equivariance-defect conformal calibration loop.

    Iterates the calibration loader once (no gradients), computes the per-sample
    equivariance-defect score, and returns the conformal report including the
    threshold :math:`q_\alpha`.

    The per-sample seed is ``seed + running_index`` so every calibration point
    sees a *different but reproducible* rotation set; the rotation distribution
    is identical across points, preserving exchangeability.

    Args:
        recon_fn: Frozen reconstructor ``G(y) -> image``.
        dataloader: Iterable of batches (tensor / tuple / dict).
        alpha: Target miscoverage.
        num_rotations: ``K`` rotations per score.
        group: ``"SO2"`` or ``"SE2"``.
        seed: Base seed.
        max_shift_px: Max integer shift for ``SE2``.
        device: Device to move measurements onto.
        exchangeability_test: ``"ks_test"`` or ``"none"``.
        exchangeability_ks_pvalue_threshold: Flag threshold for the KS test.

    Returns:
        A JSON-serialisable report dict with keys ``quantile``,
        ``n_calibration``, ``alpha``, ``num_rotations_K``, ``group``,
        ``mean_score``, ``score_fn``, ``coverage_at_alpha`` /
        ``empirical_coverage`` (held-out split estimate), and (when run)
        ``exchangeability``.
    """
    all_scores: list[float] = []
    running = 0
    with torch.no_grad():
        for batch in dataloader:
            y = _extract_measurement(batch).to(device)
            if y.ndim == 3:  # [C, H, W] -> [1, C, H, W]
                y = y.unsqueeze(0)
            per_sample = equivariance_defect_score(
                recon_fn,
                y,
                num_rotations=num_rotations,
                group=group,
                seed=seed + running,
                max_shift_px=max_shift_px,
            )
            all_scores.extend(float(v) for v in per_sample.flatten().tolist())
            running += y.shape[0]

    if not all_scores:
        raise ValueError("Calibration loader yielded no samples.")

    scores_t = torch.tensor(all_scores, dtype=torch.float64)
    q = conformal_quantile(scores_t, alpha)

    report: dict[str, Any] = {
        "quantile": q,
        "n_calibration": len(all_scores),
        "alpha": alpha,
        "num_rotations_K": num_rotations,
        "group": group,
        "mean_score": float(scores_t.mean().item()),
        "score_fn": "equivariance_defect",
    }

    # Empirical coverage at alpha — the metric the equivariance-conformal arms
    # actually grade on (best_metric_name: coverage_at_alpha). It was never
    # produced (the report only carried quantile/mean_score), so the arm
    # certified a metric it never computed (#18, metric<->claim mismatch).
    # Estimate it unbiasedly on a held-out split: fit the threshold on a random
    # half (seeded for reproducibility), then measure the fraction of the other
    # half it covers (score <= q). The deployment ``quantile`` above is the
    # full-data fit and is unchanged.
    if len(all_scores) >= 2:
        gen = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(len(all_scores), generator=gen)
        shuffled = scores_t[perm]
        half = len(all_scores) // 2
        q_split = conformal_quantile(shuffled[:half], alpha)
        coverage = float((shuffled[half:] <= q_split).double().mean().item())
        report["coverage_at_alpha"] = coverage
        report["empirical_coverage"] = coverage  # alias for badge aggregation
    else:
        report["coverage_at_alpha"] = None
        report["empirical_coverage"] = None

    if exchangeability_test == "ks_test" and len(all_scores) >= 4:
        # Split the calibration scores in half and KS-test the two halves as a
        # cheap self-consistency proxy for exchangeability of the score.
        half = len(all_scores) // 2
        a = scores_t[:half]
        b = scores_t[half:]
        ks_stat, p_value = _two_sample_ks(a, b)
        report["exchangeability"] = {
            "test": "ks_test",
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "threshold": exchangeability_ks_pvalue_threshold,
            "flagged": p_value < exchangeability_ks_pvalue_threshold,
        }

    return report


def _two_sample_ks(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic + asymptotic p-value (torch-only).

    A dependency-free implementation of the KS two-sample test so the strategy
    does not require SciPy. The p-value uses the asymptotic Kolmogorov
    distribution :math:`Q_{KS}(\\lambda)=2\\sum_{j\\ge 1}(-1)^{j-1}e^{-2j^2\\lambda^2}`.
    """
    a, _ = torch.sort(a.flatten())
    b, _ = torch.sort(b.flatten())
    n, m = a.numel(), b.numel()
    grid = torch.cat([a, b]).unique()
    cdf_a = torch.searchsorted(a, grid, right=True).double() / n
    cdf_b = torch.searchsorted(b, grid, right=True).double() / m
    d = float((cdf_a - cdf_b).abs().max().item())
    en = (n * m / (n + m)) ** 0.5
    lam = (en + 0.12 + 0.11 / en) * d
    # Kolmogorov asymptotic survival function Q_KS(lam). The series only
    # converges to the correct value for lam > 0; in the lam -> 0 limit the
    # exact survival function is 1 (identical empirical CDFs => p = 1), so we
    # short-circuit small lam rather than summing a non-convergent alternating
    # series.
    if lam < 1e-3:
        return d, 1.0
    p = 0.0
    for j in range(1, 101):
        p += ((-1) ** (j - 1)) * torch.exp(torch.tensor(-2.0 * j * j * lam * lam))
    p_value = float(max(0.0, min(1.0, 2.0 * p.item())))
    return d, p_value


class EquivarianceConformalCalibrationStrategy(BaseTrainingStrategy, MetricsMixin, ValidationMixin):
    """Calibration-only strategy; performs no gradient updates.

    The training loop is degenerate: :meth:`train_step` is a no-op and
    :meth:`_compute_losses_impl` returns a zero total loss so the base
    contract holds if the orchestrator ever calls it. The science happens in
    :meth:`run_calibration`, which loads the frozen reconstructor, computes the
    equivariance-defect scores, and writes the conformal quantile artefact.
    """

    def _setup_strategy_specific_components(self) -> None:
        cfg = getattr(self.config.training, "equivariance_conformal", None)
        if cfg is None:
            # Tolerate absence so the class is importable / testable without the
            # sub-block; run_calibration() will raise if invoked without it.
            self._ec_cfg = None
            logger.warning(
                "EquivarianceConformalCalibrationStrategy initialised without a "
                "training.equivariance_conformal block — strategy is a no-op."
            )
            return
        # The mode-dispatch to the typed TrainingConfigEquivarianceConformal does
        # not fire for these arms (config.training loads as the generic
        # TrainingStrategyConfigSchema), so the sub-block arrives as a raw dict
        # whose ``.alpha``/``.num_rotations_K`` attribute access crashed with
        # "'dict' object has no attribute 'alpha'" (cluster smoke 20260605, only
        # reachable once the upstream 'conditioning' kwarg leak was fixed).
        # Coerce to the SSOT schema (extra='forbid' validates the block).
        if isinstance(cfg, dict):
            from spectramr.config.schemas.training.equivariance_conformal import (
                EquivarianceConformalConfig,
            )

            cfg = EquivarianceConformalConfig(**cfg)
        self._ec_cfg = cfg
        logger.info(
            "EquivarianceConformalCalibrationStrategy: alpha=%s K=%s group=%s",
            cfg.alpha,
            cfg.num_rotations_K,
            cfg.group,
        )

    # ------------------------------------------------------------------
    # Degenerate training loop — no parameter updates.
    # ------------------------------------------------------------------
    @accepts_step_io
    def train_step(
        self, batch: Any, epoch: int = 0, step: int = 0, **kwargs: Any
    ) -> list:  # pragma: no cover - integration-only
        """No-op train step (no optimizer steps); calibration runs in
        :meth:`run_calibration`. Returns an EMPTY step-config list — the old
        dict return hit KeyError('optimizer') in Trainer.execute_step."""
        return []

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:  # pragma: no cover - integration-only
        """Zero loss — the strategy never optimises anything."""
        ref = input_batch if torch.is_tensor(input_batch) else target_batch
        device = ref.device if torch.is_tensor(ref) else self.device
        return {"g_total_loss": torch.zeros((), device=device, requires_grad=False)}

    # ------------------------------------------------------------------
    # The calibration pass.
    # ------------------------------------------------------------------
    def _frozen_reconstructor(self) -> Callable[..., torch.Tensor]:
        """Return the frozen reconstructor callable ``G(y) -> image``.

        Prefers the generator wired onto the strategy context (loaded from the
        configured checkpoint by the model builder); falls back to the
        ``generator_model`` property. The returned callable detaches its output
        so no gradients leak into the (frozen) network.
        """
        _ctx = getattr(self, "context", None)
        model = getattr(_ctx, "generator", None)
        if model is None:
            model = self.generator_model
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def _fn(y: torch.Tensor, csm: torch.Tensor | None = None) -> torch.Tensor:
            out = model(y) if csm is None else model(y, csm)
            return out.detach()

        return _fn

    def run_calibration(self, dataloader: Any) -> dict[str, Any]:
        """Compute the conformal quantile over ``dataloader`` and write the artefact.

        Returns the report dict and, when ``output_artefact`` is set, writes it
        as JSON.

        Raises:
            ValueError: if the ``equivariance_conformal`` block is absent.
        """
        cfg = self._ec_cfg
        if cfg is None:
            raise ValueError("run_calibration requires a training.equivariance_conformal block.")
        recon_fn = self._frozen_reconstructor()
        report = run_equivariance_calibration(
            recon_fn,
            dataloader,
            alpha=cfg.alpha,
            num_rotations=cfg.num_rotations_K,
            group=cfg.group,
            seed=cfg.seed,
            max_shift_px=cfg.max_shift_px,
            device=self.device,
            exchangeability_test=cfg.exchangeability_test,
            exchangeability_ks_pvalue_threshold=cfg.exchangeability_ks_pvalue_threshold,
        )
        self._write_artefact(getattr(cfg, "output_artefact", None), report)
        logger.info(
            "Equivariance-defect conformal quantile q_alpha=%.6g over %d samples.",
            report["quantile"],
            report["n_calibration"],
        )
        return report

    @staticmethod
    def _write_artefact(path: str | None, payload: dict[str, Any]) -> None:
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Wrote equivariance-conformal artefact to %s", out)


__all__ = [
    "EquivarianceConformalCalibrationStrategy",
    "run_equivariance_calibration",
]
