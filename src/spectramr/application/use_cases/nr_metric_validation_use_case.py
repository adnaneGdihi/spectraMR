"""No-reference metric validation use case (spec §8).

Runs the label-free NR metric battery through the six-axis meta-evaluation
subsystem (CDSCR, LGDR, SFA, ESD, FSPD, Sim2Rank) on a physics-anchored
degradation sweep, producing a reproducible concordance ranking of how well
each NR metric tracks degradation severity.

This is the §8.1 "physics-anchored concordance" tier: the simulator's
``operator_library`` is the *audited digital twin*
(:func:`spectramr.infrastructure.physics.degradations.apply_degradation`,
the deterministic ``(name, theta, seed)`` SSOT), injected via the
dependency-inversion seam so ``core/`` never imports ``infrastructure/`` at
module scope. The sweep, metric pre-computation (with a per-sample
``MetricContext`` synthesised for measurement-dependent NR metrics), and
ranking all reuse the existing pipeline — this use case only wires the NR
battery + the physics oracle into it.

Layer: ``application/`` may import ``infrastructure/`` and ``core/`` (deps
point inward). The CLI ``spectramr meta-evaluate --nr-battery`` delegates here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

# Default physics degradation families (digital-twin keys) mapped to the §8.1
# campaign axes: noise / motion / gibbs / aliasing / bias / blur.
DEFAULT_FAMILIES: tuple[str, ...] = (
    "rician",
    "rigid_motion",
    "gibbs",
    "cartesian_undersamp",
    "bias",
    "t2star_blur",
)

# The asset-free NR battery (the prior-geometry / cross-contrast-predictor
# metrics need trained checkpoints, so they are excluded from the default
# label-free sweep — enable them explicitly with their assets configured).
DEFAULT_NR_BATTERY: tuple[str, ...] = (
    # consistency
    "ndcr",
    "prp_snr",
    "fmrs",
    "csoe",
    "aoqv",
    # artifact
    "pegc",
    "are",
    "gri",
    "cpge",
    "rar",
    # spectral
    "csr",
    "sher",
    "bpci",
    "ssel",
    # stability
    "lrjs",
    "fmed",
    "pedu",
    # cross (asset-free subset)
    "ccsa",
    "brc",
    # texture
    "wlms",
    "wpde",
    "pgle",
    "pid",
    "bemd_ier",
    "pcd",
    "savs",
    # physics-nr extensions
    "psdc",
    "ksk",
    "ciea",
    "hfrw",
    "qvcr",
    # no-reference extensions
    "ecfd",
    "bnac",
)


def build_physics_operator_library(
    families: Iterable[str],
) -> dict[str, Any]:
    """Adapt the digital-twin ``apply_degradation`` oracle to the simulator's
    ``(clean, theta, generator) -> tensor`` operator contract.

    Each adapter derives a deterministic integer seed from the simulator's
    ``torch.Generator`` so the ``(theta, seed)`` reproducibility contract of the
    twin holds end-to-end. Unknown family names raise (the twin's
    ``apply_degradation`` validates against its registry — no silent fallback).
    """
    from spectramr.infrastructure.physics.degradations import apply_degradation, list_degradations

    known = set(list_degradations())
    families = list(families)
    unknown = [f for f in families if f not in known]
    if unknown:
        raise ValueError(
            f"unknown degradation families {unknown}; valid digital-twin keys: {sorted(known)}"
        )

    def _make(name: str):
        def _op(clean: torch.Tensor, theta: float, generator: torch.Generator) -> torch.Tensor:
            seed = int(generator.initial_seed()) & 0xFFFFFFFF
            x = clean
            squeeze_to = x.dim()
            while x.dim() < 4:
                x = x.unsqueeze(0)
            out = apply_degradation(name, x, float(theta), seed=seed)
            while out.dim() > squeeze_to:
                out = out.squeeze(0)
            # twin may return complex; the magnitude sweep keeps the input dtype.
            if out.is_complex() and not clean.is_complex():
                out = out.abs()
            return out.to(clean.dtype).reshape(clean.shape)

        return _op

    return {name: _make(name) for name in families}


# The §8 label-free validation tiers the full harness can run. Each maps to a
# scorer in ``spectramr.application.use_cases.nr_validation`` returning a per-metric
# ``TierResult``. Tier 5 (defensibility ranking) is the meta-evaluation pipeline
# itself and is always produced; §8.6 (borrowed radiologist labels) needs external
# downloads and is intentionally NOT in this self-contained set.
HARNESS_TIERS: tuple[str, ...] = (
    "concordance",  # §8.1 physics-anchored severity concordance
    "fr_proxy",  # §8.2 full-reference proxy agreement (+ VIF)
    "cho",  # §8.3 channelized-Hotelling lesion-detection task
    "construct_validity",  # §8.4 test-retest ICC + flip-invariance (+ redundancy)
)


@dataclass
class NRMetricValidationConfig:
    metrics: tuple[str, ...] = DEFAULT_NR_BATTERY
    families: tuple[str, ...] = DEFAULT_FAMILIES
    n_severities_per_family: int = 12
    seed: int = 42
    eval_mode: str = "2d"
    aggregator: str = "z"
    # --- full §8 harness knobs (consumed by run_validation; pitfall #15: every
    # knob is read + validated + stamped into the result provenance) ---
    tiers: tuple[str, ...] = HARNESS_TIERS
    fr_proxies: tuple[str, ...] = ("psnr", "ssim", "ms_ssim", "lpips", "dists")
    fr_proxy_rho_threshold: float = 0.6
    concordance_rho_threshold: float = 0.6
    cho_rho_threshold: float = 0.5
    cho_families: tuple[str, ...] = ("rician_noise", "gaussian_blur", "motion")
    cho_n_severities: int = 6
    cho_n_ensemble: int = 24
    icc_threshold: float = 0.75
    invariance_rel_tol: float = 0.05
    redundancy_corr_threshold: float = 0.95
    # test-retest replicate: a second sweep at seed + offset yields sample-aligned
    # (same content/family/theta, different noise) replicate columns for ICC(2,1).
    retest_seed_offset: int = 9973
    fit_aggregator: bool = True
    # register the fitted nr_quality_index into the global metric registry. Off by
    # default — the meta-metric is research-mode and we don't pollute the registry
    # during a normal validation run unless explicitly asked.
    register_aggregator: bool = False
    run_redundancy_gate: bool = True

    def __post_init__(self) -> None:
        unknown = tuple(t for t in self.tiers if t not in HARNESS_TIERS)
        if unknown:
            raise ValueError(
                f"unknown NR validation tier(s) {unknown}; valid tiers: {HARNESS_TIERS}"
            )


@dataclass
class NRMetricValidationUseCase:
    """Orchestrate the §8.1 physics-anchored NR-metric concordance ranking."""

    config: NRMetricValidationConfig = field(default_factory=NRMetricValidationConfig)

    def run(self, clean_volumes: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
        """Run the degradation sweep + six-axis ranking; return the summary dict.

        Args:
            clean_volumes: iterable of ``(content_id, tensor)`` pristine
                references (2-D/3-D). The simulator degrades each across every
                configured family x severity.

        Returns:
            ``MetaEvaluationOutput.to_summary()`` — final ranks/scores per
            ranker, the aggregate ranking, and defensibility flags.
        """
        import logging

        logging.getLogger(__name__).warning(
            "NR metric battery is RESEARCH-MODE / validation-pending: this offline "
            "concordance run is HOW the battery is validated; do not treat the "
            "metrics as production-certified, and never enable them during training."
        )

        from spectramr.core.metrics.meta_evaluation.metric_adapter import (
            build_safe_metric_set,
        )
        from spectramr.core.metrics.meta_evaluation.pipeline import (
            MetaEvaluationPipeline,
        )
        from spectramr.core.metrics.meta_evaluation.simulator import SimulatorConfig
        from spectramr.core.metrics.meta_evaluation.types import MetricSet

        metric_set = MetricSet(**build_safe_metric_set(list(self.config.metrics)))
        operator_library = build_physics_operator_library(self.config.families)

        pipeline = MetaEvaluationPipeline.from_defaults()
        pipeline.eval_mode = self.config.eval_mode
        pipeline.aggregator = self.config.aggregator
        pipeline.simulator_config = SimulatorConfig(
            families=list(self.config.families),
            n_severities_per_family=self.config.n_severities_per_family,
            seed=self.config.seed,
            operator_library=operator_library,
        )
        output = pipeline.run(metric_set, clean_volumes)
        return output.to_summary()

    # ----------------------------------------------------------------- full §8
    def _build_dataset(
        self,
        clean_volumes: list[tuple[str, torch.Tensor]],
        metric_names: list[str],
        operator_library: dict[str, Any],
        *,
        seed: int,
    ) -> Any:
        """Run the simulator + metric pre-computation into a MetricEvaluationDataset.

        Used for the Tier-4 replicate sweep (same content/family/theta grid, a
        different per-sample noise seed) without paying for the rankers. The
        severity grid is seed-independent (Halton), so samples align with the
        primary sweep one-for-one.
        """
        from spectramr.core.metrics.meta_evaluation.metric_adapter import (
            build_safe_metric_set,
        )
        from spectramr.core.metrics.meta_evaluation.simulator import (
            SimulatorConfig,
            precompute_metric_values,
            run_simulator,
        )
        from spectramr.core.metrics.meta_evaluation.types import (
            MetricEvaluationDataset,
        )

        sim_cfg = SimulatorConfig(
            families=list(self.config.families),
            n_severities_per_family=self.config.n_severities_per_family,
            seed=seed,
            operator_library=operator_library,
        )
        samples = run_simulator(clean_volumes, sim_cfg)
        metric_set = build_safe_metric_set(metric_names)
        values = precompute_metric_values(samples, metric_set["metrics"])
        return MetricEvaluationDataset(samples=samples, metric_values=values)

    def run_validation(self, clean_volumes: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
        """Run the full §8 label-free validation ladder; emit per-metric cards.

        The shared degradation sweep is built **once** (it carries both the NR
        battery and the full-reference proxy columns), then every label-free tier
        reads off it:

        * Tier 5 defensibility ranking — the meta-evaluation pipeline output;
        * Tier 1 concordance (§8.1) — ``score_concordance`` on the sweep;
        * Tier 2 FR-proxy (§8.2) — ``score_fr_proxy`` (NR↔FR Spearman + VIF);
        * Tier 3 CHO (§8.3) — ``score_cho_concordance`` (own lesion task);
        * Tier 4 construct validity (§8.4) — ``score_construct_validity`` over the
          primary sweep + a seed-offset replicate sweep, plus the redundancy gate;
        * §8.7 aggregator — ``fit_nr_quality_index`` predicting severity θ.

        Returns a JSON-able dict: the ranking summary, a per-metric card
        (pass/fail across the tiers that were run), the redundancy report, the
        aggregator metrics, and the resolved knob provenance.
        """
        import logging

        logging.getLogger(__name__).warning(
            "NR metric battery is RESEARCH-MODE / validation-pending: this offline "
            "multi-tier harness is HOW the battery is validated; do not treat the "
            "metrics as production-certified, and never enable them during training."
        )

        from spectramr.application.use_cases.nr_validation import (
            build_cards,
            redundancy_gate,
            score_cho_concordance,
            score_concordance,
            score_construct_validity,
            score_fr_proxy,
        )
        from spectramr.core.metrics.meta_evaluation.metric_adapter import (
            build_safe_metric_set,
        )
        from spectramr.core.metrics.meta_evaluation.pipeline import (
            MetaEvaluationPipeline,
        )
        from spectramr.core.metrics.meta_evaluation.simulator import SimulatorConfig
        from spectramr.core.metrics.meta_evaluation.types import MetricSet

        clean_volumes = list(clean_volumes)
        nr_names = list(self.config.metrics)
        tiers = tuple(self.config.tiers)

        # FR proxies are folded into the sweep ONLY when Tier 2 runs — they are
        # the surrogate-truth columns, not metrics under validation.
        from spectramr.core.metrics import list_available

        available = set(list_available())
        fr_names = (
            [f for f in self.config.fr_proxies if f in available] if "fr_proxy" in tiers else []
        )
        sweep_metric_names = list(dict.fromkeys([*nr_names, *fr_names]))

        operator_library = build_physics_operator_library(self.config.families)

        # Primary sweep + Tier 5 ranking (the pipeline builds the dataset we reuse).
        metric_set = MetricSet(**build_safe_metric_set(sweep_metric_names))
        pipeline = MetaEvaluationPipeline.from_defaults()
        pipeline.eval_mode = self.config.eval_mode
        pipeline.aggregator = self.config.aggregator
        pipeline.simulator_config = SimulatorConfig(
            families=list(self.config.families),
            n_severities_per_family=self.config.n_severities_per_family,
            seed=self.config.seed,
            operator_library=operator_library,
        )
        output = pipeline.run(metric_set, clean_volumes)
        dataset_a = output.dataset

        per_tier: dict[str, dict[str, Any]] = {}
        if "concordance" in tiers:
            per_tier["concordance"] = score_concordance(
                dataset_a,
                nr_names,
                rho_threshold=self.config.concordance_rho_threshold,
            )
        if "fr_proxy" in tiers:
            per_tier["fr_proxy"] = score_fr_proxy(
                dataset_a,
                nr_names,
                fr_names or None,
                rho_threshold=self.config.fr_proxy_rho_threshold,
            )
        if "cho" in tiers:
            per_tier["cho"] = score_cho_concordance(
                clean_volumes,
                nr_names,
                families=list(self.config.cho_families),
                n_severities=self.config.cho_n_severities,
                n_ensemble=self.config.cho_n_ensemble,
                rho_threshold=self.config.cho_rho_threshold,
                seed=self.config.seed,
            )

        redundancy: dict[str, Any] | None = None
        if "construct_validity" in tiers:
            dataset_b = self._build_dataset(
                clean_volumes,
                sweep_metric_names,
                operator_library,
                seed=self.config.seed + self.config.retest_seed_offset,
            )
            per_tier["construct_validity"] = score_construct_validity(
                dataset_a,
                dataset_b,
                clean_volumes,
                nr_names,
                icc_threshold=self.config.icc_threshold,
                rel_tol=self.config.invariance_rel_tol,
            )
        if self.config.run_redundancy_gate:
            redundancy = redundancy_gate(
                dataset_a,
                nr_names,
                corr_threshold=self.config.redundancy_corr_threshold,
            )

        cards = build_cards(per_tier)

        aggregator: dict[str, Any] | None = None
        aggregator_model: dict[str, Any] | None = None
        if self.config.fit_aggregator:
            from spectramr.application.use_cases.nr_validation.aggregator import (
                fit_nr_quality_index,
                register_nr_quality_index,
            )

            model, agg_metrics = fit_nr_quality_index(dataset_a, nr_names, seed=self.config.seed)
            if self.config.register_aggregator:
                register_nr_quality_index(model)
                agg_metrics = {**agg_metrics, "registered_as": "nr_quality_index"}
            aggregator = agg_metrics
            # JSON-able serialisation so the CLI persists the fitted model
            # alongside the cards (§8.7; loadable via NRQualityIndexModel.from_dict).
            try:
                aggregator_model = model.to_dict()
            except Exception:
                aggregator_model = None

        return {
            "research_mode": True,
            "tiers_run": list(tiers),
            "n_samples": dataset_a.n_samples,
            "nr_metrics": nr_names,
            "fr_proxies": fr_names,
            "ranking": output.to_summary(),
            "cards": {m: card.to_dict() for m, card in cards.items()},
            "redundancy": redundancy,
            "aggregator": aggregator,
            "aggregator_model": aggregator_model,
            "provenance": {
                "seed": self.config.seed,
                "families": list(self.config.families),
                "n_severities_per_family": self.config.n_severities_per_family,
                "retest_seed_offset": self.config.retest_seed_offset,
                "thresholds": {
                    "concordance_rho": self.config.concordance_rho_threshold,
                    "fr_proxy_rho": self.config.fr_proxy_rho_threshold,
                    "cho_rho": self.config.cho_rho_threshold,
                    "icc": self.config.icc_threshold,
                    "invariance_rel_tol": self.config.invariance_rel_tol,
                    "redundancy_corr": self.config.redundancy_corr_threshold,
                },
            },
        }


__all__ = [
    "DEFAULT_FAMILIES",
    "DEFAULT_NR_BATTERY",
    "HARNESS_TIERS",
    "NRMetricValidationConfig",
    "NRMetricValidationUseCase",
    "build_physics_operator_library",
]
