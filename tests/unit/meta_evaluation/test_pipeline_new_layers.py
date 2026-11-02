"""End-to-end integration of the four new reliability layers in the pipeline.

Exercises the L2⁺ betting certification, L3⁺ Kemeny aggregator, and L4 sim-to-real
transfer certificate through the real :class:`MetaEvaluationPipeline.run`, on the
shared synthetic dataset fixtures (``clean_volumes`` / ``metric_set``).
"""

from __future__ import annotations

import pytest

from spectramr.core.metrics.meta_evaluation import (
    MetaEvaluationPipeline,
    resolve_eval_mode,
)
from spectramr.core.metrics.meta_evaluation.pipeline import MetaEvaluationPipeline as Pipe
from spectramr.core.metrics.meta_evaluation.simulator import SimulatorConfig


def _pipeline(**kw) -> MetaEvaluationPipeline:
    base = MetaEvaluationPipeline.from_defaults()
    cfg = SimulatorConfig(
        families=["gaussian_noise", "gaussian_blur", "gamma", "kspace_undersample"],
        n_severities_per_family=4,
        seed=7,
    )
    return Pipe(rankers=base.rankers, simulator_config=cfg, **kw)


def test_default_pipeline_uses_z_aggregator(metric_set, clean_volumes) -> None:
    out = _pipeline().run(metric_set, clean_volumes)
    assert out.aggregate.final_ranks  # non-empty
    assert out.betting == {}  # off by default


def test_kemeny_aggregator_runs_end_to_end(metric_set, clean_volumes) -> None:
    out = _pipeline(aggregator="kemeny").run(metric_set, clean_volumes)
    assert sorted(out.aggregate.final_ranks) == sorted(metric_set.names())
    # mse (real distortion metric) should not be ranked last by the consensus.
    assert out.aggregate.final_ranks[-1] != "mse"


def test_unknown_aggregator_raises(metric_set, clean_volumes) -> None:
    import pytest

    with pytest.raises(ValueError):
        _pipeline(aggregator="bogus").run(metric_set, clean_volumes)


def test_betting_certification_emitted_when_enabled(metric_set, clean_volumes) -> None:
    out = _pipeline(use_betting=True).run(metric_set, clean_volumes)
    assert "aggregate" in out.betting
    bo = out.betting["aggregate"]
    assert bo.n_certified + bo.n_ambiguous == 6  # C(4, 2) pairs over 4 metrics


def test_transfer_certificate_perfect_twin(metric_set, clean_volumes) -> None:
    # A "real" cohort byte-identical to the simulated one -> TV = 0 -> every
    # strictly-ordered pair transfers (Lean ranking_transfer_recovers_sim).
    # Reuse the run's own metric_values as the real cohort so the twin is exact
    # (re-running the simulator would diverge on the nondeterministic ``noise``
    # metric, which is the point of L4 — a non-tracking metric inflates the TV).
    out = _pipeline().run(metric_set, clean_volumes)
    real_values = dict(out.dataset.metric_values)
    pipe = _pipeline()
    certs = pipe.transfer_certificates(out, real_values)
    assert len(certs) == 6  # C(4, 2)
    # identical cohort -> tv == 0 -> all transfer
    assert all(c.transfers for c in certs)


def test_transfer_certificate_shifted_cohort(metric_set, clean_volumes) -> None:
    # A real cohort with shifted metric values -> larger TV -> fewer transfers.
    out = _pipeline().run(metric_set, clean_volumes)
    shifted = {
        m: [v + 100.0 for v in vals] for m, vals in out.dataset.metric_values.items()
    }
    pipe = _pipeline()
    certs = pipe.transfer_certificates(out, shifted)
    # Large law shift: at least one close pair should fail to transfer.
    assert any(not c.transfers for c in certs)


def test_transfer_certificate_tolerates_crashed_metric(
    metric_set, clean_volumes
) -> None:
    # crash->NaN contract: a metric whose real cohort is partly NaN (crashed on some
    # samples) must not abort transfer_certificates with "cannot convert float NaN to
    # integer". The finite subsample still yields a valid TV (here, a perfect twin).
    out = _pipeline().run(metric_set, clean_volumes)
    real_values = {m: list(vals) for m, vals in out.dataset.metric_values.items()}
    poisoned = next(iter(real_values))
    real_values[poisoned][0] = float("nan")  # one crashed sample
    certs = _pipeline().transfer_certificates(out, real_values)
    assert len(certs) == 6  # C(4, 2) — produced, not crashed
    assert all(c.transfers for c in certs)  # twin minus one dropped sample -> TV ~ 0


def test_transfer_certificate_skips_fully_crashed_metric(
    metric_set, clean_volumes
) -> None:
    # A metric that crashed on the WHOLE real cohort (all NaN) has no estimable TV.
    # It must be SKIPPED, not (a) crash the run, nor (b) dominate the max-pool and
    # spuriously void every certificate. Result equals the perfect-twin case.
    out = _pipeline().run(metric_set, clean_volumes)
    real_values = {m: list(vals) for m, vals in out.dataset.metric_values.items()}
    dead = next(iter(real_values))
    real_values[dead] = [float("nan")] * len(real_values[dead])  # whole metric dead
    certs = _pipeline().transfer_certificates(out, real_values)
    assert len(certs) == 6
    # The surviving metrics are a perfect twin -> every pair still transfers.
    assert all(c.transfers for c in certs)


def test_transfer_certificate_joint_tv_failure_is_logged_not_swallowed(
    metric_set, clean_volumes, monkeypatch, caplog
) -> None:
    # #298: the joint-TV term was wrapped in contextlib.suppress(Exception), so a
    # failing joint estimator silently dropped the tighter bound, leaving only the
    # marginal-max TV, which UNDER-estimates the joint TV and certifies MORE pairs
    # (an optimistic certificate). The fallback must still produce a certificate
    # (never abort) BUT the downgrade must be AUDITABLE — a named warning.
    import spectramr.core.metrics.meta_evaluation.pipeline as pipeline_mod

    def _boom(*_a, **_k):
        raise RuntimeError("joint estimator exploded")

    monkeypatch.setattr(pipeline_mod, "tv_distance_joint_samples", _boom)

    out = _pipeline().run(metric_set, clean_volumes)
    real_values = dict(out.dataset.metric_values)
    with caplog.at_level("WARNING"):
        certs = _pipeline().transfer_certificates(out, real_values)

    assert len(certs) == 6, "the certificate must still be produced, never aborted"
    assert "joint-TV estimator failed" in caplog.text, "the downgrade must be logged"
    assert "#298" in caplog.text


def test_to_summary_includes_betting(metric_set, clean_volumes) -> None:
    out = _pipeline(use_betting=True).run(metric_set, clean_volumes)
    summary = out.to_summary()
    assert "betting" in summary
    assert "aggregate" in summary["betting"]
    assert set(summary["final_mid_ranks"]) == set(summary["final_ranks"])


def test_betting_increments_are_per_sample_not_per_method(
    metric_set, clean_volumes
) -> None:
    """A1: betting increments come from the severity trajectories, not a length-6
    sign-vs-median-z proxy. They are per-sample (anytime-valid) and aligned across
    metrics (identical length), which the paired confidence sequence relies on."""
    pipe = _pipeline(use_betting=True)
    out = pipe.run(metric_set, clean_volumes)
    inc = pipe._severity_concordance_increments(out.dataset, metric_set.higher_is_better)
    # One adjacent-pair increment per (content, family) sub-sweep, summed.
    groups = {(s.content_id, s.degradation_family) for s in out.dataset.samples}
    expected_len = out.dataset.n_samples - len(groups)
    lengths = {len(v) for v in inc.values()}
    assert lengths == {expected_len}  # all metrics aligned, same length
    # Far longer than the old per-ranker proxy (len == number of rankers == 6).
    assert expected_len > 6
    # Increments are bounded concordance signs.
    assert all(all(x in (-1.0, 0.0, 1.0) for x in v) for v in inc.values())


def test_to_summary_stamps_defensibility_semantics(metric_set, clean_volumes) -> None:
    """A2: the summary self-describes which event ``defensibility_flags`` encodes."""
    z_summary = _pipeline().run(metric_set, clean_volumes).to_summary()
    assert z_summary["defensibility_semantics"] == "breadth"
    k_summary = (
        _pipeline(aggregator="kemeny").run(metric_set, clean_volumes).to_summary()
    )
    assert k_summary["defensibility_semantics"] == "condorcet"


def test_compute_betting_shared_helper(metric_set, clean_volumes) -> None:
    """compute_betting (shared with the legacy novel-meta path) gates on use_betting."""
    off = _pipeline()
    out = off.run(metric_set, clean_volumes)
    assert off.compute_betting(out.dataset) == {}
    on = _pipeline(use_betting=True)
    out2 = on.run(metric_set, clean_volumes)
    assert "aggregate" in on.compute_betting(out2.dataset, metric_set.higher_is_better)


# ---------------------------------------------------------------------------
# eval_mode dimensionality knob (pitfall #15 — wired, validated, stamped)
# ---------------------------------------------------------------------------


def test_resolve_eval_mode_2d_passthrough() -> None:
    assert resolve_eval_mode("2d") == "2d"


def test_resolve_eval_mode_3d_fails_loud() -> None:
    """'3d' is deferred — it must raise, never silently degrade to 2-D."""
    with pytest.raises(NotImplementedError, match="3d"):
        resolve_eval_mode("3d")


def test_resolve_eval_mode_unknown_raises() -> None:
    with pytest.raises(ValueError, match="not a recognised evaluation mode"):
        resolve_eval_mode("2.5d")


def test_default_eval_mode_is_2d_and_stamped(metric_set, clean_volumes) -> None:
    out = _pipeline().run(metric_set, clean_volumes)
    assert out.eval_mode == "2d"
    assert out.to_summary()["eval_mode"] == "2d"


def test_pipeline_run_3d_fails_loud_before_work(metric_set, clean_volumes) -> None:
    """A pipeline configured for 3d raises at run() start, not mid-sweep."""
    with pytest.raises(NotImplementedError, match="3d"):
        _pipeline(eval_mode="3d").run(metric_set, clean_volumes)
