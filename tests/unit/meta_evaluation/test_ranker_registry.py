"""Tests for the unified ranker registry (one registry, one BaseRanker contract).

Phase 1 of the ranker unification: every ranker registers on ONE core registry
under the ``BaseRanker`` contract, tagged by ``generation`` (metadata, not a
pipeline fork). Today the Gen-3 structural rankers are registered; Gen-1/2/4
join as they are ported. ``MetaEvaluationPipeline.from_registry`` resolves every
available ranker — the wired entry point that replaces the hard-coded list.
"""

from __future__ import annotations

import pytest

from spectramr.core.metrics.meta_evaluation import (
    MetaEvaluationPipeline,
    available_rankers,
    check_data_dependencies,
    get_ranker,
    has_ranker,
    list_rankers,
    register_ranker,
)
from spectramr.core.metrics.meta_evaluation.rankers import registry as _registry_mod
from spectramr.core.metrics.meta_evaluation.rankers.base import BaseRanker
from spectramr.core.metrics.meta_evaluation.types import RankingResult

GEN3 = {"cdscr", "esd", "fspd", "lgdr", "sfa", "sim2rank"}


@pytest.fixture
def clean_registry():
    """Snapshot the global registry and drop any names a test adds."""
    before = set(_registry_mod._REGISTRY)
    yield
    for name in set(_registry_mod._REGISTRY) - before:
        del _registry_mod._REGISTRY[name]


class _DummyRanker(BaseRanker):
    method_name = "dummy"

    def rank(self, metric_set, dataset) -> RankingResult:
        scores = {m: 0.0 for m in metric_set.metrics}
        return RankingResult(
            method=self.method_name, scores=scores, ranks=list(scores), diagnostics={}
        )


# ---------------------------------------------------------------------------
# Registration contract
# ---------------------------------------------------------------------------


def test_gen3_structural_rankers_registered() -> None:
    names = {s.name for s in list_rankers()}
    assert GEN3 <= names
    for name in GEN3:
        spec = get_ranker(name)
        assert spec.generation == 3
        assert issubclass(spec.cls, BaseRanker)


def test_get_ranker_unknown_lists_names() -> None:
    with pytest.raises(KeyError, match="not registered"):
        get_ranker("does_not_exist")


def test_reregister_same_name_raises(clean_registry) -> None:
    register_ranker("dummy_x", generation=1)(_DummyRanker)
    with pytest.raises(ValueError, match="already registered"):
        register_ranker("dummy_x", generation=1)(_DummyRanker)


def test_register_non_baseranker_raises(clean_registry) -> None:
    class NotARanker:  # noqa: D401 - intentional non-conformant
        pass

    with pytest.raises(TypeError, match="must subclass BaseRanker"):
        register_ranker("bad", generation=1)(NotARanker)


def test_gen3_registration_site_does_not_guard_the_strict_raise() -> None:
    """REGRESSION #258: `if not has_ranker(_name)` defeated the strict-registry raise.

    The registry's documented contract is that re-registering a name RAISES. The
    guard turned a genuine double-registration (two classes racing for one name)
    into a silent no-op where the first registration wins.
    """
    import inspect

    from spectramr.core.metrics.meta_evaluation import rankers as pkg

    src = inspect.getsource(pkg)
    assert "if not has_ranker(_name)" not in src
    # And the strict raise really does fire on the live registry.
    with pytest.raises(ValueError, match="already registered"):
        register_ranker("cdscr", generation=3)(_DummyRanker)


def test_capability_flag_without_data_kwargs_raises(clean_registry) -> None:
    """REGRESSION #258: a requires_* flag with no data_kwargs is a decorative gate.

    Without the kwarg name, `from_registry` can only OMIT the ranker or hand back
    one whose rank() will raise — it can never SUPPLY the data.
    """
    with pytest.raises(ValueError, match="no data_kwargs"):
        register_ranker("bad_gate", generation=4, requires_task_net=True)(_DummyRanker)

    with pytest.raises(ValueError, match="no data_kwargs"):
        register_ranker("bad_gate2", generation=4, requires_likert=True)(_DummyRanker)


# ---------------------------------------------------------------------------
# Capability gating
# ---------------------------------------------------------------------------


def test_available_rankers_gates_on_likert(clean_registry) -> None:
    register_ranker(
        "needs_likert",
        generation=4,
        requires_likert=True,
        data_kwargs=("likert",),
    )(_DummyRanker)
    without = {s.name for s in available_rankers()}
    assert "needs_likert" not in without
    withl = {s.name for s in available_rankers(have_likert=True)}
    assert "needs_likert" in withl
    # Gen-3 rankers with no deps are always available. `sim2rank` is the one
    # exception: it is a FUSION of adr/scvr/cdrs and is dropped while all three of
    # its constituents vote (issue #256).
    assert (GEN3 - {"sim2rank"}) <= without
    assert "sim2rank" not in without


def test_check_data_dependencies_reports_unmet(clean_registry) -> None:
    register_ranker(
        "needs_task",
        generation=4,
        requires_task_net=True,
        data_kwargs=("task_performance",),
    )(_DummyRanker)
    errs = check_data_dependencies(
        ["needs_task", "cdscr", "ghost"], have_likert=False, have_task_net=False
    )
    assert any("needs_task" in e for e in errs)  # unmet task-net dep
    assert any("ghost" in e for e in errs)  # unregistered name
    assert not any("cdscr" in e for e in errs)  # satisfied


def test_check_injected_data_catches_declared_but_absent_ground_truth(
    clean_registry,
) -> None:
    """REGRESSION #258: requires_task_net=True never checked that data ARRIVED."""
    from spectramr.core.metrics.meta_evaluation.rankers import check_injected_data

    register_ranker(
        "needs_task2",
        generation=4,
        requires_task_net=True,
        data_kwargs=("task_performance",),
    )(_DummyRanker)
    spec = get_ranker("needs_task2")

    assert check_injected_data([spec], {})  # nothing injected -> error
    assert check_injected_data([spec], {"needs_task2": {"task_performance": None}})
    assert not check_injected_data([spec], {"needs_task2": {"task_performance": [1.0]}})


# ---------------------------------------------------------------------------
# Wired pipeline entry point
# ---------------------------------------------------------------------------


def test_from_registry_builds_available_rankers() -> None:
    pipe = MetaEvaluationPipeline.from_registry()
    methods = {type(r).__name__ for r in pipe.rankers}
    # Gen-3 structural + Gen-1 statistical are the always-present core.
    core = {
        "CDSCRRanker",
        "ESDRanker",
        "FSPDRanker",
        "LGDRRanker",
        "SFARanker",
        "ADRRanker",
        "SCVRRanker",
        "CDRSRanker",
    }
    assert core <= methods
    # The Task ranker is capability-gated on a frozen task net (issue #253): it
    # must NOT appear in a pool that has no measured task-performance curve.
    assert "TaskScoreRanker" not in methods
    # Nor may any Gen-4 clinical ranker (no Likert / task net here).
    assert {"DCMSRanker", "ACSSRanker", "HiBSLRanker"}.isdisjoint(methods)


def test_from_registry_drops_the_redundant_fusion_ranker() -> None:
    """REGRESSION #256: Sim2Rank == 0.5*ADR + 0.3*SCVR + 0.2*CDRS.

    Keeping the fusion next to all three of its constituents gave ADR 1.5x the
    weight of a genuinely independent ranker, SCVR 1.3x and CDRS 1.2x — so 4 of the
    14 "independent" voters were one axis counted twice, and the "N independent
    rankers concur" claim was inflated.
    """
    from spectramr.core.metrics.meta_evaluation.rankers import redundant_rankers

    specs = available_rankers()
    names = {s.name for s in specs}
    assert {"adr", "scvr", "cdrs"} <= names  # the constituents vote ...
    assert "sim2rank" not in names  # ... so the fusion does not.
    assert not redundant_rankers(specs)  # the resolved pool is non-redundant

    # The fusion is still registered, and still declares what it fuses.
    assert get_ranker("sim2rank").fuses == ("adr", "scvr", "cdrs")
    # Opting out reproduces the legacy (redundant) pool.
    legacy = {s.name for s in available_rankers(drop_redundant_fusions=False)}
    assert "sim2rank" in legacy


def test_from_registry_gate_raises_when_data_not_injected() -> None:
    """REGRESSION #258: the gate could say 'omit' or 'crash', never 'supply'.

    ``from_registry(have_task_net=True)`` used to pass ``**kwargs`` to the PIPELINE
    dataclass, not to ``spec.instantiate()`` — so it built
    ``DCMSRanker(task_performance=None)`` and ``run()`` died with "inject via
    from_registry override", an override that did not exist.
    """
    with pytest.raises(ValueError, match="was never injected"):
        MetaEvaluationPipeline.from_registry(have_task_net=True)

    with pytest.raises(ValueError, match="was never injected"):
        MetaEvaluationPipeline.from_registry(have_likert=True)


def test_from_registry_injects_ranker_kwargs() -> None:
    """The Gen-4 core-pipeline path is reachable now: data actually gets in."""
    import numpy as np

    pipe = MetaEvaluationPipeline.from_registry(
        have_task_net=True,
        ranker_kwargs={
            "dcms": {"task_performance": np.zeros(4)},
            "task": {"task_perf_drop": {"seg": [0.0, 0.5]}},
        },
    )
    by_name = {type(r).__name__: r for r in pipe.rankers}
    assert by_name["DCMSRanker"].task_performance is not None
    assert by_name["TaskScoreRanker"].config.task_perf_drop is not None


def test_from_registry_rejects_unknown_ranker_kwargs() -> None:
    with pytest.raises(ValueError, match="not in the resolved pool"):
        MetaEvaluationPipeline.from_registry(ranker_kwargs={"nope": {"x": 1}})


def test_from_registry_runs_end_to_end(metric_set, clean_volumes) -> None:
    from spectramr.core.metrics.meta_evaluation.simulator import SimulatorConfig

    pipe = MetaEvaluationPipeline.from_registry()
    pipe.simulator_config = SimulatorConfig(
        families=["gaussian_noise", "gaussian_blur"], n_severities_per_family=3, seed=0
    )
    out = pipe.run(metric_set, clean_volumes)
    assert out.aggregate.final_ranks

    # #256: the ranker-pool redundancy is MEASURED and REPORTED, so a future
    # near-clone voter cannot silently double an axis's weight again.
    red = out.ranker_redundancy
    assert set(red["methods"]) == {r.method for r in out.per_method}
    assert red["spearman"]
    assert "flagged_pairs" in red
    assert red["threshold"] == pytest.approx(0.95)
    assert "ranker_redundancy" in out.to_summary()


def test_opt_in_ranker_is_registered_but_never_pooled(clean_registry) -> None:
    pool_before = [s.name for s in available_rankers()]
    register_ranker("dummy_opt_in", generation=5, opt_in=True)(_DummyRanker)
    assert get_ranker("dummy_opt_in").opt_in
    assert [s.name for s in available_rankers()] == pool_before
    assert [s.name for s in available_rankers(have_likert=True, have_task_net=True)].count(
        "dummy_opt_in"
    ) == 0
