"""Revealed-design LP: the control is coherent, planted incoherence is caught."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectramr.core.metrics.meta_evaluation.blackwell.catalogue import (
    build_catalogue,
    info_groups,
    mixture,
)
from spectramr.core.metrics.meta_evaluation.blackwell.coherence import (
    group_profiles,
    preferences_by_t,
    revealed_design_lp,
)
from spectramr.core.metrics.meta_evaluation.blackwell.evaluate import oracle_lsi_control
from spectramr.core.metrics.meta_evaluation.blackwell.order import certify

pytestmark = pytest.mark.unit
SHAPE = (24, 24)


def _contents() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return [torch.rand(SHAPE, generator=g) + 0.5 for _ in range(3)]


def test_oracle_control_preferences_are_coherent_and_verified() -> None:
    cat = build_catalogue(SHAPE)
    prof, sizes, exact = group_profiles(cat, info_groups(cat), 4096)
    prefs = preferences_by_t(oracle_lsi_control(_contents(), cat), 3.0)
    res = revealed_design_lp(prof, sizes, prefs, exact=exact)
    assert exact and res.coherent is True and res.status == "feasible"
    assert res.mass is not None and res.mass > 0


def test_preferring_a_dominated_experiment_is_incoherent() -> None:
    """Planted violation: strictly preferring the less informative side of a relation."""
    cat = build_catalogue(SHAPE)
    prof, sizes, exact = group_profiles(cat, info_groups(cat), 4096)
    better, worse = certify(cat).strict[0]
    res = revealed_design_lp(prof, sizes, [(worse, better)], exact=exact)
    assert res.coherent is False and res.status == "infeasible"


def test_rating_a_mixture_below_both_members_is_incoherent() -> None:
    """Planted violation of Theorem C1.8(a): a convex combination rated below both ends."""
    cat = build_catalogue(SHAPE)
    names = [e.name for e in cat]
    a, b = cat[names.index("noise_0.1")], cat[names.index("blur_low_noise_1")]
    trio = [a, b, mixture(a, b)]
    prof, sizes, exact = group_profiles(trio, info_groups(trio), 4096)
    assert revealed_design_lp(prof, sizes, [(0, 2), (1, 2)], exact=exact).coherent is False
    assert revealed_design_lp(prof, sizes, [(2, 0), (2, 1)], exact=exact).coherent is True


def test_binned_infeasibility_is_inconclusive() -> None:
    cat = build_catalogue(SHAPE)
    prof, sizes, exact = group_profiles(cat, info_groups(cat), 4)
    assert not exact and prof.shape[1] <= 4
    better, worse = certify(cat).strict[0]
    res = revealed_design_lp(prof, sizes, [(worse, better)], exact=exact)
    assert res.coherent is None


def test_group_profiles_conserve_total_information() -> None:
    cat = build_catalogue(SHAPE)[:3]
    prof, sizes, _ = group_profiles(cat, info_groups(cat), 4096)
    assert sizes.sum() == SHAPE[0] * SHAPE[1]
    for e, row in zip(cat, prof, strict=True):
        assert np.isclose(row.sum(), float(torch.exp(e.log_info).sum()), rtol=1e-10)


def test_preferences_by_t_reads_consistent_gaps_only() -> None:
    v = torch.tensor([[3.0, 1.0, 2.0], [3.1, 1.2, 1.0], [2.9, 0.9, 3.0]])
    prefs = set(preferences_by_t(v, 3.0))
    assert (0, 1) in prefs and (0, 2) not in prefs and (2, 1) not in prefs
    with pytest.raises(ValueError):
        preferences_by_t(v[:1], 3.0)


def test_no_preferences_is_trivially_coherent() -> None:
    res = revealed_design_lp(np.ones((2, 3)), np.ones(3), [])
    assert res.coherent is True and res.n_preferences == 0


def test_preferences_by_t_drops_a_nan_content_per_pair_only() -> None:
    """Regression: one content a metric declines (NaN) used to erase every preference."""
    v = torch.tensor([[3.0, 1.0], [3.1, 1.2], [2.9, 0.9], [float("nan"), float("nan")]])
    assert preferences_by_t(v, 3.0) == [(0, 1)]


def test_split_preferences_sets_aside_equivalent_and_reversed_pairs() -> None:
    from spectramr.core.metrics.meta_evaluation.blackwell.coherence import split_preferences
    from spectramr.core.metrics.meta_evaluation.blackwell.order import RelationGraph

    graph = RelationGraph(strict=((0, 1),), equivalent=((2, 3),), incomparable=((0, 2),))
    kept, n_eq, n_rev = split_preferences([(0, 1), (1, 0), (3, 2), (0, 2)], graph)
    assert kept == [(0, 1), (0, 2)] and n_eq == 1 and n_rev == 1
