"""Incoherence certificates: cycles and convex dominance, planted and absent."""

from __future__ import annotations

import pytest

from spectramr.core.metrics.meta_evaluation.blackwell.catalogue import (
    build_catalogue,
    info_groups,
    mixture,
)
from spectramr.core.metrics.meta_evaluation.blackwell.certificate import (
    find_cycle,
    incoherence_certificate,
    preference_closure,
)
from spectramr.core.metrics.meta_evaluation.blackwell.coherence import group_profiles

pytestmark = pytest.mark.unit
SHAPE = (24, 24)


def _trio():
    cat = {e.name: e for e in build_catalogue(SHAPE)}
    a, b = cat["noise_0.1"], cat["blur_low_noise_1"]
    exps = [a, b, mixture(a, b)]
    prof, _, exact = group_profiles(exps, info_groups(exps), 4096)
    return exps, prof, exact


def test_closure_follows_chains() -> None:
    c = preference_closure([(0, 1), (1, 2)], 3)
    assert c[0, 2] and not c[2, 0] and find_cycle(c) is None


def test_planted_cycle_is_named() -> None:
    exps, prof, exact = _trio()
    cert = incoherence_certificate(prof, [(0, 1), (1, 0)], [e.name for e in exps], exact=exact)
    assert cert["status"] == "cycle" and cert["cycle"][0] == cert["cycle"][-1]


def test_mixture_rated_below_both_members_yields_a_dominance_certificate() -> None:
    """Planted violation of Theorem C1.8(a): the certificate recovers the mixture."""
    exps, prof, exact = _trio()
    names = [e.name for e in exps]
    cert = incoherence_certificate(prof, [(0, 2), (1, 2)], names, exact=exact)
    assert cert["status"] == "dominance"
    first = cert["dominance"][0]
    assert first["dominated"] == names[2]
    weights = first["preferred_mixture"]
    assert set(weights) == {names[0], names[1]}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_coherent_preferences_have_no_certificate() -> None:
    exps, prof, exact = _trio()
    cert = incoherence_certificate(prof, [(2, 0), (2, 1)], [e.name for e in exps], exact=exact)
    assert cert["status"] == "none found" and cert["dominance"] == []


def test_dominance_is_not_checked_on_binned_profiles() -> None:
    exps, prof, _ = _trio()
    cert = incoherence_certificate(prof, [(0, 2), (1, 2)], [e.name for e in exps], exact=False)
    assert cert["status"] == "not checked (binned)"
