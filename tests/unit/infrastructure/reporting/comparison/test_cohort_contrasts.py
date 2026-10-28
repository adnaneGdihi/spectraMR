"""``comparison/cohort_contrasts.py``: contrasts for any cohort directory, without a file."""

import pytest

from spectramr.infrastructure.reporting.comparison.cohort_contrasts import cohort_contrasts

from ._runs import make_run


@pytest.fixture
def cohort(tmp_path):
    for name in ("alpha", "beta", "gamma"):
        make_run(tmp_path, name, pngs=False)
    return tmp_path


def test_vs_baseline_tests_every_other_arm_in_one_holm_family(cohort):
    cs = cohort_contrasts(cohort, baseline="alpha")
    assert [c.name for c in cs.contrasts] == ["beta__vs__alpha", "gamma__vs__alpha"]
    assert {c.family for c in cs.contrasts} == {"vs_alpha"}
    assert all(c.baseline_label == "alpha" for c in cs.contrasts)
    assert cs.metrics is None


def test_pairs_writes_every_pair_uncorrected(cohort):
    cs = cohort_contrasts(cohort, mode="pairs")
    assert len(cs.contrasts) == 3
    assert {c.family for c in cs.contrasts} == {None}
    assert {frozenset(c.labels) for c in cs.contrasts} == {
        frozenset(p) for p in (("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma"))
    }


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "names the arm"),
        ({"baseline": "nope"}, "not a run"),
        ({"mode": "all"}, "unknown contrast mode"),
    ],
)
def test_an_incomplete_request_raises(cohort, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cohort_contrasts(cohort, **kwargs)


def test_a_cohort_with_one_run_raises(tmp_path):
    make_run(tmp_path, "only", pngs=False)
    with pytest.raises(ValueError, match="at least two"):
        cohort_contrasts(tmp_path, mode="pairs")


def test_the_baseline_is_its_own_figure_group_and_the_others_are_ungrouped(cohort):
    contrast = cohort_contrasts(cohort, baseline="alpha").contrasts[0]
    assert contrast.group_map == {"alpha": "baseline"}
