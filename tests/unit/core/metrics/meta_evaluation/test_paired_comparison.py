"""lesion_vs_control: the paired test, and why unpaired would be wrong."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.meta_evaluation.paired_comparison import (  # noqa: E402
    _tied_ranks,
    lesion_vs_control,
    spearman_rho,
)


class TestSpearman:
    def test_a_monotone_relation_is_plus_one(self) -> None:
        assert spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_an_antitone_relation_is_minus_one(self) -> None:
        assert spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_it_is_rank_based_not_linear(self) -> None:
        assert spearman_rho([1, 2, 3, 4], [1, 10, 100, 1000]) == pytest.approx(1.0)

    def test_a_constant_input_is_nan_not_zero(self) -> None:
        """A metric with zero rank variance has no correlation. Reporting 0.0 would
        claim 'this metric is uncorrelated with severity' -- a FINDING -- when the
        truth is that the quantity is undefined."""
        assert math.isnan(spearman_rho([1, 2, 3], [5, 5, 5]))

    def test_too_few_points_is_nan(self) -> None:
        assert math.isnan(spearman_rho([1], [2]))

    def test_ties_get_midranks_not_an_invented_order(self) -> None:
        """Regression. `argsort().argsort()` is NOT a rank function -- it breaks ties
        arbitrarily, turning [5,5,5] into ranks [0,1,2] and inventing an ordering
        that is not in the data. A saturated or dead metric is constant by
        definition, so this would hand it a confident +-1 correlation that is a pure
        artifact of tie-break order."""
        # A fully tied y has no rank variance -- undefined, not +-1.
        assert math.isnan(spearman_rho([1, 2, 3, 4], [7, 7, 7, 7]))

        # Tied values share the midrank of the block they span.
        assert _tied_ranks(torch.tensor([1.0, 1.0, 2.0, 2.0])).tolist() == [
            0.5,
            0.5,
            2.5,
            2.5,
        ]

        # A tied y CANNOT reach |rho| = 1 against an untied x -- ties genuinely
        # reduce the attainable correlation. -4/sqrt(20) = -0.894, and reporting a
        # confident -1.0 here would overstate the evidence.
        assert spearman_rho([1, 2, 3, 4], [2, 2, 1, 1]) == pytest.approx(-4 / math.sqrt(20))

        # Sanity: with no ties, a reversal is still exactly -1.
        assert spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_a_nan_input_propagates(self) -> None:
        assert math.isnan(spearman_rho([1, 2, 3], [1, float("nan"), 3]))

    def test_misaligned_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            spearman_rho([1, 2], [1, 2, 3])


def _paired(lesion_gain: float, n_slices: int = 40, steps: int = 8, seed: int = 0):
    """Severity sweep over `n_slices` slices; the lesion tracks severity `lesion_gain`
    times as strongly as the control."""
    g = torch.Generator().manual_seed(seed)
    sev, les, ctl, sids = [], [], [], []
    for s in range(n_slices):
        offset = float(torch.randn(1, generator=g))  # per-slice difficulty
        for t in range(steps):
            theta = t / (steps - 1)
            sev.append(theta)
            noise_l = float(torch.randn(1, generator=g)) * 0.05
            noise_c = float(torch.randn(1, generator=g)) * 0.05
            les.append(offset + lesion_gain * theta + noise_l)
            ctl.append(offset + 1.0 * theta + noise_c)
            sids.append(f"slice_{s}")
    return sev, les, ctl, sids


class TestPairedDelta:
    def test_no_tissue_difference_gives_a_ci_that_contains_zero(self) -> None:
        sev, les, ctl, sids = _paired(lesion_gain=1.0)
        d = lesion_vs_control(
            "psnr",
            "gaussian",
            "AXT2",
            severity=sev,
            lesion_scores=les,
            control_scores=ctl,
            slice_ids=sids,
            n_boot=300,
            seed=1,
        )
        assert d.ci_low <= 0.0 <= d.ci_high
        assert not d.significant

    def test_a_real_tissue_difference_is_detected(self) -> None:
        """The lesion tracks severity in the opposite direction from the control."""
        sev, les, ctl, sids = _paired(lesion_gain=-3.0)
        d = lesion_vs_control(
            "psnr",
            "gaussian",
            "AXT2",
            severity=sev,
            lesion_scores=les,
            control_scores=ctl,
            slice_ids=sids,
            n_boot=300,
            seed=1,
        )
        assert d.rho_lesion < d.rho_control
        assert d.delta < 0
        assert d.significant
        assert d.ci_high < 0.0

    def test_the_bootstrap_resamples_slices_not_observations(self) -> None:
        """Resampling observations would break the pairing: a draw could hold a
        lesion score from slice 7 and a control score from slice 12, which is not a
        difference between tissues at all."""
        sev, les, ctl, sids = _paired(lesion_gain=1.0, n_slices=6)
        d = lesion_vs_control(
            "psnr",
            "gaussian",
            "AXT2",
            severity=sev,
            lesion_scores=les,
            control_scores=ctl,
            slice_ids=sids,
            n_boot=50,
            seed=0,
        )
        assert d.n_slices == 6  # the resampling unit, not len(sev) == 48

    def test_it_is_deterministic_under_a_seed(self) -> None:
        sev, les, ctl, sids = _paired(lesion_gain=2.0)
        kw = {
            "severity": sev,
            "lesion_scores": les,
            "control_scores": ctl,
            "slice_ids": sids,
            "n_boot": 100,
            "seed": 7,
        }
        a = lesion_vs_control("psnr", "gaussian", "AXT2", **kw)
        b = lesion_vs_control("psnr", "gaussian", "AXT2", **kw)
        assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


class TestMisalignmentRaises:
    def test_ragged_inputs_raise(self) -> None:
        """A misalignment would compare a lesion on one slice against a control on
        another -- which is not a tissue difference at all."""
        with pytest.raises(ValueError, match="not a tissue difference"):
            lesion_vs_control(
                "psnr",
                "gaussian",
                "AXT2",
                severity=[0.0, 1.0],
                lesion_scores=[1.0, 2.0],
                control_scores=[1.0],
                slice_ids=["a", "b"],
            )


class TestDegenerateInputs:
    def test_a_single_slice_yields_no_ci(self) -> None:
        """One slice cannot be bootstrapped -- there is nothing to resample."""
        d = lesion_vs_control(
            "psnr",
            "gaussian",
            "AXT2",
            severity=[0.0, 0.5, 1.0],
            lesion_scores=[3.0, 2.0, 1.0],
            control_scores=[3.0, 2.0, 1.0],
            slice_ids=["s"] * 3,
        )
        assert d.n_slices == 1
        assert math.isnan(d.ci_low)
        assert not d.significant  # a NaN CI is never 'significant'

    def test_a_constant_metric_yields_no_delta(self) -> None:
        d = lesion_vs_control(
            "flat",
            "gaussian",
            "AXT2",
            severity=[0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            lesion_scores=[1.0] * 6,
            control_scores=[3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
            slice_ids=["a", "a", "a", "b", "b", "b"],
        )
        assert math.isnan(d.rho_lesion)
        assert not d.significant


class TestSubjectResampling:
    """Slices of one volume share anatomy, so the bootstrap unit is the subject."""

    def test_the_resampling_units_are_the_subjects(self) -> None:
        sev, les, ctl, sids = _paired(lesion_gain=1.0, n_slices=6)
        subject_of = {s: f"subj{int(s.split('_')[-1]) // 3}" for s in set(sids)}
        d = lesion_vs_control(
            "psnr",
            "gaussian",
            "AXT2",
            severity=sev,
            lesion_scores=les,
            control_scores=ctl,
            slice_ids=sids,
            subject_ids=[subject_of[s] for s in sids],
            n_boot=50,
            seed=0,
        )
        assert d.n_slices == 6
        assert d.resampling_unit == "subject"
        assert d.n_units == 2
        assert d.as_row()["n_units"] == 2

    def test_without_subjects_the_unit_is_the_slice(self) -> None:
        sev, les, ctl, sids = _paired(lesion_gain=1.0, n_slices=6)
        d = lesion_vs_control(
            "psnr",
            "gaussian",
            "AXT2",
            severity=sev,
            lesion_scores=les,
            control_scores=ctl,
            slice_ids=sids,
            n_boot=20,
        )
        assert (d.resampling_unit, d.n_units) == ("slice", 6)

    def test_a_slice_under_two_subjects_raises(self) -> None:
        with pytest.raises(ValueError, match="subject"):
            lesion_vs_control(
                "psnr",
                "gaussian",
                "AXT2",
                severity=[0.0, 1.0],
                lesion_scores=[1.0, 2.0],
                control_scores=[1.0, 2.0],
                slice_ids=["a", "a"],
                subject_ids=["s1", "s2"],
            )

    def test_an_empty_subject_raises(self) -> None:
        with pytest.raises(ValueError, match="empty subject"):
            lesion_vs_control(
                "psnr",
                "gaussian",
                "AXT2",
                severity=[0.0, 1.0],
                lesion_scores=[1.0, 2.0],
                control_scores=[1.0, 2.0],
                slice_ids=["a", "b"],
                subject_ids=["s1", ""],
            )
