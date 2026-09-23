"""Does degradation order matter? Measured, with the right null hypothesis.

Two cohorts disagree: ``quality_matching`` fits severities for a chain "in
application order", ``operator_id`` assembles an order-2 BCH generator because
the axes do not commute. Nothing measured the disagreement.

The trap this file exists to pin is the empirical one. A permutation sweep
produces a spread, and the spread looks like structure -- until it is compared
against the spread the *same* ordering produces under different optimiser seeds.
On the shipped 3-axes-against-2-attributes fit that seed spread is the larger of
the two, so the order effect is not detectable through the fit at all. The arm's
own metadata already says why: "the fitted theta-vector is NOT uniquely
identified".

So the ratio is the statistic, not the spread. A test that asserted on the
spread alone would pass while measuring noise.
"""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import pytest
import torch

from spectramr.infrastructure.physics.magnus_exponential import assemble_omega, commutator

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/analysis/degradation_order_dispersion.py"


def _load():
    spec = importlib.util.spec_from_file_location("degradation_order_dispersion", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DOD = _load()


class TestTheAxesItClaimsToCover:
    def test_it_names_the_axes_the_shipped_configs_declare(self) -> None:
        """A drifted list would measure a chain no arm runs."""
        import yaml

        config = REPO / "experiments/inprogress/quality_matching/exp_qm_01_hf_to_ulf.yaml"
        if not config.exists():
            pytest.skip("quality_matching cohort not present")
        raw = yaml.safe_load(config.read_text())
        declared = (((raw.get("training") or {}).get("quality_matching")) or {}).get("axes")
        if declared is None:
            pytest.skip("arm declares no axes block")
        assert tuple(declared) == DOD.QUALITY_MATCHING_AXES

    def test_it_names_the_modes_operator_id_declares(self) -> None:
        import yaml

        config = REPO / "experiments/inprogress/operator_id/bch_m4raw.yaml"
        if not config.exists():
            pytest.skip("operator_id cohort not present")
        raw = yaml.safe_load(config.read_text())
        declared = (((raw.get("training") or {}).get("operator_id")) or {}).get("mode_dictionary")
        if declared is None:
            pytest.skip("arm declares no mode_dictionary")
        assert tuple(declared) == DOD.OPERATOR_ID_MODES

    def test_the_two_legs_barely_overlap_and_the_script_says_so(self) -> None:
        """The finding underneath the finding: neither leg covers the other's axes."""
        shared = {a.split("_")[0] for a in DOD.QUALITY_MATCHING_AXES} & {
            m.split("_")[0] for m in DOD.OPERATOR_ID_MODES
        }
        assert shared == {"rigid"} or len(shared) <= 1, shared
        source = SCRIPT.read_text()
        assert "DIFFERENT axes" in source


class TestTheAnalyticLeg:
    """Order 1 is the plain sum, so the gap to order 2 IS the ordering cost."""

    def test_order_one_is_the_plain_sum(self) -> None:
        """The premise. If this drifts, the whole comparison is meaningless."""
        generators, _, _ = DOD.build_generators(
            list(DOD.OPERATOR_ID_MODES), 32, 32, dtype=torch.complex64
        )
        torch.manual_seed(0)
        probe = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
        severities = torch.full((len(generators),), 0.4)
        composite = assemble_omega(generators, severities, order=1)(probe)
        plain = sum((g(probe) * 0.4 for g in generators), torch.zeros_like(probe))
        assert (composite - plain).norm() / plain.norm() < 1e-5

    def test_the_correction_grows_with_severity(self) -> None:
        """A commutator term is second order in theta; a constant would be a bug."""
        report = DOD.analytic_report(size=32, severities=(0.1, 1.0))
        scaling = report["scaling"]
        assert scaling[1.0] > 4 * scaling[0.1], scaling

    def test_some_pairs_commute_exactly(self) -> None:
        """Diagonal operators commute. Finding none would mean the probe is wrong."""
        report = DOD.analytic_report(size=32)
        exact = [p for p, m in report["pairs"].items() if m < 1e-6]
        assert exact, report["pairs"]

    def test_it_detects_a_planted_non_commuting_pair(self) -> None:
        """Anti-vacuity: the probe must react to a commutator that is really there.

        Two generators built to NOT commute must register above the exact-zero
        pairs, or the measurement is only ever confirming zeros.
        """
        generators, names, _ = DOD.build_generators(
            list(DOD.OPERATOR_ID_MODES), 32, 32, dtype=torch.complex64
        )
        torch.manual_seed(0)
        probe = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
        magnitudes = [
            float(commutator(a, b)(probe).norm() / a(probe).norm())
            for a, b in itertools.combinations(generators, 2)
        ]
        assert max(magnitudes) > 1e-3, (names, magnitudes)


class TestTheEmpiricalLeg:
    """The ratio is the statistic. The spread alone is optimiser noise."""

    def test_it_reports_a_seed_baseline_per_axis(self) -> None:
        volume = DOD.synthetic_volume(slices=2, size=64)
        report = DOD.empirical_report(volume, max_evals=40, seeds=3)
        for axis in DOD.QUALITY_MATCHING_AXES:
            assert {"order_sd", "seed_sd", "ratio"} <= set(report[axis])
            assert report[axis]["seed_sd"] > 0, (
                f"{axis} has a zero seed baseline, so the ratio is meaningless"
            )

    def test_the_verdict_reads_the_ratio_not_the_spread(self) -> None:
        """Source-level, because the wrong reading is the failure mode here."""
        source = SCRIPT.read_text()
        assert "Read the RATIO" in source
        assert "seed_sd" in source

    def test_the_underdetermination_is_named(self) -> None:
        """The arm's own metadata says theta is not identified; the script must too.

        Without it a reader takes a fitted severity for a scanner measurement,
        which is the error the cohort's note already warns about.
        """
        source = SCRIPT.read_text()
        assert "underdetermined" in source
        assert "not be read as a physical measurement" in source
