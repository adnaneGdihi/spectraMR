"""Schedule-certification witnesses (C1 leak-freeness, inert levels, C4).

Severity contract first: all three run at WARNING, because the corpus ships 62
known-defective ladder arms (issue #534) and these witnesses must diagnose,
never block — the hard gate stays with the ratcheted CI script. Each detector
gets a sensitivity pair: green on a healthy arm AND firing on a planted or
known-defective one, so a witness that never fires cannot pass this file.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.validation.witness.checks.schedule_certification_checks import (
    build_process_from_config,
    schedule_line_allocation,
    schedule_nesting_leakfree,
    schedule_no_inert_steps,
    schedule_step_to_reach_cap,
    schedule_tangential_defect_margin,
    synthetic_spectral_prior,
)
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Subject,
    get_witness_registry,
)
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

_WITNESS_NAMES = (
    "schedule.nesting_leakfree",
    "schedule.no_inert_steps",
    "schedule.line_allocation",
)
_CHECKS = (schedule_nesting_leakfree, schedule_no_inert_steps, schedule_line_allocation)


def _exp11_doc(*, enforce_nested, timesteps=28, **undersampling_overrides):
    """The exp_11 acceleration block as a raw config doc, at a fast 64x64."""
    undersampling = {
        "acceleration_type": "equispaced",
        "base_acceleration": 2.0,
        "max_acceleration": 32.0,
        "center_fraction": 0.08,
        "min_center_fraction": 0.02,
        "acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0],
        "schedule_type": "step",
        "mask_seed": 42,
        "enforce_nested": enforce_nested,
    }
    undersampling.update(undersampling_overrides)
    return {
        "model": {"model_type": "kspace_cold_diffusion"},
        "data": {"sampling": {"patch_size": [64, 64]}},
        "undersampling": undersampling,
        "training": {"diffusion": {"timesteps": timesteps}},
    }


def _linear_doc(timesteps):
    """A short variable-density linear ladder: leak-free and inert-free."""
    doc = _exp11_doc(
        enforce_nested=True,
        timesteps=timesteps,
        acceleration_type="variable_density",
        schedule_type="linear",
        max_acceleration=8.0,
    )
    del doc["undersampling"]["acceleration_range"]
    return doc


def _subject(doc):
    return WitnessSubject.for_ci(None, doc)


class TestRegistration:
    def test_all_three_are_registered_at_warning(self):
        """WARNING is the load-bearing bit: 62 baselined arms must stay online."""
        registry = get_witness_registry()
        for name in _WITNESS_NAMES:
            witness = registry.get(name)
            assert witness is not None, f"{name} not registered"
            assert witness.severity is Severity.WARNING
            assert witness.stage is Stage.CONSTRUCT
            assert Subject.CONFIG in witness.subjects
            assert witness.category == "schedule_certification"


class TestApplicability:
    def test_non_cold_diffusion_arm_passes_through(self):
        subject = _subject({"model": {"model_type": "unet"}})
        for check in _CHECKS:
            verdict = check(subject)
            assert verdict.passed, verdict.message
            assert "does not apply" in verdict.message

    def test_build_process_returns_none_off_scope(self):
        assert build_process_from_config({"model": {"model_type": "unet"}}) is None

    @staticmethod
    def _fixed_4x_doc(model_type):
        """A fixed 4x mask (base == max): every level after t=0 repeats the mask."""
        return {
            "model": {"model_type": model_type},
            "data": {"sampling": {"patch_size": [64, 64]}},
            "undersampling": {
                "acceleration_type": "variable_density",
                "base_acceleration": 4.0,
                "max_acceleration": 4.0,
                "center_fraction": 0.08,
            },
            "training": {"diffusion": {"timesteps": 8}},
        }

    def test_gaussian_cold_diffusion_builds_no_ladder(self):
        """The Gaussian ``cold_diffusion`` wrapper noises by q_sample and owns no mask cascade."""
        doc = self._fixed_4x_doc("cold_diffusion")
        assert build_process_from_config(doc) is None
        verdict = schedule_no_inert_steps(_subject(doc))
        assert verdict.passed and "does not apply" in verdict.message

    def test_kspace_cold_diffusion_with_the_same_block_still_fires(self):
        """Anti-vacuity: the mask-cascade generator on the same fixed mask is inert."""
        verdict = schedule_no_inert_steps(_subject(self._fixed_4x_doc("kspace_cold_diffusion")))
        assert not verdict.passed
        assert "inert" in verdict.message

    def test_build_process_honours_config_timesteps(self):
        process = build_process_from_config(_exp11_doc(enforce_nested=True))
        assert process is not None
        assert process.num_timesteps == 28


class TestNestingLeakfreeWitness:
    def test_enforced_arm_is_green(self):
        verdict = schedule_nesting_leakfree(_subject(_exp11_doc(enforce_nested=True)))
        assert verdict.passed, verdict.message

    def test_unenforced_equispaced_arm_fires_as_warning(self):
        """The shipped defect: equispaced stride sets do not nest."""
        verdict = schedule_nesting_leakfree(_subject(_exp11_doc(enforce_nested=False)))
        assert verdict.passed is False
        assert verdict.severity is Severity.WARNING, "must diagnose, not block"
        assert "re-introduce" in verdict.message
        assert "undersampling.enforce_nested" in verdict.yaml_keys


class TestNoInertStepsWitness:
    def test_short_linear_ladder_is_green(self):
        verdict = schedule_no_inert_steps(_subject(_linear_doc(timesteps=6)))
        assert verdict.passed, verdict.message

    def test_step_schedule_pigeonhole_fires_as_warning(self):
        """28 levels over 7 rungs: most levels remove nothing."""
        verdict = schedule_no_inert_steps(_subject(_exp11_doc(enforce_nested=True)))
        assert verdict.passed is False
        assert verdict.severity is Severity.WARNING
        assert "inert" in verdict.message


class TestLineAllocationWitness:
    def test_exp11_spread_is_green_under_the_prior(self):
        verdict = schedule_line_allocation(_subject(_exp11_doc(enforce_nested=True)))
        assert verdict.passed, verdict.message

    def test_single_transition_ladder_fires_as_warning(self):
        """T=2 dumps the whole removed band into one level: share 100% (C4)."""
        verdict = schedule_line_allocation(
            _subject(_linear_doc(timesteps=2))
        )
        assert verdict.passed is False
        assert verdict.severity is Severity.WARNING
        assert "100" in verdict.message


def _certified_doc(per_level):
    doc = _exp11_doc(enforce_nested=True)
    doc["undersampling"]["certification"] = {"per_level": per_level}
    return doc


class TestStepToReachCapWitness:
    def test_registered_at_info(self):
        witness = get_witness_registry().get("schedule.step_to_reach_cap")
        assert witness is not None
        assert witness.severity is Severity.INFO

    def test_no_declared_estimates_passes_as_not_evaluable(self):
        verdict = schedule_step_to_reach_cap(_subject(_exp11_doc(enforce_nested=True)))
        assert verdict.passed, verdict.message
        assert "not evaluable" in verdict.message

    def test_declared_block_does_not_perturb_the_built_process(self):
        """extra: ignore — the certification block must be invisible to the schema."""
        process = build_process_from_config(
            _certified_doc([{"t": 1, "delta_hat": 1.0, "tau_hat": 4.0}])
        )
        assert process is not None
        assert process.num_timesteps == 28

    def test_compliant_levels_are_green(self):
        verdict = schedule_step_to_reach_cap(
            _subject(
                _certified_doc(
                    [
                        {"t": 1, "delta_hat": 1.0, "tau_hat": 4.0},   # kappa 0.25
                        {"t": 2, "delta_hat": 1.0, "tau_hat": 2.0},   # kappa 0.50
                    ]
                )
            )
        )
        assert verdict.passed, verdict.message

    def test_over_cap_level_fires_at_info(self):
        verdict = schedule_step_to_reach_cap(
            _subject(_certified_doc([{"t": 3, "delta_hat": 3.0, "tau_hat": 4.0}]))
        )
        assert verdict.passed is False
        assert verdict.severity is Severity.INFO, "C2 must diagnose, not block"
        assert "0.750" in verdict.message

    def test_ill_posed_level_is_called_out(self):
        verdict = schedule_step_to_reach_cap(
            _subject(_certified_doc([{"t": 4, "delta_hat": 5.0, "tau_hat": 4.0}]))
        )
        assert verdict.passed is False
        assert "well-posedness" in verdict.message


class TestTangentialDefectMarginWitness:
    def test_no_theta_passes_as_not_evaluable(self):
        """delta/tau alone cannot evaluate C3 — theta_hat is required."""
        verdict = schedule_tangential_defect_margin(
            _subject(_certified_doc([{"t": 1, "delta_hat": 1.0, "tau_hat": 4.0}]))
        )
        assert verdict.passed, verdict.message
        assert "not evaluable" in verdict.message

    def test_margin_cleared_is_green(self):
        # kappa 0.25 -> threshold 0.75; theta 0.5 clears it.
        verdict = schedule_tangential_defect_margin(
            _subject(
                _certified_doc(
                    [{"t": 1, "delta_hat": 1.0, "tau_hat": 4.0, "theta_hat": 0.5}]
                )
            )
        )
        assert verdict.passed, verdict.message

    def test_violated_margin_fires_as_warning(self):
        # kappa 0.5 -> threshold 0.5; theta 0.6 violates (2.9).
        verdict = schedule_tangential_defect_margin(
            _subject(
                _certified_doc(
                    [{"t": 2, "delta_hat": 1.0, "tau_hat": 2.0, "theta_hat": 0.6}]
                )
            )
        )
        assert verdict.passed is False
        assert verdict.severity is Severity.WARNING
        assert "contract" in verdict.message

    def test_non_cold_diffusion_arm_passes_through(self):
        verdict = schedule_tangential_defect_margin(
            _subject({"model": {"model_type": "unet"}})
        )
        assert verdict.passed
        assert "does not apply" in verdict.message


class TestSyntheticSpectralPrior:
    def test_prior_is_centred_complex_and_peaked_at_dc(self):
        """Peak at (H//2, W//2) — the fft2c shifted convention the masks use."""
        prior = synthetic_spectral_prior((8, 8))
        assert prior.shape == (1, 1, 8, 8)
        assert prior.is_complex()
        energy = prior.real**2 + prior.imag**2
        peak = torch.argmax(energy.view(-1)).item()
        assert (peak // 8, peak % 8) == (4, 4)

    def test_energy_follows_the_inverse_square_law(self):
        prior = synthetic_spectral_prior((8, 8))
        energy = (prior.real**2 + prior.imag**2)[0, 0]
        assert energy[4, 4] == pytest.approx(1.0)
        assert energy[4, 5] == pytest.approx(1.0 / 2.0)  # r^2 = 1
        assert energy[5, 5] == pytest.approx(1.0 / 3.0)  # r^2 = 2
