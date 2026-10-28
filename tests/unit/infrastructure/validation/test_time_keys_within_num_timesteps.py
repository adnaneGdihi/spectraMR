"""Unit tests for ``check_time_keys_within_num_timesteps``.

Anchor: experiment_11_kspace_cold_diffusion mosaic triage 2026-05-28.

Pins the four cross-block t-valued-key consistency checks:

* ``training.curriculum_start_timestep < T``
* ``training.sampling_steps <= T``
* ``training.diffusion.sampling_steps <= T``
* ``model.model_kwargs.timesteps == T``
"""

from __future__ import annotations

from spectramr.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
)


class _Diffusion:
    def __init__(self, timesteps=28, sampling_steps=None):
        self.timesteps = timesteps
        if sampling_steps is not None:
            self.sampling_steps = sampling_steps


class _Training:
    def __init__(
        self,
        timesteps=28,
        curriculum_start_timestep=None,
        sampling_steps=None,
        diffusion_sampling_steps=None,
    ):
        self.diffusion = _Diffusion(
            timesteps=timesteps,
            sampling_steps=diffusion_sampling_steps,
        )
        if curriculum_start_timestep is not None:
            self.curriculum_start_timestep = curriculum_start_timestep
        if sampling_steps is not None:
            self.sampling_steps = sampling_steps


class _Model:
    def __init__(self, model_timesteps=None):
        if model_timesteps is not None:
            self.model_kwargs = {"timesteps": model_timesteps}
        else:
            self.model_kwargs = {}


class _Config:
    def __init__(self, training, model=None):
        self.training = training
        self.model = model


def _check(**training_kwargs):
    model_timesteps = training_kwargs.pop("model_timesteps", None)
    cfg = _Config(
        training=_Training(**training_kwargs),
        model=_Model(model_timesteps=model_timesteps),
    )
    return ConfigHealthChecker().check_time_keys_within_num_timesteps(cfg)


class TestPassingCases:
    def test_experiment_11_post_fix_passes(self):
        result = _check(
            timesteps=28,
            curriculum_start_timestep=4,
            sampling_steps=10,
            diffusion_sampling_steps=10,
            model_timesteps=28,
        )
        assert result.passed, result.message

    def test_all_unset_passes(self):
        result = _check(timesteps=28)
        assert result.passed


class TestCurriculumStartTimestep:
    def test_start_timestep_equals_T_fails(self):
        result = _check(timesteps=28, curriculum_start_timestep=28)
        assert not result.passed
        assert "curriculum_start_timestep=28" in result.message

    def test_start_timestep_greater_than_T_fails(self):
        # Original experiment_11 state — start=100, T=28.
        result = _check(timesteps=28, curriculum_start_timestep=100)
        assert not result.passed
        assert "saturates at the first iteration" in result.message

    def test_start_timestep_less_than_T_passes(self):
        result = _check(timesteps=28, curriculum_start_timestep=4)
        assert result.passed


class TestSamplingStepsBound:
    def test_top_level_above_T_fails(self):
        # Original experiment_11 — training.sampling_steps=50, T=28.
        result = _check(timesteps=28, sampling_steps=50)
        assert not result.passed
        assert "training.sampling_steps=50" in result.message

    def test_diffusion_sub_above_T_fails(self):
        result = _check(timesteps=28, diffusion_sampling_steps=50)
        assert not result.passed
        assert "training.diffusion.sampling_steps=50" in result.message

    def test_sampling_steps_equal_T_passes(self):
        # Equal is fine — it's exactly the full reverse loop.
        result = _check(
            timesteps=28, sampling_steps=28, diffusion_sampling_steps=28
        )
        assert result.passed


class TestModelTimestepsConsistency:
    def test_mismatch_fails(self):
        # Original experiment_11 state — model says 1000, training says 28.
        result = _check(timesteps=28, model_timesteps=1000)
        assert not result.passed
        assert "collapses all" in result.message

    def test_match_passes(self):
        result = _check(timesteps=28, model_timesteps=28)
        assert result.passed

    def test_num_timesteps_spelling_mismatch_fails(self):
        # The builder refuses either spelling, so the audit must too (#2468).
        model = _Model()
        model.model_kwargs = {"num_timesteps": 1000}
        cfg = _Config(training=_Training(timesteps=28), model=model)
        result = ConfigHealthChecker().check_time_keys_within_num_timesteps(cfg)
        assert not result.passed
        assert "model.model_kwargs.num_timesteps=1000" in result.message


class TestExperiment11OriginalStateFails:
    """The pre-fix experiment_11 YAML triggered ALL FOUR problems."""

    def test_all_four_keys_fail_together(self):
        result = _check(
            timesteps=28,
            curriculum_start_timestep=100,
            sampling_steps=50,
            diffusion_sampling_steps=10,
            model_timesteps=1000,
        )
        assert not result.passed
        # Three of four problems should be reported (sampling_steps=10 is fine).
        msg = result.message
        assert "curriculum_start_timestep=100" in msg
        assert "training.sampling_steps=50" in msg
        assert "model.model_kwargs.timesteps=1000" in msg
