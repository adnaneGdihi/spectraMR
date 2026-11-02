"""Unit tests for the progressive-GAN phase scheduler.

The full :class:`GANTrainingStrategy` requires a heavily-mocked training
environment, so these tests exercise the scheduler logic (phase / alpha
computation and phase application) on an instance built via ``__new__``
with the minimal attributes the pure methods touch. This isolates the
load-bearing growth schedule from the parent's optimisation machinery.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies import gan as gan_mod
from spectramr.infrastructure.training.strategies.progressive_gan_strategy import (
    ProgressiveGANStrategy,
)


class _FakeModel:
    def __init__(self, num_phases: int) -> None:
        self.num_phases = num_phases
        self.phase: int | None = None
        self.current_phase = 0
        self.alpha: float | None = None

    def set_phase(self, phase: int, alpha: float = 1.0) -> None:
        self.phase = phase
        self.current_phase = phase
        self.alpha = alpha

    def current_resolution(self) -> int:
        return 4 * 2 ** (self.phase or 0)


def _make_strategy(num_phases: int, schedule: list[int] | None, max_iters: int = 0):
    """Build a ProgressiveGANStrategy without running the GAN __init__."""
    strat = ProgressiveGANStrategy.__new__(ProgressiveGANStrategy)
    gen = _FakeModel(num_phases)
    disc = _FakeModel(num_phases)
    strat.env = SimpleNamespace(generator=gen, discriminator=disc)
    gan_cfg = SimpleNamespace(phase_schedule=schedule) if schedule is not None else None
    strat.config = SimpleNamespace(
        training=SimpleNamespace(gan=gan_cfg, max_iterations=max_iters)
    )
    strat._last_step_metrics = {}
    # Reproduce the relevant __init__ body.
    strat._phase_schedule = strat._resolve_phase_schedule()
    strat._phase_boundaries = strat._cumulative(strat._phase_schedule)
    return strat, gen, disc


class TestPhaseSchedule:
    def test_explicit_schedule_used(self) -> None:
        strat, _, _ = _make_strategy(3, [10, 20, 30])
        assert strat._phase_schedule == [10, 20, 30]
        assert strat._phase_boundaries == [0, 10, 30, 60]

    def test_uniform_schedule_from_max_iters(self) -> None:
        strat, _, _ = _make_strategy(4, None, max_iters=40)
        assert strat._phase_schedule == [10, 10, 10, 10]

    def test_rejects_nonpositive_entries(self) -> None:
        with pytest.raises(ValueError):
            _make_strategy(2, [10, 0])

    def test_phase0_no_fade(self) -> None:
        strat, _, _ = _make_strategy(3, [10, 10, 10])
        phase, alpha = strat._phase_and_alpha(0)
        assert phase == 0
        assert alpha == 1.0

    def test_alpha_ramps_within_phase(self) -> None:
        strat, _, _ = _make_strategy(2, [10, 10])
        # Phase 1 spans steps [10, 20); fade over first half (5 steps).
        p_start, a_start = strat._phase_and_alpha(10)
        assert p_start == 1 and a_start == pytest.approx(0.0)
        _, a_mid = strat._phase_and_alpha(12)
        assert 0.0 < a_mid < 1.0
        _, a_end = strat._phase_and_alpha(18)
        assert a_end == pytest.approx(1.0)

    def test_advances_through_phases(self) -> None:
        strat, _, _ = _make_strategy(3, [10, 10, 10])
        assert strat._phase_and_alpha(0)[0] == 0
        assert strat._phase_and_alpha(15)[0] == 1
        assert strat._phase_and_alpha(25)[0] == 2

    def test_apply_phase_mutates_both_models(self) -> None:
        strat, gen, disc = _make_strategy(3, [10, 10, 10])
        phase, alpha = strat._apply_phase(15)
        assert gen.phase == phase == 1
        assert disc.phase == phase
        assert gen.alpha == alpha
        assert disc.alpha == alpha

    def test_clamps_step_beyond_total(self) -> None:
        strat, _, _ = _make_strategy(2, [5, 5])
        phase, alpha = strat._phase_and_alpha(10_000)
        assert phase == 1
        assert alpha == pytest.approx(1.0)


class TestPhaseResolutionTargets:
    """#2385: the target and the critic's real batch follow the active phase."""

    def test_target_is_box_downsampled_to_the_phase(self) -> None:
        strat, gen, _ = _make_strategy(3, [10, 10, 10])
        gen.set_phase(0)
        target = torch.arange(16.0 * 16).reshape(1, 1, 16, 16)
        out = strat._to_phase_resolution(target)
        assert out.shape == (1, 1, 4, 4)
        assert torch.isclose(out.mean(), target.mean())

    def test_indivisible_target_raises(self) -> None:
        strat, gen, _ = _make_strategy(3, [10, 10, 10])
        gen.set_phase(1)
        with pytest.raises(ValueError, match="not a multiple"):
            strat._to_phase_resolution(torch.zeros(1, 1, 12, 12))

    def test_train_step_resizes_target_and_logs_phase(self, monkeypatch) -> None:
        strat, gen, _ = _make_strategy(3, [10, 10, 10])
        strat.loop_state = SimpleNamespace(iteration=15)
        strat._to_device = lambda t: t
        seen = {}

        def fake_step(self, batch, epoch, *, input_batch, target_batch, **kwargs):
            seen["target"] = target_batch
            self._last_step_metrics = {}  # the base step resets the dict
            return []

        monkeypatch.setattr(gan_mod.GANTrainingStrategy, "train_step", fake_step)
        x = torch.rand(2, 1, 16, 16)
        strat.train_step(None, 0, input_batch=x, target_batch=x, iteration=0)
        assert gen.phase == 1  # from loop_state, not the passed iteration
        assert seen["target"].shape == (2, 1, 8, 8)
        assert strat._last_step_metrics["progan_phase"] == 1.0
        assert 0.0 <= strat._last_step_metrics["progan_alpha"] <= 1.0

    def test_validation_critic_sees_phase_metrics_see_full_resolution(self, monkeypatch) -> None:
        strat, gen, _ = _make_strategy(3, [10, 10, 10])
        gen.set_phase(0)
        seen = {}

        def fake_views(self, pred, target, metrics_config, **kwargs):
            seen["metric"] = (tuple(pred.shape), tuple(target.shape))

        def fake_val(self, i, t, **k):
            seen["critic"] = tuple(t.shape)
            self._metric_transform_views(torch.rand(2, 1, 4, 4), t, None)
            return {}

        monkeypatch.setattr(gan_mod.GANTrainingStrategy, "_metric_transform_views", fake_views)
        monkeypatch.setattr(gan_mod.GANTrainingStrategy, "validation_step", fake_val)
        x = torch.rand(2, 1, 16, 16)
        out = strat.validation_step(x, x)
        assert seen["critic"] == (2, 1, 4, 4)
        assert seen["metric"] == ((2, 1, 16, 16), (2, 1, 16, 16))
        assert out["progan_phase"] == 0.0
        assert strat._val_full_target is None

    def test_visual_pair_is_cached_at_phase_resolution(self, monkeypatch) -> None:
        strat, _, _ = _make_strategy(3, [10, 10, 10])
        monkeypatch.setattr(
            gan_mod.GANTrainingStrategy,
            "_log_validation_images_to_tensorboard",
            lambda self, p, t, i, m: None,
        )
        pred, tgt = torch.rand(2, 1, 4, 4), torch.rand(2, 1, 4, 4)
        strat._log_validation_images_to_tensorboard(pred, tgt, torch.rand(2, 1, 16, 16), {})
        assert strat._last_visual_pred.shape == strat._last_visual_target.shape == (2, 1, 4, 4)
