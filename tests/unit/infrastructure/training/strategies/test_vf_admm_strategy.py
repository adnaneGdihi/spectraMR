"""``ConcreteVFADMMStrategy``: the OOD acceleration readout seam (VF review 2026-09-03)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.physics.digital_twin_simulator import DigitalTwinSimulator
from spectramr.infrastructure.training.strategies.ood_acceleration_readout import (
    ood_acceleration_readout,
    ood_accelerations,
)
from spectramr.infrastructure.training.strategies.vf_admm_strategy import ConcreteVFADMMStrategy


class _RealStackedIdentity(nn.Module):
    """Echoes the real-stacked input, as the ADMM generator's output contract."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_masks: list[bool] = []

    def forward(self, x, marker_prior=None, **kwargs):
        self.seen_masks.append("undersampling_mask" in kwargs or bool(kwargs))
        return x


def _admm(rng):
    class _Probe(ConcreteVFADMMStrategy):
        validation_metrics_computer = None  # the base property needs a validation block

    s = _Probe.__new__(_Probe)
    s.env = SimpleNamespace(generator=_RealStackedIdentity())
    s.device = torch.device("cpu")
    s.simulator = DigitalTwinSimulator(
        im_size=(32, 32),
        enable_motion=False,
        snr_range=(100.0, 100.0),
        enable_undersampling=True,
        acceleration=4.0,
    )
    s.config = SimpleNamespace(
        physics=SimpleNamespace(
            digital_twin=SimpleNamespace(ood_acceleration_range=rng, enable_undersampling=True)
        )
    )
    return s


def test_ood_readout_scores_every_rung_on_the_real_twin_and_restores_it() -> None:
    strategy = _admm([16.0])
    target = torch.complex(torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32))
    with torch.no_grad():
        scored = strategy._score_at_current_twin(target, cache_visuals=True)
        out = ood_acceleration_readout(
            strategy.simulator,
            ood_accelerations(strategy.config),
            lambda: strategy._score_at_current_twin(target, cache_visuals=False),
        )
    assert set(scored) == {"val_psnr"}
    assert strategy._last_visual_pred is not None
    assert set(out) == {"val_ood_16x_psnr", "val_ood_accelerations"}
    assert out["val_ood_accelerations"] == 1.0 and torch.isfinite(
        torch.tensor(out["val_ood_16x_psnr"])
    )
    assert strategy.simulator.acceleration == 4.0
    assert strategy.generator_model.seen_masks == [True, True], (
        "the twin mask reaches the generator on both passes"
    )


def test_no_declared_range_gives_a_zero_count_and_no_extra_pass() -> None:
    strategy = _admm(None)
    out = ood_acceleration_readout(
        strategy.simulator,
        ood_accelerations(strategy.config),
        lambda: pytest.fail("no rung, no pass"),
    )
    assert out == {"val_ood_accelerations": 0.0}


def test_admm_reads_the_range_and_does_not_claim_the_undersampling_block() -> None:
    assert ConcreteVFADMMStrategy.reads_ood_acceleration_range is True
    assert ConcreteVFADMMStrategy.applies_undersampling is False
    assert "applies_undersampling" not in ConcreteVFADMMStrategy.__dict__


# --------------------------------------------------------------------------- #
# The loss-computer iteration seam (#1937)
#
# ``_compute_losses_impl`` read ``getattr(self.env, "step", 0)``. ``TrainingEnvironment``
# is ``frozen=True`` and declares no ``step`` field, so the value was a constant ``0``
# and the computer's warm-up gate never opened: ``resolve_loss_weight`` returns ``0.0``
# while ``iteration < warmup_iterations`` (default 1000), and
# ``_compute_reconstruction_losses`` gates on ``if lambda_l1 > 0`` -- so the ``l1`` term
# was ABSENT from ``components``, not scaled to zero, for every step of every run.
#
# ``l1`` is declared by none of the five live ``vf_admm`` arms; it reaches the gate
# through ``resolve_loss_weight``'s ``spec is None`` branch, which falls back to the
# ``lambda_l1`` schema default of 10.0 AND re-applies the gate. That is why a census
# over each arm's declared losses scores this defect at zero.
# --------------------------------------------------------------------------- #


class _SpyLossComputer:
    """Records the kwargs the hook hands the real ``UnifiedReconstructionLossComputer``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def compute(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(total=torch.zeros((), requires_grad=True), components={})


def _admm_with_spy(iteration: int) -> tuple[object, _SpyLossComputer]:
    """The house probe, plus only what the loss branch of the hook needs."""
    from spectramr.infrastructure.training.loop_state import LoopState

    s = _admm([16.0])
    s.env = SimpleNamespace(generator=_RealStackedIdentity(), losses={})
    s.config.data = SimpleNamespace(dataset_type="image")
    s.loop_state = LoopState(iteration=iteration, epoch=2)
    spy = _SpyLossComputer()
    s.loss_computer = spy
    s.loss_marker_prior = lambda *a, **k: torch.zeros(())
    s.loss_marker = lambda *a, **k: torch.zeros(())
    s.noise_estimator = lambda *a, **k: torch.zeros(1)
    s._lambda_prior = 0.1
    s._lambda_marker = 0.1
    return s, spy


#: Past the schema-default ``warmup_iterations`` of 1000, and not 0.
_LIVE_ITERATION = 1337


def test_loss_computer_receives_the_live_iteration_not_a_frozen_zero() -> None:
    strategy, spy = _admm_with_spy(_LIVE_ITERATION)
    target = torch.complex(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
    strategy._compute_losses_impl(input_batch=target, target_batch=target, epoch=2)
    assert len(spy.calls) == 1
    assert spy.calls[0]["iteration"] == _LIVE_ITERATION, (
        "the computer must receive the live loop iteration; frozen at 0 the warm-up "
        "gate stays shut for the whole run and the gated `l1` term (schema default "
        "10.0, undeclared by these arms) never enters components (#1937)"
    )


def test_the_iteration_advances_with_the_loop() -> None:
    """Two steps, two values -- a single-point assertion cannot tell live from constant."""
    strategy, spy = _admm_with_spy(0)
    target = torch.complex(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
    strategy._compute_losses_impl(input_batch=target, target_batch=target, epoch=0)
    strategy.loop_state.iteration = _LIVE_ITERATION
    strategy._compute_losses_impl(input_batch=target, target_batch=target, epoch=0)
    assert [c["iteration"] for c in spy.calls] == [0, _LIVE_ITERATION]
