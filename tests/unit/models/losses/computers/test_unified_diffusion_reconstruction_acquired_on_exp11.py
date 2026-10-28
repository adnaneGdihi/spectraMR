"""The acquired-line plane reaches ``frequency_weighted_l1_kspace`` on the exp_11 loss stack.

``FrequencyWeightedL1Loss.forward`` takes ``acquired`` and the computer forwards it
through ``_call_safe_loss`` to every loss whose signature names it. A unit test of
the loss alone cannot show that: the term is built by ``LossBuilder`` from the arm's
YAML and may sit behind a bridge wrapper whose ``**kwargs`` narrows what passes.
This builds the reference arm's losses exactly as ``BuilderDirector`` does, hands
them to ``UnifiedDiffusionLossComputer.compute`` the way the diffusion training
step does, and watches the plane arrive and change the value.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder
from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
    UnifiedDiffusionLossComputer,
)
from spectramr.models.losses.kspace_physics_losses import FrequencyWeightedL1Loss

_ARM = (
    Path(__file__).resolve().parents[5]
    / "experiments/inprogress/kspace_filling/experiment_11_kspace_cold_diffusion.yaml"
)
_B, _COILS, _S = 2, 4, 32


@pytest.fixture(scope="module")
def stack() -> tuple[TrainingSettings, dict[str, torch.nn.Module]]:
    cfg = TrainingSettings.from_yaml(str(_ARM))
    device = torch.device("cpu")
    losses = (
        LossBuilder(cfg, device)
        .build_reconstruction_losses()
        .build_adversarial_losses()
        .build_physics_losses()
        .build_regularization_losses()
        .validate()
        .build()
    )
    return cfg, losses


def _stacked(z: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(z).movedim(-1, 2).flatten(1, 2)


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    shape = (_B, _COILS, _S, _S)
    target = _stacked(torch.randn(shape, dtype=torch.complex64, generator=generator))
    pred = target + 0.1 * torch.randn(target.shape, generator=generator)
    smaps = torch.randn(shape, dtype=torch.complex64, generator=generator)
    mask = torch.zeros(_B, 1, _S, _S)
    mask[..., ::4] = 1.0
    acquired = torch.ones(_B, 1, _S, _S)
    acquired[..., :6] = 0.0  # padded phase-encode lines the scanner never wrote
    return {"pred": pred, "target": target, "smaps": smaps, "mask": mask, "acquired": acquired}


def _compute(cfg, losses, inputs, *, with_acquired: bool):
    computer = UnifiedDiffusionLossComputer(config=cfg, device=torch.device("cpu"))
    extra = {"acquired": inputs["acquired"]} if with_acquired else {}
    return computer.compute(
        pred=inputs["pred"],
        target=inputs["target"],
        iteration=5000,
        timesteps=torch.tensor([3, 20]),
        losses_dict=dict(losses),
        smaps=inputs["smaps"],
        mask=inputs["mask"],
        **extra,
    )


def test_the_arm_builds_the_term_unwrapped(stack) -> None:
    _, losses = stack
    assert isinstance(losses["frequency_weighted_l1_kspace"], FrequencyWeightedL1Loss)


def test_the_plane_reaches_forward_on_the_production_call(stack, monkeypatch) -> None:
    cfg, losses = stack
    inputs = _inputs()
    seen: list[torch.Tensor | None] = []
    original = FrequencyWeightedL1Loss.forward

    @functools.wraps(original)
    def spy(self, pred, target, acquired=None):
        seen.append(acquired)
        return original(self, pred, target, acquired=acquired)

    monkeypatch.setattr(FrequencyWeightedL1Loss, "forward", spy)
    _compute(cfg, losses, inputs, with_acquired=True)

    assert len(seen) == 1
    assert seen[0] is inputs["acquired"]


def test_the_plane_changes_the_component(stack) -> None:
    cfg, losses = stack
    inputs = _inputs()
    with_plane = _compute(cfg, losses, inputs, with_acquired=True)
    without = _compute(cfg, losses, inputs, with_acquired=False)
    key = "frequency_weighted_l1_kspace"
    assert float(with_plane.components[key]) != pytest.approx(
        float(without.components[key]), rel=1e-3
    )
