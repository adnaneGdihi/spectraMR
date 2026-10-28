"""The VAE computers weight the total at the live ``iteration`` (#1648).

``UnifiedVAELossComputer.compute`` and ``UnifiedVQVAELossComputer.compute`` gated
each term on ``_get_loss_weight(name, epoch, iteration)`` and then called
``_stack_components(components, epoch=epoch)``, which resolves every weight again
at ``iteration=0``. So a term could appear in ``components`` at full value and
contribute nothing to the total:

* a warm-up-gated term (``perceptual``, ``l1``, ``adversarial``) weighed 0.0 for
  the whole run;
* with ``anneal_kl_beta`` the KL weight stayed at ``kl_beta_start`` -- measured
  on 4 of the 5 inprogress VAE arms (``1e-6`` instead of ``1e-4`` on the three
  ``ldm_two_stage`` stage-1 arms, ``0.0`` instead of ``1.0`` on
  ``exp_pillar_05``).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.config.settings import TrainingSettings
from spectramr.models.losses.computers import UnifiedVAELossComputer, UnifiedVQVAELossComputer


class _Constant(nn.Module):
    """Grad-connected constant, so the arithmetic below is exact."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return pred.sum() * 0.0 + self._value


def _settings(losses: dict, training: dict | None = None) -> TrainingSettings:
    data = {
        "model": {"model_type": "unet"},
        "data": {"dataset_type": "synthetic"},
        "optimization": {},
        "logging": {},
        "losses": {"output_domain": "image", **losses},
    }
    if training is not None:
        data["training"] = training
    return TrainingSettings.settings_from_dict(data)


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    generator = nn.Conv2d(1, 1, 3, padding=1)
    return generator(torch.rand(2, 1, 8, 8)), torch.rand(2, 1, 8, 8)


_PERCEPTUAL = {
    "image_losses": [{"name": "mse", "weight": 1.0}, {"name": "perceptual", "weight": 3.7}]
}


def _computer(cls: type, losses: dict) -> UnifiedVAELossComputer | UnifiedVQVAELossComputer:
    """Built config-less so ``LossBuilder`` loads no VGG; the weights are the config's."""
    computer = cls(config=None)
    computer.config = _settings(losses)
    return computer


@pytest.mark.parametrize("cls", [UnifiedVAELossComputer, UnifiedVQVAELossComputer])
def test_a_warmup_gated_term_counts_once_the_gate_opens(cls) -> None:
    computer = _computer(cls, _PERCEPTUAL)
    computer.reconstruction_loss_fn = _Constant(5.0)
    computer.perceptual_loss_fn = _Constant(2.0)
    pred, target = _pair()

    out = computer.compute(pred, target, iteration=5000)

    assert float(out.components["perceptual"]) == pytest.approx(2.0)
    assert float(out.total) == pytest.approx(5.0 + 3.7 * 2.0)


@pytest.mark.parametrize("cls", [UnifiedVAELossComputer, UnifiedVQVAELossComputer])
def test_inside_the_warmup_window_the_gated_term_is_absent(cls) -> None:
    computer = _computer(cls, _PERCEPTUAL)
    computer.reconstruction_loss_fn = _Constant(5.0)
    computer.perceptual_loss_fn = _Constant(2.0)
    pred, target = _pair()

    out = computer.compute(pred, target, iteration=0)

    assert "perceptual" not in out.components
    assert float(out.total) == pytest.approx(5.0)


def test_the_kl_term_is_weighted_at_the_annealed_beta() -> None:
    computer = UnifiedVAELossComputer(
        config=_settings(
            {"image_losses": [{"name": "mse", "weight": 1.0}]},
            training={
                "vae": {
                    "anneal_kl_beta": True,
                    "kl_beta_start": 0.0,
                    "kl_beta_end": 1.0,
                    "kl_anneal_steps": 100,
                }
            },
        )
    )
    computer.reconstruction_loss_fn = _Constant(5.0)
    pred, target = _pair()
    mu = torch.ones(2, 4, requires_grad=True)
    logvar = torch.zeros(2, 4)

    for iteration, beta in ((50, 0.5), (200, 1.0)):
        out = computer.compute(pred, target, iteration=iteration, posterior=(mu, logvar))
        kl = float(out.components["kl"])
        assert kl == pytest.approx(2.0)  # 0.5 * sum(mu**2) over 4 dims
        assert float(out.total) == pytest.approx(5.0 + beta * kl)
