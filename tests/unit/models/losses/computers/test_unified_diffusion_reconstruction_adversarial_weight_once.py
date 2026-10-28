"""The diffusion computer's adversarial term carries ``lambda_adv`` once, not squared.

Both branches of the adversarial block store a value that already carries
``lambda_adv`` -- the library dict's ``g_adv_loss`` is pre-weighted, and the bare
tensor branch multiplies it in -- and ``_stack_components`` then multiplied by
``lambda_adv`` again. ``experiment_11_sense_bridge_critic`` declares
``lambda_adv: 0.01`` and so trained its adversarial term at ``1e-4``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.models.losses.computers import UnifiedDiffusionLossComputer
from spectramr.models.losses.weights import LossWeightSpec, LossWeightTable

_LAMBDA_ADV = 0.1
_DIFFUSION_WEIGHT = 1.0


class _Critic(nn.Module):
    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return x.mean(dim=(1, 2, 3), keepdim=True)


class _TensorAdversarial:
    """The bare-tensor shape: owes ``lambda_adv``, which the computer applies."""

    def __call__(self, fake_pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        return fake_pred.mean() * 0.0 + 4.0


class _LibraryAdversarial:
    """The ``CompositeGANLoss`` shape: a dict whose ``g_adv_loss`` is pre-weighted."""

    def compute_generator_loss(self, fake_outputs_d: torch.Tensor, **_: object) -> dict:
        g_adv = (fake_outputs_d.mean() * 0.0 + 4.0) * _LAMBDA_ADV
        return {"g_adv_loss": g_adv, "g_total_loss": g_adv}


def _computer(adversarial: object) -> UnifiedDiffusionLossComputer:
    computer = UnifiedDiffusionLossComputer(config=None)
    computer._loss_weight_table = LossWeightTable(
        {
            "adversarial": LossWeightSpec(
                name="adversarial",
                weight=_LAMBDA_ADV,
                enabled=True,
                source="test",
                warmup_gated=False,
            )
        },
        warmup_iterations=0,
        warmup_losses=frozenset(),
    )
    computer.adversarial_loss_fn = adversarial
    return computer


@pytest.mark.parametrize("adversarial", [_TensorAdversarial(), _LibraryAdversarial()])
def test_the_adversarial_term_is_weighted_once(adversarial: object) -> None:
    computer = _computer(adversarial)
    pred = torch.zeros(2, 1, 4, 4, requires_grad=True)
    target = torch.ones(2, 1, 4, 4)

    out = computer.compute(pred=pred, target=target, discriminator=_Critic(), losses_dict={})

    diffusion = float(out.components["diffusion"])  # the MSE fallback: 1.0
    assert float(out.components["adversarial"]) == pytest.approx(_LAMBDA_ADV * 4.0)
    assert float(out.total) == pytest.approx(_DIFFUSION_WEIGHT * diffusion + _LAMBDA_ADV * 4.0), (
        "lambda_adv applied twice"
    )
