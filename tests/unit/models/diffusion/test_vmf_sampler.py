"""``vmf_reverse``: registry resolution, level grid and the reverse chain."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spectramr.models.diffusion.samplers import get_sampler
from spectramr.models.diffusion.vmf_sampler import VMFReverseSampler
from spectramr.models.diffusion.vmf_schedule import VMFNoiseSchedule

pytestmark = pytest.mark.unit


class _Oracle(torch.nn.Module):
    """Predicts a fixed direction regardless of the input; counts calls."""

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.schedule = VMFNoiseSchedule(timesteps=20)
        self.target = target
        self.calls: list[int] = []

    def forward(self, z_t, t, cond=None):
        self.calls.append(int(t[0]))
        return self.target.expand_as(z_t)


def test_resolves_through_registry() -> None:
    assert isinstance(get_sampler("vmf_reverse", steps=3), VMFReverseSampler)


def test_rejects_bad_steps() -> None:
    with pytest.raises(ValueError, match="steps >= 1"):
        VMFReverseSampler(steps=0)
    with pytest.raises(ValueError, match="exceeds"):
        VMFReverseSampler(steps=30).levels(20)


def test_levels_descend_to_zero() -> None:
    assert VMFReverseSampler(steps=5).levels(20) == [19, 14, 10, 5, 0]


def test_chain_visits_levels_and_returns_unit_prediction() -> None:
    torch.manual_seed(0)
    target = F.normalize(torch.randn(1, 4, 1, 1), dim=1)
    model = _Oracle(target)
    out = get_sampler("vmf_reverse", steps=4).sample(model, (2, 4, 3, 3))
    assert model.calls == [19, 13, 6, 0]
    assert torch.allclose(out, target.expand(2, 4, 3, 3))
