"""vMF forward process: schedule validation and the noise endpoints."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spectramr.models.diffusion.vmf_schedule import KAPPA_SCHEDULES, VMFNoiseSchedule

pytestmark = pytest.mark.unit


def test_unknown_schedule_raises() -> None:
    with pytest.raises(ValueError, match="Unknown kappa_schedule"):
        VMFNoiseSchedule(schedule="cosine")


@pytest.mark.parametrize(("kmin", "kmax"), [(0.0, 10.0), (5.0, 5.0), (10.0, 1.0)])
def test_bad_kappa_range_raises(kmin: float, kmax: float) -> None:
    with pytest.raises(ValueError, match="kappa_min < kappa_max"):
        VMFNoiseSchedule(kappa_min=kmin, kappa_max=kmax)


@pytest.mark.parametrize("schedule", KAPPA_SCHEDULES)
def test_kappa_table_descends_between_endpoints(schedule: str) -> None:
    s = VMFNoiseSchedule(timesteps=10, kappa_max=100.0, kappa_min=0.1, schedule=schedule)
    k = s.kappas
    assert k.shape == (10,)
    assert torch.all(k[1:] < k[:-1])
    assert k[0].item() == pytest.approx(100.0, rel=1e-5)
    assert k[-1].item() == pytest.approx(0.1, rel=1e-4)


def test_q_sample_endpoints_and_sphere() -> None:
    torch.manual_seed(0)
    s = VMFNoiseSchedule(timesteps=100)
    z0 = F.normalize(torch.randn(4, 16, 8, 8), dim=1)
    clean = s.q_sample(z0, torch.zeros(4, dtype=torch.long))
    noisy = s.q_sample(z0, torch.full((4,), 99, dtype=torch.long))
    assert clean.shape == z0.shape
    assert torch.allclose(noisy.norm(dim=1), torch.ones(4, 8, 8), atol=1e-5)
    assert (clean * z0).sum(1).mean().item() > 0.99
    assert abs((noisy * z0).sum(1).mean().item()) < 0.05


def test_prior_sample_is_unit_norm() -> None:
    z = VMFNoiseSchedule.prior_sample((2, 3, 4, 4), device=torch.device("cpu"))
    assert torch.allclose(z.norm(dim=1), torch.ones(2, 4, 4), atol=1e-5)
