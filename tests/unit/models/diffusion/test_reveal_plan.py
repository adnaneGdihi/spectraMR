"""The reveal plan reproduces the per-step walk and reads the host once (#1162, #1388)."""

from __future__ import annotations

import sys

import pytest
import torch

from spectramr.models.diffusion.kspace_process import (
    KSpaceUndersamplingProcess,
    PhysicsInformedColdDiffusion,
)
from spectramr.models.diffusion.reveal_plan import plan_reveals

pytestmark = pytest.mark.unit

T = 12
_HOST_READS = ("item", "__bool__", "__int__", "__float__", "__index__", "tolist")
_LOOP_FILES = ("kspace_process.py", "reveal_plan.py")


class _Stub(torch.nn.Module):
    def __init__(self, process: KSpaceUndersamplingProcess) -> None:
        super().__init__()
        self.kspace_process = process

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_kw: object) -> torch.Tensor:
        return x * 1.01


def _sampler(mode: str = "replace_freeze_dc", steps: int = T) -> PhysicsInformedColdDiffusion:
    process = KSpaceUndersamplingProcess(
        num_timesteps=T,
        max_acceleration=16.0,
        base_acceleration=2.0,
        center_fraction=0.08,
        min_center_fraction=0.02,
        mask_type="equispaced",
        seed=42,
    ).eval()
    return PhysicsInformedColdDiffusion(
        _Stub(process).eval(),
        num_timesteps=T,
        dc_method="hard",
        sampling_steps=steps,
        reverse_mode=mode,
        reverse_clip_ratio=1.3,
        kspace_log_scaled=False,
    )


def _measurement(sampler, t_s: int):
    torch.manual_seed(0)
    full = torch.randn(2, 2, 32, 32)
    meas, mask = sampler.process.q_sample(full, torch.full((2,), t_s, dtype=torch.long))
    return meas, sampler.process.mask_generator.expand_mask_to_channels(mask, 2)


@pytest.mark.parametrize("t_s", [1, 6, T - 1])
def test_the_plan_reproduces_the_per_step_walk(t_s: int) -> None:
    sampler = _sampler()
    meas, mask = _measurement(sampler, t_s)
    obs = (mask > 0).float()
    schedule = list(range(t_s, -1, -1))
    committed = torch.zeros_like(obs)
    plan = plan_reveals(sampler, schedule, committed, obs, meas, 2, meas.device)
    for i in range(len(schedule)):
        reveal = sampler._reveal_support(i, schedule, committed, obs, meas, 2, meas.device)
        assert torch.equal(plan.reveal(i, committed.dtype), reveal), f"step {i}"
        assert plan.active[i] == bool(reveal.any())
        committed = torch.clamp(committed + reveal, max=1.0)
    assert torch.equal((plan.first >= 0).float() + obs, torch.ones_like(obs))


def test_the_plan_reads_the_host_once(monkeypatch) -> None:
    sampler = _sampler()
    meas, mask = _measurement(sampler, T - 1)
    obs = (mask > 0).float()
    reads: list[str] = []
    _count(monkeypatch, reads)
    plan_reveals(
        sampler, list(range(T - 1, -1, -1)), torch.zeros_like(obs), obs, meas, 2, meas.device
    )
    assert reads == ["tolist"]


def test_an_empty_schedule_raises() -> None:
    sampler = _sampler()
    meas, mask = _measurement(sampler, 3)
    obs = (mask > 0).float()
    with pytest.raises(ValueError, match="empty timestep schedule"):
        plan_reveals(sampler, [], torch.zeros_like(obs), obs, meas, 2, meas.device)


@pytest.mark.parametrize("mode", ["replace_freeze_dc", "replace_freeze"])
def test_the_freeze_loop_reads_do_not_scale_with_the_schedule(monkeypatch, mode) -> None:
    """Red on the per-step gate: ``bool(reveal.any())`` ran once per scheduled step."""

    def reads_for(steps: int) -> int:
        sampler = _sampler(mode, steps)
        meas, mask = _measurement(sampler, T - 1)
        sampler.sample(meas, mask, start_timestep=T - 1)
        reads: list[str] = []
        with monkeypatch.context() as m:
            _count(m, reads)
            sampler.sample(meas, mask, start_timestep=T - 1)
        return len(reads)

    assert reads_for(4) == reads_for(T)


def _count(monkeypatch, reads: list[str]) -> None:
    for name in _HOST_READS:
        original = getattr(torch.Tensor, name)

        def read(tensor, *args, _name=name, _original=original, **kwargs):
            if sys._getframe(1).f_code.co_filename.endswith(_LOOP_FILES):
                reads.append(_name)
            return _original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, read)
