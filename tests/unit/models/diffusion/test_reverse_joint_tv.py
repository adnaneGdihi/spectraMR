"""Joint y-TV projection (Algorithm H): the operator, the prox and the objective guard."""

from __future__ import annotations

import pytest
import torch

import spectramr.models.diffusion.reverse_joint_tv as jtv
from spectramr.models.diffusion.reverse_domain import linear_kspace
from spectramr.models.diffusion.reverse_integrity import ReverseIntegrityError
from tests.unit.models.diffusion import _reverse_phantom as P

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def setting():
    proc = P.process(8.0)
    target = P.kspace()
    mask = P.mask(proc)
    return proc, target, target * mask, mask


def _sampler(proc, target):
    return P.sampler(
        P.Oracle(target, proc), "replace_freeze_dc", projection="joint_tv", integrity_check="raise"
    )


def _complex(*shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.complex(*torch.randn(2, *shape, generator=gen))


def test_grad_y_adjoint_is_exact() -> None:
    u, p = _complex(2, 3, 16, 12, seed=0), _complex(2, 3, 16, 12, seed=1)
    lhs = (jtv.grad_y(u) * p.conj()).sum()
    rhs = (u * jtv.grad_y_adjoint(p).conj()).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-5)


def test_grad_y_differences_rows_only() -> None:
    u = torch.arange(12.0).view(1, 1, 3, 4).to(torch.complex64)
    d = jtv.grad_y(u)
    assert torch.equal(d[..., :-1, :], torch.full_like(d[..., :-1, :], 4.0))
    assert torch.equal(d[..., -1, :], torch.zeros_like(d[..., -1, :]))


def test_the_prox_never_raises_the_norm_and_lowers_tv() -> None:
    f = _complex(2, 4, 32, 32, seed=2)
    lam = torch.full((2, 1, 1, 1), 0.3)
    u = jtv.prox_tv_y(f, lam)
    norm = lambda z: z.abs().pow(2).sum(dim=(1, 2, 3))  # noqa: E731
    assert bool((norm(u) <= norm(f) * (1 + 1e-5)).all())
    assert bool((jtv.tv_y(u) < jtv.tv_y(f)).all())


def test_an_image_constant_along_y_is_a_fixed_point() -> None:
    f = _complex(2, 3, 1, 16, seed=3).expand(2, 3, 16, 16).clone()
    u = jtv.prox_tv_y(f, torch.full((2, 1, 1, 1), 1.0))
    assert torch.allclose(u, f, atol=1e-5)


def test_the_projection_moves_toward_the_target_and_holds_the_measured_bins(setting) -> None:
    proc, target, meas, mask = setting
    project = jtv.build_joint_tv(_sampler(proc, target), meas, mask)
    out = project(meas, mask, where="test")
    free = (mask == 0).float()

    def error(x):
        return float((linear_kspace(x - target, log_scaled=False).abs() * free).pow(2).sum())

    assert error(out) < error(meas)
    held = mask.bool().expand_as(out)
    assert torch.equal(out[held], meas[held])


def test_an_expansive_prox_is_rejected_and_counted(setting, monkeypatch) -> None:
    """A prox output that raises the majorizer is not taken, and the refusal is reported."""
    proc, target, meas, mask = setting
    monkeypatch.setattr(jtv, "prox_tv_y_correction", lambda f, lam: f)
    sampler = _sampler(proc, target)
    sampler.last_projection_rejected = None
    project = jtv.build_joint_tv(sampler, meas, mask)
    out = project(meas, mask, where="test")
    assert torch.allclose(out, meas, atol=1e-5)
    assert int(sampler.last_projection_rejected) == jtv.ISTA_STEPS * meas.shape[0]


def test_an_accepted_step_is_counted_as_nothing(setting) -> None:
    proc, target, meas, mask = setting
    sampler = _sampler(proc, target)
    sampler.last_projection_rejected = None
    jtv.build_joint_tv(sampler, meas, mask)(meas, mask, where="test")
    assert int(sampler.last_projection_rejected) == 0


def test_a_broken_safeguard_is_caught(setting, monkeypatch) -> None:
    """PLANTED: a safeguard that accepts an expansive prox raises the objective, and the check raises."""
    proc, target, meas, mask = setting
    monkeypatch.setattr(jtv, "prox_tv_y_correction", lambda f, lam: f)
    monkeypatch.setattr(
        jtv, "accept_prox", lambda candidate, current: torch.ones_like(current, dtype=torch.bool)
    )
    project = jtv.build_joint_tv(_sampler(proc, target), meas, mask)
    with pytest.raises(ReverseIntegrityError) as info:
        project(meas, mask, where="test")
    assert info.value.invariant == "projection_residual"


def test_the_sampler_reports_rejections_per_run(setting) -> None:
    """The count is reset per ``sample`` and absent when no projection ran."""
    proc, target, meas, mask = setting
    sampler = _sampler(proc, target)
    for _ in range(2):
        sampler.sample(meas, mask, start_timestep=P.HEAD)
        assert sampler.reverse_stats["projection_rejected_steps"] == 0
    plain = P.sampler(P.Oracle(target, proc), "replace_freeze_dc")
    plain.sample(meas, mask, start_timestep=P.HEAD)
    assert plain.reverse_stats["projection_rejected_steps"] is None
