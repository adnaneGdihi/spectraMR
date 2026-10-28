"""Regression tests for the 2026-06 audit fixes to ``SpinSDEStrategy``.

* Longitudinal relaxation must recover toward the equilibrium ``M_eq`` (= M0),
  not a hardcoded 1.0.
* The diffusion coefficient is a learnable nn.Parameter (its prior term must
  carry a gradient), registered on the generator optimizer.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from spectramr.infrastructure.training.strategies.spin_sde_strategy import SpinSDEStrategy


def _bare(n_steps: int = 32, dt: float = 1e-3, echo_times=None) -> SpinSDEStrategy:
    """A SpinSDEStrategy with __init__ bypassed and only the attributes the
    readout/integration helpers touch set (no env / optimizer needed)."""
    s = object.__new__(SpinSDEStrategy)
    s.diffusion = 0.0  # deterministic: disables the SDE noise term
    s.n_steps = n_steps
    s.dt = dt
    s.echo_times = echo_times
    s._readout_steps_cache = None
    return s


def test_integrate_recovers_toward_m_eq_not_one() -> None:
    s = object.__new__(SpinSDEStrategy)  # bypass __init__ (no env needed)
    s.diffusion = 0.0  # disable SDE noise for a deterministic check
    s.n_steps = 80
    s.dt = 0.02

    shape = (1, 1, 2, 2)
    T1 = torch.full(shape, 0.05)
    T2 = torch.full(shape, 0.05)
    m_eq = torch.full(shape, 0.8)
    m0 = torch.cat([torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)], dim=1)

    traj = s.integrate(m0, T1, T2, m_eq=m_eq)
    mz = traj[:, 2:3]
    # mz must approach M_eq (0.8), and be far from the old hardcoded 1.0.
    assert torch.allclose(mz, m_eq, atol=1e-2)
    assert (mz - 0.8).abs().mean() < (mz - 1.0).abs().mean()


def test_diffusion_prior_is_learnable() -> None:
    src = inspect.getsource(SpinSDEStrategy)
    # The prior uses the learnable parameter through its owned-module holder, and the
    # parameter is registered on the optimizer in __init__.
    assert "diffusion.pow(2)" in src
    assert 'self.call_owned("diffusion")' in src
    assert "torch.nn.Parameter" in src
    assert "add_param_group" in src


# --- Design item D2: per-contrast Bloch-consistency readout (audit 2026-06) ---


def test_integrate_multicontrast_reads_distinct_steps() -> None:
    """The multi-contrast readout returns one channel per requested step, and
    those readouts genuinely differ across steps (NOT a single value broadcast
    against C contrasts — the bug D2 fixes)."""
    s = _bare(n_steps=30, dt=0.01)
    shape = (1, 1, 2, 2)
    T1 = torch.full(shape, 0.5)
    T2 = torch.full(shape, 0.1)
    M0 = torch.full(shape, 1.0)
    # Excite the transverse plane (Mx = M0) so |M_xy| decays with T2 — without
    # off-resonance the magnitude is strictly decreasing along the trajectory.
    m0 = torch.cat([M0.clone(), torch.zeros(shape), torch.zeros(shape)], dim=1)

    out = s.integrate_multicontrast(m0, T1, T2, readout_steps=[1, 15, 30], m_eq=M0)
    assert out.shape == (1, 3, 2, 2)
    mags = out[0, :, 0, 0]
    assert mags[0] > mags[1] > mags[2]  # distinct, monotonically decaying


def test_integrate_multicontrast_matches_full_integrate_at_final_step() -> None:
    """Reading out at the final step must equal the transverse magnitude of the
    full ``integrate`` output (the helpers share one stepping kernel)."""
    s = _bare(n_steps=20, dt=0.01)
    shape = (1, 1, 3, 3)
    T1 = torch.full(shape, 0.4)
    T2 = torch.full(shape, 0.2)
    M0 = torch.full(shape, 0.9)
    m0 = torch.cat([M0.clone(), 0.3 * M0.clone(), torch.zeros(shape)], dim=1)

    m_final = s.integrate(m0, T1, T2, m_eq=M0)
    expected = (m_final[:, :2] ** 2).sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
    got = s.integrate_multicontrast(m0, T1, T2, readout_steps=[s.n_steps], m_eq=M0)
    assert torch.allclose(got, expected, atol=1e-6)


def test_resolve_readout_steps_evenly_spaced_spans_trajectory() -> None:
    s = _bare(n_steps=32)
    steps = s._resolve_readout_steps(4)
    assert len(steps) == 4
    assert steps[0] == 1 and steps[-1] == 32  # spans first..final step
    assert steps == sorted(steps)


def test_resolve_readout_steps_single_contrast_is_legacy_final() -> None:
    """C == 1 must reduce exactly to the legacy single final-state readout."""
    s = _bare(n_steps=32)
    assert s._resolve_readout_steps(1) == [32]


def test_echo_times_map_to_nearest_step_and_clamp() -> None:
    s = _bare(n_steps=32, dt=1e-3, echo_times=(0.005, 0.010, 0.040))
    # round(t/dt): 5, 10, 40 -> 40 clamped to n_steps (32).
    assert s._resolve_readout_steps(3) == [5, 10, 32]


def test_echo_times_length_must_match_contrasts() -> None:
    s = _bare(n_steps=32, dt=1e-3, echo_times=(0.005, 0.010))
    with pytest.raises(ValueError, match="echo_times"):
        s._resolve_readout_steps(3)


def test_readout_cache_rejects_changed_contrast_count() -> None:
    s = _bare(n_steps=16)
    first = s._resolve_readout_steps(3)
    assert s._resolve_readout_steps(3) == first  # cached, same object/value
    with pytest.raises(ValueError, match="constant across the run"):
        s._resolve_readout_steps(4)


def test_diffusion_is_an_owned_module_with_an_unchanged_checkpoint_key(monkeypatch) -> None:
    """#2629: the DDP holder is not an attribute, so the checkpoint keeps
    ``params.diffusion`` and the loss reaches it through ``call_owned``."""
    from tests.unit.infrastructure.training.strategies.test_multi_param_mapping_strategy import (
        _built_via_init,
    )

    s = _built_via_init(monkeypatch, SpinSDEStrategy)
    assert s.owned_modules()["diffusion"].value is s.diffusion
    assert sorted(s.strategy_state_dict()["params"]) == ["_log_sigmas", "diffusion"]

    calls: list[str] = []
    real = type(s).call_owned

    def _spy(self, name, *a, **k):
        calls.append(name)
        return real(self, name, *a, **k)

    monkeypatch.setattr(SpinSDEStrategy, "call_owned", _spy)
    shape = (1, 1, 4, 4)
    x = torch.randn(1, 2, 4, 4)
    batch = {
        "observed_contrasts": x,
        "t1": torch.zeros(shape),
        "t2": torch.ones(shape),
        "T1_map": torch.full(shape, 0.5),
        "T2_map": torch.full(shape, 0.1),
        "M0_map": torch.ones(shape),
        "multi_contrast_signal": torch.rand(1, 2, 4, 4),
    }
    s._compute_losses_impl(x, x, 0, batch=batch)["loss_total"].backward()
    assert sorted(calls) == ["_log_sigmas", "diffusion"]  # one DDP forward each per step
    assert s.diffusion.grad is not None
