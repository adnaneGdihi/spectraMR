"""Tests for scripts/diagnostics/rung_exposure.py (#2066)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[3] / "scripts" / "diagnostics" / "rung_exposure.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("rung_exposure", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _cfg(rate, strategy="uniform", iters=10_000, horizon=9):
    return SimpleNamespace(
        training=SimpleNamespace(
            curriculum_start_timestep=1 if rate else None,
            curriculum_ramp_rate=rate,
            max_iterations=iters,
            timestep_sampling_strategy=strategy,
            diffusion=SimpleNamespace(timesteps=horizon),
        )
    )


@pytest.fixture
def mod(monkeypatch):
    m = _load()
    monkeypatch.setattr(m, "_process_floor", lambda cfg: 0)
    return m


def test_no_curriculum_uniform_is_flat(mod) -> None:
    e = mod.measure(_cfg(None))
    assert sum(e.iterations) == pytest.approx(10_000)
    assert max(e.iterations) == pytest.approx(min(e.iterations))


def test_a_late_ramp_starves_the_top_rung(mod) -> None:
    """PLANTED: the shape the kspace_filling arms had, top rung opening at 80 %."""
    e = mod.measure(_cfg(1e-3, iters=10_000))
    assert e.first_drawable[-1] == 8_000
    assert e.share(8) < e.share(1) / 4


def test_first_drawable_matches_the_curriculum_state(mod) -> None:
    from spectramr.core.curriculum import resolve_curriculum_state

    cfg = _cfg(1e-3, iters=10_000)
    assert mod.measure(cfg).first_drawable[-1] == resolve_curriculum_state(cfg).top_opens_at
