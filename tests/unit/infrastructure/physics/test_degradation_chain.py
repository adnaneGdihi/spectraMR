"""Tests for the compounded degradation chain and its config emission."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradation_chain import (
    ChainLink,
    DegradationChain,
    UnapplicableAxisError,
    UnreplayableAxisError,
)
from spectramr.infrastructure.physics.degradations import is_identity_at_theta_zero


def _chain() -> DegradationChain:
    """Noise + blur: the canonical two-axis compound used across these tests."""
    return DegradationChain(
        links=(
            ChainLink(axis="complex_gaussian", theta=0.4),
            ChainLink(axis="t2star_blur", theta=0.25),
        )
    )


# ── validation ────────────────────────────────────────────────────────


def test_rejects_unknown_axis():
    with pytest.raises(UnapplicableAxisError, match="not a known degradation"):
        DegradationChain(links=(ChainLink(axis="not_an_axis", theta=0.5),))


def test_rejects_native_only_axis_with_actionable_message():
    # 'motion' is a native DigitalTwinSimulator axis, not a DEGRADATION_REGISTRY key,
    # so apply_degradation cannot run it standalone. The message must say so rather
    # than surfacing a bare KeyError from inside the registry.
    with pytest.raises(UnapplicableAxisError, match="native"):
        DegradationChain(links=(ChainLink(axis="motion", theta=0.5),))


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_rejects_theta_outside_unit_interval(bad):
    with pytest.raises(ValueError, match="theta"):
        DegradationChain(links=(ChainLink(axis="complex_gaussian", theta=bad),))


def test_rejects_an_empty_chain():
    with pytest.raises(ValueError, match="at least one link"):
        DegradationChain(links=())


# ── application ───────────────────────────────────────────────────────


def test_apply_is_deterministic_for_a_fixed_seed():
    x = torch.rand(1, 1, 32, 32)
    a = _chain().apply(x, seed=7)
    b = _chain().apply(x, seed=7)
    assert torch.equal(a, b)


def test_apply_differs_across_seeds():
    x = torch.rand(1, 1, 32, 32)
    a = _chain().apply(x, seed=7)
    b = _chain().apply(x, seed=8)
    assert not torch.equal(a, b)


def test_apply_actually_degrades_at_high_theta():
    x = torch.rand(1, 1, 32, 32)
    out = _chain().with_thetas((1.0, 1.0)).apply(x, seed=3)
    assert not torch.allclose(out.abs().float(), x.abs().float(), atol=1e-3)


def test_theta_is_monotonic_in_its_effect():
    # A larger theta must move the image further from the original. Without this a
    # chain could 'run' while theta did nothing -- the facade failure mode.
    x = torch.rand(1, 1, 32, 32)
    base = _chain()
    lo = base.with_thetas((0.2, 0.2)).apply(x, seed=1).abs().float()
    hi = base.with_thetas((0.9, 0.9)).apply(x, seed=1).abs().float()
    ref = x.abs().float()
    assert (hi - ref).norm() > (lo - ref).norm()


@pytest.mark.parametrize(
    "axis", ["t2star_blur", "resolution_snr", "cartesian_undersamp"]
)
def test_declared_identity_axes_are_honest_at_theta_zero(axis):
    """An axis declaring identity-at-zero must actually be identity at theta=0.

    Parametrised over the registry's own declaration rather than a hand-picked pair,
    so the test checks the CONTRACT, not one example. Note complex_gaussian is
    deliberately absent: it declares a non-zero floor (see the test below).
    """
    assert is_identity_at_theta_zero(axis), f"{axis} no longer declares identity-at-0"
    x = torch.rand(1, 1, 32, 32)
    chain = DegradationChain(links=(ChainLink(axis=axis, theta=0.0),))
    out = chain.apply(x, seed=3)
    assert torch.allclose(out.abs().float(), x.abs().float(), atol=1e-4)


def test_complex_gaussian_has_a_declared_nonzero_floor():
    """complex_gaussian is NOT identity at theta=0 -- it injects noise at its floor.

    Pinned so a future change to the identity-floor semantics is caught here rather
    than silently shifting every fitted chain that includes this axis.
    """
    assert not is_identity_at_theta_zero("complex_gaussian")


# ── config emission (the F1 mechanism) ────────────────────────────────


def test_to_digital_twin_config_lists_every_axis_with_a_degenerate_range():
    from spectramr.infrastructure.physics.digital_twin_simulator import NATIVE_DEGRADATION_AXES

    cfg = _chain().to_digital_twin_config()
    assert cfg["progressive_degradations"] == ["complex_gaussian", "t2star_blur"]
    assert cfg["degradation_ranges"] == {
        # every native stage is silenced — see the noise test below
        **dict.fromkeys(NATIVE_DEGRADATION_AXES, (0.0, 0.0)),
        "complex_gaussian": (0.4, 0.4),
        "t2star_blur": (0.25, 0.25),
    }


def test_emitted_block_silences_a_native_stage_the_consumer_enables():
    """A stage the fit never saw (here B1) must not replay at full strength."""
    from spectramr.config.schemas.physics import DigitalTwinConfig
    from spectramr.infrastructure.physics.digital_twin_simulator import DigitalTwinSimulator

    vol = torch.zeros(1, 1, 32, 32)
    vol[0, 0, 8:24, 8:24] = 1.0
    chain = DegradationChain(links=(ChainLink(axis="complex_gaussian", theta=0.0),))
    cfg = DigitalTwinConfig(**{**chain.to_digital_twin_config(), "enable_b1": True})
    sim = DigitalTwinSimulator.from_config(cfg, (32, 32))
    replayed, _markers, clean = sim(vol.to(torch.complex64), degradation_only=True, seed=0)
    fitted = chain.apply(clean, seed=0)
    assert ((replayed - fitted).abs().norm() / fitted.abs().norm()).item() < 1e-4


def test_every_chain_axis_appears_in_the_emitted_ranges():
    # An axis omitted from degradation_ranges defaults to (0.0, 1.0) and would
    # silently track the corruption factor instead of holding its fitted severity.
    chain = DegradationChain(
        links=tuple(
            ChainLink(axis=a, theta=0.3)
            for a in ("complex_gaussian", "t2star_blur", "resolution_snr")
        )
    )
    cfg = chain.to_digital_twin_config()
    assert set(cfg["progressive_degradations"]) <= set(cfg["degradation_ranges"])


def test_emitted_block_is_enabled():
    """A chain that is not enabled is a contradiction.

    ``enabled`` defaults False and gates the twin on the VF route, so an emitted
    block without it is inert wherever it is pasted — the artifact would advertise
    a replayable calibration that does nothing.
    """
    assert _chain().to_digital_twin_config()["enabled"] is True


def test_emitted_block_pins_the_simulators_own_noise_stage():
    """The AWGN stage has no enable flag, so silence has to be requested.

    ``_get_effective_cf`` holds any feature ABSENT from a non-empty
    ``progressive_degradations`` at 1.0 — static, not skipped. A block naming only
    the fitted axes therefore replayed them plus a fresh 10-25 dB scanner-noise
    draw that the fit never scored, and the synthesised volume sat at a different
    quality operating point than the one that was matched.
    """
    cfg = _chain().to_digital_twin_config()
    assert cfg["degradation_ranges"]["noise"] == (0.0, 0.0)
    assert "noise" not in cfg["progressive_degradations"]  # a pin, not an axis


@pytest.mark.parametrize(
    "axis", ["b0", "chemical_shift", "gibbs", "gradient_nonlinearity", "magic_angle"]
)
def test_dual_bank_axis_is_refused_because_the_replay_would_drop_it(axis):
    """These five are in the registry AND the native bank.

    The simulator filters dual-bank names out of its registry loop and defers to a
    native branch gated on an enable flag that defaults False, so the fit would
    score the registry operator and the replay would apply nothing whatsoever —
    measured for ``b0`` as 0.59 versus 0.00 relative L2 on a nested-block phantom.
    Refusing at construction also rejects it at CONFIG LOAD, because
    ``QualityMatchingConfig`` validates ``axes`` by building this object.
    """
    with pytest.raises(UnreplayableAxisError, match="would not reproduce"):
        DegradationChain(links=(ChainLink(axis=axis, theta=0.5),))


# ── reconstruction helpers ────────────────────────────────────────────


def test_with_thetas_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        _chain().with_thetas((0.1,))


def test_with_thetas_preserves_axes_and_order():
    updated = _chain().with_thetas((0.9, 0.1))
    assert updated.axes == ("complex_gaussian", "t2star_blur")
    assert updated.thetas == (0.9, 0.1)


# ── the seam: emitted config vs the REAL simulator ────────────────────


def test_replay_at_theta_zero_reproduces_the_chain_not_more():
    """Pixels, at the one theta where a pixel comparison is well-posed.

    The spy test below asserts the (axis, theta) pairs the simulator dispatches,
    which cannot see anything the NATIVE pipeline adds around them. At theta = 0
    both sides must be near-identity, so any residual is exactly what the emitted
    block contributes on its own — and before the noise pin that was 0.169 relative
    L2 against the chain's own 0.0087, a 19x unfitted term hiding in plain sight.

    A structured phantom, never torch.rand: with no dominant edge the cohort's
    sharpness attribute moves the wrong way under noise (see the cohort README).
    """
    from spectramr.config.schemas.physics import DigitalTwinConfig
    from spectramr.infrastructure.physics.digital_twin_simulator import (
        DigitalTwinSimulator,
    )

    vol = torch.zeros(1, 1, 32, 32)
    vol[0, 0, 4:28, 4:28] = 0.4
    vol[0, 0, 10:22, 10:22] = 1.0
    vol = vol.to(torch.complex64)

    chain = DegradationChain(
        links=tuple(
            ChainLink(axis=a, theta=0.0)
            for a in ("complex_gaussian", "resolution_snr", "rigid_motion")
        )
    )
    sim = DigitalTwinSimulator.from_config(
        DigitalTwinConfig(**chain.to_digital_twin_config()), (32, 32)
    )
    torch.manual_seed(0)
    replayed, _markers, clean = sim(vol, degradation_only=True)

    def _rel(a, b):
        return (
            torch.linalg.vector_norm(a.abs() - b.abs())
            / torch.linalg.vector_norm(b.abs())
        ).item()

    assert _rel(replayed, clean) == pytest.approx(_rel(chain.apply(vol, seed=0), vol), abs=5e-3)


def test_emitted_config_pins_theta_in_the_real_simulator(monkeypatch):
    """The F1 claim: a degenerate range holds theta at ANY corruption factor.

    Asserts the ``(axis, theta)`` pairs the simulator actually dispatches, not
    pixels. A pixel comparison against ``chain.apply()`` would compare two different
    computations -- the simulator's forward pass also embeds fiducial markers and
    runs its native pipeline -- and would fail for the wrong reason.

    Without this test the calibration artifact is an unverified claim: a degenerate
    range that silently tracked ``cf`` would still produce a plausible image.
    """
    from spectramr.infrastructure.physics import digital_twin_simulator as dts
    from spectramr.infrastructure.physics.degradations import registry as dte

    real_apply = dte.apply_degradation  # capture BEFORE patching
    chain = _chain()
    cfg = chain.to_digital_twin_config()

    seen: list[tuple[str, float]] = []

    def _spy(name, x, theta, seed=0, **kw):
        seen.append((name, round(float(theta), 6)))
        return real_apply(name, x, theta, seed=seed, **kw)

    monkeypatch.setattr(dte, "apply_degradation", _spy)

    sim = dts.DigitalTwinSimulator(
        im_size=(32, 32),
        progressive_degradations=cfg["progressive_degradations"],
        degradation_ranges=cfg["degradation_ranges"],
    )

    for cf in (0.0, 0.5, 1.0):
        seen.clear()
        sim.forward(
            torch.rand(1, 1, 32, 32),
            corruption_factor=cf,
            degradation_only=True,
            seed=11,
        )
        assert seen == [("complex_gaussian", 0.4), ("t2star_blur", 0.25)], (
            f"at cf={cf} the simulator dispatched {seen}; the degenerate range did "
            "not pin theta, so an emitted calibration would not replay"
        )
