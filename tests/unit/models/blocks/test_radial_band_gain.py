"""The band gain is phase-exact, identity at init, and reached by the generator.

Three properties carry this block, and each is the reason a different cheaper
design was rejected:

* **Phase is algebraically unchanged.** The gain is real and positive, so a
  drifting ``val_band_*`` argument localises a fault upstream instead of being
  confounded with this block. A complex gain would give no such separation, and
  the reverse loop's magnitude ceiling is phase-invariant, so an energy check
  alone cannot tell a correct band from a rotated one.
* **Identity at initialisation.** ``gamma`` starts at zero, so the arm is
  bit-identical to its control until it learns otherwise -- which is what makes
  the A/B attributable to the mechanism rather than to a reseeded model.
* **It fires on the production forward.** Registering a module and having the
  training path call it are different claims (non-negotiable 16), so the wiring
  is observed with a hook rather than read off the constructor.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.blocks.radial_band_gain import RadialBandGain  # noqa: E402

COMMON = {
    "in_channels": 4,
    "out_channels": 4,
    "features": (8, 16),
    "force_pure_kspace": True,
    "attention_type": "none",
    "use_dc": False,
    "kspace_log_scaled": False,
    "condition_with_smaps": False,
}


def _kspace(batch=2, coils=2, size=32):
    torch.manual_seed(0)
    return torch.randn(batch, coils, size, size, dtype=torch.complex64)


def _excited(n_bands=8, gamma=0.7):
    """A gain whose ``gamma`` has moved off its zero init."""
    torch.manual_seed(1)
    block = RadialBandGain(n_bands=n_bands)
    with torch.no_grad():
        block.gamma.fill_(gamma)
    return block


# ── phase exactness ───────────────────────────────────────────────────────────
def test_phase_is_unchanged_once_the_gain_is_active():
    block, x = _excited(), _kspace()
    with torch.no_grad():
        y = block(x)
    assert torch.allclose(torch.angle(x), torch.angle(y), atol=1e-5)


def test_the_magnitude_really_does_move():
    """Guards the test above from passing because the block is a no-op."""
    block, x = _excited(), _kspace()
    with torch.no_grad():
        y = block(x)
    assert not torch.allclose(x.abs(), y.abs(), atol=1e-4)


def test_every_gain_is_strictly_positive():
    """A sign flip is a pi phase error wearing a magnitude's name."""
    assert bool((_excited().band_gains(_kspace()) > 0).all())


def test_the_gain_is_bounded_by_max_log_gain():
    torch.manual_seed(2)
    block = RadialBandGain(n_bands=8, max_log_gain=0.1)
    with torch.no_grad():
        block.gamma.fill_(50.0)  # far past any bound
    gains = block.band_gains(_kspace())
    # fp32 lands ~5e-8 past exp(0.1), so the bound is checked to float tolerance
    # rather than exactly -- the property is "cannot run away", not "cannot round".
    assert float(gains.max()) <= math.exp(0.1) * (1 + 1e-6)
    assert float(gains.min()) >= math.exp(-0.1) * (1 - 1e-6)
    # And the clamp is doing the work: gamma=50 would otherwise be astronomical.
    assert float(gains.max()) == pytest.approx(math.exp(0.1), rel=1e-5)


# ── identity at init ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("complex_input", [True, False])
def test_identity_at_initialisation(complex_input):
    torch.manual_seed(3)
    block = RadialBandGain(n_bands=8)
    x = _kspace() if complex_input else torch.randn(2, 4, 32, 32)
    with torch.no_grad():
        assert torch.allclose(block(x), x, atol=1e-6)


def test_gains_are_exactly_one_at_initialisation():
    assert torch.allclose(RadialBandGain(n_bands=8).band_gains(_kspace()), torch.ones(2, 8))


def test_the_layout_round_trips():
    """Interleaved in, interleaved out -- a silent dtype change is a shape bug later."""
    block = _excited()
    x = torch.randn(2, 4, 32, 32)
    y = block(x)
    assert y.shape == x.shape and y.dtype == x.dtype and not y.is_complex()


# ── the bands must differentiate, or K=1 is hiding inside K=8 ─────────────────
def test_the_bands_do_not_all_take_the_same_gain():
    """The degenerate solution this mechanism has to beat is a global scalar."""
    gains = _excited().band_gains(_kspace())[0]
    assert float(gains.std()) > 1e-3


def test_corners_outside_the_disc_are_left_alone():
    """Bands were never fitted there; scaling it would be inventing a band."""
    block, x = _excited(), _kspace(batch=1, coils=1)
    with torch.no_grad():
        y = block(x)
    assert y[0, 0, 0, 0] == pytest.approx(complex(x[0, 0, 0, 0]), rel=1e-5)


# ── refusals ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n_bands", [0, 1, -2])
def test_fewer_than_two_bands_raises(n_bands):
    with pytest.raises(ValueError, match="n_bands must be >= 2"):
        RadialBandGain(n_bands=n_bands)


def test_an_empty_annulus_raises_rather_than_fitting_on_nothing():
    with pytest.raises(ValueError, match="empty on a"):
        RadialBandGain(n_bands=64)(_kspace(size=4))


def test_a_missing_time_embedding_raises_when_one_was_declared():
    """A timestep-conditioned gain that drops the timestep is a different block."""
    with pytest.raises(ValueError, match="received no time_embedding"):
        RadialBandGain(n_bands=4, time_embed_dim=8)(_kspace())


def test_a_mismatched_time_embedding_width_raises():
    block = RadialBandGain(n_bands=4, time_embed_dim=8)
    with pytest.raises(ValueError, match="does not match time_embed_dim"):
        block(_kspace(), time_embedding=torch.randn(2, 5))


def test_an_odd_channel_count_raises():
    with pytest.raises(ValueError, match="even channel"):
        RadialBandGain(n_bands=4)(torch.randn(2, 3, 32, 32))


# ── no allocation in the loop ─────────────────────────────────────────────────
def test_the_bin_grid_is_built_once_per_grid(monkeypatch):
    """Rebuilding the index every step would allocate in the training loop.

    Counts calls into ``radial_bins`` rather than the cache's length: a cache
    that is written and never read leaves the length at 1 while rebuilding on
    every forward, and the first version of this check passed that planted
    violation.
    """
    import spectramr.models.blocks.radial_band_gain as mod

    calls = []
    real = mod.radial_bins
    monkeypatch.setattr(mod, "radial_bins", lambda *a, **k: (calls.append(a[:2]), real(*a, **k))[1])
    block, x = _excited(), _kspace()
    with torch.no_grad():
        for _ in range(6):
            block(x)
    assert len(calls) == 1, f"bin grid rebuilt {len(calls)} times for one grid"


def test_a_second_grid_size_builds_its_own_index(monkeypatch):
    """The cache is keyed on the grid, so a new size must not reuse the old one."""
    import spectramr.models.blocks.radial_band_gain as mod

    calls = []
    real = mod.radial_bins
    monkeypatch.setattr(mod, "radial_bins", lambda *a, **k: (calls.append(a[:2]), real(*a, **k))[1])
    block = _excited()
    with torch.no_grad():
        block(_kspace(size=32))
        block(_kspace(size=16))
    assert len(calls) == 2 and calls[0] != calls[1]


# ── reachability: the generator builds it and the forward calls it ────────────
def _generator(**extra):
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceColdDiffusionGenerator,
    )

    torch.manual_seed(0)
    return KSpaceColdDiffusionGenerator(**COMMON, **extra).eval()


def test_the_knob_is_off_by_default():
    assert _generator().radial_band_gain is None


def test_declaring_bands_builds_the_block():
    assert isinstance(_generator(radial_band_gain_bands=8).radial_band_gain, RadialBandGain)


def test_the_block_fires_during_the_generator_forward():
    """Observed, not inferred: registering a module is the easy half."""
    gen = _generator(radial_band_gain_bands=8)
    fired: list[tuple[int, ...]] = []
    gen.radial_band_gain.register_forward_hook(lambda m, i, o: fired.append(tuple(o.shape)))
    with torch.no_grad():
        gen(torch.randn(1, 4, 32, 32), torch.zeros(1, dtype=torch.long))
    assert len(fired) == 1


def test_an_arm_with_the_block_reproduces_its_control_at_init():
    """The A/B is attributable only if step 0 is bit-identical."""
    x, t = torch.randn(1, 4, 32, 32), torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        control = _generator()(x, t)
        gained = _generator(radial_band_gain_bands=8)(x, t)
    assert torch.allclose(control, gained, atol=1e-6)


def test_a_trained_gain_changes_the_generator_output():
    """Otherwise the arm is an expensive way to rerun the control."""
    x, t = torch.randn(1, 4, 32, 32), torch.zeros(1, dtype=torch.long)
    gen = _generator(radial_band_gain_bands=8)
    with torch.no_grad():
        control = _generator()(x, t)
        gen.radial_band_gain.gamma.fill_(0.5)
        gained = gen(x, t)
    assert not torch.allclose(control, gained, atol=1e-5)


def test_the_generator_refuses_a_single_band():
    with pytest.raises(ValueError, match="0 \\(disabled\\) or >= 2"):
        _generator(radial_band_gain_bands=1)


def test_the_gain_is_applied_before_the_pre_dc_capture():
    """`lambda_pre_dc_kspace` must score what the model emits, not one stage earlier."""
    import ast
    import inspect
    import pathlib

    from spectramr.models.generators import kspace_cold_diffusion_generator as mod

    src = pathlib.Path(inspect.getfile(mod)).read_text()
    tree = ast.parse(src)
    gain_line = next(
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "radial_band_gain"
    )
    capture_line = next(
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "x_pre_dc" for t in n.targets)
    )
    assert gain_line < capture_line
