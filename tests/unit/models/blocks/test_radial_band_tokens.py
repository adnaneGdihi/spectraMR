"""The token bank is phase-exact, identity at init, and differentiates on both axes.

Four properties carry this block, and each is the reason a cheaper design was
rejected:

* **Phase is algebraically unchanged**, because the scale is real and strictly
  positive -- so a drifting ``val_band_*`` argument localises a fault elsewhere
  instead of being confounded with this block. The claim is scoped to the block:
  downstream ``ComplexConv2d`` and ``ModReLU`` still rotate phase.
* **Identity at initialisation, bit-exact.** ``exp(0) == 1.0`` and ``z * 1.0 == z``
  are exact in IEEE-754, so the assertion is ``torch.equal``, not ``allclose``.
  An arm carrying this reproduces its control at step 0 or the A/B is not
  attributable to the mechanism.
* **Exactly one zero-initialised group, and it is not a saddle.** Issue #471 is
  what happens when a zero gate is *composed with* an identity-at-init inner
  block: both gradients vanish together and nothing ever moves. Here the inner
  path is random, so the write head moves at step 0 and the bank at step 1 --
  asserted over two optimiser steps rather than argued.
* **Both axes differentiate.** A bank that is constant across annuli is a global
  embedding; one constant across rungs has an axis it is not using. Both are read
  from the bank without a forward, because a mechanism whose own decision can
  only be inferred from a ratio of outputs is graded by proxy.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.blocks.radial_band_tokens import (  # noqa: E402
    COMPOUNDED_LOG_CEILING,
    DEFAULT_MAX_LOG_SCALE,
    RadialBandTokens,
)

#: Complex widths of a three-level decoder, in run order, and the grid each runs
#: on against a 64-square input -- so a level with a 1/2 and a 1/4 crop factor is
#: actually exercised. At two levels every grid is full-resolution and the
#: per-level binning is never tested.
LEVELS = (32, 16, 8)
GRIDS = ((16, 32), (32, 16), (64, 8))
FULL = (64, 64)


def _block(n_bands=8, n_rungs=29, **kwargs):
    torch.manual_seed(0)
    return RadialBandTokens(level_channels=LEVELS, n_bands=n_bands, n_rungs=n_rungs, **kwargs)


def _kspace(batch=2, channels=8, size=64):
    torch.manual_seed(0)
    return torch.randn(batch, channels, size, size, dtype=torch.complex64)


def _rung(batch=2):
    return torch.arange(batch, dtype=torch.long) * 7


def _excited(n_bands=8, **kwargs):
    """A block whose write head and bank have moved off their initialisation."""
    block = _block(n_bands=n_bands, **kwargs)
    torch.manual_seed(1)
    with torch.no_grad():
        for head in block.write_heads:
            head.weight.normal_(0, 0.5)
        block.bank.normal_(0, 1.0)
    return block


# ── phase exactness ───────────────────────────────────────────────────────────
def test_phase_is_unchanged_once_the_head_is_active():
    """The algebraic statement, not ``angle``, which is ill-defined near zero."""
    block, x = _excited(), _kspace()
    with torch.no_grad():
        ratio = block(x, _rung(), level=2, full_size=FULL) / x
    assert ratio.imag.abs().max() < 1e-6
    assert bool((ratio.real > 0).all())


def test_the_magnitude_really_does_move():
    """Without this the phase test above passes for a block that does nothing."""
    block, x = _excited(), _kspace()
    with torch.no_grad():
        y = block(x, _rung(), level=2, full_size=FULL)
    assert (y.abs() - x.abs()).abs().max() > 1e-3


def test_every_scale_is_strictly_positive():
    """A sign flip is a pi phase error wearing a magnitude's name."""
    assert bool((_excited().band_scales(level=2) > 0).all())


# ── the bound ─────────────────────────────────────────────────────────────────
def test_the_scale_is_bounded_by_max_log_scale():
    block = _block(max_log_scale=0.1)
    torch.manual_seed(2)
    with torch.no_grad():
        block.write_heads[2].weight.normal_(0, 50.0)
        block.bank.normal_(0, 5.0)
    scales = block.band_scales(level=2).detach()
    ceiling, floor = float(torch.exp(torch.tensor(0.1))), float(torch.exp(torch.tensor(-0.1)))
    # Saturated at both ends, so the clamp is load-bearing rather than
    # incidentally satisfied by a head that never got near the bound.
    assert float(scales.max()) == pytest.approx(ceiling, rel=1e-5)
    assert float(scales.min()) == pytest.approx(floor, rel=1e-5)


def test_the_compounded_ceiling_matches_a_single_site_gain():
    """``n`` sites reach ``exp(n * L)``; the default is chosen so four reach A's.

    A later default has to argue with this test rather than drift past it.
    """
    assert pytest.approx(COMPOUNDED_LOG_CEILING) == 4 * DEFAULT_MAX_LOG_SCALE


# ── identity at initialisation ────────────────────────────────────────────────
@pytest.mark.parametrize("complex_input", [True, False])
def test_identity_at_initialisation(complex_input):
    """Bit-exact, so the A/B delta is the mechanism and not a reseeded model."""
    block = _block()
    x = _kspace() if complex_input else torch.randn(2, 16, 64, 64)
    with torch.no_grad():
        assert torch.equal(block(x, _rung(), level=2, full_size=FULL), x)


def test_scales_are_exactly_one_at_initialisation():
    assert torch.equal(_block().band_scales(level=2), torch.ones(29, 8, 8))


def test_exactly_one_parameter_group_is_zero_initialised():
    """The #471 detector: a second zero-init added later turns this red."""
    zeroed = [n for n, p in _block().named_parameters() if float(p.detach().abs().max()) == 0.0]
    assert zeroed == [
        f"write_heads.{i}.{w}" for i in range(len(LEVELS)) for w in ("weight", "bias")
    ]


def test_the_zero_init_is_not_a_saddle():
    """#471 needs BOTH gradients to vanish; here only the upstream one does."""
    block, x, rung = _block(), _kspace(), _rung()

    def backward():
        block.zero_grad(set_to_none=True)
        block(x, rung, level=2, full_size=FULL).abs().square().mean().backward()

    backward()
    assert block.write_heads[2].weight.grad.abs().max() > 0
    assert block.bank.grad.abs().max() == 0
    torch.optim.SGD(block.parameters(), lr=1e-2).step()
    backward()
    assert block.bank.grad.abs().max() > 0


# ── differentiation on both axes, readable without a forward ──────────────────
def test_band_scales_reads_without_a_forward():
    assert _block().band_scales(level=2).shape == (29, 8, 8)


def test_the_annuli_do_not_all_take_the_same_scale():
    """The degenerate solution to beat is a global embedding."""
    assert _excited().differentiation(level=2)["across_annuli"] > 1e-3


def test_the_rungs_do_not_all_take_the_same_scale():
    """The degenerate solution to beat is one compromise across R=2 and R=32."""
    assert _excited().differentiation(level=2)["across_rungs"] > 1e-3


def test_differentiation_is_exactly_zero_at_initialisation():
    """So a non-zero reading is attributable to training, not to the draw."""
    assert _block().differentiation(level=2) == {"across_annuli": 0.0, "across_rungs": 0.0}


# ── layout and partition ──────────────────────────────────────────────────────
def test_the_layout_round_trips():
    block, x = _excited(), torch.randn(2, 16, 64, 64)
    with torch.no_grad():
        y = block(x, _rung(), level=2, full_size=FULL)
    assert y.shape == x.shape and y.dtype == x.dtype and not y.is_complex()


def test_corners_outside_the_disc_are_left_alone():
    """The bands were never fitted there, so scaling them would invent a band."""
    block, x = _excited(), _kspace()
    with torch.no_grad():
        y = block(x, _rung(), level=2, full_size=FULL)
    assert torch.equal(y[0, 0, 0, 0], x[0, 0, 0, 0])


@pytest.mark.parametrize(
    ("level", "size", "channels", "expected"), [(0, 16, 32, 2), (1, 32, 16, 4), (2, 64, 8, 8)]
)
def test_a_cropped_level_carries_its_own_share_of_the_global_annuli(
    level, size, channels, expected
):
    """``n_bands * (H_i/H_full)`` bins, so annulus k is one physical band throughout."""
    block = _block()
    x = torch.randn(1, channels, size, size, dtype=torch.complex64)
    block(x, torch.tensor([0]), level=level, full_size=FULL)
    assert block._grid(size, size, FULL, x.device)[3] == expected


def test_the_vectorised_pooling_matches_a_per_sample_reduction():
    """Defends the scatter by correctness, including the corner masking."""
    from spectramr.infrastructure.physics.radial_bands import band_reduce

    block, x = _block(), _kspace()
    index, inside, counts, n_level = block._grid(64, 64, FULL, x.device)
    amplitude = x.abs().mean(dim=1)
    reference = torch.stack(
        [
            band_reduce(amplitude[i], index, inside, n_level) / counts.clamp_min(1.0)
            for i in range(2)
        ]
    )
    pooled = torch.zeros(2, n_level).index_add_(
        1, index.reshape(-1), (amplitude * inside).reshape(2, -1).float()
    ) / counts.clamp_min(1.0)
    assert torch.allclose(pooled, reference, atol=1e-5)


# ── refusals ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n_bands", [0, 1, -2])
def test_fewer_than_two_bands_raises(n_bands):
    with pytest.raises(ValueError, match="n_bands must be >= 2"):
        _block(n_bands=n_bands)


@pytest.mark.parametrize("n_rungs", [0, -1])
def test_a_non_positive_rung_count_raises(n_rungs):
    with pytest.raises(ValueError, match="n_rungs must be >= 1"):
        _block(n_rungs=n_rungs)


def test_no_levels_raises():
    with pytest.raises(ValueError, match="at least one up-step"):
        RadialBandTokens(level_channels=())


def test_a_band_count_that_does_not_divide_at_a_cropped_level_raises():
    block = _block(n_bands=6)
    with pytest.raises(ValueError, match="does not divide"):
        block(
            torch.randn(1, 32, 16, 16, dtype=torch.complex64),
            torch.tensor([0]),
            level=0,
            full_size=FULL,
        )


def test_a_level_reduced_to_one_annulus_raises():
    """One annulus is a global scalar, which the mechanism already has in A."""
    block = _block(n_bands=4)
    with pytest.raises(ValueError, match="a global scalar"):
        block(
            torch.randn(1, 32, 16, 16, dtype=torch.complex64),
            torch.tensor([0]),
            level=0,
            full_size=FULL,
        )


def test_an_empty_annulus_raises_rather_than_fitting_on_nothing():
    block = RadialBandTokens(level_channels=(4,), n_bands=16)
    with pytest.raises(ValueError, match="are empty"):
        block(
            torch.randn(1, 4, 8, 8, dtype=torch.complex64),
            torch.tensor([0]),
            level=0,
            full_size=(8, 8),
        )


def test_a_non_uniform_crop_raises():
    block = RadialBandTokens(level_channels=(4,), n_bands=8)
    with pytest.raises(ValueError, match="not a uniform crop"):
        block(
            torch.randn(1, 4, 32, 64, dtype=torch.complex64),
            torch.tensor([0]),
            level=0,
            full_size=FULL,
        )


@pytest.mark.parametrize("rung", [torch.tensor([29]), torch.tensor([-1])])
def test_a_rung_outside_the_bank_raises(rung):
    """Clamping would silently re-map every step past the last rung (pitfall 9)."""
    block = _block()
    with pytest.raises(ValueError, match="outside the bank"):
        block(_kspace(batch=1), rung, level=2, full_size=FULL)


def test_a_float_rung_raises():
    """A normalised t collapses every sample onto rung 0, wired but inert."""
    with pytest.raises(ValueError, match="integer step index"):
        _block()(_kspace(batch=1), torch.tensor([0.5]), level=2, full_size=FULL)


@pytest.mark.parametrize(
    "rung", [torch.zeros(3, dtype=torch.long), torch.zeros(2, 1, dtype=torch.long)]
)
def test_a_rung_of_the_wrong_shape_raises(rung):
    with pytest.raises(ValueError, match=r"rung must be \[B\]"):
        _block()(_kspace(), rung, level=2, full_size=FULL)


def test_a_channel_count_that_disagrees_with_the_declared_width_raises():
    """A mismatched width broadcasts silently instead of failing."""
    with pytest.raises(ValueError, match="complex channels"):
        _block()(_kspace(channels=4), _rung(), level=2, full_size=FULL)


def test_an_odd_channel_count_raises():
    with pytest.raises(ValueError, match="even channels"):
        _block()(torch.randn(2, 15, 64, 64), _rung(), level=2, full_size=FULL)


# ── the grid is built once ────────────────────────────────────────────────────
def test_the_bin_grid_is_built_once_per_grid(monkeypatch):
    """Counts calls into ``radial_bins``, not ``len(_grid_cache)``.

    A cache that is written and never read leaves the length at 1 while
    rebuilding every forward -- the planted violation the first version of the
    equivalent check for ``RadialBandGain`` passed.
    """
    import spectramr.models.blocks.radial_band_tokens as module

    calls: list[tuple[int, int, int]] = []
    real = module.radial_bins

    def counted(height, width, n_bins, device="cpu"):
        calls.append((height, width, n_bins))
        return real(height, width, n_bins, device)

    monkeypatch.setattr(module, "radial_bins", counted)
    block, x, rung = _block(), _kspace(), _rung()
    with torch.no_grad():
        for _ in range(6):
            block(x, rung, level=2, full_size=FULL)
    assert len(calls) == 1


def test_each_decoder_level_builds_its_own_index(monkeypatch):
    import spectramr.models.blocks.radial_band_tokens as module

    calls: list[tuple[int, int, int]] = []
    real = module.radial_bins

    def counted(height, width, n_bins, device="cpu"):
        calls.append((height, width, n_bins))
        return real(height, width, n_bins, device)

    monkeypatch.setattr(module, "radial_bins", counted)
    block = _block()
    with torch.no_grad():
        for level, (size, channels) in enumerate(GRIDS):
            block(
                torch.randn(1, channels, size, size, dtype=torch.complex64),
                torch.tensor([0]),
                level=level,
                full_size=FULL,
            )
    assert calls == [(16, 16, 2), (32, 32, 4), (64, 64, 8)]


def test_the_mechanism_can_reach_the_band_the_probe_measures():
    """The executable form of the backlog's ``outer_band_retention`` row.

    That row cannot grade a trained arm: the spectral-transfer probe measures
    UNTRAINED networks by design, and this block is identity at initialisation,
    so from a config alone it reads its control exactly. What the probe can
    answer is whether the mechanism is able to move the band at all -- so the
    write head is perturbed off zero and the retention re-measured. No movement
    means the block never reached the band, whatever else it learned.

    The perturbation is a random draw, so the *sign* of the change carries
    nothing: the assertion is two-sided on purpose, and a reading that falls is
    as much evidence of reach as one that rises.
    """
    from spectramr.infrastructure.validation.spectral_transfer_probe import (
        measure_radial_transfer,
    )
    from spectramr.models.generators.complex_unet import ComplexUNet

    common = {
        "in_channels": 8,
        "out_channels": 8,
        "features": (16, 32, 64),
        "time_embedding_dim": 32,
        "feature_domain": "kspace",
        "kspace_feature_norm": "none",
    }
    torch.manual_seed(3)
    net = ComplexUNet(**common, radial_band_tokens_bands=8, radial_band_tokens_rungs=29).eval()

    def retention():
        # n_bins from the grid: the probe's default 32 leaves annuli empty at
        # this size and raises rather than dividing by zero.
        torch.manual_seed(3)
        return measure_radial_transfer(
            net, size=64, channels=8, n_bins=8, repeats=2, timestep=14
        ).outer_band_retention

    at_init = retention()
    with torch.no_grad():
        net.radial_band_tokens.write_heads[2].weight.normal_(0, 1.5)
        net.radial_band_tokens.bank.normal_(0, 1.5)
    assert abs(retention() - at_init) > 1e-3
