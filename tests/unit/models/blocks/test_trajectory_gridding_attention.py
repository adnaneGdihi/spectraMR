"""Contract tests for gridding-by-attention over off-grid samples."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.blocks.trajectory_gridding_attention import (
    KERNEL_SIGMA_CELLS,
    TrajectoryFourierFeatures,
    TrajectoryGriddingAttention,
)

IM = (32, 32)
SPOKES, READOUT = 64, 64


def _block(**overrides) -> TrajectoryGriddingAttention:
    kwargs = {
        "im_size": IM,
        "complex_channels": 1,
        "num_features": 64,
        "num_heads": 4,
    }
    kwargs.update(overrides)
    return TrajectoryGriddingAttention(**kwargs)


def _trajectory() -> torch.Tensor:
    from spectramr.infrastructure.physics.nufft_ops import GoldenAngleTrajectory

    return GoldenAngleTrajectory(IM, SPOKES, READOUT).generate(num_frames=1).squeeze(0)


def _phantom() -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, IM[0]), torch.linspace(-1.0, 1.0, IM[1]), indexing="ij"
    )
    radius = (xx.pow(2) + yy.pow(2)).sqrt()
    magnitude = torch.nn.functional.avg_pool2d(
        ((radius < 0.7).float() * (1.0 - 0.5 * radius))[None, None], 5, 1, 2
    )[0, 0]
    return (magnitude * torch.exp(1j * (1.2 * xx + 0.8 * yy)))[None, None]


def _as_complex(interleaved: torch.Tensor) -> torch.Tensor:
    return torch.complex(interleaved[:, 0::2], interleaved[:, 1::2])


def test_a_global_phase_roll_passes_through_the_block() -> None:
    """f(e^{i*phi} y) == e^{i*phi} f(y).

    Scores are inner products of positive features of the COORDINATES, so they
    cannot see the data phase at all; the values stay complex. The density
    weight and the deapodization are both real, so neither breaks it either. A
    block that scored on the real part would rotate with the input and have to
    learn this invariance from data instead of holding it by construction.
    """
    block = _block()
    torch.manual_seed(0)
    kdata = torch.randn(2, 1, SPOKES * READOUT, dtype=torch.complex64)
    traj = _trajectory()
    mask = torch.ones(2, SPOKES * READOUT)
    phi = 0.7331
    rotation = torch.exp(torch.tensor(1j * phi))

    with torch.no_grad():
        plain = _as_complex(block(kdata, traj, mask))
        rolled = _as_complex(block(kdata * rotation, traj, mask))
    delta = (rolled - plain * rotation).abs().max().item()
    assert delta < 1e-5, f"global phase was not carried through: max delta {delta:.3e}"


def test_unacquired_samples_do_not_reach_the_grid() -> None:
    """The mask enters the key sum, so a dropped sample contributes nothing."""
    block = _block()
    torch.manual_seed(1)
    count = SPOKES * READOUT
    kdata = torch.randn(1, 1, count, dtype=torch.complex64)
    traj = _trajectory()
    mask = torch.zeros(1, count)
    mask[:, : count // 2] = 1.0

    corrupted = kdata.clone()
    corrupted[..., count // 2 :] += 500.0
    with torch.no_grad():
        assert torch.allclose(block(kdata, traj, mask), block(corrupted, traj, mask), atol=1e-4)


def test_duplicating_a_sample_does_not_double_its_contribution() -> None:
    """Planted violation for the density-compensation property.

    The linear-attention denominator sums kernel mass over the samples a grid
    point sees, which is what a density compensation function does. Remove it
    and the sampling density leaks into the VALUE.

    The probe is a constant unit field: its gridded value must be that constant
    however the samples fall, so any shift under duplication is density leaking
    through. Repeating an eighth of the samples gives a median shift of 0.0000%
    with the normaliser and 12.58% without -- the duplication ratio passed
    through untouched.

    ``sigma_cells`` is widened for this test and the reason is itself a
    finding: at the production width and 32x32, a grid point's kernel reaches
    essentially no sample, so BOTH variants read zero and the violation stops
    turning red. A vacuous test is the failure mode non-negotiable 15 exists to
    prevent, so the regime here is chosen to make the property observable.
    """
    import spectramr.models.blocks.trajectory_gridding_attention as module

    block = _block(sigma_cells=8.0)
    torch.manual_seed(2)
    count = 4000
    repeat = count // 8
    kdata = torch.ones(1, 1, count, dtype=torch.complex64)
    traj = (torch.rand(1, 2, count) * 2.0 - 1.0) * math.pi
    doubled_k = torch.cat([kdata, kdata[..., :repeat]], dim=-1)
    doubled_t = torch.cat([traj, traj[..., :repeat]], dim=-1)

    def _median_shift() -> float:
        with torch.no_grad():
            plain = block(kdata, traj, torch.ones(1, count))
            doubled = block(doubled_k, doubled_t, torch.ones(1, count + repeat))
        return ((doubled - plain).abs() / (plain.abs() + 1e-6)).median().item()

    assert _median_shift() < 0.01, "the normaliser is not compensating for density"

    def _unnormalised(q, k, v, observed, *, eps=1e-6):
        k = k * observed.reshape(observed.shape[0], 1, 1, -1)
        if torch.is_complex(v):
            q, k = q.to(v.dtype), k.to(v.dtype)
        kv = torch.einsum("bhdn,bhen->bhde", k, v)
        return torch.einsum("bhdn,bhde->bhen", q, kv)

    original = module.sample_density_attention
    module.sample_density_attention = _unnormalised
    try:
        assert _median_shift() > 0.05, (
            "dropping the denominator did not make duplication visible, so this "
            "test is vacuous and watches nothing"
        )
    finally:
        module.sample_density_attention = original


def test_the_kernel_is_compact_even_though_its_width_is_wrong() -> None:
    """What survives of the Bochner claim: the kernel is still local.

    ``<phi(k_i), phi(k_j)>`` is a shift-invariant kernel in the coordinate
    difference, so the block interpolates rather than mixing all of k-space.
    The WIDTH of that kernel is a separate claim and a false one at the
    gridding scale -- see
    ``test_the_realised_kernel_is_far_from_the_gaussian_it_claims``.
    """
    block = _block()
    features = block.features
    radii = torch.tensor([0.0, 1.0, 3.0, 6.0, 20.0]) * 2.0 * math.pi / IM[0]
    origin = torch.zeros(1, 2, 1)
    offsets = torch.stack([radii, torch.zeros_like(radii)]).unsqueeze(0)
    with torch.no_grad():
        f0, fr = features(origin), features(offsets)
    kernel = torch.einsum("bfn,bfm->nm", f0, fr)[0] / features.num_features
    kernel = kernel / kernel[0]
    assert kernel[0] == pytest.approx(1.0, abs=1e-5)
    assert kernel[4].abs() < 0.2, (
        f"kernel is still {kernel[4]:.3f} at 20 grid cells; it is not compact "
        f"and the block is not interpolating"
    )


def test_the_interpolation_kernel_is_learnable() -> None:
    """The Fourier frequencies are a parameter; a buffer would pin the kernel."""
    block = _block()
    assert block.features.frequencies.requires_grad
    assert any(p is block.features.frequencies for p in block.parameters())


def test_real_input_raises_rather_than_dropping_phase() -> None:
    """No silent phase loss (non-negotiable 3)."""
    block = _block()
    with pytest.raises(ValueError, match="must be complex"):
        block(torch.zeros(1, 1, 64), _trajectory(), torch.ones(1, 64))


def test_a_channel_count_mismatch_raises() -> None:
    """A silent broadcast here would grid the wrong coil."""
    block = _block(complex_channels=2)
    with pytest.raises(ValueError, match="complex channels"):
        block(torch.zeros(1, 1, 64, dtype=torch.complex64), _trajectory(), torch.ones(1, 64))


def test_coordinates_must_be_two_dimensional() -> None:
    """A 3-D trajectory would silently index the wrong axis."""
    features = TrajectoryFourierFeatures(num_features=8)
    with pytest.raises(ValueError, match="2-D k-space coordinates"):
        features(torch.zeros(1, 3, 16))


def test_the_untrained_block_is_measured_against_the_fixed_kernel() -> None:
    """Characterisation: where the untrained block stands against gridding.

    The block's whole claim is that it replaces a fixed Kaiser-Bessel kernel,
    so the number that matters is how it compares to one BEFORE training. The
    operator form is now right -- a density-compensated sum with deapodization
    -- and the residual gap is the random-feature estimator, which cannot reach
    a kernel this peaked at any feature count (see the underflow test below).

    This asserts the measured state rather than the intended one, so a change
    that closes the gap turns this red and has to update the number. An
    assertion of the intended state would have been green on a block that does
    not interpolate.
    """
    from spectramr.infrastructure.physics.fft_ops import ifft2c
    from spectramr.infrastructure.physics.gridding import RadialDCF
    from spectramr.infrastructure.physics.nufft_ops import NUFFTForwardModel

    traj = _trajectory()
    nufft = NUFFTForwardModel(im_size=IM)
    dcf = RadialDCF(SPOKES, READOUT).forward(traj.transpose(0, 1))
    x0 = _phantom()
    y = nufft.forward_project(x0, traj)

    def error(estimate: torch.Tensor) -> float:
        gain = (estimate.conj() * x0).sum() / (
            (estimate.conj() * estimate).sum().real.clamp_min(1e-12)
        )
        return ((x0 - gain * estimate).abs().norm() / x0.abs().norm()).item()

    baseline = error(nufft.adjoint_project(y * dcf.to(y.dtype), traj))
    with torch.no_grad():
        gridded = _block(num_features=256)(y, traj, torch.ones(1, SPOKES * READOUT))
    learned = error(ifft2c(torch.complex(gridded[:, 0::2], gridded[:, 1::2])))

    assert baseline < 0.10, f"the fixed-kernel baseline moved: {baseline:.4f}"
    assert learned / baseline > 10.0, (
        f"the untrained block is now within {learned / baseline:.1f}x of the "
        f"fixed kernel (was 22.8x). If this is a real improvement, update the "
        f"number and the module docstring; the random-feature estimator is the "
        f"cause, and an exact kernel reaches 1.3x."
    )


def test_the_random_features_underflow_at_the_gridding_width() -> None:
    """Characterisation: why the block is not ready to be an arm.

    ``exp(w . u - |u|^2)`` computes an order-1 kernel as a tiny number times a
    huge one. A gridding kernel is narrow, ``u`` scales as ``1/sigma``, and the
    exponent leaves float32's range -- so the features the attention multiplies
    are mostly exactly zero.

    This asserts the measured state. A change that rescues the estimator turns
    it red and must update the number and the module docstring.
    """
    block = _block(num_features=256)
    features = block.features(_trajectory()[None])
    zero_fraction = float((features == 0).float().mean())
    assert zero_fraction > 0.4, (
        f"only {zero_fraction:.2%} of features underflow now (was 57%). If the "
        f"estimator was fixed, update this and the docstring."
    )


def test_the_realised_kernel_is_far_from_the_gaussian_it_claims() -> None:
    """The underflow above shows up as the wrong kernel, both too narrow and too flat."""
    import math

    block = _block(num_features=256)
    features = block.features
    sigma = float(features.log_width.exp())
    origin = features(torch.zeros(1, 2, 1))[0]
    reference = float((origin * origin).sum())

    half_cell = 0.5 * (2.0 * math.pi / IM[0])
    near = features(torch.tensor([[0.0], [half_cell]])[None])[0]
    realised = float((origin * near).sum()) / max(reference, 1e-30)
    exact = math.exp(-(half_cell**2) / (2 * sigma**2))
    assert realised < 0.5 * exact, (
        f"realised {realised:.4f} is no longer far below the Gaussian's {exact:.4f}; "
        f"the estimator may have been fixed -- re-measure the gridding ratio."
    )


def test_more_features_do_not_rescue_the_estimator() -> None:
    """Why the fix is not `num_features`, stated as arithmetic rather than opinion.

    The random-feature sum must estimate ``E[exp(w . 2u)] = exp(2|u|^2)``, which
    at the k-space edge of this block's regime is ``e^1045``. Computed in EXACT
    log domain -- so precision and accumulation order are both out of the
    picture -- 256 features fall short by around 938 in the log, and a
    four-thousand-fold increase closes barely a tenth of it. The shortfall is
    variance, and only a compactly-supported evaluation avoids it.
    """
    import math

    sigma = KERNEL_SIGMA_CELLS * (2.0 * math.pi / IM[0])
    u = torch.tensor([0.0, math.pi], dtype=torch.float64) / sigma
    exact_log = 2.0 * float(u.pow(2).sum())

    torch.manual_seed(0)
    shortfalls = []
    for count in (256, 65536):
        frequencies = torch.randn(2, count, dtype=torch.float64)
        estimate = float(
            torch.logsumexp(frequencies.T @ (2 * u), dim=0) - math.log(count)
        )
        shortfalls.append(exact_log - estimate)

    assert shortfalls[0] > 500.0, f"estimator is no longer short: {shortfalls[0]:.0f}"
    assert shortfalls[1] > 0.8 * shortfalls[0], (
        f"256x more features closed {shortfalls[0] - shortfalls[1]:.0f} of "
        f"{shortfalls[0]:.0f}; if that ratio has changed, re-derive the conclusion"
    )
