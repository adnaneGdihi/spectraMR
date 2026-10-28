"""Round-trip correctness for faithful multi-acquisition synthesis.

The decisive proof that the synthesis is *faithful*: synthesise the acquisition
stack from a KNOWN field, run the method's textbook closed-form inversion, and
assert it recovers the field. No M4Raw data needed — pure physics round-trip.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.multi_acquisition import (
    PAIRED_METHODS,
    MultiAcquisitionSimulator,
    invert_afi,
    invert_bloch_siegert,
    invert_bssfp_banding,
    invert_dual_echo,
    invert_double_angle,
    invert_mrf,
)


def _phantom(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    img = 0.5 + torch.rand(2, 1, 32, 32, generator=g)  # M0 proxy, strictly > 0
    return img, torch.Generator().manual_seed(seed + 100)


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown multi-acquisition method"):
        MultiAcquisitionSimulator("triple_angle")


def test_double_angle_round_trip_recovers_b1() -> None:
    sim = MultiAcquisitionSimulator("double_angle")
    img, g = _phantom()
    res = sim(img, generator=g)
    assert res.stack.shape == (2, 2, 32, 32)
    b1_hat = invert_double_angle(res.stack, sim.flip_rad)
    # long-TR double angle is exact up to numerical noise
    assert torch.allclose(b1_hat, res.field, atol=2e-2)


def test_dual_echo_round_trip_recovers_b0() -> None:
    sim = MultiAcquisitionSimulator("dual_echo")
    img, g = _phantom(1)
    res = sim(img, generator=g)
    assert res.echoes is not None and res.echoes.shape == (2, 2, 32, 32)
    b0_hat = invert_dual_echo(res.stack, sim.te1_ms, sim.te2_ms)
    # phase difference recovers B0 (Hz) directly; allow sub-Hz numerical error
    assert torch.allclose(b0_hat, res.field, atol=1.0)


def test_bssfp_banding_is_a_paired_method() -> None:
    assert "bssfp_banding" in PAIRED_METHODS


def test_bssfp_banding_stack_is_complex_n_phase_cycles() -> None:
    sim = MultiAcquisitionSimulator("bssfp_banding", n_phase_cycles=4)
    img, g = _phantom(11)
    res = sim(img, generator=g)
    # phase-cycled stack [B, N, H, W], complex (DFT needs magnitude + phase)
    assert res.stack.shape == (2, 4, 32, 32)
    assert res.stack.is_complex()
    # field is the B0 offset [B, 1, H, W] in Hz, within the synthesis range
    assert res.field.shape == (2, 1, 32, 32)
    assert not res.field.is_complex()


def test_bssfp_banding_round_trip_recovers_b0_structure() -> None:
    # Synthesise a phase-cycled bSSFP stack from a KNOWN B0 field, then invert
    # with the first-harmonic phase-cycle DFT and recover the field's structure.
    # The DFT recovers Δf up to a constant phase (angle(c1)); compare after
    # removing a constant (Pearson correlation + demeaned RMSE).
    sim = MultiAcquisitionSimulator("bssfp_banding", n_phase_cycles=8)
    img, g = _phantom(12)
    # a known smooth B0 ramp well inside one DFT period (|Δf| < 1/(2·TR))
    ramp = torch.linspace(-40.0, 40.0, 32).view(1, 1, 1, 32).expand(2, 1, 32, 32)
    res = sim(img, generator=g, external_field=ramp)
    b0_hat = invert_bssfp_banding(
        res.stack,
        tr_ms=sim.tr_bssfp_ms,
        t1_ms=sim.t1_ms,
        t2_ms=sim.t2_ms,
        flip_rad=sim.flip_rad,
    )
    assert b0_hat.shape == res.field.shape
    a = (b0_hat - b0_hat.mean()).flatten()
    b = (res.field - res.field.mean()).flatten()
    corr = (a * b).sum() / (a.norm() * b.norm() + 1e-8)
    assert float(corr) > 0.95
    # demeaned (constant-offset-free) error is a small fraction of the range
    resid = (b0_hat - b0_hat.mean()) - (res.field - res.field.mean())
    assert float(resid.std()) < 5.0  # Hz, << 80 Hz dynamic range


def test_lowrank_temporal_series_is_low_rank() -> None:
    sim = MultiAcquisitionSimulator("lowrank_temporal", n_frames=8)
    img, g = _phantom(6)
    res = sim(img, generator=g)
    assert res.stack.shape == (2, 8, 32, 32)  # noisy series (model input)
    assert res.field.shape == (2, 8, 32, 32)  # clean series (recon target)
    # the clean series must genuinely have a low-rank temporal subspace
    m = res.field[0].reshape(8, -1)
    sv = torch.linalg.svdvals(m)
    assert (sv[:3] ** 2).sum() / (sv**2).sum() > 0.97
    # the input is a noisy version of the target (denoising target ≠ input)
    assert not torch.allclose(res.stack, res.field)


def test_subvoxel_sr_frames_are_distinct_lowres_views() -> None:
    sim = MultiAcquisitionSimulator("subvoxel_sr", n_frames=8, sr_scale=2)
    img, g = _phantom(7)
    res = sim(img, generator=g)
    assert res.stack.shape == (2, 8, 16, 16)  # 8 low-res frames (H/scale)
    assert res.field.shape == (2, 1, 32, 32)  # high-res target
    # the sub-pixel-shifted frames differ from each other
    assert not torch.allclose(res.stack[:, 0], res.stack[:, 3])


def test_subvoxel_sr_dither_repeats_under_a_fixed_generator_seed() -> None:
    """Validation seeds its generator so one checkpoint scores the same twice."""
    sim = MultiAcquisitionSimulator("subvoxel_sr", n_frames=4, sr_scale=2)
    img, _ = _phantom(3)
    a = sim(img, generator=torch.Generator().manual_seed(0))
    b = sim(img, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a.shifts, b.shifts)
    assert torch.equal(a.stack, b.stack)


def test_subvoxel_sr_accepts_cpu_generator() -> None:
    """A CPU generator must drive subvoxel_sr without device errors (regression
    guard that the offset draws now pass an explicit device matching the
    generator). On CPU this exercises the exact ``torch.rand(..., device=...)``
    code path fixed for the CUDA-generator mismatch."""
    sim = MultiAcquisitionSimulator("subvoxel_sr", n_frames=4, sr_scale=2)
    img = 0.5 + torch.rand(2, 1, 32, 32)
    g = torch.Generator(device="cpu").manual_seed(0)
    res = sim(img, generator=g)  # device='cpu' (generator) must match draw device
    assert res.stack.shape == (2, 4, 16, 16)
    assert res.stack.device.type == "cpu"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_subvoxel_sr_accepts_cuda_generator() -> None:
    """A CUDA generator must not raise 'Expected a generator on device cpu but
    found cuda' in the subvoxel_sr offset draws (the device= fix)."""
    sim = MultiAcquisitionSimulator("subvoxel_sr", n_frames=4, sr_scale=2)
    g = torch.Generator(device="cuda").manual_seed(0)
    img = 0.5 + torch.rand(2, 1, 32, 32, device="cuda")
    res = sim(img, generator=g)
    assert res.stack.shape == (2, 4, 16, 16)
    assert res.stack.device.type == "cuda"


def test_mrf_round_trip_recovers_tissue_parameter() -> None:
    sim = MultiAcquisitionSimulator("mrf", n_timepoints=16)
    img, g = _phantom(5)
    res = sim(img, generator=g)
    assert res.stack.shape == (2, 16, 32, 32)  # [B, T, H, W] transient
    assert res.field.shape == (2, 2, 32, 32)  # (T1_norm, T2_norm)
    f_rec = invert_mrf(res.stack, dict_size=256)
    # dictionary matching recovers the (T1,T2)-coupled parameter (grid-quantised)
    assert ((f_rec - res.field[:, 0:1]).abs().mean()).item() < 0.03


def test_bloch_siegert_round_trip_recovers_b1() -> None:
    sim = MultiAcquisitionSimulator("bloch_siegert")
    img, g = _phantom(3)
    res = sim(img, generator=g)
    assert res.echoes is not None
    b1_hat = invert_bloch_siegert(res.stack, sim.k_bs)
    # phase-difference B1 is exact (tissue-free); allow numerical noise
    assert torch.allclose(b1_hat, res.field, atol=2e-2)


def test_afi_round_trip_recovers_b1() -> None:
    sim = MultiAcquisitionSimulator("afi")
    img, g = _phantom(2)
    res = sim(img, generator=g)
    b1_hat = invert_afi(res.stack, sim.flip_rad, sim.tr_ratio)
    rel = ((b1_hat - res.field).abs() / res.field.abs()).mean().item()
    # Yarnykh dual-TR steady state inverts to <1% in the TR<<T1 regime
    # (measured ~0.7% at T1_WM(0.3T)=412 ms, TR1/TR2=20/100 ms).
    assert rel < 0.03, f"AFI relative B1 error {rel:.3f} too high"


# ── real-field injection (multi-acq real-reference seam) ──────────────────────


def test_external_field_overrides_synthetic_b0_dual_echo():
    """external_field makes the simulator synthesise from / supervise against a
    REAL B0 instead of a random smooth field."""
    sim = MultiAcquisitionSimulator("dual_echo", field_strength_t=3.0)
    img = torch.rand(1, 1, 16, 16)
    real_b0 = torch.full((1, 1, 16, 16), 42.0)
    res = sim(img, external_field=real_b0)
    assert torch.allclose(res.field, real_b0, atol=1e-3)


def test_external_field_overrides_synthetic_b1_double_angle():
    sim = MultiAcquisitionSimulator("double_angle", field_strength_t=3.0)
    img = torch.rand(1, 1, 16, 16)
    real_b1 = torch.full((1, 1, 16, 16), 0.85)
    res = sim(img, external_field=real_b1)
    assert torch.allclose(res.field, real_b1, atol=1e-3)


def test_external_field_resized_and_5d_reduced():
    sim = MultiAcquisitionSimulator("dual_echo", field_strength_t=3.0)
    img = torch.rand(1, 1, 16, 16)
    real_b0 = torch.full((1, 1, 8, 8, 4), 30.0)  # 5-D, different res
    res = sim(img, external_field=real_b0)
    assert tuple(res.field.shape) == (1, 1, 16, 16)
    assert torch.allclose(res.field, torch.full((1, 1, 16, 16), 30.0), atol=1.0)


def test_without_external_field_is_random_default():
    sim = MultiAcquisitionSimulator("dual_echo", field_strength_t=3.0)
    res = sim(torch.rand(1, 1, 16, 16))
    assert res.field.std() > 0


# ---------------------------------------------------------------------------
# subvoxel_sr dither + virtual fiducial (2026-07-26)
# ---------------------------------------------------------------------------
def _hr_image(b: int = 3, h: int = 128, w: int = 128) -> torch.Tensor:
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(0)
    return F.interpolate(
        torch.rand(b, 1, 16, 16, generator=g), size=(h, w), mode="bicubic",
        align_corners=False,
    )


def _sim(**kw):
    from spectramr.infrastructure.physics.multi_acquisition import (
        MultiAcquisitionSimulator,
    )

    return MultiAcquisitionSimulator("subvoxel_sr", n_frames=8, sr_scale=2, **kw)


def test_subvoxel_returns_its_ground_truth_shifts() -> None:
    res = _sim()(_hr_image())
    assert res.shifts is not None and res.shifts.shape == (3, 8, 2)


def test_subvoxel_shifts_vary_per_sample_not_just_per_frame() -> None:
    """Before 2026-07-26 one scalar offset per frame was applied to the WHOLE
    batch, so a batch of B saw n dither patterns instead of n*B."""
    res = _sim()(_hr_image())
    assert (res.shifts[0] - res.shifts[1]).abs().max().item() > 1e-6


def test_subvoxel_shifts_respect_max_shift_px() -> None:
    res = _sim(subvoxel_max_shift_px=0.25)(_hr_image())
    assert res.shifts.abs().max().item() <= 0.25 + 1e-6


def test_subvoxel_frames_are_distinct_and_low_res() -> None:
    res = _sim()(_hr_image())
    assert res.stack.shape == (3, 8, 64, 64)
    assert (res.stack[:, 0] - res.stack[:, 1]).abs().max().item() > 1e-6


def test_marker_is_absent_unless_enabled() -> None:
    assert _sim()(_hr_image()).marker_stack is None


def test_marker_carries_the_same_offsets_as_the_anatomy() -> None:
    """The mechanism-fires probe at the physics layer: registering the fiducial
    against the un-shifted reference must reproduce the simulator's own shifts
    without ever reading them."""
    from spectramr.infrastructure.physics.subpixel_registration import (
        estimate_subpixel_shifts,
    )

    sim = _sim(marker_enabled=True, marker_jitter=0.35)
    hr = _hr_image()
    res = sim(hr)
    assert res.marker_stack.shape == res.stack.shape
    ref = sim.marker_reference((128, 128), hr.device, hr.dtype).expand(3, 1, 64, 64)
    est = estimate_subpixel_shifts(ref, res.marker_stack) * sim.sr_scale
    assert (est - res.shifts).abs().max().item() < 0.05


def test_marker_reference_is_stable_across_calls() -> None:
    """An ABSOLUTE registration anchor that drifted between calls would silently
    bias every recovered shift."""
    sim = _sim(marker_enabled=True)
    hr = _hr_image(b=1)
    a = sim.marker_reference((128, 128), hr.device, hr.dtype)
    b = sim.marker_reference((128, 128), hr.device, hr.dtype)
    assert torch.equal(a, b)


def test_marker_on_a_non_subvoxel_method_raises() -> None:
    """Advertising the fiducial where nothing registers it would be an inert
    knob (CLAUDE.md #15)."""
    from spectramr.infrastructure.physics.multi_acquisition import (
        MultiAcquisitionSimulator,
    )

    with pytest.raises(ValueError, match="only implemented for 'subvoxel_sr'"):
        MultiAcquisitionSimulator("afi", marker_enabled=True)


def test_dither_preserves_high_frequency_content() -> None:
    """The dither must be an exact translation. The previous bilinear resample
    sheds ~5% of total energy per frame, i.e. exactly the signal multi-frame SR
    is trying to recover."""
    hr = _hr_image(b=1)
    res = _sim()(hr)
    import torch.nn.functional as F

    plain = F.avg_pool2d(hr, 2)
    per_frame = res.stack.pow(2).sum(dim=(-2, -1))
    assert torch.allclose(
        per_frame, plain.pow(2).sum(dim=(-2, -1)).expand_as(per_frame), rtol=0.05
    )


def test_physical_marker_geometry_reaches_the_simulator() -> None:
    """PR-B mechanism-fires probe: declaring an effective voxel must actually
    change the marker the simulator builds, not just sit in the config.

    A marker sized to a 1.6 mm effective resolution on a 0.49 mm grid is ~3.3
    grid-pixels wide; the pixel-mode default is 2.0. If the physical path were
    ignored the two fields would be identical and every ULF arm would silently
    carry a marker 3x too fine to survive at 64mT.
    """
    from spectramr.infrastructure.physics.fft_ops import fft2c

    pixel_mode = _sim(marker_enabled=True, marker_sigma=2.0)
    physical = _sim(
        marker_enabled=True,
        marker_voxel_mm=(0.49, 0.49),
        marker_effective_voxel_mm=(1.6, 1.6),
    )
    hr = _hr_image(b=1)
    a = pixel_mode._fiducial_field((128, 128), hr.device, hr.dtype)
    b = physical._fiducial_field((128, 128), hr.device, hr.dtype)
    assert not torch.allclose(a, b), "physical geometry was ignored"

    # The physical marker is the WIDER one, so its spectrum is more compact.
    def bandwidth(field):
        k = fft2c(field).abs()[0, 0]
        n = k.shape[0]
        fy = (torch.arange(n) - n // 2).float().view(-1, 1)
        fx = (torch.arange(n) - n // 2).float().view(1, -1)
        r2 = (fy**2 + fx**2) / n**2
        return float((r2 * k**2).sum() / (k**2).sum())

    assert bandwidth(b) < bandwidth(a)


def test_pixel_mode_is_unchanged_by_the_physical_option() -> None:
    """exp_vf_01 must be bit-identical: it declares no effective voxel."""
    a = _sim(marker_enabled=True)._fiducial_field(
        (64, 64), torch.device("cpu"), torch.float32
    )
    b = _sim(marker_enabled=True, marker_voxel_mm=None)._fiducial_field(
        (64, 64), torch.device("cpu"), torch.float32
    )
    assert torch.equal(a, b)



def test_marker_hr_is_the_unpooled_reference_the_probe_reconstructs() -> None:
    """``marker_reference`` returns the POOLED anchor used for registration;
    the super-Nyquist probe needs the HR marker its frames are pooled views of.
    Confusing the two would grade the network against its own input."""
    sim = MultiAcquisitionSimulator(
        "subvoxel_sr", n_frames=4, sr_scale=2, marker_enabled=True, marker_jitter=0.35
    )
    hr_size = (64, 64)
    dev, dt = torch.device("cpu"), torch.float32
    from torch.nn.functional import avg_pool2d

    hr = sim.marker_hr(hr_size, dev, dt)
    pooled = sim.marker_reference(hr_size, dev, dt)
    assert hr.shape == (1, 1, 64, 64)
    assert pooled.shape == (1, 1, 32, 32)
    assert torch.allclose(avg_pool2d(hr, 2), pooled, atol=1e-6)


def test_field_transfer_makes_the_frames_a_low_field_view() -> None:
    """The synthesised frames are the LOW-field view, so the high-field target
    is divided by kappa. With no transfer installed the branch is exactly inert,
    which is what keeps every pre-2026-07-26 arm unchanged."""
    sim = MultiAcquisitionSimulator(
        "subvoxel_sr", n_frames=4, sr_scale=2, subvoxel_max_shift_px=1e-6,
        marker_enabled=True, marker_jitter=0.35,
    )
    pd = torch.rand(2, 1, 32, 32) + 0.1
    torch.manual_seed(0)
    plain = sim(pd).stack

    sim.set_field_transfer(
        {"white_matter": 0.5, "gray_matter": 0.5, "csf": 0.5}, marker_gain=0.25
    )
    torch.manual_seed(0)
    scaled = sim(pd).stack
    # a uniform gain of 0.5 doubles the low-field view
    assert torch.allclose(scaled, plain * 2.0, atol=1e-4)

    sim.set_field_transfer(None, None)
    torch.manual_seed(0)
    assert torch.allclose(sim(pd).stack, plain, atol=1e-5)


def test_tissue_gain_field_is_a_three_class_intensity_proxy() -> None:
    """Not a segmentation. What it must deliver is three DISTINCT gains, so the
    transfer is a contrast change rather than a global scale."""
    sim = MultiAcquisitionSimulator("subvoxel_sr", n_frames=2, sr_scale=2)
    sim.set_field_transfer(
        {"white_matter": 0.40, "gray_matter": 0.45, "csf": 0.86}, marker_gain=0.67
    )
    pd = torch.linspace(0.0, 1.0, 32 * 32).reshape(1, 1, 32, 32)
    field = sim._tissue_gain_field(pd)
    assert sorted(float(v) for v in field.unique()) == pytest.approx([0.40, 0.45, 0.86], abs=1e-6)


def test_tissue_gain_table_is_built_once_and_reset_on_reinstall() -> None:
    """The per-step transfer reuses one device table; a new transfer replaces it."""
    sim = MultiAcquisitionSimulator("subvoxel_sr", n_frames=2, sr_scale=2)
    sim.set_field_transfer({"white_matter": 0.4, "gray_matter": 0.45, "csf": 0.86}, None)
    pd = torch.linspace(0.0, 1.0, 32 * 32).reshape(1, 1, 32, 32)
    sim._tissue_gain_field(pd)
    table = sim._gain_tables[(pd.device, pd.dtype)]
    sim._tissue_gain_field(pd)
    assert sim._gain_tables[(pd.device, pd.dtype)] is table
    sim.set_field_transfer({"white_matter": 1.0, "gray_matter": 2.0, "csf": 3.0}, None)
    assert sorted(float(v) for v in sim._tissue_gain_field(pd).unique()) == [1.0, 2.0, 3.0]


def test_marker_gain_is_declared_not_proxied() -> None:
    """The fiducial's kappa is exact while the anatomy's is approximate — that
    asymmetry is the whole point of carrying a calibrated instrument."""
    sim = MultiAcquisitionSimulator(
        "subvoxel_sr", n_frames=2, sr_scale=2, subvoxel_max_shift_px=1e-6,
        marker_enabled=True, marker_jitter=0.35,
    )
    pd = torch.rand(1, 1, 32, 32) + 0.1
    torch.manual_seed(0)
    plain = sim(pd).marker_stack
    sim.set_field_transfer(
        {"white_matter": 0.4, "gray_matter": 0.4, "csf": 0.4}, marker_gain=0.5
    )
    # the marker follows ITS declared gain (0.5), not the anatomy's (0.4)
    torch.manual_seed(0)
    assert torch.allclose(sim(pd).marker_stack, plain * 2.0, atol=1e-4)
