"""Tests for the transversal DigitalTwinDegradation data transform.

Mirrors the accelerator (PhysicsInformedMasking) but applies the digital-twin
corruption in the data pipeline so any experiment can opt in. Pins: it
produces a corrupted ``input`` and a preserved clean ``target``, shape is
preserved, and the round-trip is k-space→image→k-space.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
tio = pytest.importorskip("torchio")

from spectramr.config.schemas.physics import DigitalTwinConfig  # noqa: E402
from spectramr.data.transforms.tio_physics import DigitalTwinDegradation  # noqa: E402
from spectramr.infrastructure.physics.digital_twin_simulator import (  # noqa: E402
    DigitalTwinSimulator,
)


def _subject(c: int = 2, w: int = 32, h: int = 32, d: int = 1) -> tio.Subject:
    torch.manual_seed(0)
    return tio.Subject(kspace=tio.ScalarImage(tensor=torch.randn(c, w, h, d)))


def _sim(**kw) -> DigitalTwinSimulator:
    cfg = DigitalTwinConfig(enable_b0=False, enable_b1=False, **kw)
    return DigitalTwinSimulator.from_config(cfg, (32, 32))


def test_produces_corrupted_input_and_clean_target() -> None:
    subj = _subject()
    raw = subj["kspace"].data.clone()
    out = DigitalTwinDegradation(
        _sim(enable_motion=True, motion_severity=3.0), degradation_only=True, pe_axis="W"
    )(subj)

    assert {"input", "target"} <= set(out.keys())
    inp, tgt = out["input"].data, out["target"].data
    assert inp.shape == raw.shape  # shape preserved
    assert not torch.allclose(inp, tgt, atol=1e-4)  # corruption applied
    assert torch.allclose(tgt, raw, atol=1e-6)  # clean target preserved


def test_higher_severity_corrupts_more() -> None:
    """``enable_motion`` must be ON, or this asserts nothing about severity.

    Its default flipped to False on 2026-08-03 and these two tests kept passing
    ``motion_severity`` into a disabled branch. Measured: with motion off, severity
    0.0 and 5.0 produce BYTE-IDENTICAL output, so the only difference the assertion
    could see was the ambient noise draw between the two calls — a coin flip, and
    the reason this test was red on dev. With motion on it separates 6/6 with a wide
    margin (0.95 vs 0.15 mean absolute change).
    """
    raw = _subject()["kspace"].data
    mild = DigitalTwinDegradation(
        _sim(enable_motion=True, motion_severity=0.0), degradation_only=True, pe_axis="W"
    )(tio.Subject(kspace=tio.ScalarImage(tensor=raw.clone())))["input"].data
    severe = DigitalTwinDegradation(
        _sim(enable_motion=True, motion_severity=5.0), degradation_only=True, pe_axis="W"
    )(tio.Subject(kspace=tio.ScalarImage(tensor=raw.clone())))["input"].data
    assert (severe - raw).abs().mean() > (mild - raw).abs().mean()


def test_odd_channels_rejected() -> None:
    subj = tio.Subject(kspace=tio.ScalarImage(tensor=torch.randn(3, 16, 16, 1)))
    with pytest.raises(ValueError, match="even channels"):
        DigitalTwinDegradation(_sim(), degradation_only=True, pe_axis="W")(subj)


def test_missing_acquisition_raises_rather_than_passing_clean_data_through() -> None:
    """Silence here trained an arm on clean data while the log said otherwise.

    One step earlier the transform builder logs "[PHYSICS] Injecting transversal
    DigitalTwinDegradation" at INFO. This branch then returned the subject
    untouched at DEBUG, so at default verbosity the run REPORTED a degradation it
    never applied — pitfall #16 with a receipt. The message has to name what is
    missing and both ways out, because the arm author's mistake is a dataset_type
    that serves images, not a bug in the transform.
    """
    subj = tio.Subject(image=tio.ScalarImage(tensor=torch.randn(2, 16, 16, 1)))
    with pytest.raises(ValueError, match="no acquisition to corrupt"):
        DigitalTwinDegradation(_sim(), degradation_only=True, pe_axis="W")(subj)


def test_the_removed_source_domain_knob_cannot_come_back_silently() -> None:
    """``source_domain`` was accepted-but-broken and is gone.

    Its ``"image"`` value resolved its source from the k-space key anyway, so it
    meant "read k-space and pretend it is an image", and no caller ever passed it:
    the transform builder hardcoded the k-space route. What matters now is that
    passing it FAILS rather than being swallowed by ``**kwargs`` into an attribute
    nobody reads — which is how an unwired knob is born (pitfall #15).
    """
    with pytest.raises(TypeError, match="source_domain"):
        DigitalTwinDegradation(_sim(), pe_axis="W", source_domain="image")


def test_the_phase_encode_axis_is_declared_not_inferred() -> None:
    with pytest.raises(TypeError, match="pe_axis"):
        DigitalTwinDegradation(_sim(), degradation_only=True)
    with pytest.raises(ValueError, match="pe_axis"):
        DigitalTwinDegradation(_sim(), pe_axis="D")


def test_the_phase_encode_axis_is_stamped() -> None:
    assert DigitalTwinDegradation(_sim(), pe_axis="W").applied_record()["pe_axis"] == "W"


def _twin(im_size, acquisition=None, **kw) -> DigitalTwinSimulator:
    cfg = DigitalTwinConfig(
        enable_motion=False, enable_b0=False, acquisition=acquisition, **kw
    )
    return DigitalTwinSimulator.from_config(cfg, im_size, field_strength_t=3.0)


def _complex_canonical(raw: torch.Tensor) -> torch.Tensor:
    from spectramr.data.transforms.layout import tio_to_canonical
    from spectramr.infrastructure.physics.fft_ops import ifft2c

    c, w, h, d = raw.shape
    k = torch.view_as_complex(raw.view(c // 2, 2, w, h, d).permute(0, 2, 3, 4, 1).contiguous())
    return ifft2c(tio_to_canonical(k, "W"))


def test_without_a_volume_mode_each_slice_is_an_independent_2d_slice(monkeypatch) -> None:
    """W=6 (PE), H=8 (RO), D=5: each depth slice reaches the twin as one (PE, RO) slice."""
    raw = torch.randn(2, 6, 8, 5, generator=torch.Generator().manual_seed(2))
    sim = _twin((6, 8))
    calls = []
    forward = sim.forward

    def spy(x, **kw):
        out = forward(x, **kw)
        calls.append((x, out[0]))
        return out

    monkeypatch.setattr(sim, "forward", spy)
    out = DigitalTwinDegradation(sim, pe_axis="W")(
        tio.Subject(kspace=tio.ScalarImage(tensor=raw.clone()))
    )["input"].data
    assert out.shape == raw.shape
    (seen, corrupted), = calls
    assert torch.allclose(seen, _complex_canonical(raw)[0].permute(1, 0, 2, 3), atol=1e-6)
    assert torch.allclose(
        _complex_canonical(out)[0].permute(1, 0, 2, 3), corrupted, atol=1e-5
    )


def test_a_multislice_profile_hands_the_twin_the_canonical_volume(monkeypatch) -> None:
    raw = torch.randn(2, 6, 8, 5, generator=torch.Generator().manual_seed(2))
    block = {"readout_bw_hz_per_px": 200.0, "sequence": "spin_echo", "mode": "multislice_2d"}
    sim = _twin((6, 8), acquisition=block, progressive_degradations=["bias"])
    seen = []
    forward = sim.forward
    monkeypatch.setattr(
        sim, "forward", lambda x, **kw: seen.append(tuple(x.shape)) or forward(x, **kw)
    )
    out = DigitalTwinDegradation(sim, pe_axis="W")(
        tio.Subject(kspace=tio.ScalarImage(tensor=raw.clone()))
    )["input"].data
    assert seen == [(1, 1, 5, 6, 8)]
    assert out.shape == raw.shape
    assert not torch.allclose(out, raw, atol=1e-3)


def test_physics_masking_undersamples_width_constant_along_height() -> None:
    """Pins the documented axis convention (review 2026-07-01).

    ``PhysicsInformedMasking._generate_mask`` undersamples along WIDTH (this
    transform's phase-encode axis) and holds the pattern constant along HEIGHT
    (the fully-sampled/readout axis). Guards against a silent axis flip and
    documents the assumption the physics-SSOT consolidation must reconcile.
    """
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    torch.manual_seed(0)
    m = PhysicsInformedMasking(acceleration=4, center_fraction=0.25)
    width, height = 40, 24
    mask = m._generate_mask(width, height)

    assert mask.shape == (width, height)
    # Constant along height (dim=1): every column equals the first.
    assert torch.all(mask == mask[:, :1])
    # The center fraction is fully acquired along width (the ACS band).
    center = width // 2
    cf_w = int(width * 0.25)
    assert torch.all(mask[center - cf_w // 2 : center + cf_w // 2, 0] == 1)
    # Acceleration undersamples: not every width line is acquired.
    assert mask[:, 0].sum() < width


def test_physics_informed_mask_is_deterministic_and_cached() -> None:
    """Behavior + perf regression (2026-07-02): the mask used ``torch.randperm``
    UNSEEDED → a different mask per item, unlike every seeded generator in the
    physics SSOT. It is now seeded per width (reproducible) and cached per
    ``(width, height)`` so it is not recomputed in every ``apply_transform``."""
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    m = PhysicsInformedMasking(acceleration=4, center_fraction=0.1)
    a = m._generate_mask(48, 16)
    # Cache hit returns the same object (no recompute).
    b = m._generate_mask(48, 16)
    assert b is a
    assert (48, 16) in m._mask_cache

    # A fresh instance produces an identical mask (seeded, not global-RNG).
    m2 = PhysicsInformedMasking(acceleration=4, center_fraction=0.1)
    assert torch.equal(m2._generate_mask(48, 16), a)


def test_physics_informed_mask_seed_untouched_by_global_rng() -> None:
    """The seeded per-width generator must not depend on global torch RNG, so
    two instances agree regardless of intervening global draws."""
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    torch.manual_seed(1)
    a = PhysicsInformedMasking(acceleration=6)._generate_mask(64, 8)
    torch.manual_seed(999)
    _ = torch.rand(100)  # perturb global RNG
    b = PhysicsInformedMasking(acceleration=6)._generate_mask(64, 8)
    assert torch.equal(a, b)


# --- PhysicsInformedMasking: the measurement travels under its own key (#2324) ---


def _kspace_subject(c: int = 4, w: int = 32, h: int = 32, d: int = 1) -> tio.Subject:
    torch.manual_seed(1)
    return tio.Subject(kspace=tio.ScalarImage(tensor=torch.randn(c, w, h, d)))


def test_masking_writes_the_measurement_beside_the_input() -> None:
    from spectramr.data.transforms.tio_physics import MEASURED_KSPACE_KEY, PhysicsInformedMasking

    subject = _kspace_subject()
    full = subject["kspace"].data.clone()
    out = PhysicsInformedMasking(acceleration=4, center_fraction=0.08)(subject)

    measured = out[MEASURED_KSPACE_KEY].data
    assert torch.equal(measured, out["input"].data)
    assert (measured * (1 - out["mask"].data)).abs().max() == 0
    assert not torch.equal(measured, full), "the measurement is the fully sampled k-space"
    # `kspace` stays the reference; that is exactly why a resolver must not read it.
    assert torch.equal(out["kspace"].data, full)


def test_the_measurement_is_its_own_tensor() -> None:
    """A later in-place edit of `input` (normalisation, a registry transform) must not
    silently rewrite what data consistency is anchored to."""
    from spectramr.data.transforms.tio_physics import MEASURED_KSPACE_KEY, PhysicsInformedMasking

    out = PhysicsInformedMasking(acceleration=4, center_fraction=0.08)(_kspace_subject())
    out["input"].data.mul_(0)
    assert out[MEASURED_KSPACE_KEY].data.abs().sum() > 0


def test_mask_direction_x_undersamples_the_last_axis_constant_along_the_first() -> None:
    """``undersampling.mask_direction: x`` runs the line pattern along the last
    spatial axis, M4Raw's phase-encode axis (#2327); unset keeps the first."""
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    width, height = 40, 24
    mask = PhysicsInformedMasking(acceleration=4, center_fraction=0.25, mask_direction="x")
    m = mask._generate_mask(width, height)
    assert m.shape == (width, height)
    assert torch.all(m == m[:1, :])  # constant along the first axis
    assert 0 < m[0].sum() < height  # undersampled along the last
    center = height // 2
    assert torch.all(m[0, center - 3 : center + 3] == 1)
    subject = tio.Subject(kspace=tio.ScalarImage(tensor=torch.randn(2, width, height, 1)))
    out = mask(subject)["input"].data
    zero_cols = (out.abs().sum(dim=(0, 1, 3)) == 0).sum()
    zero_rows = (out.abs().sum(dim=(0, 2, 3)) == 0).sum()
    assert zero_cols > 0 and zero_rows == 0


def test_an_unknown_mask_direction_is_refused() -> None:
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    with pytest.raises(ValueError, match="mask_direction"):
        PhysicsInformedMasking(acceleration=4, mask_direction="diagonal")


@pytest.mark.parametrize(("axis", "expected"), [(-1, "x"), (-2, "y")])
def test_phase_resolves_through_the_declared_phase_encode_axis(axis: int, expected: str) -> None:
    """``phase`` follows ``data.acquisition.phase_encode_axis``; it was pinned to ``y`` (#2160)."""
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    mask = PhysicsInformedMasking(acceleration=4, mask_direction="phase", phase_encode_axis=axis)
    assert mask.line_axis == expected
    m = mask._generate_mask(40, 64)
    lines = m[:, :1] if expected == "y" else m[:1, :]
    assert torch.equal(m, lines.expand_as(m))  # constant across the other axis


def test_phase_without_a_declared_axis_is_refused() -> None:
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    with pytest.raises(ValueError, match="phase_encode_axis"):
        PhysicsInformedMasking(acceleration=4, mask_direction="phase")


@pytest.mark.parametrize(("acceleration", "masks"), [(1, False), (1.0, False), (2.0, True), (4, True)])
def test_the_loader_masks_lines_only_above_base_one(acceleration, masks) -> None:
    """#1082: the audit and the transform builder share this one predicate."""
    from spectramr.data.transforms.tio_physics import loader_masks_lines

    assert loader_masks_lines(acceleration) is masks


# --- PhysicsInformedMasking: declared R applies to the outer lines (#2382) ---


def _kspace_subject(width: int = 256, height: int = 8) -> tio.Subject:
    return tio.Subject(kspace=tio.ScalarImage(tensor=torch.randn(2, width, height, 1)))


def test_r4_at_256_lines_realises_3_241_and_the_subject_records_it() -> None:
    """The planted case from #2382: 20 ACS + 59 outer lines = 79, so R 3.241.

    Fails on the old transform, which stamped nothing on the subject.
    """
    from spectramr.data.transforms.tio_physics import (
        REALISED_ACCELERATION_KEY,
        PhysicsInformedMasking,
    )

    masking = PhysicsInformedMasking(acceleration=4, center_fraction=0.08)
    out = masking(_kspace_subject())

    assert int(out["mask"].data[0, :, 0, 0].sum()) == 79
    recorded = float(out[REALISED_ACCELERATION_KEY])
    assert recorded == pytest.approx(256 / 79)
    assert round(recorded, 3) == 3.241
    assert masking.realised_acceleration(256, 8) == pytest.approx(recorded)


def test_the_realised_factor_survives_loader_collation() -> None:
    """The value reaches the training step only if collation keeps it."""
    from spectramr.data.transforms.tio_physics import (
        REALISED_ACCELERATION_KEY,
        PhysicsInformedMasking,
    )

    dataset = tio.SubjectsDataset(
        [_kspace_subject(), _kspace_subject()],
        transform=PhysicsInformedMasking(acceleration=4, center_fraction=0.08),
    )
    batch = next(iter(tio.SubjectsLoader(dataset, batch_size=2)))
    assert batch[REALISED_ACCELERATION_KEY].tolist() == pytest.approx([256 / 79] * 2)

    # The queue builder's loaders collate through the repo's own strategy.
    from spectramr.data.collation.strategies import CollateStrategyFactory

    collated = CollateStrategyFactory.create("image").collate([dataset[0], dataset[1]])
    assert collated[REALISED_ACCELERATION_KEY].tolist() == pytest.approx([256 / 79] * 2)


def test_args_names_are_the_constructor_kwargs() -> None:
    """TorchIO replays history by passing ``args_names`` back to the constructor."""
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    masking = PhysicsInformedMasking(acceleration=8, center_fraction=0.04)
    replay = PhysicsInformedMasking(**masking._get_named_arguments())
    assert torch.equal(replay._generate_mask(64, 4), masking._generate_mask(64, 4))
    assert masking.applied_record()["acceleration_applies_to"] == "non_acs_lines"


def test_physics_masking_acs_band_is_not_the_acs_band_rows_band() -> None:
    """Characterization (#2692): the ACS here truncates its width and loses a line when it is odd.

    ``acs_band_rows`` rounds and centres on ``N // 2``. The two coincide only for some N and
    center fractions; the mask is kept as published, so the difference is pinned rather than
    fixed. Combine this mask with the ``acs`` scale domain only after aligning them.
    """
    from spectramr.data.transforms.normalization import acs_band_rows
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    n, cf = 100, 0.15  # width 15, odd
    m = PhysicsInformedMasking(acceleration=1_000_000, center_fraction=cf)
    mask = m._generate_mask(n, 4)[:, 0]
    mask_band = torch.nonzero(mask).flatten()
    band = acs_band_rows(n, cf)

    assert len(mask_band) == int(n * cf) - 1  # 14 lines: the odd width loses one
    assert band.stop - band.start == round(n * cf)  # 15 lines
    assert (int(mask_band[0]), int(mask_band[-1]) + 1) != (band.start, band.stop)


@pytest.mark.parametrize(
    ("direction", "pe_axis", "line_axis", "plane_dim", "lines_along"),
    [
        ("phase", -1, "x", "-1", "phase_encode"),
        ("phase", -2, "y", "-2", "phase_encode"),
        ("readout", -1, "y", "-2", "readout"),
        ("readout", -2, "x", "-1", "readout"),
    ],
)
def test_physics_masking_states_where_its_lines_run(
    direction, pe_axis, line_axis, plane_dim, lines_along
) -> None:
    """The applied record names the axis the lines vary along and whether it is phase-encode."""
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    record = PhysicsInformedMasking(
        mask_direction=direction, phase_encode_axis=pe_axis
    ).applied_record()
    assert (record["line_axis"], record["line_axis_plane_dim"], record["lines_along"]) == (
        line_axis,
        plane_dim,
        lines_along,
    )


def test_physics_masking_with_no_direction_says_it_cannot_tell() -> None:
    """Planted: unset ``mask_direction`` keeps the first spatial axis and must not claim it is PE."""
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    record = PhysicsInformedMasking().applied_record()
    assert record["line_axis"] == "y" and record["line_axis_plane_dim"] == "-2"
    assert record["lines_along"].startswith("undeclared")


def test_physics_masking_unset_direction_on_a_declared_axis_reports_the_mismatch() -> None:
    """PE on dim -1 (M4Raw) with the default first-axis lines: the lines run along the readout."""
    from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

    record = PhysicsInformedMasking(phase_encode_axis=-1).applied_record()
    assert record["lines_along"] == "readout"


def test_a_volume_3d_profile_is_refused_at_the_boundary() -> None:
    block = {"readout_bw_hz_per_px": 200.0, "sequence": "spin_echo", "mode": "volume_3d"}
    with pytest.raises(ValueError, match="multislice"):
        DigitalTwinDegradation(_twin((6, 8), acquisition=block), pe_axis="W")


def test_the_phase_encode_axis_source_is_stamped() -> None:
    twin = DigitalTwinDegradation(
        _sim(), pe_axis="H", pe_axis_source="data.acquisition.phase_encode_axis"
    )
    record = twin.applied_record()
    assert (record["pe_axis"], record["pe_axis_source"]) == (
        "H",
        "data.acquisition.phase_encode_axis",
    )


def test_the_profile_survives_the_provenance_cap() -> None:
    from spectramr.infrastructure.training.snapshot_provenance import _scalarize

    block = {"scanner_profile": "fastmri_3T"}
    sim = DigitalTwinSimulator.from_config(
        DigitalTwinConfig(enable_b0=False, enable_b1=False, acquisition=block),
        (32, 32),
        field_strength_t=3.0,
    )
    record = DigitalTwinDegradation(sim, pe_axis="W").applied_record()
    assert _scalarize(record) == record
    assert record["scanner_profile_sha256"] == sim.scanner_profile.meta_sha256()
