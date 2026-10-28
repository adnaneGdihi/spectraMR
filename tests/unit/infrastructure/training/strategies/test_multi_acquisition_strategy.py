"""Tests for the faithful multi-acquisition field-mapping strategy.

Exercises the synth -> model -> field-loss path on a synthetic k-space target
with the real arm models, using the lightweight ``__new__`` + stub-env pattern
(same convention as ``test_vf_admm_strategy_audit_2026_06.py``) to avoid the
full DI / dataloader machinery.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spectramr.config.schemas.physics import (  # noqa: E402
    BandProbeConfig,
    MultiAcquisitionConfig,
    RelaxometricCalibrationConfig,
    SubvoxelRegistrationConfig,
)

# The real schema object every stub inherits its defaults from. Adding fields to
# `MultiAcquisitionConfig` broke this file twice (45 tests in #512, 3 more in
# #520) while it hand-listed them; a stub cannot drift from what it is built
# from. See issue #501 section 1.
_MACQ_DEFAULTS = MultiAcquisitionConfig(enabled=True, method="afi")
from spectramr.infrastructure.training.strategies.multi_acquisition_strategy import (
    ConcreteMultiAcquisitionStrategy,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
from spectramr.models.registry import get_model_class

# (method, arm model, in channels, out channels, target channels, extra kwargs)
CASES = [
    ("afi", "afi_ratio_cnn", 2, 2, 1, {}),
    ("double_angle", "dam_trigfit", 2, 2, 1, {}),
    ("dual_echo", "resnet_unwrap", 2, 2, 1, {}),
    ("bloch_siegert", "bloch_siegert_algebraic", 2, 2, 1, {}),
    ("mrf", "mrf_dict_matcher", 16, 16, 2, {"dict_size": 64}),
    ("lowrank_temporal", "temporal_coherence_4d", 8, 8, 8, {}),
    # 8 frames + 2*8 shift-conditioning maps; restormer since 2026-07-26
    ("subvoxel_sr", "restormer", 24, 1, 1, {"scale": 2, "dim": 24}),
]


def _cfg(
    method: str,
    lambda_smooth: float = 0.01,
    normalize_magnitude: bool = True,
    shift_source: str = "blind",
    in_channels: int = 24,
    marker_jitter: float = 0.35,
    band_probe: bool = False,
    lambda_band: float = 0.0,
    marker_channels: bool = False,
    relax: RelaxometricCalibrationConfig | None = None,
    model_type: str = "restormer",
) -> SimpleNamespace:
    """Stub config for the __new__ + stub-env pattern.

    The two nested physics blocks are the REAL pydantic models, not namespaces.
    Hand-written stubs of them drifted from the schema twice: adding four
    physical-units fields in #512 turned 45 tests red with ``AttributeError``,
    the failure class #501 section 1 documents. A stub cannot drift from the
    source it is constructed from. The outer block stays a namespace so
    ``_cfg("triple_angle")`` can still build an arm the Literal rejects, which
    is what the simulator's own unknown-method guard is tested against.
    """
    macq = SimpleNamespace(
        # Every field of the real schema, at its real default, so a new schema
        # field cannot silently be missing here. Only the ones a test varies are
        # overridden below. `method` stays free-form so _cfg("triple_angle") can
        # still build an arm the Literal rejects, which is what the simulator's
        # own unknown-method guard is tested against.
        **{f: getattr(_MACQ_DEFAULTS, f) for f in type(_MACQ_DEFAULTS).model_fields}
    )
    macq.method = method
    macq.n_frames = 8
    macq.sr_scale = 2
    macq.lambda_smooth = lambda_smooth
    macq.normalize_magnitude = normalize_magnitude
    macq.marker_channels = marker_channels
    macq.subvoxel_registration = SubvoxelRegistrationConfig(
        shift_source=shift_source,
        marker_grid_spacing=16,
        marker_sigma=2.0,
        marker_jitter=marker_jitter,
        marker_seed=0,
        max_shift_px=1.0,
    )
    macq.band_probe = BandProbeConfig(enabled=band_probe, lambda_band=lambda_band)
    macq.relaxometric_calibration = relax or RelaxometricCalibrationConfig()
    return SimpleNamespace(
        physics=SimpleNamespace(
            multi_acquisition=macq,
            field_strength=0.3,
        ),
        data=SimpleNamespace(dataset_type="kspace"),
        # model_type is read by base.assert_input_contract; without it the
        # 7 train_step cases die on AttributeError before reaching the
        # behaviour they assert (#501).
        model=SimpleNamespace(in_channels=in_channels, model_type=model_type),
    )


def _strategy(
    method: str,
    model_name: str,
    in_ch: int,
    out_ch: int,
    mk: dict,
    lambda_smooth: float = 0.01,
    normalize_magnitude: bool = True,
    shift_source: str = "blind",
    band_probe: bool = False,
    lambda_band: float = 0.0,
    marker_channels: bool = False,
    relax: RelaxometricCalibrationConfig | None = None,
) -> ConcreteMultiAcquisitionStrategy:
    st = ConcreteMultiAcquisitionStrategy.__new__(ConcreteMultiAcquisitionStrategy)
    st.config = _cfg(
        method,
        lambda_smooth=lambda_smooth,
        normalize_magnitude=normalize_magnitude,
        shift_source=shift_source,
        in_channels=in_ch,
        band_probe=band_probe,
        lambda_band=lambda_band,
        marker_channels=marker_channels,
        relax=relax,
    )
    st.config.model.out_channels = out_ch
    st.device = torch.device("cpu")
    st._to_device = lambda t: t  # type: ignore[method-assign]
    st._setup_strategy_specific_components()
    st.env = SimpleNamespace(
        generator=get_model_class(model_name)(
            in_channels=in_ch, out_channels=out_ch, **mk
        )
    )
    return st


def _disk_kspace(
    b: int = 2, h: int = 32, w: int = 32, amplitude: float = 1.0
) -> torch.Tensor:
    """k-space of a bright centred disk on a zero background.

    ``dataset_type='kspace'`` → ``_clean_magnitude`` IFFTs the target back, so
    feeding ``fft2c(disk)`` reconstructs a clean object/air split the display
    mask can key off (random k-space has no object support to test against).

    ``amplitude`` sets the image-domain intensity, standing in for the arbitrary
    scanner/normalisation scale a real loaded target carries.
    """
    from spectramr.infrastructure.physics.fft_ops import fft2c

    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    r = ((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2).sqrt()
    disk = (r < h * 0.3).to(torch.complex64) * amplitude
    img = disk.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w).contiguous()
    return fft2c(img)


def test_factory_registers_multi_acquisition() -> None:
    assert "multi_acquisition" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS


@pytest.mark.parametrize("method, model_name, in_ch, out_ch, tch, mk", CASES)
def test_field_loss_path_runs_and_is_finite(
    method, model_name, in_ch, out_ch, tch, mk
) -> None:
    st = _strategy(method, model_name, in_ch, out_ch, mk)
    target_k = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    out = st._compute_losses_impl(target_k, target_k, epoch=0)
    assert "g_total_loss" in out
    assert torch.isfinite(out["g_total_loss"])
    # supervision target is the synthesised target (tch channels), not raw image
    assert st._last_visual_target.shape[1] == tch
    assert st._last_visual_pred.shape[1] == tch


@pytest.mark.parametrize("method, model_name, in_ch, out_ch, tch, mk", CASES)
def test_validation_emits_field_metrics(
    method, model_name, in_ch, out_ch, tch, mk
) -> None:
    st = _strategy(method, model_name, in_ch, out_ch, mk)
    target_k = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    val = st.validation_step(target_k, target_k)
    for key in (
        "val_field_mse",
        "val_field_bias",
        "val_field_abs_bias",
        "val_field_mae",
    ):
        assert key in val and isinstance(val[key], float)
    # The selectable form is non-negative by construction; the signed one is not.
    assert val["val_field_abs_bias"] >= 0.0
    assert val["val_field_abs_bias"] == pytest.approx(abs(val["val_field_bias"]))
    # qMRI agreement battery on the (pred_field, field) pair — resolves the
    # deferred backlog item for the multi-acquisition arms (21-24, 28).
    for key in (
        "val_icc_3_1",
        "val_bland_altman_bias",
        "val_coefficient_of_variation",
        "val_limits_of_agreement_upper",
        "val_limits_of_agreement_lower",
    ):
        assert key in val and isinstance(val[key], float)


def test_b1_field_target_in_physical_range() -> None:
    # double-angle synthesises a B1 transmit scale ~[0.7, 1.3]; the strategy
    # must be supervising against that field, not an image. The visual target is
    # now object-masked (air → 0) for honest display, so the physical-range check
    # is on the object (non-zero) support, not the whole frame.
    st = _strategy("double_angle", "dam_trigfit", 2, 2, {})
    target_k = _disk_kspace()
    st._compute_losses_impl(target_k, target_k, epoch=0)
    field = st._last_visual_target
    obj = field[field.abs() > 0]
    assert float(obj.min()) > 0.6 and float(obj.max()) < 1.4


def test_field_visuals_masked_to_object_support() -> None:
    """The saved validation real/fake panels are B1+ FIELD maps, not images.

    Regression for the 2026-06-21 vf_22-24 report ("target more degraded than
    output"): an external/real B1 map is the ratio of two noise floors *outside*
    the object, so displaying it unmasked through the per-sample image-windowing
    in ``MetricsTracker._normalize_images`` lifts the air to mid-grey noise and
    makes the *reference* look worse than the model's smoother estimate. The fix
    masks the VISUAL copies (``_last_visual_pred`` / ``_last_visual_target``) to
    the anatomical object support so both panels render on the same black-air
    background. It is display-only — the field used for the loss/metrics is
    untouched.
    """
    st = _strategy("afi", "afi_ratio_cnn", 2, 2, {})
    target_k = _disk_kspace()
    st._compute_losses_impl(target_k, target_k, epoch=0)
    vis_t = st._last_visual_target
    vis_p = st._last_visual_pred
    # air (corner, outside the disk) is suppressed to zero in BOTH panels
    assert float(vis_t[..., :4, :4].abs().max()) == 0.0
    assert float(vis_p[..., :4, :4].abs().max()) == 0.0
    # the object region (disk centre) retains the field signal
    assert float(vis_t[..., 14:18, 14:18].abs().max()) > 0.0


def _grad_norm(st: ConcreteMultiAcquisitionStrategy) -> float:
    return float(
        torch.sqrt(
            sum(
                (p.grad**2).sum()
                for p in st.env.generator.parameters()
                if p.grad is not None
            )
        )
    )


def test_field_loss_is_invariant_to_the_scanner_intensity_scale() -> None:
    """The objective must not inherit the arbitrary scale of the k-space data.

    Regression for the 2026-07 exp_vf_01 run, which logged on its FIRST step::

        GRADIENT EXPLOSION DETECTED in generator: total_norm=6476.9888
          — clipping active (1)

    ``data.normalization_kwargs.clamp: false`` is correct for k-space (clamping
    at the 99th percentile would clip the DC peak), so the loaded target keeps a
    scanner-dependent scale — the run's image-domain magnitude spanned
    ``[0, 92.7]``. Both the supervision target AND the synthesised frame stack
    derive from it, so the loss grows QUADRATICALLY with that scale: on this
    fixture the gradient norm went 2.7 -> 20124 for a 92x brighter object. With
    ``gradient_clip_value: 1.0`` the whole early trajectory is clip-dominated and
    the configured learning rate is not the effective one.

    Normalising the magnitude once, at the ``_clean_magnitude`` seam, fixes the
    target and the model input together — so the gradient norm must now be
    essentially the same at both scales.
    """
    # seed BEFORE constructing, so both arms get identical weights AND identical
    # sub-pixel shift draws — the intensity scale is then the only difference
    torch.manual_seed(0)
    lo = _strategy("subvoxel_sr", "restormer", 24, 1, {"scale": 2, "dim": 24})
    lo._field_loss(_disk_kspace(amplitude=1.0))[0].backward()

    torch.manual_seed(0)
    hi = _strategy("subvoxel_sr", "restormer", 24, 1, {"scale": 2, "dim": 24})
    hi._field_loss(_disk_kspace(amplitude=92.0))[0].backward()

    g_lo, g_hi = _grad_norm(lo), _grad_norm(hi)
    assert g_hi < 100.0, f"scanner-scale data still explodes: grad_norm={g_hi:.1f}"
    assert g_hi == pytest.approx(g_lo, rel=0.05), (
        f"objective still tracks the scanner scale: {g_lo:.3f} (amp 1) vs "
        f"{g_hi:.3f} (amp 92)"
    )


def test_clean_magnitude_normalisation_can_be_disabled() -> None:
    """The knob is real in both directions (CLAUDE.md #15), so pre-2026-07 runs
    remain reproducible by setting it false."""
    raw = _strategy(
        "subvoxel_sr",
        "restormer",
        24,
        1,
        {"scale": 2, "dim": 24},
        normalize_magnitude=False,
    )._clean_magnitude(_disk_kspace(amplitude=92.0))
    assert float(raw.max()) > 50.0  # untouched scanner scale

    normed = _strategy(
        "subvoxel_sr",
        "restormer",
        24,
        1,
        {"scale": 2, "dim": 24},
        normalize_magnitude=True,
    )._clean_magnitude(_disk_kspace(amplitude=92.0))
    assert float(normed.max()) == pytest.approx(1.0, abs=0.15)


def test_object_mask_survives_a_bright_outlier() -> None:
    """A single hot voxel must not collapse the display mask to nothing.

    Regression for the 2026-07 exp_vf_01 run: ``_object_mask`` thresholded at
    ``0.05 * amax``. The percentile-normalised M4Raw target carries a tail well
    above the tissue bulk (``target_mag`` ranged [0, 92.74] with the brain near
    1), so the threshold landed *above* the entire object, the mask kept a
    handful of pixels, and every saved validation PNG — real and fake — came out
    pure black while ``val/difference`` showed the few surviving pixels. The
    reference must be a robust upper quantile, not the raw maximum.
    """
    h = w = 32
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    r = ((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2).sqrt()
    anat = (r < h * 0.3).to(torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    clean_frac = float(
        ConcreteMultiAcquisitionStrategy._object_mask(anat, anat.shape).mean()
    )
    assert clean_frac > 0.2, "sanity: the disk itself must be retained"

    spiked = anat.clone()
    spiked[..., 0, 0] = 100.0  # one hot voxel, 100x the tissue bulk
    spiked_frac = float(
        ConcreteMultiAcquisitionStrategy._object_mask(spiked, spiked.shape).mean()
    )
    # the outlier must not wipe out the object support
    assert spiked_frac > 0.2, (
        f"one hot voxel collapsed the display mask to {spiked_frac:.4%} of the "
        "frame — every validation PNG renders black"
    )


def test_lambda_smooth_is_read_from_config_not_hardcoded() -> None:
    """The smoothness weight must come from ``physics.multi_acquisition``.

    Companion to ``test_multi_acquisition_lambda_smooth_is_a_declared_knob``:
    the schema declares the knob, and the strategy must actually read it
    (CLAUDE.md #15 — declare, read, validate, stamp).
    """
    st = _strategy("afi", "afi_ratio_cnn", 2, 2, {}, lambda_smooth=0.25)
    assert st._lambda_smooth == 0.25

    # and it changes the objective: a larger weight on a non-smooth prediction
    # must raise the total loss (the term is live, not decorative).
    target_k = _disk_kspace()
    torch.manual_seed(0)
    loss_lo, _, _ = _strategy(
        "afi", "afi_ratio_cnn", 2, 2, {}, lambda_smooth=0.0
    )._field_loss(target_k)
    torch.manual_seed(0)
    loss_hi, _, _ = _strategy(
        "afi", "afi_ratio_cnn", 2, 2, {}, lambda_smooth=1.0
    )._field_loss(target_k)
    assert float(loss_hi.detach()) > float(loss_lo.detach())


def test_field_visual_masking_is_display_only() -> None:
    """Masking the visuals must not change the graded field metrics."""
    st = _strategy("afi", "afi_ratio_cnn", 2, 2, {})
    target_k = _disk_kspace()
    val = st.validation_step(target_k, target_k)
    # air is masked in the display copy …
    assert float(st._last_visual_target[..., :4, :4].abs().max()) == 0.0
    # … yet the field metrics (computed on the unmasked field) are real numbers
    # spanning the full frame: MAE/MSE reflect the whole field, not a masked one.
    assert val["val_field_mae"] > 0.0
    assert torch.isfinite(torch.tensor(val["val_field_mse"]))


@pytest.mark.parametrize("method, model_name, in_ch, out_ch, tch, mk", CASES)
def test_train_step_bypasses_raw_batch_channel_guard(
    method, model_name, in_ch, out_ch, tch, mk
) -> None:
    """Regression for the 2026-06-03 cluster smoke crash.

    ``base.train_step`` validates the *loaded* batch channels against
    ``model.in_channels`` and 4-D-unpacks it; for multi-acquisition the model
    consumes a synthesised stack whose channel count is decoupled from the raw
    coil count (8 coils here). The override must run the step WITHOUT raising
    ``[DomainMismatch]`` / a 4-D unpack error, and return a live loss closure.
    """
    st = _strategy(method, model_name, in_ch, out_ch, mk)
    st.amp_helper = SimpleNamespace(get_autocast_context=contextlib.nullcontext)
    st.env.opt_g = SimpleNamespace()  # only identity-referenced in the step config
    # Raw 8-coil k-space batch: channels (8) != model.in_channels — exactly the
    # shape base.train_step would reject.
    raw = torch.randn(2, 8, 32, 32, dtype=torch.complex64)
    step_configs = st.train_step((raw, raw), epoch=0, iteration=1)
    assert isinstance(step_configs, list) and step_configs
    cfg = step_configs[0]
    assert cfg["model"] is st.generator_model and cfg["name"] == "generator"
    loss = cfg["closure"]()
    assert torch.isfinite(loss) and loss.requires_grad


def test_unknown_method_raises_at_setup() -> None:
    st = ConcreteMultiAcquisitionStrategy.__new__(ConcreteMultiAcquisitionStrategy)
    st.config = _cfg("triple_angle")  # not an implemented paired method
    st.device = torch.device("cpu")
    with pytest.raises(ValueError, match="method"):
        st._setup_strategy_specific_components()


# ── real-reference seam: grade against a real B0/B1 field from the batch ───────


def test_validation_flags_real_reference_when_b0_map_present() -> None:
    st = _strategy("dual_echo", "resnet_unwrap", 2, 2, {})
    target_k = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
    real_b0 = torch.full((2, 1, 32, 32), 25.0)
    val = st.validation_step(target_k, target_k, b0_map=real_b0)
    assert val["val_field_reference_real"] == 1.0
    val0 = st.validation_step(target_k, target_k)  # no real field → self-consistency
    assert val0["val_field_reference_real"] == 0.0


def test_real_field_selector_picks_b0_for_b0_method_b1_for_b1_method() -> None:
    st_b0 = _strategy("dual_echo", "resnet_unwrap", 2, 2, {})
    st_b1 = _strategy("double_angle", "dam_trigfit", 2, 2, {})
    batch = {
        "b0_map": torch.ones(1, 1, 8, 8),
        "b1_map": torch.full((1, 1, 8, 8), 0.9),
    }
    assert float(st_b0._real_field_from_batch(batch).mean()) == 1.0
    assert abs(float(st_b1._real_field_from_batch(batch).mean()) - 0.9) < 1e-5
    assert st_b0._real_field_from_batch(None) is None


def test_use_real_stack_grades_real_stack_vs_b0(monkeypatch) -> None:
    # Path A (oracle_bssfp): feed a REAL phase-cycled stack to the model and grade
    # the recovered field vs the real b0_map — no synthesis.
    cfg = _cfg("bssfp_banding")
    cfg.physics.multi_acquisition.use_real_stack = True
    st = ConcreteMultiAcquisitionStrategy.__new__(ConcreteMultiAcquisitionStrategy)
    st.config = cfg
    st.device = torch.device("cpu")
    st._to_device = lambda t: t  # type: ignore[method-assign]
    st._setup_strategy_specific_components()
    st.env = SimpleNamespace(
        generator=get_model_class("bssfp_b0_regressor")(in_channels=4, out_channels=1)
    )
    real_stack = torch.randn(2, 4, 16, 16, dtype=torch.complex64)  # [B, N, H, W]
    real_b0 = torch.full((2, 1, 16, 16), 18.0)
    val = st.validation_step(real_stack, real_stack, b0_map=real_b0)
    assert val["val_field_reference_real"] == 1.0
    assert "val_b0_field_rmse" in val
    assert val["val_b0_field_rmse"] == val["val_b0_field_rmse"]  # not NaN


def test_bssfp_banding_emits_guarded_b0_field_rmse() -> None:
    # The full Path-B slice: bssfp_banding synthesis + the BSSFPB0Regressor head
    # graded against a REAL b0_map via the guarded Hz metric (must fire + finite).
    st = _strategy("bssfp_banding", "bssfp_b0_regressor", 4, 1, {})
    img = 0.5 + torch.rand(2, 1, 32, 32)
    real_b0 = torch.full((2, 1, 32, 32), 20.0)
    val = st.validation_step(img, img, b0_map=real_b0)
    assert val["val_field_reference_real"] == 1.0
    assert "val_b0_field_rmse" in val
    assert val["val_b0_field_rmse"] == val["val_b0_field_rmse"]  # not NaN


def test_bssfp_banding_routes_b0_map() -> None:
    # bssfp_banding recovers off-resonance → must select the real b0_map
    # (not b1_map) for both training supervision and validation grading.
    st = _strategy("bssfp_banding", "resnet_unwrap", 4, 1, {})
    batch = {
        "b0_map": torch.full((1, 1, 8, 8), 30.0),
        "b1_map": torch.full((1, 1, 8, 8), 0.9),
    }
    picked = st._real_field_from_batch(batch)
    assert picked is not None
    assert float(picked.mean()) == 30.0
    assert st.macq.n_phase_cycles == 4


def test_real_field_seam_reads_training_batch_not_only_dict() -> None:
    """Regression for the 2026-06 dead real-reference seam.

    At runtime the trainer converts the loaded batch to a ``TrainingBatch``
    BEFORE handing it to the strategy / before the validation field-extraction.
    The old ``isinstance(batch, dict)`` guard rejected that ``TrainingBatch``
    form, so ``_real_field_from_batch`` always returned ``None`` and every
    real-B0/B1 arm silently self-graded on the synthesised field. The seam must
    read the field from a ``TrainingBatch`` whose ``b0_map`` lives in metadata
    (where ``BatchAdapter.from_dict`` stores all non-core keys)."""
    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(
        {
            "input": torch.randn(1, 2, 8, 8),
            "target": torch.randn(1, 2, 8, 8),
            "b0_map": torch.full((1, 1, 8, 8), 42.0),
        }
    )
    # The trainer-side extraction invariants the train.py fix relies on:
    assert "b0_map" in tb  # __contains__ falls back to metadata
    assert float(tb.get("b0_map").mean()) == 42.0

    # The train-side strategy seam must now pick it up (was None pre-fix).
    st = _strategy("bssfp_banding", "resnet_unwrap", 4, 1, {})
    picked = st._real_field_from_batch(tb)
    assert picked is not None and float(picked.mean()) == 42.0


# ---------------------------------------------------------------------------
# Shift-knowledge ladder (2026-07-26). The three rungs must be identical in
# every respect EXCEPT the content of the conditioning maps, or the ablation
# does not isolate what it claims to (CLAUDE.md pitfall #17).
# ---------------------------------------------------------------------------
_SUBVOXEL = {
    "model_name": "restormer",
    "in_ch": 24,
    "out_ch": 1,
    "mk": {"scale": 2, "dim": 24},
}


def _conditioned(shift_source: str):
    st = _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        _SUBVOXEL["in_ch"],
        _SUBVOXEL["out_ch"],
        _SUBVOXEL["mk"],
        shift_source=shift_source,
    )
    m0 = st._clean_magnitude(_disk_kspace(b=2, h=64, w=64))
    res = st.macq(m0)
    return st, res, st._condition_on_shifts(res)


@pytest.mark.parametrize("source", ["blind", "recovered", "oracle"])
def test_every_rung_produces_the_same_input_shape(source: str) -> None:
    """Identical geometry and parameter count across the ladder is the whole
    point: only the map CONTENT may differ."""
    _st, res, x = _conditioned(source)
    b, n, h, w = res.stack.shape
    assert x.shape == (b, 3 * n, h, w)
    assert torch.equal(x[:, :n], res.stack)


def test_blind_rung_zero_fills_the_conditioning_maps() -> None:
    _st, res, x = _conditioned("blind")
    assert x[:, res.stack.shape[1] :].abs().max().item() == 0.0


def test_oracle_rung_passes_the_true_shifts() -> None:
    st, res, x = _conditioned("oracle")
    n = res.stack.shape[1]
    got = x[:, n:, 0, 0].reshape(res.shifts.shape)
    assert torch.allclose(got, res.shifts / st._max_shift_px, atol=1e-6)


def test_recovered_rung_reconstructs_the_shifts_from_the_marker_alone() -> None:
    """The mechanism-fires probe. If this regressed to zeros or noise the arm
    would silently degrade to the blind rung while still claiming a fiducial."""
    st, res, x = _conditioned("recovered")
    n = res.stack.shape[1]
    got = x[:, n:, 0, 0].reshape(res.shifts.shape) * st._max_shift_px
    assert (got - res.shifts).abs().max().item() < 0.05
    # ...and it is measurably better than the blind rung's implicit zeros
    assert (got - res.shifts).abs().mean() < res.shifts.abs().mean() / 10


def test_recovered_rung_never_reads_the_ground_truth() -> None:
    """Registration must come from the fiducial. Blanking `shifts` after
    synthesis must not change the recovered conditioning maps."""
    st, res, x_ref = _conditioned("recovered")
    blanked = type(res)(
        stack=res.stack,
        field=res.field,
        echoes=res.echoes,
        shifts=torch.zeros_like(res.shifts),
        marker_stack=res.marker_stack,
        marker_hr=res.marker_hr,
    )
    x_blanked = st._condition_on_shifts(blanked)
    assert torch.allclose(x_ref, x_blanked, atol=1e-6)


def test_recovered_rung_requires_a_marker_stack() -> None:
    st, res, _x = _conditioned("recovered")
    stripped = type(res)(
        stack=res.stack,
        field=res.field,
        echoes=res.echoes,
        shifts=res.shifts,
        marker_stack=None,
    )
    with pytest.raises(ValueError, match="no marker stack"):
        st._condition_on_shifts(stripped)


def test_shift_mae_is_reported_and_ranks_the_rungs() -> None:
    """`shift_mae_px` is a diagnostic, never a loss (phase correlation has no
    parameters), and it is what makes a null PSNR result attributable."""
    maes = {}
    for source in ("blind", "recovered", "oracle"):
        st, res, _x = _conditioned(source)
        assert st._last_shift_mae is not None
        maes[source] = float(st._last_shift_mae)
    assert maes["oracle"] == pytest.approx(0.0, abs=1e-6)
    assert maes["recovered"] < maes["blind"] / 10


def test_channel_count_mismatch_raises_at_setup() -> None:
    """n_frames*3 is a contract, not a convention: an 8-channel model would
    silently drop every conditioning map."""
    with pytest.raises(ValueError, match="shift-conditioning maps"):
        _strategy(
            "subvoxel_sr",
            "restormer",
            8,
            1,
            {"scale": 2, "dim": 24},
            shift_source="oracle",
        )


# ── super-Nyquist band probe (PR-1) ───────────────────────────────────────────


def _probe_strategy(
    lambda_band: float = 0.0, band_probe: bool = True, marker_channels: bool = False
):
    return _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        _SUBVOXEL["in_ch"] * (4 if marker_channels else 3) // 3,
        _SUBVOXEL["out_ch"],
        _SUBVOXEL["mk"],
        shift_source="recovered",
        band_probe=band_probe,
        lambda_band=lambda_band,
        marker_channels=marker_channels,
    )


def test_probe_reports_a_spectrum_and_the_interpolation_floor() -> None:
    """The floor is not zero and is arm-dependent: boxcar pooling leaves aliased
    super-Nyquist energy that bilinear interpolation partially unfolds. Quoting
    an absolute gain without it would read as recovery when it is interpolation."""
    st = _probe_strategy()
    st._field_loss(_disk_kspace(b=2, h=64, w=64))
    spec = st._last_band_spectrum
    assert sum(k.startswith("snf_band_") for k in spec) == 4
    assert "snf_super_nyquist" in spec and "snf_sub_nyquist" in spec
    assert "snf_floor_super_nyquist" in spec
    assert spec["snf_gain_over_floor"] == pytest.approx(
        spec["snf_super_nyquist"] - spec["snf_floor_super_nyquist"], abs=1e-6
    )


def test_probe_off_reports_nothing_rather_than_a_stale_spectrum() -> None:
    """A spectrum left over from a previous step is indistinguishable from a
    real measurement of the current one."""
    st = _probe_strategy(band_probe=False)
    st._field_loss(_disk_kspace(b=2, h=64, w=64))
    assert st._last_band_spectrum == {}
    assert st._last_band_loss is None
    assert "val_snf_super_nyquist" not in st.validation_step(
        _disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64)
    )


def test_probe_term_enters_the_objective_only_when_weighted() -> None:
    """enabled + lambda_band=0 is the CONTROL: identical measurement, no
    constraint, so the arms differ on exactly one knob (#17)."""
    torch.manual_seed(0)
    control = _probe_strategy(lambda_band=0.0)
    target = _disk_kspace(b=2, h=64, w=64)
    loss_ctrl, _p, _f = control._field_loss(target)
    assert control._last_band_loss is not None  # measured
    # the measured probe value is NOT added to the control's objective
    assert float(loss_ctrl.detach()) == pytest.approx(
        float(control._last_task_loss), abs=1e-6
    )

    torch.manual_seed(0)
    weighted = _probe_strategy(lambda_band=0.5)
    loss_w, _p, _f = weighted._field_loss(target)
    assert float(loss_w.detach()) > float(weighted._last_task_loss)


def test_probe_term_reaches_the_generator_weights() -> None:
    """A loss with no gradient is a facade (#16). Cosine similarity is scale
    invariant BY DESIGN, so this asserts a shape-changing gradient, not a
    scale one."""
    st = _probe_strategy(lambda_band=1.0)
    loss, _p, _f = st._field_loss(_disk_kspace(b=2, h=64, w=64))
    st.generator_model.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in st.generator_model.parameters() if p.grad is not None]
    assert grads, "the probe term reached no parameter at all"
    assert sum(float(g.norm()) for g in grads) > 0.0


def test_probe_spectrum_surfaces_in_validation_metrics() -> None:
    """Declared-but-unemitted metrics are pitfall #18; the val_ keys must exist."""
    st = _probe_strategy()
    val = st.validation_step(
        _disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64)
    )
    assert "val_snf_super_nyquist" in val
    assert "val_snf_floor_super_nyquist" in val
    assert -1.0 <= val["val_snf_super_nyquist"] <= 1.0


def test_probe_reuses_the_anatomy_path_input_construction() -> None:
    """If the probe built its own input the measurement would describe a
    different network than the one being trained."""
    st = _probe_strategy()
    m0 = st._clean_magnitude(_disk_kspace(b=2, h=64, w=64))
    res = st.macq(m0)
    anat = st._condition_on_shifts(res)
    marker = st._append_shift_maps(res.marker_stack, st._last_shifts)
    assert marker.shape == anat.shape
    assert torch.equal(marker[:, res.stack.shape[1] :], anat[:, res.stack.shape[1] :])


# ── fiducial channels fed to the model (PR-2) ─────────────────────────────────


def test_marker_channels_widen_the_input_to_four_per_frame() -> None:
    """8 anatomy + 8 fiducial + 2*8 shift maps. The fiducial rides beside the
    anatomy so a marker-keyed attention has something to key on."""
    st = _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        32,
        1,
        {"scale": 2, "dim": 24},
        shift_source="recovered",
        marker_channels=True,
    )
    m0 = st._clean_magnitude(_disk_kspace(b=2, h=64, w=64))
    res = st.macq(m0)
    x = st._condition_on_shifts(res)
    b, n, h, w = res.stack.shape
    assert x.shape == (b, 4 * n, h, w)
    assert torch.equal(x[:, :n], res.stack)
    assert torch.equal(x[:, n : 2 * n], res.marker_stack)


def test_marker_channels_carry_the_real_fiducial_not_zeros() -> None:
    """A zero-filled block would make every routing of the attention ablation
    identical while still reporting three different arms."""
    st = _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        32,
        1,
        {"scale": 2, "dim": 24},
        shift_source="recovered",
        marker_channels=True,
    )
    res = st.macq(st._clean_magnitude(_disk_kspace(b=2, h=64, w=64)))
    n = res.stack.shape[1]
    marker = st._condition_on_shifts(res)[:, n : 2 * n]
    assert float(marker.abs().max()) > 0.0
    assert float(marker.std()) > 0.0


def test_channel_contract_accounts_for_the_marker_block() -> None:
    """n_frames*4 with the fiducial, n_frames*3 without: declaring the old width
    must raise rather than silently truncating the marker."""
    with pytest.raises(ValueError, match="fiducial frames"):
        _strategy(
            "subvoxel_sr",
            _SUBVOXEL["model_name"],
            24,
            1,
            {"scale": 2, "dim": 24},
            shift_source="recovered",
            marker_channels=True,
        )


def _blind_marker_strategy() -> ConcreteMultiAcquisitionStrategy:
    return _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        32,
        1,
        {"scale": 2, "dim": 24},
        shift_source="blind",
        marker_channels=True,
    )


def test_blind_marker_channels_carry_the_marker_and_zero_maps() -> None:
    """VFU-08: under 'blind' the marker is still synthesised and fed, while the
    shift maps stay zero, so the marker is the only registration source."""
    st = _blind_marker_strategy()
    assert st.macq.marker_enabled
    res = st.macq(st._clean_magnitude(_disk_kspace(b=2, h=32, w=32)))
    x = st._condition_on_shifts(res)
    n = res.stack.shape[1]
    assert float(x[:, n : 2 * n].std()) > 0.0
    assert torch.equal(x[:, 2 * n :], torch.zeros_like(x[:, 2 * n :]))


def test_blind_marker_channels_change_the_model_output() -> None:
    """The marker block reaches the network: zeroing it moves the prediction."""
    st = _blind_marker_strategy()
    res = st.macq(st._clean_magnitude(_disk_kspace(b=2, h=32, w=32)))
    x = st._condition_on_shifts(res)
    n = res.stack.shape[1]
    ablated = x.clone()
    ablated[:, n : 2 * n] = 0.0
    with torch.no_grad():
        delta = (st.generator_model(x) - st.generator_model(ablated)).abs().max()
    assert float(delta) > 0.0


def test_band_probe_matches_the_model_width_when_marker_channels_are_on() -> None:
    """The exp_vfulf_02 cohort enables BOTH. The probe feeds the fiducial as the
    signal under test, so with marker channels it must occupy the instrument
    slot too — otherwise it builds a 3n input against a 4n model and every arm
    in that cohort dies at the first probe."""
    st = _probe_strategy(band_probe=True, marker_channels=True)
    res = st.macq(st._clean_magnitude(_disk_kspace(b=2, h=64, w=64)))
    st._condition_on_shifts(res)
    probe = st._band_probe(res)
    assert probe is not None and torch.isfinite(probe)
    assert "snf_super_nyquist" in st._last_band_spectrum


# ── per-sample marker realisations (VFU-02) ───────────────────────────────────


def _per_sample_strategy(target: str = "matched", lambda_band: float = 0.1, realisation="per_sample"):
    st = _probe_strategy(lambda_band=lambda_band)
    macq = st.config.physics.multi_acquisition
    macq.subvoxel_registration = SubvoxelRegistrationConfig(
        shift_source="recovered",
        marker_grid_spacing=16,
        marker_sigma=2.0,
        marker_jitter=0.35,
        max_shift_px=1.0,
        marker_realisation=realisation,
    )
    macq.band_probe = BandProbeConfig(
        enabled=True, lambda_band=lambda_band, target_realisation=target
    )
    st._setup_strategy_specific_components()
    return st


def test_training_draws_a_fresh_even_seed_per_sample_and_step() -> None:
    st = _per_sample_strategy()
    a = st._marker_seeds("train", 0, 2)
    b = st._marker_seeds("train", 0, 2)
    assert len(set(a + b)) == 4 and all(x % 2 == 0 for x in a + b)
    assert st._marker_seeds("train", 1, 2) != a  # a new epoch restarts the ordinals elsewhere


def test_validation_markers_are_odd_stable_and_keyed_on_batch_idx() -> None:
    st = _per_sample_strategy()
    v = st._marker_seeds("val", 0, 2, batch_idx=3)
    assert v == st._marker_seeds("val", 0, 2, batch_idx=3)
    assert all(x % 2 == 1 for x in v)
    with pytest.raises(ValueError, match="batch_idx"):
        st._marker_seeds("val", 0, 2)


def test_fixed_realisation_draws_no_seeds() -> None:
    assert _per_sample_strategy(realisation="fixed")._marker_seeds("train", 0, 2) is None


def test_registration_stays_accurate_on_per_sample_markers() -> None:
    """Every consumer reads res.marker_hr; a stale fixed reference would wreck this."""
    st = _per_sample_strategy()
    m0 = st._clean_magnitude(_disk_kspace(b=2, h=32, w=32))
    res = st.macq(m0, marker_seeds=st._marker_seeds("train", 0, 2))
    st._condition_on_shifts(res)
    assert float(st._last_shift_mae) < 0.1 * float(res.shifts.abs().mean())


def test_per_sample_markers_change_the_probe_target() -> None:
    torch.manual_seed(0)
    fixed = _per_sample_strategy(realisation="fixed")
    torch.manual_seed(0)
    per = _per_sample_strategy()
    per.env.generator.load_state_dict(fixed.generator_model.state_dict())
    target = _disk_kspace(b=2, h=32, w=32)
    torch.manual_seed(0)
    fixed._field_loss(target)
    torch.manual_seed(0)
    per._field_loss(target, marker_seeds=per._marker_seeds("train", 0, 2))
    assert not torch.allclose(fixed._last_band_loss, per._last_band_loss)


def test_mismatched_target_changes_the_loss_not_the_reported_spectrum() -> None:
    """Wrong-marker control: same input, same weights, a wrong optimisation
    target; the reported val_snf_* still score the matched marker."""
    torch.manual_seed(0)
    matched = _per_sample_strategy()
    torch.manual_seed(0)
    wrong = _per_sample_strategy(target="mismatched")
    wrong.env.generator.load_state_dict(matched.generator_model.state_dict())
    target = _disk_kspace(b=2, h=32, w=32)
    seeds = (2, 4)
    torch.manual_seed(0)
    matched._field_loss(target, marker_seeds=seeds)
    torch.manual_seed(0)
    wrong._field_loss(target, marker_seeds=seeds)
    assert not torch.allclose(matched._last_band_loss, wrong._last_band_loss)
    assert matched._last_band_spectrum["snf_super_nyquist"] == pytest.approx(
        wrong._last_band_spectrum["snf_super_nyquist"], abs=1e-6
    )


def test_validation_step_runs_on_the_held_out_family() -> None:
    st = _per_sample_strategy()
    val = st.validation_step(_disk_kspace(b=2, h=32, w=32), _disk_kspace(b=2, h=32, w=32), batch_idx=0)
    assert st._last_result.marker_seeds is not None
    assert all(s % 2 == 1 for s in st._last_result.marker_seeds)
    assert "val_snf_super_nyquist" in val


# ── held-out-subject conformal calibration (VFU-09) ───────────────────────────


def _heldout_strategy():
    from spectramr.config.schemas.physics import AnchorConformalConfig

    st = _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        _SUBVOXEL["in_ch"],
        _SUBVOXEL["out_ch"],
        _SUBVOXEL["mk"],
        shift_source="recovered",
    )
    st.config.physics.multi_acquisition.anchor_conformal = AnchorConformalConfig(
        enabled=True, n_strata=1, calibration_source="held_out_subjects"
    )
    st._setup_strategy_specific_components()
    return st


def _val(st, subject: str, idx: int) -> dict:
    target = _disk_kspace(b=2, h=32, w=32)
    return st.validation_step(target, target, batch_idx=idx, batch_data={"subject_id": [subject] * 2})


def test_heldout_calibration_reports_once_per_pass_on_other_subjects() -> None:
    st = _heldout_strategy()
    st.reset_validation_metrics()
    per_batch = _val(st, "s1", 0)
    _val(st, "s2", 1)
    assert not any(k.startswith("val_conformal") for k in per_batch)  # no per-batch refit
    out = st.finalize_validation()
    assert out["val_conformal_n_calibration_subjects"] == 1.0
    assert out["val_conformal_n_test_subjects"] == 1.0
    assert "val_conformal_worst_coverage" in out


def test_heldout_calibration_raises_on_a_training_subject() -> None:
    st = _heldout_strategy()
    target = _disk_kspace(b=2, h=32, w=32)
    st._compute_losses_impl(None, target, 0, batch={"subject_id": ["s1", "s1"]})
    st.reset_validation_metrics()
    _val(st, "s1", 0)
    _val(st, "s2", 1)
    with pytest.raises(ValueError, match="overlaps training"):
        st.finalize_validation()


def test_heldout_calibration_needs_subject_ids() -> None:
    st = _heldout_strategy()
    target = _disk_kspace(b=2, h=32, w=32)
    with pytest.raises(ValueError, match="needs subject_id"):
        st.validation_step(target, target, batch_idx=0)


# ── acquired 64 mT input stream (VFU-07) ──────────────────────────────────────


def _stream_strategy(stream: str, fov_mask: str = "acquired_support"):
    st = _strategy(
        "subvoxel_sr", _SUBVOXEL["model_name"], 24, 1, _SUBVOXEL["mk"], shift_source="blind"
    )
    macq = st.config.physics.multi_acquisition
    macq.n_frames = 1
    macq.input_stream = stream
    macq.fov_mask = fov_mask
    st.config.model.in_channels = 3
    st.env = SimpleNamespace(
        generator=get_model_class(_SUBVOXEL["model_name"])(
            in_channels=3, out_channels=1, **_SUBVOXEL["mk"]
        )
    )
    st._setup_strategy_specific_components()
    return st


def _half_fov(x: torch.Tensor) -> torch.Tensor:
    """An image-domain ULF stand-in whose right half lies outside the FOV (exact 0)."""
    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    img = ifft2c(x).abs()
    img[..., img.shape[-1] // 2 :] = 0.0
    return fft2c(img.to(torch.complex64))


def test_acquired_stream_feeds_the_loaded_input_not_the_target() -> None:
    """The model input is the pooled 64 mT slice: at a fixed target, changing the
    INPUT moves the prediction on the acquired stream and not on the twin one."""
    target = _disk_kspace(b=2, h=32, w=32)
    inputs = (_disk_kspace(b=2, h=32, w=32), _half_fov(target))
    for stream, fov, moves in (("acquired", "acquired_support", True), ("twin", "none", False)):
        st = _stream_strategy(stream, fov)
        preds = []
        for x in inputs:
            torch.manual_seed(0)
            preds.append(st._field_loss(target, acquired_input=x)[1])
        assert (not torch.allclose(*preds)) == moves, stream
    res = _stream_strategy("acquired")._acquired_result(
        torch.ones(2, 1, 32, 32), torch.ones(2, 1, 32, 32)
    )
    assert res.stack.shape == (2, 1, 16, 16) and float(res.shifts.abs().sum()) == 0.0


def test_fov_mask_zeroes_both_sides_outside_the_ulf_field_of_view() -> None:
    st = _stream_strategy("acquired")
    target = _disk_kspace(b=2, h=32, w=32)
    _l, pred_field, field = st._field_loss(target, acquired_input=_half_fov(target))
    right = (..., slice(16, None))
    assert float(pred_field[right].abs().max()) == 0.0
    assert float(field[right].abs().max()) == 0.0
    assert float(field[..., :16].abs().max()) > 0.0


def test_acquired_stream_without_an_input_raises() -> None:
    with pytest.raises(ValueError, match="need the loaded 64 mT input"):
        _stream_strategy("acquired")._field_loss(_disk_kspace(b=2, h=32, w=32))


# ── relaxometric fiducial calibration (PR-3) ──────────────────────────────────

_ACQ = {"tr_ms": 500.0, "te_ms": 15.0, "flip_deg": 90.0}


def _relax_cfg(factored: bool, factor_form: str = "scalar") -> RelaxometricCalibrationConfig:
    return RelaxometricCalibrationConfig(
        enabled=True,
        factored=factored,
        factor_form=factor_form,
        source={"field_strength_t": 0.064, **_ACQ},
        target={"field_strength_t": 3.0, **_ACQ},
        marker_t1_ms=500.0,
        marker_t1_target_ms=900.0,
        marker_t2_ms=80.0,
    )


def _relax_strategy(factored: bool) -> ConcreteMultiAcquisitionStrategy:
    return _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        _SUBVOXEL["in_ch"],
        _SUBVOXEL["out_ch"],
        _SUBVOXEL["mk"],
        shift_source="recovered",
        relax=_relax_cfg(factored),
    )


def _tissue_strategy(out_ch: int = 4) -> ConcreteMultiAcquisitionStrategy:
    return _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        _SUBVOXEL["in_ch"],
        out_ch,
        _SUBVOXEL["mk"],
        shift_source="recovered",
        relax=_relax_cfg(True, factor_form="tissue_map"),
    )


def test_tissue_map_head_scales_g_by_a_per_pixel_declared_gain() -> None:
    """VFU-05: pred = g(x) * kappa(x), kappa(x) a softmax over the DECLARED
    per-tissue gains taken from channels 1..3, so it varies per pixel."""
    from spectramr.infrastructure.physics.tissue_factor import tissue_kappa_map

    st = _tissue_strategy()
    target = _disk_kspace(b=2, h=32, w=32)
    torch.manual_seed(0)
    _l, pred_field, _f = st._field_loss(target)
    torch.manual_seed(0)
    res = st.macq(st._clean_magnitude(target))
    with torch.no_grad():
        raw = st.generator_model(st._condition_on_shifts(res))
    kappa = tissue_kappa_map(raw[:, 1:4], st._tissue_gains)
    assert torch.allclose(pred_field.detach(), raw[:, :1] * kappa, atol=1e-5)
    lo, hi = float(st._tissue_gains.min()), float(st._tissue_gains.max())
    assert lo - 1e-6 <= float(kappa.min()) and float(kappa.max()) <= hi + 1e-6
    assert float(kappa.std()) > 0.0


def test_tissue_map_differs_from_the_scalar_control_on_the_same_weights() -> None:
    torch.manual_seed(0)
    tissue = _tissue_strategy()
    scalar = _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        _SUBVOXEL["in_ch"],
        4,
        _SUBVOXEL["mk"],
        shift_source="recovered",
        relax=_relax_cfg(True),
    )
    scalar.env.generator.load_state_dict(tissue.generator_model.state_dict())
    target = _disk_kspace(b=2, h=32, w=32)
    torch.manual_seed(0)
    loss_t, _p, _f = tissue._field_loss(target)
    torch.manual_seed(0)
    loss_s, _p, _f = scalar._field_loss(target)
    assert not torch.allclose(loss_t, loss_s)


def test_tissue_map_refuses_a_one_channel_model() -> None:
    with pytest.raises(ValueError, match="out_channels=4"):
        _tissue_strategy(out_ch=1)


def test_tissue_map_reports_class_agreement() -> None:
    val = _tissue_strategy().validation_step(
        _disk_kspace(b=2, h=32, w=32), _disk_kspace(b=2, h=32, w=32)
    )
    assert 0.0 <= val["val_kappa_class_agreement"] <= 1.0


def test_declared_gain_is_recovered_from_the_data_on_marker_support() -> None:
    """The mechanism-fires check: the constant handed to the model must match
    the one the fiducial actually shows. A factored model applying a WRONG
    constant confidently is worse than not factoring at all, which is why this
    is measured on both arms."""
    st = _relax_strategy(factored=True)
    val = st.validation_step(
        _disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64)
    )
    assert val["val_kappa_predicted"] == pytest.approx(0.674, abs=0.01)
    assert abs(val["val_kappa_error"]) < 0.02


def test_kappa_is_reported_on_the_unfactored_control_too() -> None:
    """It is a property of the simulator, not of the model, so both arms must
    report it or the comparison has no shared reference."""
    val = _relax_strategy(factored=False).validation_step(
        _disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64)
    )
    assert "val_kappa_predicted" in val and "val_kappa_measured" in val


def test_factoring_scales_the_prediction_by_exactly_kappa() -> None:
    """Identical weights and identical input, so the ONLY difference is the
    constant — that is what makes the ablation attributable."""
    torch.manual_seed(0)
    plain = _relax_strategy(factored=False)
    torch.manual_seed(0)
    factored = _relax_strategy(factored=True)
    factored.env.generator.load_state_dict(plain.generator_model.state_dict())
    target = _disk_kspace(b=2, h=64, w=64)
    torch.manual_seed(0)
    _l, p_plain, _f = plain._field_loss(target)
    torch.manual_seed(0)
    _l, p_fact, _f = factored._field_loss(target)
    assert torch.allclose(p_fact, p_plain * factored._marker_kappa, atol=1e-5)


def test_no_kappa_reported_when_calibration_is_off() -> None:
    """A stale constant reported by an arm that never applied one is
    indistinguishable from a real measurement."""
    val = _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        _SUBVOXEL["in_ch"],
        _SUBVOXEL["out_ch"],
        _SUBVOXEL["mk"],
        shift_source="recovered",
    ).validation_step(_disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64))
    assert "val_kappa_predicted" not in val and "val_kappa_measured" not in val


# ── anchor-calibrated conformal prediction (PR-4) ─────────────────────────────


def _conformal_strategy(n_strata: int = 4):
    from spectramr.config.schemas.physics import AnchorConformalConfig

    st = _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        _SUBVOXEL["in_ch"],
        _SUBVOXEL["out_ch"],
        _SUBVOXEL["mk"],
        shift_source="recovered",
    )
    st.config.physics.multi_acquisition.anchor_conformal = AnchorConformalConfig(
        enabled=True, alpha=0.1, n_strata=n_strata
    )
    st._setup_strategy_specific_components()
    return st


def test_conformal_coverage_is_reported_per_stratum() -> None:
    """A single verdict hides which difficulty regime failed, and that regime
    is usually the interesting one."""
    st = _conformal_strategy(n_strata=4)
    val = st.validation_step(
        _disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64)
    )
    assert sum(k.startswith("val_conformal_coverage_s") for k in val) == 4
    assert "val_conformal_guaranteed" in val
    assert "val_conformal_mean_half_width" in val


def test_conformal_gate_can_fail_and_says_so() -> None:
    """An untrained network's anatomy residuals are nothing like its marker
    residuals, so the guarantee must NOT hold here. A calibrator that always
    reports success is not a guarantee."""
    st = _conformal_strategy(n_strata=4)
    val = st.validation_step(
        _disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64)
    )
    assert val["val_conformal_guaranteed"] == 0.0
    assert val["val_conformal_worst_coverage"] < 0.85
    # "not certified" and "certified to fail" must stay distinguishable
    assert "val_conformal_n_unmeasured" in val


def test_no_conformal_keys_when_disabled() -> None:
    """A stale coverage number from another arm is indistinguishable from a
    real measurement."""
    val = _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        _SUBVOXEL["in_ch"],
        _SUBVOXEL["out_ch"],
        _SUBVOXEL["mk"],
        shift_source="recovered",
    ).validation_step(_disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64))
    assert not any(k.startswith("val_conformal_") for k in val)


def test_calibration_never_touches_the_anatomy_target() -> None:
    """Calibrating on the anatomy would assume exactly what the arm tests, so
    the quantiles must depend on the MARKER only.

    The dither is re-seeded between the two calls: the simulator draws fresh
    offsets every forward pass, so without that the marker frames differ too
    and the comparison would be measuring the draw rather than the target.
    """
    st = _conformal_strategy(n_strata=2)
    torch.manual_seed(0)
    st.validation_step(_disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64))
    first = st._conformal.quantiles.clone()
    torch.manual_seed(0)
    st.validation_step(
        _disk_kspace(b=2, h=64, w=64, amplitude=7.0),
        _disk_kspace(b=2, h=64, w=64, amplitude=7.0),
    )
    assert torch.allclose(first, st._conformal.quantiles)


# ── fiducial-measured forward operator (PR-5) ─────────────────────────────────


def _psf_strategy(source: str = "measured", true_sigma: float | None = 2.4):
    from spectramr.config.schemas.physics import ForwardPsfConfig

    st = _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        _SUBVOXEL["in_ch"],
        _SUBVOXEL["out_ch"],
        _SUBVOXEL["mk"],
        shift_source="recovered",
    )
    st.config.physics.multi_acquisition.subvoxel_registration = (
        SubvoxelRegistrationConfig(
            shift_source="recovered",
            marker_grid_spacing=16,
            marker_sigma=1.0,
            marker_jitter=0.45,
            max_shift_px=1.0,
        )
    )
    st.config.physics.multi_acquisition.forward_psf = ForwardPsfConfig(
        enabled=True,
        source=source,
        sigma_px=1.5,
        true_sigma_px=true_sigma,
        kernel_size=9,
        mu=1e-3,
    )
    # +1 input channel: the forward-PSF |OTF| map the model is conditioned on.
    st.config.model.in_channels = _SUBVOXEL["in_ch"] + 1
    st.env = SimpleNamespace(
        generator=get_model_class(_SUBVOXEL["model_name"])(
            in_channels=_SUBVOXEL["in_ch"] + 1, out_channels=_SUBVOXEL["out_ch"], **_SUBVOXEL["mk"]
        )
    )
    st._setup_strategy_specific_components()
    return st


def _conditioned_psf(st, seed: int = 0):
    torch.manual_seed(seed)
    res = st.macq(st._clean_magnitude(_disk_kspace(b=2, h=32, w=32)))
    return res, st._condition_on_shifts(res)


def test_psf_channel_is_the_last_input_channel_on_both_sources() -> None:
    """VFU-01: same channel count on both arms; only the kernel content differs."""
    for source in ("assumed", "measured"):
        st = _psf_strategy(source=source)
        res, x = _conditioned_psf(st)
        n = res.stack.shape[1]
        assert x.shape[1] == 3 * n + 1
        h, w = x.shape[-2:]
        assert torch.allclose(x[:, -1, h // 2, w // 2], torch.ones(2), atol=1e-5)


def test_assumed_psf_channel_is_rendered_once_per_grid() -> None:
    """The assumed kernel is constant, so its |OTF| is cached, not rebuilt per step."""
    from spectramr.infrastructure.physics.psf_conditioning import otf_channel

    st = _psf_strategy(source="assumed")
    res, _x = _conditioned_psf(st)
    first = st._last_psf_channel
    _conditioned_psf(st, seed=1)
    assert st._last_psf_channel.data_ptr() == first.data_ptr()
    h, w = res.stack.shape[-2:]
    assert torch.equal(first[:1], otf_channel(st._assumed_lr_kernel, (h, w)))


def test_measured_psf_channel_tracks_the_true_blur_not_the_assumption() -> None:
    """With true_sigma != sigma, the measured channel sits nearer the true MTF
    than the assumed channel does -- the information the assumed arm lacks."""
    from spectramr.infrastructure.physics.psf_conditioning import otf_channel
    from spectramr.infrastructure.physics.psf_estimation import gaussian_psf

    _assumed_res, assumed_x = _conditioned_psf(_psf_strategy(source="assumed"))
    _measured_res, measured_x = _conditioned_psf(_psf_strategy(source="measured"))
    h, w = assumed_x.shape[-2:]
    truth = otf_channel(gaussian_psf(9, 2.4 / 2).reshape(1, 1, 1, 9, 9), (h, w))
    err_assumed = (assumed_x[:, -1:] - truth).abs().mean()
    err_measured = (measured_x[:, -1:] - truth).abs().mean()
    assert err_measured < 0.5 * err_assumed


def test_psf_source_changes_the_model_output() -> None:
    """The declared source reaches the network: same weights, same frames,
    a different PSF channel, a different prediction."""
    st = _psf_strategy(source="assumed")
    _res, x = _conditioned_psf(st)
    measured = _psf_strategy(source="measured")
    _res_m, x_m = _conditioned_psf(measured)
    swapped = x.clone()
    swapped[:, -1:] = x_m[:, -1:]
    with torch.no_grad():
        delta = (st.generator_model(x) - st.generator_model(swapped)).abs().max()
    assert float(delta) > 0.0


def test_null_control_channels_agree_when_the_assumption_is_exact() -> None:
    """true_sigma unset: the simulator applies the assumed blur, so the two
    sources disagree only by estimation error -- well under the gap a real
    model error opens (0.06 against 0.14 on this 32x32, 4-peak marker)."""

    def gap(true_sigma):
        _r, a = _conditioned_psf(_psf_strategy(source="assumed", true_sigma=true_sigma))
        _r, m = _conditioned_psf(_psf_strategy(source="measured", true_sigma=true_sigma))
        return float((a[:, -1] - m[:, -1]).abs().mean())

    assert gap(None) < 0.6 * gap(2.4)


def test_psf_conditioning_refuses_a_control_grid() -> None:
    from pydantic import ValidationError

    from spectramr.config.schemas.physics import ForwardPsfConfig

    with pytest.raises(ValidationError, match="control_rows/control_cols must be 1"):
        ForwardPsfConfig(enabled=True, source="measured", control_rows=2, control_cols=2)


def test_measured_psf_tracks_the_simulated_blur_not_the_assumption() -> None:
    """The mechanism-fires check. The simulator applies sigma=2.4 HR pixels;
    on the pooled grid at sr_scale=2 that is FWHM 2.83, against an assumed 1.77.
    A measurement that returned the assumption would prove nothing."""
    st = _psf_strategy()
    val = st.validation_step(
        _disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64)
    )
    assert val["val_psf_fwhm_measured"] == pytest.approx(2.4 / 2 * 2.3548, rel=0.10)
    assert val["val_psf_fwhm_assumed"] == pytest.approx(1.5 / 2 * 2.3548, rel=0.05)
    assert val["val_psf_fwhm_error"] > 0.5


def test_identifiability_is_reported_beside_the_kernel() -> None:
    """A kernel estimated where the marker has no spectral energy is
    interpolation, and a FWHM quoted without this hides that."""
    val = _psf_strategy().validation_step(
        _disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64)
    )
    assert 0.0 <= val["val_psf_identifiability"] <= 1.0


def test_psf_is_reported_on_the_assumed_arm_too() -> None:
    """It is a property of the simulator, not of the model, so both arms must
    carry it or the comparison has no shared reference."""
    val = _psf_strategy(source="assumed").validation_step(
        _disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64)
    )
    assert "val_psf_fwhm_measured" in val and "val_psf_fwhm_assumed" in val


def test_no_psf_keys_when_disabled() -> None:
    val = _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        _SUBVOXEL["in_ch"],
        _SUBVOXEL["out_ch"],
        _SUBVOXEL["mk"],
        shift_source="recovered",
    ).validation_step(_disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64))
    assert not any(k.startswith("val_psf_") for k in val)


def test_estimator_bias_shrinks_as_the_blur_outgrows_the_pooling() -> None:
    """A real limit of measuring on the pooled grid, quantified rather than
    hidden.

    Pooling and convolution do not commute, so the LR-domain effective kernel of
    an HR-domain blur is not exactly the downscaled kernel. The residual reads
    as a positive FWHM bias that is LARGE when the blur is comparable to the
    pooling factor and small when it dominates: +0.63 px at a simulated
    sigma of 1.5 (the null control, where the assumption is exact by
    construction) against +0.05 px at 2.4, the width the arms declare.

    So the null control does NOT tie exactly, and an arm must not read a small
    positive error as a detected model error. The arms are configured where the
    bias is negligible, and this test pins the ordering that makes that safe.
    """

    def _err(true_sigma: float | None) -> float:
        val = _psf_strategy(source="assumed", true_sigma=true_sigma).validation_step(
            _disk_kspace(b=2, h=64, w=64), _disk_kspace(b=2, h=64, w=64)
        )
        return val["val_psf_fwhm_measured"] - ((true_sigma or 1.5) / 2 * 2.3548)

    near_pooling = _err(1.5)
    well_resolved = _err(2.4)
    assert near_pooling > 0.0, "the bias is positive: pooling widens the estimate"
    assert abs(well_resolved) < abs(near_pooling) / 3.0
    assert abs(well_resolved) < 0.2


# ── anatomy high-frequency term (2026-07-26) ──────────────────────────────────


def _hf_strategy(lambda_anatomy: float, shift_source: str = "recovered"):
    st = _strategy(
        "subvoxel_sr",
        _SUBVOXEL["model_name"],
        _SUBVOXEL["in_ch"],
        _SUBVOXEL["out_ch"],
        _SUBVOXEL["mk"],
        shift_source=shift_source,
    )
    st.config.physics.multi_acquisition.lambda_smooth = 0.0
    st.config.physics.multi_acquisition.band_probe = BandProbeConfig(
        enabled=True, lambda_anatomy=lambda_anatomy
    )
    st._setup_strategy_specific_components()
    return st


def test_anatomy_hf_term_enters_the_objective_only_when_weighted() -> None:
    """0.0 measures and reports; >0 optimises. The control and the treatment
    differ by exactly this weight."""
    torch.manual_seed(0)
    off = _hf_strategy(0.0)
    target = _disk_kspace(b=2, h=64, w=64)
    loss_off, _p, _f = off._field_loss(target)
    assert off._last_hf_loss is not None  # measured
    assert float(loss_off.detach()) == pytest.approx(
        float(off._last_task_loss), abs=1e-6
    )

    torch.manual_seed(0)
    on = _hf_strategy(0.5)
    loss_on, _p, _f = on._field_loss(target)
    assert float(loss_on.detach()) > float(on._last_task_loss)


def test_anatomy_hf_term_reaches_the_generator() -> None:
    """A loss with no gradient is a facade (#16)."""
    st = _hf_strategy(1.0)
    loss, _p, _f = st._field_loss(_disk_kspace(b=2, h=64, w=64))
    st.generator_model.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in st.generator_model.parameters() if p.grad is not None]
    assert grads and sum(float(g.norm()) for g in grads) > 0.0


@pytest.mark.parametrize("source", ["blind", "recovered", "oracle"])
def test_hf_term_works_on_every_rung_of_the_ladder(source: str) -> None:
    """The band PARTITION is defined by the decimation, not by the marker, so
    the detail term must not require a fiducial. Coupling the two would have
    denied the blind and oracle rungs the high-frequency loss and confounded
    the ladder with an objective difference."""
    st = _hf_strategy(0.5, shift_source=source)
    loss, _p, _f = st._field_loss(_disk_kspace(b=2, h=64, w=64))
    assert st._last_hf_loss is not None and torch.isfinite(loss)


def test_smoothness_term_is_reportable_but_off_by_configuration() -> None:
    """lambda_smooth remains available (the field-mapping methods need it) and
    the SR arms set it to 0; this pins that the strategy honours 0.0 rather
    than falling back to its old hardcoded 0.01."""
    st = _hf_strategy(0.0)
    assert st._lambda_smooth == 0.0
    pred = torch.randn(2, 1, 16, 16)
    assert float(st._spatial_smoothness(pred)) > 0.0  # the term still computes
