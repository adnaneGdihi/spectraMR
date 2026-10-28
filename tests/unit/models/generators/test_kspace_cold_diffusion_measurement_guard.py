"""Declared physics must not silently no-op (experiment_11 collapse triage).

``KSpaceColdDiffusionGenerator`` gates BOTH data consistency and the output
magnitude bound on ``kwargs["kspace_measured"]``. Both used to skip in silence
when it was missing or batch-mismatched, so ``dc_method``,
``physics.data_consistency.enabled`` and ``output_kspace_clip_ratio`` could all
be advertised in the YAML, accepted by the audit and stamped into provenance
while the forward returned the raw backbone proposal (pitfall #16).

With the measurement present an untrained net is pinned to the band-local
ceiling; without it the output runs free, which for a log-scaling arm is then
fed to an ``expm1`` at the k-space -> image boundary.

Deliberately NOT claimed here: that this is what blacked out the experiment_11
validation images. An out-of-range but finite prediction still renders --
``MetricsTracker._normalize_images`` uses percentile windowing, and a prediction
with up to 10% of its bins past ``DECOMPRESS_MAGNITUDE_CEILING`` saves as an
ordinary full-range PNG. A bit-exact black image needs an all-zero, constant, or
NaN prediction. These tests pin the mechanism they can actually demonstrate.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.diffusion.kspace_process import band_local_magnitude_ceiling
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY

# The cohort's shape, shrunk: 4 complex coils, the two knobs under test, and a
# backbone small enough to instantiate in a unit test.
_MODEL_KWARGS = {
    "base_channels": 8,
    "num_res_blocks": 1,
    "timesteps": 28,
    "sampling_steps": 28,
    "time_embedding_dim": 32,
    "num_physical_coils": 4,
    "use_complex_conv": True,
    "activation": "complex",
    "force_pure_kspace": True,
    "backbone_type": "complex_unet",
    "attention_type": "none",
    "process_type": "cold_diffusion",
    "dc_method": "hard",
    "dc_weight": 0.5,
    "output_kspace_clip_ratio": 1.3,
    "output_kspace_clip_reference": "band_local",
    # These fixtures build PHYSICAL (un-log1p'd) synthetic k-space, so the
    # ceiling is a plain ratio. Declared rather than defaulted because the
    # generator now refuses to guess the domain (issue #1281).
    "kspace_log_scaled": False,
    "kspace_feature_norm": "rms",
}


@pytest.fixture(scope="module")
def generator():
    # Seeded before construction, which is load-bearing rather than tidiness:
    # `test_the_bound_actually_holds_when_the_measurement_is_supplied` asserts the
    # output bound is INDEPENDENT of the timestep, and with unseeded weights that
    # comparison occasionally failed (observed once at |k|max 5.008 vs a differing
    # peak) purely because a different random net binds the band-local ceiling
    # differently at t=6 and t=27. An unseeded fixture under a determinism
    # assertion is a latent flake, and one that reads as a real regression.
    populate_model_registry()
    torch.manual_seed(0)
    cls = MODEL_REGISTRY["kspace_cold_diffusion"]["class"]
    return cls(in_channels=8, out_channels=8, **_MODEL_KWARGS)


def _measurement(batch: int = 2, size: int = 64):
    """Compressed-domain k-space (peak ~4.7) and a phase-encode mask."""
    torch.manual_seed(0)
    k = torch.randn(batch, 8, size, size) * 0.09
    c = size // 2
    k[:, :, c - 4 : c + 4, c - 4 : c + 4] += 4.0
    mask = torch.zeros(batch, 1, size, size)
    mask[:, :, :, c - 3 : c + 3] = 1.0
    mask[:, :, :, ::8] = 1.0
    return k * mask, mask


def test_training_forward_raises_when_the_measurement_is_absent(generator) -> None:
    """The arm declares hard DC and a clip; neither can run without it."""
    measured, mask = _measurement()
    generator.train()
    with pytest.raises(ValueError, match="kspace_measured"):
        generator(measured, torch.tensor([6, 6]), mask=mask)


def test_batch_mismatch_warns_instead_of_raising(generator, caplog) -> None:
    """A mismatch has a legitimate cause (D>1), so it must warn, not raise.

    Skipping the ceiling for a genuine 5-D batch was a priced decision that
    predates this guard; raising would break every D>1 arm to fix a D=1 cohort's
    problem. Only the silence was the defect.
    """
    measured, mask = _measurement(batch=2)
    generator.train()
    generator._warned_measurement_batch = False
    with caplog.at_level("WARNING"):
        generator(
            measured, torch.tensor([6, 6]), mask=mask, kspace_measured=measured[:1]
        )
    assert "cannot broadcast" in caplog.text


def test_the_batch_mismatch_warning_is_emitted_once_per_instance(
    generator, caplog
) -> None:
    """30,000 identical warnings would bury the log this exists to make readable."""
    measured, mask = _measurement(batch=2)
    generator.train()
    generator._warned_measurement_batch = False
    with caplog.at_level("WARNING"):
        for _ in range(3):
            generator(
                measured, torch.tensor([6, 6]), mask=mask, kspace_measured=measured[:1]
            )
    assert caplog.text.count("cannot broadcast") == 1


def test_the_bound_actually_holds_when_the_measurement_is_supplied(generator) -> None:
    """The guard is about absence; with the measurement the clip is exact.

    Asserted at two widely separated timesteps because that invariance is what
    made the experiment_11 snapshot readable: the measurement-present path is
    pinned to the ceiling regardless of ``t``, so a recorded ``|k|max`` far above
    it cannot be explained by "a harder timestep".
    """
    measured, mask = _measurement()
    ceiling = float(band_local_magnitude_ceiling(measured, 1.3, log_scaled=False).max())
    generator.train()
    peaks = []
    for t in (6, 27):
        with torch.no_grad():
            out = generator(
                measured, torch.tensor([t, t]), mask=mask, kspace_measured=measured
            )
        x_out = out[0] if isinstance(out, tuple) else out
        peaks.append(float(x_out.abs().max()))
        assert peaks[-1] <= ceiling * 1.01, (
            f"t={t}: output |k|max {peaks[-1]} exceeds the band-local ceiling "
            f"{ceiling}; the output magnitude bound is not being applied"
        )
    assert peaks[0] == pytest.approx(peaks[1], rel=1e-3), (
        f"the bound should not depend on the timestep, got {peaks}"
    )


def test_eval_forward_is_unaffected(generator) -> None:
    """The reverse sampler runs without a measurement and applies its own bound."""
    measured, mask = _measurement()
    generator.eval()
    with torch.no_grad():
        out = generator(measured, torch.tensor([6, 6]), mask=mask)
    x_out = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(x_out).all()


# --------------------------------------------------------------------------
# The probe contract: the guard above made every such arm false-fail Tier 2
# --------------------------------------------------------------------------
#
# ``synthetic_forward_probe`` builds its forward call from
# ``inspect.signature(model.forward)``. Both kwargs the guard requires arrive
# through ``**kwargs``, which a signature cannot enumerate, so the probe called
# ``model(x)`` and tripped the raise above -- reporting a healthy model as a
# Tier-2 ``error`` whose ``fix_hint`` blamed ``patch_size``. The fix is a
# model-side declaration, NOT suppressing the guard for probes: suppression
# would have made the probe grade a forward with DC and the output bound
# switched off, which is the facade this whole file exists to prevent.


def test_the_probe_hook_declares_both_kwargs_the_guard_requires(generator) -> None:
    """Sensitivity pair, half 1: this arm declares DC + a clip, so both are due."""
    measured, _ = _measurement()
    kwargs = generator.synthetic_forward_probe_kwargs(measured)
    assert set(kwargs) == {"kspace_measured", "mask"}
    assert kwargs["kspace_measured"].shape == measured.shape
    assert kwargs["mask"].shape == (measured.shape[0], 1, *measured.shape[-2:])


def test_the_probe_hook_is_empty_when_no_mechanism_is_declared() -> None:
    """Sensitivity pair, half 2: an arm with neither knob is probed as before.

    Without this the hook would inject kwargs into every k-space cold-diffusion
    forward, changing what unaffected arms are graded on. The declaration has to
    track the constructed instance, which is why it is a method and not a
    ``ClassVar`` beside ``synthetic_forward_probe_skip``.

    Note ``dc_method`` must be set to ``"none"`` EXPLICITLY, not omitted: it
    defaults to ``"hard"`` (``kspace_cold_diffusion_generator.py:2324``), so an
    arm that never mentions data consistency still builds a ``HardDataConsistency``
    layer and is therefore correctly in scope for the hook.
    """
    populate_model_registry()
    cls = MODEL_REGISTRY["kspace_cold_diffusion"]["class"]
    bare_kwargs = {
        k: v
        for k, v in _MODEL_KWARGS.items()
        if k not in ("output_kspace_clip_ratio", "output_kspace_clip_reference")
    }
    bare_kwargs["dc_method"] = "none"
    bare = cls(in_channels=8, out_channels=8, **bare_kwargs)
    assert bare.dc_layer is None, "dc_method='none' must disable model-internal DC"
    assert bare._output_kspace_clip_ratio is None
    assert bare._dc_passthrough_size is None
    # No measurement: only the mask the v2 geometry planes are built from.
    assert set(bare.synthetic_forward_probe_kwargs(torch.randn(2, 8, 64, 64))) == {"mask"}


def test_the_probe_mask_is_sparse_so_dc_is_a_constraint_not_a_copy(generator) -> None:
    """An all-ones mask would make hard DC overwrite the whole prediction.

    The probe would then grade a copy of its own input -- the identity path,
    which is exactly what a probe must not be allowed to pass on.
    """
    measured, _ = _measurement()
    mask = generator.synthetic_forward_probe_kwargs(measured)["mask"]
    fraction = float(mask.mean())
    assert 0.0 < fraction < 0.75, (
        f"probe mask samples {fraction:.3f} of k-space; at ~1.0 hard DC becomes "
        "a full overwrite and the probe grades its own input"
    )


def test_the_probe_mask_is_deterministic(generator) -> None:
    """The probe re-runs the forward to check determinism; an RNG mask breaks it."""
    measured, _ = _measurement()
    first = generator.synthetic_forward_probe_kwargs(measured)["mask"]
    second = generator.synthetic_forward_probe_kwargs(measured)["mask"]
    assert torch.equal(first, second)


def test_the_hook_kwargs_actually_satisfy_the_guard(generator) -> None:
    """End to end: the declared contract is the one the guard demands."""
    measured, _ = _measurement()
    generator.train()
    with torch.no_grad():
        out = generator(
            measured,
            torch.tensor([14, 14]),
            **generator.synthetic_forward_probe_kwargs(measured),
        )
    x_out = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(x_out).all()


def test_the_hook_makes_the_declared_mechanisms_change_the_output(generator) -> None:
    """A satisfied guard is not the same as a fired mechanism.

    The guard only checks that the kwargs are present. This asserts the stronger
    property the probe needs: running WITH the hook's kwargs produces a different
    tensor than running without them, i.e. data consistency and the output bound
    are on the path rather than merely un-complained-about.
    """
    measured, _ = _measurement()
    generator.train()
    hook_kwargs = generator.synthetic_forward_probe_kwargs(measured)
    original = generator._assert_measurement_reaches_declared_mechanisms
    with torch.no_grad():
        with_dc = generator(measured, torch.tensor([14, 14]), **hook_kwargs)
        # Bypass the GUARD (not the mechanisms) to obtain the unconstrained path.
        generator._assert_measurement_reaches_declared_mechanisms = (
            lambda *a, **k: None
        )
        try:
            without = generator(measured, torch.tensor([14, 14]))
        finally:
            generator._assert_measurement_reaches_declared_mechanisms = original
    a = with_dc[0] if isinstance(with_dc, tuple) else with_dc
    b = without[0] if isinstance(without, tuple) else without
    assert not torch.allclose(a, b), (
        "the forward produced an identical tensor with and without the measured "
        "k-space -- the declared mechanisms are not on the path"
    )


def test_declared_dc_without_a_mask_raises_rather_than_skipping(generator) -> None:
    """The facade one level behind the measurement check.

    The data-consistency block is gated on ``mask is not None`` as well as the
    measurement, so a forward carrying ``kspace_measured`` and no ``mask`` used
    to pass the guard and still skip DC in silence.
    """
    measured, _ = _measurement()
    generator.train()
    with pytest.raises(ValueError, match="no `mask`"):
        generator(measured, torch.tensor([14, 14]), kspace_measured=measured)


def test_the_output_clip_alone_does_not_require_a_mask() -> None:
    """Scope control: only the mechanisms that READ the mask may demand it.

    The output magnitude bound is per-sample off ``max|measured|`` and never
    touches the mask, so an arm declaring the clip without DC must not be
    reddened by the check above.
    """
    populate_model_registry()
    cls = MODEL_REGISTRY["kspace_cold_diffusion"]["class"]
    clip_only = dict(_MODEL_KWARGS)
    clip_only["dc_method"] = "none"
    model = cls(in_channels=8, out_channels=8, **clip_only)
    if model.dc_layer is not None:
        pytest.skip("dc_method='none' still built a DC layer; not this subject")
    assert model._output_kspace_clip_ratio is not None
    measured, _ = _measurement()
    model.train()
    with torch.no_grad():
        out = model(measured, torch.tensor([14, 14]), kspace_measured=measured)
    x_out = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(x_out).all()
