"""Hard data consistency pins to the measurement, and to nothing else.

``HardDataConsistency`` added Gaussian noise to ``kspace_obs`` before pinning it,
at levels no arm ever declared: 0 of the corpus sets either knob, so all 68
``dc_method: hard`` arms inherited 0.01 / 0.005 from a module default.

Three things made that a defect rather than an augmentation:

* **The level is absolute and the k-space is log1p-compressed**, so one number is
  a different corruption in every annulus -- 0.9% of the mean ``|k|`` at DC and
  23.2% in the outer band at eval, 46.5% there at train.
* **It randomises phase on the observed bins** -- up to 23.9 degrees at the outer
  band during training -- which is exactly the support hard DC exists to hold,
  and which the model therefore cannot correct.
* **The reverse loop never did it.** ``_apply_observed_dc`` is
  ``x0*(1-obs) + measurement*obs``, so training saw corrupted measurements and
  the validation that reports the numbers saw clean ones.

Electing the reverse loop's semantics (non-negotiable 17) makes them one
mechanism. The knobs stay read so an arm can still opt in.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.data_consistency import (  # noqa: E402
    HardDataConsistency,
)


def _batch(size=32):
    torch.manual_seed(0)
    pred = torch.randn(1, 2, size, size, dtype=torch.complex64)
    measured = torch.randn(1, 2, size, size, dtype=torch.complex64)
    mask = torch.zeros(1, 1, size, size)
    mask[..., ::2, :] = 1.0
    return pred, measured, mask


@pytest.mark.parametrize("training", [True, False])
def test_observed_bins_come_back_bit_exact(training):
    """The whole contract, in one assertion, on both paths."""
    layer = HardDataConsistency()
    layer.train(training)
    pred, measured, mask = _batch()
    out = layer(pred, measured, mask, is_kspace_domain=True)
    obs = mask.bool().expand_as(out)
    assert torch.equal(out[obs], measured[obs])


@pytest.mark.parametrize("training", [True, False])
def test_the_layer_is_deterministic(training):
    """`sampler_sigma: 0.0` is declared by every cohort arm (#1689)."""
    layer = HardDataConsistency()
    layer.train(training)
    pred, measured, mask = _batch()
    a = layer(pred, measured, mask, is_kspace_domain=True)
    b = layer(pred, measured, mask, is_kspace_domain=True)
    assert torch.equal(a, b)


def test_unobserved_bins_keep_the_prediction():
    """Guards the pin tests from passing on a layer that overwrites everything."""
    layer = HardDataConsistency().eval()
    pred, measured, mask = _batch()
    out = layer(pred, measured, mask, is_kspace_domain=True)
    free = ~mask.bool().expand_as(out)
    assert torch.equal(out[free], pred[free])


def test_the_default_is_no_noise_on_either_level():
    layer = HardDataConsistency()
    assert layer.train_noise_level == 0.0
    assert layer.eval_noise_level == 0.0


def test_a_declared_level_is_still_honoured():
    """The knob stays wired: opting in must still work, or this is pitfall 15."""
    layer = HardDataConsistency(train_noise_level=0.5, eval_noise_level=0.25)
    layer.train()
    pred, measured, mask = _batch()
    out = layer(pred, measured, mask, is_kspace_domain=True)
    obs = mask.bool().expand_as(out)
    assert not torch.equal(out[obs], measured[obs])


def test_the_generator_default_is_also_no_noise():
    """The 68 `dc_method: hard` arms reach the generator's default, not the class's."""
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceColdDiffusionGenerator,
    )

    torch.manual_seed(0)
    gen = KSpaceColdDiffusionGenerator(
        in_channels=4,
        out_channels=4,
        features=(8, 16),
        force_pure_kspace=True,
        attention_type="none",
        dc_method="hard",
        kspace_log_scaled=False,
        condition_with_smaps=False,
    )
    assert gen.dc_train_noise_level == 0.0
    assert gen.dc_eval_noise_level == 0.0
    assert gen.dc_layer.train_noise_level == 0.0
    assert gen.dc_layer.eval_noise_level == 0.0


def test_training_and_the_reverse_loop_now_agree():
    """`_apply_observed_dc`'s formula, run against the layer training uses.

    These were two owners of one invariant: the reverse loop pinned cleanly and
    the forward path pinned a noised copy, so the model trained against a
    corruption that was absent wherever the numbers came from.
    """
    layer = HardDataConsistency()
    layer.train()
    pred, measured, mask = _batch()
    forward_path = layer(pred, measured, mask, is_kspace_domain=True)
    reverse_path = pred * (1.0 - mask) + measured * mask
    assert torch.allclose(forward_path, reverse_path, atol=1e-6)


def test_noise_on_log_compressed_kspace_is_band_dependent():
    """Why an absolute level could not have been the intended augmentation.

    Pins the asymmetry rather than a specific number: the same std is a far
    larger fraction of the outer band than of DC, so one knob cannot mean one
    thing. A relative level is the shape a real augmentation would need.
    """
    from spectramr.data.transforms.normalization import compress_kspace_log
    from spectramr.infrastructure.physics.fft_ops import fft2c
    from spectramr.infrastructure.physics.radial_bands import (
        band_counts,
        band_reduce,
        radial_bins,
    )

    size = 128
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    img = (((yy / 0.75) ** 2 + (xx / 0.6) ** 2) <= 1).float()
    kspace = compress_kspace_log(fft2c(img.to(torch.complex64)[None, None]))

    index, inside, _edges = radial_bins(size, size, 8, "cpu")
    counts = band_counts(index, inside, 8)
    amplitude = band_reduce(kspace[0, 0].abs(), index, inside, 8) / counts.clamp_min(1)

    dc_band, outer_band = float(amplitude[0]), float(amplitude[-1])
    assert dc_band > 5 * outer_band, (
        "log-compressed k-space no longer has the radial falloff that makes an "
        "absolute noise level band-dependent; re-read this test's premise"
    )


# ── the layer nobody thinks of: the schema default the resolver forwards ──────
def test_the_resolver_forwards_no_noise():
    """The owner that actually reaches the arms, and the one that hid a facade.

    `resolve_generator_kwargs` step 3c forwards every `DC_SSOT_KEYS` entry from
    `physics.data_consistency` into the constructor kwargs — **including fields
    the arm never declared**, because a Pydantic default is indistinguishable
    from a declaration once the model is built. So `kwargs.get(name, 0.0)` in the
    generator never sees its own default when the block exists, which it does on
    every cohort arm.

    Fixing the generator and the class alone left this untouched and the change
    inert. Measured before the schema moved: the resolver handed back
    {'train_noise_level': 0.01, 'eval_noise_level': 0.005}.
    """
    from types import SimpleNamespace

    from spectramr.config.schemas.physics import DataConsistencyConfig
    from spectramr.infrastructure.builders.generator_kwargs import (
        resolve_generator_kwargs,
    )
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceColdDiffusionGenerator,
    )

    config = SimpleNamespace(
        model=SimpleNamespace(model_type="kspace_cold_diffusion", model_kwargs={}),
        physics=SimpleNamespace(data_consistency=DataConsistencyConfig()),
    )
    resolved = resolve_generator_kwargs(config, model_cls=KSpaceColdDiffusionGenerator)
    assert resolved["train_noise_level"] == 0.0
    assert resolved["eval_noise_level"] == 0.0


def test_a_declared_level_still_survives_the_resolver():
    """Opting in must work through the same seam, or the knob is decorative."""
    from types import SimpleNamespace

    from spectramr.config.schemas.physics import DataConsistencyConfig
    from spectramr.infrastructure.builders.generator_kwargs import (
        resolve_generator_kwargs,
    )
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceColdDiffusionGenerator,
    )

    config = SimpleNamespace(
        model=SimpleNamespace(model_type="kspace_cold_diffusion", model_kwargs={}),
        physics=SimpleNamespace(data_consistency=DataConsistencyConfig(train_noise_level=0.03)),
    )
    resolved = resolve_generator_kwargs(config, model_cls=KSpaceColdDiffusionGenerator)
    assert resolved["train_noise_level"] == 0.03


def test_the_three_defaults_agree():
    """Three owners of one value; aligned here so a drift shows up as a failure.

    The schema, the generator's `kwargs.get` fallback and the layer's own
    signature each carry this default, and they disagreed: consolidating them is
    a larger change than this, so the ratchet is that they must at least match.
    """
    import inspect

    from spectramr.config.schemas.physics import DataConsistencyConfig

    schema = DataConsistencyConfig()
    layer_defaults = inspect.signature(HardDataConsistency.__init__).parameters
    assert schema.train_noise_level == 0.0
    assert schema.eval_noise_level == 0.0
    assert layer_defaults["train_noise_level"].default == 0.0
    assert layer_defaults["eval_noise_level"].default == 0.0


# ── DC must not manufacture a degenerate reconstruction ──────────────────────
def _phantom_kspace(size=64):
    """Interleaved, log-compressed k-space of a two-ellipse phantom."""
    from spectramr.data.transforms.normalization import compress_kspace_log
    from spectramr.infrastructure.physics.fft_ops import fft2c

    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    img = (((yy / 0.75) ** 2 + (xx / 0.6) ** 2) <= 1).float()
    img = img + 0.5 * (((yy / 0.3) ** 2 + ((xx - 0.2) / 0.2) ** 2) <= 1).float()
    kspace = compress_kspace_log(
        fft2c((img / img.max()).to(torch.complex64)[None, None].repeat(1, 2, 1, 1))
    )
    return torch.stack([kspace.real, kspace.imag], 2).flatten(1, 2)


def _accelerated_mask(size=64):
    """R~8 with an ACS block, the shape the cohort actually runs."""
    mask = torch.zeros(1, 1, size, size)
    mask[..., ::8, :] = 1.0
    mask[..., size // 2 - 3 : size // 2 + 3, :] = 1.0
    return mask


def _image_relative_std(interleaved):
    """Image-space std / mean. The statistic that names a degenerate output.

    The "DC blob" is a reconstruction that has collapsed to a DC-dominated
    constant, and a constant image has relative std **exactly 0**. Measured on
    this phantom: 0.931 for the full k-space, 0.895 for the zero-filled R~8
    acquisition, 0.820 for an ACS-block-only reconstruction (blurry but real),
    and 0.000 for a DC-bin-only one.

    A centre-patch-over-mean ratio does NOT work here and was the first thing
    tried: a DC-dominated k-space inverse-transforms to a FLATTER image, not a
    centre-brighter one, so that ratio moves toward 1.0 rather than up and a
    blob slips through. The planted-violation test below is what caught it.
    """
    from spectramr.infrastructure.physics.fft_ops import ifft2c

    complex_k = torch.complex(interleaved[:, 0::2], interleaved[:, 1::2])
    image = ifft2c(complex_k)[0, 0].abs()
    return float(image.std() / image.mean().clamp_min(1e-12))


def _dc_generator():
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceColdDiffusionGenerator,
    )

    torch.manual_seed(0)
    return KSpaceColdDiffusionGenerator(
        in_channels=4,
        out_channels=4,
        features=(8, 16),
        force_pure_kspace=True,
        attention_type="none",
        dc_method="hard",
        kspace_log_scaled=False,
        condition_with_smaps=False,
    )


def _forward(gen, x, mask, measured):
    out = gen(x, torch.zeros(1, dtype=torch.long), mask=mask, kspace_measured=measured)
    return out[0] if isinstance(out, tuple) else out


def test_dc_does_not_collapse_the_image_to_a_constant():
    """The failure this cohort names "DC blob" / "white spot"."""
    gen = _dc_generator().train()
    kspace, mask = _phantom_kspace(), _accelerated_mask()
    zero_filled = kspace * mask
    with torch.no_grad():
        out = _forward(gen, zero_filled, mask, zero_filled)
    before, after = _image_relative_std(zero_filled), _image_relative_std(out)
    assert after > 0.5 * before, (
        f"data consistency flattened the reconstruction toward a constant: "
        f"relative std {before:.3f} -> {after:.3f}"
    )


def test_the_collapse_check_can_actually_fail():
    """A DC-dominated k-space must trip the check the test above relies on.

    Without this, the previous metric passed a planted blob and the guard above
    would have been decorative (non-negotiable 15).
    """
    size = 64
    kspace = _phantom_kspace(size)
    dc_only = torch.zeros_like(kspace)
    centre = slice(size // 2, size // 2 + 1)
    dc_only[..., centre, centre] = kspace[..., centre, centre]
    healthy = _image_relative_std(kspace * _accelerated_mask(size))
    assert _image_relative_std(dc_only) < 0.5 * healthy


def test_the_output_is_not_merely_the_zero_filled_input():
    """Hard DC pins the measurement; the null space must still carry the model.

    With the noise gone this is exact rather than approximate, so a model that
    contributed nothing would now be *bit-identical* to its input instead of
    being hidden behind a perturbation.
    """
    gen = _dc_generator().train()
    kspace, mask = _phantom_kspace(), _accelerated_mask()
    zero_filled = kspace * mask
    with torch.no_grad():
        out = _forward(gen, zero_filled, mask, zero_filled)
    assert not torch.allclose(out, zero_filled, atol=1e-4)


def test_the_output_responds_to_its_input():
    """Measurement-independence, at a single rung.

    `_apply_input_dependence_gate` catches the cross-rung form -- an output that
    is the same whatever the acceleration. It cannot see this one, because an
    output that merely tracks its own zero-filled input differs across rungs and
    reads as healthy spread.
    """
    gen = _dc_generator().train()
    kspace, mask = _phantom_kspace(), _accelerated_mask()
    zero_filled = kspace * mask
    torch.manual_seed(5)
    perturbed = zero_filled + 0.2 * torch.randn_like(zero_filled)
    with torch.no_grad():
        a = _forward(gen, zero_filled, mask, zero_filled)
        b = _forward(gen, perturbed, mask, zero_filled)
    assert not torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("training", [True, False])
def test_two_forwards_on_one_input_agree(training):
    """Training mode is the half that was non-deterministic before this change."""
    gen = _dc_generator().train(training)
    kspace, mask = _phantom_kspace(), _accelerated_mask()
    zero_filled = kspace * mask
    with torch.no_grad():
        a = _forward(gen, zero_filled, mask, zero_filled)
        b = _forward(gen, zero_filled, mask, zero_filled)
    assert torch.allclose(a, b, atol=1e-6)


def test_the_output_is_finite():
    gen = _dc_generator().train()
    kspace, mask = _phantom_kspace(), _accelerated_mask()
    with torch.no_grad():
        out = _forward(gen, kspace * mask, mask, kspace * mask)
    assert bool(torch.isfinite(out).all())
