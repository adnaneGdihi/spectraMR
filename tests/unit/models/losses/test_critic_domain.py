"""The critic seam converts domains because BOTH sides declared one (#1920).

Three failure shapes are planted here rather than described, because each one
produces a finite, correctly-shaped tensor and none of them raises:

1. **converting the wrong way** -- an ``fft2c`` where an ``ifft2c`` belonged;
2. **realifying before converting** -- ``cat([real, imag])`` destroys the
   complex structure the FFT needs, so the transform runs on channels that no
   longer mean real and imaginary parts;
3. **pairing channels as a block** -- reading ``[R1..Rn,I1..In]`` out of a
   tensor the generator wrote as ``[R1,I1,R2,I2,...]``.

Each test computes the wrong answer inline and asserts the implementation
differs from it. That is what stops the assertions being vacuous: a test that
only checked shape, finiteness, or "it changed" would pass on all three.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c  # noqa: E402
from spectramr.models.losses.critic_domain import (  # noqa: E402
    DomainAdaptedCritic,
    critic_component_name,
    resolve_conversion,
    to_critic_input,
)

#: 4 coils written interleaved, the layout the generator emits.
_COILS = 4


def _fake() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 2 * _COILS, 16, 16)


def _hand_complex(x: torch.Tensor) -> torch.Tensor:
    """Pair channels INTERLEAVED, by hand.

    Deliberately not ``fft_ops._to_complex``: using the implementation's own
    helper to build the expected value makes the assertion vacuous whichever
    pairing it uses.
    """
    return torch.complex(x[:, 0::2], x[:, 1::2])


# --------------------------------------------------------------------------
# resolve_conversion: when a transform is owed at all
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("generator", "critic", "expected"),
    [
        ("kspace", "image", "image"),
        ("image", "kspace", "kspace"),
        ("kspace", "kspace", None),
        ("image", "image", None),
        # image and complex_image differ in REPRESENTATION, not domain: crossing
        # between them must never insert a Fourier transform.
        ("complex_image", "image", None),
        ("image", "complex_image", None),
        # Undeclared on either side -> nothing converts. This is the path all 18
        # critics that declare no domain take, and all 58 kspace_filling arms.
        (None, "image", None),
        ("kspace", None, None),
        # A tuple that already contains the generator's domain means the critic
        # is happy as it is.
        ("kspace", ("image", "kspace"), None),
        ("kspace", ("image",), "image"),
    ],
)
def test_resolve_conversion_truth_table(generator, critic, expected):
    assert resolve_conversion(generator, critic) == expected


@pytest.mark.parametrize("pair", [("latent", "image"), ("kspace", "mesh"), ("spectrum", "kspace")])
def test_a_domain_with_no_fourier_relationship_raises(pair):
    """RAISE, never convert: there is no transform between these, so a
    conversion would invent data. Mirrors the bridge matrix in
    ``LossBuilder._build_list_based_losses``, which also raises here.
    """
    with pytest.raises(ValueError, match="no Fourier relationship"):
        resolve_conversion(*pair)


# --------------------------------------------------------------------------
# to_critic_input: the transform itself
# --------------------------------------------------------------------------


def test_kspace_generator_to_image_critic_applies_the_inverse_transform():
    """PLANTED SHAPE 1 -- the wrong direction is finite and correctly shaped."""
    fake = _fake()
    got = to_critic_input(fake, from_domain="kspace", to_domain="image")

    expected = ifft2c(_hand_complex(fake))
    expected = torch.cat([expected.real, expected.imag], dim=1)
    assert torch.allclose(got, expected, atol=1e-6)

    wrong_direction = fft2c(_hand_complex(fake))
    wrong_direction = torch.cat([wrong_direction.real, wrong_direction.imag], dim=1)
    assert wrong_direction.shape == got.shape, "the wrong direction is not caught by shape"
    assert torch.isfinite(wrong_direction).all(), "nor by finiteness"
    assert not torch.allclose(got, wrong_direction, atol=1e-6)


def test_image_generator_to_kspace_critic_applies_the_forward_transform():
    fake = _fake()
    got = to_critic_input(fake, from_domain="image", to_domain="kspace")
    expected = fft2c(_hand_complex(fake))
    expected = torch.cat([expected.real, expected.imag], dim=1)
    assert torch.allclose(got, expected, atol=1e-6)


def test_the_domain_is_converted_before_the_tensor_is_realified():
    """PLANTED SHAPE 2 -- and it takes a COMPLEX input to see it.

    ``cat([real, imag])`` doubles the channel axis, so an FFT applied afterwards
    re-pairs channels that are no longer a real/imag interleaving: channel 0 of
    the block half gets paired with channel 1 of the same half. The result is
    the same shape and finite.

    The input MUST be complex. An earlier version of this test drove the
    already-real ``fake`` through, where ``torch.is_complex`` is False and
    realifying first is a no-op -- the whole suite stayed green with the two
    steps transposed in ``to_critic_input``. That is why the mutation is
    reproduced literally here rather than approximated.
    """
    torch.manual_seed(1)
    x = torch.randn(2, _COILS, 16, 16, dtype=torch.complex64)

    got = to_critic_input(x, from_domain="kspace", to_domain="image")
    expected = ifft2c(x)
    expected = torch.cat([expected.real, expected.imag], dim=1)
    assert torch.allclose(got, expected, atol=1e-6)

    # Exactly what the transposed implementation computes.
    realified_first = torch.cat([x.real, x.imag], dim=1)
    realified_first = ifft2c(realified_first)
    realified_first = torch.cat([realified_first.real, realified_first.imag], dim=1)
    assert realified_first.shape == got.shape, "the transposition is not caught by shape"
    assert torch.isfinite(realified_first).all(), "nor by finiteness"
    assert not torch.allclose(got, realified_first, atol=1e-6)


def test_channels_are_paired_interleaved_not_as_a_block():
    """PLANTED SHAPE 3 -- block pairing gives a plausible, wrong image."""
    fake = _fake()
    got = to_critic_input(fake, from_domain="kspace", to_domain="image")

    half = fake.shape[1] // 2
    block_paired = torch.complex(fake[:, :half], fake[:, half:])
    block_paired = ifft2c(block_paired)
    block_paired = torch.cat([block_paired.real, block_paired.imag], dim=1)
    assert block_paired.shape == got.shape
    assert torch.isfinite(block_paired).all()
    assert not torch.allclose(got, block_paired, atol=1e-6)


def test_a_critic_that_accepts_complex_still_gets_its_domain_converted():
    """``accepts_complex`` is about REPRESENTATION; the domain is separate.

    The critic owns the realification, not the transform -- it declared which
    domain it reads, and honouring one declaration while ignoring the other
    would hand it complex data from the wrong space.
    """
    fake = _fake()
    got = to_critic_input(fake, from_domain="kspace", to_domain="image", takes_complex=True)
    assert torch.is_complex(got)
    assert torch.allclose(got, ifft2c(_hand_complex(fake)), atol=1e-6)


def test_no_declaration_returns_the_very_same_object():
    """REGRESSION: identity, not just equality.

    58 kspace_filling arms and 18 of 19 registered critics take this path, and
    ``test_adversarial_diffusion`` asserts ``out_f is fake``.
    """
    fake = _fake()
    assert to_critic_input(fake) is fake
    assert to_critic_input(fake, from_domain="kspace", to_domain="kspace") is fake
    assert to_critic_input(fake, from_domain="kspace", to_domain=None) is fake


# --------------------------------------------------------------------------
# The two feed points must not diverge
# --------------------------------------------------------------------------


def test_the_g_step_adapter_hands_the_critic_what_the_d_step_hands_it():
    """The invariant the whole seam exists to hold (#1920).

    The critic is fed twice per step from two places: ``_align_for_critic`` on
    the D step, and ``discriminator(pred)`` inside
    ``UnifiedDiffusionLossComputer.compute`` on the G step. Converting on one
    and not the other is worse than converting on neither -- the critic would
    score images while training and k-space while scoring the generator.
    """
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    fake = _fake()
    real = torch.randn(2, _COILS, 16, 16, dtype=torch.complex64)

    d_fake, d_real = DiffusionTrainingStrategy._align_for_critic(
        fake, real, False, generator_domain="kspace", critic_domain="image"
    )

    seen: dict[str, torch.Tensor] = {}

    def spy(x: torch.Tensor) -> torch.Tensor:
        seen["x"] = x
        return x.mean()

    DomainAdaptedCritic(spy, from_domain="kspace", to_domain="image", takes_complex=False)(fake)

    assert torch.equal(d_fake, seen["x"]), "the D and G steps disagree about the critic's input"
    assert d_fake.shape[1] == d_real.shape[1], (
        "converted sides must share a layout: both pass through complex and come "
        "out block, which is what makes them comparable"
    )


def test_converting_for_the_critic_leaves_the_loss_terms_in_the_declared_domain():
    """The critic's domain is the CRITIC's; the loss terms keep the generator's.

    ``losses.policy.output_domain`` decides which of ``kspace_losses`` /
    ``image_losses`` / ``complex_losses`` needs a Fourier bridge
    (``LossBuilder._build_list_based_losses``), and those bridges are built for
    the tensor the generator emits. The critic seam must therefore convert at
    the critic BOUNDARY only -- the same ``pred`` goes on to every declared loss
    unchanged, in the domain that arm declared.

    This is why the G step wraps the critic instead of converting ``pred``
    before ``UnifiedDiffusionLossComputer.compute``: converting the tensor would
    silently move every reconstruction term into the critic's domain, which is a
    different arm than the one the user configured.

    Pinned against an in-place "optimization", which is the one change that
    would break it without changing any call site.
    """
    x = torch.randn(2, _COILS, 16, 16, dtype=torch.complex64)
    reference = x.clone()

    converted = to_critic_input(x, from_domain="kspace", to_domain="image")

    assert torch.equal(x, reference), "the tensor the loss terms see was mutated"
    assert converted is not x
    assert not torch.allclose(
        converted, torch.cat([reference.real, reference.imag], dim=1), atol=1e-6
    ), "nothing was converted, so this test proves nothing"


def test_the_adapter_does_not_mutate_what_the_g_step_hands_it():
    """Same invariant at the G step's own call site."""
    x = torch.randn(2, _COILS, 16, 16, dtype=torch.complex64)
    reference = x.clone()

    DomainAdaptedCritic(
        lambda t: t.real.mean(), from_domain="kspace", to_domain="image", takes_complex=False
    )(x)

    assert torch.equal(x, reference)


# ---------------------------------------------------------------------------
# The elected owner (#1921, non-negotiable 17), exercised against a real schema.
# ---------------------------------------------------------------------------


def test_a_real_schema_instance_round_trips_through_the_owner():
    """The owner is exercised against a constructed schema, not a stand-in.

    A ``SimpleNamespace`` shaped like a config agrees with the reader by
    construction, so it cannot catch the field being RENAMED or RE-NESTED --
    both sides would move together. Building the real
    ``ModelConfigSchema``/``ModelComponentSchema`` is what makes this test able
    to fail: if ``discriminator_component`` is renamed or moves under another
    block, the construction below stops producing a readable critic name.
    """
    from spectramr.config.schemas.model import ModelComponentSchema, ModelConfigSchema

    model = ModelConfigSchema(
        discriminator_component=ModelComponentSchema(name="patch_gan"),
    )
    config = types.SimpleNamespace(model=model)

    assert critic_component_name(config) == "patch_gan"

    blank = types.SimpleNamespace(
        model=ModelConfigSchema(discriminator_component=ModelComponentSchema(name=""))
    )
    assert critic_component_name(blank) is None, "a blank name is not a critic"
    assert critic_component_name(types.SimpleNamespace(model=None)) is None, (
        "no model block is a legitimate 'no critic', not a schema drift"
    )


def test_a_model_without_the_field_raises_rather_than_reporting_no_critic():
    """A drifted schema must be LOUD here, exactly as it is on the strategy side.

    This is the invariant ``2e765cc4b`` established and that
    ``test_a_renamed_discriminator_component_raises_rather_than_reporting_false``
    and ``test_the_conditioning_reader_is_as_loud_as_the_capability_reader``
    (``tests/unit/infrastructure/training/strategies/test_adversarial_diffusion.py``)
    pin on the readers that delegate here. Electing this function as the one
    owner (non-negotiable 17) inherits that bar; it does not lower it
    (non-negotiable 20).

    **Why the loud half matters more than it looks.** A ``getattr`` fallback
    answers ``None`` for a renamed field exactly as for an absent one. Every
    caller then degrades in silence -- ``critic_input_domain`` reports no
    domain, so no conversion is owed; ``critic_accepts_complex`` reports False,
    so a complex-capable critic is fed realified input; ``_critic_conditioning``
    reports nothing, so a conditioned critic scores every batch of the run
    unconditioned. The run completes, the audit stays green, and the science is
    wrong (non-negotiable 3).

    Pydantic materializes every declared field on every instance, so a real
    config ALWAYS carries ``discriminator_component`` -- defaulted to ``None``
    when unset. An object that lacks the attribute entirely therefore cannot
    have come from the loader; it is either schema drift or a test double
    asserting a shape the schema does not have. Both deserve the raise.
    """
    drifted = types.SimpleNamespace(
        model=types.SimpleNamespace(critic_component=types.SimpleNamespace(name="x"))
    )
    with pytest.raises(AttributeError, match="discriminator_component"):
        critic_component_name(drifted)
