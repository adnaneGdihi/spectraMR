"""A discriminator on a diffusion arm must TRAIN, or not be accepted at all.

`DiffusionTrainingStrategy` advertised adversarial support -- its class docstring
says "**Adversarial Loss**: Optional GAN-style discriminator loss", and
`_compute_losses_impl` passes `discriminator=` to the loss computer -- while
providing no way to update the critic: it did not inherit `AdversarialMixin`,
defined no discriminator step, and had no `train_step` of its own. A critic
attached to a diffusion arm stayed at its initialisation and fed the generator a
meaningless signal (pitfall #16).

There were TWO gates, both keyed on the paradigm NAME, and fixing either alone
changes nothing observable:

* the strategy consulted a discriminator it never updated;
* ``fit()`` wired a discriminator only when ``paradigm == "gan"``, so
  ``fit(paradigm="diffusion", discriminator=d)`` accepted the argument and
  silently dropped it -- no ``opt_d``, no model in the env.

Only both together move a weight, which is why these tests assert on weights
rather than on configuration.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

pytestmark = [pytest.mark.slow]


class _Pairs(Dataset):
    def __init__(self, n: int = 4, size: int = 64) -> None:
        self.x = torch.randn(n, 1, size, size)
        self.y = torch.randn(n, 1, size, size)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {"input": self.x[i], "target": self.y[i]}


class _TimestepConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


def _adversarial_config():
    from spectramr.pipelines.fit import _PARADIGM_DEFAULTS

    return {
        "losses": {
            **_PARADIGM_DEFAULTS["gan"]["losses"],
            # The adversarial term is warm-up gated for 1000 iterations by
            # default; a short run has to step outside that to observe it.
            "reconstruction": {"warmup_iterations": 0},
        },
        "training": {
            "strategy_class": "diffusion",
            **_PARADIGM_DEFAULTS["diffusion"]["training"],
        },
    }


def test_discriminator_on_a_diffusion_arm_actually_trains():
    """THE regression: the critic's weights must move.

    Asserting that a discriminator is present, or that a loss was computed,
    would both have passed against the facade. Only the weights distinguish
    "trained" from "consulted".
    """
    from spectramr.pipelines.fit import fit

    gen, disc = _TimestepConv(), nn.Conv2d(1, 1, 3, padding=1)
    disc_before = disc.weight.detach().clone()
    gen_before = gen.conv.weight.detach().clone()

    result = fit(
        gen,
        DataLoader(_Pairs(), batch_size=2),
        paradigm="diffusion",
        discriminator=disc,
        opt_d=torch.optim.Adam(disc.parameters(), lr=1e-3),
        device="cpu",
        max_iterations=3,
        config=_adversarial_config(),
    )

    assert result.get("success") is True, result.get("error")
    assert not torch.equal(disc_before, disc.weight.detach()), (
        "the discriminator was built and consulted but never updated — the facade"
    )
    assert not torch.equal(gen_before, gen.conv.weight.detach())


def test_a_diffusion_arm_without_a_discriminator_is_unchanged():
    """ADDITIVE: no discriminator means the base generator-only step, as before.

    This is what makes the feature opt-in rather than a change of paradigm, and
    it is the property most at risk from a `train_step` override.
    """
    from spectramr.pipelines.fit import fit

    result = fit(
        _TimestepConv(),
        DataLoader(_Pairs(), batch_size=2),
        paradigm="diffusion",
        device="cpu",
        max_iterations=2,
    )
    assert result.get("success") is True, result.get("error")


def test_step_configs_are_n_critic_updates_then_one_generator():
    """The cadence comes from ``losses.gan.disc_updates``, shared with GAN."""
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    assert hasattr(DiffusionTrainingStrategy, "train_step")
    import inspect

    source = inspect.getsource(DiffusionTrainingStrategy.train_step)
    assert "assemble_adversarial_step_configs" in source, (
        "the diffusion adversarial step must use the shared cadence composer, "
        "not a third copy of the N-then-one assembly"
    )
    assert "_resolve_disc_updates" in source


def test_fit_wires_a_discriminator_for_any_paradigm_not_just_gan():
    """The second gate: ``fit`` used to drop the argument unless paradigm=='gan'."""
    import inspect

    from spectramr.pipelines import fit as fit_mod

    source = inspect.getsource(fit_mod.fit)
    assert 'if disc is not None:\n        models["discriminator"] = disc' in source, (
        "discriminator wiring is gated on the paradigm name again; "
        "fit(paradigm='diffusion', discriminator=d) would silently drop it"
    )


# ---------------------------------------------------------------------------
# The image-space critic seam.
#
# A critic that bridges k-space into image space needs two things the diffusion
# path did not provide: the pair still complex (so it can tell interleaved from
# block-stacked, which a realified tensor no longer distinguishes), and the coil
# sensitivity maps of the current batch. Both are opt-in, keyed on a capability
# the critic declares, so every existing critic is untouched.
# ---------------------------------------------------------------------------


def _pair():
    real_interleaved = torch.randn(2, 8, 8, 8)
    complex_target = torch.randn(2, 4, 8, 8, dtype=torch.complex64)
    return real_interleaved, complex_target


def test_align_for_critic_hands_a_declaring_critic_the_untouched_pair():
    """``accepts_complex`` means "I own the conversion" -- so nothing is stacked.

    This is the whole point of the flag. ``fake`` arrives real and INTERLEAVED
    from the generator; ``target_batch`` may still be complex. Stacking only the
    complex side leaves the two in different layouts, and a critic then
    separates real from fake on channel order alone.
    """
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    fake, real = _pair()
    out_f, out_r = DiffusionTrainingStrategy._align_for_critic(
        fake, real, critic_takes_complex=True
    )
    assert out_f is fake
    assert torch.is_complex(out_r), "the complex side was realified anyway"
    assert out_r.shape[1] == 4


def test_align_for_critic_still_stacks_for_a_critic_that_declares_nothing():
    """REGRESSION. Every existing critic declares nothing, and 25 GAN arms rely on this.

    The block layout here is shared with ``GANTrainingStrategy._train_discriminator_step``;
    the new seam must not change it, only bypass it on request.
    """
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    fake, real = _pair()
    out_f, out_r = DiffusionTrainingStrategy._align_for_critic(fake, real)
    assert out_f is fake, "an already-real fake must pass through untouched"
    assert not torch.is_complex(out_r)
    assert out_r.shape[1] == 8, "real/imag must still be stacked on the channel axis"
    assert torch.equal(out_r, torch.cat([real.real, real.imag], dim=1))


def test_align_for_critic_uses_one_domain_for_both():
    """PIN. ``real`` is converted from the GENERATOR's domain, not one of its own.

    The tempting "fix" here is to give ``real`` a domain resolved from
    ``data.domain.output``, on the reading that ``real`` is data so its domain
    is a data fact. That reading was true of the raw ``target_batch`` this
    parameter used to receive and is false of the ``prepared_target`` it
    receives now: the prepared target is the tensor the reconstruction losses
    are computed against, and ``DifferentiableFourierBridge.forward`` projects
    ``k_pred`` and ``k_target`` through the same ``_kspace_to_image``, so the
    loss pipeline already requires both to sit in ``losses.policy.output_domain``.

    A second resolver would not raise -- it would compute a plausible answer and
    diverge silently (non-negotiable 17), which is why the pin asserts the
    TRANSFORMED VALUE rather than merely that the call was made. With
    ``data.domain.output`` defaulting to ``image``, a second resolver makes
    ``real``'s conversion same-side and therefore a no-op, and the ``ifft2c``
    comparison below goes red.
    """
    from spectramr.infrastructure.physics.fft_ops import ifft2c
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    fake, real = _pair()
    out_f, out_r = DiffusionTrainingStrategy._align_for_critic(
        fake, real, generator_domain="kspace", critic_domain="image"
    )

    # Both sides IFFT'd, then realified as a block -- the converted path's
    # symmetry, which is the thing a second resolver would break.
    expect_r = ifft2c(real)
    expect_f = ifft2c(fake)
    assert torch.allclose(out_r, torch.cat([expect_r.real, expect_r.imag], dim=1)), (
        "`real` was not converted out of the generator's domain -- something is "
        "resolving a domain for it separately"
    )
    assert torch.allclose(out_f, torch.cat([expect_f.real, expect_f.imag], dim=1))
    assert out_r.shape == out_f.shape, (
        "conversion is what makes the two sides comparable; they diverged"
    )


def test_both_capability_readers_agree_now_that_one_owner_is_elected():
    """The two readers must return the SAME answer -- that is the invariant.

    History, because the flip in this test is the interesting part. This case
    used to assert ``model_supports(...) is False`` and explain why: the helper
    looked the flag up as ``entry.get(capability)`` at the *top level* of the
    registry entry, while the capability flags lived nested in
    ``entry["capabilities"]``. It therefore reported False for every one of the
    20 models that declare ``accepts_complex`` -- a wrong answer shaped exactly
    like a legitimate "not supported", which is why it survived so long. The
    assertion was left here deliberately, with a message telling whoever fixed
    it that the seam needed revisiting (#1916).

    #1916 elected one owner (non-negotiable 17): ``register_model`` no longer
    fans flags out to the entry dict, and ``model_supports`` reads the nested
    dataclass. So the two halves now agree, and *agreement* is what this pins.
    A regression in either direction turns it red -- a second top-level surface
    reappearing, or the strategy's own accessor drifting off the dataclass.
    """
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import model_supports

    populate_model_registry()

    def _cfg(name):
        return SimpleNamespace(
            model=SimpleNamespace(discriminator_component=SimpleNamespace(name=name))
        )

    assert DiffusionTrainingStrategy._critic_accepts_complex(_cfg("sense_bridge_patchgan"))
    assert model_supports("sense_bridge_patchgan", "accepts_complex") is True, (
        "model_supports disagrees with the dataclass again — a second "
        "declaration surface has reappeared (#1916 elected one owner)"
    )
    assert not DiffusionTrainingStrategy._critic_accepts_complex(_cfg("patch_gan"))
    assert not DiffusionTrainingStrategy._critic_accepts_complex(SimpleNamespace(model=None))


def test_a_renamed_discriminator_component_raises_rather_than_reporting_false():
    """A missing config field must be LOUD, not a False that disables the seam.

    The accessor used to read
    ``getattr(getattr(config, "model", None), "discriminator_component", None)``.
    That chain answers ``None`` for a field that was renamed exactly as it does
    for one that is legitimately absent, so a schema rename would have silently
    turned every complex-capable critic into a real-valued one -- a different
    training run, no error, non-negotiable 3's silent fallback.

    ``model=None`` stays a legitimate False (an arm with no model block, pinned
    above); a model that HAS no ``discriminator_component`` attribute at all is
    the schema drift, and it must raise.
    """
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    renamed = SimpleNamespace(model=SimpleNamespace(critic_component=SimpleNamespace(name="x")))
    with pytest.raises(AttributeError, match="discriminator_component"):
        DiffusionTrainingStrategy._critic_accepts_complex(renamed)


def test_the_conditioning_reader_is_as_loud_as_the_capability_reader():
    """Both readers of the critic name must fail a rename the SAME way (#1931).

    ``_critic_accepts_complex`` was made loud in 2e765cc4b. ``_critic_conditioning``
    -- added later, for the t/contrast payload -- kept the
    ``getattr(getattr(config, "model", None), "discriminator_component", None)``
    chain that commit had just removed, so the identical schema drift produced
    two different verdicts: an ``AttributeError`` from one seam and a silent
    empty payload from the other. The silent one is the dangerous half: the arm
    declares a conditioned critic, the audit stays green, and the critic scores
    every batch of the run unconditioned (non-negotiable 3).
    """
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    renamed = SimpleNamespace(model=SimpleNamespace(critic_component=SimpleNamespace(name="x")))
    strategy = SimpleNamespace(
        env=None,
        config=renamed,
        _critic_component_name=DiffusionTrainingStrategy._critic_component_name,
    )
    with pytest.raises(AttributeError, match="discriminator_component"):
        DiffusionTrainingStrategy._critic_conditioning(strategy, None, None)


def test_an_arm_with_no_critic_configured_still_gets_an_empty_payload():
    """The loudness must not swallow the two states the schema really allows.

    ``model=None`` (no model block) and ``discriminator_component=None`` (no
    critic) are legitimate, and both must stay a quiet ``{}`` -- that is the
    path every unconditioned arm in the corpus takes, and turning it into a
    raise would break far more than it fixed.
    """
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    for cfg in (
        SimpleNamespace(model=None),
        SimpleNamespace(model=SimpleNamespace(discriminator_component=None)),
        SimpleNamespace(model=SimpleNamespace(discriminator_component=SimpleNamespace(name=""))),
    ):
        strategy = SimpleNamespace(
            env=None,
            config=cfg,
            _critic_component_name=DiffusionTrainingStrategy._critic_component_name,
        )
        assert DiffusionTrainingStrategy._critic_conditioning(strategy, None, None) == {}
        assert DiffusionTrainingStrategy._critic_accepts_complex(cfg) is False


def test_neither_critic_reader_re_derives_the_component_name():
    """Non-negotiable 17: one owner, and the losers' resolution is DELETED.

    The owner MOVED in #1921. ``models/losses/computers/unified_gan.py`` needs
    the same fact and may not import ``infrastructure/`` (non-negotiable 5), so
    the resolution now lives at
    ``models.losses.critic_domain.critic_component_name`` and the strategy-side
    ``DiffusionTrainingStrategy._critic_component_name`` is a thin delegate.
    A move is not a second owner: this test follows the owner rather than
    grandfathering a copy, and still fails if any reader re-derives the name.

    Structural, not prose: every body is unparsed from the AST with the
    docstring dropped, because the docstrings legitimately *mention*
    ``model.discriminator_component.name`` -- a source-text grep would go red on
    the explanation rather than on a second resolver. ``ast.unparse`` also drops
    comments, so only executable code is inspected.
    """
    import ast
    import inspect
    import textwrap

    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )
    from spectramr.models.losses import critic_domain

    def _executable_body(fn) -> str:
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        return "\n".join(ast.unparse(n) for n in body)

    owner = _executable_body(critic_domain.critic_component_name)
    assert "discriminator_component" in owner, (
        "the elected owner stopped reading the field it owns -- this test is "
        "no longer checking anything"
    )
    assert "getattr" not in owner, (
        "the owner reads declared schema fields directly so a RENAME raises; a "
        "getattr fallback would report 'this arm has no critic' instead (#368)"
    )

    readers = {
        "critic_domain.critic_accepts_complex": critic_domain.critic_accepts_complex,
        "critic_domain.critic_input_domain": critic_domain.critic_input_domain,
        "DiffusionTrainingStrategy._critic_component_name": (
            DiffusionTrainingStrategy._critic_component_name
        ),
        "DiffusionTrainingStrategy._critic_accepts_complex": (
            DiffusionTrainingStrategy._critic_accepts_complex
        ),
        "DiffusionTrainingStrategy._critic_conditioning": (
            DiffusionTrainingStrategy._critic_conditioning
        ),
    }
    for name, fn in readers.items():
        code = _executable_body(fn)
        assert "discriminator_component" not in code, (
            f"{name} re-derives the critic name instead of calling the one "
            "owner critic_component_name; two resolvers of one fact drift "
            "silently and both keep passing their own tests"
        )
        assert "critic_component_name" in code or "critic_accepts_complex" in code, (
            f"{name} no longer routes through the elected owner"
        )


def test_the_unified_gan_computer_shares_the_elected_owner():
    """The chokepoint consumes the same resolver, not a third spelling.

    #1921 wires the critic-domain seam into ``UnifiedGANLossComputer``. The
    failure mode this guards is the one non-negotiable 17 names: the computer
    growing its own ``config.model.discriminator_component.name`` read, which
    would pass its own tests while drifting from the strategy's answer.
    """
    import ast
    import inspect

    from spectramr.models.losses.computers import unified_gan

    src = inspect.getsource(unified_gan)
    assert "discriminator_component" not in src, (
        "unified_gan.py re-derives the critic name; it must call "
        "critic_domain.critic_input_domain / critic_accepts_complex"
    )
    imported = {
        n.name
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("critic_domain")
        for n in node.names
    }
    assert {"critic_input_domain", "generator_output_domain", "resolve_conversion"} <= imported, (
        f"unified_gan.py must consume the elected seam helpers; imports {imported}"
    )


def test_the_sense_bridge_critic_fires_with_smaps_inside_both_closures():
    """Non-negotiable 16: observed firing on the production path, not a resolvable class.

    Registering the critic, and even constructing it, proves nothing -- three
    k-space discriminators in this package are registered and reached by no arm.
    What is asserted here is that a real ``fit`` run drives the bridge, that the
    maps are present every time it runs (never the None that would mean an RSS
    image), and that the critic's own weights move.

    Four calls per iteration is the shape to expect: real and fake in the D
    closure, real and fake again in the G closure.
    """
    from spectramr.models.discriminators import SenseBridgeDiscriminator
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.pipelines.fit import _PARADIGM_DEFAULTS, fit

    populate_model_registry()
    seen: list[tuple | None] = []

    class _SpyCritic(SenseBridgeDiscriminator):
        def bridge(self, x):
            # Read through ``sense_bridge``: the provider lives on the extracted
            # bridge, not on the critic, so both critics can share one copy.
            provider = self.sense_bridge._smaps_provider
            maps = provider(x.shape[0]) if provider is not None else None
            seen.append(None if maps is None else tuple(maps.shape))
            return super().bridge(x)

    class _CoilPairs(Dataset):
        """Carries ``smaps``, which is what makes the bridge meaningful."""

        def __init__(self, n: int = 4, size: int = 32, coils: int = 4) -> None:
            self.x = torch.randn(n, 2 * coils, size, size)
            self.y = torch.randn(n, 2 * coils, size, size)
            self.s = torch.randn(n, coils, size, size, dtype=torch.complex64)

        def __len__(self) -> int:
            return len(self.x)

        def __getitem__(self, i: int) -> dict:
            return {"input": self.x[i], "target": self.y[i], "smaps": self.s[i]}

    class _CoilGen(nn.Module):
        # 16 in: the strategy concatenates the conditioning input onto the noisy
        # sample before the forward. 8 out: 4 coils, real/imag interleaved.
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(16, 8, 3, padding=1)

        def forward(self, x, *args, **kwargs):
            return self.conv(x)

    critic = _SpyCritic(in_channels=1, critic_kwargs={"ndf": 8, "n_layers": 2})
    before = [p.detach().clone() for p in critic.critic.parameters()]

    result = fit(
        _CoilGen(),
        DataLoader(_CoilPairs(), batch_size=2),
        paradigm="diffusion",
        discriminator=critic,
        opt_d=torch.optim.Adam(critic.parameters(), lr=1e-3),
        device="cpu",
        max_iterations=2,
        config={
            # kspace_cold_diffusion is the branch that populates _current_smaps.
            "model": {
                "model_type": "kspace_cold_diffusion",
                "in_channels": 8,
                "out_channels": 8,
            },
            # Equal to in_channels so the TI-CCD asymmetric-mask branch stays out
            # of this test; it is unrelated to the critic.
            "data": {"domain": {"target_channels": 8}},
            "losses": {
                **_PARADIGM_DEFAULTS["gan"]["losses"],
                "reconstruction": {"warmup_iterations": 0},
            },
            "training": {
                "strategy_class": "diffusion",
                **_PARADIGM_DEFAULTS["diffusion"]["training"],
            },
        },
    )

    assert result.get("success") is True, result.get("error")
    assert seen, "the bridge never ran — the critic was built but not consulted"
    assert len(seen) >= 4, (
        f"only {len(seen)} bridge calls over 2 iterations; expected at least 4 "
        "(real and fake, in the D closure and again in the G closure)"
    )
    assert all(s is not None for s in seen), (
        "the bridge ran without sensitivity maps — the strategy failed to install "
        "the provider, or installed it too late for one of the two closures"
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(before, critic.critic.parameters(), strict=True)
    ), "the bridged critic was consulted but never updated"


def _cfg_with_critic(critic_name: str, output_domain: str):
    """A config carrying only the two keys the #1920 seam reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        model=SimpleNamespace(discriminator_component=SimpleNamespace(name=critic_name)),
        losses=SimpleNamespace(policy=SimpleNamespace(output_domain=output_domain)),
    )


def _resolve_critic(cfg, critic):
    from types import SimpleNamespace

    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    stub = SimpleNamespace(
        env=SimpleNamespace(config=cfg),
        config=cfg,
        _critic_accepts_complex=DiffusionTrainingStrategy._critic_accepts_complex,
    )
    return DiffusionTrainingStrategy._critic_for_domain(stub, critic)


def test_a_matching_pair_reaches_the_g_step_as_the_bare_critic():
    """REGRESSION, and the reason it is worth a test of its own.

    Wrapping unconditionally would put a converter in front of every critic on
    every arm -- including the 58 that need none -- and the wrapper is not an
    ``nn.Module``, so anything downstream reaching for ``.parameters()`` or
    ``.train()`` would break on arms this change was never meant to touch.

    ``sense_bridge_patchgan`` consumes k-space and the generator emits k-space,
    so nothing is owed.
    """
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    critic = nn.Conv2d(1, 1, 1)
    assert _resolve_critic(_cfg_with_critic("sense_bridge_patchgan", "kspace"), critic) is critic


def test_a_mismatched_pair_reaches_the_g_step_through_the_converter():
    """``kspace_discriminator`` consumes IMAGES despite its name, so an ifft2c is owed.

    Without this the G step would hand the critic raw k-space while the D step
    handed it an image -- the critic scoring one space while training and
    another while scoring the generator.
    """
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.losses.critic_domain import (
        DomainAdaptedCritic,
    )

    populate_model_registry()
    critic = nn.Conv2d(1, 1, 1)
    wrapped = _resolve_critic(_cfg_with_critic("kspace_discriminator", "kspace"), critic)
    assert isinstance(wrapped, DomainAdaptedCritic)
    assert wrapped.critic is critic


def test_an_undeclared_critic_is_never_wrapped():
    """A critic that declares no ``input_domain`` is handed through untouched.

    ``domain_discriminator`` is DELIBERATELY undeclared (#1920): its first layer is
    ``AdaptiveAvgPool2d(1) -> Flatten -> Linear``, so it scores an encoder feature
    map, and no word in the ``Domain`` vocabulary names that input. Guessing one
    would FFT a feature map into nonsense, so ``None`` must mean "do not convert".

    This test used to name ``patch_gan``, which was undeclared when it was written
    ("18 of 19 registered discriminators declare no domain"). #1920 declares it
    ``image``, so that name now exercises the OPPOSITE path -- see the sibling
    below, which asserts exactly that.

    The two registry assertions are load-bearing, not decoration.
    ``get_model_capabilities`` returns ``None`` for BOTH "not registered" and
    "registered but unannotated" (non-negotiable 18), so without them a typo in
    the critic name would take the no-conversion branch and pass this test
    vacuously.
    """
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY, get_model_capabilities

    populate_model_registry()
    assert "domain_discriminator" in MODEL_REGISTRY, "the name must be REGISTERED"
    assert get_model_capabilities("domain_discriminator") is None, (
        "...and genuinely UNANNOTATED, or this test proves nothing"
    )

    critic = nn.Conv2d(1, 1, 1)
    assert _resolve_critic(_cfg_with_critic("domain_discriminator", "kspace"), critic) is critic


def test_a_newly_declared_critic_is_wrapped_where_it_previously_was_not():
    """#1920's declaration turns a real production path from no-op to converting.

    ``patch_gan`` is a plain 2-D convolution stack: its ``forward`` calls no
    ``fft2c``, so the class body cannot say which space it scores in and it
    accepted a k-space tensor as readily as an image. Paired with a k-space
    generator it silently scored raw k-space with image weights -- the exact
    defect #1920 was filed about.

    Declaring ``input_domain="image"`` on the registration makes
    ``resolve_conversion("kspace", "image")`` owe an ``ifft2c``, and
    ``_resolve_critic`` -- the PRODUCTION resolver, not a fixture -- wraps it.
    This is the "observed to fire" evidence for non-negotiable 16: no corpus arm
    exercises the conversion (all 23 critic-bearing arms are same-side), so this
    pairing is where the seam is watched working.
    """
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.losses.critic_domain import DomainAdaptedCritic
    from spectramr.models.registry import get_model_capabilities

    populate_model_registry()
    caps = get_model_capabilities("patch_gan")
    assert caps is not None and caps.input_domain == "image"

    critic = nn.Conv2d(1, 1, 1)
    wrapped = _resolve_critic(_cfg_with_critic("patch_gan", "kspace"), critic)
    assert isinstance(wrapped, DomainAdaptedCritic), "a mismatched pair must convert"
    assert wrapped.critic is critic


# --------------------------------------------------------------------------- #
# #1931 half 2: the conditioning payload, observed on the production path.
#
# The sibling test above proves the BRIDGE fires in both closures. These prove
# the CONDITIONING does — a separate claim, because the payload travels a
# different route (strategy -> loss computer -> critic) and is dropped by a
# different mechanism (a computer that forgets to forward it, or a
# DomainAdaptedCritic whose __call__ takes only one argument).
# --------------------------------------------------------------------------- #


class _CoilPairsWithSmaps(Dataset):
    """Carries ``smaps`` AND ``contrast_idx`` — the latter is what makes this a
    multi-contrast batch, and its absence is not a detail the critic tolerates.

    Written without ``contrast_idx`` first, this dataset made the conditioned
    critic RAISE (`...was called without contrast_idx`) rather than score the
    marginal. That is the designed behaviour and worth stating: the payload
    reaching the critic as ``None`` is a loud failure, not a silent one.
    """

    def __init__(self, n: int = 4, size: int = 32, coils: int = 4) -> None:
        self.x = torch.randn(n, 2 * coils, size, size)
        self.y = torch.randn(n, 2 * coils, size, size)
        self.s = torch.randn(n, coils, size, size, dtype=torch.complex64)
        # Varies within a batch, as pairing.single_contrast: true produces.
        self.c = torch.arange(n) % 3

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {
            "input": self.x[i],
            "target": self.y[i],
            "smaps": self.s[i],
            "contrast_idx": self.c[i],
        }


class _CoilGen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(16, 8, 3, padding=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


def _run_fit_with_critic(critic, critic_name: str, data_extra: dict | None = None):
    """Drive a real 2-iteration diffusion fit with ``critic`` as the D.

    ``data_extra`` merges into the ``data`` block. It defaults to ``None`` so
    every existing caller keeps the exact config it had.
    """
    from spectramr.pipelines.fit import _PARADIGM_DEFAULTS, fit

    return fit(
        _CoilGen(),
        DataLoader(_CoilPairsWithSmaps(), batch_size=2),
        paradigm="diffusion",
        discriminator=critic,
        opt_d=torch.optim.Adam(critic.parameters(), lr=1e-3),
        device="cpu",
        max_iterations=2,
        config={
            "model": {
                "model_type": "kspace_cold_diffusion",
                "in_channels": 8,
                "out_channels": 8,
                # The registry gate reads THIS name, not the object.
                "discriminator_component": {"name": critic_name},
            },
            "data": {
                "domain": {"target_channels": 8},
                "multi_contrast": {"enabled": True, "n_contrasts": 3},
                **(data_extra or {}),
            },
            "losses": {
                **_PARADIGM_DEFAULTS["gan"]["losses"],
                "reconstruction": {"warmup_iterations": 0},
            },
            "training": {
                "strategy_class": "diffusion",
                **_PARADIGM_DEFAULTS["diffusion"]["training"],
            },
        },
    )


def test_a_conditioned_critic_is_given_its_payload_in_both_closures():
    """Non-negotiable 16: observed firing, not a forwarded argument read in source.

    The D step and the G step reach the critic through *different* loss
    computers -- ``UnifiedGANLossComputer.compute_discriminator_loss`` and
    ``UnifiedDiffusionLossComputer.compute`` -- so forwarding in one says
    nothing about the other. Half of this change exists because that asymmetry
    was mis-read as one owner.
    """
    from spectramr.models.discriminators import ContrastConditionedSenseBridgeDiscriminator
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    seen: list[dict] = []

    class _SpyCritic(ContrastConditionedSenseBridgeDiscriminator):
        def forward(self, x, timesteps=None, contrast_idx=None):
            seen.append(
                {
                    "t": None if timesteps is None else tuple(timesteps.shape),
                    "c": None if contrast_idx is None else tuple(contrast_idx.shape),
                    "batch": x.shape[0],
                }
            )
            return super().forward(x, timesteps=timesteps, contrast_idx=contrast_idx)

    critic = _SpyCritic(in_channels=1, critic_kwargs={"ndf": 8, "n_layers": 2}, num_contrasts=3)
    result = _run_fit_with_critic(critic, "sense_bridge_patchgan_conditioned")

    assert result.get("success") is True, result.get("error")
    assert seen, "the critic was never consulted"
    unconditioned = [s for s in seen if s["t"] is None or s["c"] is None]
    assert not unconditioned, (
        f"{len(unconditioned)} of {len(seen)} critic calls arrived without "
        "conditioning — the payload is dropped somewhere between "
        "_critic_conditioning and the critic"
    )
    assert len(seen) >= 4, (
        f"only {len(seen)} critic calls over 2 iterations; expected at least 4 "
        "(real and fake in the D closure, and the fake again in the G closure). "
        "Fewer means one of the two closures is unconditioned."
    )
    # Every call must label exactly as many samples as it scores; a broadcast
    # mismatch here is the silent-corruption case.
    assert all(s["t"] == (s["batch"],) and s["c"] == (s["batch"],) for s in seen), (
        f"conditioning shape disagrees with the scored batch: {seen}"
    )


def test_an_unaware_critic_is_called_exactly_as_before():
    """The other 57 arms must be byte-identical, and nothing else asserts that.

    ``_critic_conditioning`` returns ``{}`` for a critic that does not declare
    the flag, and ``**{}`` is the unconditioned call. This is the test that
    would catch a payload sent unconditionally -- which would ``TypeError``
    every existing arm on step 1.
    """
    from spectramr.models.discriminators import SenseBridgeDiscriminator
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    kwargs_seen: list[dict] = []

    class _StrictCritic(SenseBridgeDiscriminator):
        def forward(self, x, **kwargs):
            kwargs_seen.append(dict(kwargs))
            return super().forward(x)

    critic = _StrictCritic(in_channels=1, critic_kwargs={"ndf": 8, "n_layers": 2})
    result = _run_fit_with_critic(critic, "sense_bridge_patchgan")

    assert result.get("success") is True, result.get("error")
    assert kwargs_seen, "the critic was never consulted"
    assert all(k == {} for k in kwargs_seen), (
        "an unconditioned critic received keyword arguments: "
        f"{[k for k in kwargs_seen if k]}. Every arm whose critic does not "
        "declare supports_contrast_conditioning must be called exactly as before."
    )


def test_the_critic_scores_the_target_its_fake_was_prepared_against(monkeypatch):
    """The D step's ``real`` must be the target the ``fake`` was built from.

    ``_fake_for_critic`` runs the full preparation -- including
    ``apply_kspace_normalization``, which REBINDS ``target_batch`` -- and used
    to return only ``(fake, critic_cond)``, dropping the prepared target.
    ``d_closure`` then scored that fake against ``train_step``'s own
    ``target_batch``, which is only device-moved. Two owners for "the tensor
    the critic calls real"; the loser was silently discarded (non-negotiable
    17).

    Measured on this harness before the fix: the fake was built against a
    normalized target of absmax 7.50 while the critic was handed the raw one at
    absmax 22.81, against a fake at 0.475. A critic separates those on
    magnitude alone and never learns realism -- and nothing raises, because
    both tensors have the same shape and dtype.

    The k-space normalization branch is the reachable route (this arm is cold
    diffusion, and a batch carrying no ``kspace_normalized`` marker makes
    ``_batch_is_already_normalized`` return False). The 5D-flatten rebinds at
    the same seam crash instead, so this is the silent half.
    """
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )
    from spectramr.models.discriminators import ContrastConditionedSenseBridgeDiscriminator
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()

    prepared: list[torch.Tensor] = []
    original_normalize = DiffusionTrainingStrategy.apply_kspace_normalization

    def _record_prepared(self, input_batch, target_batch, **kwargs):
        new_input, new_target, scale = original_normalize(self, input_batch, target_batch, **kwargs)
        prepared.append(new_target.detach().clone())
        return new_input, new_target, scale

    monkeypatch.setattr(DiffusionTrainingStrategy, "apply_kspace_normalization", _record_prepared)

    scored: list[torch.Tensor] = []
    original_align = DiffusionTrainingStrategy.__dict__["_align_for_critic"].__func__

    def _record_real(fake, real, *args, **kwargs):
        scored.append(real.detach().clone())
        return original_align(fake, real, *args, **kwargs)

    monkeypatch.setattr(DiffusionTrainingStrategy, "_align_for_critic", staticmethod(_record_real))

    critic = ContrastConditionedSenseBridgeDiscriminator(
        in_channels=1, critic_kwargs={"ndf": 8, "n_layers": 2}, num_contrasts=3
    )
    result = _run_fit_with_critic(
        critic,
        "sense_bridge_patchgan_conditioned",
        data_extra={"processing": {"enable_kspace_normalization": True}},
    )

    assert result.get("success") is True, result.get("error")
    # Precondition: the rebind this test is about must actually have happened.
    assert prepared, (
        "apply_kspace_normalization never ran, so target_batch was never "
        "rebound and this test cannot observe the defect it exists for"
    )
    assert scored, "the D closure never reached _align_for_critic"

    for i, real in enumerate(scored):
        assert any(real.shape == p.shape and torch.allclose(real, p) for p in prepared), (
            f"D-step call {i}: the critic's `real` is not the target the fake "
            f"was prepared against. real absmax={float(real.abs().max()):.4f}, "
            f"prepared absmax="
            f"{[round(float(p.abs().max()), 4) for p in prepared]}. The critic "
            "is scoring a raw target against a normalized fake and can "
            "separate them on scale alone."
        )


def test_a_batch_that_needs_no_preparation_reaches_the_critic_unchanged(monkeypatch):
    """The other half: on the path every shipped arm takes, nothing moves.

    ``experiment_11_sense_bridge_critic`` is the only arm in the corpus that
    reaches this seam, and its batches arrive already normalized (the loader's
    ``KSpaceNormalizationTransform`` sets ``kspace_normalized``), 4D, and with
    the generator's channel count. The preparation therefore rebinds nothing and
    the prepared target IS the caller's target. Carrying the prepared one must
    be a no-op there -- otherwise the fix is a behaviour change dressed as a
    correction, and the arm's numbers would move.
    """
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )
    from spectramr.models.discriminators import ContrastConditionedSenseBridgeDiscriminator
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()

    normalized_calls: list[int] = []
    original_normalize = DiffusionTrainingStrategy.apply_kspace_normalization

    def _count(self, *args, **kwargs):
        normalized_calls.append(1)
        return original_normalize(self, *args, **kwargs)

    monkeypatch.setattr(DiffusionTrainingStrategy, "apply_kspace_normalization", _count)

    scored: list[torch.Tensor] = []
    original_align = DiffusionTrainingStrategy.__dict__["_align_for_critic"].__func__

    def _record_real(fake, real, *args, **kwargs):
        scored.append(real.detach().clone())
        return original_align(fake, real, *args, **kwargs)

    monkeypatch.setattr(DiffusionTrainingStrategy, "_align_for_critic", staticmethod(_record_real))

    critic = ContrastConditionedSenseBridgeDiscriminator(
        in_channels=1, critic_kwargs={"ndf": 8, "n_layers": 2}, num_contrasts=3
    )
    # No ``enable_kspace_normalization``: the preparation has nothing to do.
    result = _run_fit_with_critic(critic, "sense_bridge_patchgan_conditioned")

    assert result.get("success") is True, result.get("error")
    assert not normalized_calls, (
        "this test is only meaningful while the preparation is a no-op, and "
        "apply_kspace_normalization ran"
    )
    assert scored, "the D closure never reached _align_for_critic"
    # The dataset's targets are the only tensors that can legitimately appear.
    assert all(r.ndim == 4 and r.shape[1] == 8 for r in scored), (
        f"unexpected real shapes reaching the critic: {[tuple(r.shape) for r in scored]}"
    )
