"""The critic seam: convert what feeds the CRITIC, never what feeds the loss (#1921).

``UnifiedGANLossComputer`` is the one chokepoint every GAN-family strategy
reaches, and it hands the same two tensors to two consumers with **opposite**
domain requirements:

* the critic scores them in the domain it declared (``input_domain``, #1920);
* the reconstruction losses -- L1, perceptual, SSIM, MS-SSIM, LPIPS -- compare
  them in the domain the generator emits.

The two are not distinguishable by parameter name. ``CompositeGANLoss``
exposes ``real_images=``/``fake_images=`` on *both* ``compute_generator_loss``
and ``compute_discriminator_loss``, so a uniform "convert everything" rewrite
type-checks, runs, and silently computes L1 in k-space. On all 23 critic-bearing
``inprogress`` arms the generator and critic are same-side, so that defect is
**invisible in the entire live corpus** -- which is exactly why it is planted
here rather than left to a future cross-domain arm to discover.

The oracle is deliberately not a tolerance. An image->kspace conversion is an
``fft2c``, so it turns a REAL tensor into a COMPLEX one: ``is_complex()`` on what
each consumer actually received separates "the conversion ran" from "it did not"
with no numerical judgement at all.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

UNIFIED_GAN = "spectramr.models.losses.computers.unified_gan"


class _SpyCritic(nn.Module):
    """Records every tensor it is asked to score, then returns a scalar map."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        self.seen.append(x)
        # Real-valued output regardless of input dtype: a critic emits scores,
        # and the adversarial loss must not have to care that the seam converted.
        return torch.view_as_real(x).sum() if x.is_complex() else x.sum()


def _real_computer():
    """A computer built the way production builds it -- no ``object.__new__``."""
    from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer
    from spectramr.pipelines.fit import _resolve_fit_config

    config = _resolve_fit_config(None, paradigm="gan", epochs=1, max_iterations=2)
    return UnifiedGANLossComputer(config, torch.device("cpu")), config


def _force_cross_domain(monkeypatch, computer, *, takes_complex: bool = True) -> None:
    """Re-resolve the seam as an image generator against a k-space critic.

    The three readers are patched and ``_resolve_critic_seam`` is re-run, rather
    than the four seam attributes being assigned directly: the seam's own
    docstring names that method as its ONLY writer, and a test that writes the
    attributes itself would keep passing after the resolution logic broke.
    """
    monkeypatch.setattr(f"{UNIFIED_GAN}.generator_output_domain", lambda _c: "image")
    monkeypatch.setattr(f"{UNIFIED_GAN}.critic_input_domain", lambda _c: "kspace")
    monkeypatch.setattr(f"{UNIFIED_GAN}.critic_accepts_complex", lambda _c: takes_complex)
    computer._resolve_critic_seam()
    assert computer._critic_conversion is not None, "fixture failed to owe a conversion"


def _pair(size: int = 64):
    """64px, not 16: the default gan config enables the VGG perceptual loss, whose
    five pooling stages reduce a 16px input to 0x0 and raise before any assertion
    here is reached."""
    return torch.randn(2, 1, size, size), torch.randn(2, 1, size, size)


# ---------------------------------------------------------------------------
# The class defaults are only safe because production always overwrites them.
# ---------------------------------------------------------------------------


def test_the_production_path_resolves_the_seam():
    """Non-negotiable 18: the guarantee is asserted, never inferred.

    The four seam attributes are declared on the CLASS, not in ``__init__``,
    because this computer is an ``nn.Module``: ``nn.Module.__getattr__`` raises
    for a missing attribute, so the ``object.__new__`` shells four existing unit
    tests use to exercise one method in isolation would ``AttributeError`` inside
    ``_to_critic`` instead of running the arithmetic under test.

    That is only safe while every REAL construction resolves the seam, and a
    class default is exactly the kind of thing that silently becomes the value
    production reads. ``_critic_seam_resolved`` exists to make the two states
    distinguishable, and this test is what turns "every real build resolves it"
    from an assumption into a checked fact.
    """
    from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer

    assert UnifiedGANLossComputer._critic_seam_resolved is False, (
        "the class default is no longer False, so this test can no longer tell a "
        "resolved seam from an unresolved one"
    )

    computer, _ = _real_computer()
    assert computer._critic_seam_resolved is True, (
        "a computer built through the production path did NOT resolve the critic "
        "seam -- every arm would fall back to the class default (no conversion)"
    )


def test_a_same_side_arm_is_byte_for_byte_untouched():
    """All 23 live arms are same-side, so the seam must be object identity.

    ``is`` rather than ``allclose``: an ``ifft2c(fft2c(x))`` round trip would
    satisfy a numerical check while still perturbing every arm in the corpus at
    float32 precision, and re-interleaving a complex tensor to 2C would change
    the channel count without changing any value.
    """
    computer, _ = _real_computer()
    assert computer._critic_conversion is None, "the default gan config is not same-side"
    x, _ = _pair()
    assert computer._to_critic(x) is x


# ---------------------------------------------------------------------------
# The seam fires where the critic is fed, and nowhere else.
# ---------------------------------------------------------------------------


def test_the_discriminator_step_scores_the_converted_tensor(monkeypatch):
    """The D step's critic calls must arrive in the critic's declared domain."""
    computer, _ = _real_computer()
    _force_cross_domain(monkeypatch, computer)

    critic = _SpyCritic()
    real, fake = _pair()
    assert not real.is_complex() and not fake.is_complex()

    computer.compute_discriminator_loss(
        real=real, fake=fake, discriminator=critic, epoch=0, iteration=1
    )

    assert critic.seen, "the discriminator was never called -- nothing was verified"
    assert all(t.is_complex() for t in critic.seen), (
        "the critic declared it scores in k-space and was handed image-space "
        "tensors: it would train on a domain it never declared (#1921)"
    )


def test_the_generator_step_scores_the_converted_tensor(monkeypatch):
    """The G step feeds the critic too, through a different call site."""
    computer, _ = _real_computer()
    _force_cross_domain(monkeypatch, computer)

    critic = _SpyCritic()
    pred, target = _pair()
    computer.compute_generator_loss(
        pred=pred, target=target, discriminator=critic, epoch=99, iteration=99_999
    )

    assert critic.seen, "the discriminator was never called -- nothing was verified"
    assert all(t.is_complex() for t in critic.seen)


def test_the_reconstruction_loss_never_sees_the_converted_tensor(monkeypatch):
    """The defect a uniform rewrite would introduce, planted.

    L1 between an image-space prediction and its target is the generator's own
    objective. Routing it through the critic seam computes it in k-space instead
    -- a different loss surface, a different trained model, and no error anywhere.
    """
    computer, _ = _real_computer()
    _force_cross_domain(monkeypatch, computer)

    seen: list[torch.Tensor] = []

    class _SpyRec(nn.Module):
        """A Module, not a bare function, and that is not a style choice.

        ``reconstruction_loss_fn`` is a submodule slot on an ``nn.Module``, and
        ``nn.Module.__setattr__`` raises ``TypeError`` when a non-Module is
        assigned over one -- before any assertion below could run.
        """

        def forward(self, pred, target, *a, **k):
            seen.extend([pred, target])
            return torch.tensor(0.0, requires_grad=True)

    computer.reconstruction_loss_fn = _SpyRec()
    pred, target = _pair()
    computer.compute_generator_loss(
        pred=pred, target=target, discriminator=_SpyCritic(), epoch=99, iteration=99_999
    )

    assert seen, "the reconstruction loss never ran -- nothing was verified"
    assert not any(t.is_complex() for t in seen), (
        "the reconstruction loss was handed k-space tensors: L1/SSIM/LPIPS would "
        "be computed in the critic's domain, not the generator's (#1921)"
    )


# ---------------------------------------------------------------------------
# Logits scored elsewhere cannot be vouched for.
# ---------------------------------------------------------------------------


def test_pre_scored_logits_are_refused_when_a_conversion_is_owed(monkeypatch):
    """``discriminator_outputs=`` hands this computer scores it did not produce.

    On a cross-domain arm there is no way to tell which space they were scored
    in, and trusting them is how a critic ends up trained in one domain and
    scored in another. Refusing is the only honest answer (non-negotiable 3).
    """
    computer, _ = _real_computer()
    _force_cross_domain(monkeypatch, computer)

    # ``discriminator=`` is passed alongside the logits because that is what the
    # two production callers do (``strategies/gan.py`` and
    # ``strategies/mixins/adversarial.py``, which score ``fake_pred`` themselves
    # and hand both in). The refusal lives on the branch that CONSUMES the
    # logits, so a test omitting the discriminator would skip that branch and
    # pass for the wrong reason.
    pred, target = _pair()
    with pytest.raises(ValueError, match="discriminator_outputs"):
        computer.compute(
            pred=pred,
            target=target,
            epoch=99,
            iteration=99_999,
            discriminator=_SpyCritic(),
            discriminator_outputs={"fake_pred": torch.randn(2, 1)},
        )


def test_pre_scored_logits_are_not_refused_on_a_same_side_arm():
    """The negative control: the refusal above must not be a blanket ban.

    Without this, deleting the ``_critic_conversion is not None`` guard and
    refusing unconditionally would pass the test above while breaking all 23
    live arms -- every one of which is same-side and passes pre-scored logits.

    **Why this asserts a TypeError rather than a clean return.** The
    ``discriminator_outputs=`` path does not currently survive to a result on
    ANY arm, and that predates this branch: probed at ``origin/dev`` and at this
    PR's base, the production call shape (``strategies/gan.py`` and
    ``strategies/mixins/adversarial.py`` both build ``{"fake_pred": ...}`` and
    nothing else) reaches
    ``LSGANLoss.compute_discriminator_loss(real_outputs_d=None)`` and dies on
    ``None - 1.0``; supplying ``real_pred`` as well only gets as far as the
    ``g_adv_loss`` weight refusal. Both are filed separately.

    That pre-existing crash is what makes this a real oracle rather than a
    shrug. It happens **downstream** of the guard under test, so arriving at it
    proves execution passed the guard without being refused -- a positive
    observation, not an absence. If the crash is ever fixed this test goes red,
    which is the correct prompt to tighten it into a clean-return assertion.
    """
    computer, _ = _real_computer()
    assert computer._critic_conversion is None, "the default gan config is not same-side"

    pred, target = _pair()
    with pytest.raises(TypeError, match="NoneType"):
        computer.compute(
            pred=pred,
            target=target,
            epoch=99,
            iteration=99_999,
            discriminator=_SpyCritic(),
            discriminator_outputs={"fake_pred": torch.randn(2, 1)},
        )


# ---------------------------------------------------------------------------
# The trap this module's docstring names: ONE pair of parameter names, TWO
# opposite domain requirements.
# ---------------------------------------------------------------------------


def _spy_adversarial(monkeypatch, computer) -> tuple[dict, dict]:
    """Record what each adversarial method received, changing nothing else.

    The real ``CompositeGANLoss`` is kept and its two methods are wrapped, rather
    than the whole object being replaced by a double. That is deliberate: the
    D-side ``real_images``/``fake_images`` are spent on the gradient penalty and
    the G-side pair on L1/perceptual/SSIM, so a stand-in would have to reproduce
    both contracts to stay honest -- and the moment it drifted, this test would
    be pinning the double instead of the computer.

    Plain functions are assigned over the two *methods*. ``nn.Module.__setattr__``
    intercepts only ``Parameter``/``Module`` values and these names hold neither,
    so the assignment lands as an ordinary instance attribute (contrast
    ``reconstruction_loss_fn`` above, a submodule slot, which rejects a bare
    function).
    """
    fn = computer.adversarial_loss_fn
    g_seen: dict = {}
    d_seen: dict = {}
    original_g = fn.compute_generator_loss
    original_d = fn.compute_discriminator_loss

    def spy_g(**kwargs):
        g_seen.update(kwargs)
        return original_g(**kwargs)

    def spy_d(**kwargs):
        d_seen.update(kwargs)
        return original_d(**kwargs)

    monkeypatch.setattr(fn, "compute_generator_loss", spy_g)
    monkeypatch.setattr(fn, "compute_discriminator_loss", spy_d)
    return g_seen, d_seen


def _assert_opposition(g_seen: dict, d_seen: dict) -> None:
    """The generator pair stays raw; the discriminator pair is converted."""
    assert g_seen, "compute_generator_loss never ran -- nothing was verified"
    assert d_seen, "compute_discriminator_loss never ran -- nothing was verified"
    for key in ("real_images", "fake_images"):
        assert not g_seen[key].is_complex(), (
            f"compute_generator_loss received a converted {key}: it spends that "
            "on L1/perceptual/SSIM/LPIPS, so the generator's own objective would "
            "be computed in the critic's domain (#1921)"
        )
        assert d_seen[key].is_complex(), (
            f"compute_discriminator_loss received a raw {key}: it spends that on "
            "the gradient penalty, which calls the critic -- scoring it outside "
            "the domain the critic declared (#1921)"
        )


def test_the_two_identically_named_pairs_go_opposite_ways(monkeypatch):
    """The defect a uniform rewrite introduces, and the reason it is invisible.

    ``CompositeGANLoss`` names the same two arguments ``real_images``/
    ``fake_images`` on both methods, and the correct treatment is **opposite**:
    the D-side pair reaches the critic (convert), the G-side pair reaches the
    reconstruction losses (do not). Nothing in the signature says so, no type
    checker can tell them apart, and ``lsgan`` -- the default this fixture
    builds -- ignores both pairs entirely, so a wrong answer produces no error.

    Without this test the four ``_to_critic`` decisions at those call sites are
    asserted only by their own comments. A rewrite that converted all four would
    leave every other test in this file green.

    Both live methods are exercised: ``compute_generator_loss`` owns the raw
    pair, ``compute_discriminator_loss`` the converted one.
    """
    computer, _ = _real_computer()
    _force_cross_domain(monkeypatch, computer)
    g_seen, d_seen = _spy_adversarial(monkeypatch, computer)

    pred, target = _pair()
    computer.compute_generator_loss(
        pred=pred, target=target, discriminator=_SpyCritic(), epoch=99, iteration=99_999
    )
    computer.compute_discriminator_loss(
        real=target, fake=pred, discriminator=_SpyCritic(), epoch=99, iteration=99_999
    )

    _assert_opposition(g_seen, d_seen)


def test_the_same_opposition_holds_inside_compute(monkeypatch):
    """``compute`` carries the pair too, and there both halves sit in ONE method.

    ``compute_generator_loss`` and ``compute_discriminator_loss`` are far enough
    apart that their opposite treatment reads as deliberate. Inside ``compute``
    the two calls are forty lines apart with identical keyword names, which is
    where a uniform rewrite is most likely to land -- so it is pinned separately.

    **The absorbed error is pre-existing and measured, not a shrug.** With
    ``lambda_adv > 0`` this method raises ``ConfigurationError`` on
    ``g_adv_loss``: ``gan_loss_library.compute_generator_loss`` returns
    already-weighted sub-terms, ``components.update(g_adv)`` folds them in under
    their own names, and no ``lambda_g_adv_loss`` schema field exists to resolve
    them from. Reproduced identically on this branch and on its base, so it is
    filed rather than fixed here. It happens strictly **after** both spied calls,
    which is why the assertions below are still reached; the ``try`` (rather than
    ``pytest.raises``) means this test keeps testing what it names on the day
    that defect is fixed.
    """
    computer, _ = _real_computer()
    _force_cross_domain(monkeypatch, computer)
    g_seen, d_seen = _spy_adversarial(monkeypatch, computer)

    critic = _SpyCritic()
    pred, target = _pair()
    try:
        computer.compute(
            pred=pred,
            target=target,
            epoch=99,
            iteration=99_999,
            discriminator=critic,
        )
    except Exception as exc:  # documented above, and asserted on immediately
        assert "g_adv_loss" in str(exc), (
            "compute() failed for a reason this test did not predict, so the "
            f"spied tensors cannot be trusted: {type(exc).__name__}: {exc}"
        )

    # ``compute`` scores the critic ITSELF, on two lines no sibling test in this
    # file executes: ``compute_discriminator_loss``/``compute_generator_loss`` are
    # separate methods with their own critic calls. Reverting either of these two
    # to a raw feed left every other test in this file green, which is how the
    # gap was found (non-negotiable 15) rather than reasoned about.
    assert critic.seen, "compute() never scored the critic -- nothing was verified"
    assert all(t.is_complex() for t in critic.seen), (
        "compute() scored the critic on image-space tensors although it declared "
        "k-space: the on-the-fly branch bypassed the seam (#1921)"
    )

    _assert_opposition(g_seen, d_seen)


def test_a_same_side_complex_tensor_is_not_interleaved(monkeypatch):
    """``_to_critic``'s short-circuit is load-bearing, and this is the only case
    that can tell.

    Its docstring claims deleting the short-circuit "would be a NUMBER CHANGE on
    a same-side arm that feeds a complex tensor today", because ``to_critic_input``
    does two things: it converts domains *and*, when the critic does not accept
    complex, interleaves a complex tensor into 2C along the channel axis. With no
    conversion owed the first is a no-op and the second is not.

    Every other same-side assertion in this file passes a REAL tensor, for which
    the interleave branch is also a no-op -- so deleting the short-circuit left
    them all green. That was measured by planting it (non-negotiable 15), and it
    is why this test exists rather than the claim standing on its docstring
    (an invariant that is published must be executed).

    The oracle is shape as well as identity: interleaving 1 complex channel
    yields 2 real ones, so a regression changes what the critic is even shaped
    to receive.
    """
    computer, _ = _real_computer()
    monkeypatch.setattr(f"{UNIFIED_GAN}.generator_output_domain", lambda _c: "image")
    monkeypatch.setattr(f"{UNIFIED_GAN}.critic_input_domain", lambda _c: "image")
    monkeypatch.setattr(f"{UNIFIED_GAN}.critic_accepts_complex", lambda _c: False)
    computer._resolve_critic_seam()
    assert computer._critic_conversion is None, "fixture is not same-side"

    x = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    out = computer._to_critic(x)

    assert out is x, (
        "a same-side arm's complex tensor was rewritten: to_critic_input "
        "interleaved it into 2C, changing both the dtype and the channel count "
        "the critic receives, on an arm where nothing was owed (#1921)"
    )
    assert out.is_complex() and out.shape == x.shape
