"""The GAN computer's G-side methods must forward critic conditioning too.

``UnifiedGANLossComputer.compute_discriminator_loss`` accepts ``critic_cond`` and
forwards it to every critic call it makes -- real, fake, gradient-penalty
interpolate (#1931). Its two GENERATOR-side entry points, ``compute`` and
``compute_generator_loss``, called the same critic bare:

* ``compute``  -- ``discriminator(pred.detach())`` / ``discriminator(target)``
  for the on-the-fly outputs, and ``self.r1_regularizer(discriminator, target)``;
* ``compute_generator_loss`` -- ``discriminator(pred)``.

One class, two answers to "does this critic take conditioning" (non-negotiable
17). The asymmetry is what these tests pin.

**This is a consistency fix, not a live-bug fix, and the tests are written to
say so.** A conditioned critic reached bare on this path does not train
unconditioned -- it RAISES. Reproduced against the real producer:

    ContrastConditionedSenseBridgeDiscriminator was called without `timesteps`.
    This model declares supports_contrast_conditioning and scores a t-labelled
    distribution; there is no meaningful unconditioned behaviour to fall back to.

so non-negotiable 3 is already satisfied and no shipped arm is mis-training. The
defect these close is the SIGNATURE one: both methods absorb stray keywords into
``**kwargs``, so a caller that passed ``critic_cond`` had it accepted and
silently dropped. That is the shape that turns into a wrong number rather than
an error the moment a caller acquires a payload to pass -- and one already
exists in reach: ``diffusion.py`` constructs a ``UnifiedGANLossComputer`` inside
a strategy that holds ``timesteps`` and ``contrast_idx``, and already passes
``critic_cond`` to its D-side method.

Lives beside ``test_unified_gan.py`` rather than inside it. Both suites were
written against the same class on two branches and collided as an add/add on
2026-09-08; the canonical name keeps the #1921 domain-seam suite (it is the
pairing-gate partner for ``unified_gan.py``), and this file keeps the #1931
conditioning suite. Fusing them would have produced an ~800-line test module
and a helper collision (both defined ``_pair`` with different meanings).

The two are complementary at the same call sites: the seam suite asserts WHICH
DOMAIN the critic was fed, this one asserts WHAT CONDITIONING came with it. The
merge resolution passes both -- ``discriminator(self._to_critic(x), **cond)``.

Every assertion is on what the CRITIC RECEIVED, not on what was passed in: a
test that checks the call was made passes against a signature that drops the
payload, which is the exact defect.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


class _RecordingCritic(nn.Module):
    """Scores like a critic; remembers the keyword payload of every call.

    A real ``nn.Module`` rather than a stub or a Mock: ``R1RegularizationLoss``
    opens with ``if not isinstance(discriminator, nn.Module): return 0.0``, so a
    non-Module spy would make the R1 leg pass vacuously by never reaching the
    critic at all.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.calls: list[dict] = []

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self.calls.append(dict(kwargs))
        return self.conv(x)


COND = {"timesteps": torch.tensor([3, 7]), "contrast_idx": torch.tensor([1, 0])}


def _same(seen: dict, expected: dict) -> bool:
    if set(seen) != set(expected):
        return False
    return all(torch.equal(seen[k], expected[k]) for k in expected)


@pytest.fixture
def gan_computer():
    from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer
    from spectramr.pipelines.fit import _resolve_fit_config

    config = _resolve_fit_config(None, paradigm="gan", epochs=1, max_iterations=2)
    return UnifiedGANLossComputer(config, torch.device("cpu")), config


class _ScoresNothing(nn.Module):
    """An ``adversarial_loss_fn`` that opens the branch and delegates nothing.

    An ``nn.Module`` because the computer registers ``adversarial_loss_fn`` as a
    submodule and ``nn.Module.__setattr__`` refuses anything else.

    ``compute``'s adversarial branch does two things: it calls the critic on
    both sides (the seam under test), then hands the outputs to
    ``adversarial_loss_fn``. Substituting an object with neither
    ``compute_generator_loss`` nor ``compute_discriminator_loss`` keeps the
    first and skips the second, so these tests fail only for the reason they
    name.

    That substitution is load-bearing, not cosmetic: with the real
    ``CompositeGANLoss`` this branch raises ``ConfigurationError: Loss
    'g_adv_loss' is active but its weight is declared nowhere``, because
    ``compute`` folds the library's pre-weighted dict straight into
    ``components`` instead of routing it through ``_absorb_preweighted`` the way
    ``compute_generator_loss`` (unified_gan.py:412) and
    ``compute_discriminator_loss`` (:565) both do. That is a separate defect
    from the one under test here and is reported separately; pinning it from
    this file would make these tests assert two unrelated things at once.
    """


def _pair(size: int = 64):
    return torch.randn(2, 1, size, size), torch.randn(2, 1, size, size)


# --------------------------------------------------------------------------
# compute_generator_loss
# --------------------------------------------------------------------------


def test_compute_generator_loss_forwards_critic_cond(gan_computer):
    """REGRESSION. Red before the fix: ``**kwargs`` swallowed the payload.

    Pre-fix ``critic_cond`` was not a named parameter, so it landed in
    ``**kwargs``, was never read, and ``discriminator(pred)`` ran bare -- the
    call succeeded and the critic saw ``{}``.
    """
    computer, config = gan_computer
    critic = _RecordingCritic()
    x, y = _pair()

    computer.compute_generator_loss(
        pred=x,
        target=y,
        discriminator=critic,
        epoch=0,
        iteration=config.losses.reconstruction.warmup_iterations + 1,
        critic_cond=COND,
    )

    assert critic.calls, "the adversarial branch never ran -- the test is vacuous"
    assert all(_same(c, COND) for c in critic.calls), (
        f"critic received {critic.calls} -- conditioning was dropped"
    )


def test_compute_generator_loss_without_cond_is_unchanged(gan_computer):
    """The no-op leg. Every caller in the tree today passes nothing."""
    computer, config = gan_computer
    critic = _RecordingCritic()
    x, y = _pair()

    computer.compute_generator_loss(
        pred=x,
        target=y,
        discriminator=critic,
        epoch=0,
        iteration=config.losses.reconstruction.warmup_iterations + 1,
    )

    assert critic.calls, "the adversarial branch never ran -- the test is vacuous"
    assert all(c == {} for c in critic.calls), (
        f"an unconditioned call must stay byte-identical, got {critic.calls}"
    )


def test_compute_generator_loss_names_critic_cond_explicitly():
    """The signature IS the fix: ``**kwargs`` accepting-and-dropping is the defect.

    Pinned by name because a "simplification" back to a ``kwargs.get`` read here
    would restore silent acceptance while every value assertion above stayed
    green -- the payload would still arrive, just via a route that a typo or a
    rename makes invisible. ``compute_discriminator_loss`` is pinned alongside
    it so the two halves of this class cannot drift apart again.
    """
    import inspect

    from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer

    for name in ("compute_generator_loss", "compute_discriminator_loss"):
        params = inspect.signature(getattr(UnifiedGANLossComputer, name)).parameters
        assert "critic_cond" in params, f"{name} does not name critic_cond"
        assert params["critic_cond"].default is None, (
            f"{name}'s critic_cond must default to None so existing callers are unaffected"
        )


# --------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------


def test_compute_forwards_critic_cond_to_both_on_the_fly_calls(gan_computer):
    """REGRESSION. ``compute`` scores BOTH sides itself when no outputs are supplied.

    Both calls are asserted: conditioning one side and not the other would let
    the critic separate real from fake by reading the label rather than the
    image, which is worse than conditioning neither.
    """
    computer, config = gan_computer
    critic = _RecordingCritic()
    x, y = _pair()

    computer.adversarial_loss_fn = _ScoresNothing()

    computer.compute(
        pred=x,
        target=y,
        epoch=0,
        iteration=config.losses.reconstruction.warmup_iterations + 1,
        discriminator=critic,
        critic_cond=COND,
    )

    assert len(critic.calls) >= 2, (
        f"expected the fake and real calls; saw {len(critic.calls)} -- test is vacuous"
    )
    assert all(_same(c, COND) for c in critic.calls), (
        f"critic received {critic.calls} -- conditioning was dropped"
    )


def test_compute_without_cond_is_unchanged(gan_computer):
    """The no-op leg for ``compute``."""
    computer, config = gan_computer
    critic = _RecordingCritic()
    x, y = _pair()

    computer.adversarial_loss_fn = _ScoresNothing()

    computer.compute(
        pred=x,
        target=y,
        epoch=0,
        iteration=config.losses.reconstruction.warmup_iterations + 1,
        discriminator=critic,
    )

    assert len(critic.calls) >= 2, "the adversarial branch never ran -- the test is vacuous"
    assert all(c == {} for c in critic.calls), (
        f"an unconditioned call must stay byte-identical, got {critic.calls}"
    )


# --------------------------------------------------------------------------
# R1
# --------------------------------------------------------------------------


def test_compute_forwards_critic_cond_to_r1(gan_computer, monkeypatch):
    """REGRESSION, and the leg most likely to stay dark.

    R1 differentiates THROUGH the critic call, so an unconditioned call here
    regularizes a gradient the critic never takes -- a wrong penalty rather than
    a missing one. The branch is gated on ``_should_apply_r1``, which is False
    for most (epoch, iteration) pairs, so it is forced open here rather than
    hoped for; ``assert critic.calls`` below is what proves the forcing worked.
    """
    from spectramr.models.losses.gan_loss_library import R1RegularizationLoss

    computer, config = gan_computer
    critic = _RecordingCritic()
    x, y = _pair()

    computer.r1_regularizer = R1RegularizationLoss(weight=10.0)
    monkeypatch.setattr(computer, "_should_apply_r1", lambda *a, **k: True)
    # Silence the adversarial branch so every recorded call is R1's -- otherwise
    # section 2's two calls would satisfy the assertion below on their own and
    # the R1 leg could regress unnoticed.
    computer.adversarial_loss_fn = None

    out = computer.compute(
        pred=x,
        target=y,
        epoch=0,
        iteration=config.losses.reconstruction.warmup_iterations + 1,
        discriminator=critic,
        critic_cond=COND,
    )

    assert "r1_penalty" in out.components, (
        f"the R1 branch never fired; components={sorted(out.components)}"
    )
    assert len(critic.calls) == 1, (
        f"expected exactly R1's single critic call, saw {len(critic.calls)}"
    )
    assert all(_same(c, COND) for c in critic.calls), (
        f"critic received {critic.calls} -- R1 dropped the conditioning"
    )


# --------------------------------------------------------------------------
# ``critic_cond`` is read with ``get``, so it stays in ``**kwargs``
# --------------------------------------------------------------------------


def test_critic_cond_left_in_kwargs_cannot_break_a_losses_dict_entry():
    """PIN on the PREMISE of the ``get``-not-``pop`` read in ``compute``.

    ``compute`` reads ``critic_cond`` with ``get``, so the key survives in
    ``**kwargs``, which ``compute`` forwards wholesale to ``_call_safe_loss``
    for every ``losses_dict`` entry (``unified_gan.py:323`` and ``:458``). That
    is only safe because ``_call_safe_loss`` filters kwargs against the callee's
    signature -- and *that* is what this test pins.

    **Scope, stated honestly: this does not detect a change of ``get`` to
    ``pop``.** Pinning the read itself needs a ``losses_dict`` entry that
    survives ``compute``'s skip list AND carries a declared weight; the default
    GAN config declares only ``adversarial`` and ``l1``, and ``l1`` is skipped
    ("Handled by reconstruction"), so such a case has to be manufactured from a
    bespoke config. What is pinned instead is the property that makes the
    forward harmless, which is the half that can silently stop being true.

    The consequence of getting this wrong is not a visible crash. Both
    ``_call_safe_loss`` call sites sit inside a bare ``try/except`` that
    swallows any loss exception and skips the term -- the behaviour
    ``test_dynamic_loss_dispatch.py::test_unified_gan_must_not_silently_drop_a_failing_loss``
    carries as a known xfail (issue #289). A ``TypeError: unexpected keyword
    argument 'critic_cond'`` would therefore delete a reconstruction term from
    the objective and log nothing, which is why the filtering is pinned here
    rather than left to the comment that asserts it.

    Resolved through the ``unified_gan`` module object on purpose: this pins the
    exact function ``compute`` will call, so swapping the import for a helper
    that does not filter goes red.
    """
    from spectramr.models.losses.computers import unified_gan

    x, y = _pair()
    cond = {"timesteps": torch.tensor([3, 7])}

    # (i) a callee that does NOT declare it: the key must be filtered away.
    def plain_loss(pred, target):
        return (pred - target).abs().mean()

    val = unified_gan._call_safe_loss(plain_loss, x, y, critic_cond=cond)
    assert torch.isfinite(val), "the loss did not compute"

    # (ii) a callee that declares ``**kwargs``: it receives the key, exactly as
    # it did before ``critic_cond`` existed. Popping would change this.
    seen: dict = {}

    def varkw_loss(pred, target, **kwargs):
        seen.update(kwargs)
        return (pred - target).abs().mean()

    unified_gan._call_safe_loss(varkw_loss, x, y, critic_cond=cond)
    assert "critic_cond" in seen, (
        "a ``**kwargs`` callee stopped receiving the payload -- the filter now "
        "drops keys it used to forward, which is a behaviour change for every "
        "signature-aware loss, not just this one"
    )
