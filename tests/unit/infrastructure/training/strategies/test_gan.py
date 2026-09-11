"""Regression: GANTrainingStrategy.train_step must thread the LIVE iteration.

Review finding (2026-07-01): ``train_step`` declares an explicit ``iteration``
parameter, so the training loop's ``strategy.train_step(..., iteration=iteration)``
call binds the real value to that parameter and leaves ``**kwargs`` empty. The
old ``iteration = kwargs.get("iteration", kwargs.get("step", 0))`` therefore
clobbered the correct value back to 0, silently degrading the R1-regularization
cadence (``UnifiedGANLossComputer._should_apply_r1``) from
every-``r1_interval``-iterations to every step of every ``r1_interval``-th epoch.

The fix reads the loop_state seam (``resolve_loop_iteration``) — the same source
already used elsewhere in the file — so the live iteration reaches the
discriminator/generator loss closures.
"""

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.loop_state import LoopState  # noqa: E402
from spectramr.infrastructure.training.strategies import gan as gan_mod  # noqa: E402
from spectramr.infrastructure.training.strategies.gan import (  # noqa: E402
    GANTrainingStrategy,
)


def test_train_step_uses_live_loop_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    # One discriminator update per step keeps the loop deterministic.
    monkeypatch.setattr(gan_mod, "_resolve_disc_updates", lambda cfg: 1)

    strategy = object.__new__(GANTrainingStrategy)
    strategy._step_counter = 0
    strategy.config = MagicMock()
    strategy.loop_state = LoopState(iteration=42)

    env = MagicMock()
    env.discriminator = MagicMock()
    env.generator = MagicMock()
    env.losses = {}
    strategy.env = env

    strategy._to_device = lambda x: x  # identity; inputs already on CPU

    captured: dict[str, int] = {}

    def _capture_disc(inp, tgt, disc, epoch, iteration, losses):  # noqa: ANN001
        captured["disc_iteration"] = iteration
        return lambda: {}

    def _capture_gen(inp, tgt, disc, epoch, iteration, losses):  # noqa: ANN001
        captured["gen_iteration"] = iteration
        return lambda: {}

    strategy._train_discriminator_step = _capture_disc
    strategy._train_generator_step = _capture_gen

    inp = torch.zeros(2, 1, 8, 8)
    tgt = torch.zeros(2, 1, 8, 8)
    # Pass a bogus iteration= keyword: it binds to the named parameter and must
    # be IGNORED in favour of the loop_state seam (the exact bug being fixed).
    strategy.train_step(None, 0, input_batch=inp, target_batch=tgt, iteration=999)

    assert captured["disc_iteration"] == 42, "R1 gate must see the live iteration"
    assert captured["gen_iteration"] == 42


# --- #707: the per-step metrics accessor must not sync the GPU ---------------


def test_get_last_metrics_does_not_sync_the_gpu(no_gpu_sync):
    from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy

    no_gpu_sync(GANTrainingStrategy.get_last_metrics)


# --- the strategy must not score the critic itself ---------------------------
#
# ``_train_generator_step`` / ``_train_discriminator_step`` used to guard the
# loss-computer hooks with ``hasattr(...) else None`` and, on the None branch,
# feed the critic themselves. That branch was unreachable -- ``setup_adversarial``
# installs ``UnifiedGANLossComputer`` unconditionally and it defines both hooks --
# but while it existed it was a second, divergent critic feed (this file
# realified a complex fake, ``AdversarialMixin`` handed it over raw), and with no
# critic it silently replaced the adversarial objective with plain L1 (NN3).


def _critic_calls_in(func) -> list[int]:
    """Lines where ``func`` CALLS the critic rather than delegating it.

    AST, not source text: a comment or docstring naming ``discriminator(`` must
    not be able to satisfy -- or break -- this pin.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in {"discriminator", "critic"}:
            hits.append(node.lineno)
        elif isinstance(fn, ast.Attribute) and fn.attr in {
            "discriminator",
            "discriminator_model",
        }:
            hits.append(node.lineno)
    return hits


def test_generator_step_delegates_the_critic_instead_of_scoring_it():
    assert _critic_calls_in(GANTrainingStrategy._train_generator_step) == []


def test_discriminator_step_delegates_the_critic_instead_of_scoring_it():
    assert _critic_calls_in(GANTrainingStrategy._train_discriminator_step) == []


def test_the_installed_computer_defines_the_hooks_the_steps_now_call_unguarded():
    """What licenses deleting the guards: the computer always has both hooks."""
    from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer

    assert hasattr(UnifiedGANLossComputer, "compute_generator_loss")
    assert hasattr(UnifiedGANLossComputer, "compute_discriminator_loss")


def test_every_gan_subclass_reaches_the_unconditional_setup_that_installs_it():
    """The guards were dead only because no subclass bypasses ``setup_adversarial``.

    A subclass that overrode ``_setup_strategy_specific_components`` without
    calling it would install a different computer and hit the deleted branch, so
    this is the precondition, not a restatement of the fix.
    """
    import inspect

    from spectramr.infrastructure.training.strategies.betavaegan_strategy import (
        BetaVAEGANStrategy,
    )
    from spectramr.infrastructure.training.strategies.progressive_gan_strategy import (
        ProgressiveGANStrategy,
    )

    for cls in (GANTrainingStrategy, BetaVAEGANStrategy, ProgressiveGANStrategy):
        src = inspect.getsource(cls._setup_strategy_specific_components)
        assert "setup_adversarial(" in src, cls.__name__
