"""#1950 -- ``UnifiedGANLossComputer.compute`` folded pre-weighted terms raw.

``compute`` and ``compute_generator_loss`` both fold a generator adversarial
result into ``components``. They had drifted into different answers:

* ``compute_generator_loss`` absorbed a library dict via ``_absorb_preweighted``
  (excluding its keys from the stack, adding its authoritative total once) and
  applied ``lambda_adv`` to the tensor form.
* ``compute`` did ``components.update(g_adv)`` for the dict, and stored the
  tensor form **unweighted** under ``adv_generator``.

Both of ``compute``'s branches were wrong, and neither failed quietly at the
end: ``_stack_components`` asks ``resolve_loss_weight`` for a weight per key,
and neither ``g_adv_loss`` (dict) nor ``adv_generator`` (tensor) has a
``lambda_<name>`` schema field, so it refused to invent one and RAISED. The
raise is reachable only past warm-up -- below ``warmup_iterations`` the
``lambda_adv > 0`` guard never opens and the whole block is skipped -- which is
why nothing caught it: no test drove ``compute`` on a GAN config past warm-up.

The third half, which the issue does not name: ``compute`` spent ``iteration``
on every ``_get_loss_weight`` call and then dropped it at the final
``_stack_components``, so the stacked terms alone resolved their weights at step
0 forever. A warm-up-gated term therefore appeared in ``components`` with its
real value while contributing exactly nothing to ``total``.

**Reachability, stated rather than assumed.** ``LossBuilder`` has exactly one
assignment to ``_losses["adversarial"]`` and it always builds ``gan_composite``,
so in production ``adversarial_loss_fn`` is always a ``CompositeGANLoss`` and
always returns a dict. The tensor branch is reachable by direct assignment
(tests, scripting) only -- it is covered here because it was wrong, not because
an arm takes it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


class _Constant(nn.Module):
    """A grad-connected constant, so ``total`` keeps a graph to assert on."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return pred.sum() * 0.0 + self._value


class _TinyCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return self.conv(x).mean(dim=(1, 2, 3), keepdim=True)


class _TensorAdversarial(nn.Module):
    """The shape ``StandardGANLoss`` / ``LSGANLoss`` / ``WGANLoss`` / ``HingeLoss``
    / ``RALSGANLoss`` return: a bare tensor that still owes ``lambda_adv``."""

    def compute_generator_loss(self, fake_outputs_d: torch.Tensor, **_kw: object):
        return fake_outputs_d.mean() * 0.0 + 4.0


def _settings(losses: dict):
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings.settings_from_dict(
        {
            "model": {"model_type": "unet"},
            "data": {"dataset_type": "synthetic"},
            "optimization": {},
            "logging": {},
            "losses": {"output_domain": "image", **losses},
        }
    )


def _gan_settings():
    """A GAN arm: ``CompositeGANLoss``, no perceptual (so no VGG in a unit test)."""
    return _settings(
        {
            "image_losses": [{"name": "mse", "weight": 1.0}],
            "gan": {
                "enable_adversarial": True,
                "lambda_adv": 1.0,
                "gan_loss_type": "lsgan",
                "enable_gradient_penalty": False,
            },
        }
    )


def _perceptual_settings():
    """No adversarial block at all, so the ONLY thing under test is the stack.

    ``perceptual`` is in ``LEGACY_WARMUP_LOSSES`` and lands in ``components``
    RAW, so its weight is resolved inside ``_stack_components`` -- which is
    precisely the call that dropped ``iteration``. The weight is **3.7**, not
    the ``lambda_perceptual`` schema default of 10.0, so agreement cannot come
    from a coincident default.
    """
    return _settings(
        {
            "image_losses": [
                {"name": "mse", "weight": 1.0},
                {"name": "perceptual", "weight": 3.7, "enabled": True},
            ]
        }
    )


def _computer(settings):
    from spectramr.models.losses.computers import UnifiedGANLossComputer

    computer = UnifiedGANLossComputer(config=settings, device=torch.device("cpu"))
    # Stand-ins with known values. The WEIGHTS stay the config's real ones --
    # only the loss magnitudes are pinned, so the arithmetic below is checkable.
    computer.reconstruction_loss_fn = _Constant(5.0)
    return computer


def _pred_and_target():
    generator = nn.Conv2d(1, 1, 3, padding=1)
    return generator(torch.rand(2, 1, 16, 16)), torch.rand(2, 1, 16, 16)


# --------------------------------------------------------------------------
# The premise. If these stop holding, every assertion below loses its point.
# --------------------------------------------------------------------------


def test_the_warmup_gate_is_what_hid_this():
    """Below ``warmup_iterations`` the adversarial block is skipped entirely."""
    computer = _computer(_gan_settings())
    assert computer._get_loss_weight("adversarial", 0, 0) == 0.0
    assert computer._get_loss_weight("adversarial", 0, 5000) > 0.0


def test_perceptual_resolves_to_the_declared_non_default_weight():
    from spectramr.models.losses.weights import LEGACY_WARMUP_LOSSES

    computer = _computer(_perceptual_settings())
    assert "perceptual" in LEGACY_WARMUP_LOSSES
    assert computer._get_loss_weight("perceptual", 0, 0) == 0.0
    assert computer._get_loss_weight("perceptual", 0, 5000) == pytest.approx(3.7)


# --------------------------------------------------------------------------
# The library-dict fold.
# --------------------------------------------------------------------------


def test_compute_past_warmup_no_longer_raises_on_a_library_key():
    """On ``dev`` this raised ``ConfigurationError`` on ``g_adv_loss``."""
    computer = _computer(_gan_settings())
    pred, target = _pred_and_target()

    output = computer.compute(
        pred=pred, target=target, epoch=0, iteration=5000, discriminator=_TinyCritic()
    )

    assert "g_total_loss" in output.components
    assert "g_adv_loss" in output.components


def test_the_library_total_is_added_exactly_once():
    """The whole point of absorbing: sub-terms reach ``components`` for logging
    but must not also be summed beside the authoritative total they add up to."""
    computer = _computer(_gan_settings())
    pred, target = _pred_and_target()

    output = computer.compute(
        pred=pred, target=target, epoch=0, iteration=5000, discriminator=_TinyCritic()
    )

    library_total = float(output.components["g_total_loss"])
    # ``reconstruction`` is the only term the stack owns here: weight 1.0 on a
    # stand-in pinned at 5.0.
    assert float(output.total) == pytest.approx(5.0 + library_total, rel=1e-5)

    # Falsify the double-count directly: the sub-terms are non-trivial, so a
    # total that had summed them too would be strictly larger.
    assert float(output.components["l1_loss"]) > 0.0
    subterm_sum = float(output.components["l1_loss"]) + float(output.components["g_adv_loss"])
    assert library_total == pytest.approx(subterm_sum, rel=1e-5)
    assert float(output.total) != pytest.approx(5.0 + library_total + subterm_sum, rel=1e-5)


def test_discriminator_terms_are_reported_but_never_summed_into_the_generator_total():
    computer = _computer(_gan_settings())
    pred, target = _pred_and_target()

    output = computer.compute(
        pred=pred, target=target, epoch=0, iteration=5000, discriminator=_TinyCritic()
    )

    assert "d_d_total_loss" in output.components, "still reported"
    assert "d_d_total_loss" in output.metrics
    assert float(output.components["d_d_total_loss"]) > 0.0
    # ...and absent from the total, which is what backward() runs on for G.
    assert float(output.total) == pytest.approx(
        5.0 + float(output.components["g_total_loss"]), rel=1e-5
    )


def test_the_total_stays_grad_connected():
    """PR #1993 (#1952) makes a severed total raise at ``backward()``; the fold
    must not produce one."""
    computer = _computer(_gan_settings())
    pred, target = _pred_and_target()

    output = computer.compute(
        pred=pred, target=target, epoch=0, iteration=5000, discriminator=_TinyCritic()
    )

    assert output.total.grad_fn is not None


# --------------------------------------------------------------------------
# The tensor fold (test/scripting-reachable -- see the module docstring).
# --------------------------------------------------------------------------


def test_a_tensor_adversarial_result_is_weighted_once_under_the_shared_key():
    computer = _computer(_gan_settings())
    computer.adversarial_loss_fn = _TensorAdversarial()
    pred, target = _pred_and_target()

    output = computer.compute(
        pred=pred, target=target, epoch=0, iteration=5000, discriminator=_TinyCritic()
    )

    # The name ``compute_generator_loss`` already used. ``adv_generator`` had no
    # ``lambda_adv_generator`` schema field, so the stack raised on it.
    assert "adv_generator" not in output.components
    lambda_adv = computer._get_loss_weight("adversarial", 0, 5000)
    assert float(output.components["adversarial"]) == pytest.approx(4.0 * lambda_adv)
    assert float(output.total) == pytest.approx(5.0 + 4.0 * lambda_adv, rel=1e-5)


# --------------------------------------------------------------------------
# The dropped ``iteration`` inside ``compute`` itself.
# --------------------------------------------------------------------------


def test_the_stack_resolves_weights_at_the_live_iteration():
    """``compute`` spent ``iteration`` on every gate, then withheld it from the
    stack -- so a gated term sat in ``components`` at full value and
    contributed 0.0 to ``total``. One value, two answers, inside one method."""
    computer = _computer(_perceptual_settings())
    computer.perceptual_loss_fn = _Constant(2.0)
    pred, target = _pred_and_target()

    output = computer.compute(pred=pred, target=target, epoch=0, iteration=5000)

    assert float(output.components["perceptual"]) == pytest.approx(2.0)
    # 1.0 * 5.0 + 3.7 * 2.0. Frozen at iteration 0 the second term is 0.0 and
    # the total reads 5.0 while ``components`` still advertises perceptual.
    assert float(output.total) == pytest.approx(12.4, rel=1e-5)


def test_during_warmup_the_gated_term_is_absent_not_zero():
    """The complement, and the reason this is silent: a gated loss does not
    appear scaled to zero, it does not appear at all."""
    computer = _computer(_perceptual_settings())
    computer.perceptual_loss_fn = _Constant(2.0)
    pred, target = _pred_and_target()

    output = computer.compute(pred=pred, target=target, epoch=0, iteration=0)

    assert "perceptual" not in output.components
    assert float(output.total) == pytest.approx(5.0)
