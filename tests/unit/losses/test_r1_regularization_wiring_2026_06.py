"""Regression: R1 regularization must actually fire, and apply lambda_r1 once.

Two bugs in ``UnifiedGANLossComputer``:

* ``_should_apply_r1`` did ``hasattr(self.config, "r1_interval")`` on the full
  ``TrainingSettings`` — but the R1 knobs live at ``losses.gan.r1_interval`` /
  ``losses.gan.enable_r1``, so the gate was always False and R1 never fired
  (CLAUDE.md pitfall #15: advertised-but-unread knob).
* Behind that gate, the discriminator path multiplied the R1 penalty by
  ``lambda_r1`` even though the R1 module already bakes ``lambda_r1`` into its
  output, and ``_stack_components`` weighted the ``r1`` key by ``lambda_r1`` a
  third time → a ``lambda_r1**3`` (~1000x at the default 10) over-penalty.
"""

from __future__ import annotations

import types

import torch

from spectramr.models.losses.computers.unified_gan import UnifiedGANLossComputer


class _GanLosses:
    """Fake ``config.losses`` exposing both the ``.gan`` attribute (used by the
    R1 gate) and ``model_dump()`` (used by ``_get_loss_weight``)."""

    def __init__(self, **gan):
        self._gan = gan

    @property
    def gan(self):
        return types.SimpleNamespace(**self._gan)

    def model_dump(self):
        return {"gan": dict(self._gan)}


def _config(**gan):
    return types.SimpleNamespace(losses=_GanLosses(**gan))


def test_r1_gate_reads_nested_gan_config():
    c = object.__new__(UnifiedGANLossComputer)
    c.config = _config(r1_interval=16, enable_r1=True, lambda_r1=10.0)
    assert c._should_apply_r1(epoch=0, iteration=16) is True
    assert c._should_apply_r1(epoch=0, iteration=15) is False


def test_r1_disabled_when_enable_flag_false():
    c = object.__new__(UnifiedGANLossComputer)
    c.config = _config(r1_interval=16, enable_r1=False, lambda_r1=10.0)
    assert c._should_apply_r1(epoch=0, iteration=16) is False


def test_r1_penalty_applied_once_not_cubed():
    c = object.__new__(UnifiedGANLossComputer)
    c.config = _config(r1_interval=1, enable_r1=True, lambda_r1=10.0)
    c.device = torch.device("cpu")
    c.adversarial_loss_fn = None
    # Stand-in for the real R1 module, whose output already includes lambda_r1.
    # ``**_`` is load-bearing, not decoration: the production call site forwards
    # ``critic_cond=`` to the regularizer, and this double pinned the older
    # two-positional-arg signature -- so it raised ``TypeError: got an
    # unexpected keyword argument 'critic_cond'`` and had stopped testing the
    # arithmetic it names. Absorbing the kwargs restores the assertion below.
    c.r1_regularizer = lambda disc, real, **_: torch.tensor(2.0)

    out = c.compute_discriminator_loss(
        real=torch.randn(1, 1, 4, 4),
        fake=torch.randn(1, 1, 4, 4),
        discriminator=torch.nn.Identity(),
        epoch=0,
        iteration=1,
    )

    # The baked penalty (2.0) must be applied exactly once. The pre-fix code
    # produced lambda_r1**2 * 2.0 = 200 (or 0 because the gate never opened).
    assert torch.allclose(out.total, torch.tensor(2.0)), out.total
