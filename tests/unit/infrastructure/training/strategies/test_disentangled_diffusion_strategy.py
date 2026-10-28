"""Unit tests for DisentangledDiffusionStrategy.

Regression coverage for SGV-001: the validation entry point must be named
``validation_step`` (not ``validate_step``) so the training engine actually
dispatches into the full DDPM reverse-sampling path with Bloch DC. When the
method was named ``validate_step`` the engine silently fell through to the
mixin's generic metric-only ``validation_step``, leaving the DDPM sampling
logic permanently unreachable.
"""

from unittest.mock import MagicMock

import torch

from spectramr.infrastructure.training.strategies.disentangled_diffusion_strategy import (
    DisentangledDiffusionStrategy,
)
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
    MetricsMixin,
)


def _make_strategy() -> DisentangledDiffusionStrategy:
    """Build a DisentangledDiffusionStrategy without the heavy base ``__init__``.

    ``object.__new__`` skips ``BaseTrainingStrategy.__init__`` (which needs a DI
    container / real models); we wire only the attributes ``validation_step``
    touches.
    """
    strategy = object.__new__(DisentangledDiffusionStrategy)
    strategy.device = torch.device("cpu")

    # Generator: tracks ddpm_sample invocation and returns a tissue dict.
    generator = MagicMock()
    # A bare module: unwrap_model must stop here, not descend into an auto-created mock child.
    for attr in ("module", "_orig_mod", "_fsdp_wrapped_module", "_checkpoint_wrapped_module"):
        setattr(generator, attr, None)
    generator.tissue_channels = 6
    generator.ddpm_sample = MagicMock(
        return_value={"rho": torch.zeros(2, 6, 8, 8)}
    )

    env = MagicMock()
    env.generator = generator
    env.losses = {}
    strategy.env = env

    # Bloch layer maps the tissue dict back to an image matching the target.
    strategy._bloch_layer = MagicMock(return_value=torch.zeros(2, 1, 8, 8))

    # Loss computer returns a scalar-bearing object with ``.total``.
    loss_computer = MagicMock()
    loss_computer.compute = MagicMock(
        return_value=MagicMock(total=torch.tensor(0.25))
    )
    strategy.loss_computer = loss_computer

    return strategy


def test_validation_step_is_the_public_name():
    """SGV-001: the strategy must expose ``validation_step`` (engine contract).

    The old name ``validate_step`` was never called by the engine, so the
    DDPM-sampling override was dead code. After the rename the strategy's own
    method must override the mixin default — i.e. it must not be the inherited
    ``metrics_mixin.MetricsMixin.validation_step``.
    """
    assert hasattr(DisentangledDiffusionStrategy, "validation_step")
    # Old (broken) name must be gone.
    assert not hasattr(DisentangledDiffusionStrategy, "validate_step")
    # The strategy's own method overrides the generic mixin default.
    assert (
        DisentangledDiffusionStrategy.validation_step
        is not MetricsMixin.validation_step
    )


def test_validation_step_reaches_ddpm_sampling():
    """SGV-001: calling ``validation_step`` reaches the DDPM sampling path."""
    strategy = _make_strategy()
    batch = {
        "input": torch.zeros(2, 1, 8, 8),
        "target": torch.zeros(2, 1, 8, 8),
    }

    metrics = strategy.validation_step(batch)

    strategy.env.generator.ddpm_sample.assert_called_once()
    strategy._bloch_layer.assert_called_once()
    assert "val_mse" in metrics
    assert "val_psnr" in metrics


def test_compute_losses_impl_uses_live_loop_iteration() -> None:
    """Regression (2026-07-01): ``_compute_losses_impl`` must forward the LIVE
    ``loop_state.iteration`` to ``compute_generator_loss`` — NOT the old
    ``kwargs.get("step", 0)``, which read a key the training loop never sets and
    so forced iteration=0 every step, firing the Bloch-DC throttle
    (``iteration % _bloch_dc_interval == 0``) on every step instead of every N.
    """
    from spectramr.infrastructure.training.loop_state import LoopState

    strategy = object.__new__(DisentangledDiffusionStrategy)
    strategy.device = torch.device("cpu")
    strategy.loop_state = LoopState(iteration=7)

    captured: dict[str, object] = {}

    def _capture(batch: object, **kwargs: object) -> MagicMock:
        captured["iteration"] = kwargs.get("iteration")
        result = MagicMock()
        result.losses = {}
        result.g_total_loss = torch.zeros(())
        return result

    strategy.compute_generator_loss = _capture

    inp = torch.zeros(2, 1, 8, 8)
    tgt = torch.zeros(2, 1, 8, 8)
    # A bogus ``step`` kwarg must be IGNORED in favour of the loop_state seam.
    strategy._compute_losses_impl(
        input_batch=inp, target_batch=tgt, epoch=0, step=999
    )

    assert captured["iteration"] == 7, "must read loop_state.iteration, not 0 or step"
