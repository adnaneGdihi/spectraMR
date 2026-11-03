"""Regression: unified GAN/VAE/disentangled computers must fail loud on a
LossBuilder error, not silently collapse to a bare L1 stack.

Review finding (2026-07-01): all three computers wrapped
``LossBuilder(...).build()`` in a bare ``except Exception`` and fell through to a
minimal L1-only (or L1+MSE+KL) loss dict on ANY failure — including
LossBuilder's own fail-fast ``ConfigurationError("Unknown gan_loss_type")``.
A typo in ``losses.gan.gan_loss_type`` (or a missing VGG weight, a bad loss
sub-block) therefore trained the whole GAN family on vanilla L1 while logging
identically to a correct adversarial run (pitfall #9/#16).

The ONLY sanctioned fallback is "no config supplied" (``config is None``); a
build failure with a real config must propagate.
"""

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.builders import (  # noqa: E402
    loss_builder as loss_builder_mod,
)
from spectramr.models.losses.computers.unified_disentangled import (  # noqa: E402
    UnifiedDisentangledLossComputer,
)
from spectramr.models.losses.computers.unified_gan import (  # noqa: E402
    UnifiedGANLossComputer,
)
from spectramr.models.losses.computers.unified_vae import (  # noqa: E402
    UnifiedVAELossComputer,
)


class _ExplodingLossBuilder:
    """Stand-in for LossBuilder whose first build step fails, exactly as the
    real one raises ``ConfigurationError`` on an unknown ``gan_loss_type``."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def build_reconstruction_losses(self) -> "_ExplodingLossBuilder":
        raise RuntimeError("boom: simulated unknown gan_loss_type")


@pytest.fixture()
def _explode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loss_builder_mod, "LossBuilder", _ExplodingLossBuilder)


@pytest.mark.parametrize(
    "computer_cls",
    [UnifiedGANLossComputer, UnifiedVAELossComputer, UnifiedDisentangledLossComputer],
)
def test_builder_error_propagates_with_real_config(
    _explode: None, computer_cls: type
) -> None:
    """A LossBuilder failure with a (non-None) config must raise, NOT collapse
    to a bare L1 stack."""
    config = MagicMock()  # truthy, `is not None` → the real-config branch
    if computer_cls is UnifiedVAELossComputer:
        # The VAE computer first checks for a declared reconstruction loss (decision 16).
        from spectramr.config.schemas.loss import LossComponentConfig, LossConfigSchema

        config.losses = LossConfigSchema(
            output_domain="image", image_losses=[LossComponentConfig(name="mse", weight=1.0)]
        )
    with pytest.raises(RuntimeError, match="boom"):
        computer_cls(config=config, device=torch.device("cpu"))


# The VAE computer has no config-less fallback: it raises (decision 16; test_unified_vae.py).
@pytest.mark.parametrize(
    "computer_cls",
    [UnifiedGANLossComputer, UnifiedDisentangledLossComputer],
)
def test_no_config_uses_minimal_fallback(computer_cls: type) -> None:
    """``config is None`` is the ONLY sanctioned minimal-fallback path — it must
    still construct cleanly (no raise) so direct scripting/tests keep working."""
    comp = computer_cls(config=None, device=torch.device("cpu"))
    assert comp is not None
