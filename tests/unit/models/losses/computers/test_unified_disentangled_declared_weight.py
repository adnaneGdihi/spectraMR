"""The weight a disentangled arm declares for its reconstruction loss is the weight it trains at.

``compute`` aliases the declared loss (``losses.reconstruction.loss_type``) as a component of
that name and weighted it through ``lambda_<name>`` lookups that ``losses.image_losses`` never
reached; for ``mse`` nothing matched and the lookup's ``return 1.0`` applied, so
``image_losses: [{name: mse, weight: 5.0}]`` trained at 1.0 with nothing saying so.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn

from spectramr.config.schemas.loss import (
    LatentLossesConfig,
    LossComponentConfig,
    ReconstructionLossesConfig,
)
from spectramr.models.losses.computers.unified_disentangled import (
    UnifiedDisentangledLossComputer,
)


def _computer(loss_type: str, image_losses: list[LossComponentConfig], **recon):
    comp = UnifiedDisentangledLossComputer.__new__(UnifiedDisentangledLossComputer)
    nn.Module.__init__(comp)
    reconstruction = SimpleNamespace(
        **{**ReconstructionLossesConfig().model_dump(), "loss_type": loss_type, **recon}
    )
    comp.config = SimpleNamespace(
        losses=SimpleNamespace(
            reconstruction=reconstruction,
            latent=LatentLossesConfig(),
            image_losses=image_losses,
        )
    )
    return comp


def test_a_declared_mse_weight_is_the_weight_applied() -> None:
    comp = _computer("mse", [LossComponentConfig(name="mse", weight=5.0)])
    assert comp._get_loss_weight("mse") == 5.0


def test_a_declared_l1_weight_beats_the_lambda_l1_default() -> None:
    comp = _computer("l1", [LossComponentConfig(name="l1", weight=3.0)])
    assert comp._get_loss_weight("l1") == 3.0


def test_only_the_declared_reconstruction_loss_is_reweighted() -> None:
    """A second image loss in the list is not this alias; it keeps its own lookup."""
    comp = _computer(
        "mse",
        [LossComponentConfig(name="mse", weight=5.0), LossComponentConfig(name="ssim", weight=7.0)],
    )
    assert comp._get_loss_weight("ssim") == pytest.approx(0.0)  # lambda_ssim default, not 7.0


def test_a_disabled_declaration_is_not_a_weight() -> None:
    comp = _computer("mse", [LossComponentConfig(name="mse", weight=5.0, enabled=False)])
    assert comp._get_loss_weight("mse") == 1.0  # the pre-existing lookup


def test_without_a_declaration_the_old_lookup_is_unchanged() -> None:
    assert _computer("mse", [])._get_loss_weight("mse") == 1.0
    assert (
        _computer("l1", [])._get_loss_weight("l1") == 10.0
    )  # ReconstructionLossesConfig.lambda_l1


def test_the_arms_in_the_corpus_keep_their_weight() -> None:
    """The one corpus arm with a non-unit declaration (l1, 10.0) already trained at 10.0."""
    comp = _computer("l1", [LossComponentConfig(name="l1", weight=10.0)])
    assert comp._get_loss_weight("l1") == 10.0
