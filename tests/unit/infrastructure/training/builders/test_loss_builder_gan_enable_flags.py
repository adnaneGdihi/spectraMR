"""``enable_gradient_penalty: false`` must reach the term that gets built.

``get_enabled_losses`` pairs ``lambda_gp`` with ``enable_gradient_penalty`` and
``feature_matching`` with ``enable_feature_matching``. ``LossBuilder`` -- the
surface that actually constructs ``gan_composite`` -- read the weights raw and
ignored both flags, so an arm that switched the penalty off in writing was
handed one at the schema default of **10.0**. Two owners of one declaration,
and the weaker one is the one that ran (non-negotiable 17).

The cost is not only a wrong number. ``experiment_11_sense_bridge_critic``
declares ``enable_gradient_penalty: false`` next to ``parallel.strategy:
deepspeed`` / ``zero_stage: 2``; the penalty's ``autograd.grad`` double-backward
is not reducible under ZeRO-2, and the 2026-09-16 run died in
``stage_1_and_2.py::reduce_ipg_grads`` with ``IndexError: list index out of
range`` -- for a term the arm had never asked for.
"""

from __future__ import annotations

import pytest

from spectramr.config.schemas.loss import LossConfigSchema
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder


class _TrainingShim:
    """The two fields ``TrainingStrategyFactory.get_strategy_class`` reads."""

    strategy_class = None
    training_mode = "gan"


class _ConfigShim:
    def __init__(self, losses: LossConfigSchema) -> None:
        self.losses = losses
        self.training = _TrainingShim()


def _composite(**gan: object):
    losses = LossConfigSchema(gan={"enable_adversarial": True, "gan_loss_type": "hinge", **gan})
    built = LossBuilder(_ConfigShim(losses), device="cpu").build_adversarial_losses().build()
    composite = built.get("adversarial")
    assert composite is not None, "the adversarial composite must be built at all"
    return composite


@pytest.mark.parametrize("weight", [10.0, 0.5])
def test_a_disabled_gradient_penalty_is_not_built_whatever_its_weight(weight: float):
    """The planted violation: the crash shape, at the schema default and below it."""
    assert _composite(enable_gradient_penalty=False, lambda_gp=weight).lambda_gp == 0.0


def test_an_enabled_gradient_penalty_keeps_its_declared_weight():
    """The flag gates the term; it must not also silence a term the arm asked for."""
    assert _composite(enable_gradient_penalty=True, lambda_gp=7.5).lambda_gp == 7.5


def test_the_schema_default_alone_does_not_buy_a_penalty():
    """``lambda_gp`` defaults to 10.0 and ``enable_gradient_penalty`` to False."""
    assert _composite().lambda_gp == 0.0


def test_a_disabled_feature_matching_is_not_built():
    """Same shape, same fix: the second raw reader in the same call."""
    assert _composite(enable_feature_matching=False, feature_matching=2.0).lambda_feat_match == 0.0


def test_an_enabled_feature_matching_keeps_its_declared_weight():
    assert _composite(enable_feature_matching=True, feature_matching=2.0).lambda_feat_match == 2.0
