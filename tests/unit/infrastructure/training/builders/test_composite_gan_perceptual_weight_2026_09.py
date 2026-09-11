"""``gan_composite`` takes its perceptual weight from the DECLARED table (#1923).

``LossBuilder._build_composite_gan`` used to read ``recon_config.lambda_perceptual``
raw. That field's schema default is **10.0** while its sibling ``enable_perceptual``
defaults ``False``, so every arm declaring ``losses.gan`` and never mentioning
perceptual built a VGG and *trained* it at weight 10.0 — the constructor argument is
consumed by ``CompositeGANLoss.compute_generator_loss``, which the unified GAN, VAE
and diffusion-reconstruction computers all drive.

Measured on ``dev`` @ ``ddeda3b0b`` over the 641 ``inprogress`` arms: 25 declare
``losses.gan``; **24 of them declare perceptual nowhere at all** and so trained an
undeclared VGG term. The 25th (``kspace_filling/experiment_11_sense_bridge_critic``)
writes ``lambda_perceptual: 0.0`` explicitly and is unaffected.

Why the weight and not ``resolve_loss_weight``: that helper applies the warm-up gate
against an ``iteration``, and this composite is constructed **once** with a scalar.
Resolving at iteration 0 would bake a warm-up 0.0 in permanently — 22 of the 24 arms
above run a warm-up — which is worse than the defect it would be fixing.

The declared probes below use **0.37**, which is not the schema default for any
lambda in the file. A coincident default is how a weight assertion goes vacuous here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

_GAN = {"enable_adversarial": True, "lambda_adv": 1.0, "gan_loss_type": "lsgan"}
_BASE: dict = {
    "image_losses": [{"name": "mse", "weight": 1.0, "enabled": True}],
    "kspace_losses": [],
    "complex_losses": [],
    "policy": {"output_domain": "image"},
    "gan": _GAN,
}


def _with_perceptual(weight: float, *, enabled: bool) -> dict:
    losses = dict(_BASE)
    losses["image_losses"] = [
        *_BASE["image_losses"],
        {"name": "perceptual", "weight": weight, "enabled": enabled},
    ]
    return losses


@pytest.fixture
def build(monkeypatch):
    """Drive the builder chain, stubbing only the VGG construction.

    ``create_loss("perceptual")`` takes ~12 s and loads real VGG weights. The stub
    keeps the test hermetic AND turns "was it built at all?" into an observable —
    which is the whole point for the undeclared case.
    """
    import spectramr.infrastructure.training.builders.loss_builder as loss_builder
    from spectramr.config.schemas.data import DataConfigSchema
    from spectramr.config.schemas.logging import LoggingConfigSchema
    from spectramr.config.schemas.loss import LossConfigSchema
    from spectramr.config.schemas.metrics import MetricsConfigSchema
    from spectramr.config.schemas.model import ModelConfigSchema
    from spectramr.config.schemas.optimization import OptimizationConfigSchema
    from spectramr.config.settings import TrainingSettings

    real = loss_builder.create_loss
    built: list[str] = []

    def spy(name, **kwargs):
        built.append(name)
        return nn.Identity() if name == "perceptual" else real(name, **kwargs)

    monkeypatch.setattr(loss_builder, "create_loss", spy)

    def _build(losses_block: dict):
        settings = TrainingSettings(
            model=ModelConfigSchema(),
            data=DataConfigSchema(),
            optimization=OptimizationConfigSchema(),
            logging=LoggingConfigSchema(),
            metrics=MetricsConfigSchema(),
            losses=LossConfigSchema(**losses_block),
        )
        losses = (
            loss_builder.LossBuilder(settings, device=torch.device("cpu"))
            .build_reconstruction_losses()
            .build_adversarial_losses()
            .build_physics_losses()
            .build_regularization_losses()
            .build_diffusion_losses()
            .build_latent_losses()
            .build_ssl_losses()
            .build()
        )
        return losses["adversarial"], built

    return _build


class TestUndeclaredPerceptualIsNotBuiltAtAll:
    """The planted violation: today's defect, committed as a test."""

    def test_a_gan_arm_that_never_mentions_perceptual_gets_weight_zero(self, build):
        adversarial, _ = build(dict(_BASE))
        assert adversarial.lambda_perceptual == 0.0, (
            "an undeclared loss must not inherit the 10.0 schema default"
        )

    def test_and_no_vgg_is_constructed(self, build):
        """The stronger witness: not merely weighted zero — never built.

        A fix that passed 0.0 but still constructed the VGG would leave the model on
        the device and the cost in the run. ``lambda_perceptual > 0`` gating alone
        cannot be trusted to show that, because the old code built the module first.
        """
        _, constructed = build(dict(_BASE))
        assert "perceptual" not in constructed, constructed

    def test_the_perceptual_slot_is_empty_rather_than_a_zero_weighted_module(self, build):
        adversarial, _ = build(dict(_BASE))
        assert adversarial.perceptual_loss is None


class TestDeclaredPerceptualKeepsItsDeclaredWeight:
    def test_a_domain_list_entry_reaches_the_composite_verbatim(self, build):
        adversarial, constructed = build(_with_perceptual(0.37, enabled=True))
        assert adversarial.lambda_perceptual == 0.37, (
            "the composite must use the DECLARED weight, not a default"
        )
        assert adversarial.perceptual_loss is not None
        assert "perceptual" in constructed

    def test_the_module_is_reused_not_rebuilt(self, build):
        """One construction site. A second VGG is silent, and costs a model's memory.

        NOT a witness for #1923, and the only one here that is not: measured GREEN
        against ``dev`` @ ``ddeda3b0b`` as well (``PYTHONPATH`` pointed at the dev
        worktree, 2026-09-08), because the old raw read also reused whatever
        ``self._losses["perceptual"]`` already held. The other five go red there.
        It is kept as a regression guard on the fallback branch the fix left in
        place -- if that branch ever starts firing, this is what turns red.
        """
        _, constructed = build(_with_perceptual(0.37, enabled=True))
        assert constructed.count("perceptual") == 1, constructed

    def test_an_explicitly_disabled_entry_is_off_not_defaulted(self, build):
        """``enabled: false`` must reach 0.0 — the shape the raw read turned into 10.0."""
        adversarial, constructed = build(_with_perceptual(0.37, enabled=False))
        assert adversarial.lambda_perceptual == 0.0
        assert adversarial.perceptual_loss is None
        assert "perceptual" not in constructed
