"""The declared image losses reach CycleBlochStrategy's generator objective on the train path."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.config.schemas.training.cycle_bloch import CycleBlochConfig
from spectramr.infrastructure.training.strategies.cycle_bloch_objective import (
    SOURCE_ANCHORED,
    resolve_segmenter,
    split_image_losses,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
from spectramr.models.losses.registry import create_loss


def _strategy_cls():
    path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS["cycle_bloch"]
    module, name = path.rsplit(".", 1)
    return getattr(__import__(module, fromlist=[name]), name)


def _env(losses, image_losses, cycle_cfg=None):
    env = MagicMock()
    env.device = torch.device("cpu")
    env.generator = MagicMock(return_value=torch.rand(2, 1, 8, 8))
    env.discriminator = torch.nn.Conv2d(1, 1, 3, padding=1)
    env.losses = losses
    cfg = env.config
    cfg.optimization.gradient.clip.enabled = False
    cfg.optimization.gradient.clip.method = "norm"
    cfg.optimization.gradient.clip.value = 1.0
    cfg.optimization.gradient.activation_offload = "none"
    cfg.optimization.precision.dtype = "float32"
    cfg.losses.gan = SimpleNamespace(
        lambda_cycle_bloch=1.0,
        lambda_cycle_adv=1.0,
        gan_loss_type="lsgan",
        enable_gradient_penalty=False,
        lambda_gp=10.0,
        label_smoothing=0.0,
    )
    cfg.losses.image_losses = image_losses
    cfg.losses.kspace_losses = []
    cfg.losses.complex_losses = []
    cfg.training.cycle_bloch = cycle_cfg
    cfg.physics.field_strength = None
    cfg.parallel = None
    return env


def _entry(name, weight):
    return SimpleNamespace(name=name, weight=weight, enabled=True, kwargs={})


def _closure_metrics(losses, image_losses, cycle_cfg=None):
    torch.manual_seed(0)
    env = _env(losses, image_losses, cycle_cfg)
    with patch("spectramr.infrastructure.di.di_container.resolve_service") as resolve:
        resolve.return_value = MagicMock()
        strat = _strategy_cls()(env=env)
    real_ulf = torch.rand(2, 1, 8, 8, generator=torch.Generator().manual_seed(0))
    target = torch.rand(2, 1, 8, 8, generator=torch.Generator().manual_seed(1))
    steps = strat.train_step(
        batch=None, epoch=0, iteration=0, input_batch=real_ulf, target_batch=target
    )
    total = steps[0]["closure"]()
    return total, strat.get_last_metrics(), strat


def test_declared_image_losses_fold_into_the_generator_total():
    base, _, _ = _closure_metrics({}, [])
    losses = {
        "gradient_domain_consistency": create_loss("gradient_domain_consistency"),
        "mse": create_loss("mse"),
    }
    entries = [_entry("gradient_domain_consistency", 1.0), _entry("mse", 2.0)]
    total, metrics, _ = _closure_metrics(losses, entries)
    assert "loss_gradient_domain_consistency" in metrics and "loss_mse" in metrics
    expected = base + metrics["loss_gradient_domain_consistency"] + 2.0 * metrics["loss_mse"]
    torch.testing.assert_close(total.detach(), expected.detach())


def test_segmentation_equivariance_runs_with_its_segmenter():
    losses = {"segmentation_equivariance": create_loss("segmentation_equivariance")}
    cfg = CycleBlochConfig(segmenter_backend="label_dice")
    _, metrics, strat = _closure_metrics(losses, [_entry("segmentation_equivariance", 1.0)], cfg)
    assert metrics["loss_segmentation_equivariance"] > 0
    assert strat.cycle_bloch_provenance["segmenter_backend"] == "label_dice"


def test_segmentation_equivariance_without_a_segmenter_raises():
    losses = {"segmentation_equivariance": create_loss("segmentation_equivariance")}
    with pytest.raises(ValueError, match="segmenter_backend"):
        _closure_metrics(losses, [_entry("segmentation_equivariance", 1.0)])


def test_cycle_criterion_is_read_from_config():
    _, l1, _ = _closure_metrics({}, [])
    _, mse, strat = _closure_metrics({}, [], CycleBlochConfig(cycle_criterion="mse"))
    assert strat.cycle_bloch_provenance["cycle_criterion"] == "mse"
    assert not torch.isclose(l1["loss_cycle"], mse["loss_cycle"])


def test_split_routes_by_role():
    seg = resolve_segmenter("label_dice", torch.device("cpu"))
    paired, anchored = split_image_losses(
        {
            "l1": create_loss("l1"),
            "gradient_domain_consistency": create_loss("gradient_domain_consistency"),
        },
        {},
        seg,
    )
    assert set(paired) == {"l1"} and set(anchored) == {"gradient_domain_consistency"}
    assert set(anchored) <= SOURCE_ANCHORED
    assert resolve_segmenter("none", torch.device("cpu")) is None


def test_the_seg_arm_losses_build_through_the_production_loss_builder():
    """The arm's declared block, built by LossBuilder, is what the strategy splits."""
    from pathlib import Path

    import yaml

    from spectramr.config.schemas.data import DataConfigSchema
    from spectramr.config.schemas.logging import LoggingConfigSchema
    from spectramr.config.schemas.loss import LossConfigSchema
    from spectramr.config.schemas.metrics import MetricsConfigSchema
    from spectramr.config.schemas.model import ModelConfigSchema
    from spectramr.config.schemas.optimization import OptimizationConfigSchema
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.training.builders.loss_builder import LossBuilder
    from spectramr.infrastructure.training.strategies.loss_folding import declared_loss_weights

    arm = Path("experiments/inprogress/multi_contrast/exp_cycle_seg_equivariance.yaml")
    losses = yaml.safe_load(arm.read_text())["losses"]
    settings = TrainingSettings(
        model=ModelConfigSchema(),
        data=DataConfigSchema(),
        optimization=OptimizationConfigSchema(),
        logging=LoggingConfigSchema(),
        metrics=MetricsConfigSchema(),
        losses=LossConfigSchema(**losses),
    )
    builder = LossBuilder(config=settings, device=torch.device("cpu"))
    builder._build_all_dynamic()
    built = builder._losses
    assert {"l1", "gradient_domain_consistency", "segmentation_equivariance"} <= set(built)
    assert "ssim" not in built
    assert built["gradient_domain_consistency"].operator == "sobel"
    weights = declared_loss_weights(settings)
    seg = resolve_segmenter("label_dice", torch.device("cpu"))
    paired, anchored = split_image_losses(built, weights, seg)
    assert {"segmentation_equivariance", "gradient_domain_consistency"} <= set(anchored)
    assert "l1" in paired
    with pytest.raises(ValueError, match="segmenter_backend"):
        split_image_losses(built, weights, None)
