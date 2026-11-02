"""CascadedDiffusionStrategy driven from the experiment_06 arm on small CPU tensors.

The base ``__init__`` stands up AMP, the loss computer and DI services, so the
strategy is built with ``object.__new__`` and its own setup runs against the
arm's loaded ``TrainingSettings``. The generator is the registered
``cascaded_diffusion`` class built from the arm's stage blocks at toy widths.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.strategies.cascaded_diffusion_strategy import (
    CascadedDiffusionStrategy,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

_ROOT = Path(__file__).resolve().parents[5]
_ARM = _ROOT / "experiments/inprogress/diffusion/experiment_06_trellis_cascaded_diffusion.yaml"
_CONTROL = _ROOT / "experiments/inprogress/diffusion/experiment_06_single_stage_control.yaml"
_SIZE = 72  # the arm's cutoff of 32 needs min(H, W) > 64


def _registered_generator(config: TrainingSettings) -> torch.nn.Module:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import get_model_class

    populate_model_registry()
    kwargs = dict(config.model.model_kwargs)
    kwargs.update(features=(4, 8), time_dim=8)
    torch.manual_seed(0)
    return get_model_class(config.model.model_type)(
        in_channels=config.model.in_channels, out_channels=config.model.out_channels, **kwargs
    )


def _strategy(path: Path = _ARM) -> CascadedDiffusionStrategy:
    config = TrainingSettings.from_yaml(str(path))
    strat = object.__new__(CascadedDiffusionStrategy)
    strat.config = config
    strat.device = torch.device("cpu")
    strat.env = SimpleNamespace(generator=_registered_generator(config))
    strat.logging_service = None
    strat._apply_input_conditioning = lambda image, ctx: image  # type: ignore[method-assign]
    strat._setup_strategy_specific_components()
    return strat


def _record_inputs(net: torch.nn.Module) -> list[torch.Tensor]:
    seen: list[torch.Tensor] = []
    net.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))
    return seen


def _record_outputs(net: torch.nn.Module) -> list[torch.Tensor]:
    seen: list[torch.Tensor] = []
    net.register_forward_hook(lambda _m, _a, out: seen.append(out.detach().clone()))
    return seen


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(3)
    return (torch.rand(2, 1, _SIZE, _SIZE, generator=g) for _ in range(2))  # type: ignore[return-value]


def test_arm_resolves_to_the_cascade_strategy() -> None:
    config = TrainingSettings.from_yaml(str(_ARM))
    assert TrainingStrategyFactory().get_strategy_class(config) is CascadedDiffusionStrategy


def test_setup_reads_the_declared_schedule() -> None:
    strat = _strategy()
    assert strat.num_timesteps == 1000 and strat.sampling_steps == 50
    assert strat.scheduler.beta_schedule == "linear"


def test_stage_two_is_conditioned_on_stage_one_and_regresses_the_residual() -> None:
    strat = _strategy()
    gen = strat.env.generator
    seen_1, seen_2 = _record_inputs(gen.stage_1_net), _record_inputs(gen.stage_2_net)
    out_1 = _record_outputs(gen.stage_1_net)
    y, x0 = _pair()
    pred, inter = strat._generate_predictions(y, {}, {"hr": x0})
    assert inter is None and pred.shape == x0.shape and pred.requires_grad
    assert len(seen_1) == 1 and len(seen_2) == 1
    assert torch.equal(seen_1[0][:, :1], y) and torch.equal(seen_2[0][:, :1], y)
    # Stage 2's middle channel IS the stage-1 output projected onto the low band,
    # not the ground-truth low band (teacher forcing) or the measurement's.
    cond = seen_2[0][:, 1:2]
    s1_low, _ = strat.split_bands(out_1[0])
    assert torch.allclose(cond, s1_low, atol=1e-6)
    assert not torch.allclose(cond, strat.split_bands(x0)[0], atol=1e-3)
    # The composed estimate splits into a low-band and a high-band term.
    low, high = strat.split_bands(pred.detach())
    _, low_residual = strat.split_bands(low)
    assert low_residual.abs().max() < 1e-4 and high.abs().max() > 0


def test_mse_on_the_composition_is_the_sum_of_band_errors() -> None:
    strat = _strategy()
    y, x0 = _pair()
    pred, _ = strat._generate_predictions(y, {}, {"hr": x0})
    pl, ph = strat.split_bands(pred.detach())
    xl, xh = strat.split_bands(x0)
    total = (pred.detach() - x0).square().sum()
    assert torch.allclose(total, (pl - xl).square().sum() + (ph - xh).square().sum(), rtol=1e-4)


def test_validation_runs_stage_one_then_stage_two_on_its_sample() -> None:
    strat = _strategy()
    strat._sampler.schedule = replace(strat._sampler.schedule, sampling_steps=3)
    assert strat.sampling_steps == 3  # one schedule owns the count
    gen = strat.env.generator
    seen_1, seen_2 = _record_inputs(gen.stage_1_net), _record_inputs(gen.stage_2_net)
    out_1 = _record_outputs(gen.stage_1_net)
    y, _ = _pair()
    with (
        patch.object(
            CascadedDiffusionStrategy,
            "_prepare_generator_inputs",
            lambda self, ctx, x, **k: (x, {}),
        ),
        torch.no_grad(),
    ):
        out = strat._validation_forward(y, {})
        again = strat._validation_forward(y, {})
    assert out.shape == y.shape and torch.equal(out, again)  # seeded, reproducible
    assert len(seen_1) == 6 and len(seen_2) == 6
    # Every stage-2 step is conditioned on the final stage-1 sample (its last x0 estimate).
    final_s1, _ = strat.split_bands(out_1[2])
    for step_input in seen_2[:3]:
        assert torch.allclose(step_input[:, 1:2], final_s1, atol=1e-6)


def test_single_stage_control_skips_stage_two() -> None:
    strat = _strategy(_CONTROL)
    assert strat.env.generator.stage_2_net is None
    y, x0 = _pair()
    pred, _ = strat._generate_predictions(y, {}, {"hr": x0})
    assert pred.shape == x0.shape


def test_unread_diffusion_knob_raises() -> None:
    diffusion = {"sampler": "ddim", "prediction_type": "sample", "guidance_scale": 2.0}
    strat = object.__new__(CascadedDiffusionStrategy)
    strat.config = SimpleNamespace(
        training=TrainingStrategyConfigSchema.model_validate({"diffusion": diffusion})
    )
    with pytest.raises(ValueError, match="never reads"):
        strat._refuse_inert_diffusion_knobs()


@pytest.mark.parametrize(("path", "has_stage_2"), [(_ARM, True), (_CONTROL, False)])
def test_production_builder_delivers_the_stage_blocks(path: Path, has_stage_2: bool) -> None:
    """The arm's model_kwargs survive kwargs resolution and the factory's signature filter."""
    from spectramr.infrastructure.builders.generator_kwargs import resolve_full_generator_kwargs
    from spectramr.models.factories.model_factory import ModelFactory
    from spectramr.models.generators.cascaded_diffusion_generator import (
        CascadedDiffusionGenerator,
    )
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    config = TrainingSettings.from_yaml(str(path))
    resolved = resolve_full_generator_kwargs(config, model_cls=CascadedDiffusionGenerator)
    kwargs = {**resolved.kwargs, "features": (4, 8), "time_dim": 8}
    model = ModelFactory().create_generator(
        "cascaded_diffusion",
        _declared_keys=resolved.declared_keys,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        **kwargs,
    )
    assert isinstance(model, CascadedDiffusionGenerator)
    assert model.has_stage_2 is has_stage_2
    assert model.cutoff_frequency == (32 if has_stage_2 else None)
