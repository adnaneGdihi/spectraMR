"""``spectramr infer`` on the experiment_06 arms resolves the cascade sampler, not DDPM.

Drives the production resolve path (detector -> InferenceStrategyFactory) on the
arms' loaded settings with the registered generator at toy widths, and checks
that predict runs stage 2 on the stage-1 sample and reproduces validation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.inference.cascaded_diffusion_inference_strategy import (
    CascadedDiffusionInferenceStrategy,
)
from spectramr.infrastructure.inference.inference_factory import InferenceStrategyFactory
from spectramr.infrastructure.inference.strategy_detector import create_default_detector
from spectramr.infrastructure.training.strategies.cascaded_diffusion_strategy import (
    CascadedDiffusionStrategy,
)

_ROOT = Path(__file__).resolve().parents[4]
_ARM = _ROOT / "experiments/inprogress/diffusion/experiment_06_trellis_cascaded_diffusion.yaml"
_CONTROL = _ROOT / "experiments/inprogress/diffusion/experiment_06_single_stage_control.yaml"
_SIZE = 72  # the arm's cutoff of 32 needs min(H, W) > 64


def _generator(config: TrainingSettings) -> torch.nn.Module:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import get_model_class

    populate_model_registry()
    kwargs = {**config.model.model_kwargs, "features": (4, 8), "time_dim": 8}
    torch.manual_seed(0)
    return get_model_class(config.model.model_type)(
        in_channels=config.model.in_channels, out_channels=config.model.out_channels, **kwargs
    ).eval()


@pytest.mark.parametrize("path", [_ARM, _CONTROL])
def test_detector_routes_both_arms_to_the_cascade(path: Path) -> None:
    config = TrainingSettings.from_yaml(str(path)).model_dump()
    assert create_default_detector().detect(config) == "cascaded_diffusion"


def test_infer_runs_stage_two_on_the_stage_one_sample_and_matches_validation() -> None:
    config = TrainingSettings.from_yaml(str(_ARM))
    model = _generator(config)
    strategy = InferenceStrategyFactory.create(model, torch.device("cpu"), config.model_dump())
    assert isinstance(strategy, CascadedDiffusionInferenceStrategy)
    strategy.sampler.schedule = replace(strategy.sampler.schedule, sampling_steps=3)
    seen_2: list[torch.Tensor] = []
    model.stage_2_net.register_forward_pre_hook(lambda _m, a: seen_2.append(a[0]))
    y = torch.rand(1, 1, _SIZE, _SIZE, generator=torch.Generator().manual_seed(3))

    predicted = strategy.infer(y)
    assert predicted.shape == y.shape and len(seen_2) == 3

    train = object.__new__(CascadedDiffusionStrategy)
    train.config, train.device, train.logging_service = config, torch.device("cpu"), None
    train.env = SimpleNamespace(generator=model)
    train._apply_input_conditioning = lambda image, ctx: image  # type: ignore[method-assign]
    train._setup_strategy_specific_components()
    train._sampler.schedule = strategy.sampler.schedule
    with (
        patch.object(
            CascadedDiffusionStrategy, "_prepare_generator_inputs", lambda s, c, x, **k: (x, {})
        ),
        torch.no_grad(),
    ):
        validated = train._validation_forward(y, {})
    assert torch.allclose(predicted, validated, atol=1e-6)


def test_control_infers_stage_one_only() -> None:
    config = TrainingSettings.from_yaml(str(_CONTROL))
    strategy = InferenceStrategyFactory.create(
        _generator(config), torch.device("cpu"), config.model_dump()
    )
    strategy.sampler.schedule = replace(strategy.sampler.schedule, sampling_steps=2)
    assert strategy.infer(torch.rand(1, 1, _SIZE, _SIZE)).shape == (1, 1, _SIZE, _SIZE)


def test_declared_conditioning_raises_at_infer() -> None:
    config = TrainingSettings.from_yaml(str(_ARM))
    cfg = config.model_dump()
    cfg["model"]["conditioning"] = {"enabled": True, "sources": ["field_strength"]}
    with pytest.raises(ValueError, match=r"model\.conditioning"):
        CascadedDiffusionInferenceStrategy(_generator(config), torch.device("cpu"), cfg)
