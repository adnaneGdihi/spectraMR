"""`unpaired_ulf/exp_up_01` is routed to cold diffusion on the 64 mT input, but cannot train yet.

Under ``GraphColdDiffusionStrategy`` the arm degraded the 3 T target and never read the
64 mT input. It now resolves to the cold-diffusion path of ``DiffusionTrainingStrategy``
with ``degradation_source: input``. The base ``train_step`` input preparation still
double-FFTs a 1-channel ``nifti_paired`` image, so the arm stays ``needs_implementation``;
the strict xfail below turns red the day that owner is fixed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spectramr.config.schemas.base import LAUNCH_REFUSED_STATUSES
from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.leaf.adapter_builders import (
    AdapterChainBuilder,
    apply_pre_model,
)
from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

_COHORT = Path(__file__).resolve().parents[3] / "experiments/inprogress/unpaired_ulf"
ARMS = [
    "exp_up_01_paired_kspace_cold_diffusion.yaml",
    "exp_up_01_paired_kspace_cold_diffusion_seed43.yaml",
    "exp_up_01_paired_kspace_cold_diffusion_seed44.yaml",
]


@pytest.fixture(scope="module", params=ARMS)
def settings(request: pytest.FixtureRequest) -> TrainingSettings:
    return TrainingSettings.from_yaml(str(_COHORT / request.param))


@pytest.fixture
def strategy(settings: TrainingSettings) -> DiffusionTrainingStrategy:
    cls = TrainingStrategyFactory().get_strategy_class(settings)
    assert cls is DiffusionTrainingStrategy
    st = object.__new__(cls)  # no environment or model is built
    st.config = settings
    st.device = torch.device("cpu")
    return st


def test_the_launcher_refuses_the_arm_until_the_input_path_is_fixed(
    settings: TrainingSettings,
) -> None:
    assert settings.metadata.status in LAUNCH_REFUSED_STATUSES


def test_the_cold_forward_process_degrades_the_input(strategy: DiffusionTrainingStrategy) -> None:
    assert strategy._is_cold_diffusion()
    assert not strategy._is_latent_diffusion()
    assert strategy._cold_degradation_source() == "input"


def test_the_declared_objective_is_the_one_that_runs(strategy: DiffusionTrainingStrategy) -> None:
    strategy._validate_prediction_type()  # epsilon on a cold process would raise here
    assert strategy._effective_prediction_type() == "sample"


@pytest.mark.xfail(
    strict=True,
    reason="_prepare_model_input FFTs the 1-ch image, then fft_image_to_kspace FFTs it again",
)
def test_train_step_input_preparation_feeds_the_model(
    strategy: DiffusionTrainingStrategy, settings: TrainingSettings
) -> None:
    # The base train_step order: _prepare_model_input, then pre_model, then the channel guards.
    image, target = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
    prepared = strategy._prepare_model_input(image)
    chains = AdapterChainBuilder(settings.adapters).build()
    model_in, model_target = apply_pre_model(chains, prepared, target)
    assert not strategy.builds_own_model_input()
    assert model_in.shape[1] == settings.model.in_channels
    assert model_target.shape[1] == settings.model.out_channels
    assert model_in.dtype == model_target.dtype
