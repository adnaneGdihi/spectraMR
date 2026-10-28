"""The recommender's generated config must carry the weights it recommends.

``_generate_config`` wrote only the ``enable_*`` flags, so every weight came from
the schema default instead: ``perceptual`` recommended at 0.1 trained at 10.0,
``ssim`` recommended at 0.1 resolved to 0.0, and the denoising task's primary
``mse`` was dropped because the lookup compared raw names against ``l2``.
"""

from __future__ import annotations

import pytest

from spectramr.models.losses.loss_recommender import LossRecommender, TaskAnalyzer, TaskType
from spectramr.models.losses.weights import build_loss_weight_table


def _declared(task: TaskType) -> dict[str, float]:
    config = LossRecommender().recommend(task_type=task).suggested_config
    return {name: spec.weight for name, spec in build_loss_weight_table(config).items()}


@pytest.mark.parametrize(
    "task", [t for t in TaskType if t is not TaskType.SEGMENTATION], ids=lambda t: t.value
)
def test_every_enabled_recommendation_lands_at_its_recommended_weight(task: TaskType) -> None:
    recs = LossRecommender().recommend(task_type=task).losses
    expected = {
        {"mse": "l2", "gan": "adversarial"}.get(r.name, r.name): r.weight for r in recs if r.enabled
    }
    assert _declared(task) == expected


def test_a_disabled_recommendation_declares_no_weight() -> None:
    """Writing its lambda would make the weight table read it as declared and active."""
    rs = LossRecommender().recommend(task_type=TaskType.SUPER_RESOLUTION, use_perceptual=False)
    config = rs.suggested_config
    assert config.reconstruction.enable_perceptual is False
    assert "perceptual" not in build_loss_weight_table(config)


def test_the_mse_alias_reaches_the_l2_flag() -> None:
    config = LossRecommender().recommend(task_type=TaskType.DENOISING).suggested_config
    assert config.reconstruction.enable_l2 is True
    assert config.get_enabled_losses()["l2"] == pytest.approx(1.0)


@pytest.mark.parametrize("text", ["denoising pipeline", "denoise the scan", "remove noise"])
def test_denoising_descriptions_route_to_denoising(text: str) -> None:
    assert TaskAnalyzer().analyze_task_type(task_description=text) is TaskType.DENOISING
