"""``_load_strategy_class`` reports the LOAD, not every lookup (issue #619).

The import is cached by ``sys.modules``, but the ``logger.info("Loaded strategy:
...")`` line was not, and several independent callers resolve the strategy in one
process: ``ConfigHealthChecker.check_contrast_conditioning_*``,
``validation.context_resolver``, and ``ModelInitializer``. Every cluster job log
therefore carried two or three "Loaded strategy: DiffusionTrainingStrategy" lines
and read as a repeated import.
"""

from __future__ import annotations

import logging

import pytest

from spectramr.infrastructure.training import strategy_factory
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

_LOGGER = "spectramr.infrastructure.training.strategy_factory"


@pytest.fixture
def forget_logged_paths():
    """Clear the process-wide announce set so each test starts cold."""
    saved = set(strategy_factory._LOGGED_STRATEGY_PATHS)
    strategy_factory._LOGGED_STRATEGY_PATHS.clear()
    yield
    strategy_factory._LOGGED_STRATEGY_PATHS.clear()
    strategy_factory._LOGGED_STRATEGY_PATHS.update(saved)


def test_repeated_resolution_announces_the_load_once(
    forget_logged_paths, caplog: pytest.LogCaptureFixture
) -> None:
    factory = TrainingStrategyFactory()
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        first = factory._load_strategy_class("diffusion")
        second = factory._load_strategy_class("diffusion")

    assert first is second
    announced = [
        r.message for r in caplog.records if r.message.startswith("Loaded strategy:")
    ]
    assert len(announced) == 1, f"expected one INFO load line, got {announced}"


def test_a_different_strategy_still_announces_itself(
    forget_logged_paths, caplog: pytest.LogCaptureFixture
) -> None:
    """Deduplication is per dotted path, not a global one-shot."""
    factory = TrainingStrategyFactory()
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        factory._load_strategy_class("diffusion")
        factory._load_strategy_class("gan")

    announced = [
        r.message for r in caplog.records if r.message.startswith("Loaded strategy:")
    ]
    assert len(announced) == 2, announced


def test_repeat_resolution_is_still_observable_at_debug(
    forget_logged_paths, caplog: pytest.LogCaptureFixture
) -> None:
    """Silencing the INFO must not silence the event entirely."""
    factory = TrainingStrategyFactory()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        factory._load_strategy_class("diffusion")
        factory._load_strategy_class("diffusion")

    assert any("already loaded" in r.message for r in caplog.records)


def test_unknown_strategy_still_raises(forget_logged_paths) -> None:
    """The dedup path must not swallow a genuine resolution failure (#9)."""
    factory = TrainingStrategyFactory()
    with pytest.raises(ValueError, match="Failed to load strategy class"):
        factory._load_strategy_class("spectramr.nonexistent.module.NoSuchStrategy")


def test_quality_matching_is_registered_and_loadable(forget_logged_paths) -> None:
    """A registry entry whose path does not import is dead on arrival at runtime.

    Asserting membership alone would pass on a typo'd dotted path, so this actually
    loads the class through the production resolver.
    """
    assert "quality_matching" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    factory = TrainingStrategyFactory()
    cls = factory._load_strategy_class(
        TrainingStrategyFactory.STRATEGY_CLASS_PATHS["quality_matching"]
    )
    assert cls.__name__ == "QualityMatchingStrategy"


def test_cascaded_diffusion_mode_routes_to_the_cascade_strategy() -> None:
    cls = TrainingStrategyFactory()._load_strategy_class(
        TrainingStrategyFactory.STRATEGY_CLASS_PATHS["cascaded_diffusion"]
    )
    assert cls.__name__ == "CascadedDiffusionStrategy"
