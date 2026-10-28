"""The checkpoint interface the DI container hands to training.

Implemented by ``infrastructure.checkpointing.facade.CheckpointStore``. Domain may not
import infrastructure, so the signatures carry ``Any`` and keyword arguments; the
module functions behind the facade own the typed signatures.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ICheckpointService(ABC):
    """Write a run's checkpoints and restore from them."""

    @abstractmethod
    def save_step(self, pipeline: Any, **kwargs: Any) -> Path:
        """Write a step checkpoint; see ``checkpointing.store.save_step``."""

    @abstractmethod
    def save_best(self, pipeline: Any, **kwargs: Any) -> Path:
        """Replace the best checkpoint; see ``checkpointing.store.save_best``."""

    @abstractmethod
    def resume(self, path: str | Path, pipeline: Any, **kwargs: Any) -> Any:
        """Restore every part; see ``checkpointing.resume.resume``."""

    @abstractmethod
    def best_weights(self, path: str | Path, pipeline: Any, **kwargs: Any) -> None:
        """Restore models and EMA only; see ``checkpointing.restore.best_weights``."""
