"""The DI face of the checkpoint module: forwards to the module functions, owns nothing.

Forwarding by module attribute keeps one owner per behaviour (NN17) and lets a test
that patches ``store.save_step`` patch the production path too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spectramr.domain.interfaces.checkpoint_service_interface import ICheckpointService
from spectramr.infrastructure.checkpointing import restore, resume, store


class CheckpointStore(ICheckpointService):
    def save_step(self, pipeline: Any, **kwargs: Any) -> Path:
        return store.save_step(pipeline, **kwargs)

    def save_best(self, pipeline: Any, **kwargs: Any) -> Path:
        return store.save_best(pipeline, **kwargs)

    def resume(self, path: str | Path, pipeline: Any, **kwargs: Any) -> Any:
        return resume.resume(path, pipeline, **kwargs)

    def best_weights(self, path: str | Path, pipeline: Any, **kwargs: Any) -> None:
        restore.best_weights(path, pipeline, **kwargs)


__all__ = ["CheckpointStore"]
