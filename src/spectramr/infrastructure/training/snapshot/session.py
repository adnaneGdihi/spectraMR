"""One run's snapshot session: decide the step, open the trace, write it.

Built once per run from the strategy. The training loop wraps each step in
:meth:`SnapshotSession.step`; the validation loop wraps the first batch of each
validation pass. Everything a strategy, a transform or a model module records inside
that block lands in one snapshot, under ``<run_dir>/snapshots/<run_id>/<phase>/``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from spectramr.core.snapshot_trace import SnapshotTrace, View, active, record
from spectramr.infrastructure.training.snapshot.hooks import ModelHooks
from spectramr.infrastructure.training.snapshot.policy import SnapshotSchedule, ViewPolicy
from spectramr.infrastructure.training.snapshot.writer import TraceWriter
from spectramr.infrastructure.training.snapshot_coil_maps import COIL_MAP_KEYS

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = "snapshots"
_BATCH_KSPACE = {"input", "target", "kspace", "measured_kspace", "input_alt"}


def batch_tensors(batch: Any) -> dict[str, torch.Tensor]:
    """Every tensor the loader delivered, core fields first, then metadata."""
    if isinstance(batch, dict):
        return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
    out = {
        k: getattr(batch, k, None)
        for k in ("input", "target", "mask", "coil_maps")
        if isinstance(getattr(batch, k, None), torch.Tensor)
    }
    meta = getattr(batch, "metadata", None) or {}
    out.update({k: v for k, v in meta.items() if isinstance(v, torch.Tensor) and k not in out})
    return out


def loader_compressed(batch: Any, keys: set[str]) -> set[str]:
    """``keys`` when the loader's ``KSpaceNormalizationTransform`` ran on ``batch``."""
    from spectramr.data.batch_types import read_batch_field
    from spectramr.data.transforms.normalization import LoaderNormalization, loader_normalization

    verdict = loader_normalization(batch, read_batch_field(batch, "kspace_scale"))
    return set(keys) if verdict is LoaderNormalization.APPLIED else set()


class SnapshotSession:
    def __init__(
        self, strategy: Any, root: Path, schedule: SnapshotSchedule, snap_cfg: Any
    ) -> None:
        self._strategy = strategy
        self.schedule = schedule
        self.policy = ViewPolicy.from_config(strategy.config)
        self.max_repeats = int(snap_cfg.max_model_calls)
        self.trace_loader = bool(snap_cfg.trace_loader)
        self.trace_index = int(snap_cfg.trace_index)
        self.writer = TraceWriter(
            root,
            save_images=bool(snap_cfg.save_images),
            save_json=bool(snap_cfg.save_json),
            save_tensors=bool(snap_cfg.save_tensors),
        )
        self.knobs = snap_cfg.model_dump()
        models = getattr(getattr(strategy, "env", None), "models", None) or {}
        self.hooks = ModelHooks(models, self._model_view)
        for module in (getattr(strategy.env, "physics", None) or {}).values():
            self.hooks.attach_stages(module)
        self.models = _model_widths(models)

    @classmethod
    def build(cls, strategy: Any) -> SnapshotSession | None:
        """The run's session, or ``None`` when snapshots are off or this is not rank 0."""
        from spectramr.infrastructure.distributed.distributed_training import RankUtility
        from spectramr.infrastructure.training.snapshot_provenance import run_identity

        logging_cfg = strategy.config.logging
        schedule = SnapshotSchedule.from_logging(logging_cfg)
        if not schedule.enabled or not RankUtility.is_main_rank():
            return None
        run_dir = (
            getattr(strategy.env, "run_output_dir", None)
            or getattr(strategy.config.training, "output_dir", None)
            or "experiments/outputs"
        )
        if not isinstance(run_dir, (str, Path)):
            raise TypeError(f"snapshot run_dir must be a path, got {type(run_dir).__name__}")
        root = Path(run_dir) / SNAPSHOT_DIR / str(run_identity()["run_id"])
        return cls(strategy, root, schedule, logging_cfg.snapshots)

    def _model_view(self, kind: str) -> Any:
        return self.policy.views([kind])[kind]

    @contextmanager
    def step(self, phase: str, step: int, batch: Any = None) -> Iterator[SnapshotTrace | None]:
        """Trace everything inside when ``step`` is due for ``phase``; else do nothing."""
        if not self.schedule.due(phase, step):
            yield None
            return
        trace = SnapshotTrace(phase=phase, step=int(step), max_repeats=self.max_repeats)
        with active(trace):
            if self.trace_loader:
                self._trace_loader(trace)
            if batch is not None:
                self._record_batch(batch)
            try:
                yield trace
            finally:
                self._write(trace)

    def _trace_loader(self, trace: SnapshotTrace) -> None:
        """Re-fetch ``trace_index`` from this phase's dataset so the loader stages are traced."""
        from spectramr.infrastructure.training.snapshot.loader_trace import trace_sample

        loaders = getattr(self._strategy.env, "data_loaders", None) or {}
        loader = loaders.get(trace.phase)
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            trace.extra["loader_trace"] = f"no {trace.phase} dataset on the environment"
            return
        try:
            trace_sample(trace, dataset, self.trace_index, seed=trace.step)
        except Exception as exc:  # the step's own stages are still written
            trace.extra["loader_trace"] = f"failed: {type(exc).__name__}: {exc}"
            logger.warning(
                "[snapshot] loader trace of %s[%d] failed: %s", trace.phase, self.trace_index, exc
            )
            return
        trace.extra["loader_trace"] = {"index": self.trace_index, "seed": trace.step}
        self._loader_views(trace)

    def _loader_views(self, trace: SnapshotTrace) -> None:
        """Loader stages carry no domain declaration, so the spectrum check decides;
        tensors after ``KSpaceNormalizationTransform`` are on the compressed side."""
        compressed = False
        for stage in trace.stages:
            if stage.group != "loader":
                continue
            compressed = compressed or "KSpaceNormalization" in stage.name
            stage.views = {
                k: View()
                if "mask" in k or k in COIL_MAP_KEYS
                else View(kspace=True, log_scaled=self.policy.log_scaled and compressed)
                for k in stage.tensors
            }

    def _record_batch(self, batch: Any) -> None:
        from spectramr.infrastructure.training.snapshot_provenance import realised_acceleration

        tensors = batch_tensors(batch)
        kspace = _BATCH_KSPACE & set(tensors) if self.policy.targets_kspace else None
        record(
            "batch",
            "input_raw",
            tensors,
            views=self.policy.views(
                tensors,
                in_kspace_keys=kspace,
                log_scaled_keys=loader_compressed(batch, set(tensors)),
            ),
            extra={"realised_acceleration": lambda: realised_acceleration(batch)},
        )

    def _write(self, trace: SnapshotTrace) -> None:
        from spectramr.infrastructure.training.snapshot_provenance import run_identity

        strategy = self._strategy
        self._check_model_widths(trace)
        head = {
            "run": run_identity(),
            "strategy": type(strategy).__name__,
            "snapshots": self.knobs,
            "models": self.models,
            "provenance": strategy._snapshot_provenance(trace.phase),
            "strategy_context": _strategy_context(strategy),
        }
        try:
            out = self.writer.write(trace, head)
        except Exception as exc:  # a diagnostic must not end the run, but must be seen
            logger.warning("[snapshot] %s step %d not written: %s", trace.phase, trace.step, exc)
            return
        logger.info("[snapshot] %d stages -> %s", len(trace.stages), out)

    def _check_model_widths(self, trace: SnapshotTrace) -> None:
        """Stamp each model's fed channel count against its backbone's declared width."""
        from spectramr.infrastructure.training.model_input_contract import (
            STATUS_MISMATCH,
            verify_model_input,
        )

        for stage in trace.stages:
            if stage.group != "model" or not stage.name.startswith("input["):
                continue
            name = stage.name[len("input[") :].split("]", 1)[0]
            info = self.models.get(name)
            key = next(iter(stage.tensors), None)
            if info is None or key is None:
                continue
            verdict = verify_model_input(
                tensors=stage.tensors,
                model_input_key=key,
                in_channels=info["in_channels"],
                in_channels_source=info["in_channels_source"],
            )
            stage.params["width_check"] = verdict.as_record()
            if verdict.status == STATUS_MISMATCH:
                logger.warning("[snapshot] model %r: %s", name, verdict.detail)

    def close(self) -> None:
        self.hooks.remove()


def _strategy_context(strategy: Any) -> dict[str, Any]:
    try:
        return strategy._snapshot_context()
    except Exception as exc:  # stamped, not dropped: an absent record reads as "nothing to say"
        return {"error": f"{type(exc).__name__}: {exc}"}


def _model_widths(models: Any) -> dict[str, dict[str, Any]]:
    from spectramr.infrastructure.training.model_input_contract import resolve_model_in_channels

    out = {}
    for name, module in models.items():
        width, source = resolve_model_in_channels(module)
        out[name] = {
            "class": type(module).__name__,
            "in_channels": width,
            "in_channels_source": source,
        }
    return out


__all__ = ["SNAPSHOT_DIR", "SnapshotSession", "batch_tensors", "loader_compressed"]
