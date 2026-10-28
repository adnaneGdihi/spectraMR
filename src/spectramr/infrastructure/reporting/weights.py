"""Weight-space samples of a run's safetensors checkpoints, for the report.

A checkpoint stores each tensor under ``"<index>:<path>"`` (see
``checkpointing/codecs/safetensors_codec.py``); the model parameters are the entries whose
path starts with ``models.``. Reading all of them is hundreds of MB per file, so this takes a
fixed, name-seeded subset of the elements: the same positions in every checkpoint of one
architecture, which makes two samples comparable element for element. No model is built and
no data is read.
"""

from __future__ import annotations

import functools
import logging
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    from safetensors import safe_open

    HAVE_SAFETENSORS = True
except ImportError:
    safe_open = None
    HAVE_SAFETENSORS = False
    logger.info("safetensors is not installed; the weight-space figures are unavailable")

_FLOAT_DTYPES = frozenset({"F32", "F16", "BF16"})
_MODEL_PREFIX = "models."
_STEP = re.compile(r"step_(\d+)\.safetensors$")
DEFAULT_ELEMENTS = 100_000
MAX_CHECKPOINTS = 12


@dataclass(frozen=True)
class WeightSample:
    """Sampled parameter values; ``group[i]`` indexes ``group_names`` for ``values[i]``."""

    values: np.ndarray
    group: np.ndarray
    group_names: tuple[str, ...]


def checkpoint_files(
    run_dir: str | Path, limit: int = MAX_CHECKPOINTS
) -> list[tuple[int | None, Path]]:
    """``(step, path)`` of a run's checkpoints in step order; ``best`` (step ``None``) last.

    More than ``limit`` periodic checkpoints are thinned evenly, always keeping the first and
    the last, so a long run costs a bounded number of reads.
    """
    folder = Path(run_dir) / "checkpoints"
    periodic: list[tuple[int | None, Path]] = []
    best: list[tuple[int | None, Path]] = []
    for path in sorted(folder.glob("*.safetensors")):
        match = _STEP.search(path.name)
        if match:
            periodic.append((int(match.group(1)), path))
        elif path.name == "best.safetensors":
            best.append((None, path))
    periodic.sort(key=lambda item: item[0])
    if len(periodic) > limit:
        keep = np.unique(np.linspace(0, len(periodic) - 1, limit).round().astype(int))
        periodic = [periodic[i] for i in keep]
    return periodic + best


def _group(path: str, depth: int) -> str:
    return ".".join(path.split(".")[:depth])


def sample_weights(
    path: str | Path, n_elements: int = DEFAULT_ELEMENTS, group_depth: int = 2
) -> WeightSample:
    """A name-seeded subset of the checkpoint's float parameters, with their layer group.

    Cached per ``(path, n_elements, group_depth)``: the trajectory and the drift figure of one
    report read the same files, and a checkpoint is hundreds of MB.

    Raises:
        RuntimeError: ``safetensors`` is not installed.
        ValueError: the file holds no floating-point ``models.*`` tensor.
    """
    if not HAVE_SAFETENSORS:
        raise RuntimeError("safetensors is required to read checkpoints for the weight figures")
    return _sample(str(path), n_elements, group_depth)


@functools.lru_cache(maxsize=256)
def _sample(path: str, n_elements: int, group_depth: int) -> WeightSample:
    values: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    names: dict[str, int] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        entries = []
        for key in handle.keys():  # noqa: SIM118 - safe_open handles are not iterable
            _, _, param = key.partition(":")
            sl = handle.get_slice(key)
            if param.startswith(_MODEL_PREFIX) and sl.get_dtype() in _FLOAT_DTYPES:
                entries.append((param[len(_MODEL_PREFIX) :], key, int(np.prod(sl.get_shape()))))
        entries.sort()
        total = sum(size for *_, size in entries)
        if not total:
            raise ValueError(f"{path} holds no floating-point model parameters")
        for param, key, size in entries:
            take = int(min(size, max(1, round(n_elements * size / total))))
            rng = np.random.default_rng(zlib.crc32(param.encode()))
            index = (
                np.sort(rng.choice(size, take, replace=False)) if take < size else np.arange(size)
            )
            tensor = handle.get_tensor(key).reshape(-1)
            values.append(tensor[index].float().numpy())
            name = _group(param, group_depth)
            groups.append(np.full(take, names.setdefault(name, len(names)), dtype=np.int32))
    return WeightSample(np.concatenate(values), np.concatenate(groups), tuple(names))


__all__ = ["HAVE_SAFETENSORS", "WeightSample", "checkpoint_files", "sample_weights"]
