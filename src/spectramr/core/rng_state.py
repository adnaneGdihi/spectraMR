"""RNG snapshot/restore — the one owner of the checkpoint's stochastic half.

A resume that restores weights, optimizers and schedulers but not the random
streams continues from the right *position* on a different *trajectory*:
dropout masks, augmentation draws and — for cold diffusion — the per-step
timestep and mask samples all diverge from the moment the run is requeued. The
run still trains and still reports success, so the divergence is invisible.

They live outside any writer so that ``checkpointing.capture``, which writes the
state, and ``checkpointing.resume``, which restores it, share one definition of
what "the RNG state" is (non-negotiable 17).

**What a checkpoint can carry is one rank's streams.** Under plain DDP only
rank 0 reaches ``checkpointing.store``'s save (``may_checkpoint = is_main_process or
checkpoints_need_all_ranks``), so there is no rendezvous at which the other
ranks could contribute theirs — a collective placed there would hang the job.
Rank 0's stream is therefore the one that is replayed; the callers restore it
on rank 0 only, because handing every rank rank-0's stream would erase the rank
offset ``train.py`` applies precisely to keep augmentation diverse.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Reconstructors ``torch.load(weights_only=True)`` needs for our own envelope.
#:
#: ``np.random.get_state()`` returns a tuple carrying a ``uint32`` ndarray, so
#: unpickling it needs ``numpy._core.multiarray._reconstruct``, ``np.ndarray``,
#: ``np.dtype`` and the concrete dtype class. None of those are in torch's
#: default allowlist, so from torch 2.6 on — where ``weights_only=True`` became
#: the default — *every* checkpoint carrying RNG state is refused by a plain
#: ``torch.load(path)``. These four are data-only reconstructors, so
#: allowlisting them keeps the weights-only guarantee while letting our envelope
#: through. Readers should wrap their load in
#: ``torch.serialization.safe_globals(RNG_STATE_SAFE_GLOBALS)`` rather than
#: registering them process-globally, which would leak across callers.
RNG_STATE_SAFE_GLOBALS: list[Any] = [
    np._core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    type(np.dtype(np.uint32)),
]


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG that influences training stochasticity.

    Covers the calling process only — the CUDA entry spans every device visible
    to this rank, not other ranks.
    """
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte(state: Any) -> Any:
    """Bring an RNG blob back to the CPU before the generator is handed it.

    ``checkpointing.resume`` reads through the codecs, which return CPU tensors,
    so this is defensive: a caller that loads with a GPU ``map_location`` hands
    these over on the device. The generators refuse a non-CPU tensor
    (``check_rng_state`` asserts ``device().type() == kCPU``) with
    ``TypeError: RNG state must be a torch.ByteTensor``, which would fail the
    resume and report the chain link as ``success: False``.

    This is resume-time, once per job, so the copy is not the host sync
    non-negotiable 9 forbids inside the training loop.
    """
    return state.cpu() if torch.is_tensor(state) else state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Reverse of :func:`capture_rng_state`.

    Tolerates partial state, for checkpoints that pre-date one of the streams.
    The CUDA entry is skipped when this process has no CUDA — a CPU resume of a
    GPU checkpoint is legitimate — and when the device count differs, since
    ``set_rng_state_all`` requires exactly one state per visible device and
    would otherwise raise on a resume onto a differently-sized node. That
    mismatch is logged rather than absorbed: it changes the stochastic
    trajectory, which is the one thing this function exists to preserve.
    """
    if "torch" in state:
        torch.set_rng_state(_as_cpu_byte(state["torch"]))
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])
    if "cuda" not in state or not torch.cuda.is_available():
        return
    saved = [_as_cpu_byte(entry) for entry in state["cuda"]]
    if len(saved) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(saved)
    else:
        logger.warning(
            "[RNG] checkpoint carries %d CUDA RNG state(s) but this process sees "
            "%d device(s), so the CUDA streams were NOT restored (cpu/numpy/python "
            "were). Resuming onto a different GPU count changes the stochastic "
            "trajectory.",
            len(saved),
            torch.cuda.device_count(),
        )


__all__ = [
    "RNG_STATE_SAFE_GLOBALS",
    "capture_rng_state",
    "restore_rng_state",
]
