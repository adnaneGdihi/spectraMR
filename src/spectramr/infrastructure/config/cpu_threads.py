"""The thread count ``spectramr.main`` pins OpenMP, MKL, OpenBLAS and torch to."""

from __future__ import annotations

import os
from collections.abc import Mapping

from spectramr.core.resources import ALLOC_CPU_ENV, env_int

ENV_VAR = "SPECTRAMR_CPU_THREADS"


def resolve_cpu_threads(environ: Mapping[str, str] | None = None) -> int:
    """Threads per rank: the override, else the scheduler's grant, else ``1``.

    ``SPECTRAMR_CPU_THREADS`` wins when set. Otherwise the cores the scheduler
    granted this task (``SLURM_CPUS_PER_TASK`` and its peers) are divided by the
    ranks sharing the node (``LOCAL_WORLD_SIZE``, set by ``torchrun``), the same
    split the generated sbatch applies to ``OMP_NUM_THREADS``. Outside a
    scheduler allocation nothing is known, so the result is ``1``.

    A malformed override raises: an invalid thread count must not quietly become
    the default and leave a run on one core.
    """
    env = os.environ if environ is None else environ
    raw = env.get(ENV_VAR)
    if raw is not None and raw.strip():
        try:
            threads = int(raw)
        except ValueError:
            threads = 0
        if threads < 1:
            raise ValueError(f"{ENV_VAR} must be a positive integer, got {raw!r}")
        return threads
    granted = env_int(ALLOC_CPU_ENV, env)
    if granted is None:
        return 1
    ranks = env_int(("LOCAL_WORLD_SIZE",), env) or 1
    return max(1, granted // ranks)
