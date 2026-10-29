"""Rank helpers for the training pipeline.

The model-wrapping that used to live here moved to
``spectramr.infrastructure.distributed`` (see the note below). What remains is the
pipeline-layer surface: ``unwrap_model`` (re-exported from ``core``) and
``is_rank_zero``.
"""

import logging
from typing import Any

import torch

from spectramr.core.module_utils import unwrap_model
from spectramr.core.parallel_groups import dp_rank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unwrap helper — re-exported from the SSOT in ``core/``.
#
# This used to be an ``isinstance``-based DP/DDP-only implementation living here,
# in ``pipelines/``. Two problems: it handled neither ``torch.compile``'s
# ``_orig_mod`` nor FSDP's ``_fsdp_wrapped_module``, and it sat on the wrong side
# of an import edge — ``infrastructure/`` (where every checkpoint writer lives)
# may not import from ``pipelines/``, so the writers could not reach it. It had
# ZERO callers in ``src/``.
#
# The implementation now lives in ``spectramr.core.module_utils``, the rightmost
# layer, reachable from ``pipelines/``, ``infrastructure/``, ``models/`` and
# ``cli/`` alike. The name is kept here because it is the historical import path.
# ---------------------------------------------------------------------------


# ``apply_parallelism`` and its ``_wrap_data_parallel`` / ``_wrap_ddp`` /
# ``_apply_distributed_samplers`` helpers lived here and are gone.
#
# It implemented none/dp/ddp in an if/elif/else and RAISED on anything else,
# while FSDP was reached through a separate ``parallel.fsdp.enabled`` flag read
# in ModelBuilder. So ``strategy: 'fsdp'`` -- the spelling the reference template
# advertised -- raised ValueError AFTER the whole training environment had been
# built, and ``strategy: 'none'`` + ``fsdp.enabled: true`` silently sharded.
#
# It also wrapped everything at one point in the build, which cannot be right for
# both: FSDP must wrap BEFORE optimizers exist (it re-points parameter storage),
# DP/DDP after (so downstream consumers still see a bare module).
#
# Replaced by the plugin registry in
# ``spectramr.infrastructure.distributed.strategy_registry``, dispatched from
# ``TrainingEnvironmentDirector`` at two ordered hooks.


#: Strategies that run one process per device behind a torch.distributed process
#: group, so "rank 0" is a real question. ``none``/``dp`` are single-process
#: (DataParallel threads inside ONE process), where every caller is rank 0.
_PROCESS_GROUP_STRATEGIES = frozenset({"ddp", "fsdp", "deepspeed"})


def is_rank_zero(config: Any) -> bool:
    """Check if the current process is rank 0 (or non-distributed).

    Membership test, not ``!= "ddp"``. The original form predated ``fsdp`` and
    ``deepspeed`` being reachable, so it returned True on EVERY rank for both --
    and it is live at ``train.py`` gating the TensorBoard writer, so all N ranks
    opened a writer on the same event directory and interleaved their scalars.
    """
    if not hasattr(config, "parallel") or config.parallel is None:
        return True  # No parallelism config → single process
    if config.parallel.strategy.lower() not in _PROCESS_GROUP_STRATEGIES:
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def resolve_global_rank(config: Any) -> int:
    """This process's global rank, or 0 when there is one process.

    Same four guards as ``is_rank_zero`` and deliberately so: the question "which
    rank am I" is only meaningful under exactly the conditions that make "am I
    rank 0" meaningful, and answering it twice with two guard sets is how the two
    drift apart. This returns the rank itself rather than a boolean collapse of it.

    No ``os.environ["RANK"]`` fallback on purpose. ``setup_distributed()`` runs
    before ``run_training_pipeline`` (``pipelines/distributed.py``), so the process
    group is always initialised on the only path where the rank matters, and
    ``strategy: 'none'`` with ``RANK`` set already raises there. A fallback could
    therefore only fire in states the pipeline has already refused, while adding a
    second source of truth for the rank.
    """
    if not hasattr(config, "parallel") or config.parallel is None:
        return 0
    if config.parallel.strategy.lower() not in _PROCESS_GROUP_STRATEGIES:
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def resolve_data_rank(config: Any) -> int:
    """Index of this rank's data shard: its data-parallel index, not its global rank.

    With model-parallel axes, ranks in one tensor/context/spatial/pipeline group see the
    same batches, so they must draw the same data-stream seed. The global rank would
    split them. Equal to :func:`resolve_global_rank` on a trivial mesh.
    """
    if resolve_global_rank(config) == 0:
        return 0
    return dp_rank()


#: ``unwrap_model`` is re-exported from ``spectramr.core.module_utils``; naming it
#: here makes the re-export explicit rather than an apparently-unused import.
__all__ = [
    "is_rank_zero",
    "resolve_data_rank",
    "resolve_global_rank",
    "unwrap_model",
]
