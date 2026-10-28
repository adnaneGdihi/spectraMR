"""The distributed-training helpers against a real two-rank gloo group.

gloo has no ``ReduceOp.AVG``, so the mean reductions raised there and the
callers' ``except`` blocks turned that into a logged warning. CPU gloo is enough
to exercise them for real, so this needs no accelerator.
"""

from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

WORLD = 2


def _worker(rank: int, store: str, scenario: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    try:
        if scenario == "mean":
            _check_mean(rank)
        elif scenario == "loss_sync":
            _check_loss_sync(rank)
        elif scenario == "validity":
            _check_validity(rank)
    finally:
        dist.destroy_process_group()


def _check_mean(rank: int) -> None:
    from spectramr.infrastructure.distributed.distributed_training import (
        synchronize_across_ranks,
    )

    assert synchronize_across_ranks(float(rank * 4), "mean") == 2.0
    assert synchronize_across_ranks(float(rank + 1), "sum") == 3.0
    assert synchronize_across_ranks(float(rank), "max") == 1.0


def _check_loss_sync(rank: int) -> None:
    from torch import nn

    from spectramr.infrastructure.distributed.distributed_training import DistributedLossAudit

    audit = DistributedLossAudit(nn.MSELoss())
    out = torch.full((2,), float(rank + 1))
    result = audit.audit_loss_computation({}, out, torch.zeros(2), backward=False)
    assert result.loss_valid
    # The mean over ranks, not just "did not raise": the old AVG failed on gloo and
    # the except block swallowed it, leaving this False.
    assert result.all_losses_synchronized


def _check_validity(rank: int) -> None:
    from torch import nn

    from spectramr.infrastructure.distributed.distributed_training import DistributedLossAudit

    audit = DistributedLossAudit(nn.MSELoss())
    assert audit.validate_all_ranks() is False  # nothing audited yet

    good = torch.zeros(2)
    bad = torch.full((2,), float("nan"))
    audit.audit_loss_computation({}, good, torch.zeros(2), backward=False)
    assert audit.validate_all_ranks() is True

    # One rank produces a non-finite loss: every rank must see the failure.
    audit.audit_loss_computation({}, bad if rank == 1 else good, torch.zeros(2), backward=False)
    assert audit.validate_all_ranks() is False


def _spawn(scenario: str) -> None:
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(os.path.join(tmp, "store"), scenario), nprocs=WORLD, join=True)


def test_mean_reduction_works_on_gloo() -> None:
    _spawn("mean")


def test_loss_synchronisation_reports_success_on_gloo() -> None:
    _spawn("loss_sync")


def test_validate_all_ranks_reflects_the_last_audit() -> None:
    _spawn("validity")


def test_a_failed_process_group_init_raises_instead_of_warning() -> None:
    """Warning and carrying on left every rank running as an independent job."""
    from spectramr.infrastructure.distributed.distributed_training import (
        DistributedTrainingContext,
    )

    with pytest.raises(ValueError), DistributedTrainingContext(backend="no_such_backend"):
        pass
