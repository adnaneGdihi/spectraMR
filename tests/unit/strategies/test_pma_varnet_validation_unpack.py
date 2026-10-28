"""Regression test for F3/E22 (smoke_audit_20260521) —
``ConcretePMAVarNetStrategy.validation_step`` must unwrap
``TrainingBatch`` before passing to ``_compute_losses_impl``.

Root cause: the strategy's ``validation_step`` passed ``val_batch``
(a ``TrainingBatch`` dataclass) directly as ``target_batch``, which
crashed at line 79 with ``AttributeError: 'TrainingBatch' object has
no attribute 'shape'`` across 4 experiments:

  experiment_pma_01_baseline
  experiment_pma_02_b0_corruption
  experiment_pma_03_full_m4raw
  experiment_jfe_varnet_sota

Fix: call ``self._unpack_batch(val_batch)`` (provided by the
``BatchPreparationMixin``) so the TrainingBatch is split into
``(input_batch, target_batch)`` tensors before they reach
``_compute_losses_impl``.

This test drives ``validation_step`` with a real ``TrainingBatch``-shaped object and
checks what reaches ``_compute_losses_impl``, so a revert fails here, not on the cluster.
"""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from spectramr.infrastructure.training.strategies.pma_varnet_strategy import (
    ConcretePMAVarNetStrategy,
)


class _Batch:
    """Stands in for a TrainingBatch: it has no ``.shape``, which is what crashed."""


def test_validation_step_unpacks_training_batch() -> None:
    inp, tgt = torch.zeros(1, 2, 4, 4), torch.ones(1, 1, 4, 4)
    batch = _Batch()
    seen: dict[str, object] = {}

    s = ConcretePMAVarNetStrategy.__new__(ConcretePMAVarNetStrategy)
    s._unpack_batch = lambda b: (seen.update(unpacked=b) or (inp, tgt))
    s.snapshot_source = lambda phase: nullcontext()
    s._capture_model_output = lambda **kw: nullcontext()
    s.env = SimpleNamespace(generator=None)

    def _losses(input_batch, target_batch, epoch):
        seen.update(input=input_batch, target=target_batch)
        s._last_prediction, s._last_target = tgt, tgt
        return {"loss": torch.tensor(0.0)}

    s._compute_losses_impl = _losses
    s._get_validation_metrics_computer = lambda cfg: SimpleNamespace(compute=lambda p, t: {})
    s.config = None

    s.validation_step(batch, batch_idx=0)
    assert seen["unpacked"] is batch
    assert seen["input"] is inp and seen["target"] is tgt
