"""``_run_validation`` never hands a strategy a slice the collate padded in.

Two whole volumes of 2 and 8 slices collate to depth 8; flattened into rows, six
of the sixteen were exact zeros, and the cold-diffusion reverse sampler raised on
the first of them ("sample(s) [0] have no measured k-space bin").
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from spectramr.data.collation.strategies import ImageCollateStrategy
from spectramr.pipelines.train import _run_validation
from tests.unit.pipelines.test_validation_fail_loud import _logging_cfg, _make_pipeline


class _RecordingStrategy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(logging=_logging_cfg())
        self.rows: list[torch.Tensor] = []

    def validation_step(self, input_batch, target_batch, batch_idx=0):
        self.rows.extend(input_batch)
        return {"val_loss": 0.5}

    def finalize_validation(self) -> dict:
        return {}


def _volume(depth: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    kspace = torch.randn(1, 8, 8, depth, generator=gen) + 0.5
    return {"input": kspace, "target": kspace.clone()}


def test_no_validation_row_is_collate_padding() -> None:
    batch = ImageCollateStrategy().collate([_volume(2, 1), _volume(8, 2)])
    strategy = _RecordingStrategy()
    _run_validation(_make_pipeline([batch]), strategy, iteration=10, epoch=0, logging_service=None)
    assert len(strategy.rows) == 2 + 8
    assert all(float(row.abs().sum()) > 0 for row in strategy.rows)
