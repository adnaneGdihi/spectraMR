"""Regression: MAE k-space masking must keep the CLEAN image as the loss /
validation target, not the undersampled corruption.

In ``mask_domain == "kspace"`` the strategy FFTs the clean image, undersamples
it, and IFFTs back to a corrupted image. The pre-fix code overwrote
``input_batch`` with that corruption and then used it as BOTH the generator
input and the loss target — so the optimal solution is the identity map on the
corrupted data (CLAUDE.md pitfall #20), and validation PSNR grades agreement
with the corruption rather than reconstruction quality. The corrupted image must
feed the generator while the loss/metric targets the original clean image.
"""

from __future__ import annotations

import types

import torch

from spectramr.infrastructure.training.strategies.pretraining import (
    MAEPretrainingStrategy,
)
from spectramr.models.losses.computers.base import LossOutput


class _RecordingComputer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def compute(self, **kwargs):
        self.calls.append(kwargs)
        return LossOutput(
            total=torch.zeros((), requires_grad=True), components={}, metrics={}
        )


def _kspace_mae_config():
    return types.SimpleNamespace(
        training=types.SimpleNamespace(
            mae=types.SimpleNamespace(mask_domain="kspace", mask_ratio=0.75, patch_size=16)
        )
    )


def test_kspace_mae_loss_targets_clean_image_not_corruption():
    s = object.__new__(MAEPretrainingStrategy)
    rec = _RecordingComputer()
    clean = torch.randn(1, 1, 16, 16)

    s.loss_computer = rec
    s.device = torch.device("cpu")
    s._loss_dict_reuse = {}
    s.env = types.SimpleNamespace(generator=lambda x, **k: x, losses={})
    s.config = _kspace_mae_config()
    s._mask_rng = torch.Generator().manual_seed(0)
    # _resolve_legacy_batch returns a non-dict → strategy uses target_batch.
    s._resolve_legacy_batch = lambda ib, kw: ib

    s._compute_losses_impl(clean, clean, epoch=0)

    assert rec.calls, "loss_computer.compute was never called"
    target = rec.calls[0]["target"]
    # The generator received the corrupted image; the loss target must be the
    # ORIGINAL clean image. If the strategy still overwrites the target with the
    # corruption, this fails (target != clean).
    assert torch.allclose(target, clean), (
        "MAE k-space loss target is the undersampled corruption, not the clean "
        "image — degenerate identity objective (pitfall #20)."
    )
