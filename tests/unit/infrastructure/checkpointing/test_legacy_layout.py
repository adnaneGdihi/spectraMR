"""The resume path rebuilds the module layout a checkpoint was written with (#3363)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from spectramr.infrastructure.checkpointing.codecs import model_state_keys
from spectramr.infrastructure.checkpointing.legacy_layout import (
    LEGACY_REP_FUSION,
    checkpoint_layout_kwargs,
)


def _write(path: Path, wheres: list[str]) -> Path:
    save_file({f"{i}:{w}": torch.zeros(1) for i, w in enumerate(wheres)}, str(path))
    return path


def test_model_state_keys_reads_one_model_from_the_tensor_names(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "step.safetensors",
        [
            "models.generator.backbone.conv.weight",
            "models.ema.backbone.conv.weight",
            "optimizers.g.0",
        ],
    )
    assert model_state_keys(path, "generator") == ["backbone.conv.weight"]


def test_a_pre_3363_bridge_checkpoint_asks_for_the_legacy_fusion(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "step.safetensors",
        [
            "models.generator.backbone.rep_fusion.fusion_blocks.0.conv.weight",
            "models.generator.backbone.backbone.inc.weight",
        ],
    )
    assert checkpoint_layout_kwargs(path) == {LEGACY_REP_FUSION: True}


@pytest.mark.parametrize(
    "wheres",
    [
        ["models.generator.backbone.backbone.inc.weight"],
        # The generator's own (declared) fusion is a different module.
        ["models.generator.rep_fusion.fuse.weight"],
        # Only the generator's layout is read.
        ["models.ema.backbone.rep_fusion.fusion_blocks.0.conv.weight"],
    ],
)
def test_a_current_layout_checkpoint_asks_for_nothing(tmp_path: Path, wheres: list[str]) -> None:
    assert checkpoint_layout_kwargs(_write(tmp_path / "step.safetensors", wheres)) == {}


def test_a_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        checkpoint_layout_kwargs(tmp_path / "absent.safetensors")
