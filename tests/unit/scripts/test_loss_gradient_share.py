"""``scripts/diagnostics/loss_gradient_share.py`` reads a RunState before it builds.

The environment build takes minutes, so the checkpoint is checked first; the step it
returns seeds the curriculum difficulty, and a lost step would silently score rung 0.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from tests.fixtures.run_state_files import write_run_state
from tests.utils.repo_scripts import require_repo_file

_SCRIPT = require_repo_file("scripts/diagnostics/loss_gradient_share.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("loss_gradient_share", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lgs = _load_module()


def _file(tmp_path, *, ema: bool):
    return write_run_state(
        tmp_path / "step_00000010.pt",
        {"generator": {"weight": torch.zeros(2, 2)}},
        ema={"weight": torch.ones(2, 2)} if ema else None,
        role="step",
    )


@pytest.mark.unit
def test_read_returns_the_step_the_file_was_written_at(tmp_path):
    assert lgs.read_checkpoint(_file(tmp_path, ema=False), ema=False) == 10


@pytest.mark.unit
def test_ema_without_an_ema_part_fails_before_the_build(tmp_path):
    with pytest.raises(SystemExit, match="no ema part"):
        lgs.read_checkpoint(_file(tmp_path, ema=False), ema=True)


@pytest.mark.unit
@pytest.mark.parametrize(("ema", "expected"), [(False, 0.0), (True, 1.0)])
def test_load_weights_reads_the_selected_part(tmp_path, ema, expected):
    model = nn.Linear(2, 2, bias=False)
    lgs.load_weights(model, _file(tmp_path, ema=True), ema)
    assert torch.equal(model.weight, torch.full((2, 2), expected))
