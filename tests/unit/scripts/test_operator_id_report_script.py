"""``scripts/figures/operator_id_report.py`` loads the conditioner strictly.

The conditioner is the arm's whole generator, so a file whose ``generator`` part
covers only some of its parameters must raise rather than leave the rest at init.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from tests.fixtures.run_state_files import write_run_state
from tests.utils.repo_scripts import require_repo_file

_SCRIPT = require_repo_file("scripts/figures/operator_id_report.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("operator_id_report_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load_module()


@pytest.fixture
def tiny_conditioner(monkeypatch):
    """A stand-in so the test exercises the load, not the real basis construction."""
    monkeypatch.setattr(report, "BCHOperatorConditioner", lambda **_kw: nn.Linear(2, 2))


@pytest.mark.unit
def test_the_generator_part_loads(tmp_path, tiny_conditioner):
    path = write_run_state(
        tmp_path / "best.pt",
        {"generator": {"weight": torch.full((2, 2), 3.0), "bias": torch.ones(2)}},
    )
    cond = report._load_conditioner(str(path))
    assert torch.equal(cond.weight, torch.full((2, 2), 3.0))
    assert not cond.training


@pytest.mark.unit
def test_a_partial_part_raises(tmp_path, tiny_conditioner):
    path = write_run_state(tmp_path / "best.pt", {"generator": {"weight": torch.zeros(2, 2)}})
    with pytest.raises(RuntimeError, match="bias"):
        report._load_conditioner(str(path))
