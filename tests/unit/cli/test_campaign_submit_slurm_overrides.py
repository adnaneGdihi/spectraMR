"""``campaign submit --slurm key=value``: the world size is a launch-time choice.

Under a sharded strategy ``num_devices`` sets the effective batch as well as the
topology -- there is no world-size LR scaling in this tree -- so a number
committed to a campaign YAML would fix an optimisation decision on behalf of
every later submitter, on whatever allocation they happen to hold. The flag
exists so the campaign says *what to run* and the submitter says *how wide*.

This file owns the argparse half: that the flag reaches the orchestrator as a
dict, and that a malformed ``key=value`` is refused here rather than becoming a
key nobody asked for. Key/type validation belongs to the orchestrator and is
tested against the schema in
``tests/unit/infrastructure/orchestration/test_campaign_orchestrator.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

import pytest

from spectramr.cli.app import campaign_submit


class _Recorder:
    """Stands in for CampaignOrchestrator, capturing what the CLI passed."""

    last_kwargs: ClassVar[dict] = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def submit_campaign(self, _config):
        class _S:
            @staticmethod
            def summary_table():
                return ""

        return _S()


@pytest.fixture
def recorder(monkeypatch):
    import spectramr.infrastructure.orchestration.campaign_orchestrator as mod

    _Recorder.last_kwargs = {}
    monkeypatch.setattr(mod, "CampaignOrchestrator", _Recorder)
    return _Recorder


def _args(**over) -> argparse.Namespace:
    base = {
        "config": Path("c.yaml"),
        "base_dir": None,
        "dry_run": True,
        "resume": False,
        "where": "slurm",
        "only": None,
        "include": None,
        "exclude": None,
        "slurm": None,
    }
    base.update(over)
    return argparse.Namespace(**base)


def test_repeated_flags_become_one_dict(recorder) -> None:
    assert campaign_submit(_args(slurm=["gpus=4", "cpus_per_task=32"])) == 0
    assert recorder.last_kwargs["slurm_overrides"] == {"gpus": "4", "cpus_per_task": "32"}


def test_absent_flag_passes_an_empty_dict(recorder) -> None:
    """Not ``None``: the orchestrator's 'no override' path stays one shape."""
    assert campaign_submit(_args()) == 0
    assert recorder.last_kwargs["slurm_overrides"] == {}


def test_a_value_containing_equals_survives(recorder) -> None:
    """Split once from the left -- ``mail_type=END,FAIL`` and friends."""
    assert campaign_submit(_args(slurm=["mail_type=END=FAIL"])) == 0
    assert recorder.last_kwargs["slurm_overrides"] == {"mail_type": "END=FAIL"}


def test_surrounding_whitespace_is_stripped(recorder) -> None:
    assert campaign_submit(_args(slurm=[" gpus = 4 "])) == 0
    assert recorder.last_kwargs["slurm_overrides"] == {"gpus": "4"}


def test_a_flag_without_equals_is_refused_before_submitting(recorder) -> None:
    """Planted: the shape the parse guard exists for.

    ``--slurm gpus 4`` is the natural typo, and argparse hands it over as the
    bare token ``gpus``. Without the check it would either crash on unpack or
    become a key nobody asked for; the campaign must not be submitted either
    way.
    """
    assert campaign_submit(_args(slurm=["gpus"])) == 1
    assert recorder.last_kwargs == {}, "nothing may be submitted after a parse failure"


def test_an_orchestrator_rejection_stops_the_submit(monkeypatch) -> None:
    """The second shape: the key parses but the orchestrator refuses it.

    Constructor validation raises ValueError; the CLI must return non-zero
    rather than let it escape as a traceback.
    """
    import spectramr.infrastructure.orchestration.campaign_orchestrator as mod

    def _boom(**_kwargs):
        raise ValueError("Unknown SLURM override key(s): gpu.")

    monkeypatch.setattr(mod, "CampaignOrchestrator", _boom)
    assert campaign_submit(_args(slurm=["gpu=4"])) == 1
