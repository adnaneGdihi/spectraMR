"""The pre-1.0 checkpoint fixtures exist with the names the old writers produced."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "checkpoints_pre_v1"

EXPECTED = {
    "director_run": {
        "checkpoint_epoch_0000_step_000002.pt",
        "checkpoint_epoch_0001_step_000004.pt",
        "checkpoint_best.pt",
        "best.pt",
    },
    "deepspeed_run": {
        "checkpoint_epoch_0001_step_000004.pt",
        "checkpoint_best.pt",
        "checkpoint_best_ema.pt",
        "best.pt",
        "deepspeed",
    },
    "service_pth_run": {
        "checkpoint_step_4.pth",
        "checkpoint_best.pth",
        "best.pth",
        "manifest.json",
    },
    "service_safetensors_run": {
        "checkpoint_step_4.safetensors",
        "checkpoint_best.safetensors",
        "best.safetensors",
        "manifest.json",
    },
}


@pytest.mark.parametrize("run", sorted(EXPECTED))
def test_each_run_holds_what_its_writer_wrote(run: str) -> None:
    names = {p.name for p in (ROOT / run / "checkpoints").iterdir()}
    assert names == EXPECTED[run]


def test_the_best_aliases_are_symlinks() -> None:
    for run, alias in [
        ("director_run", "best.pt"),
        ("deepspeed_run", "best.pt"),
        ("service_pth_run", "best.pth"),
        ("service_safetensors_run", "best.safetensors"),
    ]:
        link = ROOT / run / "checkpoints" / alias
        assert link.is_symlink()
        assert link.resolve().exists()


def test_deepspeed_tags_use_the_pre_1_0_spelling() -> None:
    tags = {p.name for p in (ROOT / "deepspeed_run/checkpoints/deepspeed/generator").iterdir()}
    assert tags == {"epoch_0001_step_000004", "best", "latest"}
    latest = ROOT / "deepspeed_run/checkpoints/deepspeed/generator/latest"
    assert latest.read_text().strip() == "epoch_0001_step_000004"


def test_expected_draws_are_recorded_for_every_resumable_file() -> None:
    expected = json.loads((ROOT / "expected.json").read_text())
    assert set(expected) == {
        "director_run/checkpoint_epoch_0000_step_000002.pt",
        "director_run/checkpoint_epoch_0001_step_000004.pt",
        "deepspeed_run/checkpoint_epoch_0001_step_000004.pt",
    }
    for record in expected.values():
        assert set(record) == {"epoch", "step", "draws"}
        assert set(record["draws"]) == {"torch", "numpy", "python"}


def test_recorded_draws_differ_per_file() -> None:
    # Identical draws would match even if load_from never restored the RNG streams.
    expected = json.loads((ROOT / "expected.json").read_text())
    for stream in ("torch", "numpy", "python"):
        draws = [json.dumps(r["draws"][stream]) for r in expected.values()]
        assert len(set(draws)) == len(draws), stream
