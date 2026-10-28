"""KAN grid extension must arm on a resumed run, not only at iteration 1.

The warm-up enabled sample collection on ``current_step == 1`` and re-fitted the
spline knots every 2K iterations through the first 10K. Production chains
training in wall-clock segments ("PROD implies --resume if-present"), and a
resumed segment starts at the checkpoint's step, so iteration 1 is never
emitted: collection stayed off, every update ran on an empty buffer, and
``update_kan_grids()`` returned 0 without logging. The mechanism was inert on
every arm that actually trains to completion.

Arming is therefore a state, not an edge. Pinned at source because exercising it
for real needs a full training loop over 10K iterations.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("torch")

from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)


def _kan_block() -> str:
    src = inspect.getsource(DiffusionTrainingStrategy._compute_losses_impl)
    assert "KAN grid extension" in src, "the warm-up block moved; update this test"
    return src.split("KAN grid extension", 1)[1][:1600]


def test_collection_is_not_armed_by_an_edge_on_the_first_iteration():
    """The regression: `if current_step == 1:` never fires on a resumed segment."""
    block = _kan_block()
    code = "\n".join(ln for ln in block.splitlines() if not ln.strip().startswith("#"))
    assert "current_step == 1" not in code, (
        "arming is edge-triggered again; a resumed run will collect nothing"
    )


def test_collection_is_armed_from_state_so_a_resumed_segment_catches_up():
    block = _kan_block()
    assert "_kan_collection_armed" in block
    assert "0 < current_step <= kan_warmup_iters" in block, (
        "arming must hold anywhere inside the warm-up window, not at one step"
    )


def test_collection_is_still_disabled_after_the_warm_up_window():
    """Guards the fix from leaving the CPU sample buffer on for the whole run."""
    block = _kan_block()
    assert "set_kan_sample_collection(False)" in block
    assert "current_step > kan_warmup_iters" in block


def test_the_update_still_fires_on_the_two_thousand_step_cadence():
    """Guards the arming change from having swallowed the update branch."""
    block = _kan_block()
    assert "current_step % kan_update_every == 0" in block
    assert "update_kan_grids()" in block
