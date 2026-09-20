"""Unit tests for :mod:`spectramr.core.rng_state`.

The property under test is replay: after ``restore_rng_state``, the next draws
must be the ones that *would* have followed the capture. A resume that gets
this wrong still trains and still reports success, so nothing but an equality
check on the draw sequence detects it.
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from spectramr.core.rng_state import (
    RNG_STATE_SAFE_GLOBALS,
    capture_rng_state,
    restore_rng_state,
)


def _draw() -> tuple[float, float, float]:
    """One draw from each stream a training step consumes."""
    return (
        float(torch.rand(1).item()),
        float(np.random.rand()),
        random.random(),
    )


@pytest.mark.unit
def test_restore_replays_the_captured_sequence():
    """The defining property: same state in, same numbers out."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    state = capture_rng_state()
    expected = [_draw() for _ in range(4)]

    # Advance all three streams well past the capture point.
    for _ in range(50):
        _draw()

    restore_rng_state(state)
    assert [_draw() for _ in range(4)] == expected


@pytest.mark.unit
def test_without_restore_the_streams_diverge():
    """The planted violation: this is what an unrestored resume looks like."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    capture_rng_state()
    expected = [_draw() for _ in range(4)]
    for _ in range(50):
        _draw()

    assert [_draw() for _ in range(4)] != expected


@pytest.mark.unit
def test_capture_covers_every_stream():
    captured = capture_rng_state()
    assert {"torch", "numpy", "python"} <= set(captured)


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["torch", "numpy", "python"])
def test_partial_state_is_tolerated(missing):
    """An older checkpoint lacking one stream restores the others."""
    state = capture_rng_state()
    state.pop(missing)
    restore_rng_state(state)  # must not raise


@pytest.mark.unit
def test_empty_state_is_a_noop():
    restore_rng_state({})


@pytest.mark.unit
def test_cuda_state_is_skipped_when_device_count_differs(caplog):
    """Resuming a 4-GPU checkpoint on a 2-GPU node must not raise.

    ``set_rng_state_all`` requires exactly one state per visible device, so the
    mismatch has to be detected rather than passed through — and it is logged,
    because it silently changes the stochastic trajectory.
    """
    state = capture_rng_state()
    state["cuda"] = [torch.zeros(16, dtype=torch.uint8) for _ in range(4)]

    with (
        patch("spectramr.core.rng_state.torch.cuda.is_available", return_value=True),
        patch("spectramr.core.rng_state.torch.cuda.device_count", return_value=2),
        patch("spectramr.core.rng_state.torch.cuda.set_rng_state_all") as set_all,
        caplog.at_level("WARNING"),
    ):
        restore_rng_state(state)

    set_all.assert_not_called()
    assert "NOT restored" in caplog.text


@pytest.mark.unit
def test_cuda_state_is_restored_when_the_device_count_matches():
    state = capture_rng_state()
    state["cuda"] = [torch.zeros(16, dtype=torch.uint8) for _ in range(2)]

    with (
        patch("spectramr.core.rng_state.torch.cuda.is_available", return_value=True),
        patch("spectramr.core.rng_state.torch.cuda.device_count", return_value=2),
        patch("spectramr.core.rng_state.torch.cuda.set_rng_state_all") as set_all,
    ):
        restore_rng_state(state)

    set_all.assert_called_once()


@pytest.mark.unit
def test_numpy_state_survives_a_torch_save_round_trip(tmp_path):
    """The envelope must reload under the allowlist the constant advertises.

    ``np.random.get_state()`` carries a uint32 ndarray, which torch's
    weights-only loader refuses without these four reconstructors — so a
    checkpoint that saves fine would be unloadable.
    """
    path = tmp_path / "rng.pt"
    torch.save({"rng_state": capture_rng_state()}, path)

    with torch.serialization.safe_globals(RNG_STATE_SAFE_GLOBALS):
        blob = torch.load(path, weights_only=True)

    assert {"torch", "numpy", "python"} <= set(blob["rng_state"])


@pytest.mark.unit
def test_restore_moves_a_device_resident_blob_back_to_cpu():
    """The GPU-only shape, reproduced without needing a working GPU.

    ``CheckpointDirector.load_from`` loads with
    ``map_location=pipeline.device``, so on the cluster every blob in the
    envelope arrives on ``cuda:N`` -- and the generators reject a non-CPU
    tensor (``check_rng_state`` asserts ``device().type() == kCPU``). Unfixed,
    that turns every GPU resume into a failed load, which is worse than the
    missing restore it replaced, and no CPU test can see it because there
    ``map_location`` is the identity.

    A real ``.cuda()`` transfer is deliberately NOT used: this box reports
    ``torch.cuda.is_available() == True`` and then raises
    ``cudaErrorNoKernelImageForDevice`` on the copy (Thor sm_110 against the
    pinned cu126 build), which would make the test flaky rather than skipped.
    ``spec=torch.Tensor`` is what ``torch.is_tensor`` checks, so the coercion
    sees a tensor.
    """
    cpu_blob = capture_rng_state()["torch"]
    device_blob = MagicMock(spec=torch.Tensor)
    device_blob.cpu.return_value = cpu_blob

    with patch("spectramr.core.rng_state.torch.set_rng_state") as set_state:
        restore_rng_state({"torch": device_blob})

    device_blob.cpu.assert_called_once()
    # Identity, not equality: a `spec=torch.Tensor` mock inherits Tensor's
    # `__eq__`, which returns a truthy mock and makes `assert_called_once_with`
    # pass whether or not the coercion happened.
    assert set_state.call_args.args[0] is cpu_blob


@pytest.mark.unit
def test_restore_moves_device_resident_cuda_blobs_back_to_cpu():
    """``set_rng_state_all`` refuses device tensors for the same reason."""
    cpu_blobs = [torch.zeros(16, dtype=torch.uint8) for _ in range(2)]
    device_blobs = []
    for blob in cpu_blobs:
        mock = MagicMock(spec=torch.Tensor)
        mock.cpu.return_value = blob
        device_blobs.append(mock)

    with (
        patch("spectramr.core.rng_state.torch.cuda.is_available", return_value=True),
        patch("spectramr.core.rng_state.torch.cuda.device_count", return_value=2),
        patch("spectramr.core.rng_state.torch.cuda.set_rng_state_all") as set_all,
    ):
        restore_rng_state({"cuda": device_blobs})

    passed = set_all.call_args.args[0]
    assert all(p is c for p, c in zip(passed, cpu_blobs, strict=True))
