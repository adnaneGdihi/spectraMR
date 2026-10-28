"""``select_backend`` and the torch backend's command: which profiler runs, and why it says so."""

from __future__ import annotations

import pytest

from spectramr.cli.profile_backends import (
    TORCH_MIN_ITERATIONS,
    build_torch_command,
    scalene_gpu_flag,
    select_backend,
    torch_window,
)

FULL, CPU_ONLY, CPU_GPU, CPU_MEMORY = (), ("--cpu-only",), ("--gpu",), ("--memory",)


@pytest.mark.parametrize(
    ("flags", "samples_gpu"),
    [(FULL, True), (CPU_GPU, True), (CPU_ONLY, False), (CPU_MEMORY, False)],
)
def test_scalene_gpu_flag_follows_the_mode_table(flags, samples_gpu) -> None:
    assert scalene_gpu_flag(flags) is samples_gpu


@pytest.mark.parametrize(
    ("flags", "cuda", "scalene", "applied"),
    [
        (FULL, True, True, "torch"),
        (CPU_GPU, True, True, "torch"),
        (CPU_ONLY, True, True, "scalene"),
        (CPU_MEMORY, True, True, "scalene"),
        (FULL, False, True, "scalene"),
        (CPU_ONLY, True, False, "torch"),
        (CPU_ONLY, False, False, "scalene"),
    ],
)
def test_auto_picks_by_what_scalene_would_sample_and_what_exists(
    flags, cuda, scalene, applied
) -> None:
    choice = select_backend(
        "auto", mode_flags=flags, cuda_available=cuda, scalene_available=scalene
    )
    assert choice.applied == applied
    assert choice.requested == "auto"
    assert choice.reason


def test_an_explicit_torch_without_cuda_raises_rather_than_degrading() -> None:
    with pytest.raises(RuntimeError, match="no CUDA device"):
        select_backend("torch", mode_flags=FULL, cuda_available=False, scalene_available=True)


def test_an_explicit_scalene_is_never_overridden() -> None:
    choice = select_backend("scalene", mode_flags=FULL, cuda_available=True, scalene_available=True)
    assert choice.applied == "scalene"


def test_an_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="--backend must be one of"):
        select_backend("cprofile", mode_flags=FULL, cuda_available=True, scalene_available=True)


@pytest.mark.parametrize("iterations", [None, 0, TORCH_MIN_ITERATIONS - 1])
def test_torch_window_refuses_a_cap_with_no_recorded_step(iterations) -> None:
    with pytest.raises(ValueError, match=f"--iterations >= {TORCH_MIN_ITERATIONS}"):
        torch_window(iterations)


@pytest.mark.parametrize(("iterations", "active"), [(16, 1), (25, 10), (100, 20)])
def test_torch_window_closes_inside_the_iteration_cap(iterations, active) -> None:
    wait, warmup, got = torch_window(iterations)
    assert got == active
    assert wait + warmup + got <= iterations
    assert wait + warmup >= 8  # the first 8 steps write debug snapshots


def test_torch_command_carries_the_window_and_the_verb_after_the_separator(tmp_path) -> None:
    argv = build_torch_command(["train", "--config", "a.yaml"], out_dir=tmp_path, iterations=100)
    assert argv[1:3] == ["-m", "spectramr.cli.profile_torch_runner"]
    assert argv[argv.index("--out") + 1] == str(tmp_path)
    assert argv[argv.index("--active") + 1] == "20"
    assert argv[argv.index("--") + 1 :] == ["train", "--config", "a.yaml"]


def test_torch_runner_summary_divides_device_time_by_wall_time() -> None:
    from types import SimpleNamespace

    from spectramr.cli.profile_torch_runner import summarise

    total = SimpleNamespace(self_device_time_total=2_000_000.0)  # microseconds
    prof = SimpleNamespace(key_averages=lambda: SimpleNamespace(total_average=lambda: total))
    out = summarise(prof, wall_s=4.0, steps=20)
    assert out == {
        "active_steps": 20.0,
        "wall_s": 4.0,
        "device_kernel_s": 2.0,
        "device_over_wall": 0.5,
    }
