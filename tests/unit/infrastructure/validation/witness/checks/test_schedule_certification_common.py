"""``build_process_from_config`` builds one process per distinct construction.

Paired with
``src/spectramr/infrastructure/validation/witness/checks/schedule_certification_common.py``.

Three witnesses ask for a process from the same config on every run --
``schedule_allocation_checks:50``, ``schedule_nesting_checks:48`` and
``schedule_nesting_checks:122``. Un-memoised, each built its own
``KSpaceUndersamplingProcess``, and each of those built a mask generator whose
accelerator logs ``Creating Accelerator: ... | Params: {...}`` at INFO on first
use. A single-owner run therefore printed that line once per witness that
materialised a mask, with byte-identical params -- which reads as duplicated
construction somewhere in the training path and sent the #2056 investigation at
the wrong pair for a while.
"""

from __future__ import annotations

import logging

import pytest

from spectramr.infrastructure.validation.witness.checks.schedule_certification_common import (
    _PROCESS_MEMO,
    build_process_from_config,
)

_SAMPLING_LOGGER = "spectramr.infrastructure.physics.sampling"


def _doc(max_acceleration: float = 32.0, timesteps: int = 29) -> dict:
    """An experiment_11-shaped cold-diffusion config, as the witness sees it."""
    return {
        "model": {"model_type": "kspace_cold_diffusion"},
        "training": {"diffusion": {"timesteps": timesteps}},
        "undersampling": {
            "acceleration_type": "density_nested",
            "base_acceleration": 1.0,
            "max_acceleration": max_acceleration,
            "center_fraction": 0.08,
            "min_center_fraction": 0.02,
            "mask_seed": 42,
        },
    }


@pytest.fixture(autouse=True)
def _clear_memo():
    """The memo is module-level, so a stale entry would make any test vacuous."""
    _PROCESS_MEMO.clear()
    yield
    _PROCESS_MEMO.clear()


def test_the_same_config_yields_the_same_process():
    """THE pin. Three witnesses, one construction."""
    first = build_process_from_config(_doc())
    second = build_process_from_config(_doc())
    third = build_process_from_config(_doc())
    assert first is second is third


def test_one_accelerator_is_logged_however_many_witnesses_ask(caplog):
    """The observable the cluster log showed: the line appears once, not twice.

    Materialising is what logs, so each process is asked for its accelerator --
    three separate processes would print three times with identical params.
    """
    with caplog.at_level(logging.INFO, logger=_SAMPLING_LOGGER):
        for _ in range(3):
            process = build_process_from_config(_doc())
            process.mask_generator._get_accelerator(process.mask_type)

    created = [r for r in caplog.records if "Creating Accelerator" in r.getMessage()]
    assert len(created) == 1, f"expected one accelerator, got {len(created)}"


def test_configs_that_resolve_differently_do_not_share():
    """Anti-vacuity: a memo that returned one process for everything would pass
    the two tests above and silently certify the wrong ladder."""
    a = build_process_from_config(_doc(max_acceleration=32.0))
    b = build_process_from_config(_doc(max_acceleration=8.0))
    assert a is not b
    assert a.max_accel == 32.0
    assert b.max_accel == 8.0


def test_the_timestep_count_is_part_of_the_key():
    """Two arms can declare one ladder over different horizons."""
    a = build_process_from_config(_doc(timesteps=29))
    b = build_process_from_config(_doc(timesteps=1000))
    assert a is not b
    assert (a.num_timesteps, b.num_timesteps) == (29, 1000)


def test_a_non_cold_diffusion_config_still_builds_nothing():
    """The guard runs before the memo, so it cannot cache a ``None``."""
    doc = _doc()
    doc["model"]["model_type"] = "unet"
    assert build_process_from_config(doc) is None
