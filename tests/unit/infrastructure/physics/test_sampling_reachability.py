"""Registered, reachable, firing — the three claims, kept separate.

Registering a family is the easy half. This repo is registry-dispatched and
YAML-driven, so a family can be registered, imported, tested and merged and still
never run for a config. The claims and what settles each:

* **registered** — the name is in the registry: a same-process membership check;
* **reachable** — a config can resolve it: a COLD SUBPROCESS that has not
  pre-imported the accelerator modules, driving the production path;
* **fires** — the mask actually degrades the tensor, not merely that nothing
  raised.

The cold subprocess is the one that separates registered from reachable: a
same-process test may have imported the module itself, proving nothing about what
a config would see.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from spectramr.infrastructure.physics.sampling import SUPPORTED_ACCELERATION_TYPES

#: Driven in a fresh interpreter, through AccelerationConfigSchema ->
#: resolve_undersampling_kwargs -> KSpaceUndersamplingProcess.q_sample, i.e. the
#: path a YAML arm actually takes.
_PROBE = """
import sys
import torch

DIRECTION = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
from spectramr.config.schemas.acceleration import AccelerationConfigSchema
from spectramr.infrastructure.physics.sampling import SUPPORTED_ACCELERATION_TYPES
from spectramr.models.diffusion.kspace_process import (
    KSpaceUndersamplingProcess,
    resolve_undersampling_kwargs,
)

failures = []
for name in sorted(SUPPORTED_ACCELERATION_TYPES):
    block = {
        "acceleration_type": name, "base_acceleration": 2.0, "max_acceleration": 8.0,
        "center_fraction": 0.04, "min_center_fraction": 0.02, "mask_seed": 42,
        "schedule_type": "linear",
    }
    if DIRECTION is not None:
        block["mask_direction"] = DIRECTION
    try:
        kwargs = resolve_undersampling_kwargs(
            AccelerationConfigSchema(**block), {}, phase_encode_axis=-1 if DIRECTION else None
        )
        process = KSpaceUndersamplingProcess(num_timesteps=8, **kwargs)
        x0 = torch.randn(2, 2, 64, 64)
        xt, mask = process.q_sample(x0, torch.tensor([7, 7]))
    except ValueError as exc:
        # multi_mask alternates between both axes and says so rather than
        # silently ignoring a declared one; that is correct, not a failure.
        if DIRECTION and "both k-space axes" in str(exc):
            continue
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        continue
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        continue
    kept = float(mask.float().mean())
    if kept >= 0.999:
        failures.append(f"{name}: mask keeps {kept:.3f} of k-space -- it does not degrade")
    if float((xt == 0).float().mean()) <= 0.05:
        failures.append(f"{name}: q_sample left the tensor essentially untouched")

print("FAILURES:" + ";".join(failures) if failures else "ALL_OK")
"""


@pytest.mark.parametrize("direction", [None, "phase"])
def test_every_family_is_registered_reachable_and_fires(direction: str | None) -> None:
    """One test, because the three claims are only meaningful together.

    Parametrised over ``mask_direction`` because the first version of this probe
    left it unset and therefore passed 18/18 while ``multi_mask`` was broken:
    once ``mask_direction`` became live (#948) it arrived as ``line_axis`` and
    collided with the axes that accelerator pins itself. A reachability probe
    that exercises only the default knob set is not a reachability probe.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, direction or ""],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, f"probe subprocess died:\n{completed.stderr[-2000:]}"
    verdict = completed.stdout.strip().splitlines()[-1]
    assert verdict == "ALL_OK", verdict


def test_the_probe_covers_every_registered_family() -> None:
    """A probe that silently skips families would pass while proving nothing."""
    assert len(SUPPORTED_ACCELERATION_TYPES) >= 18, (
        "families were removed from the registry; confirm that was intended"
    )
