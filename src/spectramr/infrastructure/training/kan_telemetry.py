"""KAN telemetry, collected at the logging cadence and left on the device.

The generator's ``get_kan_gate_telemetry`` / ``get_kan_trust_map_telemetry``
used to be read on every training step and drained with ``.tolist()``, three
device-to-host syncs per step whose values only a log row ever reads (#1448).
They are now read only on the steps ``logging.intervals.log`` selects (the knob
the training loop's own metric gate uses), and returned as device tensors, so the
loop's single fused host transfer is the only sync.
"""

from __future__ import annotations

from typing import Any

import torch

#: The generator helpers this module reads, in the order their keys are merged.
KAN_TELEMETRY_HELPERS: tuple[str, ...] = ("get_kan_gate_telemetry", "get_kan_trust_map_telemetry")


def kan_telemetry_due(step: int, log_interval: int) -> bool:
    """Whether *step* is one the training loop logs on its periodic cadence."""
    if log_interval < 1:
        raise ValueError(f"logging.intervals.log must be >= 1, got {log_interval!r}.")
    return step % log_interval == 0


def collect_kan_telemetry(model: Any, step: int, log_interval: int) -> dict[str, torch.Tensor]:
    """Device-tensor KAN telemetry for *step*, or ``{}`` off the logging cadence.

    *model* is the unwrapped generator. A helper the model does not expose
    contributes nothing.
    """
    if not kan_telemetry_due(step, log_interval):
        return {}
    out: dict[str, torch.Tensor] = {}
    for helper in KAN_TELEMETRY_HELPERS:
        fn = getattr(model, helper, None)
        if fn is not None:
            out.update(fn())
    return out


def kan_telemetry_keys(model: Any) -> frozenset[str]:
    """The keys :func:`collect_kan_telemetry` emits for *model*, for the CSV header.

    Read from the helpers themselves so the header and the producer cannot
    disagree; their key sets depend only on which modules the model holds.
    """
    keys: set[str] = set()
    for helper in KAN_TELEMETRY_HELPERS:
        fn = getattr(model, helper, None)
        if fn is not None:
            keys.update(fn())
    return frozenset(keys)


__all__ = [
    "KAN_TELEMETRY_HELPERS",
    "collect_kan_telemetry",
    "kan_telemetry_due",
    "kan_telemetry_keys",
]
