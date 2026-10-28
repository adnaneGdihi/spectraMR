"""KAN telemetry is read at the logging cadence and never synced off it (#1448)."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.kan_telemetry import (
    KAN_TELEMETRY_HELPERS,
    collect_kan_telemetry,
    kan_telemetry_due,
    kan_telemetry_keys,
)

pytestmark = pytest.mark.unit


def _forbid_host_syncs(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a, **_k):
        raise AssertionError("device->host sync inside the training step")

    monkeypatch.setattr(torch.Tensor, "tolist", _raise)
    monkeypatch.setattr(torch.Tensor, "item", _raise)


class _CountingGenerator:
    def __init__(self) -> None:
        self.calls = dict.fromkeys(KAN_TELEMETRY_HELPERS, 0)

    def get_kan_gate_telemetry(self) -> dict[str, torch.Tensor]:
        self.calls["get_kan_gate_telemetry"] += 1
        return {"kan_gate/img": torch.tensor(0.5)}

    def get_kan_trust_map_telemetry(self) -> dict[str, torch.Tensor]:
        self.calls["get_kan_trust_map_telemetry"] += 1
        return {"kan_trust/mean": torch.tensor(0.7)}


@pytest.mark.parametrize(("n_steps", "interval"), [(20, 5), (12, 4), (7, 1)])
def test_helpers_are_called_once_per_log_interval(monkeypatch, n_steps, interval) -> None:
    gen = _CountingGenerator()
    _forbid_host_syncs(monkeypatch)
    rows = [collect_kan_telemetry(gen, step, interval) for step in range(1, n_steps + 1)]
    assert gen.calls == dict.fromkeys(KAN_TELEMETRY_HELPERS, n_steps // interval)
    for step, row in enumerate(rows, start=1):
        if step % interval:
            assert row == {}
        else:
            assert set(row) == {"kan_gate/img", "kan_trust/mean"}
            assert all(isinstance(v, torch.Tensor) for v in row.values())


def test_planted_per_step_collection_fails_the_cadence_check() -> None:
    """PLANTED: the pre-#1448 loop read both helpers on every step."""
    gen = _CountingGenerator()
    for _ in range(1, 21):
        for helper in KAN_TELEMETRY_HELPERS:
            getattr(gen, helper)()
    assert gen.calls != dict.fromkeys(KAN_TELEMETRY_HELPERS, 20 // 5)


def test_a_model_without_the_helpers_contributes_nothing() -> None:
    assert collect_kan_telemetry(object(), 10, 5) == {}


@pytest.mark.parametrize("bad", [0, -5])
def test_a_non_positive_interval_is_refused(bad: int) -> None:
    """PLANTED: a zero interval would divide by zero; a negative one would mis-gate."""
    with pytest.raises(ValueError, match=r"intervals\.log"):
        kan_telemetry_due(10, bad)


def test_generator_helpers_return_device_tensors_without_syncing(monkeypatch) -> None:
    """The real helpers, called unbound on a tiny host module, never call .tolist()/.item()."""
    from spectramr.infrastructure.physics.kan_data_consistency import (
        KANAdaptiveDataConsistency,
    )
    from spectramr.models.blocks.dual_domain_attention_kan import KANGatedDualDomainAttention
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceColdDiffusionGenerator as Gen,
    )

    host = nn.Module()
    host.attn = KANGatedDualDomainAttention(in_channels=8, feature_domain="kspace")
    host.attn._last_gates.copy_(torch.tensor([0.2, 0.4, 0.6]))
    host.dc_layer = KANAdaptiveDataConsistency()
    host.dc_layer._last_trust_stats.copy_(torch.tensor([0.9, 0.1, 0.5, 0.2]))

    _forbid_host_syncs(monkeypatch)
    gates = Gen.get_kan_gate_telemetry(host)
    trust = Gen.get_kan_trust_map_telemetry(host)
    monkeypatch.undo()

    assert all(isinstance(v, torch.Tensor) and v.numel() == 1 for v in gates.values())
    assert float(gates["kan_gate/kspace"]) == pytest.approx(0.4)
    assert float(gates["kan_gate/img_active_blocks"]) == 1.0
    assert float(trust["kan_trust/center"]) == pytest.approx(0.9)
    host.dc_layer._last_trust_stats.zero_()
    assert float(trust["kan_trust/center"]) == pytest.approx(0.9), "must not alias the buffer"


def test_the_diffusion_step_routes_through_the_gated_collector() -> None:
    """Wiring: the per-step path calls the collector, not the helpers directly."""
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    src = inspect.getsource(DiffusionTrainingStrategy._compute_losses_impl)
    assert "collect_kan_telemetry(inner, current_step, self._cached_log_interval)" in src
    assert "torch.as_tensor(float(v))" not in src


def test_header_keys_match_what_collection_emits() -> None:
    gen = _CountingGenerator()
    assert kan_telemetry_keys(gen) == set(collect_kan_telemetry(gen, 5, 5))
    assert kan_telemetry_keys(object()) == frozenset()
