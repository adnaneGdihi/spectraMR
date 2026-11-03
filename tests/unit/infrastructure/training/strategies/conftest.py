"""Shared strategy-test fixtures.

`get_last_metrics` must not sync the GPU (#707).

`training_loop` calls it on EVERY iteration, outside the `log_interval` gate.
Returning Python floats meant `float(cuda_tensor)` -- which IS `.item()` -- once
per component metric, so a GAN step publishing ~12 of them paid 12 host round
trips per step and discarded the result on every non-logging step.

The sharp part: the closures already keep these tensors on-device *deliberately*,
with a "no sync: NN#9" comment naming this method as where conversion moved to.
The conversion moved; the per-step call site did not, so the net sync count never
changed. That is why this is asserted by COUNTING syncs rather than by checking a
type -- a type check would have passed before the fix too, on the closure side.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

N_METRICS = 12  # roughly what a GAN step publishes


class _SyncCounter(torch.Tensor):
    """A tensor that counts host round-trips (`.item()` / `float()`)."""

    calls = 0

    @staticmethod
    def make(value: float) -> _SyncCounter:
        return torch.tensor(value).as_subclass(_SyncCounter)

    def item(self):
        _SyncCounter.calls += 1
        return super().item()

    def __float__(self):
        _SyncCounter.calls += 1
        return super().__float__()


def assert_no_sync(get_last_metrics) -> None:
    """Call the accessor with counting tensors and assert zero host transfers."""
    stub = types.SimpleNamespace()
    stub._last_step_metrics = {
        f"m{i}": _SyncCounter.make(float(i)) for i in range(N_METRICS)
    }
    _SyncCounter.calls = 0

    out = get_last_metrics(stub)

    assert _SyncCounter.calls == 0, (
        f"{_SyncCounter.calls} GPU sync(s) during get_last_metrics; the loop's "
        "log_interval-gated conversion must be the only converter (#707)"
    )
    assert len(out) == N_METRICS
    assert all(isinstance(v, torch.Tensor) for v in out.values()), (
        "values must stay on-device so the loop can batch the transfer"
    )


def assert_empty_is_safe(get_last_metrics) -> None:
    """A strategy that published nothing must return an empty dict, not raise."""
    assert get_last_metrics(types.SimpleNamespace()) == {}


@pytest.fixture
def no_gpu_sync():
    """Assert an accessor performs zero host round-trips."""
    return assert_no_sync


@pytest.fixture
def empty_metrics_is_safe():
    return assert_empty_is_safe


@pytest.fixture
def declared(monkeypatch):
    """Trace the whole test; ``declared()`` returns the last ``_declare_model_input`` call.

    Returns ``(tensors, extra, in_kspace_keys)`` -- whole tensors, the declared extra,
    and the k-space keys exactly as the strategy passed them -- or ``None``.
    """
    from spectramr.core import snapshot_trace as st
    from spectramr.infrastructure.training.snapshot.policy import ViewPolicy
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    monkeypatch.setattr(BaseTrainingStrategy, "_snapshot_view_policy", ViewPolicy())
    trace = st.SnapshotTrace(phase="train", step=0, batched=False, max_repeats=1000)
    token = st._ACTIVE.set(trace)

    def _last():
        stages = [s for s in trace.stages if s.group == "strategy" and s.name.startswith("model_input")]
        if not stages:
            return None
        stage = stages[-1]
        prefix = f"strategy/{stage.name}."
        extra = {k[len(prefix) :]: v for k, v in trace.extra.items() if k.startswith(prefix)}
        keys = stage.params.get("in_kspace_keys")
        return stage.tensors, extra, None if keys is None else set(keys)

    yield _last
    st._ACTIVE.reset(token)
