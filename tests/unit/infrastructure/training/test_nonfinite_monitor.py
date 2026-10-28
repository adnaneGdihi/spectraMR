"""``NonFiniteMonitor``: the per-step NaN check without a per-step host sync.

The monitor answers one step late (lag 1), so these pin which call reports a
non-finite step and which iteration the verdict names. All run on CPU except the
sync-debug test, which needs CUDA to have a sync to catch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.nonfinite_monitor import (  # noqa: E402
    NonFiniteMonitor,
    NonFiniteVerdict,
    guard_nonfinite_loss,
)

pytestmark = pytest.mark.unit

CPU = torch.device("cpu")
NAN = float("nan")


def _monitor() -> NonFiniteMonitor:
    return NonFiniteMonitor(CPU, distributed=False)


def test_a_finite_step_returns_none():
    monitor = _monitor()
    monitor.record(torch.tensor(0.5), name="g_total_loss")
    assert monitor.end_step(1) is None
    monitor.record(torch.tensor(0.4), name="g_total_loss")
    assert monitor.end_step(2) is None
    assert monitor.resolve() is None


def test_a_nan_is_reported_by_the_next_step_and_names_its_own_iteration():
    monitor = _monitor()
    for iteration in (1, 2):
        monitor.record(torch.tensor(0.5), name="g_total_loss")
        assert monitor.end_step(iteration) is None
    monitor.record(torch.tensor(0.5), name="gen")
    monitor.record(torch.tensor(NAN), name="g_total_loss")
    # Lag 1: the NaN step's own end_step only queues its flags.
    assert monitor.end_step(3) is None
    monitor.record(torch.tensor(0.5), name="g_total_loss")
    verdict = monitor.end_step(4)
    assert verdict == NonFiniteVerdict(iteration=3, names=("g_total_loss",), on_peer_rank=False)


def test_resolve_reports_a_pending_nan_immediately():
    monitor = _monitor()
    monitor.record(torch.tensor([1.0, float("inf")]), name="gen")
    assert monitor.end_step(7) is None
    verdict = monitor.resolve()
    assert verdict is not None and verdict.iteration == 7 and verdict.names == ("gen",)
    # Drained: a second resolve has nothing left to report.
    assert monitor.resolve() is None


def test_a_step_with_no_records_is_clean():
    monitor = _monitor()
    assert monitor.end_step(1) is None
    assert monitor.end_step(2) is None
    assert monitor.resolve() is None


def test_a_flagged_step_does_not_leak_into_the_next():
    monitor = _monitor()
    monitor.record(torch.tensor(NAN), name="gen")
    monitor.end_step(1)
    assert monitor.end_step(2).iteration == 1
    assert monitor.end_step(3) is None


class TestGuardWithoutAMonitorStillRaises:
    """Direct callers outside the training loop keep the synchronous raise."""

    def test_non_finite_raises(self):
        with pytest.raises(RuntimeError, match="Non-finite loss for config 'g' at step 4"):
            guard_nonfinite_loss(
                torch.tensor(NAN), name="g", global_step=4, scaler=None, monitor=None
            )

    def test_a_scaler_defers(self):
        guard_nonfinite_loss(
            torch.tensor(NAN), name="g", global_step=4, scaler=object(), monitor=None
        )

    def test_a_monitor_records_instead_of_raising(self):
        monitor = _monitor()
        guard_nonfinite_loss(
            torch.tensor(NAN), name="g", global_step=4, scaler=None, monitor=monitor
        )
        monitor.end_step(4)
        assert monitor.resolve().names == ("g",)

    def test_a_scaler_records_nothing_even_with_a_monitor(self):
        monitor = _monitor()
        guard_nonfinite_loss(
            torch.tensor(NAN), name="g", global_step=4, scaler=object(), monitor=monitor
        )
        monitor.end_step(4)
        assert monitor.resolve() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to observe a sync")
def test_record_and_end_step_never_sync_on_cuda():
    device = torch.device("cuda")
    monitor = NonFiniteMonitor(device, distributed=False)
    losses = [torch.tensor(v, device=device) for v in (0.5, NAN, 0.25, 0.125)]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        verdicts = []
        for iteration, loss in enumerate(losses, start=1):
            monitor.record(loss, name="g_total_loss")
            monitor.record(loss * 2, name="gen")
            verdicts.append(monitor.end_step(iteration))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert verdicts[:2] == [None, None]
    assert verdicts[2] == NonFiniteVerdict(2, ("g_total_loss", "gen"), on_peer_rank=False)
    assert verdicts[3] is None
    assert monitor.resolve() is None
