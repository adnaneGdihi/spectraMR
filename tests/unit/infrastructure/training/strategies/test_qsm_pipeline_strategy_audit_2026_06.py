"""Regression tests for the 2026-06 strategy-audit fixes to
``qsm_pipeline_strategy`` (audit section ``### qsm_pipeline_strategy``).

Findings addressed here:

* [high|silent-fallback] The blanket ``try/except Exception`` in
  ``_compute_losses_impl`` swallowed any ``run_pipeline`` failure into a
  constant, grad-free ``loss_qsm_pipeline_error`` tensor, so QSM silently
  degraded to a no-op recon while smoke looked green (pitfall #10). The fix
  removes the catch so failures propagate and audit/smoke fail loudly.
* [low|silent-fallback] The same catch bound ``e`` but never logged/raised it,
  masking the diagnostic. Resolved by removing the catch.
* [critical|correctness-bug] (already fixed in-tree, pinned here as a
  regression guard) ``compute_field_map`` is called with all four required
  positional args — two single-echo phase maps plus two scalar echo times —
  not the old 2-arg call that raised ``TypeError``.
"""

from __future__ import annotations

import inspect

import torch

from spectramr.infrastructure.training.strategies import qsm_pipeline_strategy
from spectramr.infrastructure.training.strategies.qsm_pipeline_strategy import (
    QSMPipelineStrategy,
)


def _bare_strategy() -> QSMPipelineStrategy:
    """Build a QSM strategy without running the (heavy) parent ``__init__``."""
    s = object.__new__(QSMPipelineStrategy)
    s.device = torch.device("cpu")
    return s


class TestPipelineFailurePropagates:
    """Finding 2/4: run_pipeline failure must raise, not become a constant loss."""

    def test_run_pipeline_error_propagates(self, monkeypatch):
        s = _bare_strategy()

        sentinel = RuntimeError("qsm physics blew up")

        # Stub the parent loss so we isolate the QSM-specific path.
        def _fake_super_impl(**kwargs):
            return {"g_total_loss": torch.zeros(())}

        # Bind a fake super()._compute_losses_impl by patching the parent class
        # method on this instance's call path via a tiny subclass-free shim:
        # we instead drive _compute_losses_impl with run_pipeline raising and a
        # batch dict, and rely on the parent impl being stubbed below.
        s.run_pipeline = lambda batch: (_ for _ in ()).throw(sentinel)  # type: ignore[method-assign]

        # monkeypatch deletes the stub on undo; a plain assign-back would leave a
        # copy of the inherited hook in the subclass's own namespace.
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        monkeypatch.setattr(
            ReconstructionTrainingStrategy,
            "_compute_losses_impl",
            lambda self, **kw: _fake_super_impl(**kw),
        )
        raised = False
        try:
            s._compute_losses_impl(input_batch={"phase_echoes": torch.zeros(1)})
        except RuntimeError as exc:
            raised = True
            assert exc is sentinel
        assert raised, "pipeline failure must propagate, not be swallowed"

    def test_no_constant_error_loss_key_assignment_in_source(self):
        # The constant grad-free fallback assignment must be gone. (The key
        # name may still appear in an explanatory comment, so we check for the
        # active assignment idiom specifically, not a bare substring.)
        src = inspect.getsource(qsm_pipeline_strategy)
        assert 'base["loss_qsm_pipeline_error"]' not in src, (
            "the grad-free constant fallback loss assignment must be gone"
        )

    def test_no_blanket_except_in_compute_losses(self):
        src = inspect.getsource(QSMPipelineStrategy._compute_losses_impl)
        assert "except Exception" not in src, (
            "the blanket except (and its unused `e` binding) must be removed"
        )


class TestFieldMapFourArgs:
    """Finding 1 (regression guard): compute_field_map gets 4 args, no TypeError."""

    def test_run_pipeline_dual_echo_reaches_field_map(self):
        s = _bare_strategy()
        # QSM volumes are [B, n_echoes, D, H, W] (5-D) with >=2 echoes + two
        # echo times. compute_field_map slices phase[:, 0:1] / phase[:, 1:2].
        batch = {
            "phase_echoes": torch.zeros(1, 2, 4, 4, 4),
            "TE": torch.tensor([0.005, 0.010]),
        }
        out = s.run_pipeline(batch)
        # The 4-arg compute_field_map call must succeed and yield a field map
        # of the single-echo spatial shape (no TypeError from the old 2-arg call).
        assert "field_map" in out
        assert out["field_map"].shape == (1, 1, 4, 4, 4)
        # Full pipeline produced a susceptibility map without raising.
        assert "chi" in out

    def test_run_pipeline_too_few_echoes_returns_empty(self):
        s = _bare_strategy()
        batch = {
            "phase_echoes": torch.zeros(1, 1, 4, 4, 4),  # only one echo
            "TE": torch.tensor([0.005]),
        }
        assert s.run_pipeline(batch) == {}
