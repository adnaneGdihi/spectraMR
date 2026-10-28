"""Regression tests for the 2026-06 audit fixes on ``ConcretePMAVarNetStrategy``.

Covers the ``pma_varnet_strategy`` section of the 2026-05-31 strategy audit
(see ``TODO/strategy_audit_2026_05_31_unreviewed_findings.txt`` and
``TODO/_high_triage_full.txt``).

Findings status:
- [critical] Simulator 3-tuple destructured in wrong order (line 96): ALREADY
  FIXED upstream (commit abe98aafb). A source-level guard test below locks in
  the corrected unpack so a regression cannot silently re-swap the tuple.
- [high] Marker-pinning loss fed swapped tensors (lines 159-164): corollary of
  the above; guarded by the same source-level test.
- [medium] Ad-hoc ``torch.manual_seed`` in ``validation_step`` (line 194):
  FIXED here — the reseed is removed; ``validation_step`` no longer clobbers
  global RNG state (the val path is a deterministic ``no_grad`` forward and
  determinism is owned by ``initialize_accelerator``).
- [low] SSIM ``.item()`` + bare except: DEFERRED — the defect locus is
  ``models/losses/m4_losses.py``, not the strategy file under review.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from spectramr.infrastructure.training.strategies.pma_varnet_strategy import (
    ConcretePMAVarNetStrategy,
)


def test_validation_step_does_not_reseed_global_rng() -> None:
    """The medium finding: ``validation_step`` must not reseed global RNG.

    The fixed method is a thin ``no_grad`` forward; it must leave the global
    CPU RNG state untouched so it does not clobber the accelerator's seeding
    contract (performance.md). We invoke the method against an UNINITIALIZED
    real instance that stubs out ``_unpack_batch`` and ``_compute_losses_impl``,
    so the test exercises only the RNG behaviour of ``validation_step`` itself.

    A bare ``SimpleNamespace`` used to serve as that fake ``self``. It stopped
    working when #1190 wired the val path's ``model_output`` snapshot, because a
    duck-typed stub silently pins the method's ENTIRE ``self.`` surface: any new
    ``self.<helper>`` call breaks it, whether or not the behaviour under test
    changed. ``object.__new__`` keeps the real helpers (``snapshot_source``,
    ``_capture_model_output``) reachable while still constructing nothing --
    ``env.generator`` is ``None``, so the capture registers no hook, stashes
    nothing, and emits nothing.
    """
    sentinel_input = torch.tensor([1.0])
    sentinel_target = torch.tensor([2.0])
    captured: dict[str, object] = {}

    def fake_unpack(batch):  # noqa: ANN001 - tiny fake
        captured["unpacked"] = batch
        return sentinel_input, sentinel_target

    def fake_compute(input_batch, target_batch, epoch):  # noqa: ANN001
        captured["compute_args"] = (input_batch, target_batch, epoch)
        fake_self._last_prediction = torch.ones(1, 1, 8, 8)
        fake_self._last_target = torch.ones(1, 1, 8, 8)
        return {"g_total_loss": torch.tensor(0.0)}

    fake_self = object.__new__(ConcretePMAVarNetStrategy)
    fake_self._unpack_batch = fake_unpack
    fake_self._compute_losses_impl = fake_compute
    fake_self.env = SimpleNamespace(generator=None)
    fake_self.device = torch.device("cpu")
    fake_self.config = SimpleNamespace(
        validation=SimpleNamespace(
            scoring=SimpleNamespace(compute=["psnr"], primary="psnr", transform=None)
        ),
        metrics=None,
    )

    # Pin a known global RNG state, then snapshot it.
    torch.manual_seed(123456)
    before_state = torch.random.get_rng_state().clone()

    # Use a non-trivial batch_idx: the old code reseeded with 42 + batch_idx,
    # which would change the global state. The fixed code must not.
    result = ConcretePMAVarNetStrategy.validation_step(fake_self, object(), batch_idx=7)

    after_state = torch.random.get_rng_state()

    assert torch.equal(before_state, after_state), (
        "validation_step reseeded the global RNG; the ad-hoc "
        "torch.manual_seed(42 + batch_idx) must be removed."
    )
    # Sanity: it still wired the unpacked tensors through to the loss path.
    assert result == {"g_total_loss": torch.tensor(0.0)} or "g_total_loss" in result
    assert captured["compute_args"][0] is sentinel_input
    assert captured["compute_args"][1] is sentinel_target
    assert captured["compute_args"][2] == 0


def test_validation_step_source_has_no_manual_seed_call() -> None:
    """Belt-and-braces: the literal ``torch.manual_seed(...)`` *call* is gone.

    A behavioural test can pass if a reseed cancels out, so also assert the
    offending call is absent from the method source. The docstring is allowed
    to mention the removed pattern, so we strip the docstring before scanning
    and look for an actual call expression (``manual_seed(``), not the bare
    word.
    """
    method = ConcretePMAVarNetStrategy.validation_step
    src = inspect.getsource(method)
    docstring = method.__doc__ or ""
    body = src.replace(docstring, "")  # drop the docstring text
    assert "manual_seed(" not in body, (
        "validation_step still contains a torch.manual_seed(...) reseed call "
        "(audit medium finding)."
    )


def test_simulator_unpack_order_and_marker_mask_source() -> None:
    """The critical/high findings: simulator 3-tuple unpack must be correct.

    ``DigitalTwinSimulator.forward`` returns
    ``(corrupted_image, marker_prior, joint_clean)`` and the binary ROI mask is
    a separate ``simulator.marker_mask`` attribute. The old code unpacked the
    tuple as ``(corrupted, marker_mask, marker_prior)``, mislabeling the prior
    as the mask. Lock in the corrected unpack at source level so a future edit
    cannot silently re-introduce the swap.
    """
    src = inspect.getsource(ConcretePMAVarNetStrategy._compute_losses_impl)

    # The corrected unpack: second slot is marker_prior, third is discarded.
    assert "corrupted_complex, marker_prior, _joint_clean = self.simulator(" in src, (
        "the simulator 3-tuple unpack is not in the corrected "
        "(corrupted, marker_prior, joint) order."
    )
    # The binary mask must be fetched from the dedicated attribute, not the tuple.
    assert "marker_mask = self.simulator.marker_mask" in src, (
        "marker_mask must come from simulator.marker_mask, not the tuple unpack."
    )
    # The old swapped form must not be present.
    assert "corrupted_complex, marker_mask, marker_prior = self.simulator(" not in src
