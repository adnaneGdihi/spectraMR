"""Unit tests for ``dtn2s_strategy`` -- the J-invariance mask must leave context.

Paired with ``src/spectramr/infrastructure/training/strategies/dtn2s_strategy.py``,
which had no paired test until #1028. That absence is part of the defect's
history: the mask blocked 100% of voxels at every window size and nothing at the
strategy level ever asserted otherwise.
"""

from __future__ import annotations

import types

import pytest
import torch


class TestMaskLeavesVisibleContext:
    """The masked input must retain context for the model to read (#1028).

    Before the fix ``build_dtn2s_mask`` blocked 100% of voxels at every window,
    so ``s_a_masked`` was all zeros and the arm trained a map from a constant.
    These assertions are on the STRATEGY, because the mask is voxel-indexed
    while ``s_a`` is position-indexed -- a correct mask applied in the wrong
    index space is just as degenerate, and only this level catches that.
    """

    @staticmethod
    def _strategy(recv: int):
        from spectramr.config.schemas.training.strategy_knobs_2026_08 import (
            DTN2STrainingConfigSchema,
        )
        from spectramr.infrastructure.training.strategies.dtn2s_strategy import (
            DTN2SStrategy,
        )

        strat = DTN2SStrategy.__new__(DTN2SStrategy)
        strat._phi_a = strat._phi_b = strat._mask = None
        cfg = types.SimpleNamespace(
            training=types.SimpleNamespace(
                dtn2s=DTN2STrainingConfigSchema(receptive_window=recv)
            )
        )
        strat.config = cfg
        return strat

    def test_masked_sequence_is_not_all_zeros(self) -> None:
        strat = self._strategy(recv=4)
        x = torch.randn(1, 1, 16, 16)
        mask = strat._resolve_mask(x)
        s_a = strat._phi_a.linearize(x).clone()
        s_a[..., mask] = 0.0
        assert s_a.abs().sum() > 0, "whole input blanked -- model trains on zeros"
        assert mask.sum() > 0, "nothing blocked at all -- J-invariance unenforced"

    def test_declared_receptive_window_is_honoured(self) -> None:
        """A YAML value must reach the mask; it used to be discarded (#376)."""
        x = torch.randn(1, 1, 16, 16)
        narrow = self._strategy(recv=1)._resolve_mask(x).sum().item()
        wide = self._strategy(recv=8)._resolve_mask(x).sum().item()
        assert wide > narrow, (
            f"receptive_window is inert: recv=1 blocked {narrow}, recv=8 blocked {wide}"
        )


class TestBackboneMustFitTheMask:
    """A backbone reading further than the window makes the mask meaningless, so it raises."""

    @staticmethod
    def _strategy(generator: torch.nn.Module, recv: int = 4):
        strat = TestMaskLeavesVisibleContext._strategy(recv)
        strat.env = types.SimpleNamespace(generator=generator)
        return strat

    def test_a_backbone_that_declares_no_receptive_field_is_refused(self) -> None:
        strat = self._strategy(torch.nn.Conv1d(1, 1, 3, padding=1))
        with pytest.raises(ValueError, match="receptive_half_width"):
            strat._require_j_invariant_backbone(strat.env.generator)

    def test_a_backbone_wider_than_the_window_is_refused(self) -> None:
        from spectramr.models.generators.windowed_sequence_denoiser import (
            WindowedSequenceDenoiser,
        )

        strat = self._strategy(WindowedSequenceDenoiser(receptive_half_width=6), recv=4)
        with pytest.raises(ValueError, match="only covers"):
            strat._require_j_invariant_backbone(strat.env.generator)

    def test_a_backbone_inside_the_window_passes(self) -> None:
        from spectramr.models.generators.windowed_sequence_denoiser import (
            WindowedSequenceDenoiser,
        )

        strat = self._strategy(WindowedSequenceDenoiser(receptive_half_width=4), recv=4)
        strat._require_j_invariant_backbone(strat.env.generator)

    def test_an_infer_run_is_blocked_with_a_reason(self) -> None:
        from spectramr.infrastructure.training.strategies.dtn2s_strategy import DTN2SStrategy

        assert "sequence" in (DTN2SStrategy.standalone_infer_blocker(object()) or "")


class TestJInvariance:
    """The prediction at a voxel does not depend on that voxel's own input value."""

    H = W = 16
    RECV = 4

    def _predict(self, strat, x: torch.Tensor) -> torch.Tensor:
        return strat._validation_forward(x, {})

    def _changed_by_own_input(self, strat, probes: int = 12) -> int:
        torch.manual_seed(0)
        x = torch.randn(1, 1, self.H, self.W)
        base = self._predict(strat, x)
        hits = 0
        for v in torch.randperm(self.H * self.W)[:probes].tolist():
            r, c = divmod(v, self.W)
            y = x.clone()
            y[0, 0, r, c] += 5.0
            if (self._predict(strat, y)[0, 0, r, c] - base[0, 0, r, c]).abs() > 1e-6:
                hits += 1
        return hits

    def test_the_windowed_backbone_never_reads_the_voxel_it_predicts(self) -> None:
        from spectramr.models.generators.windowed_sequence_denoiser import (
            WindowedSequenceDenoiser,
        )

        torch.manual_seed(1)
        strat = TestBackboneMustFitTheMask._strategy(
            WindowedSequenceDenoiser(receptive_half_width=self.RECV, hidden_channels=8), self.RECV
        )
        assert self._changed_by_own_input(strat) == 0

    def test_planted_violation_a_global_backbone_does_read_it(self) -> None:
        """The same probe turns red for a backbone that sees the whole sequence."""

        class Global(torch.nn.Module):
            receptive_half_width = 4  # a false claim, which is exactly what the probe must catch

            def forward(self, s: torch.Tensor) -> torch.Tensor:
                return s + s.mean(dim=-1, keepdim=True)

        strat = TestBackboneMustFitTheMask._strategy(Global(), self.RECV)
        assert self._changed_by_own_input(strat) > 0


def test_a_training_step_reaches_the_backbone_and_trains_it() -> None:
    """The production loss path: linearise, mask, run the backbone, score against phi_B."""
    from spectramr.models.generators.windowed_sequence_denoiser import WindowedSequenceDenoiser

    torch.manual_seed(0)
    backbone = WindowedSequenceDenoiser(receptive_half_width=4, hidden_channels=8)
    strat = TestBackboneMustFitTheMask._strategy(backbone, recv=4)
    x = torch.randn(2, 1, 16, 16)
    losses = strat._compute_losses_impl(x, x, epoch=0)
    assert torch.isfinite(losses["g_total_loss"])
    losses["g_total_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters())
