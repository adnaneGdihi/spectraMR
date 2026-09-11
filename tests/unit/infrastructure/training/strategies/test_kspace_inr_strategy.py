"""Loss-ownership pin for ``kspace_inr_strategy`` (issue #1918)."""

from __future__ import annotations

from spectramr.infrastructure.training.strategies.kspace_inr_strategy import KSpaceINRStrategy


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: l1_loss(pred_k, kspace) is a k-space term, not the canonical image-space l1.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``ReconstructionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for
    ITSELF and becomes a lie the moment a subclass replaces
    ``_compute_losses_impl``. An inherited True makes the audit's
    ``image_losses_reach_the_objective`` witness PASS every declared
    ``losses.image_losses`` entry on this strategy's arms while the training
    step discards them.
    """
    assert KSpaceINRStrategy.__dict__["folds_image_losses"] is False
    assert KSpaceINRStrategy.__dict__["inline_losses"] == frozenset()
