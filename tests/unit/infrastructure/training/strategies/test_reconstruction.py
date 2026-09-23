"""Tests for ReconstructionTrainingStrategy's loss-folding seam.

Scoped deliberately: the strategy's forward/backward path is exercised end-to-end by
the Tier-2 audit probe and by the per-paradigm strategy tests. What is pinned here is
the ``inline_losses`` declaration — the Open-Closed hook a subclass uses to declare
which registered losses it computes inline, so that declaring those losses on
``losses.image_losses`` (the only surface ``LossScheduleController`` can resolve a
curriculum rule's base weight from) does not silently double-count them.
"""

from __future__ import annotations

from spectramr.infrastructure.training.strategies.loss_folding import (
    _INLINE_MANAGED,
    declared_inline_losses,
    inline_managed_with,
)
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


def test_base_strategy_owns_nothing_inline_and_folds_everything() -> None:
    """The base recon strategy computes every declared entry through the builder."""
    assert ReconstructionTrainingStrategy.inline_losses == frozenset()
    assert ReconstructionTrainingStrategy.folds_image_losses is True


def test_an_undeclared_subclass_keeps_the_universal_skip_set() -> None:
    """The base's empty declaration must not reach a subclass that has not declared:
    that subclass's inline l1 would be folded a second time (the fold passes
    ``below=ReconstructionTrainingStrategy``)."""

    class _Undeclared(ReconstructionTrainingStrategy):
        pass

    assert declared_inline_losses(_Undeclared, below=ReconstructionTrainingStrategy) is None
    assert inline_managed_with() == _INLINE_MANAGED == frozenset({"l1", "l2"})
    assert declared_inline_losses(_Undeclared) == frozenset()  # the witness's view


def test_a_declared_subclass_replaces_the_skip_set_with_its_own_terms() -> None:
    """Bloch synthesis computes l1 and segmentation_dice inline and declares exactly those."""
    from spectramr.infrastructure.training.strategies.bloch_synth_strategy import (
        BlochSynthesisStrategy,
    )

    assert issubclass(BlochSynthesisStrategy, ReconstructionTrainingStrategy)
    declared = declared_inline_losses(BlochSynthesisStrategy, below=ReconstructionTrainingStrategy)
    assert declared == frozenset({"l1", "segmentation_dice"})
    assert BlochSynthesisStrategy.folds_image_losses is True


def test_recoverability_vib_owns_its_l1_and_folds_nothing() -> None:
    """A subclass whose ``_compute_losses_impl`` neither calls the parent nor the fold."""
    from spectramr.infrastructure.training.strategies.recoverability_vib_strategy import (
        RecoverabilityVIBStrategy,
    )

    assert RecoverabilityVIBStrategy.inline_losses == frozenset({"l1"})
    assert RecoverabilityVIBStrategy.folds_image_losses is False


def test_the_inline_l1_weight_comes_from_the_loss_weight_table() -> None:
    """One owner (mrixfields review 2026-09-03): ``losses.image_losses[l1].weight`` through
    the table; an arm without an l1 entry cannot weight a term it never declared."""
    from types import SimpleNamespace

    import pytest

    s = ReconstructionTrainingStrategy.__new__(ReconstructionTrainingStrategy)
    s.config = SimpleNamespace(
        losses=SimpleNamespace(
            image_losses=[{"name": "l1", "weight": 0.25}], kspace_losses=[], complex_losses=[]
        )
    )
    assert s._declared_inline_l1_weight() == 0.25
    s.config = SimpleNamespace(
        losses=SimpleNamespace(
            image_losses=[{"name": "hfen", "weight": 1.0}], kspace_losses=[], complex_losses=[]
        )
    )
    with pytest.raises(ValueError, match="losses.image_losses"):
        s._declared_inline_l1_weight()


def test_multislice_flag_is_read_without_a_hasattr_fallback() -> None:
    """`multislice_enabled` was undeclared, so the `hasattr` guard was always
    False and `is_multislice` could never be True — the flag both docstrings
    advertise as replacing shape heuristics never fired (pitfall #16)."""
    import inspect

    from spectramr.config.schemas.data import DataConfigSchema
    from spectramr.infrastructure.training.strategies import reconstruction

    assert DataConfigSchema().multislice_enabled is False
    assert DataConfigSchema(multislice_enabled=True).multislice_enabled is True
    src = inspect.getsource(reconstruction)
    assert 'hasattr(config.data, "multislice_enabled")' not in src
    assert "multislice_enabled = config.data.multislice_enabled" in src


class TestCoilMapsReachTheLosses:
    """The maps reached the MODEL's forward and stopped there.

    ``_prepare_generator_inputs`` passes ``batch_context['coil_sensitivities']``
    to the model as ``sensitivity_maps``, but the declarative loss terms are
    invoked through ``UnifiedReconstructionLossComputer.compute``, whose
    ``_call_safe_loss`` filters kwargs by signature -- so a term needing the
    coil geometry was called as ``(pred, target)`` and raised. The maps are a
    loss kwarg too now.
    """

    def test_the_strategy_threads_the_maps_into_the_loss_computer(self) -> None:
        """Source-level, because constructing the strategy needs a built model."""
        import inspect

        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        source = inspect.getsource(ReconstructionTrainingStrategy._compute_losses_impl)
        assert 'batch_context["coil_sensitivities"]' in source, (
            "the strategy no longer threads the coil maps into the loss computer, so "
            "a term like coil_subspace_residual is invoked as (pred, target) and raises"
        )
        assert "**loss_context" in source, "the threaded kwargs no longer reach compute()"

    def test_it_is_conditional_so_armless_runs_are_untouched(self) -> None:
        """An arm with no maps must not gain a `coil_sensitivities=None` kwarg."""
        import inspect

        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        source = inspect.getsource(ReconstructionTrainingStrategy._compute_losses_impl)
        assert 'if batch_context.get("coil_sensitivities") is not None:' in source
