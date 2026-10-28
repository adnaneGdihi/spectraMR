"""Unit tests for the workflow maturity ledger's registry-walk.

The ledger is the anti-facade spine: it asserts each ``WorkflowProfile``'s
declared maturity against the live registries. That only works if the walk
itself is honest, which it was not before 2026-07-16:

  - ``strategies_tagged`` read ``capabilities`` via ``getattr``, which walks the
    MRO — so 64 strategies reported ``workflows={mri_structural}`` while exactly
    one declared it (the leak tests below).
  - ``_iter_strategy_classes`` swallowed every import exception, so a broken
    strategy silently vanished from the walk instead of raising (pitfall #9).

These tests pin both. They are unit-level and touch no GPU.
"""

from __future__ import annotations

import logging

import pytest

from spectramr.config.schemas.enums import Regime
from spectramr.infrastructure.validation.workflow_ledger import (
    _declared_workflows,
    _iter_strategy_classes,
    losses_tagged,
    metrics_tagged,
    operator_registered,
    strategies_tagged,
)

# ---------------------------------------------------------------------------
# The MRO tag leak: a tag must be DECLARED, never inherited.
# ---------------------------------------------------------------------------


def test_declared_workflows_reads_the_class_it_is_declared_on() -> None:
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    assert _declared_workflows(ReconstructionTrainingStrategy) == frozenset(
        {Regime.STRUCTURAL}
    )


def test_declared_workflows_ignores_inherited_tags() -> None:
    """A subclass that declares nothing is UNTAGGED, not its parent's regime.

    ``capabilities`` is a ClassVar, so ``getattr`` would report this subclass as
    ``mri_structural``. Inheriting a parent's regime is not opting into it — and
    the inherited claim was outright false for the DWI/dynamic/quantitative
    strategies that happen to subclass the structural one.
    """
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    class _UntaggedSubclass(ReconstructionTrainingStrategy):
        pass

    assert _declared_workflows(_UntaggedSubclass) is None


def test_declared_workflows_honours_an_explicit_override() -> None:
    """Overriding the ClassVar IS opting in — that must still count."""
    from typing import ClassVar

    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )
    from spectramr.models.capabilities import StrategyCapabilities

    class _RetaggedSubclass(ReconstructionTrainingStrategy):
        capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
            workflows=frozenset({Regime.FLOW}),
        )

    assert _declared_workflows(_RetaggedSubclass) == frozenset({Regime.FLOW})


def test_structural_is_declared_by_exactly_one_strategy() -> None:
    """Pins the leak fix quantitatively.

    Before the fix this count was 64 (one real declarer + 63 subclasses that
    inherited the tag). A count > 1 means either a genuine second structural
    strategy landed — update this test deliberately — or the leak is back.
    """
    declarers = [
        cls.__name__
        for cls in _iter_strategy_classes()
        if Regime.STRUCTURAL in (_declared_workflows(cls) or ())
    ]
    # The leak property is the point, so assert it directly rather than via a
    # single-name allow-list: every structural claim must come from the class's
    # OWN body. This is what would catch the 64-subclass regression; a hardcoded
    # name list only catches it by accident.
    inherited = [
        cls.__name__
        for cls in _iter_strategy_classes()
        if Regime.STRUCTURAL in (_declared_workflows(cls) or ())
        and cls.__dict__.get("capabilities") is None
    ]
    assert not inherited, f"structural claimed by INHERITANCE, the leak is back: {inherited}"

    # Pinned set, so a new declarer stays a deliberate decision. PILOT and BALD
    # are learnable-acquisition arms over structural MRI; both carry their own
    # `capabilities` (verified above, not inherited from Reconstruction), which
    # is exactly the explicit opt-in the ledger asks for. Noise2Noise denoises
    # structural repetitions and declares the same on its own body.
    assert sorted(declarers) == sorted(
        [
            "ReconstructionTrainingStrategy",
            "NoiseToNoiseStrategy",
            "PILOTStrategy",
            "BALDAcquisitionStrategy",
        ]
    ), f"structural declarer set changed, update deliberately: {declarers}"


def test_structural_stays_tagged_after_the_leak_fix() -> None:
    """The leak fix must not silently downgrade the one real LIVE claim."""
    assert strategies_tagged(Regime.STRUCTURAL)


# ---------------------------------------------------------------------------
# Import failures are loud (pitfall #9).
# ---------------------------------------------------------------------------


def test_broken_spectramr_strategy_import_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spectramr strategy that will not import must RAISE, not be skipped.

    Skipping it drops the strategy from the walk, so a regime it backs reports
    "no strategy is tagged" — an assertion failure that hides the ImportError.
    """
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    monkeypatch.setattr(
        TrainingStrategyFactory,
        "STRATEGY_CLASS_PATHS",
        {"broken": "spectramr.infrastructure.training.strategies.no_such_mod.Cls"},
    )
    with pytest.raises(ImportError, match="does not import"):
        list(_iter_strategy_classes())


def test_missing_class_in_real_module_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry naming a class that is not there is a registry bug."""
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    monkeypatch.setattr(
        TrainingStrategyFactory,
        "STRATEGY_CLASS_PATHS",
        {"ghost": "spectramr.infrastructure.training.strategies.base.NoSuchClass"},
    )
    with pytest.raises(AttributeError, match="does not exist in"):
        list(_iter_strategy_classes())


def test_absent_third_party_dependency_is_skipped_loudly(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A genuinely-absent optional third-party dep skips — at WARNING.

    DEBUG was the bug: the skip was invisible, so a regime backed only by that
    strategy silently reported as unbacked.
    """
    import importlib

    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    monkeypatch.setattr(
        TrainingStrategyFactory,
        "STRATEGY_CLASS_PATHS",
        {"optional": "spectramr.infrastructure.training.strategies.base.Whatever"},
    )

    def _raise_missing_third_party(name: str) -> object:
        raise ModuleNotFoundError("No module named 'exotic_pkg'", name="exotic_pkg")

    monkeypatch.setattr(importlib, "import_module", _raise_missing_third_party)

    with caplog.at_level(logging.WARNING):
        assert list(_iter_strategy_classes()) == []
    assert "exotic_pkg" in caplog.text
    assert "UNVERIFIED" in caplog.text


def test_walk_dedupes_aliased_dotted_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """~25 short names alias the same class; the walk must yield it once."""
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    path = (
        "spectramr.infrastructure.training.strategies.reconstruction."
        "ReconstructionTrainingStrategy"
    )
    monkeypatch.setattr(
        TrainingStrategyFactory,
        "STRATEGY_CLASS_PATHS",
        {"recon": path, "reconstruction": path, "recon_alias": path},
    )
    assert len(list(_iter_strategy_classes())) == 1


# ---------------------------------------------------------------------------
# The other three helpers.
# ---------------------------------------------------------------------------


def test_operator_registered_resolves_real_keys() -> None:
    assert operator_registered("fft2d")
    assert operator_registered("phase_contrast")


def test_operator_registered_is_false_for_none_and_unknown() -> None:
    """``None`` means 'no operator declared' — that is not 'registered'.

    Callers (the LIVE rule) must distinguish the two explicitly.
    """
    assert not operator_registered(None)
    assert not operator_registered("no_such_operator")


def test_losses_and_metrics_tagged_find_the_real_flow_tags() -> None:
    assert losses_tagged(Regime.FLOW)
    assert metrics_tagged(Regime.FLOW)


def test_losses_tagged_is_false_for_an_untagged_regime() -> None:
    """CT is a STUB — nothing is tagged for it, and nothing may be."""
    assert not losses_tagged(Regime.CT)
    assert not metrics_tagged(Regime.CT)
    assert not strategies_tagged(Regime.CT)
