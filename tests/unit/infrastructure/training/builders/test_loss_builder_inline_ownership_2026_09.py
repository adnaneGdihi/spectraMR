"""``LossBuilder.validate()`` refuses an empty loss stack only for UNDECLARED strategies.

Targets ``spectramr.infrastructure.training.builders.loss_builder``.

An empty built stack has two meanings and the builder used to collapse them into
one raise. For an arm that expects the builder to feed it, nothing built is a
defect. For an arm whose strategy computes its objective inside
``_compute_losses_impl``, nothing built is the design -- and the unconditional
raise killed those arms at the director's step 3/6, before a single batch
(#1953, #1918 phase B). The strategy's own declaration is what tells them apart:
``inline_losses`` non-None plus ``folds_image_losses`` False.

Planted shapes (non-negotiable 15) -- the exemption is a detector, so it ships
with the arms that must turn it red as well as the one that must not:

===========================================  ============  =======
declaration                                  empty stack   verdict
===========================================  ============  =======
nothing declared (both ``None``)             yes           RAISE
``inline_losses`` set, ``folds`` True        yes           RAISE
``folds`` False, ``inline_losses`` ``None``  yes           RAISE
``inline_losses`` set, ``folds`` False       yes           PASS
strategy unresolvable                        yes           RAISE
===========================================  ============  =======

The third row is the half-declaration: both readers are tri-state and ``None``
is falsy, so a predicate written with ``not folds`` rather than ``is False``
passes this row and the plant is what catches that.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import pytest

from spectramr.config.schemas.loss import LossConfigSchema
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

_BUILDER_LOGGER = "spectramr.infrastructure.training.builders.loss_builder"


class _UndeclaredStrategy(BaseTrainingStrategy):
    """Declares neither. The base's own ``None`` must not read as a declaration."""


class _FoldingStrategy(BaseTrainingStrategy):
    """Declares inline ownership but still folds the builder's modules -- it needs them."""

    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = True


class _HalfDeclaredStrategy(BaseTrainingStrategy):
    """Says it folds nothing but never says what it computes. Not a claim of ownership."""

    folds_image_losses: ClassVar[bool] = False


class _InlineOwnerStrategy(BaseTrainingStrategy):
    """The exempt shape: owns its whole objective, consumes none of the builder's modules."""

    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False


def _path(cls: type) -> str:
    """Dotted path built from ``__name__`` so it resolves under either import mode."""
    return f"{__name__}.{cls.__name__}"


class _TrainingShim:
    """The two fields ``TrainingStrategyFactory.get_strategy_class`` reads."""

    def __init__(self, strategy_class: str | None) -> None:
        self.strategy_class = strategy_class
        self.training_mode = None


class _ConfigShim:
    """Minimal config surface ``validate()`` reads, mirroring ``test_loss_builder``'s shim."""

    def __init__(self, strategy_class: str | None) -> None:
        self.losses = LossConfigSchema()
        self.training = _TrainingShim(strategy_class)


def _builder_with_no_losses(strategy_class: str | None) -> LossBuilder:
    builder = LossBuilder(_ConfigShim(strategy_class), device="cpu")  # type: ignore[arg-type]
    assert builder._losses == {}, "the plant is only meaningful on an EMPTY stack"
    return builder


@pytest.mark.parametrize(
    "cls",
    [_UndeclaredStrategy, _FoldingStrategy, _HalfDeclaredStrategy],
    ids=["undeclared", "declares-but-folds", "half-declared"],
)
def test_empty_stack_still_raises_for_a_strategy_that_has_not_declared(cls: type) -> None:
    """Three shapes that must stay fatal. Only the full declaration is exempt."""
    with pytest.raises(ConfigurationError) as exc:
        _builder_with_no_losses(_path(cls)).validate()

    msg = str(exc.value)
    assert cls.__name__ in msg, f"the message must name the strategy it judged: {msg}"
    # It must name the real schema key and both remedies, not the free-text label
    # ``objectives`` the old message pointed at (pitfall #9).
    assert "'losses:'" in msg, msg
    assert "inline_losses" in msg, msg


def test_empty_stack_passes_for_a_strategy_that_declares_inline_ownership(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The exemption: declared ownership makes an empty stack the design, not a failure."""
    builder = _builder_with_no_losses(_path(_InlineOwnerStrategy))

    with caplog.at_level(logging.INFO, logger=_BUILDER_LOGGER):
        assert builder.validate() is builder, "validate() must stay chainable"

    # The pass is not silent: it says which declaration bought the exemption, so an
    # empty objective can never be read off the logs as an ordinary successful build.
    lines = [r.getMessage() for r in caplog.records]
    assert any(
        _InlineOwnerStrategy.__name__ in line and "empty stack" in line for line in lines
    ), lines


def test_unresolvable_strategy_on_an_empty_stack_still_raises_configuration_error() -> None:
    """Resolution happens only on the empty branch, and its failure keeps the diagnosis.

    ``get_strategy_class`` raises ``ConfigurationError`` and ``_load_strategy_class``
    raises ``ValueError``; both chain into one ``ConfigurationError`` so an arm with a
    bad dotted path and no losses does not change exception type.
    """
    with pytest.raises(ConfigurationError) as exc:
        _builder_with_no_losses(None).validate()

    assert "No losses were built" in str(exc.value)


def test_a_non_empty_stack_never_resolves_the_strategy() -> None:
    """A built stack is valid on its own -- an arm that declares losses must not start
    needing a resolvable strategy to pass validation."""
    builder = _builder_with_no_losses(None)
    builder._losses["l1"] = object()  # type: ignore[assignment]

    assert builder.validate() is builder
