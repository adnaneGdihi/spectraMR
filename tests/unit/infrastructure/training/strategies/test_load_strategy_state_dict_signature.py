"""Every strategy's ``load_strategy_state_dict`` takes resume's ``load_optimizer`` gate.

``resume`` calls ``strategy.load_strategy_state_dict(state, load_optimizer=...)`` (#2964),
so an override that drops the keyword raises ``TypeError`` at the first resume of
that strategy, on the cluster. This guard imports every strategy module and checks
the method each class resolves through its MRO, mixins included.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import spectramr.infrastructure.training.strategies as strategies_pkg
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy


def _import_every_strategy_module() -> None:
    # A module that fails to import would hide its classes, so it fails the guard.
    for info in pkgutil.walk_packages(strategies_pkg.__path__, f"{strategies_pkg.__name__}."):
        importlib.import_module(info.name)


def _subclasses(root: type) -> set[type]:
    found: set[type] = set()
    stack = [root]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in found:
                found.add(sub)
                stack.append(sub)
    return found


def _drops_load_optimizer(cls: type) -> bool:
    params = inspect.signature(cls.load_strategy_state_dict).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return False
    param = params.get("load_optimizer")
    return param is None or param.kind not in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def violations(classes: set[type]) -> list[str]:
    return sorted(cls.__qualname__ for cls in classes if _drops_load_optimizer(cls))


def _framework_strategies() -> set[type]:
    _import_every_strategy_module()
    return {c for c in _subclasses(BaseTrainingStrategy) if c.__module__.startswith("spectramr.")}


def test_every_strategy_accepts_load_optimizer() -> None:
    classes = _framework_strategies()
    names = {c.__name__ for c in classes}
    # Not vacuous: the overrides that motivated the guard are in the set.
    assert {"CalibratedQMapStrategy", "MultiTrainingStrategy"} <= names
    assert violations(classes) == []


def test_the_guard_turns_red_on_an_override_without_the_keyword() -> None:
    class DropsTheGate(BaseTrainingStrategy):
        def load_strategy_state_dict(self, state: dict[str, Any] | None) -> None:
            super().load_strategy_state_dict(state)

    assert violations({DropsTheGate}) == [DropsTheGate.__qualname__]


def test_the_guard_turns_red_on_a_mixin_override_too() -> None:
    class _Mixin:
        def load_strategy_state_dict(self, state: dict[str, Any] | None) -> None:
            return None

    class ThroughAMixin(_Mixin, BaseTrainingStrategy):
        pass

    assert violations({ThroughAMixin}) == [ThroughAMixin.__qualname__]


def test_the_guard_accepts_the_keyword_and_a_kwargs_catch_all() -> None:
    class Keyword(BaseTrainingStrategy):
        def load_strategy_state_dict(self, state, *, load_optimizer: bool = True) -> None:
            return None

    class CatchAll(BaseTrainingStrategy):
        def load_strategy_state_dict(self, state, **kwargs: Any) -> None:
            return None

    assert violations({Keyword, CatchAll}) == []
