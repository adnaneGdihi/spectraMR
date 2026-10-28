"""Optimizer classes and the step policies that drive them.

Two halves, deliberately in one package:

* **Optimizer classes** — ``lars``, ``lamb``, ``lion``, ``lookahead``, and the
  optional-extra wrappers. Canonical home, beside the registry that dispatches
  them (``../optimizer_registry.py``).
* **Step policies / steppers** — ``amp_policy``, ``standard_optimizer_stepper``.

The submodule imports below are **load-bearing, not tidiness**: they are what
runs the ``@register_optimizer`` decorators. A decorator in a module nothing
imports is dead, and the failure is silent — the name simply never appears in
``list_available()``, so a YAML naming it gets "unknown optimizer" and reads as a
typo rather than a missing import. ``optimizer_registry`` asserts at import that
every ``OptimizerType`` member is registered, which turns that silence into a
loud ImportError.
"""

from . import (  # noqa: F401  (imported for @register_optimizer side effects)
    bnb_optimizers,
    lamb,
    lars,
    lion,
    lookahead,
    schedulefree_optimizers,
)
from .amp_policy import AMPPolicy
from .lamb import LAMB
from .lars import LARS
from .lion import Lion
from .lookahead import Lookahead
from .schedulefree_optimizers import supports_schedule_free_modes
from .standard_optimizer_stepper import StandardOptimizerStepper

__all__ = [
    "LAMB",
    "LARS",
    "AMPPolicy",
    "Lion",
    "Lookahead",
    "StandardOptimizerStepper",
    "supports_schedule_free_modes",
]
