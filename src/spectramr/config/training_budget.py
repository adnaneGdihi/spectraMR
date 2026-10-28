"""One owner for "does this arm declare a training budget" (cohort review 2026-09-02).

``ValidatorRegistry.epochs_valid`` (the train-time gate) and the
``training_budget_is_positive`` witness (the audit) each decided the rule on
their own, and only the witness knew that a calibration mode optimises nothing.
Both read this predicate now; a mode is added here, not in either caller.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["NO_TRAINING_MODES", "zero_budget_defect"]

#: Modes that optimise nothing (split-conformal, physics-residual conformal and
#: equivariance-conformal calibration, and the validation badge, all over a frozen
#: checkpoint: ``train_step`` returns no loss and ``run_calibration`` runs after
#: the loop): ``epochs: 0``
#: is their contract, not a missing budget. ``twin_dps`` is NOT here: its
#: strategy inherits the diffusion train step and its arm is
#: ``needs_implementation`` until a sampler loop exists (VF review 2026-09-03).
NO_TRAINING_MODES: frozenset[str] = frozenset(
    {"calibration", "phys_residual_conformal", "equivariance_conformal", "validation_badge"}
)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def zero_budget_defect(training: Mapping[str, object] | None) -> str | None:
    """The finding, or None: ``epochs`` at or below zero with no positive ``max_iterations``.

    Reads the raw ``training`` block (a mapping), so the registry's dict view and
    the witness's raw YAML see the same rule. A missing ``epochs`` is the schema
    default's business, not a zero budget.
    """
    training = training or {}
    # ``training_mode: calibration`` and ``strategy_class: calibration`` resolve to the
    # same class through the factory (a bare token, not a dotted path), so both
    # spellings name a no-training mode here (mrixfields b17, 2026-09-03).
    mode = str(training.get("training_mode") or "").lower()
    alias = str(training.get("strategy_class") or "").lower()
    if mode in NO_TRAINING_MODES or ("." not in alias and alias in NO_TRAINING_MODES):
        return None
    epochs = _number(training.get("epochs"))
    max_iter = _number(training.get("max_iterations"))
    if epochs is not None and epochs <= 0 and (max_iter is None or max_iter <= 0):
        return (
            f"training.epochs={int(epochs)} and no positive training.max_iterations: a "
            "budget of zero steps trains nothing and still writes a run directory."
        )
    return None
