"""Registry of **signal models** — the forward physics that has no adjoint.

The framework has two kinds of forward model, and conflating them is what kept
three regimes off LIVE:

* A :class:`~spectramr.infrastructure.physics.registry.BaseForwardOperator` is a
  **linear measurement operator** (``fft2d``, ``nufft``, ``phase_contrast``). Its
  contract *promises* ``adjoint()``, and generic callers — ``OperatorProjectionDataConsistency``,
  ``FISTAMBIRSolver``, ``NullSpaceProjection`` — consume that promise assuming
  ``<Ax, y> = <x, A'y>``.
* A **signal model** is the other kind: ``parameters -> signal``, nonlinear,
  **no adjoint**. Extended Tofts, ADC mono-exponential, SPGR. These are real,
  load-bearing forward physics, but they are not operators, and registering one
  in the ``OperatorRegistry`` with a fabricated ``adjoint`` would not crash — it
  would be silently consumed as if the inner-product identity held (pitfall #16).

So they get their own registry, whose contract has **no adjoint in it at all**.
Nothing here can be mistaken for a linear operator, because nothing here claims
to be invertible.

This is what the maturity ledger's LIVE rule means by "declares a real forward
model": a linear operator **or** a signal model. Before this existed the rule
asked only for an operator, so *every* MR regime declared ``fft2d`` (which tests
"is this MR?", not "is this regime's physics wired?") and ``mri_perfusion`` —
which has the richest regime-specific physics in the repo — could not reach LIVE
at all, because the honest answer to "what linear operator inverts a
tracer-kinetic map?" is "none".

Registration is by decorator over the **pure forward function**, which is
returned unchanged, so the physics stays directly importable and unit-testable:

    @register_signal_model(
        name="extended_tofts",
        regime=Regime.PERFUSION,
        parameters=("ktrans", "ve", "vp"),
        signal="concentration_time_curve",
        reference="Tofts et al., JMRI 10(3):223-232, 1999",
    )
    def extended_tofts_forward(t_s, aif, ktrans, ve, vp): ...
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

from spectramr.config.schemas.enums import Regime

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True, slots=True)
class SignalModelSpec:
    """The frozen facts about one registered signal model.

    ``parameters`` is load-bearing, not documentation: two signal models for the
    same regime are **not** interchangeable just because they share a regime.
    ``extended_tofts_forward(t_s, aif, ktrans, ve, vp)`` and
    ``gamma_variate(t_s, amplitude, t0_s, alpha, beta_s)`` are both PERFUSION
    physics with completely different signatures, so a consumer that dispatches
    on a config knob must check the parameter contract before calling ``fn``.
    """

    name: str
    fn: Callable[..., Any]
    regime: Regime
    #: The physical parameters this model maps FROM, in call order.
    parameters: tuple[str, ...]
    #: What it maps TO, e.g. "concentration_time_curve" — a free-form label.
    signal: str
    #: Literature citation. A signal model without one is a guess.
    reference: str


class SignalModelRegistry:
    """Name -> :class:`SignalModelSpec`. No adjoint anywhere in the contract."""

    _models: ClassVar[dict[str, SignalModelSpec]] = {}

    @classmethod
    def register(cls, spec: SignalModelSpec) -> None:
        """Register ``spec``, rejecting a duplicate name.

        A duplicate is always a bug: either two models genuinely collide (and the
        second would silently shadow the first, so a config naming that model
        would dispatch to whichever module imported last — see
        ``reference_import_time_table_vs_registry``), or the same module was
        imported twice under different paths, which is its own problem.
        """
        existing = cls._models.get(spec.name)
        if existing is not None and existing.fn is not spec.fn:
            raise ValueError(
                f"Signal model {spec.name!r} is already registered to "
                f"{existing.fn.__module__}.{existing.fn.__qualname__}; refusing "
                f"to shadow it with {spec.fn.__module__}.{spec.fn.__qualname__}. "
                "Dispatch would depend on import order."
            )
        cls._models[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> SignalModelSpec:
        """Return the spec for ``name``.

        Raises:
            KeyError: if unregistered. Never returns a default — a missing signal
                model means the physics is absent, and silently substituting one
                would refit to a different answer with a healthy residual
                (pitfall #9).
        """
        try:
            return cls._models[name]
        except KeyError:
            raise KeyError(
                f"Signal model {name!r} is not registered. Available: "
                f"{sorted(cls._models)}. A signal model is the forward physics "
                "itself — there is no sensible default to fall back to."
            ) from None

    @classmethod
    def list_models(cls) -> dict[str, SignalModelSpec]:
        """Every registered spec, keyed by name (a copy)."""
        return dict(cls._models)

    @classmethod
    def for_regime(cls, regime: Regime) -> dict[str, SignalModelSpec]:
        """Every registered spec tagged for ``regime``."""
        return {n: s for n, s in cls._models.items() if s.regime is regime}


def register_signal_model(
    *,
    name: str,
    regime: Regime,
    parameters: tuple[str, ...],
    signal: str,
    reference: str,
) -> Callable[[F], F]:
    """Register the decorated forward function as a signal model.

    The function is returned **unchanged** — this only records metadata, so the
    physics stays a plain importable function that tests can call directly
    without touching the registry.
    """

    def decorator(fn: F) -> F:
        SignalModelRegistry.register(
            SignalModelSpec(
                name=name,
                fn=fn,
                regime=regime,
                parameters=parameters,
                signal=signal,
                reference=reference,
            )
        )
        return fn

    return decorator


def get_signal_model(name: str) -> SignalModelSpec:
    """Resolve ``name`` to its spec, importing the package so decorators fire."""
    import spectramr.infrastructure.physics.signal_models  # noqa: F401

    return SignalModelRegistry.get(name)


def list_signal_models() -> dict[str, SignalModelSpec]:
    """Every registered signal model, importing the package so decorators fire.

    The import is what makes this safe to call from the maturity ledger: a bare
    read of the class attribute would be order-dependent, reporting "not
    registered" purely because nothing had imported the physics module yet.
    """
    import spectramr.infrastructure.physics.signal_models  # noqa: F401

    return SignalModelRegistry.list_models()


__all__ = [
    "SignalModelRegistry",
    "SignalModelSpec",
    "get_signal_model",
    "list_signal_models",
    "register_signal_model",
]
