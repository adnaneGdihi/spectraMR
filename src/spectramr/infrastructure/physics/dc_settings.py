"""One owner for "which ``physics.data_consistency`` key feeds which DC kwarg".

Three consumers used to answer that question, and only one of them was right:

* :mod:`spectramr.infrastructure.builders.generator_kwargs` forwards the SSOT
  block into the generator's ``**kwargs`` (the production path);
* :class:`~spectramr.models.generators.kspace_cold_diffusion_generator.KSpaceColdDiffusionGenerator`
  reads those kwargs back and dispatches to a concrete DC layer;
* the ``config_health_checker`` audit wants the same answer to decide whether a
  declared knob can possibly matter.

A second resolver does not announce itself -- both compute a plausible value and
the divergence surfaces as a wrong number, not an error (non-negotiable 17). So
the mapping and the "does this method read this knob" table live here, once.

Inert by method
---------------
``dc_weight`` is the term this module exists to make observable.
It is **not** a universal blend fraction: it is ``lambda_init`` (the trust
temperature) for :class:`SoftDataConsistency`, ``beta`` for
:class:`NoiseAdaptiveDataConsistency`, ``hf_lambda`` for
:class:`TargetAwareFSDC`, and ``weight`` for :class:`SimpleDataConsistency`.
:class:`HardDataConsistency` takes **no weight at all** -- its blend is
``(1 - m) * recon + m * obs``, which *is* weight 1.0 by construction, so there is
no coefficient for a declared value to occupy.

The same holds on the reverse path: ``p_sample`` and ``_apply_observed_dc``
(``models/diffusion/kspace_process.py:1508``, ``:1888``) both branch on
``dc_method == "hard"`` *before* reaching the ``self.dc_weight`` term, so a
declared weight is inert under ``hard`` at sampling as well as at training.

Declaring ``dc_weight: 0.5`` under ``dc_method: hard`` is therefore not a bug to
be honoured -- honouring it would silently convert hard DC into soft DC. It is a
knob that reads as a chosen experimental variable, is stamped into provenance,
and changes nothing (non-negotiable 8, pitfall #15/#16).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "DC_SSOT_KEYS",
    "SUPPORTED_NOISE_TYPES",
    "DCKnobReadership",
    "inert_dc_knobs",
    "schema_defaults",
    "resolve_effective_dc",
    "ssot_pairs",
]

#: ``(generator kwarg, physics.data_consistency field)`` pairs forwarded by
#: ``generator_kwargs`` step 3c. Adding a row here is what makes a schema field
#: reachable by the generator -- a field absent from this tuple is validated,
#: documented and never delivered.
#:
#: ``apply_at_predict`` is absent on purpose: it is read by the inference
#: strategies (``infrastructure/inference/predict_data_consistency.py``), not by
#: the generator's DC layer, so a row here would forward a kwarg no generator
#: accepts. ``tests/unit/infrastructure/physics/test_dc_settings.py`` pins that.
DC_SSOT_KEYS: tuple[tuple[str, str], ...] = (
    ("use_dc", "enabled"),
    ("dc_method", "method"),
    ("dc_weight", "weight"),
    ("train_noise_level", "train_noise_level"),
    ("eval_noise_level", "eval_noise_level"),
    ("noise_type", "noise_type"),
)

#: Noise models the DC layers actually implement.
#:
#: The schema's ``noise_type`` description used to also advertise ``'rician'``;
#: no layer implements it. Report the shipped vocabulary, not the designed one.
SUPPORTED_NOISE_TYPES: frozenset[str] = frozenset({"gaussian"})

#: Which ``dc_method`` values READ which forwarded kwarg.
#:
#: Derived from the dispatch in ``KSpaceColdDiffusionGenerator.__init__`` and the
#: constructor signatures in ``infrastructure/physics/data_consistency.py``.
#: ``use_dc`` and ``dc_method`` are read by every method and so are absent.
DCKnobReadership: dict[str, frozenset[str]] = {
    # Every branch that passes the value somewhere: lambda_init / beta /
    # hf_lambda / weight. 'hard' is deliberately NOT here.
    "dc_weight": frozenset(
        {"soft", "noise_adjusted", "noise_adaptive", "target_aware_fsdc", "wiener_gain"}
    ),
    # Only the layers whose constructors accept a noise level: HardDataConsistency
    # and the SimpleDataConsistency fallback. The learned/soft layers do not.
    "train_noise_level": frozenset({"hard"}),
    "eval_noise_level": frozenset({"hard"}),
    "noise_type": frozenset({"hard"}),
}

#: Methods handled by a branch that constructs ``SimpleDataConsistency``, which
#: accepts both a weight and the noise levels. Anything not named in the
#: generator's explicit branches lands here.
_EXPLICIT_METHODS = frozenset(
    {
        "soft",
        "noise_adjusted",
        "adaptive",
        "kan_adaptive",
        "noise_adaptive",
        "hard",
        "target_aware_fsdc",
        "wiener_gain",
    }
)

_DISABLE_SENTINELS = frozenset({"", "none", "off", "disabled"})


@dataclass(frozen=True)
class ResolvedDC:
    """The DC settings the generator will actually be constructed with.

    Attributes:
        enabled: Whether a DC layer is built at all.
        method: The resolved ``dc_method``, or ``None`` when disabled.
        values: Resolved value per generator kwarg name.
        source: ``"physics"`` when the SSOT block supplied the values,
            ``"model_kwargs"`` when it was absent and the model block did.
    """

    enabled: bool
    method: str | None
    values: dict[str, Any]
    source: str


def ssot_pairs(dc: Any) -> tuple[tuple[str, Any], ...]:
    """``(kwarg_name, value)`` for every key ``physics.data_consistency`` owns.

    Args:
        dc: A ``DataConsistencyConfig`` (or anything exposing its fields).

    Returns:
        One pair per row of :data:`DC_SSOT_KEYS` the object actually carries.
    """
    pairs = []
    for kwarg, field in DC_SSOT_KEYS:
        if hasattr(dc, field):
            pairs.append((kwarg, getattr(dc, field)))
    return tuple(pairs)


def reads_knob(method: str | None, knob: str) -> bool:
    """Whether ``method``'s DC layer reads ``knob``.

    Args:
        method: Resolved ``dc_method``, or ``None``/a disable sentinel.
        knob: A generator kwarg name from :data:`DC_SSOT_KEYS`.

    Returns:
        False when DC is disabled. True when the method is not one of the
        generator's explicit branches, because the ``SimpleDataConsistency``
        fallback accepts every knob here.
    """
    if method is None or str(method).lower() in _DISABLE_SENTINELS:
        return False
    if knob not in DCKnobReadership:
        return True
    if method not in _EXPLICIT_METHODS:
        return True  # SimpleDataConsistency fallback accepts all of them
    return method in DCKnobReadership[knob]


def resolve_effective_dc(config: Any) -> ResolvedDC:
    """Resolve the DC settings the generator will see, with SSOT precedence.

    Mirrors ``generator_kwargs`` step 3c: ``physics.data_consistency`` wins
    whenever it is present (that step raises on a ``model_kwargs`` conflict
    rather than merging), and ``model.model_kwargs`` supplies the values only
    when the physics block is absent entirely.

    Args:
        config: A ``TrainingSettings``-shaped object.

    Returns:
        The resolved settings. ``enabled`` is False when no DC layer is built.
    """
    model = getattr(config, "model", None)
    model_kwargs = dict(getattr(model, "model_kwargs", {}) or {}) if model else {}

    physics = getattr(config, "physics", None)
    dc = getattr(physics, "data_consistency", None) if physics is not None else None

    if dc is not None:
        values = dict(ssot_pairs(dc))
        source = "physics"
    else:
        values = {k: model_kwargs[k] for k, _ in DC_SSOT_KEYS if k in model_kwargs}
        source = "model_kwargs"

    enabled = bool(values.get("use_dc", model_kwargs.get("use_dc", True)))
    method = values.get("dc_method", model_kwargs.get("dc_method", "hard"))
    if method is not None and str(method).lower() in _DISABLE_SENTINELS:
        enabled = False
        method = None

    return ResolvedDC(enabled=enabled, method=method, values=values, source=source)


def schema_defaults() -> dict[str, Any]:
    """Generator kwarg -> ``DataConsistencyConfig`` default, for every SSOT row.

    Read off the schema rather than restated: a copy here kept the noise levels
    at 0.01 / 0.005 after the schema moved them to 0.0, so every non-hard arm was
    reported for noise knobs it never declared.
    """
    from spectramr.config.schemas.physics import DataConsistencyConfig

    fields = DataConsistencyConfig.model_fields
    return {kwarg: fields[field].default for kwarg, field in DC_SSOT_KEYS}


def inert_dc_knobs(config: Any) -> tuple[tuple[str, Any, str], ...]:
    """DC knobs this arm declares that its resolved ``dc_method`` cannot read.

    Only knobs whose declared value **differs from the schema default** are
    reported: emitting every defaulted field would make the finding a tautology
    (a ``model_dump`` presence test is always true), and a defaulted value is not
    something the author chose.

    Args:
        config: A ``TrainingSettings``-shaped object.

    Returns:
        ``(kwarg_name, declared_value, reason)`` triples, empty when clean.
    """
    resolved = resolve_effective_dc(config)
    if not resolved.enabled or resolved.method is None:
        return ()

    defaults = schema_defaults()
    findings = []
    for knob in DCKnobReadership:
        if knob not in resolved.values:
            continue
        value = resolved.values[knob]
        if value is None or value == defaults.get(knob):
            continue
        if reads_knob(resolved.method, knob):
            continue
        findings.append(
            (
                knob,
                value,
                f"dc_method={resolved.method!r} builds a layer that takes no {knob!r} parameter",
            )
        )
    return tuple(findings)
