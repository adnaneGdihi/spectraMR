"""Optimizer resolution SSOT — the structural peer of :mod:`scheduler_resolution`.

One place turns an ``OptimizationConfigSchema`` into a concrete optimizer, so
there is exactly one answer to "which knobs reach torch?".

There used to be two dispatch mechanisms and the live path used the wrong one.
:class:`~spectramr.infrastructure.training.optimizer_registry.OptimizerRegistry`
was built to be the SSOT — its own docstring says so — but nothing routed
through it; the leaf ``OptimizerBuilder`` did a reflective
``getattr(torch.optim, name)`` instead, and then hand-picked which kwargs to
forward:

    if canonical_name in ("Adam", "AdamW"):
        opt_kwargs["betas"] = self._betas

``beta1``/``beta2`` were set unconditionally by the caller, so for **nadam,
radam and adamax** — all of which accept ``betas`` — they were silently dropped.
``weight_decay`` went the other way, forwarded unconditionally, so ``lbfgs`` and
``sparseadam`` (which accept none) would ``TypeError``.

That hand-maintained tuple is the defect, not a detail: it was correct when
written and became wrong the moment three more optimizers were registered.

**Kwarg forwarding is therefore signature-driven**, and the repo already trusts
introspection on this exact object — ``_unconsumed_optimizer_kwargs`` validates
``optimizer_kwargs`` against ``inspect.signature(cls.__init__)``. Until now the
two disagreed: one validated against the signature, the other forwarded against
a tuple.

Two guards keep "signature-driven" from becoming a rubber stamp:

1. A ``**kwargs`` in the signature would accept everything, so an optimizer with
   ``VAR_KEYWORD`` must declare its accepted set explicitly on
   ``@register_optimizer(..., params=frozenset({...}))``.
2. Filtering decides only what to *forward*, never what to *ignore*. The split
   is on ``model_fields_set`` — what the YAML actually said:

   ===========================  =====================================
   knob origin                  unaccepted by the optimizer
   ===========================  =====================================
   explicitly declared          **raises**
   left at its schema default   dropped, recorded at ``info``
   ===========================  =====================================

   So ``optimizer_type: lbfgs`` quietly drops the default ``weight_decay`` it
   cannot take, while ``optimizer_type: lbfgs`` *with* ``weight_decay: 0.01``
   raises rather than pretending to honour it.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

from spectramr.infrastructure.training.optimizer_registry import (
    OptimizerRegistry,
    accepted_optimizer_kwargs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LookaheadSpec",
    "OptimizerSpec",
    "ParamGroupSpec",
    "build_optimizer_from_spec",
    "resolve_optimizer_spec",
]


class OptimizerConfigurationError(ValueError):
    """A declared optimizer knob cannot be honoured."""


#: Typed schema fields that map onto an optimizer constructor argument of the
#: same name. ``betas`` is resolved separately (it has two homes).
_SCALAR_KNOBS: tuple[str, ...] = (
    "weight_decay",
    "eps",
    "momentum",
    "nesterov",
    "amsgrad",
    # `fused` defaults to None for the same reason as nesterov/amsgrad: "unset"
    # must stay distinguishable from "explicitly false". Unset is dropped as a
    # schema default; `fused: true` on an optimizer that has no such argument
    # RAISES through _partition_kwargs rather than silently training un-fused.
    # That polarity matters more here than for a numerical knob: fused is a
    # THROUGHPUT claim, and a claim that quietly does not hold is the same
    # defect class as an eager run reporting compiled numbers.
    "fused",
)


@dataclass(frozen=True)
class ParamGroupSpec:
    """One resolved parameter group."""

    name: str
    lr: float
    weight_decay: float
    n_params: int

    def as_provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "n_params": self.n_params,
        }


@dataclass(frozen=True)
class LookaheadSpec:
    """Lookahead wrapper settings (applied after the base optimizer is built)."""

    sync_period: int
    alpha: float


@dataclass(frozen=True)
class OptimizerSpec:
    """Everything needed to construct one optimizer, and nothing else.

    Deliberately inert: resolving is pure and testable without torch modules,
    which is what lets the golden-value tests assert *which* knobs reach the
    optimizer rather than only that one was built.
    """

    name: str
    lr: float
    params: dict[str, Any] = field(default_factory=dict)
    param_groups: tuple[ParamGroupSpec, ...] = ()
    lookahead: LookaheadSpec | None = None
    role: str = "generator"
    dropped_defaults: tuple[str, ...] = ()
    #: Knobs the framework chose because the YAML left them unset (e.g. ``fused`` on CUDA).
    applied_defaults: tuple[str, ...] = ()

    def as_provenance(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "optimizer": self.name,
            "role": self.role,
            "lr": self.lr,
            **self.params,
        }
        if "betas" not in record:
            # Undeclared betas run at the constructor's default, which no record
            # named, so no run could say which beta1 it trained with (#2308).
            default = _constructor_default(self.name, "betas")
            if isinstance(default, tuple | list):
                record["betas"] = tuple(default)
                record["betas_source"] = "optimizer constructor default"
        if self.param_groups:
            record["param_groups"] = [g.as_provenance() for g in self.param_groups]
        if self.lookahead is not None:
            record["lookahead"] = {
                "sync_period": self.lookahead.sync_period,
                "alpha": self.lookahead.alpha,
            }
        if self.dropped_defaults:
            # Recorded, not hidden: "adafactor ignored your weight_decay default"
            # is exactly the kind of thing that is invisible at 3am otherwise.
            record["dropped_defaults"] = list(self.dropped_defaults)
        if self.applied_defaults:
            record["applied_defaults"] = list(self.applied_defaults)
        return record


def _optimizer_block(optimization: Any) -> Any:
    """The ``optimization.optimizer`` sub-block every knob below lives on.

    Phase 8 moved the optimizer knobs off the ``optimization`` block into their
    own. Resolved once here and threaded down rather than re-fetched, and it
    RAISES on a caller that hands over an object without one: a
    ``getattr(..., default)`` would turn the move into a silent reversion to
    schema defaults, which is pitfall #9 with the whole optimizer behind it.
    """
    optimizer = getattr(optimization, "optimizer", None)
    if optimizer is None:
        raise OptimizerConfigurationError(
            "resolve_optimizer_spec expects the `optimization` block, whose "
            "`optimizer` sub-block carries the knobs. Got "
            f"{type(optimization).__name__} with no `.optimizer`."
        )
    return optimizer


def _explicitly_set(optimizer: Any) -> set[str]:
    """Field names the YAML actually declared (vs. left at their default)."""
    fields_set = getattr(optimizer, "model_fields_set", None)
    return set(fields_set) if fields_set else set()


#: torch's Adam-family beta1, which an undeclared beta1 already runs. Named so a
#: half-declared pair (beta2 only) resolves to the same number rather than to a
#: schema default that no optimizer applies (#2308).
_TORCH_ADAM_BETA1 = 0.9


def _resolve_betas(optimizer: Any, declared: set[str]) -> tuple[float, float] | None:
    """Reconcile the two homes for the Adam beta pair.

    ``betas`` had a validator, was set by 27 experiment YAMLs, and was read by
    NOTHING — its docstring named ``OptimizerSetupMixin._build_optimizer``, a
    class that has never existed here.

    Returns ``None`` when neither home was declared, so the pair is treated as a
    droppable default rather than an explicit request.
    """
    pair = getattr(optimizer, "betas", None)
    has_pair = pair is not None
    # ``declared`` is empty for a non-Pydantic stand-in (tests, scripting
    # callers). Presence must still forward the pair — dropping betas for Adam
    # because we could not introspect the config object would recreate the very
    # bug this function exists to fix. Only the *raise* policy keys off
    # ``declared``; forwarding keys off presence.
    has_scalars = bool({"beta1", "beta2"} & declared) or (
        not declared
        and getattr(optimizer, "beta1", None) is not None
        and getattr(optimizer, "beta2", None) is not None
    )

    beta1 = getattr(optimizer, "beta1", None)
    scalars = (
        _TORCH_ADAM_BETA1 if beta1 is None else float(beta1),
        float(getattr(optimizer, "beta2", 0.999)),
    )
    if has_pair and has_scalars and {"betas"} & declared and tuple(pair) != scalars:
        raise OptimizerConfigurationError(
            f"optimization.optimizer.betas={tuple(pair)} disagrees with "
            f"optimization.optimizer.beta1/beta2={scalars}. Declare the beta pair in "
            "ONE place: either `betas: [b1, b2]` or the scalar beta1/beta2 "
            "fields."
        )
    if has_pair:
        return (float(pair[0]), float(pair[1]))
    if has_scalars:
        return scalars
    return None


def _candidate_kwargs(optimizer: Any, declared: set[str]) -> tuple[dict[str, Any], set[str]]:
    """Collect every knob that *might* reach the optimizer, with its origin.

    Returns ``(candidates, explicit_names)``.
    """
    candidates: dict[str, Any] = {}
    explicit: set[str] = set()

    for knob in _SCALAR_KNOBS:
        value = getattr(optimizer, knob, None)
        if value is None:
            # nesterov/amsgrad default to None precisely so "unset" is
            # distinguishable from "set to False".
            continue
        candidates[knob] = value
        if knob in declared:
            explicit.add(knob)

    betas = _resolve_betas(optimizer, declared)
    if betas is not None:
        candidates["betas"] = betas
        if {"betas", "beta1", "beta2"} & declared:
            explicit.add("betas")

    # optimizer.kwargs is always an explicit request — the user typed every key.
    extra = dict(getattr(optimizer, "kwargs", None) or {})
    for key, value in extra.items():
        # Conflict only with what the YAML actually SAID. Comparing against
        # `candidates` compared against untouched schema defaults too, so
        # optimizer_kwargs could only ever agree with a default, never override
        # one — which is what left `adafactor` with no way to declare the tuple
        # `eps` its constructor requires.
        # A YAML list and the resolved tuple are the same pair.
        comparable = tuple(value) if isinstance(value, list) else value
        if key in explicit and candidates[key] != comparable:
            raise OptimizerConfigurationError(
                f"optimization.optimizer.kwargs['{key}']={value!r} disagrees with "
                f"optimization.{key}={candidates[key]!r}. Declare it once."
            )
        candidates[key] = value
        explicit.add(key)

    return candidates, explicit


def _accepts_value_shape(name: str, key: str, value: Any) -> bool:
    """Whether ``name``'s constructor can take ``value`` for ``key``.

    Name-matching is one check short. ``torch.optim.Adafactor`` accepts ``eps``,
    but as a 2-tuple ``(float | None, float)`` rather than the scalar every
    other optimizer takes and the scalar ``optimization.optimizer.eps`` holds. The name
    matched, the value was forwarded, and a tier-1 "always constructible" name
    died with ``TypeError: 'float' object is not subscriptable``.

    The constructor's own default carries the shape, so this stays
    introspection-driven like the rest of the module rather than adding a
    hand-maintained per-optimizer table — the exact thing this module exists to
    delete.
    """
    default = _constructor_default(name, key)
    if default is inspect.Parameter.empty or default is None:
        # No shape to compare against (`foreach=None`, positional-only, ...).
        return True
    return isinstance(default, tuple | list) == isinstance(value, tuple | list)


def _constructor_default(name: str, key: str) -> Any:
    """``name``'s constructor default for ``key``, or ``Parameter.empty``."""
    cls = OptimizerRegistry.get(name)
    if cls is None:  # pragma: no cover - unknown names raise earlier
        return inspect.Parameter.empty
    try:
        return inspect.signature(cls.__init__).parameters[key].default
    except (KeyError, TypeError, ValueError):  # pragma: no cover - defensive
        return inspect.Parameter.empty


def _partition_kwargs(
    name: str,
    candidates: dict[str, Any],
    explicit: set[str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Split candidates into forwarded kwargs and dropped defaults."""
    accepted = accepted_optimizer_kwargs(name)

    forwarded: dict[str, Any] = {}
    dropped: list[str] = []
    rejected: list[str] = []
    misshaped: list[str] = []

    for key, value in candidates.items():
        if key not in accepted:
            (rejected if key in explicit else dropped).append(key)
        elif not _accepts_value_shape(name, key, value):
            # Accepted by NAME, incompatible by SHAPE. Same policy as an
            # unaccepted knob: a schema default it cannot take is dropped, an
            # explicit declaration raises.
            (misshaped if key in explicit else dropped).append(key)
        else:
            forwarded[key] = value

    if rejected:
        raise OptimizerConfigurationError(
            f"optimizer_type={name!r} does not accept {sorted(rejected)}, but the "
            f"config declares them explicitly. It accepts {sorted(accepted)}. "
            "A declared knob that cannot reach the optimizer is a silent no-op "
            "(pitfall #15) — remove it, or choose an optimizer that takes it."
        )
    if misshaped:
        shapes = ", ".join(
            f"{key}={candidates[key]!r} (it wants {_constructor_default(name, key)!r})"
            for key in sorted(misshaped)
        )
        raise OptimizerConfigurationError(
            f"optimizer_type={name!r} accepts {sorted(misshaped)} but not with "
            f"the declared shape: {shapes}. Declare it under "
            "optimization.optimizer.kwargs in the shape the optimizer wants."
        )
    return forwarded, tuple(sorted(dropped))


def _resolve_param_groups(
    optimizer: Any,
    model: Any,
    base_lr: float,
    base_weight_decay: float,
) -> tuple[ParamGroupSpec, ...]:
    """Turn ``optimizer.param_groups`` into concrete groups.

    Keys are ``named_parameters()`` prefixes. A key matching zero parameters
    RAISES: a renamed submodule must not silently degrade the arm to a uniform
    learning rate, which is exactly the failure this knob exists to prevent.
    """
    overrides = getattr(optimizer, "param_groups", None)
    if not overrides or model is None:
        return ()

    if hasattr(model, "get_differential_lr_param_groups") or hasattr(model, "get_parameter_groups"):
        raise OptimizerConfigurationError(
            "optimization.optimizer.param_groups is declared, but the model also "
            "exposes get_differential_lr_param_groups/get_parameter_groups. That "
            "is two sources of truth for one decision; silently preferring "
            "either is how `betas` died. Remove one."
        )

    named = list(model.named_parameters())
    if not named:
        raise OptimizerConfigurationError(
            "optimization.optimizer.param_groups is declared but the model exposes "
            "no named parameters."
        )

    assignment: dict[str, str] = {}
    for param_name, _ in named:
        for key in overrides:
            if param_name == key or param_name.startswith(f"{key}."):
                if param_name in assignment:
                    raise OptimizerConfigurationError(
                        f"parameter {param_name!r} matches two param_group_overrides "
                        f"keys ({assignment[param_name]!r} and {key!r}). Prefixes "
                        "must be unambiguous."
                    )
                assignment[param_name] = key

    groups: list[ParamGroupSpec] = []
    for key, override in overrides.items():
        members = [n for n, k in assignment.items() if k == key]
        if not members:
            available = sorted({n.split(".")[0] for n, _ in named})
            raise OptimizerConfigurationError(
                f"param_group_overrides key {key!r} matches no parameter. "
                f"Top-level module prefixes are {available}. A key that matches "
                "nothing would silently leave the arm on a uniform learning rate."
            )
        if override.freeze:
            # requires_grad_(False), not lr=0: a zero-LR group still accrues
            # optimizer state and is still decoupled-weight-decayed by AdamW.
            for param_name, param in named:
                if assignment.get(param_name) == key:
                    param.requires_grad_(False)
            continue
        lr = (
            override.learning_rate
            if override.learning_rate is not None
            else base_lr * (override.lr_multiplier or 1.0)
        )
        wd = override.weight_decay if override.weight_decay is not None else base_weight_decay
        groups.append(
            ParamGroupSpec(
                name=key,
                lr=float(lr),
                weight_decay=float(wd),
                n_params=len(members),
            )
        )

    default_members = [n for n, _ in named if n not in assignment]
    if default_members:
        groups.append(
            ParamGroupSpec(
                name="__default__",
                lr=base_lr,
                weight_decay=base_weight_decay,
                n_params=len(default_members),
            )
        )
    return tuple(groups)


#: Optimizers whose torch implementation has a fused CUDA kernel worth defaulting to.
_FUSED_BY_DEFAULT = frozenset({"adam", "adamw"})


def _fused_supported(model: Any) -> bool:
    """Whether every trainable parameter is a real floating tensor on a CUDA device.

    ``fused`` rejects complex, CPU and meta parameters, so the default applies only when it
    is certain to be accepted; an explicit ``fused: true`` elsewhere still raises as before.
    """
    if model is None or not hasattr(model, "parameters"):
        return False
    params = [p for p in model.parameters() if p.requires_grad]
    return bool(params) and all(
        p.is_cuda and p.is_floating_point() and not p.is_complex() for p in params
    )


def resolve_optimizer_spec(
    optimization: Any,
    *,
    role: str = "generator",
    lr_multiplier: float = 1.0,
    lr_override: float | None = None,
    model: Any = None,
) -> OptimizerSpec:
    """Resolve the config into an :class:`OptimizerSpec`. Pure; raises on conflict."""
    optimizer = _optimizer_block(optimization)
    name = str(optimizer.type).lower()
    if OptimizerRegistry.get(name) is None:
        raise OptimizerConfigurationError(
            f"Unknown optimizer {name!r}. Available: {OptimizerRegistry.list_available()}."
        )

    declared = _explicitly_set(optimizer)
    candidates, explicit = _candidate_kwargs(optimizer, declared)
    forwarded, dropped = _partition_kwargs(name, candidates, explicit)

    # Explicit per-optimizer LR (discriminator TTUR) wins over base * multiplier.
    lr = (
        float(lr_override)
        if lr_override is not None
        else float(optimizer.learning_rate) * float(lr_multiplier)
    )

    # param_groups is keyed by the generator's named_parameters(); handed the
    # discriminator it raised on an unmatched prefix, or silently gave D the
    # generator's per-group lr/weight decay/freeze on a matching one (#2308).
    groups = _resolve_param_groups(
        optimizer,
        model if role == "generator" else None,
        base_lr=lr,
        base_weight_decay=float(forwarded.get("weight_decay", 0.0)),
    )

    lookahead_cfg = getattr(optimizer, "lookahead", None)
    lookahead = (
        LookaheadSpec(sync_period=lookahead_cfg.sync_period, alpha=lookahead_cfg.alpha)
        if lookahead_cfg is not None and lookahead_cfg.enabled
        else None
    )

    if dropped:
        logger.info(
            "optimizer %s does not accept %s; dropped (left at schema defaults, "
            "not declared by this config).",
            name,
            list(dropped),
        )

    applied: tuple[str, ...] = ()
    if name in _FUSED_BY_DEFAULT and "fused" not in forwarded and _fused_supported(model):
        forwarded = {**forwarded, "fused": True}
        applied = ("fused",)

    return OptimizerSpec(
        name=name,
        lr=lr,
        params=forwarded,
        param_groups=groups,
        lookahead=lookahead,
        role=role,
        dropped_defaults=dropped,
        applied_defaults=applied,
    )


def _param_group_dicts(spec: OptimizerSpec, model: Any) -> list[dict[str, Any]]:
    """Materialise ``spec.param_groups`` against a live model."""
    named = list(model.named_parameters())
    buckets: dict[str, list[Any]] = {g.name: [] for g in spec.param_groups}
    lookup = {g.name: g for g in spec.param_groups}
    explicit_names = [g.name for g in spec.param_groups if g.name != "__default__"]

    for param_name, param in named:
        if not param.requires_grad:
            continue
        target = "__default__"
        for key in explicit_names:
            if param_name == key or param_name.startswith(f"{key}."):
                target = key
                break
        if target in buckets:
            buckets[target].append(param)

    return [
        {
            "params": members,
            "lr": lookup[name].lr,
            "weight_decay": lookup[name].weight_decay,
        }
        for name, members in buckets.items()
        if members
    ]


def build_optimizer_from_spec(spec: OptimizerSpec, model_or_params: Any) -> Any:
    """Construct the optimizer described by *spec*.

    The single dispatch point: every entry point routes here, so the answer to
    "which kwargs did this optimizer receive?" is one function, not three.
    """
    import torch.nn as nn

    params: Any
    if spec.param_groups and isinstance(model_or_params, nn.Module):
        params = _param_group_dicts(spec, model_or_params)
        kwargs = {k: v for k, v in spec.params.items() if k != "weight_decay"}
    elif isinstance(model_or_params, nn.Module):
        if hasattr(model_or_params, "get_differential_lr_param_groups"):
            params = model_or_params.get_differential_lr_param_groups(
                base_lr=spec.lr,
                weight_decay=spec.params.get("weight_decay", 0.0),
            )
        elif hasattr(model_or_params, "get_parameter_groups"):
            params = model_or_params.get_parameter_groups()
        else:
            params = model_or_params.parameters()
        kwargs = dict(spec.params)
    else:
        params = model_or_params
        kwargs = dict(spec.params)

    optimizer = OptimizerRegistry.create(spec.name, params, lr=spec.lr, **kwargs)

    if spec.lookahead is not None:
        from spectramr.infrastructure.training.optimizers.lookahead import Lookahead

        optimizer = Lookahead(
            optimizer,
            la_steps=spec.lookahead.sync_period,
            la_alpha=spec.lookahead.alpha,
        )
    return optimizer
