"""LR-scheduler spec resolution — the single place a scheduler config is read.

Why this module exists (issue #533): ``OptimizationBuilder.build_schedulers``
used to read the scheduler config as ``scheduler["type"]`` and
``scheduler["kwargs"]``. Every experiment YAML in the corpus declares the block
*flat*::

    optimization:
      lr_scheduler_strategy: cosine
      scheduler:
        warmup_steps: 3500
        T_0: 50000
        T_mult: 2
        eta_min: 1.0e-06
        warmup_start_lr: 1.0e-06

226/226 ``inprogress`` configs declare neither ``type`` nor ``kwargs``, so every
parameter above was discarded and the built object was
``CosineAnnealingLR(T_max=100, eta_min=1e-6)`` — library defaults. Stepped once
per optimizer step that makes the LR sweep base<->eta_min with a
2*T_max*accum-iteration period for the whole run instead of annealing, and no
warmup ever ran.

Design (pitfall #15: read + validate + stamp, in the same change):

* the **flat** form is canonical (it is what the corpus uses); the legacy
  ``type``/``kwargs`` form still resolves,
* ``lr_scheduler_strategy`` is now *read* (146 configs declare the real type
  there and nothing consumed it),
* ``T_0``/``T_mult`` promote a ``cosine`` request to warm restarts, because that
  is the only factory that consumes them,
* ``T_max`` falls back to the run length in scheduler steps, never to 100,
* any key the resolved factory cannot consume **raises** — a typo or a
  wrong-family knob can no longer be silently dropped,
* the resolved spec is returned as a plain dataclass so callers can stamp it
  into provenance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.scheduler_system import (
    SCHEDULER_REGISTRY,
    WarmupScheduler,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch

logger = logging.getLogger(__name__)

__all__ = [
    "SCHEDULER_PARAM_KEYS",
    "SchedulerSpec",
    "build_scheduler_from_spec",
    "resolve_scheduler_spec",
    "resolved_schedule_record",
]


#: Parameters each registered factory actually forwards to torch. A key outside
#: the union of these (plus the warmup keys) is unroutable and raises. Keep in
#: lockstep with ``scheduler_system``'s factories — the test suite asserts every
#: registry name appears here.
SCHEDULER_PARAM_KEYS: dict[str, frozenset[str]] = {
    "constant": frozenset({"factor", "total_iters"}),
    "cosine_annealing": frozenset({"T_max", "eta_min"}),
    "cosine_annealing_warm_restarts": frozenset({"T_0", "T_mult", "eta_min"}),
    "step_lr": frozenset({"step_size", "gamma"}),
    "exponential": frozenset({"gamma"}),
    "linear": frozenset({"start_factor", "end_factor", "total_iters"}),
}

#: Names this resolver used to accept and now rejects with a reason. The
#: plateau scheduler was built but never stepped with a validation metric, so
#: every arm naming it trained at a constant LR (#2301).
_REMOVED_NAMES: dict[str, str] = dict.fromkeys(
    ("plateau", "reduce_on_plateau", "reduce_lr_on_plateau"),
    "ReduceLROnPlateau was removed: nothing steps it with a validation metric, "
    "so it trained at a constant LR (#2301). Declare a clock-driven family "
    "(cosine, step, linear_decay) instead.",
)

#: Names accepted in ``lr_scheduler_strategy`` / ``scheduler.type`` that are not
#: themselves registry keys. A warmup-bearing name resolves to its *decay* family
#: only -- the warmup half is supplied by the wrapper that ``warmup_steps``
#: selects -- so those names are listed in :data:`_WARMUP_BEARING_NAMES` and
#: checked, never silently stripped.
_NAME_ALIASES: dict[str, str] = {
    "cosine": "cosine_annealing",
    "warmup_cosine": "cosine_annealing",
    "cosine_warm_restarts": "cosine_annealing_warm_restarts",
    "step": "step_lr",
    "linear_decay": "linear",
    # Warmup with no decay family: the wrapper alone, over a flat base LR.
    "warmup": "constant",
    "linear_warmup": "constant",
}

#: Declared names whose distinguishing half is the warmup, not the decay family.
#: ``_NAME_ALIASES`` maps them onto a decay key, which drops the warmup unless
#: ``warmup_steps`` is declared somewhere -- so 15 arms asked for
#: ``warmup_cosine`` and silently got plain cosine annealing. Resolving one of
#: these with no warmup length is a contradiction, and contradictions raise
#: (non-negotiable #3: no silent fallbacks).
_WARMUP_BEARING_NAMES: frozenset[str] = frozenset({"warmup_cosine", "warmup", "linear_warmup"})

#: Historical spellings of the warmup knobs found in the corpus (17 configs).
_WARMUP_ALIASES: dict[str, str] = {
    "warm_up_epochs": "warmup_steps",
    "warm_up_start_lr": "warmup_start_lr",
    "warm_up_end_lr": "warmup_end_lr",
}

_WARMUP_KEYS: frozenset[str] = frozenset({"warmup_steps", "warmup_start_lr", "warmup_end_lr"})

#: Factories whose period must span the run when the config does not say so.
_RUN_LENGTH_KEY: dict[str, str] = {
    "cosine_annealing": "T_max",
    "linear": "total_iters",
}


@dataclass(frozen=True)
class SchedulerSpec:
    """The fully-resolved scheduler request. Safe to stamp into provenance."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    warmup_steps: int = 0
    warmup_start_lr: float | None = None
    warmup_end_lr: float | None = None

    def as_provenance(self) -> dict[str, Any]:
        """Flat, JSON-serialisable record of what was actually built."""
        rec: dict[str, Any] = {"scheduler": self.name, **self.params}
        if self.warmup_steps:
            rec["warmup_steps"] = self.warmup_steps
            rec["warmup_start_lr"] = self.warmup_start_lr
            if self.warmup_end_lr is not None:
                rec["warmup_end_lr"] = self.warmup_end_lr
        return rec


def _canonical_name(declared: str) -> str:
    """Map a declared scheduler name onto a registry key, or raise."""
    if declared.lower() in _REMOVED_NAMES:
        raise ConfigurationError(f"Scheduler {declared!r}: {_REMOVED_NAMES[declared.lower()]}")
    key = _NAME_ALIASES.get(declared.lower(), declared.lower())
    if key not in SCHEDULER_REGISTRY:
        raise ConfigurationError(
            f"Unknown scheduler {declared!r}. Valid names: "
            f"{sorted(set(SCHEDULER_REGISTRY) | set(_NAME_ALIASES))}."
        )
    return key


def _strategy_explicitly_declared(optimization: Any) -> bool:
    """True when the YAML itself named a scheduler family.

    ``lr_scheduler_strategy`` carries ``default="cosine"``, so *every* resolved
    config presents a family name whether or not the arm asked for one. 305 arms
    declare neither it nor ``scheduler:``; honouring the default would start
    annealing all of them, which is a training-behaviour change nobody requested.
    ``model_fields_set`` is the only thing that separates declared from defaulted
    (the same distinction as guarding the explicit argument rather than the
    resolved one).
    """
    fields_set = getattr(optimization, "model_fields_set", None)
    if fields_set is not None:
        return bool({"lr_scheduler_strategy", "scheduler_type"} & set(fields_set))
    # Non-Pydantic stand-in (test doubles carry no ``model_fields_set``): a name
    # that is present at all is the only signal available.
    return any(
        getattr(optimization, field_name, None) is not None
        for field_name in ("lr_scheduler_strategy", "scheduler_type")
    )


def _run_length_in_scheduler_steps(
    name: str,
    period_key: str,
    max_iterations: int | None,
    accumulation_steps: int,
    warmup_steps: int,
) -> int:
    """How many times the main scheduler steps in the run: its default period.

    ``max_iterations`` counts micro-batches, while the loop steps schedulers
    once per optimizer step and a warmup wrapper withholds its first
    ``warmup_steps`` of those, so a period of ``max_iterations`` left every
    ``accumulation_steps: 4`` arm at 85 % of its base LR at the end of the run.
    A negative ``max_iterations`` means "bounded by epochs", which is not a
    length known here; used as a period it made the cosine alternate between
    the base LR and ``eta_min`` on every step.
    """
    if max_iterations is None or max_iterations <= 0:
        raise ConfigurationError(
            f"Scheduler {name!r} needs {period_key!r} and none could be "
            f"resolved: declare optimization.scheduler.{period_key} or set a "
            f"positive training.max_iterations (got {max_iterations!r}). Falling "
            "back to the torch default of 100 is what made the LR oscillate "
            "instead of anneal (issue #533)."
        )
    steps = max_iterations // max(1, int(accumulation_steps)) - warmup_steps
    if steps <= 0:
        raise ConfigurationError(
            f"Scheduler {name!r}: warmup_steps={warmup_steps} leaves no optimizer "
            f"steps for the main schedule in a run of {max_iterations} iterations "
            f"at accumulation_steps={accumulation_steps}. Shorten the warmup or "
            f"declare optimization.scheduler.{period_key}."
        )
    return steps


def _merge(dst: dict[str, Any], src: dict[str, Any], src_label: str) -> None:
    """Merge ``src`` into ``dst``, raising on a disagreeing duplicate.

    Two homes for the same knob that disagree is the loss-weight-SSOT failure
    (pitfall #13b) in scheduler clothing: whichever the resolver happened to read
    first would silently decide the schedule.
    """
    for k, v in src.items():
        if v is None:
            continue
        if k in dst and dst[k] != v:
            raise ConfigurationError(
                f"Scheduler knob {k!r} is declared twice with different values: "
                f"{dst[k]!r} vs {v!r} (from {src_label}). Declare it once."
            )
        dst[k] = v


def resolve_scheduler_spec(
    optimization: Any,
    max_iterations: int | None,
) -> SchedulerSpec | None:
    """Resolve the ``optimization`` block into a single scheduler request.

    Args:
        optimization: the frozen ``config.optimization`` block.
        max_iterations: ``config.training.max_iterations``, from which the
            default period of a period-based scheduler is derived. ``None`` or
            negative when the run is epoch-bounded, in which case the period
            must be declared explicitly.

    Returns:
        The resolved spec, or ``None`` when no scheduler is configured.

    Raises:
        ConfigurationError: on an unknown scheduler name, a knob the resolved
            factory cannot consume, a knob declared twice with different values,
            or a period-based scheduler with no resolvable period.
    """
    raw = getattr(optimization, "scheduler", None)
    if raw is None:
        # #662: this early return used to fire before ``lr_scheduler_strategy``
        # was ever read (it is read below, at the ``strategy`` binding), so 560
        # arms that declared a family and no ``scheduler:`` dict trained at a
        # constant LR while their config said ``cosine``. An *explicitly
        # declared* family is now honoured with an empty parameter dict; a
        # *defaulted* one still means "no scheduler".
        if not _strategy_explicitly_declared(optimization):
            return None
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"optimization.scheduler must be a mapping, got {type(raw).__name__}."
        )

    declared_type = raw.get("type")
    flat_type = getattr(optimization, "scheduler_type", None)
    strategy = getattr(optimization, "lr_scheduler_strategy", None)
    if declared_type and flat_type and declared_type != flat_type:
        raise ConfigurationError(
            f"Scheduler type declared twice and disagreeing: "
            f"scheduler.type={declared_type!r} vs "
            f"optimization.scheduler_type={flat_type!r}."
        )
    declared_name = str(declared_type or flat_type or strategy or "cosine").lower()
    name = _canonical_name(declared_name)

    # ---- collect candidate params from every declared home -----------------
    params: dict[str, Any] = {}
    nested = raw.get("kwargs")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ConfigurationError(
                f"optimization.scheduler.kwargs must be a mapping, got {type(nested).__name__}."
            )
        _merge(params, dict(nested), "scheduler.kwargs")
    flat_in_block = {k: v for k, v in raw.items() if k not in ("type", "kwargs")}
    _merge(params, flat_in_block, "scheduler (flat)")
    _merge(
        params,
        dict(getattr(optimization, "lr_scheduler_kwargs", None) or {}),
        "optimization.lr_scheduler_kwargs",
    )
    # Flat optimization-level knobs. Previously documented as "not consumed by
    # the live path" — they are consumed now.
    _merge(
        params,
        {
            "T_max": getattr(optimization, "T_max", None),
            "eta_min": getattr(optimization, "eta_min", None),
        },
        "optimization (flat T_max/eta_min)",
    )

    # ---- normalise warmup spellings and split them out --------------------
    for legacy, canonical in _WARMUP_ALIASES.items():
        if legacy in params:
            _merge(params, {canonical: params.pop(legacy)}, f"scheduler.{legacy}")
    warmup_steps = int(params.pop("warmup_steps", 0) or 0)
    warmup_start_lr = params.pop("warmup_start_lr", None)
    warmup_end_lr = params.pop("warmup_end_lr", None)
    # The optimization block carries its own flat warmup_steps field.
    block_warmup = int(getattr(optimization, "warmup_steps", 0) or 0)
    if block_warmup and warmup_steps and block_warmup != warmup_steps:
        raise ConfigurationError(
            f"warmup_steps declared twice and disagreeing: "
            f"optimization.warmup_steps={block_warmup} vs "
            f"scheduler.warmup_steps={warmup_steps}."
        )
    warmup_steps = warmup_steps or block_warmup

    # ---- a warmup-bearing name must actually warm up -----------------------
    # ``warmup_cosine`` aliases onto plain ``cosine_annealing`` because the
    # warmup is the wrapper's job, not the family's. With no warmup length
    # declared anywhere, that alias silently discards the only thing the name
    # asked for -- 15 arms requested warmup_cosine and got plain cosine. The
    # name and the schedule must agree or the config is a contradiction.
    if declared_name in _WARMUP_BEARING_NAMES and warmup_steps <= 0:
        raise ConfigurationError(
            f"Scheduler {declared_name!r} names a warmup but no warmup length is "
            "declared, so the warmup would be silently dropped and the run would "
            f"train under plain {name!r}. Declare optimization.warmup_steps (or "
            f"scheduler.warmup_steps) > 0, or name the decay family directly "
            f"({name!r})."
        )

    # ---- family promotion --------------------------------------------------
    # T_0/T_mult are only consumable by warm restarts. 291 configs declare
    # ``lr_scheduler_strategy: cosine`` alongside them; honour the parameters
    # rather than dropping them.
    if name == "cosine_annealing" and ("T_0" in params or "T_mult" in params):
        logger.info(
            "scheduler: promoting 'cosine_annealing' to "
            "'cosine_annealing_warm_restarts' because T_0/T_mult are declared"
        )
        name = "cosine_annealing_warm_restarts"
        params.pop("T_max", None)

    # ---- period default ----------------------------------------------------
    period_key = _RUN_LENGTH_KEY.get(name)
    if period_key is not None and params.get(period_key) is None:
        params[period_key] = _run_length_in_scheduler_steps(
            name,
            period_key,
            max_iterations,
            optimization.gradient.accumulation_steps,
            warmup_steps,
        )

    # ---- reject anything unroutable ---------------------------------------
    allowed = SCHEDULER_PARAM_KEYS[name]
    # Record into the ledger BEFORE raising. The raise stops this run, but the
    # record is what makes the drop visible in `resolved_config.json` for the
    # runs that came before the raise existed — and it routes through the one
    # helper, instead of a third hand-rolled copy of the same check.
    from spectramr.core.execution_ledger import unconsumed_keys

    unroutable = unconsumed_keys(
        params,
        set(allowed) | set(_WARMUP_KEYS),
        site="spectramr.infrastructure.training.scheduler_resolution",
        stage="scheduler_build",
        consumer=f"scheduler {name!r}",
        path_prefix="optimization.",
        severity="error",
    )
    unroutable = sorted(k for k in params if k not in allowed)
    if unroutable:
        raise ConfigurationError(
            f"Scheduler {name!r} cannot consume {unroutable}. It accepts "
            f"{sorted(allowed)} plus {sorted(_WARMUP_KEYS)}. Either remove the "
            "key or declare a scheduler family that reads it — a knob the "
            "scheduler never reads is a silent no-op (pitfall #15)."
        )

    return SchedulerSpec(
        name=name,
        params=params,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        warmup_end_lr=warmup_end_lr,
    )


def resolved_schedule_record(config: Any) -> dict[str, Any]:
    """The LR schedule a run actually builds, as the record its artifacts stamp.

    ``{"scheduler": None}`` when no scheduler is built: the schema default of
    ``lr_scheduler_strategy`` is not honoured (#662), so any artifact that shows
    that default instead of this record misreports a constant-LR run (#2760).

    Raises:
        ConfigurationError: propagated from :func:`resolve_scheduler_spec`.
    """
    spec = resolve_scheduler_spec(
        config.optimization,
        max_iterations=getattr(getattr(config, "training", None), "max_iterations", None),
    )
    return spec.as_provenance() if spec is not None else {"scheduler": None}


def build_scheduler_from_spec(
    spec: SchedulerSpec,
    optimizer: torch.optim.Optimizer,
) -> Any:
    """Instantiate the scheduler described by ``spec``, wrapping it for warmup."""
    main = SCHEDULER_REGISTRY[spec.name](optimizer, dict(spec.params))
    if spec.warmup_steps <= 0:
        return main
    return WarmupScheduler(
        optimizer,
        main_scheduler=main,
        warmup_steps=spec.warmup_steps,
        warmup_start_lr=spec.warmup_start_lr,
        warmup_end_lr=spec.warmup_end_lr,
    )
