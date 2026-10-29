"""Refuse a profiling run that cannot succeed, before Scalene starts.

``spectramr profile --target train`` pays for the entire training environment —
data pipeline, model build, DI container, first loader batch — before the
training loop reaches either of the two conditions checked here. On the arm that
motivated this module that was **11.5 minutes** to a ``ConfigurationError``.

The wasted time is the smaller half of the problem. Scalene profiles the process
it launched, so a child that dies at minute twelve still produces a
``scalene-profile.json`` — a syntactically valid profile of environment
construction and nothing else, sitting in a properly-named directory beside a
manifest. Nothing about that artifact says "this run never trained"; you find
out by reading ``outcome.exit_code``. A profile that is wrong rather than absent
is exactly the shape non-negotiable 3 exists to prevent, so the refusal happens
up front, in about a second, with no directory created.

Both checks are **reused verdicts, never restated ones** (non-negotiable 17):

* the process-group requirement is asked of the strategy plugins themselves via
  :attr:`~spectramr.infrastructure.distributed.strategy_registry.ParallelStrategyPlugin.requires_process_group`,
  so a backend added tomorrow answers for itself. Spelling it here as
  ``strategy != "none"`` would be a second owner *and* wrong: ``dp`` is
  ``nn.DataParallel``, genuinely single-process, and profiles fine.
* the validation-gate arithmetic is :func:`spectramr.pipelines.train.validation_can_fire`
  — the same predicate the loop consults — so the pre-flight verdict and the
  runtime verdict cannot drift apart.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Targets that build a training environment and therefore apply
#: ``parallel.strategy``. ``infer`` is absent because it never touches
#: parallelism — ``pipelines/infer.py`` resolves no strategy — so refusing it
#: would be a veto the runtime itself would not have cast.
PARALLELISM_TARGETS = frozenset({"train", "sanity_check"})

#: Targets whose iteration budget is the one the config declares.
#:
#: ``sanity_check`` is deliberately ABSENT. That mode OVERWRITES
#: ``training.max_iterations`` with 5000 *after* overrides are applied
#: (``pipelines/training_loop.py`` ~line 774), so the two numbers this check
#: would compare are not the two numbers the loop ends up using: an arm with a
#: consistent 15000/15000 pair is made inconsistent BY THE MODE. The loop makes
#: exactly this exemption — it warns for a sanity check and raises only for a
#: real train — and this set is how that exemption is honoured here.
INTERVAL_TARGETS = frozenset({"train"})

#: Every spelling argparse accepts for the child's ``--override`` / ``-O``,
#: which is ``action="append"`` on the train parser. All five were verified
#: against ``argparse`` rather than assumed, because a spelling this extractor
#: misses does not fail — it makes the whole interval check silently pass:
#:     --override k=v   --override=k=v   -O k=v   -Ok=v   -O=k=v
#: Note the last one: argparse strips exactly one ``=`` after a short option, so
#: ``-O=a=1`` yields ``a=1`` and not ``=a=1``.
_LONG, _SHORT = "--override", "-O"


def extract_overrides(extra: list[str] | None) -> list[str]:
    """Pull the ``key=value`` strings out of the passthrough argv.

    Returns them in order, so a later override of the same key wins the way it
    does in the child — this list is fed to the real
    :func:`spectramr.config.overrides.apply_overrides`, never interpreted here.
    """
    tokens = list(extra or [])
    found: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in (_LONG, _SHORT):
            if i + 1 < len(tokens):
                found.append(tokens[i + 1])
                i += 2
                continue
        elif tok.startswith(f"{_LONG}="):
            found.append(tok[len(_LONG) + 1 :])
        elif tok.startswith(f"{_SHORT}=") and len(tok) > 3:
            found.append(tok[len(_SHORT) + 1 :])
        elif tok.startswith(_SHORT) and len(tok) > len(_SHORT):
            found.append(tok[len(_SHORT) :])
        i += 1
    return found


def effective_settings(settings: Any, extra: list[str] | None) -> Any:
    """Apply the passthrough overrides, so the checks see what the child will.

    An override that cannot be applied here degrades this to a check of the
    arm's *declared* values and says so. It never silently skips: the child
    applies the same overrides itself and reports the real failure, so
    swallowing the reason would hide which of the two happened
    (non-negotiable 18 — absent is a state to report, never one to infer).
    """
    overrides = extract_overrides(extra)
    if not overrides:
        return settings
    from spectramr.config.overrides import apply_overrides

    try:
        return apply_overrides(settings, list(overrides))
    except Exception as exc:
        logger.warning(
            "Pre-flight could not apply %d passthrough override(s) (%s: %s). "
            "Checking the arm's DECLARED values instead — the profiled child "
            "applies them itself and will report the failure properly.",
            len(overrides),
            type(exc).__name__,
            exc,
        )
        return settings


def process_group_strategies() -> tuple[str, ...]:
    """Registered strategies that require a ``torch.distributed`` process group.

    Asked of the registry, so adding a backend cannot leave this list stale.
    """
    from spectramr.infrastructure.distributed.strategy_registry import (
        list_parallel_strategies,
        resolve_parallel_strategy,
    )

    return tuple(
        name
        for name in list_parallel_strategies()
        if getattr(resolve_parallel_strategy(name), "requires_process_group", False)
    )


def check_parallel_strategy(*, settings: Any, target: str) -> None:
    """Refuse a process-group strategy: ``profile`` launches a single process.

    Scalene profiles the process it launches and *not* that process's children,
    which is why ``train-distributed`` is not a profileable target at all
    (``profile_command.PROFILE_TARGETS``). So a profiling run is always
    single-process, and an arm declaring ddp/fsdp/deepspeed will reach
    ``_require_process_group`` and die — correctly, but only after the
    environment is built.
    """
    if target not in PARALLELISM_TARGETS:
        return
    parallel = getattr(settings, "parallel", None)
    strategy = getattr(parallel, "strategy", None)
    if strategy is None or strategy not in process_group_strategies():
        return
    raise ValueError(
        f"parallel.strategy={strategy!r} cannot be profiled: it requires an "
        f"initialized torch.distributed process group, and `spectramr profile` "
        f"launches a single process. The run would build the whole training "
        f"environment and then raise from _require_process_group.\n"
        f"  Fix: pass --single-process, which profiles the arm with "
        f"parallel.strategy=none and its {strategy} block disabled. The profile "
        f"then measures one plain rank, not the {strategy} optimizer path.\n"
        f"  Strategies that profile as-is: none, dp."
    )


def single_process_overrides(settings: Any) -> list[str]:
    """The ``--override`` argv that turns the arm's parallelism off, or ``[]``.

    Empty for a strategy that needs no process group, so the flag is safe on a
    whole folder. The sub-block named like the strategy (``fsdp``, ``deepspeed``)
    carries the ``enabled`` flag the schema requires to agree with ``strategy``.
    """
    parallel = getattr(settings, "parallel", None)
    strategy = getattr(parallel, "strategy", None)
    if strategy is None or strategy not in process_group_strategies():
        return []
    argv = ["--override", "parallel.strategy=none"]
    if getattr(getattr(parallel, strategy, None), "enabled", False):
        argv += ["--override", f"parallel.{strategy}.enabled=false"]
    return argv


def check_validation_interval(*, settings: Any, target: str) -> None:
    """Refuse a budget the validation gate can never fire inside.

    Reproduces the loop's own verdict by calling the loop's own predicate, and
    only where that verdict is *knowable* up front. Three things make it
    unknowable, and each returns instead of guessing:

    * ``training.max_iterations`` is ``None`` — the budget is then derived at
      runtime from ``epochs x len(train_loader)``, and the loader does not exist
      yet;
    * ``validation.schedule.on_epoch`` is on — the epoch gate can rescue the
      run, but whether it does depends on ``len(train_loader)``, same problem;
    * ``target`` is ``sanity_check`` — see :data:`INTERVAL_TARGETS`.

    An arm declaring ``validation.enabled: false`` passes: the data builder attaches no val
    loader, so the loop never applies the gate (issue #673).

    The one residual imprecision is stated rather than hidden: the loop also
    requires a validation loader to exist before it raises, and that too needs a
    built pipeline. An arm with no validation split would therefore be refused
    here and would have run there — so the message names the override that
    resolves it either way rather than telling the operator to edit the arm.
    """
    if target not in INTERVAL_TARGETS:
        return
    from spectramr.config.schemas.validation import validation_declared_off

    training = getattr(settings, "training", None)
    validation = getattr(settings, "validation", None)
    if validation_declared_off(validation):
        return
    schedule = getattr(validation, "schedule", None)
    max_iterations = getattr(training, "max_iterations", None)
    eval_interval = getattr(schedule, "interval_steps", None)
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        return
    if not isinstance(eval_interval, int):
        return
    if getattr(schedule, "on_epoch", False):
        return

    from spectramr.pipelines.train import validation_can_fire

    # `eval_on_epoch=False` is not a restatement of the branch above -- it is
    # what makes this call answer the STEP gate alone, the only half of
    # `validation_can_fire` that is decidable without a built train loader.
    if validation_can_fire(
        eval_interval=eval_interval,
        max_iterations=max_iterations,
        eval_on_epoch=False,
    ):
        return

    suggested = max(1, max_iterations // 2)
    raise ValueError(
        f"validation.schedule.interval_steps={eval_interval} exceeds this run's "
        f"whole budget of {max_iterations} iterations, so the validation gate "
        f"can NEVER fire: the profiled run would train to completion, evaluate "
        f"early stopping zero times, write no best checkpoint, and the loop "
        f"would raise ConfigurationError after the environment is built.\n"
        f"  Fix: pass `-O validation.schedule.interval_steps={suggested}` "
        f"(>= 2 events) alongside the budget you already set. "
        f"`-O validation.schedule.on_epoch=true` is the other route, but it is "
        f"a weaker one: the epoch gate only fires if a whole epoch fits inside "
        f"the budget, and this check goes silent once on_epoch is set because "
        f"the train loader's length is not knowable here.\n"
        f"  Refused here rather than at runtime because the environment build "
        f"costs minutes and Scalene would still write a profile of it."
    )


def check_parameter_free(*, config: Any, target: str) -> None:
    """Refuse to profile training of a classical solver: ``train`` raises on it.

    The model class declares ``parameter_free``; the dispatcher already answers
    that per arm (:func:`manifest_dispatch.declares_parameter_free_arm`), so the
    verdict is reused, not restated. Without this, the child builds the data
    pipeline (minutes on M4Raw) and only then raises "nothing to train".
    """
    if target not in PARALLELISM_TARGETS:
        return
    from spectramr.cli.manifest_dispatch import declares_parameter_free_arm

    try:
        parameter_free = declares_parameter_free_arm(config)
    except ImportError as exc:
        logger.warning("Parameter-free check skipped: %s is not importable.", exc.name)
        return
    if parameter_free:
        raise ValueError(
            f"{config} names a parameter-free model: there is nothing to train, so "
            f"there is no training loop to profile. Grade it with "
            f"`spectramr train --config {config} --validate-only`, and leave it "
            f"out of the profile array (EXCLUDE='*<name>*')."
        )


def run_preflight(args: argparse.Namespace, settings: Any) -> None:
    """Every pre-flight check, against the settings the child will actually see.

    Called for a dry run too. Printing an argv that is guaranteed to fail is the
    same trap with an extra step in front of it, and the whole point of
    ``--dry-run`` is to inspect a command worth running.
    """
    resolved = effective_settings(settings, getattr(args, "extra", None))
    check_parallel_strategy(settings=resolved, target=args.target)
    check_validation_interval(settings=resolved, target=args.target)
    check_parameter_free(config=args.config, target=args.target)


__all__ = [
    "INTERVAL_TARGETS",
    "PARALLELISM_TARGETS",
    "check_parallel_strategy",
    "check_parameter_free",
    "check_validation_interval",
    "effective_settings",
    "extract_overrides",
    "process_group_strategies",
    "run_preflight",
    "single_process_overrides",
]
