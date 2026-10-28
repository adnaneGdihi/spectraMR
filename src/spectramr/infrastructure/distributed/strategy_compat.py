"""Which parallel backends a training strategy declares it can run under.

A strategy is built *after* the models are wrapped, and the wrappers only reach
what is in ``env.models``. Three things a strategy does therefore break silently:

* it owns a module outside ``env.models`` (CycleGAN's second generator, JEPA's
  predictor): DDP/ZeRO never all-reduce its gradients, so each rank trains a
  different network and nothing errors;
* it calls a non-forward method on the generator (``encode``, ``sample``): the
  wrapper does not forward the attribute, or, under FSDP/ZeRO-3, the call skips
  the all-gather and reads sharded views;
* it adds a parameter group after the optimizer was adopted: ZeRO never sees it.

``BaseTrainingStrategy.parallel_compat`` is the declaration. Two tokens, because
the hazards differ by backend, not by name:

``"replicated"``
    DDP, DP, and DeepSpeed ZeRO 0-2. Every rank holds whole parameters.
``"sharded"``
    FSDP and DeepSpeed ZeRO-3. Parameters are partitioned, so a call outside the
    wrapped ``forward`` reads a shard.

``None`` means undeclared. A **restriction** (a declared set lacking the token)
is inherited by subclasses; a **permission** is not, because a subclass can add
the very module that made its parent safe. Single-process runs are never gated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spectramr.infrastructure.distributed.owned_modules import (
    OWNED_BUFFERS,
    OWNED_CALL,
    OWNED_MODULE,
    owned_verdict,
    refusal_reason,
)

REPLICATED = "replicated"
SHARDED = "sharded"
VALID_TOKENS: frozenset[str] = frozenset({REPLICATED, SHARDED})


@dataclass(frozen=True)
class CompatVerdict:
    """Outcome of one strategy x backend question.

    ``refused`` stops the run; ``undeclared`` is advisory only (a replicated
    backend with no declaration is the status quo for most arms, so it must not
    fail an ``--strict`` audit).
    """

    refused: bool
    undeclared: bool
    kind: str | None
    message: str


def backend_kind(parallel_strategy: str | None, zero_stage: int | None = None) -> str | None:
    """Classify a ``parallel.strategy`` as replicated / sharded, or ``None`` for single-process."""
    if parallel_strategy in (None, "none"):
        return None
    if parallel_strategy == "fsdp":
        return SHARDED
    if parallel_strategy == "deepspeed":
        return SHARDED if (zero_stage or 0) >= 3 else REPLICATED
    if parallel_strategy in ("dp", "ddp"):
        return REPLICATED
    raise ValueError(
        f"unknown parallel.strategy {parallel_strategy!r}; cannot classify its sharding"
    )


def _declared(cls: type) -> frozenset[str] | None:
    declared = vars(cls).get("parallel_compat")
    if declared is None:
        return None
    unknown = set(declared) - VALID_TOKENS
    if unknown:
        raise ValueError(
            f"{cls.__name__}.parallel_compat has unknown token(s) {sorted(unknown)}; "
            f"valid tokens are {sorted(VALID_TOKENS)}."
        )
    return frozenset(declared)


def evaluate_parallel_compat(
    strategy_cls: type, parallel_strategy: str | None, zero_stage: int | None = None
) -> CompatVerdict:
    """Decide whether ``strategy_cls`` may run under ``parallel_strategy``."""
    kind = backend_kind(parallel_strategy, zero_stage)
    name = strategy_cls.__name__
    if kind is None:
        return CompatVerdict(False, False, None, f"{name}: single-process, not gated.")

    label = f"parallel.strategy={parallel_strategy!r}" + (
        f" (zero_stage={zero_stage})" if parallel_strategy == "deepspeed" else ""
    )
    for cls in strategy_cls.__mro__:
        declared = _declared(cls)
        if declared is not None and kind not in declared:
            return CompatVerdict(
                True,
                False,
                kind,
                f"{name} cannot run under {label}: {cls.__name__}.parallel_compat="
                f"{sorted(declared)} does not include {kind!r}. A module the strategy "
                "owns outside env.models or a call outside the wrapped forward is not "
                "reached by the wrapper, so ranks would diverge without an error.",
            )

    own = _declared(strategy_cls)
    if own is not None:
        return CompatVerdict(False, False, kind, f"{name} declares {kind!r} support.")
    if kind == SHARDED:
        return CompatVerdict(
            True,
            True,
            kind,
            f"{name} has not declared parallel_compat, and {label} shards parameters: "
            "a call outside the wrapped forward would read a shard. Declare "
            f"parallel_compat = frozenset({{{SHARDED!r}}}) on the class once it is audited, "
            "or use a replicated backend (ddp, or deepspeed zero_stage <= 2).",
        )
    return CompatVerdict(
        False,
        True,
        kind,
        f"{name} has not declared parallel_compat; {label} replicates parameters, "
        "so it runs, but the strategy is unaudited for it.",
    )


def check_strategy_for_parallel(config: Any, strategy_cls: type) -> CompatVerdict:
    """:func:`evaluate_parallel_compat` for a loaded config's ``parallel`` block.

    The class token is judged first. Then each parameter the config makes the strategy
    build itself is judged by :func:`owned_verdict`, the same decision the runtime makes,
    so a config the run would refuse is refused here too.
    """
    parallel = getattr(config, "parallel", None)
    if parallel is None:
        return evaluate_parallel_compat(strategy_cls, "none")
    deepspeed = getattr(parallel, "deepspeed", None)
    backend = getattr(parallel, "strategy", "none")
    verdict = evaluate_parallel_compat(
        strategy_cls, backend, getattr(deepspeed, "zero_stage", None)
    )
    if verdict.refused:
        return verdict
    for name, mechanism in strategy_cls.owned_module_declarations(config):
        if owned_verdict(mechanism, backend) == "refuse":
            what, where = {
                OWNED_CALL: ("calls", "outside the wrapped forward"),
                OWNED_BUFFERS: ("adds", "to the wrapped model after the wrap"),
            }.get(mechanism, ("builds", "outside env.models"))
            synced = mechanism in (OWNED_MODULE, OWNED_BUFFERS)
            usable = "none, dp or ddp" if synced else "none or dp"
            return CompatVerdict(
                True,
                False,
                verdict.kind,
                f"{strategy_cls.__name__} {what} {name!r} {where} for this config, and "
                f"parallel.strategy={backend!r} cannot run it: "
                f"{refusal_reason(mechanism, backend)}. Turn the feature off, or use "
                f"parallel.strategy: {usable}.",
            )
    return verdict
