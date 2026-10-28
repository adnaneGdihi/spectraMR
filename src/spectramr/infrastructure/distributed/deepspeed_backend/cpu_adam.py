"""The client optimizer offload hands DeepSpeed: a CPU Adam rebuilt from the torch one.

``deepspeed.initialize`` refuses a torch optimizer under ``offload_optimizer`` (see
``optimizer_support``), so the offload path rebuilds the Adam/AdamW that
``OptimizationBuilder`` already built as DeepSpeedCPUAdam, carrying the same
parameter groups and hyperparameters. The scheduler is rebuilt on the new
optimizer, because it holds a reference to the one it was built on.
"""

from __future__ import annotations

import logging
from typing import Any

from spectramr.infrastructure.distributed.deepspeed_backend.optimizer_support import (
    qualified_name,
    zero_optimizer_problems,
)
from spectramr.infrastructure.distributed.deepspeed_backend.runtime import (
    CPU_ADAM_HINT,
    cpu_adam_class,
)
from spectramr.infrastructure.training.scheduler_resolution import (
    build_scheduler_from_spec,
    resolve_scheduler_spec,
)

logger = logging.getLogger(__name__)

__all__ = ["prepare_client_optimizer"]

#: Per-group keys DeepSpeedCPUAdam takes.
_HYPER_KEYS = ("lr", "betas", "eps", "weight_decay")
#: Torch execution knobs with no CPU-Adam meaning, plus scheduler bookkeeping.
#: ``decoupled_weight_decay`` is the AdamW mode itself, checked against the class below.
_EXECUTION_KEYS = frozenset(
    {"foreach", "fused", "capturable", "differentiable", "initial_lr", "decoupled_weight_decay"}
)
#: Knobs that change the update rule. Set to a true value, they are refused, not dropped.
_REFUSED_IF_TRUE = ("amsgrad", "maximize")


def _build_cpu_adam(torch_optimizer: Any, *, zenflow: bool) -> tuple[Any, dict[str, Any]]:
    source = qualified_name(type(torch_optimizer))
    adamw_mode = source == "torch.optim.adamw.AdamW"
    groups: list[dict[str, Any]] = []
    for group in torch_optimizer.param_groups:
        unknown = (
            set(group) - set(_HYPER_KEYS) - _EXECUTION_KEYS - set(_REFUSED_IF_TRUE) - {"params"}
        )
        if unknown:
            raise ValueError(
                f"{source} param group carries {sorted(unknown)}, which the CPU Adam rebuild "
                "does not map. Refusing to drop them silently (pitfall #15)."
            )
        for key in _REFUSED_IF_TRUE:
            if group.get(key):
                raise ValueError(
                    f"{source} sets {key}=True, which DeepSpeedCPUAdam does not implement; "
                    "under optimizer offload it cannot be honoured."
                )
        if group.get("decoupled_weight_decay", False) and not adamw_mode:
            raise ValueError(f"{source} applies decoupled weight decay, but only AdamW does.")
        groups.append({"params": list(group["params"]), **{k: group[k] for k in _HYPER_KEYS}})
        groups[-1]["betas"] = tuple(groups[-1]["betas"])
    defaults = torch_optimizer.defaults
    cls = cpu_adam_class(zenflow=zenflow)
    try:
        optimizer = cls(
            groups,
            lr=defaults["lr"],
            betas=tuple(defaults["betas"]),
            eps=defaults["eps"],
            weight_decay=defaults["weight_decay"],
            amsgrad=False,
            adamw_mode=adamw_mode,
        )
    except Exception as exc:
        raise RuntimeError(f"{cls.__name__} construction failed: {exc}\n{CPU_ADAM_HINT}") from exc
    record = {
        "client": cls.__name__,
        "source": source,
        "adamw_mode": adamw_mode,
        "param_groups": len(groups),
    }
    return optimizer, record


def _describe_hint() -> str:
    from spectramr.infrastructure.distributed.deepspeed_backend.runtime import _CPU_ADAM_HINT

    return _CPU_ADAM_HINT


def prepare_client_optimizer(
    torch_optimizer: Any,
    scheduler: Any,
    *,
    config: Any,
    zero_stage: int,
    offload_optimizer: str,
    zenflow: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    """Return ``(client_optimizer, scheduler, record)`` for one optimizer.

    With no offload the torch optimizer passes through unchanged. Under offload it
    is rebuilt as DeepSpeedCPUAdam and the scheduler is re-resolved from the same
    SSOT (``resolve_scheduler_spec``) onto it. Refuses any optimizer
    ``zero_optimizer_problems`` refuses, the same rule the audit reports.
    """
    source = qualified_name(type(torch_optimizer))
    if offload_optimizer == "none":
        return torch_optimizer, scheduler, {"client": source, "source": source}
    problems = zero_optimizer_problems(
        type(torch_optimizer), zero_stage=zero_stage, offload_optimizer=offload_optimizer
    )
    if problems:
        raise ValueError("DeepSpeed would refuse this optimizer: " + " ".join(problems))
    optimizer, record = _build_cpu_adam(torch_optimizer, zenflow=zenflow)
    new_scheduler = _rebuild_scheduler(config, optimizer) if scheduler is not None else None
    _log_once(record)
    return optimizer, new_scheduler, record


def _rebuild_scheduler(config: Any, optimizer: Any) -> Any:
    spec = resolve_scheduler_spec(
        config.optimization,
        max_iterations=getattr(getattr(config, "training", None), "max_iterations", None),
    )
    if spec is None:
        raise RuntimeError("a scheduler was built but resolve_scheduler_spec now returns None")
    return build_scheduler_from_spec(spec, optimizer)


def _log_once(record: dict[str, Any]) -> None:
    """Every rank builds its optimizer; only rank 0 says so."""
    from spectramr.infrastructure.distributed.distributed_training import (
        RankUtility,
    )

    if RankUtility.is_main_rank():
        logger.info(
            "[DeepSpeed] offload_optimizer: %s rebuilt as %s (adamw_mode=%s, %d param groups)",
            record["source"],
            record["client"],
            record["adamw_mode"],
            record["param_groups"],
        )
