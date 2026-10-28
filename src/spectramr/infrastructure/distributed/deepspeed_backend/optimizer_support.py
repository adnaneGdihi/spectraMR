"""Which optimizers DeepSpeed accepts under ZeRO, and under optimizer offload.

``deepspeed.initialize`` rejects both cases below from *inside* the engine, after
the model, data and physics have been built:

* ZeRO stage >= 1 asserts unless the client optimizer is one of
  ``ZERO_SUPPORTED_OPTIMIZERS`` (``zero_allow_untested_optimizer`` is the escape,
  and this repo does not emit it);
* ``offload_optimizer`` other than ``none`` raises ``ZeRORuntimeException`` unless
  the client optimizer is ``DeepSpeedCPUAdam`` or ``DeepSpeedCPULion``
  (``zero_force_ds_cpu_optimizer`` defaults to true). This repo therefore rebuilds
  a torch Adam/AdamW as ``DeepSpeedCPUAdam`` under offload (``cpu_adam.py``), and
  refuses every other optimizer.

Matching is by qualified class name so the audit can run without DeepSpeed
installed. The sets mirror ``deepspeed/runtime/zero/utils.py`` at 0.19.6.
"""

from __future__ import annotations

ZERO_SUPPORTED: frozenset[str] = frozenset(
    {
        "torch.optim.adam.Adam",
        "torch.optim.adamw.AdamW",
        "torch.optim.adagrad.Adagrad",
        "deepspeed.ops.adam.fused_adam.FusedAdam",
        "deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam",
        "deepspeed.ops.adam.zenflow_cpu_adam.ZenFlowCPUAdam",
        "deepspeed.ops.adagrad.cpu_adagrad.DeepSpeedCPUAdagrad",
        "deepspeed.ops.lion.cpu_lion.DeepSpeedCPULion",
        "deepspeed.ops.lion.fused_lion.FusedLion",
        "deepspeed.runtime.zero.muon.muon_optimizer.MuonWithAuxAdam",
    }
)

#: The optimizers ``cpu_adam.py`` can rebuild as DeepSpeedCPUAdam under offload.
#: Only the exact torch classes: a subclass has its own step and is refused.
CPU_OFFLOAD_REBUILDABLE: frozenset[str] = frozenset(
    {
        "torch.optim.adam.Adam",
        "torch.optim.adamw.AdamW",
    }
)


def qualified_name(optimizer_cls: type) -> str:
    return f"{optimizer_cls.__module__}.{optimizer_cls.__qualname__}"


def zero_optimizer_problems(
    optimizer_cls: type, *, zero_stage: int, offload_optimizer: str
) -> list[str]:
    """What DeepSpeed would refuse about this optimizer, as sentences. Empty means accepted."""
    name = qualified_name(optimizer_cls)
    problems: list[str] = []
    if zero_stage >= 1 and name not in ZERO_SUPPORTED:
        problems.append(
            f"{name} is not a ZeRO-supported optimizer; deepspeed.initialize asserts at "
            f"zero_stage={zero_stage}. Use adam, adamw or adagrad."
        )
    if offload_optimizer != "none" and name not in CPU_OFFLOAD_REBUILDABLE:
        problems.append(
            f"offload_optimizer={offload_optimizer!r} needs an Adam or AdamW optimizer, "
            f"which is rebuilt as DeepSpeedCPUAdam; this arm builds {name}, and "
            "deepspeed.initialize raises ZeRORuntimeException for any other client optimizer."
        )
    return problems
