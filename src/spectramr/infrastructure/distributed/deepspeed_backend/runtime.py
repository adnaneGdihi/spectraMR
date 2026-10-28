"""DeepSpeed engine construction and the step policy that drives it.

``initialize_deepspeed_engine`` is deliberately the **single import site** for
``deepspeed`` in this package. That is not tidiness -- it is what makes the whole
backend testable without the dependency: a fake engine monkeypatched onto this
one symbol exercises every wiring decision (accumulation ownership, zero_grad
ownership, checkpoint routing) on CPU.
"""

from __future__ import annotations

import logging
from typing import Any

from spectramr.infrastructure.training.strategy_interfaces import IStepPolicy

logger = logging.getLogger(__name__)

__all__ = [
    "DeepSpeedStepPolicy",
    "cpu_adam_class",
    "deepspeed_available",
    "initialize_deepspeed_engine",
    "require_deepspeed",
]

#: DeepSpeedCPUAdam is a C++ extension built on first construction, not at import.
CPU_ADAM_HINT = (
    "parallel.deepspeed.offload_optimizer needs DeepSpeedCPUAdam, "
    "a C++ extension that DeepSpeed JIT-builds on first use. It needs a C++ toolchain "
    "(ninja, g++, and nvcc matching the CUDA build) and a writable extension cache "
    "(TORCH_EXTENSIONS_DIR, else XDG_CACHE_HOME). Pre-building it: "
    "DS_BUILD_CPU_ADAM=1 pip install --no-build-isolation deepspeed. "
    "Refusing to fall back to torch.optim: the engine rejects it under offload."
)

_INSTALL_HINT = (
    "parallel.strategy='deepspeed' requires the [deepspeed] extra. Install with:\n"
    '    pip install -e ".[deepspeed]"\n'
    "Refusing to fall back to another strategy: ZeRO sharding is the arm's memory "
    "claim, so substituting DDP would leave the run reporting numbers that belong "
    "to a different configuration."
)

#: Shown when ``deepspeed`` IS installed but importing it raised. Telling this
#: operator to reinstall the extra sends them to fix the one thing that is not
#: wrong; DeepSpeed writes caches during its own import, so the usual cause is a
#: directory it cannot write to rather than a missing or mismatched package.
_IMPORTABLE_BUT_BROKEN_HINT = (
    "parallel.strategy='deepspeed' is configured and the [deepspeed] extra IS "
    "installed, but importing it failed.\n"
    "DeepSpeed writes to TRITON_CACHE_DIR and to torch's JIT extension directory "
    "(under XDG_CACHE_HOME) while it is still importing, so a denied or full "
    "cache path presents here rather than at first use. The failing path is "
    "named below when the OS reported one.\n"
    "Refusing to fall back to another strategy: ZeRO sharding is the arm's memory "
    "claim, so substituting DDP would leave the run reporting numbers that belong "
    "to a different configuration."
)


def deepspeed_available() -> bool:
    """True when ``deepspeed`` can be imported.

    Uses ``find_spec`` so a mere availability check never pays DeepSpeed's
    import cost (it probes CUDA and can JIT-build ops).
    """
    import importlib.util

    try:
        return importlib.util.find_spec("deepspeed") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def _describe_import_failure(exc: BaseException) -> str:
    """Render an import exception WITHOUT discarding the path it names.

    ``repr()`` on an ``OSError`` drops ``filename``, and that is the single fact
    needed to act on the failure::

        repr(PermissionError(13, 'Permission denied', '/p'))
        -> "PermissionError(13, 'Permission denied')"      # path gone
        str(PermissionError(13, 'Permission denied', '/p'))
        -> "[Errno 13] Permission denied: '/p'"            # path kept

    This function used to be an inline ``{exc!r}``, which is how a DeepSpeed
    import denied on a cluster ``$HOME`` reported four identical
    ``PermissionError(13, 'Permission denied')`` lines and named no directory --
    the failing path had to be recovered by reasoning about which caches
    DeepSpeed touches at import. ``str()`` keeps it, and the type name is added
    back because ``str()`` alone drops that.
    """
    rendered = str(exc).strip()
    detail = f"{type(exc).__name__}: {rendered}" if rendered else type(exc).__name__
    filename = getattr(exc, "filename", None)
    # Belt and braces: an OSError raised with a filename but an empty strerror
    # still stringifies without the path.
    if filename and str(filename) not in detail:
        detail = f"{detail} (path: {filename})"
    return detail


def require_deepspeed() -> Any:
    """Import and return the ``deepspeed`` module, or raise with the install hint."""
    try:
        import deepspeed
    except Exception as exc:
        # Broad on purpose: DeepSpeed's import probes CUDA and can raise
        # RuntimeError/OSError rather than ImportError on a mismatched build.
        # Letting that escape would present as "spectramr is broken" instead of
        # "this extra does not fit this machine".
        #
        # Which of those two it is decides the advice, so ask rather than assume.
        # ``_INSTALL_HINT`` alone told an operator to install a package that was
        # already installed and had failed for an unrelated reason (a cache write
        # denied under $HOME), sending them to fix the wrong thing.
        hint = _INSTALL_HINT if not deepspeed_available() else _IMPORTABLE_BUT_BROKEN_HINT
        raise ImportError(f"{hint}\n(import failed: {_describe_import_failure(exc)})") from exc
    return deepspeed


def cpu_adam_class(*, zenflow: bool) -> Any:
    """The CPU Adam class offload needs: ZenFlowCPUAdam under ZenFlow, else DeepSpeedCPUAdam.

    The same single ``deepspeed`` import site as :func:`require_deepspeed`; the class
    is only looked up here, so a missing or unbuildable op raises with the hint.
    """
    require_deepspeed()
    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam, ZenFlowCPUAdam
    except Exception as exc:
        raise ImportError(f"{CPU_ADAM_HINT}\n(import failed: {_describe_import_failure(exc)})") from exc
    return ZenFlowCPUAdam if zenflow else DeepSpeedCPUAdam


def initialize_deepspeed_engine(
    *,
    model: Any,
    optimizer: Any,
    lr_scheduler: Any,
    config: dict[str, Any],
) -> tuple[Any, Any, Any]:
    """Wrap *model* in a DeepSpeed engine that ADOPTS the given optimizer.

    Returns ``(engine, optimizer, lr_scheduler)`` as DeepSpeed hands them back --
    it may replace the optimizer with a ZeRO-sharded wrapper around the one it
    was given.

    The optimizer is passed in rather than declared in ``config`` so TTUR, the LR
    multipliers and ``resolve_scheduler_spec`` keep describing what actually ran.
    """
    deepspeed = require_deepspeed()
    if config.get("compile", {}).get("deepcompile"):
        _assert_deepcompile_supported(model)
    engine, ds_optimizer, _, ds_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
    )
    logger.info(
        "[DeepSpeed] engine initialised: zero_stage=%s micro_batch=%s accum=%s",
        config.get("zero_optimization", {}).get("stage"),
        config.get("train_micro_batch_size_per_gpu"),
        config.get("gradient_accumulation_steps"),
    )
    _activate_deepcompile(engine, config)
    return engine, ds_optimizer, ds_scheduler


def _deepcompile_device(model: Any) -> Any:
    """The device this rank compiles on: the model's, else the current CUDA device."""
    import torch

    params = getattr(model, "parameters", None)
    if callable(params):
        first = next(iter(params()), None)
        if first is not None:
            return first.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _assert_deepcompile_supported(model: Any) -> None:
    """Refuse DeepCompile on a GPU family Inductor cannot target, before ``initialize``.

    DeepCompile lowers through Inductor (DeepSpeed's CUDA accelerator default), so
    it inherits Triton's sm_70 floor; on a P100 the failure otherwise surfaces at
    the first compiled step, after the allocation has been paid for.
    """
    from spectramr.core.compile_capability import compile_backend_for, triton_available
    from spectramr.core.device_capabilities import capability_of_device

    device = _deepcompile_device(model)
    compile_backend_for(
        device.type, triton=triton_available(), capability=capability_of_device(device)
    )


def _activate_deepcompile(engine: Any, config: dict[str, Any]) -> None:
    """Call ``engine.compile()`` when the ds_config asked for DeepCompile.

    **This is the half that makes the config real.** With
    ``compile.deepcompile: true`` and no ``engine.compile()``, DeepSpeed emits
    exactly one ``log_dist_once`` line on rank 0 --

        "DeepCompile is enabled but engine.compile() has not been called;
         executing without DeepCompile until compile() runs."

    -- and runs eagerly. A once-only, rank-0, startup-time log line on a cluster
    is indistinguishable from silence, so the knob would be declared, accepted
    by DeepSpeed, faithfully stamped into provenance, and inert: pitfall #16
    with a paper trail arguing it worked.

    Raises rather than warns on failure. Compilation is the arm's performance
    claim; a run that silently falls back to eager is reporting throughput
    numbers that belong to a different configuration -- the same reasoning that
    made the torch.compile path stop swallowing its exceptions.
    """
    if not config.get("compile", {}).get("deepcompile"):
        return
    compile_fn = getattr(engine, "compile", None)
    if compile_fn is None:
        raise RuntimeError(
            "parallel.deepspeed.compile.enabled is set but this DeepSpeed engine "
            "has no compile(). DeepCompile needs deepspeed>=0.17; upgrade the "
            "[deepspeed] extra or set compile.enabled: false."
        )
    compile_fn()
    logger.info(
        "[DeepSpeed] DeepCompile active: passes=%s (engine.compile() called)",
        config["compile"].get("passes") or "default",
    )


def _fp32_overflow_check_absent(engine: Any) -> bool:
    """True when DeepSpeed will not check gradients for overflow itself.

    ``engine.py`` enables ``check_grad_overflow`` only for fp16 (and bf16 when
    configured), so fp32 ZeRO never skips a non-finite step.
    """
    return not (engine.fp16_enabled() or engine.bfloat16_enabled())


def _flatten_tensors(obj: Any) -> list[Any]:
    import torch

    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, list | tuple):
        return [t for item in obj for t in _flatten_tensors(item)]
    return []


def _local_gradients(engine: Any) -> list[Any]:
    """The gradients this rank holds after ``engine.backward`` at a boundary.

    Stage 0 keeps the reduced gradients on the module parameters. ZeRO 1-3 keep
    this rank's partition in the optimizer's averaged-gradient buffers, which is
    what ``engine.step`` consumes, so checking it checks what reaches the weights.
    Finding none is an error, never a silent pass.
    """
    if engine.zero_optimization_stage() == 0:
        grads = [p.grad for p in engine.module.parameters() if p.grad is not None]
    else:
        buffers = getattr(engine.optimizer, "averaged_gradients", None)
        if buffers is None:
            raise RuntimeError(
                "fp32 DeepSpeed gradient check cannot locate this ZeRO stage's "
                "gradient partitions (optimizer has no averaged_gradients); refusing "
                "to step unverified."
            )
        grads = _flatten_tensors(buffers)
    if not grads:
        raise RuntimeError(
            "fp32 DeepSpeed gradient check found no gradients at an accumulation "
            "boundary; refusing to step unverified."
        )
    return grads


def _raise_if_nonfinite_gradients(engine: Any) -> None:
    """Raise when any rank holds a non-finite gradient at this optimizer step.

    One fused norm per (device, dtype) group; the agreed flag is all-reduced
    across ranks so every rank raises together. The single ``.item()`` is the
    deliberate per-optimizer-step host read (performance.md, rule 9 exception).
    """
    import torch

    from spectramr.infrastructure.training.nonfinite_monitor import process_group_spans_ranks

    by_group: dict[tuple[Any, Any], list[Any]] = {}
    grads = _local_gradients(engine)
    for grad in grads:
        by_group.setdefault((grad.device, grad.dtype), []).append(grad)
    # Offloaded partitions live on the CPU, and NCCL reduces only accelerator tensors,
    # so each group's verdict moves to the engine's device before stacking.
    device = getattr(engine, "device", None) or grads[0].device
    group_flags = [
        (~torch.isfinite(torch.stack(torch._foreach_norm(group)))).any().to(device)
        for group in by_group.values()
    ]
    flag = torch.stack(group_flags).any().to(torch.float32).reshape(1)
    if process_group_spans_ranks():
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    if flag.item() > 0:
        stage = engine.zero_optimization_stage()
        raise RuntimeError(
            f"Non-finite gradient at optimizer step {engine.global_steps} on at least one "
            f"rank (fp32 DeepSpeed, ZeRO stage {stage}). DeepSpeed's overflow check is off "
            "outside fp16/bf16, so engine.step() would apply NaN to every weight; "
            "refusing to step (check the loss terms and the coil maps)."
        )


class DeepSpeedStepPolicy(IStepPolicy):
    """Backward and step through the engine.

    Both ownership flags are True, and that is the entire point of the class.

    ``engine.backward(loss)`` already divides by the accumulation count and
    tracks the boundary; ``engine.step()`` zeroes gradients internally. If
    ``StepExecutor`` also did those, the run would get **1/N^2 loss scaling and
    one real optimizer step per N^2 micro-batches** -- and it would train, and
    every single-host test would pass, because nothing about the shapes or the
    control flow is wrong. Only the numbers are.
    """

    owns_gradient_accumulation = True
    owns_zero_grad = True
    #: ``engine.step()`` steps the scheduler handed to ``deepspeed.initialize``.
    owns_lr_scheduler = True

    def __init__(self, engines: dict[int, Any] | None = None) -> None:
        #: keyed by ``id(optimizer)`` so one policy can serve several engines
        #: if ``allow_multi_engine`` is ever enabled.
        self._engines: dict[int, Any] = engines or {}

    def register(self, optimizer: Any, engine: Any) -> None:
        self._engines[id(optimizer)] = engine

    def _engine_for(self, optimizer: Any) -> Any:
        engine = self._engines.get(id(optimizer))
        if engine is None:
            raise RuntimeError(
                "DeepSpeedStepPolicy received an optimizer with no registered "
                "engine. Stepping it directly would bypass ZeRO entirely and "
                "update unsharded copies of the weights."
            )
        return engine

    def guard_loss(self, loss, *, name, global_step, scaler, monitor=None) -> None:
        """No-op: DeepSpeed's fp16 path has its own overflow detect + step skip.

        Raising here, or recording into the loop's ``monitor``, would kill runs
        the engine is designed to survive -- an overflowing micro-batch is an
        expected event under dynamic loss scaling, not a corrupted one.
        """
        return

    def clip_gradients(self, model, max_norm: float) -> None:
        """No-op: clipping is declared in the ds_config and applied by the engine.

        Doing it here as well would clip already-clipped gradients.
        """
        return

    def backward_and_step(
        self,
        loss,
        optimizer,
        model=None,
        model_name: str = "model",
        epoch: int = 0,
        model_type: str = "unknown",
        scaler=None,
        perform_step: bool = True,
        gradient_clipping_fn=None,
        skip_scaler_update: bool = False,
    ) -> None:
        """``engine.backward`` then ``engine.step`` -- both, every micro-batch.

        ``perform_step`` is ignored on purpose. The engine decides accumulation
        boundaries itself (``is_gradient_accumulation_boundary``), so calling
        ``step()`` every time is correct: it is a no-op mid-window. Honouring the
        executor's boundary instead would double-gate and stall the schedule.
        """
        engine = self._engine_for(optimizer)
        engine.backward(loss)
        # Before step(): once the engine applies a NaN gradient the weights are gone.
        if engine.is_gradient_accumulation_boundary() and _fp32_overflow_check_absent(engine):
            _raise_if_nonfinite_gradients(engine)
        engine.step()
