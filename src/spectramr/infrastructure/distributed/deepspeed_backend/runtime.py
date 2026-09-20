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
    "deepspeed_available",
    "initialize_deepspeed_engine",
    "require_deepspeed",
]

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

    def guard_loss(self, loss, *, name, global_step, scaler) -> None:
        """No-op: DeepSpeed's fp16 path has its own overflow detect + step skip.

        Raising here would kill runs the engine is designed to survive -- an
        overflowing micro-batch is an expected event under dynamic loss scaling,
        not a corrupted one.
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
        engine.step()
