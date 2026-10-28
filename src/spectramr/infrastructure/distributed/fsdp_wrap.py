"""FSDP wrap helper (PR-14 / L1).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-14.

Reads ``parallel.fsdp`` from a TrainingSettings instance and applies
``torch.distributed.fsdp.FullyShardedDataParallel`` with the requested
sharding + mixed-precision policy. When ``fsdp.enabled=False`` this is
a no-op (returns the model unchanged) so single-GPU training is
unaffected.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _resolve_sharding_strategy(name: str) -> Any:
    from torch.distributed.fsdp import ShardingStrategy

    table = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    if name not in table:
        raise ValueError(f"Unknown FSDP sharding_strategy={name!r}. Use one of {list(table)}.")
    return table[name]


def _resolve_mixed_precision_policy(
    dtype_name: str | None,
    reduce_dtype: str = "float32",
    buffer_dtype: str = "float32",
) -> Any:
    if dtype_name is None:
        return None
    from torch.distributed.fsdp import MixedPrecision

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    if dtype_name not in dtype_map:
        raise ValueError(f"Unknown FSDP mixed_precision={dtype_name!r}. Use fp16 | bf16 | null.")
    dtype = dtype_map[dtype_name]
    resolved = {"float32": torch.float32, "match_param": dtype}
    for knob, value in (("reduce_dtype", reduce_dtype), ("buffer_dtype", buffer_dtype)):
        if value not in resolved:
            raise ValueError(f"Unknown FSDP {knob}={value!r}. Use float32 | match_param.")
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=resolved[reduce_dtype],
        buffer_dtype=resolved[buffer_dtype],
    )


def _resolve_layer_classes(names: tuple[str, ...]) -> set[type]:
    """Turn configured class NAMES into the classes FSDP wraps on.

    Names, not import paths, because that is what a user reading a model's
    ``print(model)`` output actually sees. Resolution scans the registered model
    modules; an unresolvable name RAISES rather than being dropped, since a
    silently-dropped name reproduces the empty-set bug one entry at a time.
    """
    import torch.nn as nn

    candidates: dict[str, type] = {}

    def _collect(module: Any) -> None:
        for attr in dir(module):
            obj = getattr(module, attr, None)
            if isinstance(obj, type) and issubclass(obj, nn.Module):
                candidates.setdefault(obj.__name__, obj)

    _collect(nn)
    for module_path in (
        "spectramr.models.blocks",
        "spectramr.models.layers",
        "spectramr.models.generators",
    ):
        try:
            _collect(__import__(module_path, fromlist=["*"]))
        except Exception:  # pragma: no cover - optional model deps
            logger.debug("could not scan %s for FSDP layer classes", module_path)

    resolved: set[type] = set()
    missing: list[str] = []
    for name in names:
        cls = candidates.get(name)
        if cls is None:
            missing.append(name)
        else:
            resolved.add(cls)
    if missing:
        raise ValueError(
            f"fsdp.transformer_layer_cls names {sorted(missing)} match no "
            "nn.Module subclass in torch.nn or the model packages. A name that "
            "resolves to nothing would silently shrink the wrap set -- the same "
            "failure as the empty set this replaced."
        )
    return resolved


def resolve_fsdp_precision(fsdp_cfg: Any, precision_cfg: Any = None) -> str | None:
    """The precision FSDP runs in: ``"bf16"`` or ``None`` (single precision).

    ``optimization.precision`` owns the arm's precision, the same block DeepSpeed reads, so
    FSDP follows it: off means float32, and ``enabled`` with no dtype means float16, the
    global convention. ``parallel.fsdp.mixed_precision`` overrides it when an arm names one.
    Every other precision has to be declared.

    Raises:
        ValueError: the resolved precision is float16. FSDP needs a ``ShardedGradScaler``
            that is not wired, so float16 would underflow its gradients unscaled.
    """
    from spectramr.infrastructure.training.mixed_precision import resolve_amp_precision

    explicit = getattr(fsdp_cfg, "mixed_precision", None)
    if explicit is not None:
        return explicit
    enabled, resolved = resolve_amp_precision(
        bool(getattr(precision_cfg, "enabled", False)),
        getattr(precision_cfg, "dtype", None),
    )
    if not enabled:
        return None
    if resolved == "fp16":
        raise ValueError(
            "optimization.precision resolves to float16 under FSDP, which needs a "
            "ShardedGradScaler that is not wired, so gradients would underflow unscaled. "
            "Declare optimization.precision.dtype: bfloat16 (Ampere or newer), or "
            "optimization.precision.enabled: false for float32."
        )
    return resolved


def _build_auto_wrap_policy(cfg: Any) -> Any:
    name = cfg.auto_wrap_policy
    if name == "none":
        return None
    if name == "size_based":
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        return functools.partial(size_based_auto_wrap_policy, min_num_params=cfg.min_num_params)
    if name == "transformer_block":
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # This used to pass ``transformer_layer_cls=set()``. An empty class set
        # matches NO module, so FSDP wrapped nothing -- effectively NO_SHARD --
        # while ``maybe_wrap_with_fsdp`` logged "FSDP wrapped:
        # sharding=full_shard". There was no config surface to populate the set,
        # so the option could not be made to work from YAML at all (#621 F1).
        declared = tuple(getattr(cfg, "transformer_layer_cls", ()) or ())
        if not declared:
            raise ValueError(
                "fsdp.auto_wrap_policy='transformer_block' requires a non-empty "
                "fsdp.transformer_layer_cls. An empty class set matches no "
                "module, so FSDP would wrap nothing and silently behave as "
                "no_shard while reporting full_shard."
            )
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=_resolve_layer_classes(declared),
        )
    raise ValueError(
        f"Unknown FSDP auto_wrap_policy={name!r}. Use size_based | transformer_block | none."
    )


def _maybe_apply_activation_checkpointing(model: nn.Module) -> None:
    """Wrap nn.Linear / nn.Conv2d submodules with activation checkpointing."""
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
    except ImportError:  # pragma: no cover
        logger.warning("torch.distributed activation_checkpointing API unavailable.")
        return

    def _check(submodule: nn.Module) -> bool:
        return isinstance(submodule, (nn.Linear, nn.Conv2d))

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=_check,
    )


def maybe_wrap_with_fsdp(
    model: nn.Module, parallel_cfg: Any, precision_cfg: Any = None
) -> nn.Module:
    """Wrap ``model`` with FSDP if ``parallel.fsdp.enabled`` is True.

    Args:
        model: The bare ``nn.Module`` returned by the model factory.
        parallel_cfg: ``TrainingSettings.parallel`` (a ParallelismConfigSchema).
        precision_cfg: ``TrainingSettings.optimization.precision``; the source of FSDP's
            precision unless ``parallel.fsdp.mixed_precision`` names one.

    Returns:
        Either the wrapped FSDP module, or the input unchanged when
        FSDP is disabled / torch.distributed is unavailable.
    """
    cfg = getattr(parallel_cfg, "fsdp", None)
    if cfg is None or not cfg.enabled:
        return model

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        # Was a warning + `return model`. That is the silent-fallback shape the
        # repo bans (non-negotiable #3), and DDP has always RAISED for the same
        # misconfiguration -- so one strategy aborted and the other trained a
        # full-length unsharded run whose provenance claimed it was sharded.
        raise RuntimeError(
            "FSDP requested but torch.distributed is not initialised. Launch "
            "with:\n"
            "    torchrun --nproc_per_node=N -m spectramr.cli train-distributed "
            "--config <arm>.yaml\n"
            "Refusing to fall back to the unwrapped model: the run would "
            "complete, report success, and not have been sharded."
        )

    from torch.distributed.fsdp import CPUOffload
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    sharding = _resolve_sharding_strategy(cfg.sharding_strategy)
    precision = resolve_fsdp_precision(cfg, precision_cfg)
    mp_policy = _resolve_mixed_precision_policy(precision, cfg.reduce_dtype, cfg.buffer_dtype)
    wrap_policy = _build_auto_wrap_policy(cfg)
    cpu_offload = CPUOffload(offload_params=True) if cfg.cpu_offload else None

    wrapped = FSDP(
        model,
        sharding_strategy=sharding,
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        cpu_offload=cpu_offload,
        limit_all_gathers=cfg.limit_all_gathers,
        use_orig_params=cfg.use_orig_params,
    )

    if cfg.activation_checkpointing:
        _maybe_apply_activation_checkpointing(wrapped)

    logger.info(
        "FSDP wrapped: sharding=%s mp=%s cpu_offload=%s ckpt=%s",
        cfg.sharding_strategy,
        precision,
        cfg.cpu_offload,
        cfg.activation_checkpointing,
    )
    return wrapped


__all__ = ["maybe_wrap_with_fsdp"]
