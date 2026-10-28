"""FSDP2 (``fully_shard``) wrap helper.

Reads ``parallel.fsdp`` and shards a model in place with
``torch.distributed.fsdp.fully_shard`` on a ``DeviceMesh``. FSDP2 adds no wrapper
class: the model keeps its type (it gains ``FSDPModule``), its parameters become
DTensors, and ``torch.nn.utils.clip_grad_norm_`` over them is already the global
norm. The names of the sharded units are recorded on the root so the EMA shadow can
be sharded identically (:func:`shard_like`).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

#: Attribute on the root holding the dotted names of its ``fully_shard`` units ("" = root).
FSDP_UNITS_ATTR = "_spectramr_fsdp_units"


def _resolve_mixed_precision_policy(dtype_name: str | None, reduce_dtype: str = "float32") -> Any:
    if dtype_name is None:
        return None
    from torch.distributed.fsdp import MixedPrecisionPolicy

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    if dtype_name not in dtype_map:
        raise ValueError(f"Unknown FSDP mixed_precision={dtype_name!r}. Use fp16 | bf16 | null.")
    dtype = dtype_map[dtype_name]
    resolved = {"float32": torch.float32, "match_param": dtype}
    if reduce_dtype not in resolved:
        raise ValueError(f"Unknown FSDP reduce_dtype={reduce_dtype!r}. Use float32 | match_param.")
    return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=resolved[reduce_dtype])


def _resolve_layer_classes(names: tuple[str, ...]) -> set[type]:
    """Turn configured class NAMES into the classes FSDP shards on.

    Names, not import paths, because that is what a user reading a model's
    ``print(model)`` output actually sees. An unresolvable name RAISES rather than
    being dropped, since a silently-dropped name shrinks the shard set unseen.
    """
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

    missing = sorted(name for name in names if name not in candidates)
    if missing:
        raise ValueError(
            f"fsdp.transformer_layer_cls names {missing} match no nn.Module subclass in "
            "torch.nn or the model packages. A name that resolves to nothing would "
            "silently shrink the shard set."
        )
    return {candidates[name] for name in names}


def resolve_fsdp_precision(fsdp_cfg: Any, precision_cfg: Any = None) -> str | None:
    """The precision FSDP runs in: ``"fp16"``, ``"bf16"`` or ``None`` (single precision).

    ``optimization.precision`` owns the arm's precision, the same block DeepSpeed reads, so
    FSDP follows it: off means float32, and ``enabled`` with no dtype means float16, the
    global convention. ``parallel.fsdp.mixed_precision`` overrides it when an arm names one.
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
    return resolved


def build_fsdp_mesh(sharding_strategy: str, device_type: str) -> Any:
    """The mesh a sharding strategy means: 1-D shard, or 2-D (replicate, shard).

    ``no_shard`` is a 2-D mesh whose shard dimension has size 1, i.e. pure
    replication, so every strategy goes through one ``fully_shard`` path.
    """
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    world = dist.get_world_size()
    if sharding_strategy in ("full_shard", "shard_grad_op"):
        return init_device_mesh(device_type, (world,), mesh_dim_names=("shard",))
    if sharding_strategy == "no_shard":
        return init_device_mesh(device_type, (world, 1), mesh_dim_names=("replicate", "shard"))
    if sharding_strategy == "hybrid_shard":
        from spectramr.core.topology import resolve_run_topology

        local = resolve_run_topology().local_world_size
        if local < 1 or world % local:
            raise ValueError(
                f"fsdp.sharding_strategy='hybrid_shard' needs the world size ({world}) to be "
                f"a multiple of the ranks per node ({local})."
            )
        return init_device_mesh(
            device_type, (world // local, local), mesh_dim_names=("replicate", "shard")
        )
    raise ValueError(
        f"Unknown FSDP sharding_strategy={sharding_strategy!r}. Use full_shard | "
        "shard_grad_op | no_shard | hybrid_shard."
    )


def _select_units(model: nn.Module, cfg: Any) -> list[str]:
    """Dotted names of the submodules that become their own shard unit, deepest first."""
    policy = cfg.auto_wrap_policy
    if policy == "none":
        return []
    named = [(name, module) for name, module in model.named_modules() if name]
    if policy == "transformer_block":
        declared = tuple(getattr(cfg, "transformer_layer_cls", ()) or ())
        if not declared:
            raise ValueError(
                "fsdp.auto_wrap_policy='transformer_block' requires a non-empty "
                "fsdp.transformer_layer_cls. An empty class set matches no module, so "
                "FSDP would shard only the root while reporting per-block units."
            )
        classes = tuple(_resolve_layer_classes(declared))
        units = [name for name, module in named if isinstance(module, classes)]
        if not units:
            raise ValueError(
                f"fsdp.transformer_layer_cls {list(declared)} matched no submodule of "
                f"{type(model).__name__}."
            )
        return sorted(units, key=lambda n: n.count("."), reverse=True)
    if policy == "size_based":
        # Post-order, as FSDP1's size policy: a module whose parameters not yet
        # claimed by a deeper unit reach min_num_params becomes a unit itself.
        claimed: set[int] = set()
        units = []
        for name, module in sorted(named, key=lambda nm: nm[0].count("."), reverse=True):
            free = [p for p in module.parameters() if id(p) not in claimed]
            if sum(p.numel() for p in free) >= cfg.min_num_params:
                units.append(name)
                claimed.update(id(p) for p in free)
        return units
    raise ValueError(
        f"Unknown FSDP auto_wrap_policy={policy!r}. Use size_based | transformer_block | none."
    )


def _apply_units(model: nn.Module, units: list[str], **shard_kwargs: Any) -> None:
    from torch.distributed.fsdp import fully_shard

    modules = dict(model.named_modules())
    for name in units:
        fully_shard(modules[name], **shard_kwargs)
    fully_shard(model, **shard_kwargs)
    setattr(model, FSDP_UNITS_ATTR, (*units, ""))


def maybe_wrap_with_fsdp(
    model: nn.Module, parallel_cfg: Any, precision_cfg: Any = None, device: Any = None
) -> nn.Module:
    """Shard ``model`` in place with ``fully_shard`` if ``parallel.fsdp.enabled``.

    Args:
        model: The bare ``nn.Module`` returned by the model factory, on its device.
        parallel_cfg: ``TrainingSettings.parallel`` (a ParallelismConfigSchema).
        precision_cfg: ``TrainingSettings.optimization.precision``; the source of FSDP's
            precision unless ``parallel.fsdp.mixed_precision`` names one.
        device: The run's device; its type selects the mesh's backend device. Defaults to
            the model's parameter device.

    Returns:
        The same module object, now an ``FSDPModule``, or the input unchanged when FSDP
        is disabled.
    """
    cfg = getattr(parallel_cfg, "fsdp", None)
    if cfg is None or not cfg.enabled:
        return model

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        raise RuntimeError(
            "FSDP requested but torch.distributed is not initialised. Launch with:\n"
            "    torchrun --nproc_per_node=N -m spectramr.cli train-distributed "
            "--config <arm>.yaml\n"
            "Refusing to fall back to the unsharded model: the run would complete, "
            "report success, and not have been sharded."
        )

    from torch.distributed.fsdp import CPUOffloadPolicy

    if device is None:
        device = next(model.parameters()).device
    mesh = build_fsdp_mesh(cfg.sharding_strategy, torch.device(device).type)
    precision = resolve_fsdp_precision(cfg, precision_cfg)
    if precision is not None and any(p.is_complex() for p in model.parameters()):
        # Neither float16 nor bfloat16 has a complex counterpart the policy could cast to.
        # Refused before any collective, so every rank raises alike.
        raise ValueError(
            f"FSDP {precision} cannot hold the complex parameters of {type(model).__name__}. "
            "Run it in float32 (optimization.precision.enabled: false)."
        )
    kwargs: dict[str, Any] = {
        "mesh": mesh,
        "reshard_after_forward": cfg.sharding_strategy != "shard_grad_op",
    }
    mp_policy = _resolve_mixed_precision_policy(precision, cfg.reduce_dtype)
    if mp_policy is not None:
        kwargs["mp_policy"] = mp_policy
    if cfg.cpu_offload:
        kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)

    units = _select_units(model, cfg)
    _apply_units(model, units, **kwargs)
    logger.info(
        "FSDP2 sharded %s: sharding=%s mesh=%s units=%d mp=%s cpu_offload=%s",
        type(model).__name__,
        cfg.sharding_strategy,
        tuple(mesh.shape),
        len(units) + 1,
        precision,
        cfg.cpu_offload,
    )
    return model


def shard_like(target: nn.Module, reference: nn.Module) -> nn.Module:
    """Shard ``target`` (a structural copy of ``reference``) onto the same units and mesh.

    For the EMA shadow: with both sharded alike, the per-step blend and the validation
    swap run shard against shard, with no gather. The shadow never runs a forward, so
    it takes no mixed-precision or offload policy.
    """
    units = getattr(reference, FSDP_UNITS_ATTR, None)
    if units is None:
        raise RuntimeError(
            f"{type(reference).__name__} was not sharded by maybe_wrap_with_fsdp, so there "
            "is no shard plan to copy onto the EMA shadow."
        )
    mesh = next(reference.parameters()).device_mesh
    _apply_units(target, [name for name in units if name], mesh=mesh)
    return target


__all__ = ["FSDP_UNITS_ATTR", "build_fsdp_mesh", "maybe_wrap_with_fsdp", "shard_like"]
