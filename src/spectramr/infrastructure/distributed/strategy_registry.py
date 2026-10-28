"""Parallel-strategy plugins: one registry, two ordered hooks.

Replaces ``pipelines.parallel.apply_parallelism``, whose ``if/elif/else raise``
implemented ``none``/``dp``/``ddp`` while FSDP was reached through a *separate*
``parallel.fsdp.enabled`` flag read in ``ModelBuilder.build()``. So
``strategy: 'fsdp'`` -- the spelling the reference template advertised -- raised
``ValueError`` **after** the whole training environment had been built, while
``strategy: 'none'`` + ``fsdp.enabled: true`` silently sharded.

The ordering is the load-bearing part, and it is not symmetric:

.. code-block:: text

    ModelBuilder: build -> compile -> EMA
      Stage A: prepare_models(models, ctx)
          SyncBatchNorm, FSDP wrap
    OptimizationBuilder: optimizers / schedulers
      Stage B: adopt(models, optimizers, schedulers, ctx)
          DataParallel / DDP wrap, DistributedSampler

**FSDP must be Stage A.** ``FSDP.__init__`` flattens parameters into a shard and
re-points their storage. An optimizer built before that holds parameters whose
storage was swapped underneath it: gradients arrive shard-shaped against
full-shape ``exp_avg``, which is a shape error on the first step -- or, on the
``foreach`` path, a silently wrong update. Full-size optimizer state would also
defeat the memory saving FSDP exists for.

**DDP must not move to Stage A.** It would work -- DDP does not change parameter
identity -- but every downstream consumer that expects a bare module
(``count_parameters``, the optimizer builder's model selection) would then see a
``DistributedDataParallel``.

Registration is an explicit table rather than a decorator scan: there are five
strategies, they are a closed vocabulary in the schema, and a plugin that fails
to import must be a loud error rather than a name that quietly resolves to
nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "ParallelContext",
    "ParallelRuntime",
    "ParallelStrategyPlugin",
    "ParallelizationResult",
    "register_parallel_strategy",
    "resolve_parallel_strategy",
]


@dataclass(frozen=True)
class ParallelContext:
    """What a plugin needs to know, and nothing more."""

    config: Any
    device: Any
    parallel: Any

    @property
    def world_size(self) -> int:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1


@dataclass
class ParallelizationResult:
    """What a plugin hands back. Mutable: the director threads it onward."""

    models: dict[str, Any]
    optimizers: dict[str, Any] = field(default_factory=dict)
    schedulers: dict[str, Any] = field(default_factory=dict)
    #: The policy that owns backward/step. Every plugin returns one, so the
    #: consumer never branches on strategy name to find it.
    step_policy: Any | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ParallelStrategyPlugin(Protocol):
    """One parallelisation backend."""

    name: str

    #: True when this backend requires an initialized ``torch.distributed``
    #: process group, i.e. a ``torchrun``/``srun`` launch. Part of the protocol
    #: rather than a detail of the built-ins so a caller can ask the registry
    #: "would this strategy survive a single-process run?" without importing the
    #: concrete classes or hardcoding a name list.
    requires_process_group: bool

    def prepare_models(self, models: dict[str, Any], ctx: ParallelContext) -> dict[str, Any]:
        """Stage A — before optimizers exist. Wrapping here changes parameters."""
        ...

    def prepare_ema(self, ema: Any, models: dict[str, Any], ctx: ParallelContext) -> None:
        """Stage A, after ``prepare_models``: adapt the EMA shadow to the wrapped models."""
        ...

    def adopt(
        self,
        models: dict[str, Any],
        optimizers: dict[str, Any],
        schedulers: dict[str, Any],
        ctx: ParallelContext,
    ) -> ParallelizationResult:
        """Stage B — after optimizers exist. Wrapping here must not change them."""
        ...

    def prepare_data_loaders(self, loaders: dict[str, Any], ctx: ParallelContext) -> dict[str, Any]:
        """Substitute samplers (DistributedSampler) where the strategy needs it."""
        ...

    def checkpoint_adapter(self, ctx: ParallelContext) -> Any:
        """How this strategy's checkpoints must be written.

        A method rather than a field on :class:`ParallelizationResult` so the
        base class can supply the default: a new plugin that forgets to build
        one inherits ``DefaultCheckpointAdapter`` instead of ``None``, and the
        director has exactly one call site to keep correct.
        """
        ...

    def provenance(self, ctx: ParallelContext) -> dict[str, Any]:
        """What to stamp into the run record."""
        ...


_REGISTRY: dict[str, ParallelStrategyPlugin] = {}


def register_parallel_strategy(plugin: ParallelStrategyPlugin) -> None:
    """Register *plugin* under its ``name``. Re-registering the same one is fine."""
    existing = _REGISTRY.get(plugin.name)
    if existing is not None and type(existing) is not type(plugin):
        raise ValueError(
            f"parallel strategy {plugin.name!r} already registered to "
            f"{type(existing).__name__}; refusing to overwrite with "
            f"{type(plugin).__name__}."
        )
    _REGISTRY[plugin.name] = plugin


def resolve_parallel_strategy(name: str) -> ParallelStrategyPlugin:
    """Look up a plugin. Raises on an unknown name -- never a silent no-op.

    The schema already closes this vocabulary, so reaching here with an unknown
    name means the two have drifted; saying so is more useful than defaulting to
    single-process and training something the config did not ask for.
    """
    _ensure_builtins_registered()
    plugin = _REGISTRY.get(name)
    if plugin is None:
        raise ValueError(
            f"Unknown parallel strategy {name!r}. Registered: "
            f"{sorted(_REGISTRY)}. The schema's ParallelStrategy Literal and this "
            "registry have drifted apart."
        )
    return plugin


def list_parallel_strategies() -> list[str]:
    _ensure_builtins_registered()
    return sorted(_REGISTRY)


def _ensure_builtins_registered() -> None:
    """Import the built-in plugins on first use.

    Deferred rather than done at module import: the plugins import this module
    for :func:`register_parallel_strategy`, so an eager import at the top would
    be a cycle.
    """
    if _REGISTRY:
        return
    from spectramr.infrastructure.distributed import strategies  # noqa: F401


@dataclass(frozen=True)
class ParallelRuntime:
    """The resolved parallelism, threaded onto ``TrainingEnvironment``."""

    strategy: str
    step_policy: Any | None = None
    checkpoint_adapter: Any | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def single_process(cls) -> ParallelRuntime:
        return cls(strategy="none")

    @property
    def checkpoints_require_all_ranks(self) -> bool:
        """True when gating checkpoints on rank 0 alone would DEADLOCK.

        Read once by the training loop and ORed with ``is_main_rank()``. It is
        a property rather than four edited call-site conditions so that the
        fifth checkpoint site added later inherits the answer instead of
        silently reintroducing the hang.
        """
        return bool(getattr(self.checkpoint_adapter, "requires_all_ranks", False))
