"""Site / domain batch balancer (Phase 4e).

Wraps two or more DataLoaders into a single iterator that emits batches
according to a declared :class:`DomainBalancingStrategy`. Three strategies:

- :class:`RoundRobinBalancer` — alternates batches across loaders.
  Step 0: loader[0]; step 1: loader[1]; step 2: loader[0]; ...
  Emits ``{"domain": name, "batch": <loader-batch>}`` dicts so the
  strategy code can dispatch on domain identity.

- :class:`StratifiedBalancer` — yields one composite batch per step that
  contains samples from every loader, weighted by per-loader weights.
  Useful when the GAN discriminator needs a mixed batch per step.

- :class:`ConcatBalancer` — zips loaders without balancing. Each yielded
  dict carries one batch from each loader simultaneously.

- :class:`DomainLabelledBalancer` — concatenates one batch from each of two
  loaders along the batch dimension into ONE flat batch and stamps a
  per-sample ``domain_label`` (1.0 for ``domains[0]``, 0.0 for ``domains[1]``),
  so a strategy that reads the ordinary batch contract can run a
  domain-adversarial term.

The strategy receives a dict keyed by domain name and dispatches the
appropriate model passes. Compare with the legacy code in
``DomainAdaptationTrainingStrategy`` which manually advanced two
DataLoader iterators — easy to desync, no per-batch domain tag.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader

__all__ = [
    "Balancer",
    "ConcatBalancer",
    "DomainLabelledBalancer",
    "RoundRobinBalancer",
    "StratifiedBalancer",
    "build_balancer",
]


class Balancer:
    """Base class — wraps two or more named loaders and emits domain-tagged batches."""

    def __init__(self, loaders: dict[str, DataLoader]):
        if len(loaders) < 2:
            raise ValueError(f"Balancer needs at least 2 loaders; got {len(loaders)}.")
        self.loaders = loaders
        self.names = list(loaders.keys())

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of steps per epoch — depends on strategy."""
        raise NotImplementedError


def _weights_are_uniform(weights: dict[str, float] | None) -> bool:
    """True when every domain shares the same weight (⇒ weighting is a no-op)."""
    if not weights:
        return True
    values = list(weights.values())
    return all(abs(v - values[0]) < 1e-9 for v in values)


class RoundRobinBalancer(Balancer):
    """Alternates batches across loaders, honouring per-domain weights.

    Each step yields a single dict ``{"domain": name, "batch": ...}``. Because
    only ONE domain appears per step, differential oversampling IS expressible
    here (unlike :class:`StratifiedBalancer`): a domain's weight sets how many
    times it appears per rotation cycle. ``{source: 1.0, target: 3.0}`` visits
    ``target`` 3x as often as ``source`` -- the domain-adaptation oversampling
    the ``DomainConfigSchema.weight`` knob advertises. Weights are quantised to
    integer visit counts per cycle (``round``); uniform weights reproduce the
    original strict round-robin exactly. One epoch = the *longest* loader times
    the visit-schedule length (shorter loaders cycle).
    """

    def __init__(
        self,
        loaders: dict[str, DataLoader],
        weights: dict[str, float] | None = None,
    ):
        super().__init__(loaders)
        self.weights = weights or dict.fromkeys(self.names, 1.0)
        # Visit schedule: each domain appears ``round(weight)`` times, evenly
        # interleaved (not blocked as [a,a,a,b]) so oversampling is spread
        # across the epoch rather than clustered.
        self._schedule = self._build_schedule()

    def _build_schedule(self) -> list[str]:
        counts = {name: max(1, round(self.weights.get(name, 1.0))) for name in self.names}
        schedule: list[str] = []
        remaining = dict(counts)
        while any(remaining.values()):
            for name in self.names:
                if remaining[name] > 0:
                    schedule.append(name)
                    remaining[name] -= 1
        return schedule

    def __iter__(self) -> Iterator[dict[str, Any]]:
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        # One schedule pass per loader-length cycle; a weight-w domain is thus
        # visited ``max_len * w`` times vs. ``max_len`` for a weight-1 domain.
        max_len = max(len(loader) for loader in self.loaders.values())
        for _ in range(max_len):
            for domain in self._schedule:
                try:
                    batch = next(iterators[domain])
                except StopIteration:
                    iterators[domain] = iter(self.loaders[domain])
                    batch = next(iterators[domain])
                yield {"domain": domain, "batch": batch}

    def __len__(self) -> int:
        max_len = max(len(loader) for loader in self.loaders.values())
        return max_len * len(self._schedule)


class StratifiedBalancer(Balancer):
    """Yields one composite batch per step containing samples from every loader.

    Each composite batch is a dict ``{name: batch_from_that_loader}``.
    Strategy code is expected to consume all domains per step (e.g.,
    forward pass on each domain, then a combined loss).

    One epoch = the length of the *shortest* loader (we don't repeat the
    short side mid-batch; the strategy sees each target sample at most once).
    """

    def __init__(
        self,
        loaders: dict[str, DataLoader],
        weights: dict[str, float] | None = None,
    ):
        super().__init__(loaders)
        self.weights = weights or dict.fromkeys(self.names, 1.0)
        # A composite step yields exactly one batch per domain ({name: batch}),
        # so this balancer CANNOT express differential oversampling — a weight
        # of 3.0 on one domain has nowhere to go without changing that contract.
        # The old code silently stored non-uniform weights and ignored them: a
        # config asking for 3x target oversampling ran 1:1 (pitfall #15 -- a
        # validated-but-unwired knob). Fail loud instead; route weighted
        # oversampling through ``round_robin`` (which yields one domain/step and
        # CAN honour weights), or add a list-valued composite balancer.
        if not _weights_are_uniform(self.weights):
            raise NotImplementedError(
                "StratifiedBalancer received non-uniform domain weights "
                f"({self.weights}), but its per-step composite ({{name: batch}}) "
                "yields one batch per domain per step and cannot differentially "
                "oversample. Use uniform weights here, or select 'round_robin' "
                "balancing (which honours per-domain weights)."
            )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        min_len = min(len(loader) for loader in self.loaders.values())
        for _ in range(min_len):
            composite: dict[str, Any] = {}
            for name in self.names:
                composite[name] = next(iterators[name])
            yield composite

    def __len__(self) -> int:
        return min(len(loader) for loader in self.loaders.values())


class ConcatBalancer(Balancer):
    """No balancing — zip the loaders.

    Useful when strategy code wants explicit access to both batches every
    step without any reordering. Identical to :class:`StratifiedBalancer`
    with unit weights — kept as a distinct name for readability in YAMLs.
    """

    def __iter__(self) -> Iterator[dict[str, Any]]:
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        min_len = min(len(loader) for loader in self.loaders.values())
        for _ in range(min_len):
            composite: dict[str, Any] = {}
            for name in self.names:
                composite[name] = next(iterators[name])
            yield composite

    def __len__(self) -> int:
        return min(len(loader) for loader in self.loaders.values())


#: Label of each domain position: ``domains[0]`` is the source (synthetic) side,
#: ``domains[1]`` the target (real) side, matching the DANN objective
#: ``-log d(source) - log(1 - d(target))``.
DOMAIN_LABEL_SOURCE = 1.0
DOMAIN_LABEL_TARGET = 0.0


def _leading_size(batch: Any) -> int | None:
    """Batch-dimension size of the first tensor found in a collated batch."""
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0]) if batch.dim() > 0 else None
    if isinstance(batch, Mapping):
        for value in batch.values():
            size = _leading_size(value)
            if size is not None:
                return size
    return None


def _concat_batches(batches: list[Any], path: str) -> Any:
    """Concatenate collated batches along dim 0; raise on any shape it cannot merge."""
    first = batches[0]
    if isinstance(first, Mapping):
        keys = set(first)
        for other in batches[1:]:
            if not isinstance(other, Mapping) or set(other) != keys:
                other_keys = sorted(other) if isinstance(other, Mapping) else type(other).__name__
                raise ValueError(
                    f"DomainLabelledBalancer: domain batches disagree at '{path or '<root>'}': "
                    f"{sorted(keys)} vs {other_keys}. Every domain must emit the same keys."
                )
        return {k: _concat_batches([b[k] for b in batches], f"{path}.{k}".lstrip(".")) for k in first}
    if isinstance(first, torch.Tensor):
        return torch.cat(batches, dim=0)
    if isinstance(first, (list, tuple)):
        return [item for b in batches for item in b]
    raise TypeError(
        f"DomainLabelledBalancer: cannot concatenate '{path}' of type "
        f"{type(first).__name__}; expected a tensor, list or mapping."
    )


class DomainLabelledBalancer(Balancer):
    """One flat batch per step: ``domains[0]`` samples, then ``domains[1]`` samples.

    Each step draws one batch from each loader, concatenates every key along the
    batch dimension and adds ``domain_label`` (``float32``, shape ``[B0 + B1]``).
    One epoch is the source loader's length; the target loader cycles.
    """

    def __init__(
        self,
        loaders: dict[str, DataLoader],
        weights: dict[str, float] | None = None,
    ):
        super().__init__(loaders)
        if len(self.names) != 2:
            raise ValueError(
                "DomainLabelledBalancer stamps a binary domain label and needs exactly "
                f"2 domains; got {len(self.names)} ({self.names})."
            )
        if not _weights_are_uniform(weights):
            raise NotImplementedError(
                f"DomainLabelledBalancer received non-uniform domain weights ({weights}); "
                "it draws one batch per domain per step and cannot oversample. "
                "Use uniform weights or 'round_robin' balancing."
            )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        source, target = self.names
        target_iter = iter(self.loaders[target])
        for source_batch in self.loaders[source]:
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(self.loaders[target])
                target_batch = next(target_iter)
            yield self._merge(source_batch, target_batch)

    @staticmethod
    def _merge(source_batch: Any, target_batch: Any) -> dict[str, Any]:
        if not isinstance(source_batch, Mapping) or not isinstance(target_batch, Mapping):
            raise TypeError("DomainLabelledBalancer needs dict batches from both loaders.")
        if "domain_label" in source_batch or "domain_label" in target_batch:
            raise ValueError("DomainLabelledBalancer: a loader already emits 'domain_label'.")
        sizes = (_leading_size(source_batch), _leading_size(target_batch))
        if None in sizes:
            raise ValueError("DomainLabelledBalancer: a domain batch carries no batched tensor.")
        merged = _concat_batches([source_batch, target_batch], "")
        merged["domain_label"] = torch.cat(
            (
                torch.full((sizes[0],), DOMAIN_LABEL_SOURCE),
                torch.full((sizes[1],), DOMAIN_LABEL_TARGET),
            )
        )
        return merged

    def __len__(self) -> int:
        return len(self.loaders[self.names[0]])


_BALANCERS: dict[str, Callable[[dict[str, DataLoader], dict[str, float] | None], Balancer]] = {
    "round_robin": RoundRobinBalancer,
    "stratified": StratifiedBalancer,
    "concat": lambda loaders, _weights: ConcatBalancer(loaders),
    "domain_labelled": DomainLabelledBalancer,
}


def build_balancer(
    loaders: dict[str, DataLoader],
    strategy: str = "round_robin",
    weights: dict[str, float] | None = None,
) -> Balancer:
    """Dispatch on strategy string to the right balancer class."""
    factory = _BALANCERS.get(strategy)
    if factory is None:
        raise ValueError(
            f"Unknown balancing strategy: {strategy}. Must be one of: {', '.join(_BALANCERS)}."
        )
    return factory(loaders, weights)
