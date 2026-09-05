"""Metrics Registry - Single source of truth for all metrics.

This module provides a registry pattern for metrics, enabling:
- Dynamic metric registration via decorators
- O(1) metric lookup by name or alias
- Consistent interface (IMetric protocol)
- Full type safety

Example:
    >>> from spectramr.core.metrics import get_metric, list_available
    >>> psnr = get_metric("psnr")
    >>> score = psnr(prediction, target)

    .. mermaid::

        graph TD
            A[Metric Implementation] -->|@register_metric| B(MetricsRegistry)
            C[User Code] -->|get_metric| B
            B -->|Returns Instance| A
            D[Config System] -->|ValidationMetricsConfig| B
            B -->|Computes| E[Results]
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Protocol, runtime_checkable

import torch

from spectramr.config.schemas.enums import Regime, Task

logger = logging.getLogger(__name__)


@runtime_checkable
class IMetric(Protocol):
    """Protocol that all metrics must implement.

    Metrics must be callable, taking prediction and target tensors.
    """

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        """Compute the metric value.

        Args:
            prediction: Model output tensor
            target: Ground truth tensor
            **kwargs: Additional metric-specific arguments

        Returns:
            Scalar metric value
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name of the metric."""
        ...

    @property
    def higher_is_better(self) -> bool:
        """True if higher values indicate better performance."""
        ...


class MetricsRegistry:
    """Singleton registry for all metric implementations.

    Usage:
        @MetricsRegistry.register("psnr", aliases=["PSNR", "peak_snr"])
        class PSNRMetric:
            ...

        # Get metric by name
        metric = MetricsRegistry.get("psnr")
        score = metric(pred, target)
    """

    _instance: ClassVar[MetricsRegistry | None] = None
    _metrics: ClassVar[dict[str, type]] = {}
    _aliases: ClassVar[dict[str, str]] = {}
    _initialized: ClassVar[bool] = False
    # Capability metadata for the no-reference metric battery (spec §1.2). A
    # metric declares — via the decorator OR matching class attributes — which
    # MetricContext fields it consumes (``needs``), whether it needs the
    # measurement/prior bundle at all (``requires_measurement_context``), and
    # whether it needs a reference ``target`` (``requires_reference``). The
    # audit ladder (§6) and the validation/sim2rank loops read these to decide
    # whether a metric can be computed on a given cohort.
    _needs: ClassVar[dict[str, tuple[str, ...]]] = {}
    _requires_context: ClassVar[dict[str, bool]] = {}
    _requires_reference: ClassVar[dict[str, bool]] = {}
    # Workflow tagging (imaging-regime × task). Keyed by canonical metric name;
    # ``None`` values mean unannotated. Consumed by the maturity ledger to
    # assert an EVAL_ONLY regime is backed by real, registered metrics.
    _workflow_tags: ClassVar[dict[str, dict[str, Any]]] = {}

    # Every mutable registration table, named ONCE. ``clear()``, ``snapshot()``
    # and ``restore()`` all iterate this tuple instead of each spelling the set
    # out, because they WERE three independent enumerations and one of them was
    # wrong. The isolation helper in ``test_registry.py`` saved three of the six
    # tables ``clear()`` wipes, so ``_needs`` stayed EMPTY for the rest of the
    # process and every later ``needs()`` read returned ``()`` -- a wrong value
    # rather than an error, because ``needs()`` cannot distinguish "declares
    # nothing" from "never registered". Under ``pytest-xdist`` that made the
    # FAILURE SET depend on which files shared a worker, which is why the
    # release lane could not be parallelised (#1846). Add a table here and all
    # three helpers cover it at once (non-negotiable 17); the ratchet in
    # ``test_registry.py`` fails if a table is declared and not listed.
    _TABLES: ClassVar[tuple[str, ...]] = (
        "_metrics",
        "_aliases",
        "_needs",
        "_requires_context",
        "_requires_reference",
        "_workflow_tags",
    )

    def __new__(cls) -> MetricsRegistry:
        """__new__.

        Returns:
            MetricsRegistry: Description.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(
        cls,
        name: str,
        aliases: list[str] | None = None,
        *,
        requires_reference: bool | None = None,
        requires_measurement_context: bool | None = None,
        needs: tuple[str, ...] | None = None,
        direction: str | None = None,
        workflows: frozenset[Regime] | None = None,
        tasks: frozenset[Task] | None = None,
    ) -> Any:
        """Decorator to register a metric class.

        Args:
            name: Canonical name for the metric (lowercase)
            aliases: Optional list of alternative names
            requires_reference: ``True`` (default) for full-reference metrics
                that need ``target``; ``False`` for no-reference metrics.
                If ``None``, falls back to a ``requires_reference`` class
                attribute, else ``True``.
            requires_measurement_context: ``True`` if the metric consumes a
                :class:`~spectramr.core.metrics.context.MetricContext` (k-space,
                mask, coil maps, reconstructor, prior, …). If ``None``,
                inferred from ``needs`` (any field ⇒ ``True``) or a class attr.
            needs: tuple of ``MetricContext`` field names the metric reads
                (e.g. ``("y_kspace", "mask", "coil_maps")``). Used by the §6
                audit to verify the data pipeline exposes them. If ``None``,
                falls back to a ``needs`` class attribute, else ``()``.
            direction: optional ``"higher"``/``"lower"`` convenience that, when
                the class neither self-declares ``higher_is_better`` nor is in
                the central ``METRIC_HIGHER_IS_BETTER`` map, sets the class
                ``higher_is_better`` attribute. Prefer a self-declared
                ``higher_is_better`` property; ``direction`` is for spec-fidelity.
            workflows: Imaging regimes this metric is specific to. ``None`` =
                agnostic (skip). Backs EVAL_ONLY regimes in the maturity ledger.
            tasks: Tasks this metric is specific to. ``None`` = agnostic.

        Returns:
            Decorator function

        Example:
            @register_metric("psnr", aliases=["PSNR"])
            class PSNRMetric:
                ...

            @register_metric("ndcr", requires_reference=False,
                             requires_measurement_context=True,
                             needs=("y_kspace", "mask", "coil_maps"),
                             direction="lower")
            class NormalisedDataConsistencyResidual:
                ...
        """

        def decorator(metric_cls: type) -> type:
            """decorator.

            Args:
                metric_cls (type): Description.
            Returns:
                type: Description.
            """
            # Per CLAUDE.md #9 and TODO/audit/13_metrics.md F1+F16, a duplicate
            # registration with a *different* class raises so the audit
            # ladder catches collisions at boot rather than silently
            # flipping the resolution at runtime. Re-registering the same
            # class is idempotent (test-reload-safe).
            if name in cls._metrics:
                if cls._metrics[name] is metric_cls:
                    return metric_cls
                raise ValueError(
                    f"Metric '{name}' already registered to "
                    f"{cls._metrics[name].__module__}.{cls._metrics[name].__qualname__}; "
                    f"refusing to overwrite with "
                    f"{metric_cls.__module__}.{metric_cls.__qualname__}. "
                    "Rename one of the two registrations."
                )

            # Validate the workflow tags before storing them. This table is read
            # by exactly one consumer — the maturity ledger's anti-facade check —
            # which does `regime in workflows`. A bare string
            # (workflows="mri_flow") would be stored happily and then SUBSTRING
            # match, so `Regime.FLOW in "mri_flow"` style errors would silently
            # back a maturity claim. `direction` next door already
            # validates-or-raises; this did not (CLAUDE.md #15).
            for label, tags, member in (
                ("workflows", workflows, Regime),
                ("tasks", tasks, Task),
            ):
                if tags is None:
                    continue
                if isinstance(tags, str) or not isinstance(tags, (frozenset, set)):
                    raise TypeError(
                        f"metric {name!r}: {label}= must be a frozenset of "
                        f"{member.__name__}, got {type(tags).__name__}. A bare "
                        "string would substring-match in the maturity ledger."
                    )
                bad = [t for t in tags if not isinstance(t, member)]
                if bad:
                    raise TypeError(
                        f"metric {name!r}: {label}= contains non-{member.__name__} entries {bad!r}."
                    )

            cls._metrics[name] = metric_cls
            if workflows is not None or tasks is not None:
                cls._workflow_tags[name] = {"workflows": workflows, "tasks": tasks}

            # Ensure the IMetric contract's ``higher_is_better`` is satisfied.
            # Metrics that subclass ``BaseMetric`` historically declared no
            # direction; the value lives in the central SSOT map and is
            # injected here so the reporting/ranking layers don't fall back to
            # a wrong default (see metric_directions.py). Classes that declare
            # their own direction (attribute or property) are left untouched.
            from spectramr.core.metrics.metric_directions import METRIC_HIGHER_IS_BETTER

            # Validate the advertised ``direction`` knob unconditionally
            # (CLAUDE.md #15 — a wired knob must validate-or-raise on every
            # path). Previously the membership check lived inside the
            # ``elif`` branch, so an invalid direction was silently accepted
            # whenever the class self-declared higher_is_better or the name was
            # in METRIC_HIGHER_IS_BETTER.
            if direction is not None and direction not in ("higher", "lower"):
                raise ValueError(
                    f"metric {name!r}: direction must be 'higher' or 'lower', got {direction!r}"
                )

            declares_own = any("higher_is_better" in base.__dict__ for base in metric_cls.__mro__)
            if not declares_own and name in METRIC_HIGHER_IS_BETTER:
                metric_cls.higher_is_better = METRIC_HIGHER_IS_BETTER[name]
            elif not declares_own and direction is not None:
                # Spec-fidelity convenience: ``direction="higher"/"lower"`` maps
                # to the boolean SSOT only when no other source declares it, so
                # the metric_directions ↔ METRIC_SPECS parity test never sees a
                # third, conflicting direction source.
                metric_cls.higher_is_better = direction == "higher"

            # --- capability metadata (spec §1.2) --------------------------
            # Resolve from the decorator kwarg first, else a matching class
            # attribute, else a sensible default. This lets a metric declare
            # its contract either in the decorator (spec style) or on the class.
            resolved_needs: tuple[str, ...] = tuple(
                needs if needs is not None else getattr(metric_cls, "needs", ()) or ()
            )
            resolved_requires_ctx = (
                requires_measurement_context
                if requires_measurement_context is not None
                else bool(
                    getattr(metric_cls, "requires_measurement_context", False) or resolved_needs
                )
            )
            resolved_requires_ref = (
                requires_reference
                if requires_reference is not None
                else bool(getattr(metric_cls, "requires_reference", True))
            )
            cls._needs[name] = resolved_needs
            cls._requires_context[name] = resolved_requires_ctx
            cls._requires_reference[name] = resolved_requires_ref

            if aliases:
                for alias in aliases:
                    alias_key = alias.lower()
                    existing_alias = cls._aliases.get(alias_key)
                    if existing_alias is not None and existing_alias != name:
                        raise ValueError(
                            f"Metric alias '{alias}' already maps to canonical "
                            f"'{existing_alias}'; refusing to remap to '{name}'. "
                            "Pick a different alias."
                        )
                    cls._aliases[alias_key] = name

            logger.debug(f"Registered metric: {name}")
            return metric_cls

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> IMetric:
        """Get a metric instance by name or alias.

        Args:
            name: Metric name or alias
            **kwargs: Arguments passed to metric constructor

        Returns:
            Instantiated metric object

        Raises:
            KeyError: If metric not found
        """
        lookup_name = name.lower()

        # Check aliases first
        canonical = cls._aliases.get(lookup_name, lookup_name)

        if canonical not in cls._metrics:
            available = ", ".join(sorted(cls._metrics.keys()))
            raise KeyError(f"Unknown metric: '{name}'. Available: [{available}]")

        metric_cls = cls._metrics[canonical]
        # Filter kwargs to the constructor's signature so a generic call like
        # ``get(name, device=...)`` (the metrics computer does this for EVERY metric)
        # does not crash metrics whose __init__ doesn't declare ``device`` (e.g.
        # _QCRCBase(alpha, loss_bound)). Constructors with **kwargs receive all of
        # them. This is strictly more permissive: it only drops kwargs that would
        # otherwise raise TypeError.
        import inspect

        # Which class in the MRO actually OWNS __init__ decides what it accepts.
        # `inspect.signature` is only a static promise, and one base breaks it on
        # purpose: `torch.nn.Module.__init__` is declared `(*args, **kwargs)` but
        # raises "unexpected keyword argument" for ANY kwarg unless the subclass
        # sets `call_super_init` (it defaults False, for backward compatibility).
        #
        # So a metric that subclasses nn.Module without defining its own __init__
        # advertises a permissive signature and then refuses everything. The
        # VAR_KEYWORD branch below believed the advertisement and passed `device`
        # straight through, which is the one kwarg the computer always sends
        # (`computer.py`: `MetricsRegistry.get(spec.name, device=self.device)`).
        # Eight registered metrics died there -- the whole flow/perfusion battery
        # (velocity_rmse, vnr, net_flow_error, peak_velocity_error, cbf_rmse,
        # att_mae) plus negative_voxels and ndc_diffusion -- so an arm that asked
        # for them by name crashed in validation rather than being graded (#343,
        # and the "cannot be graded on its own physics" half of #340).
        try:
            init_owner = next((c for c in metric_cls.__mro__ if "__init__" in c.__dict__), object)
            if init_owner is object or init_owner is torch.nn.Module:
                # Inherited-only __init__: takes nothing beyond self.
                accepted = {}
            else:
                sig = inspect.signature(metric_cls)
                params = sig.parameters.values()
                if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
                    accepted = kwargs
                else:
                    names = {p.name for p in params}
                    accepted = {k: v for k, v in kwargs.items() if k in names}
        except (TypeError, ValueError):
            accepted = kwargs
        instance = metric_cls(**accepted)
        # ``device`` is advertised on this API but was silently not honoured, in
        # two ways, both stranding weights on the CPU so a CUDA input dies with
        # "Input type ... and weight type ... should be the same" (pitfall #15):
        #   * the signature filter above drops the kwarg for any ctor that does
        #     not declare it (e.g. CWSSIM, whose Gabor bank stayed on the CPU);
        #   * BaseMetric.__init__ only records ``self.device`` -- it cannot move
        #     buffers its subclasses register *after* super().__init__().
        # Placing the fully-built module is idempotent and covers both. An
        # invalid device raises here rather than degrading to CPU (pitfall #9).
        device = kwargs.get("device")
        if device is not None and isinstance(instance, torch.nn.Module):
            instance = instance.to(device)
        return instance

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered metric names."""
        return sorted(cls._metrics.keys())

    @classmethod
    def get_aliases(cls, name: str) -> list[str]:
        """Get all aliases for a metric."""
        canonical = cls._aliases.get(name.lower(), name.lower())
        return [alias for alias, target in cls._aliases.items() if target == canonical]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a metric is registered."""
        lookup = name.lower()
        return lookup in cls._metrics or lookup in cls._aliases

    @classmethod
    def _canonical(cls, name: str) -> str:
        """Resolve an alias to its canonical key (lower-cased)."""
        lookup = name.lower()
        return cls._aliases.get(lookup, lookup)

    @classmethod
    def needs(cls, name: str) -> tuple[str, ...]:
        """Return the ``MetricContext`` field names ``name`` consumes.

        Empty tuple for full-reference / no-context metrics or unknown names.
        Used by the §6 audit (``check_nr_metric_context_wired``) and the
        validation / sim2rank loops to assemble the right context.
        """
        return cls._needs.get(cls._canonical(name), ())

    @classmethod
    def needs_context(cls, name: str) -> bool:
        """True iff ``name`` requires a :class:`MetricContext` to compute."""
        return cls._requires_context.get(cls._canonical(name), False)

    @classmethod
    def workflows(cls, name: str) -> frozenset[Regime] | None:
        """Imaging regimes ``name`` is specific to, or ``None`` if agnostic.

        The public read side of the ``workflows=`` tag. It existed only in the
        private ``_workflow_tags`` dict, so every consumer that wanted to ask
        "does this metric apply to the regime I am measuring?" had no way to,
        and sim2rank simply did not ask: it scored ``cbf_rmse`` (perfusion) and
        ``velocity_rmse`` (4D-flow) on structural brain magnitude images, where
        both reduce to plain RMSE — bit-identical to the registered ``rmse``,
        not merely correlated with it.

        ``None`` means agnostic and applies everywhere; an unknown name is also
        ``None``, so this can never *narrow* an unrecognised metric out of a
        run (the loud direction — a silent exclusion would be worse than a
        wrong inclusion).
        """
        tags = cls._workflow_tags.get(cls._canonical(name))
        return tags.get("workflows") if tags else None

    @classmethod
    def applies_to_regime(cls, name: str, regime: Regime | None) -> bool:
        """Is ``name`` meaningful for ``regime``?

        ``True`` when the metric is regime-agnostic, when no regime is declared
        for the run, or when ``regime`` is among the metric's declared
        ``workflows``.
        """
        if regime is None:
            return True
        declared = cls.workflows(name)
        return declared is None or regime in declared

    @classmethod
    def requires_reference(cls, name: str) -> bool:
        """True iff ``name`` needs a reference ``target`` (full-reference)."""
        return cls._requires_reference.get(cls._canonical(name), True)

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing).

        ``_workflow_tags`` is cleared too. It was omitted, which left the
        maturity ledger reading tags for metrics that no longer existed: after a
        test called ``clear()``, ``metrics_tagged(regime)`` still returned True
        and the ledger reported coverage backed by nothing. The one consumer of
        this table is the anti-facade check, so a stale entry here is a facade in
        the guard itself.
        """
        for table in cls._TABLES:
            getattr(cls, table).clear()

    @classmethod
    def snapshot(cls) -> dict[str, dict[str, Any]]:
        """A copy of every registration table, to hand back to :meth:`restore`.

        The counterpart to :meth:`clear` for any test that needs a registry it
        can scribble on. Hand-rolling the save/restore is what went wrong: the
        obvious spelling saves the tables the test happens to be *about*, and
        the ones it is not about are the ones that stay broken -- silently, in
        the same process, for every test that runs afterwards.

        Copies each table one level deep, which is what the tables hold: name ->
        class, name -> tuple, name -> bool. ``_workflow_tags`` maps to a dict,
        so a mutation *inside* one tag entry would survive a restore; nothing
        mutates a tag in place, and a deep copy here would silently deep-copy
        the registered metric CLASSES, which is worse.
        """
        return {table: dict(getattr(cls, table)) for table in cls._TABLES}

    @classmethod
    def restore(cls, snapshot: dict[str, dict[str, Any]]) -> None:
        """Put every registration table back exactly as ``snapshot`` found it.

        REPLACES rather than merges. Updating in place leaves behind whatever
        the test registered after taking the snapshot, so a probe metric
        outlives the test that defined it and the next reader sees a registry
        that never existed.

        A snapshot missing a table **raises**: it means the snapshot predates a
        table being added, and restoring the rest would put the registry into a
        state that is half old and half empty -- exactly the failure this whole
        mechanism exists to remove. Absent is a state to report, not to infer.
        """
        missing = [table for table in cls._TABLES if table not in snapshot]
        if missing:
            raise KeyError(
                f"snapshot is missing {missing}; it does not cover the registry's "
                f"current tables {list(cls._TABLES)}. Take it with "
                "MetricsRegistry.snapshot() rather than by hand."
            )
        for table in cls._TABLES:
            live = getattr(cls, table)
            live.clear()
            live.update(snapshot[table])


# Convenience functions for cleaner API
def register_metric(
    name: str,
    aliases: list[str] | None = None,
    *,
    requires_reference: bool | None = None,
    requires_measurement_context: bool | None = None,
    needs: tuple[str, ...] | None = None,
    direction: str | None = None,
    workflows: frozenset[Regime] | None = None,
    tasks: frozenset[Task] | None = None,
) -> Any:
    """Decorator to register a metric.

    Example:
        @register_metric("ssim", aliases=["SSIM"])
        class SSIMMetric:
            ...

        @register_metric("ndcr", requires_reference=False,
                         needs=("y_kspace", "mask", "coil_maps"),
                         direction="lower")
        class NormalisedDataConsistencyResidual:
            ...
    """
    return MetricsRegistry.register(
        name,
        aliases,
        requires_reference=requires_reference,
        requires_measurement_context=requires_measurement_context,
        needs=needs,
        direction=direction,
        workflows=workflows,
        tasks=tasks,
    )


def get_metric(name: str, **kwargs: Any) -> IMetric:
    """Get a metric instance by name.

    Args:
        name: Metric name or alias
        **kwargs: Arguments for metric constructor

    Returns:
        Metric instance

    Example:
        psnr = get_metric("psnr")
        score = psnr(pred, target)
    """
    return MetricsRegistry.get(name, **kwargs)


def compute_metric(
    name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    **kwargs: Any,
) -> float:
    """Compute a metric by name.

    Args:
        name: Metric name or alias
        prediction: Model output
        target: Ground truth
        **kwargs: Arguments for metric

    Returns:
        Metric value
    """
    metric = MetricsRegistry.get(name, **kwargs)
    return metric(prediction, target)


def list_available() -> list[str]:
    """List all available metric names."""
    return MetricsRegistry.list_available()


__all__ = [
    "IMetric",
    "MetricsRegistry",
    "compute_metric",
    "get_metric",
    "list_available",
    "register_metric",
]
