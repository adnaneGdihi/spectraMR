"""Loss Registry - decorator-based registry for loss functions.

This module provides a clean decorator-based API for registering and
accessing loss functions, following the same pattern as MetricsRegistry.

``@register_loss`` is the ONLY way in: ``create`` resolves against
``_custom_losses`` and raises on a miss. Earlier docstrings here described a
fallback to a ``UnifiedLossFunctionFactory`` for "built-in" losses -- that class
does not exist anywhere in the tree, and every loss including ``l1`` / ``l2``
arrives through the decorator. The promise of a second lookup path was the same
defect class as a knob with no reader (CLAUDE.md #15).

Example:
    >>> from spectramr.models.losses import create_loss, list_available
    >>> loss = create_loss("l1")
    >>> composite = create_composite_loss({"l1": {"weight": 0.5}, "ssim": {"weight": 0.5}})
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Protocol, runtime_checkable

import torch
import torch.nn as nn

from spectramr.config.schemas.enums import Regime, Task
from spectramr.models.capabilities import Domain, LossCapabilities

logger = logging.getLogger(__name__)

#: ``domain=`` value declaring a loss operates on ANY tensor, so it is legal
#: under every loss block. Deliberately NOT a ``Domain`` member -- see the note
#: on :attr:`LossCapabilities.domain_agnostic`.
AGNOSTIC_DOMAIN = "agnostic"

# Bridge the legacy free-string ``domain=`` plumbing (which predates the
# capability dataclasses) onto the closed ``Domain`` literal used by
# ``LossCapabilities``. ``AGNOSTIC_DOMAIN`` is carried on its own flag instead
# of being lost here; the non-signal domains below are handled likewise.
_LEGACY_DOMAIN_TO_LITERAL: dict[str, Domain] = {
    "image": "image",
    "kspace": "kspace",
    "complex": "complex_image",
    "complex_image": "complex_image",
    "latent": "latent",
}

#: ``domain=`` values that are deliberately NOT signal domains. These losses do
#: not grade the reconstruction signal at all -- they grade gradient waveforms,
#: Hamiltonians and Bloch simulations -- so no ``losses.<domain>_losses`` block
#: corresponds to them and the block check must not apply.
#:
#: They previously fell out of the map above as ``domain=None``, which the block
#: check reads as "unannotated, skip" -- the right OUTCOME reached by accident,
#: and indistinguishable from a typo. Four losses sat there:
#: ``hamiltonian_energy_conservation``, ``gradient_hardware_compliance``,
#: ``bloch_consistency`` and ``bloch_signal_synthesis_consistency``. Carrying
#: them on ``domain_agnostic`` instead makes the pass deliberate, and protects
#: them from the same tightening of the ``None`` branch that ``AGNOSTIC_DOMAIN``
#: was introduced to survive.
NON_SIGNAL_DOMAINS = frozenset({"physics", "bloch_synthesis"})

#: Every ``domain=`` string a registration may declare. Anything else raises at
#: import time rather than silently becoming "unannotated" -- the advertised set
#: and the honoured set had drifted apart (``register_loss``'s own docstring
#: offered ``'physics'`` while the map dropped it), which is CLAUDE.md #9/#15.
REGISTRABLE_DOMAINS = frozenset(_LEGACY_DOMAIN_TO_LITERAL) | {AGNOSTIC_DOMAIN} | NON_SIGNAL_DOMAINS


@runtime_checkable
class ILoss(Protocol):
    """Protocol that all losses should implement.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{ILoss} = \\text{abstract}"""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the loss value."""
        ...


class LossRegistry:
    """Decorator-based registry for loss functions.

    Decorator registration is the only entry point; there is no second
    lookup path behind it.

    Phase 3 Enhancement: Supports domain metadata for runtime validation.

    Usage:
        @LossRegistry.register("my_loss", aliases=["MyLoss"])
        class MyLossFunction(nn.Module):
            ...

        # Get loss by name
        loss = LossRegistry.create("my_loss")
    """

    _instance: ClassVar[LossRegistry | None] = None
    _custom_losses: ClassVar[dict[str, type[nn.Module]]] = {}
    _aliases: ClassVar[dict[str, str]] = {}
    # Phase 3: Domain metadata for each registered loss
    _loss_domains: ClassVar[dict[str, dict[str, Any]]] = {}
    # Cached-cascade WS-X: explicit LossCapabilities contracts (opt-in). The
    # retrieval SSOT is ``get_loss_capabilities`` below, which falls back to
    # synthesizing from ``_loss_domains`` so the legacy ``domain=`` plumbing
    # and the new ``capabilities=`` contract present a single surface.
    _loss_capabilities: ClassVar[dict[str, LossCapabilities]] = {}

    def __new__(cls) -> LossRegistry:
        """__new__.

        Returns:
            LossRegistry: Description.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(
        cls,
        name: str,
        aliases: list[str] | None = None,
        domain: str | None = None,
        compatible_with: list[str] | None = None,
        capabilities: LossCapabilities | None = None,
        workflows: frozenset[Regime] | None = None,
        tasks: frozenset[Task] | None = None,
    ) -> Any:
        """Decorator to register a custom loss class.

        Args:
            name: Canonical name for the loss (lowercase)
            aliases: Optional list of alternative names
            domain: One of :data:`REGISTRABLE_DOMAINS`. Anything else raises.
                   Signal domains ('image', 'kspace', 'complex',
                   'complex_image', 'latent') bind the loss to the matching
                   ``losses.<domain>_losses`` block; 'agnostic' and the
                   non-signal domains are exempt from that check.
            compatible_with: List of input domains this loss works with
            capabilities: Optional explicit :class:`LossCapabilities` contract
                (cached-cascade WS-X). When omitted, ``get_loss_capabilities``
                synthesizes one from ``domain``, so existing registrations need
                no change.

        Returns:
            Decorator function

        Example:
            @register_loss("my_loss", aliases=["custom"], domain="image")
            class MyLoss(nn.Module):
                ...
        """

        # Validate the vocabulary BEFORE the decorator body, so a bad
        # ``domain=`` fails at import rather than at the first block check.
        # Silently dropping an unknown string made a typo and a deliberate
        # non-signal domain indistinguishable: both became "unannotated", and a
        # loss that meant `domain="imagee"` skipped every check it was meant to
        # face. CLAUDE.md #9.
        if domain is not None and domain not in REGISTRABLE_DOMAINS:
            raise ValueError(
                f"Loss '{name}' declares domain={domain!r}, which is not a "
                f"registrable domain. Use one of {sorted(REGISTRABLE_DOMAINS)}. "
                "Signal domains place the loss under the matching "
                "`losses.<domain>_losses` block; 'agnostic' and the non-signal "
                f"domains {sorted(NON_SIGNAL_DOMAINS)} are exempt from that check."
            )

        def decorator(loss_cls: type[nn.Module]) -> type[nn.Module]:
            # Register in local custom dict.
            #
            # Per CLAUDE.md #9 (silent fallbacks forbidden) and TODO/audit/12_losses.md F4,
            # a duplicate registration with a *different* class is a hard
            # error: it would silently flip ``create_loss(name)`` between
            # two classes depending on import order. Re-registering the
            # same class (test reloads, idempotent imports) is fine.
            existing = cls._custom_losses.get(name)
            if existing is not None and existing is not loss_cls:
                raise ValueError(
                    f"Loss '{name}' already registered to "
                    f"{existing.__module__}.{existing.__qualname__}; "
                    f"refusing to overwrite with "
                    f"{loss_cls.__module__}.{loss_cls.__qualname__}. "
                    "Rename one of the two registrations."
                )

            # ``create`` consults the alias table before the canonical one, so a
            # name that is both an alias and another loss's canonical key resolves
            # to whichever registered first. Every name check runs before the first
            # write, so a refused registration leaves no partial entry behind.
            shadowed_by = cls._aliases.get(name)
            if shadowed_by is not None and shadowed_by != name:
                raise ValueError(
                    f"Loss name '{name}' is already an alias of '{shadowed_by}'; "
                    f"create('{name}') would resolve to '{shadowed_by}'. "
                    "Rename the loss or drop the alias."
                )
            alias_keys = [alias.lower() for alias in aliases or []]
            for alias_key in alias_keys:
                if alias_key != name and alias_key in cls._custom_losses:
                    raise ValueError(
                        f"Loss alias '{alias_key}' for '{name}' is already the canonical "
                        f"name of another loss; it would hijack create('{alias_key}'). "
                        "Pick a different alias."
                    )
                existing_alias = cls._aliases.get(alias_key)
                if existing_alias is not None and existing_alias != name:
                    raise ValueError(
                        f"Loss alias '{alias_key}' already maps to canonical "
                        f"'{existing_alias}'; refusing to remap to '{name}'. "
                        "Pick a different alias."
                    )

            cls._custom_losses[name] = loss_cls

            # Phase 3: Store domain metadata if provided. Workflow tags
            # (imaging-regime × task) are stored here too, so a loss may be
            # regime-tagged without carrying a signal ``domain``.
            if domain is not None or workflows is not None or tasks is not None:
                cls._loss_domains[name] = {
                    "domain": domain,
                    "compatible_with": compatible_with or [],
                    "class": loss_cls.__name__,
                    "workflows": workflows,
                    "tasks": tasks,
                }
                logger.debug(
                    f"Registered loss metadata: {name} (domain={domain}, workflows={workflows})"
                )

            # Cached-cascade WS-X: store an explicit capability contract if the
            # caller opted in. Retrieval still goes through
            # ``get_loss_capabilities`` (which synthesizes from ``domain`` when
            # this is absent), so the two surfaces stay reconciled.
            if capabilities is not None:
                cls._loss_capabilities[name] = capabilities

            for alias_key in alias_keys:
                cls._aliases[alias_key] = name

            logger.debug(f"Registered custom loss: {name}")
            return loss_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> nn.Module:
        """Create a loss instance by name.

        Resolves ``name`` through the alias table, then the registry. A miss
        raises with the available set -- there is no fallback lookup.

        Args:
            name: Loss name or alias
            **kwargs: Arguments passed to loss constructor

        Returns:
            Instantiated loss module

        Raises:
            ValueError: If loss not found
        """
        lookup_name = name.lower()

        # Check aliases
        canonical = cls._aliases.get(lookup_name, lookup_name)

        # Custom losses (primary path)
        if canonical in cls._custom_losses:
            loss_cls = cls._custom_losses[canonical]

            # [LOGGING] Trace creation
            logger.info(
                f"Creating Loss: {canonical} | Class: {loss_cls.__name__} | Params: {kwargs}"
            )

            try:
                return loss_cls(**kwargs)
            except Exception as e:
                # Wrap instantiation errors for better context
                raise RuntimeError(
                    f"Failed to instantiate loss '{canonical}' ({loss_cls.__name__}): {e}"
                ) from e

        # If not found
        available = sorted(cls._custom_losses)
        raise ValueError(
            f"Unknown loss: '{name}' (canonical: '{canonical}'). Available losses: {available}"
        )

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available loss names."""
        return sorted(cls._custom_losses)

    @classmethod
    def canonical_name(cls, name: str) -> str:
        """Resolve an alias to its canonical registry name (``mse`` -> ``l2``).

        The public seam for alias resolution. Unregistered names pass through
        unchanged, so callers that legitimately handle non-registry terms (a
        strategy-inline lambda such as ``pre_dc_kspace``) can canonicalise
        unconditionally. Reach for this rather than ``_aliases.get(n, n)``: code that
        keys on the RAW name treats ``mae`` and ``l1`` as two losses and double-counts
        them.
        """
        lookup_name = name.lower()
        return cls._aliases.get(lookup_name, lookup_name)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Return whether ``name`` resolves to a registered loss.

        Mirrors the resolution order used by :meth:`create` (alias →
        canonical → custom losses) but instantiates nothing. Used by the
        startup loss audit to validate configured loss names without
        paying the construction cost (e.g. the VGG backbone of
        ``perceptual``).
        """
        return cls.canonical_name(name) in cls._custom_losses

    @classmethod
    def get_loss_class(cls, name: str) -> type[nn.Module] | None:
        """The registered class for ``name``, or ``None`` if unregistered.

        Mirrors :meth:`create`'s resolution order (alias → canonical → custom
        losses) but constructs nothing, so a caller can ask what a loss's
        ``__init__`` accepts without paying the construction cost (the VGG
        backbone of ``perceptual``, say).

        The public seam for the class itself. Reaching into ``_custom_losses``
        keys on the RAW name, which treats ``mae`` and ``l1`` as two different
        losses — the double-count :meth:`canonical_name` exists to prevent.
        """
        return cls._custom_losses.get(cls.canonical_name(name))

    @classmethod
    def get_loss_aliases(cls, name: str) -> list[str]:
        """Get all aliases for a loss."""
        canonical = cls.canonical_name(name)
        return [alias for alias, canon in cls._aliases.items() if canon == canonical]

    @classmethod
    def get_loss_capabilities(cls, name: str) -> LossCapabilities | None:
        """Resolve the :class:`LossCapabilities` contract for a loss.

        This is the single retrieval surface for loss contracts (cached-cascade
        WS-X). Resolution order:

        1. An explicit ``capabilities=`` passed at registration.
        2. Otherwise, synthesize ``LossCapabilities`` from the legacy
           ``domain=`` metadata, mapping the free string onto the closed
           ``Domain`` literal. ``domain="agnostic"`` becomes
           ``domain_agnostic=True`` (a positive "legal in every block" claim);
           strings with no ``Domain`` equivalent, like "physics", yield
           ``domain=None`` rather than a guess.
        3. ``None`` if the loss is registered but carries no domain metadata.

        Returns ``None`` for an unknown loss name as well (callers treat
        ``None`` as "unannotated, skip the check", matching the
        ``ModelCapabilities`` convention).

        The agnostic flag must survive this adaptation. Before it existed,
        "agnostic" simply failed to map and fell out as ``domain=None`` -- so
        the seven genuinely-generic losses passed the block check for the same
        reason an un-annotated one did, and any tightening of the ``None``
        branch would have started rejecting them.
        """
        canonical = cls._aliases.get(name.lower(), name.lower())
        explicit = cls._loss_capabilities.get(canonical)
        if explicit is not None:
            return explicit
        meta = cls._loss_domains.get(canonical)
        if meta is None:
            return None
        raw = meta.get("domain") or ""
        literal = _LEGACY_DOMAIN_TO_LITERAL.get(raw)
        return LossCapabilities(
            domain=literal,
            # A non-signal domain ('physics', 'bloch_synthesis') is exempt for
            # the same reason 'agnostic' is: there is no block it belongs under.
            # Stating it here rather than letting it fall out of the map as
            # `None` is what separates it from an unannotated loss -- and from a
            # typo, which now cannot reach this point at all.
            domain_agnostic=(raw == AGNOSTIC_DOMAIN or raw in NON_SIGNAL_DOMAINS),
        )

    @classmethod
    def get_loss_info(cls, name: str) -> dict[str, Any]:
        """Get metadata about a loss, resolving aliases the way :meth:`create` does."""
        loss_cls = cls.get_loss_class(name)
        if loss_cls is None:
            raise KeyError(f"Unknown loss: {name}")
        return {
            "name": name,
            "description": loss_cls.__doc__ or "",
            "aliases": cls.get_loss_aliases(name),
            "type": "custom",
        }


# Convenience functions
def register_loss(
    name: str,
    aliases: list[str] | None = None,
    domain: str | None = None,
    compatible_with: list[str] | None = None,
    capabilities: LossCapabilities | None = None,
    workflows: frozenset[Regime] | None = None,
    tasks: frozenset[Task] | None = None,
) -> Any:
    """Decorator to register a loss.

    Args:
        name: Canonical loss name (lowercase, snake_case).
        aliases: Optional alternative names.
        domain: One of :data:`REGISTRABLE_DOMAINS`; anything else raises at
            import. When it is a SIGNAL domain the LossBuilder validates that
            this loss is placed under the matching ``losses.<domain>_losses``
            block. 'agnostic' and :data:`NON_SIGNAL_DOMAINS` ('physics',
            'bloch_synthesis') carry an exemption instead -- they name no block,
            so there is nothing to validate against.
        compatible_with: Domains this loss can also operate on (e.g., a
            kspace loss that also accepts image input via internal FFT).

    Example:
        @register_loss("custom_ssim", domain="image")
        class CustomSSIMloss(nn.Module):
            ...
    """
    return LossRegistry.register(
        name,
        aliases=aliases,
        domain=domain,
        compatible_with=compatible_with,
        capabilities=capabilities,
        workflows=workflows,
        tasks=tasks,
    )


def compatible_domains(name: str) -> frozenset[str]:
    """Extra ``Domain`` values a loss declares it also accepts.

    ``compatible_with=`` is a free ``list[str]`` and has collected two different
    vocabularies. Six of the fourteen populated entries hold strategy or
    workflow names rather than domains -- ``['bloch_synth']``,
    ``['field_cocycle', 'cross_field_translation']``,
    ``[..., 'reconstruction', 'self_supervised']`` -- so the escape hatch they
    were meant to open has never once fired: the block check compares them
    against a domain, and a strategy name can never equal one.

    This filters to the entries that ARE domains (mapping the legacy spelling
    through, so ``complex`` reaches ``complex_image``). The non-domain entries
    are left in place rather than deleted: removing an exemption can only turn
    passes into failures, and each one needs its owner to say whether the loss
    meant a domain or meant a strategy. ``test_registry.py`` pins the polluted
    set so it can shrink but not grow.
    """
    canonical = LossRegistry._aliases.get(name.lower(), name.lower())
    meta = LossRegistry._loss_domains.get(canonical)
    if meta is None:
        return frozenset()
    raw = meta.get("compatible_with") or []
    return frozenset(
        mapped for entry in raw if (mapped := _LEGACY_DOMAIN_TO_LITERAL.get(entry)) is not None
    )


def get_loss_capabilities(name: str) -> LossCapabilities | None:
    """Module-level accessor for a loss's :class:`LossCapabilities` contract.

    Thin wrapper over :meth:`LossRegistry.get_loss_capabilities` -- the single
    retrieval surface used by the context resolver (cached-cascade WS-X).
    """
    return LossRegistry.get_loss_capabilities(name)


def create_loss(name: str, **kwargs: Any) -> nn.Module:
    """Create a loss instance by name.

    Args:
        name: Loss name or alias
        **kwargs: Arguments for loss constructor

    Returns:
        Loss module

    Example:
        loss = create_loss("l1")
        loss = create_loss("ssim", window_size=11)
    """
    return LossRegistry.create(name, **kwargs)


def create_composite_loss(config: dict[str, dict[str, Any]]) -> nn.Module:
    """Create a weighted composite loss from config using Phase 5 infrastructure.

    Creates a ComposedLoss with full metrics support using the registry pattern.
    Each loss component can have its own configuration including metrics enablement.

    Args:
        config: Dict mapping loss names to their config (must include "weight" key)

    Returns:
        ComposedLoss module with metrics support

    Example:
        loss = create_composite_loss({
            "l1": {"weight": 10.0, "compute_metrics": True},
            "ssim": {"weight": 0.5, "compute_metrics": True},
            "perceptual": {"weight": 0.1}
        })

        # Use with metrics
        total_loss, metrics = loss.forward_with_metrics(pred, target)
    """
    from spectramr.models.losses.composed_loss import ComposedLoss, WeightedLoss

    weighted_losses = []

    for name, cfg in config.items():
        cfg = cfg.copy()  # Don't mutate input
        weight = cfg.pop("weight", 1.0)
        loss_fn = create_loss(name, **cfg)
        weighted_losses.append(WeightedLoss(name=name, loss_fn=loss_fn, weight=weight))

    return ComposedLoss(weighted_losses)


def list_available() -> list[str]:
    """List all available loss names."""
    return LossRegistry.list_available()


def is_registered(name: str) -> bool:
    """Return whether ``name`` resolves to a registered loss (no construction)."""
    return LossRegistry.is_registered(name)


__all__ = [
    "AGNOSTIC_DOMAIN",
    "ILoss",
    "LossRegistry",
    "compatible_domains",
    "create_composite_loss",
    "create_loss",
    "get_loss_capabilities",
    "is_registered",
    "list_available",
    "register_loss",
]
