"""Registry for config-declarable data transforms.

``data.processing.transforms`` is the YAML seam for "run this transform on
every subject". Before this module existed it was typed
``list[dict[str, Any]]`` with no validator, and its only consumer scanned the
list for the single literal string ``"graph_encoding"`` and ``break``\\ ed --
so every other entry was accepted by the schema and then silently discarded.
Arms named for a transform (``exp_slice_profile_sr``, ``exp_scas_8x_brain``,
the ``geomamba_ulf`` synthetic campaign) trained without it and reported
success. That is pitfall #16 (inert mechanism) sitting behind pitfall #15
(an advertised knob nothing reads).

Registry membership is now the validator, mirroring
``MetricsRegistry`` + ``check_metric_names_are_registered``: a name that is
not registered raises instead of vanishing, and every registered transform is
reachable from YAML.

Canonical home per CLAUDE.md non-negotiable #12 -- transforms live in
``src/spectramr/data/transforms/`` and nothing else may define one.

Registering a transform::

    @register_transform("phase_residual", produces=("phase_residual",))
    class PhaseResidualTransform(tio.Transform): ...

Declaring it::

    data:
      processing:
        transforms:
          - name: phase_residual
            kwargs: {kernel_size: 9}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TRANSFORM_REGISTRY",
    "RegisteredTransform",
    "build_transform",
    "get_transform",
    "list_transforms",
    "register_transform",
    "transforms_producing",
    "transforms_without_effect",
]


@dataclass(frozen=True)
class RegisteredTransform:
    """A transform that a YAML config may name.

    Attributes:
        name: The string a config declares.
        cls: The ``tio.Transform`` subclass to construct.
        produces: Subject keys this transform ADDS. This is the anti-facade
            payload: it lets an audit answer "does anything actually produce
            the key this strategy reads?" without importing every strategy.
            Several transforms in this package are the sole documented
            producer of a key a live strategy reads -- ``phase_residual`` for
            ``inverse_bloch_phase``, ``scout`` for the SCAS hypernet,
            ``foreground_mask`` for eight no-reference metrics -- and each of
            those chains was dead at link 0 because the transform had no way
            to be constructed.
        requires: Subject keys that must already exist for it to do work.
        modifies: Subject keys it rewrites in place without adding them. Kept
            apart from ``produces`` so an in-place edit of ``input`` cannot
            pass for a producer of a key some consumer reads.
    """

    name: str
    cls: type
    produces: tuple[str, ...] = ()
    requires: tuple[str, ...] = field(default=("input",))
    modifies: tuple[str, ...] = ()


TRANSFORM_REGISTRY: dict[str, RegisteredTransform] = {}


def register_transform(
    name: str,
    *,
    produces: tuple[str, ...] = (),
    requires: tuple[str, ...] = ("input",),
    modifies: tuple[str, ...] = (),
):
    """Class decorator registering a transform under ``name``.

    Args:
        name: Unique registry key; this is what YAML declares.
        produces: Subject keys the transform adds (see
            :class:`RegisteredTransform`).
        requires: Subject keys it reads.
        modifies: Subject keys it rewrites in place.

    Raises:
        ValueError: If ``name`` is already registered by a different class.
            A duplicate is a real defect here -- two classes answering to one
            YAML name means the resolved transform depends on import order
            (the failure mode recorded for import-time tables elsewhere in
            this repo).
    """

    def _decorator(cls: type) -> type:
        existing = TRANSFORM_REGISTRY.get(name)
        if existing is not None and existing.cls is not cls:
            raise ValueError(
                f"Transform name {name!r} is already registered to "
                f"{existing.cls.__module__}.{existing.cls.__qualname__}; "
                f"{cls.__module__}.{cls.__qualname__} cannot claim it. Pick a "
                "distinct name -- two classes under one name makes the "
                "resolved transform depend on import order."
            )
        TRANSFORM_REGISTRY[name] = RegisteredTransform(
            name=name,
            cls=cls,
            produces=tuple(produces),
            requires=tuple(requires),
            modifies=tuple(modifies),
        )
        return cls

    return _decorator


def list_transforms() -> tuple[str, ...]:
    """Registered names, sorted. Use for error messages and audit checks."""
    return tuple(sorted(TRANSFORM_REGISTRY))


def get_transform(name: str) -> RegisteredTransform:
    """Resolve ``name`` to its registry entry.

    Raises:
        KeyError: If the name is unregistered. The message lists the valid
            names, and calls out the dotted-path spelling explicitly because
            four committed arms use it (``name:
            spectramr.data.transforms.slice_profile.SliceProfileTransform``) and
            it never resolved -- there is no dotted-path importer in the data
            path, so those entries were silently dropped.
    """
    entry = TRANSFORM_REGISTRY.get(name)
    if entry is not None:
        return entry
    hint = ""
    if "." in name:
        hint = (
            " Dotted import paths are not supported and never were -- the data "
            "path has no dotted-path resolver, so such an entry was silently "
            "dropped. Use the registered short name instead."
        )
    raise KeyError(
        f"Unknown transform {name!r}. Registered: {list(list_transforms())}."
        f"{hint} Register new transforms with @register_transform in "
        "src/spectramr/data/transforms/, and import the module from "
        "src/spectramr/data/transforms/__init__.py so the decorator runs."
    )


def build_transform(name: str, /, **kwargs: Any):
    """Construct a registered transform.

    Unknown keyword arguments surface as the constructor's own ``TypeError``
    rather than being swallowed -- an unread transform kwarg is the same
    pitfall-#15 shape this registry exists to close.
    """
    entry = get_transform(name)
    try:
        return entry.cls(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Transform {name!r} ({entry.cls.__qualname__}) rejected the "
            f"declared kwargs {sorted(kwargs)}: {exc}"
        ) from exc


def transforms_producing(key: str) -> tuple[str, ...]:
    """Names of registered transforms that add ``key`` to a subject.

    The seam an audit uses to answer "is there any producer for the batch key
    this strategy reads?".
    """
    return tuple(sorted(n for n, e in TRANSFORM_REGISTRY.items() if key in e.produces))


def transforms_without_effect(
    table: dict[str, RegisteredTransform] | None = None,
) -> tuple[str, ...]:
    """Names that declare neither a key they add nor a key they rewrite.

    Such a transform cannot be shown to have fired. ``table`` is injectable so
    the check can be tested against a planted entry.
    """
    src = TRANSFORM_REGISTRY if table is None else table
    return tuple(sorted(n for n, e in src.items() if not (e.produces or e.modifies)))
