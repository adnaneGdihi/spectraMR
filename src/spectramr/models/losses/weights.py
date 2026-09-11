"""Loss-weight resolution — the SSOT.

Before this module there were **eight** weight resolvers (two in ``strategies/``,
five across ``losses/computers/``, one in ``loss_folding``), each with its own
precedence and its own magic default table. The same YAML resolved to a different
effective weight depending on which code path a strategy happened to take, and an
undeclared loss silently materialised at 1.0 (pitfall #9: a silent fallback).

This module replaces all of them. Two objects:

* :class:`LossWeightTable` — every declared weight, resolved **once** from the frozen
  config at build time. Not per step: the old ``BaseLossComputer._get_loss_weight``
  called ``config.losses.model_dump()`` per loss per step (``performance.md``).
* :func:`resolve_loss_weight` — a pure lookup on that table.

Declaration surfaces and the conflict rule
------------------------------------------
A weight may be declared on exactly ONE of two surfaces:

1. ``losses.<section>.lambda_<name>`` — the explicit lambda.
2. ``losses.{image,kspace,complex}_losses[].weight`` — the declarative list entry.

Both declaring the same (canonical) loss is legal only when they AGREE. Disagreement
raises: a "warning + pick one" would be pitfall #10 wearing a raise costume. Names are
canonicalised through :class:`~spectramr.models.losses.registry.LossRegistry`, so
``image_losses: [{name: mse}]`` and ``lambda_l2`` are recognised as the same knob.

"Declared" means the author WROTE it (``model_fields_set``), never a schema default —
``lambda_hfen`` defaults to 0.0, and treating that as a declaration is exactly how a
declared ``hfen weight: 0.1`` ended up silently computing at zero.

A loss declared nowhere RAISES (pitfall #9/#15). ``enabled: false`` / ``weight: 0`` is a
declaration — it resolves to 0.0 and never raises.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, get_args

from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS
from spectramr.domain.exceptions import ConfigurationError
from spectramr.models.losses.registry import LossRegistry

if TYPE_CHECKING:  # pragma: no cover
    from spectramr.config.schemas.loss import LossConfigSchema

#: Bumped when the resolution semantics change. Stamped into run provenance (#15).
WEIGHT_SEMANTICS_VERSION = "2"

#: Every sub-block of ``losses:`` that may carry ``lambda_<name>`` fields.
#: ``pinn`` is included deliberately: no legacy resolver scanned it, so 48 arms
#: declaring ``losses.pinn.lambda_pde`` were silently ignored.
#:
#: ``"adversarial"`` was removed (#1925). It named no ``LossConfigSchema`` block --
#: the adversarial lambda is ``losses.gan.lambda_adv``, which is why
#: :data:`NAME_ALIASES` below exists -- so every walk over this tuple resolved it to
#: ``None`` and skipped it. Nothing changed when it went: the entry contributed no
#: schema default and no read path. :func:`_lambda_sections` now **raises** on such a
#: name rather than skipping it, so the constant cannot silently drift again.
LAMBDA_SECTIONS: tuple[str, ...] = (
    "reconstruction",
    "latent",
    "diffusion",
    "gan",
    "physics",
    "registration",
    "ssl",
    "spatial",
    "evidential",
    "pinn",
)

#: The declarative loss lists (v6.0+). **Derived** from the schema's SSOT, never
#: hand-written: this was a three-element literal that omitted ``latent_losses``,
#: while :data:`LAMBDA_SECTIONS` above *does* carry ``latent``. One resolver
#: reading two surfaces where only one of them knows about latents made
#: :func:`build_loss_weight_table` silently wrong in two directions --
#: ``latent_losses: [{name: kl, weight: 0.37}]`` resolved to **0.0**, and a
#: 0.5-vs-0.37 contradiction across the two surfaces raised **nothing** where the
#: byte-identical shape on ``image_losses`` raises ``ConfigurationError`` (#1924).
#:
#: Order changed with the derivation (the dict is keyed kspace/image/complex/latent).
#: That is inert by construction: :func:`_declared_list_entries` accumulates every
#: source under ``setdefault(...).append(...)`` rather than taking a first match, so
#: no resolved weight depends on the order -- only the ``"+"``-joined ``source``
#: string does.
LOSS_LISTS: tuple[str, ...] = tuple(LOSS_LIST_DOMAINS)

#: Spellings of a loss that the registry does not know, because the schema field and the
#: component the computers stack are named differently. Mirrors the remap
#: ``LossConfigSchema.get_enabled_losses()`` already applies when deciding what to BUILD;
#: without it the same knob is invisible when deciding what to WEIGH.
#:
#: This is not cosmetic. Every legacy resolver looked up ``lambda_adversarial``, but the
#: field is ``losses.gan.lambda_adv`` — so the adversarial weight was **never read from
#: config**, and every GAN arm silently ran it at the fallback 1.0. 23 arms happened to
#: declare 1.0 anyway; one declares 0.1 and has been training 10x hot.
#:
#: Applied to BOTH directions (the field ``lambda_adv`` and a computer asking for
#: ``adversarial``), so the two always land on the same entry.
NAME_ALIASES: dict[str, str] = {
    "adv": "adversarial",
    "gp": "gradient_penalty",
    "commit": "commitment",
    # UnifiedGANLossComputer stacks the R1 term as `r1_penalty`; the field is `lambda_r1`.
    "r1_penalty": "r1",
}

#: The historical hardcoded warm-up set, duplicated in ``strategies/base.py`` and
#: ``unified_diffusion_reconstruction.py``. Kept as the DEFAULT for
#: ``losses.reconstruction.warmup_losses`` so behaviour is unchanged, but it is now
#: named in one place and stamped into provenance instead of being implied.
LEGACY_WARMUP_LOSSES: frozenset[str] = frozenset(
    {
        "complex_spatial_gradient",
        "rician_consistency",
        "background_suppression",
        "perceptual",
        "adversarial",
        "l1",
    }
)

DEFAULT_WARMUP_ITERATIONS = 1000


def canonical_loss_name(name: str) -> str:
    """Resolve a loss alias to its canonical registry name (``mse`` -> ``l2``).

    Unregistered names pass through unchanged: a strategy-inline term such as
    ``pre_dc_kspace`` has a lambda but no registry entry, and that is legal.

    A ``lambda_``-prefixed query is the FIELD spelling of the same knob (some callers
    thread the schema field name straight through, e.g. the disentangled computer's
    ``{"hist": "lambda_hist"}`` map), so the prefix is stripped rather than treated as a
    distinct — and therefore undeclared — loss.
    """
    if name.startswith("lambda_"):
        name = name[len("lambda_") :]
    name = NAME_ALIASES.get(name.lower(), name)
    return LossRegistry.canonical_name(name)


def _loss_name_for_field(field: str) -> str:
    """``lambda_adv`` -> ``adversarial``; ``lambda_mse`` -> ``l2``. One place."""
    return canonical_loss_name(field)


@lru_cache(maxsize=1)
def _schema_defaults() -> dict[str, tuple[str, float]]:
    """``{canonical loss -> (section, schema default)}`` for every ``lambda_<n>`` field.

    This is the ONLY sanctioned fallback, and it is not a magic table: the value is the
    Pydantic field default, declared once in ``config/schemas/loss.py`` and visible to
    the reader of the schema. Callers routinely *probe* a term's weight and gate on
    ``> 0`` ("is this configured?"), so an undeclared term with a schema field must
    answer with that default (almost always 0.0 = not requested), not explode.

    A term with NO ``lambda_<n>`` field anywhere is the dangerous case: those are what
    fell through to the three disagreeing hardcoded tables (an undeclared ``adversarial``
    resolved to 1.0 or 0.01; ``kl_divergence`` to 1.0 or 1e-4). Those RAISE.
    """
    defaults: dict[str, tuple[str, float]] = {}
    for section_name, section_cls in _lambda_sections():
        for field_name, spec in section_cls.model_fields.items():
            if not field_name.startswith("lambda_"):
                continue
            default = spec.default
            if not isinstance(default, (int, float)):
                continue
            name = _loss_name_for_field(field_name)
            # First section wins; a clash is caught as an ambiguity at declaration time.
            defaults.setdefault(name, (section_name, float(default)))
    return defaults


def _section_type(field: Any) -> Any:
    """The concrete ``*LossesConfig`` behind an ``X | None`` annotation."""
    annotation = getattr(field, "annotation", None)
    for candidate in get_args(annotation) or (annotation,):
        if isinstance(candidate, type) and hasattr(candidate, "model_fields"):
            return candidate
    return None


def _lambda_sections() -> Iterator[tuple[str, Any]]:
    """``(name, model)`` for every entry in :data:`LAMBDA_SECTIONS`. One walk.

    Raises when a name is not a ``LossConfigSchema`` block, or when its annotation
    holds no model. That is non-negotiable 3, and it is not hypothetical: this walk
    used to carry two ``continue`` statements, and ``"adversarial"`` sat in
    :data:`LAMBDA_SECTIONS` naming no field at all -- contributing silently nothing to
    the schema defaults for the whole life of the constant. A section that names no
    block is a typo in the constant, not a configuration state, so it must be loud.

    Note the annotation is ``X | None`` for every section, so ``_section_type`` is
    load-bearing: reading ``.model_fields`` off the raw annotation raises
    ``AttributeError`` on a ``types.UnionType`` and a permissive ``getattr`` walk
    silently enumerates **zero** fields.
    """
    from spectramr.config.schemas.loss import LossConfigSchema

    for section_name in LAMBDA_SECTIONS:
        field = LossConfigSchema.model_fields.get(section_name)
        section_cls = _section_type(field) if field is not None else None
        if section_cls is None:
            raise ConfigurationError(
                f"LAMBDA_SECTIONS names {section_name!r}, which is not a block on "
                f"LossConfigSchema (or whose annotation holds no model). Every walk "
                f"over LAMBDA_SECTIONS would skip it, so its lambdas would resolve to "
                f"no schema default and appear in no read-path export. Remove the name "
                f"or add the block; do not let the walk swallow it."
            )
        yield section_name, section_cls


#: What :func:`accessor_read_paths` stamps as the reader of a ``lambda_*`` field.
_LAMBDA_READER = (
    "models.losses.weights.build_loss_weight_table -> _declared_lambdas: "
    "getattr(section, field) over model_fields_set (runtime-built name)"
)
_LIST_READER = (
    "models.losses.weights.build_loss_weight_table -> _declared_list_entries: "
    "getattr(loss_config, list_name) over LOSS_LISTS (runtime-built name)"
)
_TABLE_READER = (
    "models.losses.weights.build_loss_weight_table: getattr(reconstruction, ...) "
    "for the table-level warm-up knobs"
)


def accessor_read_paths() -> dict[str, str]:
    """Config paths :func:`build_loss_weight_table` reads, keyed by **full dotted path**.

    The reachability index matches a read by finding its field name as a *token* in the
    source (``key_reachability``). This accessor names none: it builds every field name
    at runtime from :data:`LAMBDA_SECTIONS` and ``model_fields_set``, so 52 of the 108
    paths below carry **no token anywhere in the tree** and the index reports
    ``NO_READ_FOUND`` for them -- the shape its own docstring warns "a human must read
    the would-be consumer before acting on". This function is that human's answer,
    declared beside the reader (#1925).

    Keyed by the full path, never the leaf. ``losses.physics.lambda_bloch_residual`` is
    read here; ``training.multi.stages.stage_config.loss.physics.lambda_bloch_residual``
    is the same leaf under a prefix this accessor never receives --
    ``config_health_checker`` passes the **root** ``config.losses`` -- and must stay
    unread. A leaf-keyed map would call 54 such stage-scoped paths consumed.

    Returns:
        ``{dotted path: prose naming the reader}``. The value is documentation, not a
        contract; callers branch on membership only.

    The set is derived from the same two constants the readers iterate, so it cannot
    drift from them by construction. It *can* drift from the two hand-written
    table-level fields, which is what
    ``test_weights.py::test_the_static_export_equals_the_executed_read_set`` exists to
    catch: it moves every field of the schema and asserts the set that moves the table
    is exactly this one.
    """
    paths = {
        f"losses.{section_name}.{field_name}": _LAMBDA_READER
        for section_name, section_cls in _lambda_sections()
        for field_name in section_cls.model_fields
        if field_name.startswith("lambda_")
    }
    paths.update({f"losses.{list_name}": _LIST_READER for list_name in LOSS_LISTS})
    # Read off ``reconstruction`` by literal name in ``build_loss_weight_table``; see
    # the two-oracle test above for why these two are safe to name here.
    paths["losses.reconstruction.warmup_iterations"] = _TABLE_READER
    paths["losses.reconstruction.warmup_losses"] = _TABLE_READER
    return paths


@dataclass(frozen=True, slots=True)
class LossWeightSpec:
    """One resolved loss weight, with the provenance of where it came from."""

    name: str
    """Canonical registry name (aliases resolved)."""

    weight: float
    enabled: bool
    source: str
    """e.g. ``losses.reconstruction.lambda_hfen`` or ``losses.image_losses[hfen].weight``."""

    warmup_gated: bool
    """Precomputed ``name in warmup_losses`` — keeps the hot path allocation-free."""


class LossWeightTable(Mapping[str, LossWeightSpec]):
    """Immutable, fully-resolved weight table. Built once from a frozen config."""

    __slots__ = ("_specs", "warmup_iterations", "warmup_losses")

    def __init__(
        self,
        specs: dict[str, LossWeightSpec],
        *,
        warmup_iterations: int,
        warmup_losses: frozenset[str],
    ) -> None:
        self._specs = specs
        self.warmup_iterations = warmup_iterations
        self.warmup_losses = warmup_losses

    def __getitem__(self, key: str) -> LossWeightSpec:
        return self._specs[canonical_loss_name(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def weight(self, name: str, *, iteration: int = 0) -> float:
        """Static weight for ``name``. Raises when the loss is declared nowhere."""
        return resolve_loss_weight(self, name, iteration=iteration)

    def provenance(self) -> dict[str, Any]:
        """The stamp written into the run record (#15: every knob read AND stamped)."""
        return {
            "semantics_version": WEIGHT_SEMANTICS_VERSION,
            "warmup_iterations": self.warmup_iterations,
            "warmup_losses": sorted(self.warmup_losses),
            "resolved": {
                spec.name: {
                    "weight": spec.weight,
                    "enabled": spec.enabled,
                    "source": spec.source,
                    "warmup_gated": spec.warmup_gated,
                }
                for spec in self._specs.values()
            },
        }


def _declared_lambdas(loss_config: Any) -> dict[str, list[tuple[str, float]]]:
    """Author-written ``lambda_<name>`` fields, keyed by canonical loss name.

    Only fields in ``model_fields_set`` count — a schema default is not a declaration.
    """
    found: dict[str, list[tuple[str, float]]] = {}
    for section_name in LAMBDA_SECTIONS:
        section = getattr(loss_config, section_name, None)
        if section is None:
            continue
        for field in _written_lambda_fields(section):
            value = getattr(section, field, None)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            name = _loss_name_for_field(field)
            source = f"losses.{section_name}.{field}"
            found.setdefault(name, []).append((source, float(value)))
    return found


def _written_lambda_fields(section: Any) -> list[str]:
    """The ``lambda_*`` fields the AUTHOR set on ``section``.

    For a Pydantic model that is ``model_fields_set`` — the whole point, since a schema
    default is not a declaration. Config *doubles* (``SimpleNamespace``, a test's
    ``MagicMock``) have no such attribute; for those, every ``lambda_*`` present in the
    instance dict counts as written, since a double only carries what a test put there.
    Silently treating a double as "declares nothing" would zero its weights.
    """
    written = getattr(section, "model_fields_set", None)
    if isinstance(written, (set, frozenset, list, tuple)):
        return [f for f in written if isinstance(f, str) and f.startswith("lambda_")]
    instance_dict = getattr(section, "__dict__", None)
    if isinstance(instance_dict, dict):
        return [k for k in instance_dict if isinstance(k, str) and k.startswith("lambda_")]
    return []


def _declared_list_entries(
    loss_config: Any,
) -> dict[str, list[tuple[str, float, bool]]]:
    """Declarative list entries, keyed by canonical loss name."""
    found: dict[str, list[tuple[str, float, bool]]] = {}
    for list_name in LOSS_LISTS:
        for entry in getattr(loss_config, list_name, None) or []:
            raw = getattr(entry, "name", None)
            if raw is None and isinstance(entry, dict):
                raw = entry.get("name")
            if raw is None:
                continue
            weight = getattr(entry, "weight", None)
            if weight is None and isinstance(entry, dict):
                weight = entry.get("weight")
            enabled = getattr(entry, "enabled", None)
            if enabled is None and isinstance(entry, dict):
                enabled = entry.get("enabled", True)
            name = canonical_loss_name(str(raw))
            source = f"losses.{list_name}[{raw}].weight"
            found.setdefault(name, []).append(
                (source, 1.0 if weight is None else float(weight), bool(enabled))
            )
    return found


def _agree(values: list[float]) -> bool:
    first = values[0]
    return all(math.isclose(v, first, rel_tol=1e-9, abs_tol=1e-12) for v in values)


def build_loss_weight_table(
    loss_config: LossConfigSchema | None,
) -> LossWeightTable:
    """Resolve every declared loss weight exactly once, at config-freeze time.

    Raises :class:`ConfigurationError` — listing every offender at once, so one run
    surfaces the whole problem — when a canonical loss is declared more than once at
    DIFFERENT weights, whether across sections (``lambda_l1`` in both ``reconstruction``
    and ``latent``), across surfaces (``lambda_hfen`` vs ``image_losses[hfen].weight``),
    or across aliases (``lambda_l2`` vs ``image_losses[mse]``).

    Never raises for an *undeclared* name — that is a lookup-time error (the caller may
    legitimately probe for a term it does not use).
    """
    if loss_config is None:
        return LossWeightTable(
            {},
            warmup_iterations=DEFAULT_WARMUP_ITERATIONS,
            warmup_losses=frozenset(),
        )

    lambdas = _declared_lambdas(loss_config)
    entries = _declared_list_entries(loss_config)

    recon = getattr(loss_config, "reconstruction", None)
    raw_warmup = getattr(recon, "warmup_iterations", None) if recon is not None else None
    # A config double may hand back a non-int here; fall back rather than crash.
    warmup_iterations = (
        int(raw_warmup)
        if isinstance(raw_warmup, int) and not isinstance(raw_warmup, bool)
        else DEFAULT_WARMUP_ITERATIONS
    )
    configured_warmup = getattr(recon, "warmup_losses", None) if recon else None
    if not isinstance(configured_warmup, (list, tuple, set, frozenset)):
        configured_warmup = None
    warmup_losses = frozenset(
        canonical_loss_name(n)
        for n in (LEGACY_WARMUP_LOSSES if configured_warmup is None else configured_warmup)
    )

    conflicts: list[str] = []
    specs: dict[str, LossWeightSpec] = {}

    for name in sorted(set(lambdas) | set(entries)):
        lam = lambdas.get(name, [])
        lst = entries.get(name, [])
        declarations = [(src, val) for src, val in lam] + [(src, val) for src, val, _ in lst]
        values = [val for _, val in declarations]

        if len(values) > 1 and not _agree(values):
            detail = ", ".join(f"{src} = {val}" for src, val in declarations)
            conflicts.append(
                f"  '{name}' is declared {len(values)}x at DIFFERENT weights: {detail}"
            )
            continue

        # `enabled: false` on ANY list entry disables the term; a lambda-only term is
        # enabled by declaration. `weight: 0` is a declaration too (resolves to 0.0).
        enabled = all(en for _, _, en in lst) if lst else True
        weight = values[0]
        source = (
            "+".join(src for src, _ in declarations)
            if len(declarations) > 1
            else declarations[0][0]
        )
        specs[name] = LossWeightSpec(
            name=name,
            weight=0.0 if not enabled else weight,
            enabled=enabled,
            source=source,
            warmup_gated=name in warmup_losses,
        )

    if conflicts:
        raise ConfigurationError(
            "Conflicting loss-weight declarations (a loss may be declared on the "
            "lambda surface OR the declarative-list surface, not both at different "
            "values):\n"
            + "\n".join(conflicts)
            + "\n\nFix: keep ONE declaration per loss, or make the two agree. "
            "Note that aliases collapse (`mse` and `l2` are the same loss)."
        )

    return LossWeightTable(
        specs,
        warmup_iterations=warmup_iterations,
        warmup_losses=warmup_losses,
    )


def is_loss_configured(table: LossWeightTable, name: str) -> bool:
    """Is ``name`` REQUESTED by this config? A configuration question, not a temporal one.

    ``resolve_loss_weight`` answers "what is this loss's weight RIGHT NOW", which
    folds in the warm-up gate: a warm-up-gated term resolves to 0.0 for the first
    ``warmup_iterations`` steps. That is correct for scaling a loss, and wrong for
    deciding whether to BUILD one.

    ``UnifiedReconstructionLossComputer._initialize_losses`` asked the weight
    question at construction time, where there is no iteration, so ``l1`` resolved
    to 0.0 (it is in ``LEGACY_WARMUP_LOSSES``) and read as "not requested" -- the
    reconstruction loss was never constructed AT ALL, permanently, not merely
    during warm-up. A gate meant to defer a term for 1000 steps deleted it.

    So: declared, enabled, and carrying a non-zero static weight -- with the
    warm-up gate deliberately NOT consulted, because "not yet" is not "no".
    """
    spec = table.get(canonical_loss_name(name))
    if spec is None:
        fallback = _schema_defaults().get(canonical_loss_name(name))
        return bool(fallback and fallback[1] > 0)
    return bool(spec.enabled and spec.weight > 0)


def resolve_loss_weight(
    table: LossWeightTable,
    name: str,
    *,
    scheduled: Mapping[str, float] | None = None,
    iteration: int = 0,
) -> float:
    """The hot path: a pure lookup. No config walk, no ``model_dump``, no allocation.

    Precedence:
      1. ``scheduled`` — the ``loss_schedule`` curriculum override, which supersedes
         BOTH the warm-up gate and the static weight (a schedule rule must be able to
         enable a spatial term before ``warmup_iterations``).
      2. ``enabled: false`` -> 0.0
      3. the warm-up gate -> 0.0 while ``iteration < warmup_iterations``
      4. the declared static weight

    Raises:
        ConfigurationError: ``name`` is declared nowhere. Never silently 1.0.
    """
    canonical = canonical_loss_name(name)

    if scheduled:
        override = scheduled.get(canonical, scheduled.get(name))
        if override is not None:
            return float(override)

    spec = table.get(canonical)
    if spec is None:
        # Undeclared. Fall back to the `lambda_<n>` SCHEMA default if one exists — that
        # is a single, visible, auditable value (almost always 0.0 = "not requested"),
        # and callers legitimately probe a term's weight before deciding to compute it.
        fallback = _schema_defaults().get(canonical)
        if fallback is not None:
            _section, default = fallback
            if canonical in table.warmup_losses and iteration < table.warmup_iterations:
                return 0.0
            return default

        # No declaration AND no schema field: this is precisely the class that used to
        # fall through to the three disagreeing hardcoded tables — an undeclared
        # `adversarial` resolved to 1.0 or 0.01 (100x apart), `kl_divergence` to 1.0 or
        # 1e-4 (10,000x apart), purely by which computer the strategy happened to build.
        # Refuse to guess (pitfall #9/#15).
        raise ConfigurationError(
            f"Loss '{name}' is active but its weight is declared nowhere, and no "
            f"`lambda_{canonical}` field exists in any losses section to default from. "
            f"Refusing to invent a weight (CLAUDE.md #9/#15).\n"
            f"Fix: declare it explicitly — add `- {{name: {canonical}, weight: <w>}}` to "
            f"`losses.image_losses`, or add a `lambda_{canonical}` field to the schema."
        )

    if not spec.enabled:
        return 0.0
    if spec.warmup_gated and iteration < table.warmup_iterations:
        return 0.0
    return spec.weight


def lambda_schema_default(section: str, field: str) -> float | None:
    """What ``losses.<section>.<field>`` reads once it is no longer written.

    ``None`` when the section is absent, the field is absent, or the default is
    not numeric — three states a caller must report rather than read as 0.0 (#9).
    """
    from spectramr.config.schemas.loss import LossConfigSchema

    info = LossConfigSchema.model_fields.get(section)
    if info is None:
        return None
    spec = getattr(_section_type(info), "model_fields", {}).get(field)
    default = getattr(spec, "default", None)
    return float(default) if isinstance(default, (int, float)) else None


def _materialised_values(
    losses: Any, *, dropping: tuple[str, str] | None = None
) -> dict[str, set[float]]:
    """``{canonical loss -> every value a MATERIALISED load stamps for it}``.

    The cluster loads a config with every schema default written out as an
    explicit value, so a loss with a ``lambda_`` field in two sections is
    declared twice whatever the YAML says. ``dropping`` names a
    ``(section, field)`` to read at its schema default instead of its configured
    value — what an edit that deleted that field would leave behind.
    """
    values: dict[str, set[float]] = {}
    for section_name in LAMBDA_SECTIONS:
        section = getattr(losses, section_name, None)
        if section is None or not hasattr(section, "model_dump"):
            continue
        for field, value in section.model_dump().items():
            if not field.startswith("lambda_") or not isinstance(value, (int, float)):
                continue
            if dropping == (section_name, field):
                default = lambda_schema_default(section_name, field)
                if default is None:
                    continue
                value = default
            values.setdefault(_loss_name_for_field(field), set()).add(float(value))
    return values


def materialised_weight_conflicts(losses: Any) -> dict[str, set[float]]:
    """Losses whose lambda declarations disagree once schema defaults are stamped.

    ``build_loss_weight_table`` raises when one loss is declared twice at
    different values, and it counts a field as declared when it was *written*. A
    name reported here is written in one section and defaulted in another to a
    different number, so the two agree only while the first stays written. Sole
    owner of the reading (non-negotiable 17): the cohort regression test
    ``tests/unit/config/test_exp11_kspace_filling_loss_weights.py``, the audit's
    pin rule and the lambda migration all call it.

    Bound: this compares lambda fields with each other, never with a domain-list
    entry. A lambda default disagreeing with a list weight is a second shape, and
    an empty result here is not a claim that the arm has none of it.
    """
    return {name: v for name, v in _materialised_values(losses).items() if len(v) > 1}


def deleting_lambda_would_conflict(losses: Any, section: str, field: str) -> bool:
    """Whether ``losses.<section>.<field>`` is PINNING a materialised agreement.

    A lambda whose value differs from its own schema default, on a loss another
    section also aliases, is the only thing holding the two declarations equal:
    delete it and they disagree, though the written surface showed a redundant
    field. ``reconstruction.lambda_l2`` (default 0.0) against
    ``diffusion.lambda_mse`` (default 1.0) is the live case; the divergent
    per-section defaults behind it are issue #421.

    The caller is the audit's pairing rule, which reads a ``False`` as permission
    to delete — so this answers "would deleting it introduce a disagreement",
    not "does this arm load today". The two arms it protects load either way;
    what their deletion broke was the cohort regression test above.
    """
    name = _loss_name_for_field(field)
    before = _materialised_values(losses).get(name, set())
    after = _materialised_values(losses, dropping=(section, field)).get(name, set())
    return len(after) > 1 and len(before) <= 1
