"""Dotted-path config-override utilities (config-layer SSOT).

``apply_overrides`` and ``_parse_value`` used to live in ``spectramr.main`` (the
CLI/entry layer). ``pipelines/distributed.py`` imported ``apply_overrides`` from
there, which is a **leftward** import (``pipelines/ → main``) that violates the
inward-only dependency rule (CLAUDE.md #13). They are pure config transforms —
turn ``key.subkey=value`` strings into a re-validated :class:`TrainingSettings` —
so their canonical home is the innermost ``config/`` layer, which every layer may
import rightward. ``spectramr.main`` re-exports both for backward compatibility.
"""

from __future__ import annotations

import logging
import types
import typing
from typing import Any

from pydantic import BaseModel

from spectramr.config.schemas.renames import canonical_override_path

logger = logging.getLogger(__name__)


def apply_overrides(settings: Any, overrides: list[str]) -> Any:
    """Apply type-safe dot-notation overrides to ``settings``.

    **Type Safety**: values are parsed with :func:`_parse_value` and the whole
    result is re-validated against the Pydantic schema.

    Args:
        settings: ``TrainingSettings`` object.
        overrides: List of ``'key.subkey=value'`` strings, e.g.
            ``'optimization.optimizer.learning_rate=1e-4'``.

    Returns:
        Updated ``TrainingSettings`` object (re-validated).

    Raises:
        ValueError: If an override is malformed, traverses an existing non-dict
            node, or the reconstructed config fails schema validation.

    **Why the dump excludes unset fields.** The rebuild below re-validates a
    plain dict, and Pydantic marks every key it receives as *author-set*. A
    COMPLETE ``model_dump()`` therefore hands validation ~2000 defaulted keys
    dressed up as declarations, which breaks every consumer that reads
    ``model_fields_set`` to tell "the author asked for this" from "nobody said,
    so the schema default applies":

    * ``models/losses/weights.py`` — its conflict check documents "a schema
      default is not a declaration". The poisoned set made two *defaulted*
      aliased lambdas (``lambda_perceptual`` = 10.0, ``lambda_content`` = 0.0,
      both canonicalising to ``perceptual``) look like conflicting declarations
      and raise at build time. Net effect: ``audit`` passed and
      ``train --override ...`` died, which is how three ldm_two_stage_ulf_to_hf
      arms failed on 2026-07-25 (SLURM 7796517) — the smoke dispatcher always
      injects ``--override training.max_iterations=<cap>``.
    * ``parallel.deepspeed.compile`` / ``.zenflow`` — their
      ``_knobs_require_enabled`` validators reject knobs declared under a
      disabled block. Against the complete dump they fire on knobs nobody
      wrote, so NO DeepSpeed arm could be overridden at all (issue #1113;
      measured 7 / 120 sampled ``inprogress`` arms, all DeepSpeed).
    * The 2026-05-29 ``model_domain`` round-trip break, worked around at the
      time by forcing that one field's default to ``None``.

    Dumping ``exclude_unset=True`` keeps authorship correct *by construction* —
    the rebuilt set is exactly the authored keys plus the ones this call wrote
    (writing ``a.b.c=1`` puts ``a`` in the root's set, ``b`` in ``a``'s, ``c``
    in ``a.b``'s). It replaces a ~50-line post-hoc repair pass that could not
    help anyway, because the failures above happen *during* the rebuild.

    This does NOT weaken provenance (pitfall #15c). Provenance is stamped from
    the settings OBJECT by
    :meth:`~spectramr.config.settings.TrainingSettings.get_validated_snapshot`,
    which dumps all resolved fields including defaults; it never sees this
    intermediate dict. Nor does it weaken the validators: an override that
    genuinely *introduces* a dead knob (``parallel.deepspeed.zenflow.topk_ratio``
    under ``enabled: false``) still writes that key, so it is still authored and
    still raises.
    """
    settings_dict = settings.model_dump(exclude_unset=True)
    applied: list[str] = []
    # The same overrides as ``applied``, but CANONICAL paths only and no
    # values -- this is the machine-readable half, stamped onto the returned
    # object below so a consumer can ask "was this field overridden?".
    applied_paths: list[str] = []

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format '{override}', expected 'key.subkey=value'")

        key_path, value = override.split("=", 1)
        # A staged rename (`fold`) still accepts the old spelling in YAML, so it
        # must accept it here too. Translate BEFORE the write: whenever the arm
        # authored the canonical key, it is already in the dump, and writing the
        # legacy one beside it reads to the fold validator as two spellings that
        # disagree. Translating also routes retired spellings through the
        # rename table's own error, which names the replacement.
        declared_path = key_path
        key_path = canonical_override_path(key_path)
        keys = key_path.split(".")

        # Parse value type
        parsed_value = _parse_value(value)
        # `none` is a legal Literal member of fields such as parallel.strategy
        # (#1480); there it means the string, since None would fail validation.
        if value.strip().lower() == "none" and _destination_rejects_none(keys):
            parsed_value = "none"

        # Navigate to the nested key and set value. An absent/None node may be
        # created (building a new nested path is legitimate), but an existing
        # non-dict node must never be silently replaced with ``{}`` — that
        # destroyed the subtree on typo'd paths, and on permissive fields
        # (``dict[str, Any]``) the Pydantic re-validation below cannot catch
        # the loss. Fail loud instead (pitfall #9).
        current = settings_dict
        traversed: list[str] = []
        for key in keys[:-1]:
            traversed.append(key)
            if key not in current or current[key] is None:
                current[key] = {}
            elif not isinstance(current[key], dict):
                raise ValueError(
                    f"Override '{key_path}' traverses non-dict config node "
                    f"'{'.'.join(traversed)}' (existing type: "
                    f"{type(current[key]).__name__}); refusing to overwrite "
                    f"an existing config subtree. Check the override path."
                )
            current = current[key]

        current[keys[-1]] = parsed_value
        logger.debug(
            f"Override applied: {key_path} = {parsed_value} (type: {type(parsed_value).__name__})"
        )
        # Name the canonical destination whenever a fold moved the key, so a
        # caller passing a legacy spelling can see where it actually landed
        # rather than trusting that it landed anywhere.
        shown = key_path if declared_path == key_path else f"{declared_path}→{key_path}"
        applied.append(f"{shown}={parsed_value}")
        # Canonical, so a consumer asking about `training.max_iterations`
        # gets a hit whichever spelling the caller typed.
        applied_paths.append(key_path)

    # One consolidated line at INFO, not only the per-key DEBUG above. An
    # override is the single most consequential thing a caller does to a run and
    # nothing else in the pipeline echoes it, so at any level a normal run uses
    # the log could not answer "did -O actually take effect?" — the question
    # `-O optimization.gradient.enable_checkpointing=true` exists to ask.
    # Emitted BEFORE LoggingService.setup fans the configured level onto every
    # handler, so it survives an arm that sets `logging.sinks.level: warning`.
    if applied:
        logger.info("Overrides applied (%d): %s", len(applied), "  ".join(applied))

    # Reconstruct settings with validation. Pydantic validates all overrides
    # against the schema. Imported lazily to avoid an import cycle at module load.
    try:
        from spectramr.config.settings import TrainingSettings

        updated = TrainingSettings(**settings_dict)
    except Exception as e:
        raise ValueError(f"Config validation failed after applying overrides: {e}") from e

    # Stamp the machine-readable record. Until now the ONLY trace an override
    # left was the human INFO line above, so nothing downstream could ask "did
    # the caller move this field, or did the arm declare it?" -- and
    # ``model_fields_set`` cannot answer it either, because the rebuild marks a
    # YAML declaration and an override identically by construction (that is the
    # whole point of the ``exclude_unset`` dump documented above).
    #
    # That gap is not academic. On 2026-08-21 a 4-GPU run of
    # ``experiment_11_attention_none`` was launched with
    # ``-O training.max_iterations=5000``, and sanity-check mode independently
    # forces the budget to the same 5000. The log printed the number and not its
    # origin, so it could not distinguish "the operator asked for 5000" from
    # "a mode imposed 5000" -- the exact ambiguity the launch banner in
    # ``pipelines/training_loop.py`` now resolves by reading this back.
    #
    # Merged, not assigned: private attrs do NOT survive the
    # dump/re-validate round trip above, so an object that was already
    # overridden once would silently lose its record on a second call.
    prior = tuple(getattr(settings, "_override_paths", ()) or ())
    updated._override_paths = prior + tuple(applied_paths)
    return updated


def applied_override_paths(settings: Any) -> tuple[str, ...]:
    """Canonical dotted paths that :func:`apply_overrides` wrote onto ``settings``.

    The public read of the private record ``apply_overrides`` stamps. Returns an
    empty tuple for a settings object that was never overridden, and for any
    object carrying no record at all -- callers get "nothing was overridden",
    which is the truthful answer for a config no override ever touched.

    This is deliberately NOT a "did the user type this?" oracle. ``main.py``
    injects overrides of its own beside the caller's (``training.output_dir``,
    ``training.epochs``) and the smoke dispatcher injects
    ``training.max_iterations=<cap>``; all of them land here. The question it
    answers is the one that matters to a log reader: **this value did not come
    from the config file**.

    Args:
        settings: any settings object.

    Returns:
        Canonical paths in application order; ``()`` if none.
    """
    return tuple(getattr(settings, "_override_paths", ()) or ())


def _models_in(annotation: Any) -> list[type[BaseModel]]:
    """The Pydantic model classes an annotation names, through Optional/Union/Annotated."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    if typing.get_origin(annotation) is typing.Annotated:
        return _models_in(typing.get_args(annotation)[0])
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return [m for arg in typing.get_args(annotation) for m in _models_in(arg)]
    return []


def _destination_rejects_none(keys: list[str]) -> bool:
    """Whether the schema field at ``keys`` refuses ``None`` but accepts ``'none'``.

    ``False`` whenever the path cannot be resolved to one schema field (a dict-typed
    subtree, an unknown key), which keeps the historical ``none`` → ``None`` parse.
    """
    from pydantic import TypeAdapter, ValidationError

    from spectramr.config.settings import TrainingSettings

    models: list[type[BaseModel]] = [TrainingSettings]
    field = None
    for i, key in enumerate(keys):
        field = next((m.model_fields[key] for m in models if key in m.model_fields), None)
        if field is None:
            return False
        if i < len(keys) - 1:
            models = _models_in(field.annotation)
    if field is None or _models_in(field.annotation):
        return False
    adapter: TypeAdapter[Any] = TypeAdapter(field.annotation)
    try:
        adapter.validate_python(None)
        return False
    except ValidationError:
        pass
    try:
        adapter.validate_python("none")
    except ValidationError:
        return False
    return True


def _parse_value(value: str) -> Any:
    """Parse a string override value to the appropriate Python type.

    Supports booleans (``true``/``false``/``yes``/``no``/``on``/``off``),
    ``none``/``null`` → ``None``, integers, floats (incl. ``1e-4``), and falls
    back to the raw string.
    """
    # Boolean
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False

    # None
    if value.lower() in ("none", "null"):
        return None

    # Integer (try before float to avoid parsing "10" as 10.0)
    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
    except ValueError as _exc:
        logger.debug("Suppressed exception: %s", _exc)

    # Float
    try:
        return float(value)
    except ValueError as _exc:
        logger.debug("Suppressed exception: %s", _exc)

    # String (default)
    return value
