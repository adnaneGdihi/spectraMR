"""What was done to ``input_raw`` / ``input_prepared`` / ``target`` before the snapshot.

A debug snapshot shows four numbers per tensor and a PNG. Neither says whether
the picture is what the *config* asked for: an ``input_prepared`` that looks
fully sampled is a defect if the arm declared acceleration and correct if it
declared none, and the snapshot alone cannot tell the two apart. This module
answers that in the artifact itself, recording two things side by side:

* ``declared`` — the knobs the user set (config SSOT: normalization, NEX target
  construction, augmentation).
* ``applied`` — what the pipeline actually built (the real dataset wrapper chain
  and the real ``tio.Compose``), plus any model-side space-filling-curve
  reordering (Hilbert / Morton / snake / zigzag / raster) read off the
  constructed module.

A divergence between the two halves *is* the finding. See
``docs/debug_snapshot_contract.rst`` for the key contract this supports.

Every read is best-effort and type-guarded: a training run must never die
because a diagnostic could not introspect a dataset. What cannot be resolved is
named in ``incomplete`` — a silently partial record would recreate exactly the
facade this exists to expose (pitfall #16).
"""

from __future__ import annotations

import logging
from typing import Any

from spectramr.core.module_utils import unwrap_model

logger = logging.getLogger(__name__)

#: ``config.data.processing`` fields describing what was done to intensities.
_NORMALIZATION_FIELDS = (
    "enable_kspace_normalization",
    "kspace_percentile",
    "kspace_scale_domain",
    "enable_log_scaling",
    "log_scaling_center_fraction",
    "normalization_type",
    "normalization_kwargs",
    "enable_image_normalization",
    "enable_image_rescale",
    "rescale_range",
    "rescale_percentiles",
)

#: ``config.data`` fields describing how input and target were *constructed*.
#: ``target_mode`` produced the original report: a ``phase_aligned_mean`` target
#: is a coherent sqrt(N) NEX average while the input is one noisier repetition,
#: so the two legitimately look different.
_CONSTRUCTION_FIELDS = (
    "target_mode",
    # The validation split's target may differ from training's, and under
    # `target_mode: r2r` it MUST (r2r redraws its target every __getitem__, so
    # it is not a metric reference). A snapshot that named only `target_mode`
    # would describe the wrong half of the run for every r2r arm.
    "val_target_mode",
    # Read only under `target_mode: r2r`; it splits the noise budget between
    # the two recorrupted halves, so two runs differing only in alpha are
    # different constructions that no other stamped field distinguishes.
    "r2r_alpha",
    "nex_target_exclude_input",
    "nex_fallback",
    # One record per slice (depth-1 subjects, one slice read per repetition)
    # or one per group (#1757): the record must say which index built the
    # sample it shows.
    "slice_level_records",
    "return_image_domain",
)

_JSON_SCALARS = (bool, int, float, str)
#: Depth caps. A wrapper cycle or a pathological Compose must not hang a run.
_MAX_CHAIN_DEPTH = 16
_MAX_TRANSFORMS = 64
#: Attributes a dataset wrapper delegates through, in the order they are tried.
#: ``subjects_dataset`` is TorchIO's own name on ``tio.Queue``, and it is the one
#: that matters in practice: every patch-sampled arm hands this module a
#: ``Queue``, which carries none of the other three, so the walk terminated
#: immediately and the whole applied half landed in ``incomplete``. (The repo
#: already relies on that attribute elsewhere -- see the ``Queue`` guard in
#: ``training/builders/data_builder.py``.)
_WRAPPER_ATTRS = ("inner", "dataset", "base", "base_dataset", "subjects_dataset")

#: Where a dataset keeps its ``tio.Compose``, in the order the names are tried.
#: TorchIO 1.2 made it private -- ``SubjectsDataset`` stores ``self._transform``
#: and exposes only ``set_transform()``, with no public read accessor -- while
#: this repo's own datasets (``M4RawRepetitionDataset``, ``slice_dataset``,
#: ``cine_dataset``, ...) still assign a public ``self.transform``. Both
#: spellings are the SAME declared fact, so reading one after the other is not
#: default substitution; reading only the first is how nine
#: ``tio.SubjectsDataset`` call sites report "no transform" while holding one.
_TRANSFORM_ATTRS = ("transform", "_transform")


def _is_mock(obj: Any) -> bool:
    """Is ``obj`` a test double rather than a real pipeline object?

    A ``MagicMock`` satisfies every duck-type here (``.transform``,
    ``__getitem__``, a plausible ``type(m).__name__``), so an unguarded walk
    writes ``{"name": "MagicMock"}`` into ``snapshot.json`` as though it were
    real -- a confident wrong answer in the artifact whose job is to be trusted,
    worse than recording nothing. The module path is the only discriminator: the
    mock answers every attribute-shaped check affirmatively, and there is no
    positive type to test for (#693's ``run_dir``) without importing TorchIO.
    """
    return type(obj).__module__.startswith("unittest.mock")


def _scalarize(value: Any) -> Any:
    """Coerce to something ``json.dumps`` accepts; ``None`` for anything else."""
    if value is None or isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, (list, tuple)):
        return [_scalarize(v) for v in value][:16]
    if isinstance(value, dict):
        return {
            str(k): _scalarize(v)
            for k, v in list(value.items())[:16]
            if isinstance(k, _JSON_SCALARS)
        }
    return None


def _read_block(
    block: Any,
    fields: tuple[str, ...],
    incomplete: list[str] | None = None,
    *,
    label: str = "",
) -> dict[str, Any]:
    """Read ``fields`` off a Pydantic config block, skipping what is absent.

    ``_scalarize`` returns ``None`` for two different situations -- the value
    genuinely IS ``None``, and the value is a type JSON cannot carry. Dropping
    both silently made an unrepresentable declared knob indistinguishable from
    an unset one, which is the facade this module exists to expose (pitfall
    #16): the record would read as a complete account of what was declared
    while a knob that shaped the data went unmentioned.

    So the second case is named in ``incomplete``. The first is not -- an
    unset knob is honestly absent, and listing every ``None`` would bury the
    real gaps in noise.
    """
    out: dict[str, Any] = {}
    for name in fields:
        if not hasattr(block, name):
            continue
        raw = getattr(block, name)
        coerced = _scalarize(raw)
        if coerced is not None:
            out[name] = coerced
        elif raw is not None and incomplete is not None:
            incomplete.append(
                f"{label}{name}: declared as {type(raw).__name__}, "
                "not JSON-representable — value omitted"
            )
    return out


def _declared(config: Any, incomplete: list[str]) -> dict[str, Any]:
    """The half of the record that comes from the config SSOT."""
    data = getattr(config, "data", None)
    if data is None:
        incomplete.append("config.data absent — no declared section")
        return {}
    declared: dict[str, Any] = _read_block(data, _CONSTRUCTION_FIELDS, incomplete, label="data.")
    processing = getattr(data, "processing", None)
    if processing is None:
        incomplete.append("config.data.processing absent — no normalization declared")
    else:
        declared["normalization"] = _read_block(
            processing, _NORMALIZATION_FIELDS, incomplete, label="data.processing."
        )
    augmentation = getattr(data, "augmentation", None)
    if augmentation is not None:
        enabled = _scalarize(getattr(augmentation, "enabled", None))
        if enabled is not None:
            declared["augmentation_enabled"] = enabled
    return declared


def _dataset_chain(dataset: Any, incomplete: list[str]) -> tuple[list[str], Any]:
    """Unwrap delegating dataset wrappers; return their names and the innermost.

    The wrappers matter: ``SFCConformalFMRIKeysWrapper`` attaches batch keys
    *outside* ``dataset.transform``, so a record built from the ``Compose``
    alone would omit it entirely.
    """
    chain: list[str] = []
    node = dataset
    for _ in range(_MAX_CHAIN_DEPTH):
        if node is None:
            break
        chain.append(type(node).__name__)
        nxt = None
        for attr in _WRAPPER_ATTRS:
            candidate = getattr(node, attr, None)
            # A Dataset delegate, not a tensor and not a Mock. `__getitem__` is
            # the protocol every wrapper in this repo delegates through.
            if candidate is not None and hasattr(type(candidate), "__getitem__"):
                nxt = candidate
                break
        if nxt is None or nxt is node:
            return chain, node
        node = nxt
    incomplete.append(f"dataset wrapper chain deeper than {_MAX_CHAIN_DEPTH}")
    return chain, node


def _transform_params(transform: Any) -> dict[str, Any]:
    """The declared arguments of one TorchIO transform, cheaply.

    ``args_names`` is TorchIO's own reproducibility contract (what
    ``Transform.__repr__`` and history replay use), so reading it needs no
    per-transform knowledge and stays correct as transforms are added.
    """
    names = getattr(transform, "args_names", None)
    if not isinstance(names, (list, tuple)):
        return {}
    params: dict[str, Any] = {}
    for name in names:
        if not isinstance(name, str):
            continue
        coerced = _scalarize(getattr(transform, name, None))
        if coerced is not None:
            params[name] = coerced
    return params


def _flatten_transforms(transform: Any, out: list[dict[str, Any]], depth: int) -> None:
    """Walk a (possibly nested) ``tio.Compose`` into a flat ordered list.

    Duck-typed on ``.transforms`` rather than importing TorchIO: the walk then
    also covers ``OneOf`` and any future container, and imports nothing into the
    training package that is not already there.
    """
    if transform is None or depth > _MAX_CHAIN_DEPTH or len(out) >= _MAX_TRANSFORMS:
        return
    children = getattr(transform, "transforms", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            _flatten_transforms(child, out, depth + 1)
        return
    if isinstance(children, dict):  # OneOf maps transform -> probability
        for child in children:
            _flatten_transforms(child, out, depth + 1)
        return
    record: dict[str, Any] = {"name": type(transform).__name__}
    params = _transform_params(transform)
    if params:
        record["params"] = params
    out.append(record)


def _dataset_transform(dataset: Any) -> Any:
    """Read a dataset's ``tio.Compose`` under either spelling it may carry.

    Returns ``None`` only when the dataset genuinely holds no transform, which
    the caller reports in ``incomplete`` -- naming every attribute tried, because
    a bare "exposes no .transform" is what sent this triage at the M4Raw loader
    when the real answer was that the attribute had been renamed.

    The two spellings are the SAME declared fact under different storage names,
    so reading one after the other is not the default substitution
    non-negotiable 3b forbids: there is no default in play here, only storage.
    """
    for name in _TRANSFORM_ATTRS:
        candidate = getattr(dataset, name, None)
        if candidate is None:
            continue
        # A private spelling is gated on callability -- the contract TorchIO's
        # own ``set_transform`` enforces -- so an unrelated ``_transform`` (a str
        # flag, an int) cannot write a confident wrong ``{"name": "str"}`` into
        # the artifact. The public name is deliberately NOT gated: a Compose need
        # not be callable for ``_flatten_transforms`` to walk its ``.transforms``,
        # and tightening it would silently drop the datasets already reporting
        # correctly.
        if name.startswith("_") and not callable(candidate):
            continue
        return candidate
    return None


def _applied(dataset: Any, incomplete: list[str]) -> dict[str, Any]:
    """The half of the record that comes from the objects actually built."""
    if dataset is None:
        incomplete.append("no dataset reachable — applied transforms unknown")
        return {}
    if _is_mock(dataset):
        incomplete.append("dataset is a test double — applied transforms unknown")
        return {}
    chain, innermost = _dataset_chain(dataset, incomplete)
    applied: dict[str, Any] = {"dataset_chain": chain}
    transform = _dataset_transform(innermost)
    if transform is None:
        tried = " or ".join(f".{name}" for name in _TRANSFORM_ATTRS)
        incomplete.append(f"{type(innermost).__name__} exposes no transform under {tried}")
        return applied
    flat: list[dict[str, Any]] = []
    _flatten_transforms(transform, flat, 0)
    if not flat:
        incomplete.append("transform present but not walkable (no .transforms leaf)")
        return applied
    if len(flat) >= _MAX_TRANSFORMS:
        incomplete.append(f"transform list truncated at {_MAX_TRANSFORMS}")
    applied["transforms"] = flat
    return applied


def _linearization(model: Any, incomplete: list[str]) -> list[dict[str, Any]]:
    """Space-filling-curve reordering the MODEL applies to its own input.

    ``HilbertOrder`` and ``ImageTopologyLinearizer`` both permute the spatial
    axes into curve order before the sequence backbone sees them, so on a
    Mamba/SSM arm the tensor the network consumes is not the one the snapshot
    renders. Both carry ``.mode``; reading it off the *constructed* module
    covers every curve (hilbert / morton / snake / zigzag / raster) with no
    hardcoded list, and reports what was built rather than what config asked
    for.
    """
    if model is None or _is_mock(model):
        return []
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return []
    found: list[dict[str, Any]] = []
    try:
        for module in modules():
            mode = getattr(module, "mode", None)
            if not isinstance(mode, str):
                continue
            # `mode` alone is ambiguous (an upsampler has one); the permutation
            # buffer is what makes this module a linearizer.
            if not (hasattr(module, "permutation") or hasattr(module, "forward_idx")):
                continue
            record: dict[str, Any] = {"module": type(module).__name__, "mode": mode}
            shape = _scalarize(getattr(module, "shape", None))
            if shape is not None:
                record["shape"] = shape
            if record not in found:
                found.append(record)
    except Exception as exc:  # a model that resists iteration must not kill a run
        incomplete.append(f"model linearization scan failed: {exc}")
    return found


def _timestep_floor(config: Any, model: Any, incomplete: list[str]) -> dict[str, Any] | None:
    """Declared ``train_identity_rung`` beside the floor the process resolved.

    The other declared/applied pairs in this record compare a config value with
    what the pipeline built. This one is the same shape for the diffusion
    timestep floor: ``undersampling.train_identity_rung`` is the declared half,
    and ``KSpaceUndersamplingProcess.min_meaningful_timestep()`` -- read off the
    CONSTRUCTED process, not recomputed -- is the applied half. It is the SSOT
    for both the training sampler's lower bound and the reverse schedule's
    terminus (issue #535), so a snapshot that does not carry it cannot answer
    "was the fully-sampled rung in this run?" from the artifact alone.

    Not recomputed from config on purpose: a second predictor of the floor is
    the divergence shape non-negotiable 17 forbids, and it would agree with the
    real one right up until the day it did not.

    Returns ``None`` for a non-diffusion arm (no process to read), which is most
    of the corpus -- there, absent means "not applicable" and nothing is added to
    ``incomplete``. A probe that *fails* is the other case and is NOT silent: it
    also returns ``None``, but names itself in ``incomplete`` first, so absence
    plus a clean ``incomplete`` is the only reading that means "not applicable".

    The process is reached through :func:`~spectramr.core.module_utils.unwrap_model`
    rather than a direct attribute read, because training wraps the model and the
    wrapper does not forward ``kspace_process``. This cohort declares
    ``parallel.strategy: deepspeed`` *and* ``ema.enabled: true``, so on the real
    launch path a direct read finds nothing and the key would vanish from every
    snapshot -- the silent-absence shape this record exists to prevent. Unwrapping
    via the ``core.module_utils`` SSOT rather than an inline ``.module`` hop also
    covers ``torch.compile`` (``_orig_mod``) and FSDP, which an inline hop misses
    (non-negotiable 17).
    """
    process = getattr(model, "kspace_process", None)
    if process is None and model is not None and not _is_mock(model):
        inner = unwrap_model(model)
        if inner is not model:
            process = getattr(inner, "kspace_process", None)
    resolver = getattr(process, "min_meaningful_timestep", None)
    if not callable(resolver) or _is_mock(process):
        return None
    try:
        applied = int(resolver())
    except Exception as exc:  # a provenance probe must never kill a run
        incomplete.append(f"timestep floor probe failed: {exc}")
        return None
    accel = getattr(config, "undersampling", None)
    return {
        "declared_train_identity_rung": bool(getattr(accel, "train_identity_rung", False)),
        "applied_min_timestep": applied,
    }


def build_snapshot_provenance(
    config: Any,
    *,
    dataset: Any | None = None,
    model: Any | None = None,
    source: str = "train",
) -> dict[str, Any]:
    """Record what was done to the data, as declared and as applied.

    Args:
        config: The frozen ``TrainingSettings`` SSOT.
        dataset: The dataset behind the loader this batch came from. Optional:
            the ``declared`` half is still recorded without it.
        model: The constructed generator, scanned for curve reordering.
        source: Which split the batch came from (``"train"`` / ``"val"``), so a
            val-tagged snapshot cannot silently claim train augmentation --
            they are built by *different* Compose objects.

    Returns:
        A JSON-serialisable dict with ``source``, ``declared``, ``applied``,
        ``model_input_linearization`` and ``incomplete``, plus
        ``timestep_floor`` on diffusion arms. Never raises.
    """
    incomplete: list[str] = []
    try:
        record: dict[str, Any] = {
            "source": source,
            "declared": _declared(config, incomplete),
            "applied": _applied(dataset, incomplete),
            "model_input_linearization": _linearization(model, incomplete),
        }
        floor = _timestep_floor(config, model, incomplete)
        if floor is not None:
            record["timestep_floor"] = floor
    except Exception as exc:
        logger.debug("snapshot provenance build failed: %s", exc)
        return {"source": source, "error": f"{type(exc).__name__}: {exc}"}
    record["incomplete"] = incomplete
    return record


# ──────────────────────────────────────────────────────────────────────
# Run identity (#1299)
# ──────────────────────────────────────────────────────────────────────
#
# A snapshot recorded `step`, `tag`, `paradigm`, the data-provenance block and
# tensor stats -- and no run identity at all. The one artifact that carried it,
# `provenance.json`, is written once per launch and OVERWRITTEN, so relaunching
# an arm into the same directory silently rewrites the identity of every
# snapshot already sitting there.
#
# That is not hypothetical: `experiment_11_attention_none`'s directory held
# artifacts from five launches, and its `step_004000` snapshots came from a run
# configured `timesteps: 28` / `max_iterations: 4000` -- neither matching the
# on-disk YAML. The `provenance.json` that would have said so had been
# overwritten. Comparing `step_000001` against `step_004000` in that directory
# is unsound, and nothing in either file reveals it. Mtimes do not help; they
# order writes, not runs.
#
# The fix is to make the identity travel WITH the snapshot. What it must not do
# is mint a second one: a lazily-computed id here would differ from
# `provenance.json`'s (a different timestamp alone would do it) and one run
# would have two identities, which is worse than none. So the pipeline PUBLISHES
# the record it already built, and this module hands it out.

#: Published by `pipelines/train.py` from the record `collect_run_provenance`
#: built. Process-global because the snapshot writer is called from inside a
#: training step, ten frames below anything that knows about the run.
_RUN_IDENTITY: dict[str, Any] | None = None

#: The fallback, computed at most once. Separate from the above so
#: `set_run_identity` can still override it if the pipeline publishes late.
_FALLBACK_IDENTITY: dict[str, Any] | None = None

#: Values of ``identity_source``. ``fallback`` means the snapshot was written
#: outside a published run (a test, a bare strategy, a pipeline that died before
#: provenance) -- readable, but NOT the run's canonical id, and it says so.
IDENTITY_FROM_RUN = "run_provenance"
IDENTITY_FALLBACK = "fallback"


def set_run_identity(provenance: dict[str, Any] | None) -> dict[str, Any] | None:
    """Publish the run's identity for every snapshot this process writes.

    Takes the record `collect_run_provenance` already assembled rather than
    rebuilding one, so the id in a snapshot and the id in `provenance.json` are
    the same string by construction. Returns what was published (``None`` when
    the caller had no provenance -- provenance capture is fail-open, so an empty
    record is a normal outcome and must not become an exception here).
    """
    global _RUN_IDENTITY
    if not provenance:
        _RUN_IDENTITY = None
        return None
    git = provenance.get("git") or {}
    _RUN_IDENTITY = {
        "run_id": provenance.get("run_id"),
        "run_name": provenance.get("run_name"),
        "started_at": provenance.get("started_at"),
        "config_sha256": provenance.get("config_sha256"),
        "git_sha": git.get("sha"),
        "git_branch": git.get("branch"),
        # A clean sha does not reproduce a run whose tree was dirty, so the flag
        # travels with the sha rather than being inferable from it.
        "git_dirty": git.get("dirty"),
        "identity_source": IDENTITY_FROM_RUN,
    }
    return _RUN_IDENTITY


def reset_run_identity() -> None:
    """Drop both the published and the fallback identity (tests)."""
    global _RUN_IDENTITY, _FALLBACK_IDENTITY
    _RUN_IDENTITY = None
    _FALLBACK_IDENTITY = None


def run_identity() -> dict[str, Any]:
    """The identity to stamp into a snapshot, published or fallback.

    Never raises and never returns an empty dict: a snapshot whose identity
    block is missing is indistinguishable from one written before this existed,
    which is the ambiguity the whole change removes. When nothing was published
    the record still carries a usable id and says ``identity_source: fallback``
    so a reader knows not to correlate it with `provenance.json`.
    """
    if _RUN_IDENTITY is not None:
        return _RUN_IDENTITY

    global _FALLBACK_IDENTITY
    if _FALLBACK_IDENTITY is None:
        from datetime import datetime

        started = datetime.now().astimezone()
        git: dict[str, Any] = {}
        try:
            from spectramr.infrastructure.logging.provenance import git_provenance

            git = git_provenance()
        except Exception:  # pragma: no cover - git absent / import failure
            git = {}
        _stamp = started.strftime("%Y%m%d_%H%M%S")
        _sha = git.get("sha_short") or "nogit"
        run_id = f"unpublished-{_stamp}-{_sha}"
        _FALLBACK_IDENTITY = {
            "run_id": run_id,
            "run_name": None,
            "started_at": started.isoformat(timespec="seconds"),
            "config_sha256": None,
            "git_sha": git.get("sha"),
            "git_branch": git.get("branch"),
            "git_dirty": git.get("dirty"),
            "identity_source": IDENTITY_FALLBACK,
        }
    return _FALLBACK_IDENTITY


__all__ = [
    "IDENTITY_FALLBACK",
    "IDENTITY_FROM_RUN",
    "build_snapshot_provenance",
    "reset_run_identity",
    "run_identity",
    "set_run_identity",
]
