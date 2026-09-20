"""The one live ``torch.compile`` call site.

Split out of ``ModelBuilder.compile()`` so the director can invoke it at the
point :mod:`compile_placement` names, rather than at the fixed step-1 position
that was wrong for every strategy but the single-device one.

**The failure policy is carried over verbatim and must stay that way.**
Compilation failure raises. It used to be wrapped in a blanket
``except Exception`` that logged a warning and continued eager, so a
``compile_model: true`` arm ran to completion, reported success, and had never
been compiled — the arm was simply slower than its own provenance claimed
(#619 F2). ``compile.enabled: false`` is how you ask for eager.

Two dormant modules still carry the opposite policy —
``infrastructure/services/model_compilation_service.py`` and
``infrastructure/performance_optimizer.py`` both set
``torch._dynamo.config.suppress_errors`` and return the uncompiled model. Harvest
their shape if useful; never their failure handling.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from spectramr.core.compile_capability import compile_backend_for, triton_available
from spectramr.core.module_utils import is_wrapped

from .compile_regions import (
    NoRegionsToCompileError,
    compile_regions,
    has_repeated_blocks,
)

logger = logging.getLogger(__name__)

__all__ = ["CompileRecord", "apply_compile", "compile_is_enabled"]


class CompileRecord(dict):
    """Per-model applied record, for provenance.

    A plain ``dict`` subclass so it serialises without a custom encoder; the
    class exists to give the value a name at the call sites.
    """


#: Backends whose kernels Inductor emits through Triton. Someone who asked for
#: ``eager`` or ``aot_eager`` explicitly needs no Triton, so the device
#: cross-check below is scoped to the backends that depend on it rather than
#: applied to every compiled run.
_TRITON_BACKED_BACKENDS = frozenset({"inductor"})


def _assert_device_can_compile(device: Any, backend: str) -> None:
    """Raise when *device* cannot run *backend*, before dynamo is involved.

    ``compile_backend_for`` owns the rule -- Inductor on CUDA needs Triton to
    emit kernels -- and raises ``CompileUnsupportedError``. Calling it here is
    what makes that error reachable at all: its only other caller,
    ``build_capabilities``, catches it immediately and records
    ``incomplete: ["compile_backend"]``, which is provenance rather than
    enforcement. Without this the arm gets to its first forward pass on a
    cluster node before anything notices.
    """
    if backend not in _TRITON_BACKED_BACKENDS:
        return
    device_type = getattr(device, "type", None) or str(device or "cpu").split(":")[0]
    compile_backend_for(device_type, triton=triton_available())


def _select_models(
    models: dict[str, nn.Module], compile_cfg: Any
) -> dict[str, nn.Module]:
    """The subset ``apply_to`` names, or everything when it is absent.

    Raises on a name this arm does not build. The schema closes the vocabulary
    to the four keys ``ModelBuilder`` can write, but a GAN arm naming ``encoder``
    is still asking for something that is not there -- and compiling nothing
    while reporting compilation is the failure this whole module refuses.
    """
    names = getattr(compile_cfg, "apply_to", None)
    if not names:
        return dict(models)
    built = {name for name, model in models.items() if model is not None}
    unknown = sorted(set(names) - built)
    if unknown:
        raise RuntimeError(
            f"optimization.compile.apply_to names {unknown}, which this arm does not "
            f"build (it has {sorted(built)}). Compiling nothing for a declared model "
            "would report a configuration this run did not use."
        )
    return {name: models[name] for name in names}


def _configure_dynamo(compile_cfg: Any) -> dict[str, Any]:
    """Apply the recompilation policy, and report what was applied.

    The one place in ``src/`` that writes ``torch._dynamo.config``. The two
    dormant modules set ``suppress_errors`` there, which is the opposite policy;
    keeping a single writer is what stops the two meeting (non-negotiable 17).

    ``fail_on_recompile_limit_hit`` is OFF in torch. A frame that exhausts its
    budget is then dropped to eager and the run continues -- reporting a
    compiled configuration it stopped executing. That is the build-time lie this
    module already refuses, arriving later in the run instead.
    """
    applied: dict[str, Any] = {}
    try:
        import torch._dynamo.config as dynamo_config
    except Exception:  # torch built without dynamo, or the CI shim
        logger.debug("torch._dynamo.config unavailable; recompilation policy not applied")
        return {"recompile_policy": "unavailable"}

    limit = getattr(compile_cfg, "recompile_limit", None)
    if limit is not None and hasattr(dynamo_config, "recompile_limit"):
        dynamo_config.recompile_limit = limit
        applied["recompile_limit"] = limit

    fail = bool(getattr(compile_cfg, "fail_on_recompile_limit", True))
    if hasattr(dynamo_config, "fail_on_recompile_limit_hit"):
        dynamo_config.fail_on_recompile_limit_hit = fail
        applied["fail_on_recompile_limit"] = fail
    return applied


def compile_is_enabled(config: Any) -> bool:
    """Whether this arm asked for compilation."""
    optimization = getattr(config, "optimization", None)
    compile_cfg = getattr(optimization, "compile", None)
    return bool(getattr(compile_cfg, "enabled", False))


def apply_compile(
    models: dict[str, nn.Module],
    config: Any,
    *,
    stage: str,
    device: Any = None,
) -> tuple[dict[str, nn.Module], CompileRecord]:
    """Compile every entry of *models*, returning the new mapping and a record.

    Args:
        models: name -> module. Replaced, not mutated in place.
        config: the frozen settings; ``optimization.compile`` is read.
        stage: the placement this ran at, recorded for provenance so a reader
            can tell ``DDP(compile(m))`` from ``compile(DDP(m))`` after the fact.
        device: the resolved device, cross-checked against the backend before
            anything reaches dynamo.

    Raises:
        RuntimeError: compilation was requested but is unavailable, or failed.
        CompileUnsupportedError: the device cannot run the requested backend.
    """
    if not compile_is_enabled(config):
        return models, CompileRecord()

    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "optimization.compile.enabled=true but torch.compile is unavailable "
            f"(torch {getattr(torch, '__version__', 'unknown')}; requires 2.0+). "
            "Install a newer torch or set compile.enabled: false — silently "
            "running eager would misreport what this arm executed."
        )

    compile_cfg = config.optimization.compile
    mode = compile_cfg.mode
    backend = compile_cfg.backend
    _assert_device_can_compile(device, backend)
    logger.info("Compiling models at %s with mode=%r, backend=%r", stage, mode, backend)

    regional = bool(getattr(compile_cfg, "regional", False))
    kwargs = {
        "mode": mode,
        "backend": backend,
        "fullgraph": compile_cfg.fullgraph,
        "dynamic": compile_cfg.dynamic,
    }

    selected = _select_models(models, compile_cfg)
    dynamo_policy = _configure_dynamo(compile_cfg)

    if regional:
        # Checked across every SELECTED model before the first is touched.
        # `compile_regions` rewrites in place, so discovering on the third model
        # that it does not qualify would leave the first two regionally compiled
        # and the build half-applied.
        unqualified = sorted(
            name
            for name, model in selected.items()
            if model is not None and not has_repeated_blocks(model)
        )
        if unqualified:
            raise NoRegionsToCompileError(
                "optimization.compile.regional is set, but "
                f"{unqualified} expose no nn.ModuleList of two or more identical "
                "non-leaf blocks, so there is nothing to compile regionally. Set "
                "regional: false to compile the whole model instead -- silently "
                "doing that for you would report a configuration this run did not use."
            )

    # Every model is returned; only the selected ones are compiled.
    compiled: dict[str, nn.Module] = dict(models)
    record = CompileRecord()
    for name, model in selected.items():
        if model is None:
            continue
        logger.info("Compiling model: %s", name)
        try:
            if regional:
                # In place, and the parent is left bare on purpose -- compiling
                # the root as well would reinstate the cost regional exists to
                # avoid. Raises when nothing qualifies rather than quietly
                # compiling the whole model instead.
                blocks = compile_regions(model, **kwargs)
                entry: dict[str, Any] = {"regions_compiled": blocks}
            else:
                compiled[name] = torch.compile(model, **kwargs)
                entry = {"regions_compiled": 0}
        except NoRegionsToCompileError:
            # Not a compile failure: nothing was handed to dynamo. Re-framing it
            # below appended the wrong closing instruction -- "set
            # compile.enabled: false" -- to a message whose actual fix is
            # `regional: false`, and the closing line is the one a reader acts on.
            raise
        except Exception as e:
            raise RuntimeError(
                f"torch.compile failed for model {name!r} "
                f"(mode={mode!r}, backend={backend!r}): {e}. "
                "Refusing to fall back to eager: the run would report success "
                "while executing something other than what was configured. "
                "Set optimization.compile.enabled: false to run eager on purpose."
            ) from e
        # `wrapped` is OBSERVED, not assumed: it asks the resulting module
        # whether a wrapper is actually on it. A record built purely from the
        # config would say the same thing whether or not torch.compile did
        # anything, which is the declared-vs-applied distinction this record
        # exists to carry. Under `regional` the ROOT is deliberately bare, so
        # this reads False there and `regions_compiled` is the count that
        # matters -- the two fields together say which shape ran.
        record[name] = {
            "stage": stage,
            "mode": mode,
            "backend": backend,
            "regional": regional,
            "wrapped": is_wrapped(compiled[name]),
            **dynamo_policy,
            **entry,
        }
    return compiled, record
