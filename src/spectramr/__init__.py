"""spectraMR: A multi-paradigm research framework for MRI reconstruction.

NOT FOR CLINICAL USE. See DISCLAIMER.md.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Static-only imports so type-checkers see the lazily-exported names
    # (resolved at runtime via ``__getattr__`` below — zero import cost here).
    from spectramr.config.settings import (
        TrainingSettings as TrainingSettings,
    )
    from spectramr.config.settings import (
        settings_from_dict as settings_from_dict,
    )
    from spectramr.core.metrics.registry import register_metric as register_metric
    from spectramr.models.losses.registry import register_loss as register_loss
    from spectramr.models.registry import register_model as register_model
    from spectramr.pipelines.fit import (
        Trainer as Trainer,
    )
    from spectramr.pipelines.fit import (
        fit as fit,
    )
    from spectramr.pipelines.make import (
        make_dataloader as make_dataloader,
    )
    from spectramr.pipelines.make import (
        make_dataset as make_dataset,
    )
    from spectramr.pipelines.make import (
        make_model as make_model,
    )
    from spectramr.pipelines.make import (
        make_optimizer as make_optimizer,
    )

# F-CUDART-EARLY / 2026-05-20 — the same FutureWarning is filtered in
# ``main.py`` but THIS package import happens earlier (whenever
# ``import spectramr.<anything>`` runs, including from notebooks, CLI
# wrappers, smoke scripts), so the filter there fires too late and the
# warning still lands in logs. Filter at the package-init seam so every
# downstream entry point inherits the suppression. This is a third-
# party deprecation we cannot act on (NVIDIA migrating cuda.cudart →
# cuda.bindings.runtime); silencing here keeps signal-to-noise high.
warnings.filterwarnings(
    "ignore",
    message="The cuda.cudart module is deprecated",
    category=FutureWarning,
)
# Same family — the new ``cuda.bindings`` migration emits its own
# DeprecationWarning paths via ``importlib._bootstrap``. The warning
# text starts with ``builtin type <name> has no __module__ attribute``;
# Python's ``warnings.filterwarnings`` matches the ``message`` regex
# against the warning text anchored at the start (re.match), so we
# anchor accordingly. The previous ``.*SwigPy...`` form did not match
# because `re.match` doesn't backtrack across alternatives the way
# ``re.search`` does — the second alternative ``swigvarlink...`` was
# unreachable.
warnings.filterwarnings(
    "ignore",
    message=r"builtin type (SwigPyPacked|SwigPyObject|swigvarlink) has no __module__",
    category=DeprecationWarning,
)
# Silence the pair of ``torch.jit.{script,interface} is deprecated``
# notices that fire on torch>=2.13 import — also third-party.
warnings.filterwarnings(
    "ignore",
    message=r"`?torch\.jit\.(script|interface)`? is deprecated",
    category=DeprecationWarning,
)

__version__ = "0.1.3.dev1"

# ---------------------------------------------------------------------------
# Public scripting surface — lazy re-exports (PEP 562 ``__getattr__``).
#
# ``import spectramr`` MUST stay cheap: eagerly importing these would drag in the
# whole torch + models import chain at package init (and risk an import cycle,
# since ``__init__`` runs before submodules). So each name resolves on first
# access to its canonical source module, then is cached into ``globals()`` so
# subsequent accesses skip ``__getattr__`` entirely. ``spectramr.api`` is the
# eager facade for users who prefer one import; both resolve to the SAME
# canonical objects.
# ---------------------------------------------------------------------------
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "TrainingSettings": ("spectramr.config.settings", "TrainingSettings"),
    "settings_from_dict": ("spectramr.config.settings", "settings_from_dict"),
    "register_model": ("spectramr.models.registry", "register_model"),
    "register_loss": ("spectramr.models.losses.registry", "register_loss"),
    "register_metric": ("spectramr.core.metrics.registry", "register_metric"),
    "make_model": ("spectramr.pipelines.make", "make_model"),
    "make_optimizer": ("spectramr.pipelines.make", "make_optimizer"),
    "make_dataset": ("spectramr.pipelines.make", "make_dataset"),
    "make_dataloader": ("spectramr.pipelines.make", "make_dataloader"),
    "fit": ("spectramr.pipelines.fit", "fit"),
    "Trainer": ("spectramr.pipelines.fit", "Trainer"),
}

__all__: list[str] = [
    "Trainer",
    "TrainingSettings",
    "__version__",
    "fit",
    "make_dataloader",
    "make_dataset",
    "make_model",
    "make_optimizer",
    "register_loss",
    "register_metric",
    "register_model",
    "settings_from_dict",
]


def __getattr__(name: str) -> Any:
    """Resolve a public export lazily (PEP 562)."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'spectramr' has no attribute {name!r}")
    module_path, attr = target
    value = getattr(importlib.import_module(module_path), attr)
    globals()[name] = value  # cache: subsequent accesses skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


def _in_child_process() -> bool:
    """True when this interpreter is a ``multiprocessing`` child.

    ``multiprocessing.parent_process()`` is the obvious test and it does NOT
    work here. Under ``spawn``/``forkserver`` the child re-imports this package
    while UNPICKLING the target callable, which happens strictly before
    ``BaseProcess._bootstrap`` assigns ``multiprocessing._parent_process``. So
    at the moment this module body runs -- the only moment that matters --
    ``parent_process()`` is ``None`` in the child too. Measured on 3.12::

        IMPORT parent_process_is_None=True name='ForkServerProcess-2' _inheriting=True
        IMPORT parent_process_is_None=True name='SpawnProcess-3'      _inheriting=True

    ``current_process()`` is populated from the pickled process object before
    the re-import, so its ``name`` (and the private ``_inheriting`` flag the
    spawn bootstrap sets) discriminate correctly in that window. Both are
    checked; either alone is sufficient, together they are belt-and-braces
    against one of the two changing shape.

    ``fork`` children never re-import at all, so this predicate simply never
    runs there -- which is why the fix does not depend on the start method.
    """
    proc = multiprocessing.current_process()
    if getattr(proc, "_inheriting", False):
        return True
    return proc.name != "MainProcess"


def _emit_clinical_disclaimer() -> bool:
    """Warn once that spectraMR is research-only — main process only.

    Returns ``True`` if the warning was emitted, ``False`` if suppressed.
    Three gates silence it:

    * ``SPECTRAMR_SUPPRESS_CLINICAL_WARNING`` is set (explicit opt-out).
    * We are inside a spawned child — e.g. a DataLoader ``spawn`` worker, which
      re-imports the package on startup and would otherwise re-emit this
      module-level warning once per worker. See :func:`_in_child_process` for
      why the obvious ``parent_process()`` test does not detect that.
    * We are a non-zero rank of a distributed launch: N interpreters means N
      copies of one legal notice. ``RANK``/``WORLD_SIZE`` are read straight off
      the environment rather than through the ``spectramr.core.env`` SSOT that
      declares them. That began as a cost decision: importing the SSOT executed
      an eager ``spectramr.core.__init__`` that pulled torch, measured at 2.8 s
      and 4205 modules, paid by every ``import spectramr``. #1130 made that
      package lazy, and the same import now measures 0.013 s / 102 modules with
      torch absent — so the literals are no longer forced, only still
      sufficient, and are left as they are because rewriting a legal-notice
      gate is not that change's business.
      The names are pinned to that SSOT by a unit test.
    """
    if os.environ.get("SPECTRAMR_SUPPRESS_CLINICAL_WARNING"):
        return False
    if _in_child_process():
        return False
    if "WORLD_SIZE" in os.environ and (os.environ.get("RANK", "0") != "0"):
        return False
    warnings.warn(
        "spectraMR is research software and is NOT FOR CLINICAL USE. "
        "It is not a medical device and has not been evaluated by any regulatory "
        "authority. See DISCLAIMER.md. "
        "Set SPECTRAMR_SUPPRESS_CLINICAL_WARNING=1 to silence this message in "
        "batch jobs.",
        category=UserWarning,
        stacklevel=2,
    )
    return True


_emit_clinical_disclaimer()
