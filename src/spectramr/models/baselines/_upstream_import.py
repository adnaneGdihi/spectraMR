"""Importing an upstream baseline that expects to be run from its own repository root.

Two of the three vendored baselines ship a top-level ``utils`` package, so putting both
roots on ``sys.path`` makes ``import utils`` resolve to whichever landed first. The
collision is silent: the wrong ``utils`` has the same name and different contents, so the
import succeeds and the method changes.

:func:`upstream_root` isolates one import: it removes any cached ``utils*`` modules, puts
one root at the front of ``sys.path``, and restores both on exit. Module objects the
imported code already bound stay bound, so the restore is safe.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import sys
import types
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

#: Top-level package names the vendored baselines collide on. Both CDiffMR and FDB ship a
#: ``utils`` package; neither is importable as a distinct name from outside its own root.
_COLLIDING_TOP_LEVEL = ("utils",)

#: CDiffMR's radial and spiral mask builders import OpenCV at module scope, and importing
#: any of them pulls the whole ``utils_kspace_undersampling`` package. The published
#: option file selects ``cartesian_random``, which touches none of them -- so OpenCV is a
#: dependency of code this repository never calls.
_OPTIONAL_AT_IMPORT = ("cv2",)


class _AbsentModule(types.ModuleType):
    """A placeholder that raises on first use instead of pretending to work.

    Absent is a state to report, never a state to infer (non-negotiable 18). A bare
    ``MagicMock`` here would let a radial-mask code path run and return nonsense; this
    raises with the reason and the name that was wanted.
    """

    def __getattr__(self, name: str) -> object:
        raise ImportError(
            f"{self.__name__}.{name} was requested while importing a vendored baseline. "
            f"{self.__name__} is not installed in this environment, and it is only needed "
            "by upstream code paths this repository does not select. Install it if you "
            "are adding one of those paths."
        )


def _absent_optional_modules() -> dict[str, types.ModuleType]:
    """Placeholders for optional third-party modules, one probe per package."""
    placeholders: dict[str, types.ModuleType] = {}
    for name in _OPTIONAL_AT_IMPORT:
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            continue
        logger.debug("[baselines] %s absent; installing a raise-on-use placeholder", name)
        placeholders[name] = _AbsentModule(name)
    return placeholders


@contextlib.contextmanager
def upstream_root(root: Path) -> Iterator[None]:
    """Import from ``root`` with its top-level packages isolated from the rest.

    Args:
        root: The vendored repository root to put on ``sys.path``.
    """
    saved_path = list(sys.path)
    # Prepending is NOT enough. Shen's ``utils`` has no ``__init__.py``, so it is a
    # namespace portion: Python records it and keeps scanning, and FDB's ``utils`` --
    # a regular package further down the path -- wins. The other vendored roots come
    # OFF the path for the duration, or `import utils.sample_mask` reads FDB's package
    # and raises for a file that is plainly on disk.
    vendored = root.parent
    sys.path[:] = [
        entry
        for entry in sys.path
        if not (entry and Path(entry).parent == vendored and Path(entry) != root)
    ]
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == top or name.startswith(f"{top}.") for top in _COLLIDING_TOP_LEVEL)
    }
    for name in list(saved_modules):
        del sys.modules[name]

    placeholders = _absent_optional_modules()
    sys.modules.update(placeholders)
    sys.path.insert(0, str(root))
    # Without this the finder for a directory already walked under a DIFFERENT root
    # answers from its cache, and `import utils.x` misses a module that is plainly on
    # disk -- observed as `No module named 'utils.sample_mask'` with the file present.
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path[:] = saved_path
        importlib.invalidate_caches()
        for name in [
            n
            for n in sys.modules
            if any(n == top or n.startswith(f"{top}.") for top in _COLLIDING_TOP_LEVEL)
        ]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
        for name in placeholders:
            if sys.modules.get(name) is placeholders[name]:
                del sys.modules[name]


def load_isolated_module(root: Path, module_path: Path, name: str) -> types.ModuleType:
    """Execute one upstream source file with ``root`` isolated on ``sys.path``.

    Args:
        root: The vendored repository root the file's own imports are written against.
        module_path: The file to execute.
        name: The name to register the resulting module under.

    Returns:
        The executed module.
    """
    with upstream_root(root):
        spec = importlib.util.spec_from_file_location(name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot build an import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module
