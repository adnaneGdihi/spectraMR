"""Environment-aware path resolution for configuration.

Resolves configuration paths using environment variables with fallbacks,
enabling portable configuration across cluster and local environments.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

#: Example path used in remedies. ``/tmp`` because it is node-local everywhere
#: (Triton's cache must stay off NFS -- DeepSpeed hangs at rank exit otherwise);
#: ``$USER`` because ``/tmp`` is sticky and shared, so a bare ``/tmp/<name>``
#: belongs to whoever created it first.
_EXAMPLE_ROOT = '"/tmp/$USER/spectramr_cache"'


class CacheDirectoryError(OSError):
    """A cache directory could not be created, with the knob that owns it named.

    Subclasses ``OSError`` so callers already catching ``OSError`` keep working.
    """


def _create_cache_dir(var: str, value: str, *, framework_value: str | None) -> None:
    """Create one cache directory, or raise saying which knob to change.

    Created **eagerly** so a denied path fails here, naming the directory,
    rather than inside a third-party import that reports only
    ``PermissionError(13, 'Permission denied')``. Naming the path was already
    true; what the message adds are the two facts that decide what to *do*:
    who chose the value -- the framework (derived from ``cache_root``) or an
    inherited variable kept by the ``setdefault`` carve-out -- and which knob
    moves it, since only ``SPECTRAMR_CACHE_ROOT`` (whole block) and
    ``TRITON_CACHE_DIR`` (Triton alone) are settable from a job script while
    the other four are overwritten (:data:`_CACHE_ENV_LAYOUT`). A 4-rank job
    died on ``Permission denied: '/triton_cache'`` with neither fact available
    -- a path this module cannot compute, so the obvious reading ("the
    framework put my cache at ``/``") was wrong.

    Args:
        var: Environment variable naming the directory.
        value: The path now in effect for ``var``.
        framework_value: What this module *would* have chosen, or ``None`` when
            it is not a member of the cache block (the root itself). When it
            differs from ``value``, the value was inherited from the
            environment and the remedy is a different one.

    Raises:
        CacheDirectoryError: The directory does not exist and cannot be created.
    """
    try:
        Path(value).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if framework_value is None:
            # ``var`` names the source that produced the root, not a variable
            # whose value is this path -- two of the three are not even env vars.
            raise CacheDirectoryError(
                f"Cannot create the cache root {value!r} (resolved from {var}): "
                f"{exc.strerror or exc}.\n"
                "Every other cache directory hangs off this one, so nothing else "
                "was attempted.\n"
                "Point the cache root somewhere writable:\n"
                f"    export SPECTRAMR_CACHE_ROOT={_EXAMPLE_ROOT}"
            ) from exc
        if value != framework_value:
            provenance = (
                f"{var} was INHERITED from the environment -- it is the one cache "
                "variable applied with setdefault rather than assignment (DeepSpeed "
                "asks operators to point Triton at a non-NFS path), so this value "
                "was NOT chosen by the framework."
            )
            remedy = (
                f"Either unset {var} to use the framework's own path "
                f"({framework_value}), or export it somewhere writable:\n"
                f'    export {var}="/tmp/$USER/triton_cache"\n'
                "A value whose first component is empty (e.g. '/triton_cache') is "
                'usually an unset ${PREFIX} in "${PREFIX}/triton_cache".'
            )
        else:
            provenance = (
                f"{var} is the framework's own path under the resolved cache "
                "root -- nothing in the environment overrode it."
            )
            remedy = (
                "Point the whole cache block somewhere writable. "
                "SPECTRAMR_CACHE_ROOT is the control point: exporting any member "
                "other than TRITON_CACHE_DIR is overwritten from the root.\n"
                f"    export SPECTRAMR_CACHE_ROOT={_EXAMPLE_ROOT}"
            )
        raise CacheDirectoryError(
            f"Cannot create the cache directory {value!r} for {var}: "
            f"{exc.strerror or exc}.\n{provenance}\n{remedy}"
        ) from exc


def resolve_data_root() -> Path:
    """Resolve data root directory from config or environment.

    Resolution order:
    1. SPECTRAMR_DATA_ROOT environment variable (explicit user override)
    2. Default to ~/.cache/spectramr/data (local machine)
    3. Create directory if it doesn't exist

    Returns:
        Path object for data root directory

    Example:
        >>> # User can set: SPECTRAMR_DATA_ROOT=/mnt/datasets/spectramr
        >>> # Or leave default: ~/.cache/spectramr/data
        >>> root = resolve_data_root()
        >>> print(root)  # /home/user/.cache/spectramr/data
    """
    env_path = os.environ.get("SPECTRAMR_DATA_ROOT")

    if env_path:
        data_root = Path(env_path)
        logger.info(f"Data root from SPECTRAMR_DATA_ROOT: {data_root}")
    else:
        # Fallback to local cache directory
        data_root = Path.home() / ".cache" / "spectramr" / "data"
        logger.info(f"Data root using default cache: {data_root}")

    # Ensure directory exists
    data_root.mkdir(parents=True, exist_ok=True)

    return data_root


def resolve_cache_root() -> Path:
    """Resolve torch/cuda cache directory from environment.

    Resolution order:
    1. SPECTRAMR_CACHE_ROOT environment variable (explicit user override)
    2. TMPDIR environment variable (cluster standard)
    3. Default to ~/.cache/spectramr/torch (local machine)

    Returns:
        Path object for cache root directory

    Example:
        >>> # Cluster: TMPDIR=/tmp/cluster_scratch
        >>> # Local: uses ~/.cache/spectramr/torch
        >>> cache = resolve_cache_root()
    """
    cache_path = os.environ.get("SPECTRAMR_CACHE_ROOT")

    if cache_path:
        cache_root = Path(cache_path)
        source = "SPECTRAMR_CACHE_ROOT"
        logger.info(f"Cache root from SPECTRAMR_CACHE_ROOT: {cache_root}")
    elif os.environ.get("TMPDIR"):
        # Use cluster TMPDIR if available
        cache_root = Path(os.environ.get("TMPDIR")) / "spectramr_cache"
        source = "TMPDIR"
        logger.info(f"Cache root from TMPDIR: {cache_root}")
    else:
        # Fallback to local cache
        cache_root = Path.home() / ".cache" / "spectramr" / "torch"
        source = "the ~/.cache default"
        logger.info(f"Cache root using default local: {cache_root}")

    # An unwritable root has to name itself AND which of the three sources above
    # produced it, or the operator is left guessing which knob moves it.
    _create_cache_dir(source, str(cache_root), framework_value=None)
    # Pin the resolved root so a SECOND call returns the same path.
    # ``main.py`` sets ``TMPDIR = <resolved root>`` right after the first call,
    # and ``accelerator.initialize_accelerator`` then calls this again — the
    # TMPDIR branch re-appended the suffix and produced
    # ``$TMPDIR/spectramr_cache/spectramr_cache``, so torch/CUDA caches landed under
    # a different root than the one main.py had already exported. Stamping the
    # answer makes resolution idempotent no matter who mutates TMPDIR after us,
    # and it is the documented override knob, so nothing downstream changes.
    os.environ["SPECTRAMR_CACHE_ROOT"] = str(cache_root)
    return cache_root


#: Cache variables pointed at ``cache_root``, and whether an existing value wins.
#:
#: ``SPECTRAMR_CACHE_ROOT`` is the documented control point for the first five, so
#: they are *overwritten*: an operator redirects the whole set by moving the root,
#: not by exporting one member of it. Leaving them as ``setdefault`` would also
#: defeat the purpose on the machines that need this most -- a cluster profile
#: exporting ``XDG_CACHE_HOME=$HOME/.cache`` would keep torch's JIT extension
#: build root inside ``$HOME``, which is the failure this table exists to prevent.
#:
#: ``TRITON_CACHE_DIR`` is the one exception and keeps ``setdefault`` semantics:
#: DeepSpeed's own startup warning asks operators to point it at a non-NFS path,
#: so an explicit export is an informed choice about NFS behaviour that must win.
#:
#: ``TORCHINDUCTOR_CACHE_DIR`` is pinned rather than inherited. Inductor's
#: ``default_cache_dir`` is ``tempfile.gettempdir() + "torchinductor_<user>"``, so
#: it *appears* to follow ``TMPDIR`` above -- but ``gettempdir`` memoizes on first
#: call, and anything that touches it before this function runs freezes the answer
#: at ``/tmp``. Measured: with one ``tempfile.gettempdir()`` beforehand, the env
#: var reads ``<cache_root>`` while Inductor still writes to
#: ``/tmp/torchinductor_<user>``. A coupling that silently stops holding is not
#: coverage, and compiled artifacts are exactly what must not land on a node-local
#: ``/tmp`` that the next job cannot see.
#:
#: ``CUDA_CACHE_PATH`` is the PTX JIT cache and the NVIDIA-documented spelling.
#: ``CUDA_CACHE_CONFIG`` below it is NOT one of NVIDIA's three
#: (``CUDA_CACHE_DISABLE`` / ``CUDA_CACHE_PATH`` / ``CUDA_CACHE_MAXSIZE``) and is
#: kept only because retiring it is a separate decision -- see #2178. Without
#: ``CUDA_CACHE_PATH`` the cache stays at ``~/.nv/ComputeCache`` (measured: 17
#: files there on a developer box that had this table applied).
_CACHE_ENV_LAYOUT: tuple[tuple[str, str, bool], ...] = (
    # (env var, subdirectory of cache_root, overwrite an existing value)
    ("TMPDIR", "", True),
    ("TORCH_HOME", "torch_cache", True),
    ("TORCH_METRICS_CACHE", "torchmetrics_cache", True),
    ("XDG_CACHE_HOME", ".cache", True),
    ("CUDA_CACHE_CONFIG", "cuda_cache", True),
    ("CUDA_CACHE_PATH", "cuda_cache", True),
    ("TORCHINDUCTOR_CACHE_DIR", "inductor_cache", True),
    ("TRITON_CACHE_DIR", "triton_cache", False),
)


def configure_cache_environment(cache_root: Path | None = None) -> dict[str, str]:
    """Point every library cache at ``cache_root`` instead of ``$HOME``.

    This is the SSOT for that block. It previously lived inline in
    ``spectramr/main.py`` and was therefore applied on the ``spectramr train`` path
    **only** -- ``torchrun -m spectramr.cli train-distributed`` (the documented
    multi-GPU entry point, ``scripts/training/train_distributed.sbatch:173``)
    never imports ``main.py``, and ``initialize_accelerator`` back-filled three
    of the six. Measured on a clean process before this function existed::

        import spectramr.cli   ->  TRITON_CACHE_DIR=None  XDG_CACHE_HOME=None
        import spectramr.main  ->  both set under cache_root

    The two that were missing are the two that bite. Triton does not read
    ``XDG_CACHE_HOME`` and defaults to ``~/.triton``; torch's JIT extension build
    root resolves through ``torch._appdirs.user_cache_dir('torch_extensions')``,
    which *does* follow ``XDG_CACHE_HOME`` and so fell back to
    ``~/.cache/torch_extensions``. ``import deepspeed`` writes to both while the
    module is still executing, so a home directory that is over quota or not
    writable from a compute node surfaces as an opaque third-party ``OSError``
    before any config is read or any device is selected.

    Both variables are read lazily (at kernel compile / extension load), so
    unlike ``PYTORCH_CUDA_ALLOC_CONF`` this call does not have to precede
    ``import torch`` -- it only has to precede ``import deepspeed``.

    Args:
        cache_root: Root to hang the caches off. Resolved via
            :func:`resolve_cache_root` when omitted.

    Returns:
        Mapping of env var to the value now in effect, including any pre-existing
        ``TRITON_CACHE_DIR`` this call deliberately left alone.
    """
    root = resolve_cache_root() if cache_root is None else Path(cache_root)

    applied: dict[str, str] = {}
    for var, subdir, overwrite in _CACHE_ENV_LAYOUT:
        target = root / subdir if subdir else root
        if overwrite:
            os.environ[var] = str(target)
        else:
            os.environ.setdefault(var, str(target))
        applied[var] = os.environ[var]
        # Create eagerly: a cache root that cannot be written to should fail here,
        # naming the directory, rather than inside a third-party import that
        # reports only ``PermissionError(13, 'Permission denied')``. ``target`` is
        # passed so the failure can also say whether the value was ours or the
        # environment's -- only ``TRITON_CACHE_DIR`` can differ.
        _create_cache_dir(var, applied[var], framework_value=str(target))

    return applied


def resolve_device() -> str:
    """Resolve device from environment variable or default.

    Resolution order:
    1. SPECTRAMR_DEVICE environment variable (explicit user override)
    2. Default to "auto" — resolved by the accelerated-run contract in
       :mod:`spectramr.core.compute_device`, which picks CUDA when present and
       RAISES for a heavy pipeline when no accelerator exists. The old "cpu"
       default was a silent downgrade dressed up as a "safe" fallback.
    3. Validate device is one of: cuda, cpu, auto

    Returns:
        Device string: "cuda", "cpu", or "auto"

    Raises:
        ValueError: If device is invalid

    Example:
        >>> # Override: SPECTRAMR_DEVICE=cuda
        >>> # Or default: cpu (requires explicit override to use GPU)
        >>> device = resolve_device()
        >>> print(device)  # "cpu" or "cuda" depending on env
    """
    device = os.environ.get("SPECTRAMR_DEVICE", "auto").lower()

    valid_devices = {"cuda", "cpu", "auto"}
    if device not in valid_devices:
        raise ValueError(
            f"Invalid device '{device}'. Must be one of: {valid_devices}. "
            f"Set via SPECTRAMR_DEVICE environment variable."
        )

    logger.info(f"Device from SPECTRAMR_DEVICE or default: {device}")
    return device


__all__ = [
    "CacheDirectoryError",
    "configure_cache_environment",
    "resolve_cache_root",
    "resolve_data_root",
    "resolve_device",
]
