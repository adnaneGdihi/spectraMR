"""Can this device be compiled on, and with what backend.

Split out of :mod:`spectramr.core.device_capabilities` to keep both files under
the 300-line ceiling (non-negotiable 20). The seam is the question each answers:
this one is about ``torch.compile``'s requirements, the other about the device
itself. ``device_capabilities`` imports this; nothing here imports it back.

This owns **only the device cross-check**. The backend *vocabulary* belongs to
``CompileConfigSchema._validate_backend``, which validates against
``torch._dynamo.list_backends()`` at config-load time — the same split
``resolve_distributed_backend`` uses between the ``DistBackend`` Literal and its
own ``_CUDA_ONLY`` check.
"""

from __future__ import annotations

__all__ = [
    "CompileUnsupportedError",
    "compile_backend_for",
    "triton_available",
]

_COMPILE_BACKEND = "inductor"

#: Device types whose Inductor backend needs Triton to emit kernels. Inductor on
#: CPU uses its C++/OpenMP path instead, so the requirement is per-device rather
#: than per-backend.
_TRITON_BACKED = frozenset({"cuda"})

_KNOWN_DEVICE_TYPES = frozenset({"cuda", "cpu", "mps"})


class CompileUnsupportedError(RuntimeError):
    """Compilation was requested where this device cannot support it."""


def compile_backend_for(device_type: str, *, triton: bool) -> str:
    """The ``torch.compile`` backend for *device_type*, or raise.

    Raises rather than returning a fallback: a run that asked to be compiled and
    silently was not reports throughput belonging to a different configuration —
    the reasoning that made ``ModelBuilder.compile`` stop swallowing failures.
    """
    if device_type not in _KNOWN_DEVICE_TYPES:
        raise ValueError(
            f"No torch.compile backend is defined for device type {device_type!r}. "
            f"Known types: {sorted(_KNOWN_DEVICE_TYPES)}."
        )
    if device_type in _TRITON_BACKED and not triton:
        raise CompileUnsupportedError(
            f"torch.compile on {device_type!r} needs Triton to generate kernels, and it is "
            "not importable. Triton ships with the CUDA torch wheels, so its absence "
            "usually means a CPU-only build. Install a CUDA wheel, or set "
            "optimization.compile.enabled: false to run eager on purpose."
        )
    return _COMPILE_BACKEND


def triton_available() -> bool:
    """Whether Triton can emit device kernels in this process.

    The SSOT for this question. ``models/blocks/triton_scan.py`` imports it
    rather than repeating the probe — that module held the original copy, and
    two of them would disagree the moment one grew a condition.

    Both terms are required: Triton imports fine on a CPU-only box but has no
    device to compile for, so an import check alone answers the wrong question.
    ``find_spec`` avoids importing Triton as a side effect of asking whether it
    is there.
    """
    import importlib.util

    if importlib.util.find_spec("triton") is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # torch absent, or the CI shim
        return False
