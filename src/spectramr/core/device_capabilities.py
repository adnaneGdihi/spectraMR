"""Device capability SSOT — what the resolved device can actually do.

:mod:`spectramr.core.compute_device` answers *which* device a run lands on. This
answers *what that device is capable of*, which had no owner: compute capability
was captured into provenance (``logging/provenance.py``) and never consumed by a
decision, while the choices that depend on it — the AMP dtype and the
``torch.compile`` backend — were made from static tables with no hardware term.

**The bfloat16 trap is why this exists.** ``torch.cuda.is_bf16_supported()``
takes ``including_emulation=True`` by default, and that branch merely checks a
bf16 tensor can be *created*. Measured on sm_75::

    is_bf16_supported()                           -> True
    is_bf16_supported(including_emulation=False)  -> False

The target clusters are V100s (sm_70), which is why torch is pinned to the cu126
lane at all. A gate written against the default would therefore *confirm* bf16 on
the exact hardware this project runs on — worse than no gate, because it leaves a
paper trail arguing the configuration was checked. So the test here is the one
torch itself applies before reaching its emulation branch:
``get_device_capability() >= (8, 0)``. That also keeps the probe independent of
the torch version that added the keyword.

Same split as ``compute_device``: the policy is **pure functions** of the
capability, so every branch is unit-testable with no GPU (CI has none, and
``tests/conftest.py`` installs a torch ``MagicMock``). Torch is imported lazily,
by the shell only. What the shell cannot resolve is reported in
:attr:`DeviceCapabilities.incomplete` rather than guessed — absent is a state to
report, never a state to infer.

Deliberately **not** here: ``communication_backend_name()``. DeepSpeed's
accelerator exposes one, but ``infrastructure/distributed/backend.py``
(``resolve_distributed_backend``) already owns that decision and cross-checks it
against the resolved device. A second owner is non-negotiable 17 head-on — do not
add it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from spectramr.core import env_names
from spectramr.core.compile_capability import (
    CompileUnsupportedError,
    compile_backend_for,
    triton_available,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_NATIVE_BF16_CAPABILITY",
    "TARGET_CAPABILITY_ENV",
    "DeviceCapabilities",
    "build_capabilities",
    "native_bf16_supported",
    "parse_compute_capability",
    "probe_device_capabilities",
    "supported_amp_dtypes",
    "target_capability_from_env",
]

#: Compute capability at which bfloat16 stops being emulated. Ampere (sm_80) and
#: later have native bf16 tensor cores; Volta (sm_70) and Turing (sm_75) do not.
#: This is the threshold ``torch.cuda.is_bf16_supported`` itself tests before
#: falling through to its tensor-creation probe.
MIN_NATIVE_BF16_CAPABILITY = (8, 0)

#: Declares the capability of the machine the run will land on, for the case the
#: audit cannot see it: a login node legitimately differs from a compute node.
#: A live probe always wins over this. Aliased from ``env_names`` rather than
#: respelled, so the registry stays the one place the name exists.
TARGET_CAPABILITY_ENV = env_names.SPECTRAMR_TARGET_COMPUTE_CAPABILITY

# ── pure policy ──────────────────────────────────────────────────────


def parse_compute_capability(raw: str | None) -> tuple[int, int] | None:
    """``"8.9"`` -> ``(8, 9)``. Empty means unset; anything else **raises**.

    Validated rather than coerced, for the reason ``cpu_opt_in_from_env`` gives:
    a knob that silently accepts nonsense is indistinguishable from one that
    works (pitfall #15). ``"sm_70"``, ``"8"`` and ``"8.x"`` are all rejected.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    major, sep, minor = text.partition(".")
    if not sep or not major.isdigit() or not minor.isdigit():
        raise ValueError(
            f"{TARGET_CAPABILITY_ENV}={raw!r} is not a compute capability. "
            'Use "<major>.<minor>", e.g. "7.0" for a V100 or "8.9" for an L40S.'
        )
    return int(major), int(minor)


def target_capability_from_env(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int] | None:
    """The declared target capability, or ``None`` when unset."""
    import os

    env = os.environ if environ is None else environ
    return parse_compute_capability(env.get(TARGET_CAPABILITY_ENV))


def native_bf16_supported(capability: tuple[int, int] | None) -> bool:
    """Whether *capability* has **native** bfloat16.

    ``None`` is not evidence of support, so it resolves to ``False`` — the
    caller degrades toward the safe dtype rather than the fast one. Callers that
    must distinguish "no" from "cannot tell" read
    :attr:`DeviceCapabilities.incomplete`.
    """
    return capability is not None and capability >= MIN_NATIVE_BF16_CAPABILITY


def supported_amp_dtypes(capability: tuple[int, int] | None) -> tuple[str, ...]:
    """AMP dtypes runnable without emulation, spelled as the schema spells them.

    ``float32`` is always present because it denotes the *absence* of autocast,
    not a hardware feature.
    """
    dtypes = ["float32", "float16"]
    if native_bf16_supported(capability):
        dtypes.append("bfloat16")
    return tuple(dtypes)


# ── the record ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class DeviceCapabilities:
    """Resolved capabilities, plus what could not be resolved.

    ``incomplete`` is what lets a caller tell "this device has no native bf16"
    from "we could not find out" — the distinction the audit needs in order to
    stay honest when it runs somewhere other than the compute node.
    """

    device_type: str
    capability: tuple[int, int] | None
    native_bf16: bool
    triton: bool
    compile_backend: str | None
    amp_dtypes: tuple[str, ...]
    source: str
    incomplete: tuple[str, ...] = field(default=())

    @property
    def capability_str(self) -> str | None:
        """``"8.9"`` — the spelling ``provenance.torch_runtime`` already uses."""
        return None if self.capability is None else f"{self.capability[0]}.{self.capability[1]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "capability": self.capability_str,
            "native_bf16": self.native_bf16,
            "triton": self.triton,
            "compile_backend": self.compile_backend,
            "amp_dtypes": list(self.amp_dtypes),
            "source": self.source,
            "incomplete": list(self.incomplete),
        }


def build_capabilities(
    device_type: str,
    capability: tuple[int, int] | None,
    *,
    triton: bool,
    source: str,
) -> DeviceCapabilities:
    """Assemble a record from already-probed facts. Pure, so it is testable."""
    incomplete: list[str] = []
    if device_type == "cuda" and capability is None:
        incomplete.append("capability")
    try:
        backend: str | None = compile_backend_for(device_type, triton=triton)
    except (ValueError, CompileUnsupportedError):
        backend = None
        incomplete.append("compile_backend")
    return DeviceCapabilities(
        device_type=device_type,
        capability=capability,
        native_bf16=native_bf16_supported(capability),
        triton=triton,
        compile_backend=backend,
        amp_dtypes=supported_amp_dtypes(capability),
        source=source,
        incomplete=tuple(incomplete),
    )


# ── lazy shell ───────────────────────────────────────────────────────


def _probe_capability(index: int) -> tuple[int, int] | None:
    """Live compute capability, or ``None`` when it cannot be read.

    ``get_device_capability`` rather than ``get_device_properties``: it returns
    the pair directly, and never ``is_bf16_supported``, whose default branch
    allocates a tensor to answer and reports emulation as support.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(index)
    except Exception as exc:
        logger.debug("Could not read compute capability for device %s: %s", index, exc)
        return None
    if not isinstance(major, int) or not isinstance(minor, int):
        return None  # a shimmed torch hands back mocks
    return major, minor


def probe_device_capabilities(
    device: Any = None,
    *,
    index: int = 0,
    environ: Mapping[str, str] | None = None,
) -> DeviceCapabilities:
    """Probe *device*, falling back to the declared target when it is not visible.

    Never raises: this is the reporting path, and a provenance collector that
    dies on an unreadable device loses the whole record. The decision paths
    (:func:`compile_backend_for`, the AMP gate) raise instead — reporting and
    deciding have different obligations.
    """
    device_type = _device_type_of(device)
    triton = triton_available()

    capability = _probe_capability(index) if device_type == "cuda" else None
    source = f"probe:{device_type}:{index}" if capability is not None else "unknown"

    if capability is None:
        try:
            declared = target_capability_from_env(environ)
        except ValueError as exc:
            logger.warning("%s ignored: %s", TARGET_CAPABILITY_ENV, exc)
            declared = None
        if declared is not None:
            capability, source = declared, f"env:{TARGET_CAPABILITY_ENV}"

    return build_capabilities(device_type, capability, triton=triton, source=source)


def _device_type_of(device: Any) -> str:
    """``torch.device`` / str / ``None`` -> a bare device-type string."""
    if device is not None:
        type_attr = getattr(device, "type", None)
        return type_attr if isinstance(type_attr, str) else str(device).split(":", 1)[0]
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
