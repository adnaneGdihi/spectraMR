"""Point-to-point transport of a pipeline payload between adjacent stages.

A payload is a tuple of ``Tensor | None``. The receiver cannot know its shapes, so
the sender first posts one fixed-size ``int64`` header per forward describing every
entry of a microbatch (all microbatches of a step share it), then the tensors.
Complex entries travel as their real view and bool as ``uint8``: the wire format is
real-valued on every backend. Gradients flow back over the same pairs for the
entries that carried ``requires_grad`` on the sending side.

The header and any entry held on the host while the stage runs on an accelerator (a
model's shape bookkeeping) travel over a host-side ``meta_group`` (gloo), so reading
them never waits on the device stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "EntrySpec",
    "Posted",
    "decode_header",
    "empty_wire",
    "encode_header",
    "from_wire",
    "is_host",
    "new_header",
    "recv_grads",
    "recv_payload",
    "send_grads",
    "send_payload",
    "wire",
]

_HEADER_LEN = 128
_MAX_DIMS = 6
_FIELDS = 6 + _MAX_DIMS  # present, dtype, requires_grad, batched, host, ndim, dims...
_DTYPES = (
    torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64, torch.int32,
    torch.uint8, torch.bool, torch.complex64, torch.complex128,
)  # fmt: skip


@dataclass(frozen=True)
class EntrySpec:
    """One payload entry as it is on the sender (dtype and shape before the wire view)."""

    dtype: torch.dtype
    shape: tuple[int, ...]
    requires_grad: bool
    batched: bool
    host: bool = False


def is_host(entry: torch.Tensor, device: torch.device) -> bool:
    """An entry kept on the host while the stage runs elsewhere; it stays on the host."""
    return entry.device.type == "cpu" and device.type != "cpu"


def encode_header(
    payload: tuple[Any, ...], batched: tuple[bool, ...], microbatches: int, device: torch.device
) -> torch.Tensor:
    """Describe one microbatch's payload for the receiver, as a host tensor for ``meta_group``."""
    if 2 + len(payload) * _FIELDS > _HEADER_LEN:
        raise ValueError(f"a pipeline payload of {len(payload)} entries exceeds the header.")
    header = torch.zeros(_HEADER_LEN, dtype=torch.int64)
    header[0], header[1] = len(payload), microbatches
    for i, entry in enumerate(payload):
        base = 2 + i * _FIELDS
        if entry is None:
            continue
        if entry.ndim > _MAX_DIMS:
            raise ValueError(f"payload entry {i} has {entry.ndim} dims; at most {_MAX_DIMS}.")
        host = is_host(entry, device)
        if host and entry.requires_grad:
            raise ValueError(
                f"payload entry {i} is on the host and requires grad; its gradient has no route "
                "back to the sending stage. Move it to the stage's device."
            )
        fields = [1, _DTYPES.index(entry.dtype), int(entry.requires_grad), int(batched[i]),
                  int(host), entry.ndim, *entry.shape]  # fmt: skip
        header[base : base + len(fields)] = torch.tensor(fields, dtype=torch.int64)
    return header


def new_header() -> torch.Tensor:
    """A host receive buffer for :func:`encode_header`'s output."""
    return torch.empty(_HEADER_LEN, dtype=torch.int64)


def decode_header(header: torch.Tensor) -> tuple[list[EntrySpec | None], int]:
    values = header.tolist()  # a host tensor: no device sync
    specs: list[EntrySpec | None] = []
    for i in range(values[0]):
        base = 2 + i * _FIELDS
        present, dtype, grad, batched, host, ndim = values[base : base + 6]
        if not present:
            specs.append(None)
            continue
        shape = tuple(values[base + 6 : base + 6 + ndim])
        specs.append(EntrySpec(_DTYPES[dtype], shape, bool(grad), bool(batched), bool(host)))
    return specs, values[1]


def wire(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_complex():
        return torch.view_as_real(tensor.detach().resolve_conj()).contiguous()
    if tensor.dtype == torch.bool:
        return tensor.detach().to(torch.uint8).contiguous()
    return tensor.detach().contiguous()


def empty_wire(spec: EntrySpec, device: torch.device) -> torch.Tensor:
    if spec.dtype.is_complex:
        real = torch.float32 if spec.dtype == torch.complex64 else torch.float64
        return torch.empty((*spec.shape, 2), dtype=real, device=device)
    dtype = torch.uint8 if spec.dtype == torch.bool else spec.dtype
    return torch.empty(spec.shape, dtype=dtype, device=device)


def from_wire(buffer: torch.Tensor, spec: EntrySpec) -> torch.Tensor:
    if spec.dtype.is_complex:
        return torch.view_as_complex(buffer)
    return buffer.bool() if spec.dtype == torch.bool else buffer


class Posted:
    """An ``isend`` and the buffer it reads, held together until ``wait``."""

    def __init__(self, tensor: torch.Tensor, dst: int, group: Any):
        self.tensor = tensor
        self.work = dist.isend(tensor, dst=dst, group=group)

    def wait(self) -> None:
        self.work.wait()


def send_payload(
    payload: tuple[Any, ...], dst: int, group: Any, meta_group: Any, device: torch.device
) -> list[Posted]:
    """Post every present entry to ``dst``: host entries over ``meta_group``, the rest from
    ``device`` over ``group`` (NCCL sends device tensors only)."""
    return [
        Posted(wire(t), dst, meta_group) if is_host(t, device) else Posted(wire(t), dst, group)
        for t in payload
        if t is not None
    ]


def recv_payload(
    specs: list[EntrySpec | None],
    src: int,
    group: Any,
    meta_group: Any,
    device: torch.device,
    grad: bool,
) -> tuple[tuple[Any, ...], list[torch.Tensor]]:
    """Receive one microbatch. Returns the payload and its grad-carrying leaves, in order."""
    payload: list[Any] = []
    leaves: list[torch.Tensor] = []
    for spec in specs:
        if spec is None:
            payload.append(None)
            continue
        if spec.host:
            buffer = empty_wire(spec, torch.device("cpu"))
            dist.recv(buffer, src=src, group=meta_group)
            payload.append(from_wire(buffer, spec))
            continue
        buffer = empty_wire(spec, device)
        dist.recv(buffer, src=src, group=group)
        if grad and spec.requires_grad:
            buffer.requires_grad_(True)
            leaves.append(buffer)
        payload.append(from_wire(buffer, spec))
    return tuple(payload), leaves


def send_grads(leaves: list[torch.Tensor], dst: int, group: Any) -> list[Posted]:
    """Return the gradients of the received leaves; an unused leaf sends zeros."""
    grads = [leaf.grad if leaf.grad is not None else torch.zeros_like(leaf) for leaf in leaves]
    return [Posted(g.contiguous(), dst, group) for g in grads]


def recv_grads(outputs: list[torch.Tensor], src: int, group: Any) -> list[torch.Tensor]:
    """Receive the gradient of every grad-carrying output this stage sent, in order."""
    grads = []
    for out in outputs:
        if out.is_complex():
            buffer = torch.empty((*out.shape, 2), dtype=out.real.dtype, device=out.device)
        else:
            buffer = torch.empty(out.shape, dtype=out.dtype, device=out.device)
        dist.recv(buffer, src=src, group=group)
        grads.append(torch.view_as_complex(buffer) if out.is_complex() else buffer)
    return grads
