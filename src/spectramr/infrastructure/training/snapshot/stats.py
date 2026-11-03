"""Per-tensor statistics and the JSON helpers of the snapshot manifest."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class TensorReport:
    """Per-tensor diagnostic record."""

    name: str
    shape: list[int]
    dtype: str
    device: str
    is_complex: bool
    even_channel_count: bool  # heuristic for real-stacked complex
    nan_count: int
    inf_count: int
    finite_count: int
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    abs_max: float | None
    note: str = ""  # human-readable interpretation


def tensor_report(name: str, t: torch.Tensor) -> TensorReport:
    """Compute a one-shot diagnostic for a single tensor.

    Statistics are computed in fp32 even for bf16 / fp16 inputs to
    avoid silent overflow in mean/std on very large k-space tensors.
    """
    if not isinstance(t, torch.Tensor):
        return TensorReport(
            name=name,
            shape=[],
            dtype=str(type(t).__name__),
            device="cpu",
            is_complex=False,
            even_channel_count=False,
            nan_count=0,
            inf_count=0,
            finite_count=0,
            min=None,
            max=None,
            mean=None,
            std=None,
            abs_max=None,
            note="not a torch.Tensor",
        )

    with torch.no_grad():
        x = t.detach()
        is_cx = x.is_complex()
        x_view = x.abs().float() if is_cx else x.float()

        n_total = x_view.numel()
        nan_count = int(torch.isnan(x_view).sum().item())
        inf_count = int(torch.isinf(x_view).sum().item())
        finite = n_total - nan_count - inf_count

        if finite > 0:
            x_finite = x_view[torch.isfinite(x_view)]
            mn = float(x_finite.min().item())
            mx = float(x_finite.max().item())
            mu = float(x_finite.mean().item())
            sd = float(x_finite.std().item()) if x_finite.numel() > 1 else 0.0
            am = float(x_finite.abs().max().item())
        else:
            mn = mx = mu = sd = am = None

        even_ch = x.dim() >= 4 and x.shape[1] % 2 == 0 and not is_cx

        # Annotate the most common diagnostics
        notes: list[str] = []
        if nan_count > 0:
            notes.append(f"WARNING {nan_count} NaN")
        if inf_count > 0:
            notes.append(f"WARNING {inf_count} Inf")
        if mn is not None and mx is not None:
            if abs(mx - mn) < 1e-12:
                notes.append("WARNING constant tensor (max - min < 1e-12)")
            if mn < 0 and not is_cx and x.dim() == 4 and x.shape[1] == 1:
                notes.append("negative values present (signed image)")
        if even_ch and x.dim() == 4 and x.shape[1] in (2, 4, 8, 16):
            notes.append(f"{x.shape[1]}-ch real-stacked -> likely complex (R0,I0,R1,I1,...)")

        return TensorReport(
            name=name,
            shape=list(x.shape),
            dtype=str(x.dtype),
            device=str(x.device),
            is_complex=is_cx,
            even_channel_count=even_ch,
            nan_count=nan_count,
            inf_count=inf_count,
            finite_count=finite,
            min=mn,
            max=mx,
            mean=mu,
            std=sd,
            abs_max=am,
            note="; ".join(notes),
        )


#: Stamped in place of a deferred value whose callable raised. The key SURVIVES
#: with a reason rather than vanishing: a diagnostic that silently disappears is
#: indistinguishable from one that was never requested (pitfall #16).
_UNRESOLVED_EXTRA = "<unresolved: {0}>"


def resolve_deferred_extra(
    extra: dict[str, Any] | Callable[[], dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Invoke any zero-arg callable in ``extra``; the writer calls it on a snapshot step only.

    Accepts a callable returning the dict (one host sync feeds every value) or a dict
    with callable values. A callable that raises stamps :data:`_UNRESOLVED_EXTRA`, so a
    diagnostic never breaks the step and never silently vanishes (#1188).
    """
    if callable(extra):
        try:
            extra = extra()
        except Exception as exc:  # diagnostics must never break training
            return {"extra": _UNRESOLVED_EXTRA.format(f"{type(exc).__name__}: {exc}")}
    if not isinstance(extra, dict):
        return extra
    resolved: dict[str, Any] = {}
    for key, value in extra.items():
        if not callable(value):
            resolved[key] = value
            continue
        try:
            resolved[key] = value()
        except Exception as exc:  # diagnostics must never break training
            resolved[key] = _UNRESOLVED_EXTRA.format(f"{type(exc).__name__}: {exc}")
    return resolved


def coerce_extra(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce non-JSON-able extras to strings / floats."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            try:
                out[k] = v.detach().cpu().tolist() if v.numel() < 16 else float(v.mean())
            except Exception:
                out[k] = str(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def json_safe(o: Any) -> Any:
    """Replace every non-finite float with ``None``, recursively.

    ``json.dumps`` serialises a float NaN itself, as the bare token ``NaN`` that
    strict parsers reject, so a ``default=`` hook never sees one; and the
    snapshot of a diverged run is the one a reader most needs to parse.
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    return o


def json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return json_safe(o.tolist())
    return str(o)


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def report_dict(r: TensorReport) -> dict[str, Any]:
    return asdict(r)


__all__ = [
    "TensorReport",
    "coerce_extra",
    "json_default",
    "json_safe",
    "report_dict",
    "resolve_deferred_extra",
    "safe_name",
    "tensor_report",
]
