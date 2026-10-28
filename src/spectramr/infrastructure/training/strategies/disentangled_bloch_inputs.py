"""Per-sample Bloch sequence families for ``DisentangledTrainingStrategy`` (#2611).

The strategy renders tissue maps through ``MultiPhysicsBlochLayer`` and the echo-train
models; the family of each sample has to come from the sample. The tissue decode itself
lives on the model (``DisentangledMRI.decode_tissue``), its one owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from spectramr.data.batch_types import read_batch_field
from spectramr.infrastructure.physics.multi_physics_bloch import MultiPhysicsBlochLayer

SEQUENCE_FAMILIES: dict[str, int] = {
    "SE": MultiPhysicsBlochLayer.SEQ_SPIN_ECHO,
    "IR": MultiPhysicsBlochLayer.SEQ_INVERSION_RECOVERY,
    "GRE": MultiPhysicsBlochLayer.SEQ_GRADIENT_ECHO,
}


def parse_sequence_map(raw: Mapping[str, str] | None) -> dict[str, int]:
    """Validate ``model_kwargs.bloch_sequence_by_contrast`` into contrast -> family code."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"bloch_sequence_by_contrast must be a mapping, got {type(raw).__name__}")
    out: dict[str, int] = {}
    for contrast, family in raw.items():
        code = SEQUENCE_FAMILIES.get(str(family).upper())
        if code is None:
            raise ValueError(
                f"bloch_sequence_by_contrast[{contrast!r}] = {family!r}; "
                f"expected one of {sorted(SEQUENCE_FAMILIES)}"
            )
        out[str(contrast).upper()] = code
    return out


PROTOCOL_KEYS = ("TR", "TE", "TI")


def parse_protocol_map(
    raw: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, float]]:
    """Validate ``model_kwargs.bloch_protocol_by_contrast`` into contrast -> {TR, TE, TI} (ms).

    The echo-train renders (spin echo, inversion recovery) take the acquisition each
    contrast was scanned with instead of the style code's unconstrained prediction: with a
    free protocol the tissue maps and the protocol can trade off against each other, and
    the renderer would be a self-calibrated one that nothing identifies. ``TI`` is 0 when
    not declared (spin echo).
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"bloch_protocol_by_contrast must be a mapping, got {type(raw).__name__}")
    out: dict[str, dict[str, float]] = {}
    for contrast, params in raw.items():
        if not isinstance(params, Mapping):
            raise TypeError(
                f"bloch_protocol_by_contrast[{contrast!r}] must be a mapping of TR/TE/TI in "
                f"ms, got {type(params).__name__}"
            )
        entry = {str(k).upper(): float(v) for k, v in params.items()}
        unknown = sorted(set(entry) - set(PROTOCOL_KEYS))
        if unknown:
            raise ValueError(
                f"bloch_protocol_by_contrast[{contrast!r}] has unknown key(s) {unknown}; "
                f"expected {list(PROTOCOL_KEYS)}"
            )
        if "TR" not in entry or "TE" not in entry:
            raise ValueError(f"bloch_protocol_by_contrast[{contrast!r}] needs TR and TE (ms)")
        for key in ("TR", "TE"):
            if entry[key] <= 0:
                raise ValueError(f"bloch_protocol_by_contrast[{contrast!r}].{key} must be positive")
        if "TI" in entry and entry["TI"] <= 0:
            raise ValueError(
                f"bloch_protocol_by_contrast[{contrast!r}].TI must be positive; omit it for "
                "a spin-echo contrast"
            )
        entry.setdefault("TI", 0.0)
        out[str(contrast).upper()] = entry
    return out


def check_protocol_matches_families(
    protocol: Mapping[str, Mapping[str, float]], families: Mapping[str, int]
) -> None:
    """Refuse a protocol that disagrees with ``bloch_sequence_by_contrast``.

    Every declared contrast needs a protocol, a protocol needs a declared contrast, an
    inversion-recovery contrast needs a TI and a spin-echo one may not declare it (it
    would be an unread knob). Gradient echo renders through the Bloch layer, which does not
    take this protocol.
    """
    se, ir = (
        MultiPhysicsBlochLayer.SEQ_SPIN_ECHO,
        MultiPhysicsBlochLayer.SEQ_INVERSION_RECOVERY,
    )
    for contrast, family in families.items():
        if family not in (se, ir):
            raise ValueError(
                f"bloch_protocol_by_contrast covers spin echo or inversion recovery only; "
                f"{contrast!r} is declared with another family in bloch_sequence_by_contrast"
            )
        if contrast not in protocol:
            raise ValueError(
                f"contrast {contrast!r} has no entry in bloch_protocol_by_contrast "
                f"{sorted(protocol)}"
            )
        ti = protocol[contrast]["TI"]
        if family == ir and ti <= 0:
            raise ValueError(
                f"{contrast!r} is inversion recovery and needs a TI in bloch_protocol_by_contrast"
            )
        if family == se and ti > 0:
            raise ValueError(
                f"{contrast!r} is spin echo; its TI={ti} in bloch_protocol_by_contrast would "
                "never be read"
            )
    extra = sorted(set(protocol) - set(families))
    if extra:
        raise ValueError(
            f"bloch_protocol_by_contrast names {extra!r}, which bloch_sequence_by_contrast "
            "does not declare"
        )


def check_flip_angle_is_read(learnable: bool, families: Mapping[str, int]) -> None:
    """Refuse ``learnable_flip_angle`` on an arm that never reaches the layer that owns it.

    The flip angle is a parameter of the gradient-echo Bloch layer. Spin-echo and
    inversion-recovery rows render through the echo-train paths, which fix their pulses, so
    on an arm whose declared families are all of those the parameter joins the optimiser and
    never receives a gradient. With no declared families the batch decides at run time and
    nothing is refused.
    """
    if not learnable or not families:
        return
    echo_train = {
        MultiPhysicsBlochLayer.SEQ_SPIN_ECHO,
        MultiPhysicsBlochLayer.SEQ_INVERSION_RECOVERY,
    }
    if set(families.values()) <= echo_train:
        raise ValueError(
            "learnable_flip_angle is read only by the gradient-echo Bloch layer, but every "
            "family in bloch_sequence_by_contrast is spin echo or inversion recovery, so the "
            "parameter would never receive a gradient; declare learnable_flip_angle: false"
        )


def _expand(codes: torch.Tensor, n_rows: int) -> torch.Tensor:
    if codes.numel() == n_rows:
        return codes.view(-1, 1, 1, 1)
    if codes.numel() and n_rows % codes.numel() == 0:
        return codes.repeat_interleave(n_rows // codes.numel()).view(-1, 1, 1, 1)
    raise ValueError(f"{codes.numel()} per-sample sequence codes cannot cover {n_rows} rows")


def sequence_codes(
    batch: Any, seq_map: dict[str, int], n_rows: int, device: torch.device
) -> torch.Tensor:
    """The Bloch sequence family of every row, from the batch; never a default."""
    explicit = read_batch_field(batch, "contrast_type")
    if isinstance(explicit, torch.Tensor):
        return _expand(explicit.to(device=device, dtype=torch.long).view(-1), n_rows)
    contrasts = read_batch_field(batch, "contrast")
    if contrasts is None:
        raise ValueError(
            "Bloch synthesis needs each sample's sequence family, but the batch carries "
            "neither 'contrast_type' nor 'contrast'. Use a dataset that emits the M4Raw "
            "contrast, and declare model_kwargs.bloch_sequence_by_contrast."
        )
    if isinstance(contrasts, str):
        contrasts = [contrasts]
    codes = []
    for contrast in contrasts:
        key = str(contrast).upper()
        if key not in seq_map:
            raise ValueError(
                f"contrast {contrast!r} has no entry in "
                f"model_kwargs.bloch_sequence_by_contrast {sorted(seq_map)}"
            )
        codes.append(seq_map[key])
    return _expand(torch.tensor(codes, device=device, dtype=torch.long), n_rows)


def acquisition_params(
    batch: Any, protocol: Mapping[str, Mapping[str, float]], n_rows: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """TR, TE and TI (ms) of every row as ``(n_rows, 1, 1, 1)`` tensors; never a default.

    A batch that already carries an ``acquisition`` mapping (validation hands the render a
    slice of the pass's rows) is used as given; otherwise each sample's contrast name is
    looked up in the declared protocol.
    """
    explicit = read_batch_field(batch, "acquisition")
    if isinstance(explicit, Mapping):
        return {k: explicit[k].to(device=device, dtype=torch.float32) for k in PROTOCOL_KEYS}
    contrasts = read_batch_field(batch, "contrast")
    if contrasts is None:
        raise ValueError(
            "the echo-train render needs each sample's acquisition, but the batch carries "
            "neither 'acquisition' nor 'contrast'. Use a dataset that emits the M4Raw "
            "contrast, and declare model_kwargs.bloch_protocol_by_contrast."
        )
    if isinstance(contrasts, str):
        contrasts = [contrasts]
    rows = []
    for contrast in contrasts:
        key = str(contrast).upper()
        if key not in protocol:
            raise ValueError(
                f"contrast {contrast!r} has no entry in "
                f"model_kwargs.bloch_protocol_by_contrast {sorted(protocol)}"
            )
        rows.append(protocol[key])
    return {
        k: _expand(torch.tensor([r[k] for r in rows], device=device, dtype=torch.float32), n_rows)
        for k in PROTOCOL_KEYS
    }
