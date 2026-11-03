"""Does the tensor a model was fed match the input width its backbone declares?

The snapshot session's forward hooks record the real model input; this module
compares its channel count with the backbone's declared width. It WARNS and stamps
rather than raises, because the width is a heuristic read (a complex tensor
legitimately feeds a real-view conv of twice its channel count), and reports an
unresolved width as ``unresolved`` rather than dropping the check (pitfall #16).

Everything here is shape-only: no ``.item()``, no host copy, no forward pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, NamedTuple

import torch
from torch import nn

#: Channel axis for the tensors this contract inspects (N, C, ...).
CHANNEL_DIM: Final[int] = 1

#: Verdicts. ``unresolved`` is a first-class outcome, not a silent skip.
STATUS_MATCH: Final[str] = "match"
STATUS_MISMATCH: Final[str] = "mismatch"
STATUS_UNRESOLVED: Final[str] = "unresolved"


class ModelInputVerdict(NamedTuple):
    """The declared-vs-applied comparison, in the form that reaches the artifact."""

    model_input_key: str
    declared_channels: int | None
    declared_shape: tuple[int, ...] | None
    is_complex: bool
    model_in_channels: int | None
    in_channels_source: str
    accepted_channels: tuple[int, ...]
    status: str
    detail: str

    def as_record(self) -> dict[str, Any]:
        """The nested block stamped into the snapshot manifest."""
        return {
            "model_input_key": self.model_input_key,
            "declared": {
                "channels": self.declared_channels,
                "shape": list(self.declared_shape) if self.declared_shape else None,
                "is_complex": self.is_complex,
            },
            "applied": {
                "in_channels": self.model_in_channels,
                "resolved_from": self.in_channels_source,
                "accepted": list(self.accepted_channels),
            },
            "status": self.status,
            "detail": self.detail,
        }


def resolve_model_in_channels(module: Any) -> tuple[int | None, str]:
    """Best-effort read of the width consumed by whatever eats the model input.

    Preference order, most authoritative first:

    1. ``backbone_in_channels`` -- a WRAPPER's explicit statement of the width
       its inner backbone consumes, which is what the snapshotted
       ``model_input`` is compared against;
    2. a module-level ``in_channels`` attribute -- what the architecture itself
       says it takes, and the only source that survives an unusual first layer;
    3. the first ``Conv{1,2,3}d`` in ``modules()`` order;
    4. the first ``Linear``'s ``in_features``.

    Tier 1 exists because tiers 2-4 answer a subtly different question for a
    module that adapts channels before delegating. ``KSpaceColdDiffusionGenerator``
    sets ``in_channels = 8`` (the bare measurement its ``forward`` concat gate
    keys on) while building its backbone at 16 for the concatenated
    ``[noisy_kspace || smaps]`` stack the training path actually feeds it. Read
    as "the declared input width", 8 is correct; read as "the width the tensor
    under test must match", it is wrong, and the mismatch fired on every
    smaps-conditioned cold-diffusion arm. A wrapper that adapts channels knows
    both numbers -- this tier is where it says which one this check means.

    Returns ``(None, "unresolved")`` for anything else (a transformer patchifier,
    a functional first op, a mocked module). Fail-open by construction: this is
    a diagnostic, and every caller treats ``None`` as "cannot say", never as
    "mismatch".
    """
    if module is None:
        return None, "unresolved"

    backbone = getattr(module, "backbone_in_channels", None)
    if isinstance(backbone, int) and not isinstance(backbone, bool) and backbone > 0:
        return backbone, "module.backbone_in_channels"

    declared = getattr(module, "in_channels", None)
    if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
        return declared, "module.in_channels"

    try:
        children = list(module.modules())
    except Exception:  # pragma: no cover - a mock without .modules()
        return None, "unresolved"

    for sub in children:
        if isinstance(sub, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return int(sub.in_channels), f"first {type(sub).__name__}"
    for sub in children:
        if isinstance(sub, nn.Linear):
            return int(sub.in_features), "first Linear.in_features"
    return None, "unresolved"


def verify_model_input(
    *,
    tensors: Mapping[str, Any],
    model_input_key: str,
    in_channels: int | None,
    in_channels_source: str,
) -> ModelInputVerdict:
    """Compare the named tensor's channel count against the backbone's width.

    A complex tensor accepts ``{C, 2C}``: half this repo's k-space backbones take
    a real view of a complex input, doubling the channel axis, and the snapshot
    records the tensor as the strategy holds it -- before that view is taken.
    Accepting only ``C`` would make every complex-input arm report a mismatch,
    and a check that cries wolf on a whole paradigm gets muted rather than read.

    Takes the already-resolved width: :func:`resolve_model_in_channels` walks
    ``modules()``, so the session resolves it once per run.
    """
    tensor = tensors.get(model_input_key)
    source = in_channels_source

    if not isinstance(tensor, torch.Tensor) or tensor.dim() <= CHANNEL_DIM:
        return ModelInputVerdict(
            model_input_key=model_input_key,
            declared_channels=None,
            declared_shape=tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None,
            is_complex=False,
            model_in_channels=in_channels,
            in_channels_source=source,
            accepted_channels=(),
            status=STATUS_UNRESOLVED,
            detail=(
                f"{model_input_key!r} has no channel axis to compare "
                "(not a tensor, or fewer dims than NCHW)"
            ),
        )

    channels = int(tensor.shape[CHANNEL_DIM])
    is_complex = bool(tensor.is_complex())
    accepted = (channels, 2 * channels) if is_complex else (channels,)

    if in_channels is None:
        return ModelInputVerdict(
            model_input_key=model_input_key,
            declared_channels=channels,
            declared_shape=tuple(tensor.shape),
            is_complex=is_complex,
            model_in_channels=None,
            in_channels_source=source,
            accepted_channels=accepted,
            status=STATUS_UNRESOLVED,
            detail=(
                "the backbone declares no in_channels and its first layer is "
                "neither Conv nor Linear, so the width could not be read"
            ),
        )

    if in_channels in accepted:
        return ModelInputVerdict(
            model_input_key=model_input_key,
            declared_channels=channels,
            declared_shape=tuple(tensor.shape),
            is_complex=is_complex,
            model_in_channels=in_channels,
            in_channels_source=source,
            accepted_channels=accepted,
            status=STATUS_MATCH,
            detail=(
                f"{model_input_key!r} carries {channels} channels"
                + (" (complex; a real view doubles it)" if is_complex else "")
                + f" and the backbone takes {in_channels}"
            ),
        )

    return ModelInputVerdict(
        model_input_key=model_input_key,
        declared_channels=channels,
        declared_shape=tuple(tensor.shape),
        is_complex=is_complex,
        model_in_channels=in_channels,
        in_channels_source=source,
        accepted_channels=accepted,
        status=STATUS_MISMATCH,
        detail=(
            f"{model_input_key!r} carries {channels} channels "
            f"(accepting {list(accepted)}) but the backbone takes {in_channels}, "
            f"read from {source}. Either the key names the wrong tensor -- the "
            "snapshot then records something the model never saw -- or the "
            "wiring feeds the backbone a width it did not declare."
        ),
    )


__all__ = [
    "CHANNEL_DIM",
    "STATUS_MATCH",
    "STATUS_MISMATCH",
    "STATUS_UNRESOLVED",
    "ModelInputVerdict",
    "resolve_model_in_channels",
    "verify_model_input",
]
