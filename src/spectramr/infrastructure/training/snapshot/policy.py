"""Which steps get a snapshot, and how each recorded tensor is rendered.

:class:`SnapshotSchedule` is the one cadence owner (``logging.snapshots``); no call site
decides on its own whether a snapshot is due. :class:`ViewPolicy` turns the config's
domain facts into a :class:`~spectramr.core.snapshot_trace.View` per tensor name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from spectramr.core.snapshot_trace import ChannelSegment, View

PHASES = ("train", "val")

#: Names whose k-space domain the config declares when the loader yields k-space. The
#: renderer IFFTs these without the spectrum veto, which false-negates on normalized
#: multicoil M4Raw k-space; nothing whose domain depends on a branch belongs here.
CANONICAL_KSPACE = frozenset(
    {
        "input",
        "input_raw",
        "input_prepared",
        "target",
        "kspace",
        "measured_kspace",
        "model_input",
        "noisy_kspace",
        "model_output_backbone",
        "model_output_backbone_projected",
        "model_output_pre_dc",
        "model_output_post_dc",
    }
)


@dataclass
class SnapshotSchedule:
    """Per-phase budget over an interval; train and val never share a counter."""

    enabled: bool = True
    interval_steps: int = 0
    max_calls: int = 8
    validation: bool = True
    _calls: dict[str, int] = field(default_factory=lambda: dict.fromkeys(PHASES, 0))

    @classmethod
    def from_logging(cls, logging_cfg: Any) -> SnapshotSchedule:
        snap = logging_cfg.snapshots
        return cls(
            enabled=bool(snap.enabled),
            interval_steps=int(snap.interval_steps),
            max_calls=int(snap.max_calls),
            validation=bool(snap.validation),
        )

    def due(self, phase: str, step: int) -> bool:
        """Spend one call of ``phase``'s budget if ``step`` is selected."""
        if phase not in PHASES:
            raise ValueError(f"snapshot phase must be one of {PHASES}, got {phase!r}")
        if not self.enabled or (phase == "val" and not self.validation):
            return False
        if self.interval_steps > 0 and step % self.interval_steps != 0:
            return False
        if self._calls[phase] >= self.max_calls:
            return False
        self._calls[phase] += 1
        return True


@dataclass(frozen=True)
class ViewPolicy:
    """The config's domain facts: does the loader yield k-space, does the model emit it."""

    preds_kspace: bool = False
    targets_kspace: bool = False
    log_scaled: bool = False

    @classmethod
    def from_config(cls, config: Any) -> ViewPolicy:
        from spectramr.infrastructure.training.utils.domain_inference import (
            needs_ifft_for_visualization,
        )
        from spectramr.infrastructure.training.utils.kspace_view import log_scaling_enabled

        preds, targets = needs_ifft_for_visualization(config)
        return cls(bool(preds), bool(targets), bool(log_scaling_enabled(config)))

    def views(
        self,
        names: Iterable[str],
        *,
        in_kspace_keys: set[str] | None = None,
        log_scaled_keys: set[str] | None = None,
        segments: Mapping[str, Sequence[ChannelSegment]] | None = None,
        domain_exact: bool = False,
    ) -> dict[str, View]:
        """One view per name.

        ``in_kspace_keys`` names the k-space tensors (default: names containing
        ``kspace``); on a k-space loader the canonical names are added and marked
        authoritative unless ``domain_exact`` makes the caller's set the whole answer.
        ``model_output`` follows the model's declared output domain, not the loader's.
        ``log_scaled_keys`` names the tensors on the compressed side of ``log1p``
        (default: all); the renderer decompresses only k-space ones.
        """
        names = list(names)
        authoritative: set[str] = set()
        if self.targets_kspace and not domain_exact:
            canonical = set(CANONICAL_KSPACE) | ({"model_output"} if self.preds_kspace else set())
            authoritative = canonical & set(names)
            kspace = set(names) if in_kspace_keys is None else set(in_kspace_keys) | authoritative
            if not self.preds_kspace:
                kspace.discard("model_output")
        elif in_kspace_keys is None:
            kspace = {n for n in names if "kspace" in n.lower() or "k_space" in n.lower()}
        else:
            kspace = set(in_kspace_keys)
        out: dict[str, View] = {}
        for n in names:
            is_k = n in kspace
            log = self.log_scaled and (log_scaled_keys is None or n in log_scaled_keys)
            segs = tuple(segments[n]) if segments and n in segments else None
            out[n] = View(
                kspace=is_k, authoritative=n in authoritative, log_scaled=log, segments=segs
            )
        return out


__all__ = ["CANONICAL_KSPACE", "PHASES", "SnapshotSchedule", "ViewPolicy"]
