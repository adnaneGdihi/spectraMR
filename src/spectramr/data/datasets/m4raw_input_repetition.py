"""Which acquired repetition of an M4Raw group a training sample serves as its input.

``first`` serves repetition 0, as the loader always has. ``random`` draws one of the
loaded repetitions per sample, so a slice is trained on every noise draw the scanner
recorded instead of one. The NEX target is unchanged apart from its phase frame: it is
still the phase-aligned mean over all repetitions, aligned to the chosen input.

The draw uses a private NumPy generator seeded from ``torch.initial_seed()``. That
seed already separates what must be independent: ``seed_everything`` offsets it by
the DDP rank, and ``DataLoader`` gives each worker ``base_seed + worker_id`` with a
fresh ``base_seed`` per iterator, so ranks, workers and epochs draw differently while
a run stays reproducible from ``run.seed``. No global RNG is consumed, so the masks,
augmentations and r2r draws that share those streams are not shifted by the knob.
"""

from __future__ import annotations

import os

import numpy as np
import torch

__all__ = [
    "VALID_INPUT_REPETITIONS",
    "InputRepetitionDraw",
    "input_first",
    "validate_input_repetition",
]

#: ``first``: repetition 0 (legacy). ``random``: one loaded repetition per sample.
VALID_INPUT_REPETITIONS: frozenset[str] = frozenset({"first", "random"})

#: The only target mode under which ``random`` is honoured: the target averages every
#: repetition in the input's phase frame, so re-choosing the input re-chooses the frame.
_ROTATABLE_TARGET_MODES: frozenset[str] = frozenset({"phase_aligned_mean"})

#: Mixed into the torch seed so this stream is not a copy of torch's own sequence.
_SALT = 0x4D345250  # "M4RP"


def validate_input_repetition(
    value: str,
    *,
    target_mode: str,
    use_repetitions: bool,
    single_contrast: bool,
    nex_target_exclude_input: bool,
    has_coil_maps: bool,
) -> str:
    """Return ``value`` or raise when it is unknown or would go unread.

    ``random`` is refused wherever the loader would serve repetition 0 anyway, or
    where re-choosing the input would leave another quantity in repetition 0's frame:

    * ``target_mode`` other than ``phase_aligned_mean`` (split modes split one
      repetition; ``rep_pair`` and ``complex_mean`` are not what the knob was built
      and tested for);
    * ``use_repetitions=False`` (only repetition 0 is loaded);
    * the cross-contrast route (it pairs two contrasts and serves repetition 0 of each);
    * a leave-one-out target (the excluded repetition is the input's);
    * coil maps (estimated in repetition 0's coil phase frame).
    """
    if value not in VALID_INPUT_REPETITIONS:
        raise ValueError(
            f"[M4Raw] Unknown input_repetition: {value!r}. "
            f"Valid: {sorted(VALID_INPUT_REPETITIONS)}."
        )
    if value == "first":
        return value
    reasons = []
    if target_mode not in _ROTATABLE_TARGET_MODES:
        reasons.append(
            f"target_mode={target_mode!r} (needs one of {sorted(_ROTATABLE_TARGET_MODES)})"
        )
    if not use_repetitions:
        reasons.append("use_repetitions=False loads repetition 0 only")
    if not single_contrast:
        reasons.append("the cross-contrast route serves repetition 0 of each contrast")
    if nex_target_exclude_input:
        reasons.append("a leave-one-out target excludes repetition 0, not the drawn input")
    if has_coil_maps:
        reasons.append("coil maps sit in repetition 0's coil phase frame")
    if reasons:
        raise ValueError(
            "[M4Raw] input_repetition='random' is not honoured here: "
            + "; ".join(reasons)
            + ". Use input_repetition: first."
        )
    return value


def input_first(n_loaded: int) -> int:
    """The legacy choice: repetition 0 of whatever loaded."""
    del n_loaded
    return 0


class InputRepetitionDraw:
    """Per-sample input-repetition chooser for one dataset instance.

    The generator is created lazily and re-created whenever the process or its torch
    seed changes, so a forked worker never replays the parent's stream and a
    respawned worker (new ``base_seed`` each epoch) starts a new one. Within one
    process the stream advances, which reshuffles epochs under ``num_workers=0`` and
    persistent workers alike.
    """

    def __init__(self, mode: str) -> None:
        if mode not in VALID_INPUT_REPETITIONS:
            raise ValueError(
                f"[M4Raw] Unknown input_repetition: {mode!r}. "
                f"Valid: {sorted(VALID_INPUT_REPETITIONS)}."
            )
        self.mode = mode
        self._rng: np.random.Generator | None = None
        self._key: tuple[int, int] | None = None

    def __call__(self, n_loaded: int) -> int:
        """Index into the loaded repetitions of the input for this sample."""
        if n_loaded < 1:
            raise ValueError("[M4Raw] no repetitions loaded to choose an input from.")
        if self.mode == "first" or n_loaded == 1:
            return input_first(n_loaded)
        key = (os.getpid(), int(torch.initial_seed()))
        if self._rng is None or self._key != key:
            self._rng = np.random.default_rng([key[1] % 2**63, _SALT])
            self._key = key
        return int(self._rng.integers(n_loaded))

    def __getstate__(self) -> dict:
        """Ship the mode, never a stream: every process seeds its own."""
        return {"mode": self.mode, "_rng": None, "_key": None}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
