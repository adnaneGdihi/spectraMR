"""Module layouts an older build wrote into a checkpoint, rebuilt so the run resumes exactly.

ZeRO restores fp32 master weights and optimizer moments by flat position, so a resume has
to build the parameters the checkpoint was written with, in the same registration order.
#3363 stopped building the bridge's unused ``rep_fusion``; a run whose checkpoint still
carries it is refused by DeepSpeed ("parameter-alignment layout does not match") unless
the generator builds the module again.
"""

from __future__ import annotations

import re
from pathlib import Path

from spectramr.infrastructure.checkpointing.codecs import model_state_keys

#: Generator kwarg that rebuilds the pre-#3363 bridge ``rep_fusion`` (built, never called).
LEGACY_REP_FUSION = "legacy_rep_fusion"

_BRIDGE_REP_FUSION = re.compile(r"(?:^|\.)backbone\.rep_fusion\.")


def checkpoint_layout_kwargs(path: str | Path) -> dict[str, bool]:
    """Generator kwargs that reproduce the module layout of the checkpoint at ``path``.

    Empty when the checkpoint was written by the current layout.
    """
    keys = model_state_keys(path, "generator")
    if any(_BRIDGE_REP_FUSION.search(key) for key in keys):
        return {LEGACY_REP_FUSION: True}
    return {}


__all__ = ["LEGACY_REP_FUSION", "checkpoint_layout_kwargs"]
