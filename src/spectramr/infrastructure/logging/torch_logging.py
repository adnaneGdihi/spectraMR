"""Push the log level onto torch's own logging.

Torch routes the glog-style ``I1019 ... _inductor`` lines through ``torch._logging``
with handlers of its own, which the root level does not govern.
"""

from __future__ import annotations

import logging
import os
import sys

__all__ = ["apply_level_to_torch"]

logger = logging.getLogger(__name__)


def apply_level_to_torch(level: int) -> None:
    """Apply ``level`` to torch's logging, once torch is loaded.

    A no-op on the torch-free ``--help`` path; the policy applies the level again after
    the config loads, by which time torch is in. An explicit ``TORCH_LOGS`` is the
    user's and is left alone.
    """
    if "torch" not in sys.modules or os.environ.get("TORCH_LOGS"):
        return
    try:
        from torch._logging import set_logs
    except ImportError:
        logger.warning("torch._logging unavailable: torch's own log lines keep their level")
        return
    set_logs(all=level)
