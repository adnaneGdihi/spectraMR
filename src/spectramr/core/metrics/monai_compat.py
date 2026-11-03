"""Calls into MONAI metrics that need a shim around MONAI's own deprecations."""

from __future__ import annotations

import warnings
from typing import Any

__all__ = ["call_surface_metric"]


def call_surface_metric(metric: Any, **kwargs: Any) -> Any:
    """Call a MONAI surface-distance metric without its internal ``always_return_as_numpy`` warning.

    MONAI >= 1.5 passes the deprecated argument to ``get_mask_edges`` from its own
    ``get_edge_surface_distance``, so every Hausdorff/surface call warns once per call.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=r".*always_return_as_numpy.*"
        )
        return metric(**kwargs)
