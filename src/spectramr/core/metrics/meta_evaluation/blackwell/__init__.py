"""Blackwell-Loewner certificates of cross-degradation IQM validity (Fourier-diagonal class)."""

from .catalogue import DiagonalExperiment, build_catalogue, info_groups, mixture
from .coherence import (
    CoherenceResult,
    group_profiles,
    preferences_by_t,
    revealed_design_lp,
    split_preferences,
)
from .evaluate import metric_table, oracle_lsi_control
from .order import Relation, RelationGraph, certify, compare

__all__ = [
    "CoherenceResult",
    "DiagonalExperiment",
    "Relation",
    "RelationGraph",
    "build_catalogue",
    "certify",
    "compare",
    "group_profiles",
    "info_groups",
    "metric_table",
    "mixture",
    "oracle_lsi_control",
    "preferences_by_t",
    "revealed_design_lp",
    "split_preferences",
]
