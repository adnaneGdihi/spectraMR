"""Concrete meta-evaluation rankers + the unified ranker registry.

Importing this package registers every built-in ranker on the unified
:mod:`~spectramr.core.metrics.meta_evaluation.rankers.registry` (one registry,
one ``BaseRanker`` contract, all generations). The ``generation`` tag is
lineage metadata only — there is no "legacy" vs "novel" split.
"""

# Importing each ranker module fires its @register_ranker decorator (side
# effect). Gen-1 statistical: statistical(adr/scvr/cdrs). Gen-2 info/sensitivity:
# ms3, information(mi), sobol, bradley_terry(bt), task. Gen-3 structural: the 6
# below (registered programmatically). Gen-4 clinical (gated): clinical(dcms/
# acss/hibsl). Gen-5 (opt-in, never pooled): blackwell, blackwell_relations. Imports
# are alphabetical (registration is order-independent).
from .base import BaseRanker
from .blackwell import BlackwellConfig, BlackwellConsistencyRanker, radial_catalogue
from .blackwell_relations import BlackwellRelationRanker
from .bradley_terry import BTConfig, BTRanker
from .cdscr import CDSCRConfig, CDSCRRanker
from .clinical import ACSSRanker, DCMSRanker, HiBSLRanker
from .esd import ESDConfig, ESDRanker
from .fspd import FSPDConfig, FSPDRanker
from .information import MIConfig, MIRanker
from .lgdr import LGDRConfig, LGDRRanker
from .ms3 import MS3Config, MS3Ranker
from .registry import (
    RankerSpec,
    available_rankers,
    check_data_dependencies,
    check_injected_data,
    get_ranker,
    has_ranker,
    list_rankers,
    redundant_rankers,
    register_ranker,
)
from .sfa import SFAConfig, SFARanker
from .sim2rank import Sim2RankConfig, Sim2RankRanker
from .sobol import SobolConfig, SobolRanker
from .statistical import ADRRanker, CDRSRanker, SCVRRanker
from .task import TaskScoreConfig, TaskScoreRanker

# ── Register the Gen-3 structural rankers (already BaseRanker-conformant) ──
# Done programmatically here (the canonical, single registration site) rather
# than decorating each stable class file. New-generation ranker modules
# (statistical / information / clinical) use the @register_ranker decorator
# inline at class definition.
#
# NO ``if not has_ranker(...)`` GUARD. The registry's documented contract is that
# re-registering a name RAISES (strict registry, no silent overwrite) — a guard
# here turns a genuine double-registration (two classes racing for one name, a
# module imported under two paths) into a silent no-op in which the FIRST
# registration wins and the second is discarded without a word. That is exactly
# the silent-fallback class this registry exists to forbid (pitfall #9 / #258).
# Module import is idempotent (sys.modules caches it), so this loop runs once.
#
# ``sim2rank`` declares ``fuses``: it is *by construction* the min-max fusion
# 0.5·ADR + 0.3·SCVR + 0.2·CDRS of three rankers that are themselves registered
# (gen-1). ``available_rankers`` drops it whenever all three vote, so the pool
# stays mutually non-redundant (#256).
_GEN3_RANKERS: list[tuple[str, type[BaseRanker], str, tuple[str, ...]]] = [
    ("cdscr", CDSCRRanker, "Cross-Descriptor Spectral Consistency Ranking", ()),
    (
        "lgdr",
        LGDRRanker,
        "Local Geometric Distortion Ranking (manifold eigenscale)",
        (),
    ),
    ("sfa", SFARanker, "Score-Function Alignment (manifold-gradient)", ()),
    ("esd", ESDRanker, "Equivariance/Sensitivity Decomposition", ()),
    ("fspd", FSPDRanker, "Functional-PCA Severity-mode Profile Discrepancy", ()),
    (
        "sim2rank",
        Sim2RankRanker,
        "Min-max consensus fusion (ADR/SCVR/CDRS)",
        ("adr", "scvr", "cdrs"),
    ),
]
for _name, _cls, _desc, _fuses in _GEN3_RANKERS:
    register_ranker(_name, generation=3, description=_desc, fuses=_fuses)(_cls)

__all__ = [
    "BaseRanker",
    "BlackwellConfig",
    "BlackwellConsistencyRanker",
    "BlackwellRelationRanker",
    "radial_catalogue",
    "CDSCRConfig",
    "CDSCRRanker",
    "LGDRConfig",
    "LGDRRanker",
    "SFAConfig",
    "SFARanker",
    "ESDConfig",
    "ESDRanker",
    "FSPDConfig",
    "FSPDRanker",
    "Sim2RankConfig",
    "Sim2RankRanker",
    # Gen-1 statistical rankers
    "ADRRanker",
    "SCVRRanker",
    "CDRSRanker",
    # Gen-2 info/sensitivity rankers
    "MS3Config",
    "MS3Ranker",
    "MIConfig",
    "MIRanker",
    "SobolConfig",
    "SobolRanker",
    "BTConfig",
    "BTRanker",
    "TaskScoreConfig",
    "TaskScoreRanker",
    # Gen-4 clinical/causal rankers (capability-gated)
    "DCMSRanker",
    "ACSSRanker",
    "HiBSLRanker",
    # Unified registry
    "RankerSpec",
    "register_ranker",
    "get_ranker",
    "list_rankers",
    "available_rankers",
    "has_ranker",
    "check_data_dependencies",
    "check_injected_data",
    "redundant_rankers",
]
