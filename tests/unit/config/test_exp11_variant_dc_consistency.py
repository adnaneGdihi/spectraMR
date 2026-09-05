"""Regression: exp11 k-space cold-diffusion variant DC-config consistency.

Locks in the 2026-05-31 DC-blob fix propagation across the
``experiments/inprogress/kspace_filling/`` cohort. Before the fix, all 40
backbone/attention/sampling/ablation variants carried
``model.model_kwargs.dc_method: null`` together with
``physics.data_consistency.enabled: true``. That combination is broken in two
independent ways (confirmed by the 2026-05-31 variant audit):

1. ``ModelBuilder._reconcile`` (``model_builder.py:188``) overwrites the null
   sentinel with ``physics.data_consistency.method``. 32 of the 40 arms omit
   that key, so it defaults to ``projection_2d_consistency`` (``physics.py:93``),
   which is **not** in ``KSpaceColdDiffusionGenerator._VALID_DC_METHODS`` — the
   generator raises ``ValueError`` at construction (Tier-2 probe).
2. ``check_dc_method_physics_consistency`` (``config_health_checker.py:3741``)
   emits a WARNING for every arm that pairs ``physics...enabled: true`` with a
   disable-sentinel ``dc_method`` — and under the ``--strict`` smoke audit,
   warnings exit 2 (CLAUDE.md pitfall #10), so all 40 fail Tier 0/1.

There is also a reconcile *conflict* guard (``model_builder.py:175``): if
``model_kwargs.dc_method`` is set to a non-null value that disagrees with
``physics.data_consistency.method``, construction raises.

This test asserts the three invariants that, together, keep every arm runnable
and audit-clean. It is intentionally a pure-config test (no torch / no model
construction) so it runs in the fast unit tier.

Supersedes ``test_exp11_double_dc_regression.py`` (removed 2026-07-14): that
2026-05-12 guard demanded ``dc_method`` be a null/disable sentinel whenever
``physics.data_consistency.enabled`` is true — the exact combination invariant
2 below forbids (strict-audit WARNING, exit 2). The 2026-05-31 fix resolved
the double-DC hazard the other way: ``model_kwargs.dc_method`` mirrors the
physics SSOT and ``ModelBuilder._reconcile`` guarantees the generator applies
DC exactly once, so the two tests were mutually unsatisfiable on every arm
with DC enabled, and the older one failed all 47 exp-11 parametrizations on a
clean checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.physics.data_consistency import VALID_DC_METHODS
from tests.utils.corpus import tracked_yamls
from tests.utils.repo_scripts import skip_if_public_export

# Imported, never transcribed. This was a hand-copied mirror, on the premise that
# the generator's set "lives inside ``__init__`` and is not importable" -- true
# when written, false since the set moved to physics/data_consistency.py as a
# module-level frozenset. The mirror then under-listed ``noise_adaptive`` (a real
# Wiener/SNR trust layer, distinct from the ``noise_adjusted`` alias), and the
# arm named for it -- dc_shootout/experiment_11_dc_noise_adaptive.yaml -- failed
# as if its config were wrong. It was not; the mirror was.
#
# The drift guard below it did not catch this, because it only asserted that
# every method in the MIRROR appears in the source. Nothing asserted the
# converse, so an ADDITION to the generator was invisible to it by construction.
# A one-directional check on a two-directional invariant fails only in the
# direction that did not happen.
_VALID_DC_METHODS = VALID_DC_METHODS
_DISABLE_SENTINELS = (None, "", "none", "off", "disabled")

_COHORT_DIR = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "inprogress"
    / "kspace_filling"
)


def _cohort_yamls() -> list[Path]:
    return tracked_yamls(_COHORT_DIR)


def _ids(paths: list[Path]) -> list[str]:
    return [p.relative_to(_COHORT_DIR).as_posix() for p in paths]


_YAMLS = _cohort_yamls()


def test_cohort_is_non_empty() -> None:
    skip_if_public_export("experiments/ does not ship; the exp11 cohort is empty here")
    # Guards against a glob that silently matches nothing (which would make every
    # parametrized assertion vacuously pass). The exact count varies by checkout:
    # the gitignored ``ablations_kan_dual_domain/`` dir (.gitignore '**/*_kan_*/')
    # is present in the main working tree (cohort ~42) but absent in fresh
    # worktrees / clean checkouts (cohort ~31). Floor well below the
    # always-present tracked count so the guard is robust to that variance rather
    # than failing spuriously where the gitignored dir is not checked out (the
    # original ``>= 40`` floored above the worktree count and broke there).
    assert len(_YAMLS) >= 25, f"expected the full exp11 cohort, found {len(_YAMLS)}"


@pytest.mark.parametrize("yaml_path", _YAMLS, ids=_ids(_YAMLS))
def test_variant_dc_config_is_consistent(yaml_path: Path) -> None:
    settings = TrainingSettings.from_yaml(str(yaml_path))

    dc = getattr(settings.physics, "data_consistency", None)
    if dc is None or not getattr(dc, "enabled", False):
        pytest.skip("physics.data_consistency disabled — DC invariants N/A")

    model_kwargs = dict(settings.model.model_kwargs or {})
    model_dc_method = model_kwargs.get("dc_method", "__unset__")
    physics_method = dc.method

    # Invariant 1 — physics method the generator will actually receive must be
    # constructible (else the Tier-2 probe raises). projection_2d_consistency,
    # the schema default, is deliberately excluded.
    assert physics_method in _VALID_DC_METHODS, (
        f"{yaml_path.name}: physics.data_consistency.method={physics_method!r} "
        f"is not a generator-valid DC method {sorted(_VALID_DC_METHODS)} — "
        f"construction would raise (projection_2d_consistency default leak)."
    )

    # Invariant 2 — no disable-sentinel dc_method while physics DC is enabled
    # (the strict-audit WARNING that exits 2).
    if model_dc_method != "__unset__":
        assert model_dc_method not in _DISABLE_SENTINELS, (
            f"{yaml_path.name}: model_kwargs.dc_method={model_dc_method!r} is a "
            f"disable sentinel while physics.data_consistency.enabled=true — "
            f"trips check_dc_method_physics_consistency (strict audit exits 2)."
        )

        # Invariant 3 — a set, non-sentinel dc_method must match the physics
        # method, or _reconcile's conflict guard raises.
        assert model_dc_method == physics_method, (
            f"{yaml_path.name}: model_kwargs.dc_method={model_dc_method!r} != "
            f"physics.data_consistency.method={physics_method!r} — "
            f"ModelBuilder._reconcile raises a conflict ValueError."
        )


def test_the_generator_validates_against_the_same_set_this_test_does() -> None:
    """The set is imported, so drift is impossible — pin the seam that makes it so.

    What can still break is the *seam*: the generator silently stopping using the
    physics SSOT, or the SSOT emptying out. Both would leave every assertion in
    this file trivially satisfiable, so assert them directly rather than
    re-deriving the list.
    """
    from spectramr.models.generators import kspace_cold_diffusion_generator as gen

    assert gen.VALID_DC_METHODS is VALID_DC_METHODS, (
        "the generator no longer validates against "
        "physics.data_consistency.VALID_DC_METHODS — this test is now checking a "
        "different vocabulary than the code it guards"
    )
    assert isinstance(VALID_DC_METHODS, frozenset) and VALID_DC_METHODS, (
        "VALID_DC_METHODS is empty or not a set; every DC assertion in this file "
        "would pass vacuously"
    )
    # Anti-vacuity: the member whose absence from the old hand-copied mirror is
    # what made this test file wrong about a correct arm.
    assert "noise_adaptive" in VALID_DC_METHODS
