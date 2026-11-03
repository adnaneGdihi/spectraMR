"""F6 regression — ``_resolve_legacy_batch`` helper + 3 strategy refactors.

Audit anchor: TODO/audit/smoke_audit_20260516.md §F6.

Pre-F6 (2026-05-16 smoke), legacy strategies declared
``_compute_losses_impl(self, batch, *args, **kwargs)`` but the base
orchestrator at ``BaseTrainingStrategy._compute_losses`` calls the
implementation with named kwargs only:

    losses = self._compute_losses_impl(
        input_batch=input_batch,
        target_batch=target_batch,
        epoch=epoch,
        **kwargs,
    )

That call cannot bind the legacy ``batch`` positional parameter →
TypeError at the first training step. Round 3 attempted a single-line
parent change and reverted because the strategies INTENTIONALLY read
``batch.get("subject_id")``, ``batch.get("velocity_field")``, etc.

F6 (round 5) adds a shared ``_resolve_legacy_batch`` static method to
the base class and refactors three representative strategies — mri_slam,
diffeomorphic_recon, coord_kspace_gen — to the canonical signature plus
the helper. The remaining 15 strategies are queued for round 6.

These tests pin:

1. The helper's resolution semantics (dict, kwargs["batch"],
   kwargs["input_batch"], or None).
2. The three refactored strategies expose the canonical signature.
3. Their source contains the helper invocation.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

REPO_ROOT = Path(__file__).resolve().parents[3]
# Path corrected 2026-05-24: src→spectramr refactor moved the package under
# src/spectramr/; the stale src/infrastructure/ path FileNotFoundError'd and
# silently disabled the F6 canonical-signature checks. See smoke_audit_20260524.md.
STRATEGIES_DIR = REPO_ROOT / "src" / "spectramr" / "infrastructure" / "training" / "strategies"

F6_REFACTORED = [
    # Round 5 representatives
    "mri_slam_strategy.py",
    # diffeomorphic_recon left: it reads the reconstruction seam, not the batch dict.
    "coord_kspace_gen_strategy.py",
    # Round 6 — 13 remaining legacy strategies refactored in bulk
    "spin_sde_strategy.py",
    "bloch_bottleneck_strategy.py",
    "low_rank_sparse_strategy.py",
    "stochastic_interpolants_strategy.py",
    "noise_adaptive_score_strategy.py",
    "field_probe_coupled_strategy.py",
    "synthetic_pathology_aug_strategy.py",
    "girf_aware_strategy.py",
    "qspace_diffusion_strategy.py",
    "adversarial_robustness_strategy.py",
    "scas_strategy.py",
    "qsm_pipeline_strategy.py",
    "cross_contrast_kspace_diffusion_strategy.py",
]

F6_QUEUED_FOR_ROUND_7_PLUS = [
    # JEPA uses inline resolution introduced in round 4. Could adopt the
    # shared helper for consistency, but the current code works.
    "jepa_strategy.py",
]


# ── Helper semantics ───────────────────────────────────────────────────


def test_resolve_legacy_batch_returns_dict_as_is() -> None:
    batch = {"subject_id": "s001", "image": "tensor"}
    out = BaseTrainingStrategy._resolve_legacy_batch(batch, {})
    assert out is batch


def test_resolve_legacy_batch_finds_batch_in_kwargs() -> None:
    """``kwargs["batch"]`` is the canonical fallback (mirrors JEPA precedent)."""
    expected = {"velocity_field": "tensor"}
    out = BaseTrainingStrategy._resolve_legacy_batch(None, {"batch": expected})
    assert out is expected


def test_resolve_legacy_batch_finds_input_batch_in_kwargs() -> None:
    """``kwargs["input_batch"]`` is the secondary fallback."""
    expected = {"trajectory": "tensor"}
    out = BaseTrainingStrategy._resolve_legacy_batch(
        None, {"input_batch": expected},
    )
    assert out is expected


def test_resolve_legacy_batch_prefers_batch_over_input_batch() -> None:
    """When both are present, ``batch`` wins — it's the explicit handoff name."""
    explicit = {"explicit": True}
    fallback = {"fallback": True}
    out = BaseTrainingStrategy._resolve_legacy_batch(
        None, {"batch": explicit, "input_batch": fallback},
    )
    assert out is explicit


def test_resolve_legacy_batch_returns_none_when_neither_path_yields_a_dict() -> None:
    """No batch resolves → None (caller decides what to do)."""
    out = BaseTrainingStrategy._resolve_legacy_batch(None, {})
    assert out is None


def test_resolve_legacy_batch_can_return_tensor_for_legacy_callers() -> None:
    """If ``kwargs["input_batch"]`` is a non-dict tensor, return it.

    Some strategies short-circuit on ``not isinstance(batch, dict)`` and
    that branch must remain reachable for backward compatibility.
    """
    sentinel = object()
    out = BaseTrainingStrategy._resolve_legacy_batch(
        None, {"input_batch": sentinel},
    )
    assert out is sentinel


# ── Canonical signature on refactored strategies ───────────────────────


@pytest.mark.parametrize("filename", F6_REFACTORED)
def test_f6_refactored_strategy_uses_canonical_signature(filename: str) -> None:
    """Each F6-refactored strategy declares ``_compute_losses_impl`` with the canonical signature."""
    src = (STRATEGIES_DIR / filename).read_text()
    assert "def _compute_losses_impl(" in src, (
        f"{filename}: missing _compute_losses_impl"
    )
    assert "input_batch:" in src and "target_batch:" in src and "epoch:" in src, (
        f"F6 regression: {filename} reverted to the legacy "
        f"``def _compute_losses_impl(self, batch, *args, **kwargs)`` "
        f"signature. The canonical signature is "
        f"``(self, input_batch, target_batch, epoch, **kwargs)``. "
        f"See TODO/audit/smoke_audit_20260516.md §F6."
    )
    assert "_resolve_legacy_batch(" in src, (
        f"F6 regression: {filename} does not call ``_resolve_legacy_batch`` "
        f"to fetch the batch dict for its metadata reads. The helper is "
        f"the new SSOT for legacy-batch resolution."
    )


@pytest.mark.parametrize("filename", F6_REFACTORED)
def test_f6_refactored_strategy_does_not_use_legacy_batch_positional(filename: str) -> None:
    """The legacy positional ``def ...(self, batch, *args, ...)`` pattern is gone."""
    src = (STRATEGIES_DIR / filename).read_text()
    assert "def _compute_losses_impl(self, batch:" not in src, (
        f"F6 regression: {filename} reverted to the legacy positional "
        f"``def _compute_losses_impl(self, batch, *args, **kwargs)``. "
        f"That signature fails to bind the orchestrator's kwargs and "
        f"raises TypeError at the first training step."
    )


# ── Queue-tracking ─────────────────────────────────────────────────────


def test_f6_round_6_queue_resolved() -> None:
    """All 13 round-6 strategies now refactored — only JEPA remains queued."""
    assert len(F6_REFACTORED) == 15, (
        f"Expected 15 refactored strategies (2 round-5 + 13 round-6), "
        f"got {len(F6_REFACTORED)}. Update F6_REFACTORED."
    )
    assert "jepa_strategy.py" in F6_QUEUED_FOR_ROUND_7_PLUS, (
        "JEPA still uses inline resolution; track it as queued for "
        "round-7+ helper adoption."
    )


def test_no_remaining_strategy_uses_legacy_batch_positional() -> None:
    """After round 6, no strategy declares ``(self, batch, *args, ...)``.

    JEPA's signature is ``(self, batch, *args, **kwargs)`` but it has
    its own inline resolution — exclude it from this sweep until it
    adopts the shared helper. Any OTHER hit is a regression.
    """
    leaked = []
    for p in STRATEGIES_DIR.rglob("*.py"):
        if p.name == "jepa_strategy.py":
            continue
        src = p.read_text()
        if "def _compute_losses_impl(self, batch:" in src:
            leaked.append(p.name)
    assert not leaked, (
        f"F6 regression: {leaked} reverted to the legacy positional "
        f"``batch`` signature. The orchestrator only passes kwargs, so "
        f"this raises TypeError at the first training step. See "
        f"TODO/audit/smoke_audit_20260516.md §F6 / §5g."
    )


# ── Helper is discoverable on the public base class surface ────────────


def test_helper_is_static_and_callable() -> None:
    """``_resolve_legacy_batch`` is a static method (no ``self`` binding)."""
    # ``inspect.getattr_static`` returns the underlying staticmethod
    # descriptor; calling it via the class invokes the function.
    method = inspect.getattr_static(BaseTrainingStrategy, "_resolve_legacy_batch")
    # Python wraps static methods in a ``staticmethod`` descriptor.
    assert isinstance(method, staticmethod), (
        "F6 regression: _resolve_legacy_batch must be a @staticmethod "
        "so it can be called without instantiating a strategy."
    )
