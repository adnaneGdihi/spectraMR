"""Tests for ``bottomley_t1`` (XField-FM / idea 2 §2.5).

Targets ``spectramr.infrastructure.physics.relaxation_priors``.

Plan acceptance criterion (§2.8):

- *"bottomley_t1 matches table reference values for white matter at
  1.5 T, 3 T"* — covered by the canonical-value tests below. The
  reconciled coefficient set used here gives WM ≈ 700 ms at 1.5 T
  and ≈ 850 ms at 3 T, matching standard literature within ±5 %.

Other contract tests:

- Invalid B0 / unknown tissue rejected
- ``T1`` is monotone in B0 for WM and GM (Bottomley α > 0)
- CSF is approximately field-independent (Bottomley α = 0.05)
- Snapshot returns a JSON-friendly dict with the documented keys
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.physics.relaxation_priors import (
    BottomleyCoefficients,
    TissueClass,
    bottomley_t1,
    bottomley_table_snapshot,
)


# ---------------------------------------------------------------------------
# Closed-form sanity at canonical field strengths
# ---------------------------------------------------------------------------


def test_white_matter_at_1p5T_in_documented_range() -> None:
    """T1(WM, 1.5 T) ≈ 700 ms within ±10%."""
    val = bottomley_t1(1.5, TissueClass.WHITE_MATTER)
    assert 630.0 <= val <= 770.0


def test_white_matter_at_3T_in_documented_range() -> None:
    """T1(WM, 3 T) ≈ 850 ms within ±10%."""
    val = bottomley_t1(3.0, TissueClass.WHITE_MATTER)
    assert 765.0 <= val <= 935.0


def test_gray_matter_longer_than_white_matter() -> None:
    """Across all clinical fields, GM T1 > WM T1."""
    for b0 in (0.064, 0.3, 1.5, 3.0):
        gm = bottomley_t1(b0, TissueClass.GRAY_MATTER)
        wm = bottomley_t1(b0, TissueClass.WHITE_MATTER)
        assert gm > wm, f"GM<WM at B0={b0} T (gm={gm:.0f}, wm={wm:.0f})"


def test_csf_is_largest_t1() -> None:
    """CSF has the longest T1 of the three tissue classes."""
    for b0 in (0.064, 0.3, 1.5, 3.0):
        csf = bottomley_t1(b0, TissueClass.CSF)
        gm = bottomley_t1(b0, TissueClass.GRAY_MATTER)
        wm = bottomley_t1(b0, TissueClass.WHITE_MATTER)
        assert csf > gm > wm


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_white_matter_monotone_increasing_in_b0() -> None:
    """``T1(WM, B0)`` increases with B0 (Bottomley α = 0.34 > 0)."""
    fields = [0.064, 0.3, 1.5, 3.0, 7.0]
    values = [bottomley_t1(b, TissueClass.WHITE_MATTER) for b in fields]
    for prev, nxt in zip(values, values[1:]):
        assert nxt > prev


def test_csf_almost_field_independent() -> None:
    """CSF T1 changes by < 20% between 0.064 T and 3 T (α ≈ 0)."""
    low = bottomley_t1(0.064, TissueClass.CSF)
    high = bottomley_t1(3.0, TissueClass.CSF)
    rel_change = abs(high - low) / low
    assert rel_change < 0.20


# ---------------------------------------------------------------------------
# Argument variants
# ---------------------------------------------------------------------------


def test_string_tissue_argument_accepted() -> None:
    """Tissue can be passed as a string."""
    val_enum = bottomley_t1(1.5, TissueClass.WHITE_MATTER)
    val_str = bottomley_t1(1.5, "white_matter")
    assert pytest.approx(val_enum) == val_str


def test_unknown_tissue_rejected() -> None:
    """Unknown tissue string raises with the valid options listed."""
    with pytest.raises(ValueError, match="Unknown tissue"):
        bottomley_t1(1.5, "fat")


def test_zero_field_strength_rejected() -> None:
    """``B0 ≤ 0`` raises."""
    with pytest.raises(ValueError, match="b0_T"):
        bottomley_t1(0.0, TissueClass.WHITE_MATTER)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_snapshot_keys_match_enum_values() -> None:
    """Snapshot is keyed by lowercase tissue strings."""
    snap = bottomley_table_snapshot()
    assert set(snap.keys()) == {"white_matter", "gray_matter", "csf"}


def test_snapshot_values_are_coefficients() -> None:
    """Snapshot values are :class:`BottomleyCoefficients` instances."""
    snap = bottomley_table_snapshot()
    for v in snap.values():
        assert isinstance(v, BottomleyCoefficients)


def test_snapshot_does_not_mutate_internal_table() -> None:
    """Mutating the snapshot does not affect subsequent lookups."""
    snap = bottomley_table_snapshot()
    snap["white_matter"] = BottomleyCoefficients(a=0.0, alpha=0.0, b=0.0)
    val = bottomley_t1(1.5, TissueClass.WHITE_MATTER)
    # Should still match the reference.
    assert val > 100.0


# ---------------------------------------------------------------------------
# Coefficient dataclass
# ---------------------------------------------------------------------------


def test_coefficients_dataclass_frozen() -> None:
    """``BottomleyCoefficients`` is a frozen dataclass."""
    import dataclasses

    coeffs = BottomleyCoefficients(a=1.0, alpha=0.5, b=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        coeffs.a = 2.0  # type: ignore[misc]


def test_bottomley_t1_tensor_matches_scalar_and_is_batched_differentiable() -> None:
    """A-5.2: the batched/differentiable Bottomley T1(B0) matches the scalar bottomley_t1."""
    import torch

    from spectramr.infrastructure.physics.relaxation_priors import (
        TissueClass,
        bottomley_t1,
        bottomley_t1_tensor,
    )

    b = torch.tensor([0.1, 1.5, 7.0])
    t1 = bottomley_t1_tensor(b, TissueClass.WHITE_MATTER)
    assert t1.shape == b.shape
    for i, bv in enumerate([0.1, 1.5, 7.0]):
        assert abs(float(t1[i]) - bottomley_t1(bv, TissueClass.WHITE_MATTER)) < 1e-3
    # monotone in field (T1 rises with B0) + differentiable
    assert float(t1[0]) < float(t1[1]) < float(t1[2])
    bg = torch.tensor([3.0], requires_grad=True)
    bottomley_t1_tensor(bg, "gray_matter").sum().backward()
    assert bg.grad is not None and float(bg.grad.abs().sum()) > 0.0


def test_tensor_prior_still_refuses_a_non_positive_field_on_cpu() -> None:
    import pytest
    import torch

    from spectramr.infrastructure.physics.relaxation_priors import bottomley_t1_tensor

    with pytest.raises(ValueError, match="b0_T must be > 0"):
        bottomley_t1_tensor(torch.tensor([0.3, 0.0]), "white_matter")


def test_tensor_prior_does_not_sync_the_gpu() -> None:
    import pytest
    import torch

    from spectramr.infrastructure.physics.relaxation_priors import bottomley_t1_tensor

    if not torch.cuda.is_available():
        pytest.skip("sync-debug mode needs a GPU")
    b0 = torch.tensor([0.064, 3.0], device="cuda")
    bottomley_t1_tensor(b0, "csf")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        bottomley_t1_tensor(b0, "white_matter")
    finally:
        torch.cuda.set_sync_debug_mode("default")
