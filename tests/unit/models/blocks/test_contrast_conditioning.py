"""Shared contrast-conditioning helper for field-FiLM models.

The helper centralises the ``sequence_features`` construction that every
field-aware FiLM generator needs when it opts into multi-contrast
conditioning: widen the :class:`FieldFiLMBlock` ``sequence_dim`` by
``num_contrasts`` and concatenate a per-sample contrast one-hot. It enforces
the two always-on invariants (CLAUDE.md #15 raise-on-missing, #9
raise-on-out-of-range) in one place instead of inlining them per model.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.contrast_conditioning import (
    build_contrast_sequence,
    contrast_sequence_dim,
)

# --- contrast_sequence_dim ------------------------------------------------


def test_seq_dim_no_base_disabled_is_scalar_placeholder() -> None:
    # No intrinsic sequence features + conditioning off => the FieldFiLMBlock
    # sequence_dim>=1 invariant is met by a single zero placeholder.
    assert contrast_sequence_dim(0, num_contrasts=3, enabled=False) == 1


def test_seq_dim_no_base_enabled_is_num_contrasts() -> None:
    # No base features + conditioning on => the one-hot REPLACES the placeholder
    # (matches FieldVelocityUNet's original seq_dim = num_contrasts).
    assert contrast_sequence_dim(0, num_contrasts=3, enabled=True) == 3


def test_seq_dim_with_base_disabled_is_base() -> None:
    assert contrast_sequence_dim(2, num_contrasts=3, enabled=False) == 2


def test_seq_dim_with_base_enabled_appends() -> None:
    # Real base features (e.g. a time embedding) are PRESERVED; contrast appends.
    assert contrast_sequence_dim(2, num_contrasts=3, enabled=True) == 5


# --- build_contrast_sequence: disabled path -------------------------------


def test_build_disabled_no_base_returns_zero_placeholder() -> None:
    seq = build_contrast_sequence(
        None,
        None,
        num_contrasts=3,
        enabled=False,
        batch_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert seq.shape == (4, 1)
    assert torch.equal(seq, torch.zeros(4, 1))


def test_build_disabled_with_base_is_identity() -> None:
    base = torch.randn(4, 2)
    seq = build_contrast_sequence(
        base,
        torch.tensor([0, 1, 2, 0]),
        num_contrasts=3,
        enabled=False,
        batch_size=4,
        device=base.device,
        dtype=base.dtype,
    )
    assert seq is base  # untouched passthrough


# --- build_contrast_sequence: enabled path --------------------------------


def test_build_enabled_no_base_is_one_hot() -> None:
    cid = torch.tensor([0, 2, 1])
    seq = build_contrast_sequence(
        None,
        cid,
        num_contrasts=3,
        enabled=True,
        batch_size=3,
        device=cid.device,
        dtype=torch.float32,
    )
    assert seq.shape == (3, 3)
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert torch.equal(seq, expected)


def test_build_enabled_with_base_concatenates() -> None:
    base = torch.randn(2, 4)  # e.g. a 4-dim time embedding
    cid = torch.tensor([1, 0])
    seq = build_contrast_sequence(
        base,
        cid,
        num_contrasts=3,
        enabled=True,
        batch_size=2,
        device=base.device,
        dtype=base.dtype,
    )
    assert seq.shape == (2, 7)  # 4 base + 3 contrast
    assert torch.equal(seq[:, :4], base)  # base preserved
    assert torch.equal(seq[:, 4:], torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))


def test_build_enabled_missing_contrast_id_raises() -> None:
    # #15: a wired knob with no value must fail loud, not silently drop.
    with pytest.raises(ValueError, match="contrast_id"):
        build_contrast_sequence(
            None,
            None,
            num_contrasts=3,
            enabled=True,
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_build_enabled_out_of_range_id_raises() -> None:
    # #9: no silent clamp — an id >= num_contrasts must raise (via one_hot).
    with pytest.raises(RuntimeError):
        build_contrast_sequence(
            None,
            torch.tensor([0, 5]),
            num_contrasts=3,
            enabled=True,
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_build_enabled_respects_dtype() -> None:
    cid = torch.tensor([0, 1])
    seq = build_contrast_sequence(
        None,
        cid,
        num_contrasts=2,
        enabled=True,
        batch_size=2,
        device=cid.device,
        dtype=torch.float64,
    )
    assert seq.dtype == torch.float64


# --- broadcast_conditioning_map -------------------------------------------
#
# The channel-concatenation half of the same construction, used by conditioned
# CRITICS (#1931). It cannot use FiLM: a critic wraps whatever inner critic the
# registry hands back and has no access to that critic's block structure, so it
# widens the input instead.


def _proj(out_channels: int = 4, in_dim: int | None = None):
    from spectramr.models.blocks.contrast_conditioning import contrast_sequence_dim

    dim = in_dim if in_dim is not None else contrast_sequence_dim(8, 3, enabled=True)
    torch.manual_seed(0)
    return torch.nn.Linear(dim, out_channels)


def _map(image, t, c, **kw):
    from spectramr.models.blocks.contrast_conditioning import broadcast_conditioning_map

    kwargs = {
        "time_embed_dim": 8,
        "num_contrasts": 3,
        "projection": _proj(),
        "owner": "FakeCritic",
    }
    kwargs.update(kw)
    return broadcast_conditioning_map(image, t, c, **kwargs)


def test_the_map_matches_the_image_batch_and_spatial_shape() -> None:
    image = torch.randn(2, 1, 6, 6)
    out = _map(image, torch.tensor([3, 700]), torch.tensor([0, 2]))
    assert out.shape == (2, 4, 6, 6)


def test_the_map_broadcasts_over_any_spatial_rank() -> None:
    """A 3D arm must work without a rank special-case."""
    image = torch.randn(2, 1, 4, 5, 6)
    out = _map(image, torch.tensor([1, 2]), torch.tensor([0, 1]))
    assert out.shape == (2, 4, 4, 5, 6)


def test_the_map_is_constant_across_space_and_varies_across_the_batch() -> None:
    """Constant in space is the definition of a broadcast label; varying across
    the batch is what makes it a per-sample condition rather than a bias."""
    out = _map(torch.randn(2, 1, 5, 5), torch.tensor([10, 900]), torch.tensor([0, 2]))
    assert torch.allclose(out[:, :, 0, 0][..., None, None].expand_as(out), out)
    assert not torch.allclose(out[0], out[1])


@pytest.mark.parametrize(
    ("t", "c", "match"),
    [
        (None, torch.tensor([0, 1]), "without `timesteps`"),
        (torch.tensor([0, 1]), None, "without `contrast_idx`"),
    ],
)
def test_a_missing_payload_raises_and_names_its_owner(t, c, match) -> None:
    """#3: the caller declared the flag, so silence here is a lie for a run."""
    with pytest.raises(ValueError, match=match) as exc:
        _map(torch.randn(2, 1, 4, 4), t, c)
    assert "FakeCritic" in str(exc.value), "the error must name the caller, not the helper"


def test_a_batch_mismatch_raises_rather_than_broadcasting() -> None:
    """The silent-corruption shape: (2,) against 4 samples broadcasts cleanly
    in most of torch, and would label each sample with another's condition."""
    with pytest.raises(ValueError, match="batch mismatch"):
        _map(
            torch.randn(4, 1, 4, 4),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
        )


def test_an_out_of_range_contrast_id_raises() -> None:
    """#9, inherited from build_contrast_sequence -> one_hot: never a clamp."""
    with pytest.raises(RuntimeError):
        _map(torch.randn(2, 1, 4, 4), torch.tensor([0, 1]), torch.tensor([0, 9]))


def test_the_map_follows_the_image_dtype() -> None:
    image = torch.randn(2, 1, 4, 4, dtype=torch.float64)
    out = _map(image, torch.tensor([0, 1]), torch.tensor([0, 1]), projection=_proj().double())
    assert out.dtype == torch.float64
