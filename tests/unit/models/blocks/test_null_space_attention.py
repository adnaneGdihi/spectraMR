"""Contracts for null-space dual-domain attention.

Three of these are the leak argument in executable form. The block is allowed to
read the acquisition mask -- that is the scanner's own trajectory, and every
physics-driven reconstruction consumes it -- but it must read NOTHING else about
the fully-sampled truth, and it must leave the observed support alone.
"""

import pytest
import torch

from spectramr.models.blocks.null_space_attention import (
    NullSpaceDualDomainAttention,
    ObservedKeyLinearAttention,
)

CHANNELS = 16
GRID = 32


def _mask(batch: int = 2, size: int = GRID, stride: int = 4, acs: int = 4) -> torch.Tensor:
    """A Cartesian phase-encode mask: periodic lines plus a central ACS band."""
    mask = torch.zeros(batch, 1, size, size)
    mask[..., ::stride] = 1.0
    mask[..., size // 2 - acs // 2 : size // 2 + acs // 2] = 1.0
    return mask


def _block(**kwargs) -> NullSpaceDualDomainAttention:
    torch.manual_seed(0)
    kwargs.setdefault("feature_domain", "kspace")
    return NullSpaceDualDomainAttention(CHANNELS, **kwargs).eval()


# ── the leak argument ────────────────────────────────────────────────────────


def test_observed_support_passes_through_bit_identically() -> None:
    """The ``1 - M`` write gate, asserted on the bit pattern rather than a tolerance."""
    block = _block()
    x = torch.randn(2, CHANNELS, GRID, GRID)
    mask = _mask()

    out = block(x, mask=mask)
    observed = (mask > 0).expand_as(x)

    assert torch.equal(out[observed], x[observed])
    assert not torch.equal(out[~observed], x[~observed]), "the null space must be written"


def test_output_depends_on_nothing_but_x_and_the_mask() -> None:
    """Target invariance, at the block's own boundary.

    Asserted here rather than on a strategy output on purpose: the cold-diffusion
    validation path calibrates S-maps from the fully-sampled reference and
    concatenates them to the model input, so an end-to-end target perturbation
    moves the input for reasons that have nothing to do with this block. The
    claim that belongs to the block is that its forward reads two tensors.
    """
    import inspect

    params = set(inspect.signature(NullSpaceDualDomainAttention.forward).parameters)
    assert params == {"self", "x", "mask"}, (
        "a new forward argument is a new information channel and needs its own "
        f"leak argument; got {sorted(params)}"
    )


def test_a_fully_sampled_mask_is_an_exact_no_op() -> None:
    """The ``R = 1`` rung at ``t = 0``: empty null space, so nothing may be written."""
    block = _block()
    x = torch.randn(2, CHANNELS, GRID, GRID)

    out = block(x, mask=torch.ones(2, 1, GRID, GRID))

    assert torch.equal(out, x)


def test_mask_is_mandatory() -> None:
    """Never inferred from feature magnitude: unconditioned would look conditioned."""
    with pytest.raises(ValueError, match="requires `mask`"):
        _block()(torch.randn(2, CHANNELS, GRID, GRID))


# ── mask alignment across the encoder's k-space crops ────────────────────────


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_align_mask_center_crops_and_stays_binary(level: int) -> None:
    """``KSpaceCrop`` center-crops k-space, so the level mask is the center slice.

    Never ``KSpaceCrop`` itself: it divides by ``scale_factor``, which would turn
    a binary mask into halves and make every bin partially observed.
    """
    full = _mask(batch=1, size=256)
    side = 256 // 2**level

    cropped = NullSpaceDualDomainAttention.align_mask(full, side, side)

    assert cropped.shape == (1, 1, side, side)
    assert set(cropped.unique().tolist()) <= {0.0, 1.0}
    start = (256 - side) // 2
    torch.testing.assert_close(cropped, full[..., start : start + side, start : start + side])


def test_align_mask_reduces_channels_conservatively() -> None:
    """A bin counts as observed only where EVERY channel observed it.

    ``prior_channel_range`` keeps one contrast fully sampled, so a channel-wise
    max would mark that contrast's bins observed for every channel. ``amin`` errs
    toward calling a bin unmeasured, which is the harmless direction, and costs
    no host synchronisation inside the training loop.
    """
    mask = torch.ones(1, 2, 8, 8)
    mask[:, 1, :, ::2] = 0.0

    reduced = NullSpaceDualDomainAttention.align_mask(mask, 8, 8)

    assert reduced.shape == (1, 1, 8, 8)
    torch.testing.assert_close(reduced[0, 0], mask[0, 1])


def test_align_mask_raises_on_a_grid_larger_than_the_mask() -> None:
    with pytest.raises(ValueError, match="larger than the mask"):
        NullSpaceDualDomainAttention.align_mask(torch.ones(1, 1, 16, 16), 32, 32)


def test_align_mask_raises_on_a_non_4d_mask() -> None:
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        NullSpaceDualDomainAttention.align_mask(torch.ones(1, 16, 16), 16, 16)


# ── the masked attention kernel ──────────────────────────────────────────────


@pytest.mark.parametrize("perturb", ["keys", "values"])
def test_unobserved_bins_carry_no_attention_weight(perturb: str) -> None:
    """Tested at the ``attend`` seam because ``x`` feeds q, k and v alike.

    Through ``forward`` every perturbation also moves the queries that read it
    (``InstanceNorm2d`` alone couples every position), so an input-level test
    could never isolate the restriction this block exists for.
    """
    torch.manual_seed(0)
    attn = ObservedKeyLinearAttention(CHANNELS, num_heads=4).eval()
    batch, heads, dim, side = 1, 4, 4, 8
    n = side * side
    q = torch.rand(batch, heads, dim, n) + 1.0
    k = torch.rand(batch, heads, dim, n) + 1.0
    v = torch.randn(batch, heads, dim, n)
    observed = torch.zeros(1, 1, side, side)
    observed[..., ::2] = 1.0
    unobserved = (observed.reshape(1, 1, 1, n) == 0).expand(batch, heads, dim, n)

    baseline = attn.attend(q, k, v, observed)
    if perturb == "keys":
        k = k.masked_fill(unobserved, 1e3)
    else:
        v = v.masked_fill(unobserved, 1e3)

    assert torch.equal(attn.attend(q, k, v, observed), baseline)


def test_observed_bins_do_carry_weight() -> None:
    """The complement, so the test above cannot pass by the block ignoring everything."""
    torch.manual_seed(0)
    attn = ObservedKeyLinearAttention(CHANNELS, num_heads=4).eval()
    batch, heads, dim, side = 1, 4, 4, 8
    n = side * side
    q = torch.rand(batch, heads, dim, n) + 1.0
    k = torch.rand(batch, heads, dim, n) + 1.0
    v = torch.randn(batch, heads, dim, n)
    observed = torch.zeros(1, 1, side, side)
    observed[..., ::2] = 1.0
    is_observed = (observed.reshape(1, 1, 1, n) > 0).expand(batch, heads, dim, n)

    baseline = attn.attend(q, k, v, observed)
    moved = attn.attend(q, k, v.masked_fill(is_observed, 1e3), observed)

    assert not torch.allclose(moved, baseline)


def test_empty_observed_support_yields_zero_not_nan() -> None:
    """Unreachable with a central ACS band, but it must degrade to 0, never NaN."""
    torch.manual_seed(0)
    attn = ObservedKeyLinearAttention(CHANNELS, num_heads=4).eval()

    out = attn(torch.randn(1, CHANNELS, 8, 8), torch.zeros(1, 1, 8, 8))

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, attn.proj(torch.zeros(1, CHANNELS, 8, 8)))
    assert not out.isnan().any()


# ── shape, domain and gradient contracts ─────────────────────────────────────


@pytest.mark.parametrize("feature_domain", ["kspace", "image"])
def test_output_domain_matches_input_domain(feature_domain: str) -> None:
    block = _block(feature_domain=feature_domain)
    x = torch.randn(2, CHANNELS, GRID, GRID)

    assert block(x, mask=_mask()).shape == x.shape


def test_unknown_feature_domain_raises() -> None:
    with pytest.raises(ValueError, match="Unknown feature_domain"):
        NullSpaceDualDomainAttention(CHANNELS, feature_domain="latent")


def test_odd_channel_count_raises() -> None:
    with pytest.raises(ValueError, match="even"):
        NullSpaceDualDomainAttention(15, feature_domain="kspace")


def test_both_branches_and_the_fusion_receive_gradient() -> None:
    """All three are the mechanism; a dead one would be a facade (pitfall #16)."""
    block = _block().train()
    out = block(torch.randn(2, CHANNELS, GRID, GRID), mask=_mask())
    (out - torch.randn_like(out)).pow(2).mean().backward()

    for name in ("kspace_attn.proj", "image_attn.proj", "fuse"):
        grad = max(
            0.0 if p.grad is None else p.grad.abs().max().item()
            for p in block.get_submodule(name).parameters()
        )
        assert grad > 0.0, f"{name} receives no gradient"


def test_block_is_not_an_identity_at_init() -> None:
    """It is wrapped by IdentityAtInitAttention, which owns identity-at-init (#471).

    A second zero-init here would reproduce the saddle that froze ``self``,
    ``kernelized`` and ``sparse``.
    """
    block = _block()
    x = torch.randn(2, CHANNELS, GRID, GRID)

    assert not torch.allclose(block(x, mask=_mask()), x)


def test_sample_density_attention_normalises_over_samples_not_queries() -> None:
    """The two seams differ in WHICH axis carries the normaliser.

    ``masked_linear_attention`` divides per query and returns an average;
    ``sample_density_attention`` weights per sample and returns a sum. Feeding
    both a constant value field separates them: the average reproduces the
    constant, the sum does not have to.
    """
    from spectramr.models.blocks.null_space_attention import (
        masked_linear_attention,
        sample_density_attention,
    )

    torch.manual_seed(0)
    q = torch.rand(1, 2, 4, 6).abs() + 0.1
    k = torch.rand(1, 2, 4, 9).abs() + 0.1
    v = torch.ones(1, 2, 1, 9, dtype=torch.complex64)
    observed = torch.ones(1, 9)

    average = masked_linear_attention(q, k, v, observed)
    assert average.abs().sub(1.0).max() < 1e-4, "the per-query form must average to the constant"

    summed = sample_density_attention(q, k, v, observed)
    assert summed.shape == average.shape
    assert not torch.allclose(summed.abs(), average.abs(), atol=1e-3)


def test_sample_density_attention_is_duplicate_invariant() -> None:
    """One Pipe-Menon step: a repeated sample raises density and halves its weight.

    Without that, an unnormalised sum would let the sampling pattern leak into
    the value -- the property the per-query denominator used to provide.
    """
    from spectramr.models.blocks.null_space_attention import sample_density_attention

    torch.manual_seed(1)
    q = torch.rand(1, 1, 8, 5).abs() + 0.1
    k = torch.rand(1, 1, 8, 12).abs() + 0.1
    v = torch.ones(1, 1, 1, 12, dtype=torch.complex64)

    plain = sample_density_attention(q, k, v, torch.ones(1, 12))
    doubled = sample_density_attention(
        q,
        torch.cat([k, k[..., :4]], dim=-1),
        torch.cat([v, v[..., :4]], dim=-1),
        torch.ones(1, 16),
    )
    shift = ((doubled - plain).abs() / plain.abs().clamp_min(1e-6)).median()
    assert float(shift) < 0.15, f"duplication moved the output by {float(shift):.3f}"


def test_a_dropped_sample_contributes_no_density() -> None:
    """PLANTED VIOLATION: the mask must enter the density sum, not only the values.

    A masked sample that still crowded its neighbours would depress their
    weights, so a rung's output would depend on spokes it did not acquire.
    """
    from spectramr.models.blocks.null_space_attention import sample_density_attention

    torch.manual_seed(2)
    q = torch.rand(1, 1, 8, 5).abs() + 0.1
    k = torch.rand(1, 1, 8, 12).abs() + 0.1
    v = torch.ones(1, 1, 1, 12, dtype=torch.complex64)
    keep = torch.ones(1, 12)
    keep[0, 8:] = 0.0

    masked = sample_density_attention(q, k, v, keep)
    cropped = sample_density_attention(q, k[..., :8], v[..., :8], torch.ones(1, 8))
    assert torch.allclose(masked, cropped, atol=1e-5), (
        "masking a sample must equal never having acquired it"
    )
