"""Contracts for the shared k-space attention kernels.

Each mechanism is asserted beside a planted violation -- the defect shape it
replaces -- so a tolerance that nothing could fail is not mistaken for a test
(non-negotiable 15).
"""

import pytest
import torch
from torch import nn

from spectramr.models.blocks.kspace_attention_kernels import (
    MaskedInstanceNorm2d,
    align_mask_to_grid,
    conjugate_partner,
    masked_complex_rms,
    masked_linear_attention,
    pe_biased_linear_attention,
    pe_distance_kernel,
    stratified_conjugate_subsample,
)

# ── phase-encode distance bias ───────────────────────────────────────────────


def _two_column_weights(slope: float, width: int = 128, query_col: int = 30):
    """Implied weights of a query column on acquired columns 3 and 60 away.

    Keys are identical everywhere (equal content magnitude), so any preference
    between the two columns comes from the positional bias alone. Value channel
    0 marks the near column and channel 1 the far one, which turns the output
    at the query into the two attention weights.
    """
    height = 4
    near, far = query_col + 3, query_col + 60
    observed = torch.zeros(1, 1, height, width)
    observed[..., [near, far]] = 1.0
    n = height * width
    q = torch.ones(1, 1, 2, n)
    k = torch.ones(1, 1, 2, n)
    v = torch.zeros(1, 1, 2, height, width)
    v[0, 0, 0, :, near] = 1.0
    v[0, 0, 1, :, far] = 1.0
    out = pe_biased_linear_attention(
        q, k, v.reshape(1, 1, 2, n), observed, torch.tensor([slope])
    ).reshape(1, 1, 2, height, width)
    return out[0, 0, 0, 0, query_col], out[0, 0, 1, 0, query_col]


def test_pe_bias_prefers_the_nearest_acquired_column() -> None:
    w_near, w_far = _two_column_weights(slope=1.0)
    assert w_near > w_far
    torch.testing.assert_close(w_near + w_far, torch.tensor(1.0))


def test_without_the_bias_the_two_columns_tie() -> None:
    """Planted violation: content-only linear attention cannot rank them."""
    w_near, w_far = _two_column_weights(slope=0.0)
    torch.testing.assert_close(w_near, w_far)


def test_pe_biased_attention_matches_the_dense_reference() -> None:
    """The per-column factorisation is exact, not an approximation."""
    torch.manual_seed(0)
    b, h, d, e, height, width = 2, 3, 4, 5, 3, 6
    n = height * width
    q, k = torch.rand(b, h, d, n) + 0.1, torch.rand(b, h, d, n) + 0.1
    v = torch.complex(torch.randn(b, h, e, n), torch.randn(b, h, e, n))
    observed = (torch.rand(b, 1, height, width) > 0.4).float()
    slope = torch.tensor([0.5, 1.0, 2.0])

    got = pe_biased_linear_attention(q, k, v, observed, slope)

    cols = torch.arange(n) % width
    bias = pe_distance_kernel(width, slope)[:, cols][:, :, cols]  # [h, n, n]
    scores = torch.einsum("bhdi,bhdj->bhij", q, k) * bias * observed.reshape(b, 1, 1, n)
    weights = scores / scores.sum(-1, keepdim=True)
    ref = torch.einsum("bhij,bhej->bhei", weights.to(v.dtype), v)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


def test_zero_slope_reduces_to_masked_linear_attention() -> None:
    torch.manual_seed(1)
    q, k, v = torch.rand(1, 2, 3, 24) + 0.1, torch.rand(1, 2, 3, 24) + 0.1, torch.randn(1, 2, 3, 24)
    observed = (torch.rand(1, 1, 4, 6) > 0.5).float()
    torch.testing.assert_close(
        pe_biased_linear_attention(q, k, v, observed, torch.zeros(2)),
        masked_linear_attention(q, k, v, observed),
    )


# ── normalisation over the acquired support ──────────────────────────────────


def _zero_filled(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    observed = torch.zeros(2, 1, 8, 8)
    observed[..., ::4] = 1.0
    x = (torch.randn(2, 6, 8, 8) * 3.0 + 5.0) * observed
    return x, observed


def test_masked_instance_norm_ignores_zero_filled_bins() -> None:
    x, observed = _zero_filled()
    norm = MaskedInstanceNorm2d(6)
    y = norm(x, observed)
    garbage = x + 1e3 * (1.0 - observed)
    sel = observed.bool().expand_as(x)

    torch.testing.assert_close(norm(garbage, observed)[sel], y[sel])
    mean = (y * observed).sum((-2, -1)) / observed.sum((-2, -1))
    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-5, rtol=0)


def test_plain_instance_norm_is_moved_by_zero_filled_bins() -> None:
    """Planted violation: the unmasked statistics shift with the acceleration."""
    x, observed = _zero_filled()
    y = nn.InstanceNorm2d(6)(x)
    mean = (y * observed).sum((-2, -1)) / observed.sum((-2, -1))
    assert mean.abs().max() > 0.5


def test_masked_instance_norm_with_full_support_is_instance_norm() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 6, 8, 8)
    torch.testing.assert_close(
        MaskedInstanceNorm2d(6)(x, torch.ones(2, 1, 8, 8)),
        nn.InstanceNorm2d(6, affine=True)(x),
        rtol=1e-4,
        atol=1e-5,
    )


def test_masked_complex_rms_reads_only_observed_power() -> None:
    x, observed = _zero_filled()
    gain = torch.ones(3)
    y = masked_complex_rms(x, observed, gain, 1e-5)
    y_garbage = masked_complex_rms(x + 1e3 * (1.0 - observed), observed, gain, 1e-5)
    sel = observed.bool().expand_as(x)
    torch.testing.assert_close(y_garbage[sel], y[sel])
    power = (y[:, 0::2] ** 2 + y[:, 1::2] ** 2) * observed
    rms = power.sum((1, 2, 3)) / (observed.sum((1, 2, 3)) * 3)
    torch.testing.assert_close(rms, torch.ones(2), rtol=1e-4, atol=1e-4)


# ── mask alignment ───────────────────────────────────────────────────────────


def test_align_mask_to_grid_centre_crops() -> None:
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., 8] = 1.0
    out = align_mask_to_grid(mask, 8, 8)
    assert out.shape == (1, 1, 8, 8)
    assert torch.equal(out[0, 0, :, 4], torch.ones(8)), "DC column stays at N // 2"
    with pytest.raises(ValueError, match="larger than the mask"):
        align_mask_to_grid(mask, 32, 32)


@pytest.mark.parametrize("full", [372, 396, 250, 218, 256, 320])
def test_align_mask_to_grid_follows_the_kspace_crop_chain(full: int) -> None:
    """Each level must see the columns the encoder's repeated ``KSpaceCrop`` kept."""
    from spectramr.models.generators.kspace_cold_diffusion_generator import KSpaceCrop

    columns = (
        torch.arange(full, dtype=torch.float32).reshape(1, 1, 1, full).expand(1, 1, full, full)
    )
    crop, level = KSpaceCrop(scale_factor=2), columns
    while level.shape[-1] >= 8:
        level = crop(level) * 2  # undo the amplitude rescale; only positions matter here
        height, width = level.shape[-2:]
        mask = torch.zeros(1, 1, full, full)
        mask[..., int(level[0, 0, 0, 0])] = 1.0
        aligned = align_mask_to_grid(mask, height, width)
        assert aligned[0, 0, 0, 0] == 1.0, f"W={full}: level {width} starts off the crop chain"


def test_linear_attention_sums_do_not_overflow_in_half() -> None:
    """The key sum over a 256^2 grid is ~1.3e5, past the fp16 maximum of 65504."""
    n = 256 * 256
    q = torch.full((1, 1, 2, n), 2.0, dtype=torch.float16)
    k = torch.full((1, 1, 2, n), 2.0, dtype=torch.float16)
    v = torch.ones(1, 1, 2, n, dtype=torch.float16)
    out = masked_linear_attention(q, k, v, torch.ones(1, n))
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), torch.ones(1, 1, 2, n))


# ── conjugate-pair stratified subsample ──────────────────────────────────────


@pytest.mark.parametrize("size", [16, 17])
def test_conjugate_partner_is_an_involution_about_dc(size: int) -> None:
    idx = torch.arange(size * size)
    partner = conjugate_partner(idx, size, size)
    assert torch.equal(conjugate_partner(partner, size, size), idx)
    dc = (size // 2) * size + size // 2
    assert partner[dc] == dc


def _annulus(size: int, r_in: float, r_out: float) -> torch.Tensor:
    ky, kx = torch.meshgrid(
        torch.arange(size) - size // 2, torch.arange(size) - size // 2, indexing="ij"
    )
    r = (ky**2 + kx**2).float().sqrt()
    return ((r >= r_in) & (r < r_out)).flatten().nonzero().flatten()


def test_subsample_covers_every_row_and_keeps_pairs() -> None:
    size, cap = 64, 256
    idx = _annulus(size, 16.0, 31.0)
    assert idx.numel() > cap
    chosen = stratified_conjugate_subsample(idx, size, size, cap, seed=0)

    assert chosen.numel() <= cap
    assert torch.isin(chosen, idx).all()
    assert torch.isin(conjugate_partner(chosen, size, size), chosen).all()
    assert set((chosen // size).tolist()) == set((idx // size).tolist())
    rows = chosen // size
    assert (rows < size // 2).any() and (rows > size // 2).any()
    assert torch.equal(chosen, stratified_conjugate_subsample(idx, size, size, cap, seed=0))


def test_a_raster_prefix_misses_rows() -> None:
    """Planted violation: the floor-stride + ``[:cap]`` prefix leaves rows unattended."""
    size, cap = 64, 256
    idx = _annulus(size, 16.0, 31.0)
    stride = max(1, idx.numel() // cap)
    prefix = idx[torch.arange(0, idx.numel(), stride)[:cap]]
    assert set((prefix // size).tolist()) != set((idx // size).tolist())
    assert not torch.isin(conjugate_partner(prefix, size, size), prefix).all()


def test_under_cap_returns_the_band_unchanged() -> None:
    idx = _annulus(16, 2.0, 5.0)
    assert torch.equal(stratified_conjugate_subsample(idx, 16, 16, 4096, seed=0), idx)
