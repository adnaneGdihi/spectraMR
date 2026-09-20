"""Contracts for phase-equivariant null-space attention.

The two claims that make this block different from the rest of the family are
equivariance to a global phase and an exact conjugate-symmetry prior. Each is
asserted here together with a planted violation, because a tolerance that
nothing can fail is not a test (non-negotiable 15).
"""

import pytest
import torch
from torch import nn

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.blocks.hermitian_null_space_attention import (
    HermitianNullSpaceAttention,
    PhaseInvariantScoreAttention,
    hermitian_mirror,
)

CHANNELS = 16
GRID = 32


def _mask(batch: int = 2, size: int = GRID, stride: int = 4, acs: int = 4) -> torch.Tensor:
    mask = torch.zeros(batch, 1, size, size)
    mask[..., ::stride] = 1.0
    mask[..., size // 2 - acs // 2 : size // 2 + acs // 2] = 1.0
    return mask


def _block(**kwargs) -> HermitianNullSpaceAttention:
    torch.manual_seed(0)
    kwargs.setdefault("feature_domain", "kspace")
    return HermitianNullSpaceAttention(CHANNELS, **kwargs).eval()


def _rotate(x: torch.Tensor, phi: float) -> torch.Tensor:
    """Multiply the interleaved complex channels by ``exp(i*phi)``."""
    z = torch.complex(x[:, 0::2], x[:, 1::2]) * torch.exp(1j * torch.tensor(phi))
    return torch.stack([z.real, z.imag], dim=2).flatten(1, 2)


# ── global-phase equivariance ────────────────────────────────────────────────


@pytest.mark.parametrize("phi", [0.3, 1.7, -2.5])
def test_block_is_equivariant_to_a_global_phase(phi: float) -> None:
    """``f(exp(i*phi) x) == exp(i*phi) f(x)``, which the forward operator obeys."""
    block = _block()
    x = torch.randn(2, CHANNELS, GRID, GRID)
    mask = _mask()

    with torch.no_grad():
        rotated_output = _rotate(block(x, mask=mask), phi)
        output_of_rotated = block(_rotate(x, phi), mask=mask)

    relative = (rotated_output - output_of_rotated).abs().max() / output_of_rotated.abs().max()
    assert relative.item() < 1e-5


def test_a_real_valued_projection_breaks_equivariance() -> None:
    """Planted violation: swap one ComplexConv2d for a real one and the bound fails.

    Without this the 1e-5 bound above proves only that the test input was small.
    """
    block = _block()
    inner = block.kspace_attn
    inner.v_proj = nn.Conv2d(CHANNELS, CHANNELS, 1, bias=False)
    x = torch.randn(2, CHANNELS, GRID, GRID)
    mask = _mask()

    with torch.no_grad():
        rotated_output = _rotate(block(x, mask=mask), 1.1)
        output_of_rotated = block(_rotate(x, 1.1), mask=mask)

    relative = (rotated_output - output_of_rotated).abs().max() / output_of_rotated.abs().max()
    assert relative.item() > 1e-3, "a real-valued projection must break equivariance"


def test_scores_are_invariant_while_values_rotate() -> None:
    """The mechanism behind the equivariance, not just its effect."""
    torch.manual_seed(0)
    attn = PhaseInvariantScoreAttention(CHANNELS // 2, num_features=8).eval()
    x = torch.randn(1, CHANNELS, 8, 8)

    with torch.no_grad():
        base = attn._features(attn.q_proj(attn.norm(x)))
        rotated = attn._features(attn.q_proj(attn.norm(_rotate(x, 0.8))))

    torch.testing.assert_close(base, rotated, rtol=1e-5, atol=1e-6)
    assert (base > 0).all(), "features must stay positive for the denominator"


# ── the conjugate-symmetry prior ─────────────────────────────────────────────


def test_hermitian_mirror_is_the_true_fourier_partner() -> None:
    """``flip`` alone is off by one on a centred even grid; the roll fixes it."""
    size = 8
    index = torch.arange(size * size, dtype=torch.float32).reshape(1, 1, size, size)
    mirrored = hermitian_mirror(index)
    # The DC bin of an fftshift-ed even grid is its own partner.
    centre = size // 2
    assert mirrored[0, 0, centre, centre] == index[0, 0, centre, centre]
    torch.testing.assert_close(hermitian_mirror(mirrored), index)


def test_the_prior_recovers_unmeasured_kspace_of_a_real_image_exactly() -> None:
    """A real object has Hermitian k-space, so this is not an approximation."""
    torch.manual_seed(0)
    image = torch.randn(1, 1, GRID, GRID)
    kspace = fft2c(torch.complex(image, torch.zeros_like(image)))
    mask = torch.zeros(1, 1, GRID, GRID)
    mask[..., ::3] = 1.0
    mask[..., GRID // 2 - 3 : GRID // 2 + 3] = 1.0  # asymmetric: mirrors exist

    prediction, mirror_observed = _block().hermitian_prior(kspace, mask)
    usable = ((1.0 - mask) * mirror_observed).bool().expand_as(kspace)

    assert usable.sum() > 0, "the fixture must leave null bins with observed mirrors"
    error = (prediction[usable] - kspace[usable]).abs().max() / kspace.abs().max()
    assert error.item() < 1e-5


@pytest.mark.parametrize("phi", [0.0, 0.9, 2.2])
def test_the_fitted_gain_absorbs_an_unknown_global_phase(phi: float) -> None:
    """``g`` is fitted, not assumed 1 -- which is what survives ``exp(2i*phi)``."""
    torch.manual_seed(0)
    image = torch.randn(1, 1, GRID, GRID)
    kspace = fft2c(torch.complex(image, torch.zeros_like(image))) * torch.exp(
        1j * torch.tensor(phi)
    )
    mask = torch.zeros(1, 1, GRID, GRID)
    mask[..., ::3] = 1.0
    mask[..., GRID // 2 - 3 : GRID // 2 + 3] = 1.0

    prediction, mirror_observed = _block().hermitian_prior(kspace, mask)
    usable = ((1.0 - mask) * mirror_observed).bool().expand_as(kspace)

    error = (prediction[usable] - kspace[usable]).abs().max() / kspace.abs().max()
    assert error.item() < 1e-5


def test_the_prior_is_gated_off_where_no_mirror_was_acquired() -> None:
    """A centre-symmetric mask leaves the stream nothing, and must not invent any."""
    mask = torch.zeros(1, 1, GRID, GRID)
    mask[..., GRID // 2 - 4 : GRID // 2 + 4] = 1.0
    mask = ((mask + hermitian_mirror(mask)) > 0).float()  # force exact symmetry

    _, mirror_observed = _block().hermitian_prior(
        torch.randn(1, CHANNELS // 2, GRID, GRID, dtype=torch.complex64), mask
    )

    assert (((1.0 - mask) * mirror_observed).sum().item()) == 0.0


# ── inherited null-space contracts ───────────────────────────────────────────


def test_observed_support_passes_through_bit_identically() -> None:
    block = _block()
    x = torch.randn(2, CHANNELS, GRID, GRID)
    mask = _mask()

    out = block(x, mask=mask)
    observed = (mask > 0).expand_as(x)

    assert torch.equal(out[observed], x[observed])
    assert not torch.equal(out[~observed], x[~observed])


def test_a_fully_sampled_mask_is_an_exact_no_op() -> None:
    block = _block()
    x = torch.randn(2, CHANNELS, GRID, GRID)

    assert torch.equal(block(x, mask=torch.ones(2, 1, GRID, GRID)), x)


def test_mask_is_mandatory() -> None:
    with pytest.raises(ValueError, match="requires `mask`"):
        _block()(torch.randn(2, CHANNELS, GRID, GRID))


@pytest.mark.parametrize("feature_domain", ["kspace", "image"])
def test_output_domain_matches_input_domain(feature_domain: str) -> None:
    block = _block(feature_domain=feature_domain)
    x = torch.randn(2, CHANNELS, GRID, GRID)

    assert block(x, mask=_mask()).shape == x.shape


def test_every_stream_receives_gradient() -> None:
    """Three streams; a dead one would be a facade (pitfall #16)."""
    block = _block().train()
    out = block(torch.randn(2, CHANNELS, GRID, GRID), mask=_mask())
    (out - torch.randn_like(out)).pow(2).mean().backward()

    for name in ("kspace_attn.out_proj", "image_attn.out_proj", "fuse"):
        grad = max(
            0.0 if p.grad is None else p.grad.abs().max().item()
            for p in block.get_submodule(name).parameters()
        )
        assert grad > 0.0, f"{name} receives no gradient"
    assert block.hermitian_gate.grad.abs().max().item() > 0.0


def test_block_is_not_an_identity_at_init() -> None:
    """IdentityAtInitAttention owns identity-at-init; a second one is the #471 saddle."""
    block = _block()
    x = torch.randn(2, CHANNELS, GRID, GRID)

    assert not torch.allclose(block(x, mask=_mask()), x)


def test_the_image_stream_does_not_read_the_kspace_mask() -> None:
    """Planted violation for masking the wrong domain.

    ``observed`` indexes k-space bins. Applied to the ``ifft2c`` view it would
    restrict pixel keys to "pixels whose index matches an acquired k-column",
    which is an arbitrary crop rather than physics -- and it changes with the
    rung for no physical reason. Same features in, two different masks: the
    image branch's own output must not move.
    """
    block = _block()
    torch.manual_seed(1)
    image_view = torch.randn(1, CHANNELS, GRID, GRID)

    captured: list[torch.Tensor] = []
    block.image_attn.register_forward_hook(
        lambda mod, args, out, sink=captured: sink.append(out.detach().clone())
    )

    for stride in (2, 5):
        mask = _mask(batch=1, stride=stride)
        with torch.no_grad():
            block(image_view, mask=mask)

    assert len(captured) == 2
    assert torch.equal(captured[0], captured[1]), (
        "the image branch changed with the k-space mask: it is being masked in "
        "the wrong domain"
    )
