"""Energy-stability contracts for the exp_11 shootout attention blocks.

The exp_11 energy probe (PR #398; cluster run 2026-07-20) measured
``KernelizedAttention`` at ``worst_rho ~ 3.6e3``: the block computed the
linear-attention numerator ``phi(Q) (phi(K)^T V)`` with no ``D^{-1}``
normalizer, so its gain scaled with sequence length (issue #405). These tests
pin the FAVOR+ contract (positive features, softmax-kernel estimate,
row-stochastic attention, seq-len-invariant gain) and the identity-at-init
contract for the residual family (zero-init output projections).
"""

import pytest
import torch
from torch import nn

from spectramr.models.blocks.attention import (
    KernelizedAttention,
    LinearAttention,
    WindowAttention,
)


def _gain(module: torch.nn.Module, x: torch.Tensor) -> float:
    with torch.no_grad():
        y = module(x)
    return (y.norm() / x.norm()).item()


class TestKernelizedFavorPlus:
    """FAVOR+ numerics (Choromanski et al., 2021) on the internal seam."""

    def test_features_strictly_positive(self):
        torch.manual_seed(0)
        attn = KernelizedAttention(32, num_heads=4, num_features=64)
        x = torch.randn(2, 4, 50, 8)  # [B, heads, L, head_dim]
        feat = attn._favor_plus_features(x)
        assert (feat > 0).all(), "FAVOR+ features must be strictly positive"

    def test_feature_dot_estimates_softmax_kernel(self):
        # E[phi(q) . phi(k)] = exp(q . k / sqrt(d)) -- the softmax kernel.
        torch.manual_seed(0)
        d, m = 8, 4096
        attn = KernelizedAttention(d, num_heads=1, num_features=m)
        q = 0.5 * torch.randn(1, 1, 16, d)
        k = 0.5 * torch.randn(1, 1, 16, d)
        est = torch.einsum(
            "bhlf,bhmf->bhlm",
            attn._favor_plus_features(q),
            attn._favor_plus_features(k),
        )
        truth = torch.exp(torch.einsum("bhld,bhmd->bhlm", q, k) * d**-0.5)
        torch.testing.assert_close(est, truth, rtol=0.25, atol=0.05)

    def test_row_stochastic_maps_constant_v_to_itself(self):
        # With D^{-1} normalization the implied attention matrix is
        # row-stochastic, so a constant value field is a fixed point --
        # regardless of Q, K, and sequence length.
        torch.manual_seed(0)
        attn = KernelizedAttention(32, num_heads=4, num_features=64)
        shape = (2, 4, 50, 8)  # [batch, heads, seq_len, head_dim]
        q, k = torch.randn(shape), torch.randn(shape)
        v = torch.full(shape, 3.0)
        out = attn._favor_attention(q, k, v)
        torch.testing.assert_close(out, v, rtol=1e-4, atol=1e-4)

    def test_gain_does_not_scale_with_sequence_length(self):
        # The #405 defect: without the normalizer the un-averaged sum over
        # L = H*W positions makes the gain grow ~linearly with L (measured
        # worst_rho ~3.6e3 on full-res k-space). 16x the tokens must not
        # mean ~16x the gain.
        torch.manual_seed(0)
        attn = KernelizedAttention(32, num_heads=4, num_features=64).eval()
        with torch.no_grad():
            for p in attn.parameters():
                torch.nn.init.normal_(p, std=0.05)
        g_small = _gain(attn, torch.randn(1, 32, 8, 8))
        g_large = _gain(attn, torch.randn(1, 32, 32, 32))
        assert g_large < 4.0 * g_small, (
            f"gain scales with seq_len: {g_small:.3f} @ 64 tokens vs {g_large:.3f} @ 1024 tokens"
        )

    def test_identity_at_init(self):
        # Residual + zero-init out_proj: the block starts as the identity,
        # rho = 1.0, and learns its gain during training.
        torch.manual_seed(0)
        attn = KernelizedAttention(32, num_heads=4, num_features=64)
        x = torch.randn(2, 32, 16, 16)
        torch.testing.assert_close(attn(x), x)


class TestResidualFamilyInitIdentity:
    """The residual blocks must start as the identity (zero-init proj)."""

    def test_linear_attention_identity_at_init(self):
        torch.manual_seed(0)
        attn = LinearAttention(32)
        x = torch.randn(2, 32, 16, 16)
        torch.testing.assert_close(attn(x), x)

    def test_window_attention_identity_at_init(self):
        torch.manual_seed(0)
        attn = WindowAttention(32)
        x = torch.randn(2, 32, 16, 16)
        torch.testing.assert_close(attn(x), x)

    @pytest.mark.parametrize("cls", [LinearAttention, WindowAttention])
    def test_gradients_reach_attention_path(self, cls):
        # Zero-init must not dead-end the branch: the attention parameters
        # still receive gradient through the residual sum.
        torch.manual_seed(0)
        attn = cls(32)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        attn(x).square().mean().backward()
        proj = attn.proj.weight.grad
        assert proj is not None and proj.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# IdentityAtInitAttention (issue #471)
#
# LinearAttention states the family contract in its own initialiser: "zero output
# projection => forward(x) == x, so the residual family starts at energy gain
# rho = 1.0". Only 3 of the 8 blocks the complex_unet dispatch can build honoured
# it. Measured rho at init over 5 seeds on a 1/f k-space feature map:
# self/kernelized/sparse 1.000, dual_domain ~1.05, channel ~0.65,
# wavelet_freq ~0.54, spatial ~0.96 (as low as 0.23), kan_dual_domain 7.4-9.2.
# The call site is REPLACE, so a non-identity block discards its input and emits
# that -- the shootout was comparing eight different starting points.
# ─────────────────────────────────────────────────────────────────────────────


def _kspace_feature(seed: int = 0, C: int = 32, H: int = 32, W: int = 32) -> torch.Tensor:
    """1/f magnitude, random phase: the dynamic range a k-space feature map has."""
    torch.manual_seed(seed)
    ky = torch.fft.fftshift(torch.fft.fftfreq(H))[:, None]
    kx = torch.fft.fftshift(torch.fft.fftfreq(W))[None, :]
    amp = 1.0 / (ky**2 + kx**2).sqrt().clamp_min(1e-3)
    return (amp * torch.randn(2, C, H, W)).contiguous()


def test_identity_at_init_is_bit_exact() -> None:
    """Not "close to" identity: the output must BE the input at step 0."""
    from spectramr.models.blocks.attention import ChannelAttention, IdentityAtInitAttention

    x = _kspace_feature()
    wrapped = IdentityAtInitAttention(ChannelAttention(x.shape[1])).eval()

    with torch.no_grad():
        out = wrapped(x)

    assert torch.equal(out, x), "gamma=0 must give an exact identity"
    assert (out.norm() / x.norm()).item() == pytest.approx(1.0, abs=0.0)


def test_gamma_one_reproduces_the_raw_block() -> None:
    """No expressivity is given up: gamma interpolates identity -> native block."""
    from spectramr.models.blocks.attention import ChannelAttention, IdentityAtInitAttention

    torch.manual_seed(3)
    inner = ChannelAttention(32)
    wrapped = IdentityAtInitAttention(inner).eval()
    with torch.no_grad():
        wrapped.gamma.fill_(1.0)

    x = _kspace_feature()
    with torch.no_grad():
        torch.testing.assert_close(wrapped(x), inner(x))


def test_gamma_receives_gradient_so_it_leaves_zero() -> None:
    """A zero-init knob that cannot be trained is a dead parameter (#15).

    The loss must not be purely quadratic in the block's own output -- for a
    multiplicative zero scale that gives exactly 0 and says nothing.
    """
    from spectramr.models.blocks.attention import ChannelAttention, IdentityAtInitAttention

    torch.manual_seed(0)
    wrapped = IdentityAtInitAttention(ChannelAttention(16))
    x = torch.randn(1, 16, 8, 8)
    target = torch.randn(1, 16, 8, 8)

    ((wrapped(x) - target) ** 2).sum().backward()

    assert wrapped.gamma.grad is not None
    assert wrapped.gamma.grad.abs().item() > 0.0


def test_t_emb_is_forwarded_by_signature_not_by_class_tuple() -> None:
    """The old dispatch was an isinstance check against a hardcoded class tuple.

    Wrapping made that check false, which would have silently blinded the KAN
    block's timestep conditioning -- and the same trap waits for any future
    time-conditioned block. Detection is by signature instead.
    """
    from spectramr.models.blocks.attention import ChannelAttention, IdentityAtInitAttention
    from spectramr.models.blocks.dual_domain_attention_kan import (
        KANGatedDualDomainAttention,
    )

    assert IdentityAtInitAttention(ChannelAttention(16)).takes_t_emb is False
    kan = IdentityAtInitAttention(
        KANGatedDualDomainAttention(
            in_channels=16, time_embedding_dim=32, feature_domain="kspace", num_heads=2
        )
    )
    assert kan.takes_t_emb is True

    x = _kspace_feature(C=16, H=16, W=16)
    with torch.no_grad():
        out = kan.eval()(x, torch.randn(x.shape[0], 32))
    assert torch.equal(out, x)


def test_shape_changing_inner_raises_rather_than_broadcasting() -> None:
    """A block that alters the feature shape is a wiring error, not something to
    silently broadcast against the residual (pitfall #9)."""
    from spectramr.models.blocks.attention import IdentityAtInitAttention

    class _Shrinks(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, : x.shape[1] // 2]

    wrapped = IdentityAtInitAttention(_Shrinks())
    with pytest.raises(ValueError, match="changed the feature shape"):
        wrapped(torch.randn(1, 8, 4, 4))


def test_cross_contrast_olmpa_is_identity_at_init() -> None:
    """The bottleneck block ComplexUNet adds residually (``attended + target_feat``).

    Without the zero output scale it emitted 150-190x the target's norm across
    seeds, so the bottleneck carried almost pure attention output at step 0.
    """
    from spectramr.models.blocks.attention import CrossContrastOLMPA

    for seed in range(3):
        torch.manual_seed(seed)
        block = CrossContrastOLMPA(in_channels=32, num_contrasts=1, phase_safe_dim=64).eval()
        target = torch.randn(2, 64, 32)
        with torch.no_grad():
            out = block(target, torch.randn(2, 64, 32))
        assert out.abs().max().item() == 0.0, f"seed {seed}: not identity at init"


def test_cross_contrast_olmpa_out_scale_trains_off_zero() -> None:
    from spectramr.models.blocks.attention import CrossContrastOLMPA

    torch.manual_seed(0)
    block = CrossContrastOLMPA(in_channels=8, num_contrasts=1, phase_safe_dim=8)
    target, ref = torch.randn(1, 16, 8), torch.randn(1, 16, 8)
    # Mirror the ComplexUNet call site: the block's output is added residually.
    ((block(target, ref) + target - torch.randn(1, 16, 8)) ** 2).sum().backward()

    assert block.out_scale.grad is not None
    assert block.out_scale.grad.abs().item() > 0.0


class TestPhaseSafeDualAttentionReductionIsLive:
    """``reduction`` must actually reduce (pitfall #15).

    The block computed ``hidden_dim = max(stacked_channels // reduction,
    stacked_channels)``. Since ``reduction >= 1`` makes the left operand no
    larger than the right, that ``max`` always returned ``stacked_channels`` --
    an advertised knob that provably could not change the module. It reads as
    consumed (it is stored on ``self.reduction`` and appears in the signature),
    which is why it survived: only evaluating the expression exposes it.
    """

    @staticmethod
    def _hidden_dim(block) -> int:
        # The Conv2d in query_proj maps stacked_channels -> hidden_dim.
        return block.query_proj[0].out_channels

    def test_reduction_changes_the_projection_width(self):
        from spectramr.models.blocks.attention import PhaseSafeDualAttention

        wide = PhaseSafeDualAttention(in_channels=8, num_heads=1, reduction=1)
        narrow = PhaseSafeDualAttention(in_channels=8, num_heads=1, reduction=4)
        assert self._hidden_dim(wide) == 16  # in_channels * 2
        assert self._hidden_dim(narrow) == 4
        assert self._hidden_dim(narrow) < self._hidden_dim(wide)

    def test_default_reduction_is_unchanged_so_checkpoints_still_load(self):
        """reduction=1 must give the pre-fix width, or every state_dict breaks."""
        from spectramr.models.blocks.attention import PhaseSafeDualAttention

        block = PhaseSafeDualAttention(in_channels=4, num_heads=1, reduction=1)
        assert self._hidden_dim(block) == 8  # in_channels * 2, as before

    def test_reduction_never_collapses_the_projection_to_zero(self):
        """The floor is 1 channel, not 0 -- Conv2d(out_channels=0) is invalid."""
        from spectramr.models.blocks.attention import PhaseSafeDualAttention

        block = PhaseSafeDualAttention(in_channels=1, num_heads=1, reduction=64)
        assert self._hidden_dim(block) == 1

    def test_a_reduction_below_one_raises_rather_than_degrading(self):
        """Non-negotiable #3: an illegal value raises, it does not fall back."""
        from spectramr.models.blocks.attention import PhaseSafeDualAttention

        with pytest.raises(ValueError, match="reduction must be >= 1"):
            PhaseSafeDualAttention(in_channels=8, num_heads=1, reduction=0)


# ─────────────────────────────────────────────────────────────────────────────
# The saddle the #471 wrapper created (completes #471)
#
# IdentityAtInitAttention and the three blocks' own zero-init output projection
# are two mechanisms for ONE invariant. Composed, y = x + g*(P*a + b) with
# g == P == b == 0, so dy/dg = P*a + b = 0 and dy/dP = g*a^T = 0: every gradient
# into the block is exactly zero and stays there. Measured on the real dispatch
# before the fix: after 50 AdamW steps gamma was still 0.000000 and the block was
# a bit-exact identity, i.e. the self / kernelized / sparse arms were the
# attention_none control. ChannelAttention escapes because inner(x) - x != 0,
# which is why the pre-existing wrapper test -- written against exactly that
# class -- stayed green (non-negotiable 15: a gate is only a gate for the
# violation shape you have watched it fail on).
# ─────────────────────────────────────────────────────────────────────────────

_SADDLE_PRONE = [LinearAttention, KernelizedAttention, WindowAttention]


@pytest.mark.parametrize("cls", _SADDLE_PRONE)
def test_zero_init_output_is_opt_out_not_mandatory(cls) -> None:
    """``zero_init_output=False`` must leave a block that is NOT the identity."""
    torch.manual_seed(0)
    attn = cls(32, zero_init_output=False)
    x = _kspace_feature(C=32, H=16, W=16)

    with torch.no_grad():
        assert not torch.allclose(attn(x), x), "opting out must leave a live block"


@pytest.mark.parametrize("cls", _SADDLE_PRONE)
@pytest.mark.parametrize("zero_init_output", [True, False])
def test_wrapped_block_escapes_zero_only_without_the_second_mechanism(
    cls, zero_init_output: bool
) -> None:
    """Planted violation: ``zero_init_output=True`` under the wrapper is the saddle.

    Two backward passes, not one. At ``gamma == 0`` the inner gradients are zero
    by construction on the FIRST backward whatever the inner block is, so a
    single-backward assertion cannot tell the saddle from a healthy block.
    """
    from spectramr.models.blocks.attention import IdentityAtInitAttention

    torch.manual_seed(0)
    wrapped = IdentityAtInitAttention(cls(32, zero_init_output=zero_init_output))
    opt = torch.optim.AdamW(wrapped.parameters(), lr=1e-2)
    target = torch.randn(2, 32, 16, 16)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        ((wrapped(_kspace_feature(C=32, H=16, W=16)) - target) ** 2).mean().backward()
        opt.step()

    step()
    gamma_grad = wrapped.gamma.grad.abs().item()
    step()
    inner_grad = max(
        0.0 if p.grad is None else p.grad.abs().max().item()
        for p in wrapped.inner.parameters()
    )

    if zero_init_output:
        # The planted violation: this is the state the dispatch used to build.
        assert gamma_grad == 0.0 and inner_grad == 0.0
        assert wrapped.gamma.item() == 0.0
    else:
        assert gamma_grad > 0.0, "gamma must receive gradient on the first backward"
        assert inner_grad > 0.0, "inner parameters must train once gamma leaves zero"


# ─────────────────────────────────────────────────────────────────────────────
# The FAVOR+ estimate has to be usable on k-space, not merely unbiased (#405)
#
# #405 restored the D^-1 normalizer, which fixed the gain. It did not make the
# estimate track softmax attention: k-space dynamic range is SPATIAL, so the DC
# bin drives the logits to ~5e3 and the 256-feature Monte-Carlo estimate of
# exp() is uncorrelated with what it approximates. LinearAttention carries an
# InstanceNorm2d pre-norm and this block carried none -- an asymmetry inside a
# family the shootout reads as differing only in kernel.
# ─────────────────────────────────────────────────────────────────────────────


def _favor_relative_error(attn: KernelizedAttention, x: torch.Tensor, *, prenorm: bool) -> float:
    """||FAVOR+(q,k,v) - softmax(q,k,v)|| / ||softmax(q,k,v)|| on the block's own seam."""
    import math

    b, c, h, w = x.shape
    seq = h * w
    flat = x.view(b, c, seq).transpose(1, 2)
    flat = attn.norm(flat) if prenorm else flat
    nh, hd = attn.num_heads, attn.head_dim
    q, k, v = (
        proj(flat).view(b, seq, nh, hd).transpose(1, 2)
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj)
    )
    with torch.no_grad():
        approx = attn._favor_attention(q, k, v)
        exact = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(hd), dim=-1) @ v
    return ((approx - exact).norm() / exact.norm()).item()


def test_favor_plus_tracks_softmax_attention_on_kspace_features() -> None:
    """Planted violation included: bypassing the pre-norm must blow the same bound."""
    for seed in range(4):
        attn = KernelizedAttention(64, num_heads=8, num_features=256, feature_seed=seed).eval()
        x = _kspace_feature(seed=seed, C=64, H=64, W=64)

        assert _favor_relative_error(attn, x, prenorm=True) < 0.25
        # Without the per-token norm the estimate is uncorrelated with its target.
        assert _favor_relative_error(attn, x, prenorm=False) > 0.5


def test_random_features_are_orthogonal_within_each_block() -> None:
    """ORF is the variance half of the fix; plain ``randn`` would pass nothing here."""
    from spectramr.models.blocks.attention import orthogonal_random_features

    w = orthogonal_random_features(2, 8, 8, torch.Generator().manual_seed(0))
    directions = torch.nn.functional.normalize(w, dim=-1)
    gram = directions @ directions.transpose(-2, -1)
    off_diagonal = gram - torch.eye(8).expand_as(gram)
    assert off_diagonal.abs().max().item() < 1e-5


def test_feature_draw_is_seeded_locally_not_from_global_rng() -> None:
    """Every rank must build identical features by construction, not by broadcast."""
    first = KernelizedAttention(64, feature_seed=7)
    torch.manual_seed(999)
    second = KernelizedAttention(64, feature_seed=7)
    assert torch.equal(first.rand_features, second.rand_features)
    assert not torch.equal(
        first.rand_features, KernelizedAttention(64, feature_seed=8).rand_features
    )


def test_extras_are_routed_by_name_so_a_mask_block_is_not_handed_t_emb() -> None:
    """The planted violation for the arity -> name change.

    Positional arity says "this block takes a second argument" and stops there,
    so ``forward(x, mask)`` and ``forward(x, t_emb)`` are indistinguishable under
    it and the wrapper would pass whichever it happened to hold. Both stubs below
    have arity 2; only name-based detection routes them differently.
    """
    from spectramr.models.blocks.attention import IdentityAtInitAttention

    received: dict[str, torch.Tensor | None] = {}

    class _WantsMask(nn.Module):
        def forward(self, x, mask=None):
            received["mask"] = mask
            return x * 2.0

    class _WantsTEmb(nn.Module):
        def forward(self, x, t_emb=None):
            received["t_emb"] = t_emb
            return x * 2.0

    x = torch.randn(1, 4, 4, 4)
    t_emb, mask = torch.full((1, 8), 7.0), torch.ones(1, 1, 4, 4)

    mask_block = IdentityAtInitAttention(_WantsMask())
    assert (mask_block.takes_mask, mask_block.takes_t_emb) == (True, False)
    mask_block(x, t_emb, mask=mask)
    assert torch.equal(received["mask"], mask), "the mask block was handed t_emb"

    t_block = IdentityAtInitAttention(_WantsTEmb())
    assert (t_block.takes_mask, t_block.takes_t_emb) == (False, True)
    t_block(x, t_emb, mask=mask)
    assert torch.equal(received["t_emb"], t_emb)


# ---------------------------------------------------------------------------
# ChannelAttention accepts the feature maps the encoder actually hands it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda b: b, id="contiguous"),
        pytest.param(lambda b: b.permute(0, 1, 3, 2), id="permuted"),
        pytest.param(lambda b: b[:, :, ::2, ::2], id="strided-spatial"),
        pytest.param(lambda b: b[1:3], id="batch-slice"),
    ],
)
def test_channel_attention_handles_non_contiguous_feature_maps(make) -> None:
    """``view`` refused any of these outright; the pooling needs a copy, not a view."""
    from spectramr.models.blocks.attention import ChannelAttention

    base = torch.randn(4, 32, 16, 16)
    x = make(base)
    out = ChannelAttention(32)(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
