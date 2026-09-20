"""DiffiT: the timestep enters q, k and v, not a gate on the block output.

That distinction is the whole reason this backbone exists beside
:mod:`dit_backbone`, so it is what these tests pin. A TMSA that reduced to
ordinary self-attention would still train, still denoise, and be a different
model than the arm claims to run (pitfall 16).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.diffit_backbone import (
    DiffiTBackbone,
    TimeDependentAttention,
)

SIZE, CHANNELS = 32, 8


def _net(domain: str = "kspace") -> DiffiTBackbone:
    torch.manual_seed(0)
    return DiffiTBackbone(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        image_size=SIZE,
        dim=64,
        depth=2,
        heads=4,
        patch_size=4,
        feature_domain=domain,
    )


@pytest.mark.parametrize("domain", ["kspace", "image"])
def test_output_matches_the_input_field(domain: str) -> None:
    out = _net(domain)(torch.randn(2, CHANNELS, SIZE, SIZE), timesteps=torch.zeros(2))
    assert out.shape == (2, CHANNELS, SIZE, SIZE)
    assert torch.isfinite(out).all()


def test_the_declared_domain_changes_the_output() -> None:
    x = torch.randn(2, CHANNELS, SIZE, SIZE)
    t = torch.full((2,), 7.0)
    k, i = _net("kspace"), _net("image")
    with torch.no_grad():
        for a, b in zip(k.parameters(), i.parameters(), strict=True):
            if a.dim() > 1:
                torch.nn.init.normal_(a, std=0.05)
            b.copy_(a)
    assert not torch.allclose(k(x, timesteps=t), i(x, timesteps=t), atol=1e-6)


def test_the_time_projection_starts_at_zero() -> None:
    """Untrained, TMSA must BE ordinary self-attention; a random time projection
    perturbs every attention map before the spatial weights have learned."""
    attn = TimeDependentAttention(64, 4)
    assert torch.equal(attn.time_qkv.weight, torch.zeros_like(attn.time_qkv.weight))
    assert torch.equal(attn.time_qkv.bias, torch.zeros_like(attn.time_qkv.bias))
    tokens = torch.randn(2, 16, 64)
    quiet = attn(tokens, torch.zeros(2, 64))
    loud = attn(tokens, torch.randn(2, 64) * 100)
    assert torch.allclose(quiet, loud, atol=1e-6)


def test_the_time_signal_reaches_qkv_once_trained() -> None:
    """The planted facade: a block that dropped ``time_qkv`` would tie here."""
    attn = TimeDependentAttention(64, 4)
    with torch.no_grad():
        torch.nn.init.normal_(attn.time_qkv.weight, std=0.1)
    tokens = torch.randn(2, 16, 64)
    a = attn(tokens, torch.zeros(2, 64))
    b = attn(tokens, torch.randn(2, 64))
    assert not torch.allclose(a, b, atol=1e-6)


def test_the_block_carries_no_adaln_gate() -> None:
    """Adding one back would reintroduce exactly the mechanism DiT owns, and
    stop this arm from isolating the conditioning pathway."""
    net = _net()
    assert not any(hasattr(block, "modulation") for block in net.blocks), (
        "DiffiT blocks must condition through TMSA alone"
    )


def test_the_output_depends_on_the_timestep() -> None:
    net = _net()
    for p in net.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(1, CHANNELS, SIZE, SIZE)
    assert not torch.allclose(
        net(x, timesteps=torch.tensor([0.0])), net(x, timesteps=torch.tensor([28.0])), atol=1e-8
    )


def test_an_indivisible_head_count_raises() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        TimeDependentAttention(64, heads=7)


def test_checkpointing_changes_nothing_but_memory() -> None:
    net = _net().train()
    x, t = torch.randn(2, CHANNELS, SIZE, SIZE), torch.zeros(2)
    net.set_grad_checkpointing(False)
    plain = net(x, timesteps=t)
    net.set_grad_checkpointing(True)
    assert torch.allclose(plain, net(x, timesteps=t), atol=1e-5)


def test_a_checkpointed_backward_reaches_the_parameters() -> None:
    net = _net().train()
    net.set_grad_checkpointing(True)
    for p in net.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.05)
    net(torch.randn(2, CHANNELS, SIZE, SIZE), timesteps=torch.zeros(2)).square().mean().backward()
    assert [p for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0]


class TestContrastConditioning:
    """Same reconciliation contract dit_backbone pins (finding 27), read from
    the shared owner in ``diffusion_transformer_stem``.
    """

    def _cond_at_block0(self, net, x, timesteps, contrast_emb):
        captured = {}

        def hook(_module, args):
            captured["cond"] = args[1].detach().clone()

        handle = net.blocks[0].register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                net(x, timesteps=timesteps, contrast_emb=contrast_emb)
        finally:
            handle.remove()
        return captured["cond"]

    def test_matching_width_still_adds_directly(self) -> None:
        torch.manual_seed(0)
        net = DiffiTBackbone(
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            image_size=SIZE,
            dim=64,
            depth=2,
            heads=4,
            patch_size=4,
            feature_domain="kspace",
            contrast_emb_dim=64,
        )
        assert net.contrast_proj is None
        x, t = torch.randn(2, CHANNELS, SIZE, SIZE), torch.tensor([5.0, 5.0])
        a = self._cond_at_block0(net, x, t, torch.zeros(2, 64))
        b = self._cond_at_block0(net, x, t, torch.full((2, 64), 3.0))
        assert not torch.equal(a, b)

    def test_mismatched_width_projects_instead_of_dropping(self) -> None:
        torch.manual_seed(0)
        net = DiffiTBackbone(
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            image_size=SIZE,
            dim=64,
            depth=2,
            heads=4,
            patch_size=4,
            feature_domain="kspace",
            contrast_emb_dim=256,
        )
        assert net.contrast_proj is not None
        assert net.contrast_proj.in_features == 256
        x, t = torch.randn(2, CHANNELS, SIZE, SIZE), torch.tensor([5.0, 5.0])
        a = self._cond_at_block0(net, x, t, torch.zeros(2, 256))
        b = self._cond_at_block0(net, x, t, torch.full((2, 256), 3.0))
        assert not torch.equal(a, b), (
            "a declared contrast width must actually reach the block, not be "
            "silently dropped by the width-equality guard"
        )

    def test_unreconcilable_width_raises_instead_of_dropping(self) -> None:
        net = _net()  # contrast_proj is None
        x = torch.randn(2, CHANNELS, SIZE, SIZE)
        with pytest.raises(ValueError, match="contrast_emb"):
            net(x, timesteps=torch.tensor([5.0, 5.0]), contrast_emb=torch.zeros(2, 256))


class TestForwardTimeMaxTimesteps:
    """The generator only knows the real diffusion horizon inside its own
    ``forward`` -- after this backbone is already constructed -- so
    construction-time ``max_timesteps`` is structurally ``None`` on the
    production path (dit_backbone's finding 31, read from the shared owner).
    """

    def _hook_time_mlp_input(self, net, x, **kwargs):
        captured = {}

        def hook(_module, inputs):
            captured["t"] = inputs[0].detach().clone()

        handle = net.time.mlp.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                net(x, **kwargs)
        finally:
            handle.remove()
        return captured["t"]

    def test_construction_time_none_is_overridden_by_forward_time(self) -> None:
        net = _net()  # max_timesteps=None at construction
        x = torch.randn(2, CHANNELS, SIZE, SIZE)
        got = self._hook_time_mlp_input(
            net, x, timesteps=torch.tensor([1.0, 1.0]), max_timesteps=29.0
        )
        unscaled = self._hook_time_mlp_input(net, x, timesteps=torch.tensor([1.0, 1.0]))
        assert not torch.equal(got, unscaled), (
            "forward-time max_timesteps must actually change the embedding, "
            "not be silently discarded"
        )

    def test_disagreeing_construction_and_forward_horizons_raise(self) -> None:
        torch.manual_seed(0)
        net = DiffiTBackbone(
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            image_size=SIZE,
            dim=64,
            depth=2,
            heads=4,
            patch_size=4,
            feature_domain="kspace",
            max_timesteps=29.0,
        )
        with pytest.raises(ValueError, match="disagree"):
            net(
                torch.randn(2, CHANNELS, SIZE, SIZE),
                timesteps=torch.tensor([1.0, 1.0]),
                max_timesteps=1000.0,
            )
