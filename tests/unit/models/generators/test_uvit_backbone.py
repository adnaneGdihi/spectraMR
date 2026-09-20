"""U-ViT: the long skips and the odd-depth pairing they depend on."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.uvit_backbone import UViTBackbone, UViTBlock

SIZE, CHANNELS = 32, 8


def _net(domain: str = "kspace", depth: int = 5) -> UViTBackbone:
    torch.manual_seed(0)
    return UViTBackbone(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        image_size=SIZE,
        dim=64,
        depth=depth,
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
    """Weights copied across so only the domain declaration differs."""
    x = torch.randn(2, CHANNELS, SIZE, SIZE)
    t = torch.full((2,), 7.0)
    k, i = _net("kspace"), _net("image")
    with torch.no_grad():
        for a, b in zip(k.parameters(), i.parameters(), strict=True):
            if a.dim() > 1:
                torch.nn.init.normal_(a, std=0.05)
            b.copy_(a)
    assert not torch.allclose(k(x, timesteps=t), i(x, timesteps=t), atol=1e-6)


@pytest.mark.parametrize("depth", [2, 4, 12])
def test_an_even_depth_raises(depth: int) -> None:
    """Even depth cannot pair the halves around one middle block."""
    with pytest.raises(ValueError, match="ODD depth"):
        UViTBackbone(depth=depth)


def test_a_depth_below_three_raises() -> None:
    with pytest.raises(ValueError, match="ODD depth"):
        UViTBackbone(depth=1)


@pytest.mark.parametrize("depth", [3, 5, 7])
def test_the_halves_pair_one_to_one(depth: int) -> None:
    """Every decoder block consumes exactly one encoder skip."""
    net = _net(depth=depth)
    assert len(net.encoder) == len(net.decoder) == depth // 2
    assert all(b.skip_proj is None for b in net.encoder)
    assert all(b.skip_proj is not None for b in net.decoder)
    assert net.middle.skip_proj is None


def test_a_decoder_block_without_its_skip_raises() -> None:
    """Silently skipping the merge would drop the architecture's whole point."""
    block = UViTBlock(64, 4, 2.0, takes_skip=True)
    with pytest.raises(ValueError, match="no skip tensor"):
        block(torch.randn(1, 8, 64))


def test_the_skip_merge_is_a_projection_not_an_addition() -> None:
    """A Linear over the concatenated pair is what lets it weight the shallow
    signal per channel; an add would fix that weight at one."""
    block = UViTBlock(64, 4, 2.0, takes_skip=True)
    assert block.skip_proj.in_features == 128
    assert block.skip_proj.out_features == 64


def test_the_time_token_is_dropped_before_unpatchify() -> None:
    """Leaving it in would shift every patch by one and scramble the field."""
    net = _net()
    out = net(torch.randn(2, CHANNELS, SIZE, SIZE), timesteps=torch.zeros(2))
    assert out.shape[-2:] == (SIZE, SIZE)


def test_the_output_depends_on_the_timestep() -> None:
    net = _net()
    for p in net.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(1, CHANNELS, SIZE, SIZE)
    assert not torch.allclose(
        net(x, timesteps=torch.tensor([0.0])), net(x, timesteps=torch.tensor([28.0])), atol=1e-8
    )


def test_checkpointing_changes_nothing_but_memory() -> None:
    net = _net().train()
    x, t = torch.randn(2, CHANNELS, SIZE, SIZE), torch.zeros(2)
    net.set_grad_checkpointing(False)
    plain = net(x, timesteps=t)
    net.set_grad_checkpointing(True)
    assert torch.allclose(plain, net(x, timesteps=t), atol=1e-5)


class TestContrastConditioning:
    """Same reconciliation contract dit_backbone pins (finding 27), read from
    the shared owner in ``diffusion_transformer_stem`` -- gating on a bare
    width equality silently drops ``num_contrasts`` on every arm that widens
    the trunk past ``time_embedding_dim``.
    """

    def _cond_at_head(self, net, x, timesteps, contrast_emb):
        captured = {}

        def hook(_module, args):
            captured["cond"] = args[1].detach().clone()

        handle = net.head.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                net(x, timesteps=timesteps, contrast_emb=contrast_emb)
        finally:
            handle.remove()
        return captured["cond"]

    def test_matching_width_still_adds_directly(self) -> None:
        torch.manual_seed(0)
        net = UViTBackbone(
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            image_size=SIZE,
            dim=64,
            depth=5,
            heads=4,
            patch_size=4,
            feature_domain="kspace",
            contrast_emb_dim=64,
        )
        assert net.contrast_proj is None
        x, t = torch.randn(2, CHANNELS, SIZE, SIZE), torch.tensor([5.0, 5.0])
        a = self._cond_at_head(net, x, t, torch.zeros(2, 64))
        b = self._cond_at_head(net, x, t, torch.full((2, 64), 3.0))
        assert not torch.equal(a, b)

    def test_mismatched_width_projects_instead_of_dropping(self) -> None:
        torch.manual_seed(0)
        net = UViTBackbone(
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            image_size=SIZE,
            dim=64,
            depth=5,
            heads=4,
            patch_size=4,
            feature_domain="kspace",
            contrast_emb_dim=256,
        )
        assert net.contrast_proj is not None
        assert net.contrast_proj.in_features == 256
        x, t = torch.randn(2, CHANNELS, SIZE, SIZE), torch.tensor([5.0, 5.0])
        a = self._cond_at_head(net, x, t, torch.zeros(2, 256))
        b = self._cond_at_head(net, x, t, torch.full((2, 256), 3.0))
        assert not torch.equal(a, b), (
            "a declared contrast width must actually reach the head, not be "
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
        net = UViTBackbone(
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            image_size=SIZE,
            dim=64,
            depth=5,
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
