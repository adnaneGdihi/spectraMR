"""HAT: full-resolution windows, and the conv branch that is only a correction."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.swin_windows import window_partition, window_reverse
from spectramr.models.generators.hat_backbone import HATBackbone, HybridAttentionBlock

SIZE, CHANNELS = 32, 8


def _net(domain: str = "kspace", **kw) -> HATBackbone:
    torch.manual_seed(0)
    opts = {
        "in_channels": CHANNELS,
        "out_channels": CHANNELS,
        "dim": 32,
        "depth": 2,
        "heads": 4,
        "window_size": 8,
        "feature_domain": domain,
    }
    opts.update(kw)
    return HATBackbone(**opts)


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


def test_the_trunk_never_downsamples() -> None:
    """HAT's argument is full-resolution context; a stride would forfeit it.

    Checked on the features rather than the output, which is shape-preserving
    either way.
    """
    net = _net()
    strides = {m.stride for m in net.modules() if isinstance(m, torch.nn.Conv2d)}
    assert strides == {(1, 1)}


def test_a_window_size_that_does_not_tile_raises() -> None:
    """Padding to fit would wrap k-space across the window edge."""
    with pytest.raises(ValueError, match="does not tile"):
        _net(window_size=7)(torch.randn(1, CHANNELS, SIZE, SIZE))


def test_an_odd_channel_count_raises() -> None:
    with pytest.raises(ValueError, match="even"):
        _net(in_channels=7)


def test_the_conv_branch_is_weighted_below_the_attention_branch() -> None:
    """``cab_weight`` at 1.0 drowns the attention it is meant to correct."""
    assert _net().blocks[0].cab_weight < 0.5


def test_the_cab_weight_reaches_the_arithmetic() -> None:
    """The planted facade: a block that stored the weight and added the branch
    unscaled would tie these two."""
    torch.manual_seed(0)
    block = HybridAttentionBlock(32, 4, 8, cab_weight=0.0)
    torch.manual_seed(0)
    heavy = HybridAttentionBlock(32, 4, 8, cab_weight=1.0)
    x = torch.randn(1, 32, SIZE, SIZE)
    assert not torch.allclose(block(x), heavy(x), atol=1e-6)


def test_window_partition_round_trips_at_this_geometry() -> None:
    """The shared owner's helpers, exercised at the sizes this backbone uses."""
    x = torch.randn(2, SIZE, SIZE, 16)
    restored = window_reverse(window_partition(x, 8), 8, SIZE, SIZE)
    assert torch.allclose(x, restored)


def test_the_untrained_backbone_still_responds_to_its_input() -> None:
    """The Tier-2 probe rejects a measurement-independent forward, and cannot
    tell "untrained" from "ignores its input" -- so the output head is not
    zero-initialised even though DiT's published one is."""
    net = _net()
    a = net(torch.randn(2, CHANNELS, SIZE, SIZE), timesteps=torch.zeros(2))
    b = net(torch.randn(2, CHANNELS, SIZE, SIZE), timesteps=torch.zeros(2))
    assert not torch.allclose(a, b, atol=1e-6)


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
    the shared owner in ``diffusion_transformer_stem`` -- HAT conditions
    through FiLM rather than adaLN, so the site to hook is ``self.film``.
    """

    def _cond_at_film(self, net, x, timesteps, contrast_emb):
        captured = {}

        def hook(_module, args):
            captured["cond"] = args[0].detach().clone()

        handle = net.film.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                net(x, timesteps=timesteps, contrast_emb=contrast_emb)
        finally:
            handle.remove()
        return captured["cond"]

    def test_matching_width_still_adds_directly(self) -> None:
        net = _net(contrast_emb_dim=32)
        assert net.contrast_proj is None
        x, t = torch.randn(2, CHANNELS, SIZE, SIZE), torch.tensor([5.0, 5.0])
        a = self._cond_at_film(net, x, t, torch.zeros(2, 32))
        b = self._cond_at_film(net, x, t, torch.full((2, 32), 3.0))
        assert not torch.equal(a, b)

    def test_mismatched_width_projects_instead_of_dropping(self) -> None:
        net = _net(contrast_emb_dim=256)
        assert net.contrast_proj is not None
        assert net.contrast_proj.in_features == 256
        x, t = torch.randn(2, CHANNELS, SIZE, SIZE), torch.tensor([5.0, 5.0])
        a = self._cond_at_film(net, x, t, torch.zeros(2, 256))
        b = self._cond_at_film(net, x, t, torch.full((2, 256), 3.0))
        assert not torch.equal(a, b), (
            "a declared contrast width must actually reach the FiLM projection, "
            "not be silently dropped by the width-equality guard"
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
        net = _net(max_timesteps=29.0)
        with pytest.raises(ValueError, match="disagree"):
            net(
                torch.randn(2, CHANNELS, SIZE, SIZE),
                timesteps=torch.tensor([1.0, 1.0]),
                max_timesteps=1000.0,
            )
