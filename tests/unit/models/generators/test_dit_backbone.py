"""DiT: the trunk is the published one, the stem is the k-space adaptation.

What these pin is the pair of claims the module makes -- adaLN-Zero starts as
the identity, and ``feature_domain`` reaches the arithmetic.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.dit_backbone import DiTBackbone, DiTBlock

SIZE, CHANNELS = 32, 8


def _net(domain: str = "kspace") -> DiTBackbone:
    torch.manual_seed(0)
    return DiTBackbone(
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
    """Identical weights and input; only the declaration differs.

    The head is zero-init, so both would tie at zero and pass vacuously -- the
    weights are randomised first, and copied so the two nets stay identical.
    """
    x = torch.randn(2, CHANNELS, SIZE, SIZE)
    t = torch.full((2,), 7.0)
    k, i = _net("kspace"), _net("image")
    with torch.no_grad():
        for a, b in zip(k.parameters(), i.parameters(), strict=True):
            if a.dim() > 1:
                torch.nn.init.normal_(a, std=0.05)
            b.copy_(a)
    assert not torch.allclose(k(x, timesteps=t), i(x, timesteps=t), atol=1e-6)


def test_adaln_zero_makes_every_block_the_identity_at_init() -> None:
    """The paper's stability claim, and the reason the head can be zero-init."""
    block = DiTBlock(64, 4, 4.0)
    tokens = torch.randn(2, 16, 64)
    assert torch.allclose(block(tokens, torch.randn(2, 64)), tokens, atol=1e-6)


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
    x = torch.randn(2, CHANNELS, SIZE, SIZE)
    t = torch.zeros(2)
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
    touched = [p for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert touched


class TestForwardTimeMaxTimesteps:
    """Finding 31: the generator only knows the real horizon inside its own
    ``forward`` (it is set after this backbone is already constructed), so
    construction time -- what ``backbone_builders.py`` supplies -- is
    structurally ``None`` on the production path. Forward-time must still
    scale correctly, and it must not depend on a PRIOR call's kwargs.
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

    def test_does_not_leak_across_calls(self) -> None:
        """A stored override on ``self.time`` would make call N depend on call
        N-1's kwargs -- the same batch-companion defect findings 15/28 fixed
        one level up, here shifted from batch composition to call history."""
        net = _net()
        x = torch.randn(2, CHANNELS, SIZE, SIZE)
        with_horizon = self._hook_time_mlp_input(
            net, x, timesteps=torch.tensor([1.0, 1.0]), max_timesteps=29.0
        )
        after_without = self._hook_time_mlp_input(net, x, timesteps=torch.tensor([1.0, 1.0]))
        again_with_horizon = self._hook_time_mlp_input(
            net, x, timesteps=torch.tensor([1.0, 1.0]), max_timesteps=29.0
        )
        assert torch.equal(with_horizon, again_with_horizon)
        assert not torch.equal(with_horizon, after_without)

    def test_agreeing_construction_and_forward_horizons_are_fine(self) -> None:
        torch.manual_seed(0)
        net = DiTBackbone(
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
        out = net(
            torch.randn(2, CHANNELS, SIZE, SIZE),
            timesteps=torch.tensor([1.0, 1.0]),
            max_timesteps=29.0,
        )
        assert torch.isfinite(out).all()

    def test_disagreeing_construction_and_forward_horizons_raise(self) -> None:
        torch.manual_seed(0)
        net = DiTBackbone(
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


class TestContrastConditioning:
    """Finding 27: contrast conditioning was gated on ``contrast.shape[-1] ==
    cond.shape[-1]``, structurally False whenever the generator's contrast
    embedding (built at ``time_embedding_dim``) does not match this
    backbone's token width ``dim`` -- silently dropping ``num_contrasts: 3``.
    """

    def _cond_at_block0(self, net, x, timesteps, contrast_idx):
        contrast_emb = torch.zeros(x.shape[0], net.contrast_proj.in_features
                                    if net.contrast_proj is not None else net.time.dim)
        contrast_emb[:, 0] = float(contrast_idx)
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
        net = DiTBackbone(
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
        cond0 = self._cond_at_block0(net, x, t, contrast_idx=0)
        cond1 = self._cond_at_block0(net, x, t, contrast_idx=1)
        assert not torch.equal(cond0, cond1)

    def test_mismatched_width_projects_instead_of_dropping(self) -> None:
        torch.manual_seed(0)
        net = DiTBackbone(
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
        cond0 = self._cond_at_block0(net, x, t, contrast_idx=0)
        cond1 = self._cond_at_block0(net, x, t, contrast_idx=1)
        assert not torch.equal(cond0, cond1), (
            "a declared contrast width must actually reach the block, not be "
            "silently dropped by the width-equality guard"
        )

    def test_unreconcilable_width_raises_instead_of_dropping(self) -> None:
        """No ``contrast_emb_dim`` declared: the mismatch cannot be
        reconciled, so it must be loud (pitfall 9), not a silent no-op."""
        net = _net()  # contrast_proj is None
        x = torch.randn(2, CHANNELS, SIZE, SIZE)
        with pytest.raises(ValueError, match="contrast_emb"):
            net(
                x,
                timesteps=torch.tensor([5.0, 5.0]),
                contrast_emb=torch.zeros(2, 256),
            )
