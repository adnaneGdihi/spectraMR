"""``enable_checkpointing: true`` must reach the backbone, not kill the build.

``KSpaceColdDiffusionGenerator.set_grad_checkpointing`` deliberately RAISES
rather than degrading when its backbone has no hook (non-negotiable 3): a
memory claim that silently did not happen OOMs later with nothing to point at.
The other half of that contract is that every backbone an arm can select
actually implements one -- and on the 2026-09-16 cluster run four arms found
out it did not, failing at ``Training pipeline build failed`` before their first
step: ``experiment_11a_swin_diff_rec``, ``experiment_11b_diff_varnet``,
``experiment_11_kspace_cold_diffusion_varnet`` and ``experiment_11e_nafnet``.

Three more arms declare the knob on the two KAN siblings, which had the same
gap and are covered here before they are next dispatched.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.diff_varnet import DiffVarNet
from spectramr.models.generators.diff_varnet_kan import DiffVarNetKAN
from spectramr.models.generators.nafnet_generator import NAFNetGenerator
from spectramr.models.generators.swin_diff_rec import SwinDiffRec
from spectramr.models.generators.swin_diff_rec_kan import SwinDiffRecKAN

SIZE = 32
CHANNELS = 8


def _swin() -> SwinDiffRec:
    return SwinDiffRec(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        image_size=SIZE,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        swin_depth=2,
        swin_heads=2,
        swin_window_size=4,
    )


def _swin_kan() -> SwinDiffRecKAN:
    return SwinDiffRecKAN(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        image_size=SIZE,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        swin_depth=2,
        swin_heads=2,
        swin_window_size=4,
    )


def _varnet() -> DiffVarNet:
    return DiffVarNet(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        image_size=SIZE,
        num_unrolls=2,
        base_channels=16,
    )


def _varnet_kan() -> DiffVarNetKAN:
    return DiffVarNetKAN(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        image_size=SIZE,
        num_unrolls=2,
        base_channels=16,
    )


def _nafnet() -> NAFNetGenerator:
    return NAFNetGenerator(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        width=16,
        enc_blk_nums=[1, 1],
        middle_blk_num=1,
        dec_blk_nums=[1, 1],
    )


BACKBONES = {
    "swin_diff_rec": _swin,
    "swin_diff_rec_kan": _swin_kan,
    "diff_varnet": _varnet,
    "diff_varnet_kan": _varnet_kan,
    "nafnet": _nafnet,
}


@pytest.mark.parametrize("backbone_type", list(BACKBONES))
def test_backbone_implements_the_hook_the_generator_forwards_to(backbone_type):
    """The planted violation: the ``getattr`` that raised for these five."""
    net = BACKBONES[backbone_type]()
    assert callable(getattr(net, "set_grad_checkpointing", None)), (
        f"backbone_type={backbone_type!r} would raise NotImplementedError at build "
        "time for any arm declaring optimization.gradient.enable_checkpointing"
    )


@pytest.mark.parametrize("backbone_type", list(BACKBONES))
def test_checkpointing_changes_allocation_not_the_answer(backbone_type):
    """Recompute is a memory trade; the forward must return the same tensor."""
    torch.manual_seed(0)
    net = BACKBONES[backbone_type]().train()
    x = torch.randn(1, CHANNELS, SIZE, SIZE)
    t = torch.zeros(1)

    net.set_grad_checkpointing(False)
    plain = net(x, timesteps=t)
    net.set_grad_checkpointing(True)
    checkpointed = net(x, timesteps=t)

    assert torch.allclose(plain, checkpointed, atol=1e-5)


@pytest.mark.parametrize("backbone_type", list(BACKBONES))
def test_checkpointed_forward_still_reaches_every_parameter(backbone_type):
    """A checkpointed segment that dropped its graph would backward into nothing."""
    torch.manual_seed(0)
    net = BACKBONES[backbone_type]().train()
    net.set_grad_checkpointing(True)
    net(torch.randn(1, CHANNELS, SIZE, SIZE), timesteps=torch.zeros(1)).sum().backward()

    touched = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
    assert touched, "no parameter received a gradient"
    assert any(p.grad.abs().sum() > 0 for p in touched)


@pytest.mark.parametrize("backbone_type", list(BACKBONES))
def test_checkpointing_stays_off_under_eval(backbone_type):
    """Validation runs the 28-step reverse sampler; recompute there is pure cost."""
    net = BACKBONES[backbone_type]()
    net.set_grad_checkpointing(True)
    net.eval()
    assert not net._checkpointing_active()
    net.train()
    assert net._checkpointing_active()
    with torch.no_grad():
        assert not net._checkpointing_active()


@pytest.mark.parametrize("backbone_type", list(BACKBONES))
def test_a_checkpointed_backward_reports_no_error_of_its_own(backbone_type, caplog):
    """No backbone may turn the recompute early-stop into a logged failure.

    ``torch.utils.checkpoint`` ends a non-reentrant recompute by raising
    ``_StopRecomputationError`` through the user code, so any ``except
    Exception`` inside a checkpointed block sees it. The DC layer's shape
    diagnostic did, and the 2026-09-17 run's ``experiment_11b_diff_varnet`` and
    ``experiment_11_kspace_cold_diffusion_varnet`` each logged ten [DC LAYER
    CRASH] errors while training to completion.
    """
    torch.manual_seed(0)
    net = BACKBONES[backbone_type]().train()
    net.set_grad_checkpointing(True)
    x = torch.randn(1, CHANNELS, SIZE, SIZE)
    mask = (torch.rand(1, 1, SIZE, SIZE) > 0.5).float()

    with caplog.at_level("ERROR"):
        out = net(x, timesteps=torch.zeros(1), mask=mask, measured_kspace=x.clone())
        out.square().mean().backward()

    assert caplog.records == [], [r.getMessage() for r in caplog.records]
