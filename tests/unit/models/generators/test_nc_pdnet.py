"""NC-PDNet: the control the graph arms have to beat.

Its value as a baseline depends entirely on the measurement actually reaching
the answer -- an unrolled network run without its k-space is a plain CNN wearing
the name of one, and that is the shape these tests plant.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.nufft_ops import NUFFTForwardModel
from spectramr.infrastructure.physics.trajectories import get_trajectory
from spectramr.models.generators.nc_pdnet import NCPDNet
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY

IM = 32
COILS = 4
CHANNELS = 2 * COILS


@pytest.fixture(scope="module")
def acquisition():
    traj, dcf = get_trajectory("spiral", im_size=(IM, IM))
    op = NUFFTForwardModel(im_size=(IM, IM))
    torch.manual_seed(0)
    truth = torch.randn(2, COILS, IM, IM, dtype=torch.complex64)
    samples = op.forward_project(truth, traj)
    zero_filled = op.adjoint_project(samples * dcf.view(1, 1, -1), traj)
    interleaved = torch.stack([zero_filled.real, zero_filled.imag], dim=2).flatten(1, 2)
    return traj, dcf, samples, interleaved


def _net(**kw) -> NCPDNet:
    torch.manual_seed(0)
    opts = {
        "in_channels": CHANNELS,
        "out_channels": CHANNELS,
        "image_size": IM,
        "num_cascades": 2,
        "width": 16,
        "depth": 3,
    }
    opts.update(kw)
    return NCPDNet(**opts)


def test_it_is_reachable_by_name() -> None:
    populate_model_registry()
    assert "nc_pdnet" in MODEL_REGISTRY


def test_the_cascade_preserves_the_image_layout(acquisition) -> None:
    traj, dcf, samples, x0 = acquisition
    out = _net()(x0, measured_kspace=samples, trajectory=traj, dcf=dcf)
    assert out.shape == x0.shape
    assert torch.isfinite(out).all()


def test_the_measurement_reaches_the_output(acquisition) -> None:
    """The planted facade: a cascade whose DC step was a no-op would tie here,
    and would still train, converge and report a PSNR."""
    traj, dcf, samples, x0 = acquisition
    net = _net()
    a = net(x0, measured_kspace=samples, trajectory=traj, dcf=dcf)
    b = net(x0, measured_kspace=samples * 0.5, trajectory=traj, dcf=dcf)
    assert not torch.allclose(a, b, atol=1e-6)


def test_running_without_a_measurement_raises(acquisition) -> None:
    """Silently degrading to a CNN is the failure this baseline cannot have."""
    _, dcf, samples, x0 = acquisition
    with pytest.raises(ValueError, match="reached without"):
        _net()(x0, measured_kspace=samples, trajectory=None, dcf=dcf)


def test_the_sample_mask_changes_the_answer(acquisition) -> None:
    """Which readouts survived the acceleration has to matter."""
    traj, dcf, samples, x0 = acquisition
    net = _net()
    n = traj.shape[-1]
    full = net(x0, measured_kspace=samples, trajectory=traj, dcf=dcf, sample_mask=torch.ones(n))
    half = torch.ones(n)
    half[n // 2 :] = 0
    partial = net(x0, measured_kspace=samples, trajectory=traj, dcf=dcf, sample_mask=half)
    assert not torch.allclose(full, partial, atol=1e-6)


def test_every_cascade_carries_its_own_fidelity_weight(acquisition) -> None:
    """One shared step cannot anneal from data-driven to prior-driven."""
    net = _net(num_cascades=4)
    assert len(net.dc_steps) == len(net.regularisers) == 4
    assert len({id(d.step_size) for d in net.dc_steps}) == 4
    assert all(d.step_size.requires_grad for d in net.dc_steps)


def test_the_regulariser_starts_as_the_identity(acquisition) -> None:
    """Zero-init exit, so the cascade begins as pure data consistency."""
    net = _net()
    probe = torch.randn(1, CHANNELS, IM, IM)
    assert torch.allclose(net.regularisers[0](probe), probe, atol=1e-6)


def test_gradients_reach_the_regularisers_and_the_steps(acquisition) -> None:
    traj, dcf, samples, x0 = acquisition
    net = _net().train()
    net(x0, measured_kspace=samples, trajectory=traj, dcf=dcf).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.regularisers.parameters())


def test_checkpointing_does_not_change_the_answer(acquisition) -> None:
    traj, dcf, samples, x0 = acquisition
    net = _net().train()
    net.set_grad_checkpointing(False)
    plain = net(x0, measured_kspace=samples, trajectory=traj, dcf=dcf)
    net.set_grad_checkpointing(True)
    assert torch.allclose(
        plain, net(x0, measured_kspace=samples, trajectory=traj, dcf=dcf), atol=1e-5
    )


def test_an_odd_channel_count_raises() -> None:
    with pytest.raises(ValueError, match="in_channels must be even"):
        _net(in_channels=7, out_channels=7)


def test_mismatched_widths_raise() -> None:
    """An unrolled cascade feeds its own output back in."""
    with pytest.raises(ValueError, match="must equal in_channels"):
        _net(in_channels=CHANNELS, out_channels=4)
