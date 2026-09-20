"""The authors' own training step is what runs, observed rather than read.

Every assertion here spies on an object inside the vendored upstream. A test that only
checked ``training_loss`` returns a tensor would pass against a reimplementation, which
is the exact failure #2080 records -- three papers' names on one repository's forward
process.

``monkeypatch.chdir`` is not optional: FDB's ``q_sample`` writes ``w.npy`` into the
working directory on every call, so an unchdir'd run drops it in the repository root.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from spectramr.models.baselines._base import UpstreamLossFamily
from spectramr.models.baselines.cdiffmr import CDiffMRBaseline
from spectramr.models.baselines.fdb import FDBBaseline
from spectramr.models.baselines.shen2024 import Shen2024Baseline

_SIZE = 64
_STEPS = 8


def _build(name: str):
    """Small builds of each adapter — the upstreams are 34M-164M at real resolution."""
    return {
        "cdiffmr": lambda: CDiffMRBaseline(resolution=_SIZE, num_steps=_STEPS),
        "fdb": lambda: FDBBaseline(image_size=_SIZE, bridge_steps=_STEPS, undersampling_rate=2),
        "shen2024": lambda: Shen2024Baseline(image_size=_SIZE, timesteps=_STEPS, acceleration=4),
    }[name]()


@pytest.fixture(autouse=True)
def _isolate_cwd(monkeypatch, tmp_path):
    """FDB's calibration array lands here, not in the repository root."""
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("name", ["cdiffmr", "fdb", "shen2024"])
def test_the_training_step_is_differentiable_and_finite(name: str) -> None:
    """A loss with no ``grad_fn`` would train nothing while reporting a number."""
    adapter = _build(name)
    loss = adapter.training_loss(torch.randn(1, 2, _SIZE, _SIZE))
    assert loss.ndim == 0, f"{name}: expected a scalar, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss), f"{name}: loss is not finite"
    assert loss.grad_fn is not None, f"{name}: loss carries no graph"


def test_cdiffmr_degradation_is_upstreams_own_ladder() -> None:
    """The mask stack must be upstream's ``LogSamplingRate``, not a repo accelerator.

    Checked on the object the adapter built: the sampling rate has to fall
    monotonically from fully sampled, which is what ``logspace(-2, 0)[::-1]`` gives and
    what a power-law Cartesian ladder from this repository does not.
    """
    adapter = CDiffMRBaseline(resolution=_SIZE, num_steps=_STEPS)
    adapter.training_loss(torch.randn(1, 2, _SIZE, _SIZE))
    process = adapter._process()

    assert process.ksu_routine == "LogSamplingRate"
    assert process.ksu_mask_type == "cartesian_random"
    rates = [float(m.float().mean()) for m in process.ksu_masks]
    assert rates[0] == pytest.approx(1.0), f"rung 0 must be fully sampled, got {rates[0]}"
    assert all(a >= b for a, b in itertools.pairwise(rates)), rates


def test_cdiffmr_calls_upstream_q_sample(monkeypatch) -> None:
    """Spy on the upstream degradation and require it to fire."""
    adapter = CDiffMRBaseline(resolution=_SIZE, num_steps=_STEPS)
    process = adapter._process()
    calls: list[tuple] = []
    original = process.q_sample
    monkeypatch.setattr(
        process, "q_sample", lambda *a, **k: (calls.append((a, k)), original(*a, **k))[1]
    )
    adapter.training_loss(torch.randn(1, 2, _SIZE, _SIZE))
    assert calls, "upstream q_sample never ran — the degradation is not the paper's"


def test_fdb_calls_upstream_q_sample(monkeypatch) -> None:
    """Same observation for the bridge, whose q_sample also erodes k-space."""
    adapter = FDBBaseline(image_size=_SIZE, bridge_steps=_STEPS, undersampling_rate=2)
    calls: list[tuple] = []
    original = adapter._diffusion.q_sample
    monkeypatch.setattr(
        adapter._diffusion,
        "q_sample",
        lambda *a, **k: (calls.append((a, k)), original(*a, **k))[1],
    )
    adapter.training_loss(torch.randn(1, 2, _SIZE, _SIZE))
    assert calls, "upstream q_sample never ran — the bridge is not the paper's"


def test_shen_calls_upstream_q_sample(monkeypatch) -> None:
    """Shen's degradation walks the acquisition mask's missing patches."""
    adapter = Shen2024Baseline(image_size=_SIZE, timesteps=_STEPS, acceleration=4)
    process = adapter._process()
    calls: list[tuple] = []
    original = process.q_sample
    monkeypatch.setattr(
        process, "q_sample", lambda *a, **k: (calls.append((a, k)), original(*a, **k))[1]
    )
    adapter.training_loss(torch.randn(1, 2, _SIZE, _SIZE))
    assert calls, "upstream q_sample never ran — the degradation is not the paper's"


def test_fdb_writes_its_calibration_where_it_is_told(tmp_path) -> None:
    """``w.npy`` belongs with the run, not wherever the process happened to start."""
    adapter = FDBBaseline(image_size=_SIZE, bridge_steps=_STEPS, undersampling_rate=2)
    target = tmp_path / "run" / "upstream_calibration"
    adapter.set_calibration_dir(target)
    adapter.training_loss(torch.randn(1, 2, _SIZE, _SIZE))
    assert (target / "w.npy").exists(), (
        "upstream's calibration array did not land in the declared directory; it is "
        "written by `np.save` relative to the CWD, so the chdir is what places it"
    )
    assert not (tmp_path / "w.npy").exists(), "calibration leaked outside the run"


def test_shen_refuses_a_coil_stacked_network() -> None:
    """The paper builds ``Unet(channels=2)`` and loops it over coils."""
    with pytest.raises(ValueError, match="single-coil complex"):
        Shen2024Baseline(in_channels=8, out_channels=8)


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("cdiffmr", UpstreamLossFamily.L1),
        ("fdb", UpstreamLossFamily.L2),
        ("shen2024", UpstreamLossFamily.L1),
    ],
)
def test_the_declared_objective_matches_the_upstream_source(name: str, family) -> None:
    """FDB is MSE on x_0; the other two are L1. A wrong pin here mis-gates an arm."""
    assert family == _build(name).UPSTREAM_LOSS_FAMILY


def test_shen_kspace_layout_round_trips() -> None:
    """Our ``fft2c`` and upstream's ``fastmri.ifft2c`` must agree on centering.

    Shen's ``p_losses`` calls ``fastmri.ifft2c`` on whatever k-space it is handed. If
    this repository's centering or ``norm="ortho"`` convention disagreed with fastMRI's,
    the arm would train on a shifted or rescaled target without raising
    (non-negotiable 2).
    """
    import fastmri

    image = torch.randn(2, 1, _SIZE, _SIZE, 2)
    adapter = Shen2024Baseline(image_size=_SIZE, timesteps=_STEPS)
    x_0 = torch.cat([image[..., 0], image[..., 1]], dim=1)
    kspace = adapter._to_upstream_kspace(x_0)
    recovered = fastmri.ifft2c(kspace)
    assert torch.allclose(recovered, image, atol=1e-4), (
        "fft2c -> fastmri.ifft2c is not the identity; the two FFT conventions differ "
        f"by max {float((recovered - image).abs().max()):.3e}"
    )


def test_upstream_q_sample_is_undefined_at_zero() -> None:
    """Pin the upstream defect the timestep floor exists for, and its exact extent.

    ``q_sample`` binds ``img_t_minus_1`` only inside ``if i == n - int(N/T)``, within a
    loop of ``n = int(t*N/T)`` iterations. At t=0 the loop does not run and the name is
    read unbound. t=1 IS defined -- the guard fires at ``i = 0`` -- which is why the
    floor is 1 and not 2; both halves are asserted so a wider floor cannot be
    introduced by guesswork. If a submodule bump fixes the defect, this goes red and
    the floor in :meth:`FDBBaseline.training_loss` can be removed.
    """
    adapter = FDBBaseline(image_size=_SIZE, bridge_steps=_STEPS, undersampling_rate=2)
    x_0 = torch.randn(1, 2, _SIZE, _SIZE)
    with pytest.raises(UnboundLocalError):
        adapter._diffusion.q_sample(x_0, torch.zeros(1).long())
    adapter._diffusion.q_sample(x_0, torch.ones(1).long())
