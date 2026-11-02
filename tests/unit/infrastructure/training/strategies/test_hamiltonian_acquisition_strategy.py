"""Unit tests for HamiltonianAcquisitionStrategy (B-4 / Bundle P7).

The base ``BaseTrainingStrategy.__init__`` builds heavy infra (AMP, optimizer
stepper, DI services), so we exercise the strategy's *own* logic by constructing
it via ``object.__new__`` and attaching the minimal attributes the methods read.
This isolates the component under test (config coercion, coverage rasterisation,
magnitude reduction, loss assembly + gradient flow) without standing up the
whole training environment.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.config.schemas.training.hamiltonian_acquisition import (
    HamiltonianAcquisitionConfig,
)
from spectramr.infrastructure.training.strategies.hamiltonian_acquisition_strategy import (
    HamiltonianAcquisitionStrategy,
)
from spectramr.models.acquisition.hamiltonian_trajectory_generator import (
    HamiltonianTrajectoryGenerator,
)
from spectramr.models.registry import MODEL_REGISTRY  # noqa: F401  (import side effects)


def _loss_comp(name, weight=1.0, enabled=True):
    return SimpleNamespace(name=name, weight=weight, enabled=enabled)


def _make_config(ham_block=None, image_losses=None):
    """Minimal duck-typed config the strategy reads via attribute access."""
    if image_losses is None:
        image_losses = [_loss_comp("l1", 1.0, True)]
    losses = SimpleNamespace(
        image_losses=image_losses,
        complex_losses=[],
        kspace_losses=[],
    )
    training = SimpleNamespace(
        hamiltonian_acquisition=ham_block
        if ham_block is not None
        else HamiltonianAcquisitionConfig(num_shots=2, samples_per_shot=16),
    )
    # _assert_model_kwargs_consistent reads config.model.model_kwargs and only
    # flags fields PRESENT in it; an empty dict is vacuously consistent with the
    # hamiltonian block (a real config carries the matching arch fields here).
    model = SimpleNamespace(model_kwargs={})
    data = SimpleNamespace(sampling=SimpleNamespace(patch_size=(16, 16, 1)))
    return SimpleNamespace(training=training, losses=losses, model=model, data=data)


def _build_strategy(config, generator):
    """Construct the strategy without the heavy base __init__."""
    strat = object.__new__(HamiltonianAcquisitionStrategy)
    strat.config = config
    strat.device = torch.device("cpu")
    strat.env = SimpleNamespace(generator=generator, config=config)
    # _to_device is provided by BatchPreparationMixin; emulate it simply.
    strat._to_device = lambda t: t.to(strat.device) if torch.is_tensor(t) else t
    strat._setup_strategy_specific_components()
    return strat


@pytest.fixture
def generator():
    torch.manual_seed(0)
    return HamiltonianTrajectoryGenerator(
        spatial_dims=2, potential_arch="mlp", potential_hidden_dim=16,
        potential_num_layers=2, num_shots=4, samples_per_shot=16,
    )


def test_config_coercion_from_dict(generator):
    """A raw dict block is coerced to the typed HamiltonianAcquisitionConfig."""
    cfg = _make_config(
        ham_block={
            "potential_arch": "siren",
            "integrator": "leapfrog",
            "num_shots": 2,
            "samples_per_shot": 8,
        }
    )
    strat = _build_strategy(cfg, generator)
    assert isinstance(strat.ham_cfg, HamiltonianAcquisitionConfig)
    assert strat.ham_cfg.potential_arch == "siren"


def test_missing_block_raises(generator):
    cfg = _make_config()
    cfg.training.hamiltonian_acquisition = None
    with pytest.raises(ValueError, match="Missing"):
        _build_strategy(cfg, generator)


def test_no_recon_losses_raises(generator):
    cfg = _make_config(image_losses=[])
    with pytest.raises(ValueError, match="No enabled reconstruction losses"):
        _build_strategy(cfg, generator)


def test_image_magnitude_reduction():
    """5D / complex / multi-channel inputs reduce to [B,1,H,W] real."""
    # complex 4D
    x = torch.randn(2, 1, 8, 8, dtype=torch.complex64)
    mag = HamiltonianAcquisitionStrategy._to_image_magnitude(x)
    assert mag.shape == (2, 1, 8, 8) and not torch.is_complex(mag)
    # real-stacked 2-channel
    x2 = torch.randn(2, 2, 8, 8)
    assert HamiltonianAcquisitionStrategy._to_image_magnitude(x2).shape == (2, 1, 8, 8)
    # 5D volume
    x3 = torch.randn(2, 1, 8, 8, 3)
    assert HamiltonianAcquisitionStrategy._to_image_magnitude(x3).shape == (2, 1, 8, 8)


def test_coverage_weight_bounds_and_shape(generator):
    cfg = _make_config()
    strat = _build_strategy(cfg, generator)
    k_traj = torch.rand(4, 16, 2) - 0.5  # in [-0.5, 0.5]
    w = strat._coverage_weight(k_traj, 16, 16)
    assert w.shape == (1, 1, 16, 16)
    assert (w >= 0).all() and (w <= 1.0 + 1e-6).all()


def test_compute_losses_runs_and_backprops(generator):
    """Full loss path produces a finite g_total_loss that backprops to V_theta."""
    cfg = _make_config(image_losses=[_loss_comp("l1", 1.0, True)])
    strat = _build_strategy(cfg, generator)
    target = torch.rand(2, 1, 16, 16)
    losses = strat._compute_losses_impl(target, target, epoch=0)
    assert "g_total_loss" in losses
    assert "loss_hamiltonian_energy_conservation" in losses
    assert "loss_gradient_hardware_compliance" in losses
    total = losses["g_total_loss"]
    assert torch.isfinite(total).all()
    total.backward()
    grads = [p.grad for p in generator.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_compute_losses_no_nan(generator):
    cfg = _make_config()
    strat = _build_strategy(cfg, generator)
    target = torch.rand(1, 1, 16, 16)
    losses = strat._compute_losses_impl(target, target, epoch=0)
    for k, v in losses.items():
        if torch.is_tensor(v):
            assert torch.isfinite(v).all(), f"{k} is not finite"


def test_validation_step_reports_psnr_and_coverage(generator):
    cfg = _make_config()
    strat = _build_strategy(cfg, generator)
    target = torch.rand(1, 1, 16, 16)
    result = strat.validation_step(target, target)
    assert "val_psnr" in result and "val_kspace_coverage" in result
    assert all(isinstance(v, float) for v in result.values())


def test_deterministic_losses(generator):
    """Same generator state + target => identical total loss."""
    cfg = _make_config()
    strat = _build_strategy(cfg, generator)
    torch.manual_seed(7)
    target = torch.rand(1, 1, 16, 16)
    a = strat._compute_losses_impl(target, target, epoch=0)["g_total_loss"]
    b = strat._compute_losses_impl(target, target, epoch=0)["g_total_loss"]
    assert torch.allclose(a, b)


def test_hardware_loss_takes_matrix_size_from_the_patch_grid(generator):
    """#2276: the loss converts cycles/pixel with N read from data.sampling.patch_size."""
    cfg = _make_config()
    cfg.data.sampling.patch_size = (256, 256, 1)
    strat = _build_strategy(cfg, generator)
    assert strat.loss_hardware.matrix_size == 256


@pytest.mark.parametrize("patch", [None, (256, 192, 1), (0, 0, 1)])
def test_non_square_or_missing_grid_raises(generator, patch):
    cfg = _make_config()
    cfg.data.sampling.patch_size = patch
    with pytest.raises(ValueError, match="matrix size N"):
        _build_strategy(cfg, generator)
