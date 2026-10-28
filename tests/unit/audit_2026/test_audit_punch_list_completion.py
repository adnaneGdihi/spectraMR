"""Tests covering the SFC + fMRI/MRF punch-list closure.

Verifies the components added in the final closure pass — 8 audit
validators, 1 new strategy (Teichmüller cold-diffusion), 4 reporting
plotters, 4 data transforms, 2 fMRI datasets, 4D Mamba blocks, and
the cortical-surface infrastructure.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from tests.utils.optional_backends import requires_cuda_for_mamba

# --- Validator registry -----------------------------------------------------


def test_all_eight_2026_validators_registered() -> None:
    from spectramr.config.schemas.validator_registry import get_validator_registry

    reg = get_validator_registry()
    names = set(reg._validators.keys())
    for n in (
        "adaptive_sfc_paired_with_compatible_model",
        "conformal_dc_requires_jacobian_field",
        "cortical_recon_requires_flatten_grid",
        "teichmuller_schedule_in_cold_diffusion_only",
        "epi_phase_encode_direction_required",
        "mrf_spiral_rotation_metadata_required",
        "mrf_pulse_sar_compliance",
        "mrf_harmonisation_phantom_calibration_required",
    ):
        assert n in names


def test_validator_epi_phase_encode_raises_when_missing() -> None:
    from spectramr.config.schemas.audit_plan_novel_2026_validators import (
        _validate_epi_phase_encode_direction_required,
    )

    cfg = {"training": {"training_mode": "beltrami_epi_distortion"}, "data": {}}
    msgs = _validate_epi_phase_encode_direction_required(cfg)
    assert len(msgs) == 1
    cfg["data"]["phase_encode_axis"] = -2
    assert _validate_epi_phase_encode_direction_required(cfg) == []


def test_validator_mrf_pulse_sar_required() -> None:
    from spectramr.config.schemas.audit_plan_novel_2026_validators import (
        _validate_mrf_pulse_sar_compliance,
    )

    cfg = {"training": {"training_mode": "crlb_mrf_pulse_design"}, "mrf": {}}
    msgs = _validate_mrf_pulse_sar_compliance(cfg)
    assert len(msgs) == 1
    cfg["mrf"] = {
        "sar_max_w_per_kg": 4.0,
        "gradient_slew_rate_t_per_m_per_s": 200.0,
    }
    assert _validate_mrf_pulse_sar_compliance(cfg) == []


# --- Teichmüller cold-diffusion strategy ------------------------------------


def test_teichmuller_cold_diffusion_strategy_registered() -> None:
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    assert "teichmuller_cold_diffusion" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS


def test_teichmuller_cold_diffusion_strategy_smoke() -> None:
    """Smoke: the strategy computes a finite total loss.

    This fixture builds the instance with ``__new__`` (bypassing ``__init__``) and
    hand-patches the state ``_compute_losses_impl`` reads, so it mirrors — by hand
    — ``__init__``'s attribute contract. That coupling is invisible and it rotted:
    ``__init__`` sets BOTH ``lambda_teich`` and ``lambda_degrade`` from
    ``training.teichmuller``, but this fixture only patched the first, so the test
    died with ``AttributeError: no attribute 'lambda_degrade'`` and had been red on
    ``dev`` (issue #203) — invisible because Actions is disabled on this repo.

    Every ``self.<lambda_*>`` the loss reads MUST be patched here. If you add one to
    ``__init__``, add it below or this test breaks in exactly the same way.
    """
    from spectramr.infrastructure.training.strategies.teichmuller_cold_diffusion_strategy import (
        TeichmullerColdDiffusionStrategy,
    )

    class _Generator(nn.Module):
        def forward(self, x, *a, **kw):
            return x

    inst = TeichmullerColdDiffusionStrategy.__new__(TeichmullerColdDiffusionStrategy)
    inst.config = SimpleNamespace(training=SimpleNamespace(seed=0))
    inst.device = torch.device("cpu")
    inst.env = SimpleNamespace(
        models={"generator": _Generator()},
        device=torch.device("cpu"),
        config=inst.config,
    )
    type(inst).generator_model = property(  # type: ignore[assignment]
        lambda s: s.env.models["generator"]
    )
    # Patch in the schedule head and buffers (would normally be done by __init__).
    from spectramr.models.diffusion.teichmuller_schedule import TeichmullerScheduleHead

    inst.n_steps = 4
    inst.schedule_head = TeichmullerScheduleHead(hidden=4, r_max=0.9)
    # What register_owned_module records in __init__ (the call_owned lookup reads it).
    inst._owned_modules = {"schedule_head": inst.schedule_head}
    inst._owned_wrappers = {}
    inst.mu0 = torch.zeros(8, 8, dtype=torch.complex64)
    inst.muT = 0.5 * torch.exp(1j * torch.zeros(8, 8))
    # Both weights that `_compute_losses_impl` reads. `__init__` sets them from
    # `training.teichmuller.{lambda_teichmuller,lambda_degrade}`; omitting either
    # here is an AttributeError at loss time, not a wrong number (issue #203).
    inst.lambda_teich = 0.01
    inst.lambda_degrade = 1.0
    x = torch.randn(1, 1, 8, 8)
    out = inst._compute_losses_impl(x, x, epoch=0)
    assert torch.isfinite(out["g_total_loss"])


# --- Reporting plotters ----------------------------------------------------


def test_four_2026_plotters_registered() -> None:
    from spectramr.infrastructure.reporting.plotters import PLOTTERS

    for n in (
        "fig_c1_beltrami_field",
        "fig_c2_spd_geodesic",
        "fig_c3_teichmuller_schedule",
        "fig_c4_fingerprint_embedding",
    ):
        assert n in PLOTTERS


def test_beltrami_field_plotter_renders(tmp_path: Path) -> None:
    from spectramr.infrastructure.reporting.plotters.sfc_conformal_plots import (
        make_beltrami_field,
    )

    mu = 0.3 * np.exp(1j * np.random.rand(16, 16))
    out = make_beltrami_field(None, tmp_path / "fig_c1.pdf", mu=mu)
    assert out is not None and out.is_file()


def test_teichmuller_schedule_plotter_renders(tmp_path: Path) -> None:
    from spectramr.infrastructure.reporting.plotters.sfc_conformal_plots import (
        make_teichmuller_schedule,
    )

    out = make_teichmuller_schedule(
        None, tmp_path / "fig_c3.pdf", r=np.linspace(0.0, 0.9, 16)
    )
    assert out is not None and out.is_file()


def test_fingerprint_embedding_plotter_renders(tmp_path: Path) -> None:
    from spectramr.infrastructure.reporting.plotters.sfc_conformal_plots import (
        make_fingerprint_embedding,
    )

    embedding = np.random.randn(50, 5)
    labels = np.random.randint(0, 3, size=50)
    out = make_fingerprint_embedding(
        None, tmp_path / "fig_c4.pdf", embedding=embedding, labels=labels
    )
    assert out is not None and out.is_file()


def test_spd_geodesic_plotter_renders(tmp_path: Path) -> None:
    from spectramr.infrastructure.reporting.plotters.sfc_conformal_plots import (
        make_spd_geodesic_trajectory,
    )

    out = make_spd_geodesic_trajectory(
        None, tmp_path / "fig_c2.pdf", distances=np.linspace(1.0, 0.1, 50)
    )
    assert out is not None and out.is_file()


# --- Data transforms -------------------------------------------------------


def test_attach_conformal_jacobian_refuses_to_fabricate() -> None:
    """Renamed from `..._identity_default`, which CERTIFIED the facade.

    It asserted that the attach produced an all-ones Jacobian — and an
    all-ones Jacobian weights the conformal data-consistency projection by 1,
    i.e. it is numerically the plain path. Worse, it satisfied the isinstance
    check that `ConformalDiffusionReconStrategy` raises on precisely to stop
    that ("Fail loud, do NOT warn-and-run-plain-MSE ... the 'conformal'
    paradigm is inert"). A punch-list test was pinning the thing that defeated
    the guard (C13).
    """
    import pytest

    from spectramr.data.transforms.sfc_conformal_fmri_keys import (
        attach_conformal_jacobian,
    )

    batch = {"image": torch.randn(2, 1, 16, 16)}
    with pytest.raises(ValueError, match="no real Jacobian"):
        attach_conformal_jacobian(batch)
    assert "conformal_jacobian" not in batch


def test_attach_cortex_flatten_grid_shape() -> None:
    from spectramr.data.transforms.sfc_conformal_fmri_keys import (
        attach_cortex_flatten_grid,
    )

    batch = {"image": torch.randn(3, 1, 16, 16)}
    attach_cortex_flatten_grid(batch, grid_height=8, grid_width=8)
    assert batch["cortex_flatten_grid"].shape == (3, 8, 8, 2)


def test_attach_glm_design_matrix_shape() -> None:
    from spectramr.data.transforms.sfc_conformal_fmri_keys import attach_glm_design_matrix

    batch = {"image": torch.randn(2, 1, 16, 16)}
    attach_glm_design_matrix(batch, n_timepoints=16)
    assert batch["design_matrix"].shape == (2, 16)


def test_attach_scanner_id_default() -> None:
    from spectramr.data.transforms.sfc_conformal_fmri_keys import attach_scanner_id

    batch = {"image": torch.randn(4, 1, 8, 8)}
    attach_scanner_id(batch, default_id=2)
    assert batch["scanner_id"].tolist() == [2, 2, 2, 2]


def test_attach_scanner_id_from_vendor_list() -> None:
    from spectramr.data.transforms.sfc_conformal_fmri_keys import attach_scanner_id

    batch = {"image": torch.randn(2, 1, 8, 8), "scanner": ["siemens", "ge"]}
    attach_scanner_id(batch)
    assert batch["scanner_id"].shape == (2,)
    assert batch["scanner_id"][0] != batch["scanner_id"][1]


# --- fMRI datasets ---------------------------------------------------------


def test_fmri_volume_dataset_empty_dir(tmp_path: Path) -> None:
    from spectramr.data.datasets.fmri_dataset import FMRIVolumeDataset

    ds = FMRIVolumeDataset(tmp_path)
    assert len(ds) == 0


def test_fmri_volume_dataset_reads_npy(tmp_path: Path) -> None:
    from spectramr.data.datasets.fmri_dataset import FMRIVolumeDataset

    arr = np.random.randn(4, 8, 8, 1).astype("float32")
    np.save(tmp_path / "vol.npy", arr)
    ds = FMRIVolumeDataset(tmp_path)
    assert len(ds) == 1
    sample = ds[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["phase_encode_axis"] == -2


def test_cortical_surface_dataset_loads_paired_grid(tmp_path: Path) -> None:
    from spectramr.data.datasets.fmri_dataset import CorticalSurfaceDataset

    arr = np.random.randn(4, 8, 8, 1).astype("float32")
    np.save(tmp_path / "vol.npy", arr)
    grid = np.random.randn(8, 8, 2).astype("float32")
    np.save(tmp_path / "vol_cortex_flatten.npy", grid)
    ds = CorticalSurfaceDataset(tmp_path)
    sample = ds[0]
    assert "_cortex_flatten_grid_override" in sample


# --- 4D Mamba blocks -------------------------------------------------------


@requires_cuda_for_mamba
def test_bimamba_4d_block_residual_shape() -> None:
    from spectramr.models.blocks.space_filling_curves import BiMamba4DBlock

    blk = BiMamba4DBlock(channels=4, kernel_size=5)
    x = torch.randn(2, 4, 3, 8, 8)
    out = blk(x)
    assert out.shape == x.shape


@requires_cuda_for_mamba
def test_bimamba_4d_block_uses_sfc_indices_when_provided() -> None:
    from spectramr.models.blocks.space_filling_curves import BiMamba4DBlock

    blk = BiMamba4DBlock(channels=4)
    x = torch.randn(2, 4, 2, 4, 4)
    idx = torch.argsort(torch.rand(2, 2 * 4 * 4), dim=-1)
    out = blk(x, sfc_indices=idx)
    assert out.shape == x.shape


@requires_cuda_for_mamba
def test_mrf_mamba_block_applies_rotation_to_first_channels() -> None:
    from spectramr.models.blocks.space_filling_curves import MRFMambaBlock4D

    blk = MRFMambaBlock4D(channels=4)
    x = torch.randn(1, 4, 3, 8, 8)
    angles = torch.tensor([0.0, 0.5, 1.0])
    out = blk(x, rotation_angles=angles)
    assert out.shape == x.shape


# --- Surfaces package ------------------------------------------------------


def test_cortical_mesh_round_trip() -> None:
    from spectramr.infrastructure.surfaces import CorticalMesh

    v = np.random.randn(64, 3).astype("float32")
    f = np.random.randint(0, 64, size=(120, 3)).astype("int32")
    m = CorticalMesh(vertices=v, faces=f, subject_id="sub01")
    assert m.n_vertices == 64
    assert m.n_faces == 120
    assert isinstance(m.hash(), str) and len(m.hash()) == 40


def test_conformal_flattening_produces_disk_grid() -> None:
    from spectramr.infrastructure.surfaces import ConformalFlattening, CorticalMesh

    v = np.random.randn(32, 3).astype("float32")
    f = np.random.randint(0, 32, size=(60, 3)).astype("int32")
    flat = ConformalFlattening(CorticalMesh(v, f))
    assert flat.uv.shape == (32, 2)
    grid = flat.to_disk_grid(grid_size=16)
    assert grid.shape == (16, 16, 2)


def test_surface_mamba_block_forward() -> None:
    from spectramr.infrastructure.surfaces import SurfaceMambaBlock

    blk = SurfaceMambaBlock(channels=4, kernel_size=5)
    x = torch.randn(2, 4, 8, 8)
    out = blk(x)
    assert out.shape == x.shape


# --- CLI subcommand --------------------------------------------------------


def test_design_mrf_sequence_writes_yaml(tmp_path: Path) -> None:
    import argparse
    import yaml
    from spectramr.cli.design_mrf_sequence_cli import design_mrf_sequence_cmd

    target_path = tmp_path / "prior.yaml"
    target_path.write_text(
        "T1_range: [0.5, 2.0]\nT2_range: [0.05, 0.2]\nn_samples: 8\n"
    )
    out_path = tmp_path / "designed.yaml"
    args = argparse.Namespace(
        target_params=target_path,
        output=out_path,
        n_tr=16,
        n_steps=5,
        lr=1e-2,
        lambda_beltrami=1e-3,
        device="cpu",
        json=False,
        quiet=True,
    )
    rc = design_mrf_sequence_cmd(args)
    assert rc == 0
    assert out_path.is_file()
    parsed = yaml.safe_load(out_path.read_text())
    assert "pulse_sequence" in parsed
    assert len(parsed["pulse_sequence"]) == 16
