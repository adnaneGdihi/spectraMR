"""Phase-3 tests covering the runtime-gap-closure pass for
``TODO/audit_plan_novel_fmri.md`` (real Mamba kernels, learned EPI,
4-D MaskGenerator, surface readers, Gu-Yau flattening, director
wiring, KD-tree matcher, alias / shim, campaign manifests)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tests.utils.optional_backends import requires_cuda_for_mamba
from tests.utils.repo_scripts import require_repo_file


# --- Real Mamba in BiMamba4DBlock / MRFMambaBlock4D -------------------------


@requires_cuda_for_mamba
def test_bimamba_4d_real_mamba_mode_runs() -> None:
    from spectramr.models.blocks.space_filling_curves import BiMamba4DBlock
    blk = BiMamba4DBlock(channels=4, d_state=8, use_real_mamba=True)
    x = torch.randn(1, 4, 2, 4, 4)
    out = blk(x)
    assert out.shape == x.shape
    assert blk.use_real_mamba is True


def test_bimamba_4d_conv1d_fallback_runs() -> None:
    from spectramr.models.blocks.space_filling_curves import BiMamba4DBlock
    blk = BiMamba4DBlock(channels=4, kernel_size=3, use_real_mamba=False)
    x = torch.randn(1, 4, 2, 4, 4)
    assert blk(x).shape == x.shape


# --- 1-D Mamba fingerprint encoder in MRFTangentScore ----------------------


@requires_cuda_for_mamba
def test_mrf_tangent_score_consumes_sequence_fingerprint() -> None:
    from spectramr.models.generators.mrf_models import MRFTangentScore
    m = MRFTangentScore(
        hidden=16, n_layers=2, fp_dim=8, time_dim=8, use_mamba_fp_encoder=True,
    )
    theta = torch.tensor([[1.0, 0.1, 1.0, 0.0, 1.0]])
    t = torch.tensor([0.5])
    fp_seq = torch.randn(1, 16, 8)  # sequence of length 16, fp_dim=8
    out = m(theta, t, fingerprint=fp_seq)
    assert out.shape == (1, 5)


def test_mrf_tangent_score_falls_back_to_mlp_encoder() -> None:
    from spectramr.models.generators.mrf_models import MRFTangentScore
    m = MRFTangentScore(hidden=16, n_layers=2, fp_dim=8, use_mamba_fp_encoder=False)
    theta = torch.tensor([[1.0, 0.1, 1.0, 0.0, 1.0]])
    t = torch.tensor([0.5])
    out = m(theta, t, fingerprint=torch.randn(1, 8))
    assert out.shape == (1, 5)


# --- Learned EPI forward operator ------------------------------------------


def test_learned_epi_forward_operator_registered_and_runs() -> None:
    import spectramr.models.generators  # noqa
    from spectramr.models.registry import MODEL_REGISTRY
    assert "learned_epi_forward_operator" in MODEL_REGISTRY
    from spectramr.models.generators.fmri_models import LearnedEPIForwardOperator
    m = LearnedEPIForwardOperator(in_channels=1, out_channels=2, base_width=4)
    out = m(torch.randn(2, 1, 8, 8))
    assert out.shape == (2, 2, 8, 8)


# --- MaskGenerator 4-D extension -------------------------------------------


def test_mask_generator_4d_shared_in_plane_mask() -> None:
    from spectramr.infrastructure.physics.sampling import MaskGenerator
    g = MaskGenerator(seed=0)
    mask = g.generate_mask_4d(
        mask_type="equispaced",
        shape_4d=(3, 2, 16, 16),
        acceleration_factor=4.0,
    )
    assert mask.shape == (3, 2, 16, 16)
    # Shared in-plane mask ⇒ every (t, s) slice is identical.
    assert torch.equal(mask[0, 0], mask[1, 1])


def test_mask_generator_4d_per_timepoint_runs() -> None:
    from spectramr.infrastructure.physics.sampling import MaskGenerator
    g = MaskGenerator(seed=0)
    mask = g.generate_mask_4d(
        mask_type="random",
        shape_4d=(4, 2, 16, 16),
        acceleration_factor=3.0,
        per_timepoint=True,
    )
    assert mask.shape == (4, 2, 16, 16)


# --- Surface readers (mocked since real GIFTI/CIFTI not in test env) -------


def test_cortical_surface_dataset_falls_back_when_no_companion(tmp_path: Path) -> None:
    from spectramr.data.datasets.fmri_dataset import CorticalSurfaceDataset
    np.save(tmp_path / "vol.npy", np.random.randn(4, 8, 8, 1).astype("float32"))
    ds = CorticalSurfaceDataset(tmp_path)
    sample = ds[0]
    # No companion file ⇒ no override key.
    assert "_cortex_flatten_grid_override" not in sample


def test_cortical_surface_dataset_loads_npy_companion(tmp_path: Path) -> None:
    from spectramr.data.datasets.fmri_dataset import CorticalSurfaceDataset
    np.save(tmp_path / "vol.npy", np.random.randn(4, 8, 8, 1).astype("float32"))
    grid = np.random.randn(8, 8, 2).astype("float32")
    np.save(tmp_path / "vol_cortex_flatten.npy", grid)
    ds = CorticalSurfaceDataset(tmp_path)
    sample = ds[0]
    assert "_cortex_flatten_grid_override" in sample
    assert sample["_cortex_flatten_grid_override"].shape == (8, 8, 2)


# --- Gu-Yau-style cortical flattening --------------------------------------


def test_conformal_flattening_harmonic_keeps_uv_in_disk() -> None:
    from spectramr.infrastructure.surfaces import ConformalFlattening, CorticalMesh
    v = np.random.randn(50, 3).astype("float32")
    # Simple closed mesh: connect each vertex to its 3 nearest neighbours.
    f = np.array(
        [[i, (i + 1) % 50, (i + 2) % 50] for i in range(50)], dtype="int32"
    )
    mesh = CorticalMesh(vertices=v, faces=f)
    flat = ConformalFlattening(mesh)
    assert flat.uv.shape == (50, 2)
    # All UV points must lie in the unit disk after normalisation.
    radii = np.linalg.norm(flat.uv, axis=-1)
    assert radii.max() <= 1.0 + 1e-5


# --- Metric alias ----------------------------------------------------------


def test_cross_scanner_metric_alias_resolves() -> None:
    from spectramr.core.metrics.registry import MetricsRegistry
    instance = MetricsRegistry.get("cross_scanner_t1_t2_concordance")
    assert instance.name == "cross_scanner_t1t2_concordance"


# --- Director wiring -------------------------------------------------------


def test_sfc_conformal_fmri_keys_wrapper_populates_jacobian() -> None:
    from spectramr.data.builders.sfc_conformal_fmri_keys_wrapper import (
        SFCConformalFMRIKeysWrapper,
    )

    class _Stub:
        def __len__(self):
            return 1
        def __getitem__(self, _):
            return {"image": torch.randn(1, 1, 8, 8)}

    wrapper = SFCConformalFMRIKeysWrapper(
        _Stub(), expose_conformal_jacobian=True, expose_glm_design_matrix=True,
        glm_design_n_timepoints=8,
    )
    sample = wrapper[0]
    assert "conformal_jacobian" in sample
    assert "design_matrix" in sample
    assert sample["design_matrix"].shape == (1, 8)


# --- KD-tree dictless matcher ----------------------------------------------


def test_mrf_dictless_matcher_round_trip() -> None:
    from spectramr.domain.services.mrf_dictless_matching import (
        MRFDictlessMatcher,
    )
    from spectramr.models.generators.mrf_models import ConformalFPEmbedding
    model = ConformalFPEmbedding(fingerprint_length=16, embedding_dim=5, hidden=16, n_blocks=2)
    matcher = MRFDictlessMatcher(model, device="cpu")
    dict_fp = torch.randn(8, 16)
    dict_params = torch.tensor([
        [1.0, 0.1, 1.0, 0.0, 1.0], [2.0, 0.2, 1.0, 0.0, 1.0],
        [0.5, 0.05, 1.0, 0.0, 1.0], [1.5, 0.15, 1.0, 0.0, 1.0],
        [0.8, 0.08, 1.0, 0.0, 1.0], [1.2, 0.12, 1.0, 0.0, 1.0],
        [0.3, 0.03, 1.0, 0.0, 1.0], [2.5, 0.25, 1.0, 0.0, 1.0],
    ])
    matcher.build_index(dict_fp, dict_params)
    out = matcher.query(dict_fp[:3], k=1)
    assert out["indices"].shape == (3, 1)
    assert out["parameters"].shape == (3, 1, 5)
    # Query the dictionary itself ⇒ should match each entry to itself.
    for i in range(3):
        assert out["indices"][i, 0] == i


# --- Campaign manifests parse ----------------------------------------------


def test_campaign_manifests_parse() -> None:
    root = Path(__file__).resolve().parents[3]
    for cohort in ("fmri_2026_cohort", "mrf_2026_cohort"):
        # require_repo_file, not `.is_file()`: a manifest the allowlist never
        # selects is the publication boundary; a manifest that was DELETED is the
        # defect, and stays loud in every tree that is not the export.
        path = require_repo_file(f"experiments/campaigns/{cohort}.yaml")
        parsed = yaml.safe_load(path.read_text())
        assert parsed["name"] == cohort
        assert len(parsed["experiments"]) == 10
        # Every config path referenced must actually exist.
        for entry in parsed["experiments"]:
            assert (root / entry["config"]).is_file()


# --- Reporting block on idea YAMLs -----------------------------------------


def test_reporting_block_lists_new_plotters() -> None:
    yaml_path = require_repo_file(
        "experiments/inprogress/fmri_2026/idea_1_spatiotemporal_adaptive_sfc.yaml"
    )
    cfg = yaml.safe_load(yaml_path.read_text())
    figs = cfg.get("reporting", {}).get("figures") or []
    assert "fig_c1_beltrami_field" in figs
    assert "fig_c3_teichmuller_schedule" in figs
