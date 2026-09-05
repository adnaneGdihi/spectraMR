"""Phase-2 tests covering the closing plan items for
``TODO/audit_plan_novel_fmri.md``:

* MRF Pydantic schema block + TrainingSettings mount
* Multiband-EPI forward operator + 4D Cartesian mask
* MRF acquisition manifold + Beltrami pulse curve
* Pulse-sequence-design use case
* The 10 ablation YAMLs parse + audit-clean
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from tests.utils.corpus import tracked_yamls
from tests.utils.repo_scripts import skip_if_public_export

# --- MRF schema -------------------------------------------------------------


def test_mrf_schema_field_present_on_training_settings() -> None:
    from spectramr.config.settings import TrainingSettings
    assert "mrf" in TrainingSettings.model_fields


def test_mrf_schema_round_trip_yaml() -> None:
    from spectramr.config.schemas.mrf import MRFConfigSchema
    payload = {
        "spiral_rotation_schedule": "golden_angle",
        "conformal_kspace_map": "spiral_disk_v1",
        "n_timepoints": 1000,
        "sar_max_w_per_kg": 4.0,
        "gradient_slew_rate_t_per_m_per_s": 200.0,
        "phantom_calibration_path": "/data/calibration.h5",
    }
    cfg = MRFConfigSchema(**payload)
    assert cfg.sar_max_w_per_kg == 4.0
    assert cfg.spiral_rotation_schedule == "golden_angle"


def test_mrf_schema_rejects_unknown_schedule() -> None:
    from pydantic import ValidationError

    from spectramr.config.schemas.mrf import MRFConfigSchema
    try:
        MRFConfigSchema(spiral_rotation_schedule="cubic")
    except ValidationError:
        return
    raise AssertionError("MRFConfigSchema accepted an invalid schedule")


# --- Multiband EPI simulator -----------------------------------------------


def test_cartesian_mask_4d_respects_acceleration() -> None:
    from spectramr.infrastructure.physics.epi_simulator import cartesian_mask_4d
    mask = cartesian_mask_4d(
        n_time=2, n_slice=2, height=16, width=16, in_plane_R=4, center_lines=4
    )
    assert mask.shape == (2, 2, 16, 16)
    # 4× undersampling + 4 fully-sampled centre lines ⇒ rate close to 1/4 + 1/4
    rate = mask.float().mean().item()
    assert 0.2 < rate < 0.7


def test_simulated_multiband_epi_round_trip_shape() -> None:
    from spectramr.infrastructure.physics.epi_simulator import (
        MultibandEPIConfig,
        adjoint_multiband_epi,
        simulated_multiband_epi,
    )
    cfg = MultibandEPIConfig(multiband_factor=2, caipi_shift=0.25, in_plane_R=2)
    vol = torch.randn(1, 4, 4, 8, 8)
    k = simulated_multiband_epi(vol, cfg=cfg)
    assert k.shape == (1, 1, 4, 2, 8, 8)
    back = adjoint_multiband_epi(k, cfg=cfg)
    assert back.shape == vol.shape


# --- MRF acquisition manifold ----------------------------------------------


def test_beltrami_pulse_curve_stays_close_to_base() -> None:
    from spectramr.infrastructure.physics.acquisition import beltrami_pulse_curve
    base_alpha = torch.full((8,), 0.5)
    base_TR = torch.full((8,), 1.0)
    mu = 0.0 * torch.complex(torch.zeros(8), torch.zeros(8))
    pulse = beltrami_pulse_curve(mu, base_alpha=base_alpha, base_TR=base_TR)
    assert torch.allclose(pulse[:, 0], base_alpha, atol=1e-5)
    assert torch.allclose(pulse[:, 1], base_TR, atol=1e-5)


def test_acquisition_bound_rejects_overflowing_sar() -> None:
    from spectramr.infrastructure.physics.acquisition import (
        AcquisitionBound,
        beltrami_pulse_curve,
    )
    bound = AcquisitionBound(sar_max_w_per_kg=1e-3, gradient_slew_rate_t_per_m_per_s=1e6)
    base_alpha = torch.full((8,), 1.5)
    base_TR = torch.full((8,), 1.0)
    mu = torch.zeros(8, dtype=torch.complex64)
    pulse = beltrami_pulse_curve(mu, base_alpha=base_alpha, base_TR=base_TR)
    # alpha² ≈ 2.25 ⇒ SAR way over 1e-3
    assert bound.passes(pulse) is False


def test_mrf_acquisition_manifold_forward_pass_compliance() -> None:
    from spectramr.infrastructure.physics.acquisition import (
        AcquisitionBound,
        MRFAcquisitionManifold,
    )
    m = MRFAcquisitionManifold(n_TR=8, alpha_init=0.3, TR_init=1.0)
    bound = AcquisitionBound(sar_max_w_per_kg=4.0, gradient_slew_rate_t_per_m_per_s=200.0)
    pulse, sar_ok, slew_ok = m(bound)
    assert pulse.shape == (8, 2)
    assert sar_ok is True
    assert slew_ok is True


# --- All 10 ablation YAMLs --------------------------------------------------


_BLOCKING_AUDIT_CATEGORIES = (
    "[schema]",
    "[strategy_registry]",
    "[paradigm_required_fields]",
    "[loss_domain_block_match]",
)


def _audit(path: Path) -> str:
    import subprocess
    res = subprocess.run(
        ["python", "-m", "spectramr.cli", "audit", str(path)],
        capture_output=True, text=True, check=False,
    )
    combined = res.stdout + res.stderr
    for line in combined.splitlines():
        if any(cat in line for cat in _BLOCKING_AUDIT_CATEGORIES) and "passed" not in line.lower():
            # Only treat real failures (❌) as blocking, not the ✅ passes.
            if "❌" in line or "FAIL" in line or "failed" in line.lower():
                return line
    return ""


def test_all_ten_ablation_yamls_audit_clean() -> None:
    skip_if_public_export("experiments/ does not ship; the fmri/mrf ablations are absent")
    # The fmri_2026 cohort has 5 ablation YAMLs (idea_1..idea_5); the mrf_2026
    # cohort has 6 — idea_2 was split into both a "extra_wide_diffusion" and a
    # "high_sigma_diffusion" arm in commit 4886995bc so both extremes of the
    # diffusion-schedule axis get coverage. Total: 11.
    root = Path(__file__).resolve().parents[3]
    ablation_files = sorted(
        list(tracked_yamls((root / "experiments/inprogress/fmri_2026/ablations"), recursive=False))
        + list(tracked_yamls((root / "experiments/inprogress/mrf_2026/ablations"), recursive=False))
    )
    assert len(ablation_files) == 11, (
        f"Expected 11 ablation YAMLs (5 fmri + 6 mrf), found {len(ablation_files)}: "
        f"{[p.name for p in ablation_files]}"
    )
    for path in ablation_files:
        err = _audit(path)
        assert not err, f"YAML {path.name} surfaced blocking audit error: {err}"
