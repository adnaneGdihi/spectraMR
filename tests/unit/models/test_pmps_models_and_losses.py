"""PMPS Phase 1-4 unit tests — models and losses.

These tests exercise the artefacts added by the PMPS plan that don't
need a full training-loop instantiation:

* :class:`TissueParameterDiffusion` — forward shape + physical-range
  output activation (T2 ≤ T1 enforced).
* :class:`ULFCorruptionOperator` / :class:`HFCorruptionOperator` —
  forward shape preservation, calibratable parameters are leaf
  parameters with gradients.
* :class:`MMDPairedDistributionMatch` — non-negative output, zero on
  identical batches.
* :class:`TissueParameterPhysicsConstraint` — fires on out-of-range
  voxels, zero on feasible voxels.

``SyntheticPairedDataset`` was covered here too until ``17048d39b``
deleted the class as dead (D20 — no instantiator, no factory branch, no
strategy). That commit removed the class's *dedicated* test file but not
this module's copy, so a module-level import of a deleted name survived
and aborted collection for the whole of ``tests/unit`` (#804). Nothing is
lost by dropping it: the deletion is pinned by
``tests/unit/data/datasets/test_datasets_package_surface.py::test_synthetic_paired_dataset_is_gone``
— a guard which, ironically, could never itself run while this import
stood.

End-to-end strategy / paired-synthesis behaviour is covered by the
smoke YAMLs (auto-discovered by scripts/ci/smoke_test_vf_configs.sh).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.generators.paired_synthesis_generator import (  # noqa: E402
    PairedSynthesisGenerator,
)
from spectramr.models.generators.pmps_corruption_operators import (  # noqa: E402
    HFCorruptionOperator,
    ULFCorruptionOperator,
    _hoult_snr_scale,
)
from spectramr.models.generators.tissue_parameter_diffusion import (  # noqa: E402
    TissueParameterDiffusion,
)
from spectramr.models.losses.pmps_losses import (  # noqa: E402
    MMDPairedDistributionMatch,
    PerPairConsistencyResidual,
    TissueParameterPhysicsConstraint,
    TissueParameterPriorSmoothness,
)


class TestTissueParameterDiffusion:
    def test_forward_shape(self) -> None:
        model = TissueParameterDiffusion(in_channels=3, out_channels=3)
        x = torch.randn(2, 3, 64, 64)
        out = model(x, timesteps=torch.zeros(2, dtype=torch.long))
        assert out.shape == (2, 3, 64, 64)

    def test_output_in_normalised_range(self) -> None:
        model = TissueParameterDiffusion(in_channels=3, out_channels=3)
        x = torch.randn(1, 3, 32, 32) * 5.0  # extreme inputs
        out = model(x, timesteps=torch.zeros(1, dtype=torch.long))
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_t2_lte_t1_ordering(self) -> None:
        """Physical-range activation must enforce T2 ≤ T1 voxelwise."""
        model = TissueParameterDiffusion(in_channels=3, out_channels=3)
        x = torch.randn(1, 3, 16, 16)
        out = model(x, timesteps=torch.zeros(1, dtype=torch.long))
        t1 = out[:, 0]
        t2 = out[:, 1]
        # T2 should be ≤ T1 voxelwise (with small numerical slack).
        assert (t2 <= t1 + 1e-6).all(), "T2 ≤ T1 ordering violated."

    def test_b0_conditioning_channels(self) -> None:
        """Backbone must see one extra channel when condition_on_b0=True."""
        model = TissueParameterDiffusion(
            in_channels=3, out_channels=3, condition_on_b0=True
        )
        backbone_in = model.backbone.intro.in_channels  # type: ignore[attr-defined]
        assert backbone_in == 4, (
            f"With condition_on_b0=True backbone should see in_channels+1; "
            f"got {backbone_in}."
        )


class TestCorruptionOperators:
    def test_hoult_scaling(self) -> None:
        # B0^{7/4} relative to 3 T reference.
        assert _hoult_snr_scale(3.0) == pytest.approx(1.0)
        assert _hoult_snr_scale(0.064) == pytest.approx((0.064 / 3.0) ** 1.75)
        # Lower B0 → smaller scale → bigger noise in the operator.
        assert _hoult_snr_scale(0.064) < _hoult_snr_scale(0.3) < _hoult_snr_scale(1.5)

    def test_ulf_operator_forward_preserves_shape(self) -> None:
        op = ULFCorruptionOperator(in_channels=1, out_channels=1, b0_t=0.064)
        x = torch.rand(2, 1, 32, 32) + 0.1
        out = op(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_hf_operator_forward_preserves_shape(self) -> None:
        op = HFCorruptionOperator(in_channels=1, out_channels=1)
        x = torch.rand(2, 1, 32, 32) + 0.1
        out = op(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_ulf_operator_psi_is_trainable(self) -> None:
        op = ULFCorruptionOperator(in_channels=1, out_channels=1)
        trainable = [n for n, p in op.named_parameters() if p.requires_grad]
        assert "motion_severity" in trainable
        assert "recon_artifact_severity" in trainable
        assert "coil_bias" in trainable

    def test_hf_operator_psi_is_trainable(self) -> None:
        op = HFCorruptionOperator(in_channels=1, out_channels=1)
        trainable = [n for n, p in op.named_parameters() if p.requires_grad]
        assert "rician_sigma" in trainable
        assert "bias_field_severity" in trainable


class TestMMDPaired:
    def test_zero_on_identical_batches(self) -> None:
        loss = MMDPairedDistributionMatch(bandwidths=(1.0, 2.0))
        x = torch.randn(4, 1, 16, 16)
        # MMD on (x, x) should be 0 (within float noise).
        v = loss(x, x).item()
        assert v == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self) -> None:
        loss = MMDPairedDistributionMatch(bandwidths=(1.0, 2.0))
        x = torch.randn(4, 1, 16, 16)
        y = torch.randn(4, 1, 16, 16) + 5.0  # shifted distribution
        v = loss(x, y).item()
        assert v >= 0.0


class TestTissueConstraints:
    def test_zero_inside_feasible_region(self) -> None:
        loss = TissueParameterPhysicsConstraint(normalised=True)
        # All-zeros (T2=0, T1=0) → ordering OK, all in [0,1] → penalty 0.
        x = torch.zeros(1, 3, 8, 8)
        assert loss(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_fires_on_negative_values(self) -> None:
        loss = TissueParameterPhysicsConstraint(normalised=True)
        x = -torch.ones(1, 3, 8, 8)  # negative everywhere
        assert loss(x, x).item() > 0.0

    def test_fires_on_t2_greater_than_t1(self) -> None:
        loss = TissueParameterPhysicsConstraint(normalised=True)
        x = torch.zeros(1, 3, 8, 8)
        x[:, 0] = 0.2  # T1
        x[:, 1] = 0.8  # T2 > T1 → ordering violation
        assert loss(x, x).item() > 0.0


class TestSmoothness:
    def test_zero_on_constant_field(self) -> None:
        loss = TissueParameterPriorSmoothness(order=1)
        x = torch.full((1, 3, 8, 8), 0.5)
        assert loss(x, x).item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_on_varied_field(self) -> None:
        loss = TissueParameterPriorSmoothness(order=1)
        x = torch.randn(1, 3, 16, 16)
        assert loss(x, x).item() > 0.0


class TestPerPairConsistencyResidual:
    def test_l1_norm(self) -> None:
        loss = PerPairConsistencyResidual(norm="l1")
        a = torch.zeros(2, 1, 4, 4)
        b = torch.ones(2, 1, 4, 4)
        assert loss(a, b).item() == pytest.approx(1.0)

    def test_l2_norm(self) -> None:
        loss = PerPairConsistencyResidual(norm="l2")
        a = torch.zeros(2, 1, 4, 4)
        b = torch.ones(2, 1, 4, 4) * 2.0
        assert loss(a, b).item() == pytest.approx(4.0)


class TestPairedSynthesisGenerator:
    def test_forward_dict_shape(self) -> None:
        gen = PairedSynthesisGenerator(in_channels=3, out_channels=1)
        theta = torch.rand(2, 3, 16, 16)
        out = gen(
            theta,
            p_ulf={"te": 10.0, "tr": 300.0, "fa_rad": 1.5708},
            p_hf={"te": 80.0, "tr": 3000.0, "fa_rad": 1.5708},
        )
        assert set(out.keys()) >= {"ulf", "hf", "theta", "residual", "defensibility_flag"}
        assert out["ulf"].shape == (2, 1, 16, 16)
        assert out["hf"].shape == (2, 1, 16, 16)
        assert out["defensibility_flag"].dtype == torch.bool
        assert out["defensibility_flag"].shape == (2,)
