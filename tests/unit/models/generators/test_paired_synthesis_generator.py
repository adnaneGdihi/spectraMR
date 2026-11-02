"""Regression tests for the PMPS paired-synthesis defensibility gate (#2396)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.generators.paired_synthesis_generator import (  # noqa: E402
    PairedSynthesisGenerator,
)
from spectramr.models.generators.pmps_corruption_operators import (  # noqa: E402
    HFCorruptionOperator,
    ULFCorruptionOperator,
)

P_ULF = {"te": 10.0, "tr": 300.0, "fa_rad": 1.5708}
P_HF = {"te": 80.0, "tr": 3000.0, "fa_rad": 1.5708}


class _ConstantOperator(torch.nn.Module):
    def noise_sigma(self) -> torch.Tensor:
        return torch.tensor([0.03])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, 1e6)


def test_missing_protocols_raise() -> None:
    with pytest.raises(ValueError, match="p_ulf and p_hf"):
        PairedSynthesisGenerator()(torch.rand(2, 3, 16, 16))


def test_gate_rejects_an_implausible_ulf_image() -> None:
    gen = PairedSynthesisGenerator(
        ulf_corruption=_ConstantOperator(),
        hf_corruption=HFCorruptionOperator(),
        defensibility_threshold=1.0,
    )
    out = gen(torch.rand(4, 3, 32, 32), p_ulf=P_ULF, p_hf=P_HF)
    assert not out["defensibility_flag"].any()


def test_gate_passes_noise_only_corruption() -> None:
    torch.manual_seed(0)
    ulf = ULFCorruptionOperator(initial_motion_severity=0.0, initial_recon_artifact_severity=0.0)
    hf = HFCorruptionOperator(initial_motion_severity=0.0, initial_bias_field_severity=0.0)
    gen = PairedSynthesisGenerator(ulf_corruption=ulf, hf_corruption=hf)
    with torch.no_grad():
        out = gen(torch.rand(4, 3, 32, 32), p_ulf=P_ULF, p_hf=P_HF)
    assert out["defensibility_flag"].all()
    assert (out["residual_ulf"] <= 1.1).all()


def test_operator_without_noise_sigma_raises() -> None:
    gen = PairedSynthesisGenerator(ulf_corruption=torch.nn.Identity())
    with pytest.raises(TypeError, match="noise_sigma"):
        gen(torch.rand(1, 3, 8, 8), p_ulf=P_ULF, p_hf=P_HF)


def test_distinct_protocols_give_distinct_images() -> None:
    out = PairedSynthesisGenerator()(torch.rand(2, 3, 16, 16), p_ulf=P_ULF, p_hf=P_HF)
    assert (out["ulf"] - out["hf"]).abs().max().item() > 0.0


def test_clean_renders_are_returned_for_the_per_pair_residual() -> None:
    gen = PairedSynthesisGenerator(hf_corruption=HFCorruptionOperator())
    with torch.no_grad():
        out = gen(torch.rand(2, 3, 8, 8), p_ulf=P_ULF, p_hf=P_HF)
    assert torch.equal(out["ulf"], out["ulf_clean"])  # no ULF operator: output is the render
    assert out["hf_clean"].shape == out["hf"].shape
    assert not torch.equal(out["hf"], out["hf_clean"])


def test_registry_built_generator_offers_the_optimizer_a_frozen_parameter() -> None:
    params = list(PairedSynthesisGenerator().parameters())
    assert params and not any(p.requires_grad for p in params)
    torch.optim.AdamW(params)  # an empty list would raise here
