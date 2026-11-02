"""Unit tests for the PMPS Stage-C sampling loop (knobs, protocols, prior chain)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from spectramr.config.schemas.training.pmps import ProtocolSamplingConfig
from spectramr.infrastructure.training.schedulers.diffusion_scheduler import DiffusionScheduler
from spectramr.infrastructure.training.strategies.pmps_synthesis import (
    protocol_pairs,
    require_paired_synthesis_config,
    sample_tissue_prior,
)

pytestmark = [pytest.mark.unit]

ULF = {
    "name": "ulf_t1",
    "TE": 10.0,
    "TR": 300.0,
    "FA": 90.0,
    "B0": 0.064,
    "contrast_type": "gradient_echo",
}
HF = {
    "name": "hf_t1",
    "TE": 10.0,
    "TR": 300.0,
    "FA": 90.0,
    "B0": 3.0,
    "contrast_type": "gradient_echo",
}


def stage_c_extras(tmp_path: Any, **overrides: Any) -> dict[str, Any]:
    extras: dict[str, Any] = {
        "batch_size": 2,
        "n_synthetic_pairs": 3,
        "output_root": str(tmp_path / "pairs"),
        "output_format": "npz",
        "defensibility_threshold": 1.5,
        "write_rejected": True,
        "parallel_shards": 2,
        "protocol_sampling": {"mode": "fixed", "fixed_protocols": [ULF, HF]},
        "tissue_prior": {
            "checkpoint": str(tmp_path / "stage_a"),
            "part": "ema",
            "timesteps": 4,
            "noise_schedule": "cosine",
            "prediction_type": "sample",
        },
        "corruption": {"checkpoint": str(tmp_path / "ulf.pt"), "part": "generator"},
    }
    extras.update(overrides)
    return extras


def test_validates_and_refuses_unknown_keys(tmp_path: Any) -> None:
    cfg = require_paired_synthesis_config(SimpleNamespace(model_extra=stage_c_extras(tmp_path)))
    assert cfg.tissue_prior.prediction_type == "sample"
    with pytest.raises(ValueError, match="does not read"):
        require_paired_synthesis_config(
            SimpleNamespace(model_extra=stage_c_extras(tmp_path, n_synth=1))
        )


def test_fixed_protocols_split_by_field_and_convert_flip_angle() -> None:
    pairs = protocol_pairs(ProtocolSamplingConfig(mode="fixed", fixed_protocols=[ULF, HF]))
    assert [(u, h) for u, _, h, _ in pairs] == [("ulf_t1", "hf_t1")]
    assert pairs[0][1]["fa_rad"] == pytest.approx(torch.pi / 2)


@pytest.mark.parametrize(
    "cfg, match",
    [
        ({"mode": "site_balanced"}, "acquisition corpus"),
        ({"mode": "fixed", "fixed_protocols": [ULF]}, "at least one protocol"),
        (
            {"mode": "fixed", "fixed_protocols": [ULF, {**HF, "contrast_type": "spin_echo"}]},
            "spoiled gradient echo",
        ),
    ],
)
def test_unrenderable_protocols_raise(cfg: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        protocol_pairs(ProtocolSamplingConfig(**cfg))


class ConstantX0Prior(nn.Module):
    """Predicts the same clean tissue map at every step; records the b0 it was given."""

    def __init__(self, value: float = 0.4, striped: bool = False) -> None:
        super().__init__()
        self.value = value
        self.striped = striped
        self.b0: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, b0: torch.Tensor) -> torch.Tensor:
        self.b0 = b0
        out = torch.full_like(x, self.value)
        if self.striped:
            out[..., ::2] = 0.9
        return out


def test_sample_prediction_chain_ends_at_the_predicted_x0() -> None:
    torch.manual_seed(0)
    prior = ConstantX0Prior()
    scheduler = DiffusionScheduler(num_timesteps=8, beta_schedule="cosine")
    theta = sample_tissue_prior(prior, scheduler, (2, 3, 4, 4), prediction_type="sample", b0_t=3.0)
    torch.testing.assert_close(theta, torch.full_like(theta, 0.4), atol=1e-3, rtol=0)
    assert prior.b0 is not None and torch.all(prior.b0 == 3.0)
