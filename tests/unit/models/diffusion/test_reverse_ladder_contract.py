"""Every reverse mode, on the three declared ladders (T=29, T=31, T=35).

Two parity contracts and one input contract:

* oracle denoiser (returns the target): the output reproduces the target on every
  bin the ladder acquires and the measurement does not hold;
* identity denoiser (returns its input): the output is the zero-filled measurement;
* spy denoiser: at each call the input carries the lines of ``mask(label)``, which
  is 1/R(label) of the phase-encode lines within one line.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from spectramr.infrastructure.validation.witness.checks.schedule_certification_common import (
    _timesteps,
    build_process_from_config,
)
from spectramr.models.diffusion.kspace_process import (
    VALID_REVERSE_MODES,
    PhysicsInformedColdDiffusion,
)

ROOT = Path(__file__).resolve().parents[4]
LADDERS = {
    "T29_headline": "experiments/inprogress/kspace_filling/experiment_11_kspace_cold_diffusion.yaml",
    "T31_even_ladder": (
        "experiments/inprogress/kspace_filling/ablations/reverse/"
        "experiment_11_reverse_additive_normalized_even_ladder.yaml"
    ),
    "T35_regrid": "experiments/inprogress/kspace_filling/experiment_11_regrid_r1_r2.yaml",
}
MODES = sorted(VALID_REVERSE_MODES)
SIZE = 64


class _Stand(torch.nn.Module):
    """A denoiser stand-in: ``kind`` picks what it returns, and every call is recorded."""

    def __init__(self, process, kind: str, target: torch.Tensor) -> None:
        super().__init__()
        self.kspace_process = process
        self.kind = kind
        self.target = target
        self.calls: list[tuple[int, torch.Tensor]] = []

    def forward(self, x, t, **_kw):
        self.calls.append((int(t.reshape(-1)[0]), x.detach().clone()))
        if self.kind == "oracle":
            return self.target.clone()
        return x.clone()


def _build(name: str):
    doc = yaml.safe_load((ROOT / LADDERS[name]).read_text())
    process = build_process_from_config(doc)
    assert process is not None, f"{name} does not build a mask-cascade ladder"
    return doc, process


def _sample(process, mode: str, kind: str, target: torch.Tensor, num_timesteps: int):
    model = _Stand(process, kind, target)
    sampler = PhysicsInformedColdDiffusion(
        model=model,
        num_timesteps=num_timesteps,
        max_acceleration=process.max_accel,
        center_fraction=process.center_fraction,
        dc_method="hard",
        reverse_mode=mode,
        reverse_clip_ratio=1e6,
        refine_alpha=0.5 if mode == "reveal_refine_dc" else None,
        support_radius=2 if mode == "support_ordered_dc" else None,
        kspace_log_scaled=False,
    )
    # The fixed-seed cascade, not q_sample: q_sample draws a fresh dynamic mask per call.
    mask = process.mask_at(num_timesteps - 1, (SIZE, SIZE)).float().reshape(1, 1, SIZE, SIZE)
    measurement = target * mask
    out = sampler.sample(measurement, mask, start_timestep=num_timesteps - 1)
    return out, mask, model


def _target() -> torch.Tensor:
    gen = torch.Generator().manual_seed(3)
    return torch.randn(1, 2, SIZE, SIZE, generator=gen)


def _line_count(process, support: torch.Tensor) -> int:
    """Number of phase-encode lines carrying any sample in a ``[1, C, H, W]`` support."""
    plane = support.abs().sum(dim=1)[0] > 0
    return int(plane.any(dim=1).sum()) if process.line_dim == -2 else int(plane.any(dim=0).sum())


@pytest.mark.parametrize("ladder", list(LADDERS))
@pytest.mark.parametrize("mode", MODES)
def test_oracle_denoiser_reproduces_the_target_on_unmeasured_bins(ladder, mode):
    doc, process = _build(ladder)
    target = _target()
    out, mask, _ = _sample(process, mode, "oracle", target, _timesteps(doc))
    acquirable = process.mask_at(0, (SIZE, SIZE)).float().reshape(1, 1, SIZE, SIZE)
    unmeasured = (acquirable > 0) & (mask <= 0)
    unmeasured = unmeasured.expand_as(out)
    torch.testing.assert_close(out[unmeasured], target[unmeasured], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("ladder", list(LADDERS))
@pytest.mark.parametrize("mode", MODES)
def test_identity_denoiser_reproduces_the_zero_filled_measurement(ladder, mode):
    doc, process = _build(ladder)
    target = _target()
    out, mask, _ = _sample(process, mode, "identity", target, _timesteps(doc))
    zero_fill = target * mask
    torch.testing.assert_close(out, zero_fill, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("ladder", list(LADDERS))
@pytest.mark.parametrize("mode", MODES)
def test_input_lines_match_the_rung_at_every_call(ladder, mode):
    """Spy: the input at each call carries mask(label), within one line of 1/R(label)."""
    doc, process = _build(ladder)
    num_timesteps = _timesteps(doc)
    # An oracle, not the identity: an identity denoiser writes zeros, so the state
    # carries nothing but the measurement and its nonzero support is not mask(t).
    _, _, model = _sample(process, mode, "oracle", _target(), num_timesteps)
    n_lines = SIZE  # square plane: rows and columns carry the same number of lines
    ladder_rows = {row[0]: row for row in process.describe_ladder((SIZE, SIZE))}
    assert model.calls, "the sampler made no call"
    for label, x in model.calls:
        lines = _line_count(process, x)
        label_mask = process.mask_at(label, (SIZE, SIZE)).float().reshape(1, 1, SIZE, SIZE)
        assert lines == _line_count(process, label_mask.expand(1, 2, SIZE, SIZE)), (
            f"t={label}: input has {lines} lines, mask(t) has a different count"
        )
        expected = n_lines / ladder_rows[label][2]  # realised R from the ladder
        assert abs(lines - expected) <= 1.0, f"t={label}: {lines} lines, expected {expected:.2f}"


def test_t0_terminal_rewrite_passes_the_integrity_check():
    """The terminal t=0 call of replace_freeze_dc_t0 rewrites committed bins by
    design; under integrity_check='raise' that must not trip the partition check."""
    doc, process = _build("T29_headline")
    target = _target()
    num_timesteps = _timesteps(doc)
    model = _Stand(process, "oracle", target)
    sampler = PhysicsInformedColdDiffusion(
        model=model,
        num_timesteps=num_timesteps,
        max_acceleration=process.max_accel,
        center_fraction=process.center_fraction,
        dc_method="hard",
        reverse_mode="replace_freeze_dc_t0",
        reverse_clip_ratio=1e6,
        integrity_check="raise",
        kspace_log_scaled=False,
    )
    mask = process.mask_at(num_timesteps - 1, (SIZE, SIZE)).float().reshape(1, 1, SIZE, SIZE)
    out = sampler.sample(target * mask, mask, start_timestep=num_timesteps - 1)
    assert torch.isfinite(out).all()
    assert model.calls[-1][0] == 0, "the terminal call must run at t=0"
