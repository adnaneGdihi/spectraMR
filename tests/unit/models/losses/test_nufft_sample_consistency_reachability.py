"""The sample-domain term reaches the objective on the arm that declares it.

Registering a loss and declaring it in a YAML is the easy half. Before this
wiring the arm loaded, audited clean and died at the first training step: the
term's ``sample_mask`` had no route through ``_call_safe_loss``, which binds
only ``(pred, target)`` positionally. These tests drive the real config, the
real builder and the real loss computer, so the route is observed rather than
assumed (non-negotiable 16).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder
from spectramr.models.diffusion.noncartesian_spoke_process import (
    NonCartesianSpokeProcess,
)
from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
    UnifiedDiffusionLossComputer,
)

ARM = Path("experiments/inprogress/kspace_filling/nc_graph/experiment_43_nc_control_gridded.yaml")
IM = (32, 32)
LADDER = [1.0, 2.0, 4.0, 16.0]
TERM = "nufft_sample_consistency"


def _arm_config() -> TrainingSettings:
    if not ARM.exists():
        pytest.skip(f"{ARM} not present")
    return TrainingSettings.from_yaml(str(ARM))


def _interleave(z: torch.Tensor) -> torch.Tensor:
    """Complex ``[B, C, H, W]`` -> the ``[B, 2C, H, W]`` layout the arm emits."""
    return (
        torch.view_as_real(z)
        .permute(0, 1, 4, 2, 3)
        .reshape(z.shape[0], 2 * z.shape[1], *z.shape[2:])
    )


def _degraded() -> tuple[torch.Tensor, torch.Tensor, object]:
    process = NonCartesianSpokeProcess(
        num_spokes=64,
        samples_per_spoke=32,
        im_size=IM,
        num_timesteps=len(LADDER),
        max_acceleration=16.0,
        base_acceleration=1.0,
        schedule_kwargs={"acceleration_range": list(LADDER)},
    )
    x0 = torch.randn(1, 2, *IM, dtype=torch.complex64)
    x_t, _ = process.q_sample(x0, torch.tensor([1]))
    return x0, x_t, process.last_sample_measurement


def test_the_arm_declares_the_term_and_the_builder_constructs_it() -> None:
    """The declaration must survive into a module, not just into the YAML."""
    built = LossBuilder(_arm_config(), device=torch.device("cpu"))
    modules = built.build_reconstruction_losses().build()
    assert TERM in modules, f"{TERM} declared on the arm but not built: {sorted(modules)}"


def test_it_lands_in_components_with_a_gradient() -> None:
    """The real computer must fold the term and pass a gradient back to pred."""
    config = _arm_config()
    x0, x_t, measurement = _degraded()
    pred = _interleave(x_t).clone().requires_grad_(True)
    target = _interleave(x0)

    # The FULL built set, not just this term: a single-entry dict would prove
    # the term is callable while saying nothing about the objective the arm
    # actually optimises, which is the claim that matters here.
    modules = LossBuilder(config, device=torch.device("cpu")).build_reconstruction_losses().build()
    # Real coil maps: `sense_adjoint_l1` raises on absent smaps rather than
    # contributing zero, and driving the full set is what surfaces that.
    smaps = torch.full((1, 2, *IM), 1.0 / math.sqrt(2.0), dtype=torch.complex64)
    output = UnifiedDiffusionLossComputer(config).compute(
        pred=pred,
        target=target,
        losses_dict=modules,
        timesteps=torch.tensor([1]),
        smaps=smaps,
        mask=None,
        sample_measurement=measurement,
    )

    assert TERM in output.components, f"term dropped: {sorted(output.components)}"
    missing = {n for n in modules if n not in output.components}
    assert not missing, f"declared but absent from the objective: {sorted(missing)}"
    for name, value in output.components.items():
        assert torch.isfinite(value).all(), f"{name} is not finite"
    assert float(output.components[TERM]) > 0.0
    output.total.backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0.0


def test_without_the_measurement_the_same_call_raises() -> None:
    """PLANTED VIOLATION: the pre-wiring state must not read as a working arm.

    ``_call_safe_loss`` binds only ``(pred, target)``, so before the measurement
    had a route the term raised inside the computer -- which re-raises rather
    than dropping it. A silent drop here would have left the arm training with
    no data fidelity at all.
    """
    config = _arm_config()
    x0, x_t, _ = _degraded()
    modules = LossBuilder(config, device=torch.device("cpu")).build_reconstruction_losses().build()
    with pytest.raises(ValueError, match="sample_measurement=None"):
        UnifiedDiffusionLossComputer(config).compute(
            pred=_interleave(x_t),
            target=_interleave(x0),
            losses_dict={TERM: modules[TERM]},
            timesteps=torch.tensor([1]),
            sample_measurement=None,
        )
