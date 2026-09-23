"""Fourier-bridge contract for the self-bridging physics losses.

Targets ``spectramr.models.losses.physics_losses``:

- ``DifferentiableFourierBridge`` — k-space -> image projection used by the losses
  below AND by ``LossBuilder``'s list-based wrapper. It has no domain detection:
  ``_kspace_to_image`` applies ``ifft2c`` unconditionally.
- ``ComplexGradientLoss`` (``complex_spatial_gradient``) — carries its own bridge
  by default, so it expects k-space.

The regression pinned here is issue #467: because both the loss and the builder
bridge, declaring ``complex_spatial_gradient`` under ``losses.image_losses`` with
``losses.output_domain: kspace`` inverse-transformed the tensor twice. The loss
docstring says it "enforces both magnitude sharpness (edges) and local phase
coherence" — ``test_double_bridge_is_phase_blind`` shows that under the double
bridge it is exactly phase-BLIND, which is what makes the bug worth a hard guard
rather than a comment: the term scored 0 on the error it exists to catch.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.physics_losses import (
    ComplexGradientLoss,
    DifferentiableFourierBridge,
)
from spectramr.models.losses.registry import LossRegistry

COILS = 4  # matches the kspace_filling cohort (model.in_channels: 8 real)
SHAPE = (2, COILS, 16, 16)


def _interleave(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] complex -> [B, 2C, H, W] real-interleaved (the k-space layout)."""
    return torch.stack([x.real, x.imag], dim=2).flatten(1, 2)


@pytest.fixture()
def kspace_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Two k-spaces whose per-coil image MAGNITUDES are identical but phases differ.

    This is the input that separates a phase-aware term from a magnitude-only one.
    """
    torch.manual_seed(0)
    mag = torch.rand(SHAPE) + 0.5
    phase_t = torch.rand(SHAPE) * 2 * torch.pi
    phase_p = torch.rand(SHAPE) * 2 * torch.pi

    img_target = mag * torch.exp(1j * phase_t)
    img_pred = mag * torch.exp(1j * phase_p)
    # Identical magnitudes by construction — the only difference is phase.
    torch.testing.assert_close(img_pred.abs(), img_target.abs())

    return _interleave(fft2c(img_pred)), _interleave(fft2c(img_target))


def _outer_bridge_output(kspace: torch.Tensor) -> torch.Tensor:
    """What LossBuilder's ``image_losses`` wrapper hands the inner loss."""
    return DifferentiableFourierBridge(spatial_loss_fn=None, return_complex=False)._kspace_to_image(
        kspace
    )


def test_default_is_self_bridging_and_the_flag_is_exposed() -> None:
    """The attribute is what LossBuilder's guard inspects — it must exist and be True."""
    loss = ComplexGradientLoss()
    assert loss.use_fourier_bridge is True
    assert ComplexGradientLoss(use_fourier_bridge=False).use_fourier_bridge is False


def test_registry_declares_the_kspace_domain() -> None:
    """#467: the loss self-bridges FROM k-space, so that is its declared domain.

    Undeclared is what let the misplacement look legitimate on review — 108 other
    losses declare one.
    """
    caps = LossRegistry.get_loss_capabilities("complex_spatial_gradient")
    assert caps is not None, "registration must carry domain metadata"
    assert caps.domain == "kspace"

    # ``compatible_with`` is not projected onto LossCapabilities, so read the
    # registry's own store (the same dict ``get_loss_capabilities`` synthesizes from).
    meta = LossRegistry._loss_domains["complex_spatial_gradient"]
    assert "complex" in (meta.get("compatible_with") or []), (
        "with use_fourier_bridge=False the loss consumes a complex IMAGE, so "
        "complex_losses is a legal placement"
    )


def test_single_bridge_is_phase_sensitive(
    kspace_pair: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Correct placement (kspace_losses): the term sees the phase error."""
    k_pred, k_target = kspace_pair
    loss = ComplexGradientLoss()  # self-bridges once

    value = loss(k_pred, k_target)
    assert torch.isfinite(value)
    assert value > 1e-3, (
        "identical magnitudes with scrambled phase must register on a term that "
        f"claims to enforce phase coherence; got {value.item():.3e}"
    )


def test_double_bridge_is_phase_blind(
    kspace_pair: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """#467: the pre-fix path scored ~0 on a pure phase error.

    Reproduces the old composition explicitly — outer builder bridge, then the
    loss's own bridge on top — and shows the term became a magnitude-only penalty.
    """
    k_pred, k_target = kspace_pair
    loss = ComplexGradientLoss()  # inner bridge still on: the double transform

    doubled = loss(_outer_bridge_output(k_pred), _outer_bridge_output(k_target))
    single = loss(k_pred, k_target)

    assert doubled < 1e-6, (
        "the double bridge should collapse a pure phase error to ~0 (both sides "
        f"reduce to the same magnitude image); got {doubled.item():.3e}"
    )
    assert single > 1e3 * doubled, (
        "the correctly-bridged term must dominate the double-bridged one on a "
        f"phase error: single={single.item():.3e} doubled={doubled.item():.3e}"
    )


def test_double_bridge_halves_the_coil_count() -> None:
    """The mechanical tell: 4 complex coils -> 4 real magnitudes -> 2 'complex' coils.

    The inner bridge re-pairs coil-0's magnitude as the real part and coil-1's as
    the imaginary part. That is also why ``sense_adjoint_l1`` then silently
    truncated its 4-coil sensitivity maps to 2.
    """
    torch.manual_seed(0)
    kspace = torch.randn(SHAPE[0], 2 * COILS, SHAPE[2], SHAPE[3])

    once = _outer_bridge_output(kspace)
    assert once.shape[1] == COILS and not once.is_complex()

    twice = DifferentiableFourierBridge(spatial_loss_fn=None, return_complex=True)._kspace_to_image(
        once
    )
    assert twice.shape[1] == COILS // 2 and twice.is_complex()


def test_bridge_has_no_domain_detection() -> None:
    """Why the guard lives in the builder and not in the bridge.

    k-space is real-interleaved in this codebase, so "input is real" cannot
    distinguish k-space from a magnitude image — auto-detection would be a silent
    fallback (pitfall #9). The bridge stays unconditional and the placement is
    validated instead.
    """
    torch.manual_seed(0)
    kspace = torch.randn(SHAPE[0], 2 * COILS, SHAPE[2], SHAPE[3])
    bridge = DifferentiableFourierBridge(spatial_loss_fn=None, return_complex=True)

    # Same real-valued layout, transformed again without complaint.
    assert bridge._kspace_to_image(kspace).is_complex()
    assert bridge._kspace_to_image(_outer_bridge_output(kspace)).is_complex()


def test_gradients_flow_through_the_single_bridge(
    kspace_pair: tuple[torch.Tensor, torch.Tensor],
) -> None:
    k_pred, k_target = kspace_pair
    k_pred = k_pred.clone().requires_grad_(True)

    ComplexGradientLoss()(k_pred, k_target).backward()

    assert k_pred.grad is not None
    assert torch.isfinite(k_pred.grad).all()
    assert k_pred.grad.abs().sum() > 0


class _SpyLoss(torch.nn.Module):
    """Records what the bridge hands it; names only ``coil_sensitivities``."""

    def forward(self, pred, target, coil_sensitivities=None):
        self.received = coil_sensitivities
        return pred.abs().mean()


def test_the_bridge_refiles_the_coil_maps_for_its_inner_loss() -> None:
    """The bridge is the hop that narrows kwargs to a bridged term.

    The builder's wrapper forwards ``**kwargs`` untouched, so the diffusion
    strategy's ``smaps`` reaches the bridge intact; a bridge that filtered on the
    inner signature without reconciling dropped it, and the coil barrier bound
    ``None`` on every kspace_filling arm.
    """
    maps = torch.randn(SHAPE, dtype=torch.complex64)
    spy = _SpyLoss()
    bridge = DifferentiableFourierBridge(spatial_loss_fn=spy, return_complex=True)
    kspace = torch.randn(SHAPE[0], 2 * COILS, SHAPE[2], SHAPE[3])

    bridge(kspace, kspace, smaps=maps, timesteps=torch.zeros(SHAPE[0]))

    assert spy.received is maps
