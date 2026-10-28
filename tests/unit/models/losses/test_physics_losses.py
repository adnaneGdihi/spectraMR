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


# ---------------------------------------------------------------------------
# #2338: the k-space -> image step undoes the arm's log1p compression
# ---------------------------------------------------------------------------


def _coil_kspace(seed: int = 0, coils: int = 3, n: int = 16) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(1, coils, n, n, generator=g) + 1j * torch.randn(1, coils, n, n, generator=g)
    return fft2c(img.to(torch.complex64))


def _interleave(z: torch.Tensor) -> torch.Tensor:
    b, c, h, w = z.shape
    return torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(b, 2 * c, h, w).contiguous()


class TestKspaceToImageDecompresses:
    def test_log_scaled_returns_the_image_of_the_uncompressed_plane(self):
        from spectramr.data.transforms.normalization import compress_kspace_log
        from spectramr.infrastructure.physics.fft_ops import ifft2c
        from spectramr.models.losses.physics_losses import kspace_to_image

        k = _coil_kspace()
        compressed = compress_kspace_log(k)
        got = kspace_to_image(_interleave(compressed), return_complex=True, log_scaled=True)
        torch.testing.assert_close(got, ifft2c(k), rtol=1e-4, atol=1e-5)

    def test_without_the_flag_the_compressed_plane_is_not_the_image(self):
        """The defect the flag closes: iFFT of compressed k-space is another object."""
        from spectramr.data.transforms.normalization import compress_kspace_log
        from spectramr.infrastructure.physics.fft_ops import ifft2c
        from spectramr.models.losses.physics_losses import kspace_to_image

        k = _coil_kspace()
        got = kspace_to_image(_interleave(compress_kspace_log(k)), return_complex=True)
        rel = (got - ifft2c(k)).abs().norm() / ifft2c(k).abs().norm()
        assert rel > 0.1

    def test_the_bridge_is_unchanged_when_the_flag_is_off(self):
        """Arms without log scaling keep the exact pre-#2338 transform."""
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        k = _coil_kspace()
        bridge = DifferentiableFourierBridge(spatial_loss_fn=lambda a, b: a)
        assert bridge.log_scaled is False
        assert torch.equal(bridge._kspace_to_image(_interleave(k)), torch.abs(ifft2c(k)))

    def test_the_bridge_forwards_its_flag(self):
        from spectramr.data.transforms.normalization import compress_kspace_log
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        k = _coil_kspace()
        bridge = DifferentiableFourierBridge(spatial_loss_fn=lambda a, b: a, log_scaled=True)
        got = bridge._kspace_to_image(_interleave(compress_kspace_log(k)))
        torch.testing.assert_close(got, torch.abs(ifft2c(k)), rtol=1e-4, atol=1e-5)

    def test_complex_spatial_gradient_scores_zero_on_the_target_either_way_and_images_when_flagged(self):
        from spectramr.data.transforms.normalization import compress_kspace_log

        loss = ComplexGradientLoss()
        k = _coil_kspace()
        target = _interleave(compress_kspace_log(k))
        for m in loss.modules():
            if hasattr(m, "log_scaled"):
                m.log_scaled = True
        assert float(loss(target, target)) == pytest.approx(0.0, abs=1e-7)

    def test_sense_adjoint_phase_declares_its_bridge_and_decompresses(self):
        from spectramr.data.transforms.normalization import compress_kspace_log
        from spectramr.models.losses.physics_losses import SENSEAdjointPhaseLoss

        loss = SENSEAdjointPhaseLoss()
        assert loss.use_fourier_bridge is True and loss.log_scaled is False
        k = _coil_kspace()
        maps = torch.ones_like(k)
        loss.log_scaled = True
        flagged = loss(compress_kspace_log(k), k * 0, maps)
        loss.log_scaled = False
        unflagged = loss(k, k * 0, maps)
        torch.testing.assert_close(flagged, unflagged, rtol=1e-4, atol=1e-5)
