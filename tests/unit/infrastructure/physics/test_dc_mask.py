"""The DC sampling-mask channel contract, and the crash it is here to stop.

Every arm in the 2026-09-16 cluster run that selected a LEARNED ``dc_method``
(``adaptive`` / ``target_aware_fsdc``) trained for its whole budget and then
raised in validation, because ``PhysicsInformedColdDiffusion._apply_observed_dc``
hands the layer its observed support ``obs`` in the INTERLEAVED layout the
sampler works in -- ``[B, 2*coils, H, W]`` -- while the layer has already folded
its prediction and its measurement down to ``[B, coils, H, W]`` complex.

The five arms: ``experiment_11_dc_adaptive``, ``experiment_11_dc_target_aware_fsdc``,
``experiment_11_kan_ablation_cnn_adc``, ``experiment_11_kan_ablation_legacy_dual_domain``,
``experiment_cross_contrast_kspace_diffusion``.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.data_consistency import (
    AdaptiveDataConsistency,
    NoiseAdaptiveDataConsistency,
    TargetAwareFSDC,
)
from spectramr.infrastructure.physics.data_consistency_layer import MaskedReplacementDataConsistency
from spectramr.infrastructure.physics.dc_mask import align_dc_mask
from spectramr.infrastructure.physics.kan_data_consistency import KANAdaptiveDataConsistency

COILS, H, W = 4, 32, 32
INTERLEAVED = 2 * COILS


def _line_mask(channels: int) -> torch.Tensor:
    """Every other phase-encode line, replicated over *channels* channels."""
    mask = torch.zeros(1, channels, H, W)
    mask[..., ::2, :] = 1.0
    return mask


def _learned_layers() -> list[tuple[str, torch.nn.Module]]:
    """The four DC layers ``_apply_observed_dc`` delegates to, by ``dc_method``."""
    return [
        ("adaptive", AdaptiveDataConsistency()),
        ("noise_adaptive", NoiseAdaptiveDataConsistency()),
        ("kan_adaptive", KANAdaptiveDataConsistency()),
        ("target_aware_fsdc", TargetAwareFSDC(target_channels=COILS)),
    ]


# --- the helper's own contract ---------------------------------------------


def test_already_broadcastable_masks_pass_through_untouched():
    single, per_coil = _line_mask(1), _line_mask(COILS)
    assert align_dc_mask(single, COILS) is single
    assert align_dc_mask(per_coil, COILS) is per_coil


def test_interleaved_mask_reduces_pairwise_not_by_slicing():
    """A pair is observed when EITHER half is -- observation is per coefficient."""
    mask = torch.zeros(1, INTERLEAVED, H, W)
    mask[:, 1, 4, :] = 1.0  # only the imaginary half of coil 0 is marked
    aligned = align_dc_mask(mask, COILS)
    assert aligned.shape == (1, COILS, H, W)
    assert aligned[0, 0, 4, :].all(), "slicing [:, :C] would have dropped this line"


def test_unpairable_channel_count_collapses_to_the_coil_wise_or():
    """Neither 1, C nor 2C: fall back to the logical OR over coils."""
    mask = torch.zeros(1, 3, H, W)
    mask[:, 1, 8, :] = 1.0
    aligned = align_dc_mask(mask, COILS)
    assert aligned.shape == (1, 1, H, W)
    assert aligned[0, 0, 8, :].all()


def test_complex_mask_is_read_through_its_real_part():
    mask = torch.complex(_line_mask(INTERLEAVED), torch.zeros(1, INTERLEAVED, H, W))
    aligned = align_dc_mask(mask, COILS)
    assert not torch.is_complex(aligned)
    assert aligned.shape == (1, COILS, H, W)


# --- the layers, against the shape that crashed ----------------------------


@pytest.mark.parametrize(
    ("name", "layer"), _learned_layers(), ids=[n for n, _ in _learned_layers()]
)
def test_learned_dc_layers_accept_the_samplers_interleaved_support(name, layer):
    """The planted violation: the exact ``obs`` layout ``_apply_observed_dc`` passes."""
    torch.manual_seed(0)
    x0 = torch.randn(1, INTERLEAVED, H, W)
    measured = torch.randn(1, INTERLEAVED, H, W)
    out = layer(x0, measured, _line_mask(INTERLEAVED), is_kspace_domain=True)
    assert out.shape == x0.shape
    assert not torch.allclose(out, x0), f"{name} returned the prediction unchanged"


@pytest.mark.parametrize(
    ("name", "layer"), _learned_layers(), ids=[n for n, _ in _learned_layers()]
)
def test_interleaved_and_single_channel_masks_agree(name, layer):
    """Alignment must not change the physics: same mask, two layouts, one answer."""
    torch.manual_seed(0)
    layer.eval()
    x0 = torch.randn(1, INTERLEAVED, H, W)
    measured = torch.randn(1, INTERLEAVED, H, W)
    with torch.no_grad():
        interleaved = layer(x0, measured, _line_mask(INTERLEAVED), is_kspace_domain=True)
        single = layer(x0, measured, _line_mask(1), is_kspace_domain=True)
    assert torch.allclose(interleaved, single, atol=1e-6)


def test_hard_dc_layer_shares_the_same_reduction():
    """``MaskedReplacementDataConsistency``'s own guard is now the shared one (2026-05-10 crash)."""
    torch.manual_seed(0)
    image = torch.randn(1, INTERLEAVED, H, W)
    measured = torch.randn(1, INTERLEAVED, H, W)
    dc = MaskedReplacementDataConsistency()
    assert torch.allclose(
        dc(image, measured, _line_mask(INTERLEAVED)),
        dc(image, measured, _line_mask(1)),
        atol=1e-6,
    )
