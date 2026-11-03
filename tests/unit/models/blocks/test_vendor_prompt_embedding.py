"""Tests for the vendor / protocol soft-prompt embedding.

Multi-contrast Item 4 (``TODO/backlog_multi_contrast_remaining.md``).
The block composes a learnable per-vendor offset with the existing
:class:`AcquisitionEmbedding` output. Zero-init is the load-bearing
choice -- the model starts from the physics-only baseline and only
learns vendor-specific deviations as training progresses.
"""
from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.vendor_prompt_embedding import VendorPromptEmbedding


def test_zero_init_means_identity_at_construction() -> None:
    """At t=0 the vendor offset must be 0 -> output equals the bare acquisition embedding.

    This is the non-breaking-by-construction guarantee from the spec:
    enabling the block adds parameters but does not perturb model
    behaviour until those parameters move.
    """
    block = VendorPromptEmbedding(n_vendors=3, embed_dim=8)
    acq = torch.randn(2, 8)
    vendor = torch.tensor([0, 2], dtype=torch.long)
    out = block(acq, vendor)
    assert torch.allclose(out, acq, atol=1e-6)


def test_post_training_offset_is_non_zero() -> None:
    """After perturbing the lookup, the offset must show up in the output."""
    block = VendorPromptEmbedding(n_vendors=2, embed_dim=4)
    with torch.no_grad():
        block.vendor_lookup.weight.copy_(torch.tensor([[0.0, 0.0, 0.0, 0.0],
                                                       [1.0, -1.0, 2.0, 0.5]]))
    acq = torch.zeros(2, 4)
    vendor = torch.tensor([0, 1], dtype=torch.long)
    out = block(acq, vendor)
    assert torch.allclose(out[0], torch.zeros(4))
    assert torch.allclose(out[1], torch.tensor([1.0, -1.0, 2.0, 0.5]))


def test_random_init_breaks_identity() -> None:
    """``zero_init=False`` must produce a non-identity offset at construction."""
    torch.manual_seed(0)
    block = VendorPromptEmbedding(n_vendors=3, embed_dim=8, zero_init=False)
    acq = torch.randn(2, 8)
    vendor = torch.tensor([0, 2], dtype=torch.long)
    out = block(acq, vendor)
    assert not torch.allclose(out, acq, atol=1e-3)


def test_rejects_int_vendor_id_with_wrong_dtype() -> None:
    block = VendorPromptEmbedding(n_vendors=3, embed_dim=4)
    acq = torch.randn(1, 4)
    with pytest.raises(TypeError, match="vendor_id must be a long tensor"):
        block(acq, torch.tensor([0.0]))


def test_rejects_out_of_range_vendor_id() -> None:
    """An id >= n_vendors must raise -- pitfall #9 (no silent index clamping)."""
    block = VendorPromptEmbedding(n_vendors=2, embed_dim=4)
    acq = torch.randn(2, 4)
    with pytest.raises(IndexError, match=r"vendor_id contains index"):
        block(acq, torch.tensor([0, 5], dtype=torch.long))


def test_rejects_zero_vendor_cohort() -> None:
    """Constructing the block with n_vendors=0 must raise -- there's no embedding to learn."""
    with pytest.raises(ValueError, match="n_vendors >= 1"):
        VendorPromptEmbedding(n_vendors=0, embed_dim=4)


def test_audit_check_skips_when_vendor_map_empty() -> None:
    """The audit must be a no-op when vendor conditioning is not enabled."""
    from spectramr.config.settings import TrainingSettings  # type: ignore
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    # Build a minimal settings stub with empty vendor map.
    class _D:
        class multi_contrast:  # noqa: D401
            vendor_map = {}
            n_vendors = 0

    class _Cfg:
        data = _D()
        model = None

    checker = ConfigHealthChecker()
    result = checker.check_vendor_model_support(_Cfg())  # type: ignore[arg-type]
    assert result.passed
    assert "skipping" in result.message.lower()


def test_audit_check_rejects_mismatched_cardinality() -> None:
    """vendor_map and n_vendors must agree when both are set."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    class _D:
        class multi_contrast:
            vendor_map = {"siemens": 0, "ge": 1, "philips": 2}
            n_vendors = 5

    class _M:
        model_type = "latent_diffusion_residual_head"

    class _Cfg:
        data = _D()
        model = _M()

    checker = ConfigHealthChecker()
    result = checker.check_vendor_model_support(_Cfg())  # type: ignore[arg-type]
    assert not result.passed
    assert "n_vendors" in result.message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync-debug mode needs a GPU")
def test_forward_does_not_sync_the_gpu() -> None:
    """The range check is host-only; on CUDA nn.Embedding's own assert covers it."""
    block = VendorPromptEmbedding(n_vendors=2, embed_dim=4).cuda()
    acq, ids = torch.randn(2, 4, device="cuda"), torch.tensor([0, 1], device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        block(acq, ids)
    finally:
        torch.cuda.set_sync_debug_mode("default")
