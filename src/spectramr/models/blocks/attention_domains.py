"""Feature-domain SSOT for the ComplexUNet attention dispatch.

The k-space cold-diffusion stack feeds its backbone either **k-space** or
**image** features depending on the YAML knob
``model.model_kwargs.force_pure_kspace`` (``FourierBridgeNetwork`` skips the
entry ``ifft2c`` / exit ``fft2c`` when it is true). Several attention blocks
carry an internal domain assumption (``fft2c``-first vs ``ifft2c``-first), so
the dispatch must tell each block which domain its input tensor actually is.

This module is the single source of truth for:

* the valid ``feature_domain`` values (``validate_feature_domain``),
* which attention types support which feature domains
  (``ATTENTION_DOMAIN_SUPPORT``) — consulted by BOTH the model-side dispatch
  (``KSpaceDownsampleBlock`` / ``KSpaceUpsampleBlock``) and the Tier-0/1 audit
  (``ConfigHealthChecker.check_attention_domain_compatibility``), so an
  illegal YAML combination is rejected at audit time, not at iteration 1,
* which attention types receive the ``feature_domain`` kwarg
  (``DOMAIN_AWARE_ATTENTION``),
* which attention types the ComplexUNet block dispatch can actually build
  (``COMPLEX_UNET_BLOCK_ATTENTION``) — note ``"spatial"`` exists only in the
  up-block dispatch, so it is deliberately absent here,
* the shared interleaved<->complex layout helpers used by the domain-aware
  blocks.

The ``feature_domain`` value is DERIVED (``"kspace" if force_pure_kspace else
"image"``) — it is intentionally not a YAML knob of its own, so there is no
new unread-knob surface (pitfall #15).
"""

from __future__ import annotations

import torch

FEATURE_DOMAINS: frozenset[str] = frozenset({"kspace", "image"})

_BOTH: frozenset[str] = FEATURE_DOMAINS


def validate_feature_domain(value: str) -> str:
    """Normalise and validate a ``feature_domain`` string.

    Raises on anything outside :data:`FEATURE_DOMAINS` — an unknown domain
    must never silently degrade to a default (pitfall #9).
    """
    v = str(value).lower()
    if v not in FEATURE_DOMAINS:
        raise ValueError(
            f"Unknown feature_domain {value!r}. Valid options: {sorted(FEATURE_DOMAINS)}"
        )
    return v


# Per-attention-type supported feature domains. Domain-agnostic blocks (pure
# channel/spatial gates with no FFT) support both trivially; the dual-domain
# family supports both *after* the 2026-07-03 domain-awareness change (they
# conjugate their internal transforms on ``feature_domain``). A future block
# that only implements one domain must register the narrower set here so the
# audit rejects the illegal YAML combination up front — that is the ratchet.
ATTENTION_DOMAIN_SUPPORT: dict[str, frozenset[str]] = {
    "none": _BOTH,
    "self": _BOTH,
    "channel": _BOTH,
    "spatial": _BOTH,
    "kernelized": _BOTH,
    "sparse": _BOTH,
    "cross_contrast_olmpa": _BOTH,
    "dual_domain": _BOTH,
    "kan_dual_domain": _BOTH,
    "wavelet_freq": _BOTH,
    "null_space_dual_domain": _BOTH,
    "hermitian_null_space": _BOTH,
}

# Attention types whose constructors take the ``feature_domain`` kwarg (the
# blocks with an internal FFT/domain assumption). Everything else is
# domain-agnostic and is constructed without it.
DOMAIN_AWARE_ATTENTION: frozenset[str] = frozenset(
    {
        "dual_domain",
        "kan_dual_domain",
        "wavelet_freq",
        "null_space_dual_domain",
        "hermitian_null_space",
    }
)

# What the ComplexUNet down/up block dispatch can actually construct.
# ``"spatial"`` (CBAMSpatialAttention) exists ONLY in the up-block dispatch —
# ComplexUNet builds both blocks, so requesting it crashes at build time; the
# audit rejects it via this set instead.
COMPLEX_UNET_BLOCK_ATTENTION: frozenset[str] = frozenset(
    {
        "none",
        "self",
        "channel",
        "kernelized",
        "sparse",
        "dual_domain",
        "kan_dual_domain",
        "wavelet_freq",
        "cross_contrast_olmpa",
        "null_space_dual_domain",
        "hermitian_null_space",
    }
)


# ---------------------------------------------------------------------------
# Shared layout helpers: interleaved real [B, 2C, H, W] <-> complex [B, C, H, W]
# ---------------------------------------------------------------------------


def interleaved_to_complex(x: torch.Tensor) -> torch.Tensor:
    """``[B, 2C, H, W]`` real interleaved -> ``[B, C, H, W]`` complex."""
    if x.shape[1] % 2 != 0:
        raise ValueError(f"interleaved layout requires even channels, got {x.shape[1]}")
    return torch.complex(x[:, 0::2], x[:, 1::2])


def complex_to_interleaved(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` complex -> ``[B, 2C, H, W]`` real interleaved."""
    return torch.stack([x.real, x.imag], dim=2).flatten(1, 2)
