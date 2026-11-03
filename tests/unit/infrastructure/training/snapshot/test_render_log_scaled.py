"""Per-key ``log_scaled`` in debug snapshots (experiment_11 blurry-target triage).

``data.processing.enable_log_scaling`` is an ARM-level fact: it says the pipeline
compresses k-space with ``log1p`` somewhere. It does NOT say that every tensor a
given snapshot captured is on the compressed side of that transform, and the two
base-strategy emitters routinely capture tensors that may not be: ``first_steps``
holds the batch as the dataloader delivered it, and the generic ``model_output``
snapshot pairs the generator's (compressed) output with ``target``/``input`` as
they entered ``train_step``. Those are compressed only when the loader's
``KSpaceNormalizationTransform`` ran; otherwise ``apply_kspace_normalization``
has not run yet.

Decompressing an uncompressed tensor is not a rounding error. ``expm1`` is
applied per-bin under a clamp at ``DECOMPRESS_MAGNITUDE_CEILING``, so every
coefficient above it collapses to the SAME magnitude while its phase survives —
the IFFT then renders a phase-only image: ringing, washed-out interior, no
low-frequency content. On experiment_11_attention_none that made a perfectly
good ground-truth k-space render as a blurry non-brain and sent the triage after
the M4Raw loader, which was never at fault.
"""

from __future__ import annotations

import logging

import torch

from spectramr.data.transforms.normalization import (
    DECOMPRESS_MAGNITUDE_CEILING,
    compress_kspace_log,
    decompress_kspace_log,
)
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.snapshot.render import render_image_preview


def _phantom(size: int = 64, peak: float = 2407.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-stacked 4-coil k-space of a disc phantom, plus its true RSS image.

    ``peak`` defaults to the abs-max the experiment_11 snapshot actually
    recorded, which is itself the proof that tensor was uncompressed: ``log1p``
    saturates at ``ln(FLT_MAX) ~ 88.7`` in float32, so no compressed tensor can
    reach 2407.
    """
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    r = (xx**2 + yy**2).sqrt()
    img = (
        (r < 0.75).float() * 0.6 + (r < 0.4).float() * 0.3 + ((r > 0.5) & (r < 0.6)).float() * 0.25
    )
    coils = torch.stack([img * (1.0 + 0.2 * c) for c in range(4)]).to(torch.complex64)
    k = fft2c(coils.unsqueeze(0))
    k = k * (peak / float(k.abs().max()))
    stacked = torch.zeros(1, 8, size, size)
    stacked[:, 0::2] = k.real
    stacked[:, 1::2] = k.imag
    return stacked, torch.sqrt((coils.abs() ** 2).sum(0))


def _phantom_kspace(peak: float = 2407.0) -> torch.Tensor:
    return _phantom(peak=peak)[0]


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation, which is blind to the preview's [0, 1] rescale."""
    a = a.flatten().float() - a.flatten().float().mean()
    b = b.flatten().float() - b.flatten().float().mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def test_decompressing_an_uncompressed_tensor_destroys_the_image() -> None:
    """The mechanism: ``expm1`` on physical k-space is not a scale change.

    The faithful render is a lossless IFFT round-trip, so it correlates with the
    known RSS image at exactly 1.0 and the bound needs no tuning. The corrupted
    render collapses to ~0.32 because the clamp flattens every strong bin to one
    magnitude and only the phase survives.

    ``decompress_for_view`` now REFUSES this operation (it checks ``|k|max``
    against the ceiling first), so the corruption is produced here by calling
    ``decompress_kspace_log`` directly and rendering the result faithfully. The
    physics is what this test pins — it is the reason the guard exists, and it
    has to stay true for the guard to be worth having.
    """
    k, truth = _phantom()

    faithful = render_image_preview(k, in_kspace=True, authoritative=True)
    corrupted = render_image_preview(
        decompress_kspace_log(k, channel_dim=1), in_kspace=True, authoritative=True
    )
    assert faithful is not None and corrupted is not None

    assert _corr(faithful[0, 0], truth) > 0.999, (
        "the faithful render is no longer a lossless IFFT of the phantom; this "
        "test's baseline is invalid, fix that before reading the line below"
    )
    assert _corr(corrupted[0, 0], truth) < 0.6, (
        "a spurious expm1 left the image recognisable, so the clamp-flatten "
        "mechanism this regression guards has changed"
    )


def test_render_refuses_to_decompress_what_cannot_be_compressed(caplog) -> None:
    """A declared-log-scaled tensor above the ceiling is rendered as-is.

    This is the half of the guard that was missing: the old check fired only
    when decompression failed to EXPAND ``|k|max``, and spurious ``expm1``
    always expands, so the catastrophic direction was the one that passed in
    silence. ``experiment_11_attention_none`` declared ``enable_log_scaling:
    true`` while its batch reached ``train_step`` uncompressed, and every
    declared-k-space snapshot rendered an edge map with nothing in the log.
    """
    k, truth = _phantom()  # peak 2407 — 27x the float32 log1p bound of ~88.7
    assert float(k.abs().max()) > DECOMPRESS_MAGNITUDE_CEILING

    with caplog.at_level(logging.WARNING):
        rendered = render_image_preview(k, in_kspace=True, authoritative=True, log_scaled=True)

    assert rendered is not None
    assert _corr(rendered[0, 0], truth) > 0.999, (
        "the guard did not fire: a tensor declared log-scaled at |k|max 2407 "
        "was expm1'd anyway, which is #682 wearing the fix's own clothes"
    )
    assert "refusing to decompress" in caplog.text
    # Both readings must be offered: a pipeline tensor that was never
    # compressed, OR a prediction that diverged in compressed units.
    assert "never compressed" in caplog.text and "DIVERGED" in caplog.text


def test_render_still_decompresses_a_genuinely_compressed_tensor() -> None:
    """The guard must not degrade into a blanket skip.

    A real ``log1p``-compressed spectrum peaks at ``log1p(2407) ~ 7.8``, well
    under the ceiling, and must still be decompressed — otherwise the fix for
    the spurious direction re-creates #682 in the correct direction.
    """
    k, truth = _phantom()
    compressed = compress_kspace_log(k, channel_dim=1)
    assert float(compressed.abs().max()) < DECOMPRESS_MAGNITUDE_CEILING

    rendered = render_image_preview(compressed, in_kspace=True, authoritative=True, log_scaled=True)
    assert rendered is not None
    assert _corr(rendered[0, 0], truth) > 0.99, (
        "a genuinely compressed spectrum did not survive the round trip, so the "
        "ceiling guard is skipping decompression it should be applying"
    )
