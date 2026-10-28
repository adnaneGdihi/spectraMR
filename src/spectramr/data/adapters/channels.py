"""Channel and complex-tensor adapters.

These bridge the most common channel-layout mismatches the cluster
smoke runs have surfaced over the last week:

- Multi-coil dataset target (8-ch real/imag interleaved) ↔ 1-ch
  magnitude prediction. The bloch_cycle inline ``_to_magnitude_image``
  one-off (cycle_bloch_strategy.py) is the prototype this replaces.
- Complex-tensor model output ↔ real-valued image-domain metric.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.data.adapters.registry import register_adapter


@register_adapter(
    name="rss_coils_to_magnitude",
    bridges_from={"domain": "image"},  # any-channel real-valued image
    bridges_to={"domain": "image"},
    # ``pre_model`` was added 2026-05-11 to bridge the m4raw dataset's
    # cross-contrast doubling (source||target cat + real/imag interleave
    # → 4 ch out of an "rss_image"-declared pipeline) down to the 1-ch
    # input that image-domain reconstruction models expect.  See the
    # ``[CoilAdapter]`` block in ``BaseTrainingStrategy.train_step`` and
    # the change discipline note in ``docs/losses_reference.rst``.
    #
    insertion_points=(
        "pre_model",
        "pre_loss_target",
        "pre_metric",
    ),
    invertible=False,
    side_effects=("loses_per_coil_information",),
)
class RSSCoilsToMagnitude(nn.Module):
    """Reduce ``[B, C, ...]`` → ``[B, 1, ...]`` via root-sum-of-squares.

    Σ(a²+b²) over C/2 complex coils equals Σc² over C real channels
    (real/imag interleaved), so the same operation works for either
    interpretation. Single-channel input is returned unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x
        if torch.is_complex(x):
            return torch.sqrt((x.abs() ** 2).sum(dim=1, keepdim=True))
        return torch.sqrt((x.float() ** 2).sum(dim=1, keepdim=True))


@register_adapter(
    name="magnitude_from_complex",
    bridges_from={"domain": "complex_image"},
    bridges_to={"domain": "image"},
    # ``pre_model`` added 2026-05-25 (VF smoke): feeding the magnitude of a
    # complex/2C-interleaved acquisition to a real-valued image model is a
    # legitimate input bridge — the sibling ``rss_coils_to_magnitude`` (same
    # magnitude reduction) already allows ``pre_model``. Six VF arms declare
    # it under ``adapters.pre_model`` (eval_c7, exp_c4/c6, exp_p1/p2/p7);
    # rejecting the hook was a configuration-side blocker, not a contract
    # invariant. Phase loss is the documented ``discards_phase`` side-effect.
    insertion_points=("pre_model", "pre_loss_target", "pre_metric"),
    invertible=False,
    side_effects=("discards_phase",),
)
class MagnitudeFromComplex(nn.Module):
    """Magnitude of a complex tensor or interleaved 2C real channels.

    For a torch.complex tensor: ``|x|``. For a 2C interleaved real
    tensor where the even channels hold the real part: reshape to
    complex first, then magnitude. Result keeps the channel dimension
    (single-channel magnitude per coil).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            return x.abs()
        if x.shape[1] % 2 == 0:
            real = x[:, 0::2]
            imag = x[:, 1::2]
            return torch.sqrt(real * real + imag * imag)
        # Odd channel count: a single channel is an already-magnitude image
        # (idempotent). Any other odd count cannot be interpreted as interleaved
        # real/imag pairs — raise rather than silently passing a non-magnitude
        # tensor through, since the adapter declares bridges_to image (NN#3).
        if x.shape[1] == 1:
            return x
        raise ValueError(
            "magnitude_from_complex expects a complex tensor, an even channel "
            "count (interleaved real/imag), or 1 channel (already magnitude); "
            f"got {x.shape[1]} channels (shape {tuple(x.shape)})."
        )


@register_adapter(
    name="real_imag_interleave_to_complex",
    bridges_from={"domain": "image"},
    bridges_to={"domain": "complex_image"},
    insertion_points=("pre_model",),
    invertible=True,
    side_effects=("requires_even_channel_count",),
)
class RealImagInterleaveToComplex(nn.Module):
    """``[B, 2C, ...]`` real interleaved → ``[B, C, ...]`` torch.complex.

    Mirrors the layout convention used by ``ComplexConv2d``: even
    channels are real, odd channels are imaginary.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            return x
        if x.shape[1] % 2 != 0:
            raise ValueError(
                f"real_imag_interleave_to_complex needs even channel count; got {x.shape[1]}."
            )
        return torch.complex(x[:, 0::2], x[:, 1::2])


@register_adapter(
    name="complex_to_real_imag_interleave",
    bridges_from={"domain": "complex_image"},
    bridges_to={"domain": "image"},
    # ``pre_model`` added 2026-05-12 so a ``ifft_kspace_to_image`` →
    # ``complex_to_real_imag_interleave`` chain can bridge a k-space
    # dataset to an image-domain model that wants 2-channel real input
    # (e.g. ``graph_unet`` consuming spiral k-space samples).
    insertion_points=("pre_model", "post_model"),
    invertible=True,
    side_effects=(),
)
class ComplexToRealImagInterleave(nn.Module):
    """``[B, C, ...]`` torch.complex → ``[B, 2C, ...]`` interleaved real."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise ValueError(
                f"complex_to_real_imag_interleave expects a complex tensor "
                f"(bridges_from={{domain: complex_image}}); "
                f"got real dtype={x.dtype}, shape={tuple(x.shape)}. "
                "Did you forget an ifft_kspace_to_image adapter before this step?"
            )
        b, c, *spatial = x.shape
        out = x.new_zeros((b, c * 2, *spatial), dtype=torch.float32)
        out[:, 0::2] = x.real
        out[:, 1::2] = x.imag
        return out


__all__ = [
    "ComplexToRealImagInterleave",
    "MagnitudeFromComplex",
    "RSSCoilsToMagnitude",
    "RealImagInterleaveToComplex",
]
