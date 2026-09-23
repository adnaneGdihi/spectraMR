r"""Fibre-aware null-space content loss (cold-diffusion design corollary C5).

Standard data-fidelity terms score the OBSERVED k-space; the fabrication the
trust layer hunts lives entirely in the null space :math:`(1-M)` — the bins
the measurement never constrains. C5's fibre-aware objective adds explicit
supervision there:

.. math::

    \mathcal{L}_{\mathrm{null}} =
        \bigl\lVert (1 - M)\,(\mathbf{F}\hat x - \mathbf{F}x) \bigr\rVert^2

averaged over the null bins, so the scale does not depend on the acceleration
factor. Gradients flow ONLY through unobserved bins — the term is exactly
complementary to hard data consistency (which owns the observed support) and
never fights it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss

VALID_INPUT_DOMAINS: frozenset[str] = frozenset({"kspace", "image"})
VALID_LOSS_TYPES: frozenset[str] = frozenset({"l1", "l2"})


def _to_complex_image(x: torch.Tensor) -> torch.Tensor:
    """Coerce ``[B, 2C, H, W]`` stacked real/imag (or real) to complex ``[B, C, H, W]``."""
    if torch.is_complex(x):
        return x
    c = x.shape[1]
    if c % 2 == 0 and c >= 2:
        xr = x.view(x.shape[0], c // 2, 2, x.shape[-2], x.shape[-1])
        return torch.complex(xr[:, :, 0], xr[:, :, 1])
    return torch.complex(x, torch.zeros_like(x))


@register_loss("null_space_content", aliases=["unobserved_line_loss"], domain="kspace")
class NullSpaceContentLoss(nn.Module):
    """Supervised penalty on invented content in the unobserved k-space bins.

    Args:
        input_domain: ``"kspace"`` (default — pred/target are already k-space,
            the cold-diffusion arm's native domain) or ``"image"`` (an FFT via
            the physics SSOT ``fft2c`` maps them to k-space first).
        loss_type: ``"l2"`` (default, the C5 form) or ``"l1"``.

    Shape:
        pred/target: ``[B, 2C, H, W]`` stacked real/imag or complex
        ``[B, C, H, W]``; ``mask``: broadcastable ``[B|1, 1, H, W]``, 1 =
        observed. The mask is MANDATORY at forward — a null space is undefined
        without one, and silently skipping would leave the term reading as
        "on" in the YAML while training without it (pitfall #9).

        The mask may be ``bool`` (what the loader delivers), float or complex;
        all three are cast to the k-space dtype before the complement is taken.
    """

    def __init__(
        self,
        input_domain: str = "kspace",
        loss_type: str = "l2",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if input_domain not in VALID_INPUT_DOMAINS:
            raise ValueError(
                f"Unknown input_domain {input_domain!r} for null_space_content. "
                f"Valid: {sorted(VALID_INPUT_DOMAINS)}."
            )
        if loss_type not in VALID_LOSS_TYPES:
            raise ValueError(
                f"Unknown loss_type {loss_type!r} for null_space_content. "
                f"Valid: {sorted(VALID_LOSS_TYPES)}."
            )
        self.input_domain = input_domain
        self.loss_type = loss_type
        self.eps = eps

    def _to_kspace(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_domain == "kspace":
            return x
        from spectramr.infrastructure.physics.fft_ops import fft2c

        return fft2c(_to_complex_image(x))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"null_space_content needs matching shapes; got pred "
                f"{tuple(pred.shape)} vs target {tuple(target.shape)}."
            )
        if mask is None:
            raise ValueError(
                "null_space_content requires the sampling mask: the null space "
                "(1 - M) is undefined without it. Pass mask=... at compute time "
                "or disable the loss."
            )
        k_pred = self._to_kspace(pred)
        k_target = self._to_kspace(target)
        m = mask.real if torch.is_complex(mask) else mask
        # Cast the OPERAND, not the result: torch refuses `1.0 - m` outright on a bool
        # tensor, and bool is what the loader delivers, so casting afterwards never got
        # the chance to run (#2253). `~m` is not a substitute -- the complement is used
        # as a WEIGHT below, so a soft mask has to keep its fractional values.
        weight_dtype = k_pred.real.dtype if torch.is_complex(k_pred) else k_pred.dtype
        null = 1.0 - m.to(weight_dtype)

        diff = k_pred - k_target
        if torch.is_complex(diff):
            mag2 = diff.real**2 + diff.imag**2
        else:
            mag2 = diff**2
        per_element = torch.sqrt(mag2 + self.eps**2) if self.loss_type == "l1" else mag2

        weighted = per_element * null
        # Mean over the null bins only: the loss scale is then independent of
        # the acceleration factor (a 12x mask and a 2x mask penalise invented
        # content equally per bin).
        denom = null.expand_as(per_element).sum().clamp_min(1.0)
        return weighted.sum() / denom

    def extra_repr(self) -> str:
        return f"input_domain='{self.input_domain}', loss_type='{self.loss_type}'"


__all__ = ["NullSpaceContentLoss"]
