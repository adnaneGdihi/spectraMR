"""Field-dependent spectral Wiener restorer (MICCAI MRIxFields2026, B-2.7).

A restorer that applies an EMPIRICAL Wiener filter whose noise level is a learned function of the
SOURCE field strength. A small UNet produces an initial estimate ``x0``; a spectral Wiener gain
``G(f, b) = S(f) / (S(f) + N(b))`` is then applied to ``x0``'s own spectrum (``S(f)=|X0(f)|^2`` is
the empirical signal PSD), where the noise level ``N(b)`` rises as the field ``b`` drops (low-field
acquisitions are noisier). At low field the larger ``N`` attenuates low-energy (high-frequency)
bands, shrinking the **recoverable band** — the radial frequency where the Wiener gain crosses 0.5,
a principled estimate of how much spectral content is recoverable at that field.

The Wiener filter operates on a REAL magnitude image's spectrum via ``rfft2``/``irfft2`` — a
spectral neural operator on a real feature map, which physics.md exempts from the complex-k-space
``fft2c`` rule (this is not a complex k-space round-trip). Magnitude-only (1->1); ``field_strength``
is a required forward kwarg (no default) so the Tier-2 probe injects it.

Distinct from the existing ``wiener_unet`` (:class:`~spectramr.models.generators.\
vf_reconstruction_generators.WienerUNetGenerator`), which applies a field-BLIND Wiener with a
single learned scalar noise level: ``FieldWienerNet`` makes the noise level a learned FUNCTION of
the source field and reports the field-dependent recoverable band. The noise floor is RELATIVE to
the per-sample mean signal power, so the gain is scale-invariant (not a near-no-op / kill-all that
depends on the image magnitude), and the field slope is negative-by-construction so the
"lower field -> larger noise" direction is locked throughout training.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.registry import register_model


@register_model(
    name="field_wiener_net",
    training_mode="field_wiener",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
)
class FieldWienerNet(nn.Module):
    """UNet estimate + field-conditioned empirical spectral Wiener gain (B-2.7)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 48,
        n_blocks: int = 3,
        kernel_size: int = 3,
        use_field_noise: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                f"FieldWienerNet is magnitude-only (1->1); got in={in_channels}, out={out_channels}."
            )
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}.")
        if not isinstance(use_field_noise, bool):
            raise ValueError(
                f"use_field_noise must be a bool; got {type(use_field_noise).__name__}."
            )
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        self.use_field_noise = bool(use_field_noise)
        pad = kernel_size // 2
        layers: list[nn.Module] = [nn.Conv2d(1, width, kernel_size, padding=pad), nn.SiLU()]
        for _ in range(n_blocks):
            layers += [
                nn.Conv2d(width, width, kernel_size, padding=pad),
                nn.GroupNorm(min(8, width), width),
                nn.SiLU(),
            ]
        layers += [nn.Conv2d(width, 1, 1)]
        self.unet = nn.Sequential(*layers)
        # Noise-to-mean-signal RATIO N(b) = softplus(slope*log10(b) + c) >= 0, with the field slope
        # NEGATIVE BY CONSTRUCTION (slope = -softplus(noise_a) <= 0) so "lower field -> larger N"
        # holds throughout training (noise_a can never flip the direction). c init 0 -> a non-trivial
        # ratio (~0.4-1.1 across {0.1..7}T). Field-blind control (use_field_noise=False) drops b.
        self.noise_a = nn.Parameter(torch.tensor(0.0))
        self.noise_c = nn.Parameter(torch.tensor(0.0))

    def noise_level(self, field_strength: torch.Tensor) -> torch.Tensor:
        """Per-sample noise-to-MEAN-signal RATIO N(b) >= 0 (structurally rises as the field drops)."""
        b = field_strength.reshape(-1).float()
        slope = -F.softplus(self.noise_a)  # <= 0 by construction: locks the field-dependence sign
        if self.use_field_noise:
            arg = slope * torch.log10(b.clamp_min(1e-3)) + self.noise_c
        else:
            arg = self.noise_c.expand(b.shape[0])  # field-blind constant ratio
        return F.softplus(arg)

    def wiener_gain(self, x0: torch.Tensor, field_strength: torch.Tensor) -> torch.Tensor:
        """SCALE-INVARIANT empirical Wiener gain G(f,b)=S(f)/(S(f)+N(b)*mean_f S) over rfft2(x0).

        The noise floor is N(b) times the per-sample MEAN spectral power, so the gain (and hence the
        recoverable band) is invariant to the absolute scale of ``x0`` — it depends on the
        relative SNR across frequencies, not on |x0|'s magnitude. (A raw additive N would make the
        gain a near-no-op when |x0|^2 >> N or kill everything when |x0|^2 << N, an inert facade.)
        """
        x0f = torch.fft.rfft2(x0, norm="ortho")  # spectral operator on a REAL image (exempt)
        s = x0f.abs().pow(2)  # empirical signal PSD
        s_mean = s.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)  # per-sample mean power
        n = self.noise_level(field_strength).reshape(-1, 1, 1, 1) * s_mean  # relative noise floor
        return s / (s + n + 1e-8)

    def forward(self, x: torch.Tensor, *, field_strength: torch.Tensor, **_: Any) -> torch.Tensor:
        if torch.is_complex(x):
            raise ValueError("FieldWienerNet expects magnitude (real) input.")
        if field_strength.reshape(-1).shape[0] != x.shape[0]:
            raise ValueError(
                f"field_strength batch {field_strength.reshape(-1).shape[0]} != input batch "
                f"{x.shape[0]}; pass one source field per sample."
            )
        x0 = self.unet(x)
        x0f = torch.fft.rfft2(x0, norm="ortho")
        gain = self.wiener_gain(x0, field_strength)
        return torch.fft.irfft2(gain * x0f, s=x.shape[-2:], norm="ortho")

    def recoverable_band(self, field_strength: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """Recoverable band on a FIXED reference spectrum — a DATA-INDEPENDENT witness of N(b).

        On a canonical decaying reference ``S_ref(r) = 1/(1+(r/0.15)^2)``, returns the mean radial
        frequency where the Wiener gain ``S_ref/(S_ref+N(b))`` crosses 0.5 — purely a function of
        ``N(b)``, so it is not confounded by the data spectrum (which could make a data-based band
        a false positive). Lower field -> larger N -> narrower band.
        """
        h, w = int(hw[0]), int(hw[1])
        dev = self.noise_a.device
        fy = torch.fft.fftfreq(h, device=dev).abs()[:, None]
        fx = torch.fft.rfftfreq(w, device=dev).abs()[None, :]
        r = torch.sqrt(fy**2 + fx**2)
        s_ref = 1.0 / (1.0 + (r / 0.15) ** 2)
        n = self.noise_level(field_strength).mean()  # scalar noise ratio for this field
        passband = (s_ref / (s_ref + n) > 0.5).float()
        if float(passband.sum()) < 1.0:
            return torch.zeros((), device=dev)
        return (r * passband).sum() / passband.sum()
