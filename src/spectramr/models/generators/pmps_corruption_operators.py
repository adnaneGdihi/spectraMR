"""PMPS Phase-2 — calibratable corruption operators.

Two thin wrappers around the existing physics primitives
(`noise_simulation`, `motion_simulation`, `maxwell_concomitant`,
`coil_sensitivity`) that expose a small free-parameter vector
:math:`\\psi` for MMD-based calibration against the real paired corpus.

The contribution is the composition + parameterisation, not new
physics. The physics primitives themselves remain untouched.

Both operators are registered under
``training_mode="corruption_calibration"`` so the strategy factory
dispatches to :class:`CorruptionCalibrationStrategy` (a thin alias to
the reconstruction strategy with init-time checkpoint validation).

Output channels are kept identical to the input — the operator is an
image-to-image perturbation, not a reconstructor.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import centred_freqs, fft2c, ifft2c
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

_SNR_SCALING_LAWS = ("hoult", "learned")


def _rician_noise(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Add Rician noise with magnitude ``sigma`` to ``x``.

    Rician magnitude is :math:`|x + (n_R + i n_I)|` with independent
    Gaussian real/imag parts of std :math:`\\sigma`. The function is
    differentiable in ``sigma`` (used for calibration).
    """
    if sigma.ndim == 0:
        sigma = sigma.view(1)
    real = torch.randn_like(x) * sigma
    imag = torch.randn_like(x) * sigma
    return torch.sqrt((x + real).pow(2) + imag.pow(2))


def _smooth_bias_field(x: torch.Tensor, severity: torch.Tensor, n_basis: int = 4) -> torch.Tensor:
    """Multiplicative smooth bias field via a low-rank cosine basis."""
    b, _, h, w = x.shape
    device = x.device
    dtype = x.dtype
    # Generate a small set of low-frequency 2D cosines and randomly
    # weight them per-sample by `severity`. Severity in [0, 1] scales the
    # peak-to-peak modulation; at 0 the field is identity.
    field = torch.ones(b, 1, h, w, device=device, dtype=dtype)
    ys = torch.linspace(0, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    for k in range(1, n_basis + 1):
        coef = torch.randn(b, 1, 1, 1, device=device, dtype=dtype) * severity * 0.1
        field = field + coef * torch.cos(k * torch.pi * yy).view(1, 1, h, w) * torch.cos(
            k * torch.pi * xx
        ).view(1, 1, h, w)
    return x * field.clamp_min(0.0)


def _hoult_snr_scale(b0_t: float) -> float:
    """Hoult SNR-vs-B0 scaling: SNR ∝ B0^{7/4}, relative to a 3 T reference."""
    return (b0_t / 3.0) ** 1.75


def _random_fourier_shift(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Cyclically shift each sample by ``scale * N(0, I_2)`` pixels, sub-pixel and batched.

    The shift is a k-space phase ramp, so the output stays differentiable in ``scale``;
    an integer ``torch.roll`` would cut the graph.
    """
    b, _, h, w = x.shape
    dy, dx = (torch.randn(2, b, 1, 1, 1, device=x.device, dtype=x.dtype) * scale).unbind(0)
    fy = centred_freqs(h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
    fx = centred_freqs(w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
    ramp = torch.polar(torch.ones_like(fy * dy + fx * dx), -2.0 * torch.pi * (fy * dy + fx * dx))
    return ifft2c(fft2c(torch.complex(x, torch.zeros_like(x))) * ramp).real


@register_model(
    name="ulf_corruption_operator",
    training_mode="corruption_calibration",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=True,
    supports_contrast_conditioning=False,
)
class ULFCorruptionOperator(nn.Module, IGenerator):
    """ULF (~64 mT) corruption operator: Rician noise, motion, bias field, coil gain.

    Each calibratable scalar enters the output law through its own path and receives a
    gradient (#2395). ``reference_noise_sigma`` is the noise std on a [0, 1] image at
    ``reference_b0_t``; ``snr_scaling_law="hoult"`` carries it to ``b0_t`` by
    SNR ∝ B0^{7/4} and needs a reference field distinct from ``b0_t``, ``"learned"`` uses
    it as is; the defaults (3 T, 3.5725e-5) give a noise std of about 0.03 at 64 mT.
    ``log_noise_gain`` (per channel), ``motion_severity`` (shift std = 5·m px), ``recon_artifact_severity``
    (bias-field coefficient std = 0.1·|r|) and ``coil_bias`` (global gain 1 + c_b)
    are the free parameters.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        b0_t: float = 0.064,
        initial_motion_severity: float = 0.1,
        initial_recon_artifact_severity: float = 0.05,
        snr_scaling_law: str = "hoult",
        reference_noise_sigma: float = 3.5725e-5,
        reference_b0_t: float | None = 3.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if snr_scaling_law not in _SNR_SCALING_LAWS:
            raise ValueError(
                f"ULFCorruptionOperator: snr_scaling_law={snr_scaling_law!r} is not one of "
                f"{_SNR_SCALING_LAWS}."
            )
        if reference_noise_sigma <= 0.0:
            raise ValueError("ULFCorruptionOperator: reference_noise_sigma must be > 0.")
        if snr_scaling_law == "hoult" and (reference_b0_t is None or reference_b0_t == b0_t):
            raise ValueError(
                "ULFCorruptionOperator: snr_scaling_law='hoult' needs a reference_b0_t distinct "
                f"from b0_t={b0_t}; got {reference_b0_t!r}, which makes the law inert."
            )
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.b0_t = b0_t
        self.snr_scaling_law = snr_scaling_law
        self.reference_b0_t = b0_t if reference_b0_t is None else reference_b0_t
        field_factor = (
            _hoult_snr_scale(self.reference_b0_t) / _hoult_snr_scale(b0_t)
            if snr_scaling_law == "hoult"
            else 1.0
        )
        self.sigma_base = reference_noise_sigma * field_factor

        self.motion_severity = nn.Parameter(torch.tensor(initial_motion_severity))
        self.recon_artifact_severity = nn.Parameter(torch.tensor(initial_recon_artifact_severity))
        self.coil_bias = nn.Parameter(torch.tensor(0.0))
        # Log-gain, so the parameter has a gradient at its 0 init (|.| at 0 has none).
        self.log_noise_gain = nn.Parameter(torch.zeros(in_channels))

    def noise_sigma(self) -> torch.Tensor:
        """Per-channel Gaussian noise std of the Rician stage, shape ``(C,)``."""
        return self.sigma_base * self.log_noise_gain.exp()

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def name(self) -> str:
        return "ulf_corruption_operator"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        mag = x.abs() if x.is_complex() else x
        if mag.ndim != 4 or mag.shape[1] != self._in_channels:
            raise ValueError(
                f"ULFCorruptionOperator expects (B, {self._in_channels}, H, W); "
                f"got {tuple(mag.shape)}."
            )
        noisy = _rician_noise(mag, self.noise_sigma().view(1, -1, 1, 1))
        moved = _random_fourier_shift(noisy, self.motion_severity * 5.0)
        biased = _smooth_bias_field(moved, severity=self.recon_artifact_severity.abs())
        return biased * (1.0 + self.coil_bias)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)


@register_model(
    name="hf_corruption_operator",
    training_mode="corruption_calibration",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=True,
    supports_contrast_conditioning=False,
)
class HFCorruptionOperator(nn.Module, IGenerator):
    """HF (3 T) corruption operator: Rician noise, bias field, motion (3 free scalars)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        initial_rician_sigma: float = 0.02,
        initial_bias_field_severity: float = 0.05,
        initial_motion_severity: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.rician_sigma = nn.Parameter(torch.tensor(initial_rician_sigma))
        self.bias_field_severity = nn.Parameter(torch.tensor(initial_bias_field_severity))
        self.motion_severity = nn.Parameter(torch.tensor(initial_motion_severity))

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def name(self) -> str:
        return "hf_corruption_operator"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def noise_sigma(self) -> torch.Tensor:
        """Gaussian noise std of the Rician stage, shape ``(1,)``."""
        return self.rician_sigma.abs().view(1)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        mag = x.abs() if x.is_complex() else x
        noisy = _rician_noise(mag, self.rician_sigma.abs())
        biased = _smooth_bias_field(noisy, severity=self.bias_field_severity.abs())
        return _random_fourier_shift(biased, self.motion_severity * 3.0)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)


__all__ = ["HFCorruptionOperator", "ULFCorruptionOperator"]
