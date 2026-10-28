r"""Separate magnitude and phase fidelity for predicted k-space.

A pointwise complex loss shrinks the coefficients whose phase the input does
not determine. For a target :math:`t = |t|e^{i\varphi}` with :math:`\varphi`
uniform and independent of :math:`|t|`, :math:`\mathbb E[t] = 0`, so

.. math::

    \arg\min_p \mathbb E\,|p - t|^2 = \mathbb E[t] = 0 ,

and the L1 minimiser on real and imaginary parts is the per-component median,
also 0. The unobserved high-frequency bins are exactly where the phase is least
determined, so complex L1/L2 regression attenuates mid and high frequencies.
A loss on the magnitude alone has the minimiser

.. math::

    \arg\min_{|p|} \mathbb E\,\bigl||p| - |t|\bigr| = \operatorname{median}|t| > 0 ,

so it asks for the correct spectral energy whether or not the phase is known.
The phase is scored separately, weighted by magnitude where it is determined:

.. math::

    \mathcal L = \frac{1}{|\Omega|}\sum_{k\in\Omega}\bigl||\hat k| - |k|\bigr|
      + \lambda_\varphi \frac{\sum_{k\in\Omega} \bigl(|\hat k||k|
            - \operatorname{Re}(\bar{\hat k} k)\bigr)}{\sum_{k\in\Omega} |k|^2}.

The numerator is :math:`|\hat k||k|(1 - \cos\Delta\varphi_k)`, so at
:math:`|\hat k| = |k|` the term is the :math:`|k|^2`-weighted mean of
:math:`1 - \cos\Delta\varphi`. Its gradient in :math:`\hat k` is
:math:`|k|\hat k/|\hat k| - k`, bounded by :math:`2|k|`; the form
:math:`|k|(1 - \cos\Delta\varphi)` has a gradient of order :math:`|k|/|\hat k|`,
which is unbounded on the near-zero predictions of unobserved bins and would
dominate gradient clipping. :math:`\Omega` is the unobserved support
:math:`1 - M` when a mask is supplied, else every bin. Neither form needs
``torch.angle``, whose gradient is undefined at zero. The arm's ``log1p`` compression maps :math:`|k|` monotonically
and preserves phase, so both terms are computed on the tensors as given; on
compressed k-space the magnitude term is a log-spectral distance that the radial
power law does not dominate.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from spectramr.models.losses.registry import register_loss

__all__ = ["KSpaceMagnitudePhaseLoss"]


def _as_complex(x: Tensor) -> Tensor:
    """Interleaved ``[B, 2C, H, W]`` (``R0, I0, R1, I1, ...``) or complex -> complex64.

    Computed in float32: ``eps ** 2`` underflows in fp16 and ``torch.complex``
    rejects bf16.
    """
    if torch.is_complex(x):
        return x.to(torch.complex64)
    x = x.float()
    if x.dim() != 4 or x.shape[1] % 2 != 0:
        raise ValueError(
            "kspace_magnitude_phase needs complex or real/imag-interleaved k-space "
            f"[B, 2C, H, W], got {tuple(x.shape)}."
        )
    b, c2, h, w = x.shape
    paired = x.view(b, c2 // 2, 2, h, w)
    return torch.complex(paired[:, :, 0], paired[:, :, 1])


@register_loss(
    name="kspace_magnitude_phase",
    aliases=["magnitude_phase_kspace"],
    domain="kspace",
)
class KSpaceMagnitudePhaseLoss(nn.Module):
    """Magnitude L1 plus magnitude-weighted phase cosine distance.

    Args:
        phase_weight: :math:`\\lambda_\\varphi`, the weight of the phase term.
        region: ``"unobserved"`` scores ``1 - mask`` (all bins when no mask is
            passed); ``"all"`` ignores the mask.
        eps: Floor on magnitudes and normalisers.
    """

    _REGIONS = ("unobserved", "all")

    def __init__(
        self, phase_weight: float = 0.5, region: str = "unobserved", eps: float = 1e-6
    ) -> None:
        super().__init__()
        if region not in self._REGIONS:
            raise ValueError(
                f"Unknown region {region!r} for kspace_magnitude_phase; valid: {self._REGIONS}"
            )
        if float(phase_weight) < 0.0:
            raise ValueError(f"phase_weight must be >= 0, got {phase_weight!r}")
        self.phase_weight = float(phase_weight)
        self.region = region
        self.eps = float(eps)

    def _region(self, ref: Tensor, mask: Tensor | None) -> Tensor:
        if self.region == "all" or mask is None:
            return torch.ones(1, 1, *ref.shape[-2:], device=ref.device, dtype=ref.dtype)
        m = mask.real if torch.is_complex(mask) else mask
        if m.dim() == 2:
            m = m[None, None]
        elif m.dim() == 3:
            m = m[:, None]  # [B, H, W]: the leading axis is the batch, not a coil
        # Any coil observed -> observed: the PE mask is shared across coils.
        observed = (m > 0).any(dim=1, keepdim=True)
        return (~observed).to(ref.dtype)

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
        """Score ``pred`` against ``target`` on the configured region."""
        p = _as_complex(pred)
        t = _as_complex(target).to(p.dtype)
        if p.shape != t.shape:
            raise ValueError(f"pred {tuple(p.shape)} and target {tuple(t.shape)} disagree")
        p_mag = torch.sqrt(p.real.square() + p.imag.square() + self.eps**2)
        t_mag = torch.sqrt(t.real.square() + t.imag.square() + self.eps**2)
        region = self._region(t_mag, mask).expand_as(t_mag)
        count = region.sum().clamp_min(1.0)

        magnitude = ((p_mag - t_mag).abs() * region).sum() / count
        if self.phase_weight == 0.0:
            return magnitude

        misalignment = p_mag * t_mag - (p.real * t.real + p.imag * t.imag)
        normaliser = (t_mag.detach().square() * region).sum().clamp_min(self.eps)
        phase = (misalignment * region).sum() / normaliser
        return magnitude + self.phase_weight * phase
