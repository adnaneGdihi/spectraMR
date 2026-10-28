r"""SSDU's tensor operations, and the pieces that make Robust SSDU unbiased under noise.

``split_acquired_mask`` draws Lambda/Theta, ``inject_noisier_kspace`` corrupts the
measurement, ``image_of``/``kspace_of`` are the operator pair the strategy hands
the network and the loss. The rest of this docstring is Robust SSDU.

SSDU trains :math:`f` on :math:`M_\Lambda y` and scores it on the held-out lines
:math:`\Theta=\Omega\setminus\Lambda`. When the measurement :math:`y=M_\Omega(y_0+n)`
is noisy, that recovers the missing lines but never denoises the acquired ones:
the minimiser on :math:`\Theta` is :math:`\mathbb E[y_0+n\mid\cdot]` restricted to
lines whose noise the input does not carry, and the lines the input *does* carry
are passed through with their noise (arXiv:2210.01696, Eq. 18).

Robust SSDU feeds the network a *noisier*, further-subsampled measurement

.. math::  \tilde y = M_{\Lambda\cap\Omega}(y+\tilde n),\qquad
           \tilde n\sim\mathcal{CN}(0,\alpha^2\sigma_n^2 I)       \quad\text{(Eq. 25)}

and scores it on **all** of :math:`\Omega` (Eq. 26), so the population minimiser is
:math:`f^\star(\tilde y)=\mathbb E[y_0+n\mid\tilde y]` everywhere (Eq. 28). On a line
of :math:`\Lambda` the target shares :math:`n` with the input, and for Gaussian
noise :math:`\mathbb E[n\mid\cdot]=\mathbb E[\tilde n\mid\cdot]/\alpha^2`, which gives
the Noisier2Noise correction

.. math::  \hat y_0 = M_{\Lambda\cap\Omega}\,\frac{(1+\alpha^2)f(\tilde y)-\tilde y}{\alpha^2}
           + M_{(\Lambda\cap\Omega)^c}\,f(\tilde y)                  \quad\text{(Eq. 29)}.

At inference the uncorrupted :math:`y` is fed and the correction applied on
:math:`\Omega` (Eq. 36), which the paper notes departs from the conditional-mean
argument but performs better. The weighted loss (Eq. 35) scales the
:math:`\Lambda` residual by :math:`(1+\alpha^2)/\alpha^2`; the paper's
:math:`\mathcal P^{1/2}` factor on :math:`\Omega\setminus\Lambda` is omitted, so Claim 3's
gradient equivalence does not hold here (:math:`\Omega` is drawn per slice, so :math:`P`
is not 1), and :math:`\Lambda` may take ACS points. The derivation is in
``experiments/inprogress/self_supervised/PAPER.md``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from spectramr.core.metrics.coil_manifold import as_interleaved
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c, sense_adjoint, sense_forward
from spectramr.infrastructure.physics.multicoil_ei_operator import interleaved_to_complex

__all__ = [
    "image_of",
    "inject_noisier_kspace",
    "kspace_of",
    "noisier2noise_correct",
    "robust_loss_weights",
    "split_acquired_mask",
]


def split_acquired_mask(
    acquired_mask: torch.Tensor,
    theta_fraction: float = 0.4,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an acquired-sample mask into disjoint Lambda (input) and Theta (loss) masks.

    Per sample, exactly ``round(theta_fraction * n_acquired)`` acquired points move
    into Theta, chosen by a random key's rank, so the fraction holds for any
    sampling pattern. The count stays on the device (a double ``argsort`` instead
    of ``topk`` with a host integer), so the split costs no synchronisation.

    Args:
        acquired_mask: real tensor, 1 where k-space was sampled, ``(B, 1, H, W)``,
            ``(B, H, W)`` or ``(H, W)``.
        theta_fraction: fraction of acquired points held out into Theta.
        generator: optional CPU RNG for reproducible splits.

    Returns:
        ``(lambda_mask, theta_mask)`` with ``lambda + theta == acquired`` and
        ``lambda * theta == 0``.
    """
    if not 0.0 < theta_fraction < 1.0:
        raise ValueError(f"theta_fraction must be in (0, 1), got {theta_fraction}")
    acq = (acquired_mask > 0).to(acquired_mask.dtype)
    flat = acq.reshape(acq.shape[0] if acq.dim() >= 3 else 1, -1)
    key = torch.rand(flat.shape, generator=generator).to(acq.device)
    key = torch.where(flat > 0, key, torch.full_like(key, float("inf")))
    rank = key.argsort(dim=1).argsort(dim=1)
    n_theta = torch.round(theta_fraction * flat.sum(dim=1, keepdim=True))
    theta = ((rank < n_theta) & (flat > 0)).to(acq.dtype).reshape(acq.shape)
    return acq - theta, theta


def inject_noisier_kspace(
    target_kspace: torch.Tensor,
    acquired_mask: torch.Tensor,
    noise_std: float | torch.Tensor,
    generator: torch.Generator | None = None,
    coil_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add complex-Gaussian noise of magnitude std ``noise_std`` on the acquired support.

    ``target_kspace`` is complex ``[B, C, H, W]`` or real/imag-interleaved
    ``[B, 2C, H, W]`` and the result keeps its layout. The channels are paired
    into complex coils first: drawing an "imaginary" part for each real channel
    turned an 8-channel batch into 8 complex coils (#2327). ``noise_std`` may be a
    tensor (the per-slice self-calibrated sigma, ``(B, 1, 1, 1)``), so no value leaves the device.
    A plain ``0`` returns the input unchanged, which is vanilla SSDU.

    ``coil_factor`` is an optional ``(C, C)`` complex factor ``L`` of the coil
    covariance shape (``L L^H``, trace ``C``); the white draw is coloured by it
    so the injected noise has the measurement's coil correlation.
    """
    if isinstance(noise_std, (int, float)) and noise_std <= 0.0:
        return target_kspace
    interleaved = not torch.is_complex(target_kspace)
    y = interleaved_to_complex(target_kspace)
    real_dtype = y.real.dtype
    real = torch.randn(y.shape, generator=generator).to(y.device, real_dtype)
    imag = torch.randn(y.shape, generator=generator).to(y.device, real_dtype)
    std = torch.as_tensor(noise_std, dtype=real_dtype, device=y.device)
    acq = (acquired_mask > 0).to(real_dtype)
    draw = torch.complex(real, imag)
    if coil_factor is not None:
        if coil_factor.shape != (y.shape[1], y.shape[1]):
            raise ValueError(
                f"coil_factor {tuple(coil_factor.shape)} does not match the "
                f"{y.shape[1]} coils of the measurement."
            )
        draw = torch.einsum("ij,bj...->bi...", coil_factor.to(y.device, y.dtype), draw)
    noisy = y + (std / math.sqrt(2.0)) * draw * acq
    return as_interleaved(noisy) if interleaved else noisy


def kspace_of(image: torch.Tensor, maps: torch.Tensor | None) -> torch.Tensor:
    """``F S x`` (``F x`` for one coil) of a real/imag-interleaved image."""
    x = interleaved_to_complex(image)
    return sense_forward(x, maps, None) if maps is not None else fft2c(x)


def context_metric_kspace(image: torch.Tensor, maps: torch.Tensor | None) -> torch.Tensor:
    """``F S x`` real/imag-interleaved: an SSDU image as the coil k-space Rep-BV grades (#2509)."""
    return as_interleaved(kspace_of(image, maps))


def image_of(kspace: torch.Tensor, maps: torch.Tensor | None, mask: Any) -> torch.Tensor:
    """``S^H F^H M k`` (``F^H M k`` for one coil), real/imag-interleaved."""
    if maps is not None:
        return as_interleaved(sense_adjoint(kspace, maps, mask))
    return as_interleaved(ifft2c(kspace if mask is None else kspace * mask))


def _real(mask: torch.Tensor) -> torch.Tensor:
    return mask.real if torch.is_complex(mask) else mask


def robust_loss_weights(
    input_mask: torch.Tensor, acquired: torch.Tensor, alpha: float
) -> torch.Tensor:
    r"""Eq. 35's :math:`W`: :math:`(1+\alpha^2)/\alpha^2` on :math:`\Lambda`, 1 on
    :math:`\Omega\setminus\Lambda`, 0 off :math:`\Omega`. It multiplies the residual
    inside the norm, so the squared residual on :math:`\Lambda` carries its square."""
    if alpha <= 0.0:
        raise ValueError(f"Robust SSDU needs alpha > 0, got {alpha}.")
    lam = (_real(input_mask) > 0).to(torch.float32)
    omega = (_real(acquired) > 0).to(torch.float32)
    return omega * (1.0 + lam * ((1.0 + alpha * alpha) / (alpha * alpha) - 1.0))


def noisier2noise_correct(
    predicted: torch.Tensor,
    network_input: torch.Tensor,
    correction_mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    r"""Eq. 29 / 36: ``((1+a^2) f - y~) / a^2`` on the lines the input carried.

    Args:
        predicted: :math:`f(\tilde y)` as k-space, complex ``[B, C, H, W]``.
        network_input: the k-space the network was fed, same shape (zero off
            ``correction_mask``).
        correction_mask: the lines whose noise the input carries -- :math:`\Lambda`
            at training, :math:`\Omega` at inference -- real, broadcastable.
        alpha: the noise ratio the input was corrupted with at training.
    """
    if alpha <= 0.0:
        raise ValueError(f"the Noisier2Noise correction needs alpha > 0, got {alpha}.")
    scale = 1.0 + alpha * alpha
    corrected = (scale * predicted - network_input) / (alpha * alpha)
    mask = (_real(correction_mask) > 0).to(predicted.real.dtype)
    return mask * corrected + (1.0 - mask) * predicted
