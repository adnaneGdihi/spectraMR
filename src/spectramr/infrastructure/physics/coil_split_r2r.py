r"""Null-space R2R: split ONE multi-coil acquisition into two independent halves.

R2R (``m4raw_noise.py``) splits a repetition by adding and subtracting a drawn
:math:`z\sim\mathcal{CN}(0,\Sigma_z)`, exact only where :math:`\Sigma_z` equals
the scan's own :math:`\Sigma_n`. This splits it with no draw at all, using the
scan's own coil null space.

**The construction.** At pixel :math:`r` the coil vector is
:math:`y=\hat s\,m+n`, :math:`n\sim\mathcal{CN}(0,\Sigma)`, with
:math:`\hat s=s/\lVert s\rVert` the unit ESPIRiT direction. Whiten with
:math:`W=\hat\Sigma^{-1/2}`, let :math:`\tilde s=W\hat s/\lVert W\hat s\rVert`
and pick a unit :math:`v\perp\tilde s`. For :math:`\theta\in(0,90^\circ)`

.. math::

    u_1=\cos\theta\,\tilde s+\sin\theta\,v,\qquad
    u_2=\sin\theta\,\tilde s-\cos\theta\,v,\qquad
    m_1=\frac{u_1^H W y}{\cos\theta},\quad m_2=\frac{u_2^H W y}{\sin\theta}.

Both halves carry the clean component :math:`\lVert W\hat s\rVert m`, and with
:math:`\hat\Sigma=\Sigma` their noises are uncorrelated because
:math:`u_1\perp u_2` -- the R2R pair with :math:`\tau=\tan\theta`, so at
45 degrees each half carries twice the whitened-SENSE noise variance. The target
half is re-expanded as the rank-one :math:`y_2=\hat s\,m_2/\lVert W\hat s\rVert`,
the coil structure of the clean signal. The input half keeps, on top of
:math:`\hat s\,m_1/\lVert W\hat s\rVert`, the whitened noise orthogonal to both
:math:`\tilde s` and :math:`v`, :math:`W^{-1}r` with :math:`r=P^\perp_{\{\tilde
s,v\}}Wy`. Whitened noise is isotropic, so :math:`r` is independent of both
halves' scalar noise and the pair stays an R2R pair; what it buys is that a
training input carries :math:`C-1` of the :math:`C` noise directions a
validation repetition carries, instead of being rank-one across coils while
every validation input is full-rank.

**Whitening needs Sigma's shape, not its level.** The residual correlation
under a mis-specified whitener is

.. math::

    u_1^H\bigl(\hat\Sigma^{-1/2}\,\Sigma\,\hat\Sigma^{-1/2}-I\bigr)u_2,

first order in the shape error and exactly zero for a pure gain error
:math:`\Sigma=g^2\hat\Sigma`, which leaves :math:`g^2I`. A receiver-gain change
-- the worry behind the self-calibrated arm -- therefore costs nothing under
``committed`` whitening. ``none`` (:math:`W=I`) is exact only for white
:math:`\Sigma`; on M4Raw's coupled array it is the planted violation.

**One null direction per sample, not per pixel.** :math:`v` is a single
Gaussian vector drawn per slice from the global RNG and projected orthogonal to
:math:`\tilde s` pixel by pixel, so it varies as smoothly as the maps. Pixel-wise
independence alone is not enough: with ~24% of M4Raw's phase-encode columns
zero-filled, image-domain noise is correlated across neighbouring pixels, and
the cross-pixel covariance of the halves is proportional to
:math:`u_1(r)^Hu_2(r')`, which vanishes only while :math:`\tilde s` and
:math:`v` barely change between :math:`r` and :math:`r'`.

**Outside the ESPIRiT support** the maps are zero and there is no rank-one
model. Those pixels take the slice's coherent mean map direction: the clean
component there is zero in both halves, so any unit direction keeps them
independent, and a shared one keeps the choice deterministic.

The split acts in the image domain, so a half's k-space is masked back to the
acquired phase-encode support afterwards; the per-pixel mixing is smooth, so the
energy that removes is a thin band-edge leakage, not signal.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

__all__ = [
    "VALID_WHITENING",
    "PreparedSplit",
    "check_split_angle",
    "draw_null_direction",
    "null_residual",
    "prepare_coil_split",
    "split_coil_images",
    "split_halves",
    "split_kspace",
    "split_kspace_prepared",
    "split_prepared",
    "unit_directions",
    "whitener",
]

#: ``none`` uses W = I; ``committed`` whitens with the measured M4Raw matrix;
#: ``self_calibrated`` with the scan's own null-space fit.
VALID_WHITENING: frozenset[str] = frozenset({"none", "committed", "self_calibrated"})

_EPS = 1e-8


def check_split_angle(angle_deg: float) -> float:
    """Return ``angle_deg`` in radians; raise outside the open interval (0, 90)."""
    angle = float(angle_deg)
    if not 0.0 < angle < 90.0:
        raise ValueError(
            f"split angle {angle_deg!r} deg is outside (0, 90): at either end one half "
            "carries all of the signal's noise and the other divides by zero."
        )
    return math.radians(angle)


def whitener(covariance: torch.Tensor | None, num_coils: int) -> torch.Tensor:
    """``Sigma^{-1/2}`` as a ``(C, C)`` complex matrix; the identity for ``None``.

    Raises:
        ValueError: the covariance is not ``(C, C)`` Hermitian positive-definite.
    """
    if covariance is None:
        return torch.eye(num_coils, dtype=torch.complex128)
    cov = covariance.to(torch.complex128)
    if cov.shape != (num_coils, num_coils):
        raise ValueError(f"covariance {tuple(cov.shape)} does not describe {num_coils} coils.")
    cov = (cov + cov.conj().transpose(-2, -1)) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    if not bool((eigenvalues > 0).all()):
        raise ValueError(
            f"covariance is not positive-definite (eigenvalues {eigenvalues.tolist()}); "
            "it cannot whiten."
        )
    inv_sqrt = eigenvalues.rsqrt().to(torch.complex128)
    return (eigenvectors * inv_sqrt) @ eigenvectors.conj().transpose(-2, -1)


def unit_directions(maps: torch.Tensor) -> torch.Tensor:
    """Per-pixel unit coil direction ``s / ||s||`` from ``(B, C, H, W)`` maps.

    Pixels with no map (outside the ESPIRiT support) take the slice's coherent
    mean direction, or the uniform vector when the slice has no support at all.
    """
    norm = maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    inside = norm > _EPS
    mean = maps.sum(dim=(-2, -1), keepdim=True)
    mean_norm = mean.abs().pow(2).sum(1, keepdim=True).sqrt()
    uniform = torch.full_like(mean, 1.0 / math.sqrt(maps.shape[1]))
    fallback = torch.where(mean_norm > _EPS, mean / mean_norm.clamp(min=_EPS), uniform)
    return torch.where(inside, maps / norm.clamp(min=_EPS), fallback.expand_as(maps))


def draw_null_direction(
    s_tilde: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """A unit ``v`` orthogonal to ``s_tilde`` at every pixel, smooth across pixels.

    One circular Gaussian vector per slice, projected per pixel; see the module
    docstring for why it is not drawn per pixel.
    """
    shape = (s_tilde.shape[0], s_tilde.shape[1], 1, 1)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imag = torch.randn(shape, generator=generator, dtype=torch.float64)
    g = torch.complex(real, imag).to(device=s_tilde.device, dtype=s_tilde.dtype)
    v = g - s_tilde * (s_tilde.conj() * g).sum(1, keepdim=True)
    v_norm = v.abs().pow(2).sum(1, keepdim=True).sqrt()
    if bool((v_norm < 1e-6).any()):
        raise ValueError(
            "the drawn null direction is parallel to a coil direction at some pixel; "
            "redraw with a different generator state."
        )
    return v / v_norm


def split_halves(
    whitened: torch.Tensor, s_tilde: torch.Tensor, v: torch.Tensor, angle: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(m_1, m_2)``, each ``(B, H, W)``, from whitened coil images and the basis."""
    c, s = math.cos(angle), math.sin(angle)
    u1 = c * s_tilde + s * v
    u2 = s * s_tilde - c * v
    m1 = (u1.conj() * whitened).sum(1) / c
    m2 = (u2.conj() * whitened).sum(1) / s
    return m1, m2


def null_residual(whitened: torch.Tensor, s_tilde: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The component of ``whitened`` orthogonal to both ``s_tilde`` and ``v``."""
    along_s = s_tilde * (s_tilde.conj() * whitened).sum(1, keepdim=True)
    along_v = v * (v.conj() * whitened).sum(1, keepdim=True)
    return whitened - along_s - along_v


class PreparedSplit(NamedTuple):
    """The part of a split that does not depend on the null-direction draw."""

    whitened: torch.Tensor
    s_hat: torch.Tensor
    a_norm: torch.Tensor
    s_tilde: torch.Tensor
    w_inv: torch.Tensor
    dtype: torch.dtype


def prepare_coil_split(
    images: torch.Tensor, maps: torch.Tensor, *, whitening: torch.Tensor
) -> PreparedSplit:
    """Everything :func:`split_prepared` needs that the draw of ``v`` does not change.

    Recorrupted validation splits one slice J times; this is computed once for all J.
    """
    if images.shape != maps.shape:
        raise ValueError(f"images {tuple(images.shape)} and maps {tuple(maps.shape)} differ.")
    work = torch.complex128
    w = whitening.to(device=images.device, dtype=work)
    s_hat = unit_directions(maps.to(work))
    a = torch.einsum("ij,bjhw->bihw", w, s_hat)
    a_norm = a.abs().pow(2).sum(1, keepdim=True).sqrt()
    whitened = torch.einsum("ij,bjhw->bihw", w, images.to(work))
    return PreparedSplit(whitened, s_hat, a_norm, a / a_norm, torch.linalg.inv(w), images.dtype)


def split_prepared(
    prepared: PreparedSplit,
    *,
    angle_deg: float,
    generator: torch.Generator | None = None,
    first_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """:func:`split_coil_images` from a :class:`PreparedSplit`; one fresh ``v`` per call.

    ``first_only`` skips the target half's re-expansion and returns ``(y_1, None)``.
    """
    angle = check_split_angle(angle_deg)
    whitened, s_hat, a_norm, s_tilde, w_inv, dtype = prepared
    v = draw_null_direction(s_tilde, generator)
    m1, m2 = split_halves(whitened, s_tilde, v, angle)
    scale = s_hat / a_norm
    residual = torch.einsum("ij,bjhw->bihw", w_inv, null_residual(whitened, s_tilde, v))
    first = (scale * m1.unsqueeze(1) + residual).to(dtype)
    if first_only:
        return first, None
    return first, (scale * m2.unsqueeze(1)).to(dtype)


def split_coil_images(
    images: torch.Tensor,
    maps: torch.Tensor,
    *,
    whitening: torch.Tensor,
    angle_deg: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``(B, C, H, W)`` complex coil images into two re-expanded halves.

    Args:
        images: image-domain coil images.
        maps: ESPIRiT maps, same shape.
        whitening: ``(C, C)`` whitener from :func:`whitener`.
        angle_deg: split angle; ``tan(angle)`` is the R2R ``tau``.
        generator: RNG for the null direction; ``None`` uses the global one.

    Returns:
        ``(y_1, y_2)``, each ``(B, C, H, W)`` in ``images.dtype``: the input
        half with its residual null-space noise, and the rank-one target half.
    """
    check_split_angle(angle_deg)
    prepared = prepare_coil_split(images, maps, whitening=whitening)
    first, second = split_prepared(prepared, angle_deg=angle_deg, generator=generator)
    assert second is not None
    return first, second


def split_kspace(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    support: torch.Tensor,
    *,
    whitening: torch.Tensor,
    angle_deg: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`split_coil_images` from and back to k-space, masked to ``support``.

    Args:
        kspace: ``(B, C, H, W)`` complex coil k-space, phase-encode axis last.
        maps: ``(B, C, H, W)`` complex ESPIRiT maps.
        support: 1-D bool over the phase-encode columns that were acquired.
    """
    y1, y2 = split_coil_images(
        ifft2c(kspace), maps, whitening=whitening, angle_deg=angle_deg, generator=generator
    )
    mask = support.to(device=kspace.device)
    return fft2c(y1) * mask, fft2c(y2) * mask


def split_kspace_prepared(
    prepared: PreparedSplit,
    support: torch.Tensor,
    *,
    angle_deg: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """The input half of :func:`split_kspace` from a :class:`PreparedSplit` (``ifft2c`` of the k-space)."""
    first, _ = split_prepared(prepared, angle_deg=angle_deg, generator=generator, first_only=True)
    return fft2c(first) * support.to(device=first.device)
