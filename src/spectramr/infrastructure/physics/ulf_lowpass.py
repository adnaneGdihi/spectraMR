"""The ULF image-domain forward operator: a field-dependent low-pass in the fft2c domain.

``A x = ifft2c(H . fft2c(x))`` with a real Gaussian OTF ``H`` centred on the fft2c DC
bin. Shared by the ULF MAP solve, the cold field degradation, the DPS/TTA strategies
and the MoDL ULF generator, so the operator and its closed-form prox have one owner.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

#: Cache OTF tensors by (h, w, cutoff, device, dtype). The build does an extra
#: fft2c to locate the DC bin; field_degrade/ulf_map_solve call this per step, so
#: memoising keeps the hot path allocation-free (the OTF is a read-only constant).
_OTF_CACHE: dict[tuple, torch.Tensor] = {}


def _fft2c_dc_index(h: int, w: int, device: torch.device) -> tuple[int, int]:
    """Locate the fft2c DC bin EMPIRICALLY (do not hand-derive fftshift parity).

    fft2c's centering is NOT the standard fftshift on ODD axes (it applies an
    odd-axis roll — see the physics audit), so ``h // 2`` is wrong for odd sizes
    (e.g. fft2c DC for H=7 is index 4, not 3). The DC of a constant image is its
    only nonzero centered coefficient, so its argmax is the true DC bin for both
    parities.
    """
    spec = fft2c(torch.ones(1, 1, h, w, device=device)).abs()[0, 0]
    flat = int(spec.argmax())
    return flat // w, flat % w


def _lowpass_otf(
    h: int, w: int, cutoff: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Centered real Gaussian low-pass OTF on the fft2c (centered) grid.

    ``H`` is 1 at the fft2c DC bin and decays radially; ``cutoff`` is the
    half-width in normalised frequency (smaller cutoff ⇒ more blur ⇒ lower
    effective resolution). The Gaussian is centred on the EMPIRICAL DC bin so the
    OTF is conjugate-symmetric about it. For EVEN sizes (the data case — MNI 0.5mm
    is even, e.g. 256) ``ifft2c(H·fft2c(real)).imag`` is exactly 0; for odd sizes a
    tiny residual remains (fft2c's odd-axis roll is not a clean symmetric layout),
    still ~70x smaller than the previous bug. The previous ``linspace(-1,1,h)`` grid
    placed the peak at index ``h//2-1`` for even ``h`` and missed the odd-axis roll
    entirely, discarding up to ~90% imaginary energy via ``.real``. Defining ``A``
    directly through ``H`` keeps the forward operator and closed-form inverse
    self-consistent.
    """
    key = (h, w, round(float(cutoff), 6), str(device), str(dtype))
    cached = _OTF_CACHE.get(key)
    if cached is not None:
        return cached
    dc_y, dc_x = _fft2c_dc_index(h, w, device)
    yy = (torch.arange(h, device=device, dtype=dtype) - dc_y) / max(h / 2.0, 1.0)
    xx = (torch.arange(w, device=device, dtype=dtype) - dc_x) / max(w / 2.0, 1.0)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    d2 = grid_x * grid_x + grid_y * grid_y
    cut = max(float(cutoff), 1e-3)
    h_otf = torch.exp(-d2 / (2.0 * cut * cut)).view(1, 1, h, w)
    _OTF_CACHE[key] = h_otf
    return h_otf


def ulf_degrade(x: torch.Tensor, blur_cutoff: float) -> torch.Tensor:
    """Apply the field-dependent low-pass forward operator ``A x`` (real output)."""
    h_otf = _lowpass_otf(x.shape[-2], x.shape[-1], blur_cutoff, x.device, torch.float32)
    return ifft2c(h_otf * fft2c(x)).real


def lowpass_dc_prox(
    y_k: torch.Tensor,
    z: torch.Tensor,
    h_otf: torch.Tensor,
    data_weight: float | torch.Tensor,
    prior_weight: float | torch.Tensor,
) -> torch.Tensor:
    """``argmin_x w/2 ||y - A x||^2 + rho/2 ||x - z||^2`` for ``A = ifft2c . H . fft2c``.

    Closed form in the unitary fft2c domain with ``H`` real:
    ``X = (w H Y + rho F(z)) / (w H^2 + rho)``, ``x = ifft2c(X).real``. ``y_k`` is
    ``fft2c(y)``, passed in so an unrolled caller transforms the measurement once.
    """
    x_k = (data_weight * h_otf * y_k + prior_weight * fft2c(z)) / (
        data_weight * h_otf * h_otf + prior_weight
    )
    return ifft2c(x_k).real


__all__ = ["lowpass_dc_prox", "ulf_degrade"]
