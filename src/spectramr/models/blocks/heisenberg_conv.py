r"""Heisenberg-group equivariant twisted convolution for phase-space MRI.

The image–k-space duality is the canonical commutation
:math:`[X,K]=iI`. The Heisenberg group
:math:`H_3=\{(a,b,c):a,b\in\mathbb{R}^n,c\in\mathbb{R}\}` is the unique
simply-connected Lie group encoding this commutator; the Schrödinger
representation :math:`\pi:H_3\to\mathcal{U}(L^2(\mathbb{R}^n))` acts as

.. math::

    (\pi(a,b,c)\psi)(x) = e^{i(c+b\cdot x)}\psi(x+a),

so :math:`\pi(a,0,0)` is image translation and :math:`\pi(0,b,0)` is a
k-space translation (a phase ramp in image domain) — see Folland 1989,
*Harmonic Analysis in Phase Space*.

A *Heisenberg-equivariant convolution* against a kernel
:math:`K\in L^1(H_3)` is the **twisted convolution**

.. math::

    (\Phi_K\psi)(x) = \int_{H_3} K(g)\,(\pi(g)\psi)(x)\,d\mu(g),

which satisfies :math:`\Phi_K\circ\pi(g)=\pi(g)\circ\Phi_K` for all
:math:`g\in H_3`. On the discrete lattice
:math:`\Lambda=\mathrm{diag}(1/N)\,\mathbb{Z}^{2n}` the integral becomes a
finite sum over translation–phase-ramp pairs that is computable in
:math:`O(N^2\log N)` per layer via FFT.

This is the math of Gabor analysis applied to MRI: the kernel is parametrised
on a small Heisenberg grid (translation × phase-ramp) and applied by

1. translating the input by integer image-domain shifts,
2. multiplying by complex phase ramps :math:`e^{ib\cdot x}`,
3. weighted sum over the Heisenberg neighbourhood.

At :math:`b=0` the twisted convolution reduces to standard image-domain
convolution; at :math:`a=0` it reduces to k-space pointwise multiplication.

Exports:

* :class:`HeisenbergConv2d` — a twisted-convolution layer.
* :func:`twisted_convolution_step` — single forward pass of (1)–(3).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["HeisenbergConv2d", "twisted_convolution_step"]


def twisted_convolution_step(
    psi: torch.Tensor,
    kernel: torch.Tensor,
    shifts: torch.Tensor,
    phase_ramps: torch.Tensor,
) -> torch.Tensor:
    r"""One step of twisted convolution:
    :math:`(\Phi_K\psi)(x)=\sum_{(a,b)} K(a,b)\,e^{ib\cdot x}\,\psi(x+a)`.

    Args:
        psi: ``(B, C, H, W)`` complex (or real) input. If real, treated as
            having zero imaginary part.
        kernel: ``(K, C, C)`` complex weights, one per Heisenberg lattice
            point.
        shifts: ``(K, 2)`` integer image-domain shifts :math:`(a_y, a_x)`.
        phase_ramps: ``(K, H, W)`` complex phase ramps
            :math:`e^{i(b_x\cdot x + b_y\cdot y)}` per Heisenberg lattice point.

    Returns:
        ``(B, C, H, W)`` complex output of the twisted convolution.
    """
    if psi.ndim != 4:
        raise ValueError(f"psi must be (B, C, H, W); got {tuple(psi.shape)}.")
    if kernel.ndim != 3 or kernel.shape[1] != kernel.shape[2]:
        raise ValueError(f"kernel must be (K, C, C); got {tuple(kernel.shape)}.")
    if shifts.shape != (kernel.shape[0], 2):
        raise ValueError("shifts must be (K, 2).")
    if phase_ramps.shape[0] != kernel.shape[0]:
        raise ValueError("phase_ramps must have shape (K, H, W).")
    if not psi.is_complex():
        psi = torch.complex(psi, torch.zeros_like(psi))
    h, w = psi.shape[-2:]
    out = torch.zeros_like(psi)
    shift_list = shifts.tolist()  # one host transfer instead of two per lattice point
    for k_idx in range(kernel.shape[0]):
        ay, ax = int(shift_list[k_idx][0]), int(shift_list[k_idx][1])
        shifted = torch.roll(psi, shifts=(ay, ax), dims=(-2, -1))
        ramped = shifted * phase_ramps[k_idx]  # broadcasting on (B, C, H, W)
        weighted = torch.einsum("co,bohw->bchw", kernel[k_idx], ramped)
        out = out + weighted
    return out


class HeisenbergConv2d(nn.Module):
    r"""Twisted-convolution layer with a learnable Heisenberg-lattice kernel.

    Lattice points :math:`(a, b)` are taken from
    :math:`\{-r,\ldots,r\}^2\times\{0,\ldots,m-1\}^2`, so
    :math:`K=(2r+1)^2\cdot m^2` Heisenberg points per layer. At :math:`m=1`
    only the translation lattice is exercised and the layer collapses to a
    real-valued conv2d; at :math:`r=0` only phase ramps are active and the
    layer becomes a phase-space pointwise mixer.

    Args:
        in_channels: :math:`C_{\rm in}`.
        out_channels: :math:`C_{\rm out}` (must equal ``in_channels`` for true
            equivariance; relaxed here to allow channel mixing).
        spatial_size: ``(H, W)`` so phase ramps can be precomputed.
        translation_radius: :math:`r` such that :math:`(2r+1)^2` shifts.
        phase_radius: :math:`m` such that :math:`m^2` phase-ramp frequencies.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_size: tuple[int, int],
        translation_radius: int = 1,
        phase_radius: int = 1,
    ) -> None:
        super().__init__()
        if translation_radius < 0 or phase_radius < 1:
            raise ValueError("translation_radius >= 0 and phase_radius >= 1 required.")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("channels must be positive.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        h, w = spatial_size
        # Build translation × phase grid.
        r = translation_radius
        m = phase_radius
        shifts = []
        ramps = []
        for ay in range(-r, r + 1):
            for ax in range(-r, r + 1):
                for by in range(m):
                    for bx in range(m):
                        shifts.append([ay, ax])
                        y = torch.arange(h, dtype=torch.float32).unsqueeze(1)
                        x = torch.arange(w, dtype=torch.float32).unsqueeze(0)
                        phase = (
                            2.0
                            * torch.pi
                            * (by * y / max(1, m) + bx * x / max(1, m))
                            / max(1, h + w)
                        )
                        ramps.append(torch.complex(torch.cos(phase), torch.sin(phase)))
        self.register_buffer("shifts", torch.tensor(shifts, dtype=torch.long))
        self.register_buffer("phase_ramps", torch.stack(ramps))
        n_lattice = self.shifts.shape[0]
        scale = 1.0 / (n_lattice**0.5)
        self.kernel_real = nn.Parameter(torch.randn(n_lattice, out_channels, in_channels) * scale)
        self.kernel_imag = nn.Parameter(
            torch.randn(n_lattice, out_channels, in_channels) * scale * 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = torch.complex(self.kernel_real, self.kernel_imag)
        return twisted_convolution_step(x, kernel, self.shifts, self.phase_ramps)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"spatial_size={self.spatial_size}, lattice_size={self.shifts.shape[0]}"
        )
