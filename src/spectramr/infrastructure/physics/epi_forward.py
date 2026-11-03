"""Susceptibility-aware EPI forward operator (fMRI §2).

Composes the existing FFT machinery with a displacement-induced
pixel-shift operator that models EPI geometric distortion. The
distortion field :math:`\\varphi : \\Omega \\to \\Omega` is
parameterised by the underlying B0 inhomogeneity map
:math:`\\Delta B_0(\\mathbf{x})` and the effective echo spacing
:math:`t_{\\text{esp}}`:

.. math::
    \\varphi(\\mathbf{x}) = \\mathbf{x} + \\gamma\\,t_{\\text{esp}}\\,
    \\Delta B_0(\\mathbf{x})\\,\\hat{e}_y.

The Beltrami coefficient :math:`\\mu_\\varphi` derived from this
displacement (§2 of the plan) is purely real and bounded by
``|μ_φ| ≤ k < 1`` whenever the EPI no-fold condition
``|γ t_esp ∂_y ΔB₀| < 1`` is satisfied.

References:
    [9]  J. L. R. Andersson et al., "How to correct susceptibility
         distortions in spin-echo echo-planar images", *NeuroImage*,
         20(2), 2003, 870-888.
    [10] P. Jezzard & R. S. Balaban, "Correction for geometric
         distortion in echo planar images from B0 field variations",
         *Magn. Reson. Med.*, 34(1), 1995, 65-73.
    [11] M. A. Bernstein, K. F. King, X. J. Zhou, *Handbook of MRI
         Pulse Sequences*, Elsevier, 2004.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

GAMMA_HZ_PER_T = 4.2576e7  # 1H gyromagnetic ratio (Hz/T)


def b0_to_displacement(
    delta_b0: torch.Tensor, *, t_esp: float, phase_encode_axis: int = -2
) -> torch.Tensor:
    """Convert a B0 inhomogeneity map (Hz) to a displacement field (pixels).

    The EPI phase-encode pixel shift is ``Δpix = ΔB0[Hz] · T_acq`` where the
    total echo-train traverse time ``T_acq = N_pe · t_esp`` (Jezzard & Balaban
    1995). Because ``ΔB0`` is already a *frequency* (Hz), there is **no**
    gyromagnetic factor — multiplying by ``γ`` (Hz/T) is dimensionally invalid
    (``Hz²·s/T``) and overshoots the true shift by ``γ ≈ 4.26e7``, which the
    downstream ``grid_sample`` clamp then saturates to the FOV edge (a
    degenerate, measurement-independent warp). ``N_pe`` is read from the
    field-map shape along ``phase_encode_axis``.

    Args:
        delta_b0: ``[B, 1, H, W]`` field map in Hz.
        t_esp: Effective echo spacing in seconds.
        phase_encode_axis: -2 (H) by default; -1 (W) for transverse PE.

    Returns:
        Displacement tensor with the same shape as ``delta_b0``, in pixels
        along the phase-encode direction.
    """
    n_pe = delta_b0.shape[phase_encode_axis]
    return t_esp * delta_b0 * float(n_pe)


def apply_epi_distortion(
    image: torch.Tensor,
    delta_b0: torch.Tensor,
    *,
    t_esp: float = 0.5e-3,
    phase_encode_axis: int = -2,
) -> torch.Tensor:
    """Simulate the EPI image of ``image`` under the field ``delta_b0``.

    With the phase-encode displacement ``d`` (pixels) sampled on the output
    grid, the observed intensity is ``image(y + d(y)) * (1 + d'(y))``: the
    resampled image times the Jacobian of the phase-encode map, so signal is
    conserved along each phase-encode line (Jezzard & Balaban 1995; FSL FUGUE
    ``--icorr``). Compressed bands brighten and stretched bands darken; the
    factor is clamped at zero where the field violates the no-fold condition.
    The read-out direction is unaffected (single-shot EPI).
    """
    if image.dim() != 4:
        raise ValueError("expected image of shape [B, C, H, W]")
    B, _, H, W = image.shape
    device, dtype = image.device, image.dtype
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    grid_y = Y.unsqueeze(0).expand(B, -1, -1).clone()
    grid_x = X.unsqueeze(0).expand(B, -1, -1).clone()
    # Pixel shift along PE, normalised to grid_sample's [-1, 1] (FOV spans 2.0)
    # by the PE-axis pixel count.
    n_pe = H if phase_encode_axis == -2 else W
    # align_corners=True: the n_pe pixel centres span the 2.0-wide grid in n_pe - 1 steps.
    disp_px = b0_to_displacement(delta_b0, t_esp=t_esp, phase_encode_axis=phase_encode_axis)
    disp = disp_px.squeeze(1) / float(n_pe - 1) * 2.0
    if phase_encode_axis == -2:
        grid_y = grid_y + disp
    else:
        grid_x = grid_x + disp
    grid = torch.stack([grid_x.clamp(-1, 1), grid_y.clamp(-1, 1)], dim=-1)
    warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
    jacobian = (1.0 + torch.gradient(disp_px, dim=phase_encode_axis)[0]).clamp_min(0.0)
    return warped * jacobian


def beltrami_from_b0(
    delta_b0: torch.Tensor, *, t_esp: float = 0.5e-3, phase_encode_axis: int = -2
) -> torch.Tensor:
    r"""Closed-form real Beltrami coefficient of the EPI distortion.

    For the one-axial EPI shear :math:`\varphi(x,y)=(x,\,y+d(y))` with
    phase-encode displacement :math:`d=N_{pe}\,t_\text{esp}\,\Delta B_0`
    (Hz\ :math:`\to`\ pixels), the Wirtinger derivatives give
    :math:`\partial_z\varphi=\tfrac12(2+s)`, :math:`\partial_{\bar z}\varphi=-\tfrac12 s`
    with :math:`s=\partial_y d`, so

    .. math::
        \mu_\varphi = \frac{\partial_{\bar z}\varphi}{\partial_z\varphi}
        = \frac{-s}{2+s} = \frac{1-J}{1+J},\qquad
        J = 1 + N_{pe}\,t_\text{esp}\,\partial_y \Delta B_0.

    There is **no** square root: an earlier form used :math:`\sqrt{J}` (an
    isotropic-dilation map, not the one-axial EPI shear), which halved the
    coefficient (:math:`\mu\approx-s/4` instead of :math:`-s/2`) and thus
    under-corrected the warp. This closed form now agrees with the general
    Wirtinger construction :func:`beltrami_from_b0_and_gnl` on a pure-B0 field
    (in the 1-D limit where the cross-derivative :math:`\partial_x d` vanishes).

    The displacement scale is ``N_pe`` (the phase-encode line count), **not**
    the gyromagnetic ratio ``γ`` — ``ΔB0`` is already in Hz.

    Returns a *real-valued* tensor since EPI distortion is one-axial.
    Caller can convert to complex via ``torch.complex(mu, torch.zeros_like(mu))``.
    """
    if delta_b0.dim() != 4:
        raise ValueError("expected delta_b0 of shape [B, 1, H, W]")
    n_pe = delta_b0.shape[phase_encode_axis]
    # Derivative along phase-encode direction.
    if phase_encode_axis == -2:
        dB = delta_b0[..., 1:, :] - delta_b0[..., :-1, :]
        dB = F.pad(dB, (0, 0, 0, 1), mode="replicate")
    else:
        dB = delta_b0[..., :, 1:] - delta_b0[..., :, :-1]
        dB = F.pad(dB, (0, 1, 0, 0), mode="replicate")
    # J = 1 + s (the PE-axis Jacobian); mu = (1 - J) / (1 + J) = -s / (2 + s).
    # clamp_min keeps J > 0 so 1 + J > 0 (no division by zero) and |mu| < 1.
    j = (1.0 + float(n_pe) * t_esp * dB).clamp_min(1e-6)
    return (1.0 - j) / (1.0 + j)


def beltrami_from_b0_and_gnl(
    delta_b0: torch.Tensor,
    gnl_displacement: torch.Tensor,
    *,
    t_esp: float = 0.5e-3,
    phase_encode_axis: int = -2,
    k_max: float = 0.9,
) -> torch.Tensor:
    r"""Unified COMPLEX Beltrami coefficient of the combined B0 (EPI) + gradient-nonlinearity warp.

    UBGC (A-3.1) unifies the two geometric-distortion sources of low-field MRI — B0 off-resonance
    (which shifts pixels along the phase-encode axis, EPI distortion) and gradient nonlinearity
    (which warps pixels in BOTH axes) — into a SINGLE quasiconformal map :math:`\varphi(z)=z+d(z)`
    with complex displacement :math:`d=d_x+i d_y`, and returns its Beltrami coefficient

    .. math::
        \mu = \frac{\partial_{\bar z} d}{1 + \partial_z d},\qquad
        \partial_z=\tfrac12(\partial_x-i\partial_y),\ \partial_{\bar z}=\tfrac12(\partial_x+i\partial_y),

    clamped to :math:`|\mu|<k_{\max}` (quasiconformality). The result is a complex ``[B, H, W]``
    tensor (NOTE: complex, vs the ``[B, 1, H, W]`` REAL output of :func:`beltrami_from_b0`) ready for
    :class:`~spectramr.infrastructure.physics.conformal_geometry.LinearBeltramiSolver` to recover and
    invert the unified warp — a single correction for both sources, vs the separate B0-only
    :func:`beltrami_from_b0` / GNL handling.

    This is the GENERAL quasiconformal formulation (complex Wirtinger derivatives of the 2-axis
    displacement). For a pure-B0 field varying only along PE it **agrees** with the specialised
    one-axial closed form :func:`beltrami_from_b0` (both give ``mu = -s/(2+s)`` in the 1-D limit
    where the cross-derivative ``∂_x d`` vanishes; they differ only by the finite-difference stencil
    and its real-vs-complex return type). It captures the GNL warp when ``delta_b0`` is 0, and returns
    ``mu=0`` (the identity map) when both distortions vanish. Validated against the analytic case
    ``d = c·conj(z) -> mu = c`` (a known affine warp).

    Args:
        delta_b0: ``[B, 1, H, W]`` off-resonance in Hz.
        gnl_displacement: ``[B, 2, H, W]`` gradient-nonlinearity displacement ``(d_x, d_y)`` in pixels.
        t_esp: EPI echo spacing (s). phase_encode_axis: -2 (rows/y) or -1 (cols/x). k_max: |mu| clamp.
    """
    if delta_b0.dim() != 4 or delta_b0.shape[1] != 1:
        raise ValueError("expected delta_b0 of shape [B, 1, H, W]")
    if gnl_displacement.dim() != 4 or gnl_displacement.shape[1] != 2:
        raise ValueError("expected gnl_displacement of shape [B, 2, H, W] (d_x, d_y in pixels)")
    n_pe = delta_b0.shape[phase_encode_axis]
    d_b0 = float(n_pe) * t_esp * delta_b0  # EPI pixel displacement along PE (Hz->pixels scale)
    dx_gnl = gnl_displacement[:, 0:1]
    dy_gnl = gnl_displacement[:, 1:2]
    if phase_encode_axis == -2:  # PE = y (rows)
        d_x, d_y = dx_gnl, d_b0 + dy_gnl
    else:  # PE = x (cols)
        d_x, d_y = d_b0 + dx_gnl, dy_gnl

    def _dpx(f: torch.Tensor) -> torch.Tensor:  # central difference along W (x)
        g = (f[..., :, 2:] - f[..., :, :-2]) * 0.5
        return F.pad(g, (1, 1, 0, 0), mode="replicate")

    def _dpy(f: torch.Tensor) -> torch.Tensor:  # central difference along H (y)
        g = (f[..., 2:, :] - f[..., :-2, :]) * 0.5
        return F.pad(g, (0, 0, 1, 1), mode="replicate")

    # Complex partials of d = d_x + i d_y (operate on real components, then recombine).
    dxd = torch.complex(_dpx(d_x), _dpx(d_y)).squeeze(1)  # ∂_x d, [B,H,W]
    dyd = torch.complex(_dpy(d_x), _dpy(d_y)).squeeze(1)  # ∂_y d
    dz = 0.5 * (dxd - 1j * dyd)
    dzbar = 0.5 * (dxd + 1j * dyd)
    mu = dzbar / (1.0 + dz)
    scale = torch.clamp(k_max / (mu.abs() + 1e-8), max=1.0)  # enforce |mu| < k_max
    return mu * scale


__all__ = [
    "GAMMA_HZ_PER_T",
    "apply_epi_distortion",
    "b0_to_displacement",
    "beltrami_from_b0",
    "beltrami_from_b0_and_gnl",
]
