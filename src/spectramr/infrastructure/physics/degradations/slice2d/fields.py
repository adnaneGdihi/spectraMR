"""2-D field kernels: B0 (D10, D10b), bias (D11), chemical shift (D16), susceptibility (D17),
coil miscalibration (D18), magic angle (D22)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from spectramr.infrastructure.physics.degradations._common import (
    _ensure_complex,
    _generator,
    _isolated_rng,
)
from spectramr.infrastructure.physics.digital_twin_simulator import (
    DEFAULT_CHI_PPM,
    chemical_shift_pixels,
    simulate_b0_geometric_distortion,
    simulate_b0_inhomogeneity,
    simulate_chemical_shift,
    simulate_magic_angle,
)
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fft2c, ifft2c
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand, per_sample_randn


def degrade_susceptibility(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_dropout_pct: float = 0.5,
    b0_axis: str = "y",
    slice_plane: str = "coronal",
) -> torch.Tensor:
    """D17 — Susceptibility-induced phase shift + magnitude dropout.

    Random low-frequency susceptibility map (chi) is convolved with
    a 2-D dipole kernel (B0(r) = chi * d_z(r)) to produce a ΔB0 map.
    The resulting phase modulates the image, and a magnitude dropout
    is applied where ΔB0 is large (proxy for through-slice dephasing).

    The 3-D Fourier-domain dipole is :math:`D(\\mathbf{k}) = k_z^2 /
    \\|\\mathbf{k}\\|^2 - 1/3` (Schenck 1996). For a 2-D slice we pick
    one in-plane projection; the choice depends on slice plane and
    scanner-frame :math:`B_0` axis:

    Both projections assume the object is uniform through-plane (``k_through = 0``):

    * ``slice_plane="axial"`` --- the slice is perpendicular to :math:`B_0`, so
      :math:`k_z = 0` everywhere in-plane and the kernel is the constant
      :math:`-1/3`: the field is :math:`-\\chi/3` locally, with no dipole
      structure. ``b0_axis`` is not used.
    * ``slice_plane="sagittal"`` or ``"coronal"`` --- the slice contains
      :math:`B_0`, along image axis ``b0_axis``, and the kernel is
      ``k_{b0_axis}^2 / k^2 - 1/3``.

    The default (``"coronal"``, ``b0_axis="y"``) is the kernel this axis has
    always computed. Through-plane susceptibility effects in an axial slice
    (e.g. at the sinuses) need the third dimension and are not modelled.
    """
    x = _ensure_complex(x)
    B, _, H, W = x.shape
    device = x.device
    g = _generator(device, seed)

    # Low-frequency chi map (random Gaussian then upsampled)
    chi_low = per_sample_randn(B, 1, H // 16 + 1, W // 16 + 1, generator=g, device=device)
    chi = F.interpolate(chi_low, size=(H, W), mode="bicubic", align_corners=False)

    # 2-D dipole projection — see docstring for the choice rationale.
    # The kernel must sit on the fft2c grid (DC at N // 2) to be Hermitian; on an even
    # linspace grid it was not, and ``.real`` below discarded a large imaginary part.
    ky = centred_freqs(H, device).view(-1, 1)
    kx = centred_freqs(W, device).view(1, -1)
    k2 = (kx**2 + ky**2).clamp(min=1e-12)
    if b0_axis not in ("x", "y"):
        raise ValueError(f"b0_axis must be 'x' or 'y' for 2-D, got {b0_axis!r}.")
    if slice_plane not in ("axial", "sagittal", "coronal"):
        raise ValueError(f"slice_plane must be 'axial'/'sagittal'/'coronal', got {slice_plane!r}.")
    if slice_plane == "axial":
        dipole = torch.full((H, W), -1.0 / 3.0, device=device)
    else:
        k_along_b0_sq = (ky if b0_axis == "y" else kx) ** 2
        dipole = k_along_b0_sq / k2 - 1.0 / 3.0
    # D(0) is undefined; a uniform field offset is absorbed by the receiver centre frequency.
    dipole[H // 2, W // 2] = 0.0
    dipole = dipole.view(1, 1, H, W)

    # Route through fft_ops so we get norm="ortho" + AMP-safe casting
    # (CLAUDE.md pitfall #2 + M1 in the digital-twin review). The chi
    # field is real, so we cast to complex once before the round-trip.
    chi_c = torch.complex(chi, torch.zeros_like(chi))
    chi_k = fft2c(chi_c)
    db0 = ifft2c(chi_k * dipole).real

    db0 = db0 / db0.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    phase = torch.exp(1j * 2.0 * math.pi * float(theta) * db0)

    # Magnitude dropout proportional to |db0|^2 (intra-voxel dephasing
    # near susceptibility hot-spots). Smooth quadratic shading produces
    # visible dimming over larger areas than the prior |db0|>0.6 binary
    # mask. Scale chosen so theta=1 yields up to ~80 % dropout at peaks.
    dropout = torch.exp(-8.0 * float(theta) * max_dropout_pct * db0.pow(2))
    keep = dropout.clamp(min=0.0, max=1.0)

    return x * phase * keep


def degrade_coil_sensitivity_miscal(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_relative_error: float = 1.0,
) -> torch.Tensor:
    """D18 — Coil-sensitivity miscalibration (single-coil proxy).

    Multiplies the image by a smooth ratio map ``Ŝ/S`` whose deviation
    from 1.0 grows with ``theta``. For multi-coil this would be the
    residual aliasing left after SENSE; here we emulate the effect on
    the combined image as a smooth low-frequency intensity error.
    """
    x = _ensure_complex(x)
    B, _, H, W = x.shape
    device = x.device
    g = _generator(device, seed)

    # Smooth low-frequency error (4 random modes).
    # Mod1: batch the 12 random scalars in a single on-device draw and
    # slice; the prior implementation forced 12 host syncs via .item().
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    # One map per subject; one row of draws per sample keeps sample b batch-invariant.
    params = per_sample_rand(B, 3, 4, 1, 1, generator=g, device=device)  # [B, fx/fy/ph, 4 modes]
    fxs, fys, phs = params[:, 0] * 3.0, params[:, 1] * 3.0, params[:, 2] * (2 * math.pi)
    err = torch.cos(fxs * math.pi * X + fys * math.pi * Y + phs).mean(dim=1, keepdim=True)
    ratio = 1.0 + float(theta) * max_relative_error * err
    return x * ratio


def degrade_bias_field(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    max_log_b: float = 1.2,
) -> torch.Tensor:
    """D11 — Multiplicative bias field, low-frequency log-Gaussian.

    Deterministic-seed version of :func:`simulate_b1_bias_field` that
    produces the same field for the same ``(theta, seed)``.

    Mod1 (digital-twin review): four scalar parameters are drawn in a
    single ``torch.rand`` call and sliced on device, eliminating four
    GPU→CPU host syncs per call.
    """
    x = _ensure_complex(x)
    B, _, H, W = x.shape
    device = x.device
    g = _generator(device, seed)

    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    # One field per subject, shared by its coils; one row of draws per sample.
    rand4 = per_sample_rand(B, 4, 1, 1, generator=g, device=device)
    fx, fy = rand4[:, 0] * 2.0, rand4[:, 1] * 2.0
    ph_x, ph_y = rand4[:, 2] * 2 * math.pi, rand4[:, 3] * 2 * math.pi

    log_b = (
        float(theta)
        * max_log_b
        * torch.cos(fx * math.pi * X + ph_x)
        * torch.cos(fy * math.pi * Y + ph_y)
    )
    bias = torch.exp(log_b).view(B, 1, H, W)
    return x * bias


def _delegate_b0(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        strength = float(kw.get("strength", float(theta)))
        out = ifft2c(simulate_b0_inhomogeneity(fft2c(x), x.shape[-2:], strength=strength))
    return out


def _delegate_b0_geometric(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        # dwell_time controls the geometric pixel-shift magnitude
        # (the displacement is proportional to dwell_time * Delta_B0).
        # Bumped 1e-5 -> 5e-5 so theta=1 produces a clearly visible
        # voxel shift at the FOV periphery (Halbach low-field regime).
        b0_strength = float(kw.get("b0_strength", float(theta)))
        dwell_time = float(kw.get("dwell_time", 5e-5))
        out = simulate_b0_geometric_distortion(
            x,
            x.shape[-2:],
            dwell_time=dwell_time,
            b0_strength=b0_strength,
            sequence=kw.get("sequence", "spin_warp"),
        )
    return out


def _delegate_chemical_shift(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    # Without a field and bandwidth the shift is a synthetic 0-14 px ramp. With both,
    # theta scales the physical fat shift at that B0 and bandwidth; forwarding them to
    # the primitive instead would replace the theta ramp and make the axis theta-invariant.
    has_b0, has_bw = "b0_T" in kw, "bw_per_pixel_hz" in kw
    if has_b0 != has_bw:
        raise ValueError("chemical_shift needs both b0_T and bw_per_pixel_hz, or neither.")
    if "shift_pixels" in kw and has_b0:
        raise ValueError("chemical_shift takes shift_pixels or b0_T/bw_per_pixel_hz, not both.")
    if has_b0:
        physical = chemical_shift_pixels(
            kw["b0_T"], kw["bw_per_pixel_hz"], float(kw.get("sigma_cs_ppm", DEFAULT_CHI_PPM))
        )
        shift = float(theta) * physical
    else:
        shift = float(kw.get("shift_pixels", float(theta) * 14.0))
    # Default fat_fraction lowered 0.35 -> 0.15: 35 % was non-physical for brain
    # (parenchyma is < 1 % fat) and produced a whole-image fat ghost at 35 %
    # intensity; 0.15 keeps the ghost visible while being a more plausible
    # phantom-scale global fraction. Anatomy callers should still override via the
    # (wired-through) fat_fraction / fat_fraction_map kwarg for a localized fat mask.
    ff = float(kw.get("fat_fraction", 0.15))
    return ifft2c(
        simulate_chemical_shift(
            fft2c(x),
            x.shape[-2:],
            shift_pixels=shift,
            fat_fraction=ff,
            **{k: kw[k] for k in ("fat_fraction_map",) if k in kw},
        )
    )


def _delegate_magic_angle(x, theta, seed=0, **kw):
    x = _ensure_complex(x)
    with _isolated_rng(x.device, seed):
        torch.manual_seed(int(seed))
        if x.device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        te = float(kw.get("te", 20.0))
        t2_base = float(kw.get("t2_base", 10.0))
        t2_magic = t2_base + (float(kw.get("t2_magic", 30.0)) - t2_base) * float(theta)
        # #223: 0.05 -> 0.30 of the ANATOMY SUPPORT (not of the whole FOV, most of
        # which is air). An MSK slice through a tendon/ligament-rich region — the
        # only anatomy where the magic-angle effect exists at all — is easily 30%
        # ordered collagen by area. At 0.05-of-FOV the axis peaked at a rel-L2 of
        # 0.14 even on its luckiest seed, ~5x weaker than the median axis (0.40);
        # it now lands at 0.33-0.48, in family with the rest of the taxonomy.
        out = simulate_magic_angle(
            x,
            x.shape[-2:],
            te=te,
            t2_base=t2_base,
            t2_magic=t2_magic,
            collagen_fraction=float(kw.get("collagen_fraction", 0.30)),
            tissue_class=kw.get("tissue_class", "msk"),
            foreground_mask=kw.get("foreground_mask"),
        )
    return out
