"""Digital Twin MRI Simulator — Joint Forward Physics Model.

Embeds mathematically perfect synthetic digital anchor markers into clean
anatomy and applies physically-accurate simulated degradation.

Physics Model:
    Synthetic markers are fixed in simulated FOV coordinates (corners).
    The brain MOVES but the markers DO NOT.

    Therefore:
    1. Motion artifacts → apply to anatomy ONLY (before marker embedding)
    2. Simulated scanner artifacts → apply to the JOINT image:
       - B0 inhomogeneity (geometric distortion + signal dropout)
       - B1 non-uniformity (spatially varying flip angle → intensity bias)
       - K-space undersampling (aliasing wraps across entire FOV)
       - Gibbs ringing (truncation artifacts)
       - Complex AWGN → Rician noise in magnitude
       - Chemical shift (fat-water frequency offset)
       - Eddy currents (gradient-induced linear phase)
       - Spike noise (RF zipper artifacts)
       - RF crosstalk (multi-slice signal bleeding)

Pipeline:
    Clean anatomy → Apply motion (k-space phase ramp) → IFFT
    → Embed synthetic corner markers → FFT joint image
    → Apply B0 phase → Apply undersampling mask → Add k-space noise
    → Apply chemical shift, eddy, spikes → IFFT → Apply B1 bias, crosstalk
    → Corrupted joint image

Marker Types:
    - ``gaussian``: 4 isotropic Gaussians at FOV corners (smooth, sub-pixel)
    - ``cross``:    4 thin crosshair fiducials at corners (sharp, k-space QC)

Motion Models:
    - ``rigid``:       Global translation via k-space phase ramp
    - ``periodic``:    Sinusoidal respiratory/cardiac motion per PE line
    - ``random_shot``: Independent per-line random phase jitter
    - ``elastic``:     Non-rigid deformation via low-frequency displacement field

References:
    * Epperson et al., "Creation of fully sampled MR finger dataset,"
      MRM 2021.
    * Macovski A., "Noise in MRI," MRM 36:494-497, 1996. (k-space noise stats)
    * Bernstein, King & Zhou, *Handbook of MRI Pulse Sequences*, Elsevier
      2004. (B0 distortion, eddy currents, N/2 ghost, Maxwell)
    * Pruessmann et al., "SENSE: sensitivity encoding for fast MRI,"
      MRM 42:952-962, 1999. (multi-coil model)
    * Koolstra et al., "Image distortion correction for MRI in low field
      permanent magnets with strong gradient nonlinearity," MAGMA 34:631,
      2021. (Halbach-informed B0 polynomial model)
    * Lustig, Donoho, Pauly, "Sparse MRI: The application of compressed
      sensing for rapid MR imaging," MRM 58:1182, 2007. (variable-density)
    * Andersson et al., "How to correct susceptibility distortions in
      spin-echo echo-planar images," NeuroImage 20:870, 2003.

The expanded degradation taxonomy with a deterministic ``(theta, seed)``
contract suitable for Sobol / MI / BT meta-evaluation lives in
:mod:`spectramr.infrastructure.physics.degradations`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ``_robust_quantile`` / ``_QUANTILE_MAX_ELEMS`` were hoisted to
# ``spectramr.core.quantile`` (data-layer audit D2) so the data layer can share
# them: ``data/`` must not import from ``infrastructure/`` (#5), and it had
# five unguarded ``torch.quantile`` calls of its own. Re-exported under the
# original private names so this module's call sites are unchanged.
from spectramr.core.quantile import (  # noqa: F401  (re-export: tests read the cap)
    QUANTILE_MAX_ELEMS as _QUANTILE_MAX_ELEMS,
)
from spectramr.core.quantile import robust_quantile as _robust_quantile
from spectramr.infrastructure.physics.emi_degradation import add_readout_tones
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fft2c, fftnc, ifft2c, ifftnc
from spectramr.infrastructure.physics.line_axis import LINE_AXIS_DIM, line_axis_for
from spectramr.infrastructure.physics.per_sample_rng import (
    per_sample_normal_pair,
    per_sample_rand,
    per_sample_randn,
)
from spectramr.infrastructure.physics.severity import shape_severity_ratio
from spectramr.infrastructure.physics.spherical_harmonics import (
    _build_2d_polynomial_basis,
)


if TYPE_CHECKING:
    from spectramr.infrastructure.physics.scanner_profile import ScannerProfile

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Marker Builders
# ──────────────────────────────────────────────────────────────────────


def _build_corner_gaussians(
    im_size: tuple[int, int],
    offset: float,
    sigma: float,
) -> torch.Tensor:
    """Build 4-corner isotropic Gaussian markers.

    Each Gaussian is placed at ``(±offset, ±offset)`` in normalised
    ``[-1, 1]`` coordinates and has width controlled by *sigma*.

    Args:
        im_size: Image dimensions ``(H, W)``.
        offset: Fractional offset from centre (0.8 → near corners).
        sigma: Gaussian width in normalised coordinates.

    Returns:
        Complex marker template ``[1, 1, H, W]``.
    """
    H, W = im_size
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing="ij",
    )

    marker = torch.zeros(H, W, dtype=torch.float32)
    for cx, cy in [
        (-offset, -offset),
        (-offset, offset),
        (offset, -offset),
        (offset, offset),
    ]:
        marker += torch.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / sigma)

    marker = marker / marker.max().clamp(min=1e-8)
    marker_complex = torch.complex(marker, torch.zeros_like(marker))
    return marker_complex.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def _build_corner_crosses(
    im_size: tuple[int, int],
    offset: float,
    arm_length: float = 0.06,
    arm_width: float | None = None,
) -> torch.Tensor:
    """Build 4-corner crosshair fiducial markers.

    Each marker is a thin ``+``-shaped cross at ``(±offset, ±offset)``
    in normalised ``[-1, 1]`` coordinates.  The sharp edges produce
    broadband k-space energy, useful for verifying k-space integrity
    (e.g. point-spread function, truncation artifacts).

    Args:
        im_size: Image dimensions ``(H, W)``.
        offset: Fractional offset from centre (0.8 → near corners).
        arm_length: Half-length of crosshair arms in normalised coords.
        arm_width: Half-width of crosshair arms in normalised coords.
            When ``None`` (default), set to one grid step
            (``2 / min(H, W)``) so the cross is always at least one
            pixel wide. The previous fixed default of ``0.006`` was
            thinner than a pixel at 64×64 and produced an all-zero
            marker.

    Returns:
        Complex marker template ``[1, 1, H, W]``.
    """
    H, W = im_size
    if arm_width is None:
        # 2/min(H,W) is exactly one grid step in normalised coords.
        arm_width = 2.0 / float(min(H, W))
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing="ij",
    )

    marker = torch.zeros(H, W, dtype=torch.float32)
    for cx, cy in [
        (-offset, -offset),
        (-offset, offset),
        (offset, -offset),
        (offset, offset),
    ]:
        dx = (X - cx).abs()
        dy = (Y - cy).abs()
        horizontal = (dx <= arm_length) & (dy <= arm_width)
        vertical = (dy <= arm_length) & (dx <= arm_width)
        marker = torch.where(horizontal | vertical, torch.ones_like(marker), marker)

    marker = marker / marker.max().clamp(min=1e-8)
    marker_complex = torch.complex(marker, torch.zeros_like(marker))
    return marker_complex.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


_MARKER_BUILDERS = {
    "gaussian": _build_corner_gaussians,
    "cross": _build_corner_crosses,
    "none": None,  # No markers (for evaluation/diagnostics)
}


# ──────────────────────────────────────────────────────────────────────
# Robust quantile (torch.quantile size guard)
# ──────────────────────────────────────────────────────────────────────

# ``torch.quantile`` sorts the reduced dimension and refuses any reduced
# length above 2**24 (~16.7M) elements, raising
# ``RuntimeError: quantile() input tensor is too large``. A larger
# *validation* batch tips the single-coil anatomy flatten over that cap, so
# the marker embedder hard-failed every validation batch → "zero successful
# batches" (exp_vf_ib_infonce_v2, 2026-06-22 forensics). Subsample the
# reduced dimension (evenly strided, so no RNG state is consumed and the run
# stays deterministic) before the quantile — the 0.75/0.99 quantiles the
# embedder needs are statistically unchanged under uniform subsampling.
# ──────────────────────────────────────────────────────────────────────
# Fiducial Embedder
# ──────────────────────────────────────────────────────────────────────


class CornerFiducialEmbedder(nn.Module):
    """Embeds fiducial markers in the FOV corners (signal-free air).

    Supports two marker shapes via ``marker_type``:
      - ``"gaussian"``: smooth isotropic blobs (good for sub-pixel tracking)
      - ``"cross"``:    sharp crosshair patterns (good for k-space QC)

    Markers are placed strictly in the background so they do not destroy
    anatomical signal.  Amplitude is normalised to match grey-matter
    intensity (75th percentile) to prevent gradient saturation.

    Args:
        im_size: Image dimensions ``(H, W)``.
        corner_offset: Fractional offset from edge (0.8 = 80% from centre).
        sigma: Gaussian width in normalised coordinates (gaussian only).
        arm_length: Cross arm half-length (cross only).
        arm_width: Cross arm half-width (cross only).
        marker_type: ``"gaussian"`` or ``"cross"``.
        learnable: If ``True``, marker parameters are trainable.
    """

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        corner_offset: float = 0.8,
        sigma: float = 0.03,
        arm_length: float = 0.06,
        arm_width: float | None = None,
        marker_type: str = "gaussian",
        learnable: bool = False,
        per_channel_marker_prior: bool = False,
    ) -> None:
        super().__init__()
        self.im_size = im_size
        self.corner_offset = corner_offset
        self.marker_type = marker_type
        # Mod3 (digital-twin review): when True, multi-contrast embedder
        # returns marker_prior at per-channel intensity matching what was
        # embedded; default False to preserve the legacy global-intensity
        # prior expected by older VF loss heads.
        self.per_channel_marker_prior = per_channel_marker_prior

        if marker_type not in _MARKER_BUILDERS:
            raise ValueError(
                f"Unknown marker_type={marker_type!r}. Valid: {list(_MARKER_BUILDERS)}"
            )

        if marker_type == "none":
            marker_template = torch.zeros(1, 1, *im_size)
        elif marker_type == "gaussian":
            marker_template = _build_corner_gaussians(im_size, corner_offset, sigma)
        else:
            marker_template = _build_corner_crosses(im_size, corner_offset, arm_length, arm_width)

        if learnable:
            self.marker_template = nn.Parameter(marker_template)
        else:
            self.register_buffer("marker_template", marker_template)

        mask = (marker_template.abs() > 0.01 * marker_template.abs().max()).float()
        self.register_buffer("marker_mask", mask)

    def forward(
        self,
        anatomy: torch.Tensor,
        coil_sensitivities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed markers into anatomy with contrast-matched amplitude.

        Handles three cases:
          - Single-coil / single-contrast (C=1): uniform marker scaling.
          - Multi-contrast (C>1, no coil_sensitivities): per-contrast
            intensity scaling using each channel's 75th percentile.
          - Multi-coil (C>1, coil_sensitivities provided): marker is
            modulated by each coil's spatial sensitivity S_i so that
            the embedded signal is S_i * marker (physically correct).

        Args:
            anatomy: Complex image [B, C, H, W] (may be motion-corrupted).
            coil_sensitivities: Optional [1, C, H, W] or [B, C, H, W]
                complex coil sensitivity maps.  When provided, markers
                are multiplied by S_i for each coil.

        Returns:
            joint_image: Anatomy + markers [B, C, H, W].
            marker_prior: Ideal marker (contrast-scaled) [1, 1, H, W].

        forward method for CornerFiducialEmbedder.

        Executes PyTorch tensor operations.

        Args:
            anatomy (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            coil_sensitivities (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        B, C, H, W = anatomy.shape
        marker_t = self.marker_template  # [1, 1, H, W] complex

        if coil_sensitivities is not None:
            # ── Multi-coil: marker * S_i for each coil ──
            rss = torch.sqrt((anatomy.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)
            # _robust_quantile (not raw torch.quantile): a multi-coil 3-D/5-D RSS
            # volume can exceed torch's 2**24 reduced-dim cap → "quantile() input
            # tensor is too large" (the exp_vf_ib_infonce crash). The sibling
            # single-coil / per-channel branches below already use the guard;
            # this branch was the one raw call left behind.
            tissue_intensity = _robust_quantile(rss.float(), 0.75) * 2.0
            marker_scaled = marker_t * tissue_intensity  # [1, 1, H, W]
            marker_per_coil = marker_scaled * coil_sensitivities  # [B, C, H, W]
            joint_image = anatomy + marker_per_coil
        elif C > 1:
            # ── Multi-contrast: per-channel intensity scaling ──
            # Vectorised quantile across channels (Mod2 in the digital-twin
            # review). The prior code looped over C and called torch.quantile
            # once per channel; each call host-syncs.
            per_chan_intensity = (
                _robust_quantile(anatomy.abs().float().reshape(B, C, -1), 0.75, dim=-1) * 2.0
            )  # [B, C]
            scale = per_chan_intensity.view(B, C, 1, 1).to(marker_t.dtype)
            # Per-channel embedded marker — broadcast (1, 1, H, W) marker_t.
            joint_image = anatomy + marker_t * scale
            # Mod3: return marker_prior at per-channel intensity so callers
            # can compare reconstruction against the right per-channel target.
            # Falls back to a global-intensity prior when per_channel_marker_prior
            # is False (kept for backward compatibility).
            if getattr(self, "per_channel_marker_prior", False):
                # [B, C, H, W] per-channel prior, matches what was embedded.
                marker_scaled = marker_t * scale
            else:
                tissue_intensity = per_chan_intensity.mean()
                marker_scaled = marker_t * tissue_intensity
        else:
            # ── Single-coil / single-contrast (original behaviour) ──
            tissue_intensity = _robust_quantile(anatomy.abs().float(), 0.75) * 2.0
            marker_scaled = marker_t * tissue_intensity
            joint_image = anatomy + marker_scaled.expand_as(anatomy)

        return joint_image, marker_scaled


# ──────────────────────────────────────────────────────────────────────
# Simulated Artifact Functions (affect BOTH anatomy and markers)
# ──────────────────────────────────────────────────────────────────────


def simulate_b0_inhomogeneity(
    kspace: torch.Tensor,
    im_size: tuple[int, int],
    strength: float = 0.3,
    device: torch.device | None = None,
    external_b0_field: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Simulate B0 field inhomogeneity: spatially-varying phase + dropout.

    Models two coupled artefacts:

    * **Phase modulation**: image-space phase ramp from the polynomial
      field map (geometric distortion at long TE).
    * **Magnitude dropout**: where the field gradient is large, intra-voxel
      dephasing reduces the signal magnitude. Modelled as
      :math:`s(\mathbf{r}) \propto \exp(-\|\nabla \phi(\mathbf{r})\|^2 / \sigma^2)`
      so smooth-field regions are unaffected and steep-gradient regions
      lose signal (Bernstein 2004 Ch. 13).

    Without the dropout term the function is pure phase: ``|out|`` then
    equals ``|kspace|`` up to floating-point noise, hiding the artefact
    from magnitude reconstructions.

    Args:
        kspace: Complex k-space ``[B, C, H, W]``.
        im_size: Image dimensions.
        strength: Field strength (0=none, 1=severe). Drives both the
            phase amplitude AND the dropout depth.
        device: Target device.

    Returns:
        k-space with B0-induced phase + signal-dropout ``[B, C, H, W]``.
    """
    H, W = im_size
    if device is None:
        device = kspace.device
    B = kspace.shape[0]

    if external_b0_field is not None:
        # REAL B0 drives the phase modulation (VF real-reference seam). Normalise
        # the measured field's spatial structure to the [-1, 1] phase convention
        # (scaled by ``strength``, matching the random path's magnitude).
        fm = external_b0_field.to(device=device, dtype=torch.float32)
        if fm.dim() == 5:
            fm = fm[..., fm.shape[-1] // 2]
        if fm.dim() == 3:
            fm = fm.unsqueeze(1)
        if fm.shape[-2:] != (H, W):
            fm = F.interpolate(fm, size=(H, W), mode="bilinear", align_corners=False)
        if fm.shape[0] != B:
            fm = fm[:1].expand(B, -1, -1, -1)
        fm = fm - fm.mean(dim=(-2, -1), keepdim=True)
        scale = fm.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        field_map = (fm / scale) * float(strength)
    else:
        Y, X = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        coeffs = per_sample_randn(B, 6, device=device) * strength
        field_map = (
            coeffs[:, 0].view(B, 1, 1, 1) * X.unsqueeze(0).unsqueeze(0) ** 2
            + coeffs[:, 1].view(B, 1, 1, 1) * Y.unsqueeze(0).unsqueeze(0) ** 2
            + coeffs[:, 2].view(B, 1, 1, 1)
            * X.unsqueeze(0).unsqueeze(0)
            * Y.unsqueeze(0).unsqueeze(0)
            + coeffs[:, 3].view(B, 1, 1, 1) * X.unsqueeze(0).unsqueeze(0)
            + coeffs[:, 4].view(B, 1, 1, 1) * Y.unsqueeze(0).unsqueeze(0)
            + coeffs[:, 5].view(B, 1, 1, 1)
        )

    image = ifft2c(kspace)
    phase = torch.exp(1j * math.pi * field_map)

    # Magnitude dropout from intra-voxel dephasing where the field
    # gradient is large. The raw per-pixel gradient is ~1/H of the
    # polynomial peak (sub-1e-4 in absolute terms), so we normalise
    # it to [0, 1] before plugging into the exponential — this lets
    # ``strength`` directly control how aggressively the hot-spots
    # drop signal regardless of grid resolution.
    fm = field_map  # [B, 1, H, W]
    grad_y = fm[..., 1:, :] - fm[..., :-1, :]
    grad_x = fm[..., :, 1:] - fm[..., :, :-1]
    grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
    grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
    grad_sq = grad_y.pow(2) + grad_x.pow(2)
    grad_norm = grad_sq / grad_sq.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    # strength=1 -> peak dropout factor of exp(-4) = 0.018 at the
    # worst hot-spot. Lower strength scales the depth linearly.
    dropout_scale = 4.0 * float(strength)
    dropout = torch.exp(-dropout_scale * grad_norm)  # in (e^{-4}, 1]
    image_distorted = image * phase * dropout.to(image.dtype)
    return fft2c(image_distorted)


def gre_flip_modulation(kappa: torch.Tensor, nominal_alpha: float) -> torch.Tensor:
    """Long-TR (TR >> T1) GRE signal ratio ``sin(kappa * alpha) / sin(alpha)`` for a B1+ scale ``kappa``."""
    return torch.sin(nominal_alpha * kappa) / math.sin(nominal_alpha)


def simulate_b1_bias_field(
    image: torch.Tensor,
    im_size: tuple[int, int],
    strength: float = 0.2,
    nominal_alpha: float = math.pi / 2,
    *,
    b0_T: float | None = None,
) -> torch.Tensor:
    r"""Simulate B1+ transmit field inhomogeneity via flip-angle modulation.

    Models the *transmit* B1+ spatial non-uniformity that corrupts the
    effective flip angle across the FOV.  The signal modulation follows
    the Ernst steady-state relationship:

    .. math::

        \text{modulation}(\mathbf{r}) =
            \frac{\sin(\alpha \cdot B_1^+(\mathbf{r}))}
                 {\sin(\alpha)}

    where :math:`B_1^+(\mathbf{r})` is a smooth spatial efficiency map
    (1.0 = nominal flip angle achieved).

    This replaces a naive multiplicative bias with a physically correct
    non-linear intensity modulation that captures the sin(α) relationship
    between flip angle and transverse magnetization.

    Args:
        image: Complex image ``[B, C, H, W]``.
        im_size: Image dimensions ``(H, W)``.
        strength: B1+ non-uniformity strength (0=uniform, 1=severe).
        nominal_alpha: Nominal flip angle in radians (default π/2).

    Returns:
        Image with B1+ transmit modulation ``[B, C, H, W]``.
    """
    H, W = im_size
    device = image.device
    B = image.shape[0]

    # Min1 (digital-twin review): scale B1+ inhomogeneity with B0.
    # At low field the transmit field is much more uniform: empirically
    # the deviation scales roughly as (B0/3.0)^0.8 (Bernstein 2004 Ch. 27).
    # We use 0.7 to match low-field T1/T2 measurements at 0.3 T (M4Raw).
    if b0_T is not None and b0_T > 0:
        b0_scale = (float(b0_T) / 3.0) ** 0.7
        strength = float(strength) * b0_scale

    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )

    # Generate smooth B1+ efficiency map via random low-frequency modes, one per subject.
    r = per_sample_rand(B, 4, 1, 1, 1, device=device)
    freq_x, freq_y = r[:, 0] * 2, r[:, 1] * 2
    phase_x, phase_y = r[:, 2] * 2 * math.pi, r[:, 3] * 2 * math.pi

    # B1+ efficiency map: 1.0 = nominal, deviation controlled by strength
    b1_plus = 1.0 + strength * (
        torch.cos(freq_x * math.pi * X.unsqueeze(0).unsqueeze(0) + phase_x)
        * torch.cos(freq_y * math.pi * Y.unsqueeze(0).unsqueeze(0) + phase_y)
    )
    b1_plus = b1_plus.clamp(min=0.3, max=1.7)

    # Ernst-correct modulation: sin(α·B1+) / sin(α)
    sin_nominal = math.sin(nominal_alpha)
    if abs(sin_nominal) < 1e-8:
        # Degenerate flip angle, fall back to linear
        return image * b1_plus

    return image * gre_flip_modulation(b1_plus, nominal_alpha)


def outer_line_count(H: int, n_center: int, R: float) -> int:
    """Outer PE lines to acquire so the TOTAL kept count realises acceleration ``R``.

    The declared acceleration is a statement about the whole acquisition:
    ``R = H / (lines actually acquired)`` — the fastMRI convention. The lines
    already spent on the fully-sampled ACS centre therefore come *out of* the
    budget, they are not free (issue #246: the old code budgeted
    ``(H - n_center) / R`` outer lines *on top of* the ACS, so a declared R = 8
    realised R = 5.3).
    """
    n_total = max(n_center, round(H / max(R, 1e-6)))
    return max(0, n_total - n_center)


def simulate_undersampling(
    kspace: torch.Tensor,
    acceleration: float = 4.0,
    center_fraction: float = 0.08,
    return_mask: bool = False,
    line_dim: int = -2,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Simulate Cartesian k-space undersampling.

    Retains a dense center region and randomly samples outer k-space lines.
    This causes coherent aliasing that wraps across the entire FOV,
    affecting both anatomy and markers.

    The sampling mask is drawn *randomly per call* (``torch.randperm``), so a
    caller that needs to enforce data consistency against the measurement must
    capture the exact mask used — it cannot be reconstructed afterwards. Pass
    ``return_mask=True`` to receive it.

    Args:
        kspace: Complex k-space [B, C, H, W].
        acceleration: Acceleration factor.
        center_fraction: Fraction of center lines to always acquire.
        return_mask: When True, also return the binary sampling mask
            ``[1, 1, H, 1]`` (broadcasts over B/C/W).
        line_dim: The k-space dimension whose lines are kept or dropped, -2 (rows,
            the default) or -1 (columns). The twin resolves it from
            ``physics.digital_twin.mask_axis``.

    Returns:
        Undersampled k-space ``[B, C, H, W]``, or ``(kspace, mask)`` when
        ``return_mask`` is True.
    """
    if line_dim not in (-1, -2):
        raise ValueError(f"line_dim must be -1 or -2, got {line_dim!r}")
    N = kspace.shape[line_dim]
    device = kspace.device

    center_lines = int(N * center_fraction)
    center_start = N // 2 - center_lines // 2
    center_end = center_start + center_lines

    # Outer lines come out of the R budget and are drawn from outside the centre, so the
    # realised acceleration N / (lines kept) is the declared one.
    num_outer = outer_line_count(N, center_lines, acceleration)
    outer_pool = torch.cat(
        [torch.arange(0, center_start, device=device), torch.arange(center_end, N, device=device)]
    )
    outer_indices = outer_pool[torch.randperm(outer_pool.numel(), device=device)[:num_outer]]

    line = torch.zeros(N, device=device)
    line[center_start:center_end] = 1.0
    line[outer_indices] = 1.0
    mask = line.reshape(1, 1, N, 1) if line_dim == -2 else line.reshape(1, 1, 1, N)

    undersampled = kspace * mask
    if return_mask:
        return undersampled, mask
    return undersampled


def simulate_gibbs_ringing(
    kspace: torch.Tensor,
    truncation_fraction: float = 0.85,
) -> torch.Tensor:
    """Simulate Gibbs ringing by truncating high-frequency k-space.

    Low-field MRI has limited gradient performance, causing effective
    k-space truncation. This produces ringing artifacts near sharp edges.

    ``truncation_fraction`` is the retained fraction of the **maximum
    k-space radius** — the corner of the grid, not the edge midpoint. The
    radius is therefore normalised by its own maximum (``sqrt(2)`` on the
    ``[-1, 1]^2`` grid) so that ``truncation_fraction = 1.0`` is a genuine
    identity.

    Issue #246: the window used to compare an un-normalised
    ``sqrt(kx^2 + ky^2)`` (max ``sqrt(2) ~ 1.414``) against the fraction, so
    at the nominal "no truncation" setting of 1.0 the circular window still
    zeroed the k-space **corners** — 1 - pi/4 = 21.5% of the samples — and
    ``D(x, theta=0)`` came back with a rel-L2 of 0.055 against its own input.
    That pedestal exceeded the *entire* theta=1 distortion of several other
    axes and contaminated every statistic anchored on the clean baseline.

    Args:
        kspace: Complex k-space [B, C, H, W].
        truncation_fraction: Retained fraction of the maximum k-space radius
            (1.0 = no truncation, exactly the identity).

    Returns:
        Truncated k-space [B, C, H, W].
    """
    if truncation_fraction >= 1.0:
        # Exact identity — never route through a float comparison that can
        # shave the corner samples off by an ulp.
        return kspace

    H, W = kspace.shape[-2], kspace.shape[-1]
    device = kspace.device

    # Normalised k in [-1, 1) on the fft2c grid, so the window is centred on DC.
    ky = (2.0 * centred_freqs(H, device)).view(1, 1, -1, 1)
    kx = (2.0 * centred_freqs(W, device)).view(1, 1, 1, -1)
    radius = torch.sqrt(kx**2 + ky**2) / math.sqrt(2.0)  # in [0, 1]

    window = (radius <= truncation_fraction).float()
    return kspace * window


def simulate_concomitant_phase(
    image: torch.Tensor,
    im_size: tuple[int, int],
    b0_T: float = 0.3,
    gradient_amplitudes_mT_per_m: float = 20.0,
    voxel_size_mm: float = 1.0,
    duration_ms: float = 5.0,
) -> torch.Tensor:
    r"""Maxwell concomitant-phase artefact (relevant at low field).

    The transverse component of the gradient field gives rise to a
    concomitant phase term :math:`\phi_c \propto G^2 / B_0` that is
    the dominant residual at :math:`B_0 < 1~\mathrm{T}` after shimming
    (Bernstein, King & Zhou 2004, Ch. 10). The phase grows
    quadratically off-axis and is *not* removed by standard
    first-order B0 shimming.

    Min1 (digital-twin review). The framework already has
    ``concomitant_phase_operator.py`` and
    ``maxwell_concomitant.py`` in ``infrastructure/physics/``; this
    function gives the digital twin a thin entry point at the
    severity-knob level the meta-evaluation harness expects.

    The added phase at position ``(x, y)`` along an axial slice is

    .. math::

        \phi_c(x, y) \;=\; \frac{\gamma\, G^2\, \tau}{8\, B_0}\,
            (x^2 + y^2)

    where :math:`\tau` is the duration in seconds, :math:`G` the
    peak gradient in T/m, and :math:`\gamma = 2\pi \cdot
    42.577~\mathrm{MHz/T}`.

    Args:
        image: Complex image ``[B, C, H, W]``.
        im_size: ``(H, W)``.
        b0_T: Main field strength in tesla.
        gradient_amplitudes_mT_per_m: Peak gradient amplitude in mT/m.
        voxel_size_mm: Isotropic voxel size; sets the physical extent
            of the FOV at the chosen ``im_size``.
        duration_ms: Phase-accrual interval in ms.

    Returns:
        Image with the concomitant phase applied ``[B, C, H, W]``.
    """
    H, W = im_size
    device = image.device
    if b0_T <= 0:
        raise ValueError(f"b0_T must be > 0; got {b0_T}.")

    # Physical FOV in metres.
    fov_y_m = H * voxel_size_mm * 1e-3
    fov_x_m = W * voxel_size_mm * 1e-3
    y_m = torch.linspace(-fov_y_m / 2, fov_y_m / 2, H, device=device).view(1, 1, H, 1)
    x_m = torch.linspace(-fov_x_m / 2, fov_x_m / 2, W, device=device).view(1, 1, 1, W)
    r2 = x_m**2 + y_m**2  # [1, 1, H, W]

    gamma = 2 * math.pi * GAMMA_BAR_HZ_PER_T  # rad / (s * T)
    g_t_per_m = gradient_amplitudes_mT_per_m * 1e-3
    tau_s = duration_ms * 1e-3
    phase = (gamma * g_t_per_m**2 * tau_s) / (8.0 * b0_T) * r2

    return image * torch.exp(1j * phase).to(image.dtype)


#: Fat-water chemical-shift difference (ppm): the one owner of the value, re-exported by
#: ``scanner_profile`` and read by every fat axis.
DEFAULT_CHI_PPM = 3.5
#: Proton gyromagnetic ratio gamma / 2 pi (Hz/T): the one owner for the twin and the registry axes.
GAMMA_BAR_HZ_PER_T = 42.577e6


def chemical_shift_pixels(
    b0_tesla: float, bw_per_pixel_hz: float, sigma_cs_ppm: float = DEFAULT_CHI_PPM
) -> float:
    """Fat-water displacement along readout in pixels: gamma/2pi * B0 * sigma_cs / BW_pixel."""
    if bw_per_pixel_hz <= 0:
        raise ValueError(f"bw_per_pixel_hz must be > 0; got {bw_per_pixel_hz}.")
    return GAMMA_BAR_HZ_PER_T * float(b0_tesla) * (sigma_cs_ppm * 1e-6) / float(bw_per_pixel_hz)


def simulate_chemical_shift(
    kspace: torch.Tensor,
    im_size: tuple[int, int],
    shift_pixels: float = 3.0,
    fat_fraction: float = 0.15,
    *,
    b0_T: float | None = None,
    bw_per_pixel_hz: float | None = None,
    sigma_cs_ppm: float = DEFAULT_CHI_PPM,
    fat_fraction_map: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Simulate fat-water chemical shift displacement.

    Fat and water have a ~3.5 ppm (``DEFAULT_CHI_PPM``) chemical shift, which displaces the
    fat signal by a few pixels along the frequency-encode direction.
    Modeled as a shifted copy of the image scaled by ``fat_fraction``.

    M3 (digital-twin review): when ``b0_T`` and ``bw_per_pixel_hz``
    are supplied, the displacement scales with field strength:

    .. math::

        \Delta x = \gamma\, B_0\, \sigma_{\mathrm{cs}} / \mathrm{BW}_{\mathrm{pixel}}

    with :math:`\gamma / 2\pi = 42.577~\mathrm{MHz/T}`. Hard-coding
    ``shift_pixels`` for every field strength (the prior behaviour)
    gives identical artefact at 0.3 T and 3 T, which is non-physical.

    If ``fat_fraction_map`` is provided, the multiplicative
    fat fraction is taken per-voxel from the map; otherwise the
    scalar ``fat_fraction`` is applied uniformly (the prior behaviour,
    which assumes every voxel contains 15% fat --- wrong for brain).

    Args:
        kspace: Complex k-space ``[B, C, H, W]``.
        im_size: Image dimensions.
        shift_pixels: Fallback displacement in pixels along readout
            (last dim), used only when B0/bandwidth are not supplied.
        fat_fraction: Fallback scalar fraction of signal that is fat-like.
        b0_T: Optional main field strength in tesla. Together with
            ``bw_per_pixel_hz`` overrides ``shift_pixels``.
        bw_per_pixel_hz: Bandwidth per pixel along the readout axis.
        sigma_cs_ppm: Fat-water chemical shift in ppm (default ``DEFAULT_CHI_PPM``).
        fat_fraction_map: Optional ``[B|1, 1, H, W]`` (or broadcastable)
            real-valued spatial map of fat fraction. When given,
            overrides the scalar ``fat_fraction``.

    Returns:
        k-space with chemical shift artifact ``[B, C, H, W]``.
    """
    H, W = im_size
    device = kspace.device

    if b0_T is not None and bw_per_pixel_hz is not None and bw_per_pixel_hz > 0:
        shift_pixels = chemical_shift_pixels(b0_T, bw_per_pixel_hz, sigma_cs_ppm)

    # Phase ramp along readout for sub-pixel shift
    kx = centred_freqs(W, device).view(1, 1, 1, -1)
    phase_ramp = torch.exp(-1j * 2 * math.pi * kx * shift_pixels)

    # Chemical shift PARTITIONS the signal: a fraction `fat_fraction` of it resonates at
    # the fat frequency and is displaced along readout; the remaining water fraction is
    # not. Adding the shifted fat on top of the FULL water signal instead (which is what
    # this did) applies a (1 + fat_fraction) global gain -- so at shift_pixels = 0 the
    # "clean" anchor came back 1.35x brighter than its input, and rel-L2 vs the reference
    # DECREASED as severity rose. Every L2-family metric (PSNR/NMSE/MSE) then scored the
    # severity axis backwards.
    if fat_fraction_map is not None:
        # Apply the per-voxel fat fraction in image space then transform.
        image = ifft2c(kspace)
        ff_map = fat_fraction_map.to(image.dtype).to(device)
        fat_kspace = fft2c(image * ff_map) * phase_ramp
        water_kspace = fft2c(image * (1.0 - ff_map))
    else:
        fat_kspace = kspace * phase_ramp * fat_fraction
        water_kspace = kspace * (1.0 - fat_fraction)
    return water_kspace + fat_kspace


def simulate_eddy_current(
    kspace: torch.Tensor,
    strength: float = 0.1,
) -> torch.Tensor:
    """Simulate gradient-induced eddy current artifacts.

    Eddy currents from rapidly switching gradients cause a linear
    phase accumulation that varies with the PE line index, producing
    image shearing and ghosting.

    Args:
        kspace: Complex k-space ``[B, C, H, W]``.
        strength: Phase accumulation magnitude (radians per line).

    Returns:
        k-space with eddy-current phase ``[B, C, H, W]``.
    """
    H = kspace.shape[-2]
    device = kspace.device

    # Linear + quadratic phase per PE line, u in [-1, 1) with u = 0 on the k-space centre
    # line, so the linear term is an exact PE shift of alpha pixels.
    line_idx = (2.0 * centred_freqs(H, device)).view(1, 1, -1, 1)
    B = kspace.shape[0]
    ab = per_sample_randn(B, 2, device=device).view(B, 2, 1, 1, 1)
    alpha = ab[:, 0] * strength
    beta = ab[:, 1] * strength * 0.5

    phase = alpha * line_idx + beta * line_idx**2
    return kspace * torch.exp(1j * math.pi * phase)


def simulate_spike_noise(
    kspace: torch.Tensor,
    probability: float = 0.01,
    spike_amplitude: float = 5.0,
) -> torch.Tensor:
    """Simulate RF spike noise (zipper artefact) in k-space.

    Random high-intensity points in k-space caused by RF interference.
    In image space these appear as sinusoidal stripe (zipper) patterns
    across the entire FOV, affecting both anatomy and markers equally.

    A spike is one event in time, so it lands on the same k-space sample in
    every receive coil; its complex value differs per coil. The amplitude is
    relative to each sample's own RMS over its coils.

    Args:
        kspace: Complex k-space ``[B, C, H, W]``.
        probability: Per-sample-point probability of a spike.
        spike_amplitude: Relative amplitude of spikes vs signal RMS.

    Returns:
        k-space with spike noise ``[B, C, H, W]``.
    """
    if kspace.dim() != 4:
        raise ValueError(f"simulate_spike_noise expects [B, C, H, W], got {tuple(kspace.shape)}")
    return _add_kspace_spikes(kspace, probability, spike_amplitude)


def simulate_spike_noise_volume(
    kspace: torch.Tensor,
    probability: float = 0.01,
    spike_amplitude: float = 5.0,
) -> torch.Tensor:
    """:func:`simulate_spike_noise` on a 3-D k-space ``[B, C, D, H, W]``.

    A spike is one ``(kz, ky, kx)`` sample, shared by every coil; its amplitude is relative
    to each sample's RMS over coils and the whole 3-D k-space.
    """
    if kspace.dim() != 5:
        raise ValueError(
            f"simulate_spike_noise_volume expects [B, C, D, H, W], got {tuple(kspace.shape)}"
        )
    return _add_kspace_spikes(kspace, probability, spike_amplitude)


def _add_kspace_spikes(
    kspace: torch.Tensor, probability: float, spike_amplitude: float
) -> torch.Tensor:
    B, C, *grid = kspace.shape
    signal_rms = (
        kspace.abs()
        .pow(2)
        .mean(dim=tuple(range(1, kspace.dim())), keepdim=True)
        .sqrt()
        .clamp(min=1e-8)
    )
    # One row of draws per sample, so sample b does not depend on the batch size.
    n = math.prod(grid)
    draws = per_sample_rand(B, n + 2 * C * n, device=kspace.device)
    spike_mask = (draws[:, :n] < probability).view(B, 1, *grid).float()
    # Box-Muller turns the uniform draws into the complex Gaussian spike values.
    u1 = draws[:, n : n + C * n].clamp(min=1e-12).view(B, C, *grid)
    u2 = draws[:, n + C * n :].view(B, C, *grid)
    spike_vals = (
        torch.sqrt(-2.0 * torch.log(u1)) * torch.exp(2j * math.pi * u2) / math.sqrt(2.0)
    ) * (signal_rms * spike_amplitude)

    return kspace + spike_mask * spike_vals


def simulate_rf_crosstalk(
    image: torch.Tensor,
    crosstalk_fraction: float = 0.05,
) -> torch.Tensor:
    """Simulate multi-slice RF crosstalk.

    In multi-slice acquisitions, imperfect RF pulse profiles cause
    signal from neighbouring slices to leak into the current slice. A single
    2-D slice has no neighbours, so this is an in-plane proxy: a fraction of
    the slice is replaced by a blurred copy of itself, ``(1 - f) x + f blur(x)``.
    The mix conserves the mean intensity; adding the blur on top would
    brighten the slice by ``f``. A through-plane model needs the depth axis.

    Args:
        image: Complex image ``[B, C, H, W]``.
        crosstalk_fraction: Fraction ``f`` of the slice replaced by the blur.

    Returns:
        Image with RF crosstalk ``[B, C, H, W]``.
    """
    # Gaussian PSF (sigma 2 px) as a k-space window: circular, per coil, no padding.
    sigma = 2.0
    k = fft2c(image)
    fy = centred_freqs(k.shape[-2], k.device).view(-1, 1)
    fx = centred_freqs(k.shape[-1], k.device).view(1, -1)
    blurred = ifft2c(k * torch.exp(-2.0 * math.pi**2 * sigma**2 * (fx**2 + fy**2)))
    f = float(crosstalk_fraction)
    return (1.0 - f) * image + f * blurred


# ──────────────────────────────────────────────────────────────────────
# Image-Space Physical Degradations (apply BEFORE FFT)
# ──────────────────────────────────────────────────────────────────────


#: BW_FE / BW_PE per sequence: 0 means no phase-encode shift.
_B0_PE_BANDWIDTH_DIVISOR: dict[str, float] = {"spin_warp": 0.0, "epi": 8.0}


def simulate_b0_geometric_distortion(
    image: torch.Tensor,
    im_size: tuple[int, int],
    dwell_time: float = 1e-5,
    b0_strength: float = 0.3,
    return_field: bool = False,
    external_b0_field: torch.Tensor | None = None,
    sequence: str = "spin_warp",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Simulate B0-induced geometric distortion via voxel shifting.

    Uses a Halbach-informed 2nd-order polynomial field model (cf. Koolstra
    et al., MAGMA 2021) to produce realistic B0 inhomogeneity patterns.

    .. math::

        \Delta x_{\text{FE}} = \frac{\Delta B_0}{\text{BW}_{\text{pixel,FE}}}
        \quad
        \Delta y_{\text{PE}} = \frac{\Delta B_0}{\text{BW}_{\text{pixel,PE}}}

    ``sequence`` sets the phase-encode bandwidth. In Cartesian spin-warp and
    TSE every phase-encode step is a separate excitation (or a refocused echo)
    acquired at the same echo time, so off-resonance accrues no phase across
    phase-encode lines: the PE bandwidth is effectively infinite and the shift
    is along readout only (``"spin_warp"``, the default). In EPI the
    phase-encode lines are read in one echo train, so the PE bandwidth is far
    lower than the readout bandwidth and the PE shift dominates (``"epi"``,
    modelled here as ``BW_PE = BW_FE / 8``).

    The field pattern is a 2nd-order polynomial (dominant Halbach mode)
    plus low-frequency manufacturing perturbations:

    .. math::

        \Delta B_0(x,y) = c_0 + c_1 x + c_2 y + c_3 x^2 + c_4 y^2 + c_5 xy
            + \epsilon(x,y)

    Args:
        image: Complex image ``[B, C, H, W]``.
        im_size: Image dimensions ``(H, W)``.
        dwell_time: ADC dwell time in seconds.
        b0_strength: Severity multiplier (0→1). At 1.0, produces ~600 Hz
            peak offset (matching centre-slice Halbach magnet). Ignored when
            ``external_b0_field`` is supplied.
        external_b0_field: Optional REAL B0 field map ``[B,H,W]`` / ``[B,1,H,W]``
            (Hz). When given, it is applied verbatim (resized/broadcast to
            ``im_size``) instead of the synthetic Halbach field — so the VF
            field-scoring grades the model against a real off-resonance pattern
            (its spatial structure + magnitude), not a random simulation.
        sequence: ``"spin_warp"`` (readout-only shift) or ``"epi"``. Raises on
            any other value.

    Returns:
        Geometrically distorted image ``[B, C, H, W]``. When ``return_field`` is
        True, instead returns ``(distorted_image, b0_map, shift_y_pixels)`` where
        ``b0_map`` is the applied B0 field in Hz ``[B, H, W]`` and
        ``shift_y_pixels`` is the PE-axis pixel displacement field (all zeros
        for ``"spin_warp"``) — the reference an EPI tracker is expected to recover.
    """
    if sequence not in _B0_PE_BANDWIDTH_DIVISOR:
        raise ValueError(
            f"sequence must be one of {sorted(_B0_PE_BANDWIDTH_DIVISOR)}; got {sequence!r}."
        )
    H, W = im_size
    device = image.device
    B = image.shape[0]
    rdt = image.real.dtype

    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=rdt),
        torch.linspace(-1, 1, W, device=device, dtype=rdt),
        indexing="ij",
    )

    if external_b0_field is not None:
        # ── REAL B0 field (VF real-reference seam) ───────────────────
        # Apply a measured/derived B0 map verbatim instead of the synthetic
        # Halbach polynomial. The scoring then grades the model against a REAL
        # off-resonance pattern (its spatial structure + magnitude), not a
        # randomly-simulated one. Accept [B,H,W] or [B,1,H,W], any resolution.
        b0_map = external_b0_field.to(device=device, dtype=rdt)
        if b0_map.dim() == 4:
            b0_map = b0_map.squeeze(1)
        if b0_map.dim() == 2:
            b0_map = b0_map.unsqueeze(0)
        if b0_map.shape[-2:] != (H, W):
            b0_map = F.interpolate(
                b0_map.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1)
        if b0_map.shape[0] != B:
            b0_map = b0_map[:1].expand(B, -1, -1)
    else:
        # ── Halbach-informed synthetic B0 field map ──────────────────
        # 6-coefficient 2nd-order polynomial (Koolstra model)
        # Peak B0 offset in Hz: 600 Hz at b0_strength=1.0 (centre-slice Halbach)
        peak_hz = 600.0 * b0_strength

        # One row of draws per sample (6 coefficients, then the perturbation grid), so a
        # subject's field does not depend on the batch size.
        h_lo, w_lo = max(2, H // 16), max(2, W // 16)
        draws = per_sample_randn(B, 6 + h_lo * w_lo, device=device, dtype=rdt)
        coeffs = draws[:, :6]
        field = (
            coeffs[:, 0].view(B, 1, 1) * 0.05  # constant offset
            + coeffs[:, 1].view(B, 1, 1) * X.unsqueeze(0) * 0.1  # linear x
            + coeffs[:, 2].view(B, 1, 1) * Y.unsqueeze(0) * 0.1  # linear y
            + coeffs[:, 3].view(B, 1, 1) * X.unsqueeze(0) ** 2  # quadratic x²
            + coeffs[:, 4].view(B, 1, 1) * Y.unsqueeze(0) ** 2  # quadratic y²
            + coeffs[:, 5].view(B, 1, 1) * X.unsqueeze(0) * Y.unsqueeze(0) * 0.3  # xy
        )  # [B, H, W]

        # Normalise to target peak offset
        field = field - field.mean(dim=(-2, -1), keepdim=True)
        field_max = field.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        field = field / field_max * peak_hz * 0.8  # [B, H, W] in Hz

        # Add low-frequency perturbation (manufacturing imperfections)
        noise_low = draws[:, 6:].view(B, 1, h_lo, w_lo)
        noise_up = F.interpolate(noise_low, size=(H, W), mode="bicubic", align_corners=False)
        noise_up = noise_up.squeeze(1)  # [B, H, W]
        noise_max = noise_up.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        noise_up = noise_up / noise_max * peak_hz * 0.2
        b0_map = field + noise_up  # [B, H, W]

    # ── Compute pixel displacements along FE and PE axes ─────────────
    # Frequency-encode (readout) bandwidth per pixel
    bw_fe = 1.0 / (W * dwell_time)  # Hz/pixel

    # BW_PE / BW_FE; zero for spin-warp, whose PE bandwidth is infinite.
    shift_x_pixels = readout_offset_px(b0_map, bw_fe)  # FE displacement [B, H, W]
    shift_y_pixels = b0_map * (_B0_PE_BANDWIDTH_DIVISOR[sequence] / bw_fe)  # PE [B, H, W]

    # ── Build sampling grid with 2-axis displacement ─────────────────
    grid_y = Y.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    grid_x = X.unsqueeze(0).expand(B, -1, -1)

    # One pixel is 2 / (N - 1) in align_corners=True grid coordinates.
    grid_x_shifted = grid_x + shift_x_pixels * (2.0 / max(W - 1, 1))
    grid_y_shifted = grid_y + shift_y_pixels * (2.0 / max(H - 1, 1))

    grid = torch.stack([grid_x_shifted, grid_y_shifted], dim=-1)  # [B, H, W, 2]

    # Warp real and imaginary parts separately
    warped_real = F.grid_sample(
        image.real, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    warped_imag = F.grid_sample(
        image.imag, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    distorted = torch.complex(warped_real, warped_imag)
    if return_field:
        return distorted, b0_map, shift_y_pixels
    return distorted


#: Grid size at which ``severity`` was calibrated (~5% FOV at 7e-5); the displacement is
#: a fraction of the FOV at every grid size.
_GNL_REFERENCE_GRID = 256


def simulate_gradient_nonlinearity(
    image: torch.Tensor,
    im_size: tuple[int, int],
    severity: float = 2e-6,
    max_order: int = 3,
    sh_coefficients: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Simulate gradient non-linearity via spherical harmonic displacement.

    Real MRI gradient coils produce fields that deviate from linearity,
    especially at the FOV periphery.  The displacement field is modelled
    using a 2D polynomial (spherical harmonic) basis up to ``max_order``,
    which captures barrel, pincushion, and asymmetric distortions observed
    in clinical systems:

    .. math::

        \Delta\mathbf{r}(x, y) = \sum_{p+q \le L}
            c_{pq}^{(x,y)} \, x^p y^q

    When ``sh_coefficients`` are not provided, random coefficients are
    generated with amplitude controlled by ``severity``, producing
    physically plausible distortion patterns (including the classic
    :math:`\alpha r^2` barrel/pincushion as a special case of the
    2nd-order terms).

    Applied in image-space **before** FFT.

    Args:
        image: Complex image ``[B, C, H, W]``.
        im_size: Image dimensions ``(H, W)``.
        severity: Distortion strength when generating random coefficients.
        max_order: Maximum SH polynomial order (3 = cubic distortion).
        sh_coefficients: Optional ``[2, K]`` tensor of pre-computed
            spherical harmonic coefficients for (dx, dy) displacement.
            K must equal the number of basis functions for ``max_order``.
            If None, random coefficients are generated.

    Returns:
        Spatially warped image ``[B, C, H, W]``.
    """
    H, W = im_size
    device = image.device
    B = image.shape[0]

    # Build polynomial basis
    basis = _build_2d_polynomial_basis((H, W), max_order, device=device)  # [M, K]
    K = basis.shape[1]

    if sh_coefficients is not None:
        if sh_coefficients.shape[-1] != K:
            raise ValueError(
                f"sh_coefficients has {sh_coefficients.shape[-1]} terms but "
                f"max_order={max_order} requires {K} basis functions."
            )
        coeffs = sh_coefficients.to(device)  # [2, K]
    else:
        # Generate random coefficients; zero the constant term (index 0)
        # to prevent global translation, emphasise higher-order distortion.
        # The basis is in normalised [-1, 1] coordinates, so the coefficient scale must
        # not depend on the grid: severity is calibrated at a 256 grid and fixed there.
        scale = severity * _GNL_REFERENCE_GRID
        coeffs = torch.randn(2, K, device=device) * scale
        # Order 0 is a translation and order 1 (basis columns 1, 2: y, x) a gradient
        # scale/shear error; neither is non-linearity, so only order >= 2 is kept.
        coeffs[:, :3] = 0.0

    # Evaluate displacement fields
    rdt = image.real.dtype
    dx = (basis @ coeffs[0]).reshape(H, W).to(rdt)  # [H, W]
    dy = (basis @ coeffs[1]).reshape(H, W).to(rdt)  # [H, W]

    # Base grid in [-1, 1]
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=rdt),
        torch.linspace(-1, 1, W, device=device, dtype=rdt),
        indexing="ij",
    )

    x_warped = (X + dx).clamp(-1, 1)
    y_warped = (Y + dy).clamp(-1, 1)

    grid = torch.stack(
        [
            x_warped.unsqueeze(0).expand(B, -1, -1),
            y_warped.unsqueeze(0).expand(B, -1, -1),
        ],
        dim=-1,
    )  # [B, H, W, 2]

    warped_real = F.grid_sample(
        image.real, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    warped_imag = F.grid_sample(
        image.imag, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return torch.complex(warped_real, warped_imag)


def _anatomy_support(
    image: torch.Tensor,
    foreground_mask: torch.Tensor | None = None,
    rel_threshold: float = 0.05,
) -> torch.Tensor:
    """Binary support of the anatomy — the voxels that actually carry signal.

    Returns ``[B, H, W]`` in ``{0, 1}``. A voxel is in the support when its
    magnitude exceeds ``rel_threshold`` of the per-image peak. An optional
    ``foreground_mask`` (e.g. the complement of the fiducial-marker mask) is
    intersected in. An all-zero image degrades to the full FOV rather than to
    an empty support, so a degradation anchored on the support can never become
    a silent no-op (issue #223).
    """
    mag = image.abs()
    if mag.ndim == 4:
        mag = mag.amax(dim=1)  # [B, H, W] — signal in ANY channel counts
    peak = mag.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    support = (mag > rel_threshold * peak).float()

    # A support *inferred* from a degenerate (all-zero) image would make every
    # support-anchored patch a no-op — the exact failure mode #223 documents.
    # Fall back to the whole FOV. This rescues the inference ONLY; it must run
    # before the mask is applied, because a caller-supplied mask is an explicit
    # instruction and overriding it would itself be a silent fallback (#9).
    empty = support.flatten(1).sum(-1) < 1.0  # [B]
    if bool(empty.any()):
        support = torch.where(empty.view(-1, 1, 1), torch.ones_like(support), support)

    if foreground_mask is not None:
        fg = foreground_mask.to(dtype=support.dtype, device=support.device)
        if fg.ndim == 4:
            fg = fg.squeeze(1)  # [B|1, H, W]
        support = support * fg  # an empty mask means "nowhere" — honour it
    return support


def simulate_magic_angle(
    image: torch.Tensor,
    im_size: tuple[int, int],
    te: float = 20.0,
    t2_base: float = 10.0,
    t2_magic: float = 30.0,
    collagen_fraction: float = 0.05,
    *,
    tissue_class: str = "msk",
    foreground_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Simulate the magic angle effect on ordered-structure T2.

    In tendons, ligaments, and cartilage with ordered collagen, the
    dipolar interaction vanishes at exactly :math:`\theta = 54.7^{\circ}`
    (the "magic angle"), causing focal T2 prolongation that mimics pathology.

    .. math::

        T_{2,\text{eff}} = T_{2,\text{base}}
            + (T_{2,\text{magic}} - T_{2,\text{base}})
              \exp\bigl(-10\,|3\cos^2\theta - 1|\bigr)

    Signal modulation:

    .. math::

        S_{\text{boost}} = \frac{e^{-TE / T_{2,\text{eff}}}}{e^{-TE / T_{2,\text{base}}}}

    Applied in image-space **before** FFT.

    Args:
        image: Complex image ``[B, C, H, W]``.
        im_size: Image dimensions ``(H, W)``.
        te: Echo time in ms.
        t2_base: Baseline T2 for ordered collagen in ms.
        t2_magic: Maximum T2 at the magic angle in ms.
        collagen_fraction: Approximate fraction of FOV containing collagen.
        tissue_class: Anatomy class for the input. ``"msk"`` (default)
            applies the effect as in the legacy implementation; for any
            other value (e.g. ``"brain"``, ``"abdomen"``) the function
            is a no-op because the magic-angle phenomenon requires
            ordered collagen and is irrelevant outside MSK protocols
            (Fullerton, Cameron, Ord 1985). M4 (digital-twin review).
        foreground_mask: Optional ``[B|1, 1, H, W]`` real mask of
            anatomy support; when supplied the random collagen patches
            are intersected with the mask so they don't hit synthetic
            markers placed in air or the FOV background.

    Returns:
        Image with magic angle T2 modulation ``[B, C, H, W]``.
    """
    if tissue_class != "msk":
        # No magic-angle effect for non-MSK tissue. M4 fix.
        return image

    H, W = im_size
    device = image.device
    B = image.shape[0]

    # Synthesize collagen orientation map — random smooth angle field
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    # ── Collagen mask — elliptical fibre patches drawn INSIDE the anatomy ──
    #
    # Issue #223. The patches used to be centred on ``U[-1, 1]^2`` — anywhere in
    # the FOV, signal-free background included. Two patches were drawn, and at
    # ``seed=0`` — the sweep's own default — BOTH landed in the zero-valued air
    # around the brain. The modulation then multiplied nothing:
    # ``torch.equal(D(x, theta), x)`` was True for *every* theta. It was a seed
    # lottery (seeds 1/2/3/7 gave rel-L2 0.08/0.10/0.05/0.14; seed 0 gave exactly
    # 0.0), so the axis was scored as a constant and every "metric sensitivity to
    # magic angle" number was meaningless.
    #
    # Centres are now sampled from the **support of the image** (the non-background
    # voxels), so a patch always modulates real signal regardless of seed, and the
    # patch budget targets ``collagen_fraction`` of that support rather than of the
    # whole FOV (of which the anatomy is only a fraction).
    support = _anatomy_support(image, foreground_mask)  # [B, H, W] in {0, 1}

    radius_scale = 1.0 + 2.0 * max(0.0, collagen_fraction - 0.05)
    ry_mean, rx_mean = 0.07 * radius_scale, 0.105 * radius_scale
    # Ellipse area in normalised [-1, 1]^2 coords; the FOV itself has area 4.
    avg_patch_area = math.pi * ry_mean * rx_mean

    # Each subject gets its own orientation field and patches, drawn in its own pass:
    # the patch count follows that subject's support, and no draw depends on B.
    thetas, masks = [], []
    for b in range(B):
        freq = torch.rand(1, 1, 1, device=device) * 3 + 1
        phase = torch.rand(1, 1, 1, device=device) * 2 * math.pi
        # Random smooth orientation field (radians), range [-pi/2, pi/2].
        thetas.append(math.pi / 2 * torch.sin(freq * math.pi * X.unsqueeze(0) + phase))

        sup = support[b : b + 1]
        support_frac = float(sup.mean().clamp(min=1e-3))
        # Patches needed to cover `collagen_fraction` OF THE SUPPORT.
        n_patches = max(1, round(collagen_fraction * support_frac * 4.0 / avg_patch_area))
        n_patches = min(n_patches, 64)
        # Sample patch centres from the support (multinomial over its voxels).
        idx = torch.multinomial(sup.reshape(1, -1) + 1e-12, n_patches, replacement=True)
        # Pixel index -> normalised [-1, 1] coordinate (matches the meshgrid above).
        cy = (2.0 * (idx // W).float() / max(H - 1, 1) - 1.0).view(1, n_patches, 1, 1)
        cx = (2.0 * (idx % W).float() / max(W - 1, 1) - 1.0).view(1, n_patches, 1, 1)
        ry = ((torch.rand(n_patches, device=device) * 0.1 + 0.02) * radius_scale).view(
            1, n_patches, 1, 1
        )
        rx = ((torch.rand(n_patches, device=device) * 0.15 + 0.03) * radius_scale).view(
            1, n_patches, 1, 1
        )
        ellipse = ((X.view(1, 1, H, W) - cx) / rx) ** 2 + ((Y.view(1, 1, H, W) - cy) / ry) ** 2
        # Fibres exist only where there is tissue.
        masks.append((ellipse < 1.0).any(dim=1).float() * sup)
    theta = torch.cat(thetas)  # [B, H, W]
    col_mask = torch.cat(masks)  # [B, H, W]

    # Dipolar interaction term. Relaxed from exp(-10*d) to exp(-3*d)
    # so the magic-angle band is wider — the prior weighting was so
    # narrow (~10° band around 54.7°) that fewer than 1% of pixels
    # in a collagen patch saw the T2 boost at any given moment. The
    # physical narrowness still holds at the macroscopic level
    # (Fullerton 1985) but the demonstrative simulator needs a wider
    # band to be visible.
    dipolar = torch.abs(3.0 * torch.cos(theta) ** 2 - 1.0)
    t2_eff = t2_base + (t2_magic - t2_base) * torch.exp(-3.0 * dipolar)

    # Signal boost relative to baseline decay
    signal_boost = torch.exp(-te / t2_eff) / torch.exp(torch.tensor(-te / t2_base, device=device))

    # Apply only to collagen regions
    modulation = 1.0 + (signal_boost - 1.0) * col_mask  # [B, H, W]
    modulation = modulation.unsqueeze(1)  # [B, 1, H, W]

    return image * modulation


# ──────────────────────────────────────────────────────────────────────
# K-Space RF Interference
# ──────────────────────────────────────────────────────────────────────


def simulate_rf_zipper(
    kspace: torch.Tensor,
    fe_location: float = 0.75,
    intensity_ratio: float = 0.15,
) -> torch.Tensor:
    r"""Simulate RF zipper artifact from continuous-wave interference.

    An external RF source (e.g. a fluorescent light, monitoring equipment)
    leaks into the receiver at a fixed frequency. In readout time that is a
    tone, so every PE line gains a complex exponential across all of its
    :math:`k_x` samples, with a phase that is not locked to the ADC clock:

    .. math::

        K[k_y, k_x] \mathrel{+}= A \, e^{j\phi(k_y)} \, e^{j 2\pi x_0 k_x}

    In image space this is a bright vertical line (zipper) at frequency-encode
    position :math:`x_0`, distinct from random data spikes. Adding the leak to
    a single :math:`k_x` column instead would be a plane wave across the whole
    FOV, not a line. The tone model is shared with the registry ``emi`` axis
    (:func:`~spectramr.infrastructure.physics.emi_degradation.add_readout_tones`).

    Args:
        kspace: Complex k-space ``[B, C, H, W]``.
        fe_location: Image-space position of the zipper line along FE, as a
            fraction of the FOV (0 to 1, 0.5 is the centre column).
        intensity_ratio: RMS of the added interference relative to each
            sample's k-space RMS.

    Returns:
        k-space with zipper artifact ``[B, C, H, W]``.
    """
    W = kspace.shape[-1]
    position = torch.tensor([(fe_location - 0.5) * W], device=kspace.device)
    return add_readout_tones(
        kspace, position, torch.ones(1, device=kspace.device), float(intensity_ratio)
    )


# ──────────────────────────────────────────────────────────────────────
# Motion Simulators (affect ONLY anatomy, NOT markers)
# ──────────────────────────────────────────────────────────────────────


def simulate_rigid_motion_kspace(
    kspace: torch.Tensor,
    max_translation: float = 5.0,
    max_rotation: float = 0.05,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply rigid-body motion as per-PE-line image-space resampling + k-space shift.

    Motion is parameterised by an SE(2) pose
    :math:`\\boldsymbol{\\theta}_t = (R_t, \\mathbf{t}_t)` that evolves
    across the acquisition. Pose number ``t`` is associated with the
    phase-encode line ``k_y = t``. The forward model is

    .. math::

        \\mathbf{y}[k_y] = \\mathcal{F}_x\\!\\bigl(R(\\theta_{k_y})\\,
            \\mathbf{x}\\bigr)[k_y] \\, e^{-2\\pi i \\mathbf{k} \\cdot
            \\mathbf{t}_{k_y}},

    so each PE line sees a different pose --- this is what produces
    ghosting, ringing and blur, distinct from a global translation that
    a reconstructor recovers exactly.

    Translation is applied via the standard Fourier shift theorem;
    rotation is *not* a k-space phase ramp and must be done by
    resampling the image at each PE-line's rotation angle. To avoid
    rotating the entire image H times we discretise the rotation
    schedule into ``n_poses`` evenly-spaced poses across the
    acquisition and gather PE lines from the corresponding rotated
    k-space cube.

    Args:
        kspace: Complex k-space of anatomy ONLY ``[B, C, H, W]``.
        max_translation: Max per-line shift in pixels (uniform per-line draw).
        max_rotation: Max per-line rotation in radians.
        generator: Optional ``torch.Generator`` for deterministic draws.
            When ``None`` the global RNG is used (preserves legacy behaviour).

    Returns:
        Motion-corrupted k-space ``[B, C, H, W]``.
    """
    H, W = kspace.shape[-2], kspace.shape[-1]
    device = kspace.device
    B, C = kspace.shape[0], kspace.shape[1]

    # --- Per-PE-line translation (Fourier shift theorem) ---
    # One row of draws per sample, so sample b's trajectory does not depend on B.
    n_poses = min(8, H)
    draws = per_sample_rand(B, 2 * H + n_poses, generator=generator, device=device) * 2 - 1
    dx = draws[:, :H] * max_translation
    dy = draws[:, H : 2 * H] * max_translation

    kx = centred_freqs(W, device).view(1, 1, 1, -1)  # [1, 1, 1, W]
    ky = centred_freqs(H, device).view(1, 1, -1, 1)  # [1, 1, H, 1]
    dx = dx.view(B, 1, H, 1)
    dy = dy.view(B, 1, H, 1)
    trans_phase = torch.exp(-1j * 2 * math.pi * (kx * dx + ky * dy))  # [B, 1, H, W]

    # --- Per-PE-line rotation (image-space resample, gather by line) ---
    # Line k_y sees pose (R_p, t_ky): rotate the object, take that pose's line, then apply
    # the line's own translation phase. Translating first would rotate every line's
    # translation together with its neighbours'.
    if max_rotation <= 0.0:
        return kspace * trans_phase

    # Discretise into a small number of poses so we don't rotate the
    # image H times. 8 poses captures the dominant ghost spectrum and
    # keeps the cost bounded.
    thetas = draws[:, 2 * H :] * max_rotation

    image = ifft2c(kspace)  # [B, C, H, W]

    # Build [B*n_poses] affine matrices for the n_poses rotation set.
    cos_t = torch.cos(thetas).view(B * n_poses)
    sin_t = torch.sin(thetas).view(B * n_poses)
    affine = torch.zeros(B * n_poses, 2, 3, device=device, dtype=image.real.dtype)
    affine[:, 0, 0] = cos_t
    affine[:, 0, 1] = -sin_t
    affine[:, 1, 0] = sin_t
    affine[:, 1, 1] = cos_t

    img_rep = image.repeat_interleave(n_poses, dim=0)  # [B*n_poses, C, H, W]
    grid = F.affine_grid(affine, img_rep.shape, align_corners=False)
    rot_real = F.grid_sample(
        img_rep.real, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    rot_imag = F.grid_sample(
        img_rep.imag, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    rotated = torch.complex(rot_real, rot_imag).view(B, n_poses, C, H, W)

    # Transform each rotated copy to k-space and gather the PE lines that
    # were acquired during that pose.
    rotated_k = fft2c(rotated.reshape(B * n_poses, C, H, W)).view(B, n_poses, C, H, W)

    # Map each PE line (0..H-1) to its pose index (0..n_poses-1) and
    # gather along the pose axis (dim=1 of [B, n_poses, C, H, W]).
    line_to_pose = torch.linspace(0, n_poses - 1e-6, H, device=device).long()  # [H]
    gather_idx = line_to_pose.view(1, 1, 1, H, 1).expand(B, 1, C, H, W)
    gathered = torch.gather(rotated_k, dim=1, index=gather_idx)  # [B, 1, C, H, W]
    return gathered.squeeze(1) * trans_phase


def simulate_periodic_motion(
    kspace: torch.Tensor,
    amplitude: float = 3.0,
    frequency: float = 2.0,
) -> torch.Tensor:
    """Simulate respiratory/cardiac periodic motion.

    Models periodic bulk motion during k-space acquisition.  Each PE
    line acquires at a different respiratory phase, resulting in a
    sinusoidal displacement pattern that causes ghosting artefacts.

    The ghost spacing in the PE direction equals ``FOV / frequency``.

    Args:
        kspace: Complex k-space ``[B, C, H, W]``.
        amplitude: Peak displacement in pixels.
        frequency: Number of motion cycles during the acquisition.

    Returns:
        Motion-corrupted k-space ``[B, C, H, W]``.
    """
    H, W = kspace.shape[-2], kspace.shape[-1]
    device = kspace.device

    # Sinusoidal displacement per PE line
    pe_idx = torch.linspace(0, 1, H, device=device)
    displacement = amplitude * torch.sin(2 * math.pi * frequency * pe_idx)
    # displacement: [H] → phase per PE line

    # Phase ramp along readout for each PE line
    kx = centred_freqs(W, device).view(1, 1, 1, -1)
    dx = displacement.view(1, 1, -1, 1)  # [1, 1, H, 1]

    phase_shift = torch.exp(-1j * 2 * math.pi * kx * dx)
    return kspace * phase_shift


def simulate_random_shot_motion(
    kspace: torch.Tensor,
    max_shift: float = 2.0,
) -> torch.Tensor:
    """Simulate random inter-shot (per-line) motion.

    Each k-space line is acquired at a slightly different position due
    to involuntary patient movement.  This produces broadband ghosting
    that does not have the periodic structure of respiratory motion.

    Args:
        kspace: Complex k-space ``[B, C, H, W]``.
        max_shift: Maximum per-line shift in pixels.

    Returns:
        Motion-corrupted k-space ``[B, C, H, W]``.
    """
    H, W = kspace.shape[-2], kspace.shape[-1]
    device = kspace.device
    B = kspace.shape[0]

    # Random displacement per PE line (different per batch element)
    dx = (torch.rand(B, H, device=device) * 2 - 1) * max_shift  # [B, H]

    kx = centred_freqs(W, device).view(1, 1, -1)  # [1, 1, W]
    dx = dx.unsqueeze(-1)  # [B, H, 1]

    phase_shift = torch.exp(-1j * 2 * math.pi * kx * dx)  # [B, H, W]

    # Expand to match kspace channels
    C = kspace.shape[1]
    phase_shift = phase_shift.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, H, W]
    return kspace * phase_shift


def simulate_elastic_motion(
    kspace: torch.Tensor,
    im_size: tuple[int, int],
    num_modes: int = 4,
    amplitude: float = 2.0,
    displacement_out: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Simulate non-rigid elastic deformation (e.g. brain pulsation).

    Generates a smooth displacement field from random low-frequency
    Fourier modes and applies it via ``grid_sample`` in image space,
    then transforms back to k-space.

    Args:
        kspace: Complex k-space ``[B, C, H, W]``.
        im_size: Image dimensions ``(H, W)``.
        num_modes: Number of random Fourier modes for the displacement field.
        amplitude: Peak displacement in pixels.
        displacement_out: When given, receives ``"px"``: the applied displacement ``[B, 2, H, W]``
            (x, y) in pixels, so a caller can score motion recovery against ground truth.

    Returns:
        Elastically deformed k-space ``[B, C, H, W]``.
    """
    H, W = im_size
    device = kspace.device
    B = kspace.shape[0]

    # Generate smooth displacement field via random Fourier modes
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )

    disp_x = torch.zeros(B, H, W, device=device)
    disp_y = torch.zeros(B, H, W, device=device)

    for _ in range(num_modes):
        freq_x = torch.rand(B, 1, 1, device=device) * 3
        freq_y = torch.rand(B, 1, 1, device=device) * 3
        phase = torch.rand(B, 1, 1, device=device) * 2 * math.pi
        amp = torch.rand(B, 1, 1, device=device) * amplitude / H  # normalised

        disp_x += amp * torch.sin(
            freq_x * math.pi * X.unsqueeze(0) + freq_y * math.pi * Y.unsqueeze(0) + phase
        )
        disp_y += amp * torch.cos(
            freq_x * math.pi * X.unsqueeze(0) - freq_y * math.pi * Y.unsqueeze(0) + phase
        )

    # Build sampling grid
    grid_y = Y.unsqueeze(0).expand(B, -1, -1) + disp_y
    grid_x = X.unsqueeze(0).expand(B, -1, -1) + disp_x
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]
    if displacement_out is not None:
        # grid units span [-1, 1] over N pixels, so one unit is (N - 1) / 2 pixels.
        displacement_out["px"] = torch.stack(
            [disp_x * (W - 1) / 2.0, disp_y * (H - 1) / 2.0], dim=1
        )

    # Apply in image space (handle complex via real/imag split)
    image = ifft2c(kspace)
    warped_real = F.grid_sample(
        image.real, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    warped_imag = F.grid_sample(
        image.imag, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    warped_image = torch.complex(warped_real, warped_imag)

    return fft2c(warped_image)


_MOTION_SIMULATORS = {
    "rigid": simulate_rigid_motion_kspace,
    "periodic": simulate_periodic_motion,
    "random_shot": simulate_random_shot_motion,
    "elastic": simulate_elastic_motion,
}


#: Axes this simulator's OWN forward pass applies, each modulated by
#: ``_get_effective_cf`` at its call site. These are severity multipliers on the
#: built-in corruption pipeline, NOT standalone operators — which is why they are
#: coarse (``motion`` covers all four ``_MOTION_SIMULATORS``, ``noise`` is the
#: single AWGN stage) and why they are not a subset of ``DEGRADATION_REGISTRY``.
NATIVE_DEGRADATION_AXES: frozenset[str] = frozenset(
    {
        "motion",
        "noise",
        "b0",
        "b1",
        "undersampling",
        "gibbs",
        "chemical_shift",
        "eddy_current",
        "spike_noise",
        "rf_zipper",
        "rf_crosstalk",
        "b0_distortion",
        "gradient_nonlinearity",
        "magic_angle",
    }
)


def known_degradation_axes() -> frozenset[str]:
    """Every name ``progressive_degradations`` / ``degradation_ranges`` may use.

    The union of :data:`NATIVE_DEGRADATION_AXES` and the physics
    ``DEGRADATION_REGISTRY``. Before this was wired the simulator validated
    against the native 14 alone, so **26 registered degradations were
    implemented but unreachable from any config** — ``rigid_motion`` /
    ``pulsatile_motion`` / ``random_motion`` / ``sudden_motion`` /
    ``translation_drift``, ``complex_gaussian`` / ``rician`` / ``spike``, the six
    sampling operators, ``nyquist_ghost``, ``susceptibility``, ``t2star_blur``,
    and the rest. Only sim2rank's sweep could name them.

    The import is deferred on purpose: the registry's 2-D kernels
    (``degradations/slice2d``) import *this* module, so a module-level import here
    would be circular.
    """
    from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY

    return NATIVE_DEGRADATION_AXES | frozenset(DEGRADATION_REGISTRY)


# ──────────────────────────────────────────────────────────────────────
# Full Digital Twin Simulator
# ──────────────────────────────────────────────────────────────────────


#: Native stage -> the flag that enables it. The receiver noise is absent: it always runs, and
#: it is the one native stage with a volume_3d form (white noise in 3-D k-space).
_NATIVE_STAGE_FLAGS: dict[str, str] = {
    "motion": "enable_motion",
    "b0": "enable_b0",
    "b1": "enable_b1",
    "undersampling": "enable_undersampling",
    "gibbs": "enable_gibbs",
    "chemical_shift": "enable_chemical_shift",
    "eddy_current": "enable_eddy_current",
    "spike_noise": "enable_spike_noise",
    "rf_zipper": "enable_rf_zipper",
    "rf_crosstalk": "enable_rf_crosstalk",
    "b0_distortion": "enable_b0_distortion",
    "gradient_nonlinearity": "enable_gradient_nonlinearity",
    "magic_angle": "enable_magic_angle",
}

_KSPACE_3D = (-3, -2, -1)


def _fold_slices(t: torch.Tensor | None) -> torch.Tensor | None:
    """``(B, C, D, H, W)`` -> ``(B*D, C, H, W)``: each slice of a stack is its own 2-D sample."""
    if t is None or t.dim() != 5:
        return t
    b, c, d, h, w = t.shape
    return t.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)


def _unfold_slices(t: torch.Tensor, b: int, d: int) -> torch.Tensor:
    """The exact inverse of :func:`_fold_slices`; a batch-1 broadcast tensor gains a unit D."""
    if t.shape[0] == b * d:
        return t.reshape(b, d, *t.shape[1:]).permute(0, 2, 1, 3, 4)
    if t.shape[0] == 1:
        return t.unsqueeze(2)
    raise ValueError(f"cannot unfold batch {t.shape[0]} into {b} subjects x {d} slices")


def _reject_profile_conflicts(
    profile: ScannerProfile,
    im_size: tuple[int, int],
    b0_dwell_time: float,
    chemical_shift_pixels: float,
) -> None:
    """With a profile, a native dwell or fat shift away from its default must equal the profile's.

    Compared by value against the schema default: the profile owns both quantities, so a
    differing explicit value is a second owner.
    """
    from spectramr.config.schemas.physics import DigitalTwinConfig

    defaults = DigitalTwinConfig.model_fields
    owned = {
        "b0_dwell_time": (b0_dwell_time, profile.readout_dwell_s(im_size[-1])),
        "chemical_shift_pixels": (chemical_shift_pixels, profile.chemical_shift_max_px()),
    }
    for key, (given, derived) in owned.items():
        default = defaults[key].default
        if not math.isclose(given, default) and not math.isclose(given, derived):
            raise ValueError(
                f"physics.digital_twin.{key}={given} conflicts with the "
                f"physics.digital_twin.acquisition profile {profile.name!r}, which derives "
                f"{key}={derived:.6g} from B0 and readout_bw_hz_per_px. The profile owns it: "
                f"drop {key} or the acquisition block."
            )


class DigitalTwinSimulator(nn.Module):
    """Joint forward physics model with correct artifact separation.

    Physically correct corruption pipeline:
        1. Motion → anatomy ONLY (brain moves, markers don't)
        2. Embed markers into motion-corrupted anatomy
        3. Scanner artifacts → JOINT image:
           a. B0 inhomogeneity (phase distortion in k-space)
           b. Undersampling (Cartesian line skipping)
           c. Gibbs ringing (k-space truncation)
           d. Complex AWGN (→ Rician noise in magnitude)
           e. Chemical shift (fat-water displacement)
           f. Eddy currents (gradient-induced phase)
           g. Spike noise (RF zipper artefact)
        4. B1 bias field (image-space intensity modulation)
        5. RF crosstalk (multi-slice bleed-through)

    Args:
        im_size: Image dimensions (H, W).
        marker_type: ``"gaussian"`` or ``"cross"``.
        motion_type: ``"rigid"``, ``"periodic"``, ``"random_shot"``,
            or ``"elastic"``.
        snr_range: (min, max) target SNR in dB.
        max_translation: Max motion shift in pixels.
        max_rotation: Max rotation in radians.
        b0_strength: B0 inhomogeneity strength (0-1).
        b1_strength: B1 bias field strength (0-1).
        acceleration: Undersampling acceleration factor (1=fully sampled).
        gibbs_truncation: K-space truncation fraction (1=no Gibbs).
        enable_motion: Enable motion corruption.
        enable_b0: Enable B0 inhomogeneity.
        enable_b1: Enable B1 bias field.
        enable_undersampling: Enable k-space undersampling.
        enable_gibbs: Enable Gibbs ringing.
        enable_chemical_shift: Enable fat-water chemical shift.
        chemical_shift_pixels: Chemical shift displacement in pixels.
        enable_eddy_current: Enable eddy current artifacts.
        eddy_strength: Eddy current phase strength.
        enable_spike_noise: Enable RF spike noise.
        spike_probability: Per-element spike probability.
        enable_rf_crosstalk: Enable multi-slice RF crosstalk.
        crosstalk_fraction: Crosstalk signal fraction.
    """

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        marker_type: str = "gaussian",
        motion_type: str = "rigid",
        motion_composite: list[str] | None = None,
        motion_severity: float = 1.0,
        snr_range: tuple[float, float] = (10.0, 25.0),
        max_translation: float = 5.0,
        max_rotation: float = 0.05,
        motion_amplitude: float = 3.0,
        motion_frequency: float = 2.0,
        elastic_modes: int = 4,
        b0_strength: float = 0.3,
        b1_strength: float = 0.2,
        acceleration: float = 1.0,
        gibbs_truncation: float = 1.0,
        enable_motion: bool = True,
        enable_b0: bool = True,
        enable_b1: bool = True,
        enable_undersampling: bool = False,
        enable_gibbs: bool = False,
        enable_chemical_shift: bool = False,
        chemical_shift_pixels: float = 3.0,
        enable_eddy_current: bool = False,
        eddy_strength: float = 0.1,
        enable_spike_noise: bool = False,
        spike_probability: float = 0.01,
        enable_rf_crosstalk: bool = False,
        crosstalk_fraction: float = 0.05,
        # ── New corruption modules ──────────────────────
        enable_b0_distortion: bool = False,
        b0_dwell_time: float = 1e-5,
        enable_gradient_nonlinearity: bool = False,
        gradient_nonlinearity_severity: float = 2e-6,
        enable_rf_zipper: bool = False,
        rf_zipper_location: float = 0.75,
        rf_zipper_intensity: float = 0.15,
        enable_magic_angle: bool = False,
        magic_angle_te: float = 20.0,
        magic_angle_t2_base: float = 10.0,
        magic_angle_t2_magic: float = 30.0,
        magic_angle_collagen_fraction: float = 0.05,
        magic_angle_tissue_class: str = "msk",
        progressive_degradations: list[str] | None = None,
        degradation_ranges: dict[str, tuple[float, float]] | None = None,
        degradation_schedule: str = "linear",
        degradation_schedules: dict[str, str] | None = None,
        degradation_schedule_power: float = 2.0,
        # ── Wave 5 (digital-twin review) opt-in architectural flags ──
        marker_is_water_only: bool = False,
        per_channel_marker_prior: bool = False,
        coherent_3d_motion: bool = False,
        scanner_profile: ScannerProfile | None = None,
        mask_axis: str | None = None,
        phase_encode_axis: int | None = None,
    ) -> None:
        super().__init__()
        self.im_size = im_size
        self.marker_type = marker_type
        self.motion_type = motion_type
        # Composite motion: when non-empty, the listed motion models are
        # applied *sequentially* in k-space (each contributes its own per-PE
        # -line perturbation), producing mixed patterns no single type can —
        # e.g. ['rigid', 'periodic', 'random_shot'] = bulk drift +
        # respiratory bob + sparse spikes. Empty → single ``motion_type``.
        self.motion_composite = list(motion_composite) if motion_composite else []
        # Sustained motion-amplitude multiplier (translation/rotation/amplitude,
        # NOT frequency). This is a *constant floor*, not a schedule: the
        # degradation must stay large enough throughout to keep the input far
        # from the target — annealing down to a mild regime would let the
        # near-clean input leak the target (identity collapse) at the end of
        # training. Set >1 to harden; 1.0 = nominal.
        self._motion_severity = float(max(motion_severity, 0.0))
        self.snr_range = snr_range
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.motion_amplitude = motion_amplitude
        self.motion_frequency = motion_frequency
        self.elastic_modes = elastic_modes
        self.b0_strength = b0_strength
        self.b1_strength = b1_strength
        self.acceleration = acceleration
        self.gibbs_truncation = gibbs_truncation
        # ``mask_axis`` unset keeps the historical rows (dim -2) mask; set, it resolves
        # through the dataset's phase-encode axis and raises when that is undeclared.
        if mask_axis is None:
            self.undersampling_line_axis = "y"
        else:
            self.undersampling_line_axis = line_axis_for(mask_axis, phase_encode_axis)
        self.undersampling_line_dim = LINE_AXIS_DIM[self.undersampling_line_axis]

        self.enable_motion = enable_motion
        self.enable_b0 = enable_b0
        self.enable_b1 = enable_b1
        self.enable_undersampling = enable_undersampling
        self.enable_gibbs = enable_gibbs
        self.enable_chemical_shift = enable_chemical_shift
        self.chemical_shift_pixels = chemical_shift_pixels
        self.enable_eddy_current = enable_eddy_current
        self.eddy_strength = eddy_strength
        self.enable_spike_noise = enable_spike_noise
        self.spike_probability = spike_probability
        self.enable_rf_crosstalk = enable_rf_crosstalk
        self.crosstalk_fraction = crosstalk_fraction

        self.enable_b0_distortion = enable_b0_distortion
        self.b0_dwell_time = b0_dwell_time
        self.enable_gradient_nonlinearity = enable_gradient_nonlinearity
        self.gradient_nonlinearity_severity = gradient_nonlinearity_severity
        self.enable_rf_zipper = enable_rf_zipper
        self.rf_zipper_location = rf_zipper_location
        self.rf_zipper_intensity = rf_zipper_intensity
        self.enable_magic_angle = enable_magic_angle
        self.magic_angle_te = magic_angle_te
        self.magic_angle_t2_base = magic_angle_t2_base
        self.magic_angle_t2_magic = magic_angle_t2_magic
        self.magic_angle_collagen_fraction = magic_angle_collagen_fraction
        self.magic_angle_tissue_class = magic_angle_tissue_class

        # Wave 5 architectural flags.
        # M6 --- markers don't have lipid content (CuSO4/MnCl2 phantoms);
        # if True, chemical shift is applied to anatomy ONLY and embedded
        # markers are added pristine afterwards.
        self.marker_is_water_only = marker_is_water_only
        # Mod3 --- multi-contrast embedder returns per-channel marker_prior.
        self.per_channel_marker_prior = per_channel_marker_prior
        # Mod4 --- 3-D inputs use one motion trajectory per subject instead
        # of independent per-slice motion.
        self.coherent_3d_motion = coherent_3d_motion
        # Conditions the stage-5 registry operators and owns the native stages' field, dwell,
        # shift model, fat shift and fat fraction; None keeps every stage on its own knobs.
        self.scanner_profile = scanner_profile
        self._b0_sequence = "epi"
        if scanner_profile is not None:
            _reject_profile_conflicts(scanner_profile, im_size, b0_dwell_time, chemical_shift_pixels)
            # Resolved here so a non-Cartesian profile fails at build, not mid-training.
            if enable_b0_distortion or "b0_geometric_distortion" in (progressive_degradations or []):
                self._b0_sequence = b0_shift_sequence(scanner_profile.sequence)
            if scanner_profile.geometry.mode is AcquisitionMode.VOLUME_3D:
                self._reject_volume_native_stages()

        # Range configurations for progressive cold diffusion
        self.progressive_degradations = progressive_degradations or []
        self.degradation_ranges = degradation_ranges or {}
        # Severity ramp shape over diffusion timesteps -- the counterpart of
        # undersampling's acceleration_schedule. Before this the remap from the
        # corruption factor onto (vmin, vmax) was linear and unstatable.
        self.degradation_schedule = str(
            getattr(degradation_schedule, "value", degradation_schedule)
        )
        self.degradation_schedules = {
            k: str(getattr(v, "value", v)) for k, v in (degradation_schedules or {}).items()
        }
        self.degradation_schedule_power = float(degradation_schedule_power)

        # Validate degradation-feature knobs against every name the forward pass
        # can act on. A mistyped key (e.g. 'b_0') would otherwise be read and
        # silently ignored -> the advertised range becomes a no-op (pitfall
        # #15). Raise at construction, mirroring the motion_type /
        # motion_composite validation below.
        #
        # The legal set is the UNION of the native axes and DEGRADATION_REGISTRY
        # (see known_degradation_axes). It used to be the native 14 alone, which
        # is what made 26 registered degradations unnameable from any config.
        _known = known_degradation_axes()
        bad_prog = [d for d in self.progressive_degradations if d not in _known]
        if bad_prog:
            raise ValueError(
                f"Unknown progressive_degradations entries {bad_prog}. Valid: {sorted(_known)}"
            )
        bad_range = [k for k in self.degradation_ranges if k not in _known]
        if bad_range:
            raise ValueError(
                f"Unknown degradation_ranges keys {bad_range}. Valid: {sorted(_known)}"
            )
        # Same check for the per-axis schedules. The schema validates these too,
        # but half this class's callers construct it directly (tests, the chain
        # replay, the sweeps), and a key nothing looks up is silently inert.
        bad_sched = [k for k in self.degradation_schedules if k not in _known]
        if bad_sched:
            raise ValueError(
                f"Unknown degradation_schedules keys {bad_sched}. Valid: {sorted(_known)}"
            )
        # Registry-only names are applied as standalone operators after the
        # native pipeline (see the forward pass). A name present in BOTH banks —
        # b0, gibbs, chemical_shift, gradient_nonlinearity, magic_angle — is
        # handled natively, because the native stage is diffusion-coupled and
        # already wired; applying the registry copy as well would corrupt twice
        # at one severity. Native therefore wins, and this tuple is the
        # difference, not the intersection.
        self._registry_degradations: tuple[str, ...] = tuple(
            d for d in self.progressive_degradations if d not in NATIVE_DEGRADATION_AXES
        )
        # #244: the range bounds are now READ (see _get_effective_cf), so they must
        # also be VALIDATED — an out-of-range or inverted pair would silently produce
        # a negative or >1 severity multiplier.
        for feat, rng in self.degradation_ranges.items():
            if len(rng) != 2:
                raise ValueError(
                    f"degradation_ranges[{feat!r}] must be a (vmin, vmax) pair; got {rng!r}"
                )
            vmin, vmax = float(rng[0]), float(rng[1])
            if not (0.0 <= vmin <= vmax <= 1.0):
                raise ValueError(
                    f"degradation_ranges[{feat!r}] = ({vmin}, {vmax}) must satisfy "
                    "0 <= vmin <= vmax <= 1 (it remaps the corruption factor)."
                )

        if motion_type not in _MOTION_SIMULATORS:
            raise ValueError(
                f"Unknown motion_type={motion_type!r}. Valid: {list(_MOTION_SIMULATORS)}"
            )
        bad = [m for m in self.motion_composite if m not in _MOTION_SIMULATORS]
        if bad:
            raise ValueError(
                f"Unknown motion_composite entries {bad}. Valid: {list(_MOTION_SIMULATORS)}"
            )

        self.embedder = CornerFiducialEmbedder(
            im_size=im_size,
            marker_type=marker_type,
            per_channel_marker_prior=per_channel_marker_prior,
        )

        # Binary k-space sampling mask from the most recent forward pass when
        # undersampling is active, else None. Reconstruction strategies read it
        # to drive the model's data-consistency layers (see
        # ``simulator_builder.undersampling_mask_kwargs``).
        self.last_undersampling_mask: torch.Tensor | None = None
        # Applied B0 field (Hz) + dominant PE pixel-shift field, exposed for
        # field self-consistency scoring; only populated when B0 geometric
        # distortion fires this pass (see forward + _score_field). After a
        # multislice_2d forward both stay in the folded (B*D, ...) slice layout.
        self.last_b0_field: torch.Tensor | None = None
        self.last_pe_shift_field: torch.Tensor | None = None
        self.last_elastic_displacement_px: torch.Tensor | None = None

        logger.info(
            "[DigitalTwinSimulator] im_size=%s, marker=%s, motion=%s(%s), "
            "B0=%s(%.1f), B1=%s(%.1f), undersamp=%s(%.1fx), gibbs=%s(%.2f), "
            "chem_shift=%s(%.1f), eddy=%s(%.2f), spikes=%s(%.3f), crosstalk=%s(%.3f), "
            "b0_distort=%s, grad_nl=%s(%.1e), zipper=%s(%.2f), magic=%s, "
            "progressive=%s, ranges=%s, schedule=%s, per_axis_schedules=%s, "
            "scanner_profile=%s, undersamp_line_axis=%s",
            im_size,
            marker_type,
            motion_type,
            enable_motion,
            enable_b0,
            b0_strength,
            enable_b1,
            b1_strength,
            enable_undersampling,
            acceleration,
            enable_gibbs,
            gibbs_truncation,
            enable_chemical_shift,
            chemical_shift_pixels,
            enable_eddy_current,
            eddy_strength,
            enable_spike_noise,
            spike_probability,
            enable_rf_crosstalk,
            crosstalk_fraction,
            enable_b0_distortion,
            enable_gradient_nonlinearity,
            gradient_nonlinearity_severity,
            enable_rf_zipper,
            rf_zipper_intensity,
            enable_magic_angle,
            # #244: these two are now load-bearing (progressive vs static, plus a
            # severity floor), so they are stamped into the run's provenance.
            self.progressive_degradations,
            self.degradation_ranges,
            self.degradation_schedule,
            self.degradation_schedules,
            None if scanner_profile is None else scanner_profile.to_meta(),
            self.undersampling_line_axis,
        )

    @property
    def marker_mask(self) -> torch.Tensor:
        """Binary mask highlighting marker regions [1, 1, H, W]."""
        return self.embedder.marker_mask

    def _get_effective_cf(self, feat: str, base_cf: float) -> float:
        """Map the diffusion corruption factor onto one degradation's severity.

        Two knobs, with genuinely distinct effects (issue #244):

        * ``progressive_degradations`` — the features **coupled to the diffusion
          schedule**. When the list is non-empty, a feature *absent* from it is
          **static**: it is held at its configured strength (effective cf = 1.0)
          at every timestep instead of ramping with ``base_cf``. An empty list
          (the default) means *every* feature is progressive, which is the
          behaviour every existing config relies on.
        * ``degradation_ranges[feat] = (vmin, vmax)`` — remaps the effective cf
          onto ``[vmin, vmax]``. ``vmin > 0`` is a genuine severity **floor**:
          the degradation never fully vanishes, even at ``cf = 0``.

        Before #244 the two branches were byte-identical: an unlisted feature
        returned ``base_cf``, and a listed one with the range ``(0.0, 1.0)`` —
        the range *every* axis config declares — returned
        ``0 + 1 * base_cf = base_cf``. The whole mechanism was decorative
        (pitfall #15: an advertised knob that nothing reads).
        """
        progressive = not self.progressive_degradations or (feat in self.progressive_degradations)
        eff = base_cf if progressive else 1.0
        # Shape the ramp before the range remap. 'linear' returns eff unchanged
        # by early return, so every config written before this field existed is
        # bit-identical. A degenerate range (t, t) still pins t under EVERY
        # schedule, since (vmax - vmin) is 0 -- that is what keeps a fitted
        # DegradationChain replaying exactly (see degradation_chain.py).
        eff = shape_severity_ratio(
            eff,
            self._schedule_for(feat),
            self.degradation_schedule_power,
        )
        vmin, vmax = self.degradation_ranges.get(feat, (0.0, 1.0))
        return vmin + (vmax - vmin) * eff

    def _b0_field_scale(self) -> float:
        """Off-resonance scale ``B0 / 3 T`` of the profile; 1.0 without one."""
        return 1.0 if self.scanner_profile is None else self.scanner_profile.b0_field_scale()

    @property
    def acquisition_geometry(self) -> AcquisitionGeometry | None:
        """The geometry the scanner profile declares; ``None`` without a profile."""
        return None if self.scanner_profile is None else self.scanner_profile.geometry

    def _input_geometry(
        self, x: torch.Tensor, geometry: AcquisitionGeometry | None
    ) -> AcquisitionGeometry:
        """The acquisition ``x`` came from: the per-call geometry or the profile's, never both."""
        if geometry is not None and self.scanner_profile is not None:
            raise ValueError(
                "forward(geometry=...) and physics.digital_twin.acquisition both declare the "
                "acquisition; the scanner profile carries its geometry, so pass one, not both"
            )
        return resolve_geometry(x, geometry if geometry is not None else self.acquisition_geometry)

    def _reject_volume_native_stages(self, *inputs: str) -> None:
        """volume_3d has no native-stage implementation yet; ``inputs`` names native-only inputs."""
        stages = [s for s, flag in _NATIVE_STAGE_FLAGS.items() if getattr(self, flag)]
        if stages or inputs:
            raise ValueError(
                f"volume_3d input: native twin stage(s) {stages} and input(s) {list(inputs)} "
                "have no 3-D implementation until the degradation-bank consolidation "
                "(TODO/plan_degradation_bank_consolidation.md). Disable them and use the "
                "registry axes (progressive_degradations), or declare multislice_2d."
            )

    def _schedule_for(self, feat: str) -> str:
        """Per-axis ramp shape, falling back to the arm-wide default."""
        return self.degradation_schedules.get(feat, self.degradation_schedule)

    def severity_at_timestep(self, axis: str, timestep: int, num_timesteps: int) -> float:
        """Severity of ``axis`` at a diffusion timestep, without a forward pass.

        The counterpart of ``KSpaceAccelerator.get_acceleration_factor(t)``: it
        answers what a timestep-aware training will actually apply, so a ladder
        can be inspected, logged and asserted before any data moves.

        A thin wrapper over ``_get_effective_cf`` on purpose -- a second
        implementation of the ramp is how a scheduled severity and an applied
        one drift apart, which is exactly the divergence #787 had to close
        between a fitted chain and its replay.
        """
        if num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
        ratio = 0.0 if num_timesteps == 1 else float(timestep) / (num_timesteps - 1)
        return self._get_effective_cf(axis, ratio)

    @classmethod
    def from_config(
        cls,
        dt: Any,
        im_size: tuple[int, int],
        *,
        field_strength_t: float | None = None,
        phase_encode_axis: int | None = None,
    ) -> DigitalTwinSimulator:
        """Build a simulator from a ``DigitalTwinConfig`` (the SSOT mapping).

        Single source of truth for config→simulator, used by both
        ``build_simulator_from_config`` (VF strategies) and the transversal
        ``DigitalTwinDegradation`` data transform, so the two never drift.
        ``field_strength_t`` is ``physics.field_strength``, the B0 of the profile a
        declared ``dt.acquisition`` block builds; it is required when the block is.
        ``phase_encode_axis`` is ``data.acquisition.phase_encode_axis``, which
        ``physics.digital_twin.mask_axis`` resolves through.
        """
        acquisition = getattr(dt, "acquisition", None)
        profile = None
        if acquisition is not None:
            if field_strength_t is None:
                raise ValueError(
                    "physics.digital_twin.acquisition is declared, so the twin needs "
                    "physics.field_strength (the profile's B0): pass field_strength_t."
                )
            from spectramr.infrastructure.physics.scanner_profile import ScannerProfile

            profile = ScannerProfile.from_acquisition_config(
                acquisition, field_strength_t=field_strength_t
            )
        return cls(
            scanner_profile=profile,
            im_size=im_size,
            marker_type=dt.marker_type,
            motion_type=dt.motion_type,
            motion_composite=list(getattr(dt, "motion_composite", []) or []),
            motion_severity=getattr(dt, "motion_severity", 1.0),
            # Forward the diffusion-coupling fields so per-degradation ranges
            # are honoured on the from_config (VF / transform) path too. They
            # were previously dropped, leaving the schema fields dead.
            progressive_degradations=list(getattr(dt, "progressive_degradations", []) or []),
            degradation_ranges=dict(getattr(dt, "degradation_ranges", {}) or {}),
            # Same funnel that silently dropped the two fields above until it
            # was fixed; these three go through it too.
            degradation_schedule=getattr(dt, "degradation_schedule", "linear"),
            degradation_schedules=dict(getattr(dt, "degradation_schedules", {}) or {}),
            degradation_schedule_power=getattr(dt, "degradation_schedule_power", 2.0),
            snr_range=(dt.snr_min, dt.snr_max),
            max_translation=dt.max_translation,
            max_rotation=dt.max_rotation,
            motion_amplitude=dt.motion_amplitude,
            motion_frequency=dt.motion_frequency,
            elastic_modes=dt.elastic_modes,
            b0_strength=dt.b0_strength,
            b1_strength=dt.b1_strength,
            acceleration=dt.acceleration,
            mask_axis=dt.mask_axis,
            phase_encode_axis=phase_encode_axis,
            gibbs_truncation=dt.gibbs_truncation,
            enable_motion=dt.enable_motion,
            enable_b0=dt.enable_b0,
            enable_b1=dt.enable_b1,
            enable_undersampling=dt.enable_undersampling,
            enable_gibbs=dt.enable_gibbs,
            enable_chemical_shift=dt.enable_chemical_shift,
            chemical_shift_pixels=dt.chemical_shift_pixels,
            enable_eddy_current=dt.enable_eddy_current,
            eddy_strength=dt.eddy_strength,
            enable_spike_noise=dt.enable_spike_noise,
            spike_probability=dt.spike_probability,
            enable_rf_crosstalk=dt.enable_rf_crosstalk,
            crosstalk_fraction=dt.crosstalk_fraction,
            enable_b0_distortion=dt.enable_b0_distortion,
            b0_dwell_time=dt.b0_dwell_time,
            enable_gradient_nonlinearity=dt.enable_gradient_nonlinearity,
            gradient_nonlinearity_severity=dt.gradient_nonlinearity_severity,
            enable_rf_zipper=dt.enable_rf_zipper,
            rf_zipper_location=dt.rf_zipper_location,
            rf_zipper_intensity=dt.rf_zipper_intensity,
            enable_magic_angle=dt.enable_magic_angle,
            magic_angle_te=dt.magic_angle_te,
            magic_angle_t2_base=dt.magic_angle_t2_base,
            magic_angle_t2_magic=dt.magic_angle_t2_magic,
            magic_angle_collagen_fraction=dt.magic_angle_collagen_fraction,
        )

    def set_motion_severity(self, severity: float) -> None:
        """Set the constant amplitude multiplier applied to every motion model.

        ``severity=1.0`` is nominal; >1 amplifies, <1 attenuates. Affects
        translation, rotation and amplitude (NOT frequency, which is a rate).
        This is a sustained floor — keep it large enough that the corrupted
        input never collapses toward the clean target (see __init__).
        """
        self._motion_severity = float(max(severity, 0.0))

    @contextmanager
    def at_acceleration(self, acceleration: float) -> Iterator[None]:
        """Run the twin at ``acceleration`` with undersampling on, then restore it.

        The out-of-distribution readout
        (``strategies.ood_acceleration_readout``) scores the model at rungs the
        arm never trained at. The two attributes are restored in ``finally`` so
        a failing pass cannot leave the twin at the OOD rate, and the mask
        captured during the pass is dropped on exit so no later reader mistakes
        it for the in-distribution one (forward clears it again on entry).
        """
        if acceleration <= 1.0:
            raise ValueError(
                f"at_acceleration({acceleration}) is not an undersampling rate; "
                "an out-of-distribution rung must exceed 1.0."
            )
        previous = (self.acceleration, self.enable_undersampling)
        self.acceleration = float(acceleration)
        self.enable_undersampling = True
        try:
            yield
        finally:
            self.acceleration, self.enable_undersampling = previous
            self.last_undersampling_mask = None

    def degrade_at(
        self,
        clean_anatomy: torch.Tensor,
        t: int | float,
        num_timesteps: int,
        *,
        degradation_only: bool = True,
        snr_target: float | None = None,
        coil_sensitivities: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Timestep-indexed forward operator :math:`D(x_0, t)` for diffusion.

        Maps a diffusion timestep ``t in [0, num_timesteps]`` to the
        ``corruption_factor`` ``cf = t / num_timesteps`` and returns ONLY the
        corrupted image. This is the seam that lets a (cold-)diffusion strategy
        use the twin as its degradation operator: with motion now coupled to
        ``cf``, ``degrade_at(x, 0, T) ≈ x`` and ``degrade_at(x, T, T)`` is the
        fully-corrupted endpoint at the constant ``motion_severity`` floor. The
        same ``t`` should populate the conditioning context's ``diffusion_t``
        so the network is told the operator level.
        """
        if num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
        cf = min(max(float(t) / float(num_timesteps), 0.0), 1.0)
        corrupted, _marker, _clean = self.forward(
            clean_anatomy,
            snr_target=snr_target,
            coil_sensitivities=coil_sensitivities,
            corruption_factor=cf,
            degradation_only=degradation_only,
            seed=seed,
        )
        return corrupted

    def _apply_single_motion(
        self, kspace: torch.Tensor, motion_type: str, cf: float = 1.0
    ) -> torch.Tensor:
        """Apply one motion model, scaled by severity AND corruption_factor.

        ``cf`` (the diffusion ``corruption_factor``) is routed through
        ``_get_effective_cf`` exactly like every other degradation, so the
        forward operator is continuous to identity at ``cf=0`` (required for
        cold-diffusion ``D(x, 0) = x``). At the VF default ``cf=1.0`` this is a
        no-op and motion stays at the constant ``motion_severity`` floor.
        """
        s = self._motion_severity * self._get_effective_cf("motion", cf)
        if motion_type == "rigid":
            return simulate_rigid_motion_kspace(
                kspace,
                max_translation=self.max_translation * s,
                max_rotation=self.max_rotation * s,
            )
        elif motion_type == "periodic":
            return simulate_periodic_motion(
                kspace,
                amplitude=self.motion_amplitude * s,
                frequency=self.motion_frequency,
            )
        elif motion_type == "random_shot":
            return simulate_random_shot_motion(kspace, max_shift=self.motion_amplitude * s)
        elif motion_type == "elastic":
            sink: dict[str, torch.Tensor] = {}
            out = simulate_elastic_motion(
                kspace,
                im_size=self.im_size,
                num_modes=self.elastic_modes,
                amplitude=self.motion_amplitude * s,
                displacement_out=sink,
            )
            self.last_elastic_displacement_px = sink["px"]
            return out
        else:
            raise ValueError(f"Unknown motion_type: {motion_type!r}")

    def _apply_motion(self, anatomy_kspace: torch.Tensor, cf: float = 1.0) -> torch.Tensor:
        """Dispatch to the configured motion model(s).

        When ``motion_composite`` is set, each listed model is applied in
        sequence (composing in k-space); otherwise the single ``motion_type``
        is used. The constant ``motion_severity`` floor and the diffusion
        ``corruption_factor`` (``cf``) both scale every model.
        """
        models = self.motion_composite or [self.motion_type]
        kspace = anatomy_kspace
        for motion_type in models:
            kspace = self._apply_single_motion(kspace, motion_type, cf)
        return kspace

    def forward(
        self,
        clean_anatomy: torch.Tensor,
        snr_target: float | None = None,
        coil_sensitivities: torch.Tensor | None = None,
        corruption_factor: float = 1.0,
        degradation_only: bool = False,
        external_b0_field: torch.Tensor | None = None,
        seed: int | None = None,
        geometry: AcquisitionGeometry | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full Digital Twin forward model (optionally seeded).

        ``seed`` makes the whole pipeline deterministic: the same
        ``(clean_anatomy, corruption_factor, seed)`` yields the same corrupted
        output, batch after batch and process after process. Every stochastic
        stage — the motion poses, the B0/B1 field draws, the undersampling mask,
        the spike/zipper phases, the magic-angle fibre patches, the AWGN — reads
        the *global* torch RNG, so the seed is applied by forking the RNG for the
        duration of the call (``torch.random.fork_rng``) and re-seeding it inside
        the fork. Nothing leaks back out: a caller's RNG stream is untouched, so
        seeding the simulator cannot perturb dataloader shuffling or dropout.

        ``seed=None`` (the default) preserves the legacy behaviour of drawing from
        the ambient global RNG — training paths want fresh corruption each step;
        it is the *sweep* that needs reproducibility.

        Handles single-coil, multi-coil, and multi-contrast data:
          - Single-coil (C=1): standard pipeline.
          - Multi-coil (C>1 + coil_sensitivities): RSS coil-combine
            to image space, embed markers, then re-expand per-coil
            via coil sensitivity maps.
          - Multi-contrast (C>1, no coil maps): per-contrast
            intensity-matched marker embedding.

        A volume is read only in the canonical ``(B, C, D, H, W)`` layout (dim -3 the
        slice or kz axis) and only with an acquisition mode, from ``geometry`` or the
        scanner profile (not both). ``multislice_2d`` runs the native stages on each slice
        as its own 2-D sample and the registry axes on the stack; ``volume_3d`` runs the
        registry axes and the receiver noise only, and raises on any enabled native stage.

        Args:
            clean_anatomy: Clean complex image [B, C, H, W], or [B, C, D, H, W] with a mode.
            snr_target: Target SNR in dB.  If None, sampled from snr_range.
            coil_sensitivities: Optional [1|B, C, H, W] complex coil
                sensitivity maps ([B, C, D, H, W] for a stack).  When provided,
                triggers multi-coil marker embedding.
            geometry: The acquisition of a 5-D input when no scanner profile declares it.

        Returns:
            corrupted_image: Fully corrupted joint image [B, C, H, W].
            marker_prior: Ideal marker (contrast-scaled) [1, 1, H, W].
            joint_clean: Clean anatomy + markers (for supervision).

        forward method for DigitalTwinSimulator.

        Executes PyTorch tensor operations.

        Args:
            clean_anatomy (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            snr_target (float | None): Expected input tensor.
            coil_sensitivities (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            corruption_factor (float): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        if seed is None:
            return self._forward_impl(
                clean_anatomy,
                snr_target=snr_target,
                coil_sensitivities=coil_sensitivities,
                corruption_factor=corruption_factor,
                degradation_only=degradation_only,
                external_b0_field=external_b0_field,
                geometry=geometry,
            )

        device = clean_anatomy.device
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed(int(seed))
            return self._forward_impl(
                clean_anatomy,
                snr_target=snr_target,
                coil_sensitivities=coil_sensitivities,
                corruption_factor=corruption_factor,
                degradation_only=degradation_only,
                external_b0_field=external_b0_field,
                seed=int(seed),
                geometry=geometry,
            )

    def _forward_impl(
        self,
        clean_anatomy: torch.Tensor,
        snr_target: float | None = None,
        coil_sensitivities: torch.Tensor | None = None,
        corruption_factor: float = 1.0,
        degradation_only: bool = False,
        external_b0_field: torch.Tensor | None = None,
        seed: int | None = None,
        geometry: AcquisitionGeometry | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Body of :meth:`forward` (see it for the contract); ``seed`` is the registry master seed."""
        # corruption_factor (cf) is the single diffusion-time knob; it drives
        # EVERY degradation (motion included) via _get_effective_cf so the
        # operator is continuous to identity at cf=0. cf=1.0 (VF default) is a
        # no-op leaving the constant motion_severity floor in force.
        cf = corruption_factor

        # Clear any mask/field captured by a previous call; only repopulated
        # below if the corresponding degradation fires this pass (never leak a
        # stale reference into the next batch's field/mask scoring).
        self.last_undersampling_mask = None
        self.last_b0_field = None
        self.last_pe_shift_field = None
        self.last_elastic_displacement_px = None

        mode = self._input_geometry(clean_anatomy, geometry).mode
        if mode is AcquisitionMode.VOLUME_3D:
            inputs = {
                "markers (degradation_only=False)": not degradation_only,
                "coil_sensitivities": coil_sensitivities is not None,
                "external_b0_field": external_b0_field is not None,
            }
            self._reject_volume_native_stages(*(k for k, given in inputs.items() if given))
            kspace = fftnc(clean_anatomy, _KSPACE_3D)
            kspace = self._add_receiver_noise(kspace, snr_target, cf, (-4, -3, -2, -1))
            corrupted_image = ifftnc(kspace, _KSPACE_3D)
            marker_prior = torch.zeros_like(clean_anatomy[:, :1])
            joint_clean = clean_anatomy
        elif mode is AcquisitionMode.MULTISLICE_2D:
            b, d = clean_anatomy.shape[0], clean_anatomy.shape[2]
            corrupted_image, marker_prior, joint_clean = self._native_stages(
                _fold_slices(clean_anatomy),
                snr_target,
                _fold_slices(coil_sensitivities),
                cf,
                degradation_only,
                _fold_slices(external_b0_field),
                depth=d,
            )
            corrupted_image = _unfold_slices(corrupted_image, b, d)
            marker_prior = _unfold_slices(marker_prior, b, d)
            joint_clean = _unfold_slices(joint_clean, b, d)
        else:
            corrupted_image, marker_prior, joint_clean = self._native_stages(
                clean_anatomy,
                snr_target,
                coil_sensitivities,
                cf,
                degradation_only,
                external_b0_field,
                depth=None,
            )
        corrupted_image = self._apply_registry_axes(corrupted_image, cf, seed, geometry)
        return corrupted_image, marker_prior, joint_clean

    def _native_stages(
        self,
        clean_anatomy: torch.Tensor,
        snr_target: float | torch.Tensor | None,
        coil_sensitivities: torch.Tensor | None,
        cf: float,
        degradation_only: bool,
        external_b0_field: torch.Tensor | None,
        *,
        depth: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stages 0-4 on ``(N, C, H, W)``; ``depth`` is the slice count when N = B*D folds a stack."""
        device = clean_anatomy.device
        # ────────────────────────────────────────────────────
        # STEP 0: Multi-coil → image-space conversion (RSS)
        # ────────────────────────────────────────────────────
        _is_multicoil = coil_sensitivities is not None and clean_anatomy.shape[1] > 1
        if _is_multicoil:
            rss_anatomy = torch.sqrt((clean_anatomy.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)
            rss_complex = rss_anatomy.to(torch.complex64)
        else:
            rss_complex = None

        # ────────────────────────────────────────────────────
        # STEP 1: Motion → anatomy ONLY (brain moves)
        # ────────────────────────────────────────────────────
        if self.enable_motion:
            anatomy_kspace = fft2c(clean_anatomy)
            if depth is not None and self.coherent_3d_motion:
                # Mod4 (digital-twin review): one motion trajectory per
                # subject. Compute the motion's k-space modulation on a
                # uniform probe per subject, then broadcast across D slices.
                # This is exact for phase-only motions (rigid w/o rotation,
                # periodic, random_shot) and approximate for elastic /
                # rigid-with-rotation (image-space resampling).
                _, _, _Hk, _Wk = anatomy_kspace.shape
                B = anatomy_kspace.shape[0] // depth
                probe = torch.ones(
                    B,
                    1,
                    _Hk,
                    _Wk,
                    dtype=anatomy_kspace.dtype,
                    device=device,
                )
                mod_factor = self._apply_motion(probe, cf)  # [B, 1, H, W]
                # Broadcast across the D-slice / C-channel axes.
                ksp_5d = anatomy_kspace.view(B, depth, anatomy_kspace.shape[1], _Hk, _Wk)
                anatomy_kspace = (ksp_5d * mod_factor.unsqueeze(1)).view(
                    B * depth, anatomy_kspace.shape[1], _Hk, _Wk
                )
            else:
                anatomy_kspace = self._apply_motion(anatomy_kspace, cf)
            motion_corrupted = ifft2c(anatomy_kspace)
        else:
            motion_corrupted = clean_anatomy

        # ────────────────────────────────────────────────────
        # STEP 2: Embed markers (markers are fixed in scanner coords)
        # ────────────────────────────────────────────────────
        # degradation_only: skip marker embedding entirely — used when the
        # twin runs as a transversal data-pipeline degradation for non-VF
        # experiments (no fiducials), so the corruption pipeline operates on
        # the bare anatomy. marker_prior is a zero placeholder to preserve
        # the 3-tuple return contract.
        if degradation_only:
            joint_image = motion_corrupted
            joint_clean = clean_anatomy
            marker_prior = torch.zeros_like(motion_corrupted[:, :1])
        elif _is_multicoil:
            if self.enable_motion:
                rss_motion = torch.sqrt(
                    (motion_corrupted.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8
                ).to(torch.complex64)
            else:
                rss_motion = rss_complex

            _rss_joint, marker_prior = self.embedder(rss_motion)
            joint_image, _ = self.embedder(motion_corrupted, coil_sensitivities=coil_sensitivities)
            joint_clean, _ = self.embedder(clean_anatomy, coil_sensitivities=coil_sensitivities)
        else:
            joint_image, marker_prior = self.embedder(motion_corrupted)
            joint_clean, _ = self.embedder(clean_anatomy)

        # ────────────────────────────────────────────────────
        # STEP 2b: Image-space physical degradations (BEFORE FFT)
        # ────────────────────────────────────────────────────

        # B0 geometric distortion (EPI: voxel shifting along FE and PE)
        if self.enable_b0_distortion:
            profile = self.scanner_profile
            joint_image, b0_map, pe_shift = simulate_b0_geometric_distortion(
                joint_image,
                self.im_size,
                dwell_time=(
                    self.b0_dwell_time
                    if profile is None
                    else profile.readout_dwell_s(self.im_size[-1])
                ),
                b0_strength=self.b0_strength
                * self._b0_field_scale()
                * self._get_effective_cf("b0_distortion", cf),
                return_field=True,
                # When a REAL B0 map is supplied (e.g. derived from multi-echo
                # data), apply it verbatim so the field-scoring grades the model
                # against a real off-resonance pattern (VF real-reference seam).
                external_b0_field=external_b0_field,
                # EPI without a profile: the VF tracker scores the PE shift, which only
                # EPI produces. With one, the profile's sequence picks the shift model.
                sequence=self._b0_sequence,
            )
            # Expose the applied field so a tracker's estimate can be SCORED
            # against it (field self-consistency, not just image recon) — see
            # ConcreteVirtualFiducialStrategy._score_field. Without this the
            # B0-bias qMRI claim is structurally untestable (the field was
            # discarded). `b0_map` is Hz; `pe_shift` is the dominant pixel
            # displacement the EPI tracker recovers.
            self.last_b0_field = b0_map.detach()
            self.last_pe_shift_field = pe_shift.detach()

        # Gradient non-linearity (barrel/pincushion warping)
        if self.enable_gradient_nonlinearity:
            joint_image = simulate_gradient_nonlinearity(
                joint_image,
                self.im_size,
                severity=self.gradient_nonlinearity_severity
                * self._get_effective_cf("gradient_nonlinearity", cf),
            )

        # Magic angle T2 modulation (collagen near 54.7°).
        # M4 (digital-twin review): tissue_class gates non-MSK protocols
        # (e.g. brain), and the foreground_mask (anatomy support from the
        # embedder marker_mask) restricts the collagen patches so they
        # don't hit marker pixels.
        if self.enable_magic_angle:
            mcf = self._get_effective_cf("magic_angle", cf)
            # Build the anatomy-support mask from the embedder; the embedded
            # markers' mask is the complement of the anatomy support.
            anatomy_mask = 1.0 - self.embedder.marker_mask  # [1, 1, H, W]
            joint_image = simulate_magic_angle(
                joint_image,
                self.im_size,
                te=self.magic_angle_te,
                t2_base=self.magic_angle_t2_base,
                t2_magic=self.magic_angle_t2_base
                + (self.magic_angle_t2_magic - self.magic_angle_t2_base) * mcf,
                collagen_fraction=self.magic_angle_collagen_fraction,
                tissue_class=self.magic_angle_tissue_class,
                foreground_mask=anatomy_mask,
            )

        # ────────────────────────────────────────────────────
        # STEP 3: Scanner artifacts in k-space (affect BOTH)
        # ────────────────────────────────────────────────────
        # B1+ transmit non-uniformity changes the excited magnetisation, so it acts on
        # the signal before reception: the receiver noise added below is not modulated.
        if self.enable_b1:
            joint_image = simulate_b1_bias_field(
                joint_image,
                self.im_size,
                strength=self.b1_strength * self._get_effective_cf("b1", cf),
                b0_T=None if self.scanner_profile is None else self.scanner_profile.b0_T,
            )
        joint_kspace = fft2c(joint_image)

        # 3a. B0 inhomogeneity (spatially-varying phase)
        if self.enable_b0:
            joint_kspace = simulate_b0_inhomogeneity(
                joint_kspace,
                self.im_size,
                strength=self.b0_strength
                * self._b0_field_scale()
                * self._get_effective_cf("b0", cf),
                external_b0_field=external_b0_field,
            )
            # Expose the REAL B0 (Hz) as the field-scoring reference even on the
            # phase path (no geometric distortion) — _score_field grades the
            # model's field estimate against it.
            if external_b0_field is not None and self.last_b0_field is None:
                self.last_b0_field = external_b0_field.detach()

        # 3c. Gibbs ringing (k-space truncation)
        if self.enable_gibbs and self.gibbs_truncation < 1.0:
            gibbs_cf = self._get_effective_cf("gibbs", cf)
            gibbs_val = 1.0 - (1.0 - self.gibbs_truncation) * gibbs_cf
            if gibbs_val < 1.0:
                joint_kspace = simulate_gibbs_ringing(joint_kspace, truncation_fraction=gibbs_val)

        # 3d. Chemical shift (fat-water displacement).
        # M6 (digital-twin review): water-only markers (CuSO4/MnCl2) have
        # no lipid content; if marker_is_water_only is True we apply the
        # shift to the *anatomy* only and re-add the marker prior so it
        # is not duplicated by a spurious shifted ghost.
        if self.enable_chemical_shift:
            profile = self.scanner_profile
            max_shift_px = (
                self.chemical_shift_pixels if profile is None else profile.chemical_shift_max_px()
            )
            shift_px = max_shift_px * self._get_effective_cf("chemical_shift", cf)
            # The primitive's own default (0.15) without a profile, the profile's otherwise.
            fat = {} if profile is None else {"fat_fraction": profile.fat_fraction}
            if self.marker_is_water_only:
                # marker_prior may be [1, 1, H, W] or [B, C, H, W].
                joint_image_back = ifft2c(joint_kspace)
                mp = marker_prior
                if mp.shape != joint_image_back.shape:
                    mp = mp.expand_as(joint_image_back)
                anatomy_only = joint_image_back - mp
                anatomy_k = simulate_chemical_shift(
                    fft2c(anatomy_only),
                    self.im_size,
                    shift_pixels=shift_px,
                    **fat,
                )
                anatomy_shifted = ifft2c(anatomy_k)
                joint_kspace = fft2c(anatomy_shifted + mp)
            else:
                joint_kspace = simulate_chemical_shift(
                    joint_kspace,
                    self.im_size,
                    shift_pixels=shift_px,
                    **fat,
                )

        # 3e. Eddy currents (gradient-induced linear phase)
        if self.enable_eddy_current:
            joint_kspace = simulate_eddy_current(
                joint_kspace,
                strength=self.eddy_strength * self._get_effective_cf("eddy_current", cf),
            )

        # 3f. Spike noise (isolated k-space delta spikes)
        if self.enable_spike_noise:
            joint_kspace = simulate_spike_noise(
                joint_kspace,
                probability=self.spike_probability * self._get_effective_cf("spike_noise", cf),
            )

        # 3f2. RF zipper (coherent CW leak on single FE column)
        if self.enable_rf_zipper:
            joint_kspace = simulate_rf_zipper(
                joint_kspace,
                fe_location=self.rf_zipper_location,
                intensity_ratio=self.rf_zipper_intensity * self._get_effective_cf("rf_zipper", cf),
            )

        joint_kspace = self._add_receiver_noise(
            joint_kspace, snr_target, cf, (-3, -2, -1), depth=depth
        )

        # 3i. K-space undersampling (aliasing), LAST: the scanner records signal, spikes,
        # zipper and noise only on the lines it acquires. Capture the per-call random
        # sampling mask so reconstruction strategies can feed it to the model's
        # data-consistency layers (otherwise their DC is starved and no-ops).
        if self.enable_undersampling and self.acceleration > 1.0:
            accels_cf = self._get_effective_cf("undersampling", cf)
            accel_val = 1.0 + (self.acceleration - 1.0) * accels_cf
            if accel_val > 1.0:
                joint_kspace, self.last_undersampling_mask = simulate_undersampling(
                    joint_kspace,
                    acceleration=accel_val,
                    return_mask=True,
                    line_dim=self.undersampling_line_dim,
                )

        # ────────────────────────────────────────────────────
        # STEP 4: Reconstruct + image-space effects
        # ────────────────────────────────────────────────────
        corrupted_image = ifft2c(joint_kspace)

        # 4b. RF crosstalk (multi-slice signal bleeding)
        if self.enable_rf_crosstalk:
            corrupted_image = simulate_rf_crosstalk(
                corrupted_image,
                crosstalk_fraction=self.crosstalk_fraction
                * self._get_effective_cf("rf_crosstalk", cf),
            )

        return corrupted_image, marker_prior, joint_clean

    def _add_receiver_noise(
        self,
        joint_kspace: torch.Tensor,
        snr_target: float | torch.Tensor | None,
        cf: float,
        dims: tuple[int, ...],
        depth: int | None = None,
    ) -> torch.Tensor:
        """Complex AWGN at the target SNR; ``dims`` are the axes of one sample's signal RMS.

        With ``depth`` the batch folds ``depth`` slices per subject, and the noise level is
        referenced to the subject's RMS: the receiver's thermal noise is the same for every
        slice. A per-sample ``snr_target`` is then one value per subject, ``(B,)``.
        """
        device = joint_kspace.device
        # 3h. Complex AWGN (→ Rician noise in magnitude)
        noise_cf = self._get_effective_cf("noise", cf)
        if snr_target is None:
            # Drawn on the device, so no host sync on CUDA.
            snr_target = self.snr_range[0] + torch.rand(1, device=device) * (
                self.snr_range[1] - self.snr_range[0]
            )

        # Apply progressive scaling to SNR based on noise_cf
        # High SNR (clean) -> Low SNR (corrupted).
        # If noise_cf=0, SNR=100 (clean). If noise_cf=1, SNR=snr_target.
        #
        # Mod8 (digital-twin review): the 100 dB clean baseline is
        # effectively noiseless (sigma ~ 1e-5 * signal_rms). It is below
        # fp16 precision at amplitude SNR, which is fine in fp32 but
        # callers running fp16-only AMP should set snr_target with a
        # ceiling rather than relying on the 100 dB anchor.
        snr_clean = 100.0
        snr_target = snr_clean - noise_cf * (snr_clean - snr_target)
        # Mod5 (digital-twin review): `snr_target` is *amplitude* SNR ---
        # the formula below uses sigma = rms / 10^(SNR/20), the
        # 20*log10(signal_rms / sigma) convention. Complex Gaussian
        # noise has total variance 2 sigma^2, so the *power* SNR is
        # 3 dB lower than the value passed in. Macovski 1996 uses
        # power SNR; convert by SNR_power = SNR_amplitude - 3.0103 if
        # comparing.
        snr_target = torch.as_tensor(snr_target, device=device, dtype=torch.float32)
        if depth is not None and snr_target.numel() > 1:
            subjects = joint_kspace.shape[0] // depth
            if tuple(snr_target.shape) != (subjects,):
                raise ValueError(
                    f"multislice snr_target must be a scalar or one value per subject "
                    f"({subjects},); got shape {tuple(snr_target.shape)}"
                )
            snr_target = snr_target.repeat_interleave(depth)
        snr_target = snr_target.reshape((-1,) + (1,) * (joint_kspace.dim() - 1))

        # One receiver noise level per sample, shared by its coils: a weakly coupled coil
        # sees the same thermal noise with less signal, not proportionally less noise.
        if depth is None:
            signal_rms = torch.sqrt(torch.mean(joint_kspace.abs() ** 2, dim=dims, keepdim=True))
        else:
            power = joint_kspace.abs().pow(2).reshape(joint_kspace.shape[0] // depth, -1)
            signal_rms = power.mean(dim=1).sqrt().repeat_interleave(depth)
            signal_rms = signal_rms.reshape((-1,) + (1,) * (joint_kspace.dim() - 1))
        noise_std = signal_rms / (10 ** (snr_target / 20.0))
        n_re, n_im = per_sample_normal_pair(*joint_kspace.shape, device=device)
        joint_kspace = joint_kspace + torch.complex(n_re, n_im) * noise_std
        return joint_kspace

    def _apply_registry_axes(
        self,
        corrupted_image: torch.Tensor,
        cf: float,
        seed: int | None,
        geometry: AcquisitionGeometry | None,
    ) -> torch.Tensor:
        """Stage 5: the registry operators, with the per-call geometry or the profile."""
        # 5. Registry degradations — the physics operators in
        # DEGRADATION_REGISTRY that this pipeline has no native branch for.
        # Applied to the unfolded input, at 5-D with its geometry for a volume. Severity
        # is the same diffusion-coupled factor every native axis uses, so a registry
        # axis ramps with cf exactly like a native one.
        if self._registry_degradations:
            from spectramr.infrastructure.physics.degradations.registry import (
                apply_degradation,
                derive_axis_seed,
            )
            from spectramr.infrastructure.physics.degradations.registry_queries import (
                affected_components,
            )

            # forward(seed=S) uses S itself as the master seed, so the registry axes
            # draw derive_axis_seed(S, axis) -- the realisation a DegradationChain fitted
            # with seed S used. Without a seed it is fresh, drawn on CPU (no device sync).
            master = (
                int(seed)
                if seed is not None
                else int(torch.randint(0, 2**31 - 1, (1,), device="cpu").item())
            )
            for axis in self._registry_degradations:
                before = corrupted_image
                out = apply_degradation(
                    axis,
                    before,
                    theta=self._get_effective_cf(axis, cf),
                    seed=derive_axis_seed(master, axis),
                    geometry=geometry,
                    profile=self.scanner_profile,
                )
                # Some registry ops are intrinsically real (rician acts on the
                # magnitude), and this pipeline is complex end to end. Casting
                # real->complex would zero the phase and silently destroy it, so
                # the original phase is re-attached instead. That is only sound
                # when the op declares it does not touch phase, which the
                # registry records per degradation in `affects` — so the
                # contradiction is a raise, not a guess.
                if before.is_complex() and not out.is_complex():
                    if "phase" in affected_components(axis):
                        raise ValueError(
                            f"Degradation {axis!r} declares it affects phase but "
                            "returned a real tensor, so its phase change cannot "
                            "be recovered. Fix the registry entry or the op."
                        )
                    out = torch.polar(out.to(before.real.dtype), torch.angle(before))
                elif not before.is_complex() and out.is_complex():
                    out = out.abs().to(before.dtype)
                corrupted_image = out
        return corrupted_image


__all__ = [
    "CornerFiducialEmbedder",
    "DigitalTwinSimulator",
    "simulate_b0_geometric_distortion",
    "simulate_b0_inhomogeneity",
    "simulate_b1_bias_field",
    "chemical_shift_pixels",
    "simulate_chemical_shift",
    "simulate_eddy_current",
    "simulate_elastic_motion",
    "simulate_gibbs_ringing",
    "simulate_gradient_nonlinearity",
    "simulate_magic_angle",
    "simulate_periodic_motion",
    "simulate_random_shot_motion",
    "simulate_rf_crosstalk",
    "simulate_rf_zipper",
    "simulate_rigid_motion_kspace",
    "simulate_spike_noise",
    "simulate_spike_noise_volume",
    "outer_line_count",
    "simulate_undersampling",
]


# Imported last: the degradations package imports this module's primitives, so a first import
# of this module must define them before the package's catalogue reads them.
from spectramr.infrastructure.physics.degradations._common import readout_offset_px  # noqa: E402
from spectramr.infrastructure.physics.degradations.conditions import (  # noqa: E402
    b0_shift_sequence,
)
from spectramr.infrastructure.physics.degradations.geometry import (  # noqa: E402
    AcquisitionGeometry,
    AcquisitionMode,
    resolve_geometry,
)
