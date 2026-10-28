"""Faithful multi-acquisition synthesis for the VF field-mapping arms.

Several Virtual-Fiducial arms (AFI, double-angle, dual-echo) describe methods
that are mathematically defined over *multiple* acquisitions — two TRs, two
flip angles, two TEs — yet the single-frame VF pipeline feeds their models one
complex frame, so the method algebra runs on coil/real-imag channels instead
of the intended acquisitions (VF review, 2026-06-02; see
``docs/digital_twin_review_followups.rst`` Pass 5).

This module synthesises the genuine acquisition stack each method consumes,
**reusing existing, tested physics** — ``DifferentiableBlochLayer`` (steady-
state SPGR signal with explicit ``b1_map`` / ``b0_map``) and
``relaxation_priors.bottomley_t1`` (field-correct T1) — so nothing here invents
new physics. It generates a smooth ground-truth field, synthesises the
acquisitions from it, and returns both, so the field can be used as the
supervision target and the synthesis can be verified by a closed-form
round-trip (synthesise from a known field → run the method's inversion →
recover the field). See ``tests/unit/infrastructure/physics/test_multi_acquisition.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn.functional import (
    affine_grid,
    avg_pool2d,
    grid_sample,
    interpolate,
    normalize,
)

from spectramr.infrastructure.physics.differentiable_bloch import DifferentiableBlochLayer
from spectramr.infrastructure.physics.marker_realisation import marker_batch
from spectramr.infrastructure.physics.psf_estimation import apply_psf
from spectramr.infrastructure.physics.relaxation_priors import bottomley_t1
from spectramr.infrastructure.physics.subpixel_registration import fourier_shift
from spectramr.infrastructure.physics.virtual_fiducial import VirtualFiducial

# Class order of the three-class tissue proxy (index 0, 1, 2).
TISSUE_ORDER = ("white_matter", "gray_matter", "csf")

# Methods whose acquisition stack this module can synthesise faithfully
# (tissue-independent / single-tissue-T1 closed forms, or — for ``mrf`` — a
# per-pixel Bloch transient matched against a dictionary).
PAIRED_METHODS = (
    "double_angle",
    "afi",
    "dual_echo",
    "bloch_siegert",
    "mrf",
    "lowrank_temporal",
    "subvoxel_sr",
    "bssfp_banding",
)

# MRF dictionary parameters — MUST match ``MRFDictMatcher._build_bloch_dictionary``
# so a synthesised transient lands on a real dictionary atom.
_MRF_T1_LOG = (math.log(100.0), math.log(3000.0))
_MRF_T2_LOG = (math.log(20.0), math.log(500.0))
_MRF_TR_MS, _MRF_TE_MS, _MRF_FLIP_DEG = 10.0, 2.0, 60.0


def _mrf_flip_schedule(n: int) -> torch.Tensor:
    """Sinusoidal flip-angle schedule (rad), matching the model dictionary."""
    return torch.sin(torch.linspace(0.1, math.pi, n)) * math.radians(_MRF_FLIP_DEG)


@dataclass(frozen=True)
class AcquisitionResult:
    """Output of a paired-acquisition synthesis.

    Attributes:
        stack: Acquisitions stacked on the channel axis ``[B, n_acq, H, W]``
            (real magnitudes for B1 methods; per-echo phase for ``dual_echo``).
        field: Ground-truth field the method recovers — a B1 transmit scale
            (dimensionless, ~1.0) for ``afi``/``double_angle`` or a B0 offset in
            Hz for ``dual_echo`` — ``[B, 1, H, W]``. Use as the supervision target.
        echoes: For ``dual_echo``, the two complex echoes ``[B, 2, H, W]``;
            ``None`` otherwise. Kept so the inversion is testable directly.
        shifts: For ``subvoxel_sr``, the ground-truth per-sample per-frame
            sub-pixel offsets ``[B, n_frames, 2]`` in HR pixels, ``(dy, dx)``.
            The oracle rung of the shift-knowledge ladder and the supervision
            target for the recovered rung. ``None`` for methods without a
            geometric offset.
        marker_stack: For ``subvoxel_sr`` with the fiducial enabled, the virtual
            fiducial carried through the SAME offsets and the same pooling as
            the anatomy, ``[B, n_frames, H/s, W/s]``. Registering it against
            :meth:`MultiAcquisitionSimulator.marker_reference` recovers the
            offsets without ever consulting ``shifts``.
        marker_hr: The un-shifted HR fiducial each sample's ``marker_stack`` was
            built from, ``[B, 1, H, W]``. Every consumer (registration, probe
            target, PSF and kappa measurement, conformal calibration) reads it
            from here, so a per-sample realisation cannot be paired with the
            wrong reference.
        marker_seeds: The per-sample realisation seeds, or ``None`` for the
            fixed (``marker_seed``) fiducial.
    """

    stack: torch.Tensor
    field: torch.Tensor
    echoes: torch.Tensor | None = None
    shifts: torch.Tensor | None = None
    marker_stack: torch.Tensor | None = None
    marker_hr: torch.Tensor | None = None
    marker_seeds: tuple[int, ...] | None = None


def _smooth_field(
    shape: tuple[int, int, int, int],
    lo: float,
    hi: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """A smooth low-frequency field in ``[lo, hi]`` (bilinear-upsampled coarse noise)."""
    b, _, h, w = shape
    coarse = torch.rand(b, 1, 8, 8, device=device, generator=generator)
    field = interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
    fmin = field.amin(dim=(-2, -1), keepdim=True)
    fmax = field.amax(dim=(-2, -1), keepdim=True)
    field = (field - fmin) / (fmax - fmin + 1e-8)
    return lo + (hi - lo) * field


class MultiAcquisitionSimulator(torch.nn.Module):
    """Synthesise the multi-acquisition stack a VF field-mapping method needs.

    Reuses :class:`DifferentiableBlochLayer` (SPGR) + :func:`bottomley_t1`.
    The magnitude of the clean image is used as the proton-density / M0 proxy;
    a single literature T1 (Bottomley, at ``field_strength_t``) and a fixed T2
    drive the steady-state signal. The double-angle method is T1-independent by
    construction; AFI is run in the short-TR steady state where its ratio is
    near-T1-independent — both stamp the assumed T1 in provenance.
    """

    def __init__(
        self,
        method: str,
        *,
        field_strength_t: float = 0.3,
        flip_deg: float = 60.0,
        tr_ms: float = 20.0,
        tr_ratio: float = 5.0,
        te1_ms: float = 5.0,
        te2_ms: float = 10.0,
        t2_ms: float = 80.0,
        b1_range: tuple[float, float] = (0.7, 1.3),
        b0_range_hz: tuple[float, float] = (-80.0, 80.0),
        k_bs: float = 0.5,
        n_timepoints: int = 16,
        n_frames: int = 8,
        sr_scale: int = 2,
        t2_decay_ms: float = 10.0,
        n_phase_cycles: int = 4,
        tr_bssfp_ms: float = 5.0,
        b0_bssfp_range_hz: tuple[float, float] = (-60.0, 60.0),
        subvoxel_max_shift_px: float = 1.0,
        marker_enabled: bool = False,
        marker_grid_spacing: int = 16,
        marker_sigma: float = 2.0,
        marker_jitter: float = 0.35,
        marker_seed: int = 0,
        marker_voxel_mm: tuple[float, ...] | None = None,
        marker_effective_voxel_mm: tuple[float, ...] | None = None,
        marker_sigma_mm: tuple[float, ...] | None = None,
        marker_spacing_mm: tuple[float, ...] | None = None,
        marker_kappa: float = 1.0,
    ) -> None:
        super().__init__()
        if method not in PAIRED_METHODS:
            raise ValueError(
                f"Unknown multi-acquisition method {method!r}. Valid: {list(PAIRED_METHODS)}"
            )
        self.method = method
        self.field_strength_t = field_strength_t
        self.flip_rad = math.radians(flip_deg)
        self.tr_ms = tr_ms
        self.tr_ratio = tr_ratio
        self.te1_ms = te1_ms
        self.te2_ms = te2_ms
        self.t2_ms = t2_ms
        self.b1_range = b1_range
        self.b0_range_hz = b0_range_hz
        self.k_bs = k_bs
        self.n_timepoints = n_timepoints
        self.n_frames = n_frames
        self.sr_scale = sr_scale
        self.t2_decay_ms = t2_decay_ms
        # Balanced-SSFP banding synthesis. A short TR keeps the per-TR phase
        # beta = 2*pi*df*TR inside one DFT period (|df| < 1/(2*TR)); with
        # tr_bssfp_ms=5 ms the period is +/-100 Hz, so the b0 range stays inside.
        if n_phase_cycles < 4:
            raise ValueError(
                f"bssfp_banding needs n_phase_cycles >= 4 for identifiability "
                f"(elliptical-signal-model floor), got {n_phase_cycles}."
            )
        self.n_phase_cycles = n_phase_cycles
        self.tr_bssfp_ms = tr_bssfp_ms
        self.b0_bssfp_range_hz = b0_bssfp_range_hz
        # Sub-voxel dither + virtual-fiducial marker (subvoxel_sr only).
        if subvoxel_max_shift_px <= 0.0:
            raise ValueError(f"subvoxel_max_shift_px must be > 0, got {subvoxel_max_shift_px}")
        self.subvoxel_max_shift_px = subvoxel_max_shift_px
        self.marker_enabled = marker_enabled
        self.marker_grid_spacing = marker_grid_spacing
        self.marker_sigma = marker_sigma
        self.marker_jitter = marker_jitter
        self.marker_seed = marker_seed
        # Physical-units fiducial geometry (mm). `marker_effective_voxel_mm` is
        # what selects the mode; `marker_voxel_mm` defaults to it when the volume
        # has not been resampled onto a finer grid.
        self.marker_effective_voxel_mm = marker_effective_voxel_mm
        self.marker_voxel_mm = marker_voxel_mm or marker_effective_voxel_mm
        self.marker_sigma_mm = marker_sigma_mm
        self.marker_spacing_mm = marker_spacing_mm
        self.marker_kappa = marker_kappa
        # Contrast transfer, installed by the strategy when an arm declares it.
        self.field_gains: dict[str, float] | None = None
        self._gain_tables: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self.marker_gain: float | None = None
        self.forward_psf_kernel: torch.Tensor | None = None
        if marker_enabled and method != "subvoxel_sr":
            raise ValueError(
                f"marker_enabled is only implemented for 'subvoxel_sr', not {method!r}. "
                "Advertising it elsewhere would be an inert knob (CLAUDE.md #15)."
            )
        # Built once per spatial size; the pattern is a fixed, known reference.
        self._fiducial_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._fiducial_geometry: dict[tuple[int, int], tuple[tuple[float, ...], ...]] = {}
        # Long-TR (T1-independent) regime for the double-angle method.
        self.tr_long_ms = 3000.0
        self.t1_ms = bottomley_t1(field_strength_t, "white_matter")
        self._bloch = DifferentiableBlochLayer(sequence_type="SPGR")

    def _fiducial_field(
        self, size: tuple[int, int], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """The un-shifted virtual fiducial at ``size``, ``[1, 1, H, W]`` real."""
        key = (int(size[0]), int(size[1]))
        cached = self._fiducial_cache.get(key)
        if cached is None or cached.device != device:
            fiducial = self._make_fiducial(key)
            cached = fiducial(1).real.to(device=device, dtype=dtype).contiguous()
            self._fiducial_cache[key] = cached
        return cached

    def _make_fiducial(self, key: tuple[int, int]) -> VirtualFiducial:
        """The fixed fiducial at ``key``; also caches its pixel geometry."""
        # Physical mode when the arm declares the resolution the marker must
        # survive at. The synthetic M4Raw arm leaves it unset and keeps the
        # pixel geometry, where the grid IS the resolution and voxels are
        # isotropic, so both are correct in their own regime.
        if self.marker_effective_voxel_mm is not None:
            fiducial = VirtualFiducial(
                im_size=key,
                jitter=self.marker_jitter,
                seed=self.marker_seed,
                voxel_mm=self.marker_voxel_mm,
                effective_voxel_mm=self.marker_effective_voxel_mm,
                sigma_mm=self.marker_sigma_mm,
                spacing_mm=self.marker_spacing_mm,
                kappa=self.marker_kappa,
            )
        else:
            fiducial = VirtualFiducial(
                im_size=key,
                grid_spacing=self.marker_grid_spacing,
                sigma=self.marker_sigma,
                jitter=self.marker_jitter,
                seed=self.marker_seed,
            )
        self._fiducial_geometry[key] = (fiducial.spacing_px, fiducial.sigma_px)
        return fiducial

    def marker_realisations(
        self,
        seeds: tuple[int, ...],
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """One un-shifted HR fiducial per seed, ``[B, 1, H, W]``, same geometry."""
        key = (int(size[0]), int(size[1]))
        if key not in self._fiducial_geometry:
            self._make_fiducial(key)
        spacing_px, sigma_px = self._fiducial_geometry[key]
        return marker_batch(
            seeds, key, spacing_px, sigma_px, self.marker_jitter, device=device, dtype=dtype
        ).contiguous()

    def pooled(self, marker_hr: torch.Tensor) -> torch.Tensor:
        """A marker on the acquisition grid: the ABSOLUTE registration reference."""
        return avg_pool2d(marker_hr, self.sr_scale)

    def set_forward_psf(self, kernel: torch.Tensor | None) -> None:
        """Install the blur the simulator applies BEFORE pooling.

        ``None`` reproduces the pre-2026-07-26 behaviour exactly: pooling with
        no explicit PSF, which is itself an implicit boxcar assumption -- making
        it explicit is half the point of this path.
        """
        self.forward_psf_kernel = kernel

    def set_field_transfer(self, gains: dict[str, float] | None, marker_gain: float | None) -> None:
        """Install the per-tissue contrast transfer used by ``subvoxel_sr``.

        ``gains`` maps tissue class names to ``kappa``, ``marker_gain`` is the
        fiducial's own (declared, hence exactly known) factor. Set to ``None``
        to disable, which is the pre-2026-07-26 behaviour.
        """
        self.field_gains = gains
        self._gain_tables = {}
        self.marker_gain = marker_gain

    def _tissue_gain_field(self, pd: torch.Tensor) -> torch.Tensor:
        """Per-voxel ``1/kappa``: the low-field view of a high-field image.

        Tissue class is assigned by intensity quantile on the normalised
        magnitude -- a THREE-CLASS PROXY, not a segmentation. It is enough to
        make the transfer a genuine contrast change rather than a global scale
        (the CSF-to-parenchyma gain ratio is about 2.1 over 64 mT to 3 T), and
        it is stated as a proxy wherever the arm reports a result.
        """
        gains = self.field_gains
        if gains is None:
            return torch.ones_like(pd)
        # Built once per device: a host list -> CUDA copy blocks on every step.
        key = (pd.device, pd.dtype)
        table = self._gain_tables.get(key)
        if table is None:
            table = torch.tensor([gains[t] for t in TISSUE_ORDER], device=pd.device, dtype=pd.dtype)
            self._gain_tables[key] = table
        return table[self.tissue_classes(pd)]

    @staticmethod
    def tissue_classes(pd: torch.Tensor) -> torch.Tensor:
        """Per-voxel class index into :data:`TISSUE_ORDER`, ``[B, 1, H, W]`` long.

        The one owner of the quantile proxy: the transfer above and the
        tissue-map factored head's agreement report both read it.
        """
        b = pd.shape[0]
        flat = pd.reshape(b, -1)
        lo = torch.quantile(flat.float(), 0.60, dim=-1).view(b, 1, 1, 1)
        hi = torch.quantile(flat.float(), 0.95, dim=-1).view(b, 1, 1, 1)
        # dark -> white matter, mid -> grey matter, bright -> CSF-like
        return (pd >= lo).long() + (pd >= hi).long()

    def marker_hr(
        self, size: tuple[int, int], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """The un-shifted fiducial on the HR grid, ``[1, 1, H, W]``.

        The reconstruction TARGET of the super-Nyquist probe: the marker frames
        are ``sr_scale``-pooled shifted views of exactly this, so restoring it
        is the same inverse problem the anatomy path solves, on a signal whose
        every spatial frequency is known a priori.
        """
        return self._fiducial_field(size, device, dtype)

    def marker_reference(
        self, size: tuple[int, int], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Pooled, un-shifted fiducial: the ABSOLUTE registration reference.

        Registering a marker frame against this (rather than against frame 0)
        recovers each frame's absolute offset instead of an offset relative to
        an arbitrary frame. That absolute anchor is the whole point of carrying
        a known fiducial rather than doing blind frame-to-frame registration.
        """
        return self.pooled(self._fiducial_field(size, device, dtype))

    def _maps(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(pd_map, t1_map) broadcast to the image grid."""
        pd = image.abs() if image.is_complex() else image
        if pd.ndim == 3:
            pd = pd.unsqueeze(1)
        t1 = torch.full_like(pd, self.t1_ms)
        return pd, t1

    @staticmethod
    def _resolve_field(
        external: torch.Tensor | None,
        shape: torch.Size,
        default_range: tuple[float, float],
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Use a REAL field when supplied (resized/broadcast to ``shape``), else a
        random smooth field — the multi-acquisition real-reference seam."""
        if external is None:
            return _smooth_field(shape, *default_range, device, generator)
        f = external.to(device=device, dtype=torch.float32)
        if f.dim() == 5:  # [B, C, H, W, D] → middle slice
            f = f[..., f.shape[-1] // 2]
        if f.dim() == 3:  # [B, H, W] → [B, 1, H, W]
            f = f.unsqueeze(1)
        if f.shape[-2:] != shape[-2:]:
            f = interpolate(f, size=tuple(shape[-2:]), mode="bilinear", align_corners=False)
        if f.shape[0] != shape[0]:
            f = f[:1].expand(shape[0], -1, -1, -1)
        return f

    def forward(
        self,
        image: torch.Tensor,
        generator: torch.Generator | None = None,
        external_field: torch.Tensor | None = None,
        marker_seeds: tuple[int, ...] | None = None,
    ) -> AcquisitionResult:
        pd, t1 = self._maps(image)
        t2 = torch.full_like(pd, self.t2_ms)
        shape = pd.shape

        if self.method in ("double_angle", "afi"):
            b1 = self._resolve_field(external_field, shape, self.b1_range, pd.device, generator)
            if self.method == "double_angle":
                s1 = self._bloch(t1, t2, pd, self.tr_long_ms, self.te1_ms, self.flip_rad, b1_map=b1)
                s2 = self._bloch(
                    t1,
                    t2,
                    pd,
                    self.tr_long_ms,
                    self.te1_ms,
                    2.0 * self.flip_rad,
                    b1_map=b1,
                )
            else:  # afi: dual-TR interleaved steady state (Yarnykh 2007)
                s1, s2 = _afi_dual_tr_signal(
                    pd, t1, self.flip_rad * b1, self.tr_ms, self.tr_ms * self.tr_ratio
                )
            stack = torch.cat([s1.abs(), s2.abs()], dim=1)
            return AcquisitionResult(stack=stack, field=b1)

        if self.method == "mrf":
            # 1-parameter (T1,T2)-coupled tissue family (matches the model's
            # log-uniform paired dictionary); synthesise the Bloch transient.
            f = _smooth_field(shape, 0.0, 1.0, pd.device, generator)
            t1m = torch.exp(_MRF_T1_LOG[0] + (_MRF_T1_LOG[1] - _MRF_T1_LOG[0]) * f)
            t2m = torch.exp(_MRF_T2_LOG[0] + (_MRF_T2_LOG[1] - _MRF_T2_LOG[0]) * f)
            sched = _mrf_flip_schedule(self.n_timepoints).to(pd.device)
            frames = [self._bloch(t1m, t2m, pd, _MRF_TR_MS, _MRF_TE_MS, fa.item()) for fa in sched]
            stack = torch.cat(frames, dim=1)  # [B, T, H, W] per-pixel transient
            field = torch.cat([f, f], dim=1)  # (T1_norm, T2_norm), both = f
            return AcquisitionResult(stack=stack, field=field)

        if self.method == "lowrank_temporal":
            # Smooth T2 map → multi-echo decay series (low temporal rank). The
            # clean series is the recon target; the noisy series the model input.
            n = self.n_frames
            t2 = _smooth_field(shape, 20.0, 150.0, pd.device, generator)  # ms
            ts = torch.arange(n, device=pd.device).view(1, n, 1, 1) * self.t2_decay_ms
            clean = pd * torch.exp(-ts / t2)  # [B, n, H, W]
            noise = (
                0.1 * clean.std() * torch.randn(clean.shape, generator=generator, device=pd.device)
            )
            return AcquisitionResult(stack=clean + noise, field=clean)

        if self.method == "subvoxel_sr":
            # N sub-pixel-shifted, down-sampled views of the HR image; the model
            # fuses them back to the HR target (genuine multi-frame SR).
            #
            # 2026-07-26, three corrections to this branch:
            #  * ONE batched draw on-device. The previous loop called `.item()`
            #    twice per frame (2*n_frames GPU syncs per training step) and
            #    applied one scalar offset to the WHOLE batch, so a batch of B
            #    saw n distinct dither patterns rather than n*B.
            #  * Exact Fourier translation instead of bilinear `grid_sample`.
            #    A resample multiplies the spectrum by an interpolation kernel
            #    whose rolloff depends on the fractional shift: at a half-pixel
            #    dither it distorts high-frequency magnitudes by ~35% and sheds
            #    ~5% of total energy, destroying precisely the band multi-frame
            #    SR exists to recover.
            #  * The offsets are RETURNED, and optionally imprinted on a virtual
            #    fiducial, so a strategy can recover them from the marker rather
            #    than being handed them (or being blind to them).
            n, s = self.n_frames, self.sr_scale
            hr = pd  # [B, 1, H, W]
            b, _, h, w = hr.shape
            shifts = (torch.rand(b, n, 2, generator=generator, device=pd.device) - 0.5) * (
                2.0 * self.subvoxel_max_shift_px
            )
            # Relaxometric contrast transfer: the synthesised frames are the
            # LOW-field view, so the high-field target is divided by kappa. With
            # no transfer installed this is exactly 1 and the branch is inert.
            lf = hr / self._tissue_gain_field(hr).clamp(min=1e-6)
            shifted = fourier_shift(lf.expand(b, n, h, w).contiguous(), shifts)
            if self.forward_psf_kernel is not None:
                shifted = apply_psf(shifted, self.forward_psf_kernel.expand(b, -1, -1, -1, -1))
            frames = avg_pool2d(shifted, s)

            marker_stack = marker_hr = None
            if marker_seeds is not None and not self.marker_enabled:
                raise ValueError("marker_seeds given but the fiducial is not enabled")
            if self.marker_enabled:
                if marker_seeds is None:
                    marker = self._fiducial_field((h, w), pd.device, pd.dtype)
                else:
                    if len(marker_seeds) != b:
                        raise ValueError(f"{len(marker_seeds)} marker seeds for a batch of {b}")
                    marker = self.marker_realisations(marker_seeds, (h, w), pd.device, pd.dtype)
                marker_hr = marker.expand(b, 1, h, w)
                # The fiducial's gain is DECLARED, not inferred from intensity,
                # which is the whole point: its kappa is known exactly while the
                # anatomy's is only proxied.
                if self.marker_gain is not None:
                    marker = marker / max(self.marker_gain, 1e-6)
                marker_shifted = fourier_shift(marker.expand(b, n, h, w).contiguous(), shifts)
                if self.forward_psf_kernel is not None:
                    # The marker rides the SAME operator as the anatomy, which
                    # is the only reason measuring it on the marker says
                    # anything about the anatomy.
                    marker_shifted = apply_psf(
                        marker_shifted,
                        self.forward_psf_kernel.expand(b, -1, -1, -1, -1),
                    )
                marker_stack = avg_pool2d(marker_shifted, s)
            return AcquisitionResult(
                stack=frames,
                field=hr,
                shifts=shifts,
                marker_stack=marker_stack,
                marker_hr=marker_hr,
                marker_seeds=None if marker_seeds is None else tuple(marker_seeds),
            )

        if self.method == "bloch_siegert":
            # Off-resonance pulse at +/- omega_RF imprints phase +/- K_BS * B1^2;
            # the difference isolates B1 (tissue-independent — phase only).
            b1 = self._resolve_field(external_field, shape, self.b1_range, pd.device, generator)
            phi = self.k_bs * b1**2
            cplx = pd.to(torch.complex64)
            s_plus = cplx * torch.exp(1j * phi)
            s_minus = cplx * torch.exp(-1j * phi)
            stack = torch.cat([s_plus.angle(), s_minus.angle()], dim=1)
            echoes = torch.cat([s_plus, s_minus], dim=1)
            return AcquisitionResult(stack=stack, field=b1, echoes=echoes)

        if self.method == "bssfp_banding":
            # Balanced-SSFP banding -> B0. The transverse steady state traces the
            # elliptical signal model (Xiang & Hoff, MRM 2014) as the per-TR
            # off-resonance phase beta = 2*pi*df*TR sweeps a period; N RF
            # phase-cycles Delta_phi_n = 2*pi*n/N make the *absolute*
            # off-resonance identifiable via the first-harmonic phase-cycle DFT
            # (invert_bssfp_banding). field = the B0 offset (Hz); stack = the
            # complex phase-cycled images [B, N, H, W] (the DFT needs magnitude
            # AND phase, so the stack is complex, unlike the B1 magnitude stacks).
            b0 = self._resolve_field(
                external_field, shape, self.b0_bssfp_range_hz, pd.device, generator
            )
            stack = _bssfp_phase_cycled(
                pd,
                b0,
                self.n_phase_cycles,
                tr_ms=self.tr_bssfp_ms,
                t1_ms=self.t1_ms,
                t2_ms=self.t2_ms,
                flip_rad=self.flip_rad,
            )
            return AcquisitionResult(stack=stack, field=b0)

        # dual_echo: same sequence at two TEs, B0 → per-echo phase
        b0 = self._resolve_field(external_field, shape, self.b0_range_hz, pd.device, generator)
        e1 = self._bloch(t1, t2, pd, self.tr_ms, self.te1_ms, self.flip_rad, b0_map=b0)
        e2 = self._bloch(t1, t2, pd, self.tr_ms, self.te2_ms, self.flip_rad, b0_map=b0)
        echoes = torch.cat([e1, e2], dim=1)
        stack = torch.cat([e1.angle(), e2.angle()], dim=1)
        return AcquisitionResult(stack=stack, field=b0, echoes=echoes)


def _subpixel_shift(img: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Bilinear sub-pixel translation of ``img`` by ``(dx, dy)`` pixels."""
    b, _, h, w = img.shape
    theta = (
        torch.tensor(
            [[1.0, 0.0, -2.0 * dx / w], [0.0, 1.0, -2.0 * dy / h]],
            device=img.device,
            dtype=img.dtype,
        )
        .unsqueeze(0)
        .expand(b, 2, 3)
    )
    grid = affine_grid(theta, list(img.shape), align_corners=False)
    return grid_sample(img, grid, align_corners=False, padding_mode="reflection")


def _bssfp_phase_cycled(
    pd: torch.Tensor,
    b0_hz: torch.Tensor,
    n_cycles: int,
    *,
    tr_ms: float,
    t1_ms: float,
    t2_ms: float,
    flip_rad: float,
) -> torch.Tensor:
    """Synthesise an N-phase-cycled balanced-SSFP stack from a B0 field.

    Uses the elliptical signal model (Xiang & Hoff, MRM 2014). With per-TR
    relaxation ``E1 = exp(-TR/T1)``, ``E2 = exp(-TR/T2)`` and flip ``alpha``,
    the transverse steady state as a function of the off-resonance phase
    ``theta = beta + Delta_phi`` is::

        S(theta) = M * (1 - a e^{i theta}) / (1 - b cos theta),   a = E2

    with ``M`` and ``b`` the standard ESM geometry parameters. ``beta = 2 pi
    df TR`` carries the off-resonance; the N RF phase-cycles ``Delta_phi_n =
    2 pi n / N`` sweep the ellipse so the first DFT harmonic over ``n`` is
    injective in ``df`` over one period. Returns a complex ``[B, N, H, W]``.
    """
    e1 = math.exp(-tr_ms / (t1_ms + 1e-8))
    e2 = math.exp(-tr_ms / (t2_ms + 1e-8))
    ca, sa = math.cos(flip_rad), math.sin(flip_rad)
    denom = 1.0 - e1 * ca - e2**2 * (e1 - ca)
    m = pd * (1.0 - e1) * sa / (denom + 1e-8)  # [B, 1, H, W], M0-scaled
    a = e2
    b = e2 * (1.0 - e1) * (1.0 + ca) / (denom + 1e-8)  # scalar ESM eccentricity
    beta = 2.0 * math.pi * b0_hz * (tr_ms * 1e-3)  # per-TR off-resonance phase
    frames = []
    for n in range(n_cycles):
        d_phi = 2.0 * math.pi * n / n_cycles
        theta = beta + d_phi
        s = m.to(torch.complex64) * (1.0 - a * torch.exp(1j * theta)) / (1.0 - b * torch.cos(theta))
        frames.append(s)
    return torch.cat(frames, dim=1)  # complex [B, N, H, W]


def _afi_dual_tr_signal(
    pd: torch.Tensor,
    t1_ms: torch.Tensor,
    flip_eff: torch.Tensor,
    tr1_ms: float,
    tr2_ms: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spoiled dual-TR steady-state AFI signals (Yarnykh, MRM 2007).

    ``DifferentiableBlochLayer`` models the *single*-TR SPGR steady state; AFI
    interleaves two TRs and the magnetisation never fully recovers, so its
    closed form differs. With perfect spoiling and effective flip ``alpha``::

        S1 = M0 sin(a) (1 - E2 + (1-E1) E2 cos a) / (1 - E1 E2 cos^2 a)
        S2 = M0 sin(a) (1 - E1 + (1-E2) E1 cos a) / (1 - E1 E2 cos^2 a)

    with ``E_k = exp(-TR_k / T1)``. In the ``TR << T1`` regime the standard
    estimator ``cos a = (r n - 1) / (n - r)`` (``r = S2/S1``, ``n = TR2/TR1``)
    inverts it.
    """
    e1 = torch.exp(-tr1_ms / (t1_ms + 1e-8))
    e2 = torch.exp(-tr2_ms / (t1_ms + 1e-8))
    sin_a, cos_a = torch.sin(flip_eff), torch.cos(flip_eff)
    denom = 1.0 - e1 * e2 * cos_a**2 + 1e-8
    s1 = pd * sin_a * (1.0 - e2 + (1.0 - e1) * e2 * cos_a) / denom
    s2 = pd * sin_a * (1.0 - e1 + (1.0 - e2) * e1 * cos_a) / denom
    return s1, s2


# ── Closed-form inversions (the methods' textbook estimators; used by the
#    round-trip correctness tests and available to the arm models) ──────────


def invert_double_angle(stack: torch.Tensor, flip_rad: float) -> torch.Tensor:
    """Recover the B1 transmit scale from a double-angle ``[S_alpha, S_2alpha]`` stack."""
    s_a = stack[:, 0:1]
    s_2a = stack[:, 1:2]
    cos_eff = (s_2a / (2.0 * s_a + 1e-8)).clamp(-1.0, 1.0)
    alpha_eff = torch.arccos(cos_eff)  # = alpha * B1
    return alpha_eff / flip_rad


def invert_afi(stack: torch.Tensor, flip_rad: float, tr_ratio: float) -> torch.Tensor:
    """Recover the B1 transmit scale from an AFI ``[S_TR1, S_TR2]`` stack."""
    s1 = stack[:, 0:1]
    s2 = stack[:, 1:2]
    r = s2 / (s1 + 1e-8)
    n = tr_ratio
    cos_eff = ((r * n - 1.0) / (n - r + 1e-8)).clamp(-1.0, 1.0)
    alpha_eff = torch.arccos(cos_eff)
    return alpha_eff / flip_rad


def _build_mrf_dictionary(
    n_timepoints: int, dict_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference MRF dictionary ``[dict_size, T]`` + its ``f`` grid ``[dict_size]``."""
    f = torch.linspace(0.0, 1.0, dict_size, device=device)
    t1 = torch.exp(_MRF_T1_LOG[0] + (_MRF_T1_LOG[1] - _MRF_T1_LOG[0]) * f).view(-1, 1, 1, 1)
    t2 = torch.exp(_MRF_T2_LOG[0] + (_MRF_T2_LOG[1] - _MRF_T2_LOG[0]) * f).view(-1, 1, 1, 1)
    pd = torch.ones_like(t1)
    bloch = DifferentiableBlochLayer(sequence_type="SPGR")
    sched = _mrf_flip_schedule(n_timepoints).to(device)
    sigs = [bloch(t1, t2, pd, _MRF_TR_MS, _MRF_TE_MS, fa.item()).view(dict_size) for fa in sched]
    return torch.stack(sigs, dim=1), f  # [dict_size, T], [dict_size]


def invert_mrf(stack: torch.Tensor, dict_size: int = 128) -> torch.Tensor:
    """Recover the normalised tissue parameter ``f`` from an MRF transient stack.

    Per-pixel normalised-correlation matching against a reference dictionary —
    the textbook MRF inversion; ``f`` maps monotonically to (T1, T2).
    """
    b, t, h, w = stack.shape
    d, f_grid = _build_mrf_dictionary(t, dict_size, stack.device)
    x = stack.reshape(b, t, -1).permute(0, 2, 1)  # [B, HW, T]
    scores = normalize(x, dim=-1) @ normalize(d, dim=1).T  # [B, HW, dict_size]
    return f_grid[scores.argmax(dim=-1)].view(b, 1, h, w)


def invert_bloch_siegert(stack: torch.Tensor, k_bs: float) -> torch.Tensor:
    """Recover the B1 transmit scale from a Bloch-Siegert ``[phi_+, phi_-]`` stack."""
    d_phi = stack[:, 0:1] - stack[:, 1:2]
    d_phi = torch.atan2(torch.sin(d_phi), torch.cos(d_phi))  # 2 * K_BS * B1^2
    return torch.sqrt((d_phi / (2.0 * k_bs)).clamp(min=0.0))


def _bssfp_c1_phasor(e1: float, e2: float, flip_rad: float) -> complex:
    """Unit phasor of the ESM first Fourier harmonic ``c1`` of ``F(theta)``.

    ``F(theta) = (1 - a e^{i theta}) / (1 - b cos theta)`` (``a = E2``) has a
    *global* first-harmonic phase ``angle(c1)`` set by the acquisition
    parameters — for typical relaxation ``c1`` is real-negative (``a`` dominates),
    i.e. a constant ``pi`` offset on the recovered off-resonance. Demodulating by
    this phasor is the ESM calibration that yields an absolute Hz field. Computed
    by a dense numeric DFT of the closed form (cheap, parameter-only).
    """
    ca = math.cos(flip_rad)
    denom = 1.0 - e1 * ca - e2**2 * (e1 - ca) + 1e-8
    a = e2
    b = e2 * (1.0 - e1) * (1.0 + ca) / denom
    theta = torch.linspace(0.0, 2.0 * math.pi, 512)
    f = (1.0 - a * torch.exp(1j * theta)) / (1.0 - b * torch.cos(theta))
    c1 = (f * torch.exp(-1j * theta)).mean()
    return complex(c1 / (c1.abs() + 1e-8))


def invert_bssfp_banding(
    stack: torch.Tensor,
    *,
    tr_ms: float,
    t1_ms: float,
    t2_ms: float,
    flip_rad: float,
) -> torch.Tensor:
    """Recover the B0 offset (Hz) from a phase-cycled bSSFP stack.

    The first-harmonic phase-cycle DFT (the closed-form ESM estimator): for a
    complex stack ``S_n = S(beta + 2 pi n / N)``,

        summed = sum_n S_n e^{-i 2 pi n / N}  =  N c1 M(r) e^{i beta(r)} + O(|c_{1±N}|),

    so ``angle(summed) = beta + angle(c1)``. The DFT orthogonality kernel keeps
    only harmonics ``m ≡ 1 (mod N)``; dividing out the global calibration phasor
    ``c1/|c1|`` removes the constant ``angle(c1)`` offset, leaving

        df_hat = angle( summed · conj(c1/|c1|) ) / (2 pi TR).

    Identifiable over one period ``|df| < 1/(2 TR)``; the acquisition parameters
    (TR, T1, T2, flip) are passed in exactly as ``invert_dual_echo`` takes its
    echo times.
    """
    n = stack.shape[1]
    idx = torch.arange(n, device=stack.device)
    w = torch.exp(-1j * (2.0 * math.pi / n) * idx).view(1, n, 1, 1)
    summed = (stack * w).sum(dim=1, keepdim=True)  # [B, 1, H, W] complex
    e1 = math.exp(-tr_ms / (t1_ms + 1e-8))
    e2 = math.exp(-tr_ms / (t2_ms + 1e-8))
    cal = _bssfp_c1_phasor(e1, e2, flip_rad)
    demod = summed * torch.conj(torch.tensor(cal, dtype=summed.dtype))
    return demod.angle() / (2.0 * math.pi * (tr_ms * 1e-3))


def invert_dual_echo(stack: torch.Tensor, te1_ms: float, te2_ms: float) -> torch.Tensor:
    """Recover the B0 offset (Hz) from a dual-echo ``[phi_TE1, phi_TE2]`` stack."""
    d_te_s = (te2_ms - te1_ms) * 1e-3
    d_phi = stack[:, 1:2] - stack[:, 0:1]
    # wrap to (-pi, pi] before dividing
    d_phi = torch.atan2(torch.sin(d_phi), torch.cos(d_phi))
    return d_phi / (2.0 * math.pi * d_te_s)


__all__ = [
    "PAIRED_METHODS",
    "TISSUE_ORDER",
    "AcquisitionResult",
    "MultiAcquisitionSimulator",
    "invert_afi",
    "invert_bloch_siegert",
    "invert_bssfp_banding",
    "invert_double_angle",
    "invert_dual_echo",
    "invert_mrf",
]
