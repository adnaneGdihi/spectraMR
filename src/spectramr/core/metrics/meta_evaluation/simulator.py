"""Sobol-quasirandom degradation sweep used by all five rankers.

Generates ``DegradationSample`` objects across a curated set of MRI-relevant
degradation families (Gaussian noise, Rician noise, motion blur, k-space
under-sampling, bias-field, gamma, contrast). Each family takes a single
severity coordinate ``theta in [0, 1]`` so the resulting metric response
surface lives in ``[0, 1]^D`` — the FSPD ranker assumes that hypercube
structure.

The sweep is deterministic: a single seed plus the Sobol grid uniquely
determines every sample. This is what lets meta-evaluation runs be
reproducible and what makes the bootstrap variance estimates in FSPD valid.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from .damage import YARDSTICKS, damage_batch
from .types import DegradationSample

if TYPE_CHECKING:
    from spectramr.core.metrics.context import MetricContext


def _van_der_corput(n: int, base: int) -> torch.Tensor:
    """Generate the first ``n`` van der Corput points in given base.

    Used as a Sobol-replacement that has no scipy dependency. For our
    severity-grid sizes (a few hundred) the discrepancy difference is
    negligible.
    """
    out = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        f = 1.0
        v = 0.0
        k = i + 1
        while k > 0:
            f /= base
            v += f * (k % base)
            k //= base
        out[i] = v
    return out


def halton_grid(n: int, dim: int) -> torch.Tensor:
    """[N, dim] quasirandom grid in [0, 1]^dim using prime-base Halton."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if dim > len(primes):
        raise ValueError(f"halton_grid only supports up to {len(primes)} dims")
    cols = [_van_der_corput(n, primes[d]) for d in range(dim)]
    return torch.stack(cols, dim=1)


# ---------------------------------------------------------------------------
# Degradation primitives. Each takes ``clean`` and a severity in ``[0, 1]`` and
# returns a degraded tensor of the same shape. Severity is parameter-mapped
# (e.g. for Gaussian noise theta=0 -> sigma=0, theta=1 -> sigma=0.3 * std(x)).
# ---------------------------------------------------------------------------


def _signal_magnitude(clean: torch.Tensor) -> torch.Tensor:
    """Magnitude reference for the intensity-scale degraders.

    ``tensor.float()`` on a *complex* tensor keeps only the REAL part and silently
    drops the phase, so a noise scale or intensity transform derived from it would
    be wrong (and the advertised sign-/phase-preservation contract broken). Returns
    ``|clean|`` for complex input; ``clean.float()`` otherwise (callers reapply sign).
    """
    return clean.abs() if torch.is_complex(clean) else clean.float()


def _noise_sigma(clean: torch.Tensor, theta: float) -> float:
    """Noise scale that stays monotone in theta even for low-variance inputs.

    The previous formulation ``0.3 * theta * std(clean)`` collapsed to zero
    whenever the clean image was (near-)constant — a constant ROI then
    received no noise at theta=1, faking severity zero. The floor below
    keeps noise present whenever theta > 0 regardless of image content.

    Uses the signal *magnitude* (not ``.float()``, which drops the imaginary part)
    so the scale is correct for complex k-space / image inputs.
    """
    data_std = float(_signal_magnitude(clean).std().item())
    abs_floor = 0.05 * theta  # absolute fallback if std collapses
    return max(0.3 * theta * data_std, abs_floor)


# Device-awareness convention used by every degrader below:
#   • RNG and Halton-style randomness are built on a CPU generator so that a
#     given (seed, content_id, family, severity) reproduces bit-identical
#     noise on CPU and CUDA — this is the property that keeps meta-eval
#     runs comparable across hardware.
#   • Every auxiliary tensor (kernel, mask, coord grid, field, ramp) is
#     materialized on ``clean.device`` before it touches ``clean`` so the
#     math stays on a single device. Mixing CPU and CUDA tensors in arith
#     ops raises in modern PyTorch, which is what was blocking option B.


def degrade_gaussian_noise(
    clean: torch.Tensor, theta: float, generator: torch.Generator
) -> torch.Tensor:
    sigma = _noise_sigma(clean, theta)
    if torch.is_complex(clean):
        # Proper complex Gaussian noise (real + imaginary), not real-only noise
        # cast to complex — otherwise the imaginary channel stays noise-free.
        n_re = torch.randn(clean.shape, generator=generator)
        n_im = torch.randn(clean.shape, generator=generator)
        noise = torch.complex(n_re, n_im).to(device=clean.device, dtype=clean.dtype)
        return clean + sigma * noise
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.float().dtype)
    return clean + sigma * noise.to(device=clean.device, dtype=clean.dtype)


def degrade_rician_noise(
    clean: torch.Tensor, theta: float, generator: torch.Generator
) -> torch.Tensor:
    sigma = _noise_sigma(clean, theta)
    n1 = (torch.randn(clean.shape, generator=generator) * sigma).to(clean.device)
    n2 = (torch.randn(clean.shape, generator=generator) * sigma).to(clean.device)
    if torch.is_complex(clean):
        # Complex input: add complex Gaussian noise — the magnitude is then exactly
        # Rician, and the phase is preserved rather than discarded by ``.float()``.
        noise = torch.complex(n1, n2).to(dtype=clean.dtype)
        return clean + noise
    # Rician magnitude: |A + n1 + i n2| where A is the underlying clean
    # signal. We preserve the sign of A so signed inputs (e.g. residuals
    # in zero-mean normalisation) don't get folded into magnitude.
    clean_f = clean.float()
    rician = torch.sqrt((clean_f.abs() + n1) ** 2 + n2**2)
    return (torch.sign(clean_f) * rician).to(clean.dtype)


def degrade_gaussian_blur(
    clean: torch.Tensor, theta: float, generator: torch.Generator
) -> torch.Tensor:
    # Sigma in pixels; up to 4 px blur at theta=1.
    sigma = 4.0 * theta + 1e-3
    radius = max(int(math.ceil(3 * sigma)), 1)
    coords = torch.arange(-radius, radius + 1, dtype=torch.float64)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    k1 = kernel_1d.view(1, 1, 1, -1).to(device=clean.device, dtype=clean.dtype)
    k2 = kernel_1d.view(1, 1, -1, 1).to(device=clean.device, dtype=clean.dtype)
    x = clean
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    pad = (radius, radius, radius, radius)
    x_pad = torch.nn.functional.pad(x, pad, mode="reflect")
    x_blur = torch.nn.functional.conv2d(
        x_pad, k1.expand(x.shape[1], 1, 1, k1.shape[-1]), groups=x.shape[1]
    )
    x_blur = torch.nn.functional.conv2d(
        x_blur, k2.expand(x.shape[1], 1, k2.shape[-2], 1), groups=x.shape[1]
    )
    return x_blur.view(clean.shape)


def degrade_kspace_undersample(
    clean: torch.Tensor, theta: float, generator: torch.Generator
) -> torch.Tensor:
    """Cartesian random under-sampling with theta = (1 - keep_fraction).

    theta=0 keeps everything; theta=1 keeps only ~5% of phase-encode lines.
    Center 8% is always preserved. Aliasing in the image domain is the
    visible effect.

    The mask is along the **phase-encode** axis ``H`` (dim -2); the readout axis
    ``W`` is fully sampled. This is the only physical Cartesian pattern — the
    readout is acquired "for free" during the gradient and is never skipped — and
    it matches the SSOT convention (``infrastructure/physics/sampling.py``: H is
    phase-encode, W is readout) and the physics ``degrade_cartesian_undersampling``.
    """
    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    keep = max(0.05, 1.0 - theta)
    if clean.ndim == 2:
        x = clean.unsqueeze(0).unsqueeze(0)
    elif clean.ndim == 3:
        x = clean.unsqueeze(0)
    else:
        x = clean
    x_c = x.to(torch.complex64) if not torch.is_complex(x) else x
    kspace = fft2c(x_c)
    h = kspace.shape[-2]  # phase-encode axis (under-sampled)
    n_keep = max(1, int(round(keep * h)))
    center_lines = max(1, int(round(0.08 * h)))
    # Build mask on CPU so the CPU generator stays deterministic across
    # devices, then move once to kspace.device for the multiply.
    mask = torch.zeros(h, dtype=torch.bool)
    mask[(h - center_lines) // 2 : (h + center_lines) // 2] = True
    remaining = n_keep - int(mask.sum().item())
    if remaining > 0:
        rest_idx = torch.where(~mask)[0]
        perm = torch.randperm(len(rest_idx), generator=generator)[:remaining]
        mask[rest_idx[perm]] = True
    kspace_masked = kspace * mask.view(1, 1, h, 1).to(kspace.device)
    out = ifft2c(kspace_masked)
    if torch.is_complex(clean):
        return out.view(clean.shape)
    return out.abs().to(clean.dtype).view(clean.shape)


def degrade_motion(clean: torch.Tensor, theta: float, generator: torch.Generator) -> torch.Tensor:
    """Phase-encode-direction motion ghost via random k-space line phase shifts.

    Severity controls the fraction of **phase-encode** lines (rows along ``H``,
    dim -2 — the SSOT phase-encode axis) that receive a random constant phase
    offset. A constant per-PE-line phase is the textbook rigid-translation-during-
    acquisition model: it produces the discrete ghost copies replicated along the
    phase-encode direction characteristic of patient motion. (The previous version
    modulated readout *columns* with an intra-column ramp, which is not a motion
    ghost and corrupted the wrong axis.)
    """
    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    if clean.ndim == 2:
        x = clean.unsqueeze(0).unsqueeze(0)
    elif clean.ndim == 3:
        x = clean.unsqueeze(0)
    else:
        x = clean
    x_c = x.to(torch.complex64) if not torch.is_complex(x) else x
    kspace = fft2c(x_c)
    h = kspace.shape[-2]  # phase-encode axis (corrupted lines); W readout intact
    frac = float(min(max(theta, 0.0), 1.0)) * 0.5
    n_corrupt = int(round(frac * h))
    perm = torch.randperm(h, generator=generator)[:n_corrupt]
    # One random constant phase per corrupted PE line → ghosts along H.
    phase_offsets = (torch.rand(n_corrupt, generator=generator) - 0.5) * 2 * torch.pi * theta
    out = kspace.clone()
    for idx, phi in zip(perm.tolist(), phase_offsets.tolist(), strict=True):
        factor = torch.exp(torch.tensor(1j * phi)).to(out.dtype)
        out[..., idx, :] = out[..., idx, :] * factor
    img = ifft2c(out)
    if torch.is_complex(clean):
        return img.view(clean.shape)
    return img.abs().to(clean.dtype).view(clean.shape)


def degrade_bias_field(
    clean: torch.Tensor, theta: float, generator: torch.Generator
) -> torch.Tensor:
    """Smooth multiplicative bias field. theta=1 -> max 50% intensity drift."""
    h = clean.shape[-2]
    w = clean.shape[-1]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    a = (torch.rand(1, generator=generator).item() - 0.5) * 2
    b = (torch.rand(1, generator=generator).item() - 0.5) * 2
    field_ = 1.0 + theta * 0.5 * (a * xx + b * yy)
    while field_.ndim < clean.ndim:
        field_ = field_.unsqueeze(0)
    return clean * field_.to(device=clean.device, dtype=clean.dtype)


def degrade_gamma(clean: torch.Tensor, theta: float, generator: torch.Generator) -> torch.Tensor:
    """Gamma compression. theta=1 -> gamma in [0.6, 1.4].

    Previously sampled only the sign of the offset, so at fixed theta only
    two gammas were reachable across seeds. Now samples gamma continuously
    on the symmetric interval, which gives the FSPD ranker a non-trivial
    response surface across the severity grid.
    """
    offset = (torch.rand(1, generator=generator).item() - 0.5) * 2  # ∈ [-1, 1]
    gamma = 1.0 + 0.4 * theta * offset
    eps = 1e-6
    if torch.is_complex(clean):
        # Apply the gamma compression to the magnitude and re-attach the original
        # phase, instead of dropping the imaginary part via ``.float()``.
        mag = clean.abs().clamp(min=eps)
        unit_phase = clean / mag
        return (torch.pow(mag, gamma) * unit_phase).to(clean.dtype)
    x_pos = clean.float().abs().clamp(min=eps)
    return (torch.sign(clean.float()) * torch.pow(x_pos, gamma)).to(clean.dtype)


def degrade_contrast(clean: torch.Tensor, theta: float, generator: torch.Generator) -> torch.Tensor:
    """Linear contrast scaling around per-image mean."""
    # Linear, so it applies to complex inputs directly — operate on ``clean`` (not
    # ``clean.float()``, which would drop the imaginary part) and keep the complex
    # mean. ``scale`` is real, so phase is preserved.
    c = clean if torch.is_complex(clean) else clean.float()
    mean = c.mean()
    scale = 1.0 + theta * ((torch.rand(1, generator=generator).item() - 0.5) * 0.6)
    return ((c - mean) * scale + mean).to(clean.dtype)


DEGRADATION_LIBRARY: dict[str, Callable[[torch.Tensor, float, torch.Generator], torch.Tensor]] = {
    "gaussian_noise": degrade_gaussian_noise,
    "rician_noise": degrade_rician_noise,
    "gaussian_blur": degrade_gaussian_blur,
    "kspace_undersample": degrade_kspace_undersample,
    "motion": degrade_motion,
    "bias_field": degrade_bias_field,
    "gamma": degrade_gamma,
    "contrast": degrade_contrast,
}

#: Families whose random *parameter* (gamma offset, contrast scale sign, bias-field
#: plane coefficients) must stay fixed along a severity trajectory. These degraders
#: scale a single random draw by ``theta``; if the draw is re-randomised at every
#: severity step (as it is when the per-sample seed includes the severity index),
#: the degradation *magnitude* ``theta·|offset|`` is no longer monotone in
#: ``theta`` — a high-severity sample can be *less* degraded than a low-severity one
#: (critique C1). ``run_simulator`` therefore seeds these families' generator from
#: ``(content, family)`` only, so the draw is constant across the trajectory and the
#: severity axis the rankers depend on (FSPD / ADR / isotonic calibration) is
#: genuinely monotone. The stochastic families (noise/blur/undersample/motion) draw
#: fresh per-sample randomness, which is correct — each severity is an independent
#: realisation, and their expected degradation already rises with ``theta``.
TRAJECTORY_STABLE_FAMILIES: frozenset[str] = frozenset({"gamma", "contrast", "bias_field"})


@dataclass
class SimulatorConfig:
    # ``None`` → every family of the *effective* operator library (the 8
    # dependency-free core ops by default, or the injected library if given).
    families: list[str] | None = None
    n_severities_per_family: int = 16
    seed: int = 42
    # Dependency-inversion seam (CLAUDE.md pitfall #13: deps point inward only).
    # The meta-evaluation simulator lives in ``core/`` and therefore *cannot*
    # import the degradation SSOT in ``infrastructure/physics`` directly. An
    # outer layer (the ``scripts/sim2rank`` legacy pipeline or the CLI, which may
    # import physics) injects a richer operator library here — name → callable
    # ``(clean, theta, generator) -> tensor`` — so the unified physics
    # degradations run through this same simulator with no layer violation and
    # no duplicate implementation. ``None`` keeps the 8 self-contained core ops.
    # Production ``spectramr meta-evaluate`` injects the physics SSOT by default (C2);
    # the core ops are the **dependency-free fallback** for core-layer unit tests /
    # ``--core-degradations`` runs, NOT a second production degradation library.
    operator_library: (
        dict[str, Callable[[torch.Tensor, float, torch.Generator], torch.Tensor]] | None
    ) = field(default=None, repr=False)
    # Per-family ``theta_local -> {"physical": ..., "native": ...}``, injected from the
    # application layer (core cannot import the severity resolver). ``None`` adds no keys.
    severity_annotators: Mapping[str, Callable[[float], dict[str, float]]] | None = field(
        default=None, repr=False
    )
    # Yardsticks measured per sample into ``severity["damage_<name>"]``; empty adds no keys.
    damage_yardsticks: tuple[str, ...] = ()
    # Stamped verbatim into ``degradation_provenance`` by the pipeline.
    severity_provenance: Mapping[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        library = self.operator_library or DEGRADATION_LIBRARY
        if self.families is None:
            self.families = list(library.keys())
        unknown = set(self.families) - library.keys()
        if unknown:
            raise ValueError(f"unknown degradation families: {sorted(unknown)}")
        self.damage_yardsticks = tuple(self.damage_yardsticks)
        bad = [y for y in self.damage_yardsticks if y not in YARDSTICKS]
        if bad:
            raise ValueError(
                f"unknown damage yardstick(s) {bad}; expected from {sorted(YARDSTICKS)}"
            )
        if self.severity_annotators is not None:
            missing = sorted(set(self.families) - set(self.severity_annotators))
            if missing:
                raise ValueError(f"severity_annotators has no entry for families {missing}")

    @property
    def effective_library(
        self,
    ) -> dict[str, Callable[[torch.Tensor, float, torch.Generator], torch.Tensor]]:
        """The operator library actually used: injected if provided, else core."""
        return self.operator_library or DEGRADATION_LIBRARY


#: Stamped beside ``seed`` in every bundle: draws are comparable only within one scheme.
SEED_SCHEME = "blake2b_v1"


def content_seed(seed: int, content_id: str) -> int:
    """32-bit per-content seed hashed over the WHOLE id, so ids sharing a prefix differ."""
    digest = hashlib.blake2b(f"{seed}:{content_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0xFFFFFFFF


def run_simulator(
    clean_volumes: Iterable[tuple[str, torch.Tensor]],
    config: SimulatorConfig,
    assets_by_content: Mapping[str, MetricContext] | None = None,
    subject_of: Mapping[str, str] | None = None,
) -> list[DegradationSample]:
    """Sweep all (family, severity) pairs over each clean reference.

    Args:
        clean_volumes: iterable of (content_id, tensor) pairs. Tensors can be
            2-D, 3-D (channel + spatial), or 4-D (volume). Shape is preserved
            in the degraded output.
        config: simulator configuration.
        assets_by_content: optional ``{content_id: MetricContext}`` of real
            acquisition assets (coil maps, acquired k-space, reconstructor,
            noise cov). When a content_id is present, its samples carry the real
            context so the NR/physics battery is graded on the true acquisition
            rather than a synthetic stand-in. Absent content_ids fall back to the
            synthesised context (magnitude-only cohort).
        subject_of: optional ``{content_id: subject_id}``. When given it must name
            every content, so no sample silently falls back to its own content id.

    Returns:
        A flat list of ``DegradationSample`` objects.
    """
    samples: list[DegradationSample] = []
    assets_by_content = assets_by_content or {}
    severities = halton_grid(config.n_severities_per_family, dim=1).flatten()
    library = config.effective_library

    seen: set[str] = set()
    for content_id, clean in clean_volumes:
        seen.add(content_id)
        subject_id = None
        if subject_of is not None:
            if content_id not in subject_of:
                raise ValueError(f"subject_of names no subject for content {content_id!r}.")
            subject_id = subject_of[content_id]
            if not subject_id:
                raise ValueError(f"subject_of gives content {content_id!r} an empty subject id.")
        base_seed = content_seed(config.seed, content_id)
        for fi, family in enumerate(config.families or ()):
            fn = library[family]
            stable = family in TRAJECTORY_STABLE_FAMILIES
            annotate = config.severity_annotators[family] if config.severity_annotators else None
            family_samples: list[DegradationSample] = []
            for si, theta in enumerate(severities.tolist()):
                # Per-sample generator so single (content, family, theta) always
                # yields the same image. Trajectory-stable families omit the
                # severity index ``si`` from the seed so their random parameter is
                # drawn ONCE per (content, family) and only ``theta`` scales it —
                # keeping the degradation monotone in severity (critique C1).
                seed = (
                    (base_seed + fi * 7919) & 0xFFFFFFFF
                    if stable
                    else (base_seed + fi * 7919 + si * 13) & 0xFFFFFFFF
                )
                gen = torch.Generator().manual_seed(seed)
                degraded = fn(clean.detach(), float(theta), gen)
                severity: dict[str, float] = {"theta": float(theta)}
                if annotate is not None:
                    severity.update(annotate(float(theta)))
                family_samples.append(
                    DegradationSample(
                        clean=clean.detach(),
                        degraded=degraded.detach(),
                        degradation_family=family,
                        severity=severity,
                        content_id=content_id,
                        seed=seed,
                        subject_id=subject_id,
                        assets=assets_by_content.get(content_id),
                    )
                )
            # One batched yardstick call and one device->host transfer per (content, family).
            for y in config.damage_yardsticks:
                stack = torch.stack([smp.degraded for smp in family_samples])
                values = damage_batch(clean.detach(), stack, y).tolist()
                for smp, v in zip(family_samples, values, strict=True):
                    smp.severity[f"damage_{y}"] = float(v)
            samples.extend(family_samples)
    if subject_of is not None and set(subject_of) - seen:
        raise ValueError(
            f"subject_of names contents that were never simulated: {sorted(set(subject_of) - seen)}"
        )
    return samples


def _synth_smaps(img: torch.Tensor, n_coils: int = 4) -> torch.Tensor:
    """Smooth complex sensitivity maps for the NR-context measurement.

    A compact birdcage-style stand-in: ``n_coils`` low-frequency Gaussian
    bumps at distinct centres, RSS-normalised so ``sum_c |S_c|^2 ≈ 1``. Only
    used to synthesise ``y_kspace``/``coil_maps`` for the concordance sweep —
    real ESPIRiT maps are used at training time.
    """
    h, w = img.shape[-2], img.shape[-1]
    yy = torch.linspace(-1, 1, h, device=img.device).view(h, 1)
    xx = torch.linspace(-1, 1, w, device=img.device).view(1, w)
    centres = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]
    maps = []
    for i in range(n_coils):
        cy, cx = centres[i % len(centres)]
        mag = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 0.8)
        phase = torch.pi * 0.5 * (cy * yy + cx * xx)
        maps.append((mag * torch.exp(1j * phase)).to(torch.complex64))
    s = torch.stack(maps, dim=0).unsqueeze(0)  # [1, n, H, W]
    rss = torch.sqrt((s.abs() ** 2).sum(dim=1, keepdim=True)).clamp(min=1e-6)
    return s / rss


def _build_sample_context(
    clean: torch.Tensor,
    needs: tuple[str, ...],
    real: MetricContext | None = None,
) -> object:
    """Assemble a MetricContext for the declared ``needs``.

    When ``real`` (the content's prepared acquisition assets) is supplied, every
    field it populates is used verbatim — real coil maps, acquired k-space,
    reconstructor, noise_cov, acq_params — and only the STILL-missing needs are
    synthesised from ``clean``. This is what lets a multi-coil k-space cohort be
    graded on its true acquisition instead of the birdcage stand-in below.

    On a magnitude-only cohort (``real is None``) the measurement is synthesised
    from ``clean`` so a degraded recon's inconsistency rises monotonically with
    severity (the concordance property sim2rank ranks). Fields derivable from
    neither (prior_model / encoder) are left unset → the metric returns NaN
    there, which is the correct neutral on this cohort.
    """
    from dataclasses import fields as _dc_fields

    from spectramr.core.metrics.context import MetricContext

    needset = set(needs)
    c = clean.float()
    while c.dim() < 4:
        c = c.unsqueeze(0)
    if c.dim() > 4:
        c = c.reshape(c.shape[0], -1, c.shape[-2], c.shape[-1])
    c = c[:, :1]  # [1, 1, H, W] magnitude reference

    kw: dict[str, Any] = {}
    # 1. Seed from the real prepared assets (whatever fields are populated).
    if real is not None:
        for f in _dc_fields(MetricContext):
            v = getattr(real, f.name, None)
            if v is not None:
                kw[f.name] = v

    smaps = kw.get("coil_maps")
    # 2. Synthesise only the needs the real assets did not already supply.
    if "coil_maps" in needset and "coil_maps" not in kw:
        smaps = _synth_smaps(c)
        kw["coil_maps"] = smaps
    if ({"mask", "y_kspace"} & needset) and "mask" not in kw:
        kw["mask"] = None  # fully-sampled measurement
    if "y_kspace" in needset and "y_kspace" not in kw:
        from spectramr.infrastructure.physics.fft_ops import sense_forward

        kw["y_kspace"] = sense_forward(c, smaps, None)
    if "foreground_mask" in needset and "foreground_mask" not in kw:
        mag = c.abs()
        kw["foreground_mask"] = (mag > 0.05 * float(mag.amax())).float()
    if "reconstructor" in needset and "reconstructor" not in kw:
        from spectramr.infrastructure.physics.fft_ops import sense_adjoint

        def _recon(y: torch.Tensor, _s: object = smaps) -> torch.Tensor:
            return sense_adjoint(y, _s)

        kw["reconstructor"] = _recon
    return MetricContext(**kw)


def precompute_metric_values(
    samples: Sequence[DegradationSample],
    metrics: dict[str, Callable[..., float]],
    *,
    log_every: int = 32,
    complex_metric_keys: frozenset[str] | set[str] = frozenset(),
) -> dict[str, list[float]]:
    """Run every metric on every sample once, store the scalars.

    Errors in individual metric calls are caught and replaced with NaN —
    one broken metric should not abort the whole evaluation. Downstream
    rankers must handle NaN gracefully (mask before correlation /
    eigendecomposition).

    ``complex_metric_keys`` are metrics declaring ``requires_complex``. They are
    fed the sample's complex tensor when it carries one; on a magnitude-only
    cohort (the simulated clean references are magnitude pseudo-GT, which has no
    phase) they cannot be computed, so their value is recorded as ``NaN`` and the
    skip is **logged once** — honouring the spec flag (CLAUDE.md pitfall #15)
    instead of silently letting the metric crash to NaN with no breadcrumb.

    fp32 enforcement
    ~~~~~~~~~~~~~~~~

    The precompute loop runs under ``torch.amp.autocast(..., enabled=False)``
    and explicitly casts each sample tensor to ``.float()`` before the
    metric call. This defeats any outer-scope autocast context (e.g.
    a stray AMP block from upstream calibration code) so the
    leaderboard is bit-stable across runs and never picks up
    bf16/fp16-quantised metric values.

    Args:
        samples: degradation samples (N).
        metrics: ``{name: callable(pred, target) -> float}``.
        log_every: emit a progress line every N samples. The
            precompute step has total cost N · M metric calls which
            for typical sim2rank runs (~256 samples × 107 metrics ≈ 27k
            calls) easily exceeds 10 min when any metric is slow.
            Without periodic logging the run looks like a hang.

    Returns:
        ``{metric_name: [v_1, …, v_N]}``.
    """
    import logging
    import time

    log = logging.getLogger(__name__)

    # Which metrics need a MetricContext (NR battery, spec §1.2)? Resolved once
    # from the registry; the per-sample context is synthesised below.
    try:
        from spectramr.core.metrics.registry import MetricsRegistry

        context_needs: dict[str, tuple[str, ...]] = {
            name: MetricsRegistry.needs(name)
            for name in metrics
            if MetricsRegistry.needs_context(name)
        }
    except Exception:
        context_needs = {}

    N = len(samples)
    M = len(metrics)
    out: dict[str, list[float]] = {name: [] for name in metrics}
    log.info(
        "precompute_metric_values starting: %d samples × %d metrics = %d calls (fp32 enforced)",
        N,
        M,
        N * M,
    )
    t0 = time.monotonic()

    # Disable both CUDA and CPU autocast so a stray outer context can't
    # promote/demote precision under us.
    with (
        torch.amp.autocast("cuda", enabled=False),
        torch.amp.autocast("cpu", enabled=False),
    ):
        skipped_complex: set[str] = set()
        for i, sample in enumerate(samples):
            # Explicit fp32 cast (kept complex unchanged — some metrics
            # consume complex k-space directly).
            deg = sample.degraded.float() if not sample.degraded.is_complex() else sample.degraded
            clean = sample.clean.float() if not sample.clean.is_complex() else sample.clean
            sample_is_complex = deg.is_complex()
            # Build ONE MetricContext per sample (union of all NR metrics' needs)
            # from the clean reference's forward model. Lazy — only when an NR
            # context-needing metric is present.
            sample_ctx = None
            if context_needs:
                union: tuple[str, ...] = tuple(
                    {f for needs in context_needs.values() for f in needs}
                )
                try:
                    sample_ctx = _build_sample_context(clean, union, real=sample.assets)
                except Exception:
                    sample_ctx = None
            for name, fn in metrics.items():
                if name in complex_metric_keys and not sample_is_complex:
                    # requires_complex metric on a magnitude-only cohort: there is
                    # no phase to measure. Record NaN, but log the reason once so
                    # the skip is diagnosable (not a silent crash-to-NaN).
                    if name not in skipped_complex:
                        skipped_complex.add(name)
                        log.warning(
                            "metric %r requires complex input but the cohort is "
                            "magnitude-only — recording NaN (skipped). Provide "
                            "complex clean references to enable it.",
                            name,
                        )
                    out[name].append(float("nan"))
                    continue
                try:
                    if name in context_needs and sample_ctx is not None:
                        # Context MUST be keyword: raw registry metrics declare
                        # ``*, context=None`` (keyword-only), so a positional
                        # third arg TypeErrors -> caught -> silent NaN. That was
                        # NaN-ing the entire NR battery on the raw-metric (novel
                        # core) path. Keyword works for both raw and the wrapped
                        # ``safe_metric_call`` form.
                        v = float(fn(deg, clean, context=sample_ctx))
                    else:
                        v = float(fn(deg, clean))
                except Exception:
                    v = float("nan")
                out[name].append(v)
            if log_every > 0 and (i + 1) % log_every == 0:
                dt = time.monotonic() - t0
                rate = (i + 1) / max(dt, 1e-6)
                eta = (N - (i + 1)) / max(rate, 1e-6)
                log.info(
                    "precompute: %d/%d samples (%.1fs elapsed, ~%.1fs eta, %.1f sample/s)",
                    i + 1,
                    N,
                    dt,
                    eta,
                    rate,
                )
    return out
