"""Per-layer attention energy probe for the kspace_cold_diffusion cohort.

Measures the energy gain ``rho = ||output|| / ||input||`` of each attention
block in an experiment_11 arm. Training is single-step, so a per-layer
``rho > 1`` is invisible; the 20-step reverse sampler feeds the model its own
drifting out-of-distribution (OOD) states and compounds ``rho ** N`` into the
saturated central k-space band.

Since 2026-07-24 (issue #471) every block is wrapped in
``IdentityAtInitAttention`` -- ``y = x + gamma * (inner(x) - x)`` with ``gamma``
zero-initialised -- so the ``.attention`` site measures the block's **effective**
gain (``rho == 1.000`` at init, rising as ``gamma`` is learned). The raw,
unscaled block gain is still measured at the ``.attention.inner`` site, which is
what the pre-wrapper ``rho`` meant. Read both: a large ``inner`` rho with
``gamma`` near zero is a block the optimiser has chosen not to switch on, a
different diagnosis from a large effective rho.

Two measurements:

* **A (controlled):** one forward pass on an in-distribution synthetic batch and
  on magnitude-scaled / phase-perturbed OOD batches; record per-block ``rho`` and
  how it grows with the scale factor (the input-dependence signature).
* **B (trajectory, accelerator only):** run the real ``generator.sample`` reverse
  loop with the same hooks; record per-block ``rho`` and the model-output norm at
  each reverse step (the actual compounding on the model's own states).

Softmax / sigmoid families (channel, sparse=WindowAttention, self=LinearAttention)
are expected ``rho <= 1`` and input-robust; un-normalized linear (kernelized) and
KAN-gated (kan_dual_domain, wavelet_freq) are expected ``rho > 1`` on OOD inputs.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from dataclasses import dataclass, field
from typing import Any

import torch

from spectramr.infrastructure.training.debug_snapshot import _tensor_report

__all__ = [
    "ArmEnergyReport",
    "BlockEnergy",
    "ScaleEnergy",
    "StepEnergy",
    "build_ood_batch",
    "build_probe_batch",
    "build_probe_model",
    "measure_forward_energy",
    "measure_trajectory_energy",
    "probe_arm",
    "rank_arms",
    "register_energy_hooks",
]

_EPS = 1e-12


# ── data records ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BlockEnergy:
    """Energy gain of a single attention block on one forward call."""

    name: str
    kind: str
    in_norm: float
    out_norm: float
    rho: float
    in_stats: dict[str, Any] = field(default_factory=dict)
    out_stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScaleEnergy:
    """Per-block gains at one OOD magnitude scale (Measurement A)."""

    scale: float
    phase_perturbed: bool
    blocks: tuple[BlockEnergy, ...]
    max_rho: float
    output_norm: float


@dataclass(frozen=True)
class StepEnergy:
    """Per-block gains at one reverse-sampler step (Measurement B)."""

    step: int
    blocks: tuple[BlockEnergy, ...]
    max_rho: float
    model_output_norm: float


@dataclass(frozen=True)
class ArmEnergyReport:
    """Aggregated energy report for one arm."""

    arm_name: str
    attention_type: str
    model_type: str
    device: str
    per_scale: tuple[ScaleEnergy, ...]
    trajectory: tuple[StepEnergy, ...] | None
    worst_rho: float | None
    rho_growth: float | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ── energy primitives ─────────────────────────────────────────────────────────
def _energy(t: torch.Tensor) -> float:
    """Frobenius norm in fp32 (``.abs()`` first for genuine complex)."""
    x = t.detach()
    x = x.abs().float() if x.is_complex() else x.float()
    return float(x.norm().item())


def _rho(in_norm: float, out_norm: float, eps: float = _EPS) -> float:
    return out_norm / max(in_norm, eps)


def _principal_tensor(output: Any) -> torch.Tensor | None:
    """First tensor of a block's return (tuple -> [0], dict -> first value)."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0] if isinstance(output[0], torch.Tensor) else None
    if isinstance(output, dict):
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v
    return None


def _report_dict(name: str, t: torch.Tensor | None) -> dict[str, Any]:
    if not isinstance(t, torch.Tensor):
        return {}
    return dataclasses.asdict(_tensor_report(name, t))


# ── forward hooks ─────────────────────────────────────────────────────────────
def _make_hook(name: str, kind: str, records: dict[str, list[BlockEnergy]]):
    def hook(_module: Any, inputs: Any, output: Any) -> None:
        x = inputs[0] if inputs else None
        out = _principal_tensor(output)
        in_norm = _energy(x) if isinstance(x, torch.Tensor) else math.nan
        out_norm = _energy(out) if isinstance(out, torch.Tensor) else math.nan
        records[name].append(
            BlockEnergy(
                name=name,
                kind=kind,
                in_norm=in_norm,
                out_norm=out_norm,
                rho=_rho(in_norm, out_norm),
                in_stats=_report_dict(f"{name}.in", x),
                out_stats=_report_dict(f"{name}.out", out),
            )
        )

    return hook


def register_energy_hooks(
    model: torch.nn.Module,
) -> tuple[list[Any], dict[str, list[BlockEnergy]]]:
    """Hook every attention block. Returns ``(handles, records)``.

    Selects ``named_modules`` whose name ends in ``.attention`` (the effective,
    gamma-scaled site) or ``.inner`` directly beneath one (the raw block gain), and
    are not ``nn.Identity`` (the ``attention_type: none`` arm). The block class name
    is recorded so a new/unrecognised attention type surfaces rather than being
    silently dropped -- for a wrapped site the INNER class is recorded. Records are append-only: one entry per forward call, so the
    reverse-sampler trajectory (Measurement B) slices per step by call order.
    Caller MUST remove the handles (use ``try/finally``).
    """
    records: dict[str, list[BlockEnergy]] = {}
    handles: list[Any] = []
    for name, module in model.named_modules():
        if not name:
            continue
        parts = name.split(".")
        # ``.attention`` = effective (gamma-scaled) gain; ``.attention.inner`` = raw
        # block gain, which is what ``rho`` meant before the #471 wrapper.
        is_site = parts[-1] == "attention"
        is_inner = len(parts) >= 2 and parts[-1] == "inner" and parts[-2] == "attention"
        if not (is_site or is_inner):
            continue
        if isinstance(module, torch.nn.Identity):
            continue
        # Label with the MECHANISM, not the wrapper: otherwise every arm reports
        # ``IdentityAtInitAttention`` and the probe stops naming what it exists to
        # identify (issue #471).
        labelled = getattr(module, "inner", module)
        records.setdefault(name, [])
        handles.append(
            module.register_forward_hook(_make_hook(name, type(labelled).__name__, records))
        )
    return handles, records


# ── OOD batch construction ────────────────────────────────────────────────────
def build_ood_batch(
    x_base: torch.Tensor,
    *,
    scale: float,
    phase_perturb: bool,
    seed: int = 0,
) -> torch.Tensor:
    """Out-of-distribution variant of an interleaved-complex k-space batch.

    ``scale`` multiplies the magnitude (probes magnitude-dependent gain).
    ``phase_perturb`` rotates each ``[real, imag]`` channel pair by a
    deterministic linear phase ramp (magnitude-preserving; probes the
    phase-dependent gain of KAN / dual-domain blocks). Layout is
    ``[B, 2C, H, W]`` or ``[B, 2C, H, W, D]`` with channels ordered
    ``R0, I0, R1, I1, ...``; the ramp spans (H, W) and broadcasts over any
    trailing dims.

    Raises on a layout it cannot rotate rather than returning ``x`` unchanged.
    The earlier silent ``return x`` made this a no-op on **every** cohort arm:
    ``patch_size: [256, 256, 1]`` yields a 5-D batch, so the phase sweep emitted
    records labelled ``phase_perturbed=True`` that were byte-identical to the
    magnitude-only ones — a facade (pitfall #16) at 2x the compute.
    """
    x = x_base * float(scale)
    if not phase_perturb:
        return x
    if x.dim() < 4:
        raise ValueError(
            f"energy_probe: phase perturbation needs [B, 2C, H, W(, ...)]; got "
            f"{tuple(x.shape)} ({x.dim()}-D)."
        )
    if x.shape[1] % 2 != 0:
        raise ValueError(
            f"energy_probe: phase perturbation needs an even channel count "
            f"(interleaved real/imag pairs); got C={x.shape[1]} in {tuple(x.shape)}."
        )
    h, w = int(x.shape[2]), int(x.shape[3])
    trailing = (1,) * (x.dim() - 4)  # broadcast over e.g. the depth axis of [B,C,H,W,D]
    # Seeded but deterministic ramp direction (vary by seed without RNG state).
    kx = 1.0 + (seed % 3)
    ky = 1.0 + ((seed // 3) % 3)
    yy = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1, *trailing)
    xx = torch.linspace(0, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w, *trailing)
    phi = 2.0 * math.pi * (ky * yy + kx * xx)  # [1,1,H,W(,1...)]
    cos, sin = torch.cos(phi), torch.sin(phi)
    out = x.clone()
    real = x[:, 0::2]  # [B, C, H, W]
    imag = x[:, 1::2]
    out[:, 0::2] = real * cos - imag * sin
    out[:, 1::2] = real * sin + imag * cos
    return out


# ── model + synthetic batch (self-contained; mirrors forward_probe direct path)─
def _in_channels(config: Any) -> int:
    m = config.model
    mk = dict(getattr(m, "model_kwargs", None) or {})
    val = getattr(m, "in_channels", None)
    if val is None:
        val = mk.get("in_channels")
    if val is None:
        raise ValueError("energy_probe: config.model.in_channels is unset.")
    return int(val)


def _out_channels(config: Any) -> int:
    m = config.model
    mk = dict(getattr(m, "model_kwargs", None) or {})
    val = getattr(m, "out_channels", None)
    if val is None:
        val = mk.get("out_channels")
    if val is None:
        raise ValueError("energy_probe: config.model.out_channels is unset.")
    return int(val)


def build_probe_model(config: Any, device: str) -> torch.nn.Module:
    """Instantiate the arm's generator on ``device`` (direct, non-wrapper path).

    Constructor kwargs come from :func:`resolve_full_generator_kwargs`, the one
    owner ``ModelBuilder`` and ``audit --probe`` resolve through: a probe that
    builds a *different* model measures a model nothing trains. A second
    derivation here omitted ``kspace_log_scaled`` (#1281), so every arm
    declaring ``output_kspace_clip_ratio`` raised at construction and the probe
    skipped 12 of 12 arms in its own cohort (#2122).

    That SSOT withholds ``input_type`` / ``output_type`` (``SKIP_MODEL_FIELDS``),
    which strict sub-configs raise on.

    Diffusion wrapper models (a top-level ``denoising_model`` dict) raise here
    rather than mis-building.
    """
    from spectramr.infrastructure.builders.generator_kwargs import (
        resolve_full_generator_kwargs,
    )
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import get_model_class

    populate_model_registry()
    m = config.model
    model_kwargs = dict(getattr(m, "model_kwargs", None) or {})
    if getattr(m, "denoising_model", None) is not None or "denoising_model" in model_kwargs:
        raise ValueError(
            "energy_probe: diffusion-wrapper models (denoising_model) are not "
            "supported; the exp_11 attention arms are direct kspace_cold_diffusion."
        )
    cls = get_model_class(m.model_type)

    # ``device`` is forwarded so step 3d injects the probe's own device into the
    # one generator that names it, rather than leaving the constructor to
    # resolve one (non-negotiable 9b).
    ctor: dict[str, Any] = dict(
        resolve_full_generator_kwargs(config, model_cls=cls, device=device).kwargs
    )

    sig = inspect.signature(cls.__init__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    in_ok = out_ok = True
    if not has_var_kw:
        valid = {
            n
            for n, p in sig.parameters.items()
            if n != "self"
            and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        ctor = {k: v for k, v in ctor.items() if k in valid}
        in_ok = "in_channels" in valid
        out_ok = "out_channels" in valid

    head: dict[str, Any] = {}
    if in_ok:
        head["in_channels"] = _in_channels(config)
    if out_ok:
        head["out_channels"] = _out_channels(config)
    model = cls(**head, **ctor)
    return model.to(device).eval()


def _spatial(config: Any) -> list[int]:
    ps = list(config.data.sampling.patch_size or [256, 256])
    return [int(v) for v in ps]


def _batch_size(config: Any) -> int:
    return max(1, min(2, int(config.data.loader.batch_size or 2)))


def build_probe_batch(model: torch.nn.Module, config: Any, device: str) -> torch.Tensor:
    """Synthetic in-distribution **k-space** input matching the model's forward width.

    Width doubles when the generator concatenates coil maps at runtime. The
    answer comes from :func:`model_expects_smaps_concat`, the one resolver for
    that question (CLAUDE.md #17) -- **not** from ``condition_with_smaps``,
    which is the arm's *declaration* and is still ``True`` on the six
    internal-DC arms whose backbone is built at ``1x``. Reading the declaration
    here sized a ``2x`` batch for those and squeezed it through an untrained
    ``ChannelAdapter``; it now raises instead (#1326).

    The batch is built as a genuine k-space tensor, because rho for softmax /
    top-k / KAN-gated blocks is strongly input-distribution dependent and these
    are k-space models — feeding them an image-domain tensor measures the wrong
    operating point. Pipeline: 2-D Shepp-Logan phantom -> per-coil complex
    sensitivity modulation -> ``fft2c`` -> interleaved ``[R0, I0, R1, I1, ...]``.

    Two defects this replaced, both of which silently produced a degenerate input:

    * The phantom was requested at the model's 5-D spatial layout
      ``[B, C, H, W, D]``, but :func:`synthetic_phantom` reads 5-D as
      ``(B, C, D, H, W)``. With ``patch_size: [256, 256, 1]`` that built a
      256x1 phantom replicated across every row: **1 distinct row, 5 unique
      values in the whole plane**. Requesting the 4-D shape and appending the
      trailing axes afterwards gives real 2-D structure (216 distinct rows).
    * Every channel was identical, so each ``[real, imag]`` pair had
      ``imag == real`` — a constant global 45 degree phase and zero coil
      diversity.
    """
    from spectramr.infrastructure.physics.fft_ops import fft2c
    from spectramr.infrastructure.validation.phantom_builder import synthetic_phantom
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        model_expects_smaps_concat,
    )

    in_ch = _in_channels(config)
    width = in_ch * 2 if model_expects_smaps_concat(model) else in_ch
    if width % 2 != 0:
        raise ValueError(
            f"energy_probe: interleaved-complex input needs an even channel width, "
            f"got {width} (in_channels={in_ch})."
        )
    n_coils = width // 2
    batch = _batch_size(config)
    spatial = _spatial(config)
    if len(spatial) < 2:
        raise ValueError(f"energy_probe: need at least 2 spatial dims, got {spatial}.")
    h, w = int(spatial[0]), int(spatial[1])
    trailing = [int(v) for v in spatial[2:]]

    # 2-D image-domain phantom (4-D request -> real (H, W) structure).
    img = synthetic_phantom([batch, n_coils, h, w], device=device, dtype="float32").to(device)

    # Per-coil complex sensitivity: distinct smooth magnitude ramp + phase ramp, so
    # coils are not copies of one another (real arrays have spatial coil diversity).
    yy = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w)
    idx = torch.arange(n_coils, device=device, dtype=torch.float32).view(1, n_coils, 1, 1)
    ang = 2.0 * math.pi * idx / max(1, n_coils)
    sens_mag = 0.6 + 0.4 * torch.cos(ang) * yy + 0.4 * torch.sin(ang) * xx
    sens_phase = 2.0 * math.pi * ((idx + 1.0) * (0.5 * yy + 0.25 * xx))
    coil_img = (img * sens_mag).to(torch.complex64) * torch.exp(
        1j * sens_phase.to(torch.float32)
    ).to(torch.complex64)

    # Physics SSOT: centered, norm="ortho" (non-negotiable #2 -- never raw torch.fft).
    kspace = fft2c(coil_img)  # [B, n_coils, H, W] complex

    out = torch.empty(batch, width, h, w, device=device, dtype=torch.float32)
    out[:, 0::2] = kspace.real
    out[:, 1::2] = kspace.imag
    for _ in trailing:
        out = out.unsqueeze(-1)
    if trailing:
        out = out.expand(batch, width, h, w, *trailing).contiguous()
    return out


def _num_timesteps(config: Any) -> int:
    mk = dict(getattr(config.model, "model_kwargs", None) or {})
    t = mk.get("timesteps")
    if t is None:
        diff = getattr(getattr(config, "training", None), "diffusion", None)
        t = getattr(diff, "timesteps", None)
    return int(t) if t else 28


# ── measurement A: controlled OOD forward sweep ───────────────────────────────
def measure_forward_energy(
    model: torch.nn.Module,
    x_base: torch.Tensor,
    *,
    scales: list[float],
    phase_perturb: bool,
    timestep: int,
) -> tuple[ScaleEnergy, ...]:
    """Per-block ``rho`` on in-distribution + magnitude/phase OOD batches."""
    device = x_base.device
    out_scales: list[ScaleEnergy] = []
    handles, records = register_energy_hooks(model)
    try:
        variants: list[tuple[float, bool]] = [(s, False) for s in scales]
        if phase_perturb:
            variants += [(s, True) for s in scales]
        for scale, perturb in variants:
            for k in records:
                records[k].clear()
            x = build_ood_batch(x_base, scale=scale, phase_perturb=perturb, seed=int(scale))
            t = torch.full((x.shape[0],), int(timestep), device=device, dtype=torch.long)
            with torch.no_grad():
                out = model(x, timesteps=t)
            blocks = tuple(records[k][-1] for k in records if records[k])
            max_rho = max((b.rho for b in blocks), default=math.nan)
            out_norm = (
                _energy(_principal_tensor(out)) if _principal_tensor(out) is not None else math.nan
            )
            out_scales.append(
                ScaleEnergy(
                    scale=float(scale),
                    phase_perturbed=perturb,
                    blocks=blocks,
                    max_rho=max_rho,
                    output_norm=out_norm,
                )
            )
    finally:
        for h in handles:
            h.remove()
    return tuple(out_scales)


# ── measurement B: real reverse-sampler trajectory (accelerator only) ─────────
def measure_trajectory_energy(
    model: torch.nn.Module,
    config: Any,
    device: str,
) -> tuple[StepEnergy, ...]:
    """Per-block ``rho`` + model-output norm across the real reverse loop.

    Runs ``model.sample`` on a synthetic undersampled measurement; the attention
    hooks fire once per reverse step, so per-step slicing is by call order. Also
    captures the generator's output norm per step via a top-level hook.
    """
    in_ch = _in_channels(config)
    from spectramr.infrastructure.validation.phantom_builder import synthetic_phantom

    shape = [_batch_size(config), in_ch, *_spatial(config)]
    measurement = synthetic_phantom(shape, device=device, dtype="float32").to(device)

    out_norms: list[float] = []

    def _out_hook(_m: Any, _i: Any, o: Any) -> None:
        t = _principal_tensor(o)
        if isinstance(t, torch.Tensor):
            out_norms.append(_energy(t))

    # [S-MAP CONDITIONING] Gate on the generator's own contract, exactly as
    # ColdDiffusionInferenceStrategy and forward_probe do — this is a further
    # reader of one rule, not a new spelling (CLAUDE.md #17).
    #
    # ``measurement`` must stay pure k-space (``sample()`` runs its own
    # physics), so the maps are handed over SEPARATELY and the generator
    # completes the stack per reverse step. Without them the reverse loop used
    # to run through an untrained ChannelAdapter and every energy reading here
    # described a random projection rather than the trained backbone (#1326);
    # it now raises instead. Synthetic unit maps are the right calibration for a
    # phantom probe: this measures energy propagation, not reconstruction.
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        model_expects_smaps_concat,
    )

    probe_kwargs: dict[str, Any] = {"mask": None}
    if model_expects_smaps_concat(model) and in_ch % 2 == 0:
        probe_kwargs["smaps"] = (
            torch.ones(
                measurement.shape[0],
                in_ch // 2,
                *measurement.shape[2:],
                dtype=torch.complex64,
                device=measurement.device,
            )
            / max(1, in_ch // 2) ** 0.5
        )

    handles, records = register_energy_hooks(model)
    top = model.register_forward_hook(_out_hook)
    try:
        with torch.no_grad():
            model.sample(measurement, **probe_kwargs)
    finally:
        top.remove()
        for h in handles:
            h.remove()

    n_steps = min((len(v) for v in records.values()), default=0)
    steps: list[StepEnergy] = []
    for i in range(n_steps):
        blocks = tuple(records[k][i] for k in records if len(records[k]) > i)
        steps.append(
            StepEnergy(
                step=i,
                blocks=blocks,
                max_rho=max((b.rho for b in blocks), default=math.nan),
                model_output_norm=out_norms[i] if i < len(out_norms) else math.nan,
            )
        )
    return tuple(steps)


# ── aggregation + ranking ─────────────────────────────────────────────────────
def _worst_and_growth(
    per_scale: tuple[ScaleEnergy, ...],
) -> tuple[float | None, float | None]:
    if not per_scale:
        return None, None
    measured = [s.max_rho for s in per_scale]
    # A non-finite rho means the block overflowed -- the MOST unstable outcome there
    # is, so it must dominate the ranking. Filtering it out (the previous behaviour)
    # left worst=None, which `verdict` reads as "baseline(no-attn)" and `rank_arms`
    # keys as -inf: an arm whose attention blew up sorted LAST, indistinguishable
    # from attention_type: none. That inverts the tool against its own purpose.
    if any(not math.isfinite(v) for v in measured):
        return math.inf, math.inf
    worst = max(measured) if measured else None
    base = next((s.max_rho for s in per_scale if s.scale == 1.0 and not s.phase_perturbed), None)
    top = max((s.max_rho for s in per_scale if not s.phase_perturbed), default=None)
    growth = (top / base) if (base and base > 0 and top is not None) else None
    return worst, growth


def rank_arms(reports: list[ArmEnergyReport]) -> list[ArmEnergyReport]:
    """Sort most-unstable first: by ``worst_rho`` desc, tie-break ``rho_growth``.

    Arms that overflowed carry ``worst_rho = inf`` and therefore sort FIRST. Arms
    with no measured gain (``attention_type: none``) carry ``None`` and sort last.
    The two are distinct outcomes and must not collapse onto the same key.
    """

    def key(r: ArmEnergyReport) -> tuple[float, float]:
        w = r.worst_rho if r.worst_rho is not None else -math.inf
        g = r.rho_growth if r.rho_growth is not None else -math.inf
        return (w, g)

    return sorted(reports, key=key, reverse=True)


# ── per-arm orchestration ─────────────────────────────────────────────────────
def probe_arm(
    config: Any,
    device: str,
    *,
    scales: list[float],
    phase_perturb: bool,
    trajectory: bool,
    arm_name: str | None = None,
) -> ArmEnergyReport:
    """Build the arm, run Measurement A (+ B if requested), return the report."""
    model = build_probe_model(config, device)
    attn = str(
        dict(getattr(config.model, "model_kwargs", None) or {}).get("attention_type", "none")
    )
    x_base = build_probe_batch(model, config, device)
    per_scale = measure_forward_energy(
        model,
        x_base,
        scales=scales,
        phase_perturb=phase_perturb,
        timestep=_num_timesteps(config) // 2,
    )

    # Facade guard: a non-none arm that hooks zero blocks is misconfigured.
    hooked = sum(len(s.blocks) for s in per_scale[:1])
    if attn != "none" and hooked == 0:
        raise ValueError(
            f"energy_probe: arm declares attention_type={attn!r} but no attention "
            f"blocks were found in the module tree (facade / wrong backbone)."
        )

    traj: tuple[StepEnergy, ...] | None = None
    if trajectory:
        traj = measure_trajectory_energy(model, config, device)

    worst, growth = _worst_and_growth(per_scale)
    return ArmEnergyReport(
        arm_name=arm_name or getattr(getattr(config, "metadata", None), "name", "arm"),
        attention_type=attn,
        model_type=str(config.model.model_type),
        device=device,
        per_scale=per_scale,
        trajectory=traj,
        worst_rho=(None if attn == "none" else worst),
        rho_growth=(None if attn == "none" else growth),
    )
