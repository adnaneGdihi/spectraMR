"""Radial amplitude transfer function of a k-space generator.

A reconstruction that carries a fraction of the target's energy in the outer
k-space annulus has either a low-passing **objective** or a low-passing
**architecture**, and the two want opposite fixes. This probe separates them by
measuring the architecture alone: white complex k-space in, per-radius amplitude
ratio out, on an untrained network.

The measurement is only meaningful because the backbone is linear at
initialisation -- ModReLU's bias starts at zero, so it passes its input through,
and ``ComplexRMSNorm.gain`` starts at one. Measured on ``ComplexUNet``, the
normalised curve is bit-identical across ``timesteps`` 0/14/28 and across a 10x
input amplitude, which is the check to re-run before trusting a reading from a
different backbone.

Under ``force_pure_kspace`` the feature maps **are** k-space, so the tensor's
own (H, W) axes are the frequency axes with DC at the ``fft2c``/``fftshift``
centre ``(H // 2, W // 2)``. No transform is applied here and none belongs here:
a ``fft2c`` of an already-k-space feature map would measure the wrong operator.

What the probe does **not** measure: the trained network, the loss, the reverse
sampler, or any data-consistency projection. It reports one operator's shape at
init.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from typing import Any

import torch

from spectramr.infrastructure.physics.radial_bands import radial_bins

__all__ = [
    "RadialTransfer",
    "measure_radial_transfer",
    "probe_arm",
    "radial_bins",
]

_EPS = 1e-12


@dataclass(frozen=True)
class RadialTransfer:
    """Per-radius amplitude transfer of one model, plus how it was measured.

    ``ratios`` is **unnormalised** ``mean|out| / mean|in|`` per bin, so a flat
    all-pass with gain ``g`` reports ``g`` in every bin rather than hiding it,
    while ``outer_band_retention`` divides the outer-annulus mean by ``dc_gain``
    and so scores that same gain 1.0. Reading the raw curve as if it were the
    normalised one is how a gain gets misdiagnosed as a filter.
    """

    bin_edges: tuple[float, ...]
    bin_centers: tuple[float, ...]
    counts: tuple[int, ...]
    ratios: tuple[float, ...]
    dc_gain: float
    outer_band_retention: float
    outer_band_floor: float
    shape: tuple[int, ...]
    repeats: int
    seed: int
    timestep: int
    device: str
    label: str = ""

    @property
    def normalized(self) -> tuple[float, ...]:
        """``ratios`` divided by ``dc_gain`` -- the curve the summary scalar reads."""
        return tuple(r / self.dc_gain for r in self.ratios)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["normalized"] = list(self.normalized)
        return d


def _as_complex(x: torch.Tensor) -> torch.Tensor:
    """Interleaved ``[R0, I0, R1, I1, ...]`` -> complex, or pass a complex tensor."""
    if x.is_complex():
        return x
    if x.dim() < 4:
        raise ValueError(
            f"spectral_transfer_probe: need [B, C, H, W(, ...)]; got {tuple(x.shape)}."
        )
    if x.shape[1] % 2 != 0:
        raise ValueError(
            f"spectral_transfer_probe: interleaved real/imag needs an even channel "
            f"count, got C={x.shape[1]} in {tuple(x.shape)}."
        )
    return torch.complex(x[:, 0::2].float(), x[:, 1::2].float())


def _plane_amplitude(x: torch.Tensor) -> torch.Tensor:
    """Mean ``|.|`` over every axis but the two frequency axes (2, 3) -> [H, W]."""
    c = _as_complex(x)
    reduce_dims = [d for d in range(c.dim()) if d not in (2, 3)]
    return c.abs().mean(dim=reduce_dims)


def _forward(model: torch.nn.Module, x: torch.Tensor, timestep: int) -> torch.Tensor:
    """Call the model, passing ``timesteps`` only when its forward accepts it.

    Dispatch is on the signature, not on a swallowed ``TypeError``: a probe that
    retried after an exception could not tell a model that ignores the timestep
    from one that rejected the batch for an unrelated reason.
    """
    params = inspect.signature(model.forward).parameters
    accepts = "timesteps" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts:
        t = torch.full((x.shape[0],), int(timestep), device=x.device, dtype=torch.long)
        out = model(x, timesteps=t)
    else:
        out = model(x)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            f"spectral_transfer_probe: model returned {type(out).__name__}; the radial "
            "transfer is defined on a single output tensor."
        )
    return out


def measure_radial_transfer(
    model: torch.nn.Module,
    *,
    size: int | tuple[int, int],
    channels: int,
    n_bins: int = 32,
    repeats: int = 4,
    seed: int = 0,
    device: str = "cpu",
    trailing: tuple[int, ...] = (),
    timestep: int = 0,
    label: str = "",
) -> RadialTransfer:
    """Amplitude transfer per radial k-space bin, averaged over white-noise draws.

    White noise excites every radius equally, which is the whole point: an
    in-distribution phantom's k-space is ~1/f, so its outer bins are near zero
    and the ratio there is a quotient of two numbers that are both noise.

    Draws accumulate as ``sum|out| / sum|in|`` per bin rather than as a mean of
    per-draw ratios, because only the first has a variance that falls with
    ``repeats``. A single 256x256 draw moves ``outer_band_retention`` by about
    +/-10 %, so ``repeats`` is not decoration.
    """
    h, w = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    if channels % 2 != 0:
        raise ValueError(
            f"spectral_transfer_probe: interleaved-complex input needs an even channel "
            f"width, got {channels}."
        )
    if repeats < 1:
        raise ValueError(f"spectral_transfer_probe: repeats must be >= 1, got {repeats}.")

    index, inside, edges = radial_bins(h, w, n_bins, device)
    flat_index = index[inside]
    counts = torch.zeros(n_bins, device=device).index_add_(
        0, flat_index, torch.ones_like(flat_index, dtype=torch.float32)
    )
    if int((counts == 0).sum()) > 0:
        empty = [i for i, c in enumerate(counts.tolist()) if c == 0]
        raise ValueError(
            f"spectral_transfer_probe: n_bins={n_bins} leaves bins {empty} empty on a "
            f"{h}x{w} grid; reduce n_bins or raise the grid size."
        )

    shape = (1, channels, h, w, *trailing)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    num = torch.zeros(n_bins, device=device)
    den = torch.zeros(n_bins, device=device)
    was_training = model.training
    model.eval()
    try:
        for _ in range(repeats):
            x = torch.empty(shape, device=device, dtype=torch.float32).normal_(generator=generator)
            with torch.no_grad():
                y = _forward(model, x, timestep)
            den.index_add_(0, flat_index, _plane_amplitude(x)[inside])
            num.index_add_(0, flat_index, _plane_amplitude(y)[inside])
    finally:
        model.train(was_training)

    ratios = (num / den.clamp_min(_EPS)).tolist()
    centers = [0.5 * (edges[i].item() + edges[i + 1].item()) for i in range(n_bins)]
    dc_gain = float(ratios[0])
    if dc_gain <= _EPS:
        raise ValueError(
            "spectral_transfer_probe: the DC bin has no gain, so the curve cannot be "
            "normalised; the model annihilates the k-space centre."
        )
    # n_bins >= 2 puts the last centre at (n - 0.5) / n >= 0.75, so the outer
    # band is never empty and needs no guard of its own.
    outer = [r for r, c in zip(ratios, centers, strict=True) if c > 0.5]
    return RadialTransfer(
        bin_edges=tuple(float(v) for v in edges.tolist()),
        bin_centers=tuple(float(v) for v in centers),
        counts=tuple(int(v) for v in counts.tolist()),
        ratios=tuple(float(v) for v in ratios),
        dc_gain=dc_gain,
        outer_band_retention=float(sum(outer) / len(outer) / dc_gain),
        outer_band_floor=float(min(outer) / dc_gain),
        shape=shape,
        repeats=int(repeats),
        seed=int(seed),
        timestep=int(timestep),
        device=str(device),
        label=label,
    )


def _probe_timestep(config: Any) -> int:
    """Mid-schedule timestep, matching ``energy_probe``'s operating point."""
    mk = dict(getattr(config.model, "model_kwargs", None) or {})
    t = mk.get("timesteps")
    if t is None:
        diffusion = getattr(getattr(config, "training", None), "diffusion", None)
        t = getattr(diffusion, "timesteps", None)
    return int(t) // 2 if t else 0


def probe_arm(
    config: Any,
    device: str,
    *,
    n_bins: int = 32,
    repeats: int = 4,
    seed: int = 0,
    arm_name: str | None = None,
) -> RadialTransfer:
    """Build the arm's real generator and measure its radial transfer.

    The model and the input **shape** both come from ``energy_probe`` -- one
    owner for "what does this arm's backbone consume", including the smaps-concat
    channel doubling that ``condition_with_smaps`` answers wrongly (#1326). Only
    the tensor's *contents* are replaced, because a phantom does not excite the
    outer bins.
    """
    from spectramr.infrastructure.validation.energy_probe import (
        build_probe_batch,
        build_probe_model,
    )

    model = build_probe_model(config, device)
    template = build_probe_batch(model, config, device)
    if template.dim() < 4:
        raise ValueError(
            f"spectral_transfer_probe: arm batch is {tuple(template.shape)}; need "
            "[B, C, H, W(, ...)]."
        )
    return measure_radial_transfer(
        model,
        size=(int(template.shape[2]), int(template.shape[3])),
        channels=int(template.shape[1]),
        n_bins=n_bins,
        repeats=repeats,
        seed=seed,
        device=device,
        trailing=tuple(int(v) for v in template.shape[4:]),
        timestep=_probe_timestep(config),
        label=arm_name or str(getattr(getattr(config, "metadata", None), "name", "arm")),
    )
