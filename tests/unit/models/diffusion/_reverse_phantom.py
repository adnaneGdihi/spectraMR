"""Multi-coil k-space phantom and toy models shared by the reverse-loop tests.

The phantom is what the physics options assume and nothing more: a
piecewise-smooth object seen through smooth complex coil sensitivities, so its
k-space decays as a power of ``|ky|`` (Algorithm F), is a short convolution
across coils (E), and has sparse y-edges shared by the coils (H).
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.data_consistency import NoiseAdaptiveDataConsistency
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.diffusion.kspace_process import (
    KSpaceUndersamplingProcess,
    PhysicsInformedColdDiffusion,
)
from spectramr.models.diffusion.reverse_domain import stacked_kspace

H = W = 64
COILS = 4
T = 8
HEAD = T - 1
NOISE = 1e-3


def process(max_acceleration: float = 8.0, dynamic: bool = False) -> KSpaceUndersamplingProcess:
    """The ladder the ``kspace_filling`` arms declare: nested phase-encode lines.

    ``dynamic`` sets ``enable_dynamic_mask``, which every ``kspace_filling`` arm declares.
    """
    return KSpaceUndersamplingProcess(
        num_timesteps=T,
        base_acceleration=1.0,
        max_acceleration=max_acceleration,
        center_fraction=0.08,
        mask_type="density_nested",
        mask_direction="phase",
        # Rows, as before #2160: the phantom fixtures and their pinned numbers
        # are row-mask geometry, and the axis is incidental to what they test.
        phase_encode_axis=-2,
        enforce_nested=True,
        train_identity_rung=True,
        seed=42,
        enable_dynamic_mask=dynamic,
    )


def coil_images(batch: int = 2, seed: int = 0) -> torch.Tensor:
    """Complex ``[B, COILS, H, W]``: ellipses times Gaussian sensitivities with linear phase."""
    gen = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    images = []
    for _ in range(batch):
        obj = torch.zeros(H, W)
        for _ in range(4):
            cy, cx = (torch.rand(2, generator=gen) - 0.5) * 0.6
            ry, rx = 0.2 + 0.4 * torch.rand(2, generator=gen)
            obj = (
                obj
                + (0.5 + torch.rand(1, generator=gen))
                * (((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 < 1).float()
            )
        coils = []
        for c in range(COILS):
            angle = 2 * torch.pi * c / COILS
            sy, sx = 0.8 * torch.sin(torch.tensor(angle)), 0.8 * torch.cos(torch.tensor(angle))
            mag = torch.exp(-((y - sy) ** 2 + (x - sx) ** 2) / 1.5)
            phase = torch.exp(1j * (0.5 * c + 0.3 * y + 0.2 * x))
            coils.append(obj * mag * phase)
        images.append(torch.stack(coils))
    return torch.stack(images).to(torch.complex64)


def kspace(batch: int = 2, seed: int = 0, noise: float = NOISE) -> torch.Tensor:
    """Stacked real ``[B, 2*COILS, H, W]`` k-space of :func:`coil_images` plus complex noise."""
    k = fft2c(coil_images(batch, seed))
    gen = torch.Generator().manual_seed(seed + 1)
    scale = noise * k.abs().amax()
    k = k + scale * torch.complex(
        torch.randn(k.shape, generator=gen), torch.randn(k.shape, generator=gen)
    )
    return stacked_kspace(k, torch.zeros(1), log_scaled=False)


def mask(proc: KSpaceUndersamplingProcess, batch: int = 2, t: int = HEAD) -> torch.Tensor:
    return proc.mask_at(t, (H, W)).reshape(1, 1, H, W).float().expand(batch, 1, H, W)


class Oracle(torch.nn.Module):
    """Predicts the target whatever it is given."""

    def __init__(self, target: torch.Tensor, proc: KSpaceUndersamplingProcess) -> None:
        super().__init__()
        self.target = target
        self.kspace_process = proc
        self.dc_layer = NoiseAdaptiveDataConsistency()

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_kw: object) -> torch.Tensor:
        return self.target.clone()


class Feedback(torch.nn.Module):
    """``gain * input + bias``: a model that amplifies what it re-reads, the additive failure."""

    def __init__(self, proc: KSpaceUndersamplingProcess, gain: float, bias: float) -> None:
        super().__init__()
        self.gain, self.bias = gain, bias
        self.kspace_process = proc
        self.dc_layer = NoiseAdaptiveDataConsistency()

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_kw: object) -> torch.Tensor:
        return self.gain * x + self.bias


class StepSensitive(torch.nn.Module):
    """``target * (1 + t/10) + x/2``: what a coefficient receives depends on the step that writes it
    and on what earlier steps wrote, so any change in the reveal order changes the output."""

    def __init__(self, target: torch.Tensor, proc: KSpaceUndersamplingProcess) -> None:
        super().__init__()
        self.target = target
        self.kspace_process = proc
        self.inputs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_kw: object) -> torch.Tensor:
        self.inputs.append(x.clone())
        return self.target * (1 + 0.1 * t.reshape(-1, 1, 1, 1)) + 0.5 * x


def sampler(model: torch.nn.Module, mode: str, **kwargs) -> PhysicsInformedColdDiffusion:
    defaults = {
        "num_timesteps": T,
        "max_acceleration": model.kspace_process.max_accel,
        "center_fraction": 0.08,
        "dc_method": "hard",
        "reverse_mode": mode,
        "reverse_clip_ratio": 1.5,
        "kspace_log_scaled": False,
        "refine_alpha": 0.5 if mode == "reveal_refine_dc" else None,
        "support_radius": 3 if mode == "support_ordered_dc" else None,
    }
    return PhysicsInformedColdDiffusion(model=model, **{**defaults, **kwargs})


def band_local_outlier_run(mode: str, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(out, measurement, mask)`` from a hard-DC ``band_local`` loop whose outermost measured
    bin sits at 0.9x the peak: above the running-minimum envelope, so a whole-plane clamp
    after data consistency rewrites an acquired coefficient (#2310)."""
    proc = process(8.0)
    target = kspace()
    acquired = mask(proc)
    meas = (target * acquired).clone()
    support = acquired[0, 0] > 0
    ky, kx = torch.meshgrid(
        torch.arange(H) - H // 2, torch.arange(W) - W // 2, indexing="ij"
    )
    row, col = divmod(int(((ky**2 + kx**2) * support).argmax()), W)
    peak = float(meas.abs().max())
    meas[:, 0::2, row, col] = 0.9 * peak
    meas[:, 1::2, row, col] = 0.0
    run = sampler(Oracle(target, proc), mode, clip_reference="band_local", **kwargs)
    return run.sample(meas, acquired, start_timestep=HEAD), meas, acquired


def module_mode_runs(mode: str) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
    """Outputs and per-call model inputs of one eval-mode then two train-mode ``sample()`` calls
    on a dynamic-mask process: the reverse loop must read the fixed cascade in both modes (#2471)."""
    proc = process(dynamic=True)
    target = kspace()
    acquired = mask(proc)
    measurement = target * acquired
    outs, inputs = [], []
    for training in (False, True, True):
        proc.train(training)
        model = StepSensitive(target, proc)
        outs.append(sampler(model, mode).sample(measurement.clone(), acquired.clone()))
        inputs.append(model.inputs)
    return outs, inputs
