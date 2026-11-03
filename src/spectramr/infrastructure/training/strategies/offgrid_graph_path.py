"""Off-grid graph reconstruction: nodes from the acquired trajectory, image by adjoint NUFFT.

``NonCartesianSimulationTransform`` stores each slice's off-grid measurement
(``measured_kspace`` [.., 1, N], ``trajectory`` [2, N] in radians, ``dcf`` [N]). This path
turns those samples into graph nodes ``[Re, Im, kx/pi, ky/pi, dcf]`` without touching a
Cartesian grid, lets a ``node_source='trajectory'`` generator return one complex sample per
node, and maps the samples to an image with the density-weighted adjoint NUFFT. When
``physics.data_consistency`` is enabled, :class:`NonCartesianDataConsistency` then pulls the
image toward the measurement in the sample domain. When the strategy hands it a B0 map, the
measurement is re-simulated with off-resonance (time-segmented NUFFT over the spiral
readout times) before the nodes are built.

The output lies in the range of the adjoint over the acquired samples: the graph re-weights
and corrects the measured samples, it does not add samples the trajectory never visited.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.non_cartesian_dc import (
    NC_DC_MODES,
    NonCartesianDataConsistency,
)
from spectramr.infrastructure.physics.nufft.time_segmented_nufft import TimeSegmentedNUFFT
from spectramr.infrastructure.physics.nufft_ops import NUFFTForwardModel
from spectramr.infrastructure.physics.trajectories import spiral_readout_times

logger = logging.getLogger(__name__)

__all__ = ["OffGridGraphPath", "select_slice_samples"]


def select_slice_samples(
    measured: torch.Tensor, location: torch.Tensor | None, num_images: int
) -> torch.Tensor:
    """Return the measurement of each image in the batch as ``[num_images, 1, N]``.

    ``measured`` is ``[B, C, N]`` or ``[B, D, C, N]``: the transform simulates every depth
    slice of a volume, while a training patch keeps one slice, named by ``location[:, 2]``.
    """
    m = measured.unsqueeze(1) if measured.dim() == 3 else measured
    if m.dim() != 4:
        raise ValueError(
            f"measured_kspace must be [B, C, N] or [B, D, C, N]; got {tuple(measured.shape)}."
        )
    b, d, c, n = m.shape
    if c != 1:
        raise ValueError(
            f"The off-grid graph path takes one coil-combined measurement; got {c} coils. "
            "Combine coils before the simulation (data.coils.processing_mode: rss_image)."
        )
    if b * d == num_images:
        return m.reshape(b * d, 1, n)
    if b == num_images and location is not None:
        depth = location[:, 2].long().to(m.device)
        return m[torch.arange(b, device=m.device), depth]
    raise ValueError(
        f"measured_kspace holds {b}x{d} slices for {num_images} images and the batch carries "
        "no patch `location` to pick each image's slice."
    )


class OffGridGraphPath(nn.Module):
    """Run a ``node_source='trajectory'`` generator on the acquired off-grid samples."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        dc = config.physics.data_consistency
        self.dc_method: str | None = None
        self.dc_weight = float(dc.weight)
        if dc.enabled:
            if dc.method not in NC_DC_MODES:
                raise ValueError(
                    "The off-grid graph path enforces data consistency at the acquired samples, "
                    f"so physics.data_consistency.method must be one of {sorted(NC_DC_MODES)}; "
                    f"got {dc.method!r}."
                )
            if dc.method == "replace" and self.dc_weight >= 1.0:
                raise ValueError(
                    "physics.data_consistency.method 'replace' with weight >= 1.0 swaps every "
                    "predicted sample for the measurement (all trajectory samples are acquired), "
                    "so the generator never reaches the image. Use a weight in [0, 1) or 'gradient'."
                )
            self.dc_method = dc.method
        self._ops: dict[tuple[int, int], NUFFTForwardModel] = {}
        self._dc: dict[tuple[int, int], NonCartesianDataConsistency] = {}
        self._b0_ops: dict[tuple[int, int], TimeSegmentedNUFFT] = {}
        self._gain: dict[tuple, torch.Tensor] = {}

    def provenance(self) -> dict[str, Any]:
        """What the path applies, for the run record."""
        return {
            "node_source": "trajectory",
            "dc_method": self.dc_method,
            "dc_step": self.dc_weight if self.dc_method else None,
        }

    def _op(self, im_size: tuple[int, int], device: torch.device) -> NUFFTForwardModel:
        if im_size not in self._ops:
            self._ops[im_size] = NUFFTForwardModel(im_size=im_size).to(device)
        return self._ops[im_size]

    def _adjoint_gain(self, op, traj, dcf, im_size, device) -> torch.Tensor:
        """DC gain of ``A^H W A`` (its response to a constant image), cached per trajectory.

        torchkbnufft's pair is not unit-gain (about 3e4 on a 256x256 spiral), so dividing it
        out puts the adjoint image on the scale of the object that produced the samples.
        """
        key = (traj.shape[-1], im_size, str(device))
        if key not in self._gain:
            with torch.no_grad():
                one = torch.ones(1, 1, *im_size, dtype=torch.complex64, device=device)
                psf = op.adjoint_project(op.forward_project(one, traj) * dcf, traj).abs()
                h, w = im_size
                centre = psf[..., h // 4 : h - h // 4, w // 4 : w - w // 4]
                self._gain[key] = centre.mean().clamp_min(1e-12)
        return self._gain[key]

    def _with_off_resonance(self, y, hr, b0_hz, traj, im_size) -> torch.Tensor:
        """Re-simulate the measurement with the B0 map, keeping the stored measurement's scale."""
        device = y.device
        if im_size not in self._b0_ops:
            self._b0_ops[im_size] = TimeSegmentedNUFFT(image_shape=im_size).to(device)
        readout = spiral_readout_times(im_size, traj.shape[-1], device=device)
        batch_traj = traj.unsqueeze(0).expand(hr.shape[0], -1, -1)
        image = hr.to(torch.complex64)
        clean = self._op(im_size, device).forward_project(image, batch_traj)
        omega = 2.0 * math.pi * b0_hz.float()  # Hz -> rad/s, the operator's unit
        corrupted = self._b0_ops[im_size](image, omega, batch_traj, readout_times=readout)
        # alpha is 1 when hr is the image the loader simulated from; otherwise it rescales.
        dims = (1, 2)
        alpha = y.abs().pow(2).sum(dims, keepdim=True).sqrt() / clean.abs().pow(2).sum(
            dims, keepdim=True
        ).sqrt().clamp_min(1e-12)
        return y + alpha * (corrupted - clean)

    def forward(
        self,
        generator: nn.Module,
        batch_context: dict[str, Any],
        b0_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct ``[B, 1, H, W]`` magnitude images from the batch's off-grid samples."""
        hr = batch_context.get("hr")
        measured = batch_context.get("measured_kspace")
        trajectory = batch_context.get("trajectory")
        dcf = batch_context.get("dcf")
        if hr is None or measured is None or trajectory is None or dcf is None:
            missing = [
                k
                for k, v in (
                    ("hr", hr),
                    ("measured_kspace", measured),
                    ("trajectory", trajectory),
                    ("dcf", dcf),
                )
                if v is None
            ]
            raise ValueError(
                f"The off-grid graph path needs {missing} in the batch; the loader's "
                "NonCartesianSimulationTransform (data.trajectory) provides them."
            )
        im_size = (int(hr.shape[-2]), int(hr.shape[-1]))
        device = hr.device
        y = select_slice_samples(measured, batch_context.get("location"), hr.shape[0])
        y = y.to(device=device, dtype=torch.complex64)
        # One trajectory per (pattern, im_size) in the loader, so every sample shares it.
        traj = (trajectory[0] if trajectory.dim() == 3 else trajectory).to(device).float()
        weights = (dcf[0] if dcf.dim() == 2 else dcf).to(device).float().view(1, 1, -1)
        if b0_map is not None:
            y = self._with_off_resonance(y, hr, b0_map.to(device), traj, im_size)

        scale = y.abs().pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
        y_n = y / scale
        batch, _, n = y.shape
        nodes = torch.cat(
            [
                y_n.real.transpose(1, 2),
                y_n.imag.transpose(1, 2),
                (traj / math.pi).t().unsqueeze(0).expand(batch, n, 2),
                weights.view(1, n, 1).expand(batch, n, 1),
            ],
            dim=-1,
        )
        out = generator(nodes)
        if out.dim() != 3 or out.shape[-1] != 2:
            raise ValueError(
                f"A trajectory-node generator returns [B, N, 2]; got {tuple(out.shape)}."
            )
        samples = torch.complex(out[..., 0], out[..., 1]).unsqueeze(1) * scale
        dc = self._dc_layer(im_size, device) if self.dc_method is not None else None
        if self.dc_method == "replace":
            # Blend at the samples, then one gain-normalised adjoint keeps the object scale.
            samples = dc(samples, y, traj, dcf=weights, is_sample_domain=True)

        op = self._op(im_size, device)
        image = op.adjoint_project(samples * weights, traj) / self._adjoint_gain(
            op, traj, weights, im_size, device
        )
        if self.dc_method == "gradient":
            image = dc(image, y, traj, dcf=weights)
        return image.abs()

    def _dc_layer(self, im_size: tuple[int, int], device: torch.device):
        if im_size not in self._dc:
            self._dc[im_size] = NonCartesianDataConsistency(
                im_size=im_size,
                mode=self.dc_method,
                step_size=self.dc_weight,
                lambda_dc=self.dc_weight if self.dc_method == "replace" else 1.0,
                learn_step=False,
            ).to(device)
        return self._dc[im_size]
