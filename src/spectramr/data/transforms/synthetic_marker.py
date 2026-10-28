"""Synthetic privileged-marker producer for ``PrivilegedLearningStrategy``.

Emits a ``marker`` image: ``num_markers`` Gaussian dots at random in-plane positions,
scaled to ``amplitude`` times the input's peak magnitude. The strategy adds it to the
teacher's input only, so the dots are the privileged channel the student never sees.
"""

from __future__ import annotations

import copy

import torch
import torchio as tio

from spectramr.data.transforms.registry import register_transform

__all__ = ["SyntheticMarkerTransform"]


@register_transform("synthetic_marker", produces=("marker",))
class SyntheticMarkerTransform(tio.transforms.Transform):
    """Emit a sparse dot ``marker`` image shaped like ``input``.

    Args:
        num_markers: dots per sample (>= 1).
        sigma_px: Gaussian dot width in pixels (> 0).
        amplitude: dot peak as a fraction of the input's peak magnitude (> 0).
        seed: fixed seed for a reproducible dot layout; ``None`` draws from the
            worker-seeded global RNG, so every call places new dots.
    """

    def __init__(
        self,
        num_markers: int = 8,
        sigma_px: float = 2.0,
        amplitude: float = 0.5,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if num_markers < 1:
            raise ValueError(f"num_markers must be >= 1; got {num_markers}")
        if sigma_px <= 0 or amplitude <= 0:
            raise ValueError(f"sigma_px and amplitude must be > 0; got {sigma_px}, {amplitude}")
        self.num_markers = int(num_markers)
        self.sigma_px = float(sigma_px)
        self.amplitude = float(amplitude)
        self.seed = seed

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if "input" not in subject:
            raise KeyError("synthetic_marker needs subject['input'] to size the marker")
        image = subject["input"]
        out = copy.deepcopy(image)
        out.set_data(self._marker(image.data))
        subject.add_image(out, "marker")
        return subject

    def _marker(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() != 4:
            raise ValueError(f"synthetic_marker expects [C,H,W,D] data; got {tuple(data.shape)}")
        _, h, w, _ = data.shape
        gen = None if self.seed is None else torch.Generator().manual_seed(self.seed)
        centres = torch.rand(self.num_markers, 2, generator=gen) * torch.tensor([h, w])
        ys, xs = torch.arange(h)[:, None], torch.arange(w)[None, :]
        d2 = (ys - centres[:, 0, None, None]) ** 2 + (xs - centres[:, 1, None, None]) ** 2
        plane = torch.exp(-d2 / (2 * self.sigma_px**2)).amax(dim=0)
        peak = data.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
        real = data.real if torch.is_complex(data) else data
        return (self.amplitude * peak * plane[None, :, :, None]).expand_as(real).to(real.dtype)
