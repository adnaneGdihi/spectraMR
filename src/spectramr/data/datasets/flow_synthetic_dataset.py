"""``dataset_type: flow_synthetic``: readout slices of the analytic 4D-flow phantom.

Each item is one :class:`~spectramr.data.datasets.flow_sample.FlowSample`. A
subject's phantom and its k-t mask depend only on ``(seed, split, subject)``, so
every slice of a subject sees the same mask (a ``(ky, kz)`` line is acquired
along the whole readout) and the same physics, in any worker and any order; the
noise additionally depends on the slice. Train and validation draw from disjoint
seed streams. Items are subject-major, so an unshuffled loader delivers each
subject's slices consecutively.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from spectramr.infrastructure.physics.fft_ops import fftnc
from spectramr.infrastructure.physics.kt_sampling import kt_sampling_pattern
from spectramr.infrastructure.physics.signal_models.flow_encoding import (
    encoding_directions,
    referenced_encode,
)
from spectramr.infrastructure.physics.signal_models.flow_phantom import (
    FlowPhantomSpec,
    draw_flow_phantom,
    grid_basis_for,
    phantom_slice,
)

_SPLITS = {"train": 0, "val": 1}


class FlowSyntheticDataset(Dataset):
    """Readout slices of ``n_subjects`` phantoms, subject-major."""

    def __init__(self, block: Any, venc: float, encoding_scheme: str, split: str) -> None:
        if encoding_directions(encoding_scheme) != 3:
            raise ValueError(
                f"the 4D-flow phantom encodes three directions; {encoding_scheme!r} "
                f"encodes {encoding_directions(encoding_scheme)}"
            )
        self.block = block
        self.venc = float(venc)
        self.scheme = encoding_scheme
        self.split = split
        self.n_subjects = block.train_subjects if split == "train" else block.val_subjects
        self.spec = FlowPhantomSpec(
            grid=tuple(block.grid),
            voxel_size_mm=tuple(block.voxel_size_mm),
            frames=block.frames,
            coils=block.coils,
            vessels=block.vessels,
            peak_velocity=block.peak_velocity,
            background_order=block.background_order,
            background_amplitude=block.background_amplitude,
        )
        self._sampler = kt_sampling_pattern(block.sampling_pattern)
        self._grid_basis = grid_basis_for(self.spec)
        grid = self.spec.grid
        self._plane_slices = (grid[0] // 4, (3 * grid[0]) // 4)

    def __len__(self) -> int:
        return self.n_subjects * self.spec.grid[0]

    def _subject_seed(self, subject: int) -> int:
        return (self.block.seed * 1_000_003 + _SPLITS[self.split]) * 100_003 + subject

    def __getitem__(self, index: int) -> dict[str, Any]:
        n_slices = self.spec.grid[0]
        subject, x = divmod(index, n_slices)
        seed = self._subject_seed(subject)
        # One stream per subject: the phantom, then its mask, so no subject's mask
        # shares a stream with another subject's geometry.
        subject_gen = torch.Generator().manual_seed(2 * seed)
        phantom = draw_flow_phantom(self.spec, subject_gen)
        _, h, w = self.spec.grid
        mask = self._sampler(
            (4, self.spec.frames, h, w),
            self.block.acceleration,
            center_fraction=self.block.center_fraction,
            sigma=self.block.sampling_sigma,
            generator=subject_gen,
        ).float()
        fields = phantom_slice(self.spec, phantom, x, self._grid_basis)

        velocity = fields["velocity"]  # [3, T, H, W]
        encoded = referenced_encode(velocity, self.venc, self.scheme, dim=0)  # [E, T, H, W]
        background = torch.cat([torch.zeros(1, h, w), fields["background"]])[:, None]
        phase = fields["common_phase"] + background + encoded
        image_ref = (fields["magnitude"] * torch.exp(1j * phase)).to(torch.complex64)
        coil_images = fields["coil_maps"][:, None, None] * image_ref[None]  # [C, E, T, H, W]
        full = fftnc(coil_images, dims=(-2, -1))
        # Odd seeds for noise, even for subjects: the two families never share a stream.
        noise_gen = torch.Generator().manual_seed(2 * (seed * n_slices + x) + 1)
        noise = torch.complex(
            torch.randn(full.shape, generator=noise_gen),
            torch.randn(full.shape, generator=noise_gen),
        )
        target = full + (self.block.noise_std / 2**0.5) * noise

        area = self.spec.voxel_size_mm[1] * self.spec.voxel_size_mm[2]
        lumen = fields["lumen"].float()
        planes = [torch.zeros(3, h, w), torch.zeros(3, h, w)]
        for plane, x_plane in zip(planes, self._plane_slices, strict=True):
            if x == x_plane:
                plane[0] = lumen * area
        return {
            "input": target * mask,
            "target": target,
            "mask": mask,
            "coil_maps": fields["coil_maps"],
            "velocity": velocity,
            "image_ref": image_ref,
            "vessel_mask": lumen,
            "static_mask": fields["static"].float(),
            "inlet_mask": planes[0],
            "outlet_mask": planes[1],
            "venc": torch.full((3,), self.venc),
            "voxel_size_mm": torch.tensor(self.spec.voxel_size_mm),
            "encoding_scheme": self.scheme,
            "subject_id": f"{self.split}-{subject:03d}",
            "slice_index": x,
            "n_slices": n_slices,
        }


def create_flow_synthetic_datasets(config: Any, train_tfm: Any, val_tfm: Any):
    """``(train, val)`` phantom datasets. TorchIO transforms do not apply to k-space dicts."""
    del train_tfm, val_tfm
    block = config.flow_synthetic
    pc = config.phase_contrast
    if not block.enabled:
        raise ValueError("dataset_type 'flow_synthetic' requires data.flow_synthetic.enabled: true")
    if not pc.enabled:
        raise ValueError(
            "dataset_type 'flow_synthetic' reads venc and the encoding scheme from "
            "data.phase_contrast; set data.phase_contrast.enabled: true"
        )
    declared = [
        f"data.source.{name}"
        for name in ("index_path", "validation_index_path", "test_index_path")
        if getattr(config.source, name, None)
    ]
    if declared:
        raise ValueError(
            f"flow_synthetic draws its subjects from data.flow_synthetic.seed and reads no "
            f"index; {declared} would be ignored, and a held-out test pass would score the "
            "validation phantoms. Remove them."
        )
    if config.augmentation.enabled:
        raise ValueError(
            "data.augmentation is enabled, but flow_synthetic emits k-space dicts that no "
            "image-domain augmentation reads; disable it rather than leave it inert"
        )
    return (
        FlowSyntheticDataset(block, pc.venc, pc.encoding_scheme, "train"),
        FlowSyntheticDataset(block, pc.venc, pc.encoding_scheme, "val"),
    )


__all__ = ["FlowSyntheticDataset", "create_flow_synthetic_datasets"]
