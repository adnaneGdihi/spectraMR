"""The rung a cold-diffusion predict input sits at, resolved the way validation resolves it.

Validation degrades a fully sampled input to the rung for ``R`` and samples from that
rung's timestep (:mod:`spectramr.models.diffusion.cold_rung`). Predict used to infer
the mask from the input's non-zeros and start the reverse loop at ``T - 1``, so a
predict output at ``R`` was not what validation scored at ``R`` (#2883). Per file:

* **fully sampled input, ``--acceleration R``** -- the input is masked to the rung for
  ``R`` over its acquired lines and sampled from that rung's timestep, as validation does;
* **input carrying its own ``mask`` dataset** -- already undersampled, so it is never
  masked again; the mask is passed to the sampler, and the timestep comes from
  ``--acceleration`` when given, else from the mask's acceleration over the acquired
  lines (which must then be a rung of the schedule);
* **neither** -- unchanged: the sampler infers the mask and starts at ``T - 1``.

Anything else raises. ``--acceleration`` on an arm without a forward process, or at
an ``R`` the schedule does not realise, is a :class:`ConfigurationError` (it recurs on
every file, so the run aborts); an input that is not fully sampled over its acquired
lines, or whose mask is not a rung, is a per-file ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from spectramr.domain.exceptions import ConfigurationError
from spectramr.models.diffusion.acquired_line_budget import acquired_lines_of
from spectramr.models.diffusion.cold_rung import (
    Rung,
    acceleration_over_acquired,
    rung_for_acceleration,
    rung_mask,
    schedule_accelerator,
)


@dataclass(frozen=True)
class PredictRung:
    """One file's rung: where the reverse loop starts and which mask it is given.

    Attributes:
        source: ``"acceleration"`` (fully sampled, masked here) or ``"mask"`` (the
            file's own mask).
        acceleration: The acceleration the timestep was resolved from.
        timestep: The head of the reverse trajectory.
        realized: What the schedule decodes ``timestep`` back to.
        process: The generator's forward process, which builds the rung mask.
        file_mask: ``[S or 1, 1, H, W]`` float plane mask when ``source == "mask"``.
    """

    source: str
    acceleration: float
    timestep: int
    realized: float
    process: Any
    file_mask: Tensor | None = None

    def batch(
        self, batch: Tensor, acquired: Tensor | None, start: int, end: int
    ) -> tuple[Tensor, dict[str, Any]]:
        """The batch the model receives and the ``mask`` / ``start_timestep`` kwargs for it."""
        if self.file_mask is None:
            steps = torch.full(
                (batch.shape[0],), self.timestep, dtype=torch.long, device=batch.device
            )
            if acquired is not None:
                acquired = acquired.to(batch.device)
            mask = rung_mask(self.process, steps, batch, acquired)
            batch = batch * mask
        else:
            plane = self.file_mask
            plane = plane[start:end] if plane.shape[0] > 1 else plane
            mask = plane.to(batch.device).expand(batch.shape[0], batch.shape[1], -1, -1)
        return batch, {"mask": mask, "start_timestep": self.timestep}

    def provenance(self, input_file: Path) -> dict[str, Any]:
        """What predict did to this file, for the result dict and the run summary."""
        return {
            "file": str(input_file),
            "source": self.source,
            "acceleration": self.acceleration,
            "start_timestep": self.timestep,
            "acceleration_realized": self.realized,
        }


def _usable(rung: Rung, how: str, error: type[Exception] = ConfigurationError) -> Rung:
    """``rung`` when it can be sampled from; otherwise ``error``.

    A requested ``--acceleration`` fails identically on every file, so it raises the
    run-invariant :class:`ConfigurationError`; a file's own mask fails per file.
    """
    if not rung.ok:
        detail = rung.error if rung.round_trip is None else rung.round_trip.reason
        raise error(
            f"{how} R={rung.requested:g} names no usable rung of this arm's schedule "
            f"({rung.status.value}: {detail}); validation would skip this level, and predict "
            "has nothing to sample from."
        )
    return rung


def _mask_plane(mask: Tensor, tensor: Tensor, line_dim: int | None) -> Tensor:
    """The file's mask as a ``[S or 1, 1, H, W]`` float plane over ``tensor``'s grid."""
    h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
    mask = mask.float()
    if mask.dim() == 1:
        if line_dim is None or mask.shape[0] != tensor.shape[line_dim]:
            raise ValueError(
                f"a 1-D mask of {mask.shape[0]} entries cannot be placed on a {h}x{w} plane "
                f"(line dim {line_dim})."
            )
        shape = (1, 1, -1, 1) if line_dim == -2 else (1, 1, 1, -1)
        return mask.reshape(shape).expand(1, 1, h, w)
    if mask.shape[-2:] != (h, w):
        raise ValueError(f"mask plane {tuple(mask.shape[-2:])} does not match the input's {h}x{w}.")
    plane = mask.reshape(-1, 1, h, w)
    if plane.shape[0] not in (1, tensor.shape[0]):
        raise ValueError(
            f"mask has {plane.shape[0]} planes; expected 1 or one per slice ({tensor.shape[0]})."
        )
    return plane


def _mask_acceleration(plane: Tensor, tensor: Tensor, acquired: Tensor | None, dim: int) -> float:
    rows = plane.expand(tensor.shape[0], -1, -1, -1)
    support = acquired if acquired is not None else torch.ones_like(acquired_lines_of(rows, dim))
    per_row = acceleration_over_acquired(acquired_lines_of(rows, dim), support.to(rows.device))
    if not torch.allclose(per_row, per_row[:1].expand_as(per_row), rtol=0.0, atol=1e-6):
        raise ValueError(
            "the file's mask keeps a different share of the acquired lines on different "
            "slices; one start timestep cannot describe it. Pass --acceleration R."
        )
    return float(per_row[0])


def resolve_predict_rung(
    strategy: Any,
    input_file: Path,
    tensor: Tensor,
    acquired: Tensor | None,
    acceleration: float | None,
) -> PredictRung | None:
    """The file's rung, or ``None`` when neither ``--acceleration`` nor a file mask asks for one.

    Args:
        strategy: The inference strategy; a cold-diffusion one exposes ``rung_process``.
        input_file: The input, read for an optional ``mask`` dataset.
        tensor: ``[S, C, H, W]`` preprocessed input.
        acquired: ``[S, N]`` acquired lines from the file's header, or ``None``.
        acceleration: ``--acceleration``, or ``None``.

    Raises:
        ValueError: see the module docstring.
    """
    process = getattr(strategy, "rung_process", None)
    if process is None:
        if acceleration is not None:
            raise ConfigurationError(
                f"--acceleration={acceleration:g} was given, but {type(strategy).__name__} "
                "has no cold-diffusion forward process to degrade the input with."
            )
        return None

    from spectramr.data.io_strategies import load_h5_dataset_if_present

    file_mask = load_h5_dataset_if_present(input_file, "mask")
    if file_mask is None and acceleration is None:
        return None
    accelerator = schedule_accelerator(process)
    horizon = int(strategy.num_timesteps)
    dim = process.line_dim

    if file_mask is None:
        if dim is not None and acquired is not None:
            carried = acquired_lines_of(tensor, dim)
            if not torch.equal(carried, acquired.to(carried.device).bool()):
                raise ValueError(
                    f"{input_file.name}: --acceleration degrades a fully sampled input, but this "
                    "one carries data on fewer lines than its header acquired. An undersampled "
                    "scan must supply its own 'mask' dataset."
                )
        rung = _usable(rung_for_acceleration(accelerator, acceleration, horizon), "--acceleration")
        return PredictRung("acceleration", rung.requested, rung.timestep, rung.realized, process)

    plane = _mask_plane(file_mask, tensor, dim)
    if acceleration is not None:
        rung = _usable(rung_for_acceleration(accelerator, acceleration, horizon), "--acceleration")
    else:
        if dim is None:
            raise ValueError(
                f"{input_file.name} carries a mask, but this arm has no line axis to count its "
                "acceleration over. Pass --acceleration R."
            )
        measured = _mask_acceleration(plane, tensor, acquired, dim)
        rung = _usable(
            rung_for_acceleration(accelerator, measured, horizon),
            f"{input_file.name}'s mask (pass --acceleration to declare its rung):",
            ValueError,
        )
    return PredictRung("mask", rung.requested, rung.timestep, rung.realized, process, plane)


__all__ = ["PredictRung", "resolve_predict_rung"]
