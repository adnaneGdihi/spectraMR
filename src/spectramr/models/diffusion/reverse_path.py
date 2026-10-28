"""The one cold-diffusion reverse path: ``generator.sample()``, micro-batched (#2067).

Validation and predict both reconstruct through :func:`sample_reverse_path`, so a
validation metric describes the reconstruction predict returns for the same
input, seed and chunk size. Before #2067 predict ran its own strided loop, with a
different schedule, a hard-pinned DC and a final model call at t=0, while
validation called ``generator.sample()``.

The chunk size is part of the contract, not only a memory knob. ``sample()``
reseeds the C6 noise stream on every call, so with ``sampler_sigma > 0`` the
noise a row receives depends on which chunk it lands in. Both callers therefore
read the same ``validation.loader.chunk_size``. At ``sampler_sigma == 0`` the
reverse loop has no cross-batch coupling and any chunking is exact.
"""

from __future__ import annotations

from typing import Any

import torch


def _split(value: torch.Tensor | None, n: int, chunk: int, pieces: int) -> list[Any]:
    """Split a per-row tensor in step with the measurement; reuse anything else."""
    if value is not None and value.shape[0] == n:
        return list(value.split(chunk, dim=0))
    return [value] * pieces


def sample_reverse_path(
    gen: Any,
    measurement: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    steps: int | None,
    chunk_size: int,
    smaps: torch.Tensor | None = None,
    start_timestep: int | None = None,
    seed_offset: int = 0,
    contrast_idx: torch.Tensor | None = None,
    reference: torch.Tensor | None = None,
    measured_lines: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``gen.sample()`` over ``measurement`` in ``chunk_size`` micro-batches.

    The reverse sampler reconstructs each row independently, so splitting along
    dim 0 and concatenating changes nothing at ``sampler_sigma == 0`` and caps
    peak memory at one chunk's activations. ``smaps``, ``mask``,
    ``contrast_idx`` and ``reference`` are split with the measurement when they
    are per-row, so no row is conditioned on another subject's maps or contrast.
    There is no ``torch.cuda.empty_cache()`` between chunks: the allocator
    already reuses the freed blocks, so it would not lower peak allocated
    memory and would force a device sync per chunk (non-negotiable 9).

    Args:
        gen: The generator; its ``sample()`` is the reverse path.
        measurement: Measured k-space ``[B, C, H, W]`` in the model's domain.
        mask: Observed support, or ``None`` for ``sample()`` to infer it.
        steps: Reverse step count, or ``None`` for the generator's own.
        chunk_size: Rows per ``sample()`` call (``validation.loader.chunk_size``).
        smaps: Image-domain coil maps, forwarded so every step conditions on them.
        start_timestep: Head of the reverse trajectory, or ``None`` for ``T - 1``.
        seed_offset: Ensemble member index; forwarded only when non-zero, so a
            generator whose ``sample()`` predates ensembles keeps working.
        contrast_idx: Per-row contrast conditioning, forwarded only when present.
        reference: Per-row reference contrast, forwarded only when present.
        measured_lines: Per-row acquired lines ``[B, N]`` the forward process
            counted R over (#2160), forwarded only when present.

    Returns:
        The reconstruction, concatenated over chunks.

    Raises:
        ValueError: ``chunk_size`` is below 1.
    """
    if int(chunk_size) < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    chunk = int(chunk_size)
    member: dict[str, int] = {"seed_offset": int(seed_offset)} if seed_offset else {}
    per_row = {
        name: value
        for name, value in (
            ("contrast_idx", contrast_idx),
            ("reference", reference),
            ("measured_lines", measured_lines),
        )
        if value is not None
    }
    n = measurement.shape[0]
    if n <= chunk:
        return gen.sample(
            measurement=measurement,
            mask=mask,
            inference_timesteps=steps,
            smaps=smaps,
            start_timestep=start_timestep,
            **per_row,
            **member,
        )

    meas_chunks = measurement.split(chunk, dim=0)
    pieces = len(meas_chunks)
    mask_chunks = _split(mask, n, chunk, pieces)
    smap_chunks = _split(smaps, n, chunk, pieces)
    row_chunks: list[dict[str, torch.Tensor]] = [{} for _ in meas_chunks]
    for name, value in per_row.items():
        for kwargs_c, piece in zip(row_chunks, _split(value, n, chunk, pieces), strict=True):
            kwargs_c[name] = piece

    parts: list[torch.Tensor] = []
    # The schedule counters agree across chunks; the unfitted-envelope count is
    # per sample, so it is summed here or the record would hold the last chunk's.
    unfitted: int | None = None
    for meas_c, mask_c, smaps_c, row_c in zip(
        meas_chunks, mask_chunks, smap_chunks, row_chunks, strict=True
    ):
        parts.append(
            gen.sample(
                measurement=meas_c,
                mask=mask_c,
                inference_timesteps=steps,
                smaps=smaps_c,
                start_timestep=start_timestep,
                **row_c,
                **member,
            )
        )
        stats = getattr(gen, "last_reverse_stats", None) or {}
        if stats.get("decay_unfitted") is not None:
            unfitted = (unfitted or 0) + int(stats["decay_unfitted"])
    if unfitted is not None:
        gen.last_reverse_stats["decay_unfitted"] = unfitted
    return torch.cat(parts, dim=0)


def resolve_reverse_steps(
    validation_steps: int | None,
    diffusion_steps: int | None,
    num_timesteps: int | None,
) -> int | None:
    """The reverse step count both callers hand ``sample()``.

    ``validation.sampling.steps`` first, then ``training.diffusion.sampling_steps``;
    ``None`` leaves the generator's own resolved count. A count above the
    horizon is clamped to it, since a reverse pass cannot take more steps than
    the forward schedule has.
    """
    steps = validation_steps if validation_steps is not None else diffusion_steps
    if steps is not None and isinstance(num_timesteps, int) and num_timesteps > 0:
        steps = min(int(steps), num_timesteps)
    return steps


def takes_coil_maps(model_type: object) -> bool:
    """Whether a run estimates coil maps and hands them to the generator.

    Every ``kspace_cold_diffusion`` model gets them, at training, validation and
    predict alike. Whether the backbone also concatenates them is the
    generator's own decision (``model_expects_smaps_concat``); ``sample()``
    stashes them either way for the mechanisms that read them.
    """
    return "kspace_cold_diffusion" in str(model_type or "").lower()


__all__ = ["resolve_reverse_steps", "sample_reverse_path", "takes_coil_maps"]
