"""Clean-reference dataset for meta-evaluation (``.pt`` slices).

Phase 4 of ``TODO/backlog_ssot_and_layering_cleanup.md``: the CLI's
meta-evaluation path was calling ``torch.load(fp, ...)`` directly,
violating the data-SSOT rule (CLAUDE.md pitfall #11 — file → Dataset
transitions live in :mod:`spectramr.data.datasets`).

This dataset wraps a directory of ``.pt`` tensors / dicts and surfaces
them as ``(id, tensor)`` pairs. It is a non-TorchIO ``map-style`` dataset
because the meta-evaluation harness wants simple tensor access, not the
full TorchIO subject pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import torch.utils.data as _torch_data

from spectramr.shared.utils.safe_io import safe_torch_load


class MetaEvalCleanRefsDataset(_torch_data.Dataset):
    """Map-style dataset over ``.pt`` clean-reference volumes.

    Accepts raw tensors or dicts with one of the common keys (``image``,
    ``clean``, or the first value). Each item must be one slice, ``[H, W]`` or
    ``[1, H, W]``, or carry an explicit ``coil_axis``; anything else raises (see
    :func:`_as_channel_slice`). Tensors are coerced to ``float32`` and optionally
    moved to the caller's device.

    Args:
        input_dir: Directory containing the ``.pt`` files.
        max_subjects: Hard cap on how many files to enumerate (sorted by
            filename, so deterministic).
        device: Optional target device for the returned tensors.
        loader: Underlying loader function. Defaults to
            :func:`spectramr.shared.utils.safe_io.safe_torch_load` so checkpoint
            files with ``weights_only=False`` go through the same safety
            wrapper the rest of the repo uses.
    """

    def __init__(
        self,
        input_dir: str | Path,
        *,
        max_subjects: int = 50,
        device: torch.device | None = None,
        loader: Callable[[Path], object] | None = None,
    ) -> None:
        super().__init__()
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Clean-refs directory not found: {self.input_dir}")
        self.device = device
        self._loader = loader or self._default_load
        self._files: list[Path] = sorted(self.input_dir.glob("*.pt"))[:max_subjects]

    @staticmethod
    def _default_load(path: Path) -> object:
        return safe_torch_load(path, map_location="cpu", weights_only=False)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor] | None:
        """Return ``(stem, tensor)`` for ``idx`` — ``None`` if the file is malformed."""
        fp = self._files[idx]
        raw = self._loader(fp)
        coil_axis: object = None
        if isinstance(raw, dict):
            coil_axis = raw.get("coil_axis")
            if "image" in raw:
                raw = raw["image"]
            elif "clean" in raw:
                raw = raw["clean"]
            else:
                raw = next(v for k, v in raw.items() if k != "coil_axis")
        if not isinstance(raw, torch.Tensor):
            return None
        t = _as_channel_slice(raw, coil_axis, fp).float()
        if self.device is not None:
            t = t.to(self.device)
        return fp.stem, t


def _as_channel_slice(t: torch.Tensor, coil_axis: object, fp: Path) -> torch.Tensor:
    """Coerce to ``[C, H, W]``: ``[H, W]``, ``[1, H, W]``, or coils named by ``coil_axis``.

    A bare ``[S, H, W]`` is ambiguous between slices and coils; reading it as coils
    would mix noise and gains across slices, so it raises instead.
    """
    if t.is_complex():
        raise ValueError(
            f"{fp}: complex reference {tuple(t.shape)}; casting to float would drop the "
            "imaginary part. Save the magnitude (or the real image) instead."
        )
    if coil_axis is not None:
        if (
            not isinstance(coil_axis, int)
            or isinstance(coil_axis, bool)
            or t.ndim != 3
            or not -3 <= coil_axis < 3
        ):
            raise ValueError(
                f"{fp}: coil_axis={coil_axis!r} must be an int axis of a 3-D [C, H, W] "
                f"tensor; got shape {tuple(t.shape)}."
            )
        return t.movedim(coil_axis, 0)
    if t.ndim == 2:
        return t.unsqueeze(0)
    if t.ndim == 3 and t.shape[0] == 1:
        return t
    raise ValueError(
        f"{fp}: shape {tuple(t.shape)} is not a slice. meta-evaluate is slice-wise: save "
        "[H, W] or [1, H, W], or a dict {'image': t, 'coil_axis': k} for coils. A stack "
        "of slices or a volume belongs in sim2rank --mode 3d."
    )


def load_clean_volumes(
    input_dir: str | Path,
    *,
    max_subjects: int,
    device: torch.device | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Convenience factory: build the dataset and materialise its items.

    Matches the old ``_load_clean_volumes_from_pt`` contract that the
    meta-evaluation CLI consumes (a flat list of ``(id, tensor)``).

    Args:
        input_dir: Directory of ``.pt`` clean-reference files.
        max_subjects: Maximum number of files to load.
        device: Optional target device for the tensors.

    Returns:
        ``[(stem, tensor), ...]``, skipping any file that didn't decode
        to a Tensor.
    """
    ds = MetaEvalCleanRefsDataset(
        input_dir,
        max_subjects=max_subjects,
        device=device,
    )
    out: list[tuple[str, torch.Tensor]] = []
    for i in range(len(ds)):
        item = ds[i]
        if item is not None:
            out.append(item)
    return out


__all__ = ["MetaEvalCleanRefsDataset", "load_clean_volumes"]
