"""Offline 64 mT synthetic k-space corpus built from high-field fastMRI-layout volumes.

Each source volume (``kspace`` of shape ``(S, C, H, W)``) is coil-combined by RSS,
degraded by :class:`~spectramr.data.synthesis.vlf_simulator.VLFSimulator` (the one
owner of the 3 T to 64 mT degradation model), and written back as single-coil
k-space in the fastMRI h5 layout the ``dataset_type: kspace`` route reads
(``kspace`` complex64 ``(S, 1, H, W)``, ``reconstruction_rss`` float32 ``(S, H, W)``,
``ismrmrd_header`` with the field strength and coil count rewritten). Every knob
used is stamped into the file attributes, and the noise/bias draws are seeded
from the ``file_id`` so a rebuild on the cluster is byte-stable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from spectramr.data.datasets.kspace_volume_stack import read_acquisition_params
from spectramr.data.metadata.path_resolver import PathResolver
from spectramr.data.synthesis.vlf_simulator import VLFSimulator
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

CORPUS_VERSION = "1"
# fastMRI headers report nominal fields as e.g. 2.89 or 1.494 T.
FIELD_TOLERANCE_T = 0.25
_FIELD_TAG = re.compile(r"(<systemFieldStrength_T>)[^<]*(</systemFieldStrength_T>)")
_COIL_TAG = re.compile(r"(<receiverChannels>)[^<]*(</receiverChannels>)")


@dataclass(frozen=True)
class ULFCorpusParams:
    """Degradation knobs; every field is validated and stamped into each output file."""

    target_field_strength_t: float = 0.064
    source_field_strength_t: float = 3.0
    target_snr_db: float = 10.0
    resolution_scale: float = 0.6
    b0_warp_strength: float = 0.08
    bias_field_strength: float = 0.4
    # The simulator's tissue segmentation is a T1w intensity heuristic, so it is
    # wrong on the mixed T2/FLAIR/T1POST fastMRI brain corpus.
    contrast_remapping: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.target_field_strength_t < self.source_field_strength_t:
            raise ValueError(
                "target_field_strength_t must be positive and below source_field_strength_t, "
                f"got {self.target_field_strength_t} vs {self.source_field_strength_t}"
            )
        if not 0.0 < self.resolution_scale <= 1.0:
            raise ValueError(f"resolution_scale must be in (0, 1], got {self.resolution_scale}")
        if self.b0_warp_strength < 0.0 or self.bias_field_strength < 0.0:
            raise ValueError("b0_warp_strength and bias_field_strength must be >= 0")
        if self.contrast_remapping:
            raise ValueError(
                "contrast_remapping=True is not supported: VLFSimulator segments tissue with a "
                "T1w-only intensity heuristic, which mislabels T2/FLAIR source volumes"
            )


def header_field_strength(path: Path) -> float:
    """``systemFieldStrength_T`` from the file's ISMRMRD header; raise when it is absent."""
    value = read_acquisition_params(Path(PathResolver.resolve(str(path)))).get("field_strength_T")
    if not isinstance(value, float):
        raise ValueError(f"{path}: ismrmrd_header carries no systemFieldStrength_T")
    return value


def is_on_field(path: Path, field_t: float) -> bool:
    """Whether the header field strength is within :data:`FIELD_TOLERANCE_T` of ``field_t``."""
    return abs(header_field_strength(path) - field_t) <= FIELD_TOLERANCE_T


def volume_seed(file_id: str) -> int:
    """Deterministic 63-bit seed derived from the source ``file_id``."""
    return int.from_bytes(hashlib.sha256(file_id.encode()).digest()[:8], "big") >> 1


def synthesize_ulf_kspace(
    kspace: torch.Tensor, params: ULFCorpusParams, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Degrade one HF volume to synthetic 64 mT.

    Args:
        kspace: Complex ``(S, C, H, W)`` multi-coil (or ``C == 1``) k-space.
        params: Validated degradation knobs.
        seed: Seed for the simulator's noise and bias-field draws.

    Returns:
        ``(kspace_ulf, rss_ulf)``: complex64 ``(S, 1, H, W)`` and float32 ``(S, H, W)``.
    """
    if kspace.ndim != 4 or not torch.is_complex(kspace):
        raise ValueError(
            f"expected complex (S, C, H, W) k-space, got {tuple(kspace.shape)} {kspace.dtype}"
        )
    image = ifft2c(kspace.to(torch.complex64))
    rss = torch.sqrt((image.abs() ** 2).sum(dim=1, keepdim=True))
    simulator = VLFSimulator(
        target_snr=params.target_snr_db,
        resolution_scale=params.resolution_scale,
        b0_warp_strength=params.b0_warp_strength,
        bias_field_strength=params.bias_field_strength,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        ulf = simulator(rss, enable_contrast_remapping=params.contrast_remapping)
    kspace_ulf = fft2c(ulf.to(torch.complex64))
    return kspace_ulf, ulf[:, 0].to(torch.float32)


def _center_crop(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    h, w = hw
    if h > x.shape[-2] or w > x.shape[-1]:
        raise ValueError(f"crop {hw} exceeds image {tuple(x.shape[-2:])}")
    top, left = (x.shape[-2] - h) // 2, (x.shape[-1] - w) // 2
    return x[..., top : top + h, left : left + w].contiguous()


def rewrite_ismrmrd_header(header: str, *, field_strength_t: float, receiver_channels: int) -> str:
    """Set ``systemFieldStrength_T`` and ``receiverChannels``; raise when a tag is absent."""
    for pattern, value, name in (
        (_FIELD_TAG, f"{field_strength_t:g}", "systemFieldStrength_T"),
        (_COIL_TAG, str(receiver_channels), "receiverChannels"),
    ):
        header, n = pattern.subn(rf"\g<1>{value}\g<2>", header)
        if n == 0:
            raise ValueError(f"ismrmrd_header has no <{name}> tag to rewrite")
    return header


def build_ulf_volume(
    src_path: Path, dst_path: Path, *, file_id: str, params: ULFCorpusParams
) -> tuple[int, int, int]:
    """Read one source h5, write its synthetic 64 mT twin; return the output ``(S, H, W)``."""
    import h5py

    with h5py.File(PathResolver.resolve(str(src_path)), "r") as f:
        if "kspace" not in f:
            raise KeyError(f"{src_path} has no 'kspace' dataset")
        kspace = torch.from_numpy(f["kspace"][()])
        if "ismrmrd_header" not in f:
            raise KeyError(f"{src_path} has no 'ismrmrd_header'; refusing to invent one")
        header = f["ismrmrd_header"][()]
        rss_shape = f["reconstruction_rss"].shape if "reconstruction_rss" in f else None
    header = header.decode() if isinstance(header, bytes) else str(header)
    if kspace.ndim == 3:  # single-coil fastMRI layout (S, H, W)
        kspace = kspace.unsqueeze(1)
    seed = volume_seed(file_id)
    kspace_ulf, rss_ulf = synthesize_ulf_kspace(kspace, params, seed=seed)
    if rss_shape is not None:  # fastMRI stores the target centre-cropped (e.g. 640x320 -> 320x320)
        rss_ulf = _center_crop(rss_ulf, tuple(rss_shape[-2:]))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst_path, "w") as f:
        f.create_dataset("kspace", data=kspace_ulf.numpy())
        f.create_dataset("reconstruction_rss", data=rss_ulf.numpy())
        f.create_dataset(
            "ismrmrd_header",
            data=rewrite_ismrmrd_header(
                header, field_strength_t=params.target_field_strength_t, receiver_channels=1
            ),
        )
        f.attrs["synthetic_ulf_provenance"] = json.dumps(
            {
                "corpus_version": CORPUS_VERSION,
                "simulator": "VLFSimulator",
                "source_path": str(src_path),
                "source_file_id": file_id,
                "seed": seed,
                **asdict(params),
            },
            sort_keys=True,
        )
    s, _, h, w = kspace_ulf.shape
    return int(s), int(h), int(w)


def build_ulf_corpus(
    records: list[dict[str, Any]],
    *,
    source_root: Path,
    out_root: Path,
    params: ULFCorpusParams,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Synthesize every on-field record; return ``(v3 records, skipped file_ids)``.

    A source whose header field strength is not ``source_field_strength_t`` (fastMRI
    brain mixes 1.5 T and 3 T scanners) is skipped and returned for the caller to
    report. ``subject_id`` is carried from the source record (or its ``file_id``) so
    the cross-field builder can keep train and validation subject-disjoint.
    """
    out: list[dict[str, Any]] = []
    skipped: list[str] = []
    for rec in records:
        rel = rec.get("relative_path")
        if not rel:
            raise ValueError(f"source record without 'relative_path': {rec!r}")
        file_id = str(rec.get("file_id") or Path(rel).stem)
        if not is_on_field(source_root / rel, params.source_field_strength_t):
            skipped.append(file_id)
            continue
        dst_rel = Path(rel).with_suffix(".h5")
        shape = build_ulf_volume(
            source_root / rel, out_root / dst_rel, file_id=file_id, params=params
        )
        out.append(
            {
                "relative_path": str(dst_rel),
                "file_id": f"{file_id}_ulf64mT",
                "subject_id": str(rec.get("subject_id") or file_id),
                "field_strength": params.target_field_strength_t,
                "source_file_id": file_id,
                "shape": list(shape),
            }
        )
    out.sort(key=lambda r: r["file_id"])
    return out, sorted(skipped)
