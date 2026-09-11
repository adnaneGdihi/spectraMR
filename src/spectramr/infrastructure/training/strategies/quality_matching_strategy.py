"""Frozen-parameter orchestrator: fit a degradation chain, synthesise LQ volumes.

No learned parameters. The strategy walks the fit under the standard training
harness -- the ``PairedSynthesisStrategy`` precedent -- and its deliverable is the
calibration artifact plus the synthetic volumes, not a checkpoint.

The artifact carries a ``digital_twin`` block of DEGENERATE ``degradation_ranges``,
so a downstream arm replays the fitted severities through the already-wired
production simulator rather than through anything defined here.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch
import yaml

from spectramr.infrastructure.physics.chain_fitter import FitResult, fit_chain

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)

__all__ = [
    "QualityMatchingStrategy",
    "cohort_volume_paths",
    "extract_hq_header",
    "extract_hq_volume",
    "measure_cohort_acquisition",
    "measure_cohort_spacing",
    "measure_cohort_target",
    "paired_agreement",
    "require_quality_matching_config",
    "resolve_target",
    "run_quality_matching",
    "select_explicit_pairs",
    "select_field",
    "select_field_pairs",
    "synthesise_pairs",
    "write_calibration_artifact",
]


def require_quality_matching_config(config: Any) -> Any:
    """Return ``config.training.quality_matching`` or RAISE.

    The field is declared ``| None``, so ``hasattr`` is always True and an absent
    block reads as ``None``. Substituting a default chain here would let an arm
    advertise quality matching and silently run without it (pitfall #16).
    """
    cfg = getattr(getattr(config, "training", None), "quality_matching", None)
    if cfg is None:
        raise ValueError(
            "training.quality_matching is required by QualityMatchingStrategy but is "
            "absent. Declare the block; there is deliberately no default chain, "
            "because a default would make an unconfigured arm look configured."
        )
    return cfg


def cohort_volume_paths(manifest_path: Path | str) -> list[str]:
    """Volume paths listed in a low-quality-cohort manifest. Raises if unusable.

    ``read_manifest_records`` is deliberately tolerant -- it returns ``None`` for a
    missing or unparseable manifest so the *audit* can report "skipped" rather than
    crash. Here that tolerance would be dangerous: manifests are gitignored and
    regenerated on-cluster, so a silently-empty cohort would fit the chain against
    nothing and still emit a confident-looking calibration.
    """
    records = _records(manifest_path)
    root = _manifest_root(manifest_path)
    paths = [p for r in records if (p := _record_path(r, root)) is not None]
    if not paths:
        raise ValueError(
            f"cohort manifest {manifest_path!r} has {len(records)} records but none "
            f"carries a usable volume path (looked for {list(_ABSOLUTE_PATH_KEYS)}, "
            f"and {list(_RELATIVE_PATH_KEYS)} joined to the manifest's data_root)."
        )
    return paths


def _records(manifest_path: Path | str) -> list[dict[str, Any]]:
    """Manifest records, or RAISE. See :func:`cohort_volume_paths` on tolerance."""
    from spectramr.data.split_leakage import read_manifest_records

    records = read_manifest_records(manifest_path)
    if records is None:
        raise FileNotFoundError(
            f"manifest {manifest_path!r} is missing or unreadable. Manifests are "
            "gitignored and never rsynced -- regenerate it on the cluster with the "
            "committed generator (scripts/data/build_mrixfields2026_manifest.py)."
        )
    return records


#: Keys holding a path usable AS IS (the paired manifests).
_ABSOLUTE_PATH_KEYS = ("primary_path", "path", "file", "image")
#: Keys holding a FRAGMENT that must be joined to the manifest's data_root
#: (the cluster inventory manifests).
_RELATIVE_PATH_KEYS = ("relative_path", "filename")


def _manifest_root(manifest_path: Path | str) -> str | None:
    """The manifest's own declared ``data_root``, if it has one.

    ``regenerate_cluster_manifests.py`` and ``build_mrixfields2026_manifest.py`` store
    paths RELATIVE to a root recorded at the top of the document. Ignoring that root
    leaves you holding a bare filename.
    """
    try:
        payload = json.loads(Path(manifest_path).read_text())
    except (OSError, ValueError):
        return None
    if isinstance(payload, dict) and payload.get("data_root"):
        return str(payload["data_root"])
    return None


def _record_path(record: dict[str, Any], root: str | None = None) -> str | None:
    """A loadable path for one record.

    The cluster inventory manifests emit ``relative_path`` / ``filename`` plus a
    top-level ``data_root``; the paired manifests emit a ready ``primary_path``.

    Treating ``filename`` as a path -- which a naive key sweep does, because it looks
    like one -- yields a bare BASENAME that either fails to open or, far worse, opens
    a same-named file in the working directory. Hence the two-tier lookup.
    """
    for key in _ABSOLUTE_PATH_KEYS:
        value = record.get(key)
        if value:
            return str(value)
    for key in _RELATIVE_PATH_KEYS:
        value = record.get(key)
        if value:
            if root is None:
                raise ValueError(
                    f"record carries only the relative key {key!r} ({value!r}) but the "
                    "manifest declares no data_root, so it cannot be resolved to a "
                    "loadable path. Regenerate with the committed generator, which "
                    "writes data_root."
                )
            return str(Path(root) / str(value))
    return None


def select_field(
    manifest_path: Path | str,
    field_t: float,
    *,
    contrast: str | None = None,
    tol: float = 1e-6,
) -> list[str]:
    """Volume paths acquired at ``field_t`` Tesla, optionally one contrast.

    A field with no records RAISES rather than returning an empty list: an empty
    cohort silently becomes "measure nothing", and the fit would calibrate against a
    target derived from zero volumes.
    """
    records = _records(manifest_path)
    root = _manifest_root(manifest_path)
    hits = [
        p
        for r in records
        if abs(float(r.get("field_strength", float("nan"))) - float(field_t)) <= tol
        and (contrast is None or str(r.get("contrast", "")) == contrast)
        and (p := _record_path(r, root)) is not None
    ]
    if not hits:
        available = sorted({float(r["field_strength"]) for r in records if "field_strength" in r})
        raise ValueError(
            f"no volumes at {field_t} T"
            + (f" with contrast {contrast!r}" if contrast else "")
            + f" in {manifest_path!r}. Fields present: {available}."
        )
    return hits


def select_field_pairs(
    manifest_path: Path | str,
    source_field_t: float,
    target_field_t: float,
    *,
    contrast: str | None = None,
    tol: float = 1e-6,
) -> list[tuple[str, str]]:
    """``[(source_path, target_path)]`` for subjects scanned at BOTH fields.

    This is what a travelling-volunteer cohort buys and an unpaired one cannot: the
    same anatomy at both field strengths, so a synthesised low-field volume can be
    compared against the REAL low-field scan of that subject with full-reference
    metrics. No-reference attribute matching can never establish that.

    Subjects present at only one field are dropped -- they are not pairs. The count
    is logged rather than silently absorbed, because a manifest that yields far fewer
    pairs than expected usually means the cohort is retrospective (one field per
    volunteer) and was mislabelled as paired.
    """
    records = _records(manifest_path)
    root = _manifest_root(manifest_path)
    groups: dict[str, dict[float, str]] = {}
    for r in records:
        path = _record_path(r, root)
        if path is None or "field_strength" not in r:
            continue
        if contrast is not None and str(r.get("contrast", "")) != contrast:
            continue
        key = str(r.get("pairing_group") or r.get("subject_id") or path)
        groups.setdefault(key, {})[float(r["field_strength"])] = path

    def _at(fields: dict[float, str], want: float) -> str | None:
        for got, path in fields.items():
            if abs(got - want) <= tol:
                return path
        return None

    pairs: list[tuple[str, str]] = []
    for key in sorted(groups):
        src = _at(groups[key], source_field_t)
        dst = _at(groups[key], target_field_t)
        if src is not None and dst is not None:
            pairs.append((src, dst))

    if not pairs:
        raise ValueError(
            f"no subject in {manifest_path!r} appears at BOTH {source_field_t} T and "
            f"{target_field_t} T, so there are no pairs. A retrospective cohort has "
            "one field per volunteer and must be used with pairing: unpaired."
        )
    logger.info(
        "QualityMatching: %d paired subjects at %.3g T -> %.3g T (of %d groups)",
        len(pairs),
        source_field_t,
        target_field_t,
        len(groups),
    )
    return pairs


def select_explicit_pairs(
    manifest_path: Path | str,
    *,
    contrast: str | None = None,
    hq_keys: Sequence[str] = ("target_path",),
    lq_keys: Sequence[str] = ("input_path", "primary_path"),
) -> list[tuple[str, str]]:
    """``[(hq_path, lq_path)]`` from a manifest whose records ALREADY carry both sides.

    **The naming inverts against this direction, and that is the trap.** A ULF->HF
    manifest is written for RESTORATION: ``input_path`` / ``primary_path`` is the ULF
    scan (the model's input) and ``target_path`` is the HF scan (what it must
    produce). Degradation runs the other way, so the high-quality SOURCE here is the
    manifest's *target* and the low-quality TARGET is its *input*.

    Reading ``primary_path`` as the source -- the obvious first guess -- would degrade
    the ULF scan toward itself and still emit a confident-looking calibration, since
    the attributes would already nearly match.

    Records missing either side are dropped: a v4 manifest may carry unpaired ULF
    volumes (``target_path: null``) when ``allow_unpaired`` is set for the restoration
    arms, and those cannot anchor a paired check.
    """
    records = _records(manifest_path)

    def _pick(record: dict[str, Any], keys: Sequence[str]) -> str | None:
        for key in keys:
            value = record.get(key)
            if value:
                return str(value)
        return None

    pairs: list[tuple[str, str]] = []
    unpaired = 0
    for r in records:
        if contrast is not None and str(r.get("contrast", "")) != contrast:
            continue
        hq = _pick(r, hq_keys)
        lq = _pick(r, lq_keys)
        if hq and lq:
            pairs.append((hq, lq))
        else:
            unpaired += 1

    if not pairs:
        raise ValueError(
            f"{manifest_path!r} yielded no complete pairs"
            + (f" for contrast {contrast!r}" if contrast else "")
            + f" ({unpaired} record(s) carried only one side). A manifest built with "
            "allow_unpaired may hold ULF volumes with target_path: null; those cannot "
            "anchor a paired check. Use pairing: unpaired instead."
        )
    if unpaired:
        logger.info(
            "QualityMatching: %d complete pairs from %s (%d record(s) dropped as one-sided)",
            len(pairs),
            manifest_path,
            unpaired,
        )
    return pairs


def paired_agreement(
    synthetic: torch.Tensor,
    real_lq: torch.Tensor,
    *,
    metrics: Sequence[str] = ("psnr", "ssim"),
) -> dict[str, float]:
    """Full-reference agreement between a synthesised volume and the REAL low-field
    scan of the same subject.

    This is the only check that can say whether the degradation chain is *right*
    rather than merely *consistent*: the fit matches no-reference attributes, which a
    wrong chain can also match. Reported, never optimised -- folding it into the
    objective would turn an independent check into a training signal.
    """
    from spectramr.core.metrics.registry import get_metric

    a = synthetic.float()
    b = real_lq.float()
    if a.shape != b.shape:
        import torch.nn.functional as F

        a = F.interpolate(
            a[None, None], size=tuple(b.shape), mode="trilinear", align_corners=False
        )[0, 0]
    a4 = a.unsqueeze(1) if a.dim() == 3 else a
    b4 = b.unsqueeze(1) if b.dim() == 3 else b
    return {m: float(get_metric(m)(a4, b4)) for m in metrics}


def _cohort_spacing(path: str) -> tuple[float, float, float]:
    """``(slice, row, col)`` mm of one cohort volume, from its ISMRMRD header."""
    from spectramr.data.io_strategies import FastMRIH5Strategy
    from spectramr.infrastructure.physics.quality_descriptors import read_spacing_mm

    payload = FastMRIH5Strategy.load_reference_volume(path)
    return read_spacing_mm(payload["header"], payload["data"])


def measure_cohort_spacing(
    volume_paths: Sequence[str], *, max_volumes: int | None = None
) -> tuple[float, float, float]:
    """Per-axis MEDIAN voxel spacing across the low-quality cohort.

    This is the GEOMETRIC half of the target: the resolution the synthetic volume is
    resampled onto. Measured from real headers rather than assumed, for the same
    reason the quality attributes are -- an asserted resolution is a fabrication.
    """
    paths = list(volume_paths)
    if max_volumes is not None:
        paths = paths[:max_volumes]
    if not paths:
        raise ValueError("cannot measure cohort spacing from zero volumes")
    per_volume = [_cohort_spacing(p) for p in paths]
    return (
        float(statistics.median(s[0] for s in per_volume)),
        float(statistics.median(s[1] for s in per_volume)),
        float(statistics.median(s[2] for s in per_volume)),
    )


def _cohort_acquisition(path: str) -> Any:
    """Acquisition parameters of one cohort volume, from its ISMRMRD header."""
    from spectramr.data.io_strategies import FastMRIH5Strategy
    from spectramr.infrastructure.physics.acquisition_params import (
        read_acquisition_params,
    )

    return read_acquisition_params(FastMRIH5Strategy.load_reference_volume(path)["header"])


def measure_cohort_acquisition(
    volume_paths: Sequence[str], *, max_volumes: int | None = None
) -> Any:
    """Per-field MEDIAN acquisition parameters across the cohort.

    A cohort is not one protocol: subjects differ in averages and occasionally in
    bandwidth. The median is the representative acquisition, consistent with how the
    quality target and the voxel grid are derived. A field absent from EVERY header
    stays ``None`` -- the prior then treats it as neutral rather than inventing one.
    """
    import statistics as _stats

    from spectramr.infrastructure.physics.acquisition_params import AcquisitionParams

    paths = list(volume_paths)
    if max_volumes is not None:
        paths = paths[:max_volumes]
    if not paths:
        raise ValueError("cannot measure cohort acquisition from zero volumes")

    per_volume = [_cohort_acquisition(p) for p in paths]

    def _median(field: str) -> float | None:
        vals = [getattr(a, field) for a in per_volume if getattr(a, field) is not None]
        return float(_stats.median(vals)) if vals else None

    return AcquisitionParams(
        field_strength_t=_median("field_strength_t"),
        te_ms=_median("te_ms"),
        tr_ms=_median("tr_ms"),
        bandwidth_hz_px=_median("bandwidth_hz_px"),
        averages=_median("averages"),
    )


def _measure_volume(path: str, attributes: Sequence[str]) -> dict[str, float]:
    """Measure one cohort volume's attributes.

    Reading goes through the data layer's ``FastMRIH5Strategy`` so the h5py call
    stays inside ``src/spectramr/data/`` (pitfall #11 -- higher layers never open a
    file themselves). ``load_reference_volume`` reads only the reconstruction, not
    the ~40x larger k-space.
    """
    from spectramr.data.io_strategies import FastMRIH5Strategy
    from spectramr.infrastructure.physics.quality_descriptors import measure_attributes

    payload = FastMRIH5Strategy.load_reference_volume(path)
    return measure_attributes(payload["data"], attributes=attributes)


def measure_cohort_target(
    volume_paths: Sequence[str],
    *,
    attributes: Sequence[str],
    max_volumes: int | None = None,
) -> dict[str, float]:
    """Per-attribute MEDIAN across a real low-quality cohort.

    Median rather than mean: a cohort carries outliers (a motion-corrupted subject,
    a truncated volume), and a single bad volume would drag a mean target into a
    regime no real scan occupies.
    """
    paths = list(volume_paths)
    if max_volumes is not None:
        paths = paths[:max_volumes]
    if not paths:
        raise ValueError("cannot measure a cohort target from zero volumes")

    per_volume = [_measure_volume(p, attributes) for p in paths]
    return {key: float(statistics.median(v[key] for v in per_volume)) for key in attributes}


def resolve_target(
    target_cfg: Any,
    *,
    cohort_paths: Sequence[str] | None = None,
    max_volumes: int | None = None,
) -> dict[str, float]:
    """The matched quality target, from a cohort measurement or literal values.

    With ``source='cohort'`` the overrides are applied ON TOP of the measured
    values, which is the documented ablation pattern: inherit the cohort fit and pin
    one attribute at a time.
    """
    attributes = list(target_cfg.attributes)

    if target_cfg.source == "literal":
        # The schema already guarantees an override exists for every attribute.
        return {k: float(target_cfg.override[k]) for k in attributes}

    if not cohort_paths:
        raise ValueError(
            "target.source='cohort' but no cohort volume paths were supplied, so "
            "there is nothing to measure. Resolve them from "
            f"target.cohort_manifest ({target_cfg.cohort_manifest!r}) first."
        )

    measured = measure_cohort_target(cohort_paths, attributes=attributes, max_volumes=max_volumes)
    measured.update({k: float(v) for k, v in target_cfg.override.items()})
    return measured


def _paired_agreement_over(
    chain: Any,
    field_pairs: Sequence[tuple[str, str]],
    *,
    seed: int,
    dst_spacing_mm: tuple[float, float, float] | None,
    max_subjects: int | None = None,
) -> dict[str, float]:
    """Median full-reference agreement across paired subjects.

    Median rather than mean for the same reason the target is: one motion-corrupted
    volunteer should not set the headline number.
    """
    import statistics as _stats

    from spectramr.data.io_strategies import FastMRIH5Strategy, NiftiStrategy
    from spectramr.infrastructure.physics.quality_descriptors import (
        read_spacing_mm,
        resample_to_spacing,
    )

    def _load(path: str) -> tuple[torch.Tensor, Any]:
        if str(path).endswith((".nii", ".nii.gz")):
            return NiftiStrategy().load(path)["data"].squeeze().float(), None
        payload = FastMRIH5Strategy.load_reference_volume(path)
        return payload["data"].float(), payload["header"]

    pairs = list(field_pairs)
    if max_subjects is not None:
        pairs = pairs[:max_subjects]

    per_subject: list[dict[str, float]] = []
    for hq_path, lq_path in pairs:
        hq_vol, hq_hdr = _load(hq_path)
        real_lq, _ = _load(lq_path)
        vol = hq_vol
        if dst_spacing_mm is not None and hq_hdr is not None:
            vol = resample_to_spacing(hq_vol, read_spacing_mm(hq_hdr, hq_vol), dst_spacing_mm)
        synth = chain.apply(vol.unsqueeze(1), seed=seed).squeeze(1).abs()
        per_subject.append(paired_agreement(synth, real_lq))

    keys = per_subject[0].keys()
    return {k: float(_stats.median(d[k] for d in per_subject)) for k in keys}


def synthesise_pairs(
    chain: Any,
    hq_paths: Sequence[str],
    output_dir: Path | str,
    *,
    seed: int,
    dst_spacing_mm: tuple[float, float, float] | None = None,
    val_fraction: float = 0.1,
    max_volumes: int | None = None,
) -> Path:
    """Apply the fitted chain to each HQ volume; write the pairs and a v4 manifest.

    This is what turns a calibration into a usable dataset. Without it the fit is a
    number in a YAML and the downstream restoration arm has no data to read.

    Each volume is written twice -- the degraded ``primary`` and the clean ``target``
    -- on the SAME grid, because a paired restoration set needs input and target
    voxel-aligned. Writes go through the data layer's ``write_slice_first_nifti`` and
    are read back with ``verify_slice_first_nifti``: a silently transposed write would
    hand the downstream arm a cut through the wrong plane.

    Returns the manifest path, which the downstream arm consumes via
    ``data.paired_manifest_path``.
    """
    from spectramr.data.io_strategies import FastMRIH5Strategy
    from spectramr.data.nifti_export import (
        VoxelGeometry,
        verify_slice_first_nifti,
        write_slice_first_nifti,
    )
    from spectramr.infrastructure.physics.quality_descriptors import (
        read_spacing_mm,
        resample_to_spacing,
    )

    paths = list(hq_paths)
    if max_volumes is not None:
        paths = paths[:max_volumes]
    if not paths:
        raise ValueError("cannot synthesise pairs from zero high-quality volumes")

    out = Path(output_dir)
    lq_dir, hq_dir = out / "lq", out / "hq"
    lq_dir.mkdir(parents=True, exist_ok=True)
    hq_dir.mkdir(parents=True, exist_ok=True)

    n_val = max(1, round(len(paths) * float(val_fraction))) if len(paths) > 1 else 0
    records: list[dict[str, Any]] = []

    for idx, src in enumerate(paths):
        payload = FastMRIH5Strategy.load_reference_volume(src)
        volume = payload["data"].float()
        src_spacing = read_spacing_mm(payload["header"], volume)

        grid = src_spacing
        if dst_spacing_mm is not None:
            volume = resample_to_spacing(volume, src_spacing, dst_spacing_mm)
            grid = dst_spacing_mm

        # Per-volume seed: distinct artefact realisations across the cohort, but
        # reproducible for a given (seed, index). One realisation reused for every
        # subject would teach the restorer that artefact, not the degradation.
        degraded = chain.apply(volume.unsqueeze(1), seed=seed + idx).squeeze(1).abs().float()

        stem = Path(str(src)).name.replace(".h5", "")
        geom = VoxelGeometry(
            slice_mm=float(grid[0]),
            row_mm=float(grid[1]),
            col_mm=float(grid[2]),
            n_slices=int(volume.shape[0]),
            rows=int(volume.shape[1]),
            cols=int(volume.shape[2]),
            slice_gap_assumed_zero=True,
            source="quality_matching_synthesis",
        )
        lq_path = write_slice_first_nifti(lq_dir / f"{stem}.nii.gz", degraded, geom)
        hq_path = write_slice_first_nifti(hq_dir / f"{stem}.nii.gz", volume, geom)
        verify_slice_first_nifti(lq_path, degraded)
        verify_slice_first_nifti(hq_path, volume)

        records.append(
            {
                "subject_id": stem,
                "primary_path": str(lq_path),
                "target_path": str(hq_path),
                "pairing_status": "paired",
                "split_hint": "val" if idx >= len(paths) - n_val else "train",
            }
        )

    manifest = out / "quality_matching_synth_pairs.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": "4",
                "generator": "spectramr.infrastructure.training.strategies."
                "quality_matching_strategy.synthesise_pairs",
                "chain": [{"axis": link.axis, "theta": float(link.theta)} for link in chain.links],
                "seed": int(seed),
                "spacing_mm": [float(v) for v in (dst_spacing_mm or ())] or None,
                "records": records,
            },
            indent=2,
        )
    )
    logger.info(
        "QualityMatching: wrote %d synthetic pairs and a v4 manifest to %s",
        len(records),
        manifest,
    )
    return manifest


def write_calibration_artifact(
    result: FitResult,
    output_dir: Path | str,
    *,
    spacing_mm: tuple[float, float, float] | None = None,
    source_spacing_mm: tuple[float, float, float] | None = None,
    acquisition_snr_delta_db: float | None = None,
    theta0: Sequence[float] | None = None,
    pairing: str | None = None,
    paired_agreement: dict[str, float] | None = None,
) -> Path:
    """Write the auditable record of one fit. Returns the artifact path.

    Records achieved-versus-target PER ATTRIBUTE, not just the scalar residual: a
    single number hides which attribute missed and by how much. ``spacing_mm`` is the
    IMPOSED target grid and ``source_spacing_mm`` the grid it came from -- both are
    recorded so a reader can see the geometric half of the match, which no quality
    attribute reveals.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "calibration.yaml"

    doc: dict[str, Any] = {
        "chain": [{"axis": link.axis, "theta": float(link.theta)} for link in result.chain.links],
        "fit": {
            "method": result.method,
            "seed": result.seed,
            "n_evals": result.n_evals,
            "residual": float(result.residual),
            "initial_residual": float(result.initial_residual),
            "gap_closed": float(result.gap_closed),
        },
        "attributes": {
            key: {
                "target": float(result.target[key]),
                "achieved": float(result.achieved.get(key, float("nan"))),
                "weight": float(result.weights[key]),
            }
            for key in result.target
        },
        # Replayable through the production twin path: degenerate ranges pin theta
        # regardless of the diffusion corruption factor, `enabled` is set, and the
        # simulator's own unconditional AWGN stage is pinned to zero so the replay
        # reproduces the fitted degradation instead of adding an unfitted 10-25 dB
        # noise draw on top of it. The block still does not choose a ROUTE: an arm
        # consuming it sets physics.digital_twin.apply_as_transform (k-space
        # datasets) or selects a VF strategy that builds the simulator itself.
        "digital_twin": result.chain.to_digital_twin_config(),
    }
    if spacing_mm is not None:
        doc["spacing_mm"] = [float(v) for v in spacing_mm]
    if source_spacing_mm is not None:
        doc["source_spacing_mm"] = [float(v) for v in source_spacing_mm]
    if pairing is not None:
        doc["pairing"] = str(pairing)
    if paired_agreement is not None:
        # The independent check. Recorded next to the residual so a reader can see
        # that a good attribute match did NOT guarantee a good reconstruction of the
        # real low-field scan -- or that it did.
        doc["paired_agreement_vs_real"] = {k: float(v) for k, v in paired_agreement.items()}
    if acquisition_snr_delta_db is not None:
        # The prior is recorded so a reader can see WHY the search started where it
        # did, and check the prediction against what the fit actually landed on.
        doc["acquisition_prior"] = {
            "predicted_snr_delta_db": float(acquisition_snr_delta_db),
            "theta0": [float(t) for t in (theta0 or ())],
        }

    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    logger.info(
        "QualityMatching: calibration written to %s (gap closed %.1f%%, %d evals)",
        path,
        100.0 * result.gap_closed,
        result.n_evals,
    )
    return path


def extract_hq_volume(batch: Any) -> torch.Tensor:
    """Pull the high-quality volume out of a training batch, or RAISE.

    Accepts a bare tensor or a mapping under the usual keys. Guessing wrong here
    would fit the chain against the wrong tensor and still produce a confident
    calibration, so an unrecognised batch is an error rather than a fallback.
    """
    if isinstance(batch, torch.Tensor):
        return batch
    if hasattr(batch, "get"):
        for key in ("target", "image", "input", "data", "hq"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return value
        raise ValueError(
            "QualityMatchingStrategy could not find a high-quality volume in the "
            f"batch: none of target/image/input/data/hq holds a tensor (keys: "
            f"{sorted(batch.keys()) if hasattr(batch, 'keys') else '?'})."
        )
    raise ValueError(
        f"QualityMatchingStrategy needs a tensor or mapping batch; got {type(batch)!r}."
    )


def extract_hq_header(batch: Any) -> Any:
    """The ISMRMRD header carried alongside the volume, or ``None``.

    Returns ``None`` rather than raising: whether a header is REQUIRED is
    ``match_spacing``'s decision, made in :func:`run_quality_matching`. Deciding it
    here would either force a header on quality-only arms or silently skip the
    geometric match on arms that asked for it.
    """
    if hasattr(batch, "get"):
        for key in ("header", "ismrmrd_header", "hq_header"):
            value = batch.get(key)
            if value is not None:
                return value
    return None


def run_quality_matching(
    qm_config: Any,
    hq_volume: torch.Tensor,
    *,
    hq_header: Any = None,
    hq_manifest: str | None = None,
) -> FitResult:
    """Resolve the target, impose its geometry, fit the chain, write the artifact.

    Two halves, handled differently on purpose:

    * **Geometric (imposed).** When ``match_spacing`` is on and both headers are
      available, the high-quality volume is resampled onto the cohort's measured
      voxel grid BEFORE fitting. Resolution is a header fact, not a knob -- fitting
      it against a sharpness proxy would let a blur term absorb a geometry error.
    * **Quality (fitted).** The severities are then fitted so the *already
      correctly-sampled* volume matches the cohort's no-reference attributes.

    Order matters: resampling after the fit would change the sharpness the fit had
    just matched, silently invalidating the calibration.
    """
    from spectramr.infrastructure.physics.quality_descriptors import (
        read_spacing_mm,
        resample_to_spacing,
    )

    target_cfg = qm_config.target

    cohort_paths = None
    field_pairs: list[tuple[str, str]] | None = None

    if target_cfg.source == "cohort":
        if target_cfg.pairing == "paired":
            # Pairs are kept so the synthesised volume can later be checked against
            # the REAL low-quality scan of the same subject.
            if target_cfg.pair_layout == "explicit":
                # Records already carry both sides (a ULF->HF restoration manifest).
                field_pairs = select_explicit_pairs(
                    target_cfg.cohort_manifest, contrast=target_cfg.contrast
                )
            else:
                # Per-field snapshots grouped by subject (a travelling-volunteer
                # cohort such as MRIxFields2026).
                field_pairs = select_field_pairs(
                    target_cfg.cohort_manifest,
                    float(target_cfg.source_field_t),
                    float(target_cfg.target_field_t),
                    contrast=target_cfg.contrast,
                )
            cohort_paths = [lq for _hq, lq in field_pairs]
        else:
            cohort_paths = cohort_volume_paths(target_cfg.cohort_manifest)
        logger.info(
            "QualityMatching: %s target over %d cohort volumes from %s",
            target_cfg.pairing,
            len(cohort_paths),
            target_cfg.cohort_manifest,
        )

    target = resolve_target(target_cfg, cohort_paths=cohort_paths)
    logger.info("QualityMatching: target resolved -> %s", target)

    # ── geometric half: impose the cohort's voxel grid ──
    src_spacing: tuple[float, float, float] | None = None
    dst_spacing: tuple[float, float, float] | None = None
    volume = hq_volume

    if qm_config.match_spacing:
        # Resolution order, both real MEASUREMENTS -- neither is an assumption:
        #   1. a header carried on the batch, when the dataset propagates one;
        #   2. otherwise the first volume of the arm's own declared HQ manifest.
        # Most datasets (slice_dataset, which serves dataset_type: kspace) drop the
        # ISMRMRD header during collation, so (2) is the usual path. If neither
        # yields a header this RAISES -- an assumed 1 mm isotropic default would put
        # every synthetic volume on the wrong grid.
        if hq_header is not None:
            src_spacing = read_spacing_mm(hq_header, hq_volume)
        elif hq_manifest:
            src_spacing = measure_cohort_spacing(cohort_volume_paths(hq_manifest), max_volumes=1)
            logger.info(
                "QualityMatching: batch carried no header; source spacing read from %s -> %s mm",
                hq_manifest,
                src_spacing,
            )
        else:
            raise ValueError(
                "match_spacing is enabled but no ISMRMRD header is reachable: the "
                "batch carried none and no high-quality manifest was supplied. "
                "Resampling on an assumed 1 mm default would put every synthetic "
                "volume on the wrong grid. Point data.index_path at a manifest, or "
                "set match_spacing: false and say so in metadata.note."
            )
        dst_spacing = (
            tuple(target_cfg.spacing_mm)  # type: ignore[assignment]
            if target_cfg.spacing_mm is not None
            else measure_cohort_spacing(cohort_paths or [])
        )
        volume = resample_to_spacing(hq_volume, src_spacing, dst_spacing)
        logger.info(
            "QualityMatching: imposed grid %s mm -> %s mm (in-plane)",
            src_spacing,
            dst_spacing,
        )

    # ── acquisition prior: a physics-derived warm start for the noise axis ──
    # The header records HOW the scan was acquired, which PREDICTS part of the gap.
    # Feeding that in as theta0 starts the search in a physically plausible basin
    # instead of mid-box -- which matters precisely because the chain is usually
    # underdetermined, so several severity combinations reach the same measured
    # quality and something has to choose between them.
    theta0: list[float] | None = None
    snr_delta_db: float | None = None

    if qm_config.acquisition_prior_enabled:
        from spectramr.infrastructure.physics.acquisition_params import (
            predicted_snr_delta_db,
            read_acquisition_params,
        )
        from spectramr.infrastructure.physics.chain_fitter import acquisition_warm_start

        if hq_header is not None:
            hq_acq = read_acquisition_params(hq_header)
        elif hq_manifest:
            hq_acq = measure_cohort_acquisition(cohort_volume_paths(hq_manifest), max_volumes=1)
        else:
            raise ValueError(
                "the acquisition prior is enabled but no high-quality acquisition "
                "header is reachable. Point data.index_path at a manifest, or set "
                "use_acquisition_prior: false -- a guessed field strength would make "
                "the prior a fabrication."
            )

        if not cohort_paths:
            raise ValueError(
                "the acquisition prior needs the low-quality cohort's acquisition, "
                "which requires target.source='cohort'. With a literal target there "
                "is no header to read the field strength from."
            )
        lq_acq = measure_cohort_acquisition(cohort_paths)

        snr_delta_db = predicted_snr_delta_db(hq_acq, lq_acq)
        theta0 = acquisition_warm_start(list(qm_config.axes), snr_delta_db)
        logger.info(
            "QualityMatching: acquisition prior %.2f T -> %.2f T predicts "
            "%+.2f dB SNR; theta0 = %s",
            hq_acq.field_strength_t,
            lq_acq.field_strength_t,
            snr_delta_db,
            [round(t, 4) for t in theta0],
        )

    result = fit_chain(
        volume,
        axes=list(qm_config.axes),
        target=target,
        attributes=list(target_cfg.attributes),
        theta0=theta0,
        seed=int(qm_config.fit_seed),
        max_evals=int(qm_config.max_evals),
        method=str(qm_config.method),
        min_gap_closed=float(qm_config.min_gap_closed),
    )
    # ── paired check: does the chain reproduce the REAL low-field scan? ──
    # The fit matches no-reference attributes, which a WRONG chain can also match.
    # Only a paired cohort can answer whether it is right. Reported, never optimised.
    agreement: dict[str, float] | None = None
    if field_pairs:
        agreement = _paired_agreement_over(
            result.chain,
            field_pairs,
            seed=int(qm_config.fit_seed),
            dst_spacing_mm=dst_spacing,
            max_subjects=qm_config.max_synth_volumes,
        )
        logger.info("QualityMatching: paired agreement vs REAL low-field: %s", agreement)

    write_calibration_artifact(
        result,
        qm_config.output_dir,
        spacing_mm=dst_spacing,
        source_spacing_mm=src_spacing,
        acquisition_snr_delta_db=snr_delta_db,
        theta0=theta0,
        pairing=target_cfg.pairing,
        paired_agreement=agreement,
    )

    # Turn the calibration into an actual dataset. Without this the fit is a number
    # in a YAML and the downstream restoration arm has nothing to read.
    if qm_config.synthesise and hq_manifest:
        synthesise_pairs(
            result.chain,
            cohort_volume_paths(hq_manifest),
            Path(qm_config.output_dir) / "synthetic",
            seed=int(qm_config.fit_seed),
            dst_spacing_mm=dst_spacing,
            max_volumes=qm_config.max_synth_volumes,
        )
    elif qm_config.synthesise:
        raise ValueError(
            "synthesise is enabled but no high-quality manifest was supplied, so "
            "there are no volumes to degrade. Point data.index_path at a manifest, "
            "or set synthesise: false if the calibration alone is the deliverable."
        )
    return result


class QualityMatchingStrategy(ReconstructionTrainingStrategy):
    """Fit a compounded degradation chain to a measured quality target.

    A frozen-parameter orchestrator: it LEARNS NOTHING. The fit runs once, on the
    first batch (the earliest point at which real volumes are available), and the
    deliverable is the calibration artifact rather than a checkpoint. Every
    subsequent step is a no-op returning an exactly-zero, differentiable loss so the
    harness's backward/step cycle stays valid while no parameter moves.
    """

    #: Loss ownership (issue #1918). Terminates in `return {"loss_total": self._zero_loss()}` -- it computes no
    #: loss at all and folds nothing.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("quality_matching",))
        self._qm_config = require_quality_matching_config(self.config)
        self._fit_result: FitResult | None = None
        logger.info(
            "QualityMatchingStrategy: axes=%s, method=%s, budget=%d evals, "
            "min_gap_closed=%.2f, target source=%s",
            list(self._qm_config.axes),
            self._qm_config.method,
            self._qm_config.max_evals,
            self._qm_config.min_gap_closed,
            self._qm_config.target.source,
        )

    @property
    def fit_result(self) -> FitResult | None:
        """The completed fit, or ``None`` before the first batch."""
        return self._fit_result

    def _zero_loss(self) -> torch.Tensor:
        """An exactly-zero loss that is still differentiable w.r.t. the model.

        A bare ``torch.zeros(())`` has no grad_fn, so ``backward()`` would raise.
        Summing the parameters and multiplying by zero keeps the graph intact and
        yields exactly-zero gradients -- nothing moves, and nothing crashes.
        """
        params = [p for p in self.env.generator.parameters() if p.requires_grad]
        if not params:
            return torch.zeros((), requires_grad=True)
        return sum(p.sum() for p in params) * 0.0

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if self._fit_result is None:
            batch = input_batch if input_batch is not None else kwargs.get("batch")
            volume = extract_hq_volume(batch)
            # Metrics take [S, H, W]; a [B, C, H, W] batch collapses to slices.
            if volume.dim() == 4:
                volume = volume.abs().mean(dim=1)
            result = run_quality_matching(
                self._qm_config,
                volume,
                hq_header=extract_hq_header(batch),
                # The arm's own HQ manifest: the fallback source of voxel geometry
                # when the dataset drops the header during collation.
                #
                # Canonical `data.source.index_path`, read DIRECTLY. This was
                # `getattr(self.config.data, "index_path", None)` -- a string-keyed
                # read of a name that folded to `data.source.index_path`, so it
                # returned None for every arm, including arms declaring a real
                # manifest. That did not degrade quietly: the comment at the
                # `match_spacing` resolution below records that most datasets drop
                # the ISMRMRD header during collation, making this the USUAL path,
                # so the arm hit `raise ValueError(...)` telling it to "point
                # data.index_path at a manifest" -- which it had already done.
                # `synthesise` hit the same wall. `source` has a default_factory,
                # so the block is always present and a defensive getattr buys
                # nothing (non-negotiable #1: read the canonical path, fail loud).
                hq_manifest=self.config.data.source.index_path,
            )
            self._fit_result = result
            logger.info(
                "QualityMatching: chain fitted -> %s (gap closed %.1f%%)",
                [(link.axis, round(link.theta, 4)) for link in result.chain.links],
                100.0 * result.gap_closed,
            )
        return {"loss_total": self._zero_loss()}
