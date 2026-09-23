#!/usr/bin/env python3
"""
Regenerate all dataset manifests for cluster deployment (v3 JSON format).

This script should be run on the cluster where the actual data resides.
It creates v3 JSON manifests with relative paths for cross-machine portability.

Usage:
    python scripts/data/regenerate_cluster_manifests.py --data-base /path/to/databases
    python scripts/data/regenerate_cluster_manifests.py --data-base /path/to/databases --dry-run
    python scripts/data/regenerate_cluster_manifests.py --data-base /path/to/databases \
        --datasets m4raw_multicoil_train m4raw_multicoil_val

Every ``--datasets`` name above is a key of DATASET_CONFIGS below, and a test
pins that: an unknown name is a hard error, so a stale example in this docstring
is a command that fails the moment someone copies it.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Dataset configurations mapping logical names to physical cluster paths.
# Subpaths are relative to the databases/ root and match the canonical layout.
DATASET_CONFIGS = {
    # ── FastMRI Knee ──────────────────────────────────────────────
    "fastmri_knee_singlecoil": {
        "subpath": "knee/fastmri/singlecoil_train/singlecoil_train",
        "pattern": "**/*.h5",
        "manifest": "data/manifests/fastmri_knee_singlecoil.json",
        "file_type": "h5",
    },
    "fastmri_knee_multicoil": {
        "subpath": "knee/fastmri/multicoil_train/multicoil_train",
        "pattern": "**/*.h5",
        "manifest": "data/manifests/fastmri_knee_multicoil.json",
        "file_type": "h5",
    },
    # ── FastMRI Brain ─────────────────────────────────────────────
    "fastmri_brain_singlecoil": {
        "subpath": "brain/fastmri/singlecoil_train/singlecoil_train",
        "pattern": "**/*.h5",
        "manifest": "data/manifests/fastmri_brain_singlecoil.json",
        "file_type": "h5",
    },
    "fastmri_brain_multicoil": {
        "subpath": "brain/fastmri/multicoil_train/multicoil_train",
        "pattern": "**/*.h5",
        "manifest": "data/manifests/fastmri_brain_multicoil_train.json",
        "file_type": "h5",
    },
    # ── M4Raw ─────────────────────────────────────────────────────
    "m4raw_multicoil_train": {
        "subpath": "m4raw/data/multicoil_train/multicoil_train",
        "pattern": "**/*.h5",
        "manifest": "data/manifests/m4raw_train.json",
        "file_type": "h5",
    },
    "m4raw_multicoil_val": {
        "subpath": "m4raw/data/multicoil_val/multicoil_val",
        "pattern": "**/*.h5",
        "manifest": "data/manifests/m4raw_multicoil_val.json",
        "file_type": "h5",
    },
    "m4raw_multicoil_test": {
        "subpath": "m4raw/data/multicoil_test/multicoil_test",
        "pattern": "**/*.h5",
        "manifest": "data/manifests/m4raw_multicoil_test.json",
        "file_type": "h5",
    },
    "m4raw_motion": {
        "subpath": "m4raw/data/motion/motion",
        "pattern": "**/*.h5",
        "manifest": "data/manifests/m4raw_motion.json",
        "file_type": "h5",
    },
    "m4raw_gre": {
        "subpath": "m4raw/data/gre_data/gre_data",
        "pattern": "**/*.h5",
        "manifest": "data/manifests/m4raw_gre.json",
        "file_type": "h5",
    },
    "m4raw_motion_image_reconstructed": {
        "subpath": "m4raw/data/motion/motion_image/nifti_reconstructed",
        "pattern": "**/*.nii*",
        "manifest": "data/manifests/m4raw_motion_image_reconstructed.json",
        "file_type": "nifti",
    },
    # ── ULF Paired ────────────────────────────────────────────────
    "ulf_paired_brain_image_normalized": {
        "subpath": "ulf_paired/ulf_paired_64mt_3t/Data/64mT_data_image",
        "pattern": "**/*.nii*",
        "manifest": "data/manifests/ulf_paired_brain_image_normalized.json",
        "file_type": "nifti",
    },
}


def _reader_missing(package: str, suffix: str) -> str:
    """Message for a missing metadata reader.

    ``shape`` is not decoration. ``universal_dataset.py`` retains it per record
    so ``TorchIOQueueBuilder._filter_patch_compatible_subjects`` can check
    spatial extent without opening the volume; a manifest built without the
    reader loads fine and then OOMs at queue-build (F-OOM 2026-06-08).
    """
    return (
        f"{package} is required to read {suffix} metadata, and is not installed.\n"
        f"Indexing {suffix} files without it would write records with no 'shape', "
        "which loads without complaint and OOMs later at queue-build time.\n"
        f"Install it first:  pip install {package}"
    )

def repetition_group_key(file_id: str) -> str:
    """The NEX group a file belongs to, by M4Raw's own naming convention.

    Delegates to the package so this and ``M4RawRepetitionDataset`` cannot
    drift: it used to "mirror" the dataset's ``stem[:-2]``, which is the
    two-owners shape that only stays correct while nobody edits either.
    """
    from spectramr.data.datasets.m4raw_identity import repetition_group_key as _key

    return _key(file_id)


def filter_short_repetition_groups(
    records: list[dict], min_reps: int
) -> tuple[list[dict], dict[str, int]]:
    """Drop every NEX group with fewer than *min_reps* repetitions.

    A leave-one-out NEX target needs ``min_reps`` repetitions (1 input + the
    rest averaged). Where a group is short, the dataset's only two escapes both
    change the experiment: ``nex_fallback: all_reps`` swaps in a reference that
    carries the input's own noise at 1/N, and ``pairing.contrasts`` can drop
    only a whole contrast. Removing the deficient GROUP at manifest time keeps
    the reference definition intact and keeps every arm of a cohort on one
    subject set, which is what makes their metrics comparable.

    Returns the surviving records and ``{group: count}`` for those removed.
    """
    counts: dict[str, int] = {}
    for record in records:
        counts[repetition_group_key(record["file_id"])] = (
            counts.get(repetition_group_key(record["file_id"]), 0) + 1
        )
    dropped = {group: n for group, n in counts.items() if n < min_reps}
    kept = [r for r in records if repetition_group_key(r["file_id"]) not in dropped]
    return kept, dropped


def min_reps_manifest_path(output_path: Path, min_reps: int) -> Path:
    """Where a repetition-filtered manifest is written.

    Never the unfiltered name. ``m4raw_train.json`` is shared by the
    reconstruction, physics_driven, vae_latent and transformers cohorts among
    others, so filtering in place would silently change the corpus every one of
    them trains on.
    """
    return output_path.with_name(f"{output_path.stem}_nex{min_reps}{output_path.suffix}")


def create_manifest(
    data_root: Path,
    databases_root: Path,
    pattern: str,
    output_path: Path,
    file_type: str,
    min_reps: int | None = None,
) -> int:
    """Create a v3 JSON manifest with relative paths.

    Args:
        data_root: Absolute path to the directory containing the data files
        databases_root: Absolute path to the databases/ root
        pattern: Glob pattern to match files
        output_path: Path to save the JSON manifest
        file_type: Type of files ('h5', 'nifti', 'numpy', etc.)
        min_reps: Keep only NEX groups with at least this many repetitions, and
            write to the ``_nex<N>`` sibling name. ``None`` disables both.

    Returns:
        Number of files indexed
    """
    if min_reps is not None:
        output_path = min_reps_manifest_path(output_path, min_reps)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Scanning {data_root}...")
    files = sorted(data_root.glob(pattern))

    if not files:
        print(f"  ⚠ No files found matching {pattern}")
        return 0

    # Compute relative data_root from project root
    # e.g. <data-base>/m4raw/data/multicoil_train/... → databases/m4raw/data/...
    try:
        rel_data_root = str(data_root.relative_to(databases_root.parent))
    except ValueError:
        rel_data_root = str(data_root)

    records = []
    print(f"  Reading metadata from {len(files)} files...")

    for p in files:
        try:
            shape = None
            if p.suffix == ".h5":
                # SystemExit is a BaseException, so it escapes the ``except
                # Exception`` below rather than being downgraded to a per-file
                # "Skipping" line. That is deliberate: a missing reader is a
                # property of the environment, not of this one file, so it must
                # stop the run instead of silently hollowing out every record.
                try:
                    import h5py
                except ImportError as exc:  # pragma: no cover - env-dependent
                    raise SystemExit(_reader_missing("h5py", ".h5")) from exc
                with h5py.File(p, "r") as f:
                    if "kspace" in f:
                        shape = list(f["kspace"].shape)
                    elif "reconstruction_rss" in f:
                        shape = list(f["reconstruction_rss"].shape)
            elif ".nii" in p.name:
                try:
                    import nibabel as nib
                except ImportError as exc:  # pragma: no cover - env-dependent
                    raise SystemExit(_reader_missing("nibabel", ".nii")) from exc
                img = nib.load(str(p))
                shape = list(img.shape)

            rel_path = str(p.relative_to(data_root))
            record = {
                "relative_path": rel_path,
                "filename": p.name,
                "file_id": p.stem.replace(".nii", ""),
            }
            if shape is not None:
                record["shape"] = shape

            records.append(record)
        except Exception as e:
            print(f"  Skipping {p.name}: {e}")

    if not records:
        print("  ⚠ No valid files found")
        return 0

    if min_reps is not None:
        records, dropped = filter_short_repetition_groups(records, min_reps)
        if dropped:
            detail = ", ".join(f"{g} ({n})" for g, n in sorted(dropped.items()))
            print(f"  min-reps {min_reps}: dropped {len(dropped)} short group(s): {detail}")
        if not records:
            # Every group was short. Writing an empty manifest here would exit 0
            # and the arm would fail later at queue-build with no trace of why.
            print(f"  ⚠ No group survived --min-reps {min_reps}")
            return 0

    manifest = {
        "manifest_version": "3.0",
        "dataset_name": output_path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": rel_data_root,
        "file_type": file_type,
        "total_records": len(records),
        "records": records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    missing_shape = sum(1 for r in records if "shape" not in r)
    if missing_shape:
        print(
            f"  ⚠ {missing_shape}/{len(records)} records carry no 'shape'. "
            "The queue builder's patch-compatibility filter uses it to check "
            "spatial extent without loading the volume; without it the filter "
            "materialises the whole corpus at queue-build time."
        )

    print(f"  ✓ Saved {len(records)} records to {output_path}")
    return len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate dataset manifests for cluster (v3 JSON)"
    )
    parser.add_argument(
        "--data-base",
        type=str,
        default="databases",
        help="Base path to databases directory (default: databases)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Specific datasets to regenerate (default: all found)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just check what would be regenerated, don't write",
    )
    parser.add_argument(
        "--min-reps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Keep only NEX groups with at least N repetitions, and write to the "
            "'_nexN' sibling manifest (m4raw_train.json -> m4raw_train_nex3.json). "
            "The unfiltered manifest is never overwritten -- it is shared across "
            "cohorts. Use for leave-one-out NEX arms, which need N=3."
        ),
    )
    args = parser.parse_args()

    if args.min_reps is not None and args.min_reps < 2:
        raise SystemExit(f"--min-reps must be at least 2, got {args.min_reps}")

    data_base = Path(args.data_base)

    if not data_base.exists():
        # No fallback search. Substituting a different tree than the one asked
        # for writes a manifest whose ``data_root`` points somewhere the caller
        # never named, and nothing downstream can distinguish that from a
        # correct manifest -- the arm simply trains on the wrong corpus.
        raise SystemExit(
            f"--data-base does not exist: {data_base}\n"
            "Pass the directory holding the dataset trees explicitly, e.g.\n"
            "  --data-base /path/to/databases"
        )

    if args.datasets:
        unknown = sorted(set(args.datasets) - set(DATASET_CONFIGS))
        if unknown:
            raise SystemExit(
                f"unknown dataset(s): {', '.join(unknown)}\n"
                f"known: {', '.join(sorted(DATASET_CONFIGS))}"
            )

    print(f"\n{'=' * 60}")
    print(f"Regenerating v3 JSON manifests from: {data_base}")
    print(f"{'=' * 60}\n")

    requested = set(args.datasets or ())
    total_indexed = 0
    unproduced: list[tuple[str, str]] = []

    for name, cfg in DATASET_CONFIGS.items():
        if args.datasets and name not in args.datasets:
            continue

        data_root = data_base / cfg["subpath"]

        if not data_root.exists():
            print(f"[{name}] Path not found: {data_root}")
            if name in requested:
                unproduced.append((name, f"path not found: {data_root}"))
            continue

        print(f"\n[{name}]")

        if args.dry_run:
            files = list(data_root.glob(cfg["pattern"]))
            destination = Path(cfg["manifest"])
            if args.min_reps is not None:
                # Report the FILTERED count and the '_nexN' destination. Printing
                # the unfiltered pair here would make the dry run agree with a
                # real run it does not describe.
                stems = [{"file_id": p.stem.replace(".nii", "")} for p in files]
                kept, dropped = filter_short_repetition_groups(stems, args.min_reps)
                if dropped:
                    print(f"  min-reps {args.min_reps}: would drop {len(dropped)} short group(s)")
                destination = min_reps_manifest_path(destination, args.min_reps)
                count = len(kept)
            else:
                count = len(files)
            print(f"  Would index {count} files → {destination}")
            total_indexed += count
            if name in requested and not count:
                unproduced.append((name, f"no files matched {cfg['pattern']}"))
        else:
            count = create_manifest(
                data_root,
                data_base,
                cfg["pattern"],
                Path(cfg["manifest"]),
                cfg["file_type"],
                min_reps=args.min_reps,
            )
            total_indexed += count
            if name in requested and count == 0:
                unproduced.append((name, f"no files matched {cfg['pattern']}"))

    verb = "Would index" if args.dry_run else "Indexed"
    print(f"\n{'=' * 60}")
    print(f"Total: {verb} {total_indexed} files")
    print(f"{'=' * 60}")

    if unproduced:
        # A partial run is the dangerous shape: some manifests were written, the
        # total is non-zero, and the process exits 0 -- while the arm's
        # ``validation_index_path`` points at a file that was never created.
        # Naming a dataset on --datasets asserts it should exist, so producing
        # nothing for it is an error. Datasets NOT named are still skipped
        # quietly on purpose: a public checkout legitimately holds one corpus
        # and not the other eleven.
        detail = "\n".join(f"  {name}: {why}" for name, why in unproduced)
        raise SystemExit(
            f"requested dataset(s) produced no manifest:\n{detail}\n\n"
            "Nothing was written for these. Check --data-base points at the "
            "directory holding the dataset trees."
        )

    if total_indexed == 0:
        # Exiting 0 here reads as "the manifests are built". Every dataset
        # directory was absent or empty, so nothing was written at all.
        raise SystemExit(
            f"no files indexed under {data_base} -- no manifest was written.\n"
            "Check --data-base, and that the dataset subpaths exist beneath it:\n"
            + "\n".join(
                f"  {name}: {cfg['subpath']}"
                for name, cfg in DATASET_CONFIGS.items()
                if not args.datasets or name in args.datasets
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
