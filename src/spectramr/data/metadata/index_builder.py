import logging
import pickle
from pathlib import Path

import torchio as tio

from spectramr.config.data_config import DataConfig
from spectramr.data.metadata.path_resolver import PathResolver
from spectramr.data.split_utils import (
    CONFIG_OWNED_SPLIT_TYPES,
    split_index,
    split_index_grouped,
    split_index_three_way_grouped,
    subject_id_from_name,
)

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Builds dataset indices from various sources (NIfTI folders, Manifests).
    """

    @staticmethod
    def build_nifti_index(config: DataConfig, split: str) -> list:
        """
        Build index for NIfTI datasets (including paired translation).
        """
        index = []
        raw_root = config.source.root
        root_str = PathResolver.resolve(raw_root)
        root = Path(root_str)

        split_dir = root / split

        # Check if we have train/val subdirectories
        is_split_dir = False
        if split_dir.exists():
            root = split_dir
            is_split_dir = True

        # Check Layout
        layout = config.source.layout
        if layout == "bids":
            return IndexBuilder._build_bids_index(config, root, split)

        paired = config.dataset_type in (
            "nifti_paired",
            "paired_nifti",
            "paired_mri",
            "contrast_aware_paired",
        )

        if paired:
            return IndexBuilder._build_paired_nifti_index(config, root, split, is_split_dir)

        # CASE 2: Flat directory search (fallback or single folder)
        all_files = sorted(root.glob("*.nii*"))

        if not all_files:
            logger.warning(f"No NIfTI files found in {root} (after checking paired={paired})")
            return []

        # CHECK FOR 2D SLICING VARIANT
        dataset_variant = "3d_full"
        if hasattr(config, "datasets") and config.datasets:
            dataset_variant = config.datasets[0].variant

        # If config explicitly requests 2d_slices on what might be 3D data (NIfTI)
        should_slice = dataset_variant == "2d_slices"

        import nibabel as nib

        final_index = []
        if is_split_dir:
            for nii_path in all_files:
                if should_slice:
                    try:
                        img = nib.load(nii_path)
                        shape = img.shape
                        # 3D volume: (H, W, D) or (H, W, D, C)
                        # We assume slicing happens on the 3rd dimension (index 2)
                        if len(shape) >= 3 and shape[2] > 1:
                            num_slices = shape[2]
                            for i in range(num_slices):
                                final_index.append(
                                    {
                                        "primary_path": str(nii_path),
                                        "file_id": f"{nii_path.stem}_slice{i:03d}",
                                        "slice_index": i,
                                    }
                                )
                        else:
                            # Already 2D or flat 3D
                            final_index.append(
                                {
                                    "primary_path": str(nii_path),
                                    "file_id": nii_path.stem,
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Failed to read header for slicing {nii_path}: {e}")
                        # Fallback to volume
                        final_index.append(
                            {"primary_path": str(nii_path), "file_id": nii_path.stem}
                        )
                else:
                    final_index.append(
                        {
                            "primary_path": str(nii_path),
                            "file_id": nii_path.stem,
                        }
                    )
        else:
            holdout_site = config.split.holdout_site
            if holdout_site:
                # When holdout_site is set, the site filter (below) handles
                # train/val partitioning — skip manual fraction split to avoid
                # excluding holdout files that sort past the val slice boundary.
                files_subset = all_files
            else:
                # Deterministic SUBJECT-level split (never slice- or file-level):
                # a subject's T1w and T2w must land on the same side, and a
                # file whose name carries no subject label is its own group.
                train_files, val_files = split_index_grouped(
                    all_files,
                    config.split.validation_fraction,
                    key=IndexBuilder._split_group_of_path,
                )
                IndexBuilder._log_split_grouping("nifti", all_files, train_files, val_files)
                files_subset = val_files if split == "val" else train_files

            for nii_path in files_subset:
                if should_slice:
                    try:
                        img = nib.load(nii_path)
                        shape = img.shape
                        if len(shape) >= 3 and shape[2] > 1:
                            num_slices = shape[2]
                            for i in range(num_slices):
                                final_index.append(
                                    {
                                        "primary_path": str(nii_path),
                                        "file_id": f"{nii_path.stem}_slice{i:03d}",
                                        "slice_index": i,
                                    }
                                )
                        else:
                            final_index.append(
                                {
                                    "primary_path": str(nii_path),
                                    "file_id": nii_path.stem,
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Failed to read header for slicing {nii_path}: {e}")
                        final_index.append(
                            {"primary_path": str(nii_path), "file_id": nii_path.stem}
                        )
                else:
                    final_index.append(
                        {
                            "primary_path": str(nii_path),
                            "file_id": nii_path.stem,
                        }
                    )
        IndexBuilder._stamp_subject_ids(final_index)

        # CONTRAST FILTERING: Filter by config.pairing.contrasts if specified
        contrasts = config.pairing.contrasts
        target_contrasts = config.pairing.target_contrasts

        if contrasts or target_contrasts:
            contrasts_upper = [c.upper() for c in contrasts] if contrasts else None
            target_contrasts_upper = (
                [c.upper() for c in target_contrasts] if target_contrasts else None
            )
            filtered_index = []

            for record in final_index:
                file_id_upper = record["file_id"].upper()
                matches = False

                # Filter by input contrasts (if specified)
                if contrasts_upper:
                    if any(c in file_id_upper for c in contrasts_upper):
                        matches = True

                # Filter by target contrasts (if specified)
                # Use OR logic: match either input OR target contrasts
                if target_contrasts_upper:
                    if any(c in file_id_upper for c in target_contrasts_upper):
                        matches = True

                if matches:
                    filtered_index.append(record)

            final_index = filtered_index
            if contrasts or target_contrasts:
                logger.info(
                    f"[INDEX] Filtered to {len(final_index)} samples "
                    f"(contrasts={contrasts}, target_contrasts={target_contrasts})"
                )

        # [ROBUSTNESS] SITE FILTERING (Leave-One-Site-Out)
        holdout_site = config.split.holdout_site
        if holdout_site:
            final_index = IndexBuilder._filter_by_site(final_index, config, split)
            logger.info(
                f"[INDEX] Apply LOSO Site Filter (holdout={holdout_site}, split={split}) -> {len(final_index)} samples"
            )

        logger.info(
            f"[INDEX] Built {len(final_index)} samples ({'sliced' if should_slice else 'volumes'}) for {split} from {root}"
        )
        return final_index

    @staticmethod
    def _filter_by_site(index: list, config: DataConfig, split: str) -> list:
        """Filter index based on site ID for Leave-One-Site-Out validation."""
        holdout = config.split.holdout_site
        train_sites = config.split.train_sites

        if not holdout and not train_sites:
            return index

        filtered = []
        for record in index:
            file_id = record.get("file_id", "")

            matches_holdout = False
            if holdout:
                if isinstance(holdout, str):
                    if holdout.lower() in file_id.lower():
                        matches_holdout = True
                elif isinstance(holdout, list):
                    if any(h.lower() in file_id.lower() for h in holdout):
                        matches_holdout = True

            matches_train_sites = False
            if train_sites:
                for s in train_sites:
                    if s.lower() in file_id.lower():
                        matches_train_sites = True
                        break

            keep = False
            if split == "train":
                # Train: Exclude holdout, Include only train_sites if specified
                if holdout and matches_holdout:
                    keep = False
                elif train_sites:
                    keep = matches_train_sites
                else:
                    keep = True
            elif split == "val":
                # Val: Include ONLY holdout if specified
                if holdout:
                    keep = matches_holdout
                elif train_sites:
                    # If no holdout but train_sites specified, val should also match train_sites?
                    # Or val is what's left?
                    # Simpler to assume standard split if no holdout.
                    keep = matches_train_sites
                else:
                    keep = True

            if keep:
                filtered.append(record)

        return filtered

    # ------------------------------------------------------------------
    # Subject grouping for the directory routes (cohort review 2026-09-02)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_group_of_path(path: Path) -> str:
        """Group key for a crawled file: its subject, else the file itself."""
        return subject_id_from_name(str(path)) or str(path)

    @staticmethod
    def _split_group_of_record(record: dict) -> str:
        """Group key for a paired record: its subject, else its ``file_id``."""
        return subject_id_from_name(str(record.get("file_id", ""))) or str(record.get("file_id"))

    @staticmethod
    def _stamp_subject_ids(records: list) -> None:
        """Write ``subject_id`` onto every record whose path names one.

        Records without a label are left alone (``split_leakage`` then falls
        back to file identity, and ``loso_subject`` reports them as unlabeled),
        so the stamp adds evidence and never invents a subject.
        """
        for record in records:
            if record.get("subject_id"):
                continue
            sid = subject_id_from_name(
                str(record.get("primary_path") or record.get("file_id") or "")
            )
            if sid:
                record["subject_id"] = sid

    @staticmethod
    def _log_split_grouping(route: str, items: list, train: list, val: list) -> None:
        """One line saying how the split was grouped, so a leak review can read it."""
        labels = {
            IndexBuilder._split_group_of_path(p)
            if isinstance(p, Path)
            else IndexBuilder._split_group_of_record(p)
            for p in items
        }
        n_labeled = sum(
            1
            for p in items
            if subject_id_from_name(str(p if isinstance(p, Path) else p.get("file_id", "")))
        )
        logger.info(
            "[INDEX] %s route: subject-grouped split of %d file(s) into %d group(s) "
            "(%d carry a sub-* label) -> train=%d val=%d",
            route,
            len(items),
            len(labels),
            n_labeled,
            len(train),
            len(val),
        )

    @staticmethod
    def _build_paired_nifti_index(config, root: Path, split: str, is_split_dir: bool) -> list:
        """_build_paired_nifti_index.

        Args:
            config (Any): Description.
            root (Path): Description.
            split (str): Description.
            is_split_dir (bool): Description.
        Returns:
            list: Description.
        """
        index = []
        # Try to infer source/target dirs from config or defaults
        # Wait, paired datasets configurations are often passed inside TrainingSettings
        # config here might be DataConfig or custom schema.
        # Assume TrainingSettings structure implies these parameters exist in custom paired schema or DataConfig
        # Given DataConfig does not have input_hr_dir/input_lr_dir natively,
        # they might be inside `extra_kwargs` or this might be a custom schema.
        s_raw = (
            config.get("input_lr_dir", None)
            if isinstance(config, dict)
            else getattr(config, "input_lr_dir", None)
        )
        source_dir = Path(PathResolver.resolve(str(s_raw))) if s_raw else root / "source"

        t_raw = (
            config.get("input_hr_dir", None)
            if isinstance(config, dict)
            else getattr(config, "input_hr_dir", None)
        )
        target_dir = Path(PathResolver.resolve(str(t_raw))) if t_raw else root / "target"

        # If specified as relative, resolve against root
        if (
            not source_dir.is_absolute()
            and not (root / source_dir).exists()
            and source_dir.parts[0] != "databases"
        ):
            source_dir = root / source_dir
        if (
            not target_dir.is_absolute()
            and not (root / target_dir).exists()
            and target_dir.parts[0] != "databases"
        ):
            target_dir = root / target_dir

        # FALLBACK LIST: Try common directory names
        CANDIDATES = [
            ("source", "target"),
            ("input", "ground_truth"),
            ("input", "target"),
            ("low_res", "high_res"),
            ("64mT", "3T"),
            ("LF", "HF"),
            ("64MT", "3T"),
            ("LF_data", "HF_data"),
            ("Data/64mT_data", "Data/3T_data"),
            ("slice_64mT", "slice_3T"),
            ("64mt", "3t"),
            ("ulf", "hf"),
            ("ulf_paired_64mt", "ulf_paired_3t"),
            ("64mt_3t", "3t"),
        ]

        def find_case_insensitive_subdir(parent, name):
            """find_case_insensitive_subdir.

            Args:
                parent (Any): Description.
                name (Any): Description.
            Returns:
                Any: Description.
            """
            if (parent / name).exists():
                return parent / name
            for item in parent.iterdir():
                if item.is_dir() and item.name.lower() == name.lower():
                    return item
            return None

        if not source_dir.exists():
            for s_cand, t_cand in CANDIDATES:
                s_path = find_case_insensitive_subdir(root, s_cand)
                t_path = find_case_insensitive_subdir(root, t_cand)

                if s_path and t_path:
                    source_dir = s_path
                    target_dir = t_path
                    break

                # Try with split
                s_path_split = (
                    find_case_insensitive_subdir(root / split, s_cand)
                    if (root / split).exists()
                    else None
                )
                t_path_split = (
                    find_case_insensitive_subdir(root / split, t_cand)
                    if (root / split).exists()
                    else None
                )
                if s_path_split and t_path_split:
                    source_dir = s_path_split
                    target_dir = t_path_split
                    break

        if not source_dir.exists():
            logger.debug(f"Source directory not found in {root}. Falling back to flat search.")
            return []
        else:
            # Match files by name between source and target
            def normalize_filename(fname: str) -> str:
                """Remove file extensions and common suffixes for matching."""
                base = fname.replace(".nii.gz", "").replace(".nii", "")
                for suffix in [
                    "_ulf_registered",
                    "_ulf",
                    "_low_res",
                    "_low",
                    "_64mt",
                    "_64mT",
                ]:
                    if base.endswith(suffix):
                        base = base[: -len(suffix)]
                        break
                return base

            target_map = {}
            for t_path in sorted(target_dir.glob("*.nii*")):
                normalized = normalize_filename(t_path.name)
                target_map[normalized] = t_path

            for s_path in sorted(source_dir.glob("*.nii*")):
                s_normalized = normalize_filename(s_path.name)

                if s_normalized in target_map:
                    t_path = target_map[s_normalized]
                    index.append(
                        {
                            "primary_path": str(s_path),
                            "target_path": str(t_path),
                            "file_id": s_normalized,
                        }
                    )
                else:
                    logger.debug(f"No target found for {s_path.name}")
        IndexBuilder._stamp_subject_ids(index)

        # CONTRAST FILTERING: Filter by config.pairing.contrasts if specified
        contrasts = config.pairing.contrasts
        target_contrasts = config.pairing.target_contrasts

        if contrasts or target_contrasts:
            contrasts_upper = [c.upper() for c in contrasts] if contrasts else None
            target_contrasts_upper = (
                [c.upper() for c in target_contrasts] if target_contrasts else None
            )
            filtered_index = []

            for record in index:
                file_id_upper = record["file_id"].upper()
                matches = False

                # Filter by input contrasts (if specified)
                if contrasts_upper:
                    if any(c in file_id_upper for c in contrasts_upper):
                        matches = True

                # Filter by target contrasts (if specified)
                # Use OR logic: match either input OR target contrasts
                if target_contrasts_upper:
                    if any(c in file_id_upper for c in target_contrasts_upper):
                        matches = True

                if matches:
                    filtered_index.append(record)

            index = filtered_index
            if contrasts or target_contrasts:
                logger.info(
                    f"[INDEX] Filtered to {len(index)} paired samples "
                    f"(contrasts={contrasts}, target_contrasts={target_contrasts})"
                )

        # [ROBUSTNESS] SITE FILTERING (Leave-One-Site-Out)
        holdout_site = config.split.holdout_site
        if holdout_site:
            index = IndexBuilder._filter_by_site(index, config, split)
            logger.info(
                f"[INDEX] Apply LOSO Site Filter (holdout={holdout_site}, split={split}) -> {len(index)} samples"
            )

        # MANUAL SPLIT LOGIC
        # If we didn't search in a split-specific directory, we must split the data manually
        # [ROBUSTNESS] If holdout_site is set, skip manual split (split is defined by site)
        if not is_split_dir and not holdout_site:
            # Deterministic sort
            index.sort(key=lambda x: x["file_id"])

            # Three-way when the arm opts in; otherwise the two-way split
            # (``split_index_three_way`` delegates when ``test_split <= 0``).
            # Grouped by SUBJECT: a paired record is one (source, target) file
            # pair, and one subject contributes one per contrast / session.
            test_fraction = (
                float(getattr(config, "test_split", 0.0) or 0.0)
                if getattr(config, "enable_test_split", False)
                else 0.0
            )
            train_index, val_index, test_index = split_index_three_way_grouped(
                index,
                config.split.validation_fraction,
                test_fraction,
                key=IndexBuilder._split_group_of_record,
            )
            IndexBuilder._log_split_grouping("paired", index, train_index, val_index)
            # Explicit mapping, not ``val if split == "val" else train``. That
            # shape returned the TRAINING index for ``split="test"`` -- the same
            # silent-else defect this module shares with ``pipelines/make.py``.
            by_split = {"train": train_index, "val": val_index, "test": test_index}
            if split not in by_split:
                raise ValueError(f"Unknown split {split!r}. Choose one of {sorted(by_split)}.")
            if split == "test" and not test_index:
                raise ValueError(
                    "This arm declares no held-out test split, so there is no test "
                    "index to build. Set `data.enable_test_split: true` (with a "
                    "non-zero `data.test_split`) to create one. Refusing to "
                    "substitute the validation or training index."
                )
            index = by_split[split]

        if index:
            logger.info(f"[INDEX] Built {len(index)} paired samples for {split} from {root}")
        return index

    @staticmethod
    def _build_bids_index(config, root: Path, split: str) -> list:
        """
        Build index for BIDS-formatted datasets.

        Crawls: root/sub-*/ses-*/anat/*.nii.gz
        Or:     root/sub-*/anat/*.nii.gz

        Filters by split if 'participants.tsv' exists, otherwise uses validation_split config.
        """
        index = []
        logger.info(f"[INDEX] Scanning BIDS layout at {root}")

        # 1. Find all subjects
        subjects = sorted(root.glob("sub-*"))
        if not subjects:
            logger.warning(f"No 'sub-*' directories found in {root} for BIDS layout")
            return []

        # 2. Split handling via participants.tsv (Optional but recommended for BIDS)
        participants_tsv = root / "participants.tsv"
        allowed_subjects = None

        if participants_tsv.exists():
            # Very basic TSV parser (assuming 'participant_id' and 'group' or 'split' columns)
            try:
                with open(participants_tsv) as f:
                    header = f.readline().strip().split("\t")
                    lines = f.readlines()

                id_idx = -1
                group_idx = -1
                for i, col in enumerate(header):
                    if col == "participant_id":
                        id_idx = i
                    elif col.lower() in ("group", "split", "subset"):
                        group_idx = i

                if id_idx != -1 and group_idx != -1:
                    allowed_subjects = set()
                    target_group = (
                        "train" if split == "train" else "val"
                    )  # map 'val' to 'validation' if needed
                    # Handle common BIDS split names
                    map_split = {
                        "train": ["train", "training"],
                        "val": ["val", "validation", "test", "testing"],
                    }

                    for line in lines:
                        parts = line.strip().split("\t")
                        if len(parts) > max(id_idx, group_idx):
                            sub_id = parts[id_idx]
                            group = parts[group_idx].lower()
                            if group in map_split.get(split, [split]):
                                allowed_subjects.add(sub_id)
                    logger.info(
                        f"[INDEX] Found participants.tsv. Filtered to {len(allowed_subjects)} subjects for {split}"
                    )
            except Exception as e:
                logger.warning(f"[INDEX] Failed to parse participants.tsv: {e}")

        # 3. Crawl subjects
        files = []
        for sub_dir in subjects:
            if allowed_subjects is not None and sub_dir.name not in allowed_subjects:
                continue

            # Handle sessions (ses-*) or direct anat/
            # Look for 'anat' folder
            # Case 1: sub-X/anat/
            anat_dirs = [sub_dir / "anat"]
            # Case 2: sub-X/ses-Y/anat/
            ses_dirs = list(sub_dir.glob("ses-*"))
            for ses in ses_dirs:
                anat_dirs.append(ses / "anat")

            for anat in anat_dirs:
                if anat.exists():
                    # Find NIfTI files
                    # Prioritize common suffixes
                    for suffix in ["_T1w", "_T2w", "_FLAIR", ""]:
                        for ext in [".nii.gz", ".nii"]:
                            # Glob all files and filter by suffix
                            # This is slightly inefficient but flexible
                            f_candidates = sorted(list(anat.glob(f"*{suffix}{ext}")))
                            for p in f_candidates:
                                # Avoid duplicating if we matched logic loosely
                                files.append(p)

        # Remove duplicates
        files = sorted(list(set(files)))

        # If no participants.tsv, do manual split -- by SUBJECT (the sub-*
        # directory), never by file: a BIDS subject has one file per contrast
        # and session, and a file-level split put the same anatomy on both
        # sides (cohort review 2026-09-02).
        if allowed_subjects is None:
            train_files, val_files = split_index_grouped(
                files,
                config.split.validation_fraction,
                key=IndexBuilder._split_group_of_path,
            )
            IndexBuilder._log_split_grouping("bids", files, train_files, val_files)
            files = val_files if split == "val" else train_files

        for p in files:
            index.append(
                {
                    "primary_path": str(p),
                    "file_id": p.name.replace(".nii.gz", "").replace(".nii", ""),
                }
            )
        IndexBuilder._stamp_subject_ids(index)

        # CONTRAST FILTERING: Filter by config.pairing.contrasts if specified
        contrasts = config.pairing.contrasts
        target_contrasts = config.pairing.target_contrasts

        if contrasts or target_contrasts:
            contrasts_upper = [c.upper() for c in contrasts] if contrasts else None
            target_contrasts_upper = (
                [c.upper() for c in target_contrasts] if target_contrasts else None
            )
            filtered_index = []

            for record in index:
                file_id_upper = record["file_id"].upper()
                matches = False

                # Filter by input contrasts (if specified)
                if contrasts_upper:
                    if any(c in file_id_upper for c in contrasts_upper):
                        matches = True

                # Filter by target contrasts (if specified)
                # Use OR logic: match either input OR target contrasts
                if target_contrasts_upper:
                    if any(c in file_id_upper for c in target_contrasts_upper):
                        matches = True

                if matches:
                    filtered_index.append(record)

            index = filtered_index
            if contrasts or target_contrasts:
                logger.info(
                    f"[INDEX] Filtered to {len(index)} BIDS samples "
                    f"(contrasts={contrasts}, target_contrasts={target_contrasts})"
                )

        logger.info(f"[INDEX] Built {len(index)} BIDS samples for {split}")
        return index

    @staticmethod
    def load_from_manifest_roles(
        config,
        train_transform,
        val_transform,
    ) -> tuple[list, list]:
        """Load subjects from multiple manifest pickle files with role-based keys."""
        roles = config.manifest_roles

        # Resolve data root
        resolved_root_str = PathResolver.resolve(config.source.root) if config.source.root else ""
        resolved_root = Path(resolved_root_str) if resolved_root_str else None

        manifest_dir = (
            resolved_root.parent / "manifests" if resolved_root else Path("data/manifests")
        )
        manifest_dir = Path(PathResolver.resolve(str(manifest_dir)))

        # Collect all manifest-to-key mappings
        manifest_configs = []

        # Process inputs, targets, auxiliary
        for role in ["inputs", "targets", "auxiliary"]:
            items = getattr(roles, role, []) or []
            for item in items:
                if isinstance(item, dict) and item.get("manifest"):
                    manifest_configs.append(
                        {
                            "manifest": item["manifest"],
                            "key": item.get(
                                "key", role[:-1] if role != "inputs" else "input"
                            ),  # crude singularization
                            "role": (
                                role[:-1] if role != "inputs" else "input"
                            ),  # crude singularization
                        }
                    )
        # Fix crude singularization just in case
        for mc in manifest_configs:
            if mc["role"] == "target":
                pass
            elif mc["role"] == "auxiliar":  # codespell:ignore auxiliar
                mc["role"] = "auxiliary"  # fix typo from slice above

        if not manifest_configs:
            raise ValueError("No valid manifests found in manifest_roles configuration")

        samples_by_filename = {}

        for mc in manifest_configs:
            manifest_path = manifest_dir / mc["manifest"]
            if not manifest_path.exists():
                manifest_path = Path(mc["manifest"])

            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Manifest file not found: {mc['manifest']}\nSearched in: {manifest_dir}"
                )

            # Security note: pickle is used here as per original design.
            # In future, migrate to JSON or SafeTensors for manifests.
            with open(manifest_path, "rb") as f:
                samples = pickle.load(f)

            logger.info(
                f"[MANIFEST] Loaded {len(samples)} samples from {manifest_path.name} as '{mc['key']}'"
            )

            for sample in samples:
                filename = sample.get("filename", "")
                base_id = filename.replace(".h5", "").replace(".nii.gz", "").replace(".nii", "")
                for suffix in [
                    "_compressed",
                    "_reconstructed",
                    "_gt",
                    "_rss",
                    "_normalized",
                ]:
                    base_id = base_id.replace(suffix, "")

                if base_id not in samples_by_filename:
                    samples_by_filename[base_id] = {}

                samples_by_filename[base_id][mc["key"]] = sample

        # Build TorchIO subjects
        all_subjects = []
        skipped_subjects: list[tuple[str, str]] = []
        subjects_without_target: list[str] = []

        for file_id, role_samples in samples_by_filename.items():
            subject_dict = {"file_id": file_id}

            for key, sample in role_samples.items():
                raw_path = sample["path"]
                path_str = PathResolver.resolve(raw_path)
                path = Path(path_str)

                fmt = sample.get("format", "h5")

                if fmt == "h5":
                    # Route the h5py read through the canonical data-layer
                    # strategy (CLAUDE.md pitfall #11). The static helper
                    # preserves the key-precedence + complex/2D handling
                    # this site has always had.
                    from spectramr.data.io_strategies import FastMRIH5Strategy

                    try:
                        subject_dict[key] = tio.ScalarImage(
                            tensor=FastMRIH5Strategy.load_torchio_tensor(path)
                        )
                    except Exception as e:
                        logger.error(f"Failed to load H5 file {path}: {e}")
                        continue
                elif fmt in ("nifti", "nii"):
                    subject_dict[key] = tio.ScalarImage(path)
                elif fmt == "pt" or fmt == "png":
                    subject_dict[f"{key}_path"] = str(path)
                else:
                    raise ValueError(
                        f"load_from_manifest_roles: unknown format '{fmt}' for key "
                        f"'{key}' in manifest record (file_id={file_id!r}). "
                        f"Supported formats: 'h5', 'nifti', 'nii', 'pt', 'png'."
                    )

                cs_path = sample.get("coil_sensitivity_path")
                if cs_path:
                    subject_dict[f"{key}_sensitivity_path"] = PathResolver.resolve(cs_path)

            # Decide from what actually LOADED (``subject_dict``), not from what
            # the manifest promised (``role_samples``). The H5 branch above
            # ``continue``s past a failed read, so a manifest naming an input
            # that fails to open still satisfied a role_samples-derived guard --
            # and the Subject was appended with no input image at all. The
            # failure then surfaced hours later, deep in the training loop,
            # naming neither the file nor the role.
            loaded = set(subject_dict)
            has_input = any(k.startswith("input") or k == "kspace" for k in loaded)
            has_target = any(k.startswith("target") or k == "gt" for k in loaded)

            if has_input:
                subject = tio.Subject(**subject_dict)
                all_subjects.append(subject)
                if not has_target:
                    subjects_without_target.append(file_id)
            else:
                promised_input = any(k.startswith("input") or k == "kspace" for k in role_samples)
                skipped_subjects.append(
                    (
                        file_id,
                        "input failed to load" if promised_input else "no input role",
                    )
                )

        if skipped_subjects:
            # A counted census, not a per-file line: a broken manifest can drop
            # thousands, and a WARNING per file scrolls the real error away.
            preview = ", ".join(f"{fid} ({why})" for fid, why in skipped_subjects[:5])
            logger.warning(
                f"[MANIFEST] Dropped {len(skipped_subjects)} of "
                f"{len(samples_by_filename)} subject(s) with no usable input "
                f"image. First {min(5, len(skipped_subjects))}: {preview}"
                + (" ..." if len(skipped_subjects) > 5 else "")
            )
        if subjects_without_target:
            logger.warning(
                f"[MANIFEST] {len(subjects_without_target)} of {len(all_subjects)} "
                "subject(s) have an input but no target; they cannot supervise a "
                f"paired objective. First: {subjects_without_target[:5]!r}"
            )

        if not all_subjects:
            raise ValueError(
                "No valid subjects could be loaded from manifest_roles: "
                f"{len(samples_by_filename)} manifest record(s) were read and "
                f"{len(skipped_subjects)} were dropped for having no usable "
                "input image. Check the manifest paths resolve and the files "
                "open (the per-file load errors are logged above)."
            )

        # CONTRAST FILTERING: Filter by config.pairing.contrasts if specified
        contrasts = config.pairing.contrasts
        target_contrasts = config.pairing.target_contrasts

        if contrasts or target_contrasts:
            contrasts_upper = [c.upper() for c in contrasts] if contrasts else None
            target_contrasts_upper = (
                [c.upper() for c in target_contrasts] if target_contrasts else None
            )
            filtered_subjects = []

            for subject in all_subjects:
                file_id_upper = subject.get("file_id", "").upper()
                matches = False

                # Filter by input contrasts (if specified)
                if contrasts_upper:
                    if any(c in file_id_upper for c in contrasts_upper):
                        matches = True

                # Filter by target contrasts (if specified)
                # Use OR logic: match either input OR target contrasts
                if target_contrasts_upper:
                    if any(c in file_id_upper for c in target_contrasts_upper):
                        matches = True

                if matches:
                    filtered_subjects.append(subject)

            all_subjects = filtered_subjects
            if contrasts or target_contrasts:
                logger.info(
                    f"[MANIFEST] Filtered to {len(all_subjects)} subjects "
                    f"(contrasts={contrasts}, target_contrasts={target_contrasts})"
                )

        val_split = (
            config.get("validation_split", 0.1)
            if isinstance(config, dict)
            else config.split.validation_fraction
        )
        train_subjects, val_subjects = split_index(all_subjects, val_split)

        logger.info(
            f"[MANIFEST] Created {len(train_subjects)} train, {len(val_subjects)} val subjects"
        )
        return train_subjects, val_subjects

    # ------------------------------------------------------------------
    # v4 Paired Manifest Loading
    # ------------------------------------------------------------------

    @classmethod
    def load_paired_bids_manifest(
        cls,
        manifest_path: "Path",
        split: str | None,
        config,
    ) -> list[dict]:
        """Load a v4 paired JSON manifest and return records for *split*.

        Applies the following filters in order:

        1. **split_hint** — keeps only records whose ``split_hint`` matches
           *split* (``"train"`` / ``"val"``).
        2. **pairing_status** — when ``config.pairing.allow_unpaired`` is ``False``
           (default), only ``"paired"`` records are returned.
        3. **contrasts filter** — ``config.pairing.contrasts`` narrows the selection
           to specific MRI contrasts.
        3.5. **hf_resolution filter** — when ``config.pairing.hf_resolution`` is set
           (``'highres'``, ``'lowres'``, or ``'unknown'``), only records
           matching that HF resolution variant are kept.
        4. **bidirectional swap** — when
           ``config.pairing.bidirectional_mode == "hf_to_ulf"`` the
           ``primary_path`` and ``target_path`` fields are swapped so that
           HF becomes the network input.

        Parameters
        ----------
        manifest_path :
            Path to a v4 ``.json`` manifest.
        split :
            ``"train"`` or ``"val"``.
        config :
            A :class:`~spectramr.config.schemas.data.DataConfigSchema` or
            compatible object exposing a ``pairing`` block -- i.e. one
            carrying a :class:`~spectramr.config.schemas.data.DataPairingConfigSchema`.
            Build the stand-in from that schema rather than hand-rolling a
            namespace: the reads below are plain attribute access, so a
            stand-in missing a field raises instead of silently defaulting.

        Returns
        -------
        list[dict]
            Index records suitable for
            :class:`~spectramr.data.datasets.contrast_aware.ContrastAwarePairedDataset`.

        Raises
        ------
        FileNotFoundError
            If *manifest_path* does not exist.
        ValueError
            If the manifest format version is not ``"4.0"``.
        """
        import json as _json

        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"v4 paired manifest not found: {manifest_path}\n"
                "Run scripts/preprocessing/generate_paired_manifest.py to generate it."
            )

        with open(manifest_path) as fh:
            manifest = _json.load(fh)

        version = manifest.get("manifest_version", "unknown")
        if version not in ("4.0", "5.0"):
            raise ValueError(
                f"Expected manifest_version='4.0' or '5.0', got '{version}'. "
                "Re-generate the manifest with generate_paired_manifest.py."
            )

        # v5 uses 'files' key, v4 uses 'records'
        records: list[dict] = manifest.get("records", []) or manifest.get("files", [])
        logger.info(
            f"[v{version} MANIFEST] Loaded {len(records)} records from {manifest_path.name}"
        )

        # --- 0. Normalize legacy field names ---
        # Some manifests declare version 4.0 but use pre-v4 field names
        # (e.g. ``input_path`` instead of ``primary_path``, missing
        # ``split_hint`` / ``pairing_status``).  v5 manifests use
        # ``split`` and ``pairing`` which also need normalizing.
        _needs_normalize = records and "primary_path" not in records[0]
        if _needs_normalize:
            logger.info(f"[v{version} MANIFEST] Normalizing field names to canonical schema")
            # Infer train/val split from manifest metadata if available
            meta = manifest.get("metadata", {})
            train_count = meta.get("train", 0)

            for idx, r in enumerate(records):
                # Field rename: input_path → primary_path
                if "input_path" in r and "primary_path" not in r:
                    r["primary_path"] = r.pop("input_path")

                # v5 field rename: split → split_hint
                if "split" in r and "split_hint" not in r:
                    r["split_hint"] = r.pop("split")

                # v5 field rename: pairing → pairing_status
                if "pairing" in r and "pairing_status" not in r:
                    r["pairing_status"] = r.pop("pairing")

                # Infer pairing_status from presence of target_path
                if "pairing_status" not in r:
                    r["pairing_status"] = "paired" if r.get("target_path") else "unpaired_ulf"

                # Infer split_hint from positional order if metadata present
                if "split_hint" not in r:
                    if train_count > 0:
                        r["split_hint"] = "train" if idx < train_count else "val"
                    else:
                        # Default: 80/20 split
                        r["split_hint"] = "train" if idx < int(len(records) * 0.8) else "val"

                # Infer contrast from filename if missing
                if "contrast" not in r:
                    fname = r.get("filename", "")
                    for c in ("T1w", "T2w", "FLAIR", "ADC", "DWI", "SWI"):
                        if c in fname:
                            r["contrast"] = c
                            break

        # --- 1. Split filter ---
        # ``split=None`` is pool mode: every record survives the filters below and
        # the caller partitions (``data.metadata.paired_partition``). A config-owned
        # partition type never reads ``split_hint``, so it must not reach here with
        # a split name -- that would be a second partition owner.
        if split is None:
            logger.info(f"[v{version} MANIFEST] pool mode: {len(records)} records, no split filter")
        elif config.split.type in CONFIG_OWNED_SPLIT_TYPES:
            raise ValueError(
                f"split.type={config.split.type!r} owns the paired partition; call "
                "ManifestLoader.load_paired_nifti_splits, not load_paired_bids_manifest"
                f"(split={split!r}). Two readers would apply two partitions."
            )
        else:
            records = [r for r in records if r.get("split_hint") == split]
            logger.info(f"[v4 MANIFEST] After split_hint='{split}': {len(records)} records")

        # --- 2. Pairing status filter ---
        # Target-less ULF records are admitted per split: train only under
        # unpaired_training, every other split (val/test/infer) under allow_unpaired.
        # Pool mode keeps them all; the partition decides per split.
        if split is not None:
            keep_unpaired = (
                config.pairing.unpaired_training
                if split == "train"
                else config.pairing.allow_unpaired
            )
            if not keep_unpaired:
                records = [r for r in records if r.get("pairing_status") == "paired"]
                logger.info(
                    f"[v4 MANIFEST] After allow_unpaired=False: {len(records)} paired records"
                )
            else:
                n_unpaired = sum(1 for r in records if r.get("pairing_status") == "unpaired_ulf")
                logger.info(
                    f"[v4 MANIFEST] split={split!r} admits {n_unpaired} unpaired ULF records"
                )

        # --- 3. Contrast filter ---
        contrasts = config.pairing.contrasts or []
        if contrasts:
            contrasts_upper = {c.upper() for c in contrasts}
            records = [r for r in records if r.get("contrast", "").upper() in contrasts_upper]
            logger.info(f"[v4 MANIFEST] After contrast filter {contrasts}: {len(records)} records")

        # --- 3.5. HF resolution filter ---
        hf_resolution = config.pairing.hf_resolution
        if hf_resolution is not None:
            before = len(records)
            records = [
                r
                for r in records
                if (
                    # Unpaired ULF records have hf_resolution=None — always pass through
                    r.get("hf_resolution") is None or r.get("hf_resolution") == hf_resolution
                )
            ]
            logger.info(
                f"[v4 MANIFEST] After hf_resolution='{hf_resolution}': "
                f"{len(records)} records (dropped {before - len(records)})"
            )

        # --- 4. Bidirectional filter (no path swap here) ---
        # NOTE: Path swapping is handled at the dataset level in
        # ContrastAwarePairedDataset.__getitem__() via builder.build(swap=True).
        # The dataset swap also correctly reassigns contrast normalization
        # configs (input_contrast ↔ target_contrast).  Swapping paths here
        # would cause a double-swap bug (paths get swapped twice → back to
        # original direction).
        bidirectional_mode = config.pairing.bidirectional_mode
        if bidirectional_mode == "hf_to_ulf":
            before = len(records)
            # Drop unpaired records — they have no target to become input
            records = [r for r in records if r.get("target_path") is not None]
            logger.info(
                f"[v4 MANIFEST] HF→ULF mode: {len(records)} paired records "
                f"(dropped {before - len(records)} unpaired)"
            )

        if not records:
            logger.info(
                f"[v4 MANIFEST] No records remaining for split='{split}' "
                "after all filters. Check contrasts/allow_unpaired settings."
            )

        data_root = manifest.get("data_root", "")
        return cls._resolve_manifest_paths(records, data_root=data_root)

    @staticmethod
    def _resolve_manifest_paths(records: list[dict], *, data_root: str = "") -> list[dict]:
        """Resolve relative paths in manifest records to absolute paths.

        Uses :class:`~spectramr.data.metadata.path_resolver.PathResolver` which
        understands both local and cluster-relative paths.

        Parameters
        ----------
        records :
            Raw records from the v4 manifest (may contain relative paths).
        data_root :
            The ``data_root`` field from the manifest.  Relative record
            paths are joined with *data_root* **before** being handed to
            :meth:`PathResolver.resolve`, so that the resolver sees
            ``{data_root}/{record_path}`` instead of bare
            ``{record_path}``.

        Returns
        -------
        list[dict]
            Records with ``primary_path`` and ``target_path`` resolved to
            absolute strings.
        """

        def _join(raw: str) -> str:
            if not raw:
                return raw
            p = Path(raw)
            # Only prepend data_root to relative paths that aren't
            # already under data_root (avoid double-prepend).
            if data_root and not p.is_absolute() and not raw.startswith(data_root):
                raw = str(Path(data_root) / raw)
            return PathResolver.resolve(raw)

        resolved: list[dict] = []
        for r in records:
            rec = dict(r)
            rec["primary_path"] = _join(rec.get("primary_path", ""))
            rec["target_path"] = _join(rec["target_path"]) if rec.get("target_path") else None
            resolved.append(rec)
        return resolved
