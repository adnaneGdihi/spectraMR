"""Universal MRI Dataset.

The One Dataset to Rule Them All.

Uses IO Strategies for format-agnostic loading and wraps into tio.Subject.
Features:
- Handles FastMRI (H5), NIfTI, DICOM, and Synthetic data
- Integrated coil processing (RSS, Flatten, None) and k-space normalization
- Supports multi-contrast learning and paired translations (e.g. ULF -> HF)
- Intelligent path resolution for local/cluster portability

.. mermaid::

    classDiagram
        class UniversalMRIDataset {
            +__getitem__(idx)
            +dry_iter()
        }
        class SubjectBuilder {
            <<interface>>
            +build(record)
        }
        class IOStrategy {
            <<interface>>
            +load(path)
        }

        UniversalMRIDataset --> SubjectBuilder : uses
        SubjectBuilder --> IOStrategy : uses
        UniversalMRIDataset --> torch_Dataset : inherits
"""

import json
import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

import torch
import torchio as tio

from spectramr.data.builders.torchio_subject_builder import FastMRISubjectBuilder
from spectramr.data.io_strategies import IOStrategyFactory
from spectramr.data.metadata.path_resolver import PathResolver


def resolve_contrast_idx(
    record: dict[str, Any], contrast_map: dict[str, int] | None
) -> "torch.Tensor | None":
    """Map a record's ``contrast`` string to a long ``contrast_idx`` tensor.

    Used by the ``nifti_paired`` path so FiLM / contrast-guidance conditioning
    actually fires: ``GeoMambaUNet.forward`` and the latent-diffusion generator
    default to contrast 0 when ``contrast_idx`` is absent — a silent single-token
    collapse that turned every contrast-conditioned ULF→HF arm into a
    contrast-agnostic one (pitfall #16 facade).

    Args:
        record: One dataset record; must carry ``contrast`` when ``contrast_map``
            is configured.
        contrast_map: Per-arm ``data.multi_contrast.contrast_map`` (contrast name →
            embedding index). ``None`` or empty turns the feature off.

    Returns:
        A scalar ``torch.long`` tensor, or ``None`` when no map is configured.

    Raises:
        ValueError: ``contrast_map`` is set but the record has no ``contrast``, or
            the contrast is not in the map. Never silently falls back (pitfall #9).
    """
    if not contrast_map:
        return None
    contrast = record.get("contrast")
    if contrast is None:
        raise ValueError(
            "contrast_map is configured but record has no 'contrast' field "
            f"(record keys: {sorted(record)}). A contrast-conditioned arm needs "
            "the manifest to carry per-record contrast tags."
        )
    if contrast not in contrast_map:
        raise ValueError(
            f"contrast {contrast!r} is not in contrast_map "
            f"{sorted(contrast_map)}. Add it to data.multi_contrast.contrast_map "
            "or exclude it via data.contrasts — refusing to silently stamp "
            "contrast 0 (pitfall #9)."
        )
    return torch.tensor(int(contrast_map[contrast]), dtype=torch.long)


def resolve_field_strength(value: "float | None") -> "torch.Tensor | None":
    """Coerce a config field strength (Tesla) into a scalar float tensor, or None.

    Used by the ``nifti_paired`` path so field-conditioned ULF→HF strategies
    fire instead of raising on a missing ``field_strength_target`` key.
    ``ulf_dps``, ``monotone_field``, ``field_conditioned_inr``,
    ``generative_refiner`` and ``ulf_redegrad_tta`` read
    ``batch['field_strength_target']`` directly (their own errors say "set
    data.expose_field_strength_target"); the ``mrixfields`` dataset stamps it
    per-record, but ``nifti_paired`` never did — so those methods were unrunnable
    on the paired-ULF data (pitfall #16, in the "blocks the whole cohort" sense).

    A fixed 64mT→3T paired translation has a CONSTANT target field, so the value
    comes from config (``data.mrixfields_target_field``), not the manifest.

    Args:
        value: The field strength in Tesla, or ``None`` to disable the feature.

    Returns:
        A 0-dim ``torch.float32`` tensor, or ``None`` when ``value`` is ``None``.

    Raises:
        ValueError: ``value`` is non-positive — a field strength cannot be 0 or
            negative Tesla; refuse to fabricate one (pitfall #9).
    """
    if value is None:
        return None
    v = float(value)
    if v <= 0.0:
        raise ValueError(
            f"field strength must be a positive Tesla value; got {value!r}. "
            "Set data.mrixfields_target_field (e.g. 3.0) — refusing to stamp a "
            "non-physical field (pitfall #9)."
        )
    return torch.tensor(v, dtype=torch.float32)


class UniversalMRIDataset(torch.utils.data.Dataset):
    """
    Universal Dataset for MRI tasks.

    Logic:
    1. Receives an 'index': list of dicts, each describing ONE sample.
       e.g. {'path': '/data/file1.h5', 'target_path': '/data/file1_hf.h5', ...}
    2. Uses 'io_strategy' to load the primary file.
    3. Uses 'paired_io_strategy' (optional) to load a paired target (Exp 32).
    4. Wraps result in torchio.Subject.
    """

    def __init__(
        self,
        index: list[dict[str, Any]],
        io_strategy: str = "fastmri_h5",
        transform: Callable | None = None,
        paired_io_strategy: str | None = None,
        load_sensitivity: bool = True,
        coil_processing_mode: str = "none",
        num_virtual_coils: int = 4,
        svd_calibration_lines: int | None = None,
        coil_processing=None,
        normalize_kspace: bool = False,
        kspace_percentile: float = 0.99,
        log_scaling: bool = False,
        contrast_map: dict[str, int] | None = None,
        target_field_strength: float | None = None,
        image_domain_crop_or_pad: tuple[int, int] | None = None,
    ):
        """
        Initialize Universal MRI Dataset.

        Args:
            index: List of records with paths and metadata.
                   Each record should have at minimum {'path': '...'}.
                   Optional keys: 'target_path', 'sensitivity_path'
            io_strategy: Primary IO strategy name (e.g., 'fastmri_h5', 'nifti')
            transform: TorchIO transform pipeline
            paired_io_strategy: IO strategy for target loading (translation tasks)
            load_sensitivity: Whether to attempt loading sensitivity maps
            coil_processing_mode: How to handle multi-coil k-space:
                - "none": Keep original complex multi-coil data
                - "flatten": Convert (Coils, H, W, D) complex to (2*Coils, H, W, D) real
                - "rss": RSS coil combination to (2, H, W, D) real
                - "svd": Compress into virtual coils
            num_virtual_coils: Number of virtual coils when mode="svd"
            contrast_map: Optional contrast-name → embedding-index map. When set
                (the per-arm ``data.multi_contrast.contrast_map``), each sample is
                stamped with a ``contrast_idx`` long tensor so FiLM / contrast
                guidance fires; an unmapped contrast raises (pitfall #9).
            target_field_strength: Optional target field (Tesla) for the paired
                ULF→HF translation. When set (threaded from
                ``data.mrixfields_target_field`` gated on
                ``data.expose_field_strength_target``), each sample is stamped with
                a ``field_strength_target`` scalar tensor so field-conditioned
                restoration strategies (ulf_dps, monotone_field,
                field_conditioned_inr, generative_refiner, ulf_redegrad_tta) run on
                the paired-ULF data instead of raising. A non-positive value
                raises (pitfall #9).
            image_domain_crop_or_pad: Optional ``(height, width)`` the k-space is
                resized to through its image before coil processing
                (``data.processing.image_domain_crop_or_pad``).
        """
        self.index = index
        self.io = IOStrategyFactory.get(io_strategy)
        self.target_io = IOStrategyFactory.get(paired_io_strategy) if paired_io_strategy else None
        self.transform = transform
        self.load_sensitivity = load_sensitivity
        self.contrast_map = contrast_map or None
        # Validated once at construction (raises on a non-physical value) so the
        # per-sample stamp is a cheap constant lookup.
        self.target_field_strength = resolve_field_strength(target_field_strength)
        self.normalize_kspace = normalize_kspace
        self.kspace_percentile = kspace_percentile
        self.log_scaling = log_scaling
        # K-space normalization is applied by KSpaceNormalizationTransform (built
        # by TorchIOTransformBuilder for the same data.normalize_kspace flag), not
        # by the subject builder -- doing both gave a double percentile divide and
        # a double log1p. Asking for normalization with no transform to apply it
        # would silently serve raw k-space, so fail loud instead (pitfall #9).
        # -> docs/kspace_normalization_ssot.rst
        if normalize_kspace and transform is None:
            raise ValueError(
                "[UniversalMRIDataset] normalize_kspace=True but no transform was "
                "supplied. K-space normalization is performed by "
                "KSpaceNormalizationTransform, not by the dataset -- the dataset "
                "matches and serves. Pass the built transform pipeline, or set "
                "normalize_kspace=False."
            )

        # Initialize FastMRI subject builder with coil processing
        self.subject_builder = FastMRISubjectBuilder(
            primary_io=self.io,
            target_io=self.target_io,
            sensitivity_io=self.io if load_sensitivity else None,
            coil_processing_mode=coil_processing_mode,
            num_virtual_coils=num_virtual_coils,
            svd_calibration_lines=svd_calibration_lines,
            coil_processing=coil_processing,
            image_domain_crop_or_pad=image_domain_crop_or_pad,
        )

    def __len__(self) -> int:
        """__len__.

        Returns:
            int: Description.
        """
        return len(self.index)

    def __getitem__(self, idx: int) -> tio.Subject:
        """Load and return a TorchIO Subject.

        Delegates Subject creation to FastMRISubjectBuilder (Phase T4),
        eliminating 100+ lines of duplicated logic.

        Args:
            idx: Sample index

        Returns:
            TorchIO Subject with input, target, k-space, sensitivity
        """
        record = self.index[idx]

        # Use FastMRISubjectBuilder to handle all complex logic
        subject = self.subject_builder.build(record)
        # Every sample is stamped so default collation keeps the key in any batch
        # order; the loop refuses a batch holding a target-less sample.
        subject["has_target"] = torch.tensor(record.get("pairing_status") != "unpaired_ulf")

        # Add metadata
        subject["path"] = str(record["primary_path"])
        subject["file_id"] = record.get("file_id", Path(record["primary_path"]).stem)
        # Subject identity for subject-level consumers (held-out conformal calibration).
        if record.get("subject_id") is not None:
            subject["subject_id"] = str(record["subject_id"])

        # Stamp the contrast index so downstream FiLM / contrast-guidance fires
        # (raises on an unmapped contrast — never silently stamps 0). Absent map
        # → contrast-agnostic arm, no key added.
        contrast_idx = resolve_contrast_idx(record, self.contrast_map)
        if contrast_idx is not None:
            subject["contrast_idx"] = contrast_idx
            # Also emit the mrixfields key name so contrast-conditioned strategies
            # written against that dataset (field_cold_diffusion, generative_refiner)
            # fire on the paired-ULF port too — they read ``batch['contrast_id']``
            # directly, while m4raw / slice / geomamba read ``contrast_idx``. Same
            # value under both names; geomamba reads contrast_id first (identical),
            # so no regression for existing FiLM arms.
            subject["contrast_id"] = contrast_idx.clone()

        # Stamp the constant target field (Tesla) for the paired ULF→HF task so
        # field-conditioned restoration strategies fire instead of raising on a
        # missing key. Absent config → field-agnostic arm, no key added. Fresh
        # tensor per sample (the stored value is validated but immutable).
        if self.target_field_strength is not None:
            subject["field_strength_target"] = self.target_field_strength.clone()

        # Apply user transforms
        if self.transform:
            subject = self.transform(subject)

        return subject

    def dry_iter(self) -> list[tio.Subject]:
        """Return lightweight Subject shells for TorchIO Queue compatibility.

        TorchIO Queue calls ``dry_iter()`` to compute ``iterations_per_epoch``
        without loading voxel data.  It requires a ``Sequence[Subject]``
        (indexable), not a Generator, and only inspects metadata attributes
        like ``num_samples``.

        Returns:
            List of Subject objects with path metadata but no loaded images.
        """
        _stub = torch.zeros(1, 1, 1, 1)
        subjects = []
        for record in self.index:
            path = str(record.get("primary_path", record.get("path", "unknown")))
            file_id = record.get("file_id", Path(path).stem)
            subject = tio.Subject(
                input=tio.ScalarImage(tensor=_stub),
                path=path,
                file_id=file_id,
            )
            subjects.append(subject)
        return subjects


# PATH_REMAP logic moved to PathResolver


def parse_fastmri_index(
    manifest_path: str,
    sensitivity_root: str | None = None,
    sensitivity_suffix: str = "_coil_maps.pt",
    data_root: str | None = None,
    contrasts: list[str] | None = None,
    target_contrasts: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Parse FastMRI manifest pickle into index format.

    Handles both legacy (v1) and new (v2) manifest formats:
    - v1: List of dicts with absolute paths
    - v2: Dict with 'version', 'data_root', 'use_relative', 'files'

    Args:
        manifest_path: Path to pickle manifest file
        sensitivity_root: Optional path to sensitivity maps directory
        sensitivity_suffix: Suffix for sensitivity map files
        data_root: Base path to resolve relative paths (overrides manifest's data_root)
        contrasts: Optional list of input contrasts to filter (e.g. ['T1w', 'T2w'])
        target_contrasts: Optional list of target contrasts to filter (e.g. ['FLAIR'])

    Returns:
        List of record dictionaries with resolved absolute paths, filtered by contrasts if specified
    """

    manifest_p = Path(manifest_path)

    # ── JSON v3 manifest ──────────────────────────────────────────
    if manifest_p.suffix == ".json":
        with open(manifest_p, encoding="utf-8") as f:
            json_data = json.load(f)

        manifest_data_root = json_data.get("data_root", "")
        raw_index = json_data.get("records", [])

        # Remap 'relative_path' key → 'path' to match internal convention
        for rec in raw_index:
            if "relative_path" in rec and "path" not in rec:
                rec["path"] = rec["relative_path"]

        use_relative = True  # v3 always uses relative paths
        if data_root:
            base_path = Path(data_root).resolve()
        elif manifest_data_root:
            resolved_manifest_root = PathResolver.resolve(manifest_data_root)
            base_path = Path(resolved_manifest_root).resolve()
        else:
            base_path = Path(".").resolve()

    # ── Pickle (v1 / v2) manifest ────────────────────────────────
    elif manifest_p.suffix == ".pkl":
        _logger.warning(
            "Loading deprecated .pkl manifest: %s — "
            "run 'python scripts/data/migrate_manifests_to_json.py' to convert to v3 JSON.",
            manifest_p.name,
        )
        with open(manifest_p, "rb") as f:
            raw_data = pickle.load(f)

        # Detect manifest version
        if isinstance(raw_data, dict) and "version" in raw_data:
            # v2 format with relative paths
            use_relative = raw_data.get("use_relative", False)
            manifest_data_root = raw_data.get("data_root", "")
            raw_index = raw_data.get("files", [])

            # Determine base path for resolving relative paths
            if data_root:
                base_path = Path(data_root).resolve()  # Convert to absolute
            elif manifest_data_root:
                resolved_manifest_root = PathResolver.resolve(manifest_data_root)
                base_path = Path(resolved_manifest_root).resolve()
            else:
                base_path = Path(".").resolve()  # Current directory (absolute)
        else:
            # v1 legacy format: list of dicts with absolute paths
            raw_index = raw_data
            use_relative = False
            base_path = Path(data_root).resolve() if data_root else Path(".").resolve()
    else:
        raise ValueError(
            f"Unsupported manifest format: {manifest_p.suffix} (expected .json or .pkl)"
        )

    # Normalize and filter by contrasts BEFORE path resolution (optimize early filtering)
    if contrasts or target_contrasts:
        contrasts_upper = [c.upper() for c in contrasts] if contrasts else None
        target_contrasts_upper = [c.upper() for c in target_contrasts] if target_contrasts else None
        filtered_raw_index = []

        for item in raw_index:
            # Extract file_id/path for contrast matching
            if isinstance(item, dict):
                file_id = item.get("file_id", item.get("path", item.get("file_path", "")))
            else:
                file_id = str(item)

            file_id_upper = file_id.upper()

            # Check if item matches contrast filters
            matches = True

            # Filter by input contrasts (filename heuristic: T1, T2, FLAIR, STIR patterns)
            if contrasts_upper:
                if not any(c in file_id_upper for c in contrasts_upper):
                    matches = False

            # Filter by target contrasts (same pattern)
            if matches and target_contrasts_upper:
                if not any(c in file_id_upper for c in target_contrasts_upper):
                    matches = False

            if matches:
                filtered_raw_index.append(item)

        raw_index = filtered_raw_index

    index = []
    for item in raw_index:
        # Handle both dict and simple path formats
        if isinstance(item, dict):
            # v2 uses 'path', v1 might use 'file_path'
            path_value = item.get("path", item.get("file_path", str(item)))
            # CRITICAL: FastMRISubjectBuilder expects 'primary_path', not 'path'
            record = {"primary_path": path_value}
            record["file_id"] = item.get("file_id", Path(path_value).stem)
            # [METADATA] Preserve explicit keys if available (for paired translation and sensitivity mapping)
            # ``shape`` (the raw-file dims from the manifest) lets the queue
            # builder's patch-compatibility filter check spatial extent WITHOUT
            # loading the volume — without it the filter materialises the whole
            # corpus and OOMs at queue-build (F-OOM 2026-06-08). See
            # TorchIOQueueBuilder._filter_patch_compatible_subjects.
            # ``field_strength``/``contrast``/``subject_id``/``pairing_group`` are the
            # MRIxFieldsPairedDataset field-snapshot contract (it reads them at the TOP
            # LEVEL of each record). Without retaining them here the mrixfields manifest
            # path (dataset_type='mrixfields' routes through this parser) drops them and
            # the dataset KeyErrors at construction — the whole cross-field cohort. They
            # are inert for FastMRI/M4Raw manifests that don't carry them.
            for retain_key in [
                "metadata",
                "target_path",
                "sensitivity_path",
                "shape",
                "field_strength",
                "contrast",
                "subject_id",
                "pairing_group",
            ]:
                if retain_key in item:
                    record[retain_key] = item[retain_key]
            # Federated / multi-site manifests (schema ``federated_with_site_tag``,
            # e.g. cross_site_pretrain.json) carry a top-level site tag per
            # record. Surface it into ``metadata['site']`` so the site-aware
            # loader paths find it — ManifestDatasetLoader Leave-One-Site-Out
            # holdout reads ``metadata.site`` (manifest_loader.py), and
            # ``attach_site_id`` propagates it for ``data.expose_site_id``.
            # Without this the site partition was silently dropped and every
            # sample defaulted to site 0 (the federated arm trained as if
            # single-site). Only fills when absent — an explicit metadata.site
            # wins.
            _site_tag = (
                item.get("site_id")
                or item.get("site")
                or item.get("institution")
                or item.get("systemVendor")
            )
            if _site_tag is not None:
                _md = record.setdefault("metadata", {})
                if isinstance(_md, dict) and not _md.get("site"):
                    _md["site"] = _site_tag
        else:
            path_value = str(item)
            # CRITICAL: FastMRISubjectBuilder expects 'primary_path', not 'path'
            record = {"primary_path": path_value, "file_id": Path(path_value).stem}

        # --- INTELLIGENT PATH RESOLUTION ---
        # Resolve target_path if it exists
        if "target_path" in record and not Path(record["target_path"]).is_absolute():
            target_val = record["target_path"]
            target_parts = Path(target_val).parts
            base_parts = base_path.parts
            skip_join = False
            if target_parts and target_parts[0] in ("databases", "data"):
                for i, bp in enumerate(base_parts):
                    if bp in ("databases", "data"):
                        project_root = Path(*base_parts[:i])
                        record["target_path"] = str(project_root / target_val)
                        skip_join = True
                        break
            if not skip_join:
                record["target_path"] = str(base_path / target_val)
            record["target_path"] = PathResolver.resolve(record["target_path"])

        # Avoid duplication when base_path and path share a common prefix
        path_val = record["primary_path"]

        if not Path(path_val).is_absolute():
            # Check if path starts with 'databases' or 'data' and base_path ends with same
            # e.g., base_path=/project/.../databases/m4raw, path=databases/m4raw/...
            # Should NOT join them (would cause duplication)
            path_parts = Path(path_val).parts
            base_parts = base_path.parts

            # Detect duplication pattern: if path starts with 'databases' or 'data'
            # and base_path contains this directory, extract from project root
            skip_join = False
            if path_parts and path_parts[0] in ("databases", "data"):
                # Path is relative starting with 'databases' or 'data'
                # Check if base_path contains 'databases' or 'data'
                for i, bp in enumerate(base_parts):
                    if bp in ("databases", "data"):
                        # Found common component - extract project root
                        project_root = Path(*base_parts[:i])
                        record["primary_path"] = str(project_root / path_val)
                        skip_join = True
                        break

            if not skip_join:
                # No duplication pattern detected - safe to join
                record["primary_path"] = str(base_path / path_val)

        # Remap absolute paths for portability (local <-> cluster) ONCE
        record["primary_path"] = PathResolver.resolve(record["primary_path"])

        # Add sensitivity path if root provided
        if sensitivity_root:
            file_id = record["file_id"].replace(".h5", "")
            smap_path = Path(sensitivity_root) / f"{file_id}{sensitivity_suffix}"
            if smap_path.exists():
                record["sensitivity_path"] = str(smap_path)

        # SANITIZATION: Fix common path concatenation errors
        # [DISABLED] This breaks datasets that legitimately have double nesting (e.g. M4Raw)
        # Pattern 1: multicoil_train/databases (duplication from join)
        if "/multicoil_train/databases/" in record["primary_path"]:
            # Remove the duplicated structure
            record["primary_path"] = record["primary_path"].replace(
                "/multicoil_train/databases/", "/databases/"
            )
        # if "/multicoil_train/multicoil_train/" in record["primary_path"]:
        #    record["primary_path"] = record["primary_path"].replace("/multicoil_train/multicoil_train/", "/multicoil_train/")
        # Pattern 2: Standard double-slash cleanup
        if "data//project" in record["primary_path"]:
            record["primary_path"] = record["primary_path"].replace("data//project", "/project")
        if "data//home" in record["primary_path"]:
            record["primary_path"] = record["primary_path"].replace("data//home", "/home")
        # Pattern 3: Project root duplication
        while "/gan_mri/databases/m4raw/databases/" in record["primary_path"]:
            record["primary_path"] = record["primary_path"].replace(
                "/databases/m4raw/databases/", "/databases/"
            )

        index.append(record)

    return index
