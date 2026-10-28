"""Preprocessed MRI Dataset.

Unified data loader for preprocessing output directories (*_image/).
Serves data based on task type (reconstruction, super_resolution, field_translation).

Compliant with unit-test.md:
- Strict typing with explicit signatures
- No 'Any' types in public API
- Property-based testing support
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
import torchio as tio

from spectramr.data.builders.torchio_subject_builder import PreprocessedSubjectBuilder
from spectramr.data.split_utils import record_split_key, split_index_grouped

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Supported task types for preprocessing output loading."""

    RECONSTRUCTION = "reconstruction"
    SUPER_RESOLUTION = "super_resolution"
    FIELD_TRANSLATION = "field_translation"
    DENOISING = "denoising"
    COIL_COMBINATION = "coil_combination"
    KSPACE_GENERATION = "kspace_generation"
    MOTION_CORRECTION = "motion_correction"


class OutputDomain(str, Enum):
    """Output domain format for model compatibility.

    Different models require data in different formats:
    - IMAGE: Real-valued image tensor (C, H, W, D) - for standard CNNs, U-Nets
    - KSPACE_COMPLEX: Complex k-space tensor - for physics layers, data consistency
    - KSPACE_2CH: K-space as 2-channel real (real, imag) - for diffusion models
    - GRAPH: Graph representation (nodes, edges) - for Graph Neural Networks
    - HYBRID: Both image and k-space provided - for dual-domain methods
    """

    IMAGE = "image"
    KSPACE_COMPLEX = "kspace_complex"
    KSPACE_2CH = "kspace_2ch"
    GRAPH = "graph"
    HYBRID = "hybrid"


class GraphRepresentation(str, Enum):
    """Graph construction methods for GNN compatibility."""

    GRID_8 = "grid_8"  # 8-connected grid (spatial neighbors)
    GRID_4 = "grid_4"  # 4-connected grid
    KSPACE_RADIAL = "kspace_radial"  # Radial spokes in k-space
    PATCH_GRAPH = "patch_graph"  # Patches as supernodes
    LEARNED = "learned"  # Learned adjacency


@dataclass(frozen=True)
class PreprocessedSample:
    """Type-safe sample descriptor from preprocessing outputs.

    Immutable dataclass per Directive 3.0 (Immutable State by Default).
    """

    subject_id: str
    input_path: Path
    target_path: Path
    coil_sensitivity_path: Path | None = None
    statistics_path: Path | None = None
    task_type: TaskType = TaskType.RECONSTRUCTION


@dataclass(frozen=True)
class ArtifactDiscovery:
    """Result of scanning preprocessing output directory."""

    kspace: list[Path]
    coil_sensitivity: list[Path]
    nifti_reconstructed: list[Path]
    normalized: list[Path]
    registered: list[Path]
    statistics: list[Path]

    def available_artifacts(self) -> list[str]:
        """Return list of artifact types that have files."""
        result = []
        for field_name in [
            "kspace",
            "coil_sensitivity",
            "nifti_reconstructed",
            "normalized",
            "registered",
            "statistics",
        ]:
            if getattr(self, field_name):
                result.append(field_name)
        return result


class PreprocessedMRIDataset(torch.utils.data.Dataset):
    """Dataset for loading preprocessed MRI data.

    Reads from preprocessing output directories (*_image/) and serves
    data based on task type. Outputs TorchIO Subjects for transform compatibility.

    Directory structure expected:
        output_dir/
            kspace/              # Coil-compressed k-space (.pt)
            coil_sensitivity/    # ESPIRiT coil maps (.pt)
            nifti_reconstructed/ # NIfTI RSS volumes (.nii.gz)
            normalized/          # Intensity-normalized volumes (.nii.gz)
            registered/          # Registered volumes (.nii.gz, .h5)
            statistics/          # Per-file normalization stats (.json)
            manifests/           # TorchIO manifests (.csv, .pkl)

    Task mapping:
        - reconstruction: input=kspace, target=nifti_reconstructed
        - super_resolution: input=nifti_reconstructed, target=nifti_reconstructed
        - field_translation: input=registered, target=normalized
        - denoising: input=nifti_reconstructed, target=nifti_reconstructed (self-supervised)
        - coil_combination: input=kspace+coil_sensitivity, target=nifti_reconstructed

    .. mermaid::

        flowchart TD
            Dir[Output Directory] --> Discover{Discover Artifacts}
            Discover -->|Index| Samples[Sample List]

            Samples -->|__getitem__| Builder[PreprocessedSubjectBuilder]

            Builder -->|Load| Input[Input Tensor]
            Builder -->|Load| Target[Target Tensor]
            Builder -->|Load| Coils[Sensitivity Maps]

            Input & Target & Coils --> Subject[TorchIO Subject]
            Subject --> Transform[Augment/Transform]
            Transform --> Output[Final Subject]

    Example:
        >>> dataset = PreprocessedMRIDataset(
        ...     output_dir="databases/fastmri/.../singlecoil_train_image",
        ...     task_type="reconstruction",
        ... )
        >>> subject = dataset[0]
        >>> print(subject.input.shape)
    """

    # Task to artifact mapping
    TASK_ARTIFACT_MAP: dict[TaskType, tuple[str, str]] = {
        TaskType.RECONSTRUCTION: ("kspace", "nifti_reconstructed"),
        TaskType.SUPER_RESOLUTION: ("nifti_reconstructed", "nifti_reconstructed"),
        TaskType.FIELD_TRANSLATION: ("registered", "normalized"),
        TaskType.DENOISING: ("nifti_reconstructed", "nifti_reconstructed"),
        TaskType.COIL_COMBINATION: ("kspace", "nifti_reconstructed"),
        TaskType.KSPACE_GENERATION: ("kspace", "kspace"),
        TaskType.MOTION_CORRECTION: ("kspace", "nifti_reconstructed"),
    }

    def __init__(
        self,
        output_dir: str | Path,
        task_type: str | TaskType = TaskType.RECONSTRUCTION,
        output_domain: str | OutputDomain = OutputDomain.IMAGE,
        graph_type: str | GraphRepresentation | None = None,
        transform: Callable[[tio.Subject], tio.Subject] | None = None,
        split: str | None = None,
        validation_split: float = 0.1,
        max_samples: int | None = None,
        load_coil_sensitivity: bool = True,
        load_statistics: bool = False,
        contrasts: Sequence[str] | None = None,
        sessions: Sequence[str] | None = None,
        input_artifact: str | None = None,
        target_artifact: str | None = None,
        target_contrasts: Sequence[str] | None = None,
        target_sessions: Sequence[str] | None = None,
    ) -> None:
        """Initialize PreprocessedMRIDataset.

        Args:
            output_dir: Path to preprocessing output directory (*_image/)
            task_type: Type of task (reconstruction, super_resolution, etc.)
            output_domain: Output format (image, kspace_complex, kspace_2ch, graph, hybrid)
            graph_type: Graph construction method if output_domain is GRAPH
            transform: Optional TorchIO transform to apply
            split: Optional split ('train', 'val', None for all)
            validation_split: Fraction for validation if no manifest split
            max_samples: Maximum samples to load (for debugging)
            load_coil_sensitivity: Whether to load coil maps if available
            load_statistics: Whether to load per-file statistics
            contrasts: Filter specific contrasts (e.g. ['T1w', 'FLAIR'])
            sessions: Filter specific sessions (e.g. ['01', '02'])
            input_artifact: Explicitly select input artifact type (e.g. 'nifti_reconstructed')
            target_artifact: Explicitly select target artifact type (e.g. 'nifti_reconstructed')
            target_contrasts: Filter target contrasts. If None, no filter.
            target_sessions: Filter target sessions. If None, no filter.

        Raises:
            ValueError: If output_dir doesn't exist or task artifacts unavailable
        """
        self.output_dir = Path(output_dir).resolve()

        if not self.output_dir.exists():
            raise ValueError(f"Output directory does not exist: {self.output_dir}")

        # Parse task type
        if isinstance(task_type, str):
            try:
                self.task_type = TaskType(task_type)
            except ValueError:
                raise ValueError(
                    f"Unknown task_type: {task_type}. Available: {[t.value for t in TaskType]}"
                )
        else:
            self.task_type = task_type

        # Store artifact overrides
        self.input_artifact_override = input_artifact
        self.target_artifact_override = target_artifact

        # Filters
        self.contrasts = [c.upper() for c in contrasts] if contrasts else None
        self.sessions = [str(s) for s in sessions] if sessions else None
        self.target_contrasts = [c.upper() for c in target_contrasts] if target_contrasts else None
        self.target_sessions = [str(s) for s in target_sessions] if target_sessions else None

        # Parse output domain
        if isinstance(output_domain, str):
            try:
                self.output_domain = OutputDomain(output_domain)
            except ValueError:
                raise ValueError(
                    f"Unknown output_domain: {output_domain}. "
                    f"Available: {[d.value for d in OutputDomain]}"
                )
        else:
            self.output_domain = output_domain

        # Parse graph type (only required if output_domain is GRAPH)
        if graph_type is not None:
            if isinstance(graph_type, str):
                try:
                    self.graph_type = GraphRepresentation(graph_type)
                except ValueError:
                    raise ValueError(
                        f"Unknown graph_type: {graph_type}. "
                        f"Available: {[g.value for g in GraphRepresentation]}"
                    )
            else:
                self.graph_type = graph_type
        else:
            self.graph_type = (
                GraphRepresentation.GRID_8 if self.output_domain == OutputDomain.GRAPH else None
            )

        self.transform = transform
        self.split = split
        self.validation_split = validation_split
        self.load_coil_sensitivity = load_coil_sensitivity
        self.load_statistics = load_statistics

        # Initialize PreprocessedSubjectBuilder (Phase T4)
        self.subject_builder = PreprocessedSubjectBuilder()

        # Discover available artifacts
        self.artifacts = self._discover_artifacts()

        # Build sample index
        self._samples = self._build_index()

        # Apply split
        if split is not None:
            self._samples = self._apply_split(self._samples, split, validation_split)

        # Apply max_samples
        if max_samples is not None:
            self._samples = self._samples[:max_samples]

        logger.info(
            f"PreprocessedMRIDataset initialized: {len(self._samples)} samples, "
            f"task={self.task_type.value}, split={split}"
        )

    def _discover_artifacts(self) -> ArtifactDiscovery:
        """Scan output directory for available preprocessing artifacts.

        Returns:
            ArtifactDiscovery with lists of files per artifact type
        """

        def list_files(
            subdir: str, patterns: Sequence[str] = ("*.npy", "*.pt", "*.nii.gz", "*.h5")
        ) -> list[Path]:
            """list_files.

            Args:
                subdir (str): Description.
                patterns (Sequence[str]): Description.
            Returns:
                list[Path]: Description.
            """
            subdir_path = self.output_dir / subdir
            if not subdir_path.exists():
                return []
            files = []
            for pattern in patterns:
                files.extend(subdir_path.glob(pattern))
            return sorted(files)

        return ArtifactDiscovery(
            kspace=list_files("kspace", ("*.pt",)),
            coil_sensitivity=list_files("coil_sensitivity", ("*.pt",)),
            nifti_reconstructed=list_files("nifti_reconstructed", ("*.nii.gz", "*.nii")),
            normalized=list_files("normalized", ("*.nii.gz", "*.nii")),
            registered=list_files("registered", ("*.nii.gz", "*.nii", "*.h5")),
            statistics=list_files("statistics", ("*.json",)),
        )

    def _load_from_manifest(self, artifact_type: str) -> dict[str, dict]:
        """Load artifact metadata from pickle manifest if available.

        Returns:
            Dictionary mapping subject_id -> sample_metadata dict
        """
        import pickle

        manifest_dir = self.output_dir / "manifests"

        # Try to find manifest with pattern: *_{artifact_type}.pkl
        # e.g. fastmri_brain_multicoil_train_image_gt.pkl
        if not manifest_dir.exists():
            return {}

        manifest_files = list(manifest_dir.glob(f"*{artifact_type}.pkl"))
        if not manifest_files:
            return {}

        # Use the first matching manifest (usually only one per type per dataset)
        manifest_path = manifest_files[0]
        try:
            with open(manifest_path, "rb") as f:
                samples = pickle.load(f)

            # Index by file_id aka subject_id
            return {s["file_id"]: s for s in samples}
        except Exception as e:
            logger.warning(f"Failed to load manifest {manifest_path}: {e}")
            return {}

    def _build_index(self) -> list[PreprocessedSample]:
        """Build sample index from discovered artifacts.

        Returns:
            List of PreprocessedSample objects

        Raises:
            ValueError: If required artifacts for task are not available
        """
        # Determine artifacts: Use overrides if provided, else use task default
        default_input, default_target = self.TASK_ARTIFACT_MAP.get(self.task_type, (None, None))

        input_artifact = self.input_artifact_override or default_input
        target_artifact = self.target_artifact_override or default_target

        if not input_artifact or not target_artifact:
            # Fallback failed
            raise ValueError(
                f"Could not determine artifacts for task '{self.task_type}'. "
                "Please specify valid task_type or provide input_artifact/target_artifact."
            )

        # precise artifact names for manifest search
        # TASK_ARTIFACT_MAP values map to directory names, which map to manifest suffixes
        # e.g. "nifti_reconstructed" -> "_image_reconstructed.pkl" via preprocessing mapping
        # but let's try direct directory scanning first as baseline, then enrich/filter with manifest

        # Helper to resolve artifact files: check dataclass first, then filesystem
        def _resolve_files(name: str) -> list[Path]:
            # 1. Check if it's a known artifact in ArtifactDiscovery
            """_resolve_files.

            Args:
                name (str): Description.
            Returns:
                list[Path]: Description.
            """
            if hasattr(self.artifacts, name):
                return getattr(self.artifacts, name)

            # 2. Fallback: Check filesystem for custom directory
            path = self.output_dir / name
            if path.is_dir():
                # Allow common extensions
                extensions = ["*.nii.gz", "*.nii", "*.h5", "*.pt", "*.npy"]
                files = []
                for ext in extensions:
                    files.extend(path.glob(ext))
                return sorted(list(set(files)))  # Unique & sorted

            return []

        input_files = _resolve_files(input_artifact)
        target_files = _resolve_files(target_artifact)

        if not input_files:
            raise ValueError(
                f"No input files found for task '{self.task_type.value}'. "
                f"Expected files in: {self.output_dir / input_artifact}"
            )

        if not target_files:
            raise ValueError(
                f"No target files found for task '{self.task_type.value}'. "
                f"Expected files in: {self.output_dir / target_artifact}"
            )

        # Try to load manifests for filtering
        # Map directory name back to manifest suffix used in preprocessing.py
        # OUTPUT_CONFIGS in preprocessing.py:
        # "kspace": "kspace"
        # "nifti_reconstructed": "image_reconstructed"
        # "normalized": "image_normalized"
        # "registered": "image_registered"

        manifest_suffix_map = {
            "kspace": "kspace",
            "nifti_reconstructed": "image_reconstructed",
            "normalized": "image_normalized",
            "registered": "image_registered",
        }

        input_manifest_suffix = manifest_suffix_map.get(input_artifact, input_artifact)
        input_metadata = self._load_from_manifest(input_manifest_suffix)

        # Build subject ID to file mapping
        input_by_id: dict[str, Path] = {}
        input_meta_by_id: dict[str, dict] = {}

        for f in input_files:
            subject_id = self._extract_subject_id(f)

            # Metadata filtering
            meta = input_metadata.get(subject_id)
            if meta:
                # Check filters if metadata available
                if self.contrasts:
                    # Case-insensitive contrast check
                    contrast = meta.get("contrast", "unknown")
                    if contrast.upper() not in self.contrasts:
                        continue

                if self.sessions:
                    session = meta.get("session_id")
                    if str(session) not in self.sessions:
                        continue
            elif self.contrasts or self.sessions:
                # Fallback: parsing from filename if manifest missing but filters requested
                # Simple heuristic for common contrast names in filenames
                if self.contrasts:
                    fname_upper = f.name.upper()
                    if not any(c in fname_upper for c in self.contrasts):
                        continue

            input_by_id[subject_id] = f
            input_meta_by_id[subject_id] = meta or {}

        target_by_id: dict[str, Path] = {}

        # Load target metadata
        target_manifest_suffix = manifest_suffix_map.get(target_artifact, target_artifact)
        target_metadata = self._load_from_manifest(target_manifest_suffix)

        for f in target_files:
            subject_id = self._extract_subject_id(f)

            # Target Metadata filtering
            meta = target_metadata.get(subject_id)
            if meta:
                if self.target_contrasts:
                    contrast = meta.get("contrast", "unknown")
                    if contrast.upper() not in self.target_contrasts:
                        continue
                if self.target_sessions:
                    session = meta.get("session_id")
                    if str(session) not in self.target_sessions:
                        continue
            elif self.target_contrasts or self.target_sessions:
                # Fallback heuristic
                if self.target_contrasts:
                    fname_upper = f.name.upper()
                    if not any(c in fname_upper for c in self.target_contrasts):
                        continue

            target_by_id[subject_id] = f

        # Coil sensitivity mapping (optional)
        coil_by_id: dict[str, Path] = {}
        if self.load_coil_sensitivity and self.artifacts.coil_sensitivity:
            for f in self.artifacts.coil_sensitivity:
                subject_id = self._extract_subject_id(f)
                coil_by_id[subject_id] = f

        # Statistics mapping (optional)
        stats_by_id: dict[str, Path] = {}
        if self.load_statistics and self.artifacts.statistics:
            for f in self.artifacts.statistics:
                subject_id = self._extract_subject_id(f)
                stats_by_id[subject_id] = f

        # Find common subject IDs
        if self.task_type == TaskType.DENOISING:
            # Self-supervised: input == target
            common_ids = set(input_by_id.keys())
        else:
            common_ids = set(input_by_id.keys()) & set(target_by_id.keys())

        if not common_ids:
            if self.contrasts or self.sessions:
                raise ValueError(
                    f"No matching samples found after filtering. "
                    f"Filters: contrasts={self.contrasts}, sessions={self.sessions}. "
                    f"Input files matched: {len(input_by_id)}"
                )
            raise ValueError(
                f"No matching samples found between input ({len(input_by_id)}) "
                f"and target ({len(target_by_id)}) files."
            )

        # Build samples
        samples = []
        for subject_id in sorted(common_ids):
            samples.append(
                PreprocessedSample(
                    subject_id=subject_id,
                    input_path=input_by_id[subject_id],
                    target_path=target_by_id.get(subject_id, input_by_id[subject_id]),
                    coil_sensitivity_path=coil_by_id.get(subject_id),
                    statistics_path=stats_by_id.get(subject_id),
                    task_type=self.task_type,
                )
            )

        return samples

    def _extract_subject_id(self, path: Path) -> str:
        """Extract subject ID from file path.

        Removes common suffixes to match files across artifact types.
        """
        name = path.stem
        # Handle double extension (.nii.gz)
        if name.endswith(".nii"):
            name = name[:-4]

        # Remove common suffixes
        for suffix in [
            "_rss",
            "_coil_maps",
            "_compressed",
            "_stats",
            "_reconstructed",
            "_normalized",
            "_registered",
            # Common contrast suffixes for simple generic pairing
            "_T1w",
            "_t1w",
            "_T1",
            "_t1",
            "_T2w",
            "_t2w",
            "_T2",
            "_t2",
            "_FLAIR",
            "_flair",
            "_PD",
            "_pd",
        ]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]

        return name

    def _apply_split(
        self,
        samples: list[PreprocessedSample],
        split: str,
        validation_split: float,
    ) -> list[PreprocessedSample]:
        """Apply train/val split to samples, through the data-layer SSOT.

        Delegates to :func:`spectramr.data.split_utils.split_index` rather than
        slicing here. The hand-rolled version this replaces disagreed with every
        other dataset type in three ways at once:

        * it took validation from the **start** (``samples[:n_val]``) while the
          SSOT — and therefore every other loader — takes it from the end, so a
          ``preprocessed`` arm validated on different samples than an otherwise
          identical arm of any other type;
        * it truncated with ``int()`` instead of rounding, which HALVES the
          validation set at ``validation_fraction: 0.15`` on a 10-file corpus
          (1 instead of 2) — and 0.15 is one of the two commonest fractions in
          the corpus;
        * it produced a silently EMPTY validation set whenever
          ``n * fraction < 1`` (e.g. 3 files at 0.1), and an empty TRAIN set at
          ``fraction: 1.0``. The SSOT clamps both splits non-empty and raises on
          the single-file case, which is the whole reason it exists.

        Args:
            samples: Full list of samples
            split: 'train' or 'val'
            validation_split: Fraction for validation

        Returns:
            Filtered sample list

        Raises:
            ValueError: unknown ``split``, or a single-file corpus with
                ``validation_split > 0`` (raised by the SSOT).
        """
        if split not in ("train", "val"):
            raise ValueError(f"Unknown split value: {split!r}. Expected 'train' or 'val'.")
        train_samples, val_samples = split_index_grouped(
            samples, validation_split, key=record_split_key
        )
        return val_samples if split == "val" else train_samples

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> tio.Subject:
        """Load and return a TorchIO Subject.

        Delegates Subject creation to PreprocessedSubjectBuilder (Phase T4),
        eliminating 50+ lines of duplicated logic.

        Args:
            idx: Sample index

        Returns:
            TorchIO Subject with 'input', 'target', and optionally 'sensitivity' keys
        """
        sample = self._samples[idx]

        # Build record for Subject builder
        record = {
            "image_path": sample.input_path,
            "gt_image_path": sample.target_path,
        }

        # Add optional paths
        if sample.coil_sensitivity_path is not None:
            record["sensitivity_path"] = sample.coil_sensitivity_path
        if sample.statistics_path is not None:
            record["statistics_path"] = sample.statistics_path

        # Use PreprocessedSubjectBuilder to handle all complex logic
        subject = self.subject_builder.build(record)

        # Add metadata
        subject["subject_id"] = sample.subject_id
        subject["task_type"] = sample.task_type.value

        # Apply transform
        if self.transform is not None:
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
        for sample in self._samples:
            subject = tio.Subject(
                input=tio.ScalarImage(tensor=_stub),
                subject_id=sample.subject_id,
                task_type=sample.task_type.value,
            )
            subjects.append(subject)
        return subjects

    @property
    def available_artifacts(self) -> list[str]:
        """Return list of available artifact types."""
        return self.artifacts.available_artifacts()

    def get_statistics(self) -> dict[str, int]:
        """Return dataset statistics."""
        return {
            "num_samples": len(self._samples),
            "task_type": self.task_type.value,
            "input_artifact": self.TASK_ARTIFACT_MAP[self.task_type][0],
            "target_artifact": self.TASK_ARTIFACT_MAP[self.task_type][1],
            "available_artifacts": self.available_artifacts,
        }


def create_preprocessed_dataloader(
    output_dir: str | Path,
    task_type: str = "reconstruction",
    output_domain: str = "image",
    graph_type: str | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    split: str | None = None,
    transform: Callable | None = None,
    pin_memory: bool = True,
) -> tio.SubjectsLoader:
    """Create a DataLoader for preprocessed MRI data.

    Convenience function for creating ready-to-use DataLoader.

    Args:
        output_dir: Path to preprocessing output directory
        task_type: Task type (reconstruction, super_resolution, etc.)
        output_domain: Output format (image, kspace_complex, kspace_2ch, graph, hybrid)
        graph_type: Graph construction method if output_domain is "graph"
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        split: Optional split ('train', 'val')
        transform: Optional TorchIO transform
        pin_memory: Use pinned memory for faster GPU transfer

    Returns:
        PyTorch DataLoader yielding TorchIO Subjects
    """
    dataset = PreprocessedMRIDataset(
        output_dir=output_dir,
        task_type=task_type,
        output_domain=output_domain,
        graph_type=graph_type,
        split=split,
        transform=transform,
    )

    # [TORCHIO] Use SubjectsLoader for preprocessed data
    return tio.SubjectsLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )


# =============================================================================
# Domain Conversion Utilities
# =============================================================================


def complex_to_2channel(tensor: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to 2-channel real (real, imag).

    Used for k-space diffusion models (e2e_kspace_cold_diffusion).

    Args:
        tensor: Complex tensor of shape (..., H, W)

    Returns:
        Real tensor of shape (2, ..., H, W) with channel 0=real, 1=imag
    """
    if not torch.is_complex(tensor):
        return tensor

    real = tensor.real
    imag = tensor.imag
    return torch.stack([real, imag], dim=0)


def channel2_to_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Convert 2-channel real tensor to complex.

    Inverse of complex_to_2channel.

    Args:
        tensor: Real tensor of shape (2, ..., H, W)

    Returns:
        Complex tensor of shape (..., H, W)
    """
    if tensor.shape[0] != 2:
        raise ValueError(f"Expected 2 channels, got {tensor.shape[0]}")

    return torch.complex(tensor[0], tensor[1])


def _slice_to_graph(
    image: torch.Tensor,
    graph_type: GraphRepresentation,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(x, edge_index)`` for a single 2-D ``(C, H, W)`` slice."""
    C, H, W = image.shape

    if graph_type == GraphRepresentation.GRID_8 or graph_type == GraphRepresentation.GRID_4:
        # Each pixel is a node
        x = image.view(C, -1).T  # (H*W, C)

        # Build edge index for grid connectivity
        edges = []
        for i in range(H):
            for j in range(W):
                node_id = i * W + j
                # 4-connected neighbors
                if j < W - 1:  # Right
                    edges.append([node_id, node_id + 1])
                if i < H - 1:  # Down
                    edges.append([node_id, node_id + W])

                if graph_type == GraphRepresentation.GRID_8:
                    # 8-connected (add diagonals)
                    if i < H - 1 and j < W - 1:  # Down-right
                        edges.append([node_id, node_id + W + 1])
                    if i < H - 1 and j > 0:  # Down-left
                        edges.append([node_id, node_id + W - 1])

        edge_index = torch.tensor(edges, dtype=torch.long).T

    elif graph_type == GraphRepresentation.PATCH_GRAPH:
        # Each patch is a supernode
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        # Extract patch features (mean pooling)
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = image[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patches.append(patch.mean(dim=(1, 2)))

        x = torch.stack(patches, dim=0)  # (num_nodes, C)

        # 4-connected patch graph
        edges = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                node_id = i * num_patches_w + j
                if j < num_patches_w - 1:
                    edges.append([node_id, node_id + 1])
                if i < num_patches_h - 1:
                    edges.append([node_id, node_id + num_patches_w])

        edge_index = (
            torch.tensor(edges, dtype=torch.long).T
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )

    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")

    return x, edge_index


def image_to_graph(
    image: torch.Tensor,
    graph_type: GraphRepresentation = GraphRepresentation.GRID_8,
    patch_size: int = 16,
) -> dict[str, torch.Tensor]:
    """Convert image to graph representation for GNNs.

    A 3-D ``(C, H, W, D)`` volume contributes EVERY slice: one 2-D grid per
    slice, concatenated into a single graph whose per-slice sub-grids are
    disjoint (node indices offset per slice). This replaces the previous silent
    collapse to the central slice (CLAUDE.md #9), so no depth information is
    dropped on the way into a GNN.

    Args:
        image: Image tensor of shape (C, H, W) or (C, H, W, D)
        graph_type: How to construct the graph
        patch_size: Patch size for patch_graph mode

    Returns:
        Dict with 'x' (node features), 'edge_index' (connectivity),
        'num_nodes', and 'original_shape'.
    """
    if image.ndim == 4:
        depth = image.shape[-1]
        xs: list[torch.Tensor] = []
        edge_blocks: list[torch.Tensor] = []
        offset = 0
        for d in range(depth):
            x_d, edge_d = _slice_to_graph(image[..., d], graph_type, patch_size)
            xs.append(x_d)
            if edge_d.numel() > 0:
                edge_blocks.append(edge_d + offset)
            offset += x_d.shape[0]
        x = torch.cat(xs, dim=0)
        edge_index = (
            torch.cat(edge_blocks, dim=1) if edge_blocks else torch.empty((2, 0), dtype=torch.long)
        )
        return {
            "x": x,
            "edge_index": edge_index,
            "num_nodes": x.shape[0],
            "original_shape": tuple(image.shape),
        }

    x, edge_index = _slice_to_graph(image, graph_type, patch_size)
    return {
        "x": x,
        "edge_index": edge_index,
        "num_nodes": x.shape[0],
        "original_shape": tuple(image.shape),
    }


# =============================================================================
# Experiment Integration Guide
# =============================================================================

EXPERIMENT_CONFIG_EXAMPLES = """
# =============================================================================
# EXPERIMENT INTEGRATION GUIDE
# =============================================================================

This loader integrates with the following experiment types in spectramr:

## 1. K-Space Reconstruction (exp_11, FastMRI)
```yaml
data:
  dataset_type: preprocessed
  preprocessing_dir: databases/fastmri/.../singlecoil_train_image
  task_type: reconstruction
  output_domain: image  # Convert to image for U-Net
model:
  model_type: standard_unet
  in_channels: 1
  out_channels: 1
```

## 2. K-Space Cold Diffusion (e2e_kspace_cold_diffusion)
```yaml
data:
  dataset_type: preprocessed
  preprocessing_dir: databases/fastmri/.../multicoil_train_image
  task_type: reconstruction
  output_domain: kspace_2ch  # Keep in k-space, 2-channel real
model:
  model_type: standard_unet
  in_channels: 2  # Real + Imag
  out_channels: 2
training:
  training_mode: diffusion
```

## 3. Graph U-Net (exp_104)
```yaml
data:
  dataset_type: preprocessed
  preprocessing_dir: databases/fastmri/.../singlecoil_train_image
  task_type: reconstruction
  output_domain: graph
  graph_type: patch_graph  # Use patches as supernodes
  patch_size: [64, 64, 1]
model:
  model_type: graph_unet
```

## 4. ULF→HF Field Translation (exp_32)
```yaml
data:
  dataset_type: preprocessed
  preprocessing_dir: databases/open_siim/.../ulf_paired_image
  task_type: field_translation
  output_domain: image
model:
  model_type: pix2pix_generator
```

## 5. M4Raw Motion Correction (motion)
```yaml
data:
  dataset_type: preprocessed
  preprocessing_dir: databases/m4raw/.../motion_image
  task_type: motion_correction
  output_domain: hybrid  # Both k-space and image
```

## 6. BraTS Super-Resolution
```yaml
data:
  dataset_type: preprocessed
  preprocessing_dir: databases/brats_sr/BraTS-SSR_image
  task_type: super_resolution
  output_domain: image
```
"""
