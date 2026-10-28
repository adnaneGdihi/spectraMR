"""Contrast-aware paired dataset for multi-contrast MRI translation.

This module provides contrast-aware loading for paired MRI data where:
- Input and target may have different contrasts (T1/T2, ULF/HF, etc.)
- Each contrast should be normalized independently but consistently
- Contrast metadata is tracked for debugging and analysis

Example use case:
    ULF-to-HF translation where ULF (64mT) is input and HF (3T) is target.
    Each has different SNR and intensity distributions requiring separate normalization.
"""

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torchio as tio

from spectramr.data.batch_types import TargetlessSampleError
from spectramr.data.datasets.field_strength_pair import stamp_field_strength_pair

logger = logging.getLogger(__name__)

from spectramr.data.transforms.mri_transforms import ULFNoiseGate

# Valid bidirectional reconstruction-direction modes. NN#3 / pitfall #9: a
# mis-typed value (e.g. 'hf_to_ulf_wrong', 'auto') must raise rather than
# silently invert the input/target direction and corrupt training pairs.
# hf_to_hf / ulf_to_ulf are single-field autoencode modes honored only on the
# nifti_paired path (dataset_instantiator._autoencode_field); the two-file
# ContrastAwarePairedDataset raises NotImplementedError for them.
_VALID_BIDIRECTIONAL_MODES: frozenset[str] = frozenset(
    {"ulf_to_hf", "hf_to_ulf", "hf_to_hf", "ulf_to_ulf"}
)
# The subset that drops one arm and autoencodes the other (input≡target).
_AUTOENCODE_BIDIRECTIONAL_MODES: frozenset[str] = frozenset({"hf_to_hf", "ulf_to_ulf"})


@dataclass
class ContrastConfig:
    """Configuration for a specific MRI contrast.

    Attributes:
        name: Contrast identifier (e.g., 'T1w', 'T2w', '64mT', '3T')
        normalization: Normalization strategy ('percentile', 'zscore', 'minmax', 'none')
        percentile: Upper percentile for robust normalization (e.g., 99.5)
        out_range: Output range after normalization (min, max)
        clamp: Whether to clamp values to out_range
        keywords: List of keywords to match in filenames (e.g., ['64mt', 'ulf'])
    """

    name: str
    normalization: Literal["percentile", "zscore", "minmax", "none"] = "percentile"
    percentile: float = 99.5
    out_range: tuple[float, float] = (0.0, 1.0)
    clamp: bool = True
    noise_gate: bool = False
    keywords: list[str] | None = dataclasses.field(default=None)

    def __post_init__(self):
        """__post_init__.

        Returns:
            Any: Description.
        """
        if self.keywords is None:
            # Default: use name as keyword (lowercase)
            self.keywords = [self.name.lower()]


class ContrastAwareSubjectBuilder:
    """Build TorchIO subjects with contrast-aware normalization.

    This builder wraps the standard FastMRISubjectBuilder but adds:
    1. Contrast detection from filenames
    2. Contrast-specific normalization
    3. Contrast metadata tracking

    Example:
        >>> input_contrast = ContrastConfig(
        ...     name='ULF',
        ...     normalization='percentile',
        ...     percentile=99.5,
        ...     keywords=['64mt', 'ulf', 'lf']
        ... )
        >>> target_contrast = ContrastConfig(
        ...     name='HF',
        ...     normalization='percentile',
        ...     percentile=99.5,
        ...     keywords=['3t', 'hf']
        ... )
        >>> builder = ContrastAwareSubjectBuilder(
        ...     input_contrast=input_contrast,
        ...     target_contrast=target_contrast,
        ...     io_strategy='nifti'
        ... )
    """

    def __init__(
        self,
        input_contrast: ContrastConfig,
        target_contrast: ContrastConfig,
        io_strategy: str = "nifti",
        verify_contrast: bool = True,
        normalize: bool = False,
    ):
        """Initialize contrast-aware subject builder.

        Args:
            input_contrast: Configuration for input contrast
            target_contrast: Configuration for target contrast
            io_strategy: IO strategy for loading files
            verify_contrast: Whether to verify detected contrast matches config
            normalize: Normalize intensity HERE. Defaults ``False``, and the
                default is the point. Image normalization is the transform
                chain's job (``ImageNormalizationTransform``), and this dataset
                calls that chain on the last line of ``__getitem__`` — so
                normalizing here as well applied two passes back to back, on 87
                arms, with the two disagreeing on percentile (99.5 vs 99), noise
                floor and clamping (#760; the image-domain twin of #571).
                ``SliceDataset`` already takes this shape: every one of its
                construction sites passes ``normalize=False``.
        """
        self.input_contrast = input_contrast
        self.target_contrast = target_contrast
        self.io_strategy = io_strategy
        self.verify_contrast = verify_contrast
        self.normalize = normalize

        # Import IO strategy
        from spectramr.data.io_strategies import IOStrategyFactory

        self.io = IOStrategyFactory.get(io_strategy)

        logger.info(
            f"[ContrastAwareBuilder] Input={input_contrast.name}, "
            f"Target={target_contrast.name}, "
            f"IO={io_strategy}"
        )

    def detect_contrast(self, filepath: str, expected: ContrastConfig) -> bool:
        """Detect if file matches expected contrast based on keywords.

        Args:
            filepath: Path to file
            expected: Expected contrast configuration

        Returns:
            True if file matches expected contrast
        """
        filename = Path(filepath).name.lower()

        for keyword in expected.keywords:
            if keyword.lower() in filename:
                return True

        return False

    def normalize_tensor(self, tensor: torch.Tensor, contrast: ContrastConfig) -> torch.Tensor:
        """Apply contrast-specific normalization.

        Delegates to :func:`spectramr.data.transforms.normalization.normalize_tensor`
        (SSOT) for all normalization math.

        Args:
            tensor: Input tensor
            contrast: Contrast configuration

        Returns:
            Normalized tensor
        """
        from spectramr.data.transforms.normalization import (
            NormalizationConfig,
            NormalizationStrategy,
            normalize_tensor,
        )

        if hasattr(contrast, "noise_gate") and contrast.noise_gate:
            # [PREPROCESSING] Hard-clip background noise (ULF Noise Gate)
            gate = ULFNoiseGate(threshold_percentile=5.0)
            tensor = gate(tensor)

        # Map contrast config → SSOT NormalizationConfig
        percentile_val = (
            contrast.percentile / 100.0 if contrast.percentile > 1.0 else contrast.percentile
        )
        config = NormalizationConfig(
            strategy=NormalizationStrategy.from_string(contrast.normalization),
            percentile=percentile_val,
            out_range=contrast.out_range,
            clamp=getattr(contrast, "clamp", True),
            min_scale=0.05,  # Noise floor: prevent amplifying background noise
        )

        result = normalize_tensor(tensor, config)

        logger.debug(
            f"[{contrast.name}] Normalized ({contrast.normalization}): "
            f"input_max={tensor.max():.4f} → output_mean={result.mean():.4f}"
        )

        return result

    def build(self, record: dict, swap: bool = False) -> tio.Subject:
        """Build TorchIO subject with contrast-aware normalization.

        Args:
            record: Dataset record with at minimum:
                - 'primary_path': Path to input file
                - 'target_path': Path to target file (required for paired data)
                - 'file_id': Identifier for the sample
            swap: When ``True``, swap input and target paths. Used for
                bidirectional testing (``bidirectional_mode='hf_to_ulf'``).

        Returns:
            TorchIO Subject with normalized input and target
        """
        # Resolve actual input/target paths and contrasts (may be swapped)
        if swap:
            input_path_key = "target_path"
            target_path_key = "primary_path"
            in_contrast = self.target_contrast
            tgt_contrast = self.input_contrast
        else:
            input_path_key = "primary_path"
            target_path_key = "target_path"
            in_contrast = self.input_contrast
            tgt_contrast = self.target_contrast

        # Load input
        input_path = record[input_path_key]
        input_data = self.io.load(input_path)

        # Handle dict return from IO (e.g., k-space + image)
        if isinstance(input_data, dict):
            input_tensor = input_data.get("image", input_data.get("data"))
        else:
            input_tensor = input_data

        # [CRITICAL] Check for NaN in loaded data BEFORE normalization
        # Note: We DO NOT replace NaN here - let dataset handle skipping
        if torch.isnan(input_tensor).any():
            nan_count = torch.isnan(input_tensor).sum().item()
            logger.warning(
                f"[DATA CORRUPTION] NaN detected in loaded input: {input_path}\n"
                f"  NaN count: {nan_count}/{input_tensor.numel()}\n"
                f"  Sample will be skipped by dataset (if skip_nan_samples=True)"
            )
            # DO NOT replace: input_tensor = torch.nan_to_num(input_tensor, nan=0.0)

        # Verify input contrast
        if self.verify_contrast:
            if not self.detect_contrast(input_path, in_contrast):
                logger.warning(
                    f"Input file {Path(input_path).name} does not match "
                    f"expected contrast {in_contrast.name} "
                    f"(keywords: {in_contrast.keywords})"
                )

        # Normalize input with its contrast config -- ONLY when this dataset
        # owns normalization. It does not by default: the transform chain this
        # dataset is handed (and calls on the last line of __getitem__) carries
        # ImageNormalizationTransform, so normalizing here too applied two
        # passes back to back. See the `normalize` argument.
        input_normalized = (
            self.normalize_tensor(input_tensor, in_contrast) if self.normalize else input_tensor
        )

        # Load target (OPTIONAL - for inference, target may not be present)
        has_target = False
        target_normalized = None
        target_path = None

        target_raw = record.get(target_path_key)
        if target_raw:
            target_path = target_raw
            target_data = self.io.load(target_path)

            if isinstance(target_data, dict):
                target_tensor = target_data.get("image", target_data.get("data"))
            else:
                target_tensor = target_data

            # Note: We DO NOT replace NaN here - let dataset handle skipping
            if torch.isnan(target_tensor).any():
                nan_count = torch.isnan(target_tensor).sum().item()
                logger.warning(
                    f"[DATA CORRUPTION] NaN detected in loaded target: {target_path}\n"
                    f"  NaN count: {nan_count}/{target_tensor.numel()}\n"
                    f"  Sample will be skipped by dataset (if skip_nan_samples=True)"
                )
                # DO NOT replace NaN with zeros - dataset __getitem__ will skip

            # Verify target contrast
            if self.verify_contrast:
                if not self.detect_contrast(target_path, tgt_contrast):
                    logger.warning(
                        f"Target file {Path(target_path).name} does not match "
                        f"expected contrast {tgt_contrast.name} "
                        f"(keywords: {tgt_contrast.keywords})"
                    )

            # Normalize target with its contrast config (see the input branch)
            target_normalized = (
                self.normalize_tensor(target_tensor, tgt_contrast)
                if self.normalize
                else target_tensor
            )

            # Check for NaN after normalization
            if torch.isnan(target_normalized).any():
                nan_count = torch.isnan(target_normalized).sum().item()
                logger.warning(
                    f"[NORMALIZATION FAILURE] NaN in target AFTER normalization: {target_path}\n"
                    f"  NaN count: {nan_count}/{target_normalized.numel()}\n"
                    f"  Sample will be skipped by dataset (if skip_nan_samples=True)"
                )
                # DO NOT replace NaN with zeros - dataset __getitem__ will skip

            has_target = True
        else:
            logger.debug(f"[Inference Mode] No target provided for {input_path}")

        # Check for NaN in input after normalization
        if torch.isnan(input_normalized).any():
            nan_count = torch.isnan(input_normalized).sum().item()
            logger.warning(
                f"[NORMALIZATION FAILURE] NaN in input AFTER normalization: {input_path}\n"
                f"  NaN count: {nan_count}/{input_normalized.numel()}\n"
                f"  Sample will be skipped by dataset (if skip_nan_samples=True)"
            )
            # DO NOT replace: input_normalized = torch.nan_to_num(input_normalized, nan=0.0)

        # Create TorchIO subject
        subject_dict = {
            "input": tio.ScalarImage(tensor=input_normalized),
            # Metadata
            "input_contrast": in_contrast.name,
            "input_path": str(input_path),
            "file_id": record.get("file_id", Path(input_path).stem),
        }

        # Add target only if available (training mode)
        if has_target:
            subject_dict["target"] = tio.ScalarImage(tensor=target_normalized)
            # [GAP 2 FIX] Many training strategies read batch["mri"] as the
            # ground-truth reference.  Alias it to the target tensor so the
            # subject is compatible with all strategy implementations.
            subject_dict["mri"] = tio.ScalarImage(tensor=target_normalized)
            subject_dict["target_contrast"] = tgt_contrast.name
            subject_dict["target_path"] = str(target_path)

        subject = tio.Subject(**subject_dict)

        if has_target:
            logger.debug(
                f"[Subject] {record.get('file_id', 'unknown')} → "
                f"input({in_contrast.name}): mean={input_normalized.mean():.3f}, "
                f"target({tgt_contrast.name}): mean={target_normalized.mean():.3f}"
            )
        else:
            logger.debug(
                f"[Subject] {record.get('file_id', 'unknown')} → "
                f"input({in_contrast.name}): mean={input_normalized.mean():.3f} "
                f"(inference mode, no target)"
            )

        return subject


class ContrastAwarePairedDataset(torch.utils.data.Dataset):
    """Dataset for contrast-aware paired MRI data.

    This dataset handles paired MRI data where input and target have different
    contrasts and require separate normalization strategies.

    Compatible with TorchIO Queue through lazy loading and dry_iter() method.

    Example:
        >>> from spectramr.data.datasets.contrast_aware import (
        ...     ContrastConfig,
        ...     ContrastAwarePairedDataset
        ... )
        >>>
        >>> # Define contrasts
        >>> ulf_config = ContrastConfig(
        ...     name='ULF_64mT',
        ...     normalization='percentile',
        ...     percentile=99.5,
        ...     keywords=['64mt', 'ulf', 'lf']
        ... )
        >>> hf_config = ContrastConfig(
        ...     name='HF_3T',
        ...     normalization='percentile',
        ...     percentile=99.5,
        ...     keywords=['3t', 'hf']
        ... )
        >>>
        >>> # Create dataset
        >>> dataset = ContrastAwarePairedDataset(
        ...     index=paired_index,  # From IndexBuilder
        ...     input_contrast=ulf_config,
        ...     target_contrast=hf_config,
        ...     io_strategy='nifti',
        ...     transform=my_transforms
        ... )
    """

    def __init__(
        self,
        index: list[dict],
        input_contrast: ContrastConfig,
        target_contrast: ContrastConfig,
        io_strategy: str = "nifti",
        transform: tio.Transform | None = None,
        normalize: bool = False,
        verify_contrast: bool = True,
        skip_nan_samples: bool = True,
        max_skip_attempts: int = 10,
        bidirectional_mode: str = "ulf_to_hf",
        allow_unpaired: bool = False,
        expose_field_strength_pair: bool = False,
    ):
        """Initialize contrast-aware paired dataset.

        Args:
            index: List of records with 'primary_path' and optionally 'target_path'
            input_contrast: Configuration for input contrast
            target_contrast: Configuration for target contrast
            io_strategy: IO strategy name ('nifti', 'h5', etc.)
            transform: TorchIO transform pipeline (applied AFTER normalization)
            verify_contrast: Whether to verify detected contrasts
            skip_nan_samples: If True, skip samples with NaN (recommended).
                If False, replace with zeros.
            max_skip_attempts: Maximum recursive attempts to find valid sample
            bidirectional_mode: ``'ulf_to_hf'`` (default) or ``'hf_to_ulf'``.
                When ``'hf_to_ulf'``, the builder swaps primary/target paths so
                that HF volumes become the network input.
            allow_unpaired: When True, samples without a ``target_path`` are
                included as inference-only subjects (no ``'target'`` key,
                ``has_target`` False). When False, such a record raises
                :class:`TargetlessSampleError`.
        """
        self.index = index
        self.transform = transform
        self.input_contrast = input_contrast
        self.target_contrast = target_contrast
        self.io_strategy = io_strategy
        self.verify_contrast = verify_contrast
        self.skip_nan_samples = skip_nan_samples
        self.max_skip_attempts = max_skip_attempts
        self.bidirectional_mode = bidirectional_mode
        self.allow_unpaired = allow_unpaired

        # Derive swap flag for the builder
        if bidirectional_mode not in _VALID_BIDIRECTIONAL_MODES:
            raise ValueError(
                f"bidirectional_mode={bidirectional_mode!r} is not recognised. "
                f"Valid values: {sorted(_VALID_BIDIRECTIONAL_MODES)}"
            )
        # Single-field autoencode modes drop one arm; this two-file dataset has
        # no self-supervised branch, so it cannot honor them. Raise (#9/#16)
        # instead of silently falling through to _swap=False (== ulf_to_hf).
        if bidirectional_mode in _AUTOENCODE_BIDIRECTIONAL_MODES:
            raise NotImplementedError(
                f"bidirectional_mode={bidirectional_mode!r} (single-field "
                "autoencode) is implemented only on the nifti_paired path "
                "(dataset_instantiator._autoencode_field); use "
                "dataset_type: nifti_paired."
            )
        self._swap = bidirectional_mode == "hf_to_ulf"
        self.expose_field_strength_pair = expose_field_strength_pair

        # Track skipped samples
        self._skipped_indices = set()
        self._skip_count = 0

        self.builder = ContrastAwareSubjectBuilder(
            input_contrast=input_contrast,
            target_contrast=target_contrast,
            io_strategy=io_strategy,
            verify_contrast=verify_contrast,
            # Forwarded rather than left at the builder's default so the switch
            # is reachable from the dataset a caller actually constructs; an
            # unreachable parameter is not a knob (#15). Production leaves it
            # False — `ImageNormalizationTransform` owns image normalization.
            normalize=normalize,
        )

        logger.info(
            f"[ContrastAwarePairedDataset] Initialized with {len(index)} samples. "
            f"Input={input_contrast.name}, Target={target_contrast.name}, "
            f"bidirectional_mode={bidirectional_mode}, "
            f"allow_unpaired={allow_unpaired}, "
            f"skip_nan_samples={skip_nan_samples}"
        )

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.index)

    def __getitem__(self, idx: int, _attempt: int = 0) -> tio.Subject:
        """Load and return a contrast-normalized subject.

        Args:
            idx: Sample index
            _attempt: Internal recursion counter (do not use)

        Returns:
            TorchIO Subject with normalized input and (if paired) target

        Raises:
            RuntimeError: If max_skip_attempts exceeded or all samples are corrupted
        """
        # Prevent infinite recursion
        if _attempt >= self.max_skip_attempts:
            raise RuntimeError(
                f"Failed to load valid sample after {self.max_skip_attempts} attempts. "
                f"Dataset may have too many corrupted files. "
                f"Run 'python scripts/diagnose_nan_files.py' to identify and remove them."
            )

        record = self.index[idx]

        # The index builder owns admission; a target-less record here is its defect, never resampled.
        if not self.allow_unpaired and not record.get("target_path"):
            raise TargetlessSampleError(
                f"ContrastAwarePairedDataset record {record.get('file_id', idx)!r} has no "
                "target_path but this split is paired (allow_unpaired=False): the index "
                "builder admitted an unpaired record into it."
            )

        # Build subject with contrast-aware normalization (+ optional swap)
        subject = self.builder.build(record, swap=self._swap)
        subject["has_target"] = torch.tensor("target" in subject)
        if self.expose_field_strength_pair:
            stamp_field_strength_pair(subject, record, record, swap=self._swap)

        # [CRITICAL] Check for NaN in loaded subject BEFORE transforms
        if self.skip_nan_samples:
            has_nan = False

            # Check input for NaN
            if "input" in subject:
                input_data = subject["input"].data
                if torch.isnan(input_data).any():
                    nan_pct = 100 * torch.isnan(input_data).sum().item() / input_data.numel()
                    logger.warning(
                        f"[SKIP] Sample {idx} has NaN in input ({nan_pct:.2f}%). "
                        f"File: {record.get('primary_path', 'unknown')}. "
                        f"Trying next sample..."
                    )
                    has_nan = True

            # Check target for NaN
            if "target" in subject and not has_nan:
                target_data = subject["target"].data
                if torch.isnan(target_data).any():
                    nan_pct = 100 * torch.isnan(target_data).sum().item() / target_data.numel()
                    logger.warning(
                        f"[SKIP] Sample {idx} has NaN in target ({nan_pct:.2f}%). "
                        f"File: {record.get('target_path', 'unknown')}. "
                        f"Trying next sample..."
                    )
                    has_nan = True

            # If NaN detected, skip to next sample
            if has_nan:
                self._skipped_indices.add(idx)
                self._skip_count += 1

                # Try next sample (circular)
                next_idx = (idx + 1) % len(self)

                # Log warning if skipping too many
                if self._skip_count % 10 == 0:
                    logger.error(
                        f"[NaN WARNING] Skipped {self._skip_count} samples with NaN so far. "
                        f"This indicates dataset corruption. "
                        f"Run 'python scripts/diagnose_nan_files.py' to identify corrupted files."
                    )

                # Recursively try next sample
                return self.__getitem__(next_idx, _attempt + 1)

        # Apply additional transforms (geometry, augmentation, etc.)
        # NOTE: Normalization already done in builder, so transform should NOT normalize
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
