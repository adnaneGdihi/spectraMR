"""Collation Strategy Pattern - Data Loader SSOT.

Unified collation strategies for different data types:
- Images (2D/3D with padding/stacking)
- Graphs (variable node/edge counts)
- Sequences (variable length with masking)
- Mixed modalities (images + metadata)

Benefits:
- Clear separation of data type handling
- No image-specific logic leaking into graph data
- Reusable strategies across projects
- Type-safe via ABC pattern
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from spectramr.data.collation.unpad import remove_padding


class CollateStrategy(ABC):
    """Collation is the process of combining variable-sized data samples
    into fixed batches suitable for model input.

    .. mermaid::

        classDiagram
            class CollateStrategy {
                <<abstract>>
                +collate(batch)
                +validate_batch(batch)
            }
            class ImageCollateStrategy {
                +padding_mode
                +collate()
                +unpad()
            }
            class GraphCollateStrategy {
                +follow_batch
                +collate()
            }
            class SequenceCollateStrategy {
                +max_length
                +collate()
            }
            class MixedModalityCollateStrategy {
                +strategies
                +collate()
            }

            CollateStrategy <|-- ImageCollateStrategy
            CollateStrategy <|-- GraphCollateStrategy
            CollateStrategy <|-- SequenceCollateStrategy
            CollateStrategy <|-- MixedModalityCollateStrategy
    """

    @abstractmethod
    def collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate batch of samples.

        Args:
            batch: List of sample dicts from dataset

        Returns:
            Dictionary with collated tensors and metadata

        Raises:
            ValueError: If batch cannot be collated
        """
        raise NotImplementedError

    def validate_batch(self, batch: list[dict[str, Any]]) -> bool:
        """Validate that batch can be collated.

        Args:
            batch: List of samples

        Returns:
            True if batch is valid for collation

        Raises:
            ValueError: If batch is invalid
        """
        if not batch:
            raise ValueError("Empty batch")

        return True


class ImageCollateStrategy(CollateStrategy):
    """Collate 2D/3D image tensors with optional padding.

    Handles:
    - 4D tensors: (B, C, H, W)
    - 5D tensors: (B, C, D, H, W) [TorchIO format]
    - Mixed batch sizes: pads to max shape
    - Returns padding info for later unpadding
    """

    def __init__(
        self,
        padding_mode: str = "zeros",
        padding_value: float = 0.0,
        allow_variable_shapes: bool = False,
        squeeze_depth_dim: bool = False,
        flatten_3d_to_2d: bool = False,
        validate_nans: bool = True,
    ):
        """Initialize image collation strategy.

        Args:
            padding_mode: "zeros", "reflect", or "replicate"
            padding_value: Value to use for zero padding
            allow_variable_shapes: If True, allow variable shapes in batch
                                   If False, require padding to uniform shape
            squeeze_depth_dim: If True, squeeze the LAST dimension if it is 1.
                               Used for TorchIO compatibility (2D patches loaded as 3D volumes).
            flatten_3d_to_2d: If True, flattens 5D volumes [B, C, H, W, D] into
                4D slices [B*D, C, H, W]. DELIBERATELY NOT REACHABLE FROM CONFIG
                (audit A11): it is not a ``CollationConfigSchema`` field and
                ``CollationStrategySelector`` never passes it, so on every arm
                it is False. That is correct and must stay so — the production
                5D->4D flatten lives in ``pipelines/train.py`` ("Validation
                tensor prep: ComplexGuard -> 5D->4D (2D nets only) -> square-pad"),
                and exposing this knob would flatten a second time on any arm
                that set it. Reachable only via :func:`squeezing_collate`, a
                standalone helper whose shape contract is pinned by
                ``tests/integration/test_shape_contracts.py``.
                              Ensures dimension agnosticism for 2D models.
            validate_nans: If True (default), the final collation pass replaces any
                           NaN values with zeros. Set to False to pass NaNs through
                           unchanged (e.g. for debugging). Wired from
                           ``collation.validate_nans`` in the YAML config.
        """
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.allow_variable_shapes = allow_variable_shapes
        self.squeeze_depth_dim = squeeze_depth_dim
        self.flatten_3d_to_2d = flatten_3d_to_2d
        self.validate_nans = validate_nans

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate batch with support for multiple keys (image, kspace, target).

        Args:
            batch: List of dicts/Subjects

        Returns:
            Dict with collated tensors
        """
        # DEBUG: Log collation config and input
        import logging

        logger = logging.getLogger(__name__)

        if batch and isinstance(batch[0], dict):
            elem_keys = list(batch[0].keys())
            logger.debug(f"[COLLATE] batch[0] keys: {elem_keys}")

        # [NEW] Handle TrainingBatch samples by converting to dict
        from spectramr.data.batch_types import TrainingBatch

        if batch and isinstance(batch[0], TrainingBatch):
            new_batch = []
            for item in batch:
                d = {
                    "input": item.input,
                    "target": item.target,
                }
                if item.mask is not None:
                    d["mask"] = item.mask

                # Merge metadata
                if item.metadata:
                    d.update(item.metadata)
                new_batch.append(d)
            batch = new_batch

        self.validate_batch(batch)

        elem = batch[0]
        collated = {}
        depth_dim = 1  # Track depth for 3D to 2D flattening

        # The key set is checked ACROSS the batch, not read off batch[0] (B13).
        # Iterating `elem` alone silently dropped any key the first sample
        # happened not to carry — and the samples that DO carry it are exactly
        # the interesting ones (a dataset that attaches `sensitivity` only when
        # coil maps exist, a transform that adds `scout` conditionally). The
        # dropped key does not error anywhere; the mechanism behind it simply
        # never receives its input.
        _key_sets = [set(d.keys()) for d in batch]
        _union: set = set().union(*_key_sets) if _key_sets else set()
        _missing = {k: sum(k not in ks for ks in _key_sets) for k in _union}
        _partial = {k: n for k, n in _missing.items() if n}
        if _partial:
            raise ValueError(
                f"Batch samples disagree about which keys they carry: "
                f"{ {k: f'{n}/{len(batch)} samples missing' for k, n in sorted(_partial.items())} }. "
                "Collating the intersection would silently drop a mechanism's "
                "input; collating the union would stack absent tensors. Emit "
                "the same keys for every sample, or route this dataset through "
                "the 'robust' collate strategy, which is built for ragged "
                "batches."
            )

        # Iterate over all keys in the sample (e.g. image, kspace, target, sensitivity)
        for key in elem:
            # Skip internal keys if needed, but TorchIO usually keeps them clean
            items = [d[key] for d in batch]

            # Handle TorchIO Images (extract data)
            if hasattr(items[0], "data") and hasattr(items[0], "affine"):
                tensors = [item.data for item in items]
            elif isinstance(items[0], torch.Tensor):
                tensors = items
            else:
                # Non-tensor data (metadata, strings) - use list
                collated[key] = items
                continue

            # Tensor Collation Logic (Padding/Stacking)
            shapes = [t.shape for t in tensors]

            if self.allow_variable_shapes:
                if all(s == shapes[0] for s in shapes):
                    stacked = torch.stack(tensors)
                else:
                    # Variable shapes allowed -> return list
                    collated[key] = tensors
                    collated[f"{key}_shapes"] = shapes
                    continue
            else:
                # Pad to max shape if needed
                max_shape = (
                    tuple(max(s) for s in zip(*shapes, strict=False)) if shapes else shapes[0]
                )

                if all(s == max_shape for s in shapes):
                    stacked = torch.stack(tensors)
                else:
                    # A CHANNEL-count mismatch is not paddable (B4). Channels
                    # are physical: coil c of a 20-coil scan is a different
                    # element than coil c of a 16-coil scan. Centre-padding
                    # made it worse than merely wrong — a 16-coil sample padded
                    # to 20 got 2 zero channels BEFORE and 2 after, so its coil
                    # 0 landed at index 2 and was stacked against coil 2 of the
                    # 20-coil sample. Identity destroyed, silently. This is the
                    # same refusal `m4raw_dataset._select_consistent_reps` makes
                    # for NEX repetitions, for the same physics.
                    _channel_counts = {int(t.shape[0]) for t in tensors}
                    if len(_channel_counts) > 1:
                        raise ValueError(
                            f"Batch key {key!r} mixes channel counts "
                            f"{sorted(_channel_counts)}. Channels are physical "
                            "(coils / contrasts), so padding them to a common "
                            "size aligns element c of one sample against a "
                            "different element c of another — and centre "
                            "padding additionally SHIFTS the indices. Group "
                            "samples by receive array, or compress coils to a "
                            "fixed rank before collation."
                        )

                    padded = []
                    pad_record: list[list[tuple[int, int]] | None] = []
                    for t in tensors:
                        if t.shape == max_shape:
                            padded.append(t)
                            pad_record.append(None)
                        else:
                            # Pad logic — spatial dims only; dim 0 is guarded above.
                            pad_amounts = []
                            for i in range(len(max_shape)):
                                diff = max_shape[i] - t.shape[i]
                                pad_before = diff // 2
                                pad_after = diff - pad_before
                                pad_amounts.append((pad_before, pad_after))

                            pad_flat = []
                            for pad in reversed(pad_amounts):
                                pad_flat.extend(pad)

                            # F.pad uses 'constant' not 'zeros'
                            fpad_mode = (
                                "constant" if self.padding_mode == "zeros" else self.padding_mode
                            )
                            padded_t = torch.nn.functional.pad(
                                t,
                                pad_flat,
                                mode=fpad_mode,
                                value=self.padding_value,
                            )
                            padded.append(padded_t)
                            pad_record.append(pad_amounts)
                    stacked = torch.stack(padded)
                    # Record WHAT was padded. `unpad()` has always taken a
                    # `padding_info` argument and nothing ever produced one, so
                    # the padding was irreversible and invisible: val metrics on
                    # a padded cohort are computed partly over fabricated zero
                    # regions (PSNR inflated), with nothing in the batch saying
                    # so. This is that producer.
                    if any(rec is not None for rec in pad_record):
                        collated[f"{key}_padding"] = pad_record

            # Squeeze dim logic
            if self.squeeze_depth_dim and stacked.ndim == 5 and stacked.shape[-1] == 1:
                stacked = stacked.squeeze(-1)
            elif self.flatten_3d_to_2d and stacked.ndim == 5:
                # [B, C, H, W, D] -> [B, D, C, H, W] -> [B*D, C, H, W]
                B, C, H, W, D = stacked.shape
                depth_dim = D
                stacked = stacked.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)

            collated[key] = stacked

        # [NEW] Explicit contrast_idx handling
        if "contrast_idx" not in collated and any("contrast_idx" in b for b in batch):
            collated["contrast_idx"] = torch.stack(
                [b.get("contrast_idx", torch.tensor(0, dtype=torch.long)) for b in batch]
            )

        # Expand 1D tensors if we flattened 3D to 2D
        if self.flatten_3d_to_2d and depth_dim > 1:
            for key, value in collated.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim == 1
                    and value.shape[0] == len(batch)
                ):
                    # [B] -> [B, D] -> [B*D]
                    collated[key] = value.unsqueeze(1).expand(-1, depth_dim).reshape(-1)

        # [CRITICAL] Final non-finite validation before returning the batch.
        #
        # This used to log at ERROR and then ``torch.nan_to_num(value, nan=0.0)``
        # and carry on. That contradicts the dataset layer, which declines the
        # very same substitution four times over with an explicit
        # "# DO NOT replace ... let dataset handle skipping"
        # (data/datasets/contrast_aware.py). Two disjoint policies for one
        # condition, and the silent one ran last and won: a corrupt volume was
        # rewritten to zeros mid-collate and trained on, indistinguishable from
        # real background. Raising is the repo's stated policy (pitfall #9), and
        # a dataloader worker is the right place to surface it — the sample can
        # be named.
        #
        # There were in fact TWO failure modes, because the old guard was gated
        # on ``torch.isnan(...).any()``: a batch carrying +/-inf and no NaN was
        # never examined at all and flowed through unchecked, while a batch with
        # both had its infinities clamped to the dtype maximum as a side effect
        # of ``nan_to_num``'s posinf/neginf defaults — unmentioned by the ERROR
        # line. Both are covered here.
        #
        # Opt out with ``data.collation.validate_nans: false`` -- that skips the
        # check entirely and restores pass-through, which is honest; it never
        # rewrites values behind the caller's back.
        if self.validate_nans:
            for key, value in collated.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if not value.is_floating_point() and not value.is_complex():
                    continue
                # Any NaN/inf makes the sum non-finite, so a finite sum proves the
                # tensor clean in one reduction; overflow only costs the exact pass.
                if bool(torch.isfinite(value.sum())):
                    continue
                finite = torch.isfinite(value)
                if bool(finite.all()):
                    continue
                bad = int((~finite).sum().item())
                nan_count = int(torch.isnan(value).sum().item())
                raise ValueError(
                    f"[COLLATE] Key {key!r} carries {bad}/{value.numel()} "
                    f"non-finite values ({nan_count} NaN, {bad - nan_count} +/-inf) "
                    f"in a batch of {len(batch)} sample(s); shape={tuple(value.shape)}, "
                    f"dtype={value.dtype}. The batch was NOT repaired: substituting "
                    "zeros here would train the model on fabricated background and "
                    "hide the corrupt source volume. Find the offending file (the "
                    "dataset layer logs the path on load) and exclude or re-export "
                    "it, or set data.collation.validate_nans: false to pass the "
                    "values through unmodified."
                )

        return collated

    def unpad(
        self,
        padded: torch.Tensor,
        padding_info: list[list[tuple[int, int]] | None],
    ) -> list[torch.Tensor]:
        """Restore original shapes by removing padding.

        Args:
            padded: Batched padded tensor (B, ...)
            padding_info: Padding amounts for each sample

        Returns:
            List of unpadded tensors
        """
        return [
            remove_padding(img, pad_info)
            for img, pad_info in zip(padded, padding_info, strict=False)
        ]


class SlabCollateStrategy(ImageCollateStrategy):
    """Collate 2.5D slab tensors.

    Flattens the depth dimension of 3D patches into channels for 2D models
    that require spatial context (2.5D).

    Transforms:
        Input:  (B, C, H, W, D) -> (B, C*D, H, W)
        Target: (B, C, H, W, D) -> (B, C, H, W) [Middle Slice]
    """

    def __init__(
        self,
        padding_mode: str = "zeros",
        padding_value: float = 0.0,
        flat_input: bool = True,
        target_mode: str = "middle",  # "middle", "flatten", "keep"
        **kwargs,
    ):
        """__init__.

        Args:
            padding_mode (str): Description.
            padding_value (float): Description.
            flat_input (bool): Description.
            target_mode (str): Description.
        """
        super().__init__(
            padding_mode=padding_mode,
            padding_value=padding_value,
            allow_variable_shapes=False,
            squeeze_depth_dim=False,  # We handle dimensionality explicitly
            **kwargs,  # forward remaining knobs (e.g. validate_nans) to parent
        )
        self.flat_input = flat_input
        self.target_mode = target_mode

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Use parent to stack tensors first: returns dict of [B, C, H, W, D] or [B, C, H, W]
        """collate.

        Args:
            batch (list[dict[str, Any]]): Description.
        Returns:
            dict[str, torch.Tensor]: Description.
        """
        collated = super().collate(batch)

        # Process 'input': Flatten depth to channels
        if self.flat_input and "input" in collated and collated["input"].ndim == 5:
            # [B, C, H, W, D] -> [B, C, D, H, W] -> [B, C*D, H, W]
            inp = collated["input"]
            B, C, H, W, D = inp.shape

            # Permute to [B, C, D, H, W]
            inp_perm = inp.permute(0, 1, 4, 2, 3)
            # Flatten C*D
            inp_flat = inp_perm.reshape(B, C * D, H, W)

            collated["input"] = inp_flat

        # Process 'target': Extract middle slice or flatten
        if "target" in collated and collated["target"].ndim == 5:
            tgt = collated["target"]
            D = tgt.shape[-1]

            if self.target_mode == "middle":
                mid_idx = D // 2
                collated["target"] = tgt[..., mid_idx]  # [B, C, H, W]
            elif self.target_mode == "flatten":
                B, C, H, W, D = tgt.shape
                tgt_perm = tgt.permute(0, 1, 4, 2, 3)
                collated["target"] = tgt_perm.reshape(B, C * D, H, W)
            elif self.target_mode == "keep":
                # Explicit no-op: the 5-D target rides through for a volumetric
                # decoder. It used to be reached by FALLING OFF the if/elif, so
                # an unknown value ("middl") silently kept as well — the arm
                # asked for the centre slice and was handed the whole slab,
                # with nothing said (#9).
                pass
            else:
                raise ValueError(
                    f"Unknown slab target_mode {self.target_mode!r}. Valid: "
                    "'middle' (centre depth slice), 'flatten' (depth folded "
                    "into channels), 'keep' (5-D target untouched). An "
                    "unrecognised value used to fall through to 'keep' "
                    "silently."
                )

        return collated


class GraphCollateStrategy(CollateStrategy):
    """Collate graph data (variable node/edge counts).

    Uses PyG's built-in batch operations for efficient batching.
    """

    def __init__(self, follow_batch: list[str] | None = None):
        """Initialize graph collation strategy.

        Args:
            follow_batch: Node/edge attributes to follow batch
        """
        self.follow_batch = follow_batch or []
        self._check_pyg_available()

    def _check_pyg_available(self) -> None:
        """Check that PyTorch Geometric is available."""
        try:
            import torch_geometric  # noqa: F401
        except ImportError:
            raise ImportError(
                "PyTorch Geometric required for graph collation. "
                "Install with: pip install torch-geometric"
            )

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate graph batch.

        Args:
            batch: List of dicts with "graph" key (PyG Data objects)

        Returns:
            Dict with batched graph
        """
        self.validate_batch(batch)

        try:
            from torch_geometric.data import Batch
        except ImportError:
            raise ImportError("PyTorch Geometric required for graph data")

        graphs = [item["graph"] for item in batch]

        # Use PyG's Batch class for proper graph batching
        batched_graph = Batch.from_data_list(
            graphs, follow_batch=self.follow_batch if self.follow_batch else None
        )

        return {
            "graph": batched_graph,
            "num_graphs": len(graphs),
            "original_sizes": [g.num_nodes for g in graphs],
        }


class SequenceCollateStrategy(CollateStrategy):
    """Collate variable-length sequences with masking.

    Handles:
    - Variable sequence lengths
    - Padding to max length
    - Attention masks for valid positions
    """

    def __init__(
        self,
        padding_mode: str = "zeros",
        max_length: int | None = None,
        pad_token_id: int = 0,
    ):
        """Initialize sequence collation strategy.

        Args:
            padding_mode: "zeros" or "repeat"
            max_length: Maximum sequence length (None = max in batch)
            pad_token_id: Token ID for padding
        """
        self.padding_mode = padding_mode
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate sequence batch with padding and masks.

        Args:
            batch: List of dicts with "sequence" key

        Returns:
            Dict with "sequences", "masks", "lengths"
        """
        self.validate_batch(batch)

        sequences = [item["sequence"] for item in batch]
        lengths = [len(seq) for seq in sequences]

        # Determine max length
        if self.max_length is None:
            max_len = max(lengths)
        else:
            max_len = min(self.max_length, max(lengths))

        padded = []
        masks = []

        for seq, length in zip(sequences, lengths, strict=False):
            # Truncate if longer than max_len
            if length > max_len:
                seq = seq[:max_len]
                length = max_len

            # Pad to max_len
            pad_amount = max_len - length
            if pad_amount > 0:
                if self.padding_mode == "zeros":
                    padding = torch.full(
                        (pad_amount,) + seq.shape[1:],
                        self.pad_token_id,
                        dtype=seq.dtype,
                        device=seq.device,
                    )
                    padded_seq = torch.cat([seq, padding], dim=0)
                elif self.padding_mode == "repeat":
                    # Repeat last token
                    last_token = seq[-1:]
                    padding = last_token.repeat(pad_amount, *([1] * (seq.ndim - 1)))
                    padded_seq = torch.cat([seq, padding], dim=0)
                else:
                    raise ValueError(f"Unknown padding mode: {self.padding_mode}")
            else:
                padded_seq = seq

            padded.append(padded_seq)

            # Create attention mask (1 for valid, 0 for padding)
            mask = torch.cat(
                [
                    torch.ones(length, dtype=torch.bool),
                    torch.zeros(pad_amount, dtype=torch.bool),
                ]
            )
            masks.append(mask)

        return {
            "sequences": torch.stack(padded),
            "attention_mask": torch.stack(masks),
            "lengths": torch.tensor(lengths),
        }


class MixedModalityCollateStrategy(CollateStrategy):
    """Collate mixed modality batches (images + graphs, etc.).

    Coordinates collation of multiple data types in single batch.
    """

    def __init__(self, strategies: dict[str, CollateStrategy]):
        """Initialize mixed modality collation.

        Args:
            strategies: Dict mapping modality name to CollateStrategy
        """
        self.strategies = strategies

    def collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate mixed modality batch.

        Args:
            batch: List of dicts with multiple modality keys

        Returns:
            Dict with collated data for each modality
        """
        self.validate_batch(batch)

        result = {}

        for modality_name, strategy in self.strategies.items():
            if modality_name in batch[0]:
                # Extract modality-specific samples
                modality_batch = [{modality_name: item[modality_name]} for item in batch]

                # Collate using strategy
                collated = strategy.collate(modality_batch)
                result[modality_name] = collated

        return result


# =============================================================================
# Helper Functions (Sample Filtering & Padding)
# =============================================================================


def _filter_none(batch: list[Any]) -> list[Any]:
    """Filter out None samples from batch."""
    return [b for b in batch if b is not None]


def _pad_and_stack_with_mask(
    tensors: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad tensors to max size and return stacked tensor with mask.

    Used by PhysicsCollateStrategy for manual tensor collation.
    """
    if not tensors:
        return torch.tensor([]), torch.tensor([])

    ndim = tensors[0].dim()
    shapes = [t.shape for t in tensors]

    # 1D: [N]
    if ndim == 1:
        max_len = max(s[0] for s in shapes)
        padded = []
        masks = []
        for t in tensors:
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                t_p = torch.nn.functional.pad(t, (0, pad_len), value=0)
                mask = torch.cat(
                    [
                        torch.ones(t.shape[0], dtype=torch.bool),
                        torch.zeros(pad_len, dtype=torch.bool),
                    ]
                )
            else:
                t_p = t
                mask = torch.ones(t.shape[0], dtype=torch.bool)
            padded.append(t_p)
            masks.append(mask)
        return torch.stack(padded), torch.stack(masks)

    # 2D case [N, C] (dim 0 varies) or [C, N] (dim 1 varies)
    if ndim == 2:
        dim0_varies = any(s[0] != shapes[0][0] for s in shapes)
        if dim0_varies:
            max_len = max(s[0] for s in shapes)
            padded = []
            masks = []
            for t in tensors:
                pad_len = max_len - t.shape[0]
                if pad_len > 0:
                    t_p = torch.nn.functional.pad(t, (0, 0, 0, pad_len), value=0)
                    mask = torch.cat(
                        [
                            torch.ones(t.shape[0], dtype=torch.bool),
                            torch.zeros(pad_len, dtype=torch.bool),
                        ]
                    )
                else:
                    t_p = t
                    mask = torch.ones(t.shape[0], dtype=torch.bool)
                padded.append(t_p)
                masks.append(mask)
            return torch.stack(padded), torch.stack(masks)

        dim1_varies = any(s[1] != shapes[0][1] for s in shapes)
        if dim1_varies:
            max_len = max(s[1] for s in shapes)
            padded = []
            masks = []
            for t in tensors:
                pad_len = max_len - t.shape[1]
                if pad_len > 0:
                    t_p = torch.nn.functional.pad(t, (0, pad_len), value=0)
                    mask = torch.cat(
                        [
                            torch.ones(t.shape[1], dtype=torch.bool),
                            torch.zeros(pad_len, dtype=torch.bool),
                        ]
                    )
                else:
                    t_p = t
                    mask = torch.ones(t.shape[1], dtype=torch.bool)
                padded.append(t_p)
                masks.append(mask)
            return torch.stack(padded), torch.stack(masks)

    # Fallback: Pad last dimension
    max_size = max(t.shape[-1] for t in tensors)
    padded = []
    masks = []

    for t in tensors:
        n = t.shape[-1]
        pad_size = max_size - n

        if pad_size > 0:
            pad_arg = [0] * (2 * ndim)
            pad_arg[1] = pad_size
            t_padded = torch.nn.functional.pad(t, tuple(pad_arg), value=0)

            mask = torch.cat(
                [
                    torch.ones(n, dtype=torch.bool),
                    torch.zeros(pad_size, dtype=torch.bool),
                ]
            )
        else:
            t_padded = t
            mask = torch.ones(n, dtype=torch.bool)

        padded.append(t_padded)
        masks.append(mask)

    return torch.stack(padded, dim=0), torch.stack(masks, dim=0)


# =============================================================================
# Imported Strategies
# =============================================================================


class RobustCollateStrategy(CollateStrategy):
    """Robust collation for general-purpose datasets.

    Filters out None samples (corrupt/failed loads) and uses standard
    PyTorch collate. Suitable for most supervised learning tasks.
    """

    def collate(self, batch: list[Any]) -> dict | None:
        """Collate with None filtering using default PyTorch collate."""
        clean_batch = _filter_none(batch)

        if not clean_batch:
            return None

        # [NEW] Handle TorchIO Subject/Image objects by unwrapping tensors
        # This prevents pin_memory copy failures (ValueError: ScalarImage type...)
        # We check the first element to decide on processing
        first = clean_batch[0]

        # Scenario 1: Batch of Subject-like dicts containing Images
        if isinstance(first, dict) or hasattr(first, "keys"):
            new_clean_batch = []
            for item in clean_batch:
                # If it's a TrainingBatch dataclass, handle separately
                from spectramr.data.batch_types import TrainingBatch

                if isinstance(item, TrainingBatch):
                    d = {
                        "input": item.input,
                        "target": item.target,
                        "metadata": item.metadata,
                    }
                    if item.mask is not None:
                        d["mask"] = item.mask
                    item = d

                # Unwrap any TorchIO images in the dict
                new_item = {}
                for k, v in item.items():
                    # Handle ScalarImage/LabelMap
                    if hasattr(v, "data") and hasattr(v, "affine"):
                        new_item[k] = v.data
                    else:
                        new_item[k] = v
                new_clean_batch.append(new_item)
            clean_batch = new_clean_batch

        # Scenario 2: Batch of single TorchIO Image objects
        elif hasattr(first, "data") and hasattr(first, "affine"):
            clean_batch = [item.data for item in clean_batch]

        from torch.utils.data._utils.collate import default_collate

        return default_collate(clean_batch)


class PhysicsCollateStrategy(CollateStrategy):
    """Collation for MRI physics objects.

    Handles:
    - Dict samples with physics keys (trajectory, dcf, time_vec, b0_map)
    - Tuple returns (trajectory, dcf, time_vec)
    - Mixed tensor types and variable-size trajectories
    - Complex-valued k-space data
    """

    def collate(self, batch: list[dict | tuple]) -> dict | tuple | torch.Tensor | None:
        """collate.

        Args:
            batch (list[Union[dict, tuple]]): Description.
        Returns:
            Union[dict, tuple, torch.Tensor, None]: Description.
        """
        if not batch:
            return None

        clean_batch = _filter_none(batch)

        if not clean_batch:
            return None

        elem = clean_batch[0]

        # Dispatch based on element type
        if isinstance(elem, dict):
            return self._collate_physics_dict(clean_batch)
        elif isinstance(elem, tuple):
            return self._collate_physics_tuple(clean_batch)
        elif isinstance(elem, torch.Tensor):
            return self._collate_physics_tensor(clean_batch)
        else:
            # Fallback
            from torch.utils.data._utils.collate import default_collate

            return default_collate(clean_batch)

    def _collate_physics_dict(self, batch: list[dict]) -> dict:
        """Collate physics dict samples with tuple and tensor support."""
        result = {}

        for key in batch[0].keys():
            values = [b[key] for b in batch if key in b]

            if not values:
                continue

            first = values[0]

            # Handle tuples (e.g., from accelerator returns)
            if isinstance(first, tuple):
                result[key] = tuple(self.collate([v[i] for v in values]) for i in range(len(first)))
            elif isinstance(first, torch.Tensor):
                try:
                    result[key] = torch.stack(values, dim=0)
                except RuntimeError:
                    # Variable size: pad
                    result[key], result[f"{key}_mask"] = _pad_and_stack_with_mask(values)
            else:
                # Non-tensor: collect as list
                result[key] = values

        return result

    def _collate_physics_tuple(self, batch: list[tuple]) -> tuple:
        """Collate tuple samples (trajectory, dcf, time_vec)."""
        n_elements = len(batch[0])
        result = []

        for i in range(n_elements):
            elements = [b[i] for b in batch]

            if isinstance(elements[0], torch.Tensor):
                try:
                    result.append(torch.stack(elements, dim=0))
                except RuntimeError:
                    # Variable size
                    padded, _ = _pad_and_stack_with_mask(elements)
                    result.append(padded)
            else:
                result.append(elements)

        return tuple(result)

    def _collate_physics_tensor(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """Collate plain tensor samples with variable-size support."""
        try:
            return torch.stack(batch, dim=0)
        except RuntimeError:
            # Variable size: pad
            padded, _ = _pad_and_stack_with_mask(batch)
            return padded


class CollateStrategyFactory:
    """Factory for creating appropriate collation strategies."""

    STRATEGIES = {
        "image": ImageCollateStrategy,
        "graph": GraphCollateStrategy,
        "sequence": SequenceCollateStrategy,
        "mixed": MixedModalityCollateStrategy,
        "robust": RobustCollateStrategy,
        "physics": PhysicsCollateStrategy,
        "slab": SlabCollateStrategy,
    }

    @classmethod
    def create(cls, data_type: str, **kwargs: Any) -> CollateStrategy:
        """Create collation strategy for data type.

        Args:
            data_type: Type of data ("image", "graph", etc.)
            **kwargs: Arguments for strategy __init__

        Returns:
            Configured CollateStrategy instance

        Raises:
            ValueError: If data_type not recognized
        """
        if data_type not in cls.STRATEGIES:
            raise ValueError(
                f"Unknown data type: {data_type}. Supported: {list(cls.STRATEGIES.keys())}"
            )

        strategy_class = cls.STRATEGIES[data_type]
        return strategy_class(**kwargs)

    @classmethod
    def register(cls, name: str, strategy_class: type) -> None:
        """Register custom collation strategy.

        Args:
            name: Strategy name
            strategy_class: CollateStrategy subclass
        """
        cls.STRATEGIES[name] = strategy_class


# =============================================================================
# Convenience Functions
# =============================================================================


def squeezing_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Convenience collate function that applies dimension squeezing and flattening.

    This function uses ImageCollateStrategy with flatten_3d_to_2d=True
    to automatically flatten 5D volumetric tensors [B, C, H, W, D] into
    4D slice tensors [B*D, C, H, W]. This ensures dimension agnosticism
    for 2D models.

    Args:
        batch: List of sample dicts from dataset

    Returns:
        Collated batch with flattened dimensions

    See Also:
        - docs/SHAPE_CONTRACTS.md
        - ImageCollateStrategy with flatten_3d_to_2d=True
    """
    strategy = ImageCollateStrategy(
        padding_mode="zeros",
        padding_value=0.0,
        allow_variable_shapes=False,
        squeeze_depth_dim=False,
        flatten_3d_to_2d=True,
    )
    return strategy.collate(batch)


__all__ = [
    "CollateStrategy",
    "CollateStrategyFactory",
    "GraphCollateStrategy",
    "ImageCollateStrategy",
    "MixedModalityCollateStrategy",
    "PhysicsCollateStrategy",
    "RobustCollateStrategy",
    "SequenceCollateStrategy",
    "SlabCollateStrategy",
    "squeezing_collate",
]
