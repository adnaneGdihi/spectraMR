from dataclasses import dataclass, field
from typing import Any

import torch

from spectramr.config.schemas.enums import Axis

#: Re-exported from ``core.coil_map_names``, the one owner. It moved there so the
#: loss-kwarg boundary could reconcile the same five spellings -- ``models/``
#: cannot import ``data/``, so a table here could serve only one of the two.
from spectramr.core.coil_map_names import COIL_MAP_ALIASES


@dataclass
class TrainingBatch:
    """Standardized batch format for all training data.

    This replaces the loose dictionary format used previously.
    All datasets and collate functions should eventually produce this structure.

    .. mermaid::

        classDiagram
            class TrainingBatch {
                +input: Tensor
                +target: Tensor
                +mask: Tensor?
                +metadata: Dict
                +to(device)
            }
    """

    input: torch.Tensor  # The model input (e.g., undersampled k-space or LR image)
    target: torch.Tensor  # The ground truth (e.g., full k-space, HR image)
    mask: torch.Tensor | None = None  # Sampling mask (if applicable)
    # Per-coil sensitivity maps. First-class rather than a metadata entry because
    # they travelled under five names and every consumer's presence check missed
    # (C11) -- see COIL_MAP_ALIASES. `None` means the batch genuinely has none.
    coil_maps: torch.Tensor | None = None
    # Which NON-SPATIAL axes this batch's tensors carry (C8). Not a tensor: a
    # claim ABOUT the tensors, resolved once from the config by
    # ``axis_exposure.resolve_axes_for`` and carried alongside them.
    #
    # 5-D (time x contrast, echo x time) is unrepresentable today at the
    # TENSOR-CONTRACT layer, not for want of a loader: both multi-frame loaders
    # fold their extra axis into CHANNEL, and once folded nothing downstream can
    # tell a 3-frame cine from a 3-channel image. A consumer cannot unfold what
    # it cannot identify, so axis identity has to travel with the batch before a
    # second non-spatial axis is meaningful.
    #
    # ``None`` means "unresolved -- cannot vouch for the axes" and consumers must
    # SKIP, never read it as "no axes". ``frozenset()`` is the opposite: a
    # positive claim that there is no non-spatial axis. The whole axis system
    # turns on that distinction; see ``resolve_axes_for``.
    axes: frozenset[Axis] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)  # Filenames, slice indices, etc.

    def to(self, device: str | torch.device, non_blocking: bool = False) -> "TrainingBatch":
        """Move all tensors in the batch to the specified device.

        Handles both torch.Tensor objects and TorchIO Image objects.
        # TorchIO Images store tensors in .tensor attribute but don't have .to() method.
        For batched data, extracts tensors and returns them (images are single-sample only).
        """

        def move_to_device(obj, device, non_blocking):
            """Helper to move various object types to device."""
            if obj is None:
                return None

            # Handle dictionaries recursively
            if isinstance(obj, dict):
                return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}

            # Handle lists/tuples recursively
            if isinstance(obj, list):
                return [move_to_device(v, device, non_blocking) for v in obj]
            if isinstance(obj, tuple):
                return tuple(move_to_device(v, device, non_blocking) for v in obj)

            # Handle TorchIO Image objects (have .tensor attribute but no .to())
            if hasattr(obj, "tensor") and not hasattr(obj, "to"):
                # Extract tensor and move it
                # Note: Don't reconstruct Image object - batches may have extra dimensions
                tensor = obj.tensor
                return tensor.to(device, non_blocking=non_blocking)

            # Handle torch.Tensor objects (have .to() method)
            if hasattr(obj, "to") and callable(obj.to):
                return obj.to(device, non_blocking=non_blocking)

            # Return object as-is if it can't be moved
            return obj

        return TrainingBatch(
            input=move_to_device(self.input, device, non_blocking),
            target=move_to_device(self.target, device, non_blocking),
            mask=move_to_device(self.mask, device, non_blocking),
            coil_maps=move_to_device(self.coil_maps, device, non_blocking),
            # Carried, not moved: `axes` is a claim about the tensors, not a
            # tensor. Omitting it here would make a batch lose its axis identity
            # at the device boundary -- every consumer runs AFTER `.to()`, so the
            # field would read `None` at literally every point that reads it.
            axes=self.axes,
            metadata={k: move_to_device(v, device, non_blocking) for k, v in self.metadata.items()},
        )

    def __getitem__(self, key: int | str) -> Any:
        """
        Allow unpacking as (input, target) for legacy compatibility.
        Also support dict-like access for backward compatibility with tests.
        """
        # Integer indexing for tuple unpacking
        if isinstance(key, int):
            if key == 0:
                return self.input
            elif key == 1:
                return self.target
            else:
                raise IndexError("TrainingBatch only supports unpacking (input, target)")

        # String key access for dict-like interface
        if isinstance(key, str):
            # STRICT: Only canonical keys are permitted
            # Legacy aliases removed per codebase standardization
            canonical_keys = {"input", "target", "mask"}

            # Direct attribute access for canonical keys
            if key in canonical_keys:
                value = getattr(self, key, None)
                if value is not None:
                    return value

            # Every coil-map spelling resolves to the one field (C11). Consumers
            # keep their own name; the reconciliation lives here.
            if key in COIL_MAP_ALIASES and self.coil_maps is not None:
                return self.coil_maps

            # Fall back to metadata
            if key in self.metadata:
                return self.metadata[key]

            # If not found, raise KeyError (dict-like behavior)
            raise KeyError(
                f"Key '{key}' not found in TrainingBatch. Use canonical keys: {canonical_keys}"
            )

        raise TypeError(f"TrainingBatch indices must be int or str, not {type(key).__name__}")

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for checking key existence."""
        # STRICT: Only canonical keys are permitted
        canonical_keys = {"input", "target", "mask"}

        if key in canonical_keys:
            attr_value = getattr(self, key, None)
            return attr_value is not None

        # Must mirror __getitem__ exactly. A presence check that disagrees with
        # the read is how the SENSE term came to run coil-blind in the first
        # place: `"sensitivity_maps" in batch` answered False while the maps
        # were sitting in metadata under another name.
        if key in COIL_MAP_ALIASES:
            return self.coil_maps is not None

        return key in self.metadata

    def get(self, key: str, default: Any = None) -> Any:
        """Support dict.get() method for safe access."""
        try:
            return self[key]
        except (KeyError, IndexError):
            return default


def read_batch_field(batch: Any, *names: str, default: Any = None) -> Any:
    """Read the first non-``None`` field among ``names`` off a batch of any shape.

    **Use this instead of ``isinstance(batch, dict)`` plus a ``hasattr``
    fallback.** That pairing is the single most repeated defect at this seam and
    it fails in a way that reads as working: for a :class:`TrainingBatch` the
    ``isinstance`` leg is False (it is a dataclass, not a mapping) *and* the
    ``hasattr`` leg is also False, because every non-core key lives in
    ``.metadata`` and attribute lookup cannot see it. Both legs miss, the caller
    receives ``None``, and a marker the batch really did publish is read as
    absent — a silent default substitution (non-negotiable 3b).

    Concretely, on a batch carrying ``kspace_scale`` and ``kspace_normalized``::

        isinstance(batch, dict)              -> False
        hasattr(batch, "kspace_scale")       -> False   # metadata is invisible
        "kspace_scale" in batch              -> True
        batch.get("kspace_scale")            -> tensor(224.3590)

    The mapping protocol is therefore the *only* path that reaches metadata, and
    :class:`TrainingBatch` implements it (``__getitem__`` / ``__contains__`` /
    ``get``) precisely so this function can be shape-agnostic. This mirrors
    ``pipelines.train.select_validation_extra_fields``, which fixed the same bug
    class on the validation-forwarding seam.

    ``names`` is variadic so alias families resolve in one call with an explicit
    precedence order (``mask`` before ``acceleration_mask``), rather than each
    consumer re-implementing the fallback chain.

    The trailing ``getattr`` leg is deliberate and is **not** a silent fallback:
    it serves shapes that are neither mapping nor batch — argparse namespaces,
    plain config objects, and the ``Mock`` batches that unit tests feed in. It is
    reached only when the mapping protocol is genuinely unavailable, so it cannot
    mask a metadata read.

    Args:
        batch: A dict, a :class:`TrainingBatch`, or any object with attributes.
            ``None`` is accepted and yields ``default``.
        *names: Field names to try, in precedence order.
        default: Returned when no name resolves to a non-``None`` value.

    Returns:
        The first non-``None`` value found, else ``default``. A published
        ``False``/``0`` is a value and is returned as-is — only ``None`` (and a
        genuinely absent key) falls through to the next name.
    """
    if batch is None:
        return default

    # dict and TrainingBatch both implement the mapping read protocol. For
    # TrainingBatch, `.get` is the ONLY access path that reaches `.metadata`;
    # `.get` already swallows the KeyError that `__getitem__` raises for an
    # unknown key, so no membership pre-check is needed.
    if isinstance(batch, dict | TrainingBatch):
        for name in names:
            value = batch.get(name)
            if value is not None:
                return value
        return default

    for name in names:
        value = getattr(batch, name, None)
        if value is not None:
            return value
    return default


def align_scale_to_batch(
    scale: Any,
    batch_size: int,
    *,
    field: str = "scale",
    device: Any = None,
) -> torch.Tensor:
    """Reconcile a per-sample batch field with the batch axis it will multiply.

    A per-sample scalar published by the data layer -- ``kspace_scale`` above
    all -- is **per subject**. By the time a validation consumer multiplies it
    against a prediction, depth has usually been folded into the batch axis, so
    the tensor is **per slice**. The two lengths differ by exactly the depth
    factor and nothing upstream reconciles them::

        val batch from the loader      input [B=2, C, H, W, D=18]   kspace_scale [2]
        after the 5D->4D flatten       input [36, C, H, W]          kspace_scale [2]
        hr_fakes * kspace_scale        RuntimeError: 36 vs 2

    The reshape this replaces adopted the *producer's* length blindly
    (``scale.view(-1, 1, 1, 1)``), so a length-2 scale met a length-36
    prediction. Only the ``ndim == 0`` arm of that ladder expanded to the
    consumer's batch size; the ``ndim == 1``, ``ndim < 4`` and pass-through arms
    all kept the producer's, which is why a scalar scale worked and a per-sample
    one crashed.

    ``repeat_interleave``, **not** ``repeat``/``tile``, is the correct expansion,
    and getting it backwards is worse than the crash. Every 5D->4D flatten in
    this repository keeps the batch axis first and the depth axis second before
    reshaping -- ``permute(0, 4, 1, 2, 3)`` and ``permute(0, 2, 1, 3, 4)``
    alike -- so the flattened index is ``b * D + d``: subject-major, all of
    subject 0's slices, then all of subject 1's. ``repeat`` would tile
    ``[s0, s1, s0, s1, ...]`` and silently grade subject 1's slices in subject
    0's units. Same shape, no error, wrong metrics.

    **What this can and cannot check.** By the time a consumer needs the scale
    the tensor is already 4-D, so ``D`` is unrecoverable and divisibility is the
    strongest available test: ``36 % 2 == 0`` cannot distinguish a genuine
    ``B=2, D=18`` flatten from a wrong length that happens to divide. It is
    still strictly stronger than adopting the producer's length unchecked, and a
    non-divisible length -- the case that has no benign reading -- raises with
    both shapes named instead of surfacing 40 frames downstream as a bare
    ``RuntimeError`` at a multiply (non-negotiable 3: no silent fallbacks).

    Args:
        scale: The published per-sample scale. A tensor, or anything
            ``torch.as_tensor`` accepts (float, int, list).
        batch_size: The leading dimension of the tensor this scale will
            multiply -- read it off that tensor, never off the batch.
        field: Name of the field, used only in error messages so a raise is
            attributable to a producer.
        device: Optional device to move the scale to. dtype is deliberately
            left alone: callers cast to their own tensor's dtype where that
            matters, and silently adopting it would downcast a scale under AMP.

    Returns:
        A ``(batch_size, 1, 1, 1)`` tensor, broadcast-ready against
        ``(batch_size, C, H, W)``.

    Raises:
        ValueError: If the scale carries non-singleton spatial extent, is
            empty, or has a length that does not divide ``batch_size``.
    """
    if not torch.is_tensor(scale):
        scale = torch.as_tensor(scale)
    if device is not None:
        scale = scale.to(device)

    if scale.ndim == 0:
        return scale.reshape(1, 1, 1, 1).expand(batch_size, 1, 1, 1)

    # Trailing singleton axes are just the (N, 1, 1, 1) shape an earlier caller
    # gave it. A trailing axis with real extent is a scale *map*, not a
    # per-sample scalar, and flattening it would reinterpret spatial structure
    # as batch entries.
    if scale.ndim > 1 and any(size != 1 for size in scale.shape[1:]):
        raise ValueError(
            f"Cannot align {field!r} of shape {tuple(scale.shape)} to a batch of "
            f"{batch_size}: every axis after the batch axis must be singleton for "
            f"a per-sample scale. A scale with spatial extent needs a producer-side "
            f"fix, not a reshape here."
        )

    flat = scale.reshape(-1)
    published = flat.numel()

    if published == batch_size:
        return flat.view(batch_size, 1, 1, 1)

    if published > 0 and batch_size % published == 0:
        # Subject-major, per the flatten order documented above.
        return flat.repeat_interleave(batch_size // published, dim=0).view(batch_size, 1, 1, 1)

    raise ValueError(
        f"{field!r} has {published} entries but the tensor it must scale has a "
        f"batch of {batch_size}, which {published} does not divide. The published "
        f"scale is per subject and the tensor is per slice, so a whole number of "
        f"slices per subject is required; this batch has no such factor. Check "
        f"whether the producer dropped or padded samples."
    )


class BatchAdapter:
    """
    Adapter to convert dictionary batches to TrainingBatch objects.

    STRICT: Requires canonical keys (input, target, mask).
    Legacy keys are NOT supported - update your data pipeline to use canonical naming.
    """

    @staticmethod
    def from_dict(
        batch_data: dict[str, Any], axes: frozenset[Axis] | None = None
    ) -> "TrainingBatch":
        """Convert a collated batch dict from SubjectsLoader to a TrainingBatch.

        ``tio.SubjectsLoader`` wraps every ``tio.ScalarImage``/``tio.LabelMap``
        field into ``{"data": Tensor[B, C, H, W, D], "affine": Tensor}``.
        This method unwraps those nested dicts to plain tensors so that
        ``TrainingBatch.input`` and ``.target`` are always
        ``Tensor[B, C, H, W, D]`` (or squeezed to 4-D by ``_to_device``).

        REQUIRES:
        - ``"input"``  canonical key
        - ``"target"`` canonical key

        Coil sensitivity maps are bound to ``TrainingBatch.coil_maps`` under
        whichever of :data:`COIL_MAP_ALIASES` the producer used, so a consumer
        asking by any of the five spellings finds them (C11).

        All other keys are stored in ``metadata``.

        ``axes`` is passed IN rather than derived here, because nothing at this
        boundary knows the arm: the collated dict carries tensors, not a
        ``dataset_type``. The caller resolves it once from the config with
        ``axis_exposure.resolve_axes_for`` and hands it down. Defaulting to
        ``None`` keeps every existing call site behaving exactly as before --
        unresolved, therefore skipped -- rather than silently claiming the batch
        has no non-spatial axis.

        Raises:
            ValueError: If ``"input"`` or ``"target"`` keys are missing, or if
                two different coil-map spellings are present with conflicting
                values.
        """

        def _extract_tio(value: Any) -> Any:
            """Unwrap TorchIO nested image dict to its ``"data"`` tensor."""
            if isinstance(value, dict) and "data" in value:
                t = value["data"]
                if isinstance(t, torch.Tensor):
                    return t
            return value

        # Core canonical keys. The coil aliases join them so a bound map is not
        # ALSO duplicated into metadata under its original name.
        core_keys = {"input", "target", "mask", *COIL_MAP_ALIASES}

        if "input" not in batch_data or "target" not in batch_data:
            raise ValueError(
                f"BatchAdapter requires canonical keys ('input', 'target'). "
                f"Got keys: {list(batch_data.keys())}. "
                f"Update your data pipeline to use canonical naming."
            )

        # Resolve the coil maps from whichever spelling arrived. Two DIFFERENT
        # spellings both present is a producer disagreeing with itself: pick
        # neither, because guessing which is authoritative is exactly the kind
        # of silent choice that produced the five-name split.
        present = [k for k in COIL_MAP_ALIASES if batch_data.get(k) is not None]
        if len(present) > 1:
            first = _extract_tio(batch_data[present[0]])
            for other in present[1:]:
                candidate = _extract_tio(batch_data[other])
                same = (
                    isinstance(first, torch.Tensor)
                    and isinstance(candidate, torch.Tensor)
                    and first.shape == candidate.shape
                    and torch.equal(first, candidate)
                )
                if not same:
                    raise ValueError(
                        f"Batch carries conflicting coil sensitivity maps under "
                        f"{present!r}. These are aliases for one tensor "
                        "(TrainingBatch.coil_maps); emitting two different "
                        "values means the producer disagrees with itself about "
                        "which coils the sample was acquired with. Emit exactly "
                        "one."
                    )
        coil_maps = _extract_tio(batch_data[present[0]]) if present else None

        return TrainingBatch(
            coil_maps=coil_maps,
            axes=axes,
            input=_extract_tio(batch_data["input"]),
            target=_extract_tio(batch_data["target"]),
            mask=_extract_tio(batch_data.get("mask")),
            metadata={k: v for k, v in batch_data.items() if k not in core_keys},
        )
