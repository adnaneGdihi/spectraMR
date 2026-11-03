"""Universal / foundation reconstruction strategy (PR-7).

The generator is conditioned on the scan's physical acquisition parameters (TR, TE, flip
angle, B0) rather than on learned prompt tokens, so one model can serve mixed contrast,
field-strength and acceleration cohorts.

The strategy only carries the prompt: it stacks the per-sample ``acquisition_metadata`` dict
that ``data.acquisition_metadata`` emits into a ``[B, 4]`` vector and hands it to the
generator under :data:`PROMPT_KWARG`, in training, validation and the validation image
capture alike. The embedding and the LoRA adapters that consume it belong to the
prompt-aware backbone (``prompt_lora_hermitian_fno``), so they are optimised and
checkpointed with the generator. Everything else, including the declared ``losses:`` block,
runs through :class:`ReconstructionTrainingStrategy` unchanged.

Registration::

    "universal_reconstruction": "spectramr.infrastructure.training.strategies.universal_reconstruction_strategy.UniversalReconstructionStrategy",
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from spectramr.core.module_utils import unwrap_model
from spectramr.data.batch_types import read_batch_field
from spectramr.models.generators.prompt_lora_fno import PROMPT_KWARG

from .mixins.reconstruction import _as_aux_mapping
from .mixins.utils import _names_parameter
from .reconstruction import ReconstructionTrainingStrategy, _batch_contrast_id

#: Batch key of the per-sample header dict ``data.acquisition_metadata`` attaches.
ACQUISITION_METADATA_KEY = "acquisition_metadata"


def acquisition_prompt(
    metadata: Any, fields: tuple[str, ...], rows: int, device: torch.device
) -> torch.Tensor:
    """Stack the collated ``acquisition_metadata`` into a finite ``[rows, len(fields)]`` vector.

    The collate yields one dict of columns, or one dict per sample; both are accepted.

    Raises:
        KeyError: when the batch carries no metadata or a field is absent.
        ValueError: on a row-count mismatch or a non-finite value (the lenient reader's NaN).
    """
    if (
        isinstance(metadata, list | tuple)
        and metadata
        and all(isinstance(m, Mapping) for m in metadata)
    ):
        metadata = {k: [m.get(k) for m in metadata] for k in fields}
    if not isinstance(metadata, Mapping):
        raise KeyError(
            f"universal_reconstruction needs the batch's '{ACQUISITION_METADATA_KEY}' dict to "
            "build the acquisition prompt; enable data.acquisition_metadata with fields "
            f"{list(fields)}."
        )
    columns = []
    for name in fields:
        if name not in metadata or metadata[name] is None:
            raise KeyError(
                f"'{ACQUISITION_METADATA_KEY}' lacks '{name}' (has {sorted(metadata)!r}); list "
                "it under data.acquisition_metadata.fields."
            )
        columns.append(torch.as_tensor(metadata[name], dtype=torch.float32).reshape(-1))
    prompt = torch.stack(columns, dim=-1)
    if prompt.shape[0] != rows:
        raise ValueError(
            f"acquisition prompt has {prompt.shape[0]} row(s) for {rows} input row(s); "
            "refusing to broadcast a scan's parameters onto rows it was not recorded for."
        )
    message = (
        "acquisition prompt is not finite: the metadata source gave no value for one of "
        f"{list(fields)} (strict=false falls back to NaN). Set "
        "data.acquisition_metadata.strict: true and point source at the scan header."
    )
    finite = torch.isfinite(prompt).all()
    if prompt.device.type == "cpu":
        if not bool(finite):
            raise ValueError(message)
    else:
        # The loop may already have moved the metadata to the GPU: assert without a host sync.
        torch._assert_async(finite, message)
    return prompt.to(device, non_blocking=True)


class UniversalReconstructionStrategy(ReconstructionTrainingStrategy):
    """Reconstruction whose generator consumes a physics-derived acquisition prompt."""

    parallel_compat = frozenset({"replicated"})

    #: Acquisition-metadata fields of the prompt, in the order the backbone receives them.
    PROMPT_FIELDS: tuple[str, ...] = ("TR", "TE", "FA", "B0")

    def _prepare_batch_context(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        context = super()._prepare_batch_context(input_batch, target_batch, **kwargs)
        metadata = _as_aux_mapping(kwargs.get("batch", {})).get(ACQUISITION_METADATA_KEY)
        if metadata is not None:
            context[ACQUISITION_METADATA_KEY] = metadata
        return context

    def _prepare_generator_inputs(
        self,
        batch_context: dict[str, Any],
        input_batch: torch.Tensor,
        *,
        validation: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        lr_image, forward_kwargs = super()._prepare_generator_inputs(
            batch_context, input_batch, validation=validation
        )
        forward = getattr(unwrap_model(self.env.generator), "forward", None)
        if not _names_parameter(forward, PROMPT_KWARG):
            raise ValueError(
                f"universal_reconstruction conditions the generator on '{PROMPT_KWARG}', but "
                f"{type(unwrap_model(self.env.generator)).__name__}.forward does not take it, so "
                "the prompt would be inert. Use a prompt-aware backbone "
                "(model_type: prompt_lora_hermitian_fno) or training_mode: reconstruction."
            )
        forward_kwargs[PROMPT_KWARG] = acquisition_prompt(
            batch_context.get(ACQUISITION_METADATA_KEY),
            self.PROMPT_FIELDS,
            lr_image.shape[0],
            lr_image.device,
        )
        return lr_image, forward_kwargs

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validate under the batch's own prompt, as the training step conditions."""
        if batch_data is not None:
            context = dict(
                kwargs.get("batch_context") or {"use_dc": False, "measured_kspace": None}
            )
            if "contrast_id" not in context:
                contrast = _batch_contrast_id(batch_data, input_batch)
                if contrast is not None:
                    context["contrast_id"] = contrast
            metadata = read_batch_field(batch_data, ACQUISITION_METADATA_KEY)
            if metadata is not None:
                context[ACQUISITION_METADATA_KEY] = metadata
            kwargs["batch_context"] = context
        return super().validation_step(input_batch, target_batch, batch_data=batch_data, **kwargs)
