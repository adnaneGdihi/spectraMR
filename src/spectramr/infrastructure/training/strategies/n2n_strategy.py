"""Noise2Noise self-supervised denoising strategy.

Trains on pairs of independent noisy observations of the same signal. For a
squared loss and a target whose noise is zero-mean and independent of the input,

    E||f(y1) - y2||^2 = E||f(y1) - x||^2 + E||n2||^2,

so the minimiser is the supervised one although no clean target exists
(Lehtinen et al., ICML 2018). The M4Raw loader serves no target for this
strategy: it publishes a repetition stack (``data.n2n_pairing``) whose slot 0 is
the input, and the strategy forms the target from the remaining slots.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from spectramr.config.schemas.enums import Regime, Task
from spectramr.models.capabilities import StrategyCapabilities

from .base import BaseTrainingStrategy
from .mixins.model_validation import ModelValidationMixin
from .mixins.reconstruction import ReconstructionMixin
from .reconstruction import ReconstructionStepMixin

#: The batch key the loader publishes the stack under (``m4raw_dataset.REPETITION_STACK_KEY``).
_STACK_KEY = "kspace_reps"

#: Pairings this strategy implements; the schema validates the YAML against the same set.
_PAIRINGS = frozenset({"random_pair", "one_vs_rest_mean", "ordered"})


def _data_tensor(item: Any) -> torch.Tensor | None:
    """The tensor of a collated TorchIO image (``{"data": ...}``) or a bare tensor."""
    if isinstance(item, dict):
        item = item.get("data")
    return item if isinstance(item, torch.Tensor) else None


def pair_from_stack(
    input_batch: torch.Tensor,
    stack: torch.Tensor,
    pairing: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """The training target for ``input_batch`` drawn from its repetition stack.

    Args:
        input_batch: ``[B, C, ...]``, equal to slot 0 of the stack.
        stack: ``[B, N * C, ...]``, the loader's slots concatenated on the channel axis.
        pairing: ``random_pair`` (one other slot per sample, uniformly),
            ``one_vs_rest_mean`` (mean of the other slots) or ``ordered`` (slot 1).
        generator: Optional RNG for ``random_pair``.

    Returns:
        ``[B, C, ...]`` target, never slot 0.
    """
    if pairing not in _PAIRINGS:
        raise ValueError(f"Unknown n2n pairing {pairing!r}; valid: {sorted(_PAIRINGS)}")
    channels = input_batch.shape[1]
    if stack.shape[1] % channels:
        raise ValueError(
            f"repetition stack has {stack.shape[1]} channels, not a multiple of the "
            f"input's {channels}"
        )
    n_slots = stack.shape[1] // channels
    if n_slots < 2:
        raise ValueError("Noise2Noise needs at least two slots in the repetition stack")
    slots = stack.reshape(stack.shape[0], n_slots, channels, *stack.shape[2:])
    others = slots[:, 1:]
    if pairing == "ordered":
        if n_slots != 2:
            raise ValueError(f"'ordered' expects the two halves of a split, got {n_slots} slots")
        return others[:, 0]
    if pairing == "one_vs_rest_mean":
        return others.mean(dim=1)
    pick = torch.randint(0, n_slots - 1, (stack.shape[0],), device=stack.device, generator=generator)
    index = pick.view(-1, 1, *([1] * (others.dim() - 2))).expand(-1, 1, *others.shape[2:])
    return others.gather(1, index).squeeze(1)


class NoiseToNoiseStrategy(
    ReconstructionStepMixin, ModelValidationMixin, ReconstructionMixin, BaseTrainingStrategy
):
    """Noise2Noise denoising: the reconstruction step on a target drawn from a repetition stack.

    The forward path, losses and validation are the shared reconstruction step;
    this class owns only how the training target is formed (``data.n2n_pairing``).
    Validation scores against the loader's ``val_target_mode`` reference.
    """

    forwards_loss_components: ClassVar[bool] = True
    _ZF_BASELINE_METRICS: ClassVar[tuple[str, ...]] = (
        *ModelValidationMixin._ZF_BASELINE_METRICS,
        *ModelValidationMixin._ZF_OBJECT_METRICS,
    )
    parallel_compat = frozenset({"replicated", "sharded"})
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = True
    validates_on_batch_context: ClassVar[bool] = False
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.STRUCTURAL}),
        tasks=frozenset({Task.DENOISING}),
    )

    def _setup_strategy_specific_components(self) -> None:
        """Read the pairing; a Noise2Noise run without a repetition stack has no target."""
        pairing = self.config.data.n2n_pairing
        if pairing not in _PAIRINGS:
            raise ValueError(
                "NoiseToNoiseStrategy needs data.n2n_pairing in "
                f"{sorted(_PAIRINGS)}, got {pairing!r}: the loader serves the "
                "repetition stack this strategy pairs from only when it is set."
            )
        self.n2n_pairing: str = pairing
        if self.logging_service:
            self._log_config_features(self.logging_service)

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Input from ``batch["input"]``; the target from the stack under the pairing.

        A batch without the stack falls back to the loader's ``input``/``target``,
        which is what the validation split serves.
        """
        # The loop hands a TrainingBatch, which parks the stack in ``metadata``.
        if isinstance(batch, dict):
            source, raw_input = batch, batch.get("input")
        else:
            source, raw_input = getattr(batch, "metadata", None), getattr(batch, "input", None)
        if isinstance(source, dict) and _STACK_KEY in source:
            input_batch = _data_tensor(raw_input)
            stack = _data_tensor(source[_STACK_KEY])
            if input_batch is None or stack is None:
                raise ValueError("Noise2Noise batch carries a repetition stack but no input tensor")
            return input_batch, pair_from_stack(input_batch, stack, self.n2n_pairing)
        return super()._unpack_batch(batch)
