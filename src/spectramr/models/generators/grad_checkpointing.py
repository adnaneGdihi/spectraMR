"""One owner for the activation-checkpointing switch every backbone needs.

``KSpaceColdDiffusionGenerator.set_grad_checkpointing`` forwards
``optimization.gradient.enable_checkpointing`` to the backbone and RAISES when
that backbone has no hook -- a memory claim that silently did not happen OOMs
later with nothing to point at (non-negotiable 3). Four arms of the 2026-09-16
dispatch died at build for exactly that reason, so five backbones grew the hook
at once; a sixth (``ComplexUNet``) already had it. The switch itself is four
lines of state, and five copies of four lines is how they drift.

What the mixin does NOT own is the granularity. Each backbone decides which
segments it wraps in :func:`torch.utils.checkpoint.checkpoint`, because only it
knows which of its parts has an interior worth discarding: a whole res/Swin
block, one unroll of a cascade, a ``TimeAwareSequential`` stage. Wrapping a
single convolution would save nothing.
"""

from __future__ import annotations

import torch

__all__ = ["GradCheckpointingMixin"]


class GradCheckpointingMixin:
    """The ``set_grad_checkpointing`` / ``_checkpointing_active`` pair.

    Off by default, so a forward that nobody opted in for stays
    allocation-identical to the pre-checkpointing one. Measured on
    ``ComplexUNet`` at the ``experiment_11`` shape (4 coils, 256x256, fp32,
    B=2), activations saved for backward -- the term no ZeRO stage shards --
    fall from 13317 MiB to 681 MiB; on ``[2, 8, 128, 128]`` the peak allocation
    of ``swin_diff_rec`` / ``diff_varnet`` / ``nafnet`` falls by 54 % / 61 % /
    49 % with the forward output unchanged to 1e-4.
    """

    #: A class attribute, so a backbone need not remember to initialise it.
    grad_checkpointing: bool = False

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Trade recompute for activation memory in this backbone's forward.

        Args:
            enable: Whether the checkpointed segments of :meth:`forward` should
                discard their interiors and recompute them during backward.
        """
        self.grad_checkpointing = bool(enable)

    def _checkpointing_active(self) -> bool:
        """Whether this forward should actually checkpoint.

        Recompute only pays for itself when a backward follows. Under
        ``eval()`` / ``no_grad`` -- notably the multi-step reverse sampler used
        for validation -- there is nothing saved for backward to shrink, so the
        plain path is taken and the flag is ignored rather than honoured.
        """
        return (
            self.grad_checkpointing and getattr(self, "training", False) and torch.is_grad_enabled()
        )
