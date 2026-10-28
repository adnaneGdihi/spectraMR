"""What a phase-sensitive metric may be handed, and the refusal when it is not.

A metric registered with ``requires_complex=True`` measures phase. Handed a real
magnitude image it still returns a number -- ``phase_mse`` used to ``fft2c`` the
magnitude and report the k-space phase of ``|x|`` (#867) -- and that number reads
as phase fidelity while measuring nothing of the kind. This module is the one
place that decides whether a tensor carries phase, so the computer's guard and
the metrics' own input handling cannot disagree about it.
"""

from __future__ import annotations

import torch


class PhaseBlindInputError(TypeError):
    """A phase-sensitive metric was handed a tensor that carries no phase.

    A wiring defect, not an input the metric is undefined on: the caller dropped
    the phase (usually a magnitude transform) before routing. It is raised
    before the computer's per-metric ``try`` so it is never degraded to NaN.
    """


def as_phase_bearing(metric: str, tensor: torch.Tensor, role: str = "input") -> torch.Tensor:
    """Return ``tensor`` as a complex tensor, or raise if it has no phase.

    Accepted layouts:

    * a complex-dtype tensor of any shape -- returned unchanged;
    * a real ``[B, 2, H, W]`` tensor in the ``[real, imag]`` layout used by
      ``ComplexToRealTransform`` -- folded to complex ``[B, 1, H, W]``.

    Anything else is magnitude-only as far as this function can tell, and
    measuring its phase would report the phase of ``|x|``.

    Args:
        metric: Registry name of the metric, for the error text.
        tensor: The prediction or reference about to be scored.
        role: ``"prediction"`` or ``"target"``, for the error text.

    Raises:
        PhaseBlindInputError: ``tensor`` is real and not a 2-channel real/imag stack.
    """
    if torch.is_complex(tensor):
        return tensor
    if tensor.dim() == 4 and tensor.shape[1] == 2:
        return torch.complex(tensor[:, 0:1].float(), tensor[:, 1:2].float())
    raise PhaseBlindInputError(
        f"Metric '{metric}' is registered requires_complex=True but its {role} is a "
        f"real tensor of shape {tuple(tensor.shape)} (not complex, not [B, 2, H, W] "
        "real/imag). Its phase is the phase of a magnitude image, not the "
        "reconstruction's. Route the complex image to it (metric transforms keep "
        "one before taking abs()), or remove it from the metric set."
    )


__all__ = ["PhaseBlindInputError", "as_phase_bearing"]
