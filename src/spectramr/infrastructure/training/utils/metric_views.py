"""The two tensors a metric transform hands the metric set.

A metric transform (``ifft_sense_adjoint``, ``ifft_magnitude``, ``magnitude``)
ends in ``abs()``, and most metrics are defined on that magnitude. A metric
registered ``requires_complex`` measures phase, which ``abs()`` has destroyed, so
grading it on the magnitude reports the phase of ``|x|`` (#867). The transform
already holds the complex image one step before ``abs()``; :class:`MetricViews`
carries both so the computer can give each metric the tensor it is defined on.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch


class MetricViews(NamedTuple):
    """Magnitude pair for ordinary metrics, complex pair for phase metrics.

    Attributes:
        pred: The transformed prediction every ordinary metric is scored on.
        target: The transformed reference, same domain as ``pred``.
        complex_pred: The phase-preserving prediction the transform produced
            before ``abs()``, or ``None`` when it never had one (no transform on
            a real tensor, ``ifft_target_magnitude``, or a single real channel).
        complex_target: The reference counterpart of ``complex_pred``.
    """

    pred: torch.Tensor
    target: torch.Tensor
    complex_pred: torch.Tensor | None = None
    complex_target: torch.Tensor | None = None

    @property
    def complex_pair(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(complex_pred, complex_target)`` when both exist, else ``None``.

        ``None`` leaves the computer to score phase metrics on ``pred``/
        ``target``, which raises when those carry no phase -- the guard, not a
        fallback.
        """
        if self.complex_pred is None or self.complex_target is None:
            return None
        return self.complex_pred, self.complex_target

    def compute(self, computer: Any, **kwargs: Any) -> dict[str, float]:
        """Score these views with ``computer`` (a ``ValidationMetricsComputer``).

        ``complex_pair`` is passed only when there is one; ``None`` is the
        computer's own default, and omitting it keeps a computer-like object that
        never scores phase metrics free of the keyword.
        """
        pair = self.complex_pair
        if pair is not None:
            kwargs["complex_pair"] = pair
        return computer.compute(self.pred, self.target, **kwargs)


def real_stacked_to_complex(t: torch.Tensor) -> torch.Tensor:
    """Pair interleaved ``[R0, I0, R1, I1, ...]`` channels into complex.

    ``ComplexToRealTransform`` and ``ComplexGuard`` both produce this layout.
    A complex tensor, or one that is not an even-channel 4-D stack, is returned
    unchanged so the caller's own shape handling decides what it means.
    """
    if torch.is_complex(t):
        return t
    if t.dim() < 4 or t.shape[1] < 2 or t.shape[1] % 2 != 0:
        return t
    return torch.complex(t[:, 0::2], t[:, 1::2])


__all__ = ["MetricViews", "real_stacked_to_complex"]
