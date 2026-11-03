r"""Proximal Data Consistency Step for Reverse Diffusion.

Enforces measurement fidelity at every reverse diffusion timestep by
replacing sampled k-space frequencies with the measured ground-truth data:

.. math::

    \hat{k}^{DC}(\omega) =
    \begin{cases}
        y(\omega),           & \omega \in \Omega \\
        \hat{k}(\omega),     & \omega \notin \Omega
    \end{cases}

This confines the network's generative power strictly to the **null space**
of the forward operator, making structural hallucinations mathematically
impossible within the sampled bands.

Reference:
    Chung et al., "Score-MRI: Step-by-Step Problem Solving using
    Diffusion Models", *IEEE TMI* (2022).
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor, nn

from .fft_ops import _to_complex, fft2c, ifft2c


class ProximalDCStep(nn.Module):
    """Proximal Data Consistency projection.

    Supports both *hard* (exact replacement) and *soft* (learnable
    interpolation) modes.

    Hard mode:
        ``k_out = (1 - M) * k_pred + M * y``

    Soft mode (λ-weighted):
        ``k_out = (1 - λM) * k_pred + λM * y``

    where *M* is the sampling mask and *y* is the measured k-space.
    The mixing weight ``λ`` is **clamped to ``[0, 1]``** before use
    (``λ=0`` → pure prediction, ``λ=1`` → exact hard DC); ``soft_lambda``
    values outside this range are silently saturated to the nearest bound.
    Note that ``λ`` is only consulted during ``@torch.no_grad()`` sampling,
    so although it is registered as an ``nn.Parameter`` it is not trained
    by the diffusion loop.

    Example::

        dc = ProximalDCStep(mode="hard")
        # Inside reverse diffusion loop:
        x_pred = model(x_t, t)
        x_dc   = dc(x_pred, measured_kspace, mask)
    """

    #: Recorded as a ``post`` stage of the debug snapshot (``snapshot.hooks``).
    snapshot_stage: ClassVar[str] = "post"

    def __init__(self, mode: str = "hard", soft_lambda: float = 1.0) -> None:
        """Initialise the DC projection.

        Args:
            mode: ``'hard'`` for exact replacement, ``'soft'`` for learnable.
            soft_lambda: Initial mixing weight for soft mode (1.0 = hard).
        """
        super().__init__()
        if mode not in ("hard", "soft"):
            raise ValueError(f"ProximalDCStep mode must be 'hard' or 'soft', got '{mode}'")
        self.mode = mode

        if mode == "soft":
            self._lambda = nn.Parameter(torch.tensor(soft_lambda))
        else:
            self.register_buffer("_lambda", torch.tensor(1.0))

    def forward(
        self,
        predicted_image: Tensor,
        measured_kspace: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Apply data consistency projection.

        Args:
            predicted_image: Network output ``(B, C, H, W)``.
                Can be complex or 2-channel real/imag.
            measured_kspace: Acquired sensor data ``(B, C, H, W)``.
                Must be in k-space domain (complex or 2-channel real/imag).
            mask: Binary sampling mask ``(B, 1, H, W)`` or broadcastable.
                1 at sampled locations, 0 elsewhere.

        Returns:
            Data-consistent image ``(B, C, H, W)`` in the **image** domain,
            matching the input dtype (complex or 2-channel real).
        """
        is_complex_input = torch.is_complex(predicted_image)
        is_stacked_real = (
            not is_complex_input and predicted_image.dim() == 4 and predicted_image.shape[1] == 2
        )

        # Convert to complex for DC computation
        pred_c = _to_complex(predicted_image)
        meas_c = _to_complex(measured_kspace)
        m = mask.real if torch.is_complex(mask) else mask.float()

        # 1. Predicted image → k-space
        k_pred = fft2c(pred_c)

        # 2. Data consistency projection
        lam = self._lambda.clamp(0.0, 1.0)
        k_dc = (1.0 - lam * m) * k_pred + lam * m * meas_c

        # 3. Back to image domain
        x_dc = ifft2c(k_dc)

        # Return in original format
        if is_stacked_real:
            return torch.cat([x_dc.real, x_dc.imag], dim=1)
        if is_complex_input:
            return x_dc
        # Fallback: return real part for magnitude-only inputs
        return x_dc.real


__all__ = ["ProximalDCStep"]
