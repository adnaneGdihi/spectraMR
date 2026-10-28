r"""Diffusion-weighted signal models (regime: ``mri_diffusion_weighted``).

Diffusion-weighted MRI follows the mono-exponential decay

.. math::

    S(b) = S_0 \, e^{-b \cdot \mathrm{ADC}}

with ``b`` the diffusion weighting (s/mm²) and ``ADC`` the apparent diffusion
coefficient (mm²/s). This is the forward physics of the regime: nonlinear in
``ADC`` and with no adjoint, which is exactly why it is a *signal model* and not
an ``OperatorRegistry`` operator — see :mod:`.registry`.

Both directions live here because both are physics:

* :func:`adc_monoexp_forward` — parameters -> signal (the registered model).
* :func:`fit_adc_loglinear` — signal -> parameters, by log-linear least squares.
  This is the *estimator*, not an adjoint: it inverts the model by solving a
  regression, which is available only because taking a log linearises this
  particular decay. Do not mistake it for ``A'``; it does not generalise to
  IVIM/DTI and it is not what a ``OperatorProjectionDataConsistency`` wants.

``fit_adc_loglinear`` previously lived inside
``models/losses/dwi_adc_monoexp_loss.py``. Physics has one canonical home
(CLAUDE.md #6 / #12); the loss now imports it from here, as the flow and
perfusion losses already do with their signal models.
"""

from __future__ import annotations

import torch

from spectramr.config.schemas.enums import Regime

from .registry import register_signal_model


@register_signal_model(
    name="adc_monoexp",
    regime=Regime.DIFFUSION_WEIGHTED,
    parameters=("s0", "adc"),
    signal="diffusion_weighted_magnitude",
    reference="Stejskal & Tanner, J Chem Phys 42(1):288-292, 1965",
)
def adc_monoexp_forward(
    b_values: torch.Tensor,
    s0: torch.Tensor,
    adc: torch.Tensor,
) -> torch.Tensor:
    r"""Mono-exponential diffusion signal ``S(b) = S_0 e^{-b \cdot ADC}``.

    Args:
        b_values: ``[n_b]`` diffusion weightings (s/mm²).
        s0: ``[B, 1, H, W]`` (or broadcastable) non-diffusion-weighted signal.
        adc: ``[B, 1, H, W]`` (or broadcastable) apparent diffusion coefficient
            (mm²/s).

    Returns:
        ``[B, n_b, H, W]`` predicted diffusion-weighted magnitudes.

    Raises:
        ValueError: if ``b_values`` is not 1-D, or any b-value is negative.
    """
    if b_values.ndim != 1:
        raise ValueError(f"b_values must be a 1-D [n_b] axis; got {tuple(b_values.shape)}.")
    if bool((b_values < 0).any()):
        raise ValueError(
            "b_values must be non-negative (s/mm²). A negative b flips the "
            "exponent and the signal grows without bound."
        )
    b_col = b_values.reshape(1, -1, 1, 1).to(dtype=s0.dtype, device=s0.device)
    return s0 * torch.exp(-b_col * adc)


def fit_adc_loglinear(
    images: torch.Tensor, b_values: torch.Tensor, *, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Per-pixel log-linear mono-exponential fit ``ln S = ln S_0 - b·ADC``.

    The estimator paired with :func:`adc_monoexp_forward`. Taking the log turns
    the decay into a straight line in ``b``, so ``(S_0, ADC)`` fall out of an
    ordinary least-squares slope/intercept — no iteration, fully differentiable.

    Args:
        images: ``[B, n_b, H, W]`` positive diffusion-weighted magnitudes.
        b_values: ``[n_b]`` diffusion weightings (s/mm²), ``n_b >= 2`` with at
            least two distinct values.
        eps: floor on the signal before the log.

    Returns:
        ``(adc, log_s0)`` each ``[B, 1, H, W]``. ``adc`` is the fitted apparent
        diffusion coefficient (mm²/s); ``log_s0`` is ``ln S_0``.

    Raises:
        ValueError: on a non-4-D stack, a ``b_values``/channel count mismatch,
            fewer than 2 b-values, or zero b-variance. Every degenerate path
            raises rather than returning a plausible number.
    """
    if images.dim() != 4:
        raise ValueError(f"images must be [B, n_b, H, W]; got {tuple(images.shape)}.")
    n_b = images.shape[1]
    b = b_values.reshape(-1).to(images.dtype).to(images.device)
    if b.numel() != n_b:
        raise ValueError(f"b_values has {b.numel()} entries but images has {n_b} b-channels.")
    if n_b < 2:
        raise ValueError("monoexponential ADC fit needs >= 2 b-values.")
    b_col = b.reshape(1, n_b, 1, 1)
    b_mean = b.mean()
    b_centered = b_col - b_mean
    var_b = (b_centered**2).sum()
    if float(var_b) < eps:
        raise ValueError("b_values must contain at least two distinct values (var(b)=0).")

    y = torch.log(images.clamp_min(eps))  # [B, n_b, H, W]
    y_mean = y.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    # slope of ln S vs b = -ADC (least squares).
    slope = (b_centered * (y - y_mean)).sum(dim=1, keepdim=True) / (var_b + eps)
    adc = -slope
    log_s0 = y_mean - slope * b_mean
    return adc, log_s0


__all__ = ["adc_monoexp_forward", "fit_adc_loglinear"]
