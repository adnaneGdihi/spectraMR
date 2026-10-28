r"""Negative log-likelihood in bits-per-dimension.

The canonical training objective / reporting metric for exact-likelihood
generative models (normalising flows: ``glow``, ``equivariant_flow``).

For a model that returns a per-sample log-likelihood :math:`\log p(x)` in
nats, the bits-per-dimension is

.. math::
    \mathrm{bpd} = -\frac{\log p(x)}{D \,\ln 2},

where :math:`D` is the number of data dimensions. Lower is better. The
division by :math:`D\ln 2` makes the number comparable across resolutions
and is the standard reporting unit in the flow / autoregressive literature
(Kingma & Dhariwal, NeurIPS 2018).

This loss is domain-agnostic: it consumes a likelihood, not a pixel pred /
target pair. It accepts the likelihood in three forms so it slots into both
the flow forward contract and a generic loss pipeline:

* a per-sample ``log_prob`` tensor ``[B]`` (nats),
* a ``(z, log_det)`` tuple — the standard flow ``forward`` output, from
  which the standard-normal prior log-density is added, or
* a dict carrying ``log_prob`` (or ``z`` + ``log_det``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss

_LN2 = math.log(2.0)


@register_loss(name="nll_bits_per_dim", aliases=["bits_per_dim", "bpd"], domain="agnostic")
class NLLBitsPerDim(nn.Module):
    r"""Bits-per-dimension negative log-likelihood.

    **DOMAIN**: AGNOSTIC — operates on a likelihood, not a pixel grid.

    Args:
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean'|'sum'|'none', got {reduction!r}")
        self.reduction = reduction

    @staticmethod
    def _log_prob_from_z(z: torch.Tensor, log_det: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Standard-normal prior log-density + log|det J|, and dimensionality."""
        z_flat = z.reshape(z.shape[0], -1)
        d = z_flat.shape[1]
        log_pz = -0.5 * (z_flat**2).sum(dim=1) - 0.5 * d * math.log(2 * math.pi)
        return log_pz + log_det.reshape(z.shape[0]), d

    def forward(
        self,
        model_output: (
            torch.Tensor
            | tuple[torch.Tensor, torch.Tensor]
            | list[torch.Tensor]
            | dict[str, torch.Tensor]
        ),
        num_dims: int | None = None,
    ) -> torch.Tensor:
        """Compute bits-per-dimension from a likelihood.

        Args:
            model_output: per-sample ``log_prob`` ``[B]``; or a
                ``(z, log_det)`` tuple; or a dict with ``log_prob`` or
                ``z``+``log_det``.
            num_dims: data dimensionality :math:`D`. Inferred from ``z`` when
                a ``(z, log_det)`` form is given; otherwise REQUIRED for a
                bare ``log_prob`` tensor (no silent default).

        Returns:
            Bits-per-dimension, reduced per ``self.reduction``.
        """
        log_prob: torch.Tensor
        if isinstance(model_output, dict):
            if "log_prob" in model_output:
                log_prob = model_output["log_prob"]
            elif "z" in model_output and "log_det" in model_output:
                log_prob, num_dims = self._log_prob_from_z(
                    model_output["z"], model_output["log_det"]
                )
            else:
                raise ValueError(
                    "dict model_output must carry 'log_prob' or 'z'+'log_det'; "
                    f"got keys {sorted(model_output)}"
                )
        elif isinstance(model_output, (tuple, list)):
            if len(model_output) != 2:
                raise ValueError(
                    f"tuple model_output must be (z, log_det); got len {len(model_output)}"
                )
            log_prob, num_dims = self._log_prob_from_z(model_output[0], model_output[1])
        else:
            log_prob = model_output

        if num_dims is None:
            raise ValueError("num_dims is required when passing a bare log_prob tensor")
        # A (pred, target) call lands the target tensor here; it must not broadcast.
        if isinstance(num_dims, bool) or not isinstance(num_dims, int) or num_dims < 1:
            raise TypeError(f"num_dims must be a positive int, got {num_dims!r}")

        bpd = -log_prob / (num_dims * _LN2)
        if self.reduction == "mean":
            return bpd.mean()
        if self.reduction == "sum":
            return bpd.sum()
        return bpd
