r"""Bloch-bottleneck generator: predict tissue parameters, synthesise any contrast.

Implements §10a of the multi-contrast brainstorm — a generator whose
*latent space is the quantitative tissue-parameter manifold*
:math:`(\rho, T_1, T_2)` rather than weighted-image features. The forward
pass is a two-stage pipeline:

1. A per-pixel encoder–decoder backbone maps the input image to three
   parameter maps :math:`(\hat{\rho}, \hat{T}_1, \hat{T}_2)`.
2. A differentiable Bloch synthesis layer
   (:class:`~spectramr.infrastructure.physics.multi_physics_bloch.MultiPhysicsBlochLayer`)
   converts those parameters into the requested contrast at acquisition
   parameters :math:`\theta_c=(TE, TR, TI, \alpha)`.

The bottleneck is the key: any new contrast is reachable at inference by
re-running the Bloch layer with new metadata, *without retraining*. This
generalises the SynthSR / SynthSeg synthesis-from-parameters idea
[2, 3] from data augmentation into a first-class architectural choice.

Pairs naturally with
:class:`~spectramr.models.losses.bloch_consistency_loss.BlochConsistencyLoss`
which anchors the predicted parameters to observed contrast images via

.. math::

   \mathcal{L}_{\text{phys}}
   \;=\;
   \sum_c \big\| \mathbf{x}_c - B(\hat\rho, \hat T_1, \hat T_2; \theta_c)\big\|_p.

The generator exposes the predicted tissue parameters and the acquisition
metadata via the ``last_tissue_params`` / ``last_acquisition_params``
attributes so the strategy can populate the loss ``context`` dict without
rerunning the forward pass — the standard pattern for losses that need
intermediate model state in this codebase.

Channel layout
--------------
- ``in_channels``  : input image channels (typically 1 for magnitude).
- ``out_channels`` : 1 — the synthesised contrast image (the parameter
  maps are intermediate, not the final output).
- ``base_channels`` : feature width of the UNet backbone.
- ``param_clamp`` : optional ``(min, max)`` for each parameter map,
  prevents pathological exploration of T1/T2 space during early training.

References
----------
[1] E. M. Haacke, R. W. Brown, M. R. Thompson, R. Venkatesan, *Magnetic Resonance Imaging:
    Physical Principles and Sequence Design*, 2nd ed., Wiley, 2014.
[2] J. E. Iglesias *et al.*, "Joint super-resolution and synthesis of 1 mm
    isotropic MP-RAGE volumes from clinical MRI exams with scans of
    different orientation, resolution and contrast," *NeuroImage*, vol. 237,
    p. 118206, 2021.
[3] B. Billot *et al.*, "SynthSeg: Segmentation of brain MRI scans of any
    contrast and resolution without retraining," *Med. Image Anal.*,
    vol. 86, p. 102789, 2023.

§10a of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.multi_physics_bloch import MultiPhysicsBlochLayer
from spectramr.models.registry import register_model

__all__ = ["BlochParameterBottleneckGenerator"]


_CONTRAST_TYPE_TO_INT: dict[str, int] = {
    "spin_echo": MultiPhysicsBlochLayer.SEQ_SPIN_ECHO,
    "inversion_recovery": MultiPhysicsBlochLayer.SEQ_INVERSION_RECOVERY,
    "gradient_echo": MultiPhysicsBlochLayer.SEQ_GRADIENT_ECHO,
    "diffusion_weighted": MultiPhysicsBlochLayer.SEQ_DIFFUSION_WEIGHTED,
}


class _DoubleConv(nn.Module):
    """Conv → GroupNorm → SiLU twice; standard UNet primitive."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        g_in = min(groups, in_ch) if in_ch >= groups else 1
        g_out = min(groups, out_ch) if out_ch >= groups else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g_out, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g_out, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ParameterUNet(nn.Module):
    """Compact UNet that emits 3-channel parameter maps."""

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.enc1 = _DoubleConv(in_channels, c1)
        self.enc2 = _DoubleConv(c1, c2)
        self.enc3 = _DoubleConv(c2, c3)
        self.bottleneck = _DoubleConv(c3, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(c1 * 2, c1)
        self.head = nn.Conv2d(c1, 3, kernel_size=1)  # rho, T1, T2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(F.avg_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


@register_model(
    name="bloch_parameter_bottleneck",
    training_mode="reconstruction",
    supports_contrast_conditioning=True,
)
class BlochParameterBottleneckGenerator(nn.Module):
    r"""Generator factoring images through quantitative tissue parameters.

    Args:
        in_channels: Input image channels.
        out_channels: Must be 1 — output is a synthesised single-channel
            contrast image.
        base_channels: Feature width of the parameter-prediction UNet.
        rho_range: Plausible range for proton density. Sigmoid scaled.
        t1_range_ms: Plausible (min, max) for :math:`T_1` in milliseconds.
        t2_range_ms: Plausible (min, max) for :math:`T_2` in milliseconds.
        learnable_flip_angle: Forwarded to :class:`MultiPhysicsBlochLayer`.

    Forward signature
    -----------------
    ``forward(x, acquisition_params=None, contrast_idx=None)``

    - ``x``: ``(B, in_channels, H, W)`` input image.
    - ``acquisition_params``: dict with ``TE/TR/TI/contrast_type/alpha``
      (or numeric ``contrast_type`` int), scalars or per-sample ``[B]``
      tensors. Required: without it the forward raises, and the parameter
      maps are read through :meth:`predict_parameters` instead.
    - ``contrast_idx``: accepted for capability-flag compatibility and
      forwarded into ``last_acquisition_params`` so the strategy can
      look up per-sample acquisition metadata externally.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        rho_range: tuple[float, float] = (0.0, 1.0),
        t1_range_ms: tuple[float, float] = (50.0, 5000.0),
        t2_range_ms: tuple[float, float] = (5.0, 2000.0),
        learnable_flip_angle: bool = False,
    ) -> None:
        super().__init__()
        if out_channels != 1:
            raise ValueError(
                "BlochParameterBottleneckGenerator outputs a single contrast "
                f"image; got out_channels={out_channels}."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        self.parameter_net = _ParameterUNet(in_channels, base_channels)
        self.bloch = MultiPhysicsBlochLayer(learnable_flip_angle=learnable_flip_angle)

        # Buffers so the ranges move with .to(device).
        self.register_buffer(
            "_rho_lo_hi", torch.tensor(rho_range, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_t1_lo_hi", torch.tensor(t1_range_ms, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_t2_lo_hi", torch.tensor(t2_range_ms, dtype=torch.float32), persistent=False
        )

        # Inspection hooks — populated on every forward pass.
        self.last_tissue_params: dict[str, torch.Tensor] | None = None
        self.last_acquisition_params: dict[str, Any] | None = None

    @staticmethod
    def _scale_to_range(raw: torch.Tensor, lo_hi: torch.Tensor) -> torch.Tensor:
        lo, hi = lo_hi[0], lo_hi[1]
        return lo + (hi - lo) * torch.sigmoid(raw)

    def predict_parameters(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the tissue-parameter maps without synthesising any contrast."""
        raw = self.parameter_net(x)
        rho = self._scale_to_range(raw[:, 0:1], self._rho_lo_hi)
        t1 = self._scale_to_range(raw[:, 1:2], self._t1_lo_hi)
        t2 = self._scale_to_range(raw[:, 2:3], self._t2_lo_hi)
        return {"rho": rho, "t1": t1, "t2": t2}

    def forward(
        self,
        x: torch.Tensor,
        acquisition_params: dict[str, Any] | None = None,
        contrast_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if acquisition_params is None:
            # The output is one synthesised contrast; the maps alone are predict_parameters'.
            raise ValueError(
                "BlochParameterBottleneckGenerator renders its output through the Bloch "
                "layer and needs acquisition_params (TE/TR/TI/contrast_type); declare "
                "data.multi_contrast.enabled with acquisition_params, or call "
                "predict_parameters() for the (rho, T1, T2) maps."
            )

        params = self.predict_parameters(x)
        self.last_tissue_params = params
        meta = dict(acquisition_params)
        ct = meta.get("contrast_type")
        if isinstance(ct, str):
            if ct not in _CONTRAST_TYPE_TO_INT:
                raise ValueError(
                    f"Unknown contrast_type {ct!r}; expected one of {list(_CONTRAST_TYPE_TO_INT)}."
                )
            meta["contrast_type"] = _CONTRAST_TYPE_TO_INT[ct]

        signal = self.bloch(tissue_params=params, metadata=meta)
        self.last_acquisition_params = meta
        return signal
