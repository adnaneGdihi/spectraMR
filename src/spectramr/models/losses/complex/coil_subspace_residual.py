r"""Penalise reconstruction energy that no coil combination could have produced.

A physical multi-coil image is rank one in coil space, :math:`x_q=s_qm_q`, so
its component along any direction orthogonal to :math:`s_q` is exactly zero.
Noise is full rank and does not share that property: measured on the committed
M4Raw covariance, about 75% of the noise energy lands in the 3-D coil null space
and none of the signal does.

.. math::

    \mathcal L_{\mathrm{coil}}=\frac{\sum_q w_q\lVert P_q^{\perp}\hat x_q\rVert^2}
                                    {\sum_q w_q},\qquad
    P_q^{\perp}=I-\hat s_q\hat s_q^{H},\quad \hat s_q=s_q/\lVert s_q\rVert .

**Why this is not a smoothness prior.** The penalty is identically zero on every
realisable image, so it cannot trade against fidelity the way a regulariser
does. It only has a gradient where the reconstruction has put energy somewhere
the coil array cannot reach.

**Why it bites for R2R in particular.** Recorrupted-to-Recorrupted guarantees
:math:`\mathbb E[\text{target}\mid\text{input}]=x`, which constrains the
conditional *mean* and says nothing about the support. A network emitting
per-coil channels with no coupling between them can satisfy that objective while
placing invented structure off the coil manifold. Recent work reports exactly
this direction -- multi-coil reconstructions are more vulnerable to
hallucination than single-coil ones because perturbations spread across coils --
and proposes no defence.

The weight :math:`w_q=\lVert s_q\rVert^2` is free: ESPIRiT's soft-SENSE
weighting already scales its maps by the eigenvalue confidence, so the map norm
is zero outside the support and tapers at the boundary. Using it weights each
pixel by how certain the rank-one model is there, rather than imposing a hard
mask.

**Not the k-space sampling null space.** ``null_space_content`` penalises
invented content in the UNOBSERVED bins :math:`(1-M)` and needs a reference.
This is the per-pixel COIL subspace and needs none.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss

__all__ = ["CoilSubspaceResidualLoss"]

VALID_INPUT_DOMAINS: frozenset[str] = frozenset({"kspace", "image"})


def _to_complex_coils(x: torch.Tensor) -> torch.Tensor:
    """``[B, 2C, H, W]`` real-interleaved or ``[B, C, H, W]`` complex -> complex."""
    if torch.is_complex(x):
        return x
    channels = x.shape[1]
    if channels % 2 != 0:
        raise ValueError(
            f"coil_subspace_residual needs complex or real/imag-interleaved coils, "
            f"got {channels} channels. An odd count means the coils were already "
            "combined, and a combination destroys the coil vector this measures."
        )
    paired = x.view(x.shape[0], channels // 2, 2, *x.shape[-2:])
    return torch.complex(paired[:, :, 0], paired[:, :, 1])


@register_loss(
    name="coil_subspace_residual",
    aliases=["coil_null_residual"],
    domain="complex_image",
)
class CoilSubspaceResidualLoss(nn.Module):
    r"""Energy off the per-pixel coil manifold.

    Args:
        input_domain: ``"kspace"`` (default) applies ``ifft2c`` through the
            physics SSOT before measuring; ``"image"`` takes the prediction as
            already being coil images.
        weight_by_support: weight each pixel by :math:`\lVert s_q\rVert^2`, the
            ESPIRiT soft-SENSE confidence. ``False`` weights every pixel equally,
            which includes background where the maps are zero and the projector
            is the identity.
        eps: floor on the map norm before normalising.

    Shape:
        ``pred``: ``[B, 2C, H, W]`` interleaved or ``[B, C, H, W]`` complex.
        ``coil_sensitivities``: ``[B, C, H, W]``, **complex**. Mandatory at
        forward -- a coil subspace is undefined without the maps, and skipping
        silently would leave the term reading as "on" in the YAML while training
        without it (pitfall #9).
    """

    def __init__(
        self,
        input_domain: str = "kspace",
        weight_by_support: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_domain not in VALID_INPUT_DOMAINS:
            raise ValueError(
                f"Unknown input_domain {input_domain!r} for coil_subspace_residual. "
                f"Valid: {sorted(VALID_INPUT_DOMAINS)}."
            )
        self.input_domain = input_domain
        # The builder inserts its OWN iFFT bridge for `complex_losses` under
        # `output_domain: kspace`. Advertising the internal one lets it reject
        # that combination instead of inverse-transforming twice, which is
        # finite, silent, and measures nothing this term advertises (#467).
        self.use_fourier_bridge = input_domain == "kspace"
        self.weight_by_support = weight_by_support
        self.eps = eps

    def _coil_images(self, x: torch.Tensor) -> torch.Tensor:
        complex_x = _to_complex_coils(x)
        if self.input_domain == "image":
            return complex_x
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        return ifft2c(complex_x)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        coil_sensitivities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the penalty. ``target`` is accepted and unused: no reference."""
        del target
        if coil_sensitivities is None:
            raise ValueError(
                "coil_subspace_residual requires coil_sensitivities: the coil "
                "subspace is undefined without the maps. Supply them through the "
                "batch (a manifest `sensitivity_path`, resolved by "
                "ManifestLoader._extract_sensitivity_params) or disable the loss."
            )
        if not torch.is_complex(coil_sensitivities):
            raise ValueError(
                "coil_subspace_residual needs COMPLEX coil sensitivities, got "
                f"{coil_sensitivities.dtype}. One branch of the TorchIO subject "
                "builder stores `sensitivity` as a magnitude (`sens_tensor.abs()`) "
                "and the complex map under `sensitivity_complex`; a real map makes "
                "P a real projector, which under-penalises the phase half of the "
                "null space silently. Route the complex maps here."
            )

        images = self._coil_images(pred)
        if images.shape != coil_sensitivities.shape:
            raise ValueError(
                f"prediction {tuple(images.shape)} and maps "
                f"{tuple(coil_sensitivities.shape)} disagree; the projector is "
                "per-pixel and per-coil."
            )

        norm = coil_sensitivities.abs().pow(2).sum(1, keepdim=True).sqrt()
        unit = coil_sensitivities / norm.clamp(min=self.eps)
        # ||P x||^2 = ||x||^2 - |<s_hat, x>|^2, so the projector never has to be
        # formed: a (B, H, W, C, C) tensor per step would be pure overhead.
        projection = (unit.conj() * images).sum(1)
        residual = images.abs().pow(2).sum(1) - projection.abs().pow(2)
        residual = residual.clamp(min=0.0)

        if not self.weight_by_support:
            return residual.mean()
        weights = norm.squeeze(1).pow(2)
        total = weights.sum()
        if float(total) <= 0.0:
            raise ValueError(
                "coil_subspace_residual: every sensitivity map is zero, so no pixel "
                "carries a rank-one constraint. The maps did not reach this loss."
            )
        return (weights * residual).sum() / total
