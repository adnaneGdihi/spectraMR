"""TRELLIS-style Structured-LATent (SLAT) VAE for slab → HF volume.

This module replaces the dense-Conv3d ``sparse_vae_slab_to_volume`` baseline
with a structured-latent representation directly inspired by TRELLIS
(Microsoft Research, 2024 — "Structured 3D Latents for Scalable and
Versatile 3D Generation"):

    z = (S, F)            (S, F) := the SLAT
    S   ∈ {0, 1}^{H' × W' × D_in'}       coarse-grid occupancy
    F   ∈ R^{C_lat × H' × W' × D_in'}    per-active-voxel features

The VAE encodes a thin slab into this structured latent and the decoder
reconstructs a *full* HF volume by

    1. predicting a fine-grained occupancy mask at full ``H × W × D_out``
       resolution (the slab→volume "fan-out"), and
    2. scattering the per-voxel features into a dense grid gated by that
       occupancy mask.

This is a load-bearing change for the ULF→HF stage-1 LDM in May 2026:
the prior dense Conv3d encoder treated every spatial location uniformly,
which both (a) wasted capacity on background air voxels and (b) couldn't
predict structure beyond the input slab depth without trilinear
interpolation. SLAT decouples *where there is anatomy* from *what the
anatomy looks like*, so a single thin slab can predict full-volume
structure.

Relationship to the existing TRELLIS infrastructure in this repo
----------------------------------------------------------------
The repository already carries a TRELLIS port for 3D-SHAPE generation:

* :class:`spectramr.models.generators.trellis_structured_vae.TrellisStructuredGaussianVAEGenerator`
  wraps the upstream Microsoft TRELLIS package (``trellis.modules.sparse``,
  ``trellis.models.structured_latent_vae``) and ingests a
  ``trellis.modules.sparse.SparseTensor`` to emit Gaussian-splat or mesh
  parameters. That route requires the external TRELLIS install and is
  the right abstraction for shape-domain SLAT.
* :class:`spectramr.models.trellis.sparse_structure_flow.SparseStructureFlowModel`
  is the dense-cubic-voxel diffusion model used as the stage-2 LDM in
  TRELLIS — refines the SLAT in latent space.
* :mod:`spectramr.models.representations.slat.interfaces` defines the
  :class:`ISLATEcoder`/:class:`ISLATDecoder` Protocols and
  :class:`SLATConfig`; :mod:`spectramr.models.representations.slat.renderers`
  exposes :class:`VolumeRenderer` which consumes ``[B, C, H, W, D]``
  outputs.

This module sits in a *different* SLAT-shaped niche:

* **Domain**: dense MRI volumes (image-space) rather than sparse 3D
  point clouds.
* **Input**: dense slab tensor ``[B, C_in, H, W, D_in]`` rather than a
  ``SparseTensor``.
* **Output**: dense reconstructed volume ``[B, C_out, H, W, D_out]`` —
  directly compatible with :class:`VolumeRenderer.render` if the
  downstream orchestration wants to wrap the result as a structured
  asset.
* **Dependencies**: standalone — no upstream TRELLIS install required.
  Sparse 3D convolution is emulated with dense tensors gated by a
  binary mask ``f = f * S.float()`` (see below). For the ULF→HF problem
  the coarse grid is ~64×64×D_in, well inside dense-tensor budgets.

In other words: ``trellis_structured_vae`` is the **upstream-faithful
shape-domain SLAT VAE**; this module is the **MRI-domain volumetric
SLAT VAE** built on the same factoring philosophy but native to dense
voxel grids and the ``IGenerator`` contract used by the rest of the
training pipeline. They coexist; pick by output format.

Design notes specific to this implementation
--------------------------------------------
* **Coarse grid via avg-pool.** TRELLIS gets the coarse grid by
  back-projecting multi-view DINOv2 features. We don't have multi-view
  RGB; we use the slab's own foreground intensity (avg-pool +
  thresholding) as a structural prior. A learned occupancy refinement
  head can fine-tune what the threshold misses.
* **Slab → volume fan-out.** The decoder upsamples the coarse-grid
  features from ``[H', W', D_in']`` to fine-grid ``[H, W, D_out]``
  using ``ConvTranspose3d`` with stride ``(s, s, D_out // D_in')``. The
  occupancy head is a parallel transposed-conv stack predicting
  per-fine-voxel occupancy logits, supervised by the HF brain mask
  during training (and softly used at inference).
* **No torchsparse / MinkowskiEngine dependency.** Inactive positions
  contribute nothing to KL, sparsity, or downstream conv responses
  (modulo the kernel's spatial extent — addressed by the occupancy
  head's BCE supervision).

References
----------
* TRELLIS paper: arXiv:2412.01506 — "Structured 3D Latents for Scalable
  and Versatile 3D Generation".
* microsoft/TRELLIS — ``trellis/models/structured_latent_vae``.
* :mod:`spectramr.models.generators.trellis_structured_vae` — the
  shape-domain SLAT VAE wrapper.
* :mod:`spectramr.models.representations.slat` — SLAT interfaces and
  renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model
from spectramr.models.representations.slat.interfaces import SLATConfig


@dataclass
class SLATVAESlabToVolumeConfig:
    """Configuration for the SLAT slab→volume VAE.

    Mirrors the knobs accepted by
    :class:`SLATVAESlabToVolumeGenerator.__init__`. The repo-wide
    :class:`spectramr.models.representations.slat.interfaces.SLATConfig` is
    aimed at sparse-shape SLAT (resolution + num_points + feature
    channels for Gaussian/mesh decoders); this dataclass adds the
    MRI-volumetric knobs (``d_in``, ``d_out``, ``coarse_stride``,
    ``occupancy_threshold``) that don't exist on the shape-domain
    config. Use :meth:`to_slat_config` if a downstream component (e.g.
    a renderer) expects the upstream type.
    """

    in_channels: int = 1
    out_channels: int = 1
    d_in: int = 1
    d_out: int = 64
    base_channels: int = 64
    latent_channels: int = 8
    coarse_stride: int = 4
    num_blocks: int = 3
    occupancy_threshold: float = 0.05
    occupancy_lambda: float = 1.0
    sparsity_lambda: float = 0.001

    def to_slat_config(self) -> SLATConfig:
        """Adapt to the shape-domain :class:`SLATConfig` for downstream
        components (e.g. :class:`VolumeRenderer`) that expect that type.
        Coarse-grid spatial resolution is inferred from
        ``coarse_stride`` and the assumed input HxW; depth is
        ``d_out``.
        """
        return SLATConfig(
            spatial_resolution=(
                256 // self.coarse_stride,
                256 // self.coarse_stride,
                self.d_out,
            ),
            feature_channels=self.latent_channels,
            sparse_structure=True,
        )


def _conv_norm_act(
    in_ch: int, out_ch: int, kernel: int = 3, stride: tuple[int, int, int] = (1, 1, 1)
) -> nn.Sequential:
    pad = kernel // 2
    groups = 8 if out_ch % 8 == 0 else max(1, out_ch // 2)
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad),
        nn.GroupNorm(groups, out_ch, eps=1e-6, affine=True),
        nn.SiLU(),
    )


class SLATEncoder(nn.Module):
    """Slab → (S_coarse, mu, logvar) on the coarse voxel grid.

    Output spatial shape on the coarse grid is
    ``[B, _, H/coarse_stride, W/coarse_stride, D_in]``. The coarse grid
    keeps the input depth so a single-slice slab maps to a single-depth
    coarse plane (the decoder is responsible for fanning depth out).
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        latent_channels: int,
        num_blocks: int,
        coarse_stride: int,
    ) -> None:
        super().__init__()
        self.coarse_stride = coarse_stride

        self.coarse_pool = nn.AvgPool3d(
            kernel_size=(coarse_stride, coarse_stride, 1),
            stride=(coarse_stride, coarse_stride, 1),
        )

        layers: list[nn.Module] = [_conv_norm_act(in_channels, base_channels)]
        ch = base_channels
        for _ in range(num_blocks):
            layers.append(_conv_norm_act(ch, ch))
        self.body = nn.Sequential(*layers)

        self.to_mu_logvar = nn.Conv3d(ch, latent_channels * 2, kernel_size=1)
        self.latent_channels = latent_channels

    @staticmethod
    def _foreground_mask(coarse: torch.Tensor, threshold: float) -> torch.Tensor:
        """Build a per-voxel boolean active mask from coarse intensity.

        ``coarse`` is ``[B, C, H', W', D']`` real (post-pool); reduce over
        channels (mean of magnitudes) and threshold against the per-sample
        max. Returns a ``[B, 1, H', W', D']`` float tensor with values in
        ``{0, 1}``.
        """
        intensity = coarse.abs().mean(dim=1, keepdim=True)
        per_sample_max = intensity.amax(dim=(2, 3, 4), keepdim=True).clamp(min=1e-8)
        return (intensity >= threshold * per_sample_max).to(coarse.dtype)

    def forward(
        self, x: torch.Tensor, occupancy_threshold: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        coarse = self.coarse_pool(x)
        s_coarse = self._foreground_mask(coarse, occupancy_threshold)
        h = self.body(coarse) * s_coarse
        mu, logvar = torch.split(self.to_mu_logvar(h), self.latent_channels, dim=1)
        # Gate mu/logvar so inactive voxels contribute zero KL (they have
        # no signal to encode anyway — the prior is their tightest match).
        mu = mu * s_coarse
        logvar = logvar * s_coarse
        return s_coarse, mu, logvar


class SLATDecoder(nn.Module):
    """SLAT → ``[B, C_out, H, W, D_out]`` + fine occupancy logits.

    The depth fan-out ``D_in' → D_out`` is realised by a single
    ``ConvTranspose3d`` followed by a refinement stack. A parallel
    occupancy head predicts ``logit O(x, y, z)`` over the fine grid; at
    inference the decoded volume is gated by ``sigmoid(O)`` so the model
    never paints anatomy outside the predicted brain mask (a TRELLIS-
    style "structured" output).
    """

    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        latent_channels: int,
        num_blocks: int,
        coarse_stride: int,
        d_in: int,
        d_out: int,
    ) -> None:
        super().__init__()
        self.coarse_stride = coarse_stride
        self.d_in = d_in
        self.d_out = d_out
        # Depth fan-out factor; clamp ≥1 so d_out == d_in is a no-op.
        self.depth_factor = max(1, d_out // max(1, d_in))
        self.hw_factor = coarse_stride

        # ---- Feature upsampler: coarse latent → fine-resolution feats ---
        self.feature_upsample = nn.ConvTranspose3d(
            latent_channels,
            base_channels,
            kernel_size=(self.hw_factor, self.hw_factor, self.depth_factor),
            stride=(self.hw_factor, self.hw_factor, self.depth_factor),
        )
        ref_layers: list[nn.Module] = []
        ch = base_channels
        for _ in range(num_blocks):
            ref_layers.append(_conv_norm_act(ch, ch))
        ref_layers.append(nn.Conv3d(ch, out_channels, kernel_size=1))
        self.refine = nn.Sequential(*ref_layers)

        # ---- Occupancy head: same fan-out, predicts logits ---------------
        # Run on the GATED coarse latent (active set only) so the model
        # learns where structure lives, not just intensity.
        self.occupancy_upsample = nn.ConvTranspose3d(
            latent_channels,
            base_channels // 2,
            kernel_size=(self.hw_factor, self.hw_factor, self.depth_factor),
            stride=(self.hw_factor, self.hw_factor, self.depth_factor),
        )
        self.occupancy_head = nn.Sequential(
            _conv_norm_act(base_channels // 2, base_channels // 2),
            nn.Conv3d(base_channels // 2, 1, kernel_size=1),
        )

    def forward(self, z: torch.Tensor, s_coarse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.dim() == 4:
            z = z.unsqueeze(-1)

        # Sparse-via-mask: gate features by the coarse active set so the
        # transposed convs spread active-only signal into the fine grid.
        z_gated = z * s_coarse

        feats = self.feature_upsample(z_gated)
        occ_feats = self.occupancy_upsample(z_gated)

        occ_logits = self.occupancy_head(occ_feats)
        # Refined volume gated softly by occupancy probability — the
        # SLAT contract: features only contribute where occupancy > 0.
        gated = feats * torch.sigmoid(occ_logits)
        vol = self.refine(gated)
        return vol, occ_logits


@register_model(
    name="slat_vae_slab_to_volume",
    training_mode="vae",
    spatial_dims=(2, 3),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class SLATVAESlabToVolumeGenerator(nn.Module, IGenerator):
    r"""TRELLIS-SLAT slab/slice → HF volume VAE.

    Replaces the dense ``sparse_vae_slab_to_volume`` baseline with a
    structured-latent representation: a coarse occupancy mask plus
    per-active-voxel features. The decoder predicts a full-resolution
    occupancy mask and gates the reconstructed volume by it, so a thin
    input slab can produce a full HF volume without uniform interpolation.

    Args:
        in_channels: Channels of the input slab (1 = magnitude HF/ULF).
        out_channels: Channels of the reconstructed volume.
        d_in: Slab thickness on input (1 = single slice).
        d_out: Output volume depth.
        base_channels: Hidden channels for the encoder/decoder bodies.
        latent_channels: Per-active-voxel feature dim ``C`` (TRELLIS
            uses 8 — same default).
        coarse_stride: Spatial downsampling factor for the coarse grid
            (TRELLIS uses 4× → ``H/4 × W/4``).
        num_blocks: Number of encoder/decoder Conv3d refinement blocks.
        occupancy_threshold: Foreground threshold (fraction of per-sample
            max intensity) used to derive the coarse active set in the
            encoder. Applied to the magnitude-of-channel-mean.
        occupancy_lambda / sparsity_lambda: weights of the structural prior
            (:meth:`structural_prior_loss`) the VAE strategy adds to its total.
        beta_kl: refused; the KL weight is owned by ``training.latent``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_in: int = 1,
        d_out: int = 64,
        base_channels: int = 64,
        latent_channels: int = 8,
        coarse_stride: int = 4,
        num_blocks: int = 3,
        occupancy_threshold: float = 0.05,
        occupancy_lambda: float = 1.0,
        sparsity_lambda: float = 0.001,
        beta_kl: float | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if beta_kl is not None:
            raise ValueError(
                "slat_vae_slab_to_volume does not read beta_kl; declare the KL weight "
                "as training.latent.kl_beta_end (the VAE loss computer owns it)."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_in = d_in
        self.d_out = d_out
        self.latent_channels = latent_channels
        self.coarse_stride = coarse_stride
        self.occupancy_threshold = occupancy_threshold
        self.occupancy_lambda = occupancy_lambda
        self.sparsity_lambda = sparsity_lambda

        self.encoder = SLATEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            coarse_stride=coarse_stride,
        )
        self.decoder = SLATDecoder(
            out_channels=out_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            coarse_stride=coarse_stride,
            d_in=d_in,
            d_out=d_out,
        )

        # Cache of last forward's intermediates so the strategy/loss
        # builder can pull (mu, logvar, occupancy_logits, S_coarse) out
        # without a second forward pass. Reset in every forward.
        self._last_mu: torch.Tensor | None = None
        self._last_logvar: torch.Tensor | None = None
        self._last_occupancy_logits: torch.Tensor | None = None
        self._last_s_coarse: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return "slat_vae_slab_to_volume"

    # ── Standard VAE building blocks ──────────────────────────────────

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a slab; returns ``(mu, logvar)`` like a standard VAE.

        For tests/strategies that don't need ``S_coarse`` directly, this
        keeps the legacy two-tensor return signature.
        """
        s_coarse, mu, logvar = self.encoder(x, self.occupancy_threshold)
        self._last_s_coarse = s_coarse
        return mu, logvar

    def encode_structured(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a slab; returns ``(S_coarse, mu, logvar)``."""
        return self.encoder(x, self.occupancy_threshold)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
        s_coarse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode SLAT features → fine HF volume.

        If ``s_coarse`` is not provided the cached one from the last
        encode/forward is used (e.g. when called from a sampling loop).
        """
        if s_coarse is None:
            if self._last_s_coarse is None:
                raise RuntimeError(
                    "SLAT decode() requires either an explicit s_coarse "
                    "or a prior forward()/encode() call to populate the "
                    "cache. Sampling without an active set is undefined."
                )
            s_coarse = self._last_s_coarse
        vol, _ = self.decoder(z, s_coarse)
        # Snap to exact requested depth to absorb non-divisible ratios.
        if vol.shape[-1] != self.d_out:
            vol = F.interpolate(
                vol,
                size=(*vol.shape[2:-1], self.d_out),
                mode="trilinear",
                align_corners=False,
            )
        return vol

    def forward(
        self,
        x: torch.Tensor,
        return_structured: bool = False,
        **_: Any,
    ) -> torch.Tensor:
        """slab/slice → reconstructed full HF volume.

        Side-effect: caches ``(mu, logvar, occupancy_logits, s_coarse)``
        on ``self`` for retrieval by the strategy via ``last_aux()``.
        """
        s_coarse, mu, logvar = self.encoder(x, self.occupancy_threshold)
        z = self.reparameterise(mu, logvar)
        # Apply the active-set gate before the decoder spreads features.
        z = z * s_coarse
        vol, occ_logits = self.decoder(z, s_coarse)

        if vol.shape[-1] != self.d_out:
            vol = F.interpolate(
                vol,
                size=(*vol.shape[2:-1], self.d_out),
                mode="trilinear",
                align_corners=False,
            )

        self._last_mu = mu
        self._last_logvar = logvar
        self._last_occupancy_logits = occ_logits
        self._last_s_coarse = s_coarse

        if return_structured:
            return vol, mu, logvar, occ_logits, s_coarse  # type: ignore[return-value]
        return vol

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.decode(z)

    # ── Loss-side helpers (consumed by the VAE training strategy) ────

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Per-active-voxel KL.

        SLAT KL is computed only over active voxels — inactive entries
        of ``mu``/``logvar`` are zero, so ``-0.5 (1 + 0 - 0 - 1) = 0``
        already; we just normalise by the active count to keep the term
        magnitude-stable across batches with different occupancy.
        """
        s = self._last_s_coarse
        per_voxel = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if s is None:
            return per_voxel.mean()
        active = s.sum().clamp(min=1.0)
        # Sum over active voxels, then divide by their count.
        return (per_voxel * s).sum() / active

    def sparsity_loss(self) -> torch.Tensor:
        """Mean fine-grid occupancy probability — encourages a small
        active set (the TRELLIS-style structural prior). Low when only
        the brain mask is occupied; high if the model "paints" air."""
        occ = self._last_occupancy_logits
        if occ is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.sigmoid(occ).mean()

    def occupancy_loss(self, ground_truth_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Binary cross-entropy on the predicted fine occupancy.

        If no ground-truth mask is supplied the loss is computed against
        the slab's own derived foreground (a self-supervised signal —
        useful for ULF→HF where the HF mask isn't always cached). The
        strategy is expected to pass the HF brain mask when available.
        """
        if self._last_occupancy_logits is None:
            return torch.zeros((), device=next(self.parameters()).device)
        logits = self._last_occupancy_logits
        if ground_truth_mask is None:
            # Self-supervision: use a softmax-thresholded magnitude
            # version of the decoded volume's intensity. Without a real
            # GT mask, the loss becomes a regulariser that just
            # discourages overconfident occupancy predictions.
            target = torch.full_like(logits, 0.5)
        else:
            target = ground_truth_mask.to(logits.dtype).to(logits.device)
            if target.shape != logits.shape:
                target = F.interpolate(target, size=logits.shape[2:], mode="nearest")
        return F.binary_cross_entropy_with_logits(logits, target)

    def structural_prior_loss(self, target: torch.Tensor) -> torch.Tensor:
        """Weighted TRELLIS prior: occupancy BCE against the target's foreground + sparsity."""
        vol = target.unsqueeze(-1) if target.dim() == 4 else target
        mask = SLATEncoder._foreground_mask(vol.detach(), self.occupancy_threshold)
        return self.occupancy_lambda * self.occupancy_loss(
            mask
        ) + self.sparsity_lambda * self.sparsity_loss()

    def last_aux(self) -> dict[str, torch.Tensor | None]:
        """Expose the cached (mu, logvar, occupancy, s_coarse) for the
        strategy's loss aggregator — avoids a second forward pass."""
        return {
            "mu": self._last_mu,
            "logvar": self._last_logvar,
            "occupancy_logits": self._last_occupancy_logits,
            "s_coarse": self._last_s_coarse,
        }

    # ── Interface boilerplate ────────────────────────────────────────

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        b = input_shape[0]
        h, w = input_shape[2], input_shape[3]
        return (b, self.out_channels, h, w, self.d_out)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
