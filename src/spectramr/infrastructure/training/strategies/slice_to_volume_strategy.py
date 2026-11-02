r"""SliceToVolumeStrategy — Task 2.7 (slice-to-volume / 2-D → 3-D).

Implements ``IMPLEMENTATION_SPEC.md`` §2.7: takes anisotropic stacks
(thick slices with optional gaps) as input and produces an isotropic
3-D volume.

The strategy is intentionally minimal:

1. Input batch is a thick-slice stack ``(B, 1, D_lo, H, W)`` with
   ``Δz_lo > Δx_lo``.
2. Forward through the model — must accept 5-D input and emit
   ``(B, 1, D_hi, H, W)`` with ``Δz_hi = Δx_lo``.
3. Loss = composite reconstruction (MSE + SSIM) plus an explicit
   *through-plane consistency* term that compares the central axial
   slice of the predicted high-res volume against the input central
   slice — this anchors the 3-D output to the observed 2-D plane.
4. **Gradient-isotropy prior** (``lambda_ortho``): matches the
   through-plane gradient texture-energy to the in-plane energy so the
   synthesised volume is isotropically sharp (the §2.7 goal). This
   replaced a degenerate term that forced three orthogonal *mean
   projections* to be equal (minimised by a rotationally-symmetric /
   constant volume — audit 2026-06 D3).

.. note::
   The spec's full **registration consistency** metric — the *model*
   emitting sagittal + coronal reformats and registering them against
   the axial view — is NOT implemented: it needs a multi-reformat model
   head the current generators do not provide, and remains roadmap work
   (see ``docs/strategy_audit_design_decisions_2026_06.rst``). The
   gradient-isotropy prior is the well-posed regulariser achievable from
   the single emitted volume.

References
----------
* IMPLEMENTATION_SPEC.md §2.7 — 2-D to 3-D synthesis.
* Zhao et al., "SMORE: A Self-Supervised Anti-Aliasing and Super-
  Resolution Algorithm for MRI", IEEE TMI 2021 (isotropy prior).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from spectramr.data.batch_types import read_batch_field
from spectramr.data.transforms.slice_profile import (
    SLICE_PROFILE_RATIO_KEY,
    SLICE_PROFILE_Z_AXIS_KEY,
)
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.losses.computers import UnifiedReconstructionLossComputer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SliceProfileStamp:
    """Decimation applied by ``slice_profile``, plus the patch z-start when cropped."""

    ratio: int
    batch_dim: int
    z_start: torch.Tensor | None  # (B,) patch start along the through-plane axis


def _uniform(values: Any, key: str) -> int:
    vals = {int(v) for v in values} if isinstance(values, (list, tuple)) else {int(values)}
    if len(vals) != 1:
        raise ValueError(f"slice_to_volume: batch mixes {key} values {sorted(vals)}")
    return vals.pop()


def read_slice_profile_stamp(batch: Any) -> SliceProfileStamp | None:
    """Read the transform's stamp from the collated batch (dict or TrainingBatch)."""
    ratio = read_batch_field(batch, SLICE_PROFILE_RATIO_KEY)
    if ratio is None:
        return None
    ratio = _uniform(ratio, SLICE_PROFILE_RATIO_KEY)
    z_axis = _uniform(read_batch_field(batch, SLICE_PROFILE_Z_AXIS_KEY), SLICE_PROFILE_Z_AXIS_KEY)
    loc = read_batch_field(batch, "location")  # torchio.LOCATION, set by every patch sampler
    # LOCATION is (i0, j0, k0, i1, j1, k1); absent means an uncropped volume (phase 0).
    z_start = loc[:, z_axis - 1] if isinstance(loc, torch.Tensor) else None
    return SliceProfileStamp(ratio=ratio, batch_dim=z_axis + 1, z_start=z_start)


def replicated_stack_anchor(
    x_hi: torch.Tensor, x_lo: torch.Tensor, stamp: SliceProfileStamp
) -> torch.Tensor:
    """Avg-pool ``x_hi`` over each thick-slice block and compare to that block's value.

    The transform decimated the whole volume from index 0 and replicated each thick
    slice ``ratio`` times, so a patch starting at ``k0`` has its first full block at
    ``(-k0) mod ratio``. Only blocks wholly inside the patch enter the term.
    """
    r = stamp.ratio
    if r < 2:
        raise ValueError(
            f"slice_to_volume: slice_profile decimation ratio is {r}; with no "
            "decimation the anchor is an identity MSE. Raise slice_thickness_mm "
            "above the through-plane spacing."
        )
    if x_hi.shape != x_lo.shape:
        raise ValueError(
            f"slice_to_volume: a replicated stack needs D_hi == D_lo; got "
            f"{tuple(x_hi.shape)} vs {tuple(x_lo.shape)}"
        )
    hi = x_hi.movedim(stamp.batch_dim, -1)
    lo = x_lo.movedim(stamp.batch_dim, -1)
    depth, batch = hi.shape[-1], hi.shape[0]
    if stamp.z_start is None:
        phase = torch.zeros(batch, dtype=torch.long, device=hi.device)
        n_blocks = depth // r
    else:
        phase = (-stamp.z_start.to(device=hi.device, dtype=torch.long)) % r
        n_blocks = (depth - r + 1) // r  # fits every phase without a host sync
    if n_blocks < 1:
        raise ValueError(f"slice_to_volume: depth {depth} holds no full block of {r}")
    starts = phase[:, None] + r * torch.arange(n_blocks, device=hi.device)  # (B, n)
    win = (starts[:, :, None] + torch.arange(r, device=hi.device)).reshape(batch, -1)
    lead = hi.shape[1:-1]
    view = (batch,) + (1,) * len(lead)

    def take(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return t.gather(-1, idx.view(*view, -1).expand(*t.shape[:-1], idx.shape[-1]))

    pooled = take(hi, win).unflatten(-1, (n_blocks, r)).mean(-1)
    return F.mse_loss(pooled, take(lo, starts))


class SliceToVolumeStrategy(BaseTrainingStrategy):
    """Slice-to-volume isotropic upsampling (Spec §2.7)."""

    # Consistency-term weights; overridable via the typed training.slice_to_volume
    # block (previously hardcoded as inline 0.1 / 0.05 magic constants).
    lambda_through_plane: float = 0.1
    lambda_ortho: float = 0.05

    def __init__(self, env=None, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        self.loss_computer = UnifiedReconstructionLossComputer(
            config=self.config,
            device=self.device,
        )

    def _setup_strategy_specific_components(self) -> None:
        cfg = getattr(self.config.training, "slice_to_volume", None)
        if cfg is not None:
            self.lambda_through_plane = float(cfg.lambda_through_plane)
            self.lambda_ortho = float(cfg.lambda_ortho)
        logger.info(
            "SliceToVolumeStrategy (source=%s): lambda_through_plane=%.4g, lambda_ortho=%.4g",
            "config" if cfg is not None else "defaults",
            self.lambda_through_plane,
            self.lambda_ortho,
        )

    def _through_plane_consistency(
        self,
        x_hi: torch.Tensor,
        x_lo: torch.Tensor,
        stamp: SliceProfileStamp | None = None,
    ) -> torch.Tensor:
        r"""Average-pool the predicted hi-res volume back to the input's
        through-plane resolution and compare to the original.

        ``stamp`` is the decimation the ``slice_profile`` transform applied to an
        input it replicated back to the target grid (``D_hi == D_lo``).
        """
        # Both should be 5-D: (B, 1, D, H, W)
        if x_hi.dim() != 5 or x_lo.dim() != 5:
            raise ValueError(
                f"slice_to_volume expects 5-D tensors; got "
                f"x_hi {tuple(x_hi.shape)}, x_lo {tuple(x_lo.shape)}"
            )
        if stamp is not None:
            return replicated_stack_anchor(x_hi, x_lo, stamp)
        D_hi = x_hi.shape[2]
        D_lo = x_lo.shape[2]
        if D_hi == D_lo:
            # Without the decimation stamp the anchor is an identity MSE that pulls
            # the output back onto the staircase input; refuse it.
            raise ValueError(
                "slice_to_volume: input and output share a through-plane depth "
                f"({D_hi}) but the batch carries no slice_profile decimation stamp, so "
                "the input plane is unknown. Declare the slice_profile transform with "
                "decimate: true, or feed a genuinely thick-slice (D_lo < D_hi) stack."
            )
        # The slice-to-volume upsampler must produce a DENSER through-plane
        # output than its input. A hi-res volume with fewer through-plane
        # samples than the input is a contract violation (pitfall #9) — raise
        # rather than silently degrading to interpolation.
        if D_hi < D_lo:
            raise ValueError(
                f"slice_to_volume output must be denser than input: D_hi={D_hi} < D_lo={D_lo}"
            )
        # Average-pool through-plane to match D_lo. The pooling operator is the
        # only consistency operator, so the ratio MUST be an exact integer;
        # otherwise avg_pool3d would miss samples and we would silently swap to
        # a different (trilinear) operator — validate and raise instead.
        if D_hi % D_lo != 0:
            raise ValueError(
                f"slice_to_volume through-plane ratio must be an integer: "
                f"D_hi={D_hi} is not divisible by D_lo={D_lo}"
            )
        kernel = D_hi // D_lo
        x_hi_pooled = F.avg_pool3d(
            x_hi,
            kernel_size=(kernel, 1, 1),
            stride=(kernel, 1, 1),
        )
        return F.mse_loss(x_hi_pooled, x_lo)

    def _isotropy_consistency(self, x_hi: torch.Tensor, tp_dim: int = 2) -> torch.Tensor:
        r"""Gradient-isotropy prior (audit 2026-06 D3).

        The synthesised through-plane detail should carry the same texture
        energy as the in-plane detail, so the upsampled volume is isotropically
        sharp rather than blurred through-plane (the actual §2.7 goal). We
        penalise the squared gap between the mean absolute gradient along the
        through-plane (D) axis and the mean of the two in-plane (H, W) axes
        — the standard SVR isotropy prior (cf. SMORE, Zhao et al. 2021).

        This REPLACES the previous term, which forced the three orthogonal
        mean-projections to be equal. That was degenerate: it is minimised by a
        rotationally-symmetric / constant volume and biases the output away from
        real anatomy, and it did not implement the spec's
        registration-consistency metric (which requires the *model* to emit
        sagittal/coronal reformats — that remains roadmap, see the module note).
        """
        if x_hi.dim() != 5:
            raise ValueError(
                f"slice_to_volume isotropy prior expects a 5-D volume "
                f"(B, C, D, H, W); got {tuple(x_hi.shape)}"
            )
        gz = x_hi.diff(dim=tp_dim).abs().mean()  # through-plane
        gh, gw = (x_hi.diff(dim=d).abs().mean() for d in (2, 3, 4) if d != tp_dim)
        in_plane = 0.5 * (gh + gw)
        return (gz - in_plane).pow(2)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        generator = self.env.generator
        x_hi = generator(input_batch)

        # Recon loss against the isotropic target via the unified computer
        # (operates on whatever dim structure the computer accepts; for 5-D
        # we collapse the depth axis into the batch dimension).
        if target_batch.dim() == 5:
            B, C, D, H, W = target_batch.shape
            tgt_2d = target_batch.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            pred_2d = x_hi.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        else:
            tgt_2d = target_batch
            pred_2d = x_hi

        env_losses = (self.env.losses or {}) if self.env else {}
        # Live iteration (loop_state seam): the loop passes ``iteration=``,
        # never ``step=``, so this was a constant 0 and every warm-up-gated
        # loss stayed shut for the whole run (pitfall #16).
        loss_output = self.loss_computer.compute(
            pred=pred_2d,
            target=tgt_2d,
            epoch=epoch,
            iteration=resolve_loop_iteration(self),
            losses_dict=env_losses,
        )

        stamp = read_slice_profile_stamp(kwargs.get("batch"))
        if stamp is not None and not getattr(self, "_stamp_logged", False):
            logger.info(
                "SliceToVolumeStrategy: phase-aligned anchor from slice_profile stamp "
                "(ratio=%d, through-plane batch dim=%d)",
                stamp.ratio,
                stamp.batch_dim,
            )
            self._stamp_logged = True
        tp_loss = self._through_plane_consistency(x_hi, input_batch, stamp)
        iso_loss = self._isotropy_consistency(x_hi, stamp.batch_dim if stamp else 2)

        total = (
            loss_output.total + self.lambda_through_plane * tp_loss + self.lambda_ortho * iso_loss
        )
        out = {"g_total_loss": total}
        out.update(loss_output.components)
        out["g_through_plane_consistency"] = tp_loss.detach()
        out["g_isotropy_consistency"] = iso_loss.detach()
        return out


__all__ = ["SliceProfileStamp", "SliceToVolumeStrategy", "read_slice_profile_stamp"]
