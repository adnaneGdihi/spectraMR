"""Null-space marker erasure for the virtual-fiducial validation pass (arm m2)."""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.null_space_erasure import NullSpaceMarkerEraser


class NullSpaceErasureHook:
    """Project the twin's marker template out of a reconstruction and score the projection.

    The template is the twin's ``marker_prior`` (one ``[1, 1, H, W]`` pattern shared by the
    batch). Each channel is erased on its own so anatomy common to several channels is not
    mistaken for marker.
    """

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def _eraser(self, marker_prior: torch.Tensor) -> NullSpaceMarkerEraser:
        return NullSpaceMarkerEraser(marker_prior[:1, :1], epsilon=self.epsilon)

    def erase(self, pred: torch.Tensor, marker_prior: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        erased = self._eraser(marker_prior)(pred.reshape(b * c, 1, h, w))
        return erased.reshape(b, c, h, w)

    def scores(
        self, erased: torch.Tensor, target: torch.Tensor, marker_prior: torch.Tensor
    ) -> dict[str, float]:
        """``val_marker_residual_l2``: L2 norm of the marker component left in ``erased``.

        ``val_null_space_preservation_l2``: relative change erasure makes to the part of
        ``target`` that is orthogonal to the marker, which should be ~0.
        """
        m = marker_prior[:1, :1].reshape(1, -1)
        m_norm = m.norm()
        flat = erased.reshape(-1, m.shape[1])
        # <m, y> / |m|^2 is the leftover marker amplitude; times |m| gives its L2 norm.
        leftover = (m.conj() * flat).sum(dim=1).abs() / (m_norm**2 + 1e-12)
        b, c, h, w = target.shape
        t = target.reshape(b * c, 1, h, w)
        coeff = (m.conj() * t.reshape(-1, m.shape[1])).sum(dim=1, keepdim=True) / (
            m_norm**2 + 1e-12
        )
        ortho = (t.reshape(-1, m.shape[1]) - coeff * m).reshape(b * c, 1, h, w)
        kept = self._eraser(marker_prior)(ortho)
        preserve = (kept - ortho).flatten(1).norm(dim=1) / (ortho.flatten(1).norm(dim=1) + 1e-12)
        return {
            "val_marker_residual_l2": float(leftover.mean() * m_norm),
            "val_null_space_preservation_l2": float(preserve.mean()),
        }
