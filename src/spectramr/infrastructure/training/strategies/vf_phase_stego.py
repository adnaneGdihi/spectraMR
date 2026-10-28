"""Phase-steganographic marker injection for the virtual-fiducial twin pass (arm m7)."""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.phase_steganography import PhaseSteganographicMarker


class PhaseStegoHook:
    """Embed the phase grid in the clean complex anatomy before the twin corrupts it.

    Motion then warps the grid, so the corrupted input carries the deformation in its phase
    while the reconstruction target stays the unmarked anatomy. The marker is built lazily
    because its grid size is the image size, known only from the first batch.
    """

    def __init__(self, cfg, device: torch.device) -> None:
        if cfg.learnable:
            raise ValueError(
                "training.phase_steganography.learnable=true has no optimizer wiring; "
                "set learnable=false."
            )
        self._cfg = cfg
        self._device = device
        self._marker: PhaseSteganographicMarker | None = None

    def _get(self, shape: torch.Size) -> PhaseSteganographicMarker:
        size = (int(shape[-2]), int(shape[-1]))
        if self._marker is None or self._marker.im_size != size:
            self._marker = PhaseSteganographicMarker(
                im_size=size,
                spatial_freq_x=self._cfg.spatial_freq_x,
                spatial_freq_y=self._cfg.spatial_freq_y,
                amplitude=self._cfg.amplitude,
                learnable=False,
            ).to(self._device)
        return self._marker

    def inject(self, clean_complex: torch.Tensor) -> torch.Tensor:
        return self._get(clean_complex.shape).inject(clean_complex)

    def scores(self, clean_complex: torch.Tensor) -> dict[str, float]:
        """``marker_residual_after_magnitude``: mean |  |marked| - |clean|  | (0 if invisible)."""
        marker = self._get(clean_complex.shape)
        residual = (marker.erase(marker.inject(clean_complex)) - clean_complex.abs()).abs()
        return {"val_marker_residual_after_magnitude": float(residual.mean())}

    def displacement_rmse_px(
        self,
        corrupted: torch.Tensor,
        joint_clean: torch.Tensor,
        displacement_px: torch.Tensor,
    ) -> dict[str, float]:
        """RMSE (pixels) of the displacement component the phase grid can see.

        The motion-induced phase is ``angle(corrupted * conj(joint_clean))``; to first order it
        is ``grad(phi_grid) . d``, so dividing by ``|grad phi|`` gives ``d`` along the grid
        gradient. That is compared with the twin's true displacement projected on the same
        direction, where the gradient is strong enough to invert. Natural-phase warping, B0 and
        noise are inside the measured phase, so this bounds what the marker can recover.
        """
        marker = self._get(corrupted.shape)
        h, w = corrupted.shape[-2:]
        two_pi = 2.0 * torch.pi
        a = float(marker.amplitude)
        fx, fy = float(marker.freq_x), float(marker.freq_y)
        gx_n = (
            a
            * two_pi
            * fx
            * torch.cos(two_pi * fx * marker._X)
            * torch.sin(two_pi * fy * marker._Y)
        )
        gy_n = (
            a
            * two_pi
            * fy
            * torch.sin(two_pi * fx * marker._X)
            * torch.cos(two_pi * fy * marker._Y)
        )
        gx, gy = gx_n * 2.0 / (w - 1), gy_n * 2.0 / (h - 1)  # rad per pixel
        gnorm = (gx**2 + gy**2).sqrt()
        usable = gnorm > 0.25 * gnorm.max()

        dphi = torch.angle((corrupted * joint_clean.conj()).mean(dim=1))  # [B, H, W]
        est = dphi / gnorm.clamp_min(1e-12)
        true = (displacement_px[:, 0] * gx + displacement_px[:, 1] * gy) / gnorm.clamp_min(1e-12)
        err = (est - true)[:, usable]
        return {"val_stego_displacement_rmse_px": float(err.pow(2).mean().sqrt())}
