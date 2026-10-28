"""The ULF unwarp built from ``training.geomamba_ulf.unwarp``, shared by train and infer.

One owner for the unwarp: the strategy and the inference strategy both build an
:class:`UlfUnwarp` from the same :class:`UnwarpConfig` and call :meth:`apply`, so
training and inference cannot disagree about the correction.

Only the Maxwell concomitant term is computed from the config. It is a static
field of the acquisition geometry, so its map is built once per spatial shape. B0
needs a per-sample map, which must arrive in the batch under ``b0_map_key``. The
gradient-nonlinearity term has no coefficient reader yet, so declaring it raises.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.config.schemas.training.geomamba_ulf import UnwarpConfig
from spectramr.data.batch_types import read_batch_field
from spectramr.infrastructure.physics.maxwell_concomitant import maxwell_concomitant_field_hz
from spectramr.infrastructure.physics.unwarp import PhysicalUnwarp

logger = logging.getLogger(__name__)


class UlfUnwarp:
    """Applies the configured B0 and Maxwell unwarp to a ``(B, C, D, H, W)`` volume.

    Args:
        cfg: the ``training.geomamba_ulf.unwarp`` block. Invalid combinations raise here.
    """

    def __init__(self, cfg: UnwarpConfig) -> None:
        self.cfg = cfg
        if cfg.enable_grad_nonlinearity:
            raise NotImplementedError(
                "enable_grad_nonlinearity needs a gradient-nonlinearity coefficient reader, "
                "which is not wired. Set enable_grad_nonlinearity: false."
            )
        if cfg.gnl_coeff_path is not None:
            raise ValueError(
                "gnl_coeff_path is set, but no reader consumes it. Remove the key or wire a reader."
            )
        if cfg.enable_b0 and cfg.b0_map_key is None:
            raise ValueError(
                "enable_b0 is true but b0_map_key is unset, so no B0 map can be read. "
                "Set b0_map_key or set enable_b0: false."
            )
        if not cfg.enable_b0 and cfg.b0_map_key is not None:
            raise ValueError("b0_map_key is set, but enable_b0 is false; it would be inert.")
        if len(cfg.gradient_amplitudes_mt_per_m) != 3 or len(cfg.voxel_size_mm) != 3:
            raise ValueError("gradient_amplitudes_mt_per_m and voxel_size_mm need 3 entries each")
        # An inactive block (every term off) carries no echo spacing, so the warp is never built.
        self._unwarp = (
            PhysicalUnwarp(
                echo_spacing_s=cfg.echo_spacing_s,
                shift_mode=cfg.shift_mode,
                pixel_bandwidth_hz=cfg.pixel_bandwidth_hz,
            )
            if self.active
            else None
        )
        self._maxwell_cache: dict[tuple[tuple[int, int, int], str], torch.Tensor] = {}
        self._beltrami_max: float | None = None

    @property
    def active(self) -> bool:
        """True when at least one term would change the volume."""
        return self.cfg.enable_b0 or self.cfg.enable_maxwell

    def apply(self, volume: torch.Tensor, batch: Any) -> torch.Tensor:
        """Return the unwarped volume, reading any B0 map from ``batch``."""
        if volume.ndim != 5:
            raise ValueError(f"UlfUnwarp expects (B, C, D, H, W), got {tuple(volume.shape)}")
        if not self.active:
            return volume
        b0_hz = None
        if self.cfg.enable_b0:
            key = str(self.cfg.b0_map_key)
            b0_hz = read_batch_field(batch, key)
            if b0_hz is None:
                raise ValueError(
                    f"enable_b0 needs the batch field {key!r}, which is absent. A missing B0 map "
                    "must not pass as no correction."
                )
        maxwell_hz = None
        if self.cfg.enable_maxwell:
            maxwell_hz = self.maxwell_map(tuple(volume.shape[-3:]), volume.device)
            if self.cfg.emit_beltrami_diagnostic and self._beltrami_max is None:
                self._beltrami_max = self._beltrami_max_of(tuple(volume.shape[-3:]), volume.device)
                logger.info(
                    "[UlfUnwarp] Maxwell displacement max |mu| = %.4g (shift_mode=%s)",
                    self._beltrami_max,
                    self.cfg.shift_mode,
                )
        assert self._unwarp is not None  # active, checked above
        return self._unwarp(volume, b0_hz=b0_hz, maxwell_hz=maxwell_hz)

    def maxwell_map(
        self, spatial_shape: tuple[int, int, int], device: torch.device
    ) -> torch.Tensor:
        """The ``(D, H, W)`` Maxwell offset in Hz, built once per shape and device."""
        key = (spatial_shape, str(device))
        cached = self._maxwell_cache.get(key)
        if cached is None:
            cached = maxwell_concomitant_field_hz(
                spatial_shape,
                tuple(float(v) for v in self.cfg.voxel_size_mm),  # type: ignore[arg-type]
                tuple(float(v) for v in self.cfg.gradient_amplitudes_mt_per_m),  # type: ignore[arg-type]
                self.cfg.field_strength_t,
                device=device,
            )
            self._maxwell_cache[key] = cached
        return cached

    def provenance(self) -> dict[str, Any]:
        """Declared knobs and the applied Beltrami diagnostic, for the snapshot."""
        return {
            "declared": self.cfg.model_dump(),
            "beltrami_max": self._beltrami_max,
        }

    def _beltrami_max_of(self, spatial_shape: tuple[int, int, int], device: torch.device) -> float:
        """Max |mu| of the Maxwell displacement, from the same shift model as the warp."""
        from spectramr.infrastructure.physics.epi_forward import beltrami_from_b0_and_gnl

        D, H, W = spatial_shape
        maxwell = self.maxwell_map(spatial_shape, device).to(torch.float32)
        assert self._unwarp is not None  # only reached with enable_maxwell
        axis_name = self._unwarp.shift_axis
        n_axis = H if axis_name == "H" else W
        if self.cfg.shift_mode == "epi":
            t_esp = float(self.cfg.echo_spacing_s)  # type: ignore[arg-type]
        else:
            # n_axis * t_esp = 1 / bandwidth, the readout shift per Hz.
            t_esp = 1.0 / (float(self.cfg.pixel_bandwidth_hz) * n_axis)  # type: ignore[arg-type]
        delta_b0 = maxwell.unsqueeze(1)  # (D, 1, H, W): one slice per batch row
        gnl = torch.zeros(D, 2, H, W, device=device, dtype=torch.float32)
        mu = beltrami_from_b0_and_gnl(
            delta_b0, gnl, t_esp=t_esp, phase_encode_axis=-2 if axis_name == "H" else -1
        )
        return float(mu.abs().max())


__all__ = ["UlfUnwarp"]
