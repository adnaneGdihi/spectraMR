"""CS-MNO neural-operator generator.

Composes ``L`` ``CSMNOLayer`` blocks between a 1×1 lifting conv and
1×1 projection conv. The Hilbert permutation and per-token physical
Δt are pre-computed once at ``__init__`` from the spatial shape and
stored on the model as buffers.

Registered as ``"cs_mno_operator"`` with training mode
``"reconstruction"`` — the operator is a model architecture, not a
new training paradigm. The standard ``ReconstructionTrainingStrategy``
trains it like any other paired-input → paired-output model.

See ``docs/cs_mno.rst`` for math + usage and
``TODO/Mamba_NO.md`` for the full design spec (Theorems A and B).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spectramr.models.blocks.cs_mno_layer import CSMNOLayer
from spectramr.models.blocks.topology_linearizer import (
    ImageTopologyLinearizer,
    resolution_agnostic_mode,
)
from spectramr.models.generators.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.registry import register_model

#: ``physical``: the per-token arc length of the curve (Theorem A).
#: ``fixed_resolution``: one scalar step ``N**(-1/d)`` on every token, the
#: scale-matched control for the arc arm (same magnitude, no per-token term).
ARC_STEP_MODES = ("physical", "fixed_resolution")


def _arc_delta(
    linearizer: ImageTopologyLinearizer, spatial_shape: tuple[int, ...], arc_step: str
) -> torch.Tensor:
    """Per-token Δt for ``spatial_shape`` under the declared ``arc_step`` mode."""
    if arc_step == "physical":
        return linearizer.physical_arc_delta()
    n_tokens = math.prod(spatial_shape)
    return torch.full((n_tokens,), n_tokens ** (-1.0 / len(spatial_shape)), dtype=torch.float32)

__all__ = ["CSMNOOperator"]


@register_model(
    name="cs_mno_operator",
    training_mode="reconstruction",
    # Mamba is natively 1-D (selective SSM over a sequence). The
    # surrounding FNO spectral branch + lift/project convs now have
    # 1-D / 2-D / 3-D variants, so the operator works on:
    #   1-D — Burgers / Allen–Cahn benchmarks (sequence input)
    #   2-D — Darcy / MRI image data
    #   3-D — volumetric MRI data
    spatial_dims=(1, 2, 3),
    # The same operator works on PDE benchmark grids and MRI
    # image-domain data — both are real-valued [B, C, *spatial]
    # tensors. Declaring both lets the audit accept either dataset.
    input_domain=("pde_grid", "image"),
    output_domain=("pde_grid", "image"),
    accepts_complex=False,
    requires_paired_data=True,
)
class CSMNOOperator(GradCheckpointingMixin, nn.Module):
    """Continuous-SFC Spectral-Local Mamba Neural Operator.

    Args:
        in_channels:      Input channel count.
        out_channels:     Output channel count (typically equals
                          in_channels for super-resolution / denoising).
        spatial_shape:    Tuple ``(H, W)`` for 2-D or ``(D, H, W)`` for
                          3-D. Permutation is precomputed for this
                          fixed shape; if your dataloader yields a
                          different shape the model will raise.
        n_layers:         Number of CSMNOLayer stacks.
        modes:            Spectral truncation modes per axis.
        dim:              Hidden width through the stack.
        d_state:          Mamba state dimension.
        scan_mode:        Linearisation mode (e.g. ``"hilbert_2d"``,
                          ``"raster_2d"``). Forbidden values raise
                          at construction time — no silent fallback.
        use_physical_arc: If True, scale the Mamba Δt by the physical
                          arc length of the SFC step (Theorem A).
        arc_step:         ``"physical"`` (default) or ``"fixed_resolution"``.
                          The latter replaces the per-token arc with one
                          scalar step ``N**(-1/d)``; it needs
                          ``use_physical_arc=True`` and is the control arm.
        disable_spectral: If True, drop the FNO spectral branch from
                          every layer. Used by HS-MNO (no spectral,
                          discrete Δ) and C-MNO (no spectral, physical
                          arc) variants from the spec.
        activation:       Pointwise nonlinearity name.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_shape: tuple[int, ...],
        n_layers: int = 4,
        modes: int = 16,
        dim: int = 32,
        d_state: int = 16,
        scan_mode: str = "hilbert_2d",
        use_physical_arc: bool = True,
        disable_spectral: bool = False,
        activation: str = "gelu",
        arc_step: str = "physical",
        **_unused: Any,  # absorb config keys not relevant here
    ) -> None:
        super().__init__()

        spatial_dims = len(spatial_shape)
        if spatial_dims not in (1, 2, 3):
            raise ValueError(
                f"CSMNOOperator: spatial_shape must be 1-D, 2-D, or 3-D, got rank {spatial_dims}"
            )
        if arc_step not in ARC_STEP_MODES:
            raise ValueError(
                f"CSMNOOperator: arc_step must be one of {ARC_STEP_MODES}, got {arc_step!r}"
            )
        if arc_step != "physical" and not use_physical_arc:
            raise ValueError(
                f"CSMNOOperator: arc_step={arc_step!r} only applies with use_physical_arc=True; "
                "with use_physical_arc=False the step is the learnt scalar and the knob would be inert."
            )
        # Cross-check that the scan mode dimension matches the spatial rank.
        if scan_mode.endswith("_3d") and spatial_dims != 3:
            raise ValueError(
                f"scan_mode={scan_mode!r} requires 3-D spatial_shape, got {spatial_shape}"
            )
        if scan_mode.endswith("_2d") and spatial_dims != 2:
            raise ValueError(
                f"scan_mode={scan_mode!r} requires 2-D spatial_shape, got {spatial_shape}"
            )
        if scan_mode.endswith("_1d") and spatial_dims != 1:
            raise ValueError(
                f"scan_mode={scan_mode!r} requires 1-D spatial_shape, got {spatial_shape}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_shape = tuple(spatial_shape)
        self.n_layers = n_layers
        self.dim = dim

        # Mamba is natively 1-D — the only thing that needs a per-rank
        # branch is the FNO spectral path and the lift/project convs.
        if spatial_dims == 1:
            conv_cls = nn.Conv1d
        elif spatial_dims == 2:
            conv_cls = nn.Conv2d
        else:
            conv_cls = nn.Conv3d
        self.lift = conv_cls(in_channels, dim, kernel_size=1)
        self.project = conv_cls(dim, out_channels, kernel_size=1)

        self.layers = nn.ModuleList(
            [
                CSMNOLayer(
                    dim=dim,
                    modes=modes,
                    d_state=d_state,
                    spatial_dims=spatial_dims,
                    use_physical_arc=use_physical_arc,
                    disable_spectral=disable_spectral,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )

        # Permutation + per-token Δt for the construction shape: cache
        # as registered buffers so they move with .to(device) / .cuda().
        self.scan_mode = scan_mode
        linearizer = ImageTopologyLinearizer(self.spatial_shape, mode=scan_mode)
        self.register_buffer("forward_idx", linearizer.forward_idx.clone())
        self.register_buffer("inverse_idx", linearizer.inverse_idx.clone())
        self.register_buffer(
            "delta_phys",
            _arc_delta(linearizer, self.spatial_shape, arc_step)
            if use_physical_arc
            else torch.zeros(0),
        )
        self.use_physical_arc = use_physical_arc
        self.arc_step = arc_step
        # F37 / 2026-05-22 — the SFC linearization indices were fixed at
        # construction and ``forward`` hard-raised on any other shape. That
        # crashed validation for every *_mno arm: training feeds the
        # configured patch (e.g. 128×128) but validation feeds full-res
        # (221×221), so every val batch raised ValueError, validation
        # produced zero metrics, and no images were saved — yet the run
        # still exited green (smoke 20260521: c_mno / cs_mno_v0 / g_mno /
        # hs_mno / me_mno / s_mno all "passed" with no images). The Mamba
        # scan and the rfft spectral branch are both shape-agnostic; only
        # the precomputed permutation was rigid. Recompute (and cache, keyed
        # by shape) the linearization on demand so the operator is
        # resolution-agnostic. Indices are non-learnable, so they stay out
        # of the state_dict.
        self._dynamic_linearizers: dict[
            tuple[int, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def _linearization_for(
        self, spatial_shape: tuple[int, ...], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(forward_idx, inverse_idx, delta)`` for ``spatial_shape``.

        Uses the registered buffers when the shape matches the construction
        shape; otherwise builds the linearizer once and caches it. All tensors
        are returned on ``device``.
        """
        if spatial_shape == self.spatial_shape:
            delta = self.delta_phys if self.use_physical_arc else None
            return self.forward_idx, self.inverse_idx, delta

        cached = self._dynamic_linearizers.get(spatial_shape)
        if cached is None:
            # F37 follow-up (2026-06): the strict ``hilbert_2d`` / ``hilbert_3d``
            # orderings assert square / cubic power-of-2 dims, so recomputing
            # them on a non-power-of-2 full-resolution validation shape (e.g.
            # 221x221) re-raised the very ValueError F37 set out to kill -
            # validation produced zero metrics yet the run exited green. The
            # resolution-agnostic ``*_rect`` curves are byte-identical on the
            # square/pow-2 construction patch but valid on any extent, so the
            # train and val scans stay in the SAME Hilbert family.
            dyn_mode = resolution_agnostic_mode(self.scan_mode)
            linearizer = ImageTopologyLinearizer(spatial_shape, mode=dyn_mode)
            fwd = linearizer.forward_idx.clone().to(device)
            inv = linearizer.inverse_idx.clone().to(device)
            delta = (
                _arc_delta(linearizer, spatial_shape, self.arc_step).to(device)
                if self.use_physical_arc
                else torch.zeros(0, device=device)
            )
            cached = (fwd, inv, delta)
            self._dynamic_linearizers[spatial_shape] = cached
        fwd, inv, delta = cached
        if fwd.device != device:
            fwd, inv = fwd.to(device), inv.to(device)
            delta = delta.to(device)
            self._dynamic_linearizers[spatial_shape] = (fwd, inv, delta)
        return fwd, inv, (delta if self.use_physical_arc else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on a fixed-shape input.

        Args:
            x: ``[B, C_in, *spatial_shape]``. TorchIO yields 5-D tensors
                ``[B, C, H, W, D]`` even for 2-D data and ``[B, C, X, 1, 1]``
                for 1-D PDE benchmarks; trailing singleton spatial dims are
                squeezed to match ``self.spatial_shape``.

        Returns:
            ``[B, C_out, *spatial_shape]`` followed by the trailing singleton
            dims the input carried, so the output has the input's rank.
        """
        expected_rank = len(self.spatial_shape)
        squeezed = 0
        while x.ndim - 2 > expected_rank and x.shape[-1] == 1:
            x = x.squeeze(-1)
            squeezed += 1
        in_shape = tuple(x.shape[2:])
        if len(in_shape) != expected_rank:
            raise ValueError(
                f"CSMNOOperator expects a rank-{expected_rank} spatial input "
                f"(built for {self.spatial_shape}) but received spatial dims "
                f"{in_shape}. Channel/rank mismatches are not auto-resolved; "
                f"only differing extents along the same number of axes are."
            )

        # F37 — resolve the SFC permutation for the actual input shape
        # (recomputed + cached for shapes other than the construction shape).
        forward_idx, inverse_idx, delta = self._linearization_for(in_shape, x.device)

        h = self.lift(x)
        for layer in self.layers:
            if self._checkpointing_active():
                h = checkpoint(layer, h, forward_idx, inverse_idx, delta, use_reentrant=False)
            else:
                h = layer(h, forward_idx, inverse_idx, delta)
        out = self.project(h)
        # The target keeps the singleton dims; a lower-rank output cannot be scored against it.
        return out.reshape(*out.shape, *([1] * squeezed))
