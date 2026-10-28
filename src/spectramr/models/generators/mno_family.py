"""The five Mamba-Neural-Operator variants from ``TODO/Mamba_NO.md`` §3.

CS-MNO (the recommended combination) lives in
:mod:`spectramr.models.generators.cs_mno_operator`. This module ships the
remaining four variants as **registered model classes**, each a thin
wrapper around ``CSMNOOperator`` with the appropriate flag combination
preset. Adding them as separate classes (rather than YAML-only flag
flips) lets reviewers see the architectural identities cleanly in the
``EXPERIMENTS_YAML_CATALOG`` and lets the audit infer architecture
from ``model_type`` alone.

Mapping (spec §3 → flag combination):

================================================================
spec name          | scan_mode    | use_physical_arc | spectral
==================================================================
MNO-1 HS-MNO       | hilbert_2d   | False            | OFF
MNO-2 C-MNO        | hilbert_2d   | True             | OFF
MNO-3 ME-MNO       | hilbert_2d   | True             | ON  (with cubic-group averaging wrapper)
MNO-4 S-MNO        | hilbert_2d   | False            | ON
MNO-5 G-MNO        | metric SFC   | True             | ON  (geodesic-MST + DFS, image-domain)
=================================================================

CS-MNO (= C-MNO ⊕ S-MNO ⊕ Hilbert) is the proposed default; the four
alternatives ship as ablation/comparison baselines so a single
campaign can produce the comparative table from spec §5.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from spectramr.models.generators.cs_mno_operator import CSMNOOperator
from spectramr.models.registry import register_model

__all__ = [
    "CMNOOperator",
    "GMNOOperator",
    "HSMNOOperator",
    "MEMNOOperator",
    "SMNOOperator",
]


@register_model(
    name="hs_mno_operator",
    training_mode="reconstruction",
    # Capability set mirrored from ``cs_mno_operator``, the base this subclasses.
    # These were absent, which is not the same as "inherited": an undeclared field
    # reads as None, and the audit checks that consume it -- check_workflow_spatial_rank,
    # the signal-domain checks, the spec card -- do not run against None. So an arm
    # written against this name got no spatial-rank pre-flight while the identical arm
    # written against the base did (#1084).
    #
    # Copying is justified by measurement, not by inheritance: the variant differs from
    # the base ONLY in the ``use_physical_arc`` / ``disable_spectral`` kwargs it forces
    # below, and all three ranks were verified to construct AND complete a forward pass
    # -- 1-D (64,) via raster_1d, 2-D (32, 32) via hilbert_2d, 3-D (16, 16, 16) via
    # hilbert_3d -- exactly matching the base. Enforced by
    # tests/unit/models/test_registry_capability_parity.py.
    spatial_dims=(1, 2, 3),
    input_domain=("pde_grid", "image"),
    output_domain=("pde_grid", "image"),
    accepts_complex=False,
    requires_paired_data=True,
)
class HSMNOOperator(CSMNOOperator):
    """MNO-1: Hilbert-Scan Mamba Operator (no spectral, no physical arc).

    The "minimal" SFC-Mamba: just swaps raster scan for Hilbert in
    LMO. Provable locality (Proposition 1) but **not** discretisation-
    invariant. Used as the simplest baseline showing the SFC alone
    helps over raster, before either of the resolution-invariance
    mechanisms (physical Δt, spectral) is added.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("scan_mode", "hilbert_2d")
        kwargs["use_physical_arc"] = False
        kwargs["disable_spectral"] = True
        super().__init__(**kwargs)


@register_model(
    name="c_mno_operator",
    training_mode="reconstruction",
    # Capability set mirrored from ``cs_mno_operator``, the base this subclasses.
    # These were absent, which is not the same as "inherited": an undeclared field
    # reads as None, and the audit checks that consume it -- check_workflow_spatial_rank,
    # the signal-domain checks, the spec card -- do not run against None. So an arm
    # written against this name got no spatial-rank pre-flight while the identical arm
    # written against the base did (#1084).
    #
    # Copying is justified by measurement, not by inheritance: the variant differs from
    # the base ONLY in the ``use_physical_arc`` / ``disable_spectral`` kwargs it forces
    # below, and all three ranks were verified to construct AND complete a forward pass
    # -- 1-D (64,) via raster_1d, 2-D (32, 32) via hilbert_2d, 3-D (16, 16, 16) via
    # hilbert_3d -- exactly matching the base. Enforced by
    # tests/unit/models/test_registry_capability_parity.py.
    spatial_dims=(1, 2, 3),
    input_domain=("pde_grid", "image"),
    output_domain=("pde_grid", "image"),
    accepts_complex=False,
    requires_paired_data=True,
)
class CMNOOperator(CSMNOOperator):
    """MNO-2: Continuous-SFC Mamba Operator (no spectral, physical arc).

    Discretisation-invariant in the FNO sense (Proposition 2 of the
    spec) but intrinsically 1-D — restricted to operators whose
    reproducing kernel depends only on Hilbert-arc distance. Mostly
    interesting as a building block for the multi-curve extension
    (G-MNO, ME-MNO) and as the cleanest demonstration of the
    physical-Δt mechanism.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("scan_mode", "hilbert_2d")
        kwargs["use_physical_arc"] = True
        kwargs["disable_spectral"] = True
        super().__init__(**kwargs)


@register_model(
    name="s_mno_operator",
    training_mode="reconstruction",
    # Capability set mirrored from ``cs_mno_operator``, the base this subclasses.
    # These were absent, which is not the same as "inherited": an undeclared field
    # reads as None, and the audit checks that consume it -- check_workflow_spatial_rank,
    # the signal-domain checks, the spec card -- do not run against None. So an arm
    # written against this name got no spatial-rank pre-flight while the identical arm
    # written against the base did (#1084).
    #
    # Copying is justified by measurement, not by inheritance: the variant differs from
    # the base ONLY in the ``use_physical_arc`` / ``disable_spectral`` kwargs it forces
    # below, and all three ranks were verified to construct AND complete a forward pass
    # -- 1-D (64,) via raster_1d, 2-D (32, 32) via hilbert_2d, 3-D (16, 16, 16) via
    # hilbert_3d -- exactly matching the base. Enforced by
    # tests/unit/models/test_registry_capability_parity.py.
    spatial_dims=(1, 2, 3),
    input_domain=("pde_grid", "image"),
    output_domain=("pde_grid", "image"),
    accepts_complex=False,
    requires_paired_data=True,
)
class SMNOOperator(CSMNOOperator):
    """MNO-4: Spectral-SFC Hybrid Mamba Operator (no physical arc).

    The two-branch architecture but with a **discrete** scalar Δ
    instead of physical-arc Δt. Inherits FNO's universal-approximation
    guarantee on the global branch (Proposition 4) but the SFC-local
    branch is not honestly resolution-invariant — they only become so
    when combined with the C-MNO timestep, i.e. when promoted to full
    CS-MNO. Useful as the spec's pre-2025 "strong baseline" point.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("scan_mode", "hilbert_2d")
        kwargs["use_physical_arc"] = False
        kwargs["disable_spectral"] = False
        super().__init__(**kwargs)


# ── ME-MNO: cubic-group averaging wrapper ─────────────────────────────


# Cubic symmetry group on a 2-D grid: 8 elements
# (4 rotations × 2 reflections via D_4).
# Each element is a ``(rotation_k, flip_x, flip_y)`` tuple where
# rotation_k is the number of 90° rotations.
_D4_ELEMENTS: list[tuple[int, bool, bool]] = [
    (0, False, False),
    (1, False, False),
    (2, False, False),
    (3, False, False),
    (0, True, False),
    (1, True, False),
    (2, True, False),
    (3, True, False),
]


def _apply_d4(x: torch.Tensor, k: int, flip_x: bool, flip_y: bool) -> torch.Tensor:
    """Apply a D_4 group element to a ``[B, C, H, W]`` tensor."""
    if flip_x:
        x = torch.flip(x, dims=(-1,))
    if flip_y:
        x = torch.flip(x, dims=(-2,))
    if k:
        x = torch.rot90(x, k=k, dims=(-2, -1))
    return x


def _invert_d4(x: torch.Tensor, k: int, flip_x: bool, flip_y: bool) -> torch.Tensor:
    """Apply the inverse D_4 element to undo ``_apply_d4``."""
    if k:
        x = torch.rot90(x, k=-k, dims=(-2, -1))
    if flip_y:
        x = torch.flip(x, dims=(-2,))
    if flip_x:
        x = torch.flip(x, dims=(-1,))
    return x


# ── Oh: full 3-D octahedral / cubic point group (48 elements) ────────
#
# Each element is encoded as ``(perm, signs)`` where ``perm`` is a
# permutation of (D, H, W) axis indices (24 such permutations form
# the rotation subgroup once filtered by determinant) and ``signs`` is
# a tuple of ±1 sign flips per axis. We enumerate all 6×8 = 48
# perm-sign combinations, then drop the orientation-reversing ones to
# get exactly the 24 rotations; combining with a single inversion
# gives the full 48-element Oh group.


def _generate_oh_elements() -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Enumerate the 48 elements of the cubic point group Oh.

    Returns a list of ``(perm, signs)`` where:
      * ``perm`` is a permutation of (0, 1, 2) — which axis goes to
        which output position, applied to the trailing 3 spatial dims.
      * ``signs`` is a triple in ``{-1, +1}^3`` — per-axis flip.

    The 24 rotations have ``det(P · diag(signs)) == +1``; combining
    each rotation with the central inversion gives 24 more elements
    for a total of 48 (the full Oh group).
    """
    from itertools import permutations, product

    elements = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            elements.append((perm, signs))
    assert len(elements) == 48
    return elements


_OH_ELEMENTS = _generate_oh_elements()


def _apply_oh(
    x: torch.Tensor,
    perm: tuple[int, int, int],
    signs: tuple[int, int, int],
) -> torch.Tensor:
    """Apply an Oh-group element to ``[B, C, D, H, W]``.

    Order of operations: flip axes per ``signs``, then permute
    spatial dims per ``perm``.
    """
    # Flips first.
    flip_dims = tuple(-3 + i for i, s in enumerate(signs) if s == -1)
    if flip_dims:
        x = torch.flip(x, dims=flip_dims)
    # Permute the trailing three axes per ``perm`` (without disturbing
    # the leading B and C axes). torch.permute requires all dims.
    full_perm = (0, 1, 2 + perm[0], 2 + perm[1], 2 + perm[2])
    x = x.permute(*full_perm).contiguous()
    return x


def _invert_oh(
    x: torch.Tensor,
    perm: tuple[int, int, int],
    signs: tuple[int, int, int],
) -> torch.Tensor:
    """Apply the inverse of an Oh element (reverse order)."""
    # Inverse of permute first.
    inv_perm = [0, 0, 0]
    for i, p in enumerate(perm):
        inv_perm[p] = i
    full_perm = (0, 1, 2 + inv_perm[0], 2 + inv_perm[1], 2 + inv_perm[2])
    x = x.permute(*full_perm).contiguous()
    # Then flip — flips are involutions so same dims.
    flip_dims = tuple(-3 + i for i, s in enumerate(signs) if s == -1)
    if flip_dims:
        x = torch.flip(x, dims=flip_dims)
    return x


@register_model(
    name="me_mno_operator",
    training_mode="reconstruction",
    # Wraps CSMNOOperator with cubic-group averaging → same data contract as
    # cs_mno_operator (real-valued PDE-grid / MRI-image tensors), but the d4/oh
    # symmetry groups are 2-D / 3-D only.
    spatial_dims=(2, 3),
    input_domain=("pde_grid", "image"),
    output_domain=("pde_grid", "image"),
    accepts_complex=False,
    requires_paired_data=True,
)
class MEMNOOperator(nn.Module):
    """MNO-3: Multi-curve Equivariant Mamba Operator.

    Wraps a CSMNOOperator with cubic-group (D_4 in 2-D, full 48-element
    cubic group in 3-D) averaging. By the standard group-averaging
    argument (Proposition 3), the wrapped operator is **exactly**
    equivariant under the discrete cube symmetry group. The continuous-
    rotation claim from spec §3.3 is **not** asserted by this
    implementation — the spec correctly notes the lattice approximation
    is too coarse at moderate ``N`` to claim continuous equivariance.

    The cost is exactly ``|G|×`` forward + backward; in 2-D ``|G|=8``,
    in 3-D this is 48× and likely too expensive for full training but
    runnable for inference-time test-time-augmentation.

    Args:
        symmetry_group:  One of:
          * ``"identity"`` — no averaging (1 element, ablation).
          * ``"d4"`` — 2-D dihedral group, 8 elements. Use with
            2-D inputs ``[B, C, H, W]``.
          * ``"oh"`` — full 3-D octahedral / cubic group, 48 elements.
            Use with 3-D inputs ``[B, C, D, H, W]``. The architecture
            backbone must also be 3-D (``scan_mode="hilbert_3d"``,
            ``spatial_shape=(D, H, W)``).
        backbone_kwargs: passed verbatim to ``CSMNOOperator``.
    """

    def __init__(
        self,
        symmetry_group: str = "d4",
        **backbone_kwargs: Any,
    ) -> None:
        super().__init__()
        if symmetry_group not in ("d4", "identity", "oh"):
            raise ValueError(
                f"MEMNOOperator: symmetry_group must be one of "
                f"{{'identity', 'd4', 'oh'}}, got {symmetry_group!r}"
            )
        self.symmetry_group = symmetry_group
        self.backbone = CSMNOOperator(**backbone_kwargs)

        # Cross-check: 'oh' requires a 3-D backbone, 'd4' a 2-D backbone.
        backbone_rank = len(self.backbone.spatial_shape)
        if symmetry_group == "oh" and backbone_rank != 3:
            raise ValueError(
                "symmetry_group='oh' requires a 3-D backbone "
                "(spatial_shape (D, H, W) and scan_mode='hilbert_3d'); "
                f"got rank {backbone_rank} backbone"
            )
        if symmetry_group == "d4" and backbone_rank != 2:
            raise ValueError(
                "symmetry_group='d4' requires a 2-D backbone "
                "(spatial_shape (H, W) and scan_mode='hilbert_2d'); "
                f"got rank {backbone_rank} backbone"
            )

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.backbone.set_grad_checkpointing(enable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Group-averaging: f̂(x) = mean_g g⁻¹·f(g·x)
        if self.symmetry_group == "oh":
            outputs = [
                _invert_oh(self.backbone(_apply_oh(x, perm, signs)), perm, signs)
                for perm, signs in _OH_ELEMENTS
            ]
        elif self.symmetry_group == "d4":
            outputs = [
                _invert_d4(self.backbone(_apply_d4(x, k, fx, fy)), k, fx, fy)
                for k, fx, fy in _D4_ELEMENTS
            ]
        else:  # identity
            outputs = [self.backbone(x)]
        return torch.stack(outputs, dim=0).mean(dim=0)


# ── G-MNO: geodesic-SFC operator using the existing MetricSFCLinearizer ──


@register_model(name="g_mno_operator", training_mode="reconstruction")
class GMNOOperator(nn.Module):
    """MNO-5: Geodesic-Metric Mamba Operator.

    Replaces the Hilbert curve with a **geodesic** SFC that minimises a
    Riemannian path length under an intensity-aware metric, computed by
    a 2-approximate TSP via Kruskal MST + DFS. The implementation
    reuses ``MetricSFCLinearizer`` from the GeoMamba-ULF work — the
    same data structure, just used for a different purpose: the spec
    motivates it for non-rectangular domains, the GeoMamba-ULF code
    uses it for foreground-restricted ULF→HF.

    Strength (per spec §3.5): the only proposal that addresses
    non-rectangular domains, where most medical imaging lives.

    Limitations:
        - Proposition 5 requires bounded sectional curvature and
          injectivity radius bounded below; the implementation does
          not check these and silently underperforms on cusped meshes.
        - The MST+DFS is precomputed at ``__init__`` from a fixed
          mask; dynamic shapes are not supported.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_shape: tuple[int, int, int],
        n_layers: int = 4,
        modes: int = 8,
        dim: int = 32,
        d_state: int = 16,
        beta: float = 10.0,
        use_physical_arc: bool = True,
        disable_spectral: bool = False,
        activation: str = "gelu",
        **_unused: Any,
    ) -> None:
        super().__init__()

        if len(spatial_shape) != 3:
            raise ValueError(
                f"GMNOOperator is 3-D only (geodesic SFC on volumetric "
                f"masks); got shape {spatial_shape}"
            )

        # Lazy import — the GeoMamba-ULF code lives in src/models/blocks/
        # and pulls in TorchIO + scipy. Load on construction so the
        # registry decorator above doesn't drag heavy deps at import time.
        from spectramr.models.blocks.cs_mno_layer import CSMNOLayer

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_shape = tuple(spatial_shape)
        self.use_physical_arc = use_physical_arc
        self.beta = beta
        # F37 / 2026-05-22 — cache geodesic-SFC linearizations for shapes
        # other than the construction shape so validation at a different
        # resolution doesn't crash (see CSMNOOperator for the full rationale).
        # The MST geodesic build is expensive, so this is computed once per
        # unique 3-D shape and cached. Indices are non-learnable → not in
        # the state_dict.
        self._dynamic_linearizers: dict[
            tuple[int, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

        self.lift = nn.Conv3d(in_channels, dim, kernel_size=1)
        self.project = nn.Conv3d(dim, out_channels, kernel_size=1)

        self.layers = nn.ModuleList(
            [
                CSMNOLayer(
                    dim=dim,
                    modes=modes,
                    d_state=d_state,
                    spatial_dims=3,
                    use_physical_arc=use_physical_arc,
                    disable_spectral=disable_spectral,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )

        # Precompute the metric-SFC permutation. MetricSFCLinearizer
        # uses an all-ones foreground mask and a zero intensity volume
        # by default (so the metric reduces to plain Euclidean MST,
        # giving a "vanilla geodesic" curve over the whole grid).
        # Downstream applications that want intensity-aware geodesics
        # should subclass this generator and pass a real intensity
        # volume + mask at construction time.
        forward_idx, inverse_idx, delta = self._build_metric_sfc(self.spatial_shape)
        self.register_buffer("forward_idx", forward_idx)
        self.register_buffer("inverse_idx", inverse_idx)
        self.register_buffer("delta_phys", delta if use_physical_arc else torch.zeros(0))

    def _build_metric_sfc(
        self, spatial_shape: tuple[int, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ``(forward_idx, inverse_idx, delta)`` for a 3-D shape.

        MetricSFCLinearizer uses an all-ones foreground mask and a zero
        intensity volume (so the metric reduces to plain Euclidean MST — a
        "vanilla geodesic" over the whole grid). Subclasses wanting
        intensity-aware geodesics override this.
        """
        from spectramr.models.blocks.metric_sfc import MetricSFCLinearizer

        D, H, W = spatial_shape
        mask = torch.ones(D, H, W, dtype=torch.bool)
        intensity = torch.zeros(D, H, W, dtype=torch.float32)
        linearizer = MetricSFCLinearizer(
            intensity_volume=intensity,
            mask=mask,
            beta=self.beta,
            cache_dir=None,  # disable disk cache for the model-side use
        )
        forward_idx = linearizer.forward_idx.clone()
        # MetricSFCLinearizer's inverse_idx uses -1 sentinels for background
        # voxels. With an all-ones mask none remain, but clamp to be safe.
        inverse_idx = linearizer.inverse_idx.clone().clamp_min_(0)
        # Per-token physical Δt from the ordered geodesic coordinates.
        d_coords, h_coords, w_coords = np.unravel_index(
            linearizer.forward_idx.cpu().numpy(), (D, H, W)
        )
        ord_coords = np.stack([d_coords, h_coords, w_coords], axis=1).astype(np.float32)
        ord_coords = ord_coords / np.array([D, H, W], dtype=np.float32)
        steps = np.linalg.norm(np.diff(ord_coords, axis=0), axis=1)
        delta = np.concatenate([[0.0], steps]).astype(np.float32)
        return forward_idx, inverse_idx, torch.from_numpy(delta)

    def _linearization_for(
        self, spatial_shape: tuple[int, ...], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Resolve the geodesic-SFC permutation for ``spatial_shape`` (cached)."""
        if spatial_shape == self.spatial_shape:
            delta = self.delta_phys if self.use_physical_arc else None
            return self.forward_idx, self.inverse_idx, delta
        cached = self._dynamic_linearizers.get(spatial_shape)
        if cached is None:
            fwd, inv, delta = self._build_metric_sfc(spatial_shape)  # type: ignore[arg-type]
            cached = (fwd.to(device), inv.to(device), delta.to(device))
            self._dynamic_linearizers[spatial_shape] = cached
        fwd, inv, delta = cached
        if fwd.device != device:
            fwd, inv, delta = fwd.to(device), inv.to(device), delta.to(device)
            self._dynamic_linearizers[spatial_shape] = (fwd, inv, delta)
        return fwd, inv, (delta if self.use_physical_arc else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_shape = tuple(x.shape[2:])
        # F37 — GMNO is 3-D only. A rank mismatch (e.g. a 2-D [H, W] batch fed
        # to a 3-D operator) is a config/data error, not a resolution
        # difference, and must fail loudly. Same-rank extent differences
        # (training patch vs full-res validation) are resolved on demand.
        if len(in_shape) != 3:
            raise ValueError(
                f"GMNOOperator is 3-D only (built for {self.spatial_shape}) but "
                f"received spatial dims {in_shape}. This is a data/config "
                f"mismatch — the dataset is yielding a rank-{len(in_shape)} "
                f"tensor. Use a 3-D dataset/patch or a 2-D operator (cs_mno_operator)."
            )
        forward_idx, inverse_idx, delta = self._linearization_for(in_shape, x.device)
        h = self.lift(x)
        for layer in self.layers:
            h = layer(h, forward_idx, inverse_idx, delta)
        return self.project(h)
