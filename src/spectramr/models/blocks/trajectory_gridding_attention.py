"""Attention over off-grid k-space samples, without gridding them first.

Off-grid k-space is a SET of pairs ``{(k_j, y_j)}``, not an array, so a score
cannot come from an array index. The position encoding is the coordinate
itself, lifted by a random feature map whose inner product is a Gaussian kernel
in the coordinate difference.

That is what makes the block an interpolator rather than an analogy: the
attention weight over ``[H*W, N]`` IS the gridding kernel.

**It does not yet start where ``adjoint_project`` does, and the cause is now
established: the operator form.** Measured untrained on a phantom against the
fixed Kaiser-Bessel adjoint:

===========================================  =========  =========
quantity                                          32^2      256^2
===========================================  =========  =========
fixed Kaiser-Bessel adjoint (rel_err)           0.0410     0.0349
this block, untrained (rel_err)                 0.9723     0.9748
ratio                                            23.7x      28.0x
===========================================  =========  =========

Gridding is a density-compensated SUM followed by deapodization; this block
computes a normalised AVERAGE and no deapodization, and the two are not the
same operator at any kernel width. An exact dense Gaussian on the same
trajectory separates them, relative error against the same phantom:

======================  ==========  ==============  ==========  ==============
sigma (grid cells)      sum, raw    sum, deapod.    avg, raw    avg, deapod.
======================  ==========  ==============  ==========  ==============
0.5                       0.1517          0.0664      0.2767          0.2255
0.7                       0.2789          0.0550      0.4516          0.4344
0.9                       0.4147          0.0526      0.6019          0.7691
======================  ==========  ==============  ==========  ==============

So the sum form lands within 1.3x of the 0.0410 baseline once deapodized, and
the average form does not converge at any width -- deapodization makes it
worse, because dividing by the kernel's transform only inverts a convolution
and the average is not one. Earlier readings that cleared deapodization and
kernel width tested them on the average form, where neither can help.

Both fixes are now in -- :func:`sample_density_attention` is the Pipe-Menon
sum, and the image is divided by the kernel's transform -- and together they
moved the block only 23.7x -> 22.8x, because a third fault dominates and is
fatal to the approach as designed.

**The random features cannot represent a kernel this peaked in float32.**
``phi(u) = exp(w . u - |u|^2)`` computes an order-1 kernel as the product of a
very small number and a very large one. At sigma = 0.7 cells the coordinates
reach ``|u|^2 = 522``, while float32 ``exp`` underflows below about -88, so
**57% of the sample features are exactly zero** and the realised kernel is
wrong in both directions at once: 0.0802 at half a cell where the Gaussian
gives 0.7748, and a 0.0054 floor at four cells where the Gaussian is 0.
Narrowing the kernel to the width gridding needs makes this worse, since ``u``
scales as ``1/sigma``.

Neither more precision nor more features rescues it, and the arithmetic says so
rather than a guess. Factoring the per-position scalars out of the features --
``phi = s(u) psi(u)`` with ``psi`` bounded by construction -- leaves ``s`` at
``e^-467`` at the k-space edge, so ``<psi, psi>`` would have to reach ``e^933``,
405 decades, for the kernel to come out 1 at zero distance. In EXACT log-domain
arithmetic the estimate is short by 938; raising the feature count from 256 to
1,000,000 closes 128 of that. The failure is the estimator's variance at
``|u| ~ 23``, not the storage format, and no accumulation strategy addresses
variance.

The kernel itself is right: evaluated EXACTLY, as a density-compensated sum
with deapodization, it reaches 0.0526 against the fixed kernel's 0.0410. So
what has to change is the estimator, not the physics. A gridding kernel has
compact support -- a few cells -- which is the case random features are worst
at and a local gather is best at, and it is what makes real gridding
``O(H*W*n_neighbours)`` rather than ``O(H*W*N)``.

``test_the_untrained_block_is_measured_against_the_fixed_kernel`` pins the ratio
so a fix has to move it. Until then the fixed-kernel control arm is the stronger
starting point, and this block is not ready to be an arm.

Density compensation is the other half of the physics, and it belongs to the
SAMPLE. ``c_j = m_j / sum_j' K(k_j, k_j')`` is large where spokes are sparse and
small where they bunch -- the k-space centre of a radial scan -- which is the
quantity ``RadialDCF`` and ``IterativeDCF`` compute with a fixed rule, here
learned through the kernel. Normalising per GRID POINT instead, as linear
attention does by default, produces an average rather than a sum and is the
defect named above.

Phase needs no defending on this path. The scores read COORDINATES, which are
real, so nothing in them can rotate with the data; the values stay complex, so
a global ``e^{i phi}`` passes straight through.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.blocks.null_space_attention import sample_density_attention
from spectramr.models.layers.complex_conv import ComplexConv2d

#: Gaussian sigma in BASE grid cells. The reconstruction minimum against the
#: fixed-kernel adjoint is flat across 0.5-0.9 cells; wider than that and the
#: deapodization cannot be inverted at the image edge.
KERNEL_SIGMA_CELLS: float = 0.7

#: Smallest apodization value the deapodization divide may invert. Enforced as
#: a BOUND ON THE WIDTH rather than a clamp on the image: clamping per pixel
#: would leave the outer image silently deapodized by the wrong factor, and
#: checking the realised minimum would put a host sync in the training loop
#: (non-negotiable 9). At sigma = 1.5 cells the edge value is about 1e-8 and
#: the reconstruction comes back at 0.99.
DEAPODIZATION_FLOOR: float = 1e-3

__all__ = ["TrajectoryFourierFeatures", "TrajectoryGriddingAttention"]


class TrajectoryFourierFeatures(nn.Module):
    """Positive random features whose inner product is a Gaussian kernel.

    ``phi(u) = exp(w . u - |u|^2)`` with ``u = k / width`` and ``w`` standard
    normal, so ``<phi(u_i), phi(u_j)>`` estimates ``exp(-|u_i - u_j|^2 / 2)``.

    Positivity is not cosmetic. Linear attention divides by a sum of these
    features, so the textbook ``[cos(Wk), sin(Wk)]`` Bochner pair would give a
    signed normaliser that can pass through zero. It is also why the scores are
    not pushed through ``elu(|.|) + 1`` the way the grid-domain blocks do: that
    map destroys the kernel structure, and measured on a phantom it left the
    untrained block 24x worse than the fixed Kaiser-Bessel adjoint instead of
    comparable to it.

    Args:
        num_features: Output width.
        width: Kernel width in k-space units; larger is a wider kernel.
        seed: Fixes ``w`` so two constructions of one arm grid identically.
    """

    def __init__(self, num_features: int = 64, width: float = 1.0, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        # Both are PARAMETERS: the whole claim is that the interpolation kernel
        # is learned. The width moves freely DOWNWARD but meets a hard ceiling
        # upward -- see ``max_sigma`` -- so the default sits about 17% below a
        # bound the optimizer cannot cross and gets no gradient signal at.
        self.frequencies = nn.Parameter(torch.randn(2, int(num_features), generator=generator))
        self.log_width = nn.Parameter(torch.tensor(float(width)).log())
        self.num_features = int(num_features)

    def forward(self, coords: Tensor) -> Tensor:
        """Map ``[B, 2, N]`` coordinates to positive ``[B, num_features, N]``."""
        if coords.shape[-2] != 2:
            raise ValueError(
                f"expected 2-D k-space coordinates [B, 2, N], got {tuple(coords.shape)}."
            )
        scaled = coords / self.log_width.exp().clamp_min(1e-6)
        projected = torch.einsum("bcn,cf->bfn", scaled, self.frequencies)
        logits = projected - scaled.pow(2).sum(dim=-2, keepdim=True)
        # ONE scalar for the whole tensor. A per-position max would rescale each
        # sample by its own factor, which sits inside the sum over samples and so
        # does NOT cancel in the attention ratio -- it reweights the data.
        # Measured with a per-position shift the realised kernel decayed to
        # 0.511 at one grid cell where the Gaussian gives 0.986.
        return (logits - logits.amax().detach()).exp()


class TrajectoryGriddingAttention(nn.Module):
    """Cross-attend Cartesian grid queries to off-grid sample keys.

    The samples are never interpolated onto the grid before the block runs --
    the block IS the interpolation, and its kernel is learned.

    Args:
        im_size: Target matrix; the grid queries are its bin coordinates.
        complex_channels: Complex channels carried by the samples.
        num_features: Random-feature width.
        num_heads: Heads over the feature axis.
        sigma_cells: Initial Gaussian width in BASE grid cells. Learned from
            there; see :data:`KERNEL_SIGMA_CELLS` for how the default was
            chosen.
        seed: Fixes the random directions.
    """

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        complex_channels: int = 1,
        num_features: int = 64,
        num_heads: int = 4,
        sigma_cells: float = KERNEL_SIGMA_CELLS,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.im_size = (int(im_size[0]), int(im_size[1]))
        self.complex_channels = int(complex_channels)
        self.num_heads = int(num_heads)
        if num_features % num_heads != 0:
            raise ValueError(f"num_features {num_features} must divide by num_heads {num_heads}.")
        # A Gaussian sigma in BASE grid cells, not the Kaiser-Bessel support
        # `kb_numpoints` counts. Swept against the fixed-kernel adjoint on a
        # phantom, the reconstruction minimum sits between 0.5 and 0.9 cells
        # and is flat across it; the old reading put the sigma at 1.5 cells,
        # where the deapodization below has already decayed to 1e-8 at the
        # image edge and cannot be inverted.
        if float(sigma_cells) <= 0.0:
            raise ValueError(f"sigma_cells must be positive, got {sigma_cells}.")
        cell = 2.0 * math.pi / float(self.im_size[0])
        width = float(sigma_cells) * cell
        self.features = TrajectoryFourierFeatures(num_features=num_features, width=width, seed=seed)
        self.eps = 1e-6
        # Applied to the VALUES, before attention: on a multi-coil arm this is
        # where the block learns coil weighting, which an output-side scale
        # could not express. bias=False is load-bearing -- a complex linear map
        # obeys W(e^{i phi} z) == e^{i phi} (W z) and adding a bias does not, so
        # the U(1) equivariance would be lost in one line. Measured 4.5e-01 with
        # the bias against 1.9e-09 without.
        self.value_proj = ComplexConv2d(self.complex_channels, self.complex_channels, 1, bias=False)
        self.register_buffer("grid_coords", self._grid_coordinates(), persistent=False)
        self.register_buffer("image_radius_sq", self._image_radius_sq(), persistent=False)
        # The widest kernel whose apodization is still invertible at the image
        # corner. The width is learned, so this bound travels with it.
        self.max_sigma = math.sqrt(-2.0 * math.log(DEAPODIZATION_FLOOR)) / math.sqrt(
            float(self.image_radius_sq.max())
        )

    def _grid_coordinates(self) -> Tensor:
        """``[2, H*W]`` bin centres in ``[-pi, pi)``, matching the trajectory."""
        height, width = self.im_size
        ky = (torch.arange(height, dtype=torch.float32) / height - 0.5) * 2.0 * math.pi
        kx = (torch.arange(width, dtype=torch.float32) / width - 0.5) * 2.0 * math.pi
        grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
        return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)

    def _image_radius_sq(self) -> Tensor:
        """``[1, 1, H, W]`` squared image-space radius in samples, centred."""
        height, width = self.im_size
        iy = torch.arange(height, dtype=torch.float32) - height // 2
        ix = torch.arange(width, dtype=torch.float32) - width // 2
        yy, xx = torch.meshgrid(iy, ix, indexing="ij")
        return (yy.pow(2) + xx.pow(2))[None, None]

    def _heads(self, features: Tensor) -> Tensor:
        """Split the already-positive features across heads.

        No activation is applied. The features are positive by construction and
        their inner product IS the interpolation kernel, so any elementwise map
        here would break the kernel the block exists to realise.
        """
        batch, width, count = features.shape
        return features.reshape(batch, self.num_heads, width // self.num_heads, count)

    def forward(self, kdata: Tensor, trajectory: Tensor, sample_mask: Tensor) -> Tensor:
        """Grid ``kdata`` by attention.

        Args:
            kdata: Complex samples ``[B, C, N]``.
            trajectory: Coordinates ``[B, 2, N]`` or ``[2, N]`` in ``[-pi, pi]``.
            sample_mask: ``[B, N]`` or ``[N]``, 1 where acquired.

        Returns:
            Interleaved real ``[B, 2C, H, W]`` on the Cartesian grid.
        """
        if not torch.is_complex(kdata):
            raise ValueError(
                "kdata must be complex: the phase is the object's spatial-shift "
                "information and a real tensor has already discarded it."
            )
        batch, channels, count = kdata.shape
        if channels != self.complex_channels:
            raise ValueError(
                f"kdata carries {channels} complex channels, block built for "
                f"{self.complex_channels}."
            )
        traj = trajectory if trajectory.dim() == 3 else trajectory.unsqueeze(0)
        if traj.shape[0] not in (1, batch):
            raise ValueError(
                f"trajectory batch {traj.shape[0]} matches neither 1 nor the kdata "
                f"batch {batch}; einsum would fail naming neither tensor."
            )
        traj = traj.expand(batch, -1, -1).to(kdata.real.dtype)
        mask = sample_mask if sample_mask.dim() == 2 else sample_mask.unsqueeze(0)
        mask = mask.expand(batch, -1).to(kdata.real.dtype)

        keys = self._heads(self.features(traj))
        # Rebuilt per forward. The frequencies are a parameter, so a cache would
        # need invalidating every step; at O(H*W*F) against the keys' O(N*F)
        # with N about 3x H*W on a radial arm, this is the smaller of the two.
        grid = self.grid_coords.to(traj).unsqueeze(0).expand(batch, -1, -1)
        queries = self._heads(self.features(grid))

        # Complex 1x1 over the coil axis, on a [B, 2C, N, 1] view so the
        # ComplexConv2d the rest of the cohort uses applies unchanged.
        projected = self.value_proj(
            torch.view_as_real(kdata).movedim(-1, 2).flatten(1, 2).unsqueeze(-1)
        )
        values = torch.complex(projected[:, 0::2, :, 0], projected[:, 1::2, :, 0])
        values = values.reshape(batch, 1, channels, count).expand(batch, self.num_heads, -1, -1)

        # The sample axis is flat, which is exactly the shape the shared seam
        # takes; `m_t` enters the key sum rather than an [N, N] score matrix.
        gridded = sample_density_attention(queries, keys, values, mask, eps=self.eps)
        gridded = gridded.mean(dim=1).reshape(batch, channels, *self.im_size)

        # Convolving k-space by the kernel multiplies the image by the kernel's
        # transform, so the image must be divided by it again. Computed from the
        # live width because that width is a learned parameter; the floor stops
        # a widened kernel from dividing by a decayed edge value.
        radius_sq = self.image_radius_sq.to(kdata.real.dtype).to(kdata.device)
        sigma_k = self.features.log_width.exp().clamp_max(self.max_sigma)
        apodization = torch.exp(-sigma_k.pow(2) * radius_sq / 2.0)
        gridded = fft2c(ifft2c(gridded) / apodization)
        return torch.view_as_real(gridded).movedim(-1, 2).flatten(1, 2)
