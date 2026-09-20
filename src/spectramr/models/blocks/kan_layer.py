"""KAN (Kolmogorov-Arnold Network) layer.

Minimal implementation following Liu et al., "KAN: Kolmogorov-Arnold
Networks" (2024 §2). Each edge is a learnable univariate function realized
as the sum of a SiLU residual and a B-spline expansion on a fixed grid.

`forward` is fp32-internally; B-spline ratio terms lose precision in fp16/bf16
and silently NaN otherwise. Callers may run under autocast — this module
disables it locally.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _gather_samples_across_ranks(x_flat: torch.Tensor) -> torch.Tensor:
    """Make every rank fit the grid to the same sample set.

    ``update_grid_from_samples`` re-fits knots AND solves least-squares for the
    matching coefficients. Run per-rank on a ``DistributedSampler`` shard, each
    rank solves a different (grid, coefficient) pair; DDP then broadcasts
    ``grid`` -- a persistent buffer -- from rank 0 while ``spline_weight`` is a
    parameter that is not re-broadcast, so the pair that was solved together is
    split apart and the gate function every rank evaluates is one nobody fitted.

    Gathering here rather than broadcasting the result keeps the fit
    deterministic and identical on every rank without depending on which rank
    happens to be authoritative. Ranks are truncated to the global minimum count
    so the all-gather is shape-safe.
    """
    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() < 2:
        return x_flat
    counts = torch.tensor([x_flat.shape[0]], device=x_flat.device)
    dist.all_reduce(counts, op=dist.ReduceOp.MIN)
    n = int(counts.item())
    if n == 0:
        return x_flat
    local = x_flat[:n].contiguous()
    buckets = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(buckets, local)
    return torch.cat(buckets, dim=0)


class KANLayer(nn.Module):
    """A single Kolmogorov-Arnold layer.

    Output edge ``(o, i)`` is ``base_w[o,i] * SiLU(x_i) +
    sum_n spline_w[o,i,n] * B_n(x_i)``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        scale_base: float = 0.1,
        scale_spline: float = 0.1,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"KANLayer requires positive dims; got {in_dim=}, {out_dim=}")
        if spline_order < 1:
            raise ValueError(f"spline_order must be >=1; got {spline_order}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / max(grid_size, 1)
        # Extend grid by spline_order knots on either side so spline evaluation
        # has well-defined support over the full grid_range.
        knots = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        # [in_dim, num_knots]
        self.register_buffer("grid", knots.unsqueeze(0).expand(in_dim, -1).contiguous())

        coef_count = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_dim, in_dim, coef_count) * scale_spline / math.sqrt(in_dim)
        )
        self.base_weight = nn.Parameter(
            torch.randn(out_dim, in_dim) * scale_base / math.sqrt(in_dim)
        )

        # Sample collection for grid extension. Disabled by default; set
        # `collect_samples=True` to enable. Buffered samples are used by
        # `update_grid_from_samples()` when called with no `x` argument.
        self.collect_samples: bool = False
        self._collected_samples: list[torch.Tensor] = []
        self._max_buffered_samples: int = 4096

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the B-spline basis at ``x``.

        Args:
            x: ``[..., in_dim]`` real tensor.

        Returns:
            ``[..., in_dim, grid_size + spline_order]`` basis values.
        """
        # Cox-de Boor recursion. We allow arbitrary leading dims; the grid
        # broadcasts over ``in_dim``.
        # Move to fp32 for numerical stability of the recursion ratios.
        x = x.unsqueeze(-1)  # [..., in_dim, 1]
        grid = self.grid  # [in_dim, num_knots]

        # Order-0 indicator: B_n^0(x) = 1 if grid[n] <= x < grid[n+1]
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, : -(k + 1)]
            left_den = grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-8
            right_num = grid[:, k + 1 :] - x
            right_den = grid[:, k + 1 :] - grid[:, 1:-k] + 1e-8
            bases = (left_num / left_den) * bases[..., :-1] + (right_num / right_den) * bases[
                ..., 1:
            ]
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass; runs internally in fp32 to keep B-splines stable."""
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"KANLayer expected last dim {self.in_dim}, got {x.shape[-1]}")
        orig_dtype = x.dtype

        # Optionally buffer a downsampled copy of the input for grid extension.
        # We collect at most `_max_buffered_samples` rows total to keep memory
        # bounded; FIFO eviction so the buffer reflects the recent distribution.
        # Kept detached and ON-DEVICE: a ``.cpu()`` here would be a blocking
        # device->host sync on every forward (this runs for the first 10K iters
        # across ~48 KAN layers x grad-accum x grad-checkpoint re-runs), which
        # serialises an otherwise GPU-bound step (.claude/rules/performance.md).
        # The buffer is capped at `_max_buffered_samples` rows (~8 MB on GPU),
        # and `update_grid_from_samples` already moves it to the parameter
        # device (now a no-op), so nothing downstream needs the host copy.
        if self.collect_samples:
            with torch.no_grad():
                flat = x.reshape(-1, self.in_dim).detach().float()
                self._collected_samples.append(flat)
                total = sum(t.shape[0] for t in self._collected_samples)
                while total > self._max_buffered_samples and len(self._collected_samples) > 1:
                    dropped = self._collected_samples.pop(0)
                    total -= dropped.shape[0]

        # Operate in fp32 regardless of AMP context (B-splines are numerically
        # sensitive to fp16 rounding). We cast manually instead of using
        # autocast(device_type="cuda", enabled=False) to stay device-agnostic
        # (CPU, MPS, and CUDA all work without modification).
        x32 = x.float()
        base = torch.nn.functional.silu(x32) @ self.base_weight.t()
        bases = self.b_splines(x32)  # [..., in_dim, coef_count]
        spline = torch.einsum("...ic,oic->...o", bases, self.spline_weight)
        out = base + spline
        return out.to(orig_dtype)

    @torch.no_grad()
    def update_grid_from_samples(
        self,
        x: torch.Tensor | None = None,
        margin: float = 0.01,
    ) -> None:
        """Re-fit the B-spline grid to the empirical input distribution.

        Implements Liu et al. 2024 §2.2 grid extension:

        1. Evaluate the *current* spline ``f_old(x)`` (pure spline part — the
           SiLU residual is unchanged because its grid is implicit).
        2. Build a new grid based on per-input-dim quantiles of ``x`` so the
           grid is denser where the data actually lives.
        3. Solve linear least-squares for new spline coefficients that fit
           ``f_old(x)`` on the empirical support.

        The function is effectively unchanged on the support of ``x`` but
        trains faster afterward because the spline knots are aligned with
        the post-shift input distribution. Plan §9 risk mitigation #1
        recommends calling this every 2,000 iterations during the first
        10K of training.

        Args:
            x: ``[N, in_dim]`` (or any leading dims) batch of recent inputs
                to use for the grid update. Larger N = more accurate fit.
                If ``None``, uses the internal sample buffer accumulated
                during forward passes (requires ``collect_samples=True``).
            margin: Fractional padding added to either side of the input
                range so the new grid covers a slightly wider range than
                the empirical support.
        """
        if x is None:
            if not self._collected_samples:
                return  # buffer empty — nothing to fit
            x = torch.cat(self._collected_samples, dim=0)
            # Move to the parameter's device so b_splines indexing matches
            x = x.to(self.spline_weight.device)
        # Flatten leading dims to a single sample axis.
        x_flat = x.reshape(-1, self.in_dim).float()
        x_flat = _gather_samples_across_ranks(x_flat)
        if x_flat.shape[0] < self.grid_size + self.spline_order + 1:
            # Not enough samples to fit a useful grid — refuse silently
            # rather than producing a degenerate spline. Caller can detect
            # this by inspecting grid_size or by passing a larger batch.
            return

        # 1) Evaluate the OLD spline contribution at the sample points.
        #    Shape: [N, out_dim]
        old_bases = self.b_splines(x_flat)  # [N, in_dim, coef_count]
        old_spline = torch.einsum("nic,oic->no", old_bases, self.spline_weight)

        # 2) Build a NEW grid per input dim from quantiles of x_flat.
        #    Use evenly-spaced quantiles so the inner grid points carry equal
        #    sample weight; then pad both ends by `spline_order` knots so the
        #    Cox-de Boor recursion has support over the full range.
        x_sorted, _ = x_flat.sort(dim=0)  # [N, in_dim]
        n_samples = x_sorted.shape[0]
        # Inner knots: grid_size + 1 evenly-spaced quantiles in [0, n_samples-1]
        idx = torch.linspace(0.0, n_samples - 1.0, self.grid_size + 1, device=x_flat.device).long()
        inner = x_sorted[idx, :]  # [grid_size+1, in_dim]
        x_min = inner[0:1, :]  # [1, in_dim]
        x_max = inner[-1:, :]
        span = (x_max - x_min).clamp_min(1e-6)
        # Add margin so new grid covers slightly more than empirical support.
        x_min = x_min - margin * span
        x_max = x_max + margin * span

        # Linear extension by spline_order knots on each side
        h = (x_max - x_min) / max(self.grid_size, 1)
        left_pad = torch.stack(
            [x_min[0] - (self.spline_order - i) * h[0] for i in range(self.spline_order)],
            dim=0,
        )
        right_pad = torch.stack(
            [x_max[0] + (i + 1) * h[0] for i in range(self.spline_order)],
            dim=0,
        )
        # Inner already includes endpoints, so don't double them
        new_grid = torch.cat([left_pad, inner, right_pad], dim=0).T.contiguous()
        # new_grid shape: [in_dim, grid_size + 2*spline_order + 1]
        if new_grid.shape != self.grid.shape:
            # Defensive: shapes must match for in-place buffer update
            return

        # Swap in the new grid before evaluating the new basis on x.
        old_grid = self.grid.clone()
        self.grid.copy_(new_grid)

        # 3) Solve LSQ per input dim independently:
        #    For each input dim i, find spline_weight[:, i, :] (out_dim, coef_count)
        #    such that  new_bases[:, i, :] @ spline_weight[:, i, :].T  = old_spline (target).
        #
        #    But spline output sums across input dims:
        #       y_o = sum_i sum_c W[o,i,c] * B_new[n,i,c]
        #
        #    This is a single big linear system. Build the design matrix
        #    A[N, in_dim*coef_count] = flattened B_new[n,i,c], solve for
        #    W_flat[in_dim*coef_count, out_dim] s.t. A @ W_flat = old_spline.
        new_bases = self.b_splines(x_flat)  # [N, in_dim, coef_count]
        N, D, C = new_bases.shape
        A = new_bases.reshape(N, D * C)
        # Use lstsq (driver='gelsd' is robust); torch.linalg.lstsq returns a
        # named tuple with .solution.
        try:
            sol = torch.linalg.lstsq(A, old_spline)
            W_flat = sol.solution  # [D*C, out_dim]
        except Exception:
            # If lstsq fails (rank-deficient on small batches), restore the
            # old grid and bail out.
            self.grid.copy_(old_grid)
            return

        new_W = W_flat.T.reshape(self.out_dim, D, C).contiguous()
        # Replace spline_weight in-place to preserve optimizer state.
        self.spline_weight.data.copy_(new_W)
        # Buffer was consumed; clear it so the next collection cycle is fresh.
        self._collected_samples.clear()

    def set_sample_collection(self, enabled: bool) -> None:
        """Toggle the input-sample buffer used by ``update_grid_from_samples``."""
        self.collect_samples = bool(enabled)
        if not enabled:
            self._collected_samples.clear()


class KANMLP(nn.Module):
    """Sequential KAN: a stack of KANLayers with no inter-layer activation.

    The univariate functions inside each KANLayer already provide non-linearity,
    so no extra activation is required between layers (Liu et al. 2024 §2.4).
    """

    def __init__(
        self,
        dims: list[int],
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("KANMLP requires at least input and output dims")
        layers = []
        for i in range(len(dims) - 1):
            layers.append(
                KANLayer(
                    dims[i],
                    dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
