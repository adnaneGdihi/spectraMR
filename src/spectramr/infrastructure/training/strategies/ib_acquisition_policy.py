"""Line masks, the learned line prior and the policy rollouts for IB active acquisition.

The strategy in :mod:`ib_active_acquisition_strategy` trains its reconstructor on
random phase-encode masks and, at every validation pass, rolls out four policies at
the same line budget: the information-bottleneck policy (``select_next_line``), a
greedy posterior-variance policy, equispaced lines and random lines. Every policy
keeps the same fully sampled centre band, so they differ only in the outer lines.

The linear-Gaussian prior is learned from the fully sampled training k-space: the
covariance across phase-encode lines (averaged over readout positions, coils and
samples) and a per-line noise variance read from the outer readout band. That
estimate assumes the band is noise-dominated (an image-domain crop keeps the k-space
extent, so the band stays the outermost readout frequencies); a signal tail there makes
it follow each line's power, which ``noise_signal_correlation`` reports. Both accumulate
on the device with one GEMM and one reduction per step, without a host sync.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

#: Fraction of the readout axis, at each end, that the per-line noise variance is read from.
NOISE_EDGE_FRACTION = 0.05
#: Ridge added to the prior covariance, relative to its mean diagonal, so the precision exists.
PRIOR_RIDGE = 1e-6


def line_budget(width: int, acceleration: float, center_fraction: float) -> tuple[int, int]:
    """``(total lines, centre lines)`` for a ``width``-line phase-encode axis."""
    total = max(1, round(width / acceleration))
    centre = max(1, round(width * center_fraction))
    if centre > total:
        raise ValueError(
            f"IB acquisition budget: the centre band ({centre} of {width} lines at "
            f"center_fraction={center_fraction}) exceeds the total budget ({total} lines at "
            f"acceleration={acceleration}). Lower center_fraction or acceleration."
        )
    return total, centre


def centre_mask(width: int, centre: int) -> torch.Tensor:
    """Boolean ``[width]`` mask of the ``centre`` lines around ``width // 2``."""
    mask = torch.zeros(width, dtype=torch.bool)
    lo = width // 2 - centre // 2
    mask[lo : lo + centre] = True
    return mask


def random_line_masks(
    batch: int,
    width: int,
    total: int,
    centre: int,
    generator: torch.Generator,
    support: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[batch, width]`` masks: the centre band plus ``total - centre`` uniform outer lines.

    ``support`` (``[batch, width]`` bool) restricts the outer draws to acquired lines;
    a sample with fewer acquired outer lines than the budget takes all of them first.
    The masks are built on ``support``'s device: only the CPU draws travel host to device.
    """
    device = torch.device("cpu") if support is None else support.device
    seed = centre_mask(width, centre).to(device)
    draws = torch.rand(batch, width, generator=generator).to(device, non_blocking=True)
    allowed = ~seed.unsqueeze(0) if support is None else support & ~seed
    draws = draws.masked_fill(~allowed, 2.0).masked_fill(seed, 3.0)
    picks = draws.argsort(dim=1)[:, : total - centre]
    masks = seed.unsqueeze(0).repeat(batch, 1)
    masks.scatter_(1, picks, True)
    return masks


def equispaced_line_mask(
    width: int, total: int, centre: int, support: torch.Tensor | None = None
) -> torch.Tensor:
    """``[width]`` mask: the centre band plus ``total - centre`` evenly spaced outer lines.

    ``support`` (``[width]`` bool) restricts the outer lines to acquired ones.
    """
    mask = centre_mask(width, centre)
    allowed = ~mask if support is None else support.cpu() & ~mask
    outer = allowed.nonzero(as_tuple=False).squeeze(-1)
    n = min(total - centre, outer.numel())
    if n > 0:
        idx = torch.linspace(0, outer.numel() - 1, n).round().long()
        mask[outer[idx]] = True
    return mask


def apply_line_mask(kspace: torch.Tensor, mask: torch.Tensor, pe_dim: int) -> torch.Tensor:
    """Zero the unacquired phase-encode lines of ``kspace`` (``mask`` is ``[W]`` or ``[B, W]``)."""
    shape = [1] * kspace.ndim
    shape[pe_dim] = mask.shape[-1]
    if mask.ndim == 2:
        shape[0] = mask.shape[0]
    return kspace * mask.to(device=kspace.device, dtype=kspace.real.dtype).view(shape)


def _real_imag(kspace: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_complex(kspace):
        return kspace.real, kspace.imag
    if kspace.shape[1] % 2:
        raise ValueError(
            "IB line prior: real k-space must be real/imag interleaved (an even channel "
            f"count); got {kspace.shape[1]} channels."
        )
    return kspace[:, 0::2], kspace[:, 1::2]


class LinePrior:
    """Running linear-Gaussian prior over phase-encode lines, learned from training k-space."""

    def __init__(self) -> None:
        self.cov_sum: torch.Tensor | None = None
        self.rows = 0
        self.noise_sum: torch.Tensor | None = None
        self.noise_rows = 0

    @property
    def seen(self) -> bool:
        return self.rows > 0

    @torch.no_grad()
    def update(self, kspace: torch.Tensor, pe_dim: int, readout_dim: int) -> None:
        """Accumulate ``Re E[y y^H]`` across lines and the outer-readout power per line."""
        real, imag = _real_imag(kspace.detach())
        real = real.float().movedim((readout_dim, pe_dim), (-2, -1))
        imag = imag.float().movedim((readout_dim, pe_dim), (-2, -1))
        width, n_ro = real.shape[-1], real.shape[-2]
        r2, i2 = real.reshape(-1, width), imag.reshape(-1, width)
        cov = (r2.T @ r2 + i2.T @ i2).double()
        edge = max(1, round(n_ro * NOISE_EDGE_FRACTION))
        band = torch.cat([real[..., :edge, :], real[..., -edge:, :]], dim=-2)
        band_i = torch.cat([imag[..., :edge, :], imag[..., -edge:, :]], dim=-2)
        power = (band.square() + band_i.square()).reshape(-1, width).sum(0).double()
        if self.cov_sum is None or self.cov_sum.shape[-1] != width:
            self.cov_sum, self.noise_sum, self.rows, self.noise_rows = cov, power, 0, 0
        else:
            self.cov_sum += cov
            self.noise_sum += power
        self.rows += r2.shape[0]
        self.noise_rows += (r2.shape[0] // n_ro) * 2 * edge

    def covariance_and_noise(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(Sigma [W, W], sigma_N^2 [W])`` in float64 on the CPU, ridge-regularised."""
        if not self.seen or self.cov_sum is None or self.noise_sum is None:
            raise RuntimeError("IB line prior: no training k-space has been accumulated.")
        cov = (self.cov_sum / self.rows).cpu()
        cov = cov + PRIOR_RIDGE * cov.diagonal().mean() * torch.eye(cov.shape[-1], dtype=cov.dtype)
        noise = (self.noise_sum / max(self.noise_rows, 1)).cpu()
        floor = torch.finfo(noise.dtype).tiny + PRIOR_RIDGE * cov.diagonal().mean()
        return cov, noise.clamp_min(floor)

    def state_dict(self) -> dict[str, object]:
        return {
            "cov_sum": None if self.cov_sum is None else self.cov_sum.cpu(),
            "rows": self.rows,
            "noise_sum": None if self.noise_sum is None else self.noise_sum.cpu(),
            "noise_rows": self.noise_rows,
        }

    def load_state_dict(self, state: dict[str, object], device: torch.device) -> None:
        cov, noise = state.get("cov_sum"), state.get("noise_sum")
        self.cov_sum = None if cov is None else cov.to(device)  # type: ignore[union-attr]
        self.noise_sum = None if noise is None else noise.to(device)  # type: ignore[union-attr]
        self.rows = int(state.get("rows", 0))  # type: ignore[arg-type]
        self.noise_rows = int(state.get("noise_rows", 0))  # type: ignore[arg-type]


def noise_signal_correlation(cov: torch.Tensor, noise: torch.Tensor) -> float | None:
    """Pearson correlation of ``sigma_N,l^2`` with the prior power ``Sigma_ll`` over acquired lines.

    Near zero when the outer readout band is noise; near one when it carries each line's
    signal. ``None`` when it is undefined (under three acquired lines, or a constant estimate).
    """
    power = cov.diagonal()
    keep = power > PRIOR_RIDGE * 10 * power.mean()
    pair = torch.stack([noise[keep], power[keep]])
    if pair.shape[-1] < 3 or bool((pair.std(dim=-1) == 0).any()):
        return None
    return float(torch.corrcoef(pair)[0, 1])


SelectFn = Callable[..., int]


def ib_rollout(
    cov: torch.Tensor,
    noise: torch.Tensor,
    seed: torch.Tensor,
    total: int,
    beta: float,
    select_fn: SelectFn,
) -> torch.Tensor:
    """Grow ``seed`` to ``total`` lines by repeated ``select_fn`` calls on the posterior.

    The posterior precision starts at ``Sigma^-1`` plus the seed lines' ``e_l e_l^T / sigma_N^2,l``
    and takes the same rank-one update after every pick.
    """
    width = cov.shape[-1]
    eye = torch.eye(width, dtype=cov.dtype)
    precision = torch.linalg.inv(cov) + torch.diag(seed.to(cov.dtype) / noise)
    mask = seed.clone()
    while int(mask.sum()) < total:
        line = select_fn(
            mask,
            cov,
            posterior_cov_inv=precision,
            noise_var=noise,
            beta=beta,
            mode="linear_gaussian",
        )
        mask[line] = True
        precision = precision + eye[line : line + 1].T @ eye[line : line + 1] / noise[line]
    return mask


def variance_rollout(
    cov: torch.Tensor, noise: torch.Tensor, seed: torch.Tensor, total: int
) -> torch.Tensor:
    """Greedy posterior-variance baseline: acquire the unobserved line of largest variance."""
    precision = torch.linalg.inv(cov) + torch.diag(seed.to(cov.dtype) / noise)
    mask = seed.clone()
    while int(mask.sum()) < total:
        var = torch.linalg.inv(precision).diagonal().masked_fill(mask, -torch.inf)
        line = int(var.argmax())
        mask[line] = True
        precision[line, line] += 1.0 / noise[line]
    return mask


__all__ = [
    "NOISE_EDGE_FRACTION",
    "PRIOR_RIDGE",
    "LinePrior",
    "apply_line_mask",
    "centre_mask",
    "equispaced_line_mask",
    "ib_rollout",
    "line_budget",
    "noise_signal_correlation",
    "random_line_masks",
    "variance_rollout",
]
