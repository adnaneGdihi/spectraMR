r"""Two Cartesian masks per repetition pair: one for the input, one for the loss.

CoNEX (complementary-mask multi-NEX) spends a fixed scan-time budget on R
repetitions, each undersampled by R, instead of one fully sampled one. Training
feeds the network :math:`M_1 y_i` and scores it on the lines :math:`M_2` of an
independent repetition,

.. math::

    \mathcal L = \lVert M_2 \odot (F f(M_1 y_i) - y_j) \rVert^2 .

Because :math:`y_j`'s noise is independent of :math:`(x, y_i)`, the expectation
equals the supervised loss on :math:`M_2` plus a constant, so no SSDU split and no
Noisier2Noise correction is needed. Retrospectively both M4Raw repetitions are
fully sampled, so masking the loss residual with :math:`M_2` simulates the
prospective protocol exactly: this transform masks ``input`` and leaves ``target``
whole, and the loss (``masked_kspace_l2``) applies :math:`M_2`.

**The four modes and what each can identify.** A line enters the objective only
through :math:`M_2`, and is denoised only if it is also in :math:`M_1`:

* ``independent`` -- :math:`M_1, M_2` are independent draws from one mask
  distribution, so every line has :math:`P(k\in M_2)>0` (identified) and
  :math:`P(k\in M_1\cap M_2)>0` (acquired lines are denoised too). The protocol.
* ``complementary`` -- an acquired non-ACS line is never in :math:`M_2`, so the
  output on acquired lines is unsupervised: interpolation and ACS denoising are
  learned, acquired-line denoising is not. The control that isolates that gap.
* ``same`` -- :math:`M_2=M_1`, so unacquired lines never enter the loss and the
  null space is unlearnable. The negative control.
* ``full_target`` -- :math:`M_2` is every line, so the loss is plain N2N
  supervision on an undersampled input. The bound.

**Rebalancing.** For ``complementary`` at ``acceleration > 2`` the complement of
:math:`M_1` holds more lines than :math:`M_1` does, which would give that arm a
larger supervision budget than the others. Above R=2 the complement is subsampled
uniformly to :math:`|M_1\setminus\mathrm{ACS}|` lines, so :math:`|M_2|=|M_1|`.

**Support.** Masks are drawn inside the columns where ``input`` carries energy:
M4Raw zero-fills about 24% of the phase-encode axis (partial FOV), and a line
there is not an acquired line. Those structural zeros are added to ``loss_mask``
in every mode, so the network is still supervised to reproduce the zero fill --
exactly what arm B's unmasked ``l2`` teaches -- instead of emitting unsupervised
energy that the magnitude metrics would then read.

**Determinism.** Masks are seeded from ``file_id``, the slice, and ``seed``
through CRC32 (Python's ``hash`` is salted per process). A scan therefore keeps
its masks across epochs, which is the prospective protocol: an acquisition is
made once. :math:`M_1` depends only on those keys, so the four modes see
identical inputs and differ only in :math:`M_2`.

**Patterns.** ``pattern: cartesian_1d`` (the default) is the draw above, line
for line. The 2-D families (``poisson_disk``, ``radial``/``golden_angle``,
``spiral``, resolved through the sampling-pattern registry) are drawn by
:mod:`~spectramr.data.transforms.rep_mask_patterns`, which states what R and the
calibration region mean for them. A calibration region that fills the budget
raises in every pattern, and the achieved R is stamped under
``realised_acceleration`` for the snapshot provenance.

Registry transforms run on both chains. That is intended here: the validation
input must be undersampled like the training input, while the validation target
(the leave-one-out mean) stays whole because only ``input`` is masked.
"""

from __future__ import annotations

import torch
import torchio as tio

from spectramr.data.transforms.normalization import acs_band_rows
from spectramr.data.transforms.registry import register_transform
from spectramr.data.transforms.rep_mask_patterns import (
    CARTESIAN_1D,
    draw_rep_masks_2d,
    resolve_pattern,
)
from spectramr.data.transforms.rep_mask_patterns import seeded_generator as _generator
from spectramr.data.transforms.tio_physics import MEASURED_KSPACE_KEY, REALISED_ACCELERATION_KEY

#: The acquired phase-encode support, as a mask in ``mask``'s layout. Its
#: complement is the structural zero fill, known to be zero without any reference.
SUPPORT_MASK_KEY = "support_mask"

__all__ = ["VALID_MODES", "ComplementaryRepMasking", "draw_rep_masks"]

VALID_MODES: frozenset[str] = frozenset({"independent", "complementary", "same", "full_target"})


def _acs(width: int, center_fraction: float) -> torch.Tensor:
    """Centred calibration lines, on ``fft2c``'s centred k-space convention."""
    acs = torch.zeros(width, dtype=torch.bool)
    acs[acs_band_rows(width, center_fraction, min_lines=0)] = True
    return acs


def _draw(
    support: torch.Tensor, acs: torch.Tensor, acceleration: float, gen: torch.Generator
) -> torch.Tensor:
    """ACS plus a uniform draw of the remaining lines, ``|mask| = |support| / R``.

    Raises:
        ValueError: when the ACS alone fills the budget; the mask would be the
            ACS at a lower R than declared.
    """
    mask = acs.clone()
    candidates = torch.nonzero(support & ~acs).flatten()
    extra = round(int(support.sum()) / acceleration) - int(acs.sum())
    if extra <= 0:
        raise ValueError(
            f"complementary_rep_masking: the {int(acs.sum())} ACS lines fill the "
            f"{int(acs.sum()) + extra}-line budget of R={acceleration:g} on "
            f"{int(support.sum())} supported lines, so the mask would be the ACS alone at "
            f"R={int(support.sum()) / max(1, int(acs.sum())):.3g}. Lower center_fraction."
        )
    chosen = candidates[torch.randperm(candidates.numel(), generator=gen)[:extra]]
    mask[chosen] = True
    return mask


def _m2_independent(support, acs, m1, acceleration, keys) -> torch.Tensor:
    return _draw(support, acs, acceleration, _generator(*keys, "m2"))


def _m2_same(support, acs, m1, acceleration, keys) -> torch.Tensor:
    return m1.clone()


def _m2_full_target(support, acs, m1, acceleration, keys) -> torch.Tensor:
    return support.clone()


def _m2_complementary(support, acs, m1, acceleration, keys) -> torch.Tensor:
    rest = torch.nonzero(support & ~m1).flatten()
    if acceleration > 2:
        budget = int((m1 & ~acs).sum())
        rest = rest[torch.randperm(rest.numel(), generator=_generator(*keys, "m2"))[:budget]]
    m2 = acs.clone()
    m2[rest] = True
    return m2


#: How the second repetition's mask is drawn, by ``mode``. Same set as ``VALID_MODES``.
_M2_BY_MODE = {
    "independent": _m2_independent,
    "same": _m2_same,
    "full_target": _m2_full_target,
    "complementary": _m2_complementary,
}


def draw_rep_masks(
    support: torch.Tensor,
    *,
    acceleration: float,
    center_fraction: float,
    mode: str,
    keys: tuple[object, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(M1, M2)`` over one slice's phase-encode columns.

    Args:
        support: ``(W,)`` bool, the columns the acquisition covers.
        acceleration: ``R >= 1``, as a fraction of the support.
        center_fraction: ACS width as a fraction of ``W``; always in both masks.
        mode: one of :data:`VALID_MODES`.
        keys: seed material; ``M1`` is a function of these alone.

    Returns:
        ``M1`` within the support, and ``M2`` with the structural zeros
        (``~support``) added, both ``(W,)`` bool.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Valid: {sorted(VALID_MODES)}.")
    acs = _acs(support.numel(), center_fraction) & support
    m1 = _draw(support, acs, acceleration, _generator(*keys, "m1"))
    m2 = _M2_BY_MODE[mode](support, acs, m1, acceleration, keys)
    return m1, m2 | ~support


@register_transform(
    "complementary_rep_masking",
    produces=("mask", "loss_mask", MEASURED_KSPACE_KEY),
    requires=("input", "file_id"),
)
class ComplementaryRepMasking(tio.transforms.Transform):
    """Undersample ``input`` by ``M1`` and emit ``mask`` = M1, ``loss_mask`` = M2.

    The undersampled input is also written under ``measured_kspace``, replacing the
    unmasked input the M4Raw dataset publishes there, so the strategy's ``y`` is the
    masked measurement and never the reference under ``kspace`` (#2324, #2761). The CoNEX arms'
    ``complex_unet`` ignores the ``kspace_measured`` it is handed, so their numbers
    do not move; the Rep-EI arm reads it as ``y``.

    Args:
        acceleration: ``R >= 1``; the fraction ``1/R`` of the supported columns
            each mask acquires, ACS included.
        center_fraction: ACS width as a fraction of the phase-encode axis.
        mode: ``independent`` | ``complementary`` | ``same`` | ``full_target``.
        seed: salt mixed into every mask seed, for a second mask realisation of
            the same corpus.
        pattern: ``cartesian_1d`` or a 2-D family the sampling-pattern registry
            resolves (see :mod:`~spectramr.data.transforms.rep_mask_patterns`).

    Shape:
        ``input``: ``(2C, H, W, S)`` real/imag-interleaved or ``(C, H, W, S)``
        complex; the phase-encode axis is ``W``. ``mask``/``loss_mask`` are
        ``(1, H, W, S)`` float, which collation turns into ``(B, 1, H, W)`` --
        broadcastable over the ``(B, 2C, H, W)`` batch.
    """

    def __init__(
        self,
        acceleration: float = 2.0,
        center_fraction: float = 0.08,
        mode: str = "independent",
        seed: int = 0,
        pattern: str = CARTESIAN_1D,
    ) -> None:
        super().__init__()
        self.pattern = resolve_pattern(pattern)
        if mode not in VALID_MODES:
            raise ValueError(
                f"complementary_rep_masking: unknown mode {mode!r}. Valid: {sorted(VALID_MODES)}."
            )
        if not float(acceleration) >= 1.0:
            raise ValueError(f"acceleration must be >= 1, got {acceleration!r}.")
        if not 0.0 <= float(center_fraction) < 1.0:
            raise ValueError(f"center_fraction must be in [0, 1), got {center_fraction!r}.")
        self.acceleration = float(acceleration)
        self.center_fraction = float(center_fraction)
        self.mode = mode
        self.seed = int(seed)

    def _slice_ids(self, subject: tio.Subject, depth: int) -> list[int]:
        """A slice-level record carries its volume index; a volume uses its own."""
        index = subject.get("slice_index")
        if depth == 1 and index is not None:
            return [int(index)]
        return list(range(depth))

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        file_id = subject.get("file_id")
        if file_id is None:
            raise ValueError(
                "complementary_rep_masking seeds its masks from `file_id` so that "
                "validation is reproducible, and this subject carries none "
                f"(keys: {sorted(subject.keys())})."
            )
        image = subject["input"]
        data = image.data
        if data.ndim != 4:
            raise ValueError(f"expected input (C, H, W, S), got {tuple(data.shape)}.")
        _, height, width, depth = data.shape
        support = (data.abs() > 0).any(dim=1).any(dim=0)  # (W, S)

        m1 = torch.zeros(height, width, depth, dtype=torch.bool)
        m2 = torch.zeros(height, width, depth, dtype=torch.bool)
        achieved = []
        for s, slice_id in enumerate(self._slice_ids(subject, depth)):
            kwargs = {
                "acceleration": self.acceleration,
                "center_fraction": self.center_fraction,
                "mode": self.mode,
                "keys": (file_id, slice_id, self.seed),
            }
            if self.pattern == CARTESIAN_1D:
                line1, line2 = draw_rep_masks(support[:, s], **kwargs)
                m1[:, :, s], m2[:, :, s] = line1[None, :], line2[None, :]
                achieved.append(int(support[:, s].sum()) / int(line1.sum()))
            else:
                m1[:, :, s], m2[:, :, s], r = draw_rep_masks_2d(
                    self.pattern, support[:, s], height, **kwargs
                )
                achieved.append(r)
        subject[REALISED_ACCELERATION_KEY] = torch.tensor(
            sum(achieved) / len(achieved), dtype=torch.float32
        )

        mask = m1.to(torch.float32)[None]
        loss_mask = m2.to(torch.float32)[None]
        image.set_data(data * mask.to(data.dtype))
        # The second repetition the loader serves beside the input (rep_variance,
        # n2n PAPER.md 5) is the same acquisition under the same M1, or pred_alt
        # would be the estimate of an unmasked input.
        if "input_alt" in subject:
            alt = subject["input_alt"]
            alt.set_data(alt.data * mask.to(alt.data.dtype))
        affine = image.affine
        subject.add_image(tio.ScalarImage(tensor=mask.contiguous(), affine=affine), "mask")
        subject.add_image(
            tio.ScalarImage(tensor=image.data.clone(), affine=affine), MEASURED_KSPACE_KEY
        )
        subject.add_image(
            tio.ScalarImage(tensor=loss_mask.contiguous(), affine=affine), "loss_mask"
        )
        support_mask = support[None, None].expand(1, height, width, depth).to(torch.float32)
        subject.add_image(
            tio.ScalarImage(tensor=support_mask.contiguous(), affine=affine), SUPPORT_MASK_KEY
        )
        return subject
