"""Attribute cold-diffusion error to the reverse step that wrote each coefficient.

The ``replace_freeze`` / ``replace_freeze_dc`` reverse loops write every
unobserved coefficient EXACTLY ONCE and then freeze it, so the final k-space
partitions cleanly by the step that produced it. That makes "which step got it
wrong?" answerable, which no image-domain metric can be: a single PSNR over the
assembled reconstruction averages a band the model inferred confidently at a
trained timestep together with one it extrapolated.

This module is the reverse-side counterpart of
``KSpaceUndersamplingProcess.removed_line_energy_stats``, which measures the same
partition on the forward side (per-level REMOVED energy of the clean target).
Same levels, prediction error instead of target energy.

**Oracle only.** :func:`attribute_reveal_bands` reads the target, so it is a
validation diagnostic and must never steer sampling — a loop that accepted or
rewrote a band on the strength of a target comparison would leak ground truth
into the reconstruction and turn every metric downstream into an accept/reject
curve fitted on the answer.

**Cartesian line masks only.** The reported band is a group of k-space lines, so
every function here refuses a mask that is not line-structured rather than
reporting a number that reads like a band and is not one (CLAUDE.md #3). Point
patterns (Poisson-disk, 2-D variable density) and non-Cartesian trajectories
(radial, spiral, golden-angle) partition by reveal step perfectly well but have
no line to name; they need their own grouping and are not handled here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from spectramr.models.diffusion.kspace_process import paired_complex

__all__ = [
    "attribute_reveal_bands",
    "complex_gain",
    "line_indices",
    "resolve_line_axis",
    "reveal_partition",
]

#: Spatial dim each line axis indexes, on a ``[..., H, W]`` k-space tensor.
#: ``"y"`` names full rows (constant k_y, the phase-encode line an MR sequence
#: actually acquires); ``"x"`` names full columns.
_LINE_AXIS_DIM: dict[str, int] = {"y": -2, "x": -1}


def _squeeze_to_plane(mask: Tensor) -> Tensor:
    """Reduce a mask of any leading shape to one boolean ``[H, W]`` plane.

    Sampling masks arrive as ``[B, 1, H, W]``, ``[1, H, W]`` or ``[H, W]``, and
    the channel axis on a real-stacked arm repeats the same pattern per Re/Im
    channel. Reducing with ``any`` rather than indexing ``[0, 0]`` keeps the
    answer correct if a caller ever passes a per-channel mask that differs.
    """
    plane = mask > 0
    while plane.dim() > 2:
        plane = plane.any(dim=0)
    return plane


#: Acceleration families whose masks are not line-structured, so no "band of
#: lines" exists to attribute error to.
#:
#: The authority is still :func:`resolve_line_axis`, which decides from the
#: PLANE's geometry rather than from a name — a name cannot know that
#: ``density_nested`` is line-structured only once ``mask_direction`` is set.
#: This list exists so the audit layer can refuse such an arm at load instead of
#: at its first validation, hours in; it is deliberately the same vocabulary the
#: raise below reports, so the two cannot drift into naming different families.
NON_LINE_ACCELERATION_TYPES: frozenset[str] = frozenset(
    {
        "poisson_disk",
        "variable_density_2d_gaussian",
        "radial",
        "spiral",
        "golden_angle",
    }
)

#: Families that are line-structured only when ``undersampling.mask_direction``
#: names the axis; without it they fall back to a 2-D pattern.
DIRECTION_DEPENDENT_ACCELERATION_TYPES: frozenset[str] = frozenset({"density_nested"})


def resolve_line_axis(mask: Tensor) -> str:
    """Which axis this mask's kept bins form complete lines along.

    Returns ``"y"`` when every touched row is fully kept (k_y lines, the usual
    phase-encode case) or ``"x"`` for full columns.

    Raises:
        ValueError: the mask is not line-structured, i.e. some row and some
            column are partially sampled. That is a point pattern or a
            non-Cartesian trajectory, for which a "band of lines" does not
            exist — reported rather than approximated, because the nearest
            approximation (a radial annulus) mixes the sampled interior of a
            kept line with the unsampled bins of a dropped one.
    """
    plane = _squeeze_to_plane(mask)
    for axis, dim in _LINE_AXIS_DIM.items():
        reduce_dim = -1 if dim == -2 else -2
        touched = plane.any(dim=reduce_dim)
        complete = plane.all(dim=reduce_dim)
        if bool(touched.any()) and bool((touched == complete).all()):
            return axis
    kept = int(plane.sum())
    raise ValueError(
        f"mask with {kept} kept bins of {plane.numel()} is not line-structured "
        "along either axis, so it has no band of lines to attribute error to. "
        "This is a point pattern (poisson_disk / variable_density_2d_gaussian) "
        "or a non-Cartesian trajectory (radial / spiral / golden_angle); those "
        "partition by reveal step but need their own band definition. Note the "
        "geometry follows `undersampling.mask_direction` as well as the family "
        "— `density_nested` is line-structured only when a direction is set."
    )


def line_indices(mask: Tensor, axis: str) -> list[int]:
    """Indices of the kept lines, along ``axis``, in ascending order.

    Raises:
        ValueError: ``axis`` is not ``"y"`` or ``"x"``.
    """
    if axis not in _LINE_AXIS_DIM:
        raise ValueError(f"axis must be one of {sorted(_LINE_AXIS_DIM)}, got {axis!r}.")
    plane = _squeeze_to_plane(mask)
    reduce_dim = -1 if _LINE_AXIS_DIM[axis] == -2 else -2
    return torch.nonzero(plane.any(dim=reduce_dim), as_tuple=False).flatten().tolist()


#: Reverse modes whose write-once semantics :func:`reveal_partition` mirrors.
#:
#: Both freeze loops write each revealed coefficient exactly once and then hold
#: it, which is the whole premise of attributing an error to the step that wrote
#: it. ``additive`` rewrites the entire plane every step, so no coefficient has
#: "the step that wrote it" -- an attribution computed under it would be a
#: plausible number describing a partition the loop never produced. It is also
#: the constructor default, so the exclusion has to be explicit rather than
#: assumed from the corpus.
PARTITIONED_REVERSE_MODES: frozenset[str] = frozenset(
    {"replace_freeze", "replace_freeze_dc", "replace_freeze_dc_t0"}
)


def reveal_partition(next_masks: Sequence[Tensor], observed: Tensor) -> list[Tensor]:
    """Which coefficients each reverse step writes, as disjoint boolean planes.

    Mirrors the reveal arithmetic of
    ``PhysicsInformedColdDiffusion._sample_replace_freeze_dc`` exactly: step
    ``i`` reveals its keying mask minus what is already committed and minus the
    observed support, and the terminal step takes every remaining unobserved
    coefficient. The caller supplies the masks, because which rung keys a step
    is the reverse mode's decision (``CURRENT_STEP_KEYED_REVERSE_MODES``). Derived from the schedule's masks rather than from a
    sampled trajectory, so it costs ``n`` mask evaluations and no model calls.

    An empty plane is meaningful, not a gap: it is a step that reveals nothing,
    which under ``dc_method='hard'`` is exactly the step
    ``_step_reveals_anything`` skips. Keeping it preserves the index alignment
    with the timestep schedule.

    Args:
        next_masks: ``mask(schedule[i+1])`` for ``i`` in ``0 .. n-2`` — one entry
            per non-terminal step, in schedule order.
        observed: the acquired support.

    Returns:
        ``n = len(next_masks) + 1`` disjoint boolean ``[H, W]`` planes whose
        union is the complement of ``observed``.
    """
    obs = _squeeze_to_plane(observed)
    committed = torch.zeros_like(obs)
    out: list[Tensor] = []
    for mask_next in next_masks:
        reveal = _squeeze_to_plane(mask_next) & ~committed & ~obs
        out.append(reveal)
        committed = committed | reveal
    out.append(~committed & ~obs)
    return out


def complex_gain(prediction: Tensor, target: Tensor, selector: Tensor) -> tuple[float, float]:
    """Least-squares complex gain of ``prediction`` against ``target``.

    ``g = <pred, target> / <target, target>`` over the selected coefficients.
    Its MODULUS is the amplitude shrinkage a conditional-mean estimator produces
    on ambiguous lines; its ARGUMENT is the systematic phase error, which on a
    band of k_y lines is a displaced copy of the object along y by the Fourier
    shift theorem. Reporting energy alone cannot tell those apart, and the
    reverse loop's magnitude ceiling is phase-invariant by construction
    (``clamp_to_magnitude_ceiling``), so phase is the unguarded direction.

    Args:
        prediction: reconstructed k-space, complex or interleaved real-stacked.
        target: ground-truth k-space, same layout and shape.
        selector: boolean ``[H, W]`` plane naming the coefficients to score.

    Returns:
        ``(modulus, phase_radians)``. ``(nan, nan)`` when the selector is empty
        or the selected target energy is zero — an unmeasurable band is reported
        as unmeasured, never as a gain of 1.

    Raises:
        ValueError: shapes disagree.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} and target {tuple(target.shape)} "
            "must have the same shape to share a selector."
        )
    pred_c = paired_complex(prediction)
    targ_c = paired_complex(target)
    sel = _squeeze_to_plane(selector)
    if not bool(sel.any()):
        return float("nan"), float("nan")
    pred_sel = pred_c[..., sel]
    targ_sel = targ_c[..., sel]
    denom = (targ_sel.conj() * targ_sel).sum().real
    if not bool(denom > 0):
        return float("nan"), float("nan")
    gain = (pred_sel * targ_sel.conj()).sum() / denom
    return float(gain.abs()), float(torch.angle(gain))


def attribute_reveal_bands(
    prediction: Tensor,
    target: Tensor,
    next_masks: Sequence[Tensor],
    observed: Tensor,
    timesteps: Sequence[int],
) -> list[dict[str, Any]]:
    """One record per reverse step: the band it wrote and how wrong that band is.

    Oracle diagnostic — reads ``target``. See the module docstring.

    Args:
        prediction: the assembled reconstruction in k-space.
        target: ground truth in k-space, same layout and shape.
        next_masks: as :func:`reveal_partition`.
        observed: the acquired support.
        timesteps: the reverse schedule, in order; one entry per step, so
            ``len(timesteps) == len(next_masks) + 1``.

    Returns:
        One dict per step with ``step`` (index), ``timestep``, ``n_bins``,
        ``line_axis``, ``lines`` (the kept line indices), ``gain_modulus`` and
        ``gain_phase_rad``. A step that revealed nothing carries ``n_bins: 0``
        and ``nan`` gains — that is an inert step the loop skipped, reported
        rather than dropped so the record aligns with the schedule.

        The observed support is scored too, as a final record with
        ``step: -1``: under ``dc_method='hard'`` it is pinned to the measurement
        and its gain is the irreducible floor every other band is read against.

    Raises:
        ValueError: ``timesteps`` does not match the step count, or a mask is
            not line-structured (:func:`resolve_line_axis`).
    """
    if len(timesteps) != len(next_masks) + 1:
        raise ValueError(
            f"timesteps has {len(timesteps)} entries but next_masks implies "
            f"{len(next_masks) + 1} steps. The schedule and the reveal masks "
            "must be built from the same reverse trajectory or the attribution "
            "names the wrong step."
        )
    if prediction.dim() != 4:
        # A 5D ``[B, C, H, W, D]`` batch puts D where the interleaving reader
        # expects W, so ``paired_complex`` would pair the wrong axis and return
        # a plausible complex tensor of the wrong coefficients.
        raise ValueError(
            f"attribute_reveal_bands expects [B, C, H, W], got "
            f"{tuple(prediction.shape)}. Flatten the depth axis first."
        )
    axis = resolve_line_axis(observed)
    bands = reveal_partition(next_masks, observed)
    records: list[dict[str, Any]] = []
    for step, (reveal, t_idx) in enumerate(zip(bands, timesteps, strict=True)):
        modulus, phase = complex_gain(prediction, target, reveal)
        records.append(
            {
                "step": step,
                "timestep": int(t_idx),
                "n_bins": int(reveal.sum()),
                "line_axis": axis,
                "lines": line_indices(reveal, axis) if bool(reveal.any()) else [],
                "gain_modulus": modulus,
                "gain_phase_rad": phase,
            }
        )
    obs_modulus, obs_phase = complex_gain(prediction, target, observed)
    obs_plane = _squeeze_to_plane(observed)
    records.append(
        {
            "step": -1,
            "timestep": -1,
            "n_bins": int(obs_plane.sum()),
            "line_axis": axis,
            "lines": line_indices(observed, axis),
            "gain_modulus": obs_modulus,
            "gain_phase_rad": obs_phase,
        }
    )
    return records
