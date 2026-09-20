"""Score the terminal (t=0) rung of a k-space cold-diffusion arm, pre-DC.

Why this exists
---------------
Under ``dc_method: hard`` the reverse process never evaluates the model at
t=0. At i=n-2 (t=1) the reveal source ``mask_next`` is all-ones, so that step
reveals every remaining coefficient; at i=n-1 the inertness guard finds nothing
left and ``continue``s. The skip is arithmetically correct -- with ``committed``
full the step is a bit-identical no-op -- but it means the terminal rung is
never measured by the sampler.

Adding R=1.0 to the validation cascade does not fix that: an R=1.0 rung gives
``t_head == t_floor == 0``, so ``n == 1`` and ``obs`` is all-ones, the single
step hits the same inertness guard, and the rung reports its own input verbatim
-- a perfect score that means nothing. The observable has to come from a direct
forward call, and it has to be read *before* data consistency, because after
hard DC at t=0 every bin is acquired and the output is the input by
construction.

Pre-DC is also the only place a gradient exists at that rung: hard DC replaces
the proposal at every acquired bin, so every post-DC loss is a constant w.r.t.
the weights and ``losses.reconstruction.lambda_pre_dc_kspace`` carries the whole
signal. This module makes that one learning signal visible.

Three constraints the callers must respect
------------------------------------------
**The emitted key set must be rank-invariant and unconditional.**
``pipelines/train.py:_all_reduce_val_metrics`` packs ``sorted(val_accum.keys())``
into a tensor and all-reduces it *positionally*. A key set that differs between
ranks -- gated on rank, on batch content, or on a try/except -- makes ranks
reduce mismatched-length tensors: a hang or a silent misalignment of one
metric's value onto another metric's name, never an error. The only gate here is
:func:`generator_exposes_pre_dc`, which reads a property of the model class and
is therefore identical on every rank.

**Sensitivity maps must be passed explicitly.** The generator's ``forward``
stashes S-maps and, when no kwarg is supplied, *falls back to whatever is
already stashed* (see its docstring). A probe that omits them inherits whatever
the last sampler step left behind -- correct today only by accident of call
order, and stale the moment the cascade is reordered or disabled.

**The forward is chunked, and every per-row kwarg chunks with it.** This batch
is the 5D->4D depth flatten -- 36 rows for an 18-slice volume at B=2 -- and at
t=0 the revealed mask is all-ones, so it is the densest input of the whole
sweep, evaluated after the cascade has already filled the card. Splitting ``x``
alone would not be enough either: ``complex_unet`` truncates a longer
``contrast_emb`` to match ``t_emb`` rather than raising (#2055), so a full-batch
``contrast_idx`` beside a one-row ``x`` conditions every chunk on row 0's
contrast and still returns a number.

Scoring is deliberately NOT done here. The strategy owns one metrics seam
(``_compute_validation_metrics``); this module hands it a tensor and renames
what comes back, so there is no second scoring implementation to drift
(non-negotiable 17).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

__all__ = [
    "T0_PREDC_PREFIX",
    "build_t0_timesteps",
    "forward_pre_dc",
    "generator_exposes_pre_dc",
    "rename_to_probe_namespace",
    "run_t0_predc_probe",
    "slice_row_kwargs",
    "t0_predc_key",
]

T0_PREDC_PREFIX = "val_t0_predc_"

_VAL_PREFIX = "val_"

# The generator reads `sensitivity_maps` first and falls back to `smaps`, and
# the strategy's `_build_generator_kwargs` supplies `smaps` -- only its
# prior-model branch uses the long spelling. Accepting one of the two would
# have made the probe raise on the production path.
_SMAPS_KEYS = ("sensitivity_maps", "smaps")


def t0_predc_key(key: str) -> str:
    """Move one metric name into the probe namespace.

    ``val_psnr`` -> ``val_t0_predc_psnr`` and bare ``psnr`` -> ``val_t0_predc_psnr``.
    Both shapes occur: the metrics seam emits ``val_``-prefixed names, but
    callers that build a dict by hand do not always. Inserting after the
    existing ``val_`` rather than prepending keeps the CSV's ``val_`` grouping
    intact; blindly prepending would mint ``val_t0_predc_val_psnr``.

    Already-namespaced keys are returned unchanged so that applying this twice
    is a no-op rather than a nested prefix.
    """
    if key.startswith(T0_PREDC_PREFIX):
        return key
    if key.startswith(_VAL_PREFIX):
        return T0_PREDC_PREFIX + key[len(_VAL_PREFIX) :]
    return T0_PREDC_PREFIX + key


def rename_to_probe_namespace(metrics: Mapping[str, float]) -> dict[str, float]:
    """Rename every metric into the probe namespace, refusing collisions.

    A collision means two distinct source keys (``psnr`` and ``val_psnr``) map
    onto one probe key, which would silently drop one of them. That is the
    #1682 failure shape -- a computed value vanishing without a word -- so it
    raises instead (non-negotiable 3).
    """
    renamed: dict[str, float] = {}
    for key, value in metrics.items():
        new_key = t0_predc_key(key)
        if new_key in renamed:
            raise ValueError(
                f"t=0 pre-DC probe key collision: {key!r} and another source key "
                f"both map to {new_key!r}. Two metrics would be reported as one."
            )
        renamed[new_key] = value
    return renamed


def generator_exposes_pre_dc(generator: Any) -> bool:
    """Whether this generator can hand back its pre-DC proposal in eval mode.

    Read from the class, never from a trial call: this decision selects the
    emitted key set, and the all-reduce contract above requires every rank to
    reach the same answer. A model property is rank-invariant; a probing call
    whose success depends on the batch is not.
    """
    return bool(getattr(generator, "exposes_pre_dc", False))


def build_t0_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    """The terminal rung's timestep vector: t=0 for every sample in the batch."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return torch.zeros(batch_size, dtype=torch.long, device=device)


def slice_row_kwargs(
    forward_kwargs: Mapping[str, Any], *, batch_size: int, start: int, stop: int
) -> dict[str, Any]:
    """Take rows ``[start:stop)`` of every per-row entry; pass the rest through.

    Per-row means a tensor whose leading dimension is the *unchunked* batch:
    ``contrast_idx`` ``(B,)``, ``smaps`` ``(B, C, H, W)``, ``mask``,
    ``kspace_measured``. Anything else -- a scalar, a flag, a broadcastable
    ``(1, ...)`` map -- reaches every chunk untouched.

    A chunked ``x`` beside full-batch kwargs is not a shape error the model
    reports: ``complex_unet`` truncates the longer of ``contrast_emb`` /
    ``t_emb`` to the shorter (#2055), so every chunk would be conditioned on
    row 0's contrast and still score.
    """
    return {
        key: (
            value[start:stop]
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == batch_size
            else value
        )
        for key, value in forward_kwargs.items()
    }


def forward_pre_dc(
    generator: Any,
    x: torch.Tensor,
    *,
    timesteps: torch.Tensor,
    forward_kwargs: Mapping[str, Any],
) -> torch.Tensor:
    """Call the generator at t=0 and return its PRE-data-consistency proposal.

    ``sensitivity_maps`` must be present in ``forward_kwargs``: without it the
    generator silently reuses its stashed maps, so the probe would report a
    number conditioned on whatever ran last (see the module docstring).

    A generator that ignores ``return_pre_dc`` hands back a bare tensor. That is
    not treated as "no pre-DC available" -- it is a wiring defect, and returning
    the bare tensor would report the post-DC output under a pre-DC name, which
    at t=0 is the input itself and scores as a flawless reconstruction. It
    raises.
    """
    if not any(forward_kwargs.get(key) is not None for key in _SMAPS_KEYS):
        raise ValueError(
            f"t=0 pre-DC probe requires explicit sensitivity maps under one of "
            f"{_SMAPS_KEYS}; the generator falls back to its stashed maps when "
            "none is passed, which makes the measurement depend on call order."
        )

    # `timesteps` is passed positionally, so a `timesteps` key surviving in the
    # kwargs is a "multiple values for argument" TypeError. `_build_generator_kwargs`
    # merges the arm's `accelerator_kwargs` wholesale, so the key can arrive from
    # config; the strategy's own forward path pops it for the same reason.
    call_kwargs = {k: v for k, v in forward_kwargs.items() if k != "timesteps"}
    out = generator(x, timesteps, return_pre_dc=True, **call_kwargs)

    if not (isinstance(out, tuple) and len(out) == 2):
        raise TypeError(
            f"{type(generator).__name__}.forward ignored return_pre_dc=True and "
            f"returned {type(out).__name__}; the probe cannot distinguish its "
            "pre-DC proposal from its post-DC output."
        )
    _, x_pre_dc = out
    if not isinstance(x_pre_dc, torch.Tensor):
        raise TypeError(
            f"t=0 pre-DC probe expected a Tensor as the second return value, got "
            f"{type(x_pre_dc).__name__}."
        )
    return x_pre_dc


def run_t0_predc_probe(
    *,
    generator: Any,
    model_input: torch.Tensor,
    forward_kwargs: Mapping[str, Any],
    score: Callable[[torch.Tensor, torch.Tensor], Mapping[str, float]],
    chunk_size: int,
) -> dict[str, float]:
    """Measure the terminal rung pre-DC and return probe-namespaced metrics.

    ``score`` is the caller's single metrics seam, invoked as
    ``score(prediction, timesteps)`` on the reassembled full-batch prediction.
    Scoring is not reimplemented here.

    ``chunk_size`` is rows per forward -- ``validation.loader.chunk_size``, the
    knob every other validation forward already honours. It is a required
    keyword so that omitting it is a ``TypeError`` rather than a silent
    full-batch pass, which is the shape that OOMs. Non-positive values clamp to
    1, matching the multi-step sampler's ``max(1, int(...))``; the schema pins
    the field at ``ge=1``, so neither clamp is reachable from config.

    Returns an empty dict -- on every rank alike -- when the generator does not
    expose a pre-DC proposal.
    """
    if not generator_exposes_pre_dc(generator):
        return {}

    batch_size = int(model_input.shape[0])
    timesteps = build_t0_timesteps(batch_size, model_input.device)
    rows = max(1, int(chunk_size))
    with torch.no_grad():
        parts = [
            forward_pre_dc(
                generator,
                model_input[start : start + rows],
                timesteps=timesteps[start : start + rows],
                forward_kwargs=slice_row_kwargs(
                    forward_kwargs,
                    batch_size=batch_size,
                    start=start,
                    stop=start + rows,
                ),
            )
            for start in range(0, batch_size, rows)
        ]
        # `cat` on a one-element list still copies, and the whole point here is
        # to not hold a second full-batch tensor.
        x_pre_dc = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        del parts
        metrics = score(x_pre_dc, timesteps)
    del x_pre_dc
    return rename_to_probe_namespace(metrics)
