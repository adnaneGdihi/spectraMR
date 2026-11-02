"""Config schema for the sparsifying-frame coherence-optimisation strategy (A-6.5).

Read by ``SparseFrameStrategy`` via ``config.training.sparse_frame``. The headline
one-knob ablation is ``lambda_coherence`` (0 = tightness+reconstruction-only
control; >0 = also optimise mutual incoherence of the learned atoms).
``lambda_parseval`` (frame tightness) is held fixed across the ablation so the
only varied knob is the coherence penalty (#17 clean ablation).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class SparseFrameConfig(StrictSchema):
    """Knobs for learnable-frame coherence optimisation (A-6.5 SCO-Frame)."""

    lambda_coherence: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Weight on the mutual-coherence penalty selected by coherence_penalty. 0 = tightness+reconstruction-only control; >0 = "
            "coherence-optimised treatment. Linear scale; [0.01, 0.1] is typical for frame "
            "learning (upper bound 100 guards against a divergent config)."
        ),
    )
    lambda_parseval: float = Field(
        default=0.01,
        ge=0.0,
        le=100.0,
        description=(
            "Weight of the always-on Parseval frame-tightness penalty (held fixed across the "
            "ablation). Linear scale; upper bound 100 guards against a divergent config."
        ),
    )
    coherence_penalty: Literal["smooth_max", "frobenius"] = Field(
        default="smooth_max",
        description=(
            "Form of the coherence penalty. smooth_max = log-sum-exp smooth maximum "
            "beta^-1 log sum_{i<j} exp(beta G_ij^2) of the unit-atom Gram, which minimises the "
            "mutual coherence mu = max|G_ij| (PROPOSAL item e, #2368); frobenius = off-diagonal "
            "Gram energy, which lowers the average squared coherence only."
        ),
    )
    coherence_beta: float = Field(
        default=1000.0,
        gt=0.0,
        le=1.0e5,
        description=(
            "Sharpness beta of the smooth_max penalty (read only by smooth_max). The penalty "
            "exceeds max G_ij^2 by at most log(m(m-1)/2)/beta; 1000 keeps that slack below "
            "0.008 for 64 atoms. Log scale; upper bound 1e5 guards against a vanishing gradient "
            "on all but one pair."
        ),
    )
