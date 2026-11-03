"""Config schema for the certified-robust reconstruction strategy (A-8.3 CertRob).

Read via ``config.training.certified_robustness``: the adversarial and smoothing knobs by
``AdversarialRobustnessStrategy`` (and so by its subclass), the Lipschitz knobs by
``CertifiedRobustnessStrategy`` only; the parent raises on a Lipschitz knob it would not read.
The headline one-knob ablation is ``lambda_lipschitz`` (``0`` = the adversarial-
training-only control; ``>0`` = also penalise the global Lipschitz bound). The
adversarial knobs are wired here too so the strategy reads every knob from config
rather than the hardcoded class attributes its ``AdversarialRobustnessStrategy``
parent uses (CLAUDE.md pitfall #15).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class CertifiedRobustnessConfig(StrictSchema):
    """Knobs for certified-robust reconstruction (A-8.3).

    Lipschitz certification: the penalty drives every weight layer's spectral
    norm toward ``target_sigma`` so the global certified bound
    (``composed_spectral_norm_bound`` metric = product of per-layer sigmas)
    shrinks. ``lambda_lipschitz`` is the one-knob ablation.
    """

    # --- Lipschitz certification (the A-8.3 one-knob) -----------------------
    lambda_lipschitz: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight of the Lipschitz spectral penalty added to the loss. 0 = "
            "adversarial-training-only control; >0 = certified-robust treatment."
        ),
    )
    target_sigma: float = Field(
        default=1.0,
        gt=0.0,
        description="Per-layer spectral-norm budget; the penalty is 0 iff all sigmas <= this.",
    )
    n_power_iterations: int = Field(
        default=1,
        ge=1,
        description="Power-iteration steps for the differentiable per-layer sigma estimate.",
    )

    # --- adversarial training (wired from config, not hardcoded; #15) -------
    epsilon: float = Field(default=0.01, ge=0.0, description="Attack perturbation budget.")
    attack_steps: int = Field(default=5, ge=1, description="PGD ascent steps.")
    attack_norm: Literal["linf", "l2"] = Field(
        default="linf", description="Attack norm constraint (rejects unknown values; #9)."
    )
    clean_weight: float = Field(default=0.1, ge=0.0, description="Weight of the clean loss term.")
    adv_weight: float = Field(
        default=0.9, ge=0.0, description="Weight of the adversarial loss term."
    )

    # --- attack-time validation certificate --------------------------------
    sigma_smooth: float = Field(
        default=0.25, gt=0.0, description="Randomized-smoothing noise std for certified_radius."
    )
    n_smooth_samples: int = Field(
        default=32, ge=1, description="Monte-Carlo draws per input for certified_radius."
    )
