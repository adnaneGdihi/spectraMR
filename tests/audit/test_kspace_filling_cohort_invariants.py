"""Scientific-coherence invariants for the kspace_filling cold-diffusion cohort.

This driver pins the *scientific contract* shared by every cold-diffusion arm of
experiment 11 (``experiments/inprogress/kspace_filling/``) so the cohort cannot
silently drift apart again. It is the regression guard for the 2026-06-09
scientific audit, which found that the baseline arm had been fixed ~10 times
across May-June while its mask / architecture / attention variants froze at the
pre-fix state — re-introducing every bug the base had repaired (DC-blob loss
recipe, single-forward "cold" diffusion, pathological 2x selection metric, a
heavily-lagged EMA shadow). None of those is a schema error, so the audit ladder
and ``from_yaml`` accept them; only a contract test catches them.

Scope: every YAML directly under ``kspace_filling/`` (excluding ``*ablation*``
arms, which deliberately vary an extra knob) plus the ``attention_shootout/`` and
``attention_enhancements/`` sub-cohorts. The ``ablations/`` and
``ablations_kan_dual_domain/`` directories are out of scope by construction.

Each invariant maps to a numbered finding from the audit and the CLAUDE.md
pitfall taxonomy:

* A (C2, pitfall #16/#20) — ``validation.multistep_cold_sampling`` must be True:
  a single deterministic forward regresses to the posterior mean (a blur at
  heavy R). An arm named *cold diffusion* that never runs the reverse loop is a
  facade at the scientific layer.
* B (M2, pitfall #20) — ``ema.decay <= 0.999``: these are diagnostic-length runs
  (early-stop patience 1.5k-5k). At 0.9999 the EMA shadow validation grades is
  ~74% random init at the stopping point.
* C (M1, pitfall #18) — selection metric is the cascade MEAN, not a single R.
  ``val_robust_mri_psnr_2x`` is gameable by a high-R-collapsed DC blob and
  mismatches the "fills across the 2x-32x cascade" hypothesis.
* D (C1, pitfall #20) — k-space data-fidelity is present: ``kspace_losses`` is
  non-empty and contains ``complex_l1``. ``image_losses:[mse] + kspace_losses:[]``
  drops k-space fidelity, making a measurement-independent contrast-mean a
  near-optimal minimiser.
* E (M4, pitfall #19/#17) — a falsifiable claim is declared: ``hypothesis`` and
  ``primary_metric`` are present, and a ``baseline`` is named unless the arm is
  itself the cohort control (``role: baseline``).
* F (m5, pitfall #20) — ``validation.input_dependence_tol`` is set so the
  measurement-collapse gate can fire.
* G (m1, pitfall #15) — radial/spiral arms keep ``enable_dynamic_mask: false``:
  those accelerators have no RNG, so per-sample seed variation is an inert knob.
* H (kan-DC overclaim, pitfall #16) — an arm whose ``dc_method`` is not in the
  adaptive family (``adaptive`` / ``kan_adaptive``) must NOT advertise an *active*
  KAN-DC mechanism ("KAN ADC", "KAN trust map") in its provenance-stamped
  ``description`` / ``hypothesis``. The KAN data-consistency layer is only built
  under ``dc_method: kan_adaptive``; hard DC is the cohort-wide standard since
  2026-07-29 (it never re-injects an unsampled bin, and unlike soft it keeps the
  measured lines exact rather than blending them 1/3-2/3 with the prediction —
  see the DC invariants at the bottom of this module). Claiming a
  KAN ADC that never instantiates is a facade at the provenance layer. A *disclaimed*
  mention ("KAN ADC kwargs present but only consumed by...") is allowed.
* I (spread-through-acceleration, pitfall #17/#20) — the timestep curriculum that
  spreads learning across the 2x-32x cascade must match the base SSOT:
  ``training.timestep_sampling_strategy == "balanced_high_t"``, and the curriculum
  pair ``(curriculum_start_timestep, curriculum_ramp_rate)`` must open the full
  ladder INSIDE the arm's ``max_iterations`` and be uniform within the arm's own
  sub-cohort. Until 2026-08-28 this pinned ``curriculum_start_timestep == 4`` as a
  literal; that could not survive the corpus becoming multi-budget (top-level arms
  at (4, 0.005)/1000000, the attention sub-cohorts at (1, 0.0005)/70000) and it was
  blind to ``ramp_rate``, the half that decides when the ladder actually opens.
  History: the cohort first used ``high_t_emphasis`` (P(t) ∝ t,
  the 2026-06-08 negative-transfer fix for R8x/R32x collapse), but the 2026-07-24
  attention_none 30k retrain proved it OVER-corrected — once the curriculum cap
  opens the full range (~iter 4800) pure P(t) ∝ t starves t=1,2 to <2% of draws,
  the net forgets the low-t band R2x validation reverse-samples, and R2x val PSNR
  degrades over training and inverts BELOW R32x (2.7 vs 6.0 dB at 30k). The fix is
  ``balanced_high_t`` = (1-eps)·P(t) ∝ t + eps·uniform (eps=0.3): high-t keeps
  gradient priority WITHOUT starving low t. ``importance`` (P(t) ∝ (T-t)^2, biased
  toward the EASY low-R end), ``high_t_emphasis`` (starves low t), and ``uniform``
  (no high-t priority) all break one end of the cascade. None of these arms varies
  the sampler as its tested axis, so a divergent value is drift (a confound vs the
  base) that breaks the shared "fills across the cascade" claim.
  The structural schedule is asserted too: ``model_kwargs.timesteps ==
  diffusion.timesteps == acceleration.schedule_steps == len(acceleration_range)``
  and ``schedule_type == step``. If those drift apart the sinusoidal
  time-embedding collapses and the acceleration no longer maps to
  distinguishable timesteps. The ladder LENGTH was missing from that equality
  until 2026-08-28, which is why the cohort could run a 29-rung arm beside nine
  28-rung ones with this gate green throughout. The ladder HEAD is asserted as a
  coupling in the same place -- ``base_acceleration == acceleration_range[0]``,
  an R=1.0 head implies ``train_identity_rung``, and a trained identity rung
  implies ``lambda_pre_dc_kspace`` -- never as a literal, because the top-level
  arms legitimately start at 4.0 where the attention arms start at 1.0.
* J (perf, pitfall #17) — every arm whose live attention is the KAN dual-domain
  block carries ``model_kwargs.kan_dual_domain_kwargs.max_dense_attn_tokens``;
  the dense image/cross branches adaptive-pool to a ``round(sqrt(budget))²`` grid
  before attending, which bounds the O(N²) ``ComplexMHA`` score tensor (the deepest
  OOM alloc frame, ``dual_domain_attention_kan.py:891``). The whole kan-attention
  family (shootout + enhancement matrix + standalone) was reduced 4096→2304
  (64²→48²) on 2026-06-26 to fit the 32 GB cards it was OOM-ing on. Because the
  budget sets the *pooled attention resolution*, it MUST stay uniform across the
  family or a memory-vs-quality confound leaks into the head-to-heads. Output
  k-space stays full 256² regardless — this is a benign attention approximation,
  NOT a data crop (unlike ``data.patch_size``, which on this k-space pipeline
  would truncate frequencies and lower the reconstruction resolution).
* K (resolution, pitfall #17) — ``data.patch_size`` is the full 256² FOV on every
  arm. On this k-space cold-diffusion pipeline the patch IS the reconstructed
  field-of-view, so a 128² patch truncates the outer k-space (halves the
  reconstruction resolution), not merely the attention grid (invariant J). The
  cohort had drifted into a 256²/128² split (all of ``attention_enhancements/``
  plus three ``attention_shootout/`` arms ran at 128² as a 2026-05-14 OOM
  mitigation), so a metric Δ between a 256² shootout baseline and a 128²
  enhancement arm confounded the tested knob with reconstruction resolution. The
  cohort is pinned at 256² so every head-to-head is resolution-controlled
  (2026-06-27 audit).
* L (double Fourier bridge, issue #467 / pitfall #16) — no loss that bridges from
  k-space *internally* may sit under ``losses.image_losses`` or ``complex_losses``
  while ``losses.output_domain`` is ``kspace``. Both lists get an iFFT bridge for
  that output domain, so a loss carrying its own ``DifferentiableFourierBridge`` is
  inverse-transformed TWICE: the outer bridge emits a real 4-channel magnitude
  image, the inner one re-reads it as k-space and re-pairs channels as (real, imag)
  — coil-0's magnitude becomes the real part of channel 0 — then iFFTs again.
  Nothing raises and the value stays finite, so the run is green while the term
  measures nothing it advertises. The whole cohort carried this on
  ``complex_spatial_gradient`` (weight 1.0, the term that is supposed to enforce
  image-domain phase coherence — under the double bridge it is provably phase
  BLIND, see tests/unit/models/losses/test_physics_losses.py) and on
  ``sense_adjoint_l1`` (weight 0.3, whose 4-coil sensitivity maps were then
  silently sliced to 2 to match the halved channel count). Both now live under
  ``kspace_losses``, where bridge mode is ``none`` and their own bridge does the
  single iFFT. ``LossBuilder`` rejects the bad placement at build time; this
  invariant keeps the *configs* from drifting back.
* M (selection density, pitfall #18) — ``validation.eval_interval`` must divide the
  iteration budget into at least 6 validation points, and
  ``early_stopping.patience`` must exceed that count. The cohort ran
  ``eval_interval: 10000`` against ``max_iterations: 30000`` — 3 points — while
  ``val_robust_mri_psnr_mean`` (invariant C's selection metric) is stamped ONLY
  inside validation, so best-checkpoint selection chose among 3 candidates and each
  arm's curve was 3 samples. ``patience`` counts validation EVENTS, not iterations
  (``early_stopping_service.update`` runs only inside the validation block), so the
  cohort's ``5000`` was inert; it must stay ABOVE the event count so a fixed-budget
  head-to-head cannot truncate an arm. ``early_stopping.enabled`` must nonetheless
  stay True: the best-checkpoint writer is gated on the same service
  (``training_loop.py:1171`` wraps ``:1247``), so disabling early stopping to
  protect the budget would silently delete best-checkpoint selection. Arms with a
  sub-10k budget are a different class (quick diagnostics) and are exempt.
* N (EMA provenance parity, pitfall #17) — every arm pins ``ema.warmup``
  explicitly. The schema default is True, so an omission is not a behaviour
  difference — but the ``attention_shootout`` control was the ONE arm omitting it
  while all nine siblings pinned it with a comment, and that control is what every
  sibling names as its baseline. Invariant B covers ``ema.decay`` only, so nothing
  held the line. Arms that deliberately test the flag live under ``ablations*/``
  and are out of scope by construction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.physics.dc_settings import DC_SSOT_KEYS
from tests.utils.corpus import tracked_yamls
from tests.utils.repo_scripts import require_repo_file, skip_if_public_export

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COHORT_ROOT = _REPO_ROOT / "experiments" / "inprogress" / "kspace_filling"


def _discover_arms() -> list[Path]:
    """In-scope cold-diffusion arms: top-level + the two attention sub-cohorts."""
    arms: list[Path] = [
        p for p in tracked_yamls(_COHORT_ROOT, recursive=False) if "ablation" not in p.name
    ]
    for sub in ("attention_shootout", "attention_enhancements"):
        arms.extend(tracked_yamls((_COHORT_ROOT / sub), recursive=False))
    return sorted(arms)


_ARMS = _discover_arms()
_IDS = [str(p.relative_to(_COHORT_ROOT)) for p in _ARMS]


def _meta(settings: TrainingSettings, key: str) -> Any:
    """Read a metadata field from either the top-level or the free-form ``tags`` dict."""
    md = settings.metadata
    md = (
        md
        if isinstance(md, dict)
        else (md.model_dump() if hasattr(md, "model_dump") else {})
    )
    if md.get(key) not in (None, ""):
        return md[key]
    tags = md.get("tags") or {}
    if isinstance(tags, dict) and tags.get(key) not in (None, ""):
        return tags[key]
    return None


def _loss_names(losses: Any, section: str) -> list[str]:
    items = getattr(losses, section, None) or []
    out: list[str] = []
    for it in items:
        name = it.get("name") if isinstance(it, dict) else getattr(it, "name", None)
        if name:
            out.append(name)
    return out


_ADAPTIVE_DC_FAMILY = ("adaptive", "kan_adaptive")
# An *active* claim that a KAN/adaptive DC mechanism is running. A disclaimed
# mention of the present-but-unread knob ("KAN ADC kwargs present but only
# consumed by ...") is exempted via the negative lookahead.
_KAN_DC_OVERCLAIM_RE = re.compile(r"kan adc(?! kwargs)|kan trust", re.IGNORECASE)


def _dc_method(settings: TrainingSettings) -> Any:
    return (settings.model.model_kwargs or {}).get("dc_method")


def _attr(obj: Any, *names: str) -> Any:
    """Read the first present attribute (or dict key) from a config sub-object."""
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if v is not None:
            return v
    return None


#: Arms in the cohort directory that are NOT part of the single-contrast
#: exp-11 head-to-head, and so are not bound by its COMPARABILITY invariants.
#:
#: Both reconstruct one contrast using another as a prior, which changes the
#: input, the claim and the sensible selection metric. Requiring them to select
#: on the same cascade mean, or to sweep the same accelerations, compares nothing
#: -- there is no head-to-head for them to be comparable within.
#:
#: This exempts them from the comparability invariants ONLY (C, I). The
#: COHERENCE invariants -- does the EMA converge inside the budget, is there any
#: k-space data fidelity, is the collapse gate armed, is the claim falsifiable --
#: are properties of a single arm and still apply. Exempting those would silence
#: real defects: see issue #687, where `experiment_130_ti_ccd` fails four of them.
# Arms whose ``acceleration_range`` is shorter than their timestep count. This
# is the #1155 defect (21 of 28 timesteps re-emit an identical mask), NOT an
# exemption on the merits -- it is a ratchet baseline so a NEW arm cannot ship
# the shape, recorded with the issue that tracks the fix. Never add to it.
_INERT_LADDER_ARMS: dict[str, str] = {
    "experiment_11_kspace_cold_diffusion_radial.yaml": "7 rungs over 28 timesteps (#1573)",
    "experiment_11_kspace_cold_diffusion_spiral.yaml": "7 rungs over 28 timesteps (#1573)",
}

_CROSS_CONTRAST_ARMS: dict[str, str] = {
    "experiment_130_ti_ccd.yaml": (
        "Translation-Invariant Cross-Contrast K-Space Diffusion: federated "
        "T1+FLAIR multi-contrast, not a single-contrast exp-11 arm"
    ),
    "experiment_cross_contrast_kspace_diffusion.yaml": (
        "conditions a k-space cold-diffusion sampler on a fully-sampled T1 "
        "prior to reconstruct T2/FLAIR; the prior is the independent variable"
    ),
}


@pytest.fixture(scope="module", params=_ARMS, ids=_IDS)
def arm_path(request: pytest.FixtureRequest) -> Path:
    return request.param


@pytest.fixture(scope="module")
def settings(arm_path: Path) -> TrainingSettings:
    return TrainingSettings.from_yaml(str(arm_path))


def _skip_if_cross_contrast(arm_path: Path) -> None:
    """Comparability invariants do not bind arms outside the head-to-head."""
    reason = _CROSS_CONTRAST_ARMS.get(arm_path.name)
    if reason:
        pytest.skip(f"cross-contrast arm, not in the exp-11 head-to-head: {reason}")


def test_arms_discovered() -> None:
    """Guard against an empty parametrization silently passing every invariant."""
    skip_if_public_export("experiments/ does not ship; the kspace_filling cohort is empty here")
    assert (
        len(_ARMS) >= 30
    ), f"expected the full kspace_filling cohort, found {len(_ARMS)}"


def test_a_multistep_cold_sampling_enabled(settings: TrainingSettings) -> None:
    assert settings.validation.sampling.enable_multistep_cold is True, (
        "C2/#16: cold-diffusion arm must reconstruct via the multi-step reverse loop; "
        "a single deterministic forward posterior-mean-blurs at heavy R."
    )


def test_b_ema_decay_not_lagged(settings: TrainingSettings) -> None:
    """EMA must converge inside the run, measured against the run's own length.

    This asserted a flat ``decay <= 0.999``, which is not what the message
    claimed ("too high for a *diagnostic-length* run") and got both directions
    wrong. An EMA's time constant is ``1 / (1 - decay)`` iterations; the defect
    is that constant being comparable to, or larger than, the budget -- the
    averaged weights never catch up and validation grades a near-random shadow
    of the trained model (pitfall #20).

    Measured 2026-08-03, the flat threshold was simultaneously:

    * too strict -- ``experiment_cross_contrast_kspace_diffusion`` runs
      1,000,000 iterations at ``decay=0.9999`` (tau = 10,000, i.e. 1% of the
      budget). Perfectly converged, and failing.
    * too lenient in principle -- it would pass ``decay=0.999`` (tau = 1,000) on
      a 1,000-iteration run, which is exactly the failure it exists to catch.

    The rule is now ``tau <= budget / 10``: the EMA gets at least ten time
    constants to converge. ``experiment_130_ti_ccd`` still fails it, correctly --
    tau = 10,000 on a 1,500-iteration budget.
    """
    if not settings.ema.enabled:
        pytest.skip("EMA disabled for this arm")
    budget = getattr(settings.training, "max_iterations", None)
    if not budget:
        pytest.skip("arm pins no iteration budget; the ratio is undefined")
    tau = 1.0 / max(1.0 - settings.ema.decay, 1e-12)
    if tau > budget / 10.0 and getattr(settings.ema, "warmup", False):
        # `decay` is a CEILING, not the effective decay. With the ramp on,
        # `effective_decay = min(decay, (1+n)/(10+n))`, so the shadow tracks the
        # live model from step 1 and the "ten time constants" framing does not
        # apply -- the ramp IS the mitigation for a high ceiling on a short run.
        # Reading `decay` alone flags exactly the arms that already fixed this.
        pytest.skip(
            f"ema.decay={settings.ema.decay} exceeds the ratio, but ema.warmup "
            "ramps the effective decay (min(decay, (1+n)/(10+n))), so the "
            "shadow is not lagged"
        )
    assert tau <= budget / 10.0, (
        f"M2/#20: ema.decay={settings.ema.decay} gives a time constant of "
        f"{tau:,.0f} iterations on a {budget:,}-iteration budget "
        f"({tau / budget:.1%} of the run). The averaged weights never converge, "
        "so validation grades a heavily-lagged shadow. Either set "
        f"decay <= {1 - 10.0 / budget:.6f} at this budget, or enable "
        "ema.warmup to ramp the effective decay."
    )


def test_c_selection_metric_is_cascade_mean(settings: TrainingSettings, arm_path: Path) -> None:
    """Select on a cascade MEAN the run actually emits — any metric, not just PSNR.

    2026-07-29: this asserted the literal ``val_robust_mri_psnr_mean``, which
    conflated two separate requirements. Only one of them is the invariant:

    * the selection key must be a cascade **mean**, never a single-R proxy that
      high-R collapse can game (``val_robust_mri_psnr_2x``) — that is the rule;
    * it must be **PSNR** — that was never the rule, and it is actively wrong for
      this cohort. PSNR provably ranks the blurriest checkpoint first: on the
      attention_none run it scored the two cases carrying 7.8% of the target's
      high-band power at 42.2 / 41.4 dB and the one that matched it (103%) at
      33.4 dB. The attention_shootout family now selects on ``val_hfen_mean``,
      which is a cascade mean and still a fidelity error norm against the true
      target.

    The shape check is paired with a produced-ness check, so the looser name rule
    cannot let through selection on a key the run never computes (pitfall #18) —
    which the hardcoded version never verified at all.
    """
    _skip_if_cross_contrast(arm_path)
    name = settings.metrics.best_metric_name
    assert name and name.endswith("_mean"), (
        f"M1/#18: best_metric_name={name!r}; select on a cascade mean (``*_mean``), "
        "not a single-R proxy gameable by high-R collapse."
    )
    assert settings.early_stopping.metric == name, (
        f"M1/#18: early_stopping.metric={settings.early_stopping.metric!r} does not "
        f"match best_metric_name={name!r}; the two must select the same checkpoint."
    )

    base = name.removeprefix("val_").removesuffix("_mean")
    declared = {str(getattr(m, "value", m)) for m in settings.validation.scoring.compute}
    assert base in declared, (
        f"M1/#18: selects on {name!r} but validation.metrics={sorted(declared)!r} never "
        f"emits {base!r} — ``_stamp_accel_mean`` averages the CONFIGURED metrics, so "
        "this selects on a key the run does not compute."
    )


def test_d_kspace_fidelity_present(settings: TrainingSettings) -> None:
    names = _loss_names(settings.losses, "kspace_losses")
    assert "complex_l1" in names, (
        "C1/#20: k-space data-fidelity (complex_l1) is absent from kspace_losses "
        f"(found {names!r}). image-domain MSE alone admits a DC-blob minimiser."
    )


def test_e_falsifiable_claim_declared(settings: TrainingSettings) -> None:
    assert _meta(
        settings, "hypothesis"
    ), "M4/#19: metadata.hypothesis is missing (the tell)."
    assert _meta(
        settings, "primary_metric"
    ), "M4/#18: metadata.primary_metric is missing."
    if _meta(settings, "role") != "baseline":
        assert _meta(
            settings, "baseline"
        ), "M4/#17: a non-baseline arm must name its control via metadata.baseline."


def test_f_input_dependence_gate_set(settings: TrainingSettings) -> None:
    assert (
        settings.validation.gates.input_dependence_tol is not None
    ), "m5/#20: validation.input_dependence_tol is unset; the measurement-collapse gate cannot fire."


def test_g_inert_dynamic_mask_not_advertised(settings: TrainingSettings) -> None:
    if settings.undersampling.acceleration_type in ("radial", "spiral"):
        assert settings.undersampling.enable_dynamic_mask is False, (
            "m1/#15: radial/spiral accelerators have no RNG; enable_dynamic_mask is an inert knob "
            "and must stay false to avoid advertising a no-op."
        )


def test_h_no_inactive_dc_mechanism_advertised(settings: TrainingSettings) -> None:
    if _dc_method(settings) in _ADAPTIVE_DC_FAMILY:
        return  # the KAN/adaptive DC layer is genuinely built; the claim is honest
    claim = " ".join(
        str(_meta(settings, k) or "") for k in ("description", "hypothesis")
    )
    hit = _KAN_DC_OVERCLAIM_RE.search(claim)
    assert hit is None, (
        f"#16: dc_method={_dc_method(settings)!r} builds soft/hard DC, but the metadata "
        f"advertises an active KAN-DC mechanism ({hit.group(0)!r}). The KAN data-consistency "
        "layer is only instantiated at dc_method=kan_adaptive; de-claim the description/"
        "hypothesis (the live KAN mechanism here is the gated *attention*, not the DC)."
    )


def test_i_acceleration_spread_matches_base(settings: TrainingSettings, arm_path: Path) -> None:
    _skip_if_cross_contrast(arm_path)
    # (1) the learning-spread curriculum — the negative-transfer fix
    strat = _attr(settings.training, "timestep_sampling_strategy")
    assert strat == "balanced_high_t", (
        f"#17/#20: timestep_sampling_strategy={strat!r}; the cohort spreads learning across "
        "the 2x-32x cascade via 'balanced_high_t' ((1-eps)·P(t) ∝ t + eps·uniform). Pure "
        "'high_t_emphasis' starves low t (R2x val collapse); 'importance'/'uniform' bias "
        "toward the easy low-R end and let the shared net abandon high t (R8x/R32x collapse)."
    )
    # The curriculum cap is ``dynamic_max = int(start_t + iteration * rate)``
    # (diffusion.py:1142), so ``curriculum_start_timestep`` and
    # ``curriculum_ramp_rate`` are ONE design and only the DERIVED quantity --
    # the iteration at which the ladder fully opens -- is comparable across arms.
    #
    # This assertion pinned ``start_t == 4`` as "the base SSOT" until 2026-08-28.
    # A literal cannot serve this corpus: the top-level arms run (4, 0.005) at a
    # 1000000 budget and the two attention sub-cohorts run (1, 0.0005) at 70000,
    # and each is correct FOR ITS BUDGET (4800/1000000 vs 54000/70000). The pin
    # was also blind to the half that matters -- it passed (4, 0.5), which opens
    # the whole ladder by iteration 48 -- while failing arms that were merely
    # budgeted differently. Cross-arm uniformity, which is the confound the pin
    # was really protecting, is enforced per sub-cohort in
    # ``test_i2_curriculum_uniform_within_each_sub_cohort`` instead.
    cstart = _attr(settings.training, "curriculum_start_timestep")
    rate = _attr(settings.training, "curriculum_ramp_rate")
    assert cstart is not None and rate, (
        f"#17: the timestep curriculum is declared by TWO knobs "
        f"(curriculum_start_timestep={cstart!r}, curriculum_ramp_rate={rate!r}); "
        "both must be pinned, or the cap silently falls back to the full range."
    )
    assert cstart >= 1, (
        f"#17: curriculum_start_timestep={cstart!r}; both declaring schemas pin ge=1 "
        "(config/schemas/training/base.py:761, .../diffusion.py:153), so 0 is a "
        "load-time failure and a negative value is meaningless."
    )
    _tsteps = _attr(_attr(settings.training, "diffusion"), "timesteps")
    _budget = settings.training.max_iterations
    if _tsteps and _budget:
        opens_at = (_tsteps - cstart) / rate
        assert opens_at < _budget, (
            f"#17: the curriculum opens the full ladder at iteration {opens_at:.0f} "
            f"((timesteps {_tsteps} - start {cstart}) / rate {rate}), which is outside "
            f"the max_iterations={_budget} budget. The top rungs would get ZERO training "
            f"draws while validation still reverse-samples through them."
        )
    # (2) the structural schedule — three numbers must agree or the time-embedding collapses
    mk_t = (settings.model.model_kwargs or {}).get("timesteps")
    diff_t = _attr(_attr(settings.training, "diffusion"), "timesteps")
    sched_steps = _attr(settings.undersampling, "schedule_steps")
    assert mk_t == diff_t == sched_steps, (
        f"#20: model_kwargs.timesteps({mk_t}) / diffusion.timesteps({diff_t}) / "
        f"acceleration.schedule_steps({sched_steps}) must all agree, else the sinusoidal "
        "time-embedding collapses and acceleration no longer maps to distinguishable timesteps."
    )
    rungs = list(_attr(settings.undersampling, "acceleration_range") or [])
    if arm_path.name not in _INERT_LADDER_ARMS:
        assert len(rungs) == diff_t, (
            f"#20: len(acceleration_range)={len(rungs)} but diffusion.timesteps={diff_t}. "
            "`schedule_type: step` holds each rung for T/len(range) steps, so a short "
            "ladder makes most timesteps re-emit an identical mask -- the time embedding "
            "is trained to separate pixel-identical states and those reverse steps have "
            "nothing to reveal (#1155). This was NOT part of the three-way agreement "
            "above until 2026-08-28, which is exactly how a 29-rung arm sat beside nine "
            "28-rung siblings with this gate green."
        )
    sched_type = str(_attr(settings.undersampling, "schedule_type")).lower()
    assert "step" in sched_type, (
        f"#20: acceleration.schedule_type={sched_type!r}; the cohort spreads R across "
        "timesteps via the 'step' schedule, one rung per timestep."
    )

    # (3) the ladder HEAD, and the two knobs that decide whether it is real.
    # Deliberately a coupling, never a literal: the 15 top-level arms start at
    # 4.0 while the 20 attention arms start at 1.0, so `== 1.0` here would fail
    # arms this cohort never touched -- the same trap the superseded
    # `curriculum_start_timestep == 4` pin fell into.
    if rungs:
        base = _attr(settings.undersampling, "base_acceleration")
        assert base == rungs[0], (
            f"#17: base_acceleration={base!r} but acceleration_range[0]={rungs[0]!r}. "
            "R(0) is read off the LADDER (min_meaningful_timestep goes through "
            "`accelerator.get_acceleration_factor(0)`), so a base that disagrees with "
            "the head documents a schedule the run does not execute."
        )
        tir = _attr(settings.undersampling, "train_identity_rung")
        if float(rungs[0]) == 1.0:
            assert tir is True, (
                f"#16: acceleration_range starts at R=1.0 but train_identity_rung={tir!r}. "
                "min_meaningful_timestep() returns 1 at R(0) == 1, so the sampler never "
                "draws t=0 and the head is declared-but-never-trained. Measured, not "
                "theoretical: attention_none carried exactly this shape from cc170a756 "
                "and rung 0 got ZERO draws over its full 150000-iteration budget."
            )
        if tir is True:
            pre_dc = getattr(settings.losses.reconstruction, "lambda_pre_dc_kspace", None)
            assert pre_dc, (
                f"#16: train_identity_rung is on but lambda_pre_dc_kspace={pre_dc!r}. "
                "At t=0 every bin is acquired, so hard DC replaces the network's proposal "
                "everywhere and every post-DC loss is a constant with zero gradient; the "
                "pre-DC k-space term is the rung's ONLY gradient path. Without it the rung "
                "consumes its share of draws and trains nothing."
            )


# ── Full-FOV reconstruction resolution (the 2026-06-27 de-confound) ────────────
# On the k-space cold-diffusion pipeline ``data.patch_size`` sets the
# reconstructed field-of-view: a sub-256² patch truncates the outer k-space and
# halves the reconstruction resolution. Pinned cohort-wide so the shootout /
# enhancement head-to-heads are resolution-controlled (distinct from invariant J,
# which bounds only the pooled *attention* grid and leaves output k-space at 256²).
def test_i2_curriculum_uniform_within_each_sub_cohort() -> None:
    """The curriculum pair must be single-valued inside each sub-cohort.

    This is the invariant the superseded ``curriculum_start_timestep == 4``
    literal was really protecting -- "a divergent value is a confound vs the
    base" is a statement about an arm's SIBLINGS, not about a global constant.
    Expressed as uniformity it survives the corpus being multi-budget, and it
    covers ``ramp_rate`` too, which the literal never looked at.

    Grouped by directory because that is the unit of comparison: the ten
    ``attention_shootout`` arms are compared to each other, the ten
    ``attention_enhancements`` cells to each other, the top-level arms to each
    other. Cross-contrast arms are exempt for the usual reason (they are not in
    any head-to-head).
    """
    skip_if_public_export("experiments/ does not ship; there are no arms to group")
    groups: dict[str, dict[tuple[Any, Any], list[str]]] = {}
    for arm in _ARMS:
        if arm.name in _CROSS_CONTRAST_ARMS:
            continue
        s = TrainingSettings.from_yaml(str(arm))
        key = (
            _attr(s.training, "curriculum_start_timestep"),
            _attr(s.training, "curriculum_ramp_rate"),
        )
        groups.setdefault(arm.parent.name, {}).setdefault(key, []).append(arm.name)

    _assert_single_valued(groups, "curriculum pairs")


def _assert_single_valued(
    groups: dict[str, dict[Any, list[str]]], what: str
) -> None:
    """Every sub-cohort must declare exactly one value of whatever was keyed.

    Split out of ``test_i2`` so it can be planted directly: a corpus-level scan
    has no file to mutate, and an unplantable detector is its own blindness
    shape. The callers assert their own COVERAGE (that the attention groups are
    present and full), so a green plant here cannot stand in for a call site
    that silently grouped nothing.
    """
    assert groups, "no arms grouped -- the parametrization or the corpus is empty"
    for group, variants in sorted(groups.items()):
        assert len(variants) == 1, (
            f"#17: {group}/ declares {len(variants)} different {what}, so its "
            f"arms are not comparable to each other: "
            + "; ".join(
                f"{k!r} on {sorted(v)}"
                for k, v in sorted(variants.items(), key=lambda kv: str(kv[0]))
            )
        )


def test_i3_ladder_uniform_within_each_sub_cohort() -> None:
    """The whole mask ladder must be single-valued inside each sub-cohort.

    The head-to-head claim is "attention_type is the only axis", which a
    divergent ladder falsifies outright: from cc170a756 (2026-08-20) until
    2026-08-28 ``attention_none`` ran 29 rungs from ``base_acceleration: 1.0``
    against nine siblings on 28 from ``2.0``, so its rung *k* carried the
    acceleration theirs carried at *k-1* and no per-timestep curve was
    comparable. Keyed on the full tuple rather than on length alone, because the
    same divergence also appeared as one relabelled rung (29.444 vs 28.444).
    """
    groups: dict[str, dict[Any, list[str]]] = {}
    for arm in _ARMS:
        # ``_INERT_LADDER_ARMS`` skipped for the reason they are baselined at
        # all: their 7-rung ladder is the #1155 defect, so they would report as
        # a second "variant" of the top-level group and mask any real drift
        # arriving beside them. Excluded as known-defective-and-tracked, not as
        # a judgement that uniformity does not apply -- the other eleven
        # top-level arms share one ladder and are still compared here.
        if arm.name in _CROSS_CONTRAST_ARMS or arm.name in _INERT_LADDER_ARMS:
            continue
        s = TrainingSettings.from_yaml(str(arm))
        key = (
            _attr(s.undersampling, "base_acceleration"),
            tuple(_attr(s.undersampling, "acceleration_range") or ()),
            _attr(s.undersampling, "train_identity_rung"),
        )
        groups.setdefault(arm.parent.name, {}).setdefault(key, []).append(arm.name)

    # coverage: prove the scan actually reached both attention sub-cohorts at
    # full strength, so this cannot pass by grouping nothing.
    for sub_cohort in ("attention_shootout", "attention_enhancements"):
        assert sub_cohort in groups, f"{sub_cohort}/ not reached by the ladder scan"
        assert sum(len(v) for v in groups[sub_cohort].values()) == 10, (
            f"{sub_cohort}/ contributed "
            f"{sum(len(v) for v in groups[sub_cohort].values())} arms, expected 10"
        )
    _assert_single_valued(groups, "mask ladders")


# --------------------------------------------------------------------------
# Planted violations (non-negotiable 15).
#
# Each new assertion above is exercised by mutating a REAL arm and calling the
# REAL test function, so the plant lands at the call site rather than on an
# extracted helper. ``_planted`` asserts the mutation survived config load
# before the violation is asserted -- a plant that silently failed to apply
# would otherwise score green.
# --------------------------------------------------------------------------

_PLANT_BASE_REL = (
    "experiments/inprogress/kspace_filling/attention_shootout/"
    "experiment_11_attention_channel.yaml"
)


def _plant_base() -> Path:
    """The arm every NN15 plant below mutates.

    Resolved through ``require_repo_file`` rather than held as a constant: the
    plants are the detector's own proof that it goes red, and in the public
    export their subject does not ship. A skip says that; a FileNotFoundError
    from ``read_text`` says the plant is broken.
    """
    return require_repo_file(_PLANT_BASE_REL)


def _planted(tmp_path: Path, **dotted: Any) -> TrainingSettings:
    """Load ``_plant_base()`` with ``dotted`` keys overridden, proving each landed."""
    raw = yaml.safe_load(_plant_base().read_text())
    for path, value in dotted.items():
        node = raw
        *parents, leaf = path.split("__")
        for p in parents:
            node = node[p]
        assert leaf in node, f"plant target {path!r} absent from {_plant_base().name}"
        node[leaf] = value
    out = tmp_path / _plant_base().name
    out.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    settings = TrainingSettings.from_yaml(str(out))
    for path, value in dotted.items():
        *parents, leaf = path.split("__")
        node: Any = settings
        for p in parents:
            node = getattr(node, p)
        got = _attr(node, leaf)
        assert got == value, f"plant {path}={value!r} did not survive load (got {got!r})"
    return settings


def test_plant_curriculum_ladder_opening_past_the_budget_is_red(tmp_path: Path) -> None:
    """A ramp too slow to open the ladder inside the budget must turn test_i red.

    This is the shape the superseded literal pin could not see at all: start_t
    stays at its expected value and only ``ramp_rate`` moves.
    """
    s = _planted(tmp_path, training__curriculum_ramp_rate=0.0001)  # opens at 270000
    with pytest.raises(AssertionError, match="#17"):
        test_i_acceleration_spread_matches_base(s, _plant_base())


def test_plant_unpinned_ramp_rate_is_red(tmp_path: Path) -> None:
    """Dropping half the pair must turn test_i red, not fall back to full range."""
    s = _planted(tmp_path, training__curriculum_ramp_rate=None)
    with pytest.raises(AssertionError, match="#17"):
        test_i_acceleration_spread_matches_base(s, _plant_base())


def test_plant_truncating_patience_is_red(tmp_path: Path) -> None:
    """A patience that CAN fire inside the budget must turn test_m red.

    patience 8 over 70000/5000 stops at (8 + 1) * 5000 = 45000, discarding 36%
    of the budget -- the exact regression this cohort hit when the budget was
    raised from 30000 and patience was left behind.
    """
    s = _planted(tmp_path, early_stopping__patience=8)
    with pytest.raises(AssertionError, match="M/#18"):
        test_m_selection_density_and_fixed_budget(s)


def test_plant_boundary_patience_is_green(tmp_path: Path) -> None:
    """Ratchet direction: patience == points must PASS.

    The superseded ``patience > points`` failed this, which is the off-by-one
    the rewrite corrects. 14 events fit in 70000/5000 and a stop still needs 15,
    so the arm cannot truncate. Without this case the correction is unverified
    and could silently revert.
    """
    s = _planted(tmp_path, early_stopping__patience=14)
    test_m_selection_density_and_fixed_budget(s)


def test_plant_ladder_shorter_than_timesteps_is_red(tmp_path: Path) -> None:
    """A ladder with fewer rungs than timesteps must turn test_i red.

    The shape that ran unseen for eight days: the three scalars still agree
    with each other, only the LIST length is wrong.
    """
    s = _planted(tmp_path, undersampling__acceleration_range=[1.0, 2.0, 8.0, 32.0])
    with pytest.raises(AssertionError, match="#20"):
        test_i_acceleration_spread_matches_base(s, _plant_base())


def test_plant_base_disagreeing_with_the_head_is_red(tmp_path: Path) -> None:
    """base_acceleration that is not the first rung must turn test_i red."""
    s = _planted(tmp_path, undersampling__base_acceleration=2.0)
    with pytest.raises(AssertionError, match="#17"):
        test_i_acceleration_spread_matches_base(s, _plant_base())


def test_plant_untrained_identity_head_is_red(tmp_path: Path) -> None:
    """An R=1.0 head without train_identity_rung must turn test_i red.

    This is cc170a756 reproduced exactly: the ladder is well-formed and every
    scalar agrees, but min_meaningful_timestep() returns 1 so the head is never
    drawn. Nothing in the gate could see it before 2026-08-28.
    """
    s = _planted(tmp_path, undersampling__train_identity_rung=False)
    with pytest.raises(AssertionError, match="#16"):
        test_i_acceleration_spread_matches_base(s, _plant_base())


def test_plant_identity_rung_without_its_gradient_path_is_red(tmp_path: Path) -> None:
    """A trained identity rung with no pre-DC term must turn test_i red.

    The shape the ten attention_enhancements arms would have shipped: the rung
    is drawn, and every loss that could grade it is a post-DC constant.
    """
    s = _planted(tmp_path, losses__reconstruction__lambda_pre_dc_kspace=0.0)
    with pytest.raises(AssertionError, match="#16"):
        test_i_acceleration_spread_matches_base(s, _plant_base())


def test_plant_divergent_ladders_within_a_sub_cohort_are_red() -> None:
    """Two ladders in one sub-cohort must turn the uniformity checker red.

    Planted on ``_assert_single_valued`` because a corpus-level scan has no
    file to mutate. ``test_i3`` asserts its own coverage (both attention groups
    present, 10 arms each), so this cannot pass by scanning an empty corpus.
    """
    divergent = {
        "attention_shootout": {
            (1.0, (1.0, 2.0), True): ["a.yaml"],
            (2.0, (2.0,), None): ["b.yaml"],
        }
    }
    with pytest.raises(AssertionError, match="#17"):
        _assert_single_valued(divergent, "mask ladders")
    _assert_single_valued({"attention_shootout": {(1.0, (1.0, 2.0), True): ["a.yaml"]}}, "mask ladders")


_EXPECTED_PATCH_HW = (256, 256)


def _patch_hw(settings: TrainingSettings) -> tuple[int, int]:
    ps = settings.data.sampling.patch_size
    return (int(ps[0]), int(ps[1]))


def test_k_patch_size_full_fov_256(settings: TrainingSettings) -> None:
    hw = _patch_hw(settings)
    assert hw == _EXPECTED_PATCH_HW, (
        f"#17/resolution: data.patch_size={hw} crops the reconstructed FOV; the k-space "
        f"cold-diffusion cohort is pinned at {_EXPECTED_PATCH_HW} so every head-to-head is "
        "resolution-controlled. A sub-256² patch truncates outer k-space (unlike the "
        "attention-grid budget of invariant J, which keeps output k-space at 256²)."
    )


# ── Dense-attention token budget (the 2026-06-26 OOM mitigation) ───────────────
# The KAN dual-domain block pools its dense image/cross branches to a
# round(sqrt(budget))² grid when H*W exceeds the budget, capping the O(N²) score
# tensor that is the deepest OOM frame. Pinned cohort-wide so a divergent budget
# can't smuggle a memory-vs-quality confound into the shootout / matrix Δ.
_DENSE_ATTN_TOKEN_BUDGET = 2304  # 48² pooled grid; was 64²=4096 (OOM'd 32 GB cards)


def _max_dense_attn_tokens(settings: TrainingSettings) -> int | None:
    """The KAN dual-domain dense-attention token budget, or None if the arm's
    attention is some other mechanism (no ``kan_dual_domain_kwargs``)."""
    kdd = (settings.model.model_kwargs or {}).get("kan_dual_domain_kwargs") or {}
    return kdd.get("max_dense_attn_tokens")


def test_j_dense_attn_token_budget_uniform(settings: TrainingSettings) -> None:
    budget = _max_dense_attn_tokens(settings)
    if budget is None:
        pytest.skip("arm does not use the KAN dual-domain dense-attention budget")
    assert budget == _DENSE_ATTN_TOKEN_BUDGET, (
        f"#17/perf: max_dense_attn_tokens={budget}; the kan-attention family is pinned at "
        f"{_DENSE_ATTN_TOKEN_BUDGET} (48² pooled grid) cohort-wide. A divergent budget changes "
        "the pooled attention resolution and confounds memory vs quality across the head-to-head."
    )


# ── KAN dual-domain ablations (the 2026-06-12 de-confound) ─────────────────────
# The ``ablations_kan_dual_domain/`` arms each vary EXACTLY ONE declared knob
# (disable_branches / gate_type / num_bands / dc_method / a single loss term) off
# the headline control ``experiment_11_kan_dual_domain.yaml``. They are out of the
# main parametrization (they deliberately vary an extra knob), but the SHARED
# cohort invariants must STILL hold — the 2026-06-12 audit found they had drifted
# apart on every shared invariant (sampling strategy, selection metric, EMA decay,
# centre fraction, multistep sampling, collapse gate), re-introducing the exact
# pre-fix DC-blob / negative-transfer recipe the base spent ~10 fixes removing.
# The distinctive knob is disjoint from these fields, so pinning them here keeps
# each ablation a clean one-knob delta off the headline.

_KAN_ABL_DIR = _COHORT_ROOT / "ablations_kan_dual_domain"
_KAN_ARMS = tracked_yamls(_KAN_ABL_DIR, recursive=False)
_KAN_IDS = [p.name for p in _KAN_ARMS]


@pytest.fixture(scope="module", params=_KAN_ARMS, ids=_KAN_IDS)
def kan_settings(request: pytest.FixtureRequest) -> TrainingSettings:
    return TrainingSettings.from_yaml(str(request.param))


def test_kan_ablations_discovered() -> None:
    skip_if_public_export("experiments/ does not ship; the KAN ablation directory is absent here")
    assert len(_KAN_ARMS) == 11, f"expected 11 KAN ablations, found {len(_KAN_ARMS)}"


def test_kan_ablation_shares_cohort_invariants(kan_settings: TrainingSettings) -> None:
    """The shared invariants (A/B/C/D/F/I) must match the headline — an ablation
    differs from its control by exactly its one declared knob, never by the
    selection metric / sampler / EMA / collapse-gate recipe."""
    s = kan_settings
    assert (
        s.validation.sampling.enable_multistep_cold is True
    ), "A/#16: single forward blurs at heavy R"
    if s.ema.enabled:
        assert s.ema.decay <= 0.999 + 1e-12, f"B/#20: ema.decay={s.ema.decay} too high"
    assert (
        s.metrics.best_metric_name == "val_robust_mri_psnr_mean"
    ), f"C/#18: best_metric_name={s.metrics.best_metric_name!r}; select on the cascade mean."
    assert (
        s.early_stopping.metric == "val_robust_mri_psnr_mean"
    ), "C/#18: early-stop metric drift"
    assert "complex_l1" in _loss_names(
        s.losses, "kspace_losses"
    ), "D/#20: k-space fidelity dropped"
    assert (
        s.validation.gates.input_dependence_tol is not None
    ), "F/#20: collapse gate disabled"
    assert (
        _attr(s.training, "timestep_sampling_strategy") == "balanced_high_t"
    ), "I/#17: sampler drifted off 'balanced_high_t' (superseded high_t_emphasis)."
    # Literal here, unlike test_i: these 11 ablations are compared against ONE
    # control (experiment_11_kan_dual_domain.yaml), which carries (4, 0.005), so
    # the ablation set's own base value IS the invariant. Verified 2026-08-28:
    # all 11 sit at (4, 0.005, timesteps 28).
    assert (
        _attr(s.training, "curriculum_start_timestep") == 4
    ), "I/#17: curriculum_start drift vs the kan_dual_domain control"
    assert (
        _attr(s.training, "curriculum_ramp_rate") == 0.005
    ), "I/#17: curriculum_ramp_rate drift vs the kan_dual_domain control"
    assert (
        int(s.data.sampling.patch_size[0]),
        int(s.data.sampling.patch_size[1]),
    ) == _EXPECTED_PATCH_HW, (
        f"K/#17: patch_size={s.data.sampling.patch_size} crops the reconstructed FOV; pin at "
        f"{_EXPECTED_PATCH_HW} so the ablation Δ isn't confounded by resolution."
    )


def test_kan_ablation_declares_falsifiable_claim(
    kan_settings: TrainingSettings,
) -> None:
    assert _meta(
        kan_settings, "hypothesis"
    ), "#19: metadata.hypothesis missing (the tell)."
    assert _meta(
        kan_settings, "primary_metric"
    ), "#18: metadata.primary_metric missing."
    assert _meta(
        kan_settings, "baseline"
    ), "#17: ablation must name its control via metadata.baseline."


# Losses that carry their own DifferentiableFourierBridge, so a bridged list would
# transform twice. LossBuilder detects them structurally (the ``use_fourier_bridge``
# attribute); this name list is the config-side mirror.
_SELF_BRIDGING_LOSSES = frozenset(
    {
        "complex_spatial_gradient",
        "sense_adjoint_l1",
        "rician_consistency",
        "background_suppression",
    }
)

# Below this budget an arm is a quick diagnostic, not a head-to-head (invariant M).
_MIN_BUDGET_FOR_SELECTION = 10_000
_MIN_VALIDATION_POINTS = 6


def test_l_no_double_fourier_bridge(settings: TrainingSettings) -> None:
    if settings.losses.policy.output_domain != "kspace":
        pytest.skip("the iFFT bridge is only inserted for a k-space output domain")
    for section in ("image_losses", "complex_losses"):
        offenders = sorted(
            _SELF_BRIDGING_LOSSES.intersection(_loss_names(settings.losses, section))
        )
        assert not offenders, (
            f"L/#467: {offenders} bridge from k-space internally but are declared under "
            f"losses.{section}, which LossBuilder wraps in a second iFFT bridge for "
            f"output_domain='kspace' -- the tensor is inverse-transformed twice and its "
            f"coil channels re-paired as (real, imag). Declare them under "
            f"losses.kspace_losses, or set kwargs: {{use_fourier_bridge: false}} on the "
            f"entry (kwargs:, NOT config: -- see issue #468)."
        )


def test_m_selection_density_and_fixed_budget(settings: TrainingSettings) -> None:
    budget = settings.training.max_iterations
    interval = settings.validation.schedule.interval_steps
    if not budget or not interval:
        pytest.skip("arm does not pin both an iteration budget and an eval interval")
    if budget < _MIN_BUDGET_FOR_SELECTION:
        pytest.skip(f"quick-diagnostic budget ({budget} iters), not a head-to-head")

    points = budget // interval
    assert points >= _MIN_VALIDATION_POINTS, (
        f"M/#18: eval_interval={interval} over max_iterations={budget} gives {points} "
        f"validation points. val_robust_mri_psnr_mean is stamped only inside validation, "
        f"so best-checkpoint selection would choose among {points} candidates and the "
        f"arm's curve would be {points} samples."
    )
    assert settings.early_stopping.enabled is True, (
        "M/#18: the best-checkpoint writer is gated on the early-stopping service "
        "(training_loop.py:1171 wraps :1247), so enabled must stay True even when the "
        "cohort does not want plateau-stopping."
    )
    # Exact mechanism, not a floor-division proxy: one validation event sets
    # ``best``, then ``patience`` consecutive non-improving events fire the stop,
    # so the earliest possible stop is ``(patience + 1) * interval_steps``.
    #
    # Until 2026-08-28 this asserted ``patience > points``, one stricter than the
    # mechanism requires -- at ``patience == points`` a stop still needs
    # ``points + 1`` events, already past the budget. The off-by-one is why a
    # 70000/5000 arm carrying the perfectly sufficient ``patience: 14`` was
    # reported as able to truncate.
    _patience = settings.early_stopping.patience
    earliest_stop = (_patience + 1) * interval
    assert earliest_stop > budget, (
        f"M/#18: patience={_patience} counts validation EVENTS, not iterations. The "
        f"earliest possible stop is (patience + 1) * interval_steps = {earliest_stop}, "
        f"which must exceed max_iterations={budget} so a fixed-budget head-to-head "
        f"cannot truncate an arm ({points} validation events fit in this budget)."
    )


@pytest.mark.parametrize("arm", _ARMS, ids=_IDS)
def test_n_ema_warmup_pinned_explicitly(arm: Path) -> None:
    """Read the raw YAML: a validated ``ema.warmup`` is True either way."""
    raw = yaml.safe_load(arm.read_text()) or {}
    ema = raw.get("ema") or {}
    if not ema.get("enabled"):
        pytest.skip("EMA disabled on this arm")
    assert "warmup" in ema, (
        f"N/#17: {arm.name} leaves ema.warmup to the schema default. The default is True "
        f"so behaviour matches, but invariant B covers ema.decay only -- pin it so the "
        f"cohort control cannot drift from the siblings that name it as their baseline."
    )
    assert ema["warmup"] is True, (
        f"N/#17: {arm.name} disables ema.warmup; on a diagnostic-length run the shadow "
        f"validation grades then lags the live model (see invariant B). Arms that test "
        f"the flag belong under ablations*/."
    )


# ---------------------------------------------------------------------------
# Data consistency: hard is the standard, and the exemptions are enumerated.
# ---------------------------------------------------------------------------

# DC is the INDEPENDENT VARIABLE on these arms. Flipping them to hard would
# delete the experiment that measures the very claim hard rests on, so the set is
# spelled out rather than inferred -- a new arm quietly inheriting `soft` must
# fail, not join an open-ended allowlist.
#
# 11 entries, 10 of which are non-hard: `experiment_11_dc_hard` is exempt AND runs
# hard, because it is the shootout's hard control. Exempt means "this arm's method
# is a deliberate experimental choice", not "this arm is not hard".
_DC_IS_THE_VARIABLE: dict[str, str] = {
    "experiment_11_dc_adaptive.yaml": "dc_shootout: adaptive",
    "experiment_11_dc_hard.yaml": "dc_shootout: hard control",
    "experiment_11_dc_kan_adaptive.yaml": "dc_shootout: kan_adaptive",
    "experiment_11_dc_noise_adaptive.yaml": "dc_shootout: noise_adaptive",
    "experiment_11_dc_noise_adjusted.yaml": "dc_shootout: noise_adjusted",
    "experiment_11_dc_soft.yaml": "dc_shootout: soft control",
    "experiment_11_dc_target_aware_fsdc.yaml": "dc_shootout: target_aware_fsdc",
    "experiment_11_kan_ablation_cnn_adc.yaml": "ablates KAN ADC -> legacy CNN ADC",
    "experiment_11_kan_ablation_legacy_dual_domain.yaml": "legacy block + CNN ADC",
    "experiment_130_ti_ccd.yaml": "named target_aware_fsdc; divergence tracked in #611",
    "experiment_cross_contrast_kspace_diffusion.yaml": "named adaptive@0.25; #611",
}

_ALL_ARMS = tracked_yamls(_COHORT_ROOT)


@pytest.mark.parametrize("arm", _ALL_ARMS, ids=lambda p: p.name)
def test_data_consistency_is_hard_unless_dc_is_the_variable(arm: Path) -> None:
    """`soft` at w=0.5 discards 2/3 of the MEASURED k-space.

    ``SimpleDataConsistency`` returns, at sampled lines, ``mask * measured`` under
    ``hard`` and ``mask * (pred + w*measured)/(1 + w)`` under ``soft``. At w=0.5
    that is 2/3 prediction + 1/3 measurement — applied to the only high-frequency
    content the arm does not have to infer. Measured with the prediction's own
    high-band amplitude ratio (sqrt(0.078) = 0.279): 0.270x of the target's
    high-band power at the measured lines, against 1.000x under hard.

    A converged predictor is still a fixed point of soft DC, so this is not a cap
    on a perfect model — it is a 1/3 shrink of the correction toward the
    measurement at the final reverse step.
    """
    settings = TrainingSettings.from_yaml(str(arm))
    method = settings.physics.data_consistency.method
    resolved = str(getattr(method, "value", method))

    if arm.name in _DC_IS_THE_VARIABLE:
        pytest.skip(f"DC is this arm's variable ({_DC_IS_THE_VARIABLE[arm.name]})")

    assert resolved == "hard", (
        f"{arm.name}: dc_method={resolved!r}. hard is the cohort standard — it keeps "
        "the measured lines exact. If DC is genuinely this arm's independent "
        "variable, add it to _DC_IS_THE_VARIABLE with the reason."
    )


@pytest.mark.parametrize("arm", _ALL_ARMS, ids=lambda p: p.name)
def test_physics_and_model_dc_settings_agree(arm: Path) -> None:
    """Both declarations are merged into ONE layer; a mismatch is not survivable.

    ``physics.data_consistency`` is the SSOT: ``generator_kwargs`` step 3c
    forwards every row of :data:`DC_SSOT_KEYS` from it, and raises when
    ``model_kwargs`` declares the same fact differently. ``dc_method`` alone is
    reconciled silently instead -- ``model_builder._reconcile`` overwrites it,
    so a stale model-side value is not a conflict the reader can see; the YAML
    simply documents an operator the run never builds (pitfall #16).

    This iterates the tuple rather than naming a key, because the version that
    checked ``dc_method`` ALONE was blind to the five other rows for as long as
    they existed -- and `experiment_11_attention_none` sat unbuildable behind
    that blind spot, its ``physics.weight`` drifted to 1.0 against a
    ``model_kwargs.dc_weight`` of 0.5, while this guard stayed green. Naming the
    keys here would make the guard a second, unsynced owner of the mapping
    (non-negotiable 17); the noise rows joined ``DC_SSOT_KEYS`` in #1525 and no
    enumeration followed them.

    Note the two spellings differ across the blocks (``dc_weight`` <->
    ``weight``), so neither a key-name grep nor a same-name dict intersection
    can see these divergences. ``DC_SSOT_KEYS`` is the only place the pairing
    is written down.
    """
    settings = TrainingSettings.from_yaml(str(arm))
    dc = settings.physics.data_consistency
    declared_kwargs = settings.model.model_kwargs or {}

    for kwarg, field in DC_SSOT_KEYS:
        if not hasattr(dc, field):
            continue
        ssot = getattr(dc, field)
        ssot = str(getattr(ssot, "value", ssot)) if field == "method" else ssot

        if kwarg == "dc_method":
            # Presence is required for dc_method and only dc_method: every arm
            # in this cohort declares it, and dropping the ratchet to
            # "agree if present" would silently accept its removal.
            declared = declared_kwargs.get(kwarg)
            assert declared == ssot, (
                f"{arm.name}: model_kwargs.dc_method={declared!r} but "
                f"physics.data_consistency.method={ssot!r}."
            )
            continue

        if kwarg not in declared_kwargs:
            # Absent is the SSOT-clean state -- physics supplies the value.
            continue
        declared = declared_kwargs[kwarg]
        assert declared == ssot, (
            f"{arm.name}: model_kwargs.{kwarg}={declared!r} but "
            f"physics.data_consistency.{field}={ssot!r}. These describe the "
            "same fact; physics.data_consistency is the SSOT, so this arm "
            "raises at generator construction. Remove the model_kwargs copy "
            "once you have decided which value is the intended one."
        )


def test_the_cross_contrast_exemption_is_not_a_blanket_pass() -> None:
    """The exemption covers COMPARABILITY only, and must stay that way.

    Issue #687 offered two remedies for these arms: bring them onto the cohort
    recipe, or declare the divergence deliberate. Declaring it is the reversible
    half — but applied to every invariant it would hide four genuine defects on
    ``experiment_130_ti_ccd`` (no k-space data fidelity at all, no
    input-dependence gate, an EMA time constant 6.7x its own iteration budget,
    no falsifiable claim). Those are properties of one arm, not of the
    comparison, so the exemption must not reach them.

    Pinned by counting the call sites rather than trusting the convention.
    """
    import ast

    tree = ast.parse(Path(__file__).read_text())
    gated = sorted(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "_skip_if_cross_contrast"
            for c in ast.walk(node)
        )
    )
    # Counted by AST, not substring: a source-grep test whose own body contains
    # the needle matches itself, which is how this assertion first failed.
    assert gated == [
        "test_c_selection_metric_is_cascade_mean",
        "test_i_acceleration_spread_matches_base",
    ], (
        f"invariants gated on the cross-contrast exemption: {gated}. Only the "
        "two comparability ones (C selection metric, I acceleration spread) may "
        "be. Gating a third means a coherence check has been silenced — fix the "
        "arm instead, or argue the case in #687."
    )


def test_every_exempted_arm_exists() -> None:
    """Anti-rot: an exemption for a deleted or renamed arm guards nothing."""
    skip_if_public_export("experiments/ does not ship, so every exemption reads as naming a deleted arm")
    present = {p.name for p in tracked_yamls(_COHORT_ROOT)}
    missing = sorted((set(_CROSS_CONTRAST_ARMS) | set(_INERT_LADDER_ARMS)) - present)
    assert not missing, f"exemption names an arm that no longer exists: {missing}"
