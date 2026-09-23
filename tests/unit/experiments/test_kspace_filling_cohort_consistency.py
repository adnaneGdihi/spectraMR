"""Cohort-consistency guard for the experiment_11 k-space cold-diffusion arms.

Regression fence for the 2026-07-01 stale-ablation incident: 33 arms under
``experiments/inprogress/kspace_filling/`` declared their fidelity losses only in
the ``kspace_losses`` / ``image_losses`` list form (which feeds only the
LossBuilder enable-gate), leaving the COMPUTE-time ``losses.<section>.lambda_<name>``
weights at their 0.0 default. Every k-space fidelity term therefore computed at
0.0, and the models collapsed to a measurement-independent DC blob (pitfall
#16/#20 — the L4 input-dependence gate fired ``val_measurement_collapse=1.0``).

This test loads every ``kspace_cold_diffusion`` arm in the cohort and asserts:

1. **No inert fidelity loss** — every enabled loss whose weight is resolved via a
   ``lambda_<name>`` field computes at a NONZERO weight (the facade cure). The
   resolution mirrors ``BaseLossComputer._get_loss_weight`` (the compute-time
   path the cold-diffusion loss computer actually uses).
2. **target_channels == out_channels** — the (self-consistency) invariant reconciled
   across the cohort.

Deliberately scoped to this cohort (a path-based sweep), not a repo-wide audit
check — the facade is specific to the k-space cold-diffusion loss computer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from spectramr.config.schemas.loss import LossConfigSchema
from spectramr.models.losses.weights import build_loss_weight_table
from tests.utils.corpus import tracked_yamls
from tests.utils.repo_scripts import skip_if_public_export

_COHORT = Path(__file__).resolve().parents[3] / "experiments" / "inprogress" / "kspace_filling"

# The section-order table that used to live here is gone. It was a copy of
# ``_get_loss_weight``'s search order, "kept in sync so the test resolves the
# compute weight the way training does" -- the exact private-second-resolver
# that pitfall #13b forbids, and it had already fallen out of sync (see
# ``_resolved_compute_weight``). ``models/losses/weights`` is the one resolver.


def _resolved_compute_weight(losses_cfg: object, name: str) -> float | None:
    """Compute-time weight for ``name``, from the production resolver.

    This walked the ``lambda_<name>`` schema fields itself, which is the second
    resolver CLAUDE.md pitfall #13b exists to forbid — and it disagreed with the
    real one. ``lambda_hfen`` is a declared field defaulting to ``0.0``, so the
    walk reported "computes 0.0" for every arm that weights ``hfen`` through a
    ``image_losses[].weight`` list entry instead. That is 11 of 58 arms, all
    correctly configured, failing a facade check as facades:

        >>> resolve_loss_weight(build_loss_weight_table(s.losses), "hfen")
        0.3    # source='losses.image_losses[hfen].weight'

    The list entry wins in the production table. Asking it removes the
    disagreement rather than adjudicating it here — a facade check built on a
    private copy of the weight rules cannot be more right than the rules.

    Read the spec's STATIC weight, not ``resolve_loss_weight``. That resolver
    defaults to ``iteration=0``, where a warm-up-gated term correctly returns
    0.0 — deferred, not inert. Calling it here turned 53 arms red on
    ``complex_spatial_gradient``, which is warm-up gated by design. The facade
    question is "does this term ever contribute", and the static weight is what
    answers it.
    """
    table = build_loss_weight_table(losses_cfg)
    spec = table.get(name) if hasattr(table, "get") else None
    if spec is None:  # declared nowhere -> not lambda-resolved; check N/A
        return None
    if not getattr(spec, "enabled", True):
        return None  # explicitly disabled is a decision, not a facade
    return float(spec.weight)


def _cold_diffusion_arms() -> list[Path]:
    arms: list[Path] = []
    for f in tracked_yamls(_COHORT):
        try:
            cfg = yaml.safe_load(f.read_text()) or {}
        except Exception:  # pragma: no cover — malformed YAML caught elsewhere
            continue
        model_type = str((cfg.get("model") or {}).get("model_type", "")).lower()
        if "kspace_cold_diffusion" in model_type:
            arms.append(f)
    return arms


def _arms_with_data_block() -> list[Path]:
    """Every cohort YAML that declares a ``data`` block (superset of the
    cold-diffusion arms — the NEX invariant is data-side, not model-specific)."""
    arms: list[Path] = []
    for f in tracked_yamls(_COHORT):
        try:
            cfg = yaml.safe_load(f.read_text()) or {}
        except Exception:  # pragma: no cover — malformed YAML caught elsewhere
            continue
        if isinstance(cfg.get("data"), dict):
            arms.append(f)
    return arms


_ARMS = _cold_diffusion_arms()
_DATA_ARMS = _arms_with_data_block()


def _arm_id(p: Path) -> str:
    return str(p.relative_to(_COHORT))


@pytest.mark.skipif(not _ARMS, reason="kspace_filling cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_no_inert_fidelity_loss(arm: Path) -> None:
    """Every enabled, lambda-resolved loss must compute at a nonzero weight."""
    cfg = yaml.safe_load(arm.read_text()) or {}
    losses_block = cfg.get("losses")
    assert isinstance(losses_block, dict), f"{_arm_id(arm)}: no losses block"
    losses = LossConfigSchema(**losses_block)

    inert: list[str] = []
    for attr in ("kspace_losses", "image_losses", "complex_losses"):
        for entry in getattr(losses, attr, None) or []:
            if getattr(entry, "enabled", True) is False:
                continue
            name = getattr(entry, "name", None)
            weight = getattr(entry, "weight", None)
            if not name or weight is None or float(weight) <= 0.0:
                continue
            resolved = _resolved_compute_weight(losses, str(name))
            # ``None`` -> weight is not lambda-resolved (delivered elsewhere); only a
            # lambda-resolved loss that resolves to 0.0 is the inert-facade class.
            if resolved is not None and resolved <= 0.0:
                inert.append(f"{name} (list weight {weight}, computes 0.0)")

    assert not inert, (
        f"{_arm_id(arm)}: declared fidelity loss(es) are inert (compute at 0.0) — "
        f"set losses.<section>.lambda_<name>: {inert}"
    )


@pytest.mark.skipif(not _ARMS, reason="kspace_filling cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_target_channels_match_out_channels(arm: Path) -> None:
    """data.target_channels must equal model.out_channels (self-consistency)."""
    cfg = yaml.safe_load(arm.read_text()) or {}
    out_ch = (cfg.get("model") or {}).get("out_channels")
    tgt_ch = (cfg.get("data") or {}).get("target_channels")
    if out_ch is None or tgt_ch is None:
        pytest.skip("out_channels or target_channels not declared")
    assert tgt_ch == out_ch, (
        f"{_arm_id(arm)}: data.target_channels={tgt_ch} != model.out_channels={out_ch}"
    )


# The NEX-reference route guard that lived here (2026-07-06) is now the
# corpus-wide ``tests/unit/experiments/test_nex_reference_route_corpus.py``,
# driving the ``nex_reference_route`` witness's predicate over every tracked
# arm -- one owner (non-negotiable 17); 79 arms outside this cohort carried
# the same defect the cohort-scoped copy could not see.


# ---------------------------------------------------------------------------
# 2026-07-01 second-order remediation fences — scientific-validity + honesty
# invariants for the ABLATION families (beyond the loss-facade cure above).
# Scoped to attention_shootout / attention_enhancements / ablations_kan_dual_domain:
# the pattern-variant arms (gaussian/radial/...) and the alt-architecture arms
# (swin/varnet/nafnet) legitimately fix the mask / have no sibling control, so a
# cohort-wide assertion would false-fail on them.
# ---------------------------------------------------------------------------

_ABLATION_FAMILIES = (
    "attention_shootout",
    "attention_enhancements",
    "ablations_kan_dual_domain",
)


def _in_families(p: Path, families: tuple[str, ...]) -> bool:
    return any(f"/{fam}/" in str(p) for fam in families)


_FAMILY_ARMS = [a for a in _ARMS if _in_families(a, _ABLATION_FAMILIES)]
_KAN_ABLATION_ARMS = [a for a in _ARMS if _in_families(a, ("ablations_kan_dual_domain",))]


@pytest.mark.skipif(not _FAMILY_ARMS, reason="ablation-family arms not present")
@pytest.mark.parametrize("arm", _FAMILY_ARMS, ids=[_arm_id(a) for a in _FAMILY_ARMS])
def test_ablation_family_uses_dynamic_mask(arm: Path) -> None:
    """Attention/KAN ablation arms must use a dynamic mask so the R=[2,8,32]
    input-dependence gate sees diverse accelerations. A static mask is a
    confound (the no_csg regression). Pattern-variant / alt-arch arms outside
    these families legitimately fix the mask and are not swept here."""
    # Resolved, not raw YAML. This read ``.get("acceleration") or {}``, and the
    # 2026-08-02 drain renamed that block to ``undersampling``. The fallback then
    # made ``.get("enable_dynamic_mask")`` return None for every arm, and
    # ``None is not False`` is True -- the guard passed on all 33 arms while
    # checking nothing, and nothing reported it. A negative assertion over a
    # defensively-defaulted dict cannot fail once its block moves.
    from spectramr.config.settings import TrainingSettings

    settings = TrainingSettings.from_yaml(str(arm))
    assert settings.undersampling.enable_dynamic_mask is not False, (
        f"{_arm_id(arm)}: undersampling.enable_dynamic_mask is False — a static mask "
        f"confounds the input-dependence probe vs the dynamic-mask siblings."
    )


@pytest.mark.skipif(not _FAMILY_ARMS, reason="ablation-family arms not present")
@pytest.mark.parametrize("arm", _FAMILY_ARMS, ids=[_arm_id(a) for a in _FAMILY_ARMS])
def test_ablation_arm_declares_baseline(arm: Path) -> None:
    """Every ablation-family arm must declare a control for one-knob attribution
    (metadata.baseline or metadata.tags.baseline), or BE the baseline
    (metadata.role == 'baseline')."""
    md = (yaml.safe_load(arm.read_text()) or {}).get("metadata") or {}
    has_baseline = (
        "baseline" in md or "baseline" in (md.get("tags") or {}) or md.get("role") == "baseline"
    )
    assert has_baseline, (
        f"{_arm_id(arm)}: no metadata.baseline / tags.baseline / role:baseline — "
        f"an ablation delta cannot be attributed to a single knob."
    )


@pytest.mark.skipif(not _KAN_ABLATION_ARMS, reason="kan-ablation arms not present")
@pytest.mark.parametrize("arm", _KAN_ABLATION_ARMS, ids=[_arm_id(a) for a in _KAN_ABLATION_ARMS])
def test_kan_ablation_has_coil_estimation(arm: Path) -> None:
    """The ablations_kan_dual_domain family shares baseline kan_dual_domain, which
    sets physics.coil_processing.estimation (power_iter, kernel_size 12). Each
    ablation must carry it, else it silently falls back to kernel_size 6 while the
    baseline uses 12 — a family-wide smap confound (pitfall #17)."""
    phys = (yaml.safe_load(arm.read_text()) or {}).get("physics") or {}
    est = ((phys.get("coil_processing") or {}).get("estimation")) or {}
    assert est.get("method"), (
        f"{_arm_id(arm)}: missing physics.coil_processing.estimation — smap config "
        f"diverges from baseline experiment_11_kan_dual_domain (confound)."
    )


@pytest.mark.skipif(not _KAN_ABLATION_ARMS, reason="kan-ablation arms not present")
@pytest.mark.parametrize("arm", _KAN_ABLATION_ARMS, ids=[_arm_id(a) for a in _KAN_ABLATION_ARMS])
def test_kan_ablation_no_stray_pre_dc(arm: Path) -> None:
    """No kan-ablation arm may set lambda_pre_dc_kspace: the KAN family drops it
    (unlike the base SSOT), so a lone arm carrying it (the no_csg regression) is a
    second, unintended loss delta vs the shared baseline."""
    losses = (yaml.safe_load(arm.read_text()) or {}).get("losses") or {}
    assert "lambda_pre_dc_kspace" not in json.dumps(losses), (
        f"{_arm_id(arm)}: carries lambda_pre_dc_kspace — a stray loss delta vs the "
        f"KAN family (which drops it); confounds the arm's single-knob ablation."
    )


# ---------------------------------------------------------------------------
# 2026-07-15 ESPIRiT rank-viability fence.
#
# ``estimate_csm_espirit`` builds a block-Hankel calibration matrix with
# ``(acs-k+1)^2`` rows against ``k^2*coils`` unknowns and RAISES when the system
# is rank-deficient (``n_patches < unknowns``) rather than emit a silent bad map
# (#9/#16). The attention_shootout arms switched ``power_iter -> espirit`` (the
# 2026-07-09 SNR easy-win) but carried power_iter's ``kernel_size: 12`` (only a
# smoothing-window there), which at ``acs=24`` / 4-coil gives ``169 < 576`` -> a
# hard crash at training iter 1. This fence asserts every espirit arm is
# over-determined. ``power_iter`` arms are exempt: that estimator reads only
# ``kernel_size`` (never ``acs_size``) and builds no Hankel matrix, so it carries
# no rank constraint. 2026-07-15: the cohort reverted fully to power_iter, so
# this fence is dormant by design (0 espirit arms -> it skips) and now guards
# against a future espirit reintroduction landing a rank-deficient kernel.
# ---------------------------------------------------------------------------

# estimate_smaps defaults (src/spectramr/infrastructure/physics/coil_sensitivity.py).
_ESPIRIT_DEFAULT_KERNEL = 6
_ESPIRIT_DEFAULT_ACS = 24


def _espirit_arms() -> list[Path]:
    arms: list[Path] = []
    for f in tracked_yamls(_COHORT):
        try:
            cfg = yaml.safe_load(f.read_text()) or {}
        except Exception:  # pragma: no cover - malformed YAML caught elsewhere
            continue
        est = (((cfg.get("physics") or {}).get("coil_processing") or {}).get("estimation")) or {}
        if str(est.get("method", "")).lower() == "espirit":
            arms.append(f)
    return arms


_ESPIRIT_ARMS = _espirit_arms()


def _acs_hw(acs_size: int | list[int] | tuple[int, ...]) -> tuple[int, int]:
    """Mirror estimate_csm_espirit: int -> square region, (h, w) -> rectangle."""
    if isinstance(acs_size, (list, tuple)):
        return int(acs_size[0]), int(acs_size[1])
    return int(acs_size), int(acs_size)


@pytest.mark.skipif(not _ESPIRIT_ARMS, reason="no espirit arms in cohort")
@pytest.mark.parametrize("arm", _ESPIRIT_ARMS, ids=[_arm_id(a) for a in _ESPIRIT_ARMS])
def test_espirit_calibration_not_rank_deficient(arm: Path) -> None:
    """Every ``method: espirit`` arm must over-determine the ESPIRiT calibration:
    ``(acs-k+1)^2 >= k^2 * coils``, mirroring the raise in estimate_csm_espirit.
    A ``kernel_size: 12`` carried over from the power_iter tuning is the
    rank-deficient class that crashed the shootout at iter 1."""
    cfg = yaml.safe_load(arm.read_text()) or {}
    coil = (cfg.get("physics") or {}).get("coil_processing") or {}
    est = coil.get("estimation") or {}
    kernel = int(est.get("kernel_size", _ESPIRIT_DEFAULT_KERNEL))
    acs_h, acs_w = _acs_hw(est.get("acs_size", _ESPIRIT_DEFAULT_ACS))

    # ESPIRiT sees the coil axis of the ACS k-space. The espirit arms all disable
    # coil compression (compression.method: none), so that axis is the physical
    # coil count; with compression on, the count feeding ESPIRiT is not statically
    # knowable here, so skip rather than assert on a guessed value.
    compression = str((coil.get("compression") or {}).get("method", "none")).lower()
    if compression not in ("", "none"):
        pytest.skip("coil compression active - ESPIRiT input coil count not static")
    coils = ((cfg.get("model") or {}).get("model_kwargs") or {}).get("num_physical_coils")
    if coils is None:
        pytest.skip("arm does not declare num_physical_coils")
    coils = int(coils)

    n_patches = (acs_h - kernel + 1) * (acs_w - kernel + 1)
    unknowns = kernel * kernel * coils
    assert n_patches >= unknowns, (
        f"{_arm_id(arm)}: ESPIRiT rank-deficient - {n_patches} ACS patches "
        f"({acs_h}x{acs_w}, kernel {kernel}) < {unknowns} unknowns "
        f"(k^2*coils = {kernel}^2*{coils}); estimate_csm_espirit raises at train "
        f"iter 1. Reduce kernel_size (6 is the ESPIRiT default) or raise acs_size."
    )


def _arms_with_estimation() -> list[Path]:
    """Cohort YAMLs that declare an active ``coil_processing.estimation`` block."""
    arms: list[Path] = []
    for f in tracked_yamls(_COHORT):
        try:
            cfg = yaml.safe_load(f.read_text()) or {}
        except Exception:  # pragma: no cover - malformed YAML caught elsewhere
            continue
        est = ((cfg.get("physics") or {}).get("coil_processing") or {}).get("estimation")
        if isinstance(est, dict) and est.get("method") is not None:
            arms.append(f)
    return arms


_ESTIMATION_ARMS = _arms_with_estimation()


# ---------------------------------------------------------------------------
# 2026-07-15 cohort smap-method uniformity.
#
# The exp_11 kspace_filling cohort estimates coil-sensitivity maps with
# ``power_iter`` (kernel_size 12, acs_size 24) across every arm. Uniformity is
# load-bearing: the maps set the SENSE-combine that BOTH the training loss and
# the graded image use, so a divergent estimator makes absolute PSNR/SSIM
# non-comparable across arms (pitfall #17). It also keeps the cohort clear of
# ESPIRiT's Hankel rank constraint (the fence above), which crashed the shootout
# arms at iter 1 before the switch. Relaxing this to admit a non-power_iter arm
# is a deliberate, comparability-affecting decision - update this fence in the
# same change; the espirit fence above then guards that arm's rank viability.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ESTIMATION_ARMS, reason="kspace_filling cohort not present")
@pytest.mark.parametrize("arm", _ESTIMATION_ARMS, ids=[_arm_id(a) for a in _ESTIMATION_ARMS])
def test_cohort_smap_method_uniform_power_iter(arm: Path) -> None:
    """Every arm's live smap estimator is ``power_iter``.

    ``physics.coil_processing.estimation.method`` is the SSOT the diffusion
    strategy reads (``_configured_estimation_method``); the legacy
    ``physics.coil_sensitivity.estimation_method`` is intentionally NOT read, so
    only the coil_processing method is asserted here.
    """
    cfg = yaml.safe_load(arm.read_text()) or {}
    est = (((cfg.get("physics") or {}).get("coil_processing") or {}).get("estimation")) or {}
    method = str(est.get("method", "")).lower()
    assert method == "power_iter", (
        f"{_arm_id(arm)}: coil_processing.estimation.method={method!r}, expected "
        f"'power_iter' (cohort standard as of 2026-07-15). A non-power_iter method "
        f"changes the SENSE-combine reference and breaks cross-arm metric "
        f"comparability; switch back or update this fence deliberately."
    )


def test_no_cohort_arm_is_hidden_from_git() -> None:
    """Every arm on disk must be visible to git — none may be gitignored.

    A `.gitignore` rule aimed at KAN *training output* (`**/*_kan_*/`, sibling to
    `model_kanu_*/`, `output_kanu_*/`, `loss_kanu_*/`) also matched this cohort's
    SOURCE config directory `ablations_kan_dual_domain/`. Its 11 arms survived only
    because they were tracked BEFORE the rule existed — gitignore does not affect
    already-tracked files. A NEW arm added there was invisible to `git status` and
    refused by `git add` without `-f`, so it would never reach the cluster, never
    be migrated by the standing schema rule (which walks tracked files), and never
    be audited. It would simply not exist for anyone but its author.

    This asserts the property that actually matters — on-disk == tracked — rather
    than the specific rule, so any future ignore pattern that swallows a cohort
    directory fails here regardless of how it is spelled.
    """
    skip_if_public_export("experiments/ does not ship; the cohort is absent, not hidden")
    import subprocess

    on_disk = {p.relative_to(_COHORT.parents[2]) for p in _COHORT.rglob("*.yaml")}
    tracked = {
        Path(line)
        for line in subprocess.run(
            ["git", "ls-files", "experiments/inprogress/kspace_filling"],
            cwd=_COHORT.parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        if line.endswith(".yaml")
    }
    assert on_disk, "no arms found on disk — the cohort path is wrong"
    hidden = sorted(str(p) for p in on_disk - tracked)
    assert not hidden, (
        "these cohort arms exist on disk but are NOT tracked by git, so they are "
        "invisible to the migration rule, the audit and the cluster:\n  " + "\n  ".join(hidden)
    )


def test_no_cohort_directory_is_covered_by_a_gitignore_rule() -> None:
    """A NEW arm must be addable to every cohort directory.

    The companion test above asserts on-disk == tracked, which the
    `**/*_kan_*/` incident passed vacuously: gitignore does not affect files that
    are already tracked, so the 11 pre-existing arms under
    `ablations_kan_dual_domain/` looked fine while the directory was closed to
    every future one. This probes the cause instead of the symptom by asking git
    what it would do with a path that does not exist yet — no file is created.
    """
    import subprocess

    root = _COHORT.parents[2]
    dirs = sorted({_COHORT, *(p for p in _COHORT.rglob("*") if p.is_dir())})
    probes = [str((d / "__would_a_new_arm_land_here.yaml").relative_to(root)) for d in dirs]
    # `check-ignore` exits 0 when it matched something, 1 when nothing matched.
    proc = subprocess.run(
        ["git", "check-ignore", "--verbose", "--no-index", "--stdin"],
        cwd=root,
        input="\n".join(probes),
        capture_output=True,
        text=True,
    )
    assert dirs, "no cohort directories found — the cohort path is wrong"
    assert proc.returncode == 1, (
        "a .gitignore rule covers a cohort SOURCE directory, so a new arm placed "
        "there would be invisible to `git status` and refused by `git add`:\n  "
        + "\n  ".join(proc.stdout.strip().splitlines())
        + "\n\nRe-include the directory itself (`!experiments/inprogress/**/<pat>/`) "
        "-- negating the files alone cannot work, because git will not re-include "
        "a path whose parent DIRECTORY is excluded."
    )


# ── DeepSpeed wiring ─────────────────────────────────────────────────────────
# The cohort is sharded with ZeRO-2. Sharding must not change what is optimised
# (the premise of the b3 reference arm), so what these guard is not "DeepSpeed is
# declared" but the three ways declaring it silently changes the experiment.


@pytest.mark.skipif(not _DATA_ARMS, reason="kspace_filling cohort not present")
@pytest.mark.parametrize("arm", _DATA_ARMS, ids=[_arm_id(a) for a in _DATA_ARMS])
def test_arm_declares_deepspeed_zero2(arm: Path) -> None:
    """The cohort shards uniformly, and the double switch must agree.

    ``strategy`` and ``deepspeed.enabled`` used to be independent, which failed
    in opposite directions depending on which you set: ``strategy`` alone raised
    after the whole training environment was built, while ``enabled`` alone
    sharded silently under a config claiming it did not.
    """
    parallel = (yaml.safe_load(arm.read_text()) or {}).get("parallel") or {}
    ds = parallel.get("deepspeed") or {}
    assert parallel.get("strategy") == "deepspeed", (
        f"{_arm_id(arm)}: cohort arms shard with DeepSpeed; got "
        f"strategy={parallel.get('strategy')!r}"
    )
    assert ds.get("enabled") is True, (
        f"{_arm_id(arm)}: deepspeed.enabled must equal (strategy == 'deepspeed')"
    )
    assert ds.get("zero_stage") == 2, (
        f"{_arm_id(arm)}: cohort is ZeRO-2; got zero_stage={ds.get('zero_stage')!r}. "
        "Stage 3 also partitions parameters and changes the communication "
        "profile, so a lone arm on 3 is no longer comparable to its siblings."
    )


@pytest.mark.skipif(not _DATA_ARMS, reason="kspace_filling cohort not present")
@pytest.mark.parametrize("arm", _DATA_ARMS, ids=[_arm_id(a) for a in _DATA_ARMS])
def test_deepspeed_arm_is_not_fp16(arm: Path) -> None:
    """fp16 + DeepSpeed on a k-space arm produces NaNs, not a slowdown.

    There is no ``complex16``, so ``get_autocast_context`` disables autocast for
    complex+fp16 — but DeepSpeed casts weights to half from INSIDE the engine,
    where that guard cannot see it. The cohort runs fp32; bfloat16 would also be
    safe. ``check_deepspeed_precision_coherent`` makes this an audit error, and
    this test is the cheaper tripwire that fires without a strict audit run.
    """
    cfg = yaml.safe_load(arm.read_text()) or {}
    precision = ((cfg.get("optimization") or {}).get("precision")) or {}
    if not precision.get("enabled", False):
        return  # AMP off -> fp32; neither an fp16 nor a bf16 block is emitted
    assert precision.get("dtype") in {"bfloat16", "float32"}, (
        f"{_arm_id(arm)}: AMP is enabled with dtype={precision.get('dtype')!r} "
        "under parallel.strategy: deepspeed on a complex/k-space arm. Use "
        "bfloat16 — fp16 weights against complex64 activations give NaNs."
    )


@pytest.mark.skipif(not _DATA_ARMS, reason="kspace_filling cohort not present")
@pytest.mark.parametrize("arm", _DATA_ARMS, ids=[_arm_id(a) for a in _DATA_ARMS])
def test_deepcompile_and_torch_compile_are_not_both_on(arm: Path) -> None:
    """They are alternatives, not layers.

    ``optimization.compile`` compiles the bare module in Stage A, *before*
    ``deepspeed.initialize``, so the ZeRO collectives are opaque calls the
    compiler cannot schedule across. DeepCompile traces the engine graph
    *including* the allgather/reduce-scatter — and handed an already-compiled
    module it cannot do the one thing it exists for.
    """
    cfg = yaml.safe_load(arm.read_text()) or {}
    torch_compile = ((cfg.get("optimization") or {}).get("compile")) or {}
    deepcompile = (((cfg.get("parallel") or {}).get("deepspeed") or {}).get("compile")) or {}
    assert not (torch_compile.get("enabled", False) and deepcompile.get("enabled", False)), (
        f"{_arm_id(arm)}: optimization.compile and parallel.deepspeed.compile "
        "are both enabled. Pick one — DeepCompile must own the engine graph."
    )


@pytest.mark.skipif(not _ARMS, reason="kspace_filling cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_null_space_content_is_enabled_cohort_wide(arm: Path) -> None:
    """The null-band term is a loss-recipe axis, so it moves on every arm at once.

    Under ``dc_method: hard`` the measurement replaces the prediction at every
    acquired bin, so every post-DC term is a constant there and
    ``lambda_pre_dc_kspace`` is masked to ``(1 - M)`` as well. This term is the
    cohort's only weighted supervision of the unobserved bins. An arm that
    silently keeps it off is not a control -- it is a second axis inside a
    shootout whose contract is one axis, and the resulting numbers are
    unattributable rather than merely different.

    It is asserted through the resolved weight table, not the YAML key: a
    declared ``enabled: true`` that resolves to weight 0.0 reads as on and
    trains as off (pitfall 15).
    """
    cfg = yaml.safe_load(arm.read_text()) or {}
    entries = ((cfg.get("losses") or {}).get("kspace_losses")) or []
    declared = [e for e in entries if isinstance(e, dict) and e.get("name") == "null_space_content"]
    if not declared:
        pytest.skip(f"{_arm_id(arm)} does not declare null_space_content")

    assert declared[0].get("enabled") is True, (
        f"{_arm_id(arm)}: null_space_content is declared but switched off while "
        "its cohort siblings carry it."
    )
    table = build_loss_weight_table(LossConfigSchema(**cfg["losses"]))
    entry = table.get("null_space_content")
    assert entry is not None, f"{_arm_id(arm)}: enabled but absent from the weight table"
    assert float(entry.weight) > 0.0, (
        f"{_arm_id(arm)}: null_space_content resolves to weight {entry.weight} -- "
        "declared on, trains off."
    )


@pytest.mark.skipif(not _ARMS, reason="kspace_filling cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_the_coil_manifold_barrier_is_enabled_cohort_wide(arm: Path) -> None:
    """Same axis argument as the null-band term, one dimension over.

    ``sense_adjoint_l1`` supervises the single on-manifold direction and is blind
    to its 3-D orthogonal complement; ``complex_l1`` sees all four coils but
    penalises off-manifold error exactly as much as on-manifold error. This is
    the only term that couples the channels, so an arm that keeps it off is a
    second axis inside a one-axis shootout.

    ``input_domain: image`` is asserted, not assumed: the loss sets
    ``use_fourier_bridge`` from it, and under ``output_domain: kspace`` the
    ``complex_losses`` bridge is ``ifft_complex``, so ``kspace`` would trip the
    builder's double-bridge guard at construction.
    """
    cfg = yaml.safe_load(arm.read_text()) or {}
    losses = cfg.get("losses") or {}
    entries = losses.get("complex_losses") or []
    declared = [
        e for e in entries if isinstance(e, dict) and e.get("name") == "coil_subspace_residual"
    ]
    if not declared:
        pytest.skip(f"{_arm_id(arm)} does not declare coil_subspace_residual")

    entry = declared[0]
    assert entry.get("enabled") is True, f"{_arm_id(arm)}: declared but switched off"
    assert (entry.get("kwargs") or {}).get("input_domain") == "image", (
        f"{_arm_id(arm)}: input_domain must be 'image' under complex_losses, or the "
        "builder's double-bridge guard raises at construction."
    )
    table = build_loss_weight_table(LossConfigSchema(**losses))
    resolved = table.get("coil_subspace_residual")
    assert resolved is not None and float(resolved.weight) > 0.0, (
        f"{_arm_id(arm)}: resolves to {resolved} -- declared on, trains off."
    )


@pytest.mark.skipif(not _ARMS, reason="kspace_filling cohort not present")
@pytest.mark.parametrize("arm", _ARMS, ids=[_arm_id(a) for a in _ARMS])
def test_the_acquired_band_anchor_is_enabled_cohort_wide(arm: Path) -> None:
    """Every arm that supervises the null band must also anchor the acquired one.

    The two are complementary halves of one plane: under hard DC the null-band
    term is all the gradient there is, and it is masked away from the only bins
    where the target is knowable from the input. An arm carrying one and not the
    other is training on half the plane with no scale reference.
    """
    cfg = yaml.safe_load(arm.read_text()) or {}
    recon = ((cfg.get("losses") or {}).get("reconstruction")) or {}
    if "lambda_pre_dc_kspace" not in recon:
        pytest.skip(f"{_arm_id(arm)} carries a different pre-DC recipe")

    assert float(recon.get("lambda_pre_dc_acquired", 0.0)) > 0.0, (
        f"{_arm_id(arm)}: declares lambda_pre_dc_kspace={recon['lambda_pre_dc_kspace']} "
        "on the unobserved bins but nothing on the acquired ones, where hard DC "
        "leaves d(output)/d(prediction) == 0 for every other term."
    )


# ---------------------------------------------------------------------------
# 2026-09-22 cross-contrast objective fence.
#
# ``experiment_cross_contrast_kspace_diffusion`` was the one LIVE arm in the
# cohort still on the pre-#2220 recipe: its resolved table held three terms
# against the cohort's fourteen, so neither the null-space supervision (#2220)
# nor the acquired-band anchor (#2244) ever reached it. It was brought onto the
# cohort objective, MINUS the two coil-domain terms -- and that subtraction is
# the fragile half, because it reads as an oversight rather than as a fact
# about the arm's channel axis. The second test below states the fact.
# ---------------------------------------------------------------------------

_CROSS_CONTRAST = _COHORT / "experiment_cross_contrast_kspace_diffusion.yaml"

#: What the arm must share with the cohort reference, weight for weight. The
#: two coil-domain terms are absent by measurement, not by preference; `l2` is
#: the arm's own diffusion objective and has no counterpart in the reference.
_CROSS_CONTRAST_SHARED_WEIGHTS = {
    "bloch_residual": 0.0,
    "complex_l1": 1.0,
    "complex_spatial_gradient": 1.0,
    "hfen": 0.3,
    "log_spectral": 0.1,
    "null_space_content": 0.25,
    "perceptual": 0.0,
    "physics_constraint": 0.0,
    "pre_dc_acquired": 0.5,
    "pre_dc_kspace": 0.3,
    "snr_preserving": 0.0,
    "sobolev_kspace": 0.05,
}


@pytest.mark.skipif(not _CROSS_CONTRAST.exists(), reason="cross-contrast arm not present")
def test_cross_contrast_resolves_the_cohort_objective() -> None:
    """The arm trains on the cohort's terms, at the cohort's weights.

    Asserted on the RESOLVED table rather than on the YAML keys: the weight is
    declarable on two surfaces and canonicalised across aliases, so a key being
    present is not evidence that the term trains at that number.
    """
    losses = (yaml.safe_load(_CROSS_CONTRAST.read_text()) or {}).get("losses") or {}
    table = build_loss_weight_table(LossConfigSchema(**losses))
    for name, expected in sorted(_CROSS_CONTRAST_SHARED_WEIGHTS.items()):
        entry = table.get(name)
        assert entry is not None, (
            f"cross_contrast: '{name}' is not in the resolved weight table -- the arm "
            "has fallen back off the cohort objective."
        )
        assert float(entry.weight) == pytest.approx(expected), (
            f"cross_contrast: '{name}' resolves to {float(entry.weight)}, cohort weight "
            f"is {expected}."
        )


@pytest.mark.skipif(not _CROSS_CONTRAST.exists(), reason="cross-contrast arm not present")
def test_the_coil_domain_terms_cannot_run_on_the_cross_contrast_layout() -> None:
    """Why ``sense_adjoint_l1`` and ``coil_subspace_residual`` are absent there.

    ``prior_channel_range: [0, 8]`` makes channels 0-7 the fully-sampled T1
    prior and 8-15 the target contrast, so the 16 channels are TWO CONTRASTS x
    4 coils x (real, imag) -- not 8 coils. Both terms read dim 1 as the coil
    axis, so both raise against this arm's 4 sensitivity maps.

    This asserts the mechanism in both directions: red if the losses start
    accepting the layout (then the exclusion is stale and the arm should get
    them back), and red if they stop scoring the cohort's own 8-channel layout
    (then the exclusion has been misattributed). Pinning only the absence would
    catch neither.
    """
    torch = pytest.importorskip("torch")
    from spectramr.models.losses.registry import create_loss

    cfg = yaml.safe_load(_CROSS_CONTRAST.read_text()) or {}
    model_kwargs = (cfg.get("model") or {}).get("model_kwargs") or {}
    assert model_kwargs.get("prior_channel_range") == [0, 8], (
        "the prior no longer occupies channels 0-7; re-derive whether the coil "
        "terms can run before trusting this exclusion."
    )
    n_ch = int((cfg.get("model") or {}).get("out_channels"))
    n_coils = int(model_kwargs.get("num_physical_coils"))

    smaps = torch.randn(1, n_coils, 16, 16, dtype=torch.complex64)
    terms = {
        "sense_adjoint_l1": (create_loss("sense_adjoint_l1", normalize=True), "smaps"),
        "coil_subspace_residual": (
            create_loss("coil_subspace_residual", input_domain="image", weight_by_support=True),
            "coil_sensitivities",
        ),
    }
    for name, (fn, kw) in terms.items():
        with pytest.raises((RuntimeError, ValueError)):
            fn(
                torch.randn(1, n_ch, 16, 16),
                torch.randn(1, n_ch, 16, 16),
                **{kw: smaps},
            )
        # ...and the same call scores on the cohort's own layout, so the raise
        # above is about THIS arm's channel axis and not about the loss.
        value = fn(
            torch.randn(1, n_coils * 2, 16, 16),
            torch.randn(1, n_coils * 2, 16, 16),
            **{kw: smaps},
        )
        assert torch.isfinite(value), f"{name} did not score the {n_coils}-coil layout"
