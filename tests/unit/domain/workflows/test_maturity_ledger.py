"""The anti-facade spine: the workflow maturity ledger cannot lie.

Maturity is *declared* on each :class:`WorkflowProfile` (so deleting a strategy
is caught as a regression) and *asserted against the live registries* here (so a
profile cannot claim ``LIVE`` without a real registered strategy, nor
``EVAL_ONLY`` without real metrics). If this test ever needs to be relaxed to
keep a maturity claim, the claim is wrong — fix the profile, not the test.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import pathlib
from types import SimpleNamespace

import pytest

from spectramr.config.schemas.enums import Maturity, Regime
from spectramr.domain.workflows import WORKFLOW_PROFILES
from spectramr.domain.workflows.profiles import WorkflowProfile
from spectramr.infrastructure.validation.workflow_ledger import (
    forward_model_declared,
    losses_tagged,
    metrics_tagged,
    operator_registered,
    signal_model_registered,
    strategies_tagged,
)


def test_every_regime_has_exactly_one_profile() -> None:
    assert set(WORKFLOW_PROFILES) == set(Regime)
    for regime, profile in WORKFLOW_PROFILES.items():
        assert profile.name is regime


@pytest.mark.parametrize("regime", list(Regime))
def test_declared_forward_operator_is_real(regime: Regime) -> None:
    """A named forward operator must resolve in the physics registry."""
    profile = WORKFLOW_PROFILES[regime]
    if profile.forward_operator is not None:
        assert operator_registered(profile.forward_operator), (
            f"{regime.value} names forward_operator="
            f"{profile.forward_operator!r} which is not registered"
        )


@pytest.mark.parametrize("regime", list(Regime))
def test_declared_signal_model_is_real(regime: Regime) -> None:
    """A named signal model must resolve in the signal-model registry.

    The nonlinear twin of the operator check. Both fields are registry KEYS, so
    a profile naming physics nothing implements is caught here rather than at
    the first training step.
    """
    profile = WORKFLOW_PROFILES[regime]
    if profile.signal_model is not None:
        assert signal_model_registered(profile.signal_model), (
            f"{regime.value} names signal_model={profile.signal_model!r} "
            "which is not registered"
        )


@pytest.mark.parametrize("regime", list(Regime))
def test_signal_model_regime_matches_the_profile(regime: Regime) -> None:
    """A profile may not name another regime's signal model.

    ``signal_model`` is a claim about *this* regime's physics. Naming, say,
    ``adc_monoexp`` from the perfusion profile would resolve, satisfy the LIVE
    rule, and mean nothing — the registry check alone cannot catch that, because
    the key is perfectly real.
    """
    from spectramr.infrastructure.physics.signal_models.registry import (
        get_signal_model,
    )

    profile = WORKFLOW_PROFILES[regime]
    if profile.signal_model is None:
        return
    spec = get_signal_model(profile.signal_model)
    assert spec.regime is regime, (
        f"{regime.value} declares signal_model={profile.signal_model!r}, but that "
        f"model is registered for {spec.regime.value}."
    )


def test_every_maturity_member_has_a_branch() -> None:
    """A new Maturity member must force a branch, not fall through silently.

    The absent branch is exactly how this test came to check three of the four
    maturities: PARTIAL fell through the if/elif/elif with no assertions, so
    five regimes sat there for months with nothing backing them at all.
    """
    assert set(Maturity) == {
        Maturity.LIVE,
        Maturity.PARTIAL,
        Maturity.EVAL_ONLY,
        Maturity.STUB,
    }


@pytest.mark.parametrize("regime", list(Regime))
def test_maturity_is_backed_by_real_components(regime: Regime) -> None:
    profile = WORKFLOW_PROFILES[regime]
    maturity = profile.maturity

    if maturity is Maturity.LIVE:
        # LIVE has three clauses. Together they say: the regime's physics is
        # expressible, trainable, and GRADEABLE ON ITS OWN TERMS.
        assert forward_model_declared(profile), (
            f"{regime.value} claims LIVE but declares no forward model that "
            "resolves — neither a registered operator nor a registered signal "
            f"model (forward_operator={profile.forward_operator!r}, "
            f"signal_model={profile.signal_model!r})."
        )
        assert strategies_tagged(
            regime
        ), f"{regime.value} claims LIVE but no registered strategy is tagged for it"
        assert metrics_tagged(regime), (
            f"{regime.value} claims LIVE but no registered metric is tagged for "
            "it — nothing can grade this regime on its own terms, so the claim "
            "is unfalsifiable (pitfall #18)."
        )

        # Deliberately NOT asserted: losses_tagged. A regime may honestly train
        # on agnostic objectives — mri_structural trains on L1/SSIM, and that is
        # CORRECT rather than a gap. Requiring a regime-tagged loss would force
        # mis-tagging L1 as structural-only, which is false, and mass-tagging
        # generic components is exactly how an allow-list rots (the same failure
        # as the MRO tag leak, one layer up). Metrics are required and losses are
        # not because the asymmetry is real: a regime can train on a generic
        # objective, but it cannot be GRADED generically once its output stops
        # being an image — PSNR on a Ktrans map is pitfall #18. Where a regime
        # does have real losses, they are pinned per-regime below instead.

    elif maturity is Maturity.PARTIAL:
        # PARTIAL has exactly two honest backings:
        #   (a) strategy-backed — a strategy explicitly tagged for the regime;
        #   (b) physics-backed  — no strategy yet, but BOTH losses and metrics.
        #
        # NOT `strategies or losses or metrics`. That is strictly WEAKER than the
        # EVAL_ONLY branch below, which already asserts metrics AND no strategies
        # AND no losses — so a single tagged metric would satisfy it, and a
        # metrics-only regime is *literally the EVAL_ONLY state*. Since PARTIAL
        # unlocks `train` while EVAL_ONLY raises on it
        # (domain/workflows/enforcement.py), one metric tag would buy a trainable
        # claim. The `and` in tier (b) is what forbids the metrics-only state that
        # DEFINES EVAL_ONLY.
        strategy_backed = strategies_tagged(regime)
        physics_backed = losses_tagged(regime) and metrics_tagged(regime)
        assert strategy_backed or physics_backed, (
            f"{regime.value} claims PARTIAL but nothing backs it "
            f"(strategies={strategies_tagged(regime)}, "
            f"losses={losses_tagged(regime)}, metrics={metrics_tagged(regime)}). "
            "PARTIAL needs a regime-tagged strategy, or BOTH tagged losses and "
            "tagged metrics. If neither holds, the claim is wrong — downgrade the "
            "profile, do not relax this test."
        )

    elif maturity is Maturity.EVAL_ONLY:
        assert metrics_tagged(
            regime
        ), f"{regime.value} claims EVAL_ONLY but no registered metric is tagged for it"
        assert not strategies_tagged(regime), (
            f"{regime.value} is EVAL_ONLY yet a strategy is tagged for it — "
            "promote it to PARTIAL/LIVE"
        )
        assert not losses_tagged(regime), (
            f"{regime.value} is EVAL_ONLY yet a loss is tagged for it — "
            "promote it to PARTIAL/LIVE"
        )

    elif maturity is Maturity.STUB:
        assert not forward_model_declared(profile), (
            f"{regime.value} is a STUB but declares a forward model that "
            f"resolves (forward_operator={profile.forward_operator!r}, "
            f"signal_model={profile.signal_model!r})"
        )
        assert not strategies_tagged(regime)
        assert not losses_tagged(regime)
        assert not metrics_tagged(regime)

    else:  # pragma: no cover - exhaustiveness guard
        pytest.fail(
            f"{regime.value} declares maturity {maturity!r}, which this test does "
            "not assert. Every Maturity member needs a branch — a fall-through is "
            "how PARTIAL went unchecked across five regimes."
        )


@pytest.mark.parametrize("regime", list(Regime))
def test_declared_tasks_are_supported_by_the_regime(regime: Regime) -> None:
    """A strategy may not claim a task its tagged regime does not support.

    This is the mechanical guard against the CRLBMRFPulseDesign-shaped mistake:
    that strategy's task is ACQUISITION_DESIGN, which fingerprinting's
    ``supported_tasks`` does not include — so tagging it would have asserted a
    regime × task pairing the profile itself denies.
    """
    from spectramr.infrastructure.validation.workflow_ledger import (
        _declared_workflows,
        _iter_strategy_classes,
    )

    supported = WORKFLOW_PROFILES[regime].supported_tasks
    for cls in _iter_strategy_classes():
        if regime not in (_declared_workflows(cls) or ()):
            continue
        tasks = cls.__dict__["capabilities"].tasks or frozenset()
        unsupported = tasks - supported
        assert not unsupported, (
            f"{cls.__name__} is tagged for {regime.value} and claims task(s) "
            f"{sorted(t.value for t in unsupported)}, which that regime's "
            f"supported_tasks does not include "
            f"({sorted(t.value for t in supported)})."
        )


def test_structural_is_the_live_baseline() -> None:
    """Pin the one regime this slice lands LIVE, so a silent downgrade fails."""
    profile = WORKFLOW_PROFILES[Regime.STRUCTURAL]
    assert profile.maturity is Maturity.LIVE
    assert strategies_tagged(Regime.STRUCTURAL)


@pytest.mark.parametrize(
    "regime",
    [
        Regime.QUANTITATIVE,
        Regime.FINGERPRINTING,
        Regime.FUNCTIONAL,
        Regime.DIFFUSION_WEIGHTED,
        Regime.DYNAMIC,
    ],
)
def test_the_five_formerly_unbacked_partial_regimes_are_live(regime: Regime) -> None:
    """These five claimed PARTIAL for months with NOTHING tagged; now they are LIVE.

    They were never unbackable — real, dispatchable strategies existed all along
    (multi-echo B0 fit, one-shot multi-parameter, dictionary-free MRF, q-space
    SH, low-rank+sparse, Beltrami EPI). Three things hid it: the ledger never
    asserted PARTIAL at all, the capability tag leaked down the MRO so every
    subclass of the structural strategy reported ``mri_structural`` for free, and
    no metric was tagged for any of them, so nothing could grade them.

    Pinned per-regime so a lost tag names the regime it broke.
    """
    profile = WORKFLOW_PROFILES[regime]
    assert profile.maturity is Maturity.LIVE
    assert forward_model_declared(profile)
    assert strategies_tagged(regime)
    assert metrics_tagged(regime)


@pytest.mark.parametrize(
    "regime",
    [
        Regime.FLOW,
        Regime.PERFUSION,
        Regime.QUANTITATIVE,
        Regime.FINGERPRINTING,
        Regime.FUNCTIONAL,
        Regime.DIFFUSION_WEIGHTED,
        Regime.DYNAMIC,
        Regime.SPECTROSCOPIC,
    ],
)
def test_regimes_with_distinctive_physics_have_tagged_losses(regime: Regime) -> None:
    """Every regime whose physics is its own must have a loss expressing it.

    LIVE deliberately does not require ``losses_tagged`` — ``mri_structural``
    trains on agnostic L1/SSIM and that is correct, not a gap (see the LIVE
    branch above). But every regime listed here HAS distinctive physics, and a
    loss is how that physics enters the objective rather than merely being
    available to import. Without one, an arm advertising the regime trains as a
    plain regressor while the ledger says LIVE — pitfall #16 at the scientific
    layer.

    Structural is the deliberate omission from this list, and the only one.
    """
    assert losses_tagged(regime), (
        f"{regime.value} has regime-specific physics but no registered loss is "
        "tagged for it, so nothing puts that physics into the objective."
    )


def test_no_mr_regime_with_real_physics_is_left_at_eval_only() -> None:
    """EVAL_ONLY is now empty, and that is the point of the ledger.

    This replaces `test_empty_regimes_are_eval_only`, which pinned
    mri_spectroscopy at EVAL_ONLY — metrics, and nothing else. It was the last
    one, and it was never unbackable either: MRS quantification IS fitting the
    FID, the fit is analytic and self-supervised, and none of that needed data
    the repo lacks. What it needed was for somebody to write it.

    Kept as a live assertion rather than deleted, so that a regime silently
    sliding BACK to EVAL_ONLY (by losing its strategy or its losses) fails here
    and names itself.
    """
    eval_only = [
        r.value
        for r, p in WORKFLOW_PROFILES.items()
        if p.maturity is Maturity.EVAL_ONLY
    ]
    assert eval_only == [], (
        f"{eval_only} sit at EVAL_ONLY. If that is deliberate, say why here; if "
        "a regime lost its strategy or losses, restore them rather than "
        "downgrading the claim."
    )


def test_spectroscopy_is_live_on_a_signal_model_and_declares_no_operator() -> None:
    """The last EVAL_ONLY regime, completed.

    Like perfusion, LIVE via a signal model with `forward_operator=None` — the
    Lorentzian sum is nonlinear in frequency, linewidth and phase and has no
    adjoint, so None is the correct answer, not a gap. Pinned so nobody
    "completes" it by inventing an operator.
    """
    profile = WORKFLOW_PROFILES[Regime.SPECTROSCOPIC]
    assert profile.maturity is Maturity.LIVE
    assert profile.forward_operator is None  # no linear operator exists — correct
    assert profile.signal_model == "mrs_lorentzian"
    assert forward_model_declared(profile)
    assert strategies_tagged(Regime.SPECTROSCOPIC)
    assert losses_tagged(Regime.SPECTROSCOPIC)
    assert metrics_tagged(Regime.SPECTROSCOPIC)


def test_spectroscopy_declares_the_spectrum_domain_not_image_or_kspace() -> None:
    """An FID is neither an image nor 2-D k-space.

    Declaring the image/kspace family here would let a model registered for image
    reconstruction be pointed at a free induction decay, which is exactly the
    kind of mismatch signal_domains exists to catch.
    """
    assert WORKFLOW_PROFILES[Regime.SPECTROSCOPIC].signal_domains == frozenset(
        {"spectrum"}
    )


def test_flow_is_live_on_the_fourier_operator_and_the_pc_signal_model() -> None:
    """mri_flow is LIVE on a split forward model: linear Fourier + nonlinear encoding.

    Each velocity encoding is a multicoil Fourier acquisition; the encoding
    ``v -> m exp(i pi v / venc)`` has an inverse within ``|v| < venc`` but no
    adjoint, so it is a signal model and must not sit in the OperatorRegistry.
    """
    profile = WORKFLOW_PROFILES[Regime.FLOW]
    assert profile.maturity is Maturity.LIVE
    assert profile.forward_operator == "multi_coil_fft"
    assert operator_registered(profile.forward_operator)
    assert profile.signal_model == "phase_contrast"
    assert signal_model_registered(profile.signal_model)
    assert not operator_registered("phase_contrast")
    assert strategies_tagged(Regime.FLOW)
    assert losses_tagged(Regime.FLOW)
    assert metrics_tagged(Regime.FLOW)


def test_perfusion_is_live_on_a_signal_model_and_declares_no_operator() -> None:
    """mri_perfusion is LIVE *without* a forward operator, and that is correct.

    This replaces a test that pinned perfusion at PARTIAL "because the LIVE rule
    wants a linear operator". The rule was the defect: it demanded an
    ``OperatorRegistry`` entry, and the tracer-kinetic map is nonlinear with no
    adjoint, so the only way to satisfy it was to register a fake ``adjoint`` —
    which generic callers assuming ``<Ax,y> = <x,A'y>`` would have consumed
    silently. The rule *rewarded* the facade it existed to prevent.

    ``forward_operator is None`` is asserted deliberately: it is the honest
    answer for this regime, not a gap, and pinning it stops anyone from
    "completing" perfusion by inventing an operator now that LIVE no longer
    needs one.
    """
    profile = WORKFLOW_PROFILES[Regime.PERFUSION]
    assert profile.maturity is Maturity.LIVE
    assert profile.forward_operator is None  # no linear operator exists — correct
    assert profile.signal_model == "extended_tofts"
    assert signal_model_registered(profile.signal_model)
    assert forward_model_declared(profile)
    assert strategies_tagged(Regime.PERFUSION)
    assert losses_tagged(Regime.PERFUSION)
    assert metrics_tagged(Regime.PERFUSION)


def test_perfusion_config_default_matches_the_profiles_declared_signal_model() -> None:
    """The ledger's claim and the runtime dispatch must resolve the SAME key.

    A registry that only the ledger reads is a scoreboard, not a seam: the
    profile could claim ``extended_tofts`` while the strategy hardcoded
    something else, and both would look fine. ``data.perfusion.kinetic_model``
    defaults to the profile's declared model and the strategy resolves it
    through ``get_signal_model``, so this pins the two ends together.
    """
    from spectramr.config.schemas.data import PerfusionConfigSchema
    from spectramr.infrastructure.physics.signal_models.registry import get_signal_model

    declared = WORKFLOW_PROFILES[Regime.PERFUSION].signal_model
    assert PerfusionConfigSchema().kinetic_model == declared, (
        "data.perfusion.kinetic_model's default has drifted from the signal "
        "model mri_perfusion declares — the config would dispatch to physics "
        "the ledger is not vouching for."
    )
    spec = get_signal_model(declared)
    assert spec.parameters == ("ktrans", "ve", "vp")


# ---------------------------------------------------------------------------
# Which declared profile fields anything actually READS.
#
# A field nothing reads is pitfall #15 wearing a dataclass, and the specific
# failure these tests exist to stop is a *claim*: "signal_domains — the last
# declared-but-unchecked profile field" (commit 17d6c990d, 2026-07-16) was
# false when it was written, because the two fields below were already sitting
# unread and still are. A commit message can assert that; a test cannot.
# ---------------------------------------------------------------------------

#: Profile fields with NO reader under ``src/``. Each one's reason for existing
#: anyway is argued on ``WorkflowProfile``'s docstring — an argued deferral is
#: fine, an undocumented one is not.
UNREAD_PROFILE_FIELDS = frozenset({"modality"})

#: Profile fields some module under ``src/`` really consumes.
READ_PROFILE_FIELDS = frozenset(
    {
        "name",
        "maturity",
        "spatial_ranks",
        "required_axes",
        "signal_domains",
        "forward_operator",
        "signal_model",
        "supported_tasks",
    }
)


def _profile_consumer_modules() -> list[pathlib.Path]:
    """Modules under ``src/`` that can obtain a profile at all.

    Scoping the scan to real consumers is what keeps it honest: a bare grep for
    ``.modality`` across ``src/`` hits the EDA catalogue's unrelated
    ``entry.modality`` and would report the field as read.
    """
    import spectramr

    src = pathlib.Path(spectramr.__file__).resolve().parent
    definition_site = src / "domain" / "workflows" / "profiles.py"
    consumers = []
    for path in sorted(src.rglob("*.py")):
        if path == definition_site:
            continue
        text = path.read_text(encoding="utf-8")
        if any(
            token in text
            for token in ("WORKFLOW_PROFILES", "get_profile", "workflows.profiles")
        ):
            consumers.append(path)
    return consumers


def _attribute_reads(paths: list[pathlib.Path], attr: str) -> list[str]:
    """Every ``x.<attr>`` / ``getattr(x, "<attr>")`` site in ``paths``."""
    hits: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == attr:
                hits.append(f"{path.name}:{node.lineno}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == attr
            ):
                hits.append(f"{path.name}:{node.lineno} (getattr)")
    return hits


def test_every_profile_field_is_classified_as_read_or_unread() -> None:
    """A new profile field must land in exactly one bucket.

    The partition is the point. A field added with neither a reader nor a stated
    deferral fails here rather than sitting in the ledger looking load-bearing.
    """
    declared = {f.name for f in dataclasses.fields(WorkflowProfile)}
    classified = READ_PROFILE_FIELDS | UNREAD_PROFILE_FIELDS
    assert declared == classified, (
        f"WorkflowProfile fields {sorted(declared ^ classified)} are not "
        "classified. Give the field a reader (add it to READ_PROFILE_FIELDS) or "
        "an argued deferral on the WorkflowProfile docstring (add it to "
        "UNREAD_PROFILE_FIELDS). Do not delete this assertion."
    )


def test_the_unread_profile_fields_really_have_no_reader() -> None:
    """The declared-unread set is measured, not asserted.

    If someone wires ``modality``, this fails and tells them
    to move the field into ``READ_PROFILE_FIELDS`` — which is also what stops the
    set from silently rotting into a lie in the other direction.
    """
    consumers = _profile_consumer_modules()
    assert consumers, "found no profile-consumer modules — the scan is broken"

    # Control: a scan that finds nothing anywhere would "prove" every field
    # unread. `maturity` is read by enforcement.py and the health checker, so it
    # must register — otherwise the silence below means nothing.
    assert _attribute_reads(consumers, "maturity"), (
        "the scan cannot even see WorkflowProfile.maturity being read, so its "
        "silence about the unread fields is vacuous"
    )

    for fieldname in sorted(UNREAD_PROFILE_FIELDS):
        hits = _attribute_reads(consumers, fieldname)
        assert not hits, (
            f"WorkflowProfile.{fieldname} is listed as unread but is read at "
            f"{hits}. Move it to READ_PROFILE_FIELDS and drop its deferral note "
            "from the WorkflowProfile docstring."
        )


def test_signal_domains_was_not_the_last_unchecked_field() -> None:
    """Pin the refutation of the claim itself, by name.

    ``signal_domains`` genuinely IS read now (``check_workflow_signal_domain``),
    so the work was real — the *superlative* was false. Two fields remained. This
    test fails the day the claim becomes true, which is the only honest way for
    it to be made.
    """
    assert "signal_domains" in READ_PROFILE_FIELDS
    assert UNREAD_PROFILE_FIELDS, (
        "Every profile field now has a reader, so 'the last declared-but-"
        "unchecked field' is finally a claim someone may make — update this "
        "test and the WorkflowProfile docstring together."
    )


def test_optional_axes_is_gone_not_merely_empty() -> None:
    """``optional_axes`` was deleted on 2026-07-31, and stays deleted.

    It was inert twice over: no consumer read it AND no profile populated it.
    Those two facts normally have different fixes — an unread-but-populated
    field wants a reader — but here neither applied. ``required_axes`` alone
    drives ``check_workflow_required_axes``, and an *optional* axis has no rule
    to state, since by construction its absence cannot make an arm
    inadmissible. Its only possible reader would have been a rule that cannot
    fire, so the field was decoration.

    Asserted on the dataclass rather than on the values, because
    ``default_factory=frozenset`` made "every profile leaves it empty" and "the
    field does not exist" indistinguishable from the outside — which is how it
    survived two audits.
    """
    assert "optional_axes" not in WorkflowProfile.__dataclass_fields__, (
        "optional_axes is back on WorkflowProfile. It was removed because no "
        "consumer could read it: an optional axis states no rule. If a real "
        "reader now exists, add the field to READ_PROFILE_FIELDS and say what "
        "reads it."
    )


def test_dwi_live_does_not_rest_on_the_qspace_strategy() -> None:
    """DWI is LIVE on its signal model, loss and metric — not on q-space SH.

    Since #350 the q-space mechanism actually fires (the strategy reads the
    ``b_vectors`` key LoadDWIMetadata writes, and raises if it is absent — see the
    producer/consumer test below). But the LIVE claim still does not REST on it:
    these are the clauses that carry it, pinned so the claim stands on physics in
    the objective and a metric on its own terms, with the regulariser as honest
    added coverage rather than load-bearing evidence.
    """
    profile = WORKFLOW_PROFILES[Regime.DIFFUSION_WEIGHTED]
    assert profile.maturity is Maturity.LIVE
    assert profile.signal_model == "adc_monoexp"
    assert signal_model_registered(profile.signal_model)
    assert losses_tagged(Regime.DIFFUSION_WEIGHTED)  # dwi_adc_monoexp, reads b_values
    assert metrics_tagged(Regime.DIFFUSION_WEIGHTED)  # adc_mae, grades in mm²/s


def test_qspace_strategy_reads_the_directions_key_the_data_layer_writes() -> None:
    """The q-space mechanism is WIRED: consumer key == a producer key (#350).

    Was a facade — ``QSpaceDiffusionStrategy`` read ``batch["bvecs"]``, a key
    nothing produced, so it silently returned the base recon losses (pitfall #16).
    Fixed: the strategy reads ``batch["b_vectors"]``, exactly what
    ``LoadDWIMetadata`` writes under ``data/`` (the data SSOT rule), and RAISES if a
    diffusion batch lacks it. This pins producer∩consumer ≠ ∅ so the facade cannot
    silently return: an AST STORE-context scan of ``data/`` for the produced key,
    and a source check that the strategy reads that same key and no longer returns
    base on a missing one.
    """
    import spectramr
    from spectramr.infrastructure.training.strategies import qspace_diffusion_strategy

    data_root = pathlib.Path(spectramr.__file__).resolve().parent / "data"
    written: set[str] = set()
    for path in sorted(data_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                written.add(node.slice.value)

    # Producer: LoadDWIMetadata writes b_vectors under data/ (STORE context, so a
    # docstring/error-message mention cannot pass this vacuously).
    assert "b_vectors" in written, (
        "the STORE-context scan cannot see LoadDWIMetadata's b_vectors write "
        f"(saw {len(written)} keys) — the producer control is broken"
    )
    # Consumer: the strategy reads that exact key, and its old 'bvecs' facade key
    # is gone — so the get() no longer returns None on every batch.
    strat_src = inspect.getsource(qspace_diffusion_strategy)
    assert 'batch.get("b_vectors")' in strat_src, (
        "QSpaceDiffusionStrategy no longer reads the produced 'b_vectors' key — the "
        "q-space regulariser would be inert again (#350 regression)"
    )
    assert (
        'batch.get("bvecs")' not in strat_src
    ), "the dead 'bvecs' facade key is back in QSpaceDiffusionStrategy"
    # And a missing directions key now RAISES rather than silently returning base.
    impl_src = inspect.getsource(
        qspace_diffusion_strategy.QSpaceDiffusionStrategy._compute_losses_impl
    )
    assert "raise ValueError" in impl_src


def test_required_axes_guard_skips_unannotated_but_rejects_annotated() -> None:
    """The guard's polarity, which the FLOW profile used to state backwards.

    An arm is rejected only when its dataset_type IS annotated in
    ``axis_exposure`` and the annotation lacks the required axis. An UNANNOTATED
    dataset_type skips — passed, info. So "no dataset_type exposes
    VELOCITY_ENCODING" never protected mri_flow from anything.

    Uses a deliberately non-existent dataset_type rather than a real one, so this
    keeps testing the polarity rather than today's contents of the table.
    """
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    checker = ConfigHealthChecker()

    annotated = SimpleNamespace(
        workflow=SimpleNamespace(regime=Regime.FLOW),
        data=SimpleNamespace(dataset_type="image"),  # annotated: frozenset()
    )
    rejected = checker.check_workflow_required_axes(annotated)
    assert rejected.passed is False, (
        "dataset_type='image' is annotated and exposes no VELOCITY_ENCODING, so "
        "a flow arm on it must be rejected"
    )
    assert rejected.severity == "error"

    unannotated = SimpleNamespace(
        workflow=SimpleNamespace(regime=Regime.FLOW),
        data=SimpleNamespace(dataset_type="__unannotated_dataset_type__"),
    )
    skipped = checker.check_workflow_required_axes(unannotated)
    assert skipped.passed is True, (
        "an unannotated dataset_type must SKIP, not reject — if this now fails, "
        "the guard grew teeth the FLOW profile comment does not describe"
    )
    assert skipped.severity == "info"
