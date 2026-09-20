"""Unit tests for the declared-but-unread model-knob detector.

Pairs with ``src/spectramr/infrastructure/validation/inert_knobs.py``.

The detector's whole value is its *precision*: a false positive sends an author
to delete a live knob, which is strictly worse than the defect it reports. Most
of what follows therefore pins the ways a parameter legitimately counts as READ.
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.validation.inert_knobs import (
    DELIBERATELY_UNREAD,
    InertKnob,
    declared_knobs_out_of_scope,
    find_inert_declared_knobs,
    unread_init_params,
)


class _Swallows:
    """Declares a parameter, documents it, never references it."""

    def __init__(self, used: int = 1, ignored: str = "complex"):
        """Args: used (int): kept. ignored (str): documented and dropped."""
        self.used = used


class _AssignsAttribute:
    def __init__(self, knob: str = "x"):
        self.knob = knob


class _ForwardsToSuper(_AssignsAttribute):
    def __init__(self, knob: str = "x"):
        super().__init__(knob)


class _ReadsInBranchOnly:
    def __init__(self, flag: bool = False):
        self.mode = "on" if flag else "off"


class _ReadsInFString:
    def __init__(self, label: str = "a"):
        self.name = f"model-{label}"


class _UsesLocals:
    """Undecidable by AST — the detector must decline rather than accuse."""

    def __init__(self, alpha: int = 1, beta: int = 2):
        self._cfg = dict(locals())


class _NoParams:
    def __init__(self):
        self.x = 1


def test_unreferenced_parameter_is_reported():
    assert unread_init_params(_Swallows) == frozenset({"ignored"})


@pytest.mark.parametrize(
    "cls",
    [_AssignsAttribute, _ForwardsToSuper, _ReadsInBranchOnly, _ReadsInFString],
    ids=["self-assign", "super-forward", "branch-condition", "f-string"],
)
def test_genuine_reads_are_not_reported(cls):
    """Every one of these consumes its parameter; none may be flagged."""
    assert unread_init_params(cls) == frozenset()


def test_reflective_escape_declines_to_answer():
    """``locals()`` can consume a parameter without naming it.

    The empty set here means "no answer", not "no problems" — asserting it
    pins that the detector stays silent rather than reporting ``alpha``/``beta``.
    """
    assert unread_init_params(_UsesLocals) == frozenset()


def test_parameterless_init_is_empty():
    assert unread_init_params(_NoParams) == frozenset()


def test_builtin_init_is_empty():
    """A class with no Python-level ``__init__`` yields no answer, not a crash."""

    class _Plain:
        pass

    assert unread_init_params(_Plain) == frozenset()


def test_only_declared_knobs_are_reported():
    """Arm-scoped: an unread parameter the arm never set is not this check's business."""
    hits = find_inert_declared_knobs("demo", {"used": 3}, _Swallows)
    assert hits == []


def test_declared_and_unread_is_reported_with_provenance():
    hits = find_inert_declared_knobs("demo", {"used": 3, "ignored": "relu"}, _Swallows)
    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, InertKnob)
    assert hit.key == "ignored"
    assert hit.yaml_path == "model.model_kwargs.ignored"
    assert hit.declared_value == "relu"
    assert hit.model_type == "demo"
    assert hit.class_name == "_Swallows"


def test_unresolved_model_class_returns_no_answer():
    assert find_inert_declared_knobs("demo", {"ignored": "relu"}, None) == []


def test_empty_kwargs_returns_no_answer():
    assert find_inert_declared_knobs("demo", {}, _Swallows) == []
    assert find_inert_declared_knobs("demo", None, _Swallows) == []


def test_allowlist_suppresses_deliberate_parameter(monkeypatch):
    """An allowlisted entry records intent and must not be reported."""
    monkeypatch.setitem(DELIBERATELY_UNREAD, ("_Swallows", "ignored"), "on purpose")
    assert find_inert_declared_knobs("demo", {"ignored": "relu"}, _Swallows) == []


def test_results_are_sorted_by_key():
    class _TwoDead:
        def __init__(self, zebra: int = 1, alpha: int = 2, live: int = 3):
            self.live = live

    hits = find_inert_declared_knobs("demo", {"zebra": 1, "alpha": 2, "live": 3}, _TwoDead)
    assert [h.key for h in hits] == ["alpha", "zebra"]


def test_regression_kspace_cold_diffusion_swallows_four_knobs():
    """The defect that motivated the check, pinned against silent repair.

    ``KSpaceColdDiffusionGenerator.__init__`` declares and documents these four
    and references none: flipping ``activation``/``use_complex_conv`` leaves the
    module tree and forward output bit-identical. If a fix lands, this test
    fails loudly and should be updated — that is the point.
    """
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceColdDiffusionGenerator,
    )

    assert unread_init_params(KSpaceColdDiffusionGenerator) == frozenset(
        {"activation", "use_complex_conv", "time_embedding_type", "training_mode"}
    )


# ── the scope the detector cannot see (non-negotiable 15) ────────────────────
class _AbsorbsEverything:
    """One named knob; the rest arrive through ``**kwargs`` and nothing reads them."""

    def __init__(self, named: int = 1, **kwargs):
        self.named = named


def test_kwargs_absorbed_knobs_are_reported_as_out_of_scope():
    """The planted violation: a declared key nothing reads, hidden behind **kwargs.

    ``unread_init_params`` enumerates named parameters only, so this key can
    never appear in its answer. Before the scope accessor existed the audit
    still printed "all 3 declared model_kwargs are read", which is a clean
    verdict over a population the check never looked at.
    """
    declared = {"named": 1, "invisible": 2, "also_invisible": 3}
    assert find_inert_declared_knobs("demo", declared, _AbsorbsEverything) == []
    assert declared_knobs_out_of_scope(declared, _AbsorbsEverything) == frozenset(
        {"invisible", "also_invisible"}
    )


def test_a_fully_named_signature_has_nothing_out_of_scope():
    """Guards the check above from passing because the accessor returns everything."""

    class _AllNamed:
        def __init__(self, a: int = 1, b: int = 2):
            self.a, self.b = a, b

    assert declared_knobs_out_of_scope({"a": 1, "b": 2}, _AllNamed) == frozenset()


def test_the_audit_message_no_longer_claims_unmeasured_knobs_are_read():
    """The message must separate *measured* from *declared*, or it overclaims.

    Measured on the kspace_filling cohort: 1365 of 1590 declared keys are
    absorbed by ``**kwargs``, so "all N are read" spoke for 85.8 % of knobs the
    detector cannot see.
    """
    import inspect as _inspect

    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    src = _inspect.getsource(ConfigHealthChecker.check_declared_model_kwargs_are_read)
    assert "declared model_kwargs are in" in src, "the scoped message was reverted"
    assert "are NOT measured by this check" in src
    assert "all {len(declared)} declared model_kwargs are read" not in src


def test_the_scale_domain_check_fires_where_the_transform_guard_cannot():
    """The transform's refusal is unreachable; this is the owner that can answer.

    ``KSpaceNormalizationSpec`` refuses a ``processing`` block that omits
    ``kspace_scale_domain``, but training hands it ``config.data`` and inference
    hands it ``config.model_dump()`` — pydantic filled the default in both, so
    the only input that trips it is a raw mapping nothing passes.
    ``model_fields_set`` still records what the author typed, which is why the
    question is answerable here and nowhere downstream.
    """
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    arm = "experiments/inprogress/kspace_filling/experiment_11_kfn_none.yaml"
    cfg = TrainingSettings.from_yaml(arm)
    declared = ConfigHealthChecker().check_kspace_scale_domain_is_declared(cfg)
    assert "declared as 'image'" in declared.message

    processing = cfg.data.processing.model_copy()
    processing.model_fields_set.discard("kspace_scale_domain")
    data = cfg.data.model_copy(update={"processing": processing})
    undeclared = ConfigHealthChecker().check_kspace_scale_domain_is_declared(
        cfg.model_copy(update={"data": data})
    )
    assert "declares no kspace_scale_domain" in undeclared.message


def test_a_divergent_metric_transform_is_reported_not_silently_compared():
    """69 cohort arms measure train_<m> and val_<m> through different transforms.

    The training path reads ``metrics.transform`` and the validation path reads
    ``validation.scoring.output_transform``. On this cohort that is
    ``ifft_sense_adjoint`` in normalised log-compressed units against
    ``ifft_magnitude`` in denormalised physical ones, so the two series share a
    name stem and measure different quantities — and their difference reads as
    a generalisation gap when it is arithmetic.
    """
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    arm = "experiments/inprogress/kspace_filling/experiment_11_kfn_none.yaml"
    cfg = TrainingSettings.from_yaml(arm)
    assert cfg.metrics.transform != cfg.validation.scoring.output_transform
    result = ConfigHealthChecker().check_metric_transforms_agree(cfg)
    assert "different measurements under one name" in result.message

    scoring = cfg.validation.scoring.model_copy(
        update={"output_transform": cfg.metrics.transform}
    )
    validation = cfg.validation.model_copy(update={"scoring": scoring})
    agreed = ConfigHealthChecker().check_metric_transforms_agree(
        cfg.model_copy(update={"validation": validation})
    )
    assert "share one transform" in agreed.message
