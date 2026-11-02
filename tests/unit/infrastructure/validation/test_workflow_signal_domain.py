"""Tests for check_workflow_signal_domain.

`signal_domains` was the last field on `WorkflowProfile` that was declared but
never asserted — the same shape as `forward_operator` before the ledger walked
it, and as PARTIAL before its branch existed. These tests pin both what the check
catches and, just as importantly, what it must NOT reject.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.config.schemas.enums import Regime
from spectramr.config.schemas.workflow import WorkflowConfigSchema
from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker
from spectramr.models.capabilities import ModelCapabilities


def _check(
    regime: Regime | None,
    input_domain,
    *,
    model_type: str = "fake_model",
    signal_domain: str | None = None,
):
    """Run the check with a stubbed model registry entry.

    The registry is patched rather than a real model chosen, so the test states
    the (regime, input_domain) pair under test directly instead of depending on
    which registered model happens to declare what today.
    """
    checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
    workflow = (
        WorkflowConfigSchema(regime=regime, signal_domain=signal_domain)
        if regime is not None
        else None
    )
    cfg = SimpleNamespace(
        workflow=workflow,
        model=SimpleNamespace(
            model_type=model_type,
            in_channels=1,
            out_channels=1,
            spatial_dims=2,
            input_type=None,
        ),
    )
    from spectramr.models.registry import MODEL_REGISTRY

    caps = (
        ModelCapabilities(input_domain=input_domain)
        if input_domain is not None
        else ModelCapabilities()
    )
    MODEL_REGISTRY[model_type] = {"capabilities": caps, "mode": "reconstruction"}
    try:
        return checker.check_workflow_signal_domain(cfg)
    finally:
        MODEL_REGISTRY.pop(model_type, None)


class TestCatchesRealMismatches:
    def test_an_image_model_pointed_at_a_spectrum_regime_is_rejected(self) -> None:
        """THE hole this check closes.

        mri_spectroscopy's signal is an FID. An image-reconstruction UNet would
        consume it happily — 2T real channels look exactly like an image to a
        conv net — train, converge, and mean nothing.
        """
        r = _check(Regime.SPECTROSCOPIC, "image")
        assert not r.passed
        assert r.severity == "error"
        assert "disjoint" in r.message
        assert "spectrum" in r.message

    def test_a_spectrum_model_pointed_at_an_image_regime_is_rejected(self) -> None:
        """The mirror case — an FID consumer aimed at structural MRI."""
        r = _check(Regime.STRUCTURAL, "spectrum")
        assert not r.passed
        assert r.severity == "error"

    def test_the_failure_names_both_sides_and_offers_a_fix(self) -> None:
        r = _check(Regime.SPECTROSCOPIC, "kspace")
        assert "mri_spectroscopy" in r.message
        assert r.yaml_keys == ["workflow.regime", "model.model_type"]
        assert "input_domain" in (r.fix_hint or "")


class TestDoesNotRejectLegitimateArms:
    """The false-positive risk that kept this check unwritten. It was real."""

    def test_a_spectrum_model_on_the_spectroscopy_regime_passes(self) -> None:
        assert _check(Regime.SPECTROSCOPIC, "spectrum").passed

    @pytest.mark.parametrize("domain", ["image", "kspace", "complex_image"])
    def test_the_mr_image_regimes_accept_their_whole_family(self, domain: str) -> None:
        for regime in (Regime.STRUCTURAL, Regime.QUANTITATIVE, Regime.FLOW):
            assert _check(regime, domain).passed, f"{regime.value} rejected {domain}"

    def test_a_multi_domain_model_passes_if_any_domain_fits(self) -> None:
        """input_domain may be a tuple; one overlap is enough.

        cs_mno_operator genuinely handles both pde_grid and image. Requiring
        every declared domain to match would reject it from every MR regime.
        """
        assert _check(Regime.STRUCTURAL, ("pde_grid", "image")).passed

    def test_an_unannotated_model_is_skipped_never_guessed(self) -> None:
        """The escape hatch check_data_model_compatibility already establishes."""
        r = _check(Regime.SPECTROSCOPIC, None)
        assert r.passed
        assert "skipped" in r.message

    def test_a_stub_regime_declaring_no_signal_domains_is_skipped(self) -> None:
        r = _check(Regime.CT, "image")
        assert r.passed
        assert "no signal domains" in r.message

    def test_no_workflow_declared_is_skipped(self) -> None:
        assert _check(None, "image").passed


def test_it_checks_the_input_domain_not_the_output() -> None:
    """The direction is the whole design, and getting it backwards inverts it.

    `signal_domains` says what the ACQUISITION is, not what the model predicts.
    Every parameter-mapping arm emits something other than its regime's signal:
    MRSQuantificationStrategy consumes a spectrum and emits [B, 4M, H, W]
    resonance maps — domain `image`. Checking output_domain would reject the
    very arm mri_spectroscopy exists for.

    So: a model declaring input_domain="spectrum" and output_domain="image" must
    PASS on the spectroscopy regime.
    """
    checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
    from spectramr.models.registry import MODEL_REGISTRY

    MODEL_REGISTRY["_mrs_probe"] = {
        "capabilities": ModelCapabilities(
            input_domain="spectrum",  # consumes the FID
            output_domain="image",  # emits parameter maps
        ),
        "mode": "reconstruction",
    }
    try:
        cfg = SimpleNamespace(
            workflow=WorkflowConfigSchema(regime=Regime.SPECTROSCOPIC),
            model=SimpleNamespace(
                model_type="_mrs_probe",
                in_channels=1,
                out_channels=1,
                spatial_dims=2,
                input_type=None,
            ),
        )
        assert checker.check_workflow_signal_domain(cfg).passed
    finally:
        MODEL_REGISTRY.pop("_mrs_probe", None)


def test_the_check_is_wired_into_the_audit_run() -> None:
    """An unwired check is the facade this repo polices.

    check_workflow_signal_domain existing but never being called would be
    exactly the "landed but unwired" shape (#279).
    """
    import inspect

    source = inspect.getsource(ConfigHealthChecker)
    assert "self.check_workflow_signal_domain(config)" in source


def test_spectrum_is_a_real_domain_literal() -> None:
    """The profile declares signal_domains={"spectrum"} as a bare string.

    `domain/` is kept import-free, so WorkflowProfile.signal_domains cannot
    reference the Domain literal directly — which means nothing structurally
    stops the two drifting apart. This is the seam that ties them.
    """
    from typing import get_args

    from spectramr.domain.workflows import WORKFLOW_PROFILES
    from spectramr.models.capabilities import Domain

    declared = WORKFLOW_PROFILES[Regime.SPECTROSCOPIC].signal_domains
    assert declared == frozenset({"spectrum"})
    assert set(declared) <= set(get_args(Domain)), (
        "mri_spectroscopy declares a signal domain that is not a Domain literal, "
        "so no model could ever advertise it and the check would reject every arm."
    )


def test_every_profile_signal_domain_is_a_real_domain_literal() -> None:
    """The general form: no profile may name a domain no model can declare.

    Without this, a typo'd or aspirational signal_domain would make the new check
    reject every arm of that regime — the check would look like it was working
    while actually being unsatisfiable.
    """
    from typing import get_args

    from spectramr.domain.workflows import WORKFLOW_PROFILES
    from spectramr.models.capabilities import Domain

    known = set(get_args(Domain))
    for regime, profile in WORKFLOW_PROFILES.items():
        unknown = set(profile.signal_domains) - known
        assert not unknown, (
            f"{regime.value} declares signal_domains {sorted(unknown)} which are "
            f"not Domain literals {sorted(known)} — unsatisfiable by any model."
        )


class TestTheDeclaredSignalDomain:
    """``workflow.signal_domain`` narrows the regime's SET to the one domain the
    arm says it consumes.

    Semantics are pinned to *consumes* (== ``ModelCapabilities.input_domain``),
    never *emits*. That is not a style choice: every parameter-mapping arm emits
    off-regime — MRS consumes a ``spectrum`` and emits resonance maps (domain
    ``image``) — so an emits-reading would reject the arms these regimes exist
    for, the exact false positive this check's docstring was written about.
    """

    def test_a_domain_the_regime_does_not_produce_is_an_error(self) -> None:
        r = _check(Regime.SPECTROSCOPIC, "spectrum", signal_domain="kspace")
        assert not r.passed and r.severity == "error"
        assert "not produced by" in r.message
        assert r.yaml_keys == ["workflow.regime", "workflow.signal_domain"]

    def test_the_profile_check_runs_even_when_the_model_is_unannotated(self) -> None:
        """Ordering: declared-vs-profile needs no model annotation, so it must
        not sit behind the "model declares no input_domain -> skip" branch."""
        r = _check(Regime.SPECTROSCOPIC, None, signal_domain="kspace")
        assert not r.passed, (
            "an unannotated model swallowed a signal_domain the regime cannot "
            "produce; the profile check is behind the skip."
        )

    def test_declaring_narrows_and_catches_what_the_set_would_miss(self) -> None:
        """The value of the field. `image` and `kspace` are both in
        mri_structural's set, so without a declaration an image-consuming model
        passes. Declaring `kspace` makes the mismatch visible."""
        assert _check(Regime.STRUCTURAL, "image").passed
        narrowed = _check(Regime.STRUCTURAL, "image", signal_domain="kspace")
        assert not narrowed.passed and narrowed.severity == "error"
        assert "['kspace']" in narrowed.message

    def test_declaring_the_matching_domain_passes(self) -> None:
        r = _check(Regime.STRUCTURAL, "kspace", signal_domain="kspace")
        assert r.passed and r.severity == "info"

    def test_a_parameter_mapping_arm_is_judged_on_what_it_consumes(self) -> None:
        """The regression guard. An MRS arm consumes `spectrum` and emits image
        -domain maps; declaring what it CONSUMES must pass."""
        r = _check(Regime.SPECTROSCOPIC, "spectrum", signal_domain="spectrum")
        assert r.passed, (
            "an MRS arm declaring the domain it consumes was rejected — "
            "signal_domain is being read as *emits*."
        )
