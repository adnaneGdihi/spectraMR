"""Tier-1 `validation_path_is_wired`: a stubbed reverse process is a pre-flight failure.

#2255: `baseline_fdb` passed `audit --strict`, trained 999 iterations, then aborted
when all 240 validation batches hit `FDBBaseline.validation_sample`, which can only
raise. Eighteen minutes of GPU to learn something the class object says statically.

Per non-negotiable 15 each rule the detector claims gets a planted violation that
turns it red, and each shape it must NOT fire on gets one that keeps it green — the
conditional-raise case especially, since that is a live implementation with a guard
and is the obvious way this detector would go wrong.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker


def _config(model_type: str = "some_model", status: str | None = None):
    return SimpleNamespace(
        model=SimpleNamespace(model_type=model_type),
        metadata=SimpleNamespace(status=status),
    )


def _run(monkeypatch, cls, *, model_type: str = "some_model", status: str | None = None):
    import spectramr.models.registry as registry_mod

    monkeypatch.setattr(registry_mod, "MODEL_REGISTRY", {model_type: {"class": cls}}, raising=False)
    return ConfigHealthChecker().check_validation_path_is_wired(_config(model_type, status))


# --- the shapes the detector must reject ----------------------------------


class OnlyRaises:
    def validation_sample(self, input_batch, target_batch, batch=None):
        raise NotImplementedError("not wired: needs the loader's mask")


class RaisesAfterSomeWork:
    """A stub that computes before refusing — FDB's actual shape."""

    def validation_sample(self, input_batch, target_batch, batch=None):
        del target_batch, batch
        data_type = "singlecoil"
        raise NotImplementedError(f"not wired for {data_type!r}")


class InheritsAStub(OnlyRaises):
    """Adds nothing; the MRO still lands on a method that can only raise."""


# --- the shapes it must accept --------------------------------------------


class Wired:
    def validation_sample(self, input_batch, target_batch, batch=None):
        return input_batch


class RaisesConditionallyButReturns:
    """A live implementation with a guard. Firing here would be the detector's
    own defect — and a detector defect outranks an equal-scoring code defect."""

    def validation_sample(self, input_batch, target_batch, batch=None):
        if batch is None:
            raise NotImplementedError("this path needs the full batch")
        return input_batch


class NoSuchMethod:
    """Not every model is a baseline adapter; most never define this at all."""


class TestPlantedViolations:
    @pytest.mark.parametrize("cls", [OnlyRaises, RaisesAfterSomeWork, InheritsAStub])
    def test_a_stubbed_validation_sample_fails_the_check(self, monkeypatch, cls):
        result = _run(monkeypatch, cls)
        assert not result.passed
        assert result.severity == "error"
        assert result.check_name == "validation_path_is_wired"
        assert "validation_sample" in result.message
        assert "metadata.status" in result.yaml_keys

    @pytest.mark.parametrize("cls", [Wired, RaisesConditionallyButReturns, NoSuchMethod])
    def test_a_live_validation_sample_passes(self, monkeypatch, cls):
        assert _run(monkeypatch, cls).passed

    def test_a_declared_needs_implementation_arm_is_not_double_reported(self, monkeypatch):
        """The launch guard already refuses it in under two seconds; saying so twice
        would push an arm that is correctly declared into the audit's error list."""
        result = _run(monkeypatch, OnlyRaises, status="needs_implementation")
        assert result.passed
        assert "needs_implementation" in result.message

    def test_an_unresolvable_model_type_is_deferred_not_guessed(self, monkeypatch):
        import spectramr.models.registry as registry_mod

        monkeypatch.setattr(registry_mod, "MODEL_REGISTRY", {"other": {}}, raising=False)
        assert ConfigHealthChecker().check_validation_path_is_wired(_config()).passed

    def test_no_model_type_is_not_an_error(self, monkeypatch):
        assert ConfigHealthChecker().check_validation_path_is_wired(_config("")).passed


class TestAgainstTheRealAdapters:
    """The detector has to separate the four adapters that actually ship."""

    def test_fdb_is_the_stub_and_its_two_siblings_are_not(self):
        from spectramr.models.baselines._base import BaselineAdapter
        from spectramr.models.baselines.cdiffmr import CDiffMRBaseline
        from spectramr.models.baselines.fdb import FDBBaseline
        from spectramr.models.baselines.shen2024 import Shen2024Baseline

        is_stub = ConfigHealthChecker._validation_sample_is_a_stub
        assert is_stub(FDBBaseline)
        assert is_stub(BaselineAdapter)
        assert not is_stub(CDiffMRBaseline)
        assert not is_stub(Shen2024Baseline)


class TestTheCheckIsActuallyRun:
    """Non-negotiable 16: a check that is defined and never invoked is a facade.
    `meta.health_checker_no_orphan_checks` enforces this repo-wide; pinned here so
    the reason the call site exists is stated where the check is."""

    def test_the_run_method_invokes_it(self):
        from spectramr.infrastructure.validation import config_health_checker as mod

        run = next(
            fn
            for name, fn in vars(ConfigHealthChecker).items()
            if callable(fn)
            and "check_registered_model_resolves(config)" in (inspect.getsource(fn) or "")
        )
        src = inspect.getsource(run)
        assert "self.check_validation_path_is_wired(config)" in src
        del mod


class TestBaselineFdbIsDeclared:
    """The arm half of #2255: the YAML now says what the code says."""

    def test_the_arm_declares_needs_implementation(self):
        import pathlib

        import yaml

        path = (
            pathlib.Path(__file__).resolve().parents[4]
            / "experiments/inprogress/kspace_filling/baseline_fdb.yaml"
        )
        meta = yaml.safe_load(path.read_text())["metadata"]
        assert meta["status"] == "needs_implementation"
        assert meta["status_reason"]
