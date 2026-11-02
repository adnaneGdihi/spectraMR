"""Tests for the radiomic-extractor backend selector.

Closes ``TODO/backlog_fastrad_radiomics_backend.md`` Step 0: an
explicit ``backend=`` parameter with fail-loud validation. The
``fastrad`` GPU path itself is tracked separately -- selecting it
explicitly raises ``NotImplementedError`` until the wiring lands, and
the audit catches typos in the advertised set
(CLAUDE.md pitfall #15).
"""
from __future__ import annotations

import pytest


def test_allowed_backends_match_advertised_set() -> None:
    """The class-level allow-list pins the contract surface."""
    from spectramr.core.metrics.radiomic import RadiomicFeatureExtractor

    assert RadiomicFeatureExtractor.ALLOWED_BACKENDS == (
        "pyradiomics",
        "fastrad",
        "auto",
    )


def test_unknown_backend_raises_value_error() -> None:
    """Typos must raise -- no silent fallback to a default."""
    from spectramr.core.metrics.radiomic import RadiomicFeatureExtractor

    with pytest.raises(ValueError, match="backend="):
        RadiomicFeatureExtractor(backend="pyradimics")  # typo
    with pytest.raises(ValueError, match="backend="):
        RadiomicFeatureExtractor(backend="gpu")


#: These three tests are about a BRANCH on ``_fastrad_available()``, so they
#: patch it rather than depending on what happens to be installed. They asserted
#: the un-patched value before, which made them pass here (no ``fastrad``) and
#: fail on the cluster, where something answering to that name IS importable --
#: 3 of cluster job 8004252's failures. A test named "...when_fastrad_absent"
#: should MAKE it absent.
_AVAIL = "spectramr.core.metrics.radiomic.RadiomicFeatureExtractor._fastrad_available"


def test_explicit_fastrad_raises_not_implemented() -> None:
    """Selecting fastrad explicitly must fail loud while it is unwired."""
    from unittest.mock import patch

    from spectramr.core.metrics.radiomic import RadiomicFeatureExtractor

    with (
        patch(_AVAIL, return_value=False),
        pytest.raises(NotImplementedError, match="fastrad"),
    ):
        RadiomicFeatureExtractor(backend="fastrad")


def test_resolve_backend_priority_is_documented() -> None:
    """The auto-dispatch priority must put fastrad ahead of pyradiomics."""
    from spectramr.core.metrics.radiomic import RadiomicFeatureExtractor

    assert RadiomicFeatureExtractor.BACKEND_PRIORITY[0] == "fastrad"
    assert "pyradiomics" in RadiomicFeatureExtractor.BACKEND_PRIORITY


def test_resolve_backend_auto_falls_through_to_pyradiomics_when_fastrad_absent() -> None:
    """When fastrad is not installed (the current state), auto must resolve to pyradiomics."""
    from spectramr.core.metrics.radiomic import (
        RADIOMICS_AVAILABLE,
        RadiomicFeatureExtractor,
    )

    if not RADIOMICS_AVAILABLE:
        pytest.skip("pyradiomics not installed; both backends unavailable on this host")

    # Patched, not observed: the point is the fall-through BRANCH, and reading
    # the live probe made the assertion a statement about the host instead.
    from unittest.mock import patch

    with patch(_AVAIL, return_value=False):
        assert RadiomicFeatureExtractor._resolve_backend("auto") == "pyradiomics"


def test_rfs_fallback_is_bounded_near_zero_features() -> None:
    """Without a population std, a near-zero GT feature must not blow up RFS.

    Regression: the fallback ``std = |gt_feat| + 1e-8`` divided a small
    difference by ~1e-8 for near-zero features (skewness, correlation, ...),
    dragging RFS to a huge negative value. The fix floors each feature's scale
    at a fraction of the largest feature magnitude. We bypass the heavy
    pyradiomics ``__init__`` via ``__new__`` and inject a stub extractor.
    """
    import math

    import numpy as np

    from spectramr.core.metrics.radiomic import RadiomicFeatureStability

    gt_feat = np.array([1.0, 100.0, 1e-9])  # third feature ~ 0
    recon_feat = np.array([1.1, 101.0, 0.5])  # diffs: 0.1, 1.0, 0.5

    class _Feat:
        def __init__(self, features):
            self.features = features

    class _StubExtractor:
        def __init__(self, seq):
            self._seq = list(seq)
            self._i = 0

        def extract_from_array(self, _arr, _mask=None):
            f = self._seq[self._i]
            self._i += 1
            return _Feat(f)

    rfs = RadiomicFeatureStability.__new__(RadiomicFeatureStability)
    rfs.population_std = None
    rfs.extractor = _StubExtractor([recon_feat, gt_feat])

    score = rfs.compute(np.zeros((4, 4)), np.zeros((4, 4)))
    assert math.isfinite(score)
    # Pre-fix this was ~ -1.7e7; the floored scale keeps it bounded.
    assert score > -100.0


def test_resolve_backend_raises_when_no_backend_available() -> None:
    """When neither backend is importable, auto-dispatch must fail loud."""
    from spectramr.core.metrics.radiomic import (
        RADIOMICS_AVAILABLE,
        RadiomicFeatureExtractor,
    )

    if RADIOMICS_AVAILABLE:
        pytest.skip("pyradiomics is installed; cannot exercise the no-backend branch")

    with pytest.raises(ImportError, match="No radiomic backend"):
        RadiomicFeatureExtractor._resolve_backend("auto")


def test_resolved_backend_is_stamped_on_instance() -> None:
    """Every instance must expose ``self.backend`` for provenance reporting.

    Pitfall #15 -- self-describing artefacts. The cluster's results
    JSON will include this value so downstream tooling knows which
    backend produced the features.
    """
    from spectramr.core.metrics.radiomic import (
        RADIOMICS_AVAILABLE,
        RadiomicFeatureExtractor,
    )

    if not RADIOMICS_AVAILABLE:
        pytest.skip("pyradiomics not installed; cannot instantiate the CPU path")

    from unittest.mock import patch

    with patch(_AVAIL, return_value=False):
        extractor = RadiomicFeatureExtractor(backend="auto")
    assert extractor.backend == "pyradiomics"


# ---------------------------------------------------------------------------
# SimpleITK / scipy optional-dependency guards (6.1). Both are REQUIRED in
# pyproject; the guards raise a clear error when a metric is USED in a
# degraded cluster env, so the module still imports and walk-discovery still
# registers the other metrics.
# ---------------------------------------------------------------------------


def test_module_exposes_simpleitk_and_scipy_flags() -> None:
    """The availability flags exist so audits can introspect the env."""
    import spectramr.core.metrics.radiomic as r

    assert isinstance(r.SIMPLEITK_AVAILABLE, bool)
    assert isinstance(r.SCIPY_AVAILABLE, bool)


def test_extractor_raises_clear_error_without_simpleitk(monkeypatch) -> None:
    """Constructing the extractor without SimpleITK raises a clear error
    naming the missing dependency, not an obscure ``NoneType`` crash."""
    import spectramr.core.metrics.radiomic as r

    monkeypatch.setattr(r, "SIMPLEITK_AVAILABLE", False)
    with pytest.raises(ImportError, match="SimpleITK"):
        r.RadiomicFeatureExtractor(backend="auto")


def test_frd_raises_clear_error_without_scipy(monkeypatch) -> None:
    """Constructing the Fréchet distance metric without scipy raises a clear
    error naming scipy (used for the covariance matrix square root)."""
    import spectramr.core.metrics.radiomic as r

    monkeypatch.setattr(r, "SCIPY_AVAILABLE", False)
    with pytest.raises(ImportError, match="scipy"):
        r.FrechetRadiomicDistance()


def test_frechet_distance_runs_on_current_scipy() -> None:
    """``sqrtm(..., disp=False)`` raised TypeError on SciPy >= 1.18; the distance must still compute."""
    import numpy as np

    from spectramr.core.metrics.radiomic import FrechetRadiomicDistance

    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((50, 4)), rng.standard_normal((50, 4)) + 1.0
    d = FrechetRadiomicDistance._frechet_distance(
        a.mean(0), np.cov(a.T), b.mean(0), np.cov(b.T)
    )
    assert np.isfinite(d) and d > 1.0
    same = FrechetRadiomicDistance._frechet_distance(a.mean(0), np.cov(a.T), a.mean(0), np.cov(a.T))
    assert same == pytest.approx(0.0, abs=1e-6)
