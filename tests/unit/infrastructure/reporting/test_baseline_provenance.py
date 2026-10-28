"""Tests for ``baseline_provenance`` — Phase A.4 of the baseline backlog.

Locks ``TODO/backlog_baseline_replication_experiment_11.md`` Phase A.4:
every baseline run records enough metadata that a reviewer can identify
the exact code path that produced the published numbers.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.reporting.metadata import baseline_provenance
from spectramr.models.baselines import BaselineAdapter, CoilHandling, FFTNorm


class _ExampleBaselineAdapter(BaselineAdapter):
    """Reference adapter — every required attribute is set."""

    REPO_NAME = "fake_baseline"
    PAPER_REF = "arxiv:0000.00000"
    PREFERRED_MASK_TYPE = "equispaced"
    PREFERRED_FFT_NORM = FFTNorm.ORTHO
    COIL_HANDLING = CoilHandling.RSS

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return x


def test_provenance_dict_has_declared_keys() -> None:
    """The result lists every declared class attribute the contract requires."""
    prov = baseline_provenance(_ExampleBaselineAdapter)
    assert prov["repo_name"] == "fake_baseline"
    assert prov["paper_ref"] == "arxiv:0000.00000"
    assert prov["preferred_mask_type"] == "equispaced"
    assert prov["preferred_fft_norm"] == "ortho"
    assert prov["coil_handling"] == "rss"


def test_provenance_includes_adapter_module_and_qualname() -> None:
    """Module + qualname pinpoint the adapter class for git-blame lookups."""
    prov = baseline_provenance(_ExampleBaselineAdapter)
    assert prov["adapter_qualname"] == "_ExampleBaselineAdapter"
    assert "test_baseline_provenance" in prov["adapter_module"]


def test_provenance_records_this_repo_commit_when_git_available() -> None:
    """``adapter_commit`` is a SHA-like string when running inside a git repo."""
    prov = baseline_provenance(_ExampleBaselineAdapter)
    commit = prov["adapter_commit"]
    # In a clean checkout the commit string is a 40-char hex SHA. In a
    # detached / non-git environment it may be ``None``; both are OK.
    if commit is not None:
        assert len(commit) == 40
        assert all(c in "0123456789abcdef" for c in commit.lower())


def test_provenance_returns_none_for_absent_vendored_repo() -> None:
    """If ``external/baselines/<REPO_NAME>/`` does not exist, ``upstream_commit`` is None."""
    prov = baseline_provenance(_ExampleBaselineAdapter)
    # "fake_baseline" is not a real vendored repo, so upstream_commit must be None.
    assert prov["upstream_commit"] is None


def test_provenance_picks_up_vendored_cdiffmr_commit_sha() -> None:
    """When ``external/baselines/cdiffmr/`` is vendored, its commit SHA shows up."""
    from spectramr_baselines.cdiffmr import CDiffMRBaseline

    prov = baseline_provenance(CDiffMRBaseline)
    # Vendored — must produce a 40-char hex SHA, not None.
    assert prov["upstream_commit"] is not None
    assert len(prov["upstream_commit"]) == 40
    assert all(c in "0123456789abcdef" for c in prov["upstream_commit"].lower())


def test_provenance_picks_up_vendored_fdb_commit_sha() -> None:
    """When ``external/baselines/fdb/`` is vendored, its commit SHA shows up."""
    from spectramr_baselines.fdb import FDBBaseline

    prov = baseline_provenance(FDBBaseline)
    assert prov["upstream_commit"] is not None
    assert len(prov["upstream_commit"]) == 40
