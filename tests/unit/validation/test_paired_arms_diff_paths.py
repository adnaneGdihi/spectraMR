"""Unit tests for the extracted paired-arms diff-path allow-list.

The allow-list moved out of ``paired_arms_audit`` in the Wave 0 exit-criterion
split (#1400). These tests pin the two properties that make the extraction safe:
the audit module still exposes the *same object* under its historical name, and
every entry is a canonical spelling the audit's own path walker can actually
produce.
"""

from __future__ import annotations

from spectramr.infrastructure.validation.paired_arms_diff_paths import DEFAULT_DIFF_PATHS


class TestAllowListIdentity:
    def test_audit_module_reexports_the_same_object(self) -> None:
        """One frozenset, one owner (NN17) -- not a copy that can drift."""
        from spectramr.infrastructure.validation import paired_arms_audit

        assert paired_arms_audit._DEFAULT_DIFF_PATHS is DEFAULT_DIFF_PATHS

    def test_allow_list_is_a_non_empty_frozenset_of_str(self) -> None:
        assert isinstance(DEFAULT_DIFF_PATHS, frozenset)
        assert DEFAULT_DIFF_PATHS, "an empty allow-list makes every pin vacuous"
        assert all(isinstance(p, str) for p in DEFAULT_DIFF_PATHS)


class TestAllowListSpellings:
    def test_no_allowlist_entry_is_a_retired_path(self) -> None:
        """A retired spelling never matches, so its exemption is silently dead."""
        from spectramr.config.schemas.renames import RENAMES

        fold = {r.legacy: r.canonical for r in RENAMES.values() if r.posture == "fold"}
        stale = sorted(f"{p} -> {fold[p]}" for p in DEFAULT_DIFF_PATHS if p in fold)
        assert not stale, (
            "these allow-list paths are retired spellings, so the audit's path "
            "walker will never produce them and the exemption is dead:\n  "
            + "\n  ".join(stale)
        )

    def test_the_retired_path_check_can_fire(self) -> None:
        """Anti-vacuity: an empty fold table would make the check above pass blind."""
        from spectramr.config.schemas.renames import RENAMES

        assert any(r.posture == "fold" for r in RENAMES.values())

    def test_every_entry_is_dotted_and_stripped(self) -> None:
        bad = sorted(p for p in DEFAULT_DIFF_PATHS if p != p.strip() or not p)
        assert not bad, f"malformed allow-list entries: {bad}"


class TestNexTargetAblationIsRepresentable:
    """The list enumerates the factors an ablation may vary, not just identity.

    Before ``data.target_mode`` / ``data.r2r_alpha`` landed it carried three
    ``data.*`` paths and none was a target mode, so **no** repetition-target
    ablation could pass however cleanly it was built. The n2n cohort varies
    exactly those two and was reported as three spurious diffs across its three
    pairs -- a finding about the allow-list, not about the cohort.
    """

    def test_the_nex_ablation_factors_are_exempt(self) -> None:
        for path in ("data.target_mode", "data.r2r_alpha", "metadata.expected_outcome"):
            assert path in DEFAULT_DIFF_PATHS, f"{path} is not exempt"

    def test_the_n2n_cohort_passes_on_every_pair(self) -> None:
        """The corpus leg: a list entry that matches nothing exempts nothing."""
        import itertools
        from pathlib import Path

        import pytest

        from spectramr.infrastructure.validation.paired_arms_audit import audit_paired_arms

        cohort = Path(__file__).resolve().parents[3] / "experiments/inprogress/n2n"
        arms = sorted(cohort.glob("*.yaml"))
        if len(arms) < 2:
            pytest.skip("n2n cohort not present")

        for a, b in itertools.combinations(arms, 2):
            report = audit_paired_arms(str(a), str(b))
            assert report.passed, (
                f"{a.stem} vs {b.stem} reports unexempted diffs: "
                f"{getattr(report, 'diffs', None)}"
            )

    def test_an_unrelated_knob_is_still_caught(self) -> None:
        """Anti-vacuity: the exemptions must not have blanketed ``data.*``."""
        assert "data.loader.batch_size" not in DEFAULT_DIFF_PATHS
        assert "data.coils.processing_mode" not in DEFAULT_DIFF_PATHS
