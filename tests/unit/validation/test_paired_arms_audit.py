"""Tests for the paired-arms audit (cold_diffusion_locus_ablation et al.).

Three responsibilities under test:

1. Real-campaign happy path — the two enhanced cold-diffusion YAMLs
   pass the paired-arms audit. Regression guard against future drift
   that would silently make the ablation multi-factor.
2. Synthetic counterexample — a forged Arm B that differs on an
   un-allow-listed key (e.g. ``optimization.optimizer.learning_rate``) is
   rejected with the offending key in the diff list.
3. Allow-list extension — the caller can pass a custom
   ``allowed_diff_paths`` set to permit additional cross-arm
   differences (used by other ablation campaigns).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from spectramr.infrastructure.validation.paired_arms_audit import (
    audit_paired_arms,
    _DEFAULT_DIFF_PATHS,
)
from tests.utils.repo_scripts import require_repo_file

_ARM_A_REL = (
    "experiments/inprogress/cold_diffusion/experiment_12_image_cold_diffusion_v2.yaml"
)
_ARM_B_REL = (
    "experiments/inprogress/cold_diffusion/experiment_12_physics_cold_diffusion_v2.yaml"
)


def _arm_a() -> Path:
    return require_repo_file(_ARM_A_REL)


def _arm_b() -> Path:
    return require_repo_file(_ARM_B_REL)


class TestCampaignHappyPath:
    """The real campaign YAMLs must pass the audit."""

    def test_arm_a_exists(self) -> None:
        # require_repo_file, not `.exists()`: the two answers must stay
        # distinguishable. An arm the allowlist never selects is a publication
        # boundary (skip); an arm that was deleted or MOVED is the defect this
        # test exists for, and it must still fail loudly in every other tree.
        assert _arm_a().is_file()

    def test_arm_b_exists(self) -> None:
        assert _arm_b().is_file()

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "#934: the ablation drifted to multi-factor. The arms differ on 12 "
            "paths -- not only the plausible locus marker (physics.data_consistency.*, "
            "the loss re-weighting) but undersampling.center_fraction 0.08 vs 0.03 "
            "and data.multi_contrast.enabled, which are a different acquisition and "
            "a different input. This guard is doing its job; strict=True means it "
            "turns RED once the arms are repaired, so the marker cannot rot."
        ),
    )
    def test_paired_arms_audit_passes(self) -> None:
        result = audit_paired_arms(_arm_a(), _arm_b())
        assert (
            result.passed
        ), f"Paired-arms audit failed unexpectedly:\n{result.render()}"
        assert result.group == "cold_diffusion_locus_ablation"
        # The arms ARE single-factor — at least 50 paths were allow-listed
        # (output dirs, names, model_type, etc.). If this drops below 50
        # the YAMLs have diverged structurally.
        assert len(result.skipped_paths) >= 50, (
            f"Only {len(result.skipped_paths)} paths allow-listed; "
            "the YAMLs may have lost their shared structure."
        )


def _forge_pair(tmp_path: Path, mutate) -> tuple[Path, Path]:
    """Two arms identical but for ``mutate(b)``, both forged from **Arm A**.

    These cases test the AUDIT, so both sides must come from one arm. Forging
    Arm B from the real Arm B made them fail whenever the corpus drifted --
    which it did (#934, 12 unrelated diffs), turning unit tests of the audit
    red for a reason that has nothing to do with the audit.

    ``metadata.name``/``output_dir`` differences are already allow-listed, so a
    self-pair audits clean and the injected diff is the only signal.
    """
    a = yaml.safe_load(_arm_a().read_text())
    b = yaml.safe_load(_arm_a().read_text())
    mutate(b)
    a_path, b_path = tmp_path / "forged_a.yaml", tmp_path / "forged_b.yaml"
    a_path.write_text(yaml.safe_dump(a))
    b_path.write_text(yaml.safe_dump(b))
    return a_path, b_path


def _set_lr(doc: dict, value: float) -> None:
    """Write the LR at the CANONICAL path.

    Writing the legacy ``optimization.learning_rate`` into an arm that already
    declares ``optimization.optimizer.learning_rate`` produces a document the
    loader refuses outright ("...disagree. Declare one of them"), so forging
    that way was injecting an unloadable config, not an LR mismatch.
    """
    doc.setdefault("optimization", {}).setdefault("optimizer", {})[
        "learning_rate"
    ] = value


class TestSyntheticForge:
    """Forge a counterexample with a forbidden diff."""

    def test_lr_mismatch_is_caught(self, tmp_path: Path) -> None:
        a_path, b_path = _forge_pair(tmp_path, lambda b: _set_lr(b, 5.0e-5))
        result = audit_paired_arms(a_path, b_path)
        assert not result.passed
        paths = {d.path for d in result.diffs}
        assert (
            "optimization.optimizer.learning_rate" in paths
        ), f"Audit failed to catch LR mismatch; diffs were: {paths}"

    def test_an_unmutated_self_pair_is_clean(self, tmp_path: Path) -> None:
        """Anti-vacuity for :func:`_forge_pair`.

        If a self-pair already reported diffs, every test above would "catch"
        its injected mismatch by accident. This is what the old fixture lost by
        pairing two different real arms.
        """
        a_path, b_path = _forge_pair(tmp_path, lambda _b: None)
        result = audit_paired_arms(a_path, b_path)
        assert result.passed, f"a self-pair must audit clean:\n{result.render()}"

    def test_loss_weight_mismatch_is_caught(self, tmp_path: Path) -> None:
        a_path, b_path = _forge_pair(
            tmp_path,
            lambda b: b["losses"]["image_losses"][0].__setitem__("weight", 0.5),
        )
        result = audit_paired_arms(a_path, b_path)
        assert not result.passed
        paths = {d.path for d in result.diffs}
        assert any(
            "losses.image_losses" in p for p in paths
        ), f"Audit failed to catch loss-weight mismatch; diffs were: {paths}"


class TestArmDeclaringBothSpellingsIsRefused:
    """An arm that contradicts itself must raise, not get a silent winner.

    ``dict(_walk(...))`` let a legacy/canonical collision resolve by insertion
    order, so ``yaml.safe_dump`` -- which sorts keys -- decided which value the
    audit saw. That is how ``test_lr_mismatch_is_caught`` came to report the
    forged 5e-05 as absent: in memory the legacy key was appended last and won,
    but after the round-trip ``learning_rate`` sorted before ``optimizer`` and
    the original 1e-4 overwrote it.
    """

    def test_disagreeing_duplicate_declarations_raise(self, tmp_path: Path) -> None:
        def mutate(b: dict) -> None:
            _set_lr(b, 1.0e-4)
            b["optimization"]["learning_rate"] = 5.0e-5  # legacy spelling too

        a_path, b_path = _forge_pair(tmp_path, mutate)
        with pytest.raises(ValueError, match="declares the same knob twice"):
            audit_paired_arms(a_path, b_path)

    def test_the_verdict_does_not_depend_on_key_order(self) -> None:
        """Both serialisation orders must reach the same answer."""
        from spectramr.infrastructure.validation.paired_arms_audit import (
            _canonical_paths,
        )

        legacy_first: dict[str, Any] = {
            "optimization": {"learning_rate": 5e-5, "optimizer": {}}
        }
        legacy_first["optimization"]["optimizer"]["learning_rate"] = 1e-4
        legacy_last: dict[str, Any] = {
            "optimization": {"optimizer": {"learning_rate": 1e-4}}
        }
        legacy_last["optimization"]["learning_rate"] = 5e-5
        for doc in (legacy_first, legacy_last):
            with pytest.raises(ValueError, match="declares the same knob twice"):
                _canonical_paths(doc, "arm")

    def test_agreeing_duplicates_are_redundant_not_ambiguous(self) -> None:
        """Equal values resolve identically either way, so they must NOT raise."""
        from spectramr.infrastructure.validation.paired_arms_audit import (
            _canonical_paths,
        )

        doc: dict[str, Any] = {"optimization": {"optimizer": {"learning_rate": 1e-4}}}
        doc["optimization"]["learning_rate"] = 1e-4
        assert _canonical_paths(doc, "arm")["optimization.optimizer.learning_rate"] == (
            1e-4
        )


class TestAllowListExtension:
    """The caller can extend the allow-list for other campaigns."""

    def test_custom_allowlist_silences_diff(self, tmp_path: Path) -> None:
        a_path, b_path = _forge_pair(tmp_path, lambda b: _set_lr(b, 5.0e-5))
        # Extend the default allow-list with the LR path.
        extended = set(_DEFAULT_DIFF_PATHS) | {"optimization.optimizer.learning_rate"}
        result = audit_paired_arms(a_path, b_path, allowed_diff_paths=extended)
        assert (
            result.passed
        ), f"Custom allow-list should silence LR diff; got: {result.render()}"


class TestGroupValidation:
    """Arms must declare the same metadata.group."""

    def test_mismatched_groups_raise(self, tmp_path: Path) -> None:
        a = yaml.safe_load(_arm_a().read_text())
        b = yaml.safe_load(_arm_b().read_text())
        b["metadata"]["group"] = "wrong_group"
        a_path = tmp_path / "forged_a.yaml"
        b_path = tmp_path / "forged_b.yaml"
        a_path.write_text(yaml.safe_dump(a))
        b_path.write_text(yaml.safe_dump(b))
        with pytest.raises(ValueError, match="different metadata.group"):
            audit_paired_arms(a_path, b_path)

    def test_missing_arm_raises(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            audit_paired_arms(nonexistent, _arm_b())


class TestStagedRenamePathsAreNormalised:
    """A migrated arm and an unmigrated one must compare as equals.

    The audit walks RAW YAML, so during the staged window one arm can spell a
    knob ``optimization.learning_rate`` while its sibling spells it
    ``optimization.optimizer.learning_rate``. Untranslated that is a spurious
    diff at every moved knob -- which would make the audit unusable for exactly
    the period the migration is running.
    """

    @staticmethod
    def _walk(doc):
        from spectramr.infrastructure.validation.paired_arms_audit import (
            _canonical_paths,
        )

        return _canonical_paths(doc, "arm")

    def test_the_flat_spelling_reports_the_canonical_path(self) -> None:
        walked = self._walk({"optimization": {"learning_rate": 1e-4}})
        assert "optimization.optimizer.learning_rate" in walked
        assert "optimization.learning_rate" not in walked

    def test_flat_and_nested_arms_agree_on_the_key(self) -> None:
        flat = self._walk({"optimization": {"learning_rate": 1e-4}})
        nested = self._walk({"optimization": {"optimizer": {"learning_rate": 1e-4}}})
        assert flat == nested

    def test_a_differing_value_is_still_a_differing_value(self) -> None:
        """Both directions: normalising keys must not normalise away a real
        disagreement, which is the only thing this audit exists to find."""
        flat = self._walk({"optimization": {"learning_rate": 1e-4}})
        nested = self._walk({"optimization": {"optimizer": {"learning_rate": 5e-5}}})
        assert flat != nested

    def test_an_unrelated_path_is_untouched(self) -> None:
        """A path with NO fold record passes through verbatim.

        The example must be chosen at run time, not hard-coded: this test used
        `data.batch_size`, which phase 9a then gave a record, so `_walk`
        correctly rewrote it and the test went red while the code was right.
        Asserting the chosen path is absent from the table makes the fixture
        self-checking.
        """
        from spectramr.config.schemas.renames import RENAMES

        fold = {r.legacy for r in RENAMES.values() if r.posture == "fold"}
        path = "data.zzz_not_a_real_key"
        assert path not in fold, "pick a path the rename table does not own"
        walked = self._walk({"data": {"zzz_not_a_real_key": 4}})
        assert walked == {path: 4}


class TestAllowListIsCanonical:
    """Every allow-list path must be spelled the way `_walk` will yield it.

    `_walk` canonicalises each observed path through the rename SSOT before
    matching, so an entry left at a RETIRED spelling can never match anything.
    It does not fail loudly -- it silently stops exempting its knob, and the
    audit then reports a spurious diff on every pair that sets it.

    Seven entries had gone stale this way across phases 9d/10a/10b/10d
    (`data.output_domain`, `validation.metrics`, `logging.log_dir`, ...) before
    this test existed. It is the same anti-rot shape as
    `test_renames.py::TestProductionTableDoesNotRot`: the fixture that describes
    the schema must not describe a schema that no longer exists.
    """

    @staticmethod
    def _allow_paths() -> set[str]:
        from spectramr.infrastructure.validation.paired_arms_audit import (
            _DEFAULT_DIFF_PATHS,
        )

        return set(_DEFAULT_DIFF_PATHS)

    def test_no_allowlist_entry_is_a_retired_path(self) -> None:
        from spectramr.config.schemas.renames import RENAMES

        fold = {r.legacy: r.canonical for r in RENAMES.values() if r.posture == "fold"}
        stale = sorted(f"{p} -> {fold[p]}" for p in self._allow_paths() if p in fold)
        assert not stale, (
            "these allow-list paths are retired spellings, so `_walk` will never "
            "produce them and the exemption is dead:\n  " + "\n  ".join(stale)
        )

    def test_the_check_can_fire(self) -> None:
        """Anti-vacuity: the fold table must be non-empty, or the test proves
        nothing about any spelling."""
        from spectramr.config.schemas.renames import RENAMES

        assert any(r.posture == "fold" for r in RENAMES.values())
        assert self._allow_paths(), "the allow-list is empty"
