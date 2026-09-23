"""Three identities live in one M4Raw file name, and conflating them inflates a bound.

M4Raw ships ``<patient>_<contrast><NN>.h5``. The patient is the **independence**
unit a risk certificate counts; ``(patient, contrast)`` is the repetition group
the NEX target averages; the trailing digits say which acquisition. Grouping on
the wrong one does not raise -- it produces a certificate that is too tight,
which is the failure #1707 describes.

The grouping rule was written twice before this module (the dataset's
``stem[:-2]`` and the manifest generator's ``file_id[:-2]``), so the drift test
below is not hypothetical.
"""

from __future__ import annotations

import math

import pytest

from spectramr.data.datasets.m4raw_identity import (
    parse_m4raw_file_id,
    repetition_group_key,
)


class TestItParsesTheConvention:
    @pytest.mark.parametrize(
        ("file_id", "subject", "contrast", "repetition"),
        [
            ("2022061001_T101", "2022061001", "T1", "01"),
            ("2022061001_T202", "2022061001", "T2", "02"),
            ("2022061001_FLAIR01", "2022061001", "FLAIR", "01"),
            ("2022061002_T103", "2022061002", "T1", "03"),
        ],
    )
    def test_the_three_parts(
        self, file_id: str, subject: str, contrast: str, repetition: str
    ) -> None:
        identity = parse_m4raw_file_id(file_id)
        assert (identity.subject, identity.contrast, identity.repetition) == (
            subject,
            contrast,
            repetition,
        )

    def test_the_group_is_patient_and_contrast(self) -> None:
        """NOT the patient: the NEX target averages within one contrast."""
        assert parse_m4raw_file_id("2022061001_T101").repetition_group == "2022061001_T1"

    def test_two_contrasts_of_one_patient_share_a_subject(self) -> None:
        """The distinction the whole module exists for."""
        t1 = parse_m4raw_file_id("2022061001_T101")
        t2 = parse_m4raw_file_id("2022061001_T202")
        assert t1.subject == t2.subject
        assert t1.repetition_group != t2.repetition_group


class TestItDoesNotGuessOnUnparseableNames:
    """A wrong guess MERGES two patients; a refusal only splits one."""

    @pytest.mark.parametrize("file_id", ["odd", "ab", "x", ""])
    def test_a_short_name_is_its_own_group(self, file_id: str) -> None:
        identity = parse_m4raw_file_id(file_id)
        assert identity.subject == file_id
        assert identity.repetition_group == file_id
        assert identity.repetition == ""

    def test_a_non_numeric_suffix_is_not_read_as_a_repetition(self) -> None:
        identity = parse_m4raw_file_id("patient_scanAB")
        assert identity.repetition == ""
        assert identity.repetition_group == "patient_scanAB"

    def test_a_name_without_a_separator_claims_no_contrast(self) -> None:
        identity = parse_m4raw_file_id("abcdef01")
        assert identity.repetition == "01"
        assert identity.contrast == ""
        assert identity.subject == "abcdef"


class TestTheGroupingRuleHasOneOwner:
    """It was written twice, and `src/` cannot import `scripts/`, so the
    generator delegates to the package rather than mirroring it."""

    def test_the_manifest_generator_delegates(self) -> None:
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[4] / "scripts/data/regenerate_cluster_manifests.py"
        spec = importlib.util.spec_from_file_location("gen", path)
        assert spec is not None and spec.loader is not None
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        for file_id in ("2022061001_T101", "2022061001_FLAIR02", "odd", "ab"):
            assert generator.repetition_group_key(file_id) == repetition_group_key(file_id)

    def test_the_dataset_uses_the_shared_key(self) -> None:
        import inspect

        from spectramr.data.datasets import m4raw_dataset

        source = inspect.getsource(m4raw_dataset)
        assert "repetition_group_key(stem)" in source, "the dataset re-implements the rule"


class TestWhyTheUnitMatters:
    """The bound, not the grouping, is what a reader acts on."""

    @staticmethod
    def _corpus() -> list[str]:
        """M4Raw's shape: 128 patients x (T1 x3, T2 x3, FLAIR x2)."""
        return [
            f"{2022061000 + s}_{contrast}{rep:02d}"
            for s in range(1, 129)
            for contrast, reps in (("T1", 3), ("T2", 3), ("FLAIR", 2))
            for rep in range(1, reps + 1)
        ]

    @staticmethod
    def _half_width(n: int, delta: float = 0.05) -> float:
        return math.sqrt(0.5 * math.log(2 / delta) / n)

    def test_subjects_are_far_fewer_than_repetition_groups(self) -> None:
        corpus = self._corpus()
        subjects = {parse_m4raw_file_id(f).subject for f in corpus}
        groups = {parse_m4raw_file_id(f).repetition_group for f in corpus}
        assert len(subjects) == 128
        assert len(groups) == 384

    def test_the_wrong_unit_makes_the_certificate_look_tighter(self) -> None:
        """12x on slices, 1.7x on repetition groups -- neither of which raises."""
        corpus = self._corpus()
        subjects = len({parse_m4raw_file_id(f).subject for f in corpus})
        groups = len({parse_m4raw_file_id(f).repetition_group for f in corpus})
        slices = len(corpus) * 18
        assert self._half_width(subjects) > 1.7 * self._half_width(groups)
        assert self._half_width(subjects) > 10.0 * self._half_width(slices)
