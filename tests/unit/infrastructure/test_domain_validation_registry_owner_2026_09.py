"""Loss domains have one owner: the ``@register_loss(domain=...)`` annotation.

``domain_validation`` carried a second, hand-maintained table of 36 entries
beside the registry's 148. Characterized before removing it, as non-negotiable 17
requires: of the 19 names both annotated, 18 agreed and ONE disagreed --
``complex_l1``, hand-written ``kspace`` against the decorator's ``agnostic``. The
decorator is right, an elementwise L1 is defined on any tensor, and the hand
entry would have flagged a legal declaration. 184 of 220 registered losses were
absent from the hand table while reading as checked.
"""

from __future__ import annotations

import pytest

import spectramr.infrastructure.domain_validation as dv
from spectramr.infrastructure.domain_validation import _loss_domain_of


class TestTheSecondOwnerIsGone:
    @pytest.mark.parametrize("name", ["LOSS_DOMAIN_REGISTRY", "register_loss_domain"])
    def test_the_hand_maintained_surface_is_deleted(self, name) -> None:
        """Keeping a weaker checker as defence in depth is what the rule forbids;
        the loser's enforcement goes, not just its use."""
        assert not hasattr(dv, name), (
            f"{name} still exists -- a second table beside the registry is how "
            "the two start to disagree"
        )


class TestItReadsTheRegistry:
    def test_the_one_name_the_two_owners_disagreed_on(self) -> None:
        """``complex_l1``: hand table said kspace, the decorator says agnostic."""
        domain, agnostic, registered = _loss_domain_of("complex_l1")
        assert registered and agnostic and domain is None

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("hfen", "image"), ("null_space_content", "kspace")],
    )
    def test_an_annotated_loss_reports_its_domain(self, name, expected) -> None:
        assert _loss_domain_of(name)[0] == expected

    def test_a_loss_the_hand_table_never_had(self) -> None:
        """``coil_subspace_residual`` was absent from the old table entirely, so
        it was unchecked while every artifact read clean."""
        domain, _, registered = _loss_domain_of("coil_subspace_residual")
        assert registered and domain == "complex_image"


class TestAbsentIsReportedNotInferred:
    """Three different absences, three different answers (non-negotiable 18)."""

    def test_registered_but_unannotated(self) -> None:
        """``physics`` has no ``Domain`` equivalent, so the capability yields
        ``None`` rather than a guess. Skip, and say it is unannotated."""
        domain, agnostic, registered = _loss_domain_of("bloch_residual")
        assert registered and not agnostic and domain is None

    def test_not_registered_at_all(self) -> None:
        domain, agnostic, registered = _loss_domain_of("definitely_not_a_loss")
        assert not registered and not agnostic and domain is None

    def test_agnostic_is_a_positive_claim_not_an_absence(self) -> None:
        """The distinction that keeps an un-audited loss and a deliberately
        generic one from looking alike."""
        _, unannotated_agnostic, _ = _loss_domain_of("bloch_residual")
        _, declared_agnostic, _ = _loss_domain_of("complex_l1")
        assert declared_agnostic and not unannotated_agnostic


class TestTheCorpusStillPasses:
    def test_the_shipped_cohort_arm_produces_no_domain_errors(self) -> None:
        """The registry annotates 4x as many losses as the hand table, so the
        check got strictly wider. Measured across all 672 inprogress arms at the
        time of the change: 0 errors. This pins one of them."""
        from spectramr.config.settings import TrainingSettings

        cfg = TrainingSettings.from_yaml(
            "experiments/inprogress/kspace_filling/attention_shootout/"
            "experiment_11_attention_none.yaml"
        )
        result = dv.validate_loss_domains(cfg.losses, "kspace")
        assert result.errors == [], result.errors
