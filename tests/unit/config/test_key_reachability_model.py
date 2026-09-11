"""Pins for the verdict types in ``key_reachability_model``.

The module is the leaf of the chain ``model <- collect <- index <-
key_reachability`` and had no test file of its own; ``ACCESSOR_READ`` and
:func:`accessor_verdict` arrived with #1925 and are pinned here rather than
through the facade, so a change to the vocabulary fails next to the vocabulary.
"""

from __future__ import annotations

from spectramr.config.key_reachability_model import (
    ReachabilityVerdict,
    ReadEvidence,
    accessor_verdict,
)


class TestReadEvidenceVocabulary:
    def test_the_four_kinds_are_the_whole_vocabulary(self) -> None:
        """A fifth kind must be a deliberate edit, not a drive-by addition.

        Consumers branch on this enum to decide wire-or-delete, and a kind
        nobody branches on is read as "not reachable" by an ``is`` comparison
        chain that does not know about it.
        """
        assert {e.value for e in ReadEvidence} == {
            "live_read",
            "accessor_read",
            "no_live_read",
            "no_read_found",
        }

    def test_accessor_read_is_distinct_from_live_read(self) -> None:
        """Both mean "read", and the difference is what the evidence IS.

        ``LIVE_READ`` is a call-graph finding this module derived; ``ACCESSOR_READ``
        is a declaration a human made after reading the consumer. Collapsing them
        would hide which verdicts rest on an export that could be wrong.
        """
        assert ReadEvidence.ACCESSOR_READ is not ReadEvidence.LIVE_READ


class TestAccessorVerdict:
    def test_it_reports_a_read_and_names_the_accessor(self) -> None:
        verdict = accessor_verdict("losses.gan.lambda_gp", "lambda_gp", (), "the weight table")
        assert isinstance(verdict, ReachabilityVerdict)
        assert verdict.reachable
        assert verdict.evidence is ReadEvidence.ACCESSOR_READ
        assert "losses.gan.lambda_gp" in verdict.reason
        assert "the weight table" in verdict.reason

    def test_it_neither_invents_nor_drops_sites(self) -> None:
        """``sites`` are AST facts; this verdict may not manufacture or lose one.

        A token-less read has no `file:line` naming the leaf, so the empty case
        must stay empty -- a fabricated site would be read as evidence by the
        ledger tests that ask whether an entry really has read sites. And a key
        that *does* have (dead) sites must keep them, or it gets refiled under
        the wrong ledger section.
        """
        assert accessor_verdict("a.b.c", "c", (), "x").sites == ()
        sites = ("ema.py:12", "ema.py:44")
        assert accessor_verdict("a.b.c", "c", sites, "x").sites == sites
