"""``latent_losses`` is a declared loss list, so every "walk the lists" site must see it.

The schema (:data:`spectramr.config.schemas.loss.LOSS_LIST_DOMAINS`) declares **four**
declarative loss lists. Before #1924 the repo carried a second, divergent table for the
same fact -- ``weights.LOSS_LISTS``, a hand-written three-element tuple -- plus a dozen
inline re-spellings of the same three names. Two owners for one invariant (non-negotiable
17), and the loser was the one that decided what a loss *weighs*.

The failure was silent in **both** directions, because
:func:`~spectramr.models.losses.weights.build_loss_weight_table` cross-checks two
surfaces: ``_declared_lambdas`` walks :data:`LAMBDA_SECTIONS`, which *does* include
``latent``, and ``_declared_list_entries`` walks ``LOSS_LISTS``, which did not include
``latent_losses``. A conflict can only be seen when both halves see the name, so:

===========================================================  ==============  ============
config                                                       before          after
===========================================================  ==============  ============
``latent_losses: [{name: kl, weight: 0.37}]``                resolves 0.0    resolves 0.37
``... plus losses.latent.lambda_kl: 0.5``                    resolves 0.5,   ``Configuration
                                                             raises nothing  Error``
``image_losses: [{name: l1, ...}] + lambda_l1`` (control)    ``ConfigurationError`` both
===========================================================  ==============  ============

The control row is the point: the mechanism was never broken, it was *blind on one
domain*. A test that only asserted the latent rows could pass against a build in which
the detector had been disabled altogether, so the control is asserted alongside them.

Blast radius on the live corpus is **zero** -- 0 of the 1444 YAMLs carrying a ``losses:``
block declare ``latent_losses`` (measured 2026-09-08). This is a landmine, not a live
miscomputation, so no arm's numbers move.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS, LossConfigSchema
from spectramr.domain.exceptions import ConfigurationError
from spectramr.models.losses.weights import LOSS_LISTS, build_loss_weight_table

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "src" / "spectramr"

#: A weight that is not any schema default. ``LatentLossesConfig.lambda_kl`` defaults to
#: 0.0 and ``lambda_l1`` to 10.0 elsewhere, so a test probing with a default cannot tell
#: "resolved my declaration" from "fell through to the default".
PROBE_WEIGHT = 0.37
CONFLICTING_LAMBDA = 0.5


def _latent_arm(*, with_lambda: bool = False) -> LossConfigSchema:
    """A legal latent arm. ``output_domain: latent`` is required by the schema."""
    kwargs: dict = {
        "output_domain": "latent",
        "latent_losses": [{"name": "kl", "weight": PROBE_WEIGHT}],
    }
    if with_lambda:
        kwargs["latent"] = {"lambda_kl": CONFLICTING_LAMBDA}
    return LossConfigSchema(**kwargs)


class TestLossListsIsDerived:
    """``LOSS_LISTS`` is a view of the SSOT, not a copy of it."""

    def test_loss_lists_equals_the_schema_ssot(self) -> None:
        assert tuple(LOSS_LIST_DOMAINS) == LOSS_LISTS, (
            "weights.LOSS_LISTS must be tuple(LOSS_LIST_DOMAINS). If they can differ, "
            "the weight table and the schema disagree about what a declared loss is."
        )

    def test_latent_losses_is_actually_in_it(self) -> None:
        # Guards the tautology: the assertion above also passes if BOTH tables lost a
        # list. Name the member that was missing.
        assert "latent_losses" in LOSS_LISTS

    @staticmethod
    def _loss_lists_assignments(source: str) -> list[ast.expr]:
        """Every value assigned to a module-level ``LOSS_LISTS``.

        Both assignment shapes are collected. ``LOSS_LISTS: tuple[str, ...] = (...)`` is
        an :class:`ast.AnnAssign`, NOT an :class:`ast.Assign`, and the annotated form is
        the one this file actually uses -- a scan for ``Assign`` alone silently matched
        nothing and passed against the literal it exists to forbid.
        """
        values: list[ast.expr] = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if node.value is None:
                continue
            if any(isinstance(t, ast.Name) and t.id == "LOSS_LISTS" for t in targets):
                values.append(node.value)
        return values

    def test_the_constant_is_not_re_spelled_by_hand(self) -> None:
        values = self._loss_lists_assignments(
            (SRC / "models" / "losses" / "weights.py").read_text()
        )
        assert values, "no LOSS_LISTS assignment found -- the scan is looking at nothing"
        for value in values:
            assert not isinstance(value, (ast.Tuple, ast.List, ast.Set)), (
                "LOSS_LISTS is assigned a literal collection again. It must be derived "
                "from LOSS_LIST_DOMAINS so a new list cannot be forgotten here."
            )

    @pytest.mark.parametrize(
        ("source", "expect_literal"),
        [
            ('LOSS_LISTS: tuple[str, ...] = ("a", "b")', True),  # the annotated form
            ('LOSS_LISTS = ("a", "b")', True),  # the bare form
            ("LOSS_LISTS: tuple[str, ...] = tuple(LOSS_LIST_DOMAINS)", False),
        ],
    )
    def test_the_literal_scan_sees_both_assignment_shapes(
        self, source: str, expect_literal: bool
    ) -> None:
        """Plants each shape (NN15). The annotated one was invisible and passed."""
        values = self._loss_lists_assignments(source)
        assert values, source
        found = any(isinstance(v, (ast.Tuple, ast.List, ast.Set)) for v in values)
        assert found is expect_literal


class TestLatentWeightsAreResolved:
    """The two silent failures, and the control that proves the mechanism works."""

    def test_a_declared_latent_weight_is_resolved(self) -> None:
        """Was 0.0: the loss was built, then weighted to nothing."""
        table = build_loss_weight_table(_latent_arm())
        assert table.weight("kl") == pytest.approx(PROBE_WEIGHT)

    def test_latent_declared_on_both_surfaces_at_odds_raises(self) -> None:
        """Was: resolves to the lambda (0.5) and raises nothing."""
        with pytest.raises(ConfigurationError) as excinfo:
            build_loss_weight_table(_latent_arm(with_lambda=True))
        message = str(excinfo.value)
        assert "latent_losses" in message, message
        assert "losses.latent.lambda_kl" in message, message

    def test_control_the_same_shape_on_image_losses_always_raised(self) -> None:
        """Green on both sides. Isolates the defect to the missing list.

        Without this, a build that simply deleted the conflict detector would turn the
        test above red for the wrong reason and this file would report it as a fix.
        """
        both = LossConfigSchema(
            output_domain="image",
            image_losses=[{"name": "l1", "weight": PROBE_WEIGHT}],
            reconstruction={"lambda_l1": CONFLICTING_LAMBDA},
        )
        with pytest.raises(ConfigurationError):
            build_loss_weight_table(both)


class TestEveryWalkSiteSeesLatent:
    """The consumers routed to the SSOT, each exercised on a latent-only arm."""

    def test_startup_verification_sees_an_unregistered_latent_loss(self) -> None:
        """``verify_startup_losses`` walked three lists, so a bogus latent name passed."""
        from spectramr.infrastructure.loss_audit import verify_startup_losses

        bogus = LossConfigSchema(
            output_domain="latent",
            latent_losses=[{"name": "definitely_not_a_registered_loss", "weight": 1.0}],
        )
        with pytest.raises(RuntimeError, match="definitely_not_a_registered_loss"):
            verify_startup_losses(bogus)

    def test_context_resolver_collects_latent_loss_names(self) -> None:
        from spectramr.infrastructure.validation.context_resolver import _iter_loss_names

        class _Cfg:
            losses = _latent_arm()

        assert "kl" in _iter_loss_names(_Cfg())

    def test_spec_card_renders_the_latent_list(self) -> None:
        """The seed dict was the census blind spot: it mixed field names with
        ``output_domain``, so a scan for *pure* enumerations could not see it -- and
        ``out[k].append`` would KeyError on any list the seed omitted."""
        from spectramr.infrastructure.validation.spec_card import _derive_loss_form

        class _Cfg:
            losses = _latent_arm()

        form = _derive_loss_form(_Cfg())
        assert "latent_losses" in form, "the seed dict still omits a declared list"
        assert [e["name"] for e in form["latent_losses"]] == ["kl"]
        assert form["latent_losses"][0]["weight"] == pytest.approx(PROBE_WEIGHT)


# --------------------------------------------------------------------------- ratchet


#: Sites that enumerate a PROPER SUBSET of the loss lists on purpose, keyed by
#: ``(path relative to src/spectramr, enclosing function)``. Each needs a reason, because
#: a future author reading a bare allowlist cannot tell a decision from an oversight.
ALLOWED_SUBSET_SITES: dict[tuple[str, str], str] = {
    (
        "infrastructure/validation/config_health_checker.py",
        "check_loss_domain_consistency",
    ): (
        "Mirrors LossBuilder's bridge matrix, which defines no bridge into a latent. "
        "latent_losses is legal only with output_domain: latent, which `compat` does "
        "not list, so including it would flag every correct latent arm."
    ),
    ("models/losses/_legacy_weights.py", "_legacy_folding"): (
        "Frozen oracle for the PRE-SSOT semantics, which never read latent_losses. "
        "Deriving it would let the oracle drift with the code it exists to check."
    ),
}


def _subset_enumeration_sites(root: Path) -> list[tuple[str, str, int, tuple[str, ...]]]:
    """Literal collections of loss-list field names that omit at least one.

    Counts a collection whose members are field names even when it also carries
    unrelated members (the ``spec_card`` seed-dict shape), because that mixed form is
    exactly what a stricter "every member must be a field name" scan cannot see.
    """
    fields = set(LOSS_LIST_DOMAINS)
    found: list[tuple[str, str, int, tuple[str, ...]]] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a broken file is another test's job
            continue
        enclosing: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    enclosing.setdefault(id(child), node.name)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                members = [
                    e.value
                    for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
            elif isinstance(node, ast.Dict):
                members = [
                    k.value
                    for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                ]
            else:
                continue
            present = {m for m in members if m in fields}
            if len(present) >= 2 and present != fields:
                found.append(
                    (
                        str(path.relative_to(root)),
                        enclosing.get(id(node), "<module>"),
                        node.lineno,
                        tuple(sorted(fields - present)),
                    )
                )
    return found


class TestNoSecondTable:
    """A structural ratchet, driven over a parametrizable root so it is plantable."""

    def test_src_enumerates_no_proper_subset_of_the_loss_lists(self) -> None:
        offenders = [
            site
            for site in _subset_enumeration_sites(SRC)
            if (site[0], site[1]) not in ALLOWED_SUBSET_SITES
        ]
        assert not offenders, (
            "These sites enumerate some but not all declared loss lists. Derive them "
            "from LOSS_LIST_DOMAINS, or add them to ALLOWED_SUBSET_SITES with the "
            "reason the omission is correct:\n"
            + "\n".join(f"  {f}:{ln} in {fn}() omits {miss}" for f, fn, ln, miss in offenders)
        )

    def test_every_allowlist_entry_still_exists(self) -> None:
        """An allowlist that outlives its site silently widens the ratchet."""
        live = {(f, fn) for f, fn, _, _ in _subset_enumeration_sites(SRC)}
        stale = sorted(set(ALLOWED_SUBSET_SITES) - live)
        assert not stale, f"allowlisted sites that no longer enumerate a subset: {stale}"

    def test_the_scan_sees_a_planted_violation(self, tmp_path: Path) -> None:
        """The scanner is exercised against a tree built to be red (NN15).

        Without this, a scan that silently returned nothing -- a wrong root, a changed
        AST shape -- would read as a clean tree.
        """
        (tmp_path / "planted.py").write_text(
            'BAD = ("image_losses", "kspace_losses", "complex_losses")\n'
        )
        hits = _subset_enumeration_sites(tmp_path)
        assert [(h[0], h[3]) for h in hits] == [("planted.py", ("latent_losses",))]

    def test_the_scan_sees_the_mixed_collection_shape(self, tmp_path: Path) -> None:
        """The spec_card seed-dict shape: field names mixed with unrelated keys."""
        (tmp_path / "planted.py").write_text(
            'def f():\n    return {"present": True, "image_losses": [], "kspace_losses": []}\n'
        )
        hits = _subset_enumeration_sites(tmp_path)
        assert len(hits) == 1, hits
        assert hits[0][1] == "f"
        assert set(hits[0][3]) == {"complex_losses", "latent_losses"}

    def test_the_scan_passes_a_complete_enumeration(self, tmp_path: Path) -> None:
        """The other polarity: a complete enumeration must NOT be reported."""
        names = ", ".join(f'"{n}"' for n in LOSS_LIST_DOMAINS)
        (tmp_path / "fine.py").write_text(f"OK = ({names})\n")
        assert _subset_enumeration_sites(tmp_path) == []
