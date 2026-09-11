"""The guard's skip sets are a closed world over the names computers ask for.

``LossBuilder``'s ``unmigrated`` guard refuses a loss weight that no module was
built for. Three sets say when that refusal is wrong:
``STRATEGY_MANAGED_LOSSES`` (a strategy owns the term),
``COMPUTER_RESOLVED_LAMBDA_SOURCES`` (a computer reads the weight and computes
the term inline) and the module-backed names, where the refusal is correct.

Each set is a verified hand-list, the shape that already failed once: the escape
hatch shipped with one entry against five names, so ``l1``, ``l2`` and ``kl`` were
refused although each trains. The tests below enumerate the vocabulary by AST and
check every declarable source is classified.
"""

from __future__ import annotations

import ast
import pathlib
import typing

import pytest
from pydantic import BaseModel

from .test_loss_builder_unmigrated_guard_2026_09 import _build

#: Names whose term comes from a module ``LossBuilder`` puts in its dict. The
#: computer reads the weight *and* the module, so a lambda-only declaration leaves
#: the module ``None`` and the term does not train — refusing is correct.
#: ``unified_vae.py:121`` / ``unified_gan.py:143`` are the deciding ``losses.get``.
MODULE_BACKED_LOSSES = frozenset({"perceptual", "adversarial"})


def _computers_dir() -> pathlib.Path:
    """Walk up rather than count ``parents``: a fixed index breaks on a move, the
    scan returns nothing, and every assertion built on it then passes."""
    for parent in pathlib.Path(__file__).resolve().parents:
        candidate = parent / "src/spectramr/models/losses/computers"
        if candidate.is_dir():
            return candidate
    raise AssertionError("could not locate src/spectramr/models/losses/computers")


_COMPUTERS = _computers_dir()


def _weight_name_literals() -> dict[str, list[str]]:
    """Every ``self._get_loss_weight("<name>")`` literal under ``computers/``.

    Parsed, not grepped: ``unified_diffusion_reconstruction.py:377`` mentions the
    call inside a docstring, and a text scan counts that as a read site.
    """
    found: dict[str, list[str]] = {}
    for path in sorted(_COMPUTERS.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "_get_loss_weight" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.setdefault(first.value, []).append(f"{path.name}:{node.lineno}")
    return found


def _declarable_sources() -> dict[str, set[str]]:
    """Canonical loss name → every ``losses.<section>.lambda_<field>`` reaching it.

    Read off the schema, because canonicalisation erases the section:
    ``losses.gan.lambda_adv`` becomes ``adversarial``."""
    from spectramr.config.schemas.loss import LossConfigSchema
    from spectramr.models.losses.weights import LAMBDA_SECTIONS, canonical_loss_name

    out: dict[str, set[str]] = {}
    for section in LAMBDA_SECTIONS:
        field = LossConfigSchema.model_fields.get(section)
        if field is None:
            continue
        model = next(
            (
                c
                for c in (field.annotation, *typing.get_args(field.annotation))
                if isinstance(c, type) and issubclass(c, BaseModel)
            ),
            None,
        )
        if model is None:
            continue
        for name in model.model_fields:
            if name.startswith("lambda_"):
                canonical = canonical_loss_name(name[len("lambda_") :])
                out.setdefault(canonical, set()).add(f"losses.{section}.{name}")
    return out


class TestTheVocabularyIsClosed:
    """Every name a computer asks for is classified, on every surface it has."""

    def test_the_ast_scan_finds_the_read_sites(self):
        """A scan that found nothing would make every assertion below vacuous."""
        literals = _weight_name_literals()
        assert {"l1", "l2", "kl", "perceptual", "adversarial"} <= set(literals), literals

    def test_every_declarable_source_is_classified(self):
        from spectramr.infrastructure.training.builders.loss_builder import (
            COMPUTER_RESOLVED_LAMBDA_SOURCES,
            STRATEGY_MANAGED_LOSSES,
        )
        from spectramr.models.losses.weights import canonical_loss_name

        managed = {canonical_loss_name(n) for n in STRATEGY_MANAGED_LOSSES} | set(
            STRATEGY_MANAGED_LOSSES
        )
        declarable = _declarable_sources()

        unclassified: list[str] = []
        for name, sites in _weight_name_literals().items():
            # A module-backed name is refused on every surface it has, so a new
            # section growing a ``lambda_`` field for one is not a finding. The
            # sources below are checked one at a time because exemption is
            # per-surface: ``losses.diffusion.lambda_mse`` was exempt while
            # ``losses.reconstruction.lambda_l2`` reached the same name and was not.
            if name in managed or name in MODULE_BACKED_LOSSES:
                continue
            for source in sorted(declarable.get(name, set())):
                if source not in COMPUTER_RESOLVED_LAMBDA_SOURCES:
                    unclassified.append(f"{source} → '{name}' (read at {', '.join(sites)})")

        assert not unclassified, (
            "A computer reads these weights, and an author can declare each one as a "
            "lambda, but nothing says whether the term trains without a built module. "
            "Read the call site, then add the source to "
            "COMPUTER_RESOLVED_LAMBDA_SOURCES (it computes the term inline) or the "
            "name to MODULE_BACKED_LOSSES (it needs the builder's module):\n  "
            + "\n  ".join(unclassified)
        )

    def test_every_exempt_source_is_still_a_schema_field(self):
        """A renamed or deleted field would leave a dead exemption behind."""
        from spectramr.infrastructure.training.builders.loss_builder import (
            COMPUTER_RESOLVED_LAMBDA_SOURCES,
        )

        every_source = {s for sources in _declarable_sources().values() for s in sources}
        stale = sorted(set(COMPUTER_RESOLVED_LAMBDA_SOURCES) - every_source)
        assert not stale, stale

    def test_the_reinterpreted_set_is_a_strict_subset(self):
        """A lambda read under a different meaning is also one no module is built
        for, so the narrow set sits inside the wide one. The reverse does not
        hold: ``reconstruction.lambda_l1`` needs no module and still duplicates a
        list entry of the same name."""
        from spectramr.infrastructure.training.builders.loss_builder import (
            COMPUTER_RESOLVED_LAMBDA_SOURCES,
            REINTERPRETED_LAMBDA_SOURCES,
        )

        assert set(REINTERPRETED_LAMBDA_SOURCES) < set(COMPUTER_RESOLVED_LAMBDA_SOURCES)

    @pytest.mark.parametrize(
        ("relative", "required"),
        [
            (
                "src/spectramr/infrastructure/validation/config_health_checker.py",
                "REINTERPRETED_LAMBDA_SOURCES",
            ),
            (
                "scripts/migrations/migrate_loss_lambdas_to_domain_lists.py",
                "dual_surface_loss_declarations",
            ),
        ],
    )
    def test_the_duplicate_report_and_the_migration_import_one_set(self, relative, required):
        """Non-negotiable 17: the script may not delete what the audit keeps.

        Checked on the ``import``, and the two rows require DIFFERENT names on
        purpose. The checker is where the exclusion is applied, so it must import
        the narrow constant. The migration is downstream of that decision: it
        imports ``dual_surface_loss_declarations``, which has already subtracted
        the exclusion, and gates every removal on the pairs that function reports.

        Requiring the constant on the migration row instead would be satisfied by
        an unused import -- the script has nothing to do with it -- which is the
        vacuous shape this check exists to rule out. Requiring the *function* is
        the direct read: delete the call and the migration's own plants go red.

        Neither may import the wider set: naming it here would be a second
        exclusion policy, which is the two-owner split itself."""
        for parent in pathlib.Path(__file__).resolve().parents:
            candidate = parent / relative
            if candidate.exists():
                break
        else:
            raise AssertionError(f"could not locate {relative}")

        imported = {
            alias.name
            for node in ast.walk(ast.parse(candidate.read_text()))
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert required in imported, sorted(imported)
        assert "COMPUTER_RESOLVED_LAMBDA_SOURCES" not in imported

    def test_module_backed_names_are_not_also_exempt(self):
        """The two classifications are exclusive; overlap would hide a real drop."""
        from spectramr.infrastructure.training.builders.loss_builder import (
            COMPUTER_RESOLVED_LAMBDA_SOURCES,
        )

        declarable = _declarable_sources()
        for name in MODULE_BACKED_LOSSES:
            overlap = declarable.get(name, set()) & set(COMPUTER_RESOLVED_LAMBDA_SOURCES)
            assert not overlap, f"'{name}' is module-backed but exempt via {sorted(overlap)}"


def _armed(section: str, field: str, weight: float, **extra) -> dict:
    """One list entry to arm the guard, plus a single lambda-only declaration."""
    block = {
        "image_losses": [{"name": "ssim", "weight": 0.5, "enabled": True}],
        "kspace_losses": [],
        "complex_losses": [],
        "policy": {"output_domain": "image"},
        section: {field: weight, **extra},
    }
    return block


class TestInlineComputedLambdasBuild:
    """The false refusals ``d4d2619db`` shipped. Each weight is read at a cited
    line and the term is computed there, so refusing blocks a run that trains."""

    @pytest.mark.parametrize(
        "section,field",
        [
            ("reconstruction", "lambda_l1"),
            ("reconstruction", "lambda_l2"),
            ("latent", "lambda_l1"),
            ("latent", "lambda_kl"),
        ],
    )
    def test_a_lambda_only_inline_term_does_not_raise(self, section, field):
        losses = _build(_armed(section, field, 1.0))
        assert "ssim" in losses


class TestModuleBackedLambdasAreRefused:
    """The planted violations for the second pass (non-negotiable 15): a lambda
    whose term needs a module nobody built must still stop the run."""

    def test_a_lambda_only_perceptual_raises(self):
        from spectramr.domain.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match=r"not in the v6\.0"):
            _build(_armed("reconstruction", "lambda_perceptual", 1.0))

    def test_a_lambda_only_adversarial_with_the_flag_off_raises(self):
        """``enable_adversarial`` defaults False, so ``_build_composite_gan`` never
        runs and the adversarial module is absent — the weight buys nothing."""
        from spectramr.domain.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match=r"not in the v6\.0"):
            _build(_armed("gan", "lambda_adv", 1.0))

    def test_the_same_adversarial_weight_builds_once_the_flag_is_on(self):
        """The refusal above is about the missing module, not the lambda surface."""
        losses = _build(_armed("gan", "lambda_adv", 1.0, enable_adversarial=True))
        assert "adversarial" in losses

    def test_the_message_names_all_four_domain_lists(self):
        """``latent_losses`` is built by ``_build_list_based_losses`` too, so a fix
        hint naming only three sends an author to a list that cannot hold the term."""
        from spectramr.domain.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError) as excinfo:
            _build(_armed("reconstruction", "lambda_perceptual", 1.0))
        message = str(excinfo.value)
        for name in ("image_losses", "kspace_losses", "complex_losses", "latent_losses"):
            assert name in message, name


class TestDeclaredNamesIn:
    """The guard tests the name the author wrote, not the canonical one."""

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("losses.reconstruction.lambda_marker", {"marker"}),
            ("losses.kspace_losses[sobolev_kspace].weight", {"sobolev_kspace"}),
            ("losses.gan.lambda_adv", {"adv"}),
            # A loss declared on both surfaces joins its sources with '+'.
            (
                "losses.reconstruction.lambda_l1+losses.image_losses[l1].weight",
                {"l1"},
            ),
        ],
    )
    def test_raw_names_are_recovered_from_a_weight_spec_source(self, source, expected):
        from spectramr.infrastructure.training.builders.loss_builder import (
            declared_names_in,
        )

        assert declared_names_in(source) == expected

    def test_the_three_drifting_managed_names_are_authored_in_schema_spelling(self):
        """If these stop drifting, the raw-name lookup can be simplified.

        If a *fourth* name starts drifting, this test does not fail — but the
        parametrised no-raise cases elsewhere will, which is the intended alarm.
        """
        from spectramr.infrastructure.training.builders.loss_builder import (
            STRATEGY_MANAGED_LOSSES,
        )
        from spectramr.models.losses.weights import canonical_loss_name

        drifting = {n for n in STRATEGY_MANAGED_LOSSES if canonical_loss_name(n) != n}
        assert drifting == {"content", "marker", "patch_nce"}, sorted(drifting)
